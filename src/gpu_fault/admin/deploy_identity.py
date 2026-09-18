"""Persist the first public deploy request before any source/bootstrap work."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gpu_fault.admin.atomic_json import write_json_atomic
from gpu_fault.admin.bootstrap_common import Arn, BootstrapError
from gpu_fault.admin.membership_lock import administrator_operation_lock
from gpu_fault.admin.notifications import validate_admin_email
from gpu_fault.admin.site import SiteConfigError

INITIAL_REQUEST_FILE = "initial-deploy-request.json"


def initial_deploy_request(
    state_dir: Path,
    *,
    cpu_arn: str,
    gpu_arns: tuple[str, ...],
    admin_email: str,
) -> tuple[str, tuple[str, ...], str]:
    """Recover only exact, explicitly recorded inputs, never inferred resources."""
    with administrator_operation_lock(state_dir):
        path = state_dir / INITIAL_REQUEST_FILE
        recorded: dict[str, Any] | None = None
        if path.exists() or path.is_symlink():
            try:
                if (
                    path.is_symlink()
                    or not path.is_file()
                    or path.stat().st_mode & 0o077
                ):
                    raise ValueError("unsafe request file")
                recorded = json.loads(path.read_text(encoding="utf-8"))
                if (
                    not isinstance(recorded, dict)
                    or type(recorded.get("schema_version")) is not int
                    or recorded["schema_version"] != 1
                    or recorded.get("state_dir") != str(state_dir.resolve())
                    or not isinstance(recorded.get("cpu_cluster_arn"), str)
                    or not recorded["cpu_cluster_arn"]
                    or not isinstance(recorded.get("admin_email"), str)
                    or not recorded["admin_email"]
                    or not isinstance(recorded.get("gpu_cluster_arns"), list)
                    or not recorded["gpu_cluster_arns"]
                    or any(
                        not isinstance(item, str) or not item
                        for item in recorded["gpu_cluster_arns"]
                    )
                ):
                    raise ValueError("invalid request identity")
            except (OSError, ValueError) as exc:
                raise SiteConfigError(
                    "initial deploy request is invalid; reconcile the original inputs"
                ) from exc
        if recorded is not None:
            supplied = {
                "cpu_cluster_arn": cpu_arn,
                "gpu_cluster_arns": list(gpu_arns),
                "admin_email": admin_email,
            }
            for key, value in supplied.items():
                if value and value != recorded[key]:
                    raise SiteConfigError(
                        f"initial deploy request conflicts on {key}; resume the original request"
                    )
            return (
                recorded["cpu_cluster_arn"],
                tuple(recorded["gpu_cluster_arns"]),
                recorded["admin_email"],
            )
        if not cpu_arn or not gpu_arns:
            raise SiteConfigError(
                "deploy requires --cpu-cluster-arn and at least one --gpu-cluster-arn"
            )
        if not admin_email:
            raise SiteConfigError("deploy requires --admin-email")
        if len(gpu_arns) != len(set(gpu_arns)):
            raise SiteConfigError("initial deploy request contains duplicate GPU ARNs")
        try:
            identities = [Arn.parse(value) for value in (cpu_arn, *gpu_arns)]
            if (
                any(
                    item.service not in {"eks", "sagemaker"}
                    or not item.resource.startswith("cluster/")
                    or not item.region
                    or len(item.account) != 12
                    or not item.account.isdigit()
                    for item in identities
                )
                or len(
                    {(item.partition, item.region, item.account) for item in identities}
                )
                != 1
                or cpu_arn in gpu_arns
            ):
                raise BootstrapError(
                    "first deploy requires distinct, same-scope EKS/HyperPod clusters"
                )
            admin_email = validate_admin_email(admin_email)
        except BootstrapError as exc:
            raise SiteConfigError(str(exc)) from exc
        write_json_atomic(
            path,
            {
                "schema_version": 1,
                "state_dir": str(state_dir.resolve()),
                "cpu_cluster_arn": cpu_arn,
                "gpu_cluster_arns": list(gpu_arns),
                "admin_email": admin_email,
            },
        )
        return cpu_arn, gpu_arns, admin_email
