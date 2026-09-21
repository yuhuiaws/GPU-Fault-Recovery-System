"""BOOT-020 public AdminConfig apply/restore, separate from engine fault injection."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from gpu_fault.admin.config import AdminConfig, load_desired_admin_config
from gpu_fault.admin.config_parser import AdminConfigError
from gpu_fault.admin.config_file import (
    admin_config_file_path,
    load_admin_config_file,
    write_admin_config_file,
)
from gpu_fault.admin.execution import run_driver
from gpu_fault.admin.membership_lock import administrator_operation_lock
from gpu_fault.admin.operation_lock import SITE_OPERATION_LOCK_FD_ENV
from gpu_fault.admin.site import load_site
from scripts.e2e.regional.boot_acceptance_lifecycle import (
    deployment_generations,
    parse_status_report,
)
from scripts.e2e.regional.boot_membership_observation import cpu_observation


from scripts.e2e.regional.admin_cli import admin_command  # noqa: E402


def admin_config_command(state_dir: Path, reference: str, *args: str) -> list[str]:
    """The public ``gpu-fault-admin config`` entrypoint for ``state_dir``.

    Uses the state dir's own deploy-host CLI when it has one (the checkout's
    module form is refused there, live 2026-09-20); the module form otherwise.
    """

    return [
        *admin_command(state_dir),
        "config",
        "--state-dir",
        str(state_dir),
        "--reference",
        reference,
        *args,
    ]


def validate_admin_target(state_dir: Path, config_path: Path) -> None:
    site = load_site(state_dir / "site.yaml")
    value = json.loads(config_path.read_text(encoding="utf-8"))
    for field in ("cpu_eks_arn", "namespace", "aws_region", "site_name"):
        if value.get(field) != site.release_config.get(field):
            raise ValueError(
                "BOOT-020 administrator state differs from the release target"
            )
    if {
        (item["cluster_id"], item["eks_cluster_arn"]) for item in value["clusters"]
    } != {
        (item["cluster_id"], item["eks_cluster_arn"])
        for item in site.release_config["clusters"]
    }:
        raise ValueError(
            "BOOT-020 administrator GPU membership differs from the release target"
        )


def changed_roles(before: dict[str, Any], after: dict[str, Any]) -> set[str]:
    """The CPU roles whose replicas, Pod template or Pod set changed.

    ``generation`` is ignored on purpose: Kubernetes bumps a Deployment's
    generation on annotation changes as well as spec changes, and every
    ``gpu-fault-admin config`` apply re-stamps ``gpu-fault.io/admin-config-sha256``
    on ALL CPU Deployments (live 2026-09-20, BOOT-020 a3: api-ha and spool moved
    +1 per apply with no ReplicaSet and no Pod change, and the drill called that
    a rollout outside the declared scope).
    """
    if set(before) != set(after):
        raise ValueError("public config changed the CPU role inventory")
    if any(before[name]["uid"] != after[name]["uid"] for name in before):
        raise ValueError("public config replaced a CPU Deployment")
    return {
        name
        for name in before
        if _rollout_identity(before[name]) != _rollout_identity(after[name])
    }


def _rollout_identity(observation: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in observation.items() if key != "generation"}


def public_config_roundtrip(
    state_dir: Path, *, desired: AdminConfig, reference: str
) -> dict[str, Any]:
    with administrator_operation_lock(state_dir) as lock_fd:
        return _roundtrip_locked(
            state_dir, desired=desired, reference=reference, lock_fd=lock_fd
        )


def _roundtrip_locked(
    state_dir: Path, *, desired: AdminConfig, reference: str, lock_fd: int
) -> dict[str, Any]:
    if not reference.strip():
        raise ValueError("public config acceptance requires a change reference")
    site_file = state_dir / "site.yaml"
    site = load_site(site_file)
    before_config = load_desired_admin_config(state_dir)
    editable = admin_config_file_path(state_dir)
    if (state_dir / "admin-config/pending.json").exists():
        raise ValueError("public config acceptance cannot take over a pending apply")
    if (
        editable.exists()
        and load_admin_config_file(editable, base=before_config) != before_config
    ):
        raise ValueError(
            "public config acceptance cannot overwrite an administrator edit"
        )
    roles = {
        {
            "worker": "gpu-fault-control-worker",
            "ingress": "gpu-fault-api-ha",
            "spool": "gpu-fault-telemetry-spool-worker",
        }[role]
        for role, digest in before_config.role_sha256().items()
        if desired.role_sha256()[role] != digest
    }
    if not roles or desired.aurora != before_config.aurora:
        raise ValueError("BOOT-020 public config drill requires a CPU-only change")
    before = cpu_observation(site)
    gpu_before = deployment_generations(site_file)["gpu"]

    def command(*args: str) -> dict[str, Any]:
        completed = run_driver(
            admin_config_command(state_dir, reference, *args),
            env={**os.environ, SITE_OPERATION_LOCK_FD_ENV: str(lock_fd)},
            pass_fds=(lock_fd,),
            capture_output=True,
        )
        if completed.returncode:
            raise ValueError(
                "public AdminConfig command failed; preserve its pending state and audit"
            )
        return parse_status_report(completed.stdout)

    # Admit the desired config before touching the editable file: an inadmissible
    # drill config (live 2026-09-20: 6->7 workers exceeded the validated PostgreSQL
    # connection budget) would otherwise be written, refused by the administrator's
    # dry-run, and left behind as an edit the same guard cannot even parse.
    try:
        desired.validate()
    except AdminConfigError as exc:
        raise ValueError(f"BOOT-020 desired config is not admissible: {exc}") from exc
    write_admin_config_file(editable, desired, overwrite=True)
    applied = False
    result: dict[str, Any] = {}
    try:
        dry_run = command("--dry-run")
        if (
            dry_run.get("status") != "DRY_RUN"
            or (state_dir / "admin-config/pending.json").exists()
        ):
            raise ValueError("AdminConfig dry-run changed the pending transaction")
        application = command()
        applied = application.get("status") == "APPLIED"
        if not applied or load_desired_admin_config(state_dir) != desired:
            raise ValueError("public config did not commit the requested configuration")
        after = cpu_observation(load_site(site_file))
        moved = changed_roles(before, after)
        gpu_after = deployment_generations(site_file)["gpu"]
        if moved != roles or gpu_after != gpu_before:
            raise ValueError(
                "public config rolled a role outside its declared scope: "
                f"moved={sorted(moved)} expected={sorted(roles)} "
                f"gpu_generations_changed={gpu_after != gpu_before}"
            )
        if (
            command().get("status") != "NOOP"
            or cpu_observation(load_site(site_file)) != after
        ):
            raise ValueError("repeating public config was not a strict NOOP")
        audit_path = Path(str(application.get("audit") or ""))
        if not audit_path.resolve().is_relative_to(
            (state_dir / "admin-config").resolve()
        ):
            raise ValueError("public config audit is not bound to this state directory")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if not audit or (state_dir / "admin-config/pending.json").exists():
            raise ValueError("public config lacks its completed audit")
        result = {
            "entrypoint": "gpu-fault-admin config",
            "applied_config_sha256": desired.sha256(),
            "affected_deployments": sorted(roles),
            "dry_run": True,
            "repeat_noop": True,
            "audit_present": True,
        }
    finally:
        if applied:
            write_admin_config_file(editable, before_config, overwrite=True)
            restored = command()
            if (
                restored.get("status") != "APPLIED"
                or load_desired_admin_config(state_dir) != before_config
            ):
                raise ValueError("public config restore did not complete")
            final = cpu_observation(load_site(site_file))
            if (
                any(
                    final[name]["template_sha256"] != before[name]["template_sha256"]
                    or final[name]["replicas"] != before[name]["replicas"]
                    for name in before
                )
                or deployment_generations(site_file)["gpu"] != gpu_before
            ):
                raise ValueError(
                    "public config restore did not restore the original runtime scope"
                )
            result["restored_config_sha256"] = before_config.sha256()
    return result
