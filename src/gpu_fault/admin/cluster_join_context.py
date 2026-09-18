"""Bound local kubeconfig recovery and stale installer metadata for joins."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.site import RenderedSite, effective_environment
from gpu_fault_release.regional_deployment_inventory import GPU_RECONCILER_DEPLOYMENT


def ensure_kube_context(
    site: RenderedSite, kubeconfig: Path, target: Mapping[str, Any]
) -> None:
    from gpu_fault.admin import cluster_join as join

    if kubeconfig.resolve() == Path(site.release_config["cpu_kubeconfig"]).resolve():
        raise join.JoinTargetIdentityError("GPU rollback cannot rewrite CPU kubeconfig")
    context = target.get("context")
    if any(
        not isinstance(target.get(key), str) or not target[key]
        for key in ("context", "region", "eks_name", "eks_arn", "hyperpod_arn")
    ):
        raise join.JoinTargetIdentityError("GPU context recovery lacks target identity")
    prefix = ["kubectl", "--kubeconfig", str(kubeconfig), "config"]
    environment = effective_environment(site)
    result = join.run_command(
        [*prefix, "get-contexts", "-o", "name"],
        environment=environment,
        timeout_seconds=30,
    )
    if result.returncode:
        raise BootstrapError("cannot inspect GPU contexts before rollback")
    if context in (result.stdout or "").splitlines():
        return
    result = join.run_command(
        [
            "aws",
            "eks",
            "update-kubeconfig",
            "--region",
            str(target["region"]),
            "--name",
            str(target["eks_name"]),
            "--kubeconfig",
            str(kubeconfig),
            "--alias",
            str(context),
        ],
        environment=environment,
        timeout_seconds=120,
    )
    if result.returncode:
        raise BootstrapError("cannot restore the recorded GPU kubeconfig context")
    if kubeconfig.is_file():
        kubeconfig.chmod(0o600)
    result = join.run_command(
        [*prefix, "get-contexts", "-o", "name"],
        environment=environment,
        timeout_seconds=30,
    )
    if result.returncode or context not in (result.stdout or "").splitlines():
        raise BootstrapError("restored GPU context is not observable")


def restore_current_context(
    site: RenderedSite,
    kubeconfig: Path,
    *,
    deleted: str,
    remaining: Sequence[str],
) -> None:
    from gpu_fault.admin import cluster_join as join

    prefix = ["kubectl", "--kubeconfig", str(kubeconfig), "config"]
    environment = effective_environment(site)
    current = join.run_command(
        [*prefix, "view", "-o", "jsonpath={.current-context}"],
        environment=environment,
        timeout_seconds=30,
    )
    if current.returncode:
        raise BootstrapError("cannot read GPU current-context after rollback")
    if (current.stdout or "").strip() != deleted:
        return
    managed = {str(item["context"]) for item in site.release_config["clusters"]}
    names = sorted((set(remaining) & managed) - {deleted})
    command = ["use-context", names[0]] if names else ["unset", "current-context"]
    result = join.run_command(
        [*prefix, *command], environment=environment, timeout_seconds=30
    )
    if result.returncode:
        raise BootstrapError("cannot restore GPU current-context after rollback")


def clear_stale_installer_annotations(
    site: RenderedSite,
    target: Mapping[str, Any],
    nodes: list[str],
    *,
    kubeconfig: Path | None = None,
) -> None:
    from gpu_fault.admin import cluster_join as join

    if not nodes:
        return
    node_uids = target.get("expected_node_uids")
    if not isinstance(node_uids, dict) or set(node_uids) != set(nodes):
        raise join.JoinTargetIdentityError("stale installer cleanup lacks Node UIDs")
    prefix = [
        "kubectl",
        "--kubeconfig",
        str(kubeconfig or join._gpu_kubeconfig(site)),
        "--context",
        str(target["context"]),
        "-n",
        str(site.release_config["namespace"]),
    ]
    environment = effective_environment(site)
    deployment = join.run_command(
        [
            *prefix,
            "get",
            "deployment",
            GPU_RECONCILER_DEPLOYMENT,
            "--ignore-not-found",
            "-o",
            "json",
        ],
        environment=environment,
        timeout_seconds=30,
    )
    if deployment.returncode:
        raise BootstrapError("cannot prove the installer Reconciler is absent")
    if (deployment.stdout or "").strip():
        try:
            document = json.loads(deployment.stdout)
            metadata = document["metadata"]
            if (
                document.get("kind") != "Deployment"
                or metadata.get("name") != GPU_RECONCILER_DEPLOYMENT
                or metadata.get("namespace") != site.release_config["namespace"]
                or not isinstance(metadata.get("uid"), str)
                or not metadata["uid"]
            ):
                raise ValueError
        except (AttributeError, KeyError, TypeError, ValueError):
            raise BootstrapError("installer Deployment identity is malformed") from None
        return
    pods = join.run_command(
        [
            *prefix,
            "get",
            "pods",
            "-l",
            f"app={GPU_RECONCILER_DEPLOYMENT}",
            "-o",
            "json",
        ],
        environment=environment,
        timeout_seconds=30,
    )
    if pods.returncode:
        raise BootstrapError("cannot prove the installer Reconciler Pods are absent")
    try:
        document = json.loads(pods.stdout)
        items = document["items"]
        if document.get("kind") not in {"List", "PodList"} or not isinstance(
            items, list
        ):
            raise ValueError
    except (KeyError, TypeError, ValueError):
        raise BootstrapError("installer Pod inventory is malformed") from None
    if items:
        return
    from gpu_fault.admin.cluster_removal_kubernetes import clear_installer_annotations
    from gpu_fault.node_installer_reconciler import INSTALLER_NODE_ANNOTATIONS

    clear_installer_annotations(
        join.run_command,
        prefix[:-2],
        hyperpod_name=str(target["hyperpod_cluster_name"]),
        node_uids=node_uids,
        annotations=INSTALLER_NODE_ANNOTATIONS,
    )
