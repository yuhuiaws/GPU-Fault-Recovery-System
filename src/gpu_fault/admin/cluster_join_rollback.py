"""Undo helpers for a rolled-back ``join-cluster`` attempt."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.cluster_removal import _clear_installer_annotations, _gpu_kubectl
from gpu_fault.admin.site import RenderedSite, effective_environment
from gpu_fault_release.regional_deployment_inventory import GPU_RECONCILER_DEPLOYMENT


def restore_current_context(
    kubeconfig: Path, *, deleted: str, errors: list[str]
) -> None:
    """Point ``current-context`` away from the context the rollback deleted.

    ``aws eks update-kubeconfig --alias`` makes the joined cluster current;
    deleting only the context leaves that pointer dangling and every
    ``kubectl --kubeconfig`` call without ``--context`` failing. The first
    remaining context (another managed cluster) takes over; with none left the
    pointer is unset.
    """

    base = ["kubectl", "--kubeconfig", str(kubeconfig), "config"]
    current = subprocess.run(base + ["current-context"], text=True, capture_output=True)
    if current.returncode or (current.stdout or "").strip() != deleted:
        return
    remaining = subprocess.run(
        base + ["get-contexts", "-o", "name"], text=True, capture_output=True
    )
    names = [
        line.strip()
        for line in (remaining.stdout or "").splitlines()
        if line.strip() and line.strip() != deleted
    ]
    repoint = (
        base + ["use-context", names[0]]
        if names
        else base + ["unset", "current-context"]
    )
    result = subprocess.run(repoint, text=True, capture_output=True)
    if result.returncode:
        errors.append("kube current-context rollback: " + result.stderr.strip())


def rollback_command(
    arguments: list[str],
    *,
    not_found: tuple[str, ...],
) -> None:
    result = subprocess.run(arguments, text=True, capture_output=True)
    if result.returncode == 0:
        return
    message = (result.stdout or "") + "\n" + (result.stderr or "")
    if any(value in message for value in not_found):
        return
    raise BootstrapError(
        f"rollback command failed ({result.returncode}): "
        f"{' '.join(arguments[:3])}: {result.stderr.strip()}"
    )


def rollback_iam_role(role: object, *, label: str, errors: list[str]) -> None:
    """Delete a role a failed join created (inline policy first); absent is fine."""

    if not isinstance(role, dict) or not role.get("role_arn"):
        return
    role_name = str(role["role_arn"]).rsplit("/", 1)[-1]
    policy = str(role.get("inline_policy_name") or "")
    if policy:
        try:
            rollback_command(
                [
                    "aws",
                    "iam",
                    "delete-role-policy",
                    "--role-name",
                    role_name,
                    "--policy-name",
                    policy,
                ],
                not_found=("NoSuchEntity",),
            )
        except BootstrapError as exc:
            errors.append(f"{label} policy rollback: {exc}")
    try:
        rollback_command(
            ["aws", "iam", "delete-role", "--role-name", role_name],
            not_found=("NoSuchEntity",),
        )
    except BootstrapError as exc:
        errors.append(f"{label} role rollback: {exc}")


def nothing_installed(candidate: RenderedSite, config: Path) -> bool:
    """Whether the installed-resource registry lists no GPU-plane resource.

    Unreadable output counts as "something may be installed": the cleanup
    script then validates the inventory itself, fail-closed as before.
    """

    result = subprocess.run(
        [
            "python3",
            str(
                candidate.repository_root
                / "deploy/control-plane/tools/collect_installed_resource_registry.py"
            ),
            "--config",
            str(config),
        ],
        cwd=candidate.repository_root,
        env=effective_environment(candidate),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        return False
    try:
        registry = json.loads(result.stdout or "")
    except json.JSONDecodeError:
        return False
    if not isinstance(registry, dict):
        return False
    gpu = registry.get("gpu")
    resources = gpu.get("resources") if isinstance(gpu, dict) else None
    return not resources and not registry.get("unregistered_resources")


def ensure_kube_context(kubeconfig: Path, target: Mapping[str, Any]) -> None:
    """Recreate the GPU cluster's kubeconfig context if it is gone.

    A rollback deletes the context as its last Kubernetes step; when it fails
    partway and is retried (live, 2026-09-12), the annotation, namespace and
    cleanup undos would all fail on the missing context. ``update-kubeconfig``
    is the same call the join made and is idempotent.
    """

    context = str(target.get("context") or "")
    if not context:
        return
    probe = subprocess.run(
        ["kubectl", "--kubeconfig", str(kubeconfig), "config", "get-contexts", context],
        text=True,
        capture_output=True,
    )
    if probe.returncode == 0:
        return
    subprocess.run(
        [
            "aws",
            "eks",
            "update-kubeconfig",
            "--region",
            str(target.get("region") or ""),
            "--name",
            str(target.get("eks_name") or ""),
            "--kubeconfig",
            str(kubeconfig),
            "--alias",
            context,
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    if kubeconfig.exists():
        kubeconfig.chmod(0o600)


def clear_stale_installer_annotations(
    site: RenderedSite, target: Mapping[str, Any], nodes: list[str]
) -> None:
    """Drop the installer annotations on ``nodes`` when no Reconciler owns them.

    A cluster being joined has no node-installer Reconciler yet, so whatever
    installer-state its nodes carry is a leftover: remove-cluster clears the
    annotations, but a Reconciler still terminating rewrites unannotated nodes
    as "Retrying" (live 2026-09-12), and the node barrier then refuses every
    node as "installer-active". With a Reconciler present the annotations are
    live state and stay.
    """

    if not nodes:
        return
    probe = subprocess.run(
        [
            *_gpu_kubectl(site, dict(target)),
            "-n",
            str(site.release_config["namespace"]),
            "get",
            "deployment",
            GPU_RECONCILER_DEPLOYMENT,
            "-o",
            "name",
        ],
        text=True,
        capture_output=True,
    )
    if probe.returncode == 0:
        return
    if "not found" not in (probe.stderr or "").lower():
        raise BootstrapError(
            "cannot tell whether a node-installer Reconciler owns the installer "
            "annotations: " + (probe.stderr or "").strip()
        )
    _clear_installer_annotations(site, dict(target), nodes)
