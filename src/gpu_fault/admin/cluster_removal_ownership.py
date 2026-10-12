"""The OWNERSHIP RELEASE step of ``remove-cluster`` and the GPU kubectl prefix.

Kept apart from ``cluster_removal`` so the removal pipeline stays within its
size budget; the step itself is ``ownership_release.release_cluster_ownership``.
"""

from __future__ import annotations

from typing import Any, Protocol

from gpu_fault.admin.bootstrap_common import CommandRunner
from gpu_fault.admin.ownership_release import release_cluster_ownership
from gpu_fault.admin.site import RenderedSite

__all__ = ["RemovalRequest", "gpu_kubectl", "release_ownership"]


class RemovalRequest(Protocol):
    """What the release step reads from a ``RemoveClusterRequest``."""

    @property
    def site(self) -> RenderedSite: ...

    @property
    def keep_ownership_label(self) -> bool: ...


def gpu_kubectl(site: RenderedSite, target: dict[str, Any]) -> list[str]:
    """``kubectl`` for the removed cluster's context, through the site kubeconfig."""

    command = ["kubectl"]
    kubeconfig = site.release_config.get("gpu_kubeconfig") or site.environment.get(
        "KUBECONFIG"
    )
    if kubeconfig:
        command.extend(["--kubeconfig", str(kubeconfig)])
    command.extend(["--context", str(target["context"])])
    return command


def release_ownership(
    request: RemovalRequest,
    target: dict[str, Any],
    runner: CommandRunner,
) -> dict[str, Any]:
    """The OWNERSHIP RELEASE step: the label leaves the groups the site selected.

    Last on purpose: the Kubernetes cleanup and the registry drain before it
    still select the cluster's nodes by the live label. The evidence names
    every group with its label before and after and the outcome
    (``released``, ``retained`` under ``--keep-ownership-label``, ``manual``
    for a Karpenter NodePool the step does not edit).
    """

    return release_cluster_ownership(
        runner,
        region=str(request.site.release_config["aws_region"]),
        site_id=str(request.site.release_config["site_name"]),
        target=target,
        keep=request.keep_ownership_label,
        kubectl=gpu_kubectl(request.site, target),
    )
