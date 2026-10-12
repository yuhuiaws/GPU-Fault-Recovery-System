"""The OWNERSHIP RELEASE phase of ``uninstall``: every GPU cluster the site lists.

Kept apart from ``uninstall`` so the uninstall driver stays within its size
budget; the per-cluster work is ``ownership_release.release_cluster_ownership``.
"""

from __future__ import annotations

from typing import Any

from gpu_fault.admin.bootstrap_common import CommandRunner
from gpu_fault.admin.ownership_release import (
    MANUAL,
    RELEASED,
    RETAINED,
    release_cluster_ownership,
)
from gpu_fault.admin.site import RenderedSite
from gpu_fault.admin.uninstall_types import UninstallRequest
from gpu_fault.cluster_scope import OWNERSHIP_LABEL

__all__ = ["gpu_kubectl", "ownership_summary", "release_site_ownership"]


def gpu_kubectl(site: RenderedSite, context: str) -> list[str]:
    """``kubectl`` for one GPU cluster context, through the site kubeconfig."""

    command = ["kubectl"]
    kubeconfig = site.release_config.get("gpu_kubeconfig") or site.environment.get(
        "KUBECONFIG"
    )
    if kubeconfig:
        command.extend(["--kubeconfig", str(kubeconfig)])
    command.extend(["--context", context])
    return command


def release_site_ownership(
    request: UninstallRequest, runner: CommandRunner
) -> dict[str, Any]:
    """The OWNERSHIP RELEASE phase: every GPU cluster the site still lists.

    Per cluster the evidence of ``release_cluster_ownership`` (groups with
    their label before and after, the outcome, a NodePool's manual edit) plus
    the cluster's identity, so the acceptance driver can re-apply the label to
    exactly the groups released. The phase outcome is the worst of the
    clusters: ``retained`` under ``--keep-ownership-label``, ``manual`` when a
    NodePool is left to the administrator, else ``released``.
    """

    site = request.site
    clusters: list[dict[str, Any]] = []
    for target in site.release_config["clusters"]:
        evidence = release_cluster_ownership(
            runner,
            region=str(site.release_config["aws_region"]),
            site_id=str(site.release_config["site_name"]),
            target=target,
            keep=request.keep_ownership_label,
            kubectl=gpu_kubectl(site, str(target["context"])),
        )
        clusters.append(
            {
                "cluster_id": str(target["cluster_id"]),
                "eks_cluster_arn": str(target.get("eks_cluster_arn") or ""),
                "hyperpod_cluster_name": str(target.get("hyperpod_cluster_name") or ""),
                **evidence,
            }
        )
    outcomes = {str(item["outcome"]) for item in clusters}
    outcome = RELEASED
    for candidate in (RETAINED, MANUAL):
        if candidate in outcomes:
            outcome = candidate
            break
    return {"label": OWNERSHIP_LABEL, "outcome": outcome, "clusters": clusters}


def ownership_summary(released: object) -> dict[str, Any]:
    """The ``ownership_label`` of the uninstall result: outcome per cluster."""

    if not isinstance(released, dict):
        return {"outcome": "unreleased", "label": OWNERSHIP_LABEL}
    return {
        "outcome": released.get("outcome"),
        "label": released.get("label"),
        "clusters": {
            str(item.get("cluster_id")): item.get("outcome")
            for item in released.get("clusters") or []
            if isinstance(item, dict)
        },
    }
