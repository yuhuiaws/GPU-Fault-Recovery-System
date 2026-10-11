"""``join-cluster`` command surface: the parser flags and the runner.

Split out of ``cli.py`` so the plain EKS (``eks-managed``) flags have a home
that both ``join-cluster`` and ``deploy`` share without the entry module
growing past its size budget.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

from gpu_fault.admin.cluster_join import JoinClusterRequest
from gpu_fault.admin.cluster_join_types import site_allowed_namespaces
from gpu_fault.admin.profile_approval import approve_profile_plan_inline
from gpu_fault.admin.site import RenderedSite, SiteConfigError

__all__ = [
    "add_eks_managed_arguments",
    "add_join_recovery_arguments",
    "run_join_cluster",
]


def add_join_recovery_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags that re-attach a cluster the site lost; ``join-cluster`` only."""

    parser.add_argument(
        "--reclaim-cluster-token",
        action="store_true",
        help=(
            "re-attach one cluster whose site membership was lost (a rollback "
            "rebuilt the site from a stale baseline, or the site was edited) by "
            "adopting its existing token file; refused unless a committed join "
            "journal records the token for this cluster, the cluster's data plane "
            "holds the same token and the control plane still records the cluster "
            "id. Takes a single --gpu-cluster-arn"
        ),
    )


def add_eks_managed_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags that only a plain EKS (``eks-managed``) GPU cluster reads."""

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--node-selector",
        metavar="SELECTOR",
        help=(
            "GPU node label selector of a plain EKS cluster (no HyperPod "
            "orchestrator); default nvidia.com/gpu.present=true; ignored for "
            "HyperPod clusters, whose selector is the HyperPod cluster label"
        ),
    )
    group.add_argument(
        "--gpu-nodegroup",
        metavar="NODEGROUP",
        help=(
            "EKS managed node group whose nodes are the GPU nodes of a plain EKS "
            "cluster (selects eks.amazonaws.com/nodegroup=NODEGROUP)"
        ),
    )
    parser.add_argument(
        "--approve-profile-plan",
        metavar="PLAN_SHA256",
        help=(
            "approve the Runtime Profile plan a previous join stopped for (the "
            "joining cluster needs its own profile version) and continue; "
            "requires --reference"
        ),
    )
    parser.add_argument(
        "--reference",
        metavar="REFERENCE",
        help="change request reference recorded with --approve-profile-plan",
    )


def run_join_cluster(
    arguments: argparse.Namespace,
    *,
    site_file: Path,
    load_site: Callable[..., RenderedSite],
    join_cluster: Callable[[JoinClusterRequest], dict[str, Any]],
    join_clusters: Callable[[tuple[JoinClusterRequest, ...]], dict[str, Any]],
    approve_profile_plan: Callable[..., Mapping[str, Any]] | None = None,
) -> int:
    """``gpu-fault-admin join-cluster``: one or more GPU clusters, any provider.

    The collaborators are handed in by ``cli`` so that module stays the one
    place a test replaces them. ``--approve-profile-plan SHA --reference REF``
    approves the pending Runtime Profile plan (written when the joining
    cluster needs its own profile version) before the join resumes, exactly
    as ``deploy`` does.
    """

    plan_sha256 = getattr(arguments, "approve_profile_plan", None)
    reference = getattr(arguments, "reference", None)
    if plan_sha256 and not reference:
        raise SiteConfigError("--approve-profile-plan requires --reference")
    if reference and not plan_sha256:
        raise SiteConfigError("--reference is only used with --approve-profile-plan")
    if plan_sha256:
        approve = approve_profile_plan or approve_profile_plan_inline
        record = approve(
            site_file.parent, plan_sha256=str(plan_sha256), reference=str(reference)
        )
        print(
            f"gpu-fault-admin: Runtime Profile plan {record['plan_sha256']} approved "
            f"(reference {record['reference']}); continuing the join",
            file=sys.stderr,
            flush=True,
        )

    site = load_site(
        site_file,
        repository_root=arguments.repo_root,
    )
    cluster_arns = tuple(str(value) for value in arguments.gpu_cluster_arn)
    if len(cluster_arns) > 1 and arguments.cluster_id:
        raise SiteConfigError(
            "join-cluster --cluster-id cannot be shared by multiple GPU clusters"
        )
    reclaim_cluster_token = bool(getattr(arguments, "reclaim_cluster_token", False))
    if reclaim_cluster_token and len(cluster_arns) > 1:
        raise SiteConfigError(
            "--reclaim-cluster-token re-attaches one cluster; pass a single "
            "--gpu-cluster-arn"
        )
    allowed_namespaces = tuple(arguments.allowed_namespace) or site_allowed_namespaces(
        site
    )
    requests = tuple(
        JoinClusterRequest(
            site=site,
            gpu_cluster_arn=cluster_arn,
            cluster_id=arguments.cluster_id,
            allowed_namespaces=allowed_namespaces,
            node_selector=getattr(arguments, "node_selector", None),
            gpu_nodegroup=getattr(arguments, "gpu_nodegroup", None),
            reclaim_cluster_token=reclaim_cluster_token,
        )
        for cluster_arn in cluster_arns
    )
    result = (
        join_cluster(requests[0]) if len(requests) == 1 else join_clusters(requests)
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
