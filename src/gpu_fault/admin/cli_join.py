"""``join-cluster`` command surface: the parser flags and the runner.

Split out of ``cli.py`` so the plain EKS (``eks-managed``) flags have a home
that both ``join-cluster`` and ``deploy`` share without the entry module
growing past its size budget.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

from gpu_fault.admin.cluster_join import DEFAULT_ALLOWED_NAMESPACES, JoinClusterRequest
from gpu_fault.admin.site import RenderedSite, SiteConfigError

__all__ = ["add_eks_managed_arguments", "run_join_cluster"]


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


def run_join_cluster(
    arguments: argparse.Namespace,
    *,
    site_file: Path,
    load_site: Callable[..., RenderedSite],
    join_cluster: Callable[[JoinClusterRequest], dict[str, Any]],
    join_clusters: Callable[[tuple[JoinClusterRequest, ...]], dict[str, Any]],
) -> int:
    """``gpu-fault-admin join-cluster``: one or more GPU clusters, any provider.

    The collaborators are handed in by ``cli`` so that module stays the one
    place a test replaces them.
    """

    site = load_site(
        site_file,
        repository_root=arguments.repo_root,
    )
    cluster_arns = tuple(str(value) for value in arguments.gpu_cluster_arn)
    if len(cluster_arns) > 1 and arguments.cluster_id:
        raise SiteConfigError(
            "join-cluster --cluster-id cannot be shared by multiple GPU clusters"
        )
    allowed_namespaces = (
        tuple(arguments.allowed_namespace) or DEFAULT_ALLOWED_NAMESPACES
    )
    requests = tuple(
        JoinClusterRequest(
            site=site,
            gpu_cluster_arn=cluster_arn,
            cluster_id=arguments.cluster_id,
            allowed_namespaces=allowed_namespaces,
            node_selector=getattr(arguments, "node_selector", None),
            gpu_nodegroup=getattr(arguments, "gpu_nodegroup", None),
        )
        for cluster_arn in cluster_arns
    )
    result = (
        join_cluster(requests[0]) if len(requests) == 1 else join_clusters(requests)
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
