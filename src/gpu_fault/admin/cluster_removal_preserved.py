"""Post-removal proof that the GPU cluster itself was preserved.

``remove-cluster`` detaches a GPU data plane and must leave the customer's
clusters untouched: the EKS cluster for every provider, and the HyperPod
cluster for a HyperPod one (a plain EKS ``eks-managed`` cluster has none).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from gpu_fault.admin.bootstrap_common import Arn, BootstrapError
from gpu_fault.admin.cluster_removal_network import json_command
from gpu_fault.cluster_scope import is_eks_managed

if TYPE_CHECKING:
    from gpu_fault.admin.cluster_removal import RemoveClusterRequest

__all__ = ["verify_preserved_gpu_eks", "verify_preserved_hyperpod"]


def verify_preserved_gpu_eks(
    request: RemoveClusterRequest,
    target: dict[str, Any],
) -> None:
    region = str(request.site.release_config["aws_region"])
    eks_name = Arn.parse(str(target["eks_cluster_arn"])).resource_name
    document = json_command(
        [
            "aws",
            "eks",
            "describe-cluster",
            "--region",
            region,
            "--name",
            eks_name,
            "--output",
            "json",
        ],
        description="preserved GPU EKS cluster is missing",
    )
    cluster = document.get("cluster") or {}
    if (
        cluster.get("arn") != target["eks_cluster_arn"]
        or cluster.get("status") != "ACTIVE"
        or str(cluster.get("createdAt") or "") != target.get("expected_eks_created_at")
    ):
        raise BootstrapError("preserved GPU EKS cluster identity drifted")


def verify_preserved_hyperpod(
    request: RemoveClusterRequest,
    target: dict[str, Any],
) -> None:
    if is_eks_managed(target):
        # Nothing to preserve beyond the EKS cluster, verified separately.
        return
    region = str(request.site.release_config["aws_region"])
    document = json_command(
        [
            "aws",
            "sagemaker",
            "describe-cluster",
            "--region",
            region,
            "--cluster-name",
            str(target.get("expected_hyperpod_arn") or target["hyperpod_cluster_name"]),
            "--output",
            "json",
        ],
        description="preserved GPU HyperPod cluster is missing",
    )
    if (
        not target.get("expected_hyperpod_arn")
        or document.get("ClusterArn") != target["expected_hyperpod_arn"]
        or document.get("ClusterName") != target["hyperpod_cluster_name"]
        or ((document.get("Orchestrator") or {}).get("Eks") or {}).get("ClusterArn")
        != target["eks_cluster_arn"]
        or document.get("NodeRecovery") != "None"
    ):
        raise BootstrapError(
            "preserved GPU HyperPod identity or NodeRecovery policy drifted"
        )
