from __future__ import annotations

import json
from typing import Any

from regional_release_config import (
    EKS_ARN_PATTERN,
    ReleaseError,
)


def _context_eks_arn(runner: Any, kubectl: list[str], *, label: str) -> str:
    cluster = runner.run(
        kubectl
        + [
            "config",
            "view",
            "--minify",
            "-o",
            "jsonpath={.contexts[0].context.cluster}",
        ],
        capture=True,
    )
    if EKS_ARN_PATTERN.fullmatch(cluster) is None:
        raise ReleaseError(
            f"{label} kubeconfig cluster identity is not an EKS ARN; "
            "regenerate it with aws eks update-kubeconfig"
        )
    return cluster


def _validate_hyperpod_cluster(
    runner: Any,
    *,
    region: str,
    cluster_name: str,
    expected_eks_arn: str,
    require_node_recovery_none: bool,
) -> None:
    raw = runner.run(
        [
            "aws",
            "sagemaker",
            "describe-cluster",
            "--region",
            region,
            "--cluster-name",
            cluster_name,
            "--query",
            "{EksClusterArn:Orchestrator.Eks.ClusterArn,NodeRecovery:NodeRecovery}",
            "--output",
            "json",
        ],
        capture=True,
    )
    value = json.loads(raw)
    if value.get("EksClusterArn") != expected_eks_arn:
        raise ReleaseError(
            f"HyperPod cluster {cluster_name} does not target "
            f"configured EKS ARN {expected_eks_arn}"
        )
    if require_node_recovery_none and value.get("NodeRecovery") != "None":
        raise ReleaseError(
            f"HyperPod cluster {cluster_name} must set NodeRecovery=None"
        )


def ensure_region_contexts(release: Any) -> None:
    config = release.config
    runner = release.runner
    cpu_command = release._cpu
    gpu_command = release._gpu
    cpu_eks_arn = _context_eks_arn(runner, cpu_command(), label="CPU")
    if cpu_eks_arn != config.cpu_eks_arn:
        raise ReleaseError(
            "CPU kubeconfig EKS ARN does not match configured cpu_eks_arn"
        )
    _validate_hyperpod_cluster(
        runner,
        region=config.aws_region,
        cluster_name=config.cpu_hyperpod_cluster_name,
        expected_eks_arn=config.cpu_eks_arn,
        require_node_recovery_none=False,
    )
    runner.run(cpu_command("get", "--raw=/readyz"), capture=True)
    for target in config.clusters:
        gpu_eks_arn = _context_eks_arn(
            runner,
            gpu_command(target),
            label=f"GPU cluster {target.cluster_id}",
        )
        if gpu_eks_arn != target.eks_cluster_arn:
            raise ReleaseError(
                f"GPU context for {target.cluster_id} does not match "
                "configured eks_cluster_arn"
            )
        _validate_hyperpod_cluster(
            runner,
            region=target.region,
            cluster_name=target.hyperpod_cluster_name,
            expected_eks_arn=target.eks_cluster_arn,
            require_node_recovery_none=True,
        )
        runner.run(gpu_command(target, "get", "--raw=/readyz"), capture=True)
        release._validate_executor_iam_role(target)
