"""Shared business-workload classification for regional node preflight."""

from __future__ import annotations

from typing import Any

SYSTEM_NAMESPACES = frozenset(
    {
        "aws-hyperpod",
        "cert-manager",
        "hyperpod-inference-system",
        "kube-system",
        "kubeflow",
    }
)


def pod_gpu_count(item: dict[str, Any]) -> int:
    """The largest GPU request or limit across a Pod's containers."""
    gpu_count = 0
    for container in item.get("spec", {}).get("containers", []):
        resources = container.get("resources", {})
        for values in (resources.get("requests", {}), resources.get("limits", {})):
            try:
                gpu_count = max(gpu_count, int(values.get("nvidia.com/gpu", 0)))
            except (TypeError, ValueError):
                pass
    return gpu_count


def business_workload_items(
    items: list[dict[str, Any]], *, namespace: str
) -> list[dict[str, str]]:
    """Keep business Pods, including GPU jobs in the solution namespace."""
    result = []
    for item in items:
        pod_namespace = str(item["metadata"].get("namespace", ""))
        if pod_namespace in SYSTEM_NAMESPACES:
            continue
        if pod_namespace == namespace and pod_gpu_count(item) <= 0:
            continue
        result.append(
            {
                "namespace": pod_namespace,
                "name": str(item["metadata"].get("name", "")),
            }
        )
    return result
