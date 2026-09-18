"""Complete GPU membership projection for the warm-spare guard audit."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

GPU_RESOURCE = "nvidia.com/gpu"


def gpu_allocatable(node: dict[str, Any]) -> int | None:
    status = node.get("status", {})
    if not isinstance(status, dict):
        raise RuntimeError("node resource status is not an object")
    declared: dict[str, int] = {}
    for field in ("capacity", "allocatable"):
        resources = status.get(field, {})
        if not isinstance(resources, dict):
            raise RuntimeError(f"node status.{field} is not a resource map")
        if GPU_RESOURCE not in resources:
            continue
        value = resources[GPU_RESOURCE]
        if type(value) is int and value >= 0:
            declared[field] = value
        elif isinstance(value, str) and value.isascii() and value.isdecimal():
            declared[field] = int(value)
        else:
            raise RuntimeError(f"GPU resource quantity in status.{field} is invalid")
    if not declared:
        return None
    if "allocatable" not in declared:
        raise RuntimeError("GPU node has no allocatable resource declaration")
    return declared["allocatable"]


def project_gpu_nodes(
    document: object, *, ownership_annotations: Sequence[str]
) -> list[dict[str, Any]]:
    if not isinstance(document, dict) or not isinstance(document.get("items"), list):
        raise RuntimeError("node inventory is not a complete object list")
    metadata = document.get("metadata", {})
    if not isinstance(metadata, dict) or metadata.get("continue"):
        raise RuntimeError("node inventory pagination is incomplete")
    nodes: list[dict[str, Any]] = []
    for item in document["items"]:
        if not isinstance(item, dict):
            raise RuntimeError("node inventory contains a non-object")
        allocatable = gpu_allocatable(item)
        # A GPU resource declaration remains membership evidence when its value is zero.
        if allocatable is None:
            continue
        metadata = item.get("metadata", {})
        annotations = metadata.get("annotations", {})
        taints = sorted(
            (
                {
                    key: taint[key]
                    for key in ("key", "value", "effect", "timeAdded")
                    if key in taint
                }
                for taint in item.get("spec", {}).get("taints", [])
            ),
            key=lambda taint: (
                str(taint.get("key", "")),
                str(taint.get("value", "")),
                str(taint.get("effect", "")),
            ),
        )
        ready = next(
            (
                condition.get("status")
                for condition in item.get("status", {}).get("conditions", [])
                if condition.get("type") == "Ready"
            ),
            None,
        )
        nodes.append(
            {
                "name": metadata.get("name"),
                "uid": metadata.get("uid"),
                "ready": ready,
                "gpu_allocatable": allocatable,
                "unschedulable": bool(item.get("spec", {}).get("unschedulable", False)),
                "taints": taints,
                "ownership_annotations": {
                    key: annotations.get(key) for key in ownership_annotations
                },
            }
        )
    return sorted(nodes, key=lambda node: str(node["name"]))
