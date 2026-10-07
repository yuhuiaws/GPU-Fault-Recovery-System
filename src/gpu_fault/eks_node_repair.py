"""EKS managed node group auto repair: plain EKS's counterpart of HyperPod ``NodeRecovery``.

A node group with ``nodeRepairConfig.enabled: true`` lets EKS reboot or replace
an unhealthy node on its own, which would race this solution's RESTART_NODE /
warm-spare REPLACE_NODE exactly as HyperPod ``NodeRecovery=Automatic`` would.
The feature is opt-in: a group without the field (``null``) has it off, which
is recorded as ``unset`` rather than invented as ``disabled``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from gpu_fault.cluster_scope import EKS_NODEGROUP_LABEL
from gpu_fault.collectors.gpu.discovery import expected_accelerator_counts

NODE_REPAIR_DISABLED = "disabled"
NODE_REPAIR_UNSET = "unset"
NODE_REPAIR_ENABLED = "enabled"


def node_repair_state(nodegroup: Mapping[str, Any]) -> str:
    """``enabled`` / ``disabled`` / ``unset`` from a DescribeNodegroup ``nodegroup``."""

    config = nodegroup.get("nodeRepairConfig")
    if not isinstance(config, Mapping) or config.get("enabled") is None:
        return NODE_REPAIR_UNSET
    return (
        NODE_REPAIR_ENABLED if config.get("enabled") is True else NODE_REPAIR_DISABLED
    )


def aggregate_node_repair(states: Iterable[str]) -> str:
    """One state for a set of node groups: enabled if any is, else unset if any is."""

    values = set(states)
    if NODE_REPAIR_ENABLED in values:
        return NODE_REPAIR_ENABLED
    if NODE_REPAIR_UNSET in values or not values:
        return NODE_REPAIR_UNSET
    return NODE_REPAIR_DISABLED


def node_repair_error(nodegroup_name: str, eks_name: str, state: str) -> str | None:
    """The fail-closed reason when ``state`` is enabled, in the words of the
    HyperPod ``NodeRecovery=None`` guard; ``None`` otherwise."""

    if state != NODE_REPAIR_ENABLED:
        return None
    return (
        f"EKS node group {nodegroup_name} of {eks_name} has node auto repair "
        "enabled (nodeRepairConfig.enabled=true); the GPU node groups of an "
        "eks-managed cluster must set it false -- as a HyperPod cluster must set "
        "NodeRecovery=None -- or it races this solution's own recovery"
    )


def selected_nodegroup(selector: str) -> str:
    """The node group an ``eks.amazonaws.com/nodegroup=<name>`` selector names."""

    for term in (selector or "").split(","):
        key, separator, value = term.strip().partition("=")
        if separator and key.strip().rstrip("=") == EKS_NODEGROUP_LABEL:
            return value.strip()
    return ""


def hosts_gpu_instance_types(nodegroup: Mapping[str, Any]) -> bool:
    """Whether a DescribeNodegroup ``nodegroup`` runs a known accelerator type
    (``instanceTypes``, or the launch template's ``instanceType``)."""

    types = [str(item) for item in nodegroup.get("instanceTypes") or () if item]
    if nodegroup.get("instanceType"):
        types.append(str(nodegroup["instanceType"]))
    return any(
        expected_accelerator_counts(item.removeprefix("ml.")) is not None
        for item in types
    )
