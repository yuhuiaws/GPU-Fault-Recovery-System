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

from gpu_fault.cluster_scope import (
    EKS_NODEGROUP_LABEL,
    OWNERSHIP_LABEL,
    selector_matches,
    selector_ownership_value,
    selector_requirements,
)
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

    for key, operator, values in selector_requirements(selector or ""):
        if key == EKS_NODEGROUP_LABEL and operator == "in" and len(values) == 1:
            return next(iter(values))
    return ""


def nodegroup_hosts_gpu_nodes(nodegroup: Mapping[str, Any], selector: str) -> bool:
    """Whether ``selector`` (a cluster's recorded GPU selector) covers this
    DescribeNodegroup ``nodegroup``.

    A selector naming the ownership label covers the groups whose ``labels``
    carry that value; one naming a node group covers that group; any other
    (legacy) selector covers the groups running an accelerator instance type,
    which is what the retired ``nvidia.com/gpu.present=true`` default meant.
    """

    name = str(nodegroup.get("nodegroupName") or "")
    labels = {
        str(key): str(value) for key, value in (nodegroup.get("labels") or {}).items()
    }
    labels[EKS_NODEGROUP_LABEL] = name
    if selector_ownership_value(selector) is not None:
        return labels.get(OWNERSHIP_LABEL) == selector_ownership_value(
            selector
        ) and selector_matches(labels, selector)
    named = selected_nodegroup(selector)
    if named:
        return name == named
    return hosts_gpu_instance_types(nodegroup)


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
