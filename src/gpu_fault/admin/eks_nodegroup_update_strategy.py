"""EKS managed node group update strategy: GPU node groups roll with ``MINIMAL``.

``updateConfig.updateStrategy`` decides how ``update-nodegroup-version`` (a
launch template or AMI roll) replaces a managed node group's nodes.
``DEFAULT`` -- also what a group without the field does -- launches the
replacement instances first and terminates the old ones after, so it needs
spare capacity of the instance type, which GPU node groups rarely have even on
demand (the live Spot p5en group failed the roll with ``NodeCreationFailure``
after about 17 minutes, two attempts of three). ``MINIMAL`` terminates first
and launches after: no extra instance, the group is short by
``maxUnavailable`` nodes while the replacements join. A single-node group has
no GPU node then, whatever ``maxUnavailable`` says (it is at least 1): only
more nodes, or a window that accepts the outage, keep the roll harmless.

``MINIMAL`` is the product standard for ``eks-managed`` GPU node groups:
``prepare-eks-nodegroup`` sets it before any roll, keeping the group's own
``maxUnavailable``/``maxUnavailablePercentage``; the ``regional_contexts``
check (deploy preflight, ``status --full``, verify) and ``node-scope`` record
every group's strategy, and ``status --full``/``node-scope`` print an advisory
for a selected GPU group that is not ``MINIMAL``. Advice only: no check fails
or warns for it, nothing refuses it. Deploy-host code: no runtime component
imports it. HyperPod instance groups are re-imaged in place by SageMaker and
have no such setting.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = [
    "UPDATE_STRATEGY_DEFAULT",
    "UPDATE_STRATEGY_MINIMAL",
    "advisory",
    "capacity_effect",
    "minimal_update_config",
    "nodes_out",
    "single_node",
    "takes_whole_group",
    "unavailable_setting",
    "unavailable_text",
    "update_strategy",
    "update_strategy_findings",
    "whole_group_remedy",
]

UPDATE_STRATEGY_DEFAULT = "DEFAULT"
UPDATE_STRATEGY_MINIMAL = "MINIMAL"
#: Where the manual command and the reason are documented.
UPDATE_STRATEGY_SECTION = "administrator operations §5.0.1"
_UNAVAILABLE_KEYS = ("maxUnavailable", "maxUnavailablePercentage")


def _count(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def update_strategy(nodegroup: Mapping[str, Any]) -> str:
    """The strategy of a DescribeNodegroup ``nodegroup``; ``DEFAULT`` when absent."""

    config = nodegroup.get("updateConfig")
    value = config.get("updateStrategy") if isinstance(config, Mapping) else None
    return str(value or UPDATE_STRATEGY_DEFAULT)


def unavailable_setting(nodegroup: Mapping[str, Any]) -> dict[str, int]:
    """``maxUnavailable``/``maxUnavailablePercentage`` exactly as the group sets them."""

    config = nodegroup.get("updateConfig")
    if not isinstance(config, Mapping):
        return {}
    return {
        key: int(config[key])
        for key in _UNAVAILABLE_KEYS
        if _count(config.get(key)) is not None
    }


def minimal_update_config(nodegroup: Mapping[str, Any]) -> dict[str, Any]:
    """The ``--update-config`` payload: the group's own unavailability, MINIMAL.

    Neither value set leaves EKS's default in force; none is invented.
    """

    return {**unavailable_setting(nodegroup), "updateStrategy": UPDATE_STRATEGY_MINIMAL}


def unavailable_text(nodegroup: Mapping[str, Any]) -> str:
    setting = unavailable_setting(nodegroup)
    return (
        ", ".join(f"{key} {value}" for key, value in setting.items())
        or "the EKS default maxUnavailable"
    )


def _desired(nodegroup: Mapping[str, Any]) -> int | None:
    scaling = nodegroup.get("scalingConfig")
    return _count(scaling.get("desiredSize")) if isinstance(scaling, Mapping) else None


def single_node(nodegroup: Mapping[str, Any]) -> bool:
    """Whether the group runs exactly one node (``desiredSize`` 1)."""

    return _desired(nodegroup) == 1


def takes_whole_group(nodegroup: Mapping[str, Any]) -> bool:
    """Whether a MINIMAL roll terminates every node of the group at once.

    Always for a single node: the roll replaces it and MINIMAL terminates
    before it launches, whatever the unavailability says. Never without nodes.
    """

    setting = unavailable_setting(nodegroup)
    desired = _desired(nodegroup)
    if desired is not None and desired <= 1:
        return desired == 1
    if "maxUnavailable" in setting:
        return desired is not None and setting["maxUnavailable"] >= desired
    return setting.get("maxUnavailablePercentage", 0) >= 100


def whole_group_remedy(nodegroup: Mapping[str, Any]) -> str:
    """How to keep GPU nodes up during a MINIMAL roll of the whole group.

    A single-node group cannot lower ``maxUnavailable`` below its size (the
    least is 1): it needs a second node first, or a window for the outage.
    """

    if single_node(nodegroup):
        return (
            "a single-node group keeps no GPU node up with any maxUnavailable: "
            "before a roll scale it to at least 2 nodes with maxUnavailable below "
            "the new size (one more GPU instance of the type; MINIMAL then keeps "
            "the others up), or roll in a maintenance window that accepts no GPU "
            "node until the replacement joins"
        )
    return (
        "lower maxUnavailable below the group size to keep GPU nodes up during a roll"
    )


def nodes_out(nodegroup: Mapping[str, Any]) -> str:
    """How many of the group's GPU nodes a MINIMAL roll takes out at a time:
    ``1 of 2 GPU nodes out at a time``."""

    setting = unavailable_setting(nodegroup)
    desired = _desired(nodegroup)
    whole = " (the whole group)" if takes_whole_group(nodegroup) else ""
    count = setting.get("maxUnavailable")
    if count is not None:
        if desired is None:
            return f"{count} GPU node{'s' * (count != 1)} out at a time"
        return (
            f"{count} of {desired} GPU node{'s' * (desired != 1)}{whole} out at a time"
        )
    group = (
        f"{desired} GPU node{'s' * (desired != 1)}"
        if desired is not None
        else "the GPU nodes"
    )
    share = (
        f"{setting['maxUnavailablePercentage']}%"
        if "maxUnavailablePercentage" in setting
        else "the EKS default maxUnavailable"
    )
    return f"{share} of {group}{whole} out at a time"


def capacity_effect(nodegroup: Mapping[str, Any], strategy: str) -> str:
    """What a roll with ``strategy`` costs the group, in one clause."""

    if strategy == UPDATE_STRATEGY_MINIMAL:
        return (
            f"a roll terminates before it launches -- {nodes_out(nodegroup)}, no "
            "extra GPU instance needed"
        )
    return (
        "a roll launches a replacement before it terminates a node -- it needs "
        "one more GPU instance of the type"
    )


def advisory(nodegroup: str, strategy: str) -> str | None:
    """The advisory line for a selected GPU node group, ``None`` when MINIMAL."""

    if strategy == UPDATE_STRATEGY_MINIMAL:
        return None
    reason = (
        "(surge): a version update needs one more GPU instance"
        if strategy == UPDATE_STRATEGY_DEFAULT
        else "instead of MINIMAL"
    )
    return (
        f"node group {nodegroup} updates with {strategy} {reason}; set "
        f"updateStrategy=MINIMAL ({UPDATE_STRATEGY_SECTION})"
    )


def _context_providers(report: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """``details.providers`` of the report's ``regional_contexts`` check."""

    health = (report or {}).get("health")
    checks = health.get("checks") if isinstance(health, Mapping) else None
    for check in checks if isinstance(checks, list) else ():
        if isinstance(check, Mapping) and check.get("name") == "regional_contexts":
            details = check.get("details")
            providers = (
                details.get("providers") if isinstance(details, Mapping) else None
            )
            return providers if isinstance(providers, Mapping) else {}
    return {}


def update_strategy_findings(report: Mapping[str, Any] | None) -> list[str]:
    """One advisory per selected GPU node group not on MINIMAL, from the
    ``regional_contexts`` check of a full status report (absent: none)."""

    lines: list[str] = []
    for cluster, entry in sorted(_context_providers(report).items()):
        groups = entry.get("node_groups") if isinstance(entry, Mapping) else None
        for name, facts in sorted(
            (groups if isinstance(groups, Mapping) else {}).items()
        ):
            if not isinstance(facts, Mapping) or not facts.get("gpu"):
                continue
            line = advisory(str(name), str(facts.get("update_strategy") or ""))
            if "update_strategy" in facts and line is not None:
                lines.append(f"{cluster}: {line}")
    return lines
