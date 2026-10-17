"""Automatic node updates on a site's GPU nodes: recorded and announced, never refused.

GPU nodes the site manages are not replaced or re-imaged by automatic provider
schedules; administrators update node groups by hand in a maintenance window
(管理员日常运维 / administrator-operations §5.0.5). EKS node auto repair stays the
hard refusal it is (``eks_node_repair``). Everything else that replaces or
re-images those nodes on its own is an advisory this module derives from the
provider documents the product already reads, plus two small reads:

* HyperPod: an instance group's ``ScheduledUpdateConfig`` (a cron expression on
  which the service re-images every instance in place -- same instance id,
  root volume replaced, the Node Agent reinstalled by the node installer
  reconciler) and ``AutoPatchConfig`` (automatic security patch images on its
  patch schedule), both read off the ``describe-cluster`` document;
* eks-managed: the Auto Scaling group behind each selected managed node group
  (``describe-nodegroup`` ``resources.autoScalingGroups``, read with
  ``autoscaling:DescribeAutoScalingGroups``): a set ``MaxInstanceLifetime``
  replaces every instance at that age; ``CapacityRebalance``, which EKS
  enables on Spot node groups, is recorded only -- inherent Spot behaviour the
  replacement-node join window handles. A failed read records ``unknown``;
* Karpenter: the NodePools whose template labels carry the ownership label
  with the site's id -- ``expireAfter`` (Karpenter's default is 720h; expiry
  ignores disruption budgets) and the disruption budgets, which drift and
  consolidation honour.

Nothing here changes a check verdict, ``healthy``, a refusal or an exit code:
the facts sit next to the ones they extend (the ``regional_contexts`` check of
``status --full`` and deploy preflight, ``node-scope``) and the advisory lines
go to the ``status`` header, ``node-scope`` and join/deploy discovery's
stderr. The module reads nothing itself; callers hand in the documents, the
selected group names and the read callables.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from typing import Any

from gpu_fault.cluster_scope import OWNERSHIP_LABEL

__all__ = [
    "ASG_QUERY",
    "FIELD",
    "HELPER_VERB",
    "KARPENTER_NODEPOOLS",
    "UNKNOWN",
    "advisory_lines",
    "announce",
    "eks_node_group_updates",
    "guarded",
    "hyperpod_group_updates",
    "hyperpod_updates",
    "karpenter_updates",
    "nodegroup_asg_names",
    "nodepool_updates",
    "zero_budget",
]

#: The key the advisory block sits under, beside the facts it extends.
FIELD = "automatic_node_updates"
UNKNOWN = "unknown"
#: The read-only verb that prints the command disabling a HyperPod schedule.
HELPER_VERB = "hyperpod-scheduled-update"
SECTION = "administrator-operations §5.0.5"
KARPENTER_NODEPOOLS = "nodepools.karpenter.sh"
KARPENTER_DEFAULT_EXPIRE_AFTER = "720h"
KARPENTER_NEVER = "Never"
#: The reasons a Karpenter budget can name; a budget naming fewer blocks fewer.
KARPENTER_REASONS = frozenset({"Drifted", "Underutilized", "Empty"})
#: ``describe-auto-scaling-groups`` reduced to the two advisory fields (the
#: instance list stays out of the output); a JMESPath object, so the answer is
#: still one JSON object.
ASG_QUERY = (
    "{AutoScalingGroups: AutoScalingGroups[].{AutoScalingGroupName: "
    "AutoScalingGroupName, MaxInstanceLifetime: MaxInstanceLifetime, "
    "CapacityRebalance: CapacityRebalance}}"
)
_MAX_REASON = 300

AsgReader = Callable[[Sequence[str]], Mapping[str, Any]]
NodePoolReader = Callable[[], Mapping[str, Any] | None]


def _reason(exc: BaseException) -> str:
    """One line naming why an advisory read failed."""

    text = re.sub(r"\s+", " ", str(exc)).strip() or type(exc).__name__
    return text if len(text) <= _MAX_REASON else text[: _MAX_REASON - 3] + "..."


def guarded(build: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """``build()``, or ``{"state": "unknown", "error": ...}``: the advisory block
    of a caller whose verdict must not depend on it (a check, ``node-scope``)."""

    try:
        return build()
    except Exception as exc:  # noqa: BLE001 - advisory: recorded, never raised
        return {"state": UNKNOWN, "error": _reason(exc)}


# --- HyperPod -------------------------------------------------------------------


def hyperpod_group_updates(group: Mapping[str, Any]) -> dict[str, Any]:
    """``name``, ``scheduled_update`` (the cron expression) and ``auto_patch``
    (the patching strategy) of one ``describe-cluster`` ``InstanceGroups[]``
    entry; ``None`` where the group carries no such configuration."""

    schedule = group.get("ScheduledUpdateConfig")
    expression = (
        schedule.get("ScheduleExpression") if isinstance(schedule, Mapping) else None
    )
    patch = group.get("AutoPatchConfig")
    return {
        "name": str(group.get("InstanceGroupName") or ""),
        "scheduled_update": str(expression) if expression else None,
        "auto_patch": (
            str(patch.get("PatchingStrategy") or "configured")
            if isinstance(patch, Mapping)
            else None
        ),
    }


def hyperpod_updates(
    groups: Iterable[Any], *, selected: Collection[str], gpu_cluster_arn: str = ""
) -> dict[str, Any]:
    """The advisory block of a HyperPod cluster: the ``selected`` instance groups.

    ``gpu_cluster_arn`` is what the advisory line hands to the helper verb.
    """

    block: dict[str, Any] = {
        "instance_groups": [
            hyperpod_group_updates(group)
            for group in groups
            if isinstance(group, Mapping)
            and str(group.get("InstanceGroupName") or "") in selected
        ]
    }
    if gpu_cluster_arn:
        block["gpu_cluster_arn"] = gpu_cluster_arn
    return block


# --- eks-managed: the Auto Scaling groups ----------------------------------------


def nodegroup_asg_names(nodegroup: Mapping[str, Any]) -> list[str]:
    """The Auto Scaling groups ``describe-nodegroup`` names in ``resources``."""

    resources = nodegroup.get("resources")
    items = resources.get("autoScalingGroups") if isinstance(resources, Mapping) else ()
    return [
        str(item["name"])
        for item in items or ()
        if isinstance(item, Mapping) and item.get("name")
    ]


def _asg_facts(item: Mapping[str, Any]) -> dict[str, Any]:
    lifetime = item.get("MaxInstanceLifetime")
    return {
        # Unset is null; 0 is how a lifetime is cleared.
        "max_instance_lifetime": lifetime
        if type(lifetime) is int and lifetime
        else None,
        "capacity_rebalance": item.get("CapacityRebalance") is True,
    }


def eks_node_group_updates(
    nodegroups: Mapping[str, Mapping[str, Any]],
    *,
    selected: Iterable[str],
    read_asgs: AsgReader,
) -> list[dict[str, Any]]:
    """Per selected node group, its Auto Scaling groups' lifetime and rebalance.

    One read covers every group; a read that fails (most often an operator
    identity without ``autoscaling:DescribeAutoScalingGroups``) or an Auto
    Scaling group it does not return records ``unknown`` with the reason --
    an advisory read never fails its caller.
    """

    names = {
        group: nodegroup_asg_names(nodegroups[group])
        for group in sorted(set(selected))
        if group in nodegroups
    }
    wanted = sorted({asg for asgs in names.values() for asg in asgs})
    found: dict[str, dict[str, Any]] = {}
    error = "not returned by DescribeAutoScalingGroups"
    if wanted:
        try:
            response = read_asgs(wanted)
            for item in response.get("AutoScalingGroups") or ():
                if isinstance(item, Mapping) and item.get("AutoScalingGroupName"):
                    found[str(item["AutoScalingGroupName"])] = _asg_facts(item)
        except Exception as exc:  # noqa: BLE001 - advisory: recorded, never raised
            error = _reason(exc)
    unknown = {
        "max_instance_lifetime": UNKNOWN,
        "capacity_rebalance": UNKNOWN,
        "error": error,
    }
    return [
        {
            "name": group,
            "auto_scaling_groups": [
                {"name": asg, **found.get(asg, unknown)} for asg in asgs
            ],
        }
        for group, asgs in names.items()
    ]


# --- Karpenter --------------------------------------------------------------------


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _mappings(value: object) -> list[Mapping[str, Any]]:
    """The mapping items of a list (a report read back is not trusted blindly)."""

    return (
        [item for item in value if isinstance(item, Mapping)]
        if isinstance(value, list)
        else []
    )


def zero_budget(budgets: object) -> bool:
    """Whether one disruption budget allows no node at all for every reason.

    ``nodes: "0"`` (or ``"0%"``), scheduled or not -- a maintenance-window
    budget is a zero budget with a schedule -- naming no ``reasons`` or all of
    them. Absent budgets are Karpenter's default ``nodes: 10%``.
    """

    for budget in budgets if isinstance(budgets, list) else ():
        if not isinstance(budget, Mapping):
            continue
        reasons = budget.get("reasons")
        if str(budget.get("nodes", "")).strip() in {"0", "0%"} and (
            not reasons or KARPENTER_REASONS <= {str(item) for item in reasons}
        ):
            return True
    return False


def nodepool_updates(
    document: Mapping[str, Any], *, site_id: str
) -> list[dict[str, Any]]:
    """The NodePools of a ``kubectl get nodepools.karpenter.sh -o json`` list
    whose template labels carry ``OWNERSHIP_LABEL=<site_id>``.

    ``expire_after`` is ``spec.template.spec.expireAfter`` (Karpenter v1; the
    v1beta1 ``spec.disruption.expireAfter`` when absent), ``None`` when neither
    is set (Karpenter's default applies); ``budgets`` is
    ``spec.disruption.budgets`` as written.
    """

    found = []
    for item in document.get("items") or ():
        if not isinstance(item, Mapping):
            continue
        spec = _mapping(item.get("spec"))
        template = _mapping(spec.get("template"))
        labels = _mapping(_mapping(template.get("metadata")).get("labels"))
        if str(labels.get(OWNERSHIP_LABEL) or "") != site_id:
            continue
        disruption = _mapping(spec.get("disruption"))
        expire = _mapping(template.get("spec")).get("expireAfter")
        if expire is None:
            expire = disruption.get("expireAfter")
        budgets = disruption.get("budgets")
        found.append(
            {
                "name": str(_mapping(item.get("metadata")).get("name") or ""),
                "expire_after": None if expire is None else str(expire),
                "budgets": budgets if isinstance(budgets, list) else None,
                "zero_budget": zero_budget(budgets),
            }
        )
    return sorted(found, key=lambda pool: pool["name"])


def karpenter_updates(read: NodePoolReader, *, site_id: str) -> dict[str, Any]:
    """``state`` ``read`` / ``absent`` (no Karpenter CRD: ``read`` returned
    ``None``) / ``unknown`` (the read failed), with the site's NodePools."""

    try:
        document = read()
    except Exception as exc:  # noqa: BLE001 - advisory: recorded, never raised
        return {"state": UNKNOWN, "nodepools": [], "error": _reason(exc)}
    if document is None:
        return {"state": "absent", "nodepools": []}
    return {"state": "read", "nodepools": nodepool_updates(document, site_id=site_id)}


# --- advisory lines -----------------------------------------------------------------


def _hyperpod_lines(where: str, block: Mapping[str, Any]) -> list[str]:
    arn = str(block.get("gpu_cluster_arn") or "<GPU cluster ARN>")
    helper = (
        f"gpu-fault-admin {HELPER_VERB} --state-dir <state dir> "
        f"--gpu-cluster-arn {arn} prints the update-cluster command that removes it"
    )
    lines = []
    for group in _mappings(block.get("instance_groups")):
        name = group.get("name")
        if group.get("scheduled_update"):
            lines.append(
                f"advisory: HyperPod instance group {name} of {where} has "
                f"ScheduledUpdateConfig {group['scheduled_update']}: the service "
                "re-images every instance of the group on that schedule; "
                f"{helper}; update the group by hand in a maintenance window "
                f"({SECTION})"
            )
        if group.get("auto_patch"):
            lines.append(
                f"advisory: HyperPod instance group {name} of {where} has "
                f"AutoPatchConfig (PatchingStrategy {group['auto_patch']}): the "
                "service cordons and re-images its instances on the patch "
                f"schedule; {helper} ({SECTION})"
            )
    return lines


def _eks_lines(where: str, block: Mapping[str, Any]) -> list[str]:
    lines = []
    for group in _mappings(block.get("node_groups")):
        for asg in _mappings(group.get("auto_scaling_groups")):
            lifetime = asg.get("max_instance_lifetime")
            if type(lifetime) is int and lifetime:
                lines.append(
                    f"advisory: Auto Scaling group {asg.get('name')} of EKS node "
                    f"group {group.get('name')} ({where}) sets MaxInstanceLifetime "
                    f"{lifetime} s: EC2 Auto Scaling replaces every instance at "
                    "that age; clear it with aws autoscaling "
                    "update-auto-scaling-group --auto-scaling-group-name "
                    f"{asg.get('name')} --max-instance-lifetime 0 ({SECTION})"
                )
    for pool in _mappings(_mapping(block.get("karpenter")).get("nodepools")):
        expire = pool.get("expire_after")
        problems = []
        if expire != KARPENTER_NEVER:
            problems.append(
                f"expireAfter {expire}"
                if expire is not None
                else f"no expireAfter (Karpenter's default {KARPENTER_DEFAULT_EXPIRE_AFTER})"
            )
        if not pool.get("zero_budget"):
            problems.append(
                "no zero disruption budget (budgets "
                + json.dumps(pool.get("budgets"), sort_keys=True, separators=(",", ":"))
                + ")"
            )
        if problems:
            lines.append(
                f"advisory: Karpenter NodePool {pool.get('name')} of {where} "
                f"provisions site GPU nodes with {' and '.join(problems)}: "
                "Karpenter replaces them on expiry, drift and consolidation; set "
                "spec.template.spec.expireAfter: Never and a disruption budget "
                'nodes: "0" (or a maintenance-window schedule budget) '
                f"({SECTION})"
            )
    return lines


def advisory_lines(where: str, block: object) -> list[str]:
    """The advisory lines of one cluster's block: a scheduled or auto-patched
    HyperPod instance group, a set ``MaxInstanceLifetime``, a site NodePool
    with a finite ``expireAfter`` or no zero budget. ``unknown`` and
    ``CapacityRebalance`` are recorded, never announced."""

    if not isinstance(block, Mapping):
        return []
    return [*_hyperpod_lines(where, block), *_eks_lines(where, block)]


def announce(where: str, block: object, *, prefix: str = "gpu-fault-admin") -> None:
    """Print ``advisory_lines`` on stderr (discovery and ``node-scope``)."""

    for line in advisory_lines(where, block):
        print(f"{prefix}: {line}", file=sys.stderr, flush=True)
