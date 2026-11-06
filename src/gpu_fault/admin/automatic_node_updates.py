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

The site's CPU control-plane cluster is covered too (``control_plane_updates``):
every instance group of its HyperPod cluster -- the control-plane Pods carry
no node selector, so any CPU node hosts them -- and, read where the
``regional_contexts`` check runs, the Auto Scaling groups of the managed node
groups its EKS cluster may also have. A schedule that re-images every
control-plane instance in one window takes the whole control plane down for
minutes (live 2026-11-01: the CPU group's ``cron(0 8 ? * SUN *)`` took all
three nodes NotReady at once, the api-ha and control-worker Pods with them,
and the Pods could not come back until Kueue's Pod webhook answered again);
the advisory line says so in those words. A rolling schedule's line names its
rollback batch and alarms; whether they can roll an update back (present, OK,
on the current API NLB) is the ``cpu_rolling_update`` check of ``status
--full`` (``gpu_fault_release.regional_cpu_rolling_update``).

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

from gpu_fault.admin.hyperpod_rollback import rollback_alarm_names
from gpu_fault.cluster_scope import OWNERSHIP_LABEL

__all__ = [
    "ASG_QUERY",
    "CONTROL_PLANE",
    "FIELD",
    "DEPLOY_STEP",
    "HELPER_VERB",
    "HYPERPOD_UPDATE_EVIDENCE",
    "KARPENTER_NODEPOOLS",
    "UNKNOWN",
    "advisory_lines",
    "announce",
    "announce_control_plane",
    "control_plane_lines",
    "control_plane_updates",
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
#: The ``regional_contexts`` facts key of the CPU control-plane cluster.
CONTROL_PLANE = "control_plane"
UNKNOWN = "unknown"
#: The read-only verb that prints the command disabling a HyperPod schedule.
HELPER_VERB = "hyperpod-scheduled-update"
SECTION = "administrator-operations §5.0.5"
#: What fixes an all-at-once CPU schedule without a hand-made request (batch 240).
DEPLOY_STEP = (
    "with spec.cpuControlPlane.rollingUpdate mode rolling (the default) the "
    "deploy's last step makes it roll one instance at a time, rolled back on the "
    "product's two alarms, and gpu-fault-admin cpu-rolling-update --state-dir "
    "<state dir> runs that step now"
)
#: What an update of the HyperPod platform software does to the nodes it
#: updates -- a ``ScheduledUpdateConfig`` applies the same update on its cron:
#: AWS, "Updating the SageMaker HyperPod platform software", the Important box.
#: That a schedule without a ``DeploymentConfig`` updates every instance of the
#: group at once is observed (2026-11-01), not documented.
HYPERPOD_UPDATE_EVIDENCE: dict[str, Any] = {
    "vendor": "AWS",
    "url": (
        "https://docs.aws.amazon.com/sagemaker/latest/dg/"
        "sagemaker-hyperpod-eks-operate-cli-command-update-cluster-software.html"
    ),
    "page": "Updating the SageMaker HyperPod platform software",
    "section": "Important",
    "quotes": (
        "When this API is called, SageMaker HyperPod doesn\u2019t drain or "
        "redistribute the jobs (Pods) running on the nodes.",
        "The patching process replaces the root volume with the updated AMI, which "
        "means that your previous data stored in the instance root volume will be "
        "lost.",
        "All cluster nodes experience downtime (nodes appear as `<NotReady>` in the "
        "output of `kubectl get node`) while the patching is in progress.",
    ),
}
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


# --- the CPU control-plane cluster ---------------------------------------------------


def _batch(size: object) -> dict[str, Any] | None:
    kind, value = _mapping(size).get("Type"), _mapping(size).get("Value")
    if not isinstance(kind, str) or type(value) is not int:
        return None
    return {"type": kind, "value": value}


def rolling_batch(schedule: object) -> dict[str, Any] | None:
    """``{"type", "value", "wait_seconds", "rollback_batch", "rollback_alarms"}``
    of a ``ScheduledUpdateConfig``'s ``DeploymentConfig``:
    ``RollingUpdatePolicy.MaximumBatchSize``, ``WaitIntervalInSeconds``
    (``None`` when unset), ``RollbackMaximumBatchSize`` (``{"type", "value"}``
    or ``None``) and the ``AutoRollbackConfiguration`` alarm names. ``None``
    without a batch -- the service was then observed to re-image every instance
    of the group at once (2026-11-01: all three control-plane nodes NotReady
    together); no AWS document states that default."""

    deployment = _mapping(_mapping(schedule).get("DeploymentConfig"))
    policy = _mapping(deployment.get("RollingUpdatePolicy"))
    batch = _batch(policy.get("MaximumBatchSize"))
    if batch is None:
        return None
    wait = deployment.get("WaitIntervalInSeconds")
    return {
        **batch,
        "wait_seconds": wait if type(wait) is int else None,
        "rollback_batch": _batch(policy.get("RollbackMaximumBatchSize")),
        "rollback_alarms": rollback_alarm_names(schedule),
    }


def control_plane_updates(
    groups: Iterable[Any], *, hyperpod_cluster_name: str, cpu_eks_arn: str
) -> dict[str, Any]:
    """The advisory block of the site's CPU control-plane HyperPod cluster.

    Every instance group, not a selection: the control-plane Pods carry no
    node selector, so any CPU node hosts them. Per group the HyperPod facts
    (``hyperpod_group_updates``) plus ``instances`` (``CurrentCount``, ``None``
    when absent) and ``rolling_batch``. ``cpu_eks_arn`` is what the advisory
    line hands to the helper verb, which accepts the EKS ARN of any HyperPod
    cluster.
    """

    instance_groups = []
    for group in groups:
        if not isinstance(group, Mapping):
            continue
        count = group.get("CurrentCount")
        instance_groups.append(
            {
                **hyperpod_group_updates(group),
                "instances": count if type(count) is int else None,
                "rolling_batch": rolling_batch(group.get("ScheduledUpdateConfig")),
            }
        )
    return {
        "hyperpod_cluster_name": hyperpod_cluster_name,
        "cpu_eks_arn": cpu_eks_arn,
        "instance_groups": instance_groups,
    }


def announce_control_plane(
    hyperpod: Mapping[str, Any], *, prefix: str = "gpu-fault-admin"
) -> None:
    """Discovery's advisory for the CPU control-plane cluster, from the
    ``describe-cluster`` document discovery already read; never raises."""

    try:
        eks = _mapping(_mapping(hyperpod.get("Orchestrator")).get("Eks"))
        block = control_plane_updates(
            hyperpod.get("InstanceGroups") or (),
            hyperpod_cluster_name=str(hyperpod.get("ClusterName") or ""),
            cpu_eks_arn=str(eks.get("ClusterArn") or ""),
        )
        for line in control_plane_lines(block):
            print(f"{prefix}: {line}", file=sys.stderr, flush=True)
    except Exception:  # noqa: BLE001 - an advisory never fails discovery
        return


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


def _all_at_once(group: Mapping[str, Any]) -> bool:
    """Whether a scheduled update re-images every instance of ``group`` together."""

    batch = _mapping(group.get("rolling_batch"))
    value, instances = batch.get("value"), group.get("instances")
    if not batch or type(value) is not int:
        return True
    if batch.get("type") == "CAPACITY_PERCENTAGE":
        return value >= 100
    return type(instances) is int and value >= instances


def _size(batch: Mapping[str, Any]) -> str:
    value = batch.get("value")
    return (
        f"{value}%"
        if batch.get("type") == "CAPACITY_PERCENTAGE"
        else f"{value} instance(s)"
    )


def _rollback(batch: Mapping[str, Any]) -> str:
    """How a rolling batch is rolled back: its ``RollbackMaximumBatchSize`` and
    the ``AutoRollbackConfiguration`` alarms (the service requires both on a
    rolling schedule written through UpdateCluster, ``hyperpod_rollback``)."""

    alarms = [str(item) for item in batch.get("rollback_alarms") or ()]
    if not alarms:
        return (
            ", with no rollback alarm (AutoRollbackConfiguration): nothing rolls it "
            "back when the control plane loses capacity"
        )
    back = _mapping(batch.get("rollback_batch"))
    how = (
        f" {_size(back)} at a time"
        if back
        else " (no RollbackMaximumBatchSize: in the service's default batches)"
    )
    return f", rolled back{how} on alarm {' or '.join(alarms)}"


def _control_plane_window(group: Mapping[str, Any], *, whole: bool) -> tuple[str, bool]:
    """``(what one scheduled update window does to the control plane on
    ``group``, whether that is the safe form)``.

    The safe form is a rolling batch smaller than the group: the control plane
    keeps serving, with a residual named. No batch, or one that covers every
    instance, takes the control plane on the group down at once -- the whole of
    it when the group holds every CPU node.
    """

    batch = _mapping(group.get("rolling_batch"))
    count = group.get("instances")
    instances = f"all {count} instances" if type(count) is int else "every instance"
    percentage = batch.get("type") == "CAPACITY_PERCENTAGE"
    if not _all_at_once(group):
        wait = batch.get("wait_seconds")
        return (
            f"rolling {_size(batch)} at a time"
            + (f" {wait} s apart" if type(wait) is int else "")
            + _rollback(batch)
            + ": the safe form -- the control plane keeps serving from the other "
            "nodes; residual: the control-plane Pods on each batch go down with "
            "it, and the single-replica ADOT and Kueue (when present) are down "
            "while their node is re-imaged, so a Pod created meanwhile waits for "
            "Kueue's webhook",
            True,
        )
    covered = (
        "with no rolling DeploymentConfig: observed to re-image "
        f"{instances} of the group at once"
        if not batch
        else f"with a rolling batch of {batch.get('value')}"
        + ("%" if percentage else " instance(s)")
        + f", which covers {instances} of the group: the service re-images them "
        "at once"
    )
    return (
        covered
        + (
            " -- every node the control plane runs on, so the whole control "
            "plane (api-ha, control-worker) is down for minutes each time"
            if whole
            else ", so the control-plane Pods on them (api-ha, control-worker) "
            "are down for minutes each time"
        ),
        False,
    )


def control_plane_lines(block: object) -> list[str]:
    """The advisory lines of the CPU control-plane cluster's block
    (``control_plane_updates``): a scheduled or auto-patched instance group --
    with what one window does to the control plane, naming a rolling
    schedule's rollback batch and alarms -- and a managed node group's Auto
    Scaling group with a set ``MaxInstanceLifetime``. ``unknown`` is recorded,
    never announced. Whether the alarms can trip is the ``cpu_rolling_update``
    check's (``status --full``)."""

    if not isinstance(block, Mapping):
        return []
    name = block.get("hyperpod_cluster_name") or "<CPU HyperPod cluster>"
    where = f"the CPU control-plane cluster {name}"
    arn = str(block.get("cpu_eks_arn") or "<CPU EKS ARN>")
    groups = _mappings(block.get("instance_groups"))
    node_groups = _mappings(_mapping(block.get("eks_node_groups")).get("node_groups"))
    # The group is every control-plane node when no other group or managed node
    # group has instances to host the Pods.
    hosts = [group for group in groups if group.get("instances") != 0]
    lines = []
    for group in groups:
        group_name = group.get("name")
        command = (
            f"gpu-fault-admin {HELPER_VERB} --state-dir <state dir> --gpu-cluster-arn "
            f"{arn} --instance-group {group_name}"
        )
        helper = f"{command} prints the update-cluster command that removes it"
        if group.get("scheduled_update"):
            window, safe = _control_plane_window(
                group, whole=hosts == [group] and not node_groups
            )
            lines.append(
                f"advisory: HyperPod instance group {group_name} of {where} has "
                f"ScheduledUpdateConfig {group['scheduled_update']} {window}"
                + (
                    ""
                    if safe
                    else f"; {DEPLOY_STEP}; by hand, {command} --rolling "
                    "--rollback-alarm <alarm> prints the update-cluster commands "
                    "(delete the schedule, then create it again) that make it roll "
                    "one instance at a time, rolled back on that alarm (without "
                    "--rolling, the one that removes it: then update the group by "
                    "hand in a maintenance window, one instance at a time)"
                )
                + f" ({SECTION})"
            )
        if group.get("auto_patch"):
            lines.append(
                f"advisory: HyperPod instance group {group_name} of {where} has "
                f"AutoPatchConfig (PatchingStrategy {group['auto_patch']}): the "
                "service cordons and re-images its instances on the patch "
                "schedule, and the control-plane Pods on them go down with them; "
                f"{helper} ({SECTION})"
            )
    return [*lines, *_eks_lines(where, {"node_groups": node_groups})]


def announce(where: str, block: object, *, prefix: str = "gpu-fault-admin") -> None:
    """Print ``advisory_lines`` on stderr (discovery and ``node-scope``)."""

    for line in advisory_lines(where, block):
        print(f"{prefix}: {line}", file=sys.stderr, flush=True)
