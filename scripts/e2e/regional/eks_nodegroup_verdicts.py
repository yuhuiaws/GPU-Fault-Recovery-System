"""Verdicts of the plain EKS node-group preparation cases (BOOT-036, BOOT-037).

Pure functions over ``gpu-fault-admin prepare-eks-nodegroup``'s JSON document
(``eks_nodegroup_plan.GroupPlan.describe`` per node group) and over what the
runners read independently (``eks_nodegroup_case``). Each returns the list of
reasons the evidence is not the expected shape; an empty list is a pass.

* ``compliant_plan_errors``: a prepared group plans nothing -- COMPLIANT, no
  step, node auto repair off, the ownership label, ``updateStrategy`` MINIMAL,
  the managed user data part in place, in host mode the paired DCGM pin with
  the restart drop-in -- and its read-only node inventory passes every check
  on exactly the group's running instances, with no finding; the one strategy
  finding no setting avoids, a single-node group's whole-group roll, is
  recorded instead (``strategy_limitations``);
* ``refusal_errors``: a refused target stops before anything is changed and
  names the fact (an unknown group, a cluster that is not plain EKS, an
  adoption reference naming no part, a group the verb cannot prepare);
* ``snapshot_drift``: a read-only run left the provider, the Nodes and the
  tool's journals exactly as it found them;
* ``roll_plan_errors`` / ``apply_errors`` / ``replacement_errors`` /
  ``agent_errors`` / ``incident_errors`` / ``stability_errors``: the
  destructive roll of BOOT-037, from the plan that has to roll to the
  replacement nodes that have to come back prepared and adopted by the fleet;
* ``node_renames`` / ``profile_updates``: what the campaign inputs must follow.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from gpu_fault.admin.eks_nodegroup_plan import (
    CHANGES,
    COMPLIANT,
    IN_PROGRESS,
    REFUSED,
    STEP_LABELS,
    STEP_LAUNCH_TEMPLATE,
    STEP_NODE_REPAIR,
    STEP_ROLL,
    STEP_UPDATE_CONFIG,
)
from gpu_fault.admin.eks_nodegroup_update_strategy import UPDATE_STRATEGY_MINIMAL
from scripts.e2e.regional.eks_nodegroup_case import (
    DCGM_HOSTENGINE_LABEL,
    EKS_MANAGED,
    OWNERSHIP_LABEL,
    TOOL_VERB,
    SiteTarget,
    ToolRun,
)

__all__ = [
    "ROLL_STEPS",
    "Refusal",
    "agent_errors",
    "apply_errors",
    "case_verdict",
    "compliant_plan_errors",
    "incident_errors",
    "label_errors",
    "node_renames",
    "profile_updates",
    "refusal_errors",
    "replacement_errors",
    "roll_plan_errors",
    "single_node_roll",
    "snapshot_drift",
    "stability_errors",
    "strategy_limitations",
]

HOST = "host"
PAIRED = "paired"
DISABLED = "disabled"
#: The steps a roll plan may carry, in the order the verb runs them.
ROLL_STEPS = (
    STEP_LAUNCH_TEMPLATE,
    STEP_NODE_REPAIR,
    STEP_LABELS,
    STEP_UPDATE_CONFIG,
    STEP_ROLL,
)
#: What the verb exits with when it refuses (``command_log.report_failure``).
REFUSAL_EXIT_CODE = 2
#: The calls that read node groups or nodes: a cluster-level refusal comes
#: before every one of them.
NODEGROUP_READS = ("aws eks list-nodegroups", "aws eks describe-nodegroup")
NODE_READS = ("aws ssm send-command", "aws ec2 describe-instances")
NOTHING_CHANGED = "cannot be prepared automatically; nothing was changed"


def _items(value: object) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def document_errors(
    document: Mapping[str, Any] | None, *, mode: str, target: SiteTarget
) -> list[str]:
    """The document is the verb's, for this site and this plain EKS member."""

    if document is None:
        return ["the tool printed no plan document"]
    errors = []
    if document.get("verb") != TOOL_VERB or document.get("mode") != mode:
        errors.append(
            f"the document is {document.get('verb')} mode {document.get('mode')}, "
            f"not {TOOL_VERB} mode {mode}"
        )
    if document.get("site_id") != target.site_id:
        errors.append(
            f"site id {document.get('site_id')} is not the site's {target.site_id}"
        )
    if _mapping(document.get("ownership_label")).get("value") != target.site_id:
        errors.append("the ownership label value is not the site id")
    cluster = _mapping(document.get("gpu_cluster"))
    if cluster.get("arn") != target.gpu_cluster_arn:
        errors.append(f"the plan is for {cluster.get('arn')}, not the target member")
    if (
        cluster.get("provider") != EKS_MANAGED
        or cluster.get("site_managed") is not True
    ):
        errors.append(
            f"the cluster is {cluster.get('provider')} with site_managed "
            f"{cluster.get('site_managed')}, not a site-managed {EKS_MANAGED} member"
        )
    recorded = _mapping(document.get("recorded_site"))
    if recorded.get("matches") is not True:
        errors.append(
            f"the state directory records site {recorded.get('site_name')}, not the "
            "id the CPU ARN derives"
        )
    return errors


def groups_by_name(document: Mapping[str, Any] | None) -> dict[str, Mapping[str, Any]]:
    return {
        str(item.get("nodegroup")): item
        for item in _items((document or {}).get("nodegroups"))
        if isinstance(item, Mapping)
    }


def _failed_checks(node: Mapping[str, Any]) -> list[str]:
    return sorted(
        f"{name} {check.get('observed')} (expected {check.get('expected')})"
        for name, check in _mapping(node.get("checks")).items()
        if not _mapping(check).get("ok")
    )


def inventory_errors(
    name: str,
    inventory: Mapping[str, Any],
    *,
    instances: Sequence[str],
    node_instances: Sequence[str],
    desired: object,
) -> list[str]:
    """The inventory passed every check on exactly the group's instances."""

    errors = []
    if inventory.get("outcome") != "passed":
        errors.append(
            f"{name}: the node inventory is {inventory.get('outcome')}"
            + (f" ({inventory.get('reason')})" if inventory.get("reason") else "")
        )
    nodes = [
        item for item in _items(inventory.get("nodes")) if isinstance(item, Mapping)
    ]
    for node in nodes:
        failed = _failed_checks(node)
        if (
            node.get("ssm_status") != "Success"
            or node.get("passed") is not True
            or failed
        ):
            errors.append(
                f"{name}: {node.get('instance_id')} ({node.get('ssm_status')}) fails "
                + (", ".join(failed) or "its inventory")
            )
        if _items(node.get("findings")):
            errors.append(
                f"{name}: {node.get('instance_id')} findings: "
                + "; ".join(str(item) for item in node["findings"])
            )
    if _items(inventory.get("findings")):
        errors.append(
            f"{name}: node findings: "
            + "; ".join(str(item) for item in inventory["findings"])
        )
    checked = sorted(str(node.get("instance_id")) for node in nodes)
    if not checked:
        errors.append(f"{name}: the inventory checked no node")
    if checked != sorted(instances):
        errors.append(
            f"{name}: the inventory checked {checked}, the group runs {sorted(instances)}"
        )
    if sorted(node_instances) != sorted(instances):
        errors.append(
            f"{name}: the group's Kubernetes Nodes are {sorted(node_instances)}, its "
            f"running instances {sorted(instances)}"
        )
    if isinstance(desired, int) and len(instances) != desired:
        errors.append(f"{name}: {len(instances)} running instances, desired {desired}")
    return errors


def _dcgm_errors(
    name: str,
    group: Mapping[str, Any],
    *,
    mode: str,
    pin: str | None,
) -> list[str]:
    dcgm = _mapping(group.get("dcgm_hostengine"))
    if dcgm.get("mode") != mode:
        return [f"{name}: DCGM mode {dcgm.get('mode')}, the site records {mode}"]
    if mode != HOST:
        return []
    errors = []
    if _mapping(group.get("labels")).get(DCGM_HOSTENGINE_LABEL) != HOST:
        errors.append(
            f"{name}: the group does not declare {DCGM_HOSTENGINE_LABEL}=host"
        )
    if not dcgm.get("pin") or (pin is not None and dcgm.get("pin") != pin):
        errors.append(
            f"{name}: DCGM pin {dcgm.get('pin')}" + (f", expected {pin}" if pin else "")
        )
    if dcgm.get("exporter_pairing") != PAIRED:
        errors.append(
            f"{name}: the DCGM pin {dcgm.get('pin')} is "
            f"{dcgm.get('exporter_pairing')} with the exporter's DCGM client "
            f"{dcgm.get('exporter_dcgm_client')}, not paired"
        )
    if not _mapping(dcgm.get("restart_dropin")).get("path"):
        errors.append(f"{name}: host mode plans no nvidia-dcgm restart drop-in")
    return errors


def _user_data_errors(name: str, group: Mapping[str, Any]) -> list[str]:
    user_data = _mapping(group.get("user_data"))
    parts = [
        item for item in _items(user_data.get("parts")) if isinstance(item, Mapping)
    ]
    errors = []
    if user_data.get("action") != "keep":
        errors.append(f"{name}: the user data would {user_data.get('action')}")
    if sum(1 for item in parts if item.get("managed") is True) != 1:
        errors.append(f"{name}: the user data does not carry exactly one managed part")
    for part in parts:
        for line in _items(part.get("contradictions")):
            errors.append(f"{name}: user data part {part.get('index')}: {line}")
    return errors


def single_node_roll(group: Mapping[str, Any]) -> bool:
    """The plan rolls a group of exactly one node with MINIMAL, not kept.

    Such a roll takes the group's only GPU node out whatever its
    unavailability (``maxUnavailable`` is at least 1), so its whole-group
    finding names a limit, not a fix. Decided from the plan's structured
    fields (``scaling.desiredSize``, ``update_strategy``), never from the
    finding's words: tool versions word it differently.
    """

    desired = _mapping(group.get("scaling")).get("desiredSize")
    strategy = _mapping(group.get("update_strategy"))
    return (
        isinstance(desired, int)
        and not isinstance(desired, bool)
        and desired == 1
        and strategy.get("planned") == UPDATE_STRATEGY_MINIMAL
        and not strategy.get("kept")
    )


def strategy_limitations(document: Mapping[str, Any] | None) -> list[str]:
    """What the verdicts record instead of judging: per single-node group of
    the plan, that no setting keeps a GPU node up during its roll."""

    notes = []
    for name, group in sorted(groups_by_name(document).items()):
        if not single_node_roll(group):
            continue
        finding = _mapping(group.get("update_strategy")).get("finding")
        notes.append(
            f"{name} runs a single node (desiredSize 1): a MINIMAL roll takes its "
            "only GPU node out until the replacement joins, whatever maxUnavailable "
            "says; keeping one up needs a second node, so the case records this "
            "instead of failing on it"
            + (f" (the tool's finding: {finding})" if finding else "")
        )
    return notes


def compliant_group_errors(
    name: str,
    group: Mapping[str, Any],
    *,
    target: SiteTarget,
    mode: str,
    pin: str | None = None,
) -> list[str]:
    """One prepared group: COMPLIANT and every prerequisite in place."""

    errors = []
    if group.get("verdict") != COMPLIANT or group.get("refusal") is not None:
        errors.append(
            f"{name}: verdict {group.get('verdict')}"
            + (f" ({group.get('refusal')})" if group.get("refusal") else "")
        )
    steps = [str(_mapping(step).get("step")) for step in _items(group.get("steps"))]
    if steps:
        errors.append(f"{name}: the plan would run {steps}")
    if group.get("node_repair") != DISABLED:
        errors.append(f"{name}: EKS node auto repair is {group.get('node_repair')}")
    if _mapping(group.get("labels")).get(OWNERSHIP_LABEL) != target.site_id:
        errors.append(
            f"{name}: the group is not labelled {OWNERSHIP_LABEL}={target.site_id}"
        )
    strategy = _mapping(group.get("update_strategy"))
    finding = strategy.get("finding")
    planned = strategy.get("planned")
    if (
        strategy.get("current") != UPDATE_STRATEGY_MINIMAL
        or planned != UPDATE_STRATEGY_MINIMAL
        or strategy.get("kept")
        or (finding and not single_node_roll(group))
    ):
        errors.append(
            f"{name}: updateStrategy {strategy.get('current')}"
            + (f", planned {planned}" if planned != strategy.get("current") else "")
            + (" (kept)" if strategy.get("kept") else "")
            + (f" ({finding})" if finding else "")
        )
    if _mapping(group.get("roll")).get("needed") is not False:
        errors.append(f"{name}: the plan still needs a roll")
    errors += _dcgm_errors(name, group, mode=mode, pin=pin)
    errors += _user_data_errors(name, group)
    return errors


def kubernetes_errors(findings: Mapping[str, Any]) -> list[str]:
    """The read-only Kubernetes findings a joined plain EKS cluster must clear."""

    if findings.get("available") is not True:
        return [f"Kubernetes checks unavailable: {findings.get('reason')}"]
    errors = []
    missing = _items(_mapping(findings.get("namespaces")).get("missing"))
    if missing:
        errors.append(f"allowed workload namespaces missing: {missing}")
    plugins = _mapping(findings.get("device_plugins"))
    for kind in ("nvidia", "efa"):
        plugin = _mapping(plugins.get(kind))
        if plugin.get("running") is not True:
            errors.append(
                f"{kind} device plugin not running: {plugin.get('daemonsets') or 'none'}"
            )
    tolerating = _items(findings.get("quiesce_tolerating_gpu_daemonsets"))
    if tolerating:
        errors.append(f"GPU DaemonSets tolerate the quiesce taint: {tolerating}")
    return errors


@dataclass(frozen=True)
class GroupFacts:
    """What the runner read about one group itself, beside the tool."""

    instances: tuple[str, ...]
    node_instances: tuple[str, ...]
    desired: int | None


def compliant_plan_errors(
    document: Mapping[str, Any] | None,
    *,
    target: SiteTarget,
    facts: Mapping[str, GroupFacts],
    mode: str,
    pin: str | None = None,
) -> list[str]:
    """The whole read-only plan of prepared groups (BOOT-036, BOOT-037 after)."""

    errors = document_errors(document, mode="plan", target=target)
    if document is None:
        return errors
    if document.get("status") != COMPLIANT:
        errors.append(f"the plan status is {document.get('status')}, not {COMPLIANT}")
    groups = groups_by_name(document)
    if sorted(groups) != sorted(facts):
        errors.append(f"the plan covers {sorted(groups)}, the case {sorted(facts)}")
    for name in sorted(set(groups) & set(facts)):
        group = groups[name]
        errors += compliant_group_errors(name, group, target=target, mode=mode, pin=pin)
        fact = facts[name]
        errors += inventory_errors(
            name,
            _mapping(group.get("current_nodes")),
            instances=fact.instances,
            node_instances=fact.node_instances,
            desired=fact.desired,
        )
    errors += kubernetes_errors(_mapping(document.get("findings")))
    return errors


# --- refusals ---------------------------------------------------------------------


@dataclass(frozen=True)
class Refusal:
    """What one refusal probe must show.

    ``document`` is the plan status the verb prints first (``REFUSED`` for a
    group-level refusal, ``COMPLIANT`` for the adoption no-op), ``None`` when
    it refuses before printing any plan; ``text`` must appear in what it said
    (its error line, a group's refusal or a user data note);
    ``forbidden_calls`` are provider calls it must not have reached.
    """

    name: str
    text: str
    document: str | None = None
    exit_code: int = REFUSAL_EXIT_CODE
    forbidden_calls: tuple[str, ...] = ()

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "text": self.text,
            "document": self.document,
            "exit_code": self.exit_code,
            "forbidden_calls": list(self.forbidden_calls),
        }


def refusal_errors(run: ToolRun, expected: Refusal) -> list[str]:
    errors = []
    prefix = f"{expected.name}:"
    if run.returncode != expected.exit_code:
        errors.append(f"{prefix} exit {run.returncode}, expected {expected.exit_code}")
    document = run.document
    status = (document or {}).get("status")
    if status != expected.document:
        errors.append(f"{prefix} plan status {status}, expected {expected.document}")
    said = list(run.error_lines)
    for group in groups_by_name(document).values():
        said.append(str(group.get("refusal") or ""))
        said.extend(
            str(item) for item in _items(_mapping(group.get("user_data")).get("notes"))
        )
    if not any(expected.text in line for line in said):
        errors.append(f"{prefix} the tool did not say {expected.text!r}: {said}")
    if expected.document == REFUSED and not any(
        NOTHING_CHANGED in line for line in said
    ):
        errors.append(f"{prefix} the refusal does not say nothing was changed")
    reached = sorted(
        {
            call
            for call in run.provider_calls
            for forbidden in expected.forbidden_calls
            if call.startswith(forbidden)
        }
    )
    if reached:
        errors.append(f"{prefix} the tool still reached {reached}")
    return errors


def snapshot_drift(before: Mapping[str, Any], after: Mapping[str, Any]) -> list[str]:
    """Every difference between two ``eks_nodegroup_case.snapshot`` reads."""

    errors = []
    groups_before = _mapping(before.get("nodegroups"))
    groups_after = _mapping(after.get("nodegroups"))
    for name in sorted(set(groups_before) | set(groups_after)):
        old, new = _mapping(groups_before.get(name)), _mapping(groups_after.get(name))
        for key in sorted(set(old) | set(new)):
            if old.get(key) != new.get(key):
                errors.append(
                    f"{name}: {key} changed from {old.get(key)} to {new.get(key)}"
                )
    if before.get("journals") != after.get("journals"):
        errors.append(
            f"the tool's journals changed: {before.get('journals')} -> "
            f"{after.get('journals')}"
        )
    return errors


# --- the roll (BOOT-037) ----------------------------------------------------------


def roll_plan_errors(
    document: Mapping[str, Any] | None,
    *,
    target: SiteTarget,
    nodegroup: str,
    mode: str,
    desired: object,
) -> list[str]:
    """The plan of the target configuration rolls the group with MINIMAL."""

    errors = document_errors(document, mode="plan", target=target)
    if document is None:
        return errors
    status = document.get("status")
    group = _mapping(groups_by_name(document).get(nodegroup))
    if status == COMPLIANT:
        return errors + [
            f"{nodegroup} already runs the target configuration: the plan rolls "
            "nothing. Roll only a group whose plan changes it (a first preparation, "
            "a new managed part, another DCGM pin, an adopted hand-written part)"
        ]
    if status == IN_PROGRESS:
        pending = _mapping(group.get("pending_update")).get("update_id")
        return errors + [
            f"update {pending} of an earlier apply is still in progress; rerun that "
            "apply (it resumes the journaled update), then plan the case again"
        ]
    if status == REFUSED:
        return errors + [f"{nodegroup} is refused: {group.get('refusal')}"]
    if status != CHANGES or sorted(groups_by_name(document)) != [nodegroup]:
        return errors + [
            f"the plan status is {status} for {sorted(groups_by_name(document))}"
        ]
    steps = [str(_mapping(step).get("step")) for step in _items(group.get("steps"))]
    if STEP_ROLL not in steps or [item for item in steps if item not in ROLL_STEPS]:
        errors.append(f"{nodegroup}: the plan runs {steps}, not a roll")
    strategy = _mapping(group.get("update_strategy"))
    if strategy.get("planned") != UPDATE_STRATEGY_MINIMAL or strategy.get("kept"):
        errors.append(
            f"{nodegroup}: the roll would use updateStrategy {strategy.get('planned')}"
        )
    elif strategy.get("finding") and not single_node_roll(group):
        # The replan after the roll would fail the BOOT-036 verdict on it.
        errors.append(f"{nodegroup}: {strategy.get('finding')}")
    dcgm = _mapping(group.get("dcgm_hostengine"))
    if dcgm.get("mode") != mode:
        errors.append(f"{nodegroup}: DCGM mode {dcgm.get('mode')}, expected {mode}")
    elif mode == HOST and (
        not dcgm.get("pin") or dcgm.get("exporter_pairing") != PAIRED
    ):
        errors.append(
            f"{nodegroup}: the DCGM pin {dcgm.get('pin')} is "
            f"{dcgm.get('exporter_pairing')} with the exporter, not paired"
        )
    inventory = _mapping(group.get("current_nodes"))
    if inventory.get("outcome") not in ("passed", "failed"):
        errors.append(
            f"{nodegroup}: the current nodes are unreadable "
            f"({inventory.get('outcome')}: {inventory.get('reason')})"
        )
    elif isinstance(desired, int) and len(_items(inventory.get("nodes"))) != desired:
        errors.append(
            f"{nodegroup}: the inventory read {len(_items(inventory.get('nodes')))} "
            f"nodes, desired {desired}"
        )
    return errors


def apply_errors(run: ToolRun, *, nodegroup: str) -> list[str]:
    """The apply completed: prepared, rolled and verified by the tool itself."""

    document = run.document
    errors = []
    if run.returncode != 0:
        errors.append(f"the apply exited {run.returncode}: {run.error_lines[-3:]}")
    applied = _mapping((document or {}).get("apply"))
    results = [
        item for item in _items(applied.get("results")) if isinstance(item, Mapping)
    ]
    if (document or {}).get("mode") != "apply" or applied.get("status") != "COMPLETED":
        errors.append(f"the apply status is {applied.get('status')}")
    result = next((item for item in results if item.get("nodegroup") == nodegroup), {})
    if result.get("outcome") != "COMPLETED":
        errors.append(
            f"{nodegroup}: {result.get('outcome')}: {result.get('reason')} "
            f"(journal {result.get('journal')})"
        )
    rolls = [
        _mapping(step.get("update"))
        for step in _items(result.get("steps"))
        if isinstance(step, Mapping) and step.get("step") == STEP_ROLL
    ]
    if not any(item.get("status") == "Successful" for item in rolls):
        errors.append(f"{nodegroup}: no roll update ended Successful: {rolls}")
    if _mapping(result.get("verification")).get("outcome") != "passed":
        errors.append(
            f"{nodegroup}: the tool's verification is "
            f"{_mapping(result.get('verification')).get('outcome')}"
        )
    return errors


def replacement_errors(
    before: Sequence[Mapping[str, Any]],
    after: Sequence[Mapping[str, Any]],
    *,
    desired: object,
) -> list[str]:
    """Every node was replaced and every replacement is a Ready Node."""

    errors = []
    old_instances = {str(item.get("instance_id")) for item in before}
    old_names = {str(item.get("name")) for item in before}
    kept = sorted(
        str(item.get("name"))
        for item in after
        if str(item.get("instance_id")) in old_instances
        or str(item.get("name")) in old_names
    )
    if kept:
        errors.append(f"departed nodes are still Nodes of the group: {kept}")
    if isinstance(desired, int) and len(after) != desired:
        errors.append(f"the group has {len(after)} Nodes, desired {desired}")
    unready = sorted(
        str(item.get("name")) for item in after if item.get("ready") != "True"
    )
    if unready:
        errors.append(f"replacement nodes not Ready: {unready}")
    return errors


def label_errors(
    nodes: Sequence[Mapping[str, Any]], *, site_id: str, mode: str
) -> list[str]:
    errors = []
    for node in nodes:
        labels = _mapping(node.get("labels"))
        if labels.get(OWNERSHIP_LABEL) != site_id:
            errors.append(f"{node.get('name')}: no {OWNERSHIP_LABEL}={site_id}")
        if mode == HOST and labels.get(DCGM_HOSTENGINE_LABEL) != HOST:
            errors.append(f"{node.get('name')}: no {DCGM_HOSTENGINE_LABEL}=host")
    return errors


def _agents(store: Mapping[str, Any]) -> dict[str, str]:
    return {
        str(item.get("node_id")): str(item.get("lifecycle_state"))
        for item in _items(store.get("agents"))
        if isinstance(item, Mapping)
    }


def agent_errors(
    store: Mapping[str, Any],
    *,
    active: Iterable[str],
    departed: Iterable[str] = (),
) -> list[str]:
    """``active`` nodes have ACTIVE Agents; no ``departed`` node still has one."""

    agents = _agents(store)
    errors = [
        f"{node}: Agent {agents.get(node, 'absent')}, not ACTIVE"
        for node in sorted(active)
        if agents.get(node) != "ACTIVE"
    ]
    errors += [
        f"{node}: the departed node's Agent is still {agents[node]}"
        for node in sorted(departed)
        if agents.get(node) in ("ACTIVE", "DRAINING")
    ]
    return errors


def _instant(value: object) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def incident_errors(
    store: Mapping[str, Any],
    *,
    since: datetime | None = None,
    nodes: Iterable[str] = (),
    cluster_wide: bool = True,
) -> list[str]:
    """What the store holds that a case must not run beside.

    Any incident opened since ``since``; an active incident or an executable
    workflow -- anywhere on the cluster when ``cluster_wide``, else only on
    ``nodes`` -- and an escalated incident on ``nodes``.
    """

    watched = set(nodes)
    errors = []
    for item in _items(store.get("incidents")):
        if not isinstance(item, Mapping):
            continue
        created = _instant(item.get("created_at"))
        on_nodes = bool(watched & {str(node) for node in _items(item.get("node_ids"))})
        shown = (
            f"{item.get('incident_id')} {item.get('state')} {item.get('node_ids')} "
            f"{item.get('reason')}"
        )
        if since is not None and created is not None and created >= since:
            errors.append(f"incident opened during the case: {shown}")
        elif item.get("active") is True and (cluster_wide or on_nodes):
            errors.append(f"active incident: {shown}")
        elif on_nodes:
            errors.append(f"open incident on the group's nodes: {shown}")
    for item in _items(store.get("workflows")):
        if not isinstance(item, Mapping):
            continue
        if cluster_wide or watched & {
            str(node) for node in _items(item.get("node_ids"))
        }:
            errors.append(
                f"executable workflow {item.get('request_id')} {item.get('status')} "
                f"on {item.get('node_ids')}"
            )
    return errors


def stability_errors(
    nodes: Sequence[Mapping[str, Any]], *, now: datetime, ready_seconds: float
) -> list[str]:
    """Every node Ready for ``ready_seconds``, schedulable and not isolated."""

    errors = []
    for node in nodes:
        since = _instant(node.get("ready_since"))
        if node.get("ready") != "True" or since is None:
            errors.append(f"{node.get('name')}: not Ready")
        elif (now - since).total_seconds() < ready_seconds:
            errors.append(
                f"{node.get('name')}: Ready for {(now - since).total_seconds():.0f} s "
                f"only, not {ready_seconds:.0f} s"
            )
        if node.get("unschedulable"):
            errors.append(f"{node.get('name')}: cordoned")
        if _items(node.get("blocking_taints")):
            errors.append(f"{node.get('name')}: tainted {node['blocking_taints']}")
    return errors


def node_renames(
    before: Sequence[Mapping[str, Any]], after: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Departed and replacement nodes paired in name order.

    A roll keeps no identity across a replacement, so the pairing is a
    convention, the same one every time: the n-th departed name (sorted)
    becomes the n-th replacement name (sorted).
    """

    def brief(item: Mapping[str, Any] | None) -> dict[str, Any] | None:
        if item is None:
            return None
        return {"name": item.get("name"), "instance_id": item.get("instance_id")}

    old = sorted(before, key=lambda item: str(item.get("name")))
    new = sorted(after, key=lambda item: str(item.get("name")))
    return [
        {
            "departed": brief(old[index] if index < len(old) else None),
            "replacement": brief(new[index] if index < len(new) else None),
        }
        for index in range(max(len(old), len(new)))
    ]


def profile_updates(
    arguments: Mapping[str, Any], renames: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """The site profile arguments that name a departed node, with the
    replacement name the campaign inputs should carry instead."""

    mapping = {
        str(_mapping(item.get("departed")).get("name")): _mapping(
            item.get("replacement")
        ).get("name")
        for item in renames
        if item.get("departed")
    }
    updates = []
    for key, value in sorted(arguments.items()):
        for item in value if isinstance(value, list) else [value]:
            if isinstance(item, str) and item in mapping:
                updates.append(
                    {"argument": key, "departed": item, "replacement": mapping[item]}
                )
    return updates


def case_verdict(stages: Mapping[str, Sequence[str]]) -> str:
    return "PASS" if stages and not any(stages.values()) else "FAIL"
