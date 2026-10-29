"""Read-only oracle for b220's workload failure domains (GF-REGIONAL-DESTR-027).

The product rule (``execution/remediation_budget.py``, design §9.5). Before
b220 every budgeted claim also took ``region``, ``cluster:<c>``,
``class:<c>:<resource class>`` and one ``domain:<c>:<label>`` slot per node
group or instance group (the node-label map of ``failure_domains``), so two
independent faults on two nodes of one group were repaired one after the
other: the second workflow stayed PENDING behind a ``REMEDIATION_BUDGET_WAIT``
hold on that ``domain:`` scope. Since b220:

* a training-job attempt, or a GPU node without one, is one merged workflow;
* a workflow's budgeted steps -- ``QUARANTINE`` and every node-mutating
  operation but ``RESTORE_GPU_SERVICES`` -- claim ``node:<c>:<node>`` (limit 1)
  for every node they name; ``STOP_WORKLOADS`` and ``RESTART_WORKLOAD`` also
  claim ``job:<c>:<workload_id>`` and ``job:<c>:job-id/<job_id>``; the only
  ``domain:`` scope left is an SXID's node-local NVSwitch ``fabric_partition``;
* no region, cluster or resource-class scope is claimed. The cluster's storm
  breaker (``gpu_fault.remediation_storm``) holds a first admission instead
  once ``storm_threshold`` = max(3, ceil(0.2 x the cluster's GPU nodes))
  distinct nodes entered destructive remediation within 600 s across at least
  two workload domains (a job, else the incident).

What the breaker counts. ``execution/storm_admission.storm_candidate`` judges a
workflow's FIRST admission of budgeted work and counts every node of its
``node:`` claims -- every node any budgeted step still to run names. For a
running multi-node attempt the job-wide STOP_WORKLOADS and RESTART_WORKLOAD
name every node of the allocation (``orchestration/families/grouped_faults``
widens them), so a job's healthy peer is counted with its faulted node.

What the oracle judges, on durable control-plane records only (never logs),
for a case that faults each of N failure domains once (``DomainExpectation``):

1. one workflow per domain, each on its own incident and node set;
2. its claims: the node scope of the faulted node (a job's also the job
   tokens and, for its peers, their node scopes), plus a fabric-partition
   ``domain:`` only when one of its own steps names that partition; nothing of
   the retired kinds (``region``, ``cluster:``, ``class:``) and no node-label
   ``domain:``;
3. its holds: ``remediation_budget_wait_count`` and every ``HOLD`` event --
   never refused on a node-group ``domain:`` (or another retired) scope, never
   held by the storm breaker;
4. parallel work: the domains' budgeted spans (or their remediation steps)
   overlap pairwise in wall-clock time, from the step executions'
   ``started_at`` (the first attempt) and ``updated_at`` (the terminal record);
5. the storm breaker stayed CLOSED through the case, with the nodes it counted
   for these workflows (its own window entries, else their claims) below its
   threshold;
6. every workflow SUCCEEDED, every incident RECOVERED, and nothing the case
   left behind on its nodes: no open workflow, no open incident.

Outcomes. PASS is ``PARALLEL``. A FAIL names its shape so a pre-b220 or a
storm result never reads as the parallel PASS: ``SERIALIZED`` (a node-group
domain or another retired scope was claimed or held), ``BREAKER_TRIPPED`` (a
workflow was held by, or the case tripped, the storm breaker), ``NOT_PARALLEL``
(the work did not overlap), ``CONTRACT`` (anything else). A record the
judgement needs that cannot be read is ``INCONCLUSIVE`` / ``RECORD_MISSING``.

The store read (``ORACLE_PROBE``) runs in a ``gpu-fault-api-ha`` Pod through
``RegionalLiveFixture.cpu_python`` -- the store path every regional runner
uses -- and only reads: ``get_workflow``, ``get_incident``,
``list_active_workflow_incidents``, ``list_incidents_by_state``,
``list_workflows`` (the breaker's held waiters, as ``remediation-storm
--dry-run`` lists them), ``get_remediation_storm_breaker`` and ``list_agents``
(the GPU node count the breaker's threshold is taken from).

Preconditions are node counts read live (``node_group_gate``): the GPU nodes
of one node group -- ``eks.amazonaws.com/nodegroup`` on plain EKS,
``sagemaker.amazonaws.com/instance-group-name`` on HyperPod -- never a cluster
name; short of the minimum the case is ``NOT_RUN(CAPABILITY_ABSENT)``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from itertools import combinations
from typing import Any

from gpu_fault.cluster_scope import EKS_NODEGROUP_LABEL, HYPERPOD_INSTANCE_GROUP_LABEL
from gpu_fault.execution.remediation_budget import WHOLE_JOB_OPERATIONS
from gpu_fault.models import WorkflowOperation
from gpu_fault.operation_registry import NODE_MUTATING_OPERATIONS
from gpu_fault.remediation_storm import (
    STORM_SCOPE_PREFIX,
    STORM_WINDOW_SECONDS,
    storm_threshold,
)

PASS = "PASS"
FAIL = "FAIL"
INCONCLUSIVE = "INCONCLUSIVE"
NOT_RUN = "NOT_RUN"
CAPABILITY_ABSENT = "CAPABILITY_ABSENT"

PARALLEL = "PARALLEL"
SERIALIZED = "SERIALIZED"
BREAKER_TRIPPED = "BREAKER_TRIPPED"
NOT_PARALLEL = "NOT_PARALLEL"
CONTRACT = "CONTRACT"
RECORD_MISSING = "RECORD_MISSING"

#: Which span of a domain's work the overlap is judged on.
OVERLAP_BUDGETED = "budgeted"
OVERLAP_REMEDIATION = "remediation"

#: The node-group labels, HyperPod first (a HyperPod node carries no EKS
#: node-group label; read in this order a node is never in two groups).
NODE_GROUP_LABELS = (HYPERPOD_INSTANCE_GROUP_LABEL, EKS_NODEGROUP_LABEL)

#: ``remediation_budget._budgeted`` spelt with the public operation sets: the
#: operations whose steps claim scopes (pinned against the product's own
#: ``claims_in_cluster`` by the unit suite).
BUDGETED_OPERATIONS = frozenset(
    operation.value
    for operation in NODE_MUTATING_OPERATIONS
    if operation is not WorkflowOperation.RESTORE_GPU_SERVICES
) | {WorkflowOperation.QUARANTINE.value}
#: The steps that take a job's token (``remediation_budget.WHOLE_JOB_OPERATIONS``).
JOB_TOKEN_OPERATIONS = frozenset(operation.value for operation in WHOLE_JOB_OPERATIONS)
#: The steps the job family widens to the attempt's whole allocation: the only
#: steps of a job workflow that may name a node no fault was raised on.
JOB_WIDE_OPERATIONS = JOB_TOKEN_OPERATIONS | {
    WorkflowOperation.CHECKPOINT_WORKLOADS.value
}
#: Scope kinds b220 retired; a claim or a refusal on one is the pre-b220 shape.
RETIRED_SCOPE_KINDS = frozenset({"region", "cluster", "class"})
WAIT_CODE = "REMEDIATION_BUDGET_WAIT"
TRIPPED_CODE = "REMEDIATION_STORM_TRIPPED"
ABORTED_CODE = "REMEDIATION_STORM_ABORTED"
TERMINAL_STEP_STATUSES = frozenset({"SUCCEEDED", "FAILED"})
OFFICIAL_PHASES = frozenset({None, "official"})
#: Incident states that still need something: the case leaves none behind.
OPEN_INCIDENT_STATES = (
    "DETECTED",
    "ACTION_PENDING",
    "SAFETY_PENDING",
    "QUARANTINED",
    "ESCALATED",
)


# --------------------------------------------------------------------------- #
# Preconditions: GPU nodes per node group, read live
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class NodeGroup:
    """One node group's GPU nodes, as the cluster's Node labels name them."""

    label: str
    name: str
    nodes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "name": self.name,
            "nodes": list(self.nodes),
            "gpu_nodes": len(self.nodes),
        }


@dataclass(frozen=True)
class NodeGroupGate:
    """The group a case may use, or why the cluster cannot carry the case."""

    minimum: int
    group: NodeGroup | None
    groups: tuple[NodeGroup, ...]
    not_run: dict[str, str] | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "minimum_gpu_nodes_in_one_node_group": self.minimum,
            "group": self.group.as_dict() if self.group is not None else None,
            "groups": [item.as_dict() for item in self.groups],
            "not_run": self.not_run,
        }


def node_group_of(labels: Mapping[str, Any] | None) -> tuple[str, str] | None:
    """``(label, group)`` of a Node's labels, or ``None`` outside any group."""

    for key in NODE_GROUP_LABELS:
        value = str((labels or {}).get(key) or "").strip()
        if value:
            return key, value
    return None


def gpu_node_groups(gpu_nodes: Iterable[Mapping[str, Any]]) -> tuple[NodeGroup, ...]:
    """The GPU node groups of ``RegionalLiveFixture.gpu_nodes()``, largest
    first (ties in label then name order); a node outside every group is in
    none of them."""

    members: dict[tuple[str, str], set[str]] = {}
    for node in gpu_nodes:
        key = node_group_of(node.get("labels"))
        name = str(node.get("name") or "")
        if key is not None and name:
            members.setdefault(key, set()).add(name)
    groups = [
        NodeGroup(label=label, name=name, nodes=tuple(sorted(nodes)))
        for (label, name), nodes in members.items()
    ]
    return tuple(
        sorted(groups, key=lambda item: (-len(item.nodes), item.label, item.name))
    )


def capability_absent(detail: str) -> dict[str, str]:
    """The ``NOT_RUN(CAPABILITY_ABSENT)`` record a short cluster gets."""

    return {"verdict": NOT_RUN, "reason": CAPABILITY_ABSENT, "detail": detail}


def node_group_gate(
    gpu_nodes: Sequence[Mapping[str, Any]],
    *,
    minimum: int,
    cluster_id: str,
    requested: str = "",
) -> NodeGroupGate:
    """The node group a case needing ``minimum`` GPU nodes in one group runs in.

    The largest GPU node group, or the one ``requested`` names; short of
    ``minimum`` the gate carries ``NOT_RUN(CAPABILITY_ABSENT)`` naming the
    requirement and the count found. Only GPU nodes count (``gpu_nodes()``
    lists nodes that allocate ``nvidia.com/gpu``); their state is the
    preflight's business, the count is the cluster's shape.
    """

    if minimum < 1:
        raise ValueError("a node-group minimum must be positive")
    groups = gpu_node_groups(gpu_nodes)
    requirement = f"needs >={minimum} GPU nodes in one node group"
    if requested:
        chosen = next((item for item in groups if item.name == requested), None)
        if chosen is None:
            detail = (
                f"{requirement}; cluster {cluster_id} has no GPU node in node "
                f"group {requested} (groups: "
                + (
                    ", ".join(f"{item.name}={len(item.nodes)}" for item in groups)
                    or "none"
                )
                + ")"
            )
            return NodeGroupGate(minimum, None, groups, capability_absent(detail))
    elif groups:
        chosen = groups[0]
    else:
        detail = (
            f"{requirement}; cluster {cluster_id} has {len(gpu_nodes)} GPU nodes "
            f"and none carries {' or '.join(NODE_GROUP_LABELS)}"
        )
        return NodeGroupGate(minimum, None, groups, capability_absent(detail))
    if len(chosen.nodes) < minimum:
        detail = (
            f"{requirement}; cluster {cluster_id} group {chosen.name} "
            f"({chosen.label}) has {len(chosen.nodes)}"
        )
        return NodeGroupGate(minimum, chosen, groups, capability_absent(detail))
    return NodeGroupGate(minimum, chosen, groups, None)


# --------------------------------------------------------------------------- #
# The store read (runs in a gpu-fault-api-ha Pod; reads only)
# --------------------------------------------------------------------------- #
ORACLE_PROBE = r"""
import json
import sys
from datetime import datetime, timezone

from gpu_fault.app import ApplicationContext
from gpu_fault.fleet import AgentLifecycleState
from gpu_fault.models import IncidentState, WorkflowStatus
from gpu_fault.store import NotFoundError

cluster_id = sys.argv[1]
workflow_ids = [item for item in sys.argv[2].split(",") if item]
node_ids = sorted({item for item in sys.argv[3].split(",") if item})
store = ApplicationContext.from_environment().store
workflows = {}
incidents = {}
missing_workflows = []
missing_incidents = []
for request_id in workflow_ids:
    try:
        workflow = store.get_workflow(request_id)
    except NotFoundError:
        missing_workflows.append(request_id)
        continue
    workflows[request_id] = workflow.model_dump(mode="json")
    try:
        incident = store.get_incident(workflow.incident_id)
    except NotFoundError:
        missing_incidents.append(workflow.incident_id)
        continue
    incidents[incident.incident_id] = incident.model_dump(mode="json")
storm_supported = hasattr(store, "get_remediation_storm_breaker")
breaker = store.get_remediation_storm_breaker(cluster_id) if storm_supported else None
held_scope = "storm:" + cluster_id
held = sorted(
    item.request_id
    for item in store.list_workflows(
        {WorkflowStatus.PENDING, WorkflowStatus.SAFETY_PENDING},
        limit=1000,
        newest_first=True,
    )
    if item.remediation_budget_last_blocked_scope == held_scope
)
nodes = set(node_ids)
active = []
open_incidents = []
if nodes:
    for incident, workflow in store.list_active_workflow_incidents(
        cluster_id, node_ids=nodes
    ):
        active.append({
            "incident_id": incident.incident_id,
            "workflow_request_id": workflow.request_id,
            "status": workflow.status.value,
            "node_ids": list(incident.node_ids),
            "created_at": workflow.created_at.isoformat(),
        })
    states = {IncidentState(value) for value in sys.argv[4].split(",") if value}
    for item in store.list_incidents_by_state(cluster_id, states, node_ids=nodes):
        open_incidents.append({
            "incident_id": item.incident_id,
            "state": item.state.value,
            "node_ids": list(item.node_ids),
            "created_at": item.created_at.isoformat(),
        })
try:
    agents = sum(
        1
        for agent in store.list_agents(cluster_id)
        if agent.lifecycle_state
        in {AgentLifecycleState.ACTIVE, AgentLifecycleState.DRAINING}
    )
except Exception:
    agents = None
print(json.dumps({
    "read_at": datetime.now(timezone.utc).isoformat(),
    "cluster_id": cluster_id,
    "workflows": workflows,
    "incidents": incidents,
    "missing_workflows": missing_workflows,
    "missing_incidents": missing_incidents,
    "storm_supported": storm_supported,
    "breaker": breaker.model_dump(mode="json") if breaker is not None else None,
    "storm_held_workflow_ids": held,
    "active_workflow_incidents": active,
    "open_incidents": open_incidents,
    "cluster_gpu_agents": agents,
}, sort_keys=True, default=str))
"""


def read_oracle_snapshot(
    fixture: Any,
    *,
    workflow_ids: Sequence[str],
    node_ids: Sequence[str],
) -> dict[str, Any]:
    """One read-only store read of what the judges need (``ORACLE_PROBE``)."""

    for value in (*workflow_ids, *node_ids):
        if not value or "," in value:
            raise ValueError("oracle workflow and node ids must be non-empty, no comma")
    return dict(
        fixture.cpu_python(
            ORACLE_PROBE,
            fixture.settings.cluster_id,
            ",".join(workflow_ids),
            ",".join(node_ids),
            ",".join(OPEN_INCIDENT_STATES),
        )
    )


# --------------------------------------------------------------------------- #
# Record helpers (pure)
# --------------------------------------------------------------------------- #
def node_scope(cluster_id: str, node: str) -> str:
    return f"node:{cluster_id}:{node}"


def job_scopes(cluster_id: str, job_id: str, workload_ids: Iterable[str]) -> set[str]:
    """The tokens b220's STOP_WORKLOADS / RESTART_WORKLOAD claim for one job."""

    scopes = {f"job:{cluster_id}:{workload}" for workload in workload_ids if workload}
    if job_id:
        scopes.add(f"job:{cluster_id}:job-id/{job_id}")
    return scopes


def scope_kind(scope: str) -> str:
    return scope.split(":", 1)[0]


def parse_time(value: Any) -> datetime | None:
    """A record's ISO timestamp as an aware datetime; ``None`` if unreadable."""

    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def _records(value: Any) -> list[dict[str, Any]]:
    return [item for item in value or [] if isinstance(item, dict)]


def fabric_partitions(workflow: Mapping[str, Any]) -> set[str]:
    """The NVSwitch partitions the workflow's own steps name (the one
    ``domain:`` scope b220 keeps)."""

    partitions: set[str] = set()
    for step in _records(
        [*(workflow.get("official_steps") or []), *(workflow.get("safety_steps") or [])]
    ):
        raw = (step.get("parameters") or {}).get("fabric_partition")
        values = raw if isinstance(raw, list) else [raw]
        partitions.update(str(value) for value in values if value)
    return partitions


def budget_waits(workflow: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Each wait's opening ``HOLD`` event: ``{"at", "scope", "reason"}``."""

    return [
        {
            "at": event.get("at"),
            "scope": (event.get("details") or {}).get("scope"),
            "reason": event.get("reason"),
        }
        for event in _records(workflow.get("events"))
        if event.get("kind") == "HOLD" and event.get("code") == WAIT_CODE
    ]


def storm_events(workflow: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The breaker's own events on the workflow (a trip, an operator abort)."""

    return [
        {
            "at": event.get("at"),
            "code": event.get("code"),
            "reason": event.get("reason"),
        }
        for event in _records(workflow.get("events"))
        if event.get("code") in {TRIPPED_CODE, ABORTED_CODE}
    ]


def history_truncated(workflow: Mapping[str, Any]) -> bool:
    return any(
        event.get("kind") == "HISTORY_TRUNCATED"
        for event in _records(workflow.get("events"))
    )


def official_executions(
    workflow: Mapping[str, Any], operations: frozenset[str] | None = None
) -> list[dict[str, Any]]:
    return [
        item
        for item in _records(workflow.get("step_executions"))
        if item.get("phase") in OFFICIAL_PHASES
        and (operations is None or str(item.get("operation")) in operations)
    ]


@dataclass(frozen=True)
class Span:
    """Wall-clock interval of some of a workflow's step executions."""

    start: datetime
    end: datetime
    operations: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "operations": list(self.operations),
        }


def execution_span(executions: Sequence[Mapping[str, Any]]) -> Span | str:
    """The span from the first ``started_at`` to the last terminal
    ``updated_at``, or why it cannot be taken from these records."""

    if not executions:
        return "no step execution is recorded"
    open_steps = sorted(
        str(item.get("operation"))
        for item in executions
        if item.get("status") not in TERMINAL_STEP_STATUSES
    )
    if open_steps:
        return f"{', '.join(open_steps)} has no terminal step execution"
    starts = [parse_time(item.get("started_at")) for item in executions]
    ends = [parse_time(item.get("updated_at")) for item in executions]
    known_starts = [value for value in starts if value is not None]
    known_ends = [value for value in ends if value is not None]
    if len(known_starts) != len(starts) or len(known_ends) != len(ends):
        return "a step execution has no readable started_at/updated_at"
    return Span(
        start=min(known_starts),
        end=max(known_ends),
        operations=tuple(
            sorted(
                f"{item.get('step_index')}/{item.get('operation')}"
                for item in executions
            )
        ),
    )


def overlap_seconds(first: Span, second: Span) -> float:
    """Seconds both spans cover; zero or negative when they do not overlap
    (then minus the gap between them)."""

    return (min(first.end, second.end) - max(first.start, second.start)).total_seconds()


# --------------------------------------------------------------------------- #
# The judgement
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DomainExpectation:
    """One b220 failure domain the case faults once, and its workflow."""

    label: str
    faulted_node: str
    workflow_id: str
    #: The step that must run on the faulted node (RESET_GPU, RESTART_NODE).
    remediation: str = "RESET_GPU"
    #: A job domain names its job; an idle-node domain names none.
    job_id: str = ""
    workload_ids: tuple[str, ...] = ()
    #: Every node of the job's attempt (its healthy peers may be stopped,
    #: restarted and counted with it); an idle-node domain is its node alone.
    allocation: tuple[str, ...] = ()

    @property
    def is_job(self) -> bool:
        return bool(self.job_id)

    @property
    def nodes(self) -> frozenset[str]:
        return frozenset({self.faulted_node, *self.allocation})

    def storm_domain(self, incident_id: str) -> str:
        """The workload domain the breaker counts this domain's nodes under
        (``remediation_storm.storm_domain``): the job, else the incident."""

        return f"job:{self.job_id}" if self.is_job else f"incident:{incident_id}"


@dataclass
class _Findings:
    errors: list[str] = field(default_factory=list)
    inconclusive: list[str] = field(default_factory=list)
    shapes: set[str] = field(default_factory=set)

    def fail(self, shape: str, message: str) -> None:
        self.shapes.add(shape)
        self.errors.append(message)


@dataclass(frozen=True)
class OracleVerdict:
    verdict: str
    outcome: str
    errors: tuple[str, ...]
    inconclusive: tuple[str, ...]
    facts: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "outcome": self.outcome,
            "errors": list(self.errors),
            "inconclusive": list(self.inconclusive),
            "facts": self.facts,
        }


#: Which shape a FAIL reports when several apply: a storm hold or a retired
#: scope explains a missing overlap, never the other way round.
_SHAPE_ORDER = (BREAKER_TRIPPED, SERIALIZED, NOT_PARALLEL, CONTRACT)


def unreadable_verdict(reason: str) -> OracleVerdict:
    """The verdict when the store could not be read at all."""

    return OracleVerdict(INCONCLUSIVE, RECORD_MISSING, (), (reason,), {})


def judge_parallel_domains(
    snapshot: Mapping[str, Any],
    *,
    cluster_id: str,
    domains: Sequence[DomainExpectation],
    started_at: datetime,
    node_group: NodeGroup | None = None,
    overlap: str = OVERLAP_BUDGETED,
    expected_counted_nodes: int | None = None,
) -> OracleVerdict:
    """Did the faulted domains recover as separate, parallel b220 workflows?

    ``snapshot`` is ``read_oracle_snapshot``'s record set read after every
    workflow ended; ``started_at`` the injection's start (a breaker trip or
    release at or after it happened during the case); ``node_group`` the
    group the case ran in (named in a ``domain:`` claim it is the pre-b220
    slot); ``expected_counted_nodes`` the number of nodes the breaker must
    have counted for these workflows.
    """

    if len(domains) < 2:
        raise ValueError("a parallel judgement needs at least two domains")
    if overlap not in {OVERLAP_BUDGETED, OVERLAP_REMEDIATION}:
        raise ValueError(f"unknown overlap span {overlap!r}")
    findings = _Findings()
    facts: dict[str, Any] = {"domains": {}, "overlap": overlap}
    workflows = snapshot.get("workflows") or {}
    incidents = snapshot.get("incidents") or {}
    present: dict[str, dict[str, Any]] = {}
    for domain in domains:
        workflow = workflows.get(domain.workflow_id)
        if not isinstance(workflow, dict):
            findings.inconclusive.append(
                f"{domain.label}: workflow {domain.workflow_id} cannot be read from "
                "the store"
            )
            continue
        incident = incidents.get(str(workflow.get("incident_id") or ""))
        if not isinstance(incident, dict):
            findings.inconclusive.append(
                f"{domain.label}: incident {workflow.get('incident_id')} of "
                f"{domain.workflow_id} cannot be read from the store"
            )
            continue
        present[domain.label] = workflow
        facts["domains"][domain.label] = _judge_domain(
            findings, domain, workflow, incident, cluster_id, node_group
        )
    if len({domain.workflow_id for domain in domains}) != len(domains):
        findings.fail(
            CONTRACT,
            "two domains name one workflow: the faults were merged, not "
            "repaired as separate domains",
        )
    complete = len(present) == len(domains)
    if complete:
        facts["spans"] = _judge_overlap(findings, domains, present, overlap)
    facts["breaker"] = _judge_breaker(
        findings,
        snapshot,
        domains=domains,
        present=present,
        cluster_id=cluster_id,
        started_at=started_at,
        complete=complete,
        expected_counted_nodes=expected_counted_nodes,
    )
    facts["residue"] = _judge_residue(findings, snapshot, domains, started_at)
    return _verdict(findings, facts)


def _verdict(findings: _Findings, facts: dict[str, Any]) -> OracleVerdict:
    if findings.errors:
        outcome = next(
            (shape for shape in _SHAPE_ORDER if shape in findings.shapes), CONTRACT
        )
        verdict = FAIL
    elif findings.inconclusive:
        outcome, verdict = RECORD_MISSING, INCONCLUSIVE
    else:
        outcome, verdict = PARALLEL, PASS
    return OracleVerdict(
        verdict=verdict,
        outcome=outcome,
        errors=tuple(findings.errors),
        inconclusive=tuple(findings.inconclusive),
        facts=facts,
    )


def _judge_domain(
    findings: _Findings,
    domain: DomainExpectation,
    workflow: Mapping[str, Any],
    incident: Mapping[str, Any],
    cluster_id: str,
    node_group: NodeGroup | None,
) -> dict[str, Any]:
    label = domain.label
    if incident.get("cluster_id") != cluster_id:
        findings.fail(
            CONTRACT, f"{label}: incident belongs to {incident.get('cluster_id')}"
        )
    if incident.get("workflow_request_id") != domain.workflow_id:
        findings.fail(
            CONTRACT,
            f"{label}: incident {incident.get('incident_id')} names workflow "
            f"{incident.get('workflow_request_id')}, not {domain.workflow_id}",
        )
    incident_nodes = set(incident.get("node_ids") or [])
    if domain.faulted_node not in incident_nodes or not incident_nodes <= domain.nodes:
        findings.fail(
            CONTRACT,
            f"{label}: incident nodes {sorted(incident_nodes)} are not the faulted "
            f"node {domain.faulted_node}"
            + (f" of job {domain.job_id}" if domain.is_job else ""),
        )
    job_id = str(incident.get("job_id") or "")
    if job_id != domain.job_id:
        findings.fail(
            CONTRACT,
            f"{label}: incident job is {job_id or 'none'}, expected "
            f"{domain.job_id or 'none (an idle node)'}",
        )
    _judge_plan(findings, domain, workflow)
    claims = _judge_claims(findings, domain, workflow, cluster_id, node_group)
    waits = _judge_holds(findings, domain, workflow)
    if workflow.get("status") != "SUCCEEDED":
        findings.fail(
            CONTRACT, f"{label}: workflow ended {workflow.get('status')}, not SUCCEEDED"
        )
    if incident.get("state") != "RECOVERED":
        findings.fail(
            CONTRACT, f"{label}: incident is {incident.get('state')}, not RECOVERED"
        )
    return {
        "workflow_id": domain.workflow_id,
        "incident_id": incident.get("incident_id"),
        "faulted_node": domain.faulted_node,
        "job_id": domain.job_id or None,
        "status": workflow.get("status"),
        "incident_state": incident.get("state"),
        "claims": claims,
        "limits": dict(workflow.get("remediation_budget_limits") or {}),
        "wait_count": int(workflow.get("remediation_budget_wait_count") or 0),
        "waits": waits,
        "held_budget": held_budget_period(workflow),
    }


def held_budget_period(workflow: Mapping[str, Any]) -> dict[str, str | None]:
    """When the workflow first claimed (its budget is held from its first
    admission) and when it ended -- the period no other workflow could hold
    its scopes; a diagnostic beside the step spans the overlap is judged on."""

    def first(kind: str) -> str | None:
        moments = [
            moment
            for event in _records(workflow.get("events"))
            if event.get("kind") == kind
            and (moment := parse_time(event.get("at"))) is not None
        ]
        return min(moments).isoformat() if moments else None

    return {"first_claim_at": first("CLAIM"), "terminal_at": first("TERMINAL")}


def _judge_plan(
    findings: _Findings, domain: DomainExpectation, workflow: Mapping[str, Any]
) -> None:
    label = domain.label
    steps = _records(workflow.get("official_steps"))
    remediation = [
        step for step in steps if step.get("operation") == domain.remediation
    ]
    if [step.get("node_ids") for step in remediation] != [[domain.faulted_node]]:
        findings.fail(
            CONTRACT,
            f"{label}: expected one {domain.remediation} step on "
            f"{domain.faulted_node}, found "
            f"{[step.get('node_ids') for step in remediation]}",
        )
    for step in steps:
        operation = str(step.get("operation"))
        nodes = set(step.get("node_ids") or [])
        allowed = (
            domain.nodes
            if domain.is_job and operation in JOB_WIDE_OPERATIONS
            else {domain.faulted_node}
        )
        if not nodes <= allowed:
            findings.fail(
                CONTRACT,
                f"{label}: {operation} acts on {sorted(nodes - allowed)} beyond "
                f"{'the job' if domain.is_job else 'its own node'}",
            )
    operations = {str(step.get("operation")) for step in steps}
    if domain.is_job:
        missing = sorted(JOB_TOKEN_OPERATIONS - operations)
        if missing:
            findings.fail(CONTRACT, f"{label}: the job workflow has no {missing}")
    elif operations & JOB_TOKEN_OPERATIONS:
        findings.fail(
            CONTRACT,
            f"{label}: an idle node's workflow stops or restarts a workload: "
            f"{sorted(operations & JOB_TOKEN_OPERATIONS)}",
        )
    done = {
        str(item.get("operation"))
        for item in official_executions(workflow)
        if item.get("status") == "SUCCEEDED"
    }
    for operation in (
        domain.remediation,
        *(sorted(JOB_TOKEN_OPERATIONS) if domain.is_job else ()),
    ):
        if operation not in done:
            findings.fail(
                CONTRACT, f"{label}: {operation} has no SUCCEEDED step execution"
            )
    reconciled = [
        event.get("code")
        for event in _records(workflow.get("events"))
        if event.get("kind") == "OPERATOR_RECONCILED"
    ]
    if reconciled:
        findings.fail(
            CONTRACT,
            f"{label}: an operator reconciled the workflow ({reconciled}); the "
            "recovery must be the product's own",
        )


def _judge_claims(
    findings: _Findings,
    domain: DomainExpectation,
    workflow: Mapping[str, Any],
    cluster_id: str,
    node_group: NodeGroup | None,
) -> list[str]:
    label = domain.label
    claims = sorted(
        str(scope) for scope in workflow.get("remediation_budget_claims") or []
    )
    if not claims:
        findings.fail(CONTRACT, f"{label}: the workflow claimed no remediation budget")
        return claims
    own = node_scope(cluster_id, domain.faulted_node)
    required = {own}
    allowed = {own, *(node_scope(cluster_id, node) for node in domain.allocation)}
    if domain.is_job:
        tokens = job_scopes(cluster_id, domain.job_id, domain.workload_ids)
        required |= tokens
        allowed |= tokens
    partitions = {
        f"domain:{cluster_id}:{value}" for value in fabric_partitions(workflow)
    }
    allowed |= partitions
    missing = sorted(required - set(claims))
    if missing:
        findings.fail(CONTRACT, f"{label}: claims lack {missing}")
    for scope in sorted(set(claims) - allowed):
        kind = scope_kind(scope)
        if kind in RETIRED_SCOPE_KINDS:
            findings.fail(
                SERIALIZED,
                f"{label}: claims the retired {kind} scope {scope} (pre-b220 "
                "fixed limits)",
            )
        elif kind == "domain":
            group = (
                " -- the node group's pre-b220 slot"
                if node_group is not None
                and scope == f"domain:{cluster_id}:{node_group.name}"
                else ", a node-label domain no step's fabric partition names"
            )
            findings.fail(SERIALIZED, f"{label}: claims {scope}{group}")
        else:
            findings.fail(
                CONTRACT,
                f"{label}: claims {scope}, outside "
                + ("its job's nodes and tokens" if domain.is_job else "its own node"),
            )
    limits = workflow.get("remediation_budget_limits") or {}
    if own in claims and limits.get(own) not in (None, 1):
        findings.fail(CONTRACT, f"{label}: {own} limit is {limits.get(own)}, not 1")
    return claims


def _judge_holds(
    findings: _Findings, domain: DomainExpectation, workflow: Mapping[str, Any]
) -> list[dict[str, Any]]:
    label = domain.label
    waits = budget_waits(workflow)
    count = int(workflow.get("remediation_budget_wait_count") or 0)
    for event in storm_events(workflow):
        findings.fail(
            BREAKER_TRIPPED,
            f"{label}: the storm breaker held the workflow ({event['code']} at "
            f"{event['at']}): {event['reason']}",
        )
    scopes = [str(item.get("scope") or "") for item in waits]
    last = str(workflow.get("remediation_budget_last_blocked_scope") or "")
    if last:
        scopes.append(last)
    for scope in dict.fromkeys(scopes):
        kind = scope_kind(scope) if scope else ""
        if scope.startswith(STORM_SCOPE_PREFIX):
            findings.fail(
                BREAKER_TRIPPED,
                f"{label}: waited on the storm breaker ({scope}); its work did not "
                "start with the other domains'",
            )
        elif kind == "domain" or kind in RETIRED_SCOPE_KINDS:
            findings.fail(
                SERIALIZED,
                f"{label}: was held on {scope}: the pre-b220 node-group serialization",
            )
        elif scope:
            findings.fail(
                CONTRACT,
                f"{label}: waited on its own {kind} slot {scope}: another workflow "
                "held it",
            )
        else:
            findings.fail(CONTRACT, f"{label}: a budget wait names no scope")
    if count and not waits:
        if history_truncated(workflow):
            findings.inconclusive.append(
                f"{label}: refused {count} time(s); the bounded event history no "
                "longer shows the scope"
            )
        else:
            findings.inconclusive.append(
                f"{label}: refused {count} time(s) without a {WAIT_CODE} event "
                "(a release before the wait-age change refused it)"
            )
    return waits


def _domain_span(
    domain: DomainExpectation, workflow: Mapping[str, Any], overlap: str
) -> Span | str:
    if overlap == OVERLAP_REMEDIATION:
        operations = frozenset({domain.remediation})
    else:
        operations = BUDGETED_OPERATIONS
    return execution_span(official_executions(workflow, operations))


def _judge_overlap(
    findings: _Findings,
    domains: Sequence[DomainExpectation],
    present: Mapping[str, Mapping[str, Any]],
    overlap: str,
) -> dict[str, Any]:
    spans: dict[str, Span] = {}
    report: dict[str, Any] = {}
    for domain in domains:
        span = _domain_span(domain, present[domain.label], overlap)
        if isinstance(span, str):
            findings.fail(
                NOT_PARALLEL,
                f"{domain.label}: no {overlap} span to judge the overlap on: {span}",
            )
            continue
        spans[domain.label] = span
        report[domain.label] = span.as_dict()
    pairs: dict[str, float] = {}
    for first, second in combinations(
        [item for item in domains if item.label in spans], 2
    ):
        seconds = overlap_seconds(spans[first.label], spans[second.label])
        pairs[f"{first.label}|{second.label}"] = seconds
        if seconds <= 0:
            findings.fail(
                NOT_PARALLEL,
                f"{first.label} and {second.label}: the {overlap} steps did not "
                f"overlap ({-seconds:.1f}s apart): {spans[first.label].as_dict()} vs "
                f"{spans[second.label].as_dict()}",
            )
    return {"spans": report, "overlap_seconds": pairs}


def _judge_breaker(
    findings: _Findings,
    snapshot: Mapping[str, Any],
    *,
    domains: Sequence[DomainExpectation],
    present: Mapping[str, Mapping[str, Any]],
    cluster_id: str,
    started_at: datetime,
    complete: bool,
    expected_counted_nodes: int | None,
) -> dict[str, Any]:
    if snapshot.get("storm_supported") is not True:
        findings.inconclusive.append(
            "the deployed control plane has no remediation storm breaker: the "
            "closed breaker cannot be shown (deploy b220 first)"
        )
        return {}
    breaker = snapshot.get("breaker")
    if not isinstance(breaker, dict):
        findings.inconclusive.append(
            f"the storm breaker record of {cluster_id} is absent: no first "
            "admission was judged by it, so its state and count cannot be shown"
        )
        return {}
    state = breaker.get("state")
    if state != "CLOSED":
        findings.fail(
            BREAKER_TRIPPED,
            f"the storm breaker of {cluster_id} is {state} (trip "
            f"{breaker.get('trip_id')}: nodes {breaker.get('trip_node_ids')}, "
            f"{breaker.get('trip_domains')} domains, threshold "
            f"{breaker.get('trip_threshold')}, held {breaker.get('trip_workflow_id')})",
        )
    for key in ("tripped_at", "released_at", "aborted_at"):
        moment = parse_time(breaker.get(key))
        if moment is not None and moment >= started_at:
            findings.fail(
                BREAKER_TRIPPED,
                f"the storm breaker of {cluster_id} was {key.removesuffix('_at')} "
                f"during the case ({moment.isoformat()}, trip {breaker.get('trip_id')})",
            )
    held = list(snapshot.get("storm_held_workflow_ids") or [])
    if held:
        findings.fail(BREAKER_TRIPPED, f"the storm breaker holds workflows {held}")
    ours = {domain.workflow_id: domain for domain in domains}
    window = [
        entry
        for entry in _records(breaker.get("window"))
        if entry.get("workflow_id") in ours
    ]
    counted: dict[str, list[str]] = {}
    counted_domains: set[str] = set()
    sources: dict[str, str] = {}
    # The breaker's own entries are what it counted. A later admission
    # elsewhere prunes entries older than the window from the record, and
    # may prune one domain's and keep another's; for a domain the record no
    # longer lists, its claims are what was counted, since storm_candidate
    # counts the ``node:`` scopes of a first admission.
    for entry in window:
        domain = ours[str(entry["workflow_id"])]
        counted.setdefault(domain.label, []).append(str(entry.get("node_id")))
        counted_domains.add(str(entry.get("domain") or ""))
        sources[domain.label] = "breaker window"
        workflow = present.get(domain.label)
        expected = domain.storm_domain(str((workflow or {}).get("incident_id") or ""))
        if workflow is not None and entry.get("domain") != expected:
            findings.fail(
                CONTRACT,
                f"{domain.label}: the breaker counted it in workload domain "
                f"{entry.get('domain')}, not {expected}",
            )
    for domain in domains:
        workflow = present.get(domain.label)
        if workflow is not None and domain.label not in counted:
            counted[domain.label] = _claimed_nodes(workflow, cluster_id)
            sources[domain.label] = "claims"
    for domain in domains:
        nodes = set(counted.get(domain.label) or [])
        if (
            domain.label in present
            and nodes
            and not (domain.faulted_node in nodes and nodes <= domain.nodes)
        ):
            findings.fail(
                CONTRACT,
                f"{domain.label}: the breaker counted {sorted(nodes)} for it, not "
                f"its faulted node {domain.faulted_node}"
                + (" and its job's peers" if domain.is_job else ""),
            )
    total = sorted({node for nodes in counted.values() for node in nodes})
    threshold = int(breaker.get("threshold") or 0) or storm_threshold(
        snapshot.get("cluster_gpu_agents")
    )
    if (
        complete
        and expected_counted_nodes is not None
        and len(total) != expected_counted_nodes
    ):
        findings.fail(
            CONTRACT,
            f"the breaker counted {len(total)} nodes {total} for the case, "
            f"expected {expected_counted_nodes}",
        )
    admitted = [parse_time(entry.get("admitted_at")) for entry in window]
    last = max((value for value in admitted if value is not None), default=None)
    return {
        "state": state,
        "threshold": threshold,
        "cluster_gpu_agents": snapshot.get("cluster_gpu_agents"),
        "counted_source": sources,
        "counted_nodes": {label: sorted(nodes) for label, nodes in counted.items()},
        "counted_total": len(total),
        "counted_domains": sorted(counted_domains),
        "headroom": threshold - len(total),
        "trip_id": breaker.get("trip_id"),
        # When the case's own entries stop counting: the next destructive case
        # on this cluster starts after it, or the two cases count together.
        "window_clear_at": (
            (last + timedelta(seconds=STORM_WINDOW_SECONDS)).isoformat()
            if last is not None
            else None
        ),
    }


def _claimed_nodes(workflow: Mapping[str, Any], cluster_id: str) -> list[str]:
    prefix = f"node:{cluster_id}:"
    return sorted(
        str(scope)[len(prefix) :]
        for scope in workflow.get("remediation_budget_claims") or []
        if str(scope).startswith(prefix)
    )


def _judge_residue(
    findings: _Findings,
    snapshot: Mapping[str, Any],
    domains: Sequence[DomainExpectation],
    started_at: datetime,
) -> dict[str, Any]:
    """Open workflows and incidents on the case's nodes the case created."""

    ours = {domain.workflow_id for domain in domains}

    def of_case(item: Mapping[str, Any]) -> bool:
        created = parse_time(item.get("created_at"))
        return item.get("workflow_request_id") in ours or (
            created is not None and created >= started_at
        )

    active = [
        item
        for item in _records(snapshot.get("active_workflow_incidents"))
        if of_case(item)
    ]
    open_incidents = [
        item for item in _records(snapshot.get("open_incidents")) if of_case(item)
    ]
    for item in active:
        findings.fail(
            CONTRACT,
            f"residue: workflow {item.get('workflow_request_id')} is still "
            f"{item.get('status')} on {item.get('node_ids')}",
        )
    for item in open_incidents:
        findings.fail(
            CONTRACT,
            f"residue: incident {item.get('incident_id')} is still "
            f"{item.get('state')} on {item.get('node_ids')}",
        )
    return {"active_workflows": active, "open_incidents": open_incidents}


def breaker_premise_errors(
    snapshot: Mapping[str, Any],
    *,
    cluster_id: str,
    targets: Sequence[str],
    now: datetime,
) -> list[str]:
    """Why the case's own admissions could trip the breaker before it starts.

    The breaker must be CLOSED with nothing held, and the nodes it still counts
    from the last ``STORM_WINDOW_SECONDS`` together with the case's ``targets``
    must stay below the threshold -- a destructive case on another node of the
    cluster inside the window would otherwise make the case's second
    admission the trip.
    """

    if snapshot.get("storm_supported") is not True:
        return [
            "the deployed control plane has no remediation storm breaker "
            "(deploy b220 first)"
        ]
    errors: list[str] = []
    breaker = snapshot.get("breaker")
    record = breaker if isinstance(breaker, dict) else {}
    if record and record.get("state") != "CLOSED":
        errors.append(
            f"the storm breaker of {cluster_id} is {record.get('state')} (trip "
            f"{record.get('trip_id')}); an operator must release it first"
        )
    held = list(snapshot.get("storm_held_workflow_ids") or [])
    if held:
        errors.append(f"the storm breaker holds workflows {held}")
    horizon = now - timedelta(seconds=STORM_WINDOW_SECONDS)
    recent: dict[str, datetime] = {}
    for entry in _records(record.get("window")):
        moment = parse_time(entry.get("admitted_at"))
        if moment is not None and moment > horizon:
            node = str(entry.get("node_id"))
            recent[node] = max(moment, recent.get(node, moment))
    threshold = storm_threshold(snapshot.get("cluster_gpu_agents"))
    counted = set(recent) | set(targets)
    if len(counted) >= threshold:
        clear = (
            f"start after {(max(recent.values()) + timedelta(seconds=STORM_WINDOW_SECONDS)).isoformat()}"
            if recent
            else "the case alone reaches it"
        )
        errors.append(
            f"the storm breaker still counts {sorted(recent)} from the last "
            f"{STORM_WINDOW_SECONDS}s; with the case's {sorted(targets)} that is "
            f"{len(counted)} nodes, at or past its threshold {threshold}: {clear}"
        )
    return errors
