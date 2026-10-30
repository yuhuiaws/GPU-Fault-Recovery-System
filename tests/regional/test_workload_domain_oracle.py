"""The shared oracle of b220's workload failure domains.

Three layouts of independent failure domains of one node group are judged on
synthetic store records built from the product models (dumped the way the
oracle's store read dumps them):

* two idle nodes (GF-REGIONAL-DESTR-027);
* two jobs, a reboot-class fault on one node of each (GF-REGIONAL-DESTR-028):
  two two-node jobs, or a two-node and a single-node job;
* one two-node job (reboot-class fault) beside one idle node (reset).

For each: the b220 parallel outcome is PASS; a pre-b220 serialized outcome --
the second workflow held on the node group's ``domain:`` slot, its steps after
the first one's -- is FAIL; a record that cannot be read is INCONCLUSIVE; and
faults that tripped the storm breaker are a FAIL of their own shape, never the
parallel PASS. Then the records the b220 claim path itself writes: two idle
nodes end PASS, and a two-node job's healthy workload-stop peer is NOT counted
(b223) -- only its rebooted node is -- so a two-node job counts exactly as a
single-node one, two such domains stay at two counted nodes below the floor of
3, and both run in parallel. Last, the node-group precondition and the breaker
premise.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from gpu_fault.cluster_scope import EKS_NODEGROUP_LABEL, HYPERPOD_INSTANCE_GROUP_LABEL
from gpu_fault.execution.remediation_budget import (
    RemediationBudgetPolicy,
    claims_in_cluster,
)
from gpu_fault.models import (
    FaultIncident,
    IncidentState,
    WorkflowEvent,
    WorkflowEventKind,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepExecution,
    WorkflowStepSpec,
    WorkflowStepStatus,
)
from gpu_fault.remediation_storm import (
    RemediationStormBreaker,
    StormBreakerState,
    StormEntry,
)
from gpu_fault.store import InMemoryStore
from scripts.e2e.regional import workload_domain_oracle as oracle
from tests._builders import fault_incident, workflow_request, workflow_step
from tests.execution._budget_hold_support import (
    ScriptedAdapter,
    budget_dispatcher,
    seed,
)

Op = WorkflowOperation
CLUSTER = "gpu-cluster"
GROUP = oracle.NodeGroup(
    label=EKS_NODEGROUP_LABEL,
    name="gpu-group",
    nodes=("node-a", "node-b", "node-c", "node-d"),
)
T0 = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(hours=1)
STARTED = T0 - timedelta(seconds=5)
RESET_PLAN = (
    Op.FREEZE_EVIDENCE,
    Op.MARK_UNSCHEDULABLE,
    Op.QUIESCE_GPU_SERVICES,
    Op.VERIFY_NO_GPU_CLIENTS,
    Op.RESET_GPU,
    Op.RESTORE_GPU_SERVICES,
    Op.VALIDATE_GPU,
    Op.RESTORE_SCHEDULING,
)
JOB_REBOOT_PLAN = (
    Op.FREEZE_EVIDENCE,
    Op.MARK_UNSCHEDULABLE,
    Op.STOP_WORKLOADS,
    Op.RESTART_NODE,
    Op.VALIDATE_GPU,
    Op.RESTORE_SCHEDULING,
    Op.RESTART_WORKLOAD,
)
JOB_WIDE = {Op.STOP_WORKLOADS, Op.RESTART_WORKLOAD}
STEP_SECONDS = 10
# The scopes a release before b220 claimed beside the node scope.
PRE_B220 = (
    "region",
    f"cluster:{CLUSTER}",
    f"class:{CLUSTER}:GPU_NODE_MUTATION",
    f"domain:{CLUSTER}:{GROUP.name}",
)
NODE_GROUP_SCOPE = f"domain:{CLUSTER}:{GROUP.name}"
STORM = f"storm:{CLUSTER}"


def at(seconds: float) -> datetime:
    return T0 + timedelta(seconds=seconds)


# --------------------------------------------------------------------------- #
# Synthetic records
# --------------------------------------------------------------------------- #
@dataclass
class Part:
    """One failure domain's expectation and its two records."""

    expectation: oracle.DomainExpectation
    workflow: dict[str, Any]
    incident: dict[str, Any]
    window: list[StormEntry]


def _steps(node: str, job: str, allocation: tuple[str, ...]) -> list[WorkflowStepSpec]:
    if not job:
        return [
            WorkflowStepSpec(operation=op, execution_owner="owner-a", node_ids=[node])
            for op in RESET_PLAN
        ]
    workload = f"default/pytorchjob/{job}"
    return [
        WorkflowStepSpec(
            operation=op,
            execution_owner="owner-a",
            node_ids=list(allocation) if op in JOB_WIDE else [node],
            workload_ids=[workload] if op in JOB_WIDE else [],
            parameters={"job_id": job} if op is Op.RESTART_WORKLOAD else {},
        )
        for op in JOB_REBOOT_PLAN
    ]


def part(
    label: str,
    node: str,
    *,
    start: float,
    job: str = "",
    peers: tuple[str, ...] = (),
    pre_b220: bool = False,
    held_on: str = "",
    ran: bool = True,
) -> Part:
    """A domain whose steps run ``STEP_SECONDS`` each from ``start``.

    ``pre_b220`` adds the retired scopes to its claims, ``held_on`` the budget
    wait an older release (or the breaker) recorded on it; ``ran=False`` is a
    workflow the breaker held at its first admission: PENDING, no claims, no
    step ran.
    """

    allocation = (node, *peers)
    steps = _steps(node, job, allocation)
    workload = f"default/pytorchjob/{job}" if job else ""
    claims = {f"node:{CLUSTER}:{item}": 1 for item in allocation}
    if job:
        claims[f"job:{CLUSTER}:{workload}"] = 1
        claims[f"job:{CLUSTER}:job-id/{job}"] = 1
    if pre_b220:
        claims.update({scope: 1 for scope in PRE_B220})
    request_id = f"workflow-{label}"
    incident_id = f"incident-{label}"
    events = []
    if held_on:
        events.append(
            WorkflowEvent(
                kind=WorkflowEventKind.HOLD,
                at=at(1),
                code=oracle.WAIT_CODE,
                actor="remediation-budget",
                reason=f"held on {held_on}",
                details={"reason": oracle.WAIT_CODE, "scope": held_on},
            )
        )
    end = start + STEP_SECONDS * len(steps)
    if ran:
        events.append(WorkflowEvent(kind=WorkflowEventKind.CLAIM, at=at(start - 1)))
        events.append(WorkflowEvent(kind=WorkflowEventKind.TERMINAL, at=at(end + 1)))
    workflow = WorkflowRequest(
        request_id=request_id,
        incident_id=incident_id,
        status=WorkflowStatus.SUCCEEDED if ran else WorkflowStatus.PENDING,
        fencing_token=3,
        official_steps=steps,
        step_executions=[
            WorkflowStepExecution(
                step_index=index,
                operation=step.operation,
                status=WorkflowStepStatus.SUCCEEDED,
                phase="official",
                started_at=at(start + STEP_SECONDS * index),
                updated_at=at(start + STEP_SECONDS * index + STEP_SECONDS - 1),
            )
            for index, step in enumerate(steps)
        ]
        if ran
        else [],
        events=events,
        remediation_budget_claims=sorted(claims) if ran else [],
        remediation_budget_limits=claims if ran else {},
        remediation_budget_wait_count=3 if held_on else 0,
        remediation_budget_last_blocked_scope=None if ran else (held_on or None),
        remediation_budget_last_blocked_reason=None if ran else "held",
        created_at=at(-2),
    )
    incident = FaultIncident(
        incident_id=incident_id,
        event_id=f"event-{label}",
        event_type="XID",
        cluster_id=CLUSTER,
        node_ids=[node],
        job_id=job or None,
        policy_version="610",
        policy_source="NVIDIA",
        state=IncidentState.RECOVERED if ran else IncidentState.ACTION_PENDING,
        workflow_request_id=request_id,
        created_at=at(-2),
    )
    domain = f"job:{job}" if job else f"incident:{incident_id}"
    # b223: the breaker counts the faulted node only. A job's healthy peer,
    # whose claim is a workload stop/restart widened across the allocation, is
    # not entering destructive NODE remediation; an idle domain is its node.
    window = (
        [
            StormEntry(
                node_id=node,
                domain=domain,
                workflow_id=request_id,
                admitted_at=at(start - 1),
            )
        ]
        if ran and not pre_b220
        else []
    )
    expectation = oracle.DomainExpectation(
        label=label,
        faulted_node=node,
        workflow_id=request_id,
        remediation="RESTART_NODE" if job else "RESET_GPU",
        job_id=job,
        workload_ids=(workload,) if job else (),
        allocation=allocation if job else (),
    )
    return Part(
        expectation=expectation,
        workflow=workflow.model_dump(mode="json"),
        incident=incident.model_dump(mode="json"),
        window=window,
    )


def snapshot(
    *parts: Part,
    threshold: int = 3,
    agents: int = 4,
    storm_supported: bool = True,
    breaker: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = breaker
    if record is None and storm_supported:
        record = RemediationStormBreaker(
            cluster_id=CLUSTER,
            window=[entry for item in parts for entry in item.window],
            threshold=threshold,
            updated_at=at(10),
        ).model_dump(mode="json")
    return {
        "read_at": at(600).isoformat(),
        "cluster_id": CLUSTER,
        "workflows": {item.expectation.workflow_id: item.workflow for item in parts},
        "incidents": {item.incident["incident_id"]: item.incident for item in parts},
        "missing_workflows": [],
        "missing_incidents": [],
        "storm_supported": storm_supported,
        "breaker": record if storm_supported else None,
        "storm_held_workflow_ids": [],
        "active_workflow_incidents": [],
        "open_incidents": [],
        "cluster_gpu_agents": agents,
    }


def tripped(
    parts: tuple[Part, ...], held: Part, *, nodes: list[str], threshold: int = 3
) -> dict[str, Any]:
    """The records a trip leaves: the breaker TRIPPED by ``held``'s first
    admission, which waits PENDING on ``storm:<cluster>`` with the trip event."""

    value = snapshot(*parts, threshold=threshold)
    value["breaker"] = RemediationStormBreaker(
        cluster_id=CLUSTER,
        state=StormBreakerState.TRIPPED,
        window=[entry for item in parts for entry in item.window],
        threshold=threshold,
        trip_id=f"{CLUSTER}@{at(6).isoformat()}",
        tripped_at=at(6),
        trip_node_ids=nodes,
        trip_domains=2,
        trip_threshold=threshold,
        trip_cluster_nodes=4,
        trip_workflow_id=held.expectation.workflow_id,
        updated_at=at(6),
    ).model_dump(mode="json")
    value["storm_held_workflow_ids"] = [held.expectation.workflow_id]
    held.workflow["events"].append(
        WorkflowEvent(
            kind=WorkflowEventKind.HOLD,
            at=at(6),
            code=oracle.TRIPPED_CODE,
            actor="remediation-storm-breaker",
            reason="remediation storm breaker tripped",
        ).model_dump(mode="json")
    )
    return value


def judge(
    value: dict[str, Any],
    parts: tuple[Part, ...],
    *,
    overlap: str = oracle.OVERLAP_BUDGETED,
    counted: int | None = None,
) -> oracle.OracleVerdict:
    return oracle.judge_parallel_domains(
        value,
        cluster_id=CLUSTER,
        domains=[item.expectation for item in parts],
        started_at=STARTED,
        node_group=GROUP,
        overlap=overlap,
        expected_counted_nodes=counted,
    )


def two_idle(**second: Any) -> tuple[Part, Part]:
    return part("node-a", "node-a", start=0), part(
        "node-b", "node-b", **{"start": 4, **second}
    )


def two_jobs(**second: Any) -> tuple[Part, Part]:
    return part("job-a", "node-a", start=0, job="job-a", peers=("node-b",)), part(
        "job-b",
        "node-c",
        **{"start": 5, "job": "job-b", "peers": ("node-d",), **second},
    )


def two_node_and_single_node_jobs(**second: Any) -> tuple[Part, Part]:
    return part("job-a", "node-a", start=0, job="job-a", peers=("node-b",)), part(
        "job-b", "node-c", **{"start": 5, "job": "job-b", **second}
    )


def job_and_idle(**idle: Any) -> tuple[Part, Part]:
    return part("job", "node-a", start=0, job="job-a", peers=("node-b",)), part(
        "idle", "node-c", **{"start": 6, **idle}
    )


# --------------------------------------------------------------------------- #
# Two idle nodes of one node group (GF-REGIONAL-DESTR-027)
# --------------------------------------------------------------------------- #
def test_two_idle_nodes_in_parallel_are_the_pass_shape() -> None:
    parts = two_idle()

    verdict = judge(snapshot(*parts), parts, counted=2)

    assert (verdict.verdict, verdict.outcome) == (oracle.PASS, oracle.PARALLEL), (
        verdict.errors,
        verdict.inconclusive,
    )
    breaker = verdict.facts["breaker"]
    assert breaker["counted_nodes"] == {"node-a": ["node-a"], "node-b": ["node-b"]}
    assert breaker["counted_domains"] == [
        "incident:incident-node-a",
        "incident:incident-node-b",
    ]
    assert (breaker["counted_total"], breaker["threshold"]) == (2, 3)
    assert verdict.facts["spans"]["overlap_seconds"]["node-a|node-b"] > 0
    assert verdict.facts["domains"]["node-a"]["claims"] == ["node:gpu-cluster:node-a"]


def test_the_pre_b220_node_group_serialization_fails_as_serialized() -> None:
    """Before b220: both claim the node group's slot, the second waits behind
    it, and its reset starts only after the first workflow ended."""

    first = part("node-a", "node-a", start=0, pre_b220=True)
    second = part(
        "node-b", "node-b", start=120, pre_b220=True, held_on=NODE_GROUP_SCOPE
    )

    verdict = judge(snapshot(first, second, storm_supported=False), (first, second))

    assert (verdict.verdict, verdict.outcome) == (oracle.FAIL, oracle.SERIALIZED)
    text = "\n".join(verdict.errors)
    assert f"node-b: was held on {NODE_GROUP_SCOPE}" in text
    assert f"claims {NODE_GROUP_SCOPE} -- the node group's pre-b220 slot" in text
    assert "node-a and node-b: the budgeted steps did not overlap" in text


def test_a_missing_workflow_record_is_inconclusive() -> None:
    parts = two_idle()
    value = snapshot(*parts)
    del value["workflows"]["workflow-node-b"]
    value["missing_workflows"] = ["workflow-node-b"]

    verdict = judge(value, parts, counted=2)

    assert (verdict.verdict, verdict.outcome) == (
        oracle.INCONCLUSIVE,
        oracle.RECORD_MISSING,
    )
    assert verdict.errors == ()
    assert verdict.inconclusive == (
        "node-b: workflow workflow-node-b cannot be read from the store",
    )


def test_a_missing_incident_or_breaker_record_is_inconclusive() -> None:
    parts = two_idle()
    value = snapshot(*parts)
    del value["incidents"]["incident-node-a"]
    value["breaker"] = None

    verdict = judge(value, parts, counted=2)

    assert verdict.verdict == oracle.INCONCLUSIVE
    assert any("incident incident-node-a" in item for item in verdict.inconclusive), (
        verdict.inconclusive
    )
    assert any("storm breaker record" in item for item in verdict.inconclusive), (
        verdict.inconclusive
    )


def test_work_that_did_not_overlap_is_not_parallel_even_without_a_hold() -> None:
    first, second = two_idle(start=200)

    verdict = judge(snapshot(first, second), (first, second), counted=2)

    assert (verdict.verdict, verdict.outcome) == (oracle.FAIL, oracle.NOT_PARALLEL)


def test_a_fabric_partition_its_own_step_names_is_the_only_domain_allowed() -> None:
    first, second = two_idle()
    step = first.workflow["official_steps"][4]
    step["parameters"] = {"fabric_partition": "partition-7"}
    first.workflow["remediation_budget_claims"].append(f"domain:{CLUSTER}:partition-7")

    allowed = judge(snapshot(first, second), (first, second), counted=2)
    first.workflow["remediation_budget_claims"].append(f"domain:{CLUSTER}:rack-3")
    refused = judge(snapshot(first, second), (first, second), counted=2)

    assert allowed.verdict == oracle.PASS, allowed.errors
    assert (refused.verdict, refused.outcome) == (oracle.FAIL, oracle.SERIALIZED)
    assert "no step's fabric partition names" in refused.errors[0]


def test_an_idle_workflow_waiting_on_its_own_node_slot_is_a_contract_failure() -> None:
    parts = two_idle(held_on=f"node:{CLUSTER}:node-b")

    verdict = judge(snapshot(*parts), parts, counted=2)

    assert (verdict.verdict, verdict.outcome) == (oracle.FAIL, oracle.CONTRACT)
    assert "waited on its own node slot" in verdict.errors[0]


def test_residue_the_case_left_fails_and_older_records_are_not_its_residue() -> None:
    parts = two_idle()
    value = snapshot(*parts)
    value["open_incidents"] = [
        {
            "incident_id": "incident-old",
            "state": "ESCALATED",
            "node_ids": ["node-a"],
            "created_at": at(-3600).isoformat(),
        }
    ]
    clean = judge(value, parts, counted=2)
    value["active_workflow_incidents"] = [
        {
            "incident_id": "incident-new",
            "workflow_request_id": "workflow-new",
            "status": "PENDING",
            "node_ids": ["node-b"],
            "created_at": at(30).isoformat(),
        }
    ]
    stained = judge(value, parts, counted=2)

    assert clean.verdict == oracle.PASS, clean.errors
    assert (stained.verdict, stained.outcome) == (oracle.FAIL, oracle.CONTRACT)
    assert stained.errors == (
        "residue: workflow workflow-new is still PENDING on ['node-b']",
    )


def test_a_pruned_window_entry_is_counted_from_the_workflow_steps() -> None:
    """A later admission elsewhere pruned one domain's entry and kept the
    other's: the pruned one's budgeted non-workload steps are what the breaker
    counted (b223)."""

    first, second = two_idle()
    first.window = []

    verdict = judge(snapshot(first, second), (first, second), counted=2)

    assert verdict.verdict == oracle.PASS, verdict.errors
    assert verdict.facts["breaker"]["counted_source"] == {
        "node-a": "workflow steps",
        "node-b": "breaker window",
    }


def test_a_breaker_that_tripped_during_the_case_is_never_the_pass() -> None:
    parts = two_idle()
    value = snapshot(*parts)
    value["breaker"]["tripped_at"] = at(3).isoformat()
    value["breaker"]["released_at"] = at(400).isoformat()

    verdict = judge(value, parts, counted=2)

    assert (verdict.verdict, verdict.outcome) == (oracle.FAIL, oracle.BREAKER_TRIPPED)


def test_a_refusal_history_that_was_truncated_is_inconclusive() -> None:
    first, second = two_idle()
    second.workflow["remediation_budget_wait_count"] = 2
    second.workflow["events"].append(
        WorkflowEvent(kind=WorkflowEventKind.HISTORY_TRUNCATED, at=at(0)).model_dump(
            mode="json"
        )
    )

    verdict = judge(snapshot(first, second), (first, second), counted=2)

    assert verdict.verdict == oracle.INCONCLUSIVE
    assert "no longer shows the scope" in verdict.inconclusive[0]


def test_an_unreadable_store_is_inconclusive_and_one_domain_is_refused() -> None:
    assert oracle.unreadable_verdict("exec refused").as_dict() == {
        "verdict": oracle.INCONCLUSIVE,
        "outcome": oracle.RECORD_MISSING,
        "errors": [],
        "inconclusive": ["exec refused"],
        "facts": {},
    }
    with pytest.raises(ValueError, match="at least two domains"):
        oracle.judge_parallel_domains(
            {},
            cluster_id=CLUSTER,
            domains=[two_idle()[0].expectation],
            started_at=STARTED,
        )


# --------------------------------------------------------------------------- #
# Two two-node jobs, a reboot-class fault on one node of each
# --------------------------------------------------------------------------- #
def test_two_jobs_in_parallel_carry_the_job_scopes() -> None:
    """The PASS shape. Each job claims both its nodes (``node:`` keeps a second
    workflow off either) and its job tokens, but only its one rebooted node is
    counted, so two jobs stay at two counted nodes -- below the floor of 3 on
    the four-node group (b223)."""

    parts = two_jobs()

    verdict = judge(
        snapshot(*parts), parts, overlap=oracle.OVERLAP_REMEDIATION, counted=2
    )

    assert (verdict.verdict, verdict.outcome) == (oracle.PASS, oracle.PARALLEL), (
        verdict.errors
    )
    assert verdict.facts["domains"]["job-a"]["claims"] == [
        "job:gpu-cluster:default/pytorchjob/job-a",
        "job:gpu-cluster:job-id/job-a",
        "node:gpu-cluster:node-a",
        "node:gpu-cluster:node-b",
    ]
    assert verdict.facts["breaker"]["counted_total"] == 2
    assert verdict.facts["spans"]["overlap_seconds"]["job-a|job-b"] > 0


def test_a_job_workflow_without_its_job_tokens_fails() -> None:
    first, second = two_jobs()
    second.workflow["remediation_budget_claims"].remove(f"job:{CLUSTER}:job-id/job-b")

    verdict = judge(
        snapshot(first, second, threshold=5),
        (first, second),
        overlap=oracle.OVERLAP_REMEDIATION,
    )

    assert (verdict.verdict, verdict.outcome) == (oracle.FAIL, oracle.CONTRACT)
    assert verdict.errors == ("job-b: claims lack ['job:gpu-cluster:job-id/job-b']",)


def test_two_jobs_serialized_on_the_node_group_slot_fail_as_serialized() -> None:
    first = part(
        "job-a", "node-a", start=0, job="job-a", peers=("node-b",), pre_b220=True
    )
    second = part(
        "job-b",
        "node-c",
        start=150,
        job="job-b",
        peers=("node-d",),
        pre_b220=True,
        held_on=NODE_GROUP_SCOPE,
    )

    verdict = judge(
        snapshot(first, second, storm_supported=False),
        (first, second),
        overlap=oracle.OVERLAP_REMEDIATION,
    )

    assert (verdict.verdict, verdict.outcome) == (oracle.FAIL, oracle.SERIALIZED)
    assert any(f"was held on {NODE_GROUP_SCOPE}" in item for item in verdict.errors), (
        verdict.errors
    )


def test_two_jobs_with_a_missing_record_are_inconclusive() -> None:
    parts = two_jobs()
    value = snapshot(*parts)
    del value["workflows"]["workflow-job-a"]

    verdict = judge(value, parts, overlap=oracle.OVERLAP_REMEDIATION, counted=2)

    assert (verdict.verdict, verdict.outcome) == (
        oracle.INCONCLUSIVE,
        oracle.RECORD_MISSING,
    )


def test_two_jobs_held_by_a_tripped_breaker_get_their_own_verdict() -> None:
    """A correlated storm from other faults on the cluster has tripped the
    breaker, which holds the second job's first admission. One workflow ran;
    whatever tripped it, this is never the parallel PASS."""

    first, second = two_jobs(ran=False, held_on=STORM)
    value = tripped((first, second), second, nodes=["node-a", "node-c", "node-x"])

    verdict = judge(value, (first, second), overlap=oracle.OVERLAP_REMEDIATION)

    assert verdict.verdict == oracle.FAIL
    assert verdict.outcome == oracle.BREAKER_TRIPPED
    assert verdict.outcome != oracle.PARALLEL
    text = "\n".join(verdict.errors)
    assert f"job-b: waited on the storm breaker ({STORM})" in text
    assert "the storm breaker of gpu-cluster is TRIPPED" in text
    assert "held workflow-job-b" in text


# --------------------------------------------------------------------------- #
# A two-node job and a single-node job (DESTR-028's 2+1 layout)
# --------------------------------------------------------------------------- #
def test_a_two_node_and_a_single_node_job_in_parallel_are_the_pass_shape() -> None:
    """Three nodes, two job domains: the two-node job claims both its nodes,
    the single-node job its one, each its own tokens; the two rebooted nodes
    are counted, the two-node job's peer is not (b223)."""

    parts = two_node_and_single_node_jobs()

    verdict = judge(
        snapshot(*parts), parts, overlap=oracle.OVERLAP_REMEDIATION, counted=2
    )

    assert (verdict.verdict, verdict.outcome) == (oracle.PASS, oracle.PARALLEL), (
        verdict.errors
    )
    assert verdict.facts["domains"]["job-b"]["claims"] == [
        "job:gpu-cluster:default/pytorchjob/job-b",
        "job:gpu-cluster:job-id/job-b",
        "node:gpu-cluster:node-c",
    ]
    assert verdict.facts["breaker"]["counted_nodes"] == {
        "job-a": ["node-a"],
        "job-b": ["node-c"],
    }


def test_a_counted_healthy_peer_fails_even_below_a_higher_threshold() -> None:
    """The pre-b223 count of the 2+1 layout is three nodes. On the four-node
    group that is the floor and trips the breaker; on a cluster whose
    threshold is higher it would not trip, and the exact count still fails."""

    first, second = two_node_and_single_node_jobs()
    first.window.append(
        StormEntry(
            node_id="node-b",
            domain="job:job-a",
            workflow_id=first.expectation.workflow_id,
            admitted_at=at(-1),
        )
    )

    verdict = judge(
        snapshot(first, second, threshold=4, agents=20),
        (first, second),
        overlap=oracle.OVERLAP_REMEDIATION,
        counted=2,
    )

    assert (verdict.verdict, verdict.outcome) == (oracle.FAIL, oracle.CONTRACT)
    assert verdict.errors == (
        "the breaker counted 3 nodes ['node-a', 'node-b', 'node-c'] for the case, "
        "expected 2",
    )


def test_the_pre_b223_count_trips_the_floor_and_holds_the_single_node_job() -> None:
    first, second = two_node_and_single_node_jobs(ran=False, held_on=STORM)
    first.window.append(
        StormEntry(
            node_id="node-b",
            domain="job:job-a",
            workflow_id=first.expectation.workflow_id,
            admitted_at=at(-1),
        )
    )
    value = tripped((first, second), second, nodes=["node-a", "node-b", "node-c"])

    verdict = judge(
        value, (first, second), overlap=oracle.OVERLAP_REMEDIATION, counted=2
    )

    assert (verdict.verdict, verdict.outcome) == (oracle.FAIL, oracle.BREAKER_TRIPPED)
    assert f"job-b: waited on the storm breaker ({STORM})" in "\n".join(verdict.errors)


# --------------------------------------------------------------------------- #
# One two-node job beside one idle node
# --------------------------------------------------------------------------- #
def test_a_job_and_an_idle_node_in_parallel_are_the_pass_shape() -> None:
    """Different domain types: the job workflow carries its job tokens, the
    idle one its node scope only. Two nodes count -- the job's rebooted node
    and the idle node, not the job's workload-stop peer -- below the floor of
    three (b223)."""

    parts = job_and_idle()

    verdict = judge(snapshot(*parts), parts, counted=2)

    assert (verdict.verdict, verdict.outcome) == (oracle.PASS, oracle.PARALLEL), (
        verdict.errors
    )
    claims = {label: item["claims"] for label, item in verdict.facts["domains"].items()}
    assert claims["idle"] == ["node:gpu-cluster:node-c"]
    assert "job:gpu-cluster:job-id/job-a" in claims["job"]
    assert verdict.facts["breaker"]["counted_domains"] == [
        "incident:incident-idle",
        "job:job-a",
    ]


def test_an_idle_workflow_held_behind_the_job_on_the_group_slot_is_serialized() -> None:
    job = part("job", "node-a", start=0, job="job-a", peers=("node-b",), pre_b220=True)
    idle = part("idle", "node-c", start=130, pre_b220=True, held_on=NODE_GROUP_SCOPE)

    verdict = judge(snapshot(job, idle, storm_supported=False), (job, idle))

    assert (verdict.verdict, verdict.outcome) == (oracle.FAIL, oracle.SERIALIZED)
    assert any(
        f"idle: was held on {NODE_GROUP_SCOPE}" in item for item in verdict.errors
    ), verdict.errors


def test_a_job_and_an_idle_node_with_a_missing_record_are_inconclusive() -> None:
    parts = job_and_idle()
    value = snapshot(*parts)
    del value["incidents"]["incident-idle"]

    verdict = judge(value, parts, counted=2)

    assert (verdict.verdict, verdict.outcome) == (
        oracle.INCONCLUSIVE,
        oracle.RECORD_MISSING,
    )


def test_a_job_and_an_idle_node_held_by_a_tripped_breaker_are_not_the_pass() -> None:
    job, idle = job_and_idle(ran=False, held_on=STORM)
    value = tripped((job, idle), idle, nodes=["node-a", "node-c", "node-x"])

    verdict = judge(value, (job, idle))

    assert (verdict.verdict, verdict.outcome) == (oracle.FAIL, oracle.BREAKER_TRIPPED)


# --------------------------------------------------------------------------- #
# The records the b220 claim path writes
# --------------------------------------------------------------------------- #
RESET_STEPS = (
    Op.MARK_UNSCHEDULABLE,
    Op.QUIESCE_GPU_SERVICES,
    Op.RESET_GPU,
    Op.RESTORE_GPU_SERVICES,
    Op.RESTORE_SCHEDULING,
)
STORE_CLUSTER = "cluster-a"


def _store_snapshot(store: InMemoryStore, ids: list[str]) -> dict[str, Any]:
    workflows = {item: store.get_workflow(item) for item in ids}
    breaker = store.get_remediation_storm_breaker(STORE_CLUSTER)
    return {
        "workflows": {
            key: value.model_dump(mode="json") for key, value in workflows.items()
        },
        "incidents": {
            value.incident_id: store.get_incident(value.incident_id).model_dump(
                mode="json"
            )
            for value in workflows.values()
        },
        "storm_supported": True,
        "breaker": breaker.model_dump(mode="json") if breaker is not None else None,
        "storm_held_workflow_ids": sorted(
            item
            for item, value in workflows.items()
            if value.remediation_budget_last_blocked_scope == f"storm:{STORE_CLUSTER}"
        ),
        "active_workflow_incidents": [],
        "open_incidents": [],
        "cluster_gpu_agents": None,
    }


def test_two_idle_nodes_through_the_b220_claim_path_are_the_pass_shape() -> None:
    """Both workflows are admitted in one dispatch tick, both resets wait at
    once, and the records they end with are the oracle's PASS."""

    store = InMemoryStore()
    start = datetime.now(timezone.utc) - timedelta(seconds=30)
    ids = [
        seed(
            store,
            suffix,
            node_id=f"node-{suffix}",
            operations=RESET_STEPS,
            created_at=start + timedelta(milliseconds=index),
        ).request_id
        for index, suffix in enumerate("ab")
    ]
    adapter = ScriptedAdapter(RESET_STEPS, waiting=(Op.RESET_GPU,))
    dispatcher = budget_dispatcher(
        store, adapter, RemediationBudgetPolicy(), max_workers=1
    )
    dispatcher.run_once()
    admitted = [store.get_workflow(item).status for item in ids]
    adapter.released.update(ids)
    for _ in range(3):
        dispatcher.run_once()

    verdict = oracle.judge_parallel_domains(
        _store_snapshot(store, ids),
        cluster_id=STORE_CLUSTER,
        domains=[
            oracle.DomainExpectation(
                label=f"node-{suffix}", faulted_node=f"node-{suffix}", workflow_id=item
            )
            for suffix, item in zip("ab", ids, strict=True)
        ],
        started_at=start,
        expected_counted_nodes=2,
    )

    assert admitted == [WorkflowStatus.RUNNING] * 2
    assert (verdict.verdict, verdict.outcome) == (oracle.PASS, oracle.PARALLEL), (
        verdict.errors,
        verdict.inconclusive,
    )
    assert verdict.facts["breaker"]["counted_source"] == {
        "node-a": "breaker window",
        "node-b": "breaker window",
    }


def _job_workflow(
    store: InMemoryStore, name: str, allocation: tuple[str, ...], created_at: datetime
) -> oracle.DomainExpectation:
    """A job workflow as the job family compiles it for a fault on the first
    node: STOP and RESTART widened to the attempt's allocation."""

    node = allocation[0]
    workload = f"default/pytorchjob/{name}"
    incident = fault_incident(
        f"incident-{name}",
        f"event-{name}",
        cluster_id=STORE_CLUSTER,
        node_ids=[node],
        job_id=name,
        attempt_id=f"{name}-a001",
        official_action="RESTART_BM",
        state=IncidentState.ACTION_PENDING,
        fencing_token=3,
    )
    workflow = workflow_request(
        f"workflow-{name}",
        incident.incident_id,
        runtime_profile_version="active-v1",
        official_steps=[
            workflow_step(
                Op.STOP_WORKLOADS, node_ids=list(allocation), workload_ids=[workload]
            ),
            workflow_step(Op.RESTART_NODE, node_ids=[node]),
            workflow_step(
                Op.RESTART_WORKLOAD,
                node_ids=list(allocation),
                workload_ids=[workload],
                # What WorkflowBuilder.restart_step_parameters writes.
                parameters={
                    "cluster_id": STORE_CLUSTER,
                    "job_id": name,
                    "source_attempt_id": f"{name}-a001",
                    "source_gpu_count": 8 * len(allocation),
                    "restart_budget": 1,
                },
            ),
        ],
        created_at=created_at,
    )
    store.save_incident_and_workflow(
        incident.model_copy(update={"workflow_request_id": workflow.request_id}),
        workflow,
    )
    return oracle.DomainExpectation(
        label=name,
        faulted_node=node,
        workflow_id=workflow.request_id,
        remediation="RESTART_NODE",
        job_id=name,
        workload_ids=(workload,),
        allocation=allocation,
    )


def _dispatch_layout(
    layout: dict[str, tuple[str, ...] | None],
) -> tuple[Any, list[Any]]:
    """Seed one workflow per domain (a job: its allocation; ``None``: an idle
    node of that name) and run one dispatch tick, oldest first."""

    store = InMemoryStore()
    start = datetime.now(timezone.utc) - timedelta(seconds=30)
    domains = []
    for index, (name, allocation) in enumerate(layout.items()):
        created = start + timedelta(milliseconds=index)
        if allocation is None:
            row = seed(
                store, name, node_id=name, operations=RESET_STEPS, created_at=created
            )
            domains.append(
                oracle.DomainExpectation(
                    label=name, faulted_node=name, workflow_id=row.request_id
                )
            )
        else:
            domains.append(_job_workflow(store, name, allocation, created))
    adapter = ScriptedAdapter(
        (*RESET_STEPS, Op.STOP_WORKLOADS, Op.RESTART_NODE, Op.RESTART_WORKLOAD),
        waiting=(Op.RESTART_NODE, Op.RESET_GPU),
    )
    budget_dispatcher(
        store, adapter, RemediationBudgetPolicy(), max_workers=1
    ).run_once()
    return store, domains


@pytest.mark.parametrize(
    ("layout", "counted"),
    [
        ({"job-a": ("a1", "a2"), "job-b": ("b1", "b2")}, ["a1", "b1"]),
        ({"job-a": ("a1", "a2"), "job-b": ("b1",)}, ["a1", "b1"]),
        ({"job-a": ("a1", "a2"), "idle-1": None}, ["a1", "idle-1"]),
    ],
    ids=[
        "two-two-node-jobs",
        "two-node-and-single-node-job",
        "two-node-job-and-idle-node",
    ],
)
def test_a_two_node_jobs_peer_is_not_counted_so_the_floor_holds(
    layout: dict[str, tuple[str, ...] | None], counted: list[str]
) -> None:
    """b223: a job's STOP_WORKLOADS / RESTART_WORKLOAD name its whole
    allocation, but they are workload mutations -- only the rebooted node
    enters destructive NODE remediation and is counted. A two-node job counts
    exactly as a single-node one, so two such domains stay at two counted
    nodes, below the floor of 3, and both run in parallel. Before b223 the
    healthy peers counted too (3-4 nodes) and the second admission was held."""

    store, domains = _dispatch_layout(layout)
    breaker = store.get_remediation_storm_breaker(STORE_CLUSTER)
    rows = [store.get_workflow(item.workflow_id) for item in domains]

    assert breaker is not None and breaker.state is StormBreakerState.CLOSED, breaker
    assert sorted(entry.node_id for entry in breaker.window) == counted
    assert [row.status for row in rows] == [WorkflowStatus.RUNNING] * 2
    assert [row.remediation_budget_last_blocked_scope for row in rows] == [None, None]


@pytest.mark.parametrize(
    "layout",
    [{"job-a": ("a1",), "job-b": ("b1",)}, {"job-a": ("a1",), "idle-1": None}],
    ids=["two-single-node-jobs", "single-node-job-and-idle-node"],
)
def test_single_node_jobs_keep_exactly_two_nodes_counted(
    layout: dict[str, tuple[str, ...] | None],
) -> None:
    """The smallest layout of two domains, a job among them, that stays below
    the floor: each job on one node."""

    store, domains = _dispatch_layout(layout)
    breaker = store.get_remediation_storm_breaker(STORE_CLUSTER)
    rows = [store.get_workflow(item.workflow_id) for item in domains]

    assert breaker is not None and breaker.state is StormBreakerState.CLOSED, breaker
    assert sorted(entry.node_id for entry in breaker.window) == sorted(
        item.faulted_node for item in domains
    )
    assert [row.status for row in rows] == [WorkflowStatus.RUNNING] * 2
    assert [row.remediation_budget_last_blocked_scope for row in rows] == [None, None]


# --------------------------------------------------------------------------- #
# The operation sets the oracle reads
# --------------------------------------------------------------------------- #
def test_the_budgeted_operations_are_the_ones_the_product_claims_for() -> None:
    policy = RemediationBudgetPolicy()
    claiming = set()
    for operation in WorkflowOperation:
        step = WorkflowStepSpec(
            operation=operation, execution_owner="owner-a", node_ids=["node-a"]
        )
        workflow = WorkflowRequest(
            incident_id="incident-a",
            status=WorkflowStatus.PENDING,
            fencing_token=1,
            official_steps=[step],
        )
        if claims_in_cluster(policy, workflow, "cluster-a", [step]):
            claiming.add(operation.value)

    assert claiming == set(oracle.BUDGETED_OPERATIONS)
    assert oracle.JOB_TOKEN_OPERATIONS == {"STOP_WORKLOADS", "RESTART_WORKLOAD"}


# --------------------------------------------------------------------------- #
# The precondition: GPU nodes in one node group, read live
# --------------------------------------------------------------------------- #
def gpu_node(name: str, **labels: str) -> dict[str, Any]:
    return {"name": name, "uid": f"uid-{name}", "gpu_allocatable": 8, "labels": labels}


EKS_NODES = [
    gpu_node(f"eks-{index}", **{EKS_NODEGROUP_LABEL: "gpu-p5en-group"})
    for index in range(2)
]
HYPERPOD_NODES = [
    gpu_node(f"hyperpod-{index}", **{HYPERPOD_INSTANCE_GROUP_LABEL: "p5en-group"})
    for index in range(4)
]


@pytest.mark.parametrize(
    ("nodes", "minimum", "group", "not_run"),
    [
        (EKS_NODES, 2, "gpu-p5en-group", None),
        (
            EKS_NODES,
            4,
            "gpu-p5en-group",
            "needs >=4 GPU nodes in one node group; cluster gpu-cluster group "
            f"gpu-p5en-group ({EKS_NODEGROUP_LABEL}) has 2",
        ),
        (HYPERPOD_NODES, 2, "p5en-group", None),
        (HYPERPOD_NODES, 4, "p5en-group", None),
    ],
    ids=["eks-2-of-2", "eks-4-of-2", "hyperpod-2-of-4", "hyperpod-4-of-4"],
)
def test_the_precondition_is_the_live_gpu_node_count_of_one_group(
    nodes: list[dict[str, Any]], minimum: int, group: str, not_run: str | None
) -> None:
    gate = oracle.node_group_gate(nodes, minimum=minimum, cluster_id=CLUSTER)

    assert gate.group is not None and gate.group.name == group, gate
    assert (gate.not_run or {}).get("detail") == not_run
    if not_run is not None:
        assert gate.not_run == oracle.capability_absent(not_run)


def test_the_largest_group_is_chosen_and_a_named_group_must_exist() -> None:
    small = [gpu_node("x-0", **{EKS_NODEGROUP_LABEL: "a-small"})]
    unlabelled = [gpu_node("loose-0")]
    nodes = [*small, *HYPERPOD_NODES, *unlabelled]

    largest = oracle.node_group_gate(nodes, minimum=3, cluster_id=CLUSTER)
    named = oracle.node_group_gate(
        nodes, minimum=1, cluster_id=CLUSTER, requested="a-small"
    )
    absent = oracle.node_group_gate(
        nodes, minimum=1, cluster_id=CLUSTER, requested="zz"
    )
    bare = oracle.node_group_gate(unlabelled, minimum=1, cluster_id=CLUSTER)

    assert largest.group is not None and largest.group.nodes == tuple(
        item["name"] for item in HYPERPOD_NODES
    )
    assert largest.not_run is None
    assert named.group is not None and named.group.nodes == ("x-0",)
    assert absent.group is None
    assert (
        absent.not_run is not None
        and "no GPU node in node group zz" in absent.not_run["detail"]
    )
    assert bare.not_run is not None and "none carries" in bare.not_run["detail"]


def test_a_hyperpod_instance_group_wins_over_an_eks_node_group_label() -> None:
    labels = {EKS_NODEGROUP_LABEL: "eks", HYPERPOD_INSTANCE_GROUP_LABEL: "hyperpod"}

    assert oracle.node_group_of(labels) == (HYPERPOD_INSTANCE_GROUP_LABEL, "hyperpod")
    assert oracle.node_group_of({}) is None


# --------------------------------------------------------------------------- #
# The breaker premise a case checks before it injects
# --------------------------------------------------------------------------- #
def premise_snapshot(*entries: StormEntry, state: str = "CLOSED") -> dict[str, Any]:
    breaker = RemediationStormBreaker(
        cluster_id=CLUSTER,
        state=StormBreakerState(state),
        window=list(entries),
        updated_at=at(0),
    ).model_dump(mode="json")
    return {
        "storm_supported": True,
        "breaker": breaker,
        "storm_held_workflow_ids": [],
        "cluster_gpu_agents": 4,
    }


def test_a_node_still_counted_from_another_case_refuses_the_start() -> None:
    recent = StormEntry(
        node_id="node-c", domain="incident:other", workflow_id="w", admitted_at=at(-60)
    )
    stale = recent.model_copy(update={"admitted_at": at(-900)})

    refused = oracle.breaker_premise_errors(
        premise_snapshot(recent),
        cluster_id=CLUSTER,
        targets=["node-a", "node-b"],
        now=at(0),
    )
    admitted = oracle.breaker_premise_errors(
        premise_snapshot(stale),
        cluster_id=CLUSTER,
        targets=["node-a", "node-b"],
        now=at(0),
    )

    assert len(refused) == 1 and "at or past its threshold 3" in refused[0], refused
    assert f"start after {at(540).isoformat()}" in refused[0]
    assert admitted == []


def test_an_open_or_absent_breaker_refuses_the_start() -> None:
    open_breaker = premise_snapshot(state="TRIPPED")
    held = premise_snapshot()
    held["storm_held_workflow_ids"] = ["workflow-x"]

    assert (
        "an operator must release it first"
        in oracle.breaker_premise_errors(
            open_breaker, cluster_id=CLUSTER, targets=["node-a", "node-b"], now=at(0)
        )[0]
    )
    assert oracle.breaker_premise_errors(
        held, cluster_id=CLUSTER, targets=["node-a", "node-b"], now=at(0)
    ) == ["the storm breaker holds workflows ['workflow-x']"]
    assert oracle.breaker_premise_errors(
        {"storm_supported": False}, cluster_id=CLUSTER, targets=["a", "b"], now=at(0)
    ) == [
        "the deployed control plane has no remediation storm breaker (deploy b220 first)"
    ]


def test_the_store_read_refuses_ids_it_cannot_pass() -> None:
    class Fixture:
        settings = type("S", (), {"cluster_id": CLUSTER})()

        def cpu_python(self, script: str, *arguments: str) -> dict[str, Any]:
            return {"script": script, "arguments": list(arguments)}

    read = oracle.read_oracle_snapshot(
        Fixture(), workflow_ids=["w-1", "w-2"], node_ids=["node-a"]
    )

    assert read["script"] == oracle.ORACLE_PROBE
    assert read["arguments"] == [
        CLUSTER,
        "w-1,w-2",
        "node-a",
        ",".join(oracle.OPEN_INCIDENT_STATES),
    ]
    with pytest.raises(ValueError, match="no comma"):
        oracle.read_oracle_snapshot(Fixture(), workflow_ids=["w,1"], node_ids=["n"])
