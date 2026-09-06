"""RF-5 / RF-8 on the planning side: every rewrite of the plan leaves an event.

``DagBrancher`` appends, replaces, widens and queues branches by rewriting
``official_steps`` and bumping ``dag_revision``; ``DispositionApplier`` widens
and replaces flat plans; ``WorkflowMergeService`` preempts. None of them left
a machine-readable trace. Each now appends one PLAN_REWRITE or PREEMPTION
event with a stable code, the ids involved, and a digest of the plan it
produced. A rewrite that changes nothing records nothing.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from gpu_fault.models import (
    WorkflowEventCode,
    WorkflowEventKind,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepStatus,
)
from gpu_fault.operation_registry import (
    NODE_EXCLUSIVE_OPERATIONS,
    WORKLOAD_SCOPED_OPERATIONS,
)
from gpu_fault.orchestration.arbitration import RecoveryArbiter
from gpu_fault.orchestration.dag_branching import DagBrancher, plan_digest
from gpu_fault.orchestration.disposition import Disposition, DispositionApplier
from gpu_fault.orchestration.families.conflicts import NodeConflictService
from gpu_fault.orchestration.workflow_merge import WorkflowMergeService
from tests._builders import (
    copy_model,
    fault_incident,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)

NOW = datetime(2026, 9, 6, 9, 0, tzinfo=timezone.utc)
ALL_NODES = ["node-a", "node-b", "node-c"]
STOP = WorkflowOperation.STOP_WORKLOADS
RESET = WorkflowOperation.RESET_GPU
REBOOT = WorkflowOperation.RESTART_NODE
RESTART_JOB = WorkflowOperation.RESTART_WORKLOAD


def _job_workflow(
    node_id: str,
    request_id: str,
    action: WorkflowOperation = RESET,
    gpu_uuids: list[str] | None = None,
) -> WorkflowRequest:
    gpus = gpu_uuids or []
    return workflow_request(
        request_id,
        "incident-dag",
        fencing_token=1,
        runtime_profile_version="simulated-v1",
        official_action=action.value,
        official_steps=[
            workflow_step(STOP, node_ids=ALL_NODES, workload_ids=["training/job/a"]),
            workflow_step(action, node_ids=[node_id], gpu_uuids=gpus),
            workflow_step(WorkflowOperation.RESTORE_SCHEDULING, node_ids=[node_id]),
            workflow_step(
                RESTART_JOB, node_ids=ALL_NODES, workload_ids=["training/job/a"]
            ),
        ],
    )


def _running_dag() -> WorkflowRequest:
    return copy_model(
        _job_workflow("node-a", "workflow-node-a", gpu_uuids=["GPU-1"]),
        status=WorkflowStatus.RUNNING,
        completed_step_indexes=[0],
        completed_operations=[STOP],
    )


def _rewrites(workflow: WorkflowRequest):
    return [
        event
        for event in workflow.events
        if event.kind is WorkflowEventKind.PLAN_REWRITE
    ]


def _preemptions(workflow: WorkflowRequest):
    return [
        event for event in workflow.events if event.kind is WorkflowEventKind.PREEMPTION
    ]


# ------------------------------------------------------------- brancher


def test_appending_a_parallel_branch_records_what_it_appended():
    brancher = DagBrancher(RecoveryArbiter())
    existing = _running_dag()

    dag = brancher.append_parallel_job_branch(
        existing, _job_workflow("node-b", "wf-reset-b")
    )

    (event,) = _rewrites(dag)
    assert event.code == WorkflowEventCode.BRANCH_APPENDED
    assert event.dag_revision == dag.dag_revision
    assert event.details["node_id"] == "node-b"
    assert event.details["branch_id"].startswith("branch:node-b"), (
        "the escalation event names the node's branch"
    )
    assert event.details["candidate_workflow_id"] == "wf-reset-b"
    assert event.details["appended_indexes"] == list(
        range(len(existing.official_steps), len(dag.official_steps))
    )
    assert event.details["superseded_indexes"] == []
    join = next(step for step in dag.official_steps if step.operation is RESTART_JOB)
    assert event.details["join_dependencies"] == join.depends_on_step_indexes
    assert event.details["steps_digest"] == plan_digest(dag.official_steps)
    assert event.details["steps_digest"] != plan_digest(existing.official_steps)
    assert "parameters" not in event.details, "payloads never enter an event"


def test_an_already_present_branch_records_nothing():
    brancher = DagBrancher(RecoveryArbiter())
    dag = brancher.append_parallel_job_branch(
        _running_dag(), _job_workflow("node-b", "wf-reset-b")
    )

    again = brancher.append_parallel_job_branch(
        dag, _job_workflow("node-b", "wf-reset-b-dup")
    )

    assert again is dag
    assert len(_rewrites(again)) == 1


def test_replacing_a_branch_records_the_retired_indexes():
    brancher = DagBrancher(RecoveryArbiter())
    dag = brancher.append_parallel_job_branch(
        _running_dag(), _job_workflow("node-b", "wf-reset-b")
    )
    before = brancher.node_branch_step_indexes(dag, "node-b")

    replaced = brancher.replace_parallel_job_branch(
        dag, _job_workflow("node-b", "wf-reboot-b", REBOOT), "node-b"
    )

    event = _rewrites(replaced)[-1]
    assert event.code == WorkflowEventCode.BRANCH_REPLACED
    assert set(before) <= set(event.details["superseded_indexes"]), event.details
    assert event.details["appended_indexes"], event.details
    assert event.details["node_id"] == "node-b"
    assert event.details["steps_digest"] != _rewrites(dag)[-1].details["steps_digest"]


def test_queueing_a_branch_successor_records_the_predecessor_step():
    brancher = DagBrancher(RecoveryArbiter())
    dag = brancher.append_parallel_job_branch(
        _running_dag(), _job_workflow("node-b", "wf-reset-b")
    )
    tail = max(brancher.node_branch_step_indexes(dag, "node-b"))

    queued = brancher.append_parallel_job_branch_successor(
        dag, _job_workflow("node-b", "wf-reset-b-later"), "node-b"
    )

    event = _rewrites(queued)[-1]
    assert event.code == WorkflowEventCode.BRANCH_SUCCESSOR_QUEUED
    assert event.details["predecessor_step_index"] == tail
    assert ":successor:" in event.details["branch_id"], event.details


def test_widening_a_branch_records_the_indexes_and_gpus_it_widened():
    brancher = DagBrancher(RecoveryArbiter())
    dag = brancher.append_parallel_job_branch(
        _running_dag(), _job_workflow("node-b", "wf-reset-b", gpu_uuids=["GPU-2"])
    )
    indexes = brancher.node_branch_step_indexes(dag, "node-b")

    widened = brancher.widen_parallel_job_branch(
        dag,
        _job_workflow("node-b", "wf-reset-b-more", gpu_uuids=["GPU-3"]),
        "node-b",
        {"GPU-3"},
    )

    event = _rewrites(widened)[-1]
    assert event.code == WorkflowEventCode.BRANCH_WIDENED
    assert event.details["node_id"] == "node-b"
    assert event.details["gpu_uuids"] == ["GPU-3"]
    assert set(event.details["widened_indexes"]) <= set(indexes), event.details
    assert event.details["widened_indexes"], event.details
    assert event.dag_revision == widened.dag_revision


# -------------------------------------------------------------- applier


def _flat(request_id: str, gpu_uuids: list[str], **updates) -> WorkflowRequest:
    return workflow_request(
        request_id,
        "inc-a",
        status=WorkflowStatus.PENDING,
        official_action=RESET.value,
        official_steps=[
            workflow_step(WorkflowOperation.MARK_UNSCHEDULABLE, node_ids=["node-a"]),
            workflow_step(
                RESET,
                node_ids=["node-a"],
                gpu_uuids=gpu_uuids,
                depends_on_step_indexes=[0],
            ),
            workflow_step(
                WorkflowOperation.RESTORE_SCHEDULING,
                node_ids=["node-a"],
                depends_on_step_indexes=[1],
            ),
        ],
        **updates,
    )


def _applier() -> DispositionApplier:
    arbiter = RecoveryArbiter()
    return DispositionApplier(
        arbiter=arbiter,
        brancher=DagBrancher(arbiter),
        aggregation_deadlines=lambda now, _workflow: (
            now + timedelta(seconds=5),
            now + timedelta(seconds=30),
        ),
        prepare_preempting_successor=lambda _existing, successor: successor,
        preempt_parallel_job_branch=lambda existing, _candidate, _node_id: existing,
        workflow_preemption_enabled=True,
    )


def _apply(verdict: Disposition, existing: WorkflowRequest, candidate: WorkflowRequest):
    incident = fault_incident("inc-a", "evt-a")
    workflow, _ = _applier().apply(
        verdict,
        node_id="node-a",
        candidate=incident,
        candidate_workflow=candidate,
        existing_incident=incident,
        existing_workflow=existing,
        gpu_uuids=set(candidate.official_steps[1].gpu_uuids),
        mutable=True,
        now=NOW,
    )
    return workflow


def test_widening_in_place_records_the_widened_steps():
    existing = _flat("wf-a", ["GPU-1"])

    widened = _apply(Disposition.WIDEN_IN_PLACE, existing, _flat("wf-b", ["GPU-2"]))

    (event,) = _rewrites(widened)
    assert event.code == WorkflowEventCode.PLAN_WIDENED
    assert event.details["node_id"] == "node-a"
    assert event.details["gpu_uuids"] == ["GPU-2"]
    assert 1 in event.details["widened_indexes"], event.details
    assert event.details["steps_digest"] == plan_digest(widened.official_steps)


def test_widening_in_place_with_nothing_pending_records_nothing():
    existing = _flat(
        "wf-a",
        ["GPU-1"],
        completed_step_indexes=[0, 1, 2],
        step_executions=[
            workflow_step_execution(index, step.operation, WorkflowStepStatus.SUCCEEDED)
            for index, step in enumerate(_flat("wf-a", ["GPU-1"]).official_steps)
        ],
    )

    widened = _apply(Disposition.WIDEN_IN_PLACE, existing, _flat("wf-b", ["GPU-2"]))

    assert _rewrites(widened) == []


def test_replacing_in_place_inherits_the_old_trail_and_records_the_swap():
    existing = _flat("wf-a", ["GPU-1"])
    existing = _apply(Disposition.WIDEN_IN_PLACE, existing, _flat("wf-x", ["GPU-9"]))
    assert len(existing.events) == 1, "precondition: the old record has history"
    candidate = copy_model(
        _flat("wf-b", ["GPU-1"]),
        official_action=REBOOT.value,
        official_steps=[
            workflow_step(WorkflowOperation.MARK_UNSCHEDULABLE, node_ids=["node-a"]),
            workflow_step(REBOOT, node_ids=["node-a"], depends_on_step_indexes=[0]),
            workflow_step(
                WorkflowOperation.RESTORE_SCHEDULING,
                node_ids=["node-a"],
                depends_on_step_indexes=[1],
            ),
        ],
    )

    replaced = _apply(Disposition.REPLACE_IN_PLACE, existing, candidate)

    assert replaced.request_id == existing.request_id
    assert [event.code for event in replaced.events] == [
        WorkflowEventCode.PLAN_WIDENED,
        WorkflowEventCode.PLAN_REPLACED,
    ]
    event = replaced.events[-1]
    assert event.details["candidate_workflow_id"] == "wf-b"
    assert event.details["from_action"] == RESET.value
    assert event.details["to_action"] == REBOOT.value
    assert event.details["previous_steps_digest"] == plan_digest(
        existing.official_steps
    )
    assert event.details["steps_digest"] == plan_digest(replaced.official_steps)


# ---------------------------------------------------------------- merge


def _merger() -> WorkflowMergeService:
    arbiter = RecoveryArbiter()
    return WorkflowMergeService(
        arbiter,
        DagBrancher(arbiter),
        preemption_enabled=True,
        workload_scoped_operations=set(WORKLOAD_SCOPED_OPERATIONS),
        node_exclusive_operations=set(NODE_EXCLUSIVE_OPERATIONS),
        workflow_resource_claims_by_node=(
            NodeConflictService.workflow_resource_claims_by_node
        ),
    )


def _existing(operations, executions) -> WorkflowRequest:
    return workflow_request(
        "workflow-existing",
        "incident-a",
        WorkflowStatus.RUNNING,
        1,
        runtime_profile_version="simulated-v1",
        official_action=operations[0].value,
        official_steps=[workflow_step(operation) for operation in operations],
        completed_step_indexes=[0],
        completed_operations=[operations[0]],
        step_executions=executions,
    )


def _candidate() -> WorkflowRequest:
    return workflow_request(
        "workflow-candidate",
        "incident-a",
        fencing_token=1,
        runtime_profile_version="simulated-v1",
        official_action=REBOOT.value,
        official_steps=[workflow_step(REBOOT)],
    )


def test_preempting_a_branch_records_predecessor_successor_and_boundary():
    existing = _existing(
        [WorkflowOperation.QUARANTINE, RESET, WorkflowOperation.FREEZE_EVIDENCE], []
    )

    combined = _merger().preempt_parallel_branch(existing, _candidate(), "node-a")

    assert 1 in combined.superseded_step_indexes, "precondition: boundary was open"
    (event,) = _preemptions(combined)
    assert event.code == WorkflowEventCode.PREEMPTED
    assert event.details["predecessor_workflow_id"] == "workflow-existing"
    assert event.details["successor_workflow_id"] == "workflow-candidate"
    assert event.details["node_id"] == "node-a"
    assert set(event.details["replaced_indexes"]) == {1, 2}
    assert isinstance(event.reason, str) and event.reason, event
    # The brancher's own rewrite sits beside it, on the same revision.
    (rewrite,) = _rewrites(combined)
    assert rewrite.dag_revision == event.dag_revision == combined.dag_revision


def test_a_closed_boundary_is_recorded_as_such():
    existing = _existing(
        [WorkflowOperation.QUARANTINE, WorkflowOperation.CHECK_MECHANICALS, RESET],
        [
            workflow_step_execution(
                1, WorkflowOperation.CHECK_MECHANICALS, WorkflowStepStatus.WAITING
            )
        ],
    )

    combined = _merger().preempt_parallel_branch(existing, _candidate(), "node-a")

    assert 1 not in combined.superseded_step_indexes, "precondition: boundary held"
    (event,) = _preemptions(combined)
    assert event.code == WorkflowEventCode.PREEMPTION_BOUNDARY_CLOSED
    assert event.details["boundary_step_index"] == 1
    assert "CHECK_MECHANICALS" in (event.reason or ""), event


def test_a_preempting_successor_records_the_predecessor_it_preempts():
    existing = copy_model(
        _existing([WorkflowOperation.MARK_UNSCHEDULABLE, RESET], []),
        official_action=RESET.value,
    )
    candidate = workflow_request(
        "workflow-successor",
        "incident-a",
        fencing_token=1,
        official_action=REBOOT.value,
        official_steps=[
            workflow_step(WorkflowOperation.MARK_UNSCHEDULABLE),
            workflow_step(REBOOT, depends_on_step_indexes=[0]),
        ],
    )

    successor = _merger().prepare_preempting_successor(existing, candidate)

    assert successor.preempt_predecessor is True, "precondition: it preempts"
    (event,) = _preemptions(successor)
    assert event.code == WorkflowEventCode.PREEMPTED
    assert event.details["predecessor_workflow_id"] == "workflow-existing"
    assert event.details["successor_workflow_id"] == "workflow-successor"
    assert event.details["inherited_step_indexes"] == successor.inherited_step_indexes
    assert event.reason == successor.preemption_reason


def test_a_successor_that_does_not_preempt_records_nothing():
    existing = _existing([WorkflowOperation.MARK_UNSCHEDULABLE, REBOOT], [])
    weaker = workflow_request(
        "workflow-weaker",
        "incident-a",
        fencing_token=1,
        official_action=RESET.value,
        official_steps=[workflow_step(RESET)],
    )

    successor = _merger().prepare_preempting_successor(existing, weaker)

    assert successor.preempt_predecessor is False
    assert successor.events == []
