"""RF-5: a DAG rewrite leaves an event, so the plan's history is reconstructible.

Branch escalation used to mutate ``official_steps`` in place and bump
``dag_revision``; the only trace of the rungs a node walked was free text in
``incident.reasons``. Now every rewrite appends a PLAN_REWRITE event carrying
a digest of the new plan, and every rung a BRANCH_ESCALATION event naming the
node, both branch ids, and the operation it moved from and to. The node's
ladder is readable from ``events`` alone.
"""

from __future__ import annotations

from gpu_fault.execution.branch_escalation import BranchEscalator
from gpu_fault.execution.models import WorkflowExecutionRequest
from gpu_fault.models import (
    WORKFLOW_EVENTS_LIMIT,
    WorkflowEventCode,
    WorkflowEventKind,
    WorkflowOperation,
    WorkflowStatus,
)
from gpu_fault.orchestration.arbitration import RecoveryArbiter
from gpu_fault.orchestration.dag_branching import DagBrancher
from tests._builders import (
    active_workflow_executor,
    build_store,
    copy_model,
    workflow_step,
    workflow_step_execution,
)
from tests.execution._support import (
    RESTART_PARAMETERS,
    FakeAdapter,
    WorkflowStepOutcome,
    workflow_state,
)

STOP = WorkflowOperation.STOP_WORKLOADS
RESET = WorkflowOperation.RESET_GPU
REBOOT = WorkflowOperation.RESTART_NODE
REPLACE = WorkflowOperation.REPLACE_NODE
RESTART_JOB = WorkflowOperation.RESTART_WORKLOAD
BUNDLE = WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE
VALIDATIONS = (
    WorkflowOperation.VALIDATE_GPU,
    WorkflowOperation.VALIDATE_HOST,
    WorkflowOperation.VALIDATE_FABRIC,
    WorkflowOperation.RESTORE_SCHEDULING,
)
ALL_OPERATIONS = {STOP, RESET, REBOOT, REPLACE, RESTART_JOB, BUNDLE, *VALIDATIONS}


def _escalator(max_rungs: int = 2) -> BranchEscalator:
    def compile_steps(_workflow, operations, node_id, gpu_uuids):
        return [
            workflow_step(operation, node_ids=[node_id], gpu_uuids=list(gpu_uuids))
            for operation in operations
        ]

    return BranchEscalator(
        DagBrancher(RecoveryArbiter()), compile_steps, max_rungs=max_rungs
    )


def _job_dag(store):
    """STOP (shared, done) -> {node-b RESET_GPU, node-c BUNDLE} -> RESTART (join)."""

    incident, workflow = workflow_state(store, [STOP, RESET, BUNDLE, RESTART_JOB])
    steps = [
        copy_model(
            workflow.official_steps[0],
            branch_id="shared",
            node_ids=["node-b", "node-c"],
        ),
        copy_model(
            workflow.official_steps[1],
            node_ids=["node-b"],
            depends_on_step_indexes=[0],
            branch_id="branch:node-b",
        ),
        copy_model(
            workflow.official_steps[2],
            node_ids=["node-c"],
            depends_on_step_indexes=[0],
            branch_id="branch:node-c",
        ),
        copy_model(
            workflow.official_steps[3],
            depends_on_step_indexes=[1, 2],
            branch_id="join",
            node_ids=["node-b", "node-c"],
            parameters=dict(RESTART_PARAMETERS),
        ),
    ]
    workflow = copy_model(
        workflow,
        dag_enabled=True,
        dag_revision=1,
        official_steps=steps,
        completed_step_indexes=[0],
        completed_operations=[STOP],
        step_executions=[workflow_step_execution(0, STOP)],
    )
    store.save_workflow(workflow)
    return incident, workflow


def _ok(*operations):
    return {operation: WorkflowStepOutcome.succeeded() for operation in operations}


def test_two_escalations_are_reconstructible_from_events_alone():
    store = build_store()
    _, workflow = _job_dag(store)
    adapter = FakeAdapter(
        {
            RESET: WorkflowStepOutcome.failed("reset refused"),
            REBOOT: WorkflowStepOutcome.failed("node did not come back"),
            **_ok(REPLACE, BUNDLE, RESTART_JOB, *VALIDATIONS),
        }
    )
    executor = active_workflow_executor(store, [adapter], ALL_OPERATIONS)
    executor.branch_escalator = _escalator()

    result = executor.execute(
        workflow.request_id,
        WorkflowExecutionRequest(expected_fencing_token=workflow.fencing_token),
    )
    saved = store.get_workflow(workflow.request_id)

    assert result.status is WorkflowStatus.SUCCEEDED, result
    escalations = [
        event
        for event in saved.events
        if event.kind is WorkflowEventKind.BRANCH_ESCALATION
    ]
    rewrites = [
        event for event in saved.events if event.kind is WorkflowEventKind.PLAN_REWRITE
    ]
    assert len(escalations) == 2, saved.events
    assert [event.code for event in escalations] == [
        WorkflowEventCode.BRANCH_ESCALATED,
        WorkflowEventCode.BRANCH_ESCALATED,
    ]
    assert len(rewrites) >= 2, rewrites
    digests = [event.details["steps_digest"] for event in rewrites]
    assert len(set(digests)) == len(digests), digests
    assert all(len(digest) == 16 for digest in digests), digests
    # Revisions climb with the rewrites, and each event was stamped with the
    # revision it produced.
    assert [event.dag_revision for event in rewrites] == sorted(
        event.dag_revision for event in rewrites
    )
    assert rewrites[-1].dag_revision == saved.dag_revision
    # node-b's ladder, without reading ``official_steps``.
    node_b = [event for event in escalations if event.details["node_id"] == "node-b"]
    ladder = [node_b[0].details["from_operation"]] + [
        event.details["to_operation"] for event in node_b
    ]
    assert ladder == [RESET.value, REBOOT.value, REPLACE.value], ladder
    assert [event.details["rung_count"] for event in node_b] == [1, 2]
    assert all(
        event.details["branch_id"].startswith("branch:node-b") for event in node_b
    ), node_b
    assert all(event.step_index is not None for event in node_b), node_b
    assert all(event.reason for event in node_b), node_b
    # The failed attempts that triggered the rungs are in the same trail.
    failed = [
        event
        for event in saved.events
        if event.kind is WorkflowEventKind.STEP_ATTEMPT
        and event.code == WorkflowEventCode.STEP_FAILED
    ]
    assert [event.operation for event in failed] == [RESET, REBOOT], failed


def test_an_exhausted_ladder_is_one_more_branch_escalation_event():
    store = build_store()
    _, workflow = _job_dag(store)
    adapter = FakeAdapter(
        {
            RESET: WorkflowStepOutcome.failed("reset refused"),
            REBOOT: WorkflowStepOutcome.failed("node did not come back"),
            REPLACE: WorkflowStepOutcome.failed("no healthy warm spare"),
            **_ok(BUNDLE, RESTART_JOB, *VALIDATIONS),
        }
    )
    executor = active_workflow_executor(store, [adapter], ALL_OPERATIONS)
    executor.branch_escalator = _escalator()

    result = executor.execute(
        workflow.request_id,
        WorkflowExecutionRequest(expected_fencing_token=workflow.fencing_token),
    )
    saved = store.get_workflow(workflow.request_id)

    assert result.status is WorkflowStatus.FAILED, result
    codes = [
        event.code
        for event in saved.events
        if event.kind is WorkflowEventKind.BRANCH_ESCALATION
    ]
    assert codes == [
        WorkflowEventCode.BRANCH_ESCALATED,
        WorkflowEventCode.BRANCH_ESCALATED,
        WorkflowEventCode.BRANCH_EXHAUSTED,
    ], codes
    exhausted = saved.events[
        [event.code for event in saved.events].index(WorkflowEventCode.BRANCH_EXHAUSTED)
    ]
    assert exhausted.details["exhausted"] is True
    assert exhausted.details["node_id"] == "node-b"
    assert exhausted.details["branch_id"] == saved.exhausted_branch_ids[0]
    assert exhausted.details["rung_count"] == 2
    assert len(saved.events) <= WORKFLOW_EVENTS_LIMIT
