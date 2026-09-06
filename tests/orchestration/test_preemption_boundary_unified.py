"""The executor and the merge answer "may this step still be superseded?"
with one function (F-C1).

``ProductionWorkflowExecutor._supersede_if_safe`` judged every WAITING
record by whether its wait could be stopped; ``WorkflowMergeService`` judged
only the branch's records, and only by a registry flag that ignored the
command's remote state and the operator acknowledgements. A step the executor
refused to abandon was one the merge marked superseded. Both now read
``orchestration.preemption_boundary.preemption_boundary``; the cases below
show them agreeing, and pin each place the merge deliberately changed.
"""

from __future__ import annotations

import pytest

from gpu_fault.execution.models import WorkflowExecutionRequest, WorkflowStepOutcome
from gpu_fault.models import (
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepExecution,
    WorkflowStepStatus,
)
from gpu_fault.operation_registry import (
    NODE_EXCLUSIVE_OPERATIONS,
    WORKLOAD_SCOPED_OPERATIONS,
)
from gpu_fault.orchestration.arbitration import RecoveryArbiter
from gpu_fault.orchestration.dag_branching import DagBrancher
from gpu_fault.orchestration.families.conflicts import NodeConflictService
from gpu_fault.orchestration.preemption_boundary import preemption_boundary
from gpu_fault.orchestration.workflow_merge import WorkflowMergeService
from tests._builders import (
    active_workflow_executor,
    build_store,
    copy_model,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)
from tests.execution._support import FakeAdapter, _preempting_successor, workflow_state

QUARANTINE = WorkflowOperation.QUARANTINE
FREEZE = WorkflowOperation.FREEZE_EVIDENCE
RESET = WorkflowOperation.RESET_GPU
VALIDATE = WorkflowOperation.VALIDATE_GPU
CHECK = WorkflowOperation.CHECK_MECHANICALS
COLLECT = WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE
RESTART_NODE = WorkflowOperation.RESTART_NODE


def _local_wait(index: int, operation: WorkflowOperation) -> WorkflowStepExecution:
    return workflow_step_execution(index, operation, WorkflowStepStatus.WAITING)


def _remote_wait(
    index: int, operation: WorkflowOperation, remote_status: str
) -> WorkflowStepExecution:
    return workflow_step_execution(
        index,
        operation,
        WorkflowStepStatus.WAITING,
        adapter_operation_id=f"remote/cmd-{index}",
        details={"remote_status": remote_status, "remote_command_id": f"cmd-{index}"},
    )


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


def _existing(
    operations: list[WorkflowOperation], executions: list[WorkflowStepExecution]
) -> WorkflowRequest:
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
        official_action=RESTART_NODE.value,
        official_steps=[workflow_step(RESTART_NODE)],
    )


def _executor_supersedes(
    operations: list[WorkflowOperation],
    executions: list[WorkflowStepExecution],
    *,
    cancelled: list[str] | None = None,
) -> bool:
    """Run the executor against the same shape with a preempting successor."""

    store = build_store()
    incident, workflow = workflow_state(store, operations)
    running = copy_model(
        workflow,
        status=WorkflowStatus.RUNNING,
        completed_step_indexes=[0],
        completed_operations=[operations[0]],
        step_executions=executions,
    )
    store.save_workflow(running)
    _preempting_successor(store, incident, running)
    if cancelled is not None:

        def cancel(command_id: str, *, reason: str) -> bool:
            cancelled.append(command_id)
            return True

        store.cancel_remote_command = cancel
    adapter = FakeAdapter(
        {
            operation: WorkflowStepOutcome.waiting(operation_id=f"remote/cmd-{index}")
            for index, operation in enumerate(operations)
            if index > 0
        }
    )
    executor = active_workflow_executor(store, [adapter], set(operations))
    result = executor.execute(
        running.request_id,
        WorkflowExecutionRequest(expected_fencing_token=running.fencing_token),
    )
    return result.status is WorkflowStatus.SUPERSEDED


@pytest.mark.parametrize(
    "name, operations, executions, open_boundary",
    [
        ("pending step only", [QUARANTINE, FREEZE, RESET], [], True),
        (
            "local validation wait can be abandoned",
            [QUARANTINE, VALIDATE, FREEZE],
            [_local_wait(1, VALIDATE)],
            True,
        ),
        (
            "remote reset executing on the node",
            [QUARANTINE, RESET, FREEZE],
            [_remote_wait(1, RESET, "RUNNING")],
            False,
        ),
        (
            "remote reset the node has accepted",
            [QUARANTINE, RESET, FREEZE],
            [_remote_wait(1, RESET, "WAITING")],
            False,
        ),
        (
            "operator acknowledgement outstanding",
            [QUARANTINE, CHECK, FREEZE],
            [_local_wait(1, CHECK)],
            False,
        ),
    ],
)
def test_executor_and_merge_agree_on_the_boundary(
    name: str,
    operations: list[WorkflowOperation],
    executions: list[WorkflowStepExecution],
    open_boundary: bool,
) -> None:
    existing = _existing(operations, executions)

    executor_superseded = _executor_supersedes(operations, executions)
    combined = _merger().preempt_parallel_branch(existing, _candidate(), "node-a")
    merge_replaced = 1 in combined.superseded_step_indexes

    assert executor_superseded is open_boundary, name
    assert merge_replaced is open_boundary, name
    assert preemption_boundary(existing).open is open_boundary, name
    assert (
        preemption_boundary(existing, remote_cancellation_available=False).open
        is open_boundary
    ), name


def test_the_merge_now_chains_behind_an_operator_acknowledgement() -> None:
    """Deliberate change: CHECK_MECHANICALS is not flagged non-cancelable in the
    registry, so the merge used to replace a step whose operator was still
    being asked. The executor never abandoned that wait; now neither does the
    merge -- the new branch depends on the acknowledgement instead."""

    existing = _existing([QUARANTINE, CHECK, FREEZE], [_local_wait(1, CHECK)])

    combined = _merger().preempt_parallel_branch(existing, _candidate(), "node-a")

    appended = combined.official_steps[len(existing.official_steps) :]
    assert appended, "the stronger candidate still joins the workflow as a branch"
    assert 1 not in combined.superseded_step_indexes
    assert all(1 in step.depends_on_step_indexes for step in appended[:1]), (
        "the branch has to wait for the operator's answer"
    )
    boundary = preemption_boundary(existing)
    assert boundary.predecessor == 1
    assert "CHECK_MECHANICALS" in boundary.reason


def test_the_merge_holds_a_cancellable_remote_wait_the_executor_cancels() -> None:
    """Deliberate difference, made explicit: a collection the node has
    accepted can be cancelled, and the executor does so before superseding.
    Planning has no store to cancel with, so the same record is a boundary for
    the merge instead of a step it silently marks superseded while the command
    keeps running on the node."""

    operations = [QUARANTINE, COLLECT, FREEZE]
    executions = [_remote_wait(1, COLLECT, "WAITING")]
    existing = _existing(operations, executions)
    cancelled: list[str] = []

    superseded = _executor_supersedes(operations, executions, cancelled=cancelled)
    combined = _merger().preempt_parallel_branch(existing, _candidate(), "node-a")

    assert superseded, "the executor cancels the collection and gives way"
    assert cancelled == ["cmd-1"]
    assert 1 not in combined.superseded_step_indexes
    with_cancel = preemption_boundary(existing)
    without_cancel = preemption_boundary(existing, remote_cancellation_available=False)
    assert with_cancel.open, "with a store to cancel through the boundary is open"
    assert [item.remote_command_id for item in with_cancel.cancellations] == ["cmd-1"]
    assert not without_cancel.open, "without one the same record holds the line"
    assert without_cancel.predecessor == 1


def test_a_remote_wait_of_unknown_state_holds_both_sides() -> None:
    """Deliberate change on the merge side: a remote record that carries no
    ``remote_status`` used to be replaceable unless its operation was flagged
    non-cancelable. Nothing proves such a command has not started, so it holds
    -- as it always did for the executor."""

    existing = _existing(
        [QUARANTINE, COLLECT, FREEZE],
        [
            workflow_step_execution(
                1,
                COLLECT,
                WorkflowStepStatus.WAITING,
                adapter_operation_id="remote/cmd-unknown",
            )
        ],
    )

    combined = _merger().preempt_parallel_branch(existing, _candidate(), "node-a")

    assert 1 not in combined.superseded_step_indexes
    assert not preemption_boundary(existing).open, (
        "an unknown remote state is not a cancellable one"
    )


def test_a_stale_record_no_longer_holds_the_merge() -> None:
    """Deliberate change: the merge judged a WAITING record by the operation
    it recorded, the executor by the step now at that index. After a DAG
    rewrite the two can differ; a record for a step that is no longer there
    is history, not in-flight work, on both sides."""

    existing = _existing(
        [QUARANTINE, FREEZE, RESET], [_remote_wait(1, RESET, "RUNNING")]
    )

    combined = _merger().preempt_parallel_branch(existing, _candidate(), "node-a")

    assert 1 in combined.superseded_step_indexes
    boundary = preemption_boundary(existing)
    assert boundary.open, "a stale record is ignored, not a boundary"
    assert boundary.stale == frozenset({1})


def test_the_executor_cancels_nothing_while_another_branch_blocks() -> None:
    """Deliberate change: the executor used to cancel commands one record at
    a time and give up at the first it could not stop, leaving the earlier
    cancellations behind on a preemption that then did not happen. The
    boundary is judged whole first; a closed one issues no cancellation."""

    store = build_store()
    incident, workflow = workflow_state(store, [QUARANTINE, COLLECT, RESET])
    dag = copy_model(
        workflow,
        dag_enabled=True,
        status=WorkflowStatus.RUNNING,
        official_steps=[
            workflow.official_steps[0],
            copy_model(workflow.official_steps[1], depends_on_step_indexes=[0]),
            copy_model(workflow.official_steps[2], depends_on_step_indexes=[0]),
        ],
        completed_step_indexes=[0],
        completed_operations=[QUARANTINE],
        # The cancellable record comes first, as it did in the loop that
        # cancelled it before reaching the reset.
        step_executions=[
            _remote_wait(1, COLLECT, "PENDING"),
            _remote_wait(2, RESET, "RUNNING"),
        ],
    )
    store.save_workflow(dag)
    _preempting_successor(store, incident, dag)
    cancelled: list[str] = []

    def cancel(command_id: str, *, reason: str) -> bool:
        cancelled.append(command_id)
        return True

    store.cancel_remote_command = cancel
    adapter = FakeAdapter(
        {
            COLLECT: WorkflowStepOutcome.waiting(operation_id="remote/cmd-1"),
            RESET: WorkflowStepOutcome.waiting(operation_id="remote/cmd-2"),
        }
    )
    executor = active_workflow_executor(store, [adapter], {COLLECT, RESET})

    result = executor.execute(
        dag.request_id,
        WorkflowExecutionRequest(expected_fencing_token=dag.fencing_token),
    )

    assert result.status is not WorkflowStatus.SUPERSEDED
    assert cancelled == []


def test_a_record_outside_the_step_set_closes_the_whole_workflow_only() -> None:
    """The executor treated a record past the end of the step list as a reason
    not to preempt (it cannot say what is running); the merge only ever looked
    at its branch. Both readings survive: whole-workflow scope is closed,
    branch scope ignores what lies outside it."""

    existing = _existing(
        [QUARANTINE, FREEZE, RESET],
        [workflow_step_execution(7, RESET, WorkflowStepStatus.WAITING)],
    )

    whole = preemption_boundary(existing)
    branch = preemption_boundary(existing, indexes=[0, 1])

    assert not whole.open, "a record nobody can place is not one we can stop"
    assert whole.malformed == frozenset({7})
    assert "outside" in whole.reason
    assert branch.open, "a branch judgement only reads its own indexes"
    assert branch.replaceable == frozenset({1})
    assert branch.predecessor == 0
