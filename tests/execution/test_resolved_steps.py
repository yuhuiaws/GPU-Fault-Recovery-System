"""A superseded step never executes, so every "still pending" question must
treat it as resolved -- and the executor must not re-run destructive steps
when a compensation is in flight.

FINAL-建议汇总 F-C2 (P0-52A, P0-62D, P0-65B, P1-66C, P1-67C). ``completed``
and ``superseded`` were consulted inconsistently: the non-DAG loop skipped
only completed steps, the two compensation paths picked a RESTORE step that
had been superseded, the DAG termination test compared *counts* so an
out-of-range index ended a workflow early, ``_append_chain`` dropped the
candidate branch's own dependencies, and ``pending_failure_step_index`` was
written but never read at entry, so the next dispatch re-ran the failed
destructive step instead of finishing the compensation.
"""

from __future__ import annotations

from gpu_fault.execution.models import WorkflowExecutionRequest
from gpu_fault.models import (
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepSpec,
    resolved_step_indexes,
)
from gpu_fault.orchestration.arbitration import RecoveryArbiter
from gpu_fault.orchestration.dag_branching import DagBrancher
from tests._builders import build_store, copy_model, workflow_step
from tests.execution._support import (
    FakeAdapter,
    WorkflowStepOutcome,
    active_workflow_executor,
    workflow_state,
)

QUARANTINE = WorkflowOperation.QUARANTINE
RESTART = WorkflowOperation.RESTART_NODE
RESTORE_SCHED = WorkflowOperation.RESTORE_SCHEDULING
QUIESCE = WorkflowOperation.QUIESCE_GPU_SERVICES
RESTORE_GPU = WorkflowOperation.RESTORE_GPU_SERVICES


def _ok(*operations: WorkflowOperation) -> FakeAdapter:
    return FakeAdapter({op: WorkflowStepOutcome.succeeded() for op in operations})


def _execute(store, adapter, workflow):
    executor = active_workflow_executor(store, [adapter], set(adapter.outcomes))
    return executor.execute(
        workflow.request_id,
        WorkflowExecutionRequest(expected_fencing_token=workflow.fencing_token),
    )


def test_resolved_is_completed_union_superseded():
    store = build_store()
    _, workflow = workflow_state(store, [QUARANTINE, RESTART, RESTORE_SCHED])
    workflow = copy_model(
        workflow, completed_step_indexes=[0], superseded_step_indexes=[1, 1]
    )

    assert resolved_step_indexes(workflow) == frozenset({0, 1})


def test_the_sequential_loop_skips_a_superseded_step():
    store = build_store()
    _, workflow = workflow_state(store, [QUARANTINE, RESTART, RESTORE_SCHED])
    store.save_workflow(copy_model(workflow, superseded_step_indexes=[1]))
    adapter = _ok(QUARANTINE, RESTART, RESTORE_SCHED)

    result = _execute(store, adapter, workflow)

    assert result.status is WorkflowStatus.SUCCEEDED
    assert len(adapter.calls) == 2
    assert store.get_workflow(workflow.request_id).completed_step_indexes == [0, 2]


def test_dag_termination_is_set_coverage_not_a_count():
    """P0-65B: with ``completed == [0, 7]`` and two steps the count matched
    and the workflow ended SUCCEEDED without step 1 ever running."""

    store = build_store()
    _, workflow = workflow_state(store, [QUARANTINE, RESTORE_SCHED])
    steps = [
        workflow.official_steps[0],
        workflow.official_steps[1].model_copy(update={"depends_on_step_indexes": [0]}),
    ]
    store.save_workflow(
        copy_model(
            workflow,
            dag_enabled=True,
            official_steps=steps,
            completed_step_indexes=[0, 7],
            completed_operations=[QUARANTINE],
        )
    )
    adapter = _ok(QUARANTINE, RESTORE_SCHED)

    result = _execute(store, adapter, workflow)

    assert result.status is WorkflowStatus.SUCCEEDED
    assert len(adapter.calls) == 1
    assert 1 in store.get_workflow(workflow.request_id).completed_step_indexes


def test_a_pending_compensation_resumes_instead_of_re_running_the_failed_step():
    """P1-67C: the failure marker was written but never read at entry."""

    store = build_store()
    _, workflow = workflow_state(store, [QUIESCE, RESTART, RESTORE_GPU])
    store.save_workflow(
        copy_model(
            workflow,
            completed_step_indexes=[0],
            completed_operations=[QUIESCE],
            pending_failure_step_index=1,
            pending_failure_error="node did not come back",
        )
    )
    adapter = _ok(QUIESCE, RESTART, RESTORE_GPU)

    result = _execute(store, adapter, workflow)

    assert result.status is WorkflowStatus.FAILED
    assert len(adapter.calls) == 1  # only RESTORE_GPU_SERVICES
    final = store.get_workflow(workflow.request_id)
    assert final.pending_failure_step_index is None
    assert RESTORE_GPU in final.completed_operations
    assert RESTART not in final.completed_operations


def test_compensation_does_not_pick_a_superseded_restore_step():
    store = build_store()
    _, workflow = workflow_state(store, [QUIESCE, RESTART, RESTORE_GPU, RESTORE_GPU])
    store.save_workflow(
        copy_model(
            workflow,
            completed_step_indexes=[0],
            completed_operations=[QUIESCE],
            superseded_step_indexes=[2],
            pending_failure_step_index=1,
            pending_failure_error="node did not come back",
        )
    )
    adapter = _ok(QUIESCE, RESTART, RESTORE_GPU)

    result = _execute(store, adapter, workflow)

    assert result.status is WorkflowStatus.FAILED
    assert store.get_workflow(workflow.request_id).completed_step_indexes == [0, 3]


def test_append_chain_keeps_the_candidate_branch_dependencies():
    steps: list[WorkflowStepSpec] = [workflow_step(QUARANTINE)]
    values = [
        workflow_step(QUIESCE),
        workflow_step(RESTART),
        workflow_step(RESTORE_GPU, depends_on_step_indexes=[0]),
    ]
    appended: list[int] = []

    tail = DagBrancher._append_chain(steps, values, 0, "branch-a", appended)

    assert tail == 3
    assert appended == [1, 2, 3]
    # The chain still runs in order, and the candidate's own edge
    # (RESTORE after QUIESCE, local index 0 -> appended index 1) survives.
    assert steps[3].depends_on_step_indexes == [1, 2]


def test_attach_candidate_branch_returns_a_tail_at_index_zero():
    """``suffix_tail or prefix_tail`` treated index 0 as "no tail"."""

    brancher = DagBrancher(RecoveryArbiter())
    steps: list[WorkflowStepSpec] = []
    appended: list[int] = []

    tail = brancher._attach_candidate_branch(
        steps,
        [],
        [workflow_step(RESTART)],
        "branch-a",
        None,
        None,
        False,
        None,
        appended,
    )

    assert tail == 0
    assert appended == [0]
