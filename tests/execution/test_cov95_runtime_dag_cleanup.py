from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.execution import WorkflowStepOutcome
from gpu_fault.execution.restart_budget_preflight import claim_deadlines
from gpu_fault.models import WorkflowOperation, WorkflowStatus, WorkflowStepStatus
from tests._builders import workflow_request, workflow_step, workflow_step_execution
from tests.execution._cov95_runtime_workflows import FlowHarness

QUIESCE = WorkflowOperation.QUIESCE_GPU_SERVICES
RESET = WorkflowOperation.RESET_GPU
RESTORE = WorkflowOperation.RESTORE_GPU_SERVICES


def dag_harness():
    h = FlowHarness([QUIESCE, RESET, RESTORE])
    h.amend(
        dag_enabled=True,
        dag_revision=1,
        official_steps=[
            step.model_copy(
                update={
                    "branch_id": "node-a",
                    "depends_on_step_indexes": [index - 1] if index else [],
                }
            )
            for index, step in enumerate(h.workflow.official_steps)
        ],
    )
    return h


@pytest.mark.parametrize("quiesced", [False, True])
def test_withdrawn_dag_restores_only_touched_state_without_running_the_reset(quiesced):
    h = dag_harness()
    h.amend(
        status=WorkflowStatus.RUNNING,
        workload_withdrawn_at=datetime.now(timezone.utc),
        workload_withdrawn_reason="operator stopped the attempt",
        completed_step_indexes=[0] if quiesced else [],
        completed_operations=[QUIESCE] if quiesced else [],
        step_executions=[
            workflow_step_execution(0, QUIESCE, WorkflowStepStatus.SUCCEEDED)
        ]
        if quiesced
        else [],
    )
    result = h.execute()
    assert result.status is WorkflowStatus.SUPERSEDED, result
    assert [call.step.operation for call in h.adapter.calls] == (
        [RESTORE] if quiesced else []
    ), h.adapter.calls
    saved = h.store.get_workflow(h.workflow.request_id)
    assert RESET not in saved.completed_operations, saved
    assert saved.execution_owner_id is None, saved
    assert saved.superseded_step_indexes == ([1] if quiesced else [0, 1, 2]), saved


@pytest.mark.parametrize("restore_waits", [False, True])
def test_dag_reset_failure_finishes_compensation_before_landing_the_failure(
    restore_waits,
):
    h = dag_harness()
    h.adapter.outcomes[RESET] = WorkflowStepOutcome.failed("reset failed")
    if restore_waits:
        h.adapter.outcomes[RESTORE] = WorkflowStepOutcome.waiting(
            operation_id="restore-pending"
        )
    result = h.execute()
    if restore_waits:
        assert result.status is WorkflowStatus.RUNNING, result
        held = h.store.get_workflow(h.workflow.request_id)
        assert held.pending_failure_step_index == 1, held
        h.adapter.outcomes[RESTORE] = WorkflowStepOutcome.succeeded()
        result = h.execute()
    assert result.status is WorkflowStatus.FAILED, result
    assert "reset failed" in (result.error or ""), result
    assert [call.step.operation for call in h.adapter.calls] == [
        QUIESCE,
        RESET,
        RESTORE,
        *([RESTORE] if restore_waits else []),
    ], h.adapter.calls
    saved = h.store.get_workflow(h.workflow.request_id)
    assert saved.pending_failure_step_index is None, saved
    assert saved.pending_failure_error is None and saved.execution_owner_id is None, (
        saved
    )
    assert RESTORE in saved.completed_operations, saved


def test_ordinary_waiting_restore_does_not_require_a_preempting_successor():
    h = FlowHarness([QUIESCE, RESTORE])
    h.adapter.outcomes[RESTORE] = WorkflowStepOutcome.waiting(operation_id="restore")
    waiting = h.execute()
    assert waiting.status is WorkflowStatus.RUNNING, waiting
    h.adapter.outcomes[RESTORE] = WorkflowStepOutcome.succeeded()
    completed = h.execute()
    assert completed.status is WorkflowStatus.SUCCEEDED, completed
    assert [call.step.operation for call in h.adapter.calls] == [
        QUIESCE,
        RESTORE,
        RESTORE,
    ], h.adapter.calls
    assert (
        h.store.get_workflow(h.workflow.request_id).preempted_by_workflow_id is None
    ), completed


@pytest.mark.parametrize(
    ("operation", "floor", "lifetime"),
    [
        (WorkflowOperation.CHECK_MECHANICALS, 3600, 3600),
        (WorkflowOperation.REMEDIATE_DRIVER, 900, 7200),
    ],
)
def test_claim_deadlines_apply_operation_floor_only_to_the_initial_execution_claim(
    operation, floor, lifetime
):
    now = datetime(2026, 9, 12, 12, tzinfo=timezone.utc)
    workflow = workflow_request(
        "deadline-workflow",
        "deadline-incident",
        official_steps=[workflow_step(operation)],
    )
    kwargs = {
        "timeout_seconds": 60,
        "node_lifetime_seconds": lifetime,
        "job_lifetime_seconds": lifetime,
        "operator_acknowledgement_seconds": 3600,
        "install_floor_seconds": 900,
    }
    execution_deadline, lifetime_deadline = claim_deadlines(workflow, now=now, **kwargs)
    assert execution_deadline == now + timedelta(seconds=floor), execution_deadline
    assert lifetime_deadline == now + timedelta(seconds=lifetime), lifetime_deadline
    resumed = workflow.model_copy(
        update={
            "execution_deadline": execution_deadline,
            "lifetime_deadline_at": lifetime_deadline,
        }
    )
    later, _ = claim_deadlines(resumed, now=now + timedelta(seconds=30), **kwargs)
    assert later == execution_deadline, (
        "repeated claims must not slide an operation's execution deadline"
    )
