from __future__ import annotations

import pytest

from gpu_fault.execution import WorkflowStepOutcome
from gpu_fault.models import WorkflowOperation, WorkflowStatus, WorkflowStepStatus
from tests._builders import (
    fault_incident,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)
from tests.execution._cov95_runtime_workflows import FlowHarness

QUIESCE = WorkflowOperation.QUIESCE_GPU_SERVICES
RESET = WorkflowOperation.RESET_GPU
RESTORE = WorkflowOperation.RESTORE_GPU_SERVICES
REBOOT = WorkflowOperation.RESTART_NODE


def successor(h, *, operation=REBOOT, nodes=None, foreign=False):
    incident_id = "other-incident" if foreign else h.incident.incident_id
    value = workflow_request(
        "next-workflow",
        incident_id,
        fencing_token=h.workflow.fencing_token,
        predecessor_workflow_id=h.workflow.request_id,
        preempt_predecessor=True,
        preemption_reason="stronger recovery",
        official_steps=[workflow_step(operation, node_ids=nodes or ["node-a"])],
    )
    h.store.save_workflow(value)
    if foreign:
        h.store.save_incident(
            fault_incident(
                incident_id,
                "other-event",
                workflow_request_id=value.request_id,
                fencing_token=value.fencing_token,
            )
        )
    else:
        current = h.store.get_incident(h.incident.incident_id)
        h.store.save_incident(
            current.model_copy(update={"workflow_request_id": value.request_id}),
            expected=current,
        )
    return value


@pytest.mark.parametrize("restore_fails", [False, True])
def test_failed_reset_waits_for_compensation_without_replaying_the_reset(
    restore_fails: bool,
) -> None:
    h = FlowHarness([QUIESCE, RESET, RESTORE])
    h.adapter.outcomes[RESET] = WorkflowStepOutcome.failed("reset failed")
    h.adapter.outcomes[RESTORE] = WorkflowStepOutcome.waiting(
        operation_id="restore-wait"
    )
    first = h.execute()
    assert first.status is WorkflowStatus.RUNNING and first.waiting_step_index == 2, (
        first
    )
    held = h.store.get_workflow(h.workflow.request_id)
    assert (
        held.pending_failure_step_index == 1
        and held.pending_failure_error == "reset failed"
    ), held
    h.adapter.outcomes[RESTORE] = (
        WorkflowStepOutcome.failed("restore failed")
        if restore_fails
        else WorkflowStepOutcome.succeeded()
    )
    second = h.execute()
    assert second.status is WorkflowStatus.FAILED and "reset failed" in (
        second.error or ""
    ), second
    assert ("restore compensation failed" in (second.error or "")) is restore_fails, (
        second
    )
    assert [context.step.operation for context in h.adapter.calls] == [
        QUIESCE,
        RESET,
        RESTORE,
        RESTORE,
    ], h.adapter.calls
    saved = h.store.get_workflow(h.workflow.request_id)
    assert (
        saved.pending_failure_step_index is None and saved.pending_failure_error is None
    ), saved
    assert (
        saved.execution_owner_id is None and saved.execution_lease_expires_at is None
    ), saved


def test_missing_compensation_is_reported_without_inventing_successful_cleanup() -> (
    None
):
    h = FlowHarness([QUIESCE, RESET])
    h.adapter.outcomes[RESET] = WorkflowStepOutcome.failed("reset failed")
    result = h.execute()
    assert result.status is WorkflowStatus.FAILED, result
    assert "no RESTORE_GPU_SERVICES compensation step" in (result.error or ""), result
    saved = h.store.get_workflow(h.workflow.request_id)
    assert saved.completed_operations == [QUIESCE], saved
    assert [context.step.operation for context in h.adapter.calls] == [
        QUIESCE,
        RESET,
    ], h.adapter.calls


@pytest.mark.parametrize(
    "handoff",
    [
        "same-scope",
        "foreign-incident",
        "different-node",
        "replacement",
        "missing-proof",
    ],
)
def test_preemption_restores_before_transferring_unproven_quiesce_ownership(
    handoff: str,
) -> None:
    h = FlowHarness([QUIESCE, RESET, RESTORE])
    h.amend(
        status=WorkflowStatus.RUNNING,
        completed_step_indexes=[0],
        completed_operations=[QUIESCE],
        step_executions=(
            []
            if handoff == "missing-proof"
            else [workflow_step_execution(0, QUIESCE, WorkflowStepStatus.SUCCEEDED)]
        ),
    )
    next_workflow = successor(
        h,
        operation=WorkflowOperation.REPLACE_NODE
        if handoff == "replacement"
        else REBOOT,
        nodes=["node-b"] if handoff == "different-node" else ["node-a"],
        foreign=handoff == "foreign-incident",
    )
    result = h.execute()
    assert result.status is WorkflowStatus.SUPERSEDED, result
    expected = [] if handoff == "same-scope" else [RESTORE]
    assert [context.step.operation for context in h.adapter.calls] == expected, (
        h.adapter.calls
    )
    saved = h.store.get_workflow(h.workflow.request_id)
    assert saved.preempted_by_workflow_id == next_workflow.request_id, saved
    assert RESET not in saved.completed_operations, saved
    assert all(context.step.node_ids == ["node-a"] for context in h.adapter.calls), (
        h.adapter.calls
    )


@pytest.mark.parametrize("restore_fails", [False, True])
def test_preemption_compensation_waits_and_records_its_final_outcome(
    restore_fails: bool,
) -> None:
    h = FlowHarness([QUIESCE, RESET, RESTORE])
    h.amend(
        status=WorkflowStatus.RUNNING,
        completed_step_indexes=[0],
        completed_operations=[QUIESCE],
        step_executions=[
            workflow_step_execution(0, QUIESCE, WorkflowStepStatus.SUCCEEDED)
        ],
    )
    next_workflow = successor(h, operation=WorkflowOperation.REPLACE_NODE)
    h.adapter.outcomes[RESTORE] = WorkflowStepOutcome.waiting(
        operation_id="restore-wait"
    )
    first = h.execute()
    assert first.status is WorkflowStatus.RUNNING and first.waiting_step_index == 2, (
        first
    )
    h.adapter.outcomes[RESTORE] = (
        WorkflowStepOutcome.failed("preemption restore failed")
        if restore_fails
        else WorkflowStepOutcome.succeeded()
    )
    second = h.execute()
    assert second.status is (
        WorkflowStatus.FAILED if restore_fails else WorkflowStatus.SUPERSEDED
    ), second
    saved = h.store.get_workflow(h.workflow.request_id)
    assert saved.preempted_by_workflow_id == next_workflow.request_id, saved
    assert RESET not in saved.completed_operations, saved
    if restore_fails:
        assert "preemption compensation failed" in (saved.preemption_reason or ""), (
            saved
        )
    assert (
        h.store.get_incident(h.incident.incident_id).workflow_request_id
        == next_workflow.request_id
    ), saved
