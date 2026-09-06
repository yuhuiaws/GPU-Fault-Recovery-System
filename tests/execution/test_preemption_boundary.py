"""Preemption is only safe when *every* in-flight step can be stopped.

FINAL-建议汇总 F-C1 (P0-47A, P0-63A). ``_supersede_if_safe`` inspected the
WAITING record of the one step it was about to run. In a DAG another branch
may hold a remote command that is already executing on the node; superseding
the workflow at that moment abandons that command with nobody to collect it.
"""

from __future__ import annotations

from gpu_fault.execution.models import WorkflowExecutionRequest, WorkflowStepOutcome
from gpu_fault.models import WorkflowOperation, WorkflowStatus, WorkflowStepStatus
from tests._builders import build_store, copy_model, workflow_step_execution
from tests.execution._support import (
    FakeAdapter,
    _preempting_successor,
    active_workflow_executor,
    workflow_state,
)

FREEZE = WorkflowOperation.FREEZE_EVIDENCE
RESET = WorkflowOperation.RESET_GPU
QUARANTINE = WorkflowOperation.QUARANTINE


def _in_flight_remote(step_index: int, operation: WorkflowOperation):
    return workflow_step_execution(
        step_index,
        operation,
        WorkflowStepStatus.WAITING,
        adapter_operation_id=f"remote/cmd-{step_index}",
        details={"remote_status": "RUNNING", "remote_command_id": f"cmd-{step_index}"},
    )


def test_a_running_remote_command_on_another_branch_blocks_preemption():
    store = build_store()
    incident, workflow = workflow_state(store, [QUARANTINE, FREEZE, RESET])
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
        # Branch 2 already submitted a node-side reset that is executing.
        step_executions=[_in_flight_remote(2, RESET)],
    )
    store.save_workflow(dag)
    _preempting_successor(store, incident, dag)
    adapter = FakeAdapter(
        {
            FREEZE: WorkflowStepOutcome.succeeded(),
            RESET: WorkflowStepOutcome.waiting(operation_id="remote/cmd-2"),
        }
    )
    executor = active_workflow_executor(store, [adapter], {FREEZE, RESET})

    result = executor.execute(
        dag.request_id,
        WorkflowExecutionRequest(expected_fencing_token=dag.fencing_token),
    )

    # Not superseded: the reset on branch 2 cannot be cancelled, so the
    # workflow keeps executing and waits for it.
    assert result.status is not WorkflowStatus.SUPERSEDED
    assert store.get_workflow(dag.request_id).status is WorkflowStatus.RUNNING
    assert "workflow-active/1/FREEZE_EVIDENCE" in adapter.calls


def test_preemption_proceeds_when_the_other_branch_is_only_pending():
    """Control: the same shape with a cancellable (PENDING) remote command is
    superseded at the boundary as before."""

    store = build_store()
    incident, workflow = workflow_state(store, [QUARANTINE, FREEZE, RESET])
    pending = copy_model(
        _in_flight_remote(2, RESET),
        details={"remote_status": "PENDING", "remote_command_id": "cmd-2"},
    )
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
        step_executions=[pending],
    )
    store.save_workflow(dag)
    _preempting_successor(store, incident, dag)
    cancelled: list[str] = []

    def cancel(command_id: str, *, reason: str) -> bool:
        cancelled.append(command_id)
        return True

    store.cancel_remote_command = cancel
    adapter = FakeAdapter({FREEZE: WorkflowStepOutcome.succeeded()})
    executor = active_workflow_executor(store, [adapter], {FREEZE, RESET})

    result = executor.execute(
        dag.request_id,
        WorkflowExecutionRequest(expected_fencing_token=dag.fencing_token),
    )

    assert result.status is WorkflowStatus.SUPERSEDED
    assert adapter.calls == []
    assert cancelled == ["cmd-2"]
