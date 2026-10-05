"""A FAILED workflow records why on the record, not only in its TERMINAL event.

Live on HyperPod (GF-REGIONAL-COLLECT-016) both records of a reset-to-reboot
chain -- the drill whose VALIDATE_GPU hit the per-step cap and the
``reboot-after`` successor refused on its first isolation -- ended FAILED with
``terminal_failure_reason`` None; the reason lived only in the TERMINALIZED
event. The API, the acceptance verdicts and the failure handler read the field.
"""

from __future__ import annotations

from datetime import datetime, timezone

from gpu_fault.execution.terminal_state import terminal_decision
from gpu_fault.models import (
    IncidentState,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
)
from tests._builders import (
    fault_incident,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)

NOW = datetime(2026, 10, 5, 2, 36, 46, tzinfo=timezone.utc)
CAP_REASON = (
    "step 7/VALIDATE_GPU stayed non-terminal for 601s, past the 600s per-step cap"
)


def _running(*operations: WorkflowOperation, completed=()):
    return workflow_request(
        "workflow",
        "incident",
        status=WorkflowStatus.RUNNING,
        official_steps=[workflow_step(operation) for operation in operations],
        completed_operations=list(completed),
        step_executions=[
            workflow_step_execution(index, operation, phase="official")
            for index, operation in enumerate(operations)
            if operation in completed
        ],
        execution_owner_id="executor",
        execution_epoch=1,
    )


def _decide(workflow, status, **kwargs):
    return terminal_decision(
        workflow,
        fault_incident("incident", "event"),
        status,
        1,
        now=NOW,
        actor="executor",
        **kwargs,
    )


def test_a_failed_workflow_carries_its_reason_on_the_record() -> None:
    workflow = _running(
        WorkflowOperation.MARK_UNSCHEDULABLE,
        WorkflowOperation.RESET_GPU,
        WorkflowOperation.VALIDATE_GPU,
        completed=(WorkflowOperation.MARK_UNSCHEDULABLE, WorkflowOperation.RESET_GPU),
    )

    decision = _decide(workflow, WorkflowStatus.FAILED, reason=CAP_REASON)

    assert decision.workflow.terminal_failure_reason == CAP_REASON
    assert decision.workflow.events[-1].reason == CAP_REASON, (
        "the event keeps the same text; the field is no longer the only place"
    )
    assert decision.incident is not None
    assert decision.incident.state is IncidentState.QUARANTINED


def test_a_reason_set_ahead_of_time_is_not_overwritten() -> None:
    workflow = _running(
        WorkflowOperation.STOP_WORKLOADS, WorkflowOperation.RESTART_WORKLOAD
    )
    budget = "restart budget exhausted for cluster-a/job: 1/1"

    decision = _decide(
        workflow,
        WorkflowStatus.FAILED,
        reason=budget,
        updates={"terminal_failure_reason": budget},
    )

    assert decision.workflow.terminal_failure_reason == budget
    preset = _decide(
        workflow,
        WorkflowStatus.FAILED,
        reason="later text",
        updates={"terminal_failure_reason": budget},
    )
    assert preset.workflow.terminal_failure_reason == budget, (
        "the rule A / budget paths own the field when they set it"
    )


def test_only_a_failed_record_gains_the_field() -> None:
    workflow = _running(WorkflowOperation.VALIDATE_GPU, WorkflowOperation.RESET_GPU)

    succeeded = _decide(workflow, WorkflowStatus.SUCCEEDED, reason="done")
    assert succeeded.workflow.terminal_failure_reason is None
    superseded = _decide(workflow, WorkflowStatus.SUPERSEDED, reason="replaced")
    assert superseded.workflow.terminal_failure_reason is None
    blocked = _decide(workflow, WorkflowStatus.BLOCKED, reason="needs an operator")
    assert blocked.workflow.terminal_failure_reason is None, (
        "a BLOCKED record may resume; it is not a terminal failure"
    )
    silent = _decide(workflow, WorkflowStatus.FAILED)
    assert silent.workflow.terminal_failure_reason is None, "no reason, no text"


def test_a_diagnostic_only_failure_records_the_inconclusive_reason() -> None:
    workflow = _running(WorkflowOperation.VALIDATE_GPU)

    decision = _decide(workflow, WorkflowStatus.FAILED, reason="no telemetry")

    assert decision.diagnostic_inconclusive is True
    assert decision.workflow.terminal_failure_reason == (
        "diagnostic inconclusive: no telemetry"
    )
    assert (
        decision.workflow.events[-1].reason == "diagnostic inconclusive: no telemetry"
    )


def test_an_unresolved_node_action_blocks_without_a_terminal_reason() -> None:
    workflow = _running(WorkflowOperation.RESTART_NODE).model_copy(
        update={
            "step_executions": [
                workflow_step_execution(
                    0,
                    WorkflowOperation.RESTART_NODE,
                    WorkflowStepStatus.FAILED,
                    phase="official",
                    details={
                        "outcome_unknown": True,
                        "manual_confirmation_required": True,
                    },
                )
            ]
        }
    )

    decision = _decide(workflow, WorkflowStatus.FAILED, reason="reboot cap")

    assert decision.workflow.status is WorkflowStatus.BLOCKED
    assert decision.workflow.terminal_failure_reason is None
