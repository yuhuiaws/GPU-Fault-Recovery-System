from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.execution.terminal_state import terminal_decision
from gpu_fault.models import (
    BlockedKind,
    IncidentState,
    WorkflowEventKind,
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

NOW = datetime(2026, 9, 14, 12, tzinfo=timezone.utc)
RESET = WorkflowOperation.RESET_GPU
VALIDATE = WorkflowOperation.VALIDATE_GPU
QUARANTINE = WorkflowOperation.QUARANTINE
RESTORE = WorkflowOperation.RESTORE_SCHEDULING
SUPPORT = WorkflowOperation.ESCALATE_SUPPORT


@pytest.mark.parametrize(
    ("status", "planned", "completed", "expected", "inconclusive"),
    [
        (WorkflowStatus.FAILED, [VALIDATE], [], IncidentState.RECOVERED, True),
        (WorkflowStatus.FAILED, [VALIDATE, RESET], [], IncidentState.ESCALATED, False),
        (
            WorkflowStatus.FAILED,
            [QUARANTINE, RESET],
            [QUARANTINE],
            IncidentState.QUARANTINED,
            False,
        ),
        (
            WorkflowStatus.BLOCKED,
            [QUARANTINE],
            [QUARANTINE],
            IncidentState.QUARANTINED,
            False,
        ),
        (
            WorkflowStatus.SUCCEEDED,
            [QUARANTINE],
            [QUARANTINE],
            IncidentState.QUARANTINED,
            False,
        ),
        (
            WorkflowStatus.SUCCEEDED,
            [QUARANTINE, RESTORE],
            [QUARANTINE, RESTORE],
            IncidentState.RECOVERED,
            False,
        ),
        (
            WorkflowStatus.SUCCEEDED,
            [SUPPORT],
            [SUPPORT],
            IncidentState.ESCALATED,
            False,
        ),
        (
            WorkflowStatus.SUPERSEDED,
            [QUARANTINE],
            [QUARANTINE],
            IncidentState.QUARANTINED,
            False,
        ),
        (WorkflowStatus.SUPERSEDED, [RESET], [], IncidentState.RECOVERED, False),
        (WorkflowStatus.FAILED, [], [], IncidentState.ESCALATED, False),
    ],
)
def test_terminal_decision_is_pure_and_preserves_existing_state_derivation(
    status, planned, completed, expected, inconclusive
):
    workflow = workflow_request(
        "workflow",
        "incident",
        status=WorkflowStatus.RUNNING,
        official_steps=[workflow_step(operation) for operation in planned],
        completed_operations=completed,
        step_executions=[
            workflow_step_execution(index, operation, phase="official")
            for index, operation in enumerate(planned)
            if operation in completed
        ],
        execution_owner_id="executor",
        execution_epoch=7,
        execution_lease_expires_at=NOW + timedelta(minutes=1),
    )
    incident = fault_incident("incident", "event")
    original_workflow, original_incident = workflow.model_dump(), incident.model_dump()
    updates = {"terminal_failure_reason": "original failure"}
    first = terminal_decision(
        workflow,
        incident,
        status,
        7,
        now=NOW,
        actor="executor",
        reason="validation failed",
        updates=updates,
    )
    second = terminal_decision(
        workflow,
        incident,
        status,
        7,
        now=NOW,
        actor="executor",
        reason="validation failed",
        updates=updates,
    )
    assert first == second
    assert workflow.model_dump() == original_workflow
    assert incident.model_dump() == original_incident
    assert updates == {"terminal_failure_reason": "original failure"}
    assert first.workflow.execution_owner_id is None
    assert first.workflow.execution_lease_expires_at is None
    assert first.workflow.execution_epoch == 7
    assert first.workflow.status is status
    assert first.incident is not None and first.incident.state is expected
    assert first.diagnostic_inconclusive is inconclusive
    assert first.error == "validation failed"
    event = first.workflow.events[-1]
    assert event.kind is WorkflowEventKind.TERMINAL
    assert event.at == NOW and event.actor == "executor"
    assert event.details["execution_epoch"] == 7
    assert event.details["incident_state"] == expected.value
    assert event.details.get("diagnostic_inconclusive", False) is inconclusive


@pytest.mark.parametrize(
    "status",
    [
        WorkflowStatus.SUCCEEDED,
        WorkflowStatus.FAILED,
        WorkflowStatus.SUPERSEDED,
        WorkflowStatus.BLOCKED,
    ],
)
@pytest.mark.parametrize("isolated", [False, True])
def test_uncertain_physical_action_overrides_every_requested_terminal_status(
    status, isolated
):
    workflow = workflow_request(
        "workflow",
        "incident",
        official_steps=[workflow_step(RESET)],
        completed_operations=[QUARANTINE] if isolated else [],
        step_executions=[
            workflow_step_execution(
                0,
                RESET,
                WorkflowStepStatus.FAILED,
                phase="official",
                details={"outcome_unknown": True},
            )
        ],
    )
    result = terminal_decision(
        workflow,
        fault_incident("incident", "event"),
        status,
        1,
        now=NOW,
        actor="watchdog",
        updates={"blocked_kind": BlockedKind.SAFETY_SETTLED},
    )
    assert result.workflow.status is WorkflowStatus.BLOCKED
    assert result.workflow.blocked_kind is BlockedKind.NEEDS_OPERATOR
    assert result.incident is not None
    assert result.incident.state is (
        IncidentState.QUARANTINED if isolated else IncidentState.ESCALATED
    )
    assert result.error == "NODE_ACTION_OUTCOME_UNRESOLVED"
    assert not result.diagnostic_inconclusive, (
        "uncertain physical execution is not a completed inconclusive diagnostic"
    )


def test_missing_incident_and_explicit_state_do_not_invent_a_diagnostic_closure():
    workflow = workflow_request(
        "workflow", "incident", official_steps=[workflow_step(VALIDATE)]
    )
    result = terminal_decision(
        workflow,
        None,
        WorkflowStatus.FAILED,
        1,
        now=NOW,
        actor="watchdog",
        incident_state=IncidentState.ESCALATED,
    )
    assert result.incident is None
    assert not result.diagnostic_inconclusive, (
        "a missing incident must not create an inferred diagnostic closure"
    )
    assert result.workflow.events[-1].details["incident_state"] == "ESCALATED"
