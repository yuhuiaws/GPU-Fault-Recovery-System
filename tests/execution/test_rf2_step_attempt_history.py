"""RF-2: every attempt at a step leaves an event; the record still collapses.

``step_bounds.record_attempt`` keeps one ``step_executions`` record per step
identity and replaces it on every attempt, which is right for execution and
wrong for history: three WAITING passes and their reasons vanished the moment
the step succeeded. The record is unchanged here; the history moves to
``WorkflowRequest.events`` as STEP_ATTEMPT entries with a running attempt
number and a bounded, allow-listed subset of the outcome details.
"""

from __future__ import annotations

from gpu_fault.execution import step_bounds
from gpu_fault.execution.models import WorkflowStepOutcome
from gpu_fault.models import (
    WorkflowEventCode,
    WorkflowEventKind,
    WorkflowOperation,
    WorkflowStepStatus,
)
from tests._builders import workflow_request, workflow_step, workflow_step_execution

RESET = WorkflowOperation.RESET_GPU


def _workflow(**values):
    return workflow_request(
        "workflow-rf2",
        "incident-rf2",
        official_steps=[workflow_step(RESET, node_ids=["node-b"])],
        **values,
    )


def _attempts(workflow):
    return [
        event
        for event in workflow.events
        if event.kind is WorkflowEventKind.STEP_ATTEMPT
    ]


def test_three_waits_then_success_keep_one_record_and_three_attempt_events():
    """The second wait differs from the first only in ``step_waiting_seconds``:
    the same wait polled again, folded into the first event (D-10). The third
    names another reason and is recorded."""

    workflow = _workflow()
    step = workflow.official_steps[0]
    waits = (
        {"reason": "NODE_BUSY", "step_waiting_seconds": 0, "payload": {"big": 1}},
        {"reason": "NODE_BUSY", "step_waiting_seconds": 30, "payload": {"big": 2}},
        {"reason": "AGENT_LAG", "gpu_reset_commit_attempt": 2, "payload": [3]},
    )
    for details in waits:
        workflow = step_bounds.record_attempt(
            workflow,
            step,
            0,
            WorkflowStepOutcome.waiting(operation_id="cmd-1", details=dict(details)),
        )
    workflow = step_bounds.record_attempt(
        workflow, step, 0, WorkflowStepOutcome.succeeded(operation_id="cmd-1")
    )

    assert len(workflow.step_executions) == 1, workflow.step_executions
    assert workflow.step_executions[0].status is WorkflowStepStatus.SUCCEEDED
    attempts = _attempts(workflow)
    assert [event.details["attempt"] for event in attempts] == [1, 2, 3]
    assert [event.status for event in attempts] == ["WAITING", "WAITING", "SUCCEEDED"]
    assert [event.code for event in attempts] == [
        WorkflowEventCode.STEP_WAITING,
        WorkflowEventCode.STEP_WAITING,
        WorkflowEventCode.STEP_SUCCEEDED,
    ]
    assert [event.details.get("reason") for event in attempts[:2]] == [
        "NODE_BUSY",
        "AGENT_LAG",
    ]
    assert attempts[0].details["step_waiting_seconds"] == 0
    assert attempts[1].details["gpu_reset_commit_attempt"] == 2
    assert all("payload" not in event.details for event in attempts), (
        "whole outcome details must never be copied into an event"
    )
    assert all(
        event.details["adapter_operation_id"] == "cmd-1" for event in attempts
    ), attempts
    assert all(event.step_index == 0 for event in attempts), attempts
    assert all(event.operation is RESET for event in attempts), attempts
    assert all(event.phase == "official" for event in attempts), attempts


def test_a_failed_attempt_keeps_its_error_as_the_event_reason():
    workflow = _workflow()
    step = workflow.official_steps[0]

    workflow = step_bounds.record_attempt(
        workflow,
        step,
        0,
        WorkflowStepOutcome.failed("reset refused", details={"reason": "EBUSY"}),
    )

    (event,) = _attempts(workflow)
    assert event.code == WorkflowEventCode.STEP_FAILED
    assert event.reason == "reset refused"
    assert event.details["reason"] == "EBUSY"


def test_deadline_and_lifetime_failures_carry_their_own_codes():
    workflow = _workflow()
    step = workflow.official_steps[0]

    deadline = step_bounds.record_attempt(
        workflow,
        step,
        0,
        WorkflowStepOutcome.failed(
            "workflow execution deadline exceeded",
            details={
                "workflow_execution_deadline": "2026-09-06T00:00:00+00:00",
                "workflow_lifetime_exceeded": False,
            },
        ),
    )
    lifetime = step_bounds.record_attempt(
        workflow,
        step,
        0,
        WorkflowStepOutcome.failed(
            "workflow lifetime exceeded",
            details={
                "workflow_execution_deadline": "2026-09-06T00:00:00+00:00",
                "workflow_lifetime_exceeded": True,
            },
        ),
    )
    waited_out = step_bounds.record_attempt(
        workflow,
        step,
        0,
        WorkflowStepOutcome.failed(
            "stayed non-terminal past the cap",
            details={"step_waiting_seconds": 601, "step_waiting_timeout_seconds": 600},
        ),
    )

    assert _attempts(deadline)[0].code == WorkflowEventCode.DEADLINE_EXCEEDED
    assert _attempts(lifetime)[0].code == WorkflowEventCode.LIFETIME_EXCEEDED
    assert _attempts(waited_out)[0].code == WorkflowEventCode.STEP_WAITING_TIMEOUT


def test_a_legacy_record_without_a_phase_is_still_this_steps_record():
    legacy = workflow_step_execution(0, RESET, WorkflowStepStatus.WAITING, phase=None)
    workflow = _workflow(step_executions=[legacy])
    step = workflow.official_steps[0]

    workflow = step_bounds.record_attempt(
        workflow, step, 0, WorkflowStepOutcome.succeeded()
    )

    assert len(workflow.step_executions) == 1, workflow.step_executions
    assert workflow.step_executions[0].phase == "official"
    assert workflow.step_executions[0].started_at == legacy.started_at
    # The legacy record predates events, so this is attempt 1 of the history.
    assert [event.details["attempt"] for event in _attempts(workflow)] == [1]


def test_step_attempt_history_is_scoped_to_one_step_identity():
    workflow = workflow_request(
        "workflow-rf2-two",
        "incident-rf2",
        official_steps=[
            workflow_step(RESET, node_ids=["node-b"]),
            workflow_step(WorkflowOperation.VALIDATE_GPU, node_ids=["node-b"]),
        ],
    )
    reset, validate = workflow.official_steps
    workflow = step_bounds.record_attempt(
        workflow, reset, 0, WorkflowStepOutcome.waiting()
    )
    workflow = step_bounds.record_attempt(
        workflow, reset, 0, WorkflowStepOutcome.succeeded()
    )
    workflow = step_bounds.record_attempt(
        workflow, validate, 1, WorkflowStepOutcome.succeeded()
    )

    reset_history = step_bounds.step_attempt_history(workflow, 0, RESET)
    validate_history = step_bounds.step_attempt_history(
        workflow, 1, WorkflowOperation.VALIDATE_GPU, "official"
    )
    other_phase = step_bounds.step_attempt_history(workflow, 0, RESET, "safety")

    assert [event.status for event in reset_history] == ["WAITING", "SUCCEEDED"]
    assert [event.details["attempt"] for event in validate_history] == [1]
    assert other_phase == [], other_phase
