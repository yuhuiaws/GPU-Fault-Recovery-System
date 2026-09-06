"""One terminalization path, and every ending leaves exactly one TERMINAL event.

Review clean-up item 1 (RF-1). The executor's ``_terminalize`` was the funnel
for its own endings; the dispatcher's deadline reap and internal-error BLOCK
each wrote their own copy, and the reap never touched the incident, so an
incident whose workflow the watchdog failed stayed ACTION_PENDING for ever.
Both dispatcher paths now hand the claimed record to the executor's
owner-aware ``terminalize_claimed``. Every terminal write -- executor or
dispatcher -- appends one ``TERMINAL`` audit event naming its actor, and a
successful claim appends one ``CLAIM`` event.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from gpu_fault.execution.config import WorkflowDispatcherConfig
from gpu_fault.execution.dispatcher import WorkflowDispatcher
from gpu_fault.models import (
    IncidentState,
    WorkflowEventKind,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
)
from gpu_fault.store.shared.errors import WorkflowLeaseError
from tests._builders import (
    active_workflow_executor,
    build_store,
    copy_model,
    execute_workflow,
)
from tests.execution._support import (
    FakeAdapter,
    WorkflowStepOutcome,
    _preempting_successor,
    workflow_state,
)

QUARANTINE = WorkflowOperation.QUARANTINE
FREEZE = WorkflowOperation.FREEZE_EVIDENCE
RESET = WorkflowOperation.RESET_GPU
WATCHDOG = "dispatcher-watchdog"
INTERNAL = "dispatcher-internal-error"


def _events(workflow: WorkflowRequest, kind: WorkflowEventKind):
    return [event for event in workflow.events if event.kind is kind]


def _dispatcher(store, adapter, operations) -> WorkflowDispatcher:
    return WorkflowDispatcher(
        store,
        active_workflow_executor(store, [adapter], operations),
        WorkflowDispatcherConfig(enabled=True, batch_size=10, max_workers=1),
    )


def _overdue(store, operations, **values):
    _, workflow = workflow_state(store, operations)
    now = datetime.now(timezone.utc)
    overdue = copy_model(
        workflow,
        status=WorkflowStatus.RUNNING,
        execution_owner_id="executor-gone",
        execution_epoch=1,
        execution_lease_expires_at=now - timedelta(minutes=5),
        execution_deadline=now - timedelta(minutes=1),
        **values,
    )
    store.save_workflow(overdue)
    return overdue


@pytest.mark.parametrize(
    "completed, expected",
    [([], IncidentState.ESCALATED), ([QUARANTINE], IncidentState.QUARANTINED)],
    ids=["nothing-isolated-escalates", "isolated-stays-quarantined"],
)
def test_the_watchdog_reap_ends_the_incident_in_the_derived_state(
    completed, expected
) -> None:
    store = build_store()
    overdue = _overdue(
        store,
        [QUARANTINE, RESET],
        completed_step_indexes=list(range(len(completed))),
        completed_operations=completed,
    )
    assert store.get_incident(overdue.incident_id).state is IncidentState.ACTION_PENDING

    _dispatcher(store, FakeAdapter({}), [QUARANTINE, RESET]).run_once()

    failed = store.get_workflow(overdue.request_id)
    assert failed.status is WorkflowStatus.FAILED
    assert failed.execution_owner_id is None, "the funnel drops the owner"
    incident = store.get_incident(overdue.incident_id)
    assert incident.state is expected, (
        "a reaped workflow's incident must leave ACTION_PENDING through the "
        "same derivation the executor's own FAILED uses"
    )


def test_the_watchdog_reap_records_one_terminal_event_as_the_watchdog() -> None:
    store = build_store()
    overdue = _overdue(store, [QUARANTINE, RESET])

    _dispatcher(store, FakeAdapter({}), [QUARANTINE, RESET]).run_once()

    failed = store.get_workflow(overdue.request_id)
    terminal = _events(failed, WorkflowEventKind.TERMINAL)
    assert len(terminal) == 1, [event.model_dump() for event in failed.events]
    assert terminal[0].actor == WATCHDOG
    assert terminal[0].status == WorkflowStatus.FAILED.value
    assert "deadline" in (terminal[0].reason or "")
    assert terminal[0].details["incident_state"] == IncidentState.ESCALATED.value


def test_blocking_on_an_internal_error_records_one_terminal_event() -> None:
    store = build_store()
    _, workflow = workflow_state(store, [QUARANTINE])
    try:
        WorkflowRequest.model_validate({"incident_id": "x"})
    except ValidationError as error:
        invalid = error

    class _Raising:
        owner = "simulated-runtime"

        def supports(self, _step) -> bool:
            raise invalid

        def execute(self, _context):
            raise AssertionError("unreachable")

    _dispatcher(store, _Raising(), [QUARANTINE]).run_once()

    blocked = store.get_workflow(workflow.request_id)
    assert blocked.status is WorkflowStatus.BLOCKED
    terminal = _events(blocked, WorkflowEventKind.TERMINAL)
    assert len(terminal) == 1, [event.model_dump() for event in blocked.events]
    assert terminal[0].actor == INTERNAL
    assert terminal[0].status == WorkflowStatus.BLOCKED.value
    assert "ValidationError" in (terminal[0].reason or "")
    # The path names its own incident state: an internal error is an
    # operator matter, not a settled safety phase.
    assert store.get_incident(workflow.incident_id).state is IncidentState.ESCALATED


def test_the_executors_own_ending_records_a_claim_and_one_terminal_event() -> None:
    store = build_store()
    _, workflow = workflow_state(store, [FREEZE, RESET])
    adapter = FakeAdapter(
        {
            FREEZE: WorkflowStepOutcome.succeeded(),
            RESET: WorkflowStepOutcome.succeeded(),
        }
    )
    executor = active_workflow_executor(store, [adapter], [FREEZE, RESET])

    result = execute_workflow(executor, workflow.request_id)

    assert result.status is WorkflowStatus.SUCCEEDED
    saved = store.get_workflow(workflow.request_id)
    claims = _events(saved, WorkflowEventKind.CLAIM)
    assert len(claims) == 1, [event.model_dump() for event in saved.events]
    assert claims[0].actor == executor.config.executor_id
    assert claims[0].details["execution_epoch"] == saved.execution_epoch
    terminal = _events(saved, WorkflowEventKind.TERMINAL)
    assert len(terminal) == 1, [event.model_dump() for event in saved.events]
    assert terminal[0].actor == executor.config.executor_id
    assert terminal[0].status == WorkflowStatus.SUCCEEDED.value
    assert saved.events.index(claims[0]) < saved.events.index(terminal[0]), (
        "the claim is recorded before the ending"
    )


def test_a_failed_step_records_one_terminal_event_with_the_error() -> None:
    store = build_store()
    _, workflow = workflow_state(store, [FREEZE, RESET])
    adapter = FakeAdapter(
        {
            FREEZE: WorkflowStepOutcome.succeeded(),
            RESET: WorkflowStepOutcome.failed("reset refused"),
        }
    )
    executor = active_workflow_executor(store, [adapter], [FREEZE, RESET])

    result = execute_workflow(executor, workflow.request_id)

    assert result.status is WorkflowStatus.FAILED
    saved = store.get_workflow(workflow.request_id)
    terminal = _events(saved, WorkflowEventKind.TERMINAL)
    assert len(terminal) == 1, [event.model_dump() for event in saved.events]
    assert terminal[0].status == WorkflowStatus.FAILED.value
    assert terminal[0].reason == "reset refused"


def test_a_safe_supersession_records_a_terminal_event() -> None:
    store = build_store()
    incident, workflow = workflow_state(store, [FREEZE, RESET])
    _preempting_successor(store, incident, workflow)
    adapter = FakeAdapter(
        {
            FREEZE: WorkflowStepOutcome.succeeded(),
            RESET: WorkflowStepOutcome.succeeded(),
        }
    )
    executor = active_workflow_executor(store, [adapter], [FREEZE, RESET])

    result = execute_workflow(executor, workflow.request_id)

    assert result.status is WorkflowStatus.SUPERSEDED
    saved = store.get_workflow(workflow.request_id)
    terminal = _events(saved, WorkflowEventKind.TERMINAL)
    assert len(terminal) == 1, [event.model_dump() for event in saved.events]
    assert terminal[0].status == WorkflowStatus.SUPERSEDED.value
    assert terminal[0].details["preempted_by_workflow_id"] == "workflow-successor"


def test_terminalize_claimed_refuses_a_record_another_executor_holds() -> None:
    store = build_store()
    incident, workflow = workflow_state(store, [FREEZE])
    held = copy_model(
        workflow,
        status=WorkflowStatus.RUNNING,
        execution_owner_id="executor-elsewhere",
        execution_epoch=4,
        execution_lease_expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
    )
    store.save_workflow(held)
    executor = active_workflow_executor(store, [], [FREEZE])

    with pytest.raises(WorkflowLeaseError):
        executor.terminalize_claimed(
            held,
            incident,
            WorkflowStatus.FAILED,
            held.execution_epoch,
            reason="not mine",
            actor=WATCHDOG,
        )

    assert store.get_workflow(workflow.request_id).status is WorkflowStatus.RUNNING
