"""The dispatcher cancels remote commands a terminal workflow left open.

Until 2026-09-08 this was ``gpu-fault-admin workflow-reconcile --mode
orphaned-commands``, used live once: a ``CHECK_MECHANICALS`` command still
``WAITING`` thirteen hours after its workflow had ``FAILED``, blocking the
release upgrade that refuses to start while any command is open. The predicate
-- an open command whose workflow is terminal -- is a pure Store read, so the
dispatcher now runs it on ``sweep_stuck_records`` with the proofs the manual
path had: the Store's own ``cancel_remote_commands_for_workflow``, the workflow
row left as it was, an ``OPERATOR_RECONCILED`` event with actor ``dispatcher``
written in the same transaction as the amend, nothing deleted, and never by age.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.execution.config import WorkflowDispatcherConfig
from gpu_fault.execution.dispatcher import WorkflowDispatcher
from gpu_fault.models import (
    IncidentState,
    WorkflowEventKind,
    WorkflowOperation,
    WorkflowStatus,
)
from gpu_fault.orphaned_commands import (
    AUDIT_ACTION,
    DISPATCHER_ACTOR,
    cancel_orphaned_commands,
)
from gpu_fault.regional import RemoteActionCommand
from gpu_fault.remote_command_models import RemoteCommandStatus
from tests._builders import (
    active_workflow_executor,
    build_store,
    fault_incident,
    workflow_request,
    workflow_step,
)

WORKFLOW = "workflow-f48baa91-63e4-432b-9648-1469d9e7eb39"
INCIDENT = "inc-kernel-log-kmsg-xid-54"
COMMAND = "remote-2cbdab99"
STEPS = [
    workflow_step(WorkflowOperation.FREEZE_EVIDENCE),
    workflow_step(WorkflowOperation.CHECK_MECHANICALS),
]


def orphan(
    store,
    *,
    request_id: str = WORKFLOW,
    incident_id: str = INCIDENT,
    command_id: str = COMMAND,
    workflow_status=WorkflowStatus.FAILED,
    command_status=RemoteCommandStatus.WAITING,
    **workflow_values,
):
    now = datetime.now(timezone.utc)
    incident = fault_incident(
        incident_id,
        f"event-{incident_id}",
        node_ids=["node-a"],
        state=IncidentState.ESCALATED,
    )
    workflow = workflow_request(
        request_id,
        incident_id,
        status=workflow_status,
        fencing_token=1,
        official_action="CHECK_MECHANICALS",
        official_steps=STEPS,
        **workflow_values,
    )
    store.save_incident(incident)
    store.save_workflow(workflow)
    leased = command_status is RemoteCommandStatus.LEASED
    store.ensure_remote_command(
        RemoteActionCommand(
            command_id=command_id,
            cluster_id="cluster-a",
            workflow_request_id=request_id,
            incident_id=incident_id,
            step_index=1,
            fencing_token=1,
            idempotency_key=f"{request_id}/1/CHECK_MECHANICALS",
            step=STEPS[1],
            workflow=workflow,
            incident=incident,
            status=command_status,
            lease_owner="regional-executor-a" if leased else None,
            lease_expires_at=now + timedelta(minutes=1) if leased else None,
        )
    )
    return incident, workflow


def dispatcher(store) -> WorkflowDispatcher:
    return WorkflowDispatcher(
        store,
        active_workflow_executor(store, [], frozenset()),
        WorkflowDispatcherConfig(enabled=True, batch_size=100),
    )


def reconcile_events(store, request_id: str = WORKFLOW):
    return [
        event
        for event in store.get_workflow(request_id).events
        if event.kind is WorkflowEventKind.OPERATOR_RECONCILED
    ]


def test_an_open_command_of_a_failed_workflow_is_cancelled_on_the_tick() -> None:
    store = build_store()
    incident, workflow = orphan(store)

    dispatcher(store).run_once()

    command = store.get_remote_command(COMMAND)
    assert command.status is RemoteCommandStatus.FAILED
    assert command.status_source == "workflow-timeout"
    assert command.lease_owner is None
    assert f"{DISPATCHER_ACTOR} reconciliation" in (command.error or "")
    after = store.get_workflow(WORKFLOW)
    assert after.status is WorkflowStatus.FAILED, "the workflow row is not moved"
    assert after.preemption_reason == workflow.preemption_reason
    assert store.get_incident(INCIDENT) == incident, "the incident is not written"


def test_the_cancel_is_recorded_on_the_workflow_as_the_dispatcher() -> None:
    store = build_store()
    orphan(store)

    dispatcher(store).run_once()

    (event,) = reconcile_events(store)
    assert event.actor == DISPATCHER_ACTOR
    assert event.details["action"] == AUDIT_ACTION
    assert event.details["previous_status"] == WorkflowStatus.FAILED.value
    assert event.details["new_status"] == WorkflowStatus.FAILED.value
    assert event.details["cancelled_remote_commands"] == {
        "cancelled": 1,
        "cancellation_requested": 0,
    }
    assert event.details["command_ids"] == [COMMAND]


@pytest.mark.parametrize(
    "workflow_status",
    [
        WorkflowStatus.FAILED,
        WorkflowStatus.SUCCEEDED,
        WorkflowStatus.BLOCKED,
        WorkflowStatus.SUPERSEDED,
    ],
)
def test_every_terminal_status_orphans_its_open_commands(
    workflow_status: WorkflowStatus,
) -> None:
    store = build_store()
    orphan(store, workflow_status=workflow_status)

    dispatcher(store).run_once()

    assert store.get_remote_command(COMMAND).status is RemoteCommandStatus.FAILED


@pytest.mark.parametrize(
    "workflow_status",
    [WorkflowStatus.PENDING, WorkflowStatus.RUNNING, WorkflowStatus.SAFETY_PENDING],
)
def test_a_live_workflow_keeps_its_commands_however_old(
    workflow_status: WorkflowStatus,
) -> None:
    """An open command of a live workflow is work only its executor may settle."""

    store = build_store()
    stale = datetime.now(timezone.utc) - timedelta(days=2)
    orphan(
        store,
        workflow_status=workflow_status,
        execution_owner_id="executor-b",
        execution_lease_expires_at=stale,
        created_at=stale,
        updated_at=stale,
    )

    cancel_orphaned_commands(store, now=datetime.now(timezone.utc))

    assert store.get_remote_command(COMMAND).status is RemoteCommandStatus.WAITING
    assert reconcile_events(store) == []


def test_a_terminal_workflow_without_open_commands_gets_no_event() -> None:
    store = build_store()
    orphan(store, command_status=RemoteCommandStatus.SUCCEEDED)

    dispatcher(store).run_once()

    assert reconcile_events(store) == []


def test_a_leased_command_is_asked_to_cancel_once_not_every_tick() -> None:
    """The agent settles a LEASED command; the request is recorded a single time."""

    store = build_store()
    orphan(store, command_status=RemoteCommandStatus.LEASED)
    sweep = dispatcher(store)

    sweep.run_once()
    command = store.get_remote_command(COMMAND)
    assert command.cancellation_requested_at is not None
    (event,) = reconcile_events(store)
    assert event.details["cancelled_remote_commands"] == {
        "cancelled": 0,
        "cancellation_requested": 1,
    }

    sweep.run_once()

    assert len(reconcile_events(store)) == 1, (
        "a command already asked to cancel must not grow the audit every tick"
    )


def test_a_second_tick_after_the_cancel_changes_nothing() -> None:
    store = build_store()
    orphan(store)
    sweep = dispatcher(store)

    sweep.run_once()
    first = store.get_workflow(WORKFLOW)
    sweep.run_once()

    assert store.get_workflow(WORKFLOW) == first
    assert len(reconcile_events(store)) == 1


def test_a_failing_cancel_on_one_workflow_does_not_stop_the_others() -> None:
    store = build_store()
    orphan(store)
    orphan(
        store,
        request_id="workflow-second",
        incident_id="inc-second",
        command_id="remote-second",
    )
    genuine_cancel = store.cancel_remote_commands_for_workflow

    def failing_cancel(workflow_request_id, *, reason):
        if workflow_request_id == WORKFLOW:
            raise RuntimeError("store hiccup")
        return genuine_cancel(workflow_request_id, reason=reason)

    store.cancel_remote_commands_for_workflow = failing_cancel  # type: ignore[method-assign]

    cancelled = cancel_orphaned_commands(store, now=datetime.now(timezone.utc))

    assert sorted(cancelled) == ["workflow-second"]
    assert store.get_remote_command(COMMAND).status is RemoteCommandStatus.WAITING, (
        "the failed workflow is left for the next tick"
    )
    assert store.get_remote_command("remote-second").status is (
        RemoteCommandStatus.FAILED
    )
