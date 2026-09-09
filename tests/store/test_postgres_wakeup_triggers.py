"""The ``gpu_fault_objects`` wakeup trigger fires on the same condition the
in-process hub does, and ``run_wakeup_listener`` delivers it.

Schema v14. One row trigger on ``gpu_fault_objects`` publishes
``gpu_fault_workflow_dispatch`` for a workflow row that became executable (or
whose scan-ordering fields changed) and ``gpu_fault_remote_command`` for every
remote-command status transition. Lease renewals and step-evidence writes stay
quiet, otherwise the dispatcher would re-dispatch the WAITING row it had just
written, in a loop, until the remote command completed. The memory / SQLite
side of the same contract is ``tests/store/test_wakeup_listener.py``.
"""

from __future__ import annotations

import json
import os
import threading
import time
from queue import Empty, Queue

import pytest

from gpu_fault.models import IncidentState, WorkflowOperation, WorkflowStatus
from gpu_fault.regional import RemoteActionCommand
from gpu_fault.store import PostgresStore
from gpu_fault.store.contracts import WakeupChannel
from tests._builders import fault_incident, workflow_request, workflow_step
from tests.store._postgres_processor_claim_support import _truncate

POSTGRES_URL = os.getenv("GPU_FAULT_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(
    not POSTGRES_URL, reason="GPU_FAULT_TEST_POSTGRES_URL is not configured"
)

WAIT = 3.0
QUIET = 0.5


@pytest.fixture
def store():
    assert POSTGRES_URL is not None
    instance = PostgresStore(POSTGRES_URL)
    _truncate()
    try:
        yield instance
    finally:
        instance.close()
        _truncate()


def _listener(channel: WakeupChannel):
    import psycopg

    connection = psycopg.connect(POSTGRES_URL, autocommit=True)
    connection.execute(f"LISTEN {channel.value}")
    return connection


def _notifications(listener, timeout: float = WAIT) -> list[dict]:
    return [
        json.loads(notification.payload)
        for notification in listener.notifies(timeout=timeout, stop_after=16)
    ]


def _workflow(request_id: str, status=WorkflowStatus.PENDING, **values):
    return workflow_request(
        request_id,
        f"incident-{request_id}",
        status=status,
        official_steps=[workflow_step(WorkflowOperation.RESTART_NODE)],
        **values,
    )


def _command(command_id: str, request_id: str) -> RemoteActionCommand:
    incident = fault_incident(
        f"incident-{request_id}",
        f"event-{request_id}",
        state=IncidentState.ACTION_PENDING,
        fencing_token=3,
    ).model_copy(update={"workflow_request_id": request_id})
    workflow = _workflow(request_id)
    return RemoteActionCommand(
        command_id=command_id,
        cluster_id="cluster-a",
        workflow_request_id=request_id,
        incident_id=incident.incident_id,
        step_index=0,
        fencing_token=3,
        idempotency_key=f"{request_id}/0/RESTART_NODE",
        step=workflow.official_steps[0],
        workflow=workflow,
        incident=incident,
    )


def test_a_pending_workflow_notifies_and_a_terminal_one_does_not(store) -> None:
    listener = _listener(WakeupChannel.WORKFLOW_DISPATCH)
    try:
        store.save_workflow(_workflow("wf-pending"))
        pending = _notifications(listener)
        store.save_workflow(_workflow("wf-done", WorkflowStatus.SUCCEEDED))
        terminal = _notifications(listener, timeout=QUIET)
    finally:
        listener.close()

    assert pending == [
        {
            "request_id": "wf-pending",
            # WorkflowRequest carries no cluster_id; published null for symmetry.
            "cluster_id": None,
            "status": "PENDING",
            "not_before": None,
        }
    ]
    assert terminal == []


def test_lease_bookkeeping_is_quiet_but_a_handover_and_a_merge_notify(store):
    listener = _listener(WakeupChannel.WORKFLOW_DISPATCH)
    try:
        store.save_workflow(_workflow("wf-running", WorkflowStatus.RUNNING))
        _notifications(listener)
        claimed = store.claim_workflow("wf-running", "executor-a", 3)
        handover = _notifications(listener)
        store.renew_workflow_lease("wf-running", "executor-a", claimed.execution_epoch)
        store.save_workflow_if_leased(
            claimed.model_copy(update={"pending_failure_error": "step evidence"}),
            "executor-a",
            claimed.execution_epoch,
        )
        bookkeeping = _notifications(listener, timeout=QUIET)
        merged = store.get_workflow("wf-running")
        store.save_workflow(
            merged.model_copy(update={"merge_revision": merged.merge_revision + 1}),
            expected=merged,
        )
        merge = _notifications(listener)
    finally:
        listener.close()

    assert [item["request_id"] for item in handover] == ["wf-running"]
    assert bookkeeping == []
    assert [item["status"] for item in merge] == ["RUNNING"]


def test_every_remote_command_status_transition_notifies(store) -> None:
    listener = _listener(WakeupChannel.REMOTE_COMMAND)
    try:
        store.ensure_remote_command(_command("cmd-1", "wf-cmd-1"))
        created = _notifications(listener)
        store.ensure_remote_command(_command("cmd-1", "wf-cmd-1"))
        repeated = _notifications(listener, timeout=QUIET)
        (leased,) = store.claim_remote_commands(
            "cluster-a", "executor-a", limit=1, lease_seconds=60
        )
        claimed = _notifications(listener)
        store.renew_remote_command_lease(
            "cluster-a", "cmd-1", "executor-a", leased.lease_token, lease_seconds=60
        )
        renewed = _notifications(listener, timeout=QUIET)
        store.cancel_remote_commands_for_workflow("wf-cmd-1", reason="stop")
        requested = _notifications(listener, timeout=QUIET)
    finally:
        listener.close()

    assert created == [
        {
            "command_id": "cmd-1",
            "cluster_id": "cluster-a",
            "workflow_request_id": "wf-cmd-1",
            "status": "PENDING",
        }
    ]
    assert repeated == []
    assert [item["status"] for item in claimed] == ["LEASED"]
    assert renewed == [], "a lease renewal is not a status transition"
    assert requested == [], "a cancellation request leaves the row LEASED"


def test_run_wakeup_listener_delivers_the_trigger_payload(store) -> None:
    stop = threading.Event()
    payloads: Queue = Queue()
    states: list[bool] = []
    thread = threading.Thread(
        target=store.run_wakeup_listener,
        args=(WakeupChannel.REMOTE_COMMAND, stop, payloads.put),
        kwargs={"timeout_seconds": 0.05, "on_state": states.append},
        daemon=True,
    )
    thread.start()
    try:
        deadline = time.monotonic() + WAIT
        while not states and time.monotonic() < deadline:
            time.sleep(0.01)
        assert states == [True]
        # LISTEN is registered before ``on_state(True)``; a write after that
        # point is seen.
        store.ensure_remote_command(_command("cmd-live", "wf-live"))
        payload = payloads.get(timeout=WAIT)
        with pytest.raises(Empty):
            payloads.get(timeout=QUIET)
    finally:
        stop.set()
        thread.join(timeout=WAIT)

    assert not thread.is_alive(), "the listener did not exit after the stop event"
    assert payload == {
        "command_id": "cmd-live",
        "cluster_id": "cluster-a",
        "workflow_request_id": "wf-live",
        "status": "PENDING",
    }
    assert states == [True, False]


def test_a_missing_wakeup_trigger_fails_closed(store) -> None:
    import psycopg

    assert POSTGRES_URL is not None
    with psycopg.connect(POSTGRES_URL, autocommit=True) as connection:
        connection.execute(
            "DROP TRIGGER IF EXISTS gpu_fault_objects_notify_wakeup_trigger "
            "ON gpu_fault_objects"
        )
    try:
        with pytest.raises(RuntimeError, match="wakeup trigger is missing"):
            PostgresStore(POSTGRES_URL, initialize_schema=False)
    finally:
        PostgresStore(POSTGRES_URL, initialize_schema=True).close()
    PostgresStore(POSTGRES_URL, initialize_schema=False).close()
