"""The two wakeup channels wake a poller before its next interval.

The workflow dispatcher polled every 5 s and the data-plane executor every
2 s, so every remediation step paid 5-10 s of pure waiting. The processor queue
already woke its consumer with ``pg_notify``; ``WakeupChannel.WORKFLOW_DISPATCH``
and ``WakeupChannel.REMOTE_COMMAND`` are the same design for the two remaining
pollers. On PostgreSQL a row trigger on ``gpu_fault_objects`` publishes; the
memory and SQLite stores publish from the write path into an in-process hub
under the same condition, so a consumer written against
``run_wakeup_listener`` behaves the same on every backend. The PostgreSQL
trigger itself is exercised by ``test_postgres_wakeup_triggers.py``.
"""

from __future__ import annotations

import inspect
import re
import threading
import time
from pathlib import Path
from queue import Empty, Queue
from typing import get_type_hints

import pytest

from gpu_fault.models import (
    EXECUTABLE_WORKFLOW_STATUSES,
    IncidentState,
    WorkflowOperation,
    WorkflowStatus,
)
from gpu_fault.regional import RemoteActionCommand
from gpu_fault.remote_command_models import RemoteCommandStatus
from gpu_fault.store import InMemoryStore, PostgresStore, SqliteStore
from gpu_fault.store.contracts import WakeupChannel, WakeupStore
from gpu_fault.store.postgres import ddl_wakeups
from gpu_fault.store.postgres.wakeups import PostgresWakeupMixin
from gpu_fault.store.shared.wakeups import WORKFLOW_WAKEUP_FIELDS
from tests._builders import fault_incident, workflow_request, workflow_step

WAIT = 3.0
QUIET = 0.3


# ---------------------------------------------------------------------------
# Contract: one signature, every backend, both channel names.
# ---------------------------------------------------------------------------


def _shape(function):
    hints = get_type_hints(function)
    return [
        (parameter.name, parameter.kind, parameter.default, hints.get(parameter.name))
        for parameter in inspect.signature(function).parameters.values()
    ], hints.get("return")


def test_every_store_implements_the_wakeup_listener_with_one_signature() -> None:
    protocol = _shape(WakeupStore.run_wakeup_listener)
    for store in (InMemoryStore, SqliteStore, PostgresStore):
        assert issubclass(store, WakeupStore), store.__name__
        assert _shape(store.run_wakeup_listener) == protocol, store.__name__


def test_the_channel_names_are_the_postgres_channel_names() -> None:
    assert WakeupChannel.WORKFLOW_DISPATCH == "gpu_fault_workflow_dispatch"
    assert WakeupChannel.REMOTE_COMMAND == "gpu_fault_remote_command"
    assert str(WakeupChannel.REMOTE_COMMAND) == "gpu_fault_remote_command"


# ---------------------------------------------------------------------------
# The trigger and the in-process hub fire on the same condition.
# ---------------------------------------------------------------------------


def _trigger_source() -> str:
    return Path(ddl_wakeups.__file__).read_text(encoding="utf-8")


def test_the_trigger_notifies_on_the_dispatcher_executable_statuses() -> None:
    match = re.search(
        r"NEW\.payload->>'status', ''\)\s+IN \(([^)]*)\)", _trigger_source()
    )
    assert match is not None, "the workflow status list is missing from the trigger"
    literal = {item.strip().strip("'") for item in match.group(1).split(",")}
    assert literal == {status.value for status in EXECUTABLE_WORKFLOW_STATUSES}


def test_the_trigger_compares_the_same_workflow_fields_as_the_hub() -> None:
    source = _trigger_source()
    compared = set(re.findall(r"OLD\.payload->>'(\w+)'\s+IS NOT DISTINCT FROM", source))
    assert compared == {*WORKFLOW_WAKEUP_FIELDS, "status"}
    for channel in WakeupChannel:
        assert re.search(rf"pg_notify\(\s*'{channel.value}'", source), channel


# ---------------------------------------------------------------------------
# Memory and SQLite: the hub publishes from every store write path.
# ---------------------------------------------------------------------------


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    if request.param == "memory":
        instance = InMemoryStore()
    else:
        instance = SqliteStore(str(tmp_path / "wakeups.db"))
    try:
        yield instance
    finally:
        if request.param == "sqlite":
            instance.close()


class _Listener:
    """A listener thread with a queue of the payloads it received."""

    def __init__(self, store, channel: WakeupChannel) -> None:
        self.stop = threading.Event()
        self.payloads: Queue = Queue()
        self.states: list[bool] = []
        self.thread = threading.Thread(
            target=store.run_wakeup_listener,
            args=(channel, self.stop, self.payloads.put),
            kwargs={"timeout_seconds": 0.05, "on_state": self.states.append},
            daemon=True,
        )
        self.thread.start()
        deadline = time.monotonic() + WAIT
        while not self.states and time.monotonic() < deadline:
            time.sleep(0.005)
        assert self.states == [True], "the listener never reported connected"

    def next(self, timeout: float = WAIT) -> dict:
        return self.payloads.get(timeout=timeout)

    def assert_quiet(self) -> None:
        with pytest.raises(Empty):
            self.payloads.get(timeout=QUIET)

    def close(self) -> None:
        self.stop.set()
        self.thread.join(timeout=WAIT)
        assert not self.thread.is_alive(), "the listener did not stop"


@pytest.fixture
def workflow_listener(store):
    listener = _Listener(store, WakeupChannel.WORKFLOW_DISPATCH)
    try:
        yield listener
    finally:
        listener.close()


@pytest.fixture
def command_listener(store):
    listener = _Listener(store, WakeupChannel.REMOTE_COMMAND)
    try:
        yield listener
    finally:
        listener.close()


def _workflow(request_id: str, status=WorkflowStatus.PENDING, **values):
    return workflow_request(
        request_id,
        f"incident-{request_id}",
        status=status,
        official_steps=[workflow_step(WorkflowOperation.RESTART_NODE)],
        **values,
    )


def _command(store, command_id: str, request_id: str) -> RemoteActionCommand:
    incident = fault_incident(
        f"incident-{request_id}",
        f"event-{request_id}",
        state=IncidentState.ACTION_PENDING,
        fencing_token=3,
    )
    workflow = _workflow(request_id)
    incident = incident.model_copy(update={"workflow_request_id": request_id})
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


def test_a_workflow_saved_pending_wakes_the_dispatcher_channel(
    store, workflow_listener
) -> None:
    store.save_workflow(_workflow("wf-pending"))

    assert workflow_listener.next() == {
        "request_id": "wf-pending",
        # WorkflowRequest carries no cluster_id; published null for symmetry.
        "cluster_id": None,
        "status": "PENDING",
        "not_before": None,
    }


def test_a_workflow_saved_succeeded_does_not_wake_anyone(
    store, workflow_listener
) -> None:
    store.save_workflow(_workflow("wf-done", WorkflowStatus.SUCCEEDED))

    workflow_listener.assert_quiet()


def test_a_terminal_transition_of_an_executable_row_is_quiet(
    store, workflow_listener
) -> None:
    running = _workflow("wf-finish", WorkflowStatus.RUNNING)
    store.save_workflow(running)
    workflow_listener.next()

    store.save_workflow(running.model_copy(update={"status": WorkflowStatus.FAILED}))

    workflow_listener.assert_quiet()


def test_lease_bookkeeping_on_a_running_row_is_quiet_but_a_merge_wakes(
    store, workflow_listener
) -> None:
    """The executor writes a WAITING row back every dispatch (D-7) and renews
    its lease; each write is an UPDATE of an executable row. Waking the
    dispatcher on those would re-dispatch the row it just wrote, in a loop,
    until the remote command completed. Only the fields the scan orders on
    re-fire the wakeup."""

    running = _workflow("wf-running", WorkflowStatus.RUNNING)
    store.save_workflow(running)
    workflow_listener.next()

    claimed = store.claim_workflow("wf-running", "executor-a", 3)
    assert workflow_listener.next()["request_id"] == "wf-running", (
        "a lease handover (execution_owner_id) must wake the other replicas"
    )
    store.renew_workflow_lease("wf-running", "executor-a", claimed.execution_epoch)
    store.save_workflow_if_leased(
        claimed.model_copy(update={"pending_failure_error": "step evidence"}),
        "executor-a",
        claimed.execution_epoch,
    )
    workflow_listener.assert_quiet()

    merged = store.get_workflow("wf-running")
    store.save_workflow(
        merged.model_copy(update={"merge_revision": merged.merge_revision + 1}),
        expected=merged,
    )
    assert workflow_listener.next()["status"] == "RUNNING"


def test_create_incident_workflow_if_absent_wakes_the_dispatcher(
    store, workflow_listener
) -> None:
    incident = fault_incident(
        "incident-wf-created", "event-created", state=IncidentState.ACTION_PENDING
    ).model_copy(update={"workflow_request_id": "wf-created"})
    workflow = _workflow("wf-created")

    _, _, created = store.create_incident_workflow_if_absent(
        "event-created", lambda: (incident, workflow)
    )

    assert created is True
    assert workflow_listener.next()["request_id"] == "wf-created"


def test_a_remote_command_ensured_pending_wakes_the_command_channel(
    store, command_listener
) -> None:
    store.ensure_remote_command(_command(store, "cmd-1", "wf-cmd-1"))

    assert command_listener.next() == {
        "command_id": "cmd-1",
        "cluster_id": "cluster-a",
        "workflow_request_id": "wf-cmd-1",
        "status": "PENDING",
    }
    # Idempotent re-ensure writes nothing and wakes nobody.
    store.ensure_remote_command(_command(store, "cmd-1", "wf-cmd-1"))
    command_listener.assert_quiet()


def test_every_remote_command_status_transition_wakes_but_a_renewal_does_not(
    store, command_listener
) -> None:
    store.ensure_remote_command(_command(store, "cmd-2", "wf-cmd-2"))
    command_listener.next()

    (leased,) = store.claim_remote_commands(
        "cluster-a", "executor-a", limit=1, lease_seconds=60
    )
    assert command_listener.next()["status"] == RemoteCommandStatus.LEASED.value

    store.renew_remote_command_lease(
        "cluster-a", "cmd-2", "executor-a", leased.lease_token, lease_seconds=60
    )
    command_listener.assert_quiet()

    store.cancel_remote_command("cmd-2", reason="test")  # LEASED: not cancellable
    command_listener.assert_quiet()
    store.cancel_remote_commands_for_workflow("wf-cmd-2", reason="stop")
    command_listener.assert_quiet()  # LEASED only gets a cancellation request

    store.ensure_remote_command(_command(store, "cmd-3", "wf-cmd-3"))
    command_listener.next()
    assert store.cancel_remote_command("cmd-3", reason="preempted") is True
    assert command_listener.next() == {
        "command_id": "cmd-3",
        "cluster_id": "cluster-a",
        "workflow_request_id": "wf-cmd-3",
        "status": "FAILED",
    }


def test_the_channels_are_independent(store, workflow_listener, command_listener):
    store.save_workflow(_workflow("wf-only"))

    assert workflow_listener.next()["request_id"] == "wf-only"
    command_listener.assert_quiet()


def test_the_stop_event_ends_the_listener_within_its_timeout(store) -> None:
    stop = threading.Event()
    states: list[bool] = []
    thread = threading.Thread(
        target=store.run_wakeup_listener,
        args=(WakeupChannel.WORKFLOW_DISPATCH, stop, lambda _payload: None),
        kwargs={"timeout_seconds": 0.2, "on_state": states.append},
    )
    thread.start()
    time.sleep(0.05)
    started = time.monotonic()
    stop.set()
    thread.join(timeout=WAIT)

    assert not thread.is_alive(), "the listener did not exit after the stop event"
    assert time.monotonic() - started < 1.0
    assert states == [True, False]


def test_a_listener_started_after_the_write_sees_nothing(store) -> None:
    """Hints, not state: a wakeup published with no listener is gone, which is
    exactly why every consumer keeps its poll."""

    store.save_workflow(_workflow("wf-early"))
    listener = _Listener(store, WakeupChannel.WORKFLOW_DISPATCH)
    try:
        listener.assert_quiet()
    finally:
        listener.close()


def test_a_rejected_timeout_is_a_programming_error(store) -> None:
    with pytest.raises(ValueError):
        store.run_wakeup_listener(
            WakeupChannel.WORKFLOW_DISPATCH,
            threading.Event(),
            lambda _payload: None,
            timeout_seconds=0,
        )


# ---------------------------------------------------------------------------
# PostgreSQL listener loop, without a server: payload decode, reader
# rejection after a failover, reconnect, stop.
# ---------------------------------------------------------------------------


class _Notification:
    def __init__(self, payload: str) -> None:
        self.payload = payload


class _Result:
    def __init__(self, row):
        self._row = row

    def fetchone(self):
        return self._row


class FakeListenConnection:
    def __init__(self, *, in_recovery: bool = False, payloads=()) -> None:
        self.in_recovery = in_recovery
        self.pending = list(payloads)
        self.statements: list[str] = []
        self.closed = False
        self.autocommit = True

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, sql: str, params=None) -> _Result:
        flat = " ".join(sql.split())
        self.statements.append(flat)
        if "pg_is_in_recovery()" in flat:
            return _Result((self.in_recovery, "on" if self.in_recovery else "off"))
        return _Result((True,))

    def notifies(self, *, timeout: float, stop_after: int):
        if self.pending:
            batch, self.pending = self.pending[:stop_after], self.pending[stop_after:]
            return iter(_Notification(item) for item in batch)
        time.sleep(timeout)
        return iter(())

    def close(self) -> None:
        self.closed = True


class _FakeUrlStore(PostgresWakeupMixin):
    # ``PostgresStore.url`` is a property; the mixin declares it read-only.
    url = "postgresql://fake"


def _listen(connections, monkeypatch, channel=WakeupChannel.REMOTE_COMMAND):
    import psycopg

    handed_out: list[FakeListenConnection] = []

    def connect(*_args, **_kwargs):
        connection = connections[min(len(handed_out), len(connections) - 1)]
        handed_out.append(connection)
        return connection

    monkeypatch.setattr(psycopg, "connect", connect)
    store = _FakeUrlStore()
    stop = threading.Event()
    states: list[bool] = []
    payloads: Queue = Queue()
    thread = threading.Thread(
        target=store.run_wakeup_listener,
        args=(channel, stop, payloads.put),
        kwargs={"timeout_seconds": 0.01, "on_state": states.append},
    )
    thread.start()
    return stop, thread, states, payloads, handed_out


def test_the_postgres_listener_decodes_payloads_and_dedups_a_batch(monkeypatch):
    connection = FakeListenConnection(
        payloads=[
            '{"command_id": "cmd-1", "cluster_id": "c", '
            '"workflow_request_id": "wf", "status": "PENDING"}',
            '{"command_id": "cmd-1", "cluster_id": "c", '
            '"workflow_request_id": "wf", "status": "PENDING"}',
            "not json",
            '{"command_id": "cmd-2", "cluster_id": "c", '
            '"workflow_request_id": "wf", "status": "LEASED"}',
        ]
    )
    stop, thread, states, payloads, _ = _listen([connection], monkeypatch)
    try:
        first = payloads.get(timeout=WAIT)
        second = payloads.get(timeout=WAIT)
    finally:
        stop.set()
        thread.join(timeout=WAIT)

    assert first["command_id"] == "cmd-1" and second["command_id"] == "cmd-2"
    assert payloads.empty(), "the duplicate or the malformed payload was forwarded"
    assert states == [True, False]
    assert connection.statements[0] == "LISTEN gpu_fault_remote_command"


def test_a_postgres_listener_on_a_demoted_writer_reconnects(monkeypatch) -> None:
    demoted = FakeListenConnection(in_recovery=True)
    healthy = FakeListenConnection()
    monkeypatch.setattr(PostgresWakeupMixin, "_WAKEUP_WRITER_CHECK_SECONDS", 0.05)
    stop, thread, states, _payloads, handed_out = _listen(
        [demoted, healthy], monkeypatch, channel=WakeupChannel.WORKFLOW_DISPATCH
    )
    try:
        deadline = time.monotonic() + WAIT
        while time.monotonic() < deadline and len(handed_out) < 2:
            time.sleep(0.01)
    finally:
        stop.set()
        thread.join(timeout=WAIT)

    assert demoted.closed is True, "the demoted connection was kept"
    assert any("pg_is_in_recovery()" in sql for sql in demoted.statements), (
        "the listener never probed the connection for a demoted writer"
    )
    assert len(handed_out) >= 2, "the loop did not reconnect after the reader probe"
    assert states[:2] == [True, False]
    assert handed_out[0].statements[0] == "LISTEN gpu_fault_workflow_dispatch"
