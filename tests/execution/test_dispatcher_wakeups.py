"""Store wakeups reach the dispatcher's wake event (性能 A).

``wake()`` was only ever called by the ingress routes, and in the split
deployment the ingress process is not the worker process that scans, so every
remediation step paid the 5 s poll: once for the PENDING row to be noticed and
once more for the remote command's result to be picked up. The store now
publishes two wakeup channels (``WakeupChannel``); the dispatcher listens to
both and translates them into ``wake()``. The listener is a hint path: a
payload means "scan now", the poll stays the fallback, and a remote-command
payload wakes only on the terminal statuses that let a WAITING step advance.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

import pytest

from gpu_fault.execution.config import WorkflowDispatcherConfig
from gpu_fault.execution.dispatcher import WorkflowDispatcher
from gpu_fault.models import IncidentState, WorkflowOperation, WorkflowStatus
from gpu_fault.regional import RemoteActionCommand
from gpu_fault.remote_command_models import RemoteCommandStatus
from gpu_fault.store.contracts import WakeupChannel
from tests._builders import (
    active_workflow_executor,
    build_store,
    fault_incident,
    workflow_request,
    workflow_step,
)

WAIT = 5.0
QUIET = 0.3


def _dispatcher(store, poll_interval_seconds: float = 60.0) -> WorkflowDispatcher:
    return WorkflowDispatcher(
        store,
        active_workflow_executor(store, [], frozenset()),
        WorkflowDispatcherConfig(
            enabled=True, poll_interval_seconds=poll_interval_seconds
        ),
    )


def _workflow_payload(**overrides: Any) -> dict[str, Any]:
    payload = {
        "request_id": "wf-1",
        "cluster_id": None,
        "status": "PENDING",
        "not_before": None,
    }
    payload.update(overrides)
    return payload


def _command_payload(status: str) -> dict[str, Any]:
    return {
        "command_id": "cmd-1",
        "cluster_id": "cluster-a",
        "workflow_request_id": "wf-1",
        "status": status,
    }


class _ScriptedListener:
    """A store's ``run_wakeup_listener`` that delivers a fixed script of
    payloads on the calling thread, then returns."""

    def __init__(
        self,
        payloads: list[dict[str, Any]],
        *,
        fail: bool = False,
        after_state: Callable[[bool], None] | None = None,
    ):
        self.payloads = payloads
        self.fail = fail
        self.calls: list[dict[str, Any]] = []
        self.states: list[bool] = []
        # Runs after each ``on_state`` report has been delivered, so a test
        # can look at what the dispatcher now publishes for the channel.
        self.after_state = after_state

    def __call__(
        self,
        channel: WakeupChannel,
        stop_event: threading.Event,
        on_notification,
        *,
        timeout_seconds: float = 1.0,
        on_state=None,
    ) -> None:
        self.calls.append(
            {
                "channel": channel,
                "stop_event": stop_event,
                "timeout_seconds": timeout_seconds,
            }
        )
        if on_state is not None:
            self._report(on_state, True)
        for payload in self.payloads:
            on_notification(payload)
        if self.fail:
            raise RuntimeError("listener connection lost for good")
        if on_state is not None:
            self._report(on_state, False)

    def _report(self, on_state, connected: bool) -> None:
        on_state(connected)
        self.states.append(connected)
        if self.after_state is not None:
            self.after_state(connected)


# ---------------------------------------------------------------------------
# Translation: which payloads become a wake.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "status", [RemoteCommandStatus.SUCCEEDED, RemoteCommandStatus.FAILED]
)
def test_a_terminal_remote_command_transition_wakes_the_dispatcher(status) -> None:
    listener = _ScriptedListener([_command_payload(status.value)])
    dispatcher = _dispatcher(build_store(run_wakeup_listener=listener))

    dispatcher.run_wakeups(WakeupChannel.REMOTE_COMMAND)

    assert dispatcher.consume_wake() is True
    assert dispatcher.wakeups_total == {"workflow_dispatch": 0, "remote_command": 1}
    assert dispatcher.wakeups_ignored_total == 0
    assert dispatcher.wakeup_last_seen_timestamp_seconds > 0


@pytest.mark.parametrize(
    "status",
    [
        RemoteCommandStatus.PENDING,
        RemoteCommandStatus.LEASED,
        RemoteCommandStatus.WAITING,
    ],
)
def test_the_executor_side_of_the_command_channel_does_not_wake(status) -> None:
    """PENDING was written by this dispatcher's own dispatch, LEASED and
    WAITING are the executor claiming and running it; a scan on any of them
    would find the same WAITING step still waiting."""

    listener = _ScriptedListener([_command_payload(status.value)])
    dispatcher = _dispatcher(build_store(run_wakeup_listener=listener))

    dispatcher.run_wakeups(WakeupChannel.REMOTE_COMMAND)

    assert dispatcher.consume_wake() is False
    assert dispatcher.wakeups_total == {"workflow_dispatch": 0, "remote_command": 0}
    assert dispatcher.wakeups_ignored_total == 1
    assert dispatcher.wakeup_last_seen_timestamp_seconds == 0.0


@pytest.mark.parametrize(
    "payload",
    [
        _workflow_payload(),
        _workflow_payload(status="RUNNING"),
        _workflow_payload(status="SAFETY_PENDING"),
        # A row deferred into the future still wakes: ``_eligible`` holds it
        # and the poll dispatches it later; the listener does not second-guess.
        _workflow_payload(not_before="2999-01-01T00:00:00Z"),
        # An unexpected shape is still a hint that something changed.
        {},
    ],
)
def test_every_workflow_dispatch_payload_wakes(payload) -> None:
    listener = _ScriptedListener([payload])
    dispatcher = _dispatcher(build_store(run_wakeup_listener=listener))

    dispatcher.run_wakeups(WakeupChannel.WORKFLOW_DISPATCH)

    assert dispatcher.consume_wake() is True
    assert dispatcher.wakeups_total == {"workflow_dispatch": 1, "remote_command": 0}


def test_a_burst_of_wakeups_coalesces_into_one_scan_request() -> None:
    """``wake()`` sets an Event, so ten payloads between two scans ask for one
    scan, not ten; the counter still records every one."""

    listener = _ScriptedListener(
        [_workflow_payload(request_id=f"wf-{i}") for i in range(10)]
    )
    dispatcher = _dispatcher(build_store(run_wakeup_listener=listener))

    dispatcher.run_wakeups(WakeupChannel.WORKFLOW_DISPATCH)

    assert dispatcher.consume_wake() is True
    assert dispatcher.consume_wake() is False
    assert dispatcher.wakeups_total["workflow_dispatch"] == 10


def test_a_disabled_dispatcher_never_opens_a_listener() -> None:
    listener = _ScriptedListener([_workflow_payload()])
    store = build_store(run_wakeup_listener=listener)
    dispatcher = WorkflowDispatcher(
        store,
        active_workflow_executor(store, [], frozenset()),
        WorkflowDispatcherConfig(enabled=False, poll_interval_seconds=60),
    )

    dispatcher.run_wakeups(WakeupChannel.WORKFLOW_DISPATCH)

    assert listener.calls == []
    assert dispatcher.consume_wake() is False


def test_a_store_without_the_listener_is_tolerated() -> None:
    """Contract tests pin every real store to ``run_wakeup_listener``; this is
    for the fakes and stubs that build a dispatcher without one."""

    dispatcher = _dispatcher(build_store())
    dispatcher.store = object()  # type: ignore[assignment]

    dispatcher.run_wakeups(WakeupChannel.WORKFLOW_DISPATCH)

    assert dispatcher.consume_wake() is False


# ---------------------------------------------------------------------------
# Lifecycle facts the listener thread body owns.
# ---------------------------------------------------------------------------


def test_the_listener_runs_on_the_given_stop_event_and_reports_its_state() -> None:
    """Each ``on_state`` report from the store listener lands, at once and
    for that channel only, in ``wakeup_listener_connected`` -- the gauge
    ``/metrics`` exports as ``..._wakeup_listener_connected``."""

    seen: list[dict[str, bool]] = []
    listener = _ScriptedListener(
        [],
        after_state=lambda _: seen.append(dict(dispatcher.wakeup_listener_connected)),
    )
    dispatcher = _dispatcher(build_store(run_wakeup_listener=listener))
    stop = threading.Event()

    dispatcher.run_wakeups(WakeupChannel.REMOTE_COMMAND, stop)

    (call,) = listener.calls
    assert call["channel"] is WakeupChannel.REMOTE_COMMAND
    assert call["stop_event"] is stop
    assert listener.states == [True, False]
    assert seen == [
        {"workflow_dispatch": False, "remote_command": True},
        {"workflow_dispatch": False, "remote_command": False},
    ]


def test_without_a_stop_event_the_listener_stops_with_the_dispatcher() -> None:
    listener = _ScriptedListener([])
    dispatcher = _dispatcher(build_store(run_wakeup_listener=listener))

    dispatcher.run_wakeups(WakeupChannel.WORKFLOW_DISPATCH)

    (call,) = listener.calls
    assert not call["stop_event"].is_set(), (
        "the listener stop event must be clear before the dispatcher stops"
    )
    dispatcher.stop()
    assert call["stop_event"].is_set(), (
        "the listener must wait on the event dispatcher.stop() sets"
    )


def test_a_listener_that_dies_is_logged_and_reported_disconnected(caplog) -> None:
    """Same contract as the processor's queue listener: the thread body never
    raises, because a dead listener costs a poll interval per step, not
    correctness, and the poll fallback is still turning."""

    listener = _ScriptedListener([_workflow_payload()], fail=True)
    dispatcher = _dispatcher(build_store(run_wakeup_listener=listener))

    with caplog.at_level("ERROR", logger="gpu_fault.execution.dispatcher"):
        dispatcher.run_wakeups(WakeupChannel.WORKFLOW_DISPATCH)

    assert dispatcher.wakeup_listener_connected["workflow_dispatch"] is False
    assert dispatcher.consume_wake() is True, "payloads before the failure still count"
    assert any(
        "wakeup listener failed" in record.getMessage()
        and "gpu_fault_workflow_dispatch" in record.getMessage()
        for record in caplog.records
    ), "a dead listener must log the failure with its channel name"


# ---------------------------------------------------------------------------
# End to end against the memory store's hub: a scan happens well inside a
# 60 s poll interval because the store woke the dispatcher.
# ---------------------------------------------------------------------------


def _pending_workflow(request_id: str):
    return workflow_request(
        request_id,
        f"incident-{request_id}",
        official_steps=[workflow_step(WorkflowOperation.RESTART_NODE)],
    )


def _remote_command(store, command_id: str, request_id: str) -> RemoteActionCommand:
    incident = fault_incident(
        f"incident-{request_id}",
        f"event-{request_id}",
        state=IncidentState.ACTION_PENDING,
        fencing_token=3,
    ).model_copy(update={"workflow_request_id": request_id})
    workflow = _pending_workflow(request_id).model_copy(
        update={"status": WorkflowStatus.RUNNING}
    )
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


class _Cycles:
    """Counts ``run_once`` cycles and lets a test wait for the next one."""

    def __init__(self, dispatcher: WorkflowDispatcher) -> None:
        self.count = 0
        self._changed = threading.Condition()
        original = dispatcher.run_once

        def run_once():
            with self._changed:
                self.count += 1
                self._changed.notify_all()
            return original()

        dispatcher.run_once = run_once  # type: ignore[method-assign]

    def wait_for(self, count: int, timeout: float = WAIT) -> bool:
        with self._changed:
            return self._changed.wait_for(lambda: self.count >= count, timeout)


@pytest.fixture
def running_dispatcher():
    """A real dispatcher on a 60 s poll, its two listener threads, and its
    ``run_forever`` thread, all stopped and joined at teardown."""

    from gpu_fault.app.lifespan_workers import start_dispatcher_wakeup_threads

    store = build_store()
    dispatcher = _dispatcher(store, poll_interval_seconds=60.0)
    cycles = _Cycles(dispatcher)
    stop = threading.Event()
    listeners = start_dispatcher_wakeup_threads(context_for(store, dispatcher), stop)
    worker = threading.Thread(
        target=dispatcher.run_forever, name="test-run-forever", daemon=True
    )
    worker.start()
    deadline = time.monotonic() + WAIT
    while (
        not all(dispatcher.wakeup_listener_connected.values())
        and time.monotonic() < deadline
    ):
        time.sleep(0.005)
    assert all(dispatcher.wakeup_listener_connected.values()), (
        "not every wakeup channel connected within WAIT"
    )
    # The first cycle runs at once on entry; everything after it is a wake.
    assert cycles.wait_for(1), "the entry cycle did not run"
    try:
        yield store, dispatcher, cycles
    finally:
        stop.set()
        dispatcher.stop()
        for thread in (*listeners, worker):
            thread.join(timeout=WAIT)
            assert not thread.is_alive(), f"{thread.name} did not stop"
        assert dispatcher.wakeup_listener_connected == {
            "workflow_dispatch": False,
            "remote_command": False,
        }


def context_for(store, dispatcher):
    from types import SimpleNamespace

    return SimpleNamespace(store=store, dispatcher=dispatcher)


def test_saving_a_pending_workflow_scans_without_waiting_for_the_poll(
    running_dispatcher,
) -> None:
    store, dispatcher, cycles = running_dispatcher
    before = cycles.count
    started = time.monotonic()

    store.save_workflow(_pending_workflow("wf-woken"))

    assert cycles.wait_for(before + 1), "no scan followed the PENDING write"
    assert time.monotonic() - started < WAIT
    assert dispatcher.wakeups_total["workflow_dispatch"] >= 1


def test_a_remote_command_result_scans_but_its_creation_does_not(
    running_dispatcher,
) -> None:
    store, dispatcher, cycles = running_dispatcher
    before = cycles.count

    store.ensure_remote_command(_remote_command(store, "cmd-woken", "wf-cmd"))
    (leased,) = store.claim_remote_commands(
        "cluster-a", "executor-a", limit=1, lease_seconds=60
    )
    # PENDING then LEASED: two payloads on the channel, neither a scan.
    assert not cycles.wait_for(before + 1, timeout=QUIET), (
        "creating or leasing a remote command must not trigger a scan"
    )
    assert dispatcher.wakeups_ignored_total >= 2

    assert store.cancel_remote_command("cmd-woken", reason="test") is False
    store.cancel_remote_commands_for_workflow("wf-cmd", reason="stop")
    assert not cycles.wait_for(before + 1, timeout=QUIET), (
        "a cancellation request on a LEASED command is not a status change"
    )

    store.ensure_remote_command(_remote_command(store, "cmd-failed", "wf-cmd-2"))
    assert store.cancel_remote_command("cmd-failed", reason="preempted") is True

    assert cycles.wait_for(before + 1), "FAILED did not wake the dispatcher"
    assert dispatcher.wakeups_total["remote_command"] >= 1
