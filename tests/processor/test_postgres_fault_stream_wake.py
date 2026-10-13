"""A fault row queued behind a PENDING one is claimed without a notification.

Local PostgreSQL, two coordinators that each own a notification shard. The
queue trigger announces only idle->busy (path, priority) edges, so a priority-0
row admitted while another priority-0 row of its path is PENDING -- here one
deferred for an hour -- raises no NOTIFY. Every control-worker of a perf run
had backed its fault stream off to the 5 s notification fallback, so such rows
waited in 5 s steps; while a fault-tier row is PENDING the fault stream keeps
its 0.5 s idle ceiling and the row is claimed within it.

With no fault-tier row PENDING or LEASED the settled fault stream waits on the
edge instead (the fallback): it polls an idle queue about once per 5 s, a row
admitted to the empty queue is claimed as soon as its NOTIFY arrives, and the
next deferred row puts the stream back on the 0.5 s cadence.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from threading import Lock, Thread
from typing import Any

import pytest

from gpu_fault.processor import (
    ProcessorCoordinator,
    ProcessorLeaseSettings,
    ProcessorPoolSettings,
)
from tests._builders import processor_request
from tests.store._postgres_processor_claim_support import (
    _truncate,
    postgres_store_instance,
)

POSTGRES_URL = os.getenv("GPU_FAULT_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(
    not POSTGRES_URL, reason="GPU_FAULT_TEST_POSTGRES_URL is not configured"
)

# The production fault idle ceiling (0.5 s) plus scheduling slack. Measured to
# the start of the claim that returned the row: the bound is about when the loop
# looks, not how long the claim query takes on a loaded gate host.
CLAIM_BOUND_SECONDS = 0.6
# Commit, LISTEN delivery and the claim loop's wake: far below the 5 s fallback
# an idle fault stream sleeps on, with room for a loaded gate host.
NOTIFIED_CLAIM_BOUND_SECONDS = 1.0
# Production cadence: eight empty rounds (about 3 s) settle the stream, whose
# interval then doubles 1, 2, 4 s out to the 5 s fallback.
SETTLE_SECONDS = 11.0
IDLE_WINDOW_SECONDS = 5.0


@pytest.fixture
def store():
    yield from postgres_store_instance()
    _truncate()


def _connect():
    import psycopg

    assert POSTGRES_URL is not None
    return psycopg.connect(POSTGRES_URL, autocommit=True)


def _terminal(attempt_id: str):
    return processor_request(
        "/v1/attempts/terminal", body=json.dumps({"attempt_id": attempt_id}).encode()
    )


class _Response:
    status = 200
    headers = {"Content-Type": "application/json"}

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def read(self) -> bytes:
        return b"{}"


class _ClaimClock:
    """The shared store, recording when the claim that took a request started."""

    def __init__(self, store: Any) -> None:
        self._store = store
        self._lock = Lock()
        self.claimed_at: dict[str, float] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._store, name)

    def claim_active_processor_requests(self, *args: Any, **kwargs: Any) -> Any:
        started = time.monotonic()
        rows = self._store.claim_active_processor_requests(*args, **kwargs)
        with self._lock:
            for row in rows:
                self.claimed_at.setdefault(row.request_id, started)
        return rows


def _wait_until(predicate: Callable[[], bool], timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def _coordinator(store: Any, owner_id: str) -> ProcessorCoordinator:
    return ProcessorCoordinator(
        store,
        owner_id=owner_id,
        internal_token="token-" + "x" * 32,
        local_url="http://127.0.0.1:1",
        pools=ProcessorPoolSettings(
            fault_worker_count=2,
            observation_worker_count=1,
            gpu_telemetry_worker_count=1,
            host_telemetry_worker_count=1,
        ),
        lease=ProcessorLeaseSettings(),  # production ceilings and fallback
        active_consumers=True,
    )


def test_a_fault_row_behind_a_pending_one_is_claimed_without_a_notification(
    store, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "gpu_fault.processor.coordinator.urlopen",
        lambda request, **_kwargs: _Response(),
    )
    deferred = _terminal("attempt-deferred")
    store.enqueue_processor_request(deferred)
    with _connect() as connection:
        connection.execute(
            "UPDATE gpu_fault_processor_queue "
            "SET not_before = now() + interval '1 hour' WHERE request_id = %s",
            (deferred.request_id,),
        )
    clock = _ClaimClock(store)
    coordinators = [_coordinator(clock, owner) for owner in ("pod-a:1", "pod-b:1")]
    threads = [
        Thread(target=getattr(coordinator, target), daemon=True)
        for coordinator in coordinators
        for target in ("run_queue_notifications", "run_processor")
    ]
    delays: list[float] = []
    with _connect() as listener:
        listener.execute("LISTEN gpu_fault_processor_queue")
        for thread in threads:
            thread.start()
        try:
            assert _wait_until(
                lambda: all(
                    coordinator.metrics_snapshot()["notifications_enabled"] == 1
                    for coordinator in coordinators
                ),
                15,
            ), "both coordinators must own a notification shard"
            time.sleep(3.0)  # every empty stream backs off to its idle ceiling
            received = [
                coordinator.metrics_snapshot()["notifications_received_total"]
                for coordinator in coordinators
            ]
            for index, gap in enumerate((0.0, 1.3, 1.7)):
                time.sleep(gap)  # land on different phases of the backoff
                item = _terminal(f"attempt-x{index}")
                store.enqueue_processor_request(item)
                enqueued = time.monotonic()
                assert _wait_until(lambda: item.request_id in clock.claimed_at, 10), (
                    f"{item.request_id} was never claimed"
                )
                delays.append(clock.claimed_at[item.request_id] - enqueued)
            notifications = [
                json.loads(notification.payload)
                for notification in listener.notifies(timeout=0.5, stop_after=64)
            ]
            assert [
                coordinator.metrics_snapshot()["notifications_received_total"]
                for coordinator in coordinators
            ] == received, "no coordinator was woken by a notification"
        finally:
            for coordinator in coordinators:
                coordinator.stop()
            for thread in threads:
                thread.join(timeout=10)
    assert not any(thread.is_alive() for thread in threads), "a loop did not stop"
    assert notifications == [], "rows behind a PENDING one raise no edge"
    assert max(delays) < CLAIM_BOUND_SECONDS, (
        f"fault rows waited {[round(delay, 3) for delay in delays]} s to be claimed"
    )


def _fault_rounds(coordinator: ProcessorCoordinator) -> int:
    return int(
        coordinator.metrics_snapshot()["claim"]["rounds_by_stream"].get("fault", 0)
    )


def _notifications_received(coordinator: ProcessorCoordinator) -> int:
    return int(coordinator.metrics_snapshot()["notifications_received_total"])


def test_an_idle_fault_stream_waits_on_the_edge_and_polls_behind_a_pending_row(
    store, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    monkeypatch.setattr(
        "gpu_fault.processor.coordinator.urlopen",
        lambda request, **_kwargs: _Response(),
    )
    clock = _ClaimClock(store)
    coordinators = [_coordinator(clock, owner) for owner in ("pod-a:1", "pod-b:1")]
    threads = [
        Thread(target=getattr(coordinator, target), daemon=True)
        for coordinator in coordinators
        for target in ("run_queue_notifications", "run_processor")
    ]
    delays: list[float] = []
    with _connect() as listener:
        listener.execute("LISTEN gpu_fault_processor_queue")
        for thread in threads:
            thread.start()
        try:
            assert _wait_until(
                lambda: all(
                    coordinator.metrics_snapshot()["notifications_enabled"] == 1
                    for coordinator in coordinators
                ),
                15,
            ), "both coordinators must own a notification shard"
            time.sleep(SETTLE_SECONDS)
            rounds = [_fault_rounds(coordinator) for coordinator in coordinators]
            time.sleep(IDLE_WINDOW_SECONDS)
            idle_rounds = [
                _fault_rounds(coordinator) - before
                for coordinator, before in zip(coordinators, rounds)
            ]

            received = [_notifications_received(c) for c in coordinators]
            edge = _terminal("attempt-edge")
            # From before the INSERT: its NOTIFY can wake a claim loop before
            # the enqueue call has returned.
            admitted = time.monotonic()
            store.enqueue_processor_request(edge)
            assert _wait_until(lambda: edge.request_id in clock.claimed_at, 10), (
                "the row admitted to the empty queue was never claimed"
            )
            edge_delay = clock.claimed_at[edge.request_id] - admitted
            woken = [
                _notifications_received(coordinator) - before
                for coordinator, before in zip(coordinators, received)
            ]

            # Its own edge wakes both streams; settled again, the deferred row
            # is a fault backlog and rows queued behind it raise no edge.
            store.enqueue_processor_request(
                _terminal("attempt-deferred").model_copy(
                    update={
                        "not_before": datetime.now(timezone.utc) + timedelta(hours=1)
                    }
                )
            )
            time.sleep(SETTLE_SECONDS)
            for index, gap in enumerate((0.0, 1.3, 1.7)):
                time.sleep(gap)  # land on different phases of the cadence
                item = _terminal(f"attempt-x{index}")
                store.enqueue_processor_request(item)
                enqueued = time.monotonic()
                assert _wait_until(lambda: item.request_id in clock.claimed_at, 10), (
                    f"{item.request_id} was never claimed"
                )
                delays.append(clock.claimed_at[item.request_id] - enqueued)
            notifications = [
                json.loads(notification.payload)
                for notification in listener.notifies(timeout=0.5, stop_after=64)
            ]
        finally:
            for coordinator in coordinators:
                coordinator.stop()
            for thread in threads:
                thread.join(timeout=10)
    assert not any(thread.is_alive() for thread in threads), "a loop did not stop"
    request.node.user_properties.extend(
        [
            ("idle_fault_rounds", json.dumps(idle_rounds)),
            ("edge_claim_seconds", round(edge_delay, 3)),
            ("behind_backlog_claim_seconds", json.dumps([round(d, 3) for d in delays])),
        ]
    )
    assert max(idle_rounds) <= 2, (
        f"the idle fault streams polled {idle_rounds} times in "
        f"{IDLE_WINDOW_SECONDS:.0f} s instead of waiting on the edge"
    )
    assert edge_delay < NOTIFIED_CLAIM_BOUND_SECONDS, (
        f"the row admitted to the empty queue waited {edge_delay:.3f} s"
    )
    assert min(woken) >= 1, f"the edge woke {woken} coordinators' listeners"
    assert notifications == [{"priority": 0, "path": "/v1/attempts/terminal"}] * 2, (
        "only the empty queue's row and the deferred row raise an edge"
    )
    assert max(delays) < CLAIM_BOUND_SECONDS, (
        f"fault rows behind the deferred one waited "
        f"{[round(delay, 3) for delay in delays]} s to be claimed"
    )
