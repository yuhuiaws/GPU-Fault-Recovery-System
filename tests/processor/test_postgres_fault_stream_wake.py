"""A fault row queued behind a PENDING one is claimed without a notification.

Local PostgreSQL, two coordinators that each own a notification shard. The
queue trigger announces only idle->busy (path, priority) edges, so a priority-0
row admitted while another priority-0 row of its path is PENDING -- here one
deferred for an hour -- raises no NOTIFY. Every control-worker of a perf run
had backed its fault stream off to the 5 s notification fallback, so such rows
waited in 5 s steps; while a fault-tier row is PENDING the fault stream keeps
its 0.5 s idle ceiling and the row is claimed within it.

With no fault-tier row waiting to be claimed -- PENDING, or LEASED with a
lapsed lease -- the fault stream waits on the edge instead (the fallback): it
polls an idle queue about once per 5 s, doubles out to it right after an edge
rather than first polling eight rounds at 0.5 s, a row admitted to the empty
queue is claimed as soon as its NOTIFY arrives, and the next deferred row puts
the stream back on the 0.5 s cadence. A fault row a worker elsewhere still
holds is no backlog (it silences no edge; once its lease lapses it is claimed
within the fallback), and a routine row queued behind a lane held elsewhere
keeps the stream polling until the lane frees.
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
# Production cadence: with no fault backlog the interval doubles 0.2, 0.4, 0.8,
# 1.6, 3.2 s out to the 5 s fallback (batch 86 first polled eight rounds at up
# to 0.5 s and doubled from 1 s, about 10 s in all).
SETTLE_SECONDS = 11.0
IDLE_WINDOW_SECONDS = 5.0
# Fault claim rounds of one coordinator in the 5 s after an edge: the claim
# that took the row (or lost it), the round a finished worker forces, and the
# doubling above (batch 86: 9-10, eight of them at up to 0.5 s).
POST_EDGE_ROUNDS_MAX = 6
# A lease that lapses silently waits for the stream's next poll: at most the
# 5 s fallback, plus the claim loop on a loaded gate host.
LAPSED_LEASE_BOUND_SECONDS = 6.0


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
            before_edge = [_fault_rounds(coordinator) for coordinator in coordinators]
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
            time.sleep(max(0.0, admitted + IDLE_WINDOW_SECONDS - time.monotonic()))
            post_edge_rounds = [
                _fault_rounds(coordinator) - before
                for coordinator, before in zip(coordinators, before_edge)
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
            ("post_edge_fault_rounds", json.dumps(post_edge_rounds)),
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
    assert max(post_edge_rounds) <= POST_EDGE_ROUNDS_MAX, (
        f"after the edge the fault streams polled {post_edge_rounds} times in "
        f"{IDLE_WINDOW_SECONDS:.0f} s before backing off to the fallback"
    )
    assert min(woken) >= 1, f"the edge woke {woken} coordinators' listeners"
    assert notifications == [{"priority": 0, "path": "/v1/attempts/terminal"}] * 2, (
        "only the empty queue's row and the deferred row raise an edge"
    )
    assert max(delays) < CLAIM_BOUND_SECONDS, (
        f"fault rows behind the deferred one waited "
        f"{[round(delay, 3) for delay in delays]} s to be claimed"
    )


def _claim_elsewhere(store: Any, item: Any) -> Any:
    """A worker on a third process takes ``item`` and holds it for minutes."""

    store.enqueue_processor_request(item)
    (claimed,) = store.claim_active_processor_requests(
        "pod-z:1",
        now=datetime.now(timezone.utc),
        lease_duration=timedelta(minutes=2),
        limit=1,
        include_paths={item.path},
    )
    assert claimed.request_id == item.request_id
    return claimed


def _start(coordinators: list[ProcessorCoordinator]) -> list[Thread]:
    threads = [
        Thread(target=getattr(coordinator, target), daemon=True)
        for coordinator in coordinators
        for target in ("run_queue_notifications", "run_processor")
    ]
    for thread in threads:
        thread.start()
    assert _wait_until(
        lambda: all(
            coordinator.metrics_snapshot()["notifications_enabled"] == 1
            for coordinator in coordinators
        ),
        15,
    ), "both coordinators must own a notification shard"
    return threads


def _stop(coordinators: list[ProcessorCoordinator], threads: list[Thread]) -> None:
    for coordinator in coordinators:
        coordinator.stop()
    for thread in threads:
        thread.join(timeout=10)
    assert not any(thread.is_alive() for thread in threads), "a loop did not stop"


def test_a_fault_row_a_worker_holds_elsewhere_is_no_fault_backlog(
    store, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    monkeypatch.setattr(
        "gpu_fault.processor.coordinator.urlopen",
        lambda request, **_kwargs: _Response(),
    )
    # A long fault handler on a third process: LEASED under a live lease.
    held = _claim_elsewhere(store, _terminal("attempt-running"))
    clock = _ClaimClock(store)
    coordinators = [_coordinator(clock, owner) for owner in ("pod-a:1", "pod-b:1")]
    with _connect() as listener:
        listener.execute("LISTEN gpu_fault_processor_queue")
        threads = _start(coordinators)
        try:
            time.sleep(SETTLE_SECONDS)
            rounds = [_fault_rounds(coordinator) for coordinator in coordinators]
            time.sleep(IDLE_WINDOW_SECONDS)
            idle_rounds = [
                _fault_rounds(coordinator) - before
                for coordinator, before in zip(coordinators, rounds)
            ]

            # The queue trigger's busy probe reads PENDING rows only: the next
            # row of the held row's (path, priority) still raises its edge.
            nearby = _terminal("attempt-next")
            admitted = time.monotonic()
            store.enqueue_processor_request(nearby)
            assert _wait_until(lambda: nearby.request_id in clock.claimed_at, 10), (
                "the row admitted beside the held one was never claimed"
            )
            edge_delay = clock.claimed_at[nearby.request_id] - admitted

            # The holder dies: request and lane lease lapse without an update.
            time.sleep(IDLE_WINDOW_SECONDS)
            with _connect() as connection:
                connection.execute(
                    "UPDATE gpu_fault_processor_queue SET lease_expires_at = "
                    "now() - interval '1 second' WHERE request_id = %s",
                    (held.request_id,),
                )
                connection.execute(
                    "UPDATE gpu_fault_processor_lanes SET lease_expires_at = "
                    "now() - interval '1 second' WHERE ordering_key = %s",
                    (held.ordering_key(),),
                )
            lapsed = time.monotonic()
            assert _wait_until(lambda: held.request_id in clock.claimed_at, 15), (
                "the row whose lease lapsed was never claimed again"
            )
            reclaim_delay = clock.claimed_at[held.request_id] - lapsed
            notifications = [
                json.loads(notification.payload)
                for notification in listener.notifies(timeout=0.5, stop_after=64)
            ]
        finally:
            _stop(coordinators, threads)
    request.node.user_properties.extend(
        [
            ("idle_fault_rounds_beside_a_held_row", json.dumps(idle_rounds)),
            ("edge_claim_seconds", round(edge_delay, 3)),
            ("lapsed_lease_claim_seconds", round(reclaim_delay, 3)),
        ]
    )
    assert max(idle_rounds) <= 2, (
        f"beside a held fault row the idle fault streams polled {idle_rounds} "
        f"times in {IDLE_WINDOW_SECONDS:.0f} s instead of waiting on the edge"
    )
    assert edge_delay < NOTIFIED_CLAIM_BOUND_SECONDS, (
        f"the row admitted beside the held one waited {edge_delay:.3f} s"
    )
    assert notifications == [{"priority": 0, "path": "/v1/attempts/terminal"}], (
        "a LEASED row silences no edge, and a lapsing lease raises none"
    )
    assert reclaim_delay < LAPSED_LEASE_BOUND_SECONDS, (
        f"the lapsed lease waited {reclaim_delay:.3f} s to be claimed again"
    )


def test_a_routine_row_behind_a_lane_held_elsewhere_is_claimed_as_it_frees(
    store, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    monkeypatch.setattr(
        "gpu_fault.processor.coordinator.urlopen",
        lambda request, **_kwargs: _Response(),
    )
    # Admin, fleet and collector-health rows ride the fault stream; this lane
    # is held by a worker of another process (or of another pool), so no
    # finished fault worker here announces it freeing up.
    holder = _claim_elsewhere(
        store,
        processor_request(
            "/v1/markers", body=b'{"node_id":"node-a"}', cluster_id="cluster-lane"
        ),
    )
    clock = _ClaimClock(store)
    coordinators = [_coordinator(clock, owner) for owner in ("pod-a:1", "pod-b:1")]
    threads = _start(coordinators)
    try:
        queued = processor_request(
            "/v1/markers", body=b'{"node_id":"node-b"}', cluster_id="cluster-lane"
        )
        assert queued.ordering_key() == holder.ordering_key()
        store.enqueue_processor_request(queued)
        # Past the eight empty rounds that settled the stream in batch 86, which
        # then backed off towards the fallback without asking the lane probe.
        time.sleep(1.0)
        rounds = [_fault_rounds(coordinator) for coordinator in coordinators]
        time.sleep(3.0)
        blocked_rounds = [
            _fault_rounds(coordinator) - before
            for coordinator, before in zip(coordinators, rounds)
        ]
        assert holder.leader_epoch is not None and holder.lease_token is not None
        store.complete_active_processor_request(
            holder.request_id,
            "pod-z:1",
            holder.leader_epoch,
            holder.lease_token,
            response_status=200,
            response_content_type="application/json",
            response_body_base64="e30=",
        )
        freed = time.monotonic()
        assert _wait_until(lambda: queued.request_id in clock.claimed_at, 10), (
            "the row behind the freed lane was never claimed"
        )
        delay = clock.claimed_at[queued.request_id] - freed
    finally:
        _stop(coordinators, threads)
    request.node.user_properties.extend(
        [
            ("lane_blocked_fault_rounds_per_3s", json.dumps(blocked_rounds)),
            ("freed_lane_claim_seconds", round(delay, 3)),
        ]
    )
    assert delay < CLAIM_BOUND_SECONDS, (
        f"the routine row waited {delay:.3f} s after its lane was freed"
    )
