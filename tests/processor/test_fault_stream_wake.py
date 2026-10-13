"""A free fault worker is never left waiting on the notification fallback.

The queue trigger announces only idle->busy (path, priority) edges: a
priority-0 row admitted while another one is PENDING (lane-blocked, deferred,
held by the observation interlock) raises no notification. A perf run showed
every control-worker backing the fault stream off to the 5 s notification
fallback, so such rows -- and the row queued behind a fault worker that had
just freed its lane -- waited in 5 s steps (fault p99 16.2 s, 11.9 s of it
processor wait). The fault stream now keeps its own idle ceiling, a finished
fault worker wakes the claim loop, and a fault claim that returned rows
without filling its limit is re-polled one poll interval later.

That 0.5 s cadence then made 97% of a perf run's fault claim rounds empty
polls of an idle queue. A row can only be silent behind a PENDING fault-tier
row, so a settled fault stream with notifications on asks the store for its
fault backlog instead of the lane probe and, with none, waits on the edge
notification (the 5 s fallback); any fault-tier row PENDING or LEASED keeps
the 0.5 s cadence, and the depth query is bounded.
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import Future
from datetime import datetime, timedelta, timezone
from threading import Event, Lock, Thread
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from gpu_fault.processor import (
    ProcessorCoordinator,
    ProcessorLeaseSettings,
    ProcessorPoolSettings,
    telemetry_spool,
)
from gpu_fault.processor.claim_backoff import BackoffCeilings, empty_round_ceiling
from gpu_fault.processor.lane_runtime import FaultSlotWake
from tests._builders import build_store, processor_request

ROUTINE_STREAMS = (
    "observation",
    "gpu-inventory",
    "gpu-metrics",
    "host-telemetry",
    "node-log",
)


def _terminal(attempt_id: str):
    return processor_request(
        "/v1/attempts/terminal", body=json.dumps({"attempt_id": attempt_id}).encode()
    )


def _coordinator(
    store, *, fault_workers: int, host_workers: int, poll_seconds: float
) -> ProcessorCoordinator:
    return ProcessorCoordinator(
        store,
        owner_id="pod-a:1",
        internal_token="token-" + "x" * 32,
        local_url="http://127.0.0.1:1",
        pools=ProcessorPoolSettings(
            fault_worker_count=fault_workers,
            observation_worker_count=0,
            gpu_telemetry_worker_count=0,
            host_telemetry_worker_count=host_workers,
        ),
        lease=ProcessorLeaseSettings(poll_seconds=poll_seconds),
        active_consumers=True,
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


class _Replay:
    """Stands in for the local replay; holds the first attempt until released."""

    def __init__(self) -> None:
        self.entered: list[tuple[str, float]] = []
        self.first_entered = Event()
        self.release = Event()
        self._lock = Lock()

    def __call__(self, request: Any, **_kwargs: Any) -> _Response:
        attempt = json.loads(request.data)["attempt_id"]
        with self._lock:
            self.entered.append((attempt, time.monotonic()))
            first = len(self.entered) == 1
        if first:
            self.first_entered.set()
            self.release.wait(10)
        return _Response()


def _wait_until(predicate: Callable[[], bool], timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


@pytest.mark.parametrize(
    ("notifications_enabled", "fallback", "fault", "routine"),
    [
        pytest.param(False, 5.0, 0.5, 2.0, id="polling"),
        pytest.param(True, 5.0, 0.5, 5.0, id="notified"),
        pytest.param(True, 0.3, 0.3, 0.3, id="notified-short-fallback"),
        pytest.param(False, 0.3, 0.5, 2.0, id="polling-short-fallback"),
    ],
)
def test_a_fault_backlog_keeps_the_fault_stream_off_the_notification_fallback(
    notifications_enabled: bool, fallback: float, fault: float, routine: float
) -> None:
    ceilings = BackoffCeilings(
        idle=2.0,
        busy=0.4,
        fault_idle=0.5,
        fault_busy=0.1,
        notification_fallback=fallback,
        notifications_enabled=notifications_enabled,
    )

    assert ceilings.idle_ceiling("fault") == fault
    assert {stream: ceilings.idle_ceiling(stream) for stream in ROUTINE_STREAMS} == {
        stream: routine for stream in ROUTINE_STREAMS
    }
    settled: dict[str, Any] = {
        "stream": "fault",
        "next_interval": 8.0,
        "settled_empty": True,
        "pools_busy": True,
        # A fault-tier row is PENDING or LEASED somewhere.
        "fault_backlog_empty": lambda: False,
    }
    assert empty_round_ceiling(ceilings, lane_blocked=lambda: False, **settled) == fault
    assert empty_round_ceiling(ceilings, lane_blocked=lambda: True, **settled) == 0.1


class _Probes:
    """Answers an empty round's two store probes and records which it asked."""

    def __init__(self, *, lane_blocked: bool, fault_backlog_empty: bool) -> None:
        self.asked: list[str] = []
        self._lane_blocked = lane_blocked
        self._fault_backlog_empty = fault_backlog_empty

    def lane_blocked(self) -> bool:
        self.asked.append("lane")
        return self._lane_blocked

    def fault_backlog_empty(self) -> bool:
        self.asked.append("fault-backlog")
        return self._fault_backlog_empty


@pytest.mark.parametrize(
    (
        "notified",
        "settled",
        "pools_busy",
        "backlog_empty",
        "lane_blocked",
        "ceiling",
        "asked",
    ),
    [
        pytest.param(
            True, True, False, True, True, 5.0, ["fault-backlog"], id="idle-queue"
        ),
        pytest.param(
            True, True, True, True, True, 5.0, ["fault-backlog"], id="idle-busy-pools"
        ),
        pytest.param(
            True,
            True,
            False,
            False,
            False,
            0.5,
            ["fault-backlog", "lane"],
            id="backlog",
        ),
        pytest.param(
            True,
            True,
            False,
            False,
            True,
            0.1,
            ["fault-backlog", "lane"],
            id="lane-blocked-backlog",
        ),
        pytest.param(True, False, True, True, False, 0.1, [], id="busy-path"),
        pytest.param(True, False, False, True, False, 0.5, ["lane"], id="unsettled"),
        pytest.param(False, True, False, True, False, 0.5, ["lane"], id="poller"),
        pytest.param(
            False, True, False, True, True, 0.1, ["lane"], id="poller-lane-blocked"
        ),
        pytest.param(False, False, True, True, False, 0.1, [], id="poller-busy-path"),
    ],
)
def test_only_a_settled_notified_fault_stream_asks_for_the_fault_backlog(
    notified: bool,
    settled: bool,
    pools_busy: bool,
    backlog_empty: bool,
    lane_blocked: bool,
    ceiling: float,
    asked: list[str],
) -> None:
    ceilings = BackoffCeilings(
        idle=2.0,
        busy=0.4,
        fault_idle=0.5,
        fault_busy=0.1,
        notification_fallback=5.0,
        notifications_enabled=notified,
    )
    probes = _Probes(lane_blocked=lane_blocked, fault_backlog_empty=backlog_empty)

    result = empty_round_ceiling(
        ceilings,
        stream="fault",
        next_interval=1.0,
        settled_empty=settled,
        pools_busy=pools_busy,
        lane_blocked=probes.lane_blocked,
        fault_backlog_empty=probes.fault_backlog_empty,
    )

    assert result == ceiling
    assert probes.asked == asked


@pytest.mark.parametrize("stream", ROUTINE_STREAMS)
def test_the_routine_streams_never_ask_for_the_fault_backlog(stream: str) -> None:
    ceilings = BackoffCeilings(
        idle=2.0,
        busy=0.4,
        fault_idle=0.5,
        fault_busy=0.1,
        notification_fallback=5.0,
        notifications_enabled=True,
    )
    probes = _Probes(lane_blocked=False, fault_backlog_empty=False)

    for settled in (True, False):
        assert (
            empty_round_ceiling(
                ceilings,
                stream=stream,
                next_interval=1.0,
                settled_empty=settled,
                pools_busy=False,
                lane_blocked=probes.lane_blocked,
                fault_backlog_empty=probes.fault_backlog_empty,
            )
            == 5.0
        )
    assert probes.asked == ["lane", "lane"]


EMPTY_FAULT_ROUND = {"fault": 1, "observation": 0, "gpu": 0, "host": 0}


class _CountingStore:
    """The in-memory store, counting the probes the empty fault rounds ask."""

    def __init__(self, store: Any) -> None:
        self._store = store
        self.depth_reads = 0
        self.lane_probes = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._store, name)

    def processor_fault_backlog_depth(self) -> int:
        self.depth_reads += 1
        return int(self._store.processor_fault_backlog_depth())

    def active_backlog_is_lane_blocked(self, **kwargs: Any) -> bool:
        self.lane_probes += 1
        return bool(self._store.active_backlog_is_lane_blocked(**kwargs))


class _UncountedStore:
    """A store that predates ``processor_fault_backlog_depth``."""

    def __init__(self, store: Any) -> None:
        self._store = store

    def __getattr__(self, name: str) -> Any:
        if name == "processor_fault_backlog_depth":
            raise AttributeError(name)
        return getattr(self._store, name)


def _notified_coordinator(store: Any, *, shard: int | None) -> ProcessorCoordinator:
    """One fault worker and no dedicated pools: only the fault stream claims."""

    processor = _coordinator(store, fault_workers=1, host_workers=0, poll_seconds=0.1)
    processor._set_notification_state(True, shard)
    return processor


def _empty_fault_rounds(processor: ProcessorCoordinator, count: int) -> list[float]:
    """Run ``count`` back-to-back fault rounds; the fault interval after each."""

    intervals = []
    for _ in range(count):
        processor._stream_idle_until.clear()
        processor._claim_active_by_pool(
            EMPTY_FAULT_ROUND, lease_duration=timedelta(seconds=30)
        )
        intervals.append(processor.metrics_snapshot()["claim"]["backoff_seconds"])
    return intervals


# The production cadence (0.1 s poll, 0.5 s fault idle, 8 rounds to settle):
# the seven rounds before the stream settles empty double up to fault idle.
UNSETTLED_INTERVALS = [0.2, 0.4, 0.5, 0.5, 0.5, 0.5, 0.5]


def test_an_idle_shard_owner_waits_on_notifications_instead_of_polling() -> None:
    store = _CountingStore(build_store())
    processor = _notified_coordinator(store, shard=0)

    intervals = _empty_fault_rounds(processor, 12)

    assert intervals[:7] == pytest.approx(UNSETTLED_INTERVALS)
    assert intervals[7:] == pytest.approx([1.0, 2.0, 4.0, 5.0, 5.0]), (
        "with no fault-tier row the settled stream doubles out to the fallback"
    )
    assert store.lane_probes == 7, "the settled rounds skip the lane probe"
    assert store.depth_reads == 5, "an empty fault backlog is read fresh every round"
    assert processor.metrics_snapshot()["claim"]["probes"] == 12


def test_a_shardless_process_keeps_polling_the_fault_stream() -> None:
    store = _CountingStore(build_store())
    processor = _notified_coordinator(store, shard=None)

    intervals = _empty_fault_rounds(processor, 12)

    assert intervals == pytest.approx(UNSETTLED_INTERVALS + [0.5] * 5), (
        "a process that is never woken keeps the fault idle ceiling"
    )
    assert store.depth_reads == 0
    assert store.lane_probes == 12


def _deferred_terminal(attempt_id: str) -> Any:
    return _terminal(attempt_id).model_copy(
        update={"not_before": datetime.now(timezone.utc) + timedelta(hours=1)}
    )


def test_a_fault_backlog_keeps_the_cadence_and_is_read_once_per_hold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = SimpleNamespace(now=1000.0)
    monkeypatch.setattr(
        telemetry_spool, "time", SimpleNamespace(monotonic=lambda: clock.now)
    )
    store = _CountingStore(build_store())
    # PENDING, never claimable: a row queued behind it would raise no edge.
    store.enqueue_processor_request(_deferred_terminal("attempt-deferred"))
    processor = _notified_coordinator(store, shard=0)

    intervals = _empty_fault_rounds(processor, 12)

    assert intervals == pytest.approx(UNSETTLED_INTERVALS + [0.5] * 5), (
        "a PENDING fault-tier row keeps the settled stream on the 0.5 s cadence"
    )
    assert store.depth_reads == 1, "a non-empty backlog is believed for the hold"
    assert store.lane_probes == 12, "behind a backlog the lane probe still decides"
    clock.now += 4.9
    _empty_fault_rounds(processor, 3)
    assert store.depth_reads == 1
    clock.now += 0.2
    assert _empty_fault_rounds(processor, 1) == pytest.approx([0.5])
    assert store.depth_reads == 2, "the hold has expired: the backlog is read again"


def test_a_backlog_the_store_cannot_count_keeps_the_fault_stream_polling(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    uncounted = _notified_coordinator(_UncountedStore(build_store()), shard=0)
    assert _empty_fault_rounds(uncounted, 12) == pytest.approx(
        UNSETTLED_INTERVALS + [0.5] * 5
    ), "a store without the depth query keeps the fault idle ceiling"

    store = build_store()

    def broken() -> int:
        raise RuntimeError("statement timeout")

    monkeypatch.setattr(store, "processor_fault_backlog_depth", broken)
    failing = _notified_coordinator(store, shard=0)
    with caplog.at_level(logging.ERROR, logger=telemetry_spool.LOGGER.name):
        intervals = _empty_fault_rounds(failing, 12)

    assert intervals == pytest.approx(UNSETTLED_INTERVALS + [0.5] * 5), (
        "a failed depth read keeps the fault idle ceiling"
    )
    assert [
        record.getMessage()
        for record in caplog.records
        if record.name == telemetry_spool.LOGGER.name
    ] == ["processor fault backlog probe failed"] * 5


def test_finished_fault_workers_coalesce_into_one_wake() -> None:
    work_available = Event()
    slots = FaultSlotWake(work_available)
    assert slots.take() is False
    futures: list[Future[None]] = [Future() for _ in range(16)]
    for future in futures:
        future.add_done_callback(slots.worker_finished)
    threads = [Thread(target=future.set_result, args=(None,)) for future in futures]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert work_available.is_set(), "a finished fault worker wakes the claim loop"
    assert slots.take() is True
    assert slots.take() is False, "sixteen finished workers are one wake"

    work_available.clear()
    finished: Future[None] = Future()
    finished.set_result(None)
    finished.add_done_callback(slots.worker_finished)
    assert work_available.is_set(), "a future that is already done wakes at once"
    assert slots.take() is True


def _second_claim_delay(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fault_workers: int,
    host_workers: int,
    enqueue_second_after_start: bool,
) -> float:
    """Seconds from releasing the first fault row to the second one's replay."""

    store = build_store()
    replay = _Replay()
    monkeypatch.setattr("gpu_fault.processor.coordinator.urlopen", replay)
    store.enqueue_processor_request(_terminal("attempt-1"))
    if not enqueue_second_after_start:
        store.enqueue_processor_request(_terminal("attempt-2"))
    # A 2 s poll clamps every backoff ceiling to 2 s: only a wake claims sooner.
    processor = _coordinator(
        store, fault_workers=fault_workers, host_workers=host_workers, poll_seconds=2.0
    )
    thread = Thread(target=processor.run_processor, daemon=True)
    thread.start()
    try:
        assert replay.first_entered.wait(5), "the first fault row was never replayed"
        if enqueue_second_after_start:
            store.enqueue_processor_request(_terminal("attempt-2"))
        time.sleep(0.3)  # the claim loop is asleep on its poll by now
        released = time.monotonic()
        replay.release.set()
        assert _wait_until(lambda: len(replay.entered) == 2, 5), replay.entered
        return replay.entered[1][1] - released
    finally:
        replay.release.set()
        processor.stop()
        thread.join(timeout=10)
        assert not thread.is_alive(), "the claim loop did not stop"


def test_a_finished_fault_worker_wakes_a_full_pool_claim_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # One fault worker, busy: the next priority-0 row waits for it. The host
    # worker keeps the loop on its claim path (it is not out of slots).
    delay = _second_claim_delay(
        monkeypatch, fault_workers=1, host_workers=1, enqueue_second_after_start=False
    )

    assert delay < 1.0, f"the freed fault worker waited {delay:.2f} s for the poll"


def test_a_finished_fault_worker_wakes_a_loop_asleep_after_a_short_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Two fault workers, one busy: the second row arrives without a
    # notification (the in-memory store has none) after a short claim.
    delay = _second_claim_delay(
        monkeypatch, fault_workers=2, host_workers=0, enqueue_second_after_start=True
    )

    assert delay < 1.0, f"the freed fault worker waited {delay:.2f} s for the poll"


def test_an_unsaturated_fault_claim_is_repolled_after_one_poll() -> None:
    store = build_store()
    store.enqueue_processor_request(_terminal("attempt-1"))
    processor = _coordinator(store, fault_workers=2, host_workers=1, poll_seconds=0.1)
    # The routine streams sit far out at the notification fallback; a short
    # fault claim must not leave the fault stream waiting on their deadlines.
    for stream in ("host-telemetry", "node-log"):
        processor._stream_idle_until[stream] = time.monotonic() + 5.0

    claimed = processor._claim_active_by_pool(
        {"fault": 2, "observation": 0, "gpu": 0, "host": 1},
        lease_duration=timedelta(seconds=30),
    )

    assert [item.path for item in claimed] == ["/v1/attempts/terminal"]
    assert processor._next_claim_wait_seconds() == pytest.approx(0.1)
