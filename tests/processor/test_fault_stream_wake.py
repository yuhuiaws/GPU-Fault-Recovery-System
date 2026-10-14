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
row, so a fault stream with notifications on that the lane probe does not hold
asks the store for its fault backlog -- PENDING, or LEASED with a lapsed lease
-- as soon as the answer can lengthen its interval, settled or not, and with
none waits on the edge notification (the 5 s fallback). A backlog keeps the
0.5 s cadence and is believed for one fallback interval; a row under a live
lease is no backlog. A lane-blocked backlog keeps 0.1 s while a fault-tier row
waits and the routine busy ceiling, 0.4 s, while none does.
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
    ProcessorRequestStatus,
    telemetry_spool,
)
from gpu_fault.processor.claim_backoff import BackoffCeilings, empty_round_ceiling
from gpu_fault.processor.lane_runtime import FaultSlotWake
from gpu_fault.store import SqliteStore
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
    store,
    *,
    fault_workers: int,
    host_workers: int,
    poll_seconds: float,
    fallback: float = 5.0,
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
        lease=ProcessorLeaseSettings(
            poll_seconds=poll_seconds, processor_notification_fallback_seconds=fallback
        ),
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
        # A fault-tier row waits to be claimed somewhere.
        "fault_backlog_empty": lambda _hold: False,
    }
    assert empty_round_ceiling(ceilings, lane_blocked=lambda: False, **settled) == fault
    assert empty_round_ceiling(ceilings, lane_blocked=lambda: True, **settled) == 0.1


class _Probes:
    """Answers an empty round's two store probes and records which it asked."""

    def __init__(self, *, lane_blocked: bool, fault_backlog_empty: bool) -> None:
        self.asked: list[str] = []
        self.holds: list[float] = []
        self._lane_blocked = lane_blocked
        self._fault_backlog_empty = fault_backlog_empty

    def lane_blocked(self) -> bool:
        self.asked.append("lane")
        return self._lane_blocked

    def fault_backlog_empty(self, hold: float) -> bool:
        self.asked.append("fault-backlog")
        self.holds.append(hold)
        return self._fault_backlog_empty


@pytest.mark.parametrize(
    (
        "notified",
        "settled",
        "pools_busy",
        "next_interval",
        "backlog_empty",
        "lane_blocked",
        "ceiling",
        "asked",
    ),
    [
        pytest.param(
            True,
            True,
            False,
            1.0,
            True,
            False,
            5.0,
            ["lane", "fault-backlog"],
            id="idle-queue",
        ),
        pytest.param(
            True,
            True,
            True,
            1.0,
            True,
            False,
            5.0,
            ["lane", "fault-backlog"],
            id="idle-busy-pools",
        ),
        pytest.param(
            True,
            False,
            False,
            1.0,
            True,
            False,
            5.0,
            ["lane", "fault-backlog"],
            id="unsettled-idle-pools",
        ),
        pytest.param(
            True,
            False,
            False,
            0.4,
            True,
            False,
            0.5,
            ["lane"],
            id="interval-below-fault-idle",
        ),
        pytest.param(
            True,
            True,
            False,
            1.0,
            False,
            False,
            0.5,
            ["lane", "fault-backlog"],
            id="backlog",
        ),
        pytest.param(
            True,
            True,
            False,
            1.0,
            False,
            True,
            0.1,
            ["lane", "fault-backlog"],
            id="lane-blocked-backlog",
        ),
        pytest.param(
            True,
            True,
            False,
            1.0,
            True,
            True,
            0.4,
            ["lane", "fault-backlog"],
            id="lane-blocked-routine",
        ),
        pytest.param(
            True,
            False,
            False,
            0.2,
            True,
            True,
            0.4,
            ["lane", "fault-backlog"],
            id="lane-blocked-routine-early",
        ),
        pytest.param(True, False, True, 1.0, True, False, 0.1, [], id="busy-path"),
        pytest.param(False, True, False, 1.0, True, False, 0.5, ["lane"], id="poller"),
        pytest.param(
            False, True, False, 1.0, True, True, 0.1, ["lane"], id="poller-lane-blocked"
        ),
        pytest.param(
            False, False, True, 1.0, True, False, 0.1, [], id="poller-busy-path"
        ),
    ],
)
def test_a_notified_fault_stream_asks_the_lane_then_the_fault_backlog(
    notified: bool,
    settled: bool,
    pools_busy: bool,
    next_interval: float,
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
        next_interval=next_interval,
        settled_empty=settled,
        pools_busy=pools_busy,
        lane_blocked=probes.lane_blocked,
        fault_backlog_empty=probes.fault_backlog_empty,
    )

    assert result == ceiling
    assert probes.asked == asked
    assert probes.holds == [5.0] * asked.count("fault-backlog"), (
        "a backlog is believed for one notification fallback"
    )


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


@pytest.fixture
def sqlite_store(tmp_path):
    """A store that answers the fault backlog query. The in-memory store has
    no notifications and no such query: it keeps the fault stream polling."""

    store = SqliteStore(str(tmp_path / "fault-stream.db"))
    try:
        yield store
    finally:
        store.close()


class _CountingStore:
    """A store, counting the probes the empty fault rounds ask."""

    def __init__(self, store: Any) -> None:
        self._store = store
        self.backlog_reads = 0
        self.lane_probes = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._store, name)

    def has_unleased_fault_backlog(self, *, now: datetime) -> bool:
        self.backlog_reads += 1
        return bool(self._store.has_unleased_fault_backlog(now=now))

    def active_backlog_is_lane_blocked(self, **kwargs: Any) -> bool:
        self.lane_probes += 1
        return bool(self._store.active_backlog_is_lane_blocked(**kwargs))


class _UncountedStore:
    """A store that predates ``has_unleased_fault_backlog``."""

    def __init__(self, store: Any) -> None:
        self._store = store

    def __getattr__(self, name: str) -> Any:
        if name == "has_unleased_fault_backlog":
            raise AttributeError(name)
        return getattr(self._store, name)


def _notified_coordinator(
    store: Any, *, shard: int | None, fallback: float = 5.0
) -> ProcessorCoordinator:
    """One fault worker and no dedicated pools: only the fault stream claims."""

    processor = _coordinator(
        store, fault_workers=1, host_workers=0, poll_seconds=0.1, fallback=fallback
    )
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


# The production cadence (0.1 s poll, 0.5 s fault idle): the interval doubles
# from the poll and is capped at fault idle while a row can be silent.
UNSETTLED_INTERVALS = [0.2, 0.4, 0.5, 0.5, 0.5, 0.5, 0.5]
# With no fault-tier row waiting it doubles straight out to the 5 s fallback:
# no longer eight rounds at 0.5 s first.
EDGE_INTERVALS = [0.2, 0.4, 0.8, 1.6, 3.2, 5.0, 5.0]


def test_an_idle_shard_owner_waits_on_notifications_instead_of_polling(
    sqlite_store: SqliteStore,
) -> None:
    store = _CountingStore(sqlite_store)
    processor = _notified_coordinator(store, shard=0)

    intervals = _empty_fault_rounds(processor, 12)

    assert intervals == pytest.approx(EDGE_INTERVALS + [5.0] * 5), (
        "with no fault-tier row the stream doubles out to the fallback at once"
    )
    assert store.lane_probes == 12, "the lane probe decides every round first"
    assert store.backlog_reads == 10, (
        "the backlog is read once the fallback can lengthen the interval, "
        "and an empty answer is read fresh every round"
    )
    assert processor.metrics_snapshot()["claim"]["probes"] == 22


def test_a_shardless_process_keeps_polling_the_fault_stream() -> None:
    store = _CountingStore(build_store())
    processor = _notified_coordinator(store, shard=None)

    intervals = _empty_fault_rounds(processor, 12)

    assert intervals == pytest.approx(UNSETTLED_INTERVALS + [0.5] * 5), (
        "a process that is never woken keeps the fault idle ceiling"
    )
    assert store.backlog_reads == 0
    assert store.lane_probes == 12


def _deferred_terminal(attempt_id: str) -> Any:
    return _terminal(attempt_id).model_copy(
        update={"not_before": datetime.now(timezone.utc) + timedelta(hours=1)}
    )


@pytest.mark.parametrize("fallback", [5.0, 2.0])
def test_a_fault_backlog_keeps_the_cadence_and_is_read_once_per_fallback(
    monkeypatch: pytest.MonkeyPatch, sqlite_store: SqliteStore, fallback: float
) -> None:
    clock = SimpleNamespace(now=1000.0)
    monkeypatch.setattr(
        telemetry_spool, "time", SimpleNamespace(monotonic=lambda: clock.now)
    )
    store = _CountingStore(sqlite_store)
    # PENDING, never claimable: a row queued behind it would raise no edge.
    store.enqueue_processor_request(_deferred_terminal("attempt-deferred"))
    processor = _notified_coordinator(store, shard=0, fallback=fallback)

    intervals = _empty_fault_rounds(processor, 12)

    assert intervals == pytest.approx(UNSETTLED_INTERVALS + [0.5] * 5), (
        "a PENDING fault-tier row keeps the stream on the 0.5 s cadence"
    )
    assert store.backlog_reads == 1, "a non-empty backlog is believed for the hold"
    assert store.lane_probes == 12, "behind a backlog the lane probe still decides"
    clock.now += fallback - 0.1
    _empty_fault_rounds(processor, 3)
    assert store.backlog_reads == 1, (
        "GPU_FAULT_PROCESSOR_NOTIFICATION_FALLBACK_SECONDS is the hold"
    )
    clock.now += 0.2
    assert _empty_fault_rounds(processor, 1) == pytest.approx([0.5])
    assert store.backlog_reads == 2, "the hold has expired: the backlog is read again"


def _claim_elsewhere(store: Any, item: Any, *, lease: timedelta) -> Any:
    """Another process's worker takes ``item`` and holds it for ``lease``."""

    store.enqueue_processor_request(item)
    (claimed,) = store.claim_active_processor_requests(
        "pod-z:1",
        now=datetime.now(timezone.utc),
        lease_duration=lease,
        limit=1,
        include_paths={item.path},
    )
    assert claimed.request_id == item.request_id
    return claimed


def test_a_fault_row_under_a_live_lease_is_no_backlog(
    sqlite_store: SqliteStore,
) -> None:
    store = _CountingStore(sqlite_store)
    # A long fault handler on another process: it silences no edge.
    _claim_elsewhere(store, _terminal("attempt-running"), lease=timedelta(minutes=2))
    processor = _notified_coordinator(store, shard=0)

    assert store.processor_fault_backlog_depth() == 1, "the depth still counts it"
    assert _empty_fault_rounds(processor, 7) == pytest.approx(EDGE_INTERVALS), (
        "a fault row a worker holds no longer keeps every process at 0.5 s"
    )
    assert store.backlog_reads == 5


def _complete_elsewhere(store: Any, holder: Any) -> None:
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


def test_a_lane_blocked_routine_row_keeps_the_fault_stream_polling(
    sqlite_store: SqliteStore,
) -> None:
    """Admin, fleet and collector-health rows ride the fault stream. Behind a
    lane held in another pool or process nothing announces the lane freeing
    up, so the lane probe keeps the stream polling -- at the routine busy
    ceiling, the one those rows get on a routine stream, while no fault-tier
    row waits."""

    store = _CountingStore(sqlite_store)
    marker = processor_request(
        "/v1/markers", body=b'{"node_id":"node-a"}', cluster_id="cluster-a"
    )
    holder = _claim_elsewhere(store, marker, lease=timedelta(minutes=2))
    queued = processor_request(
        "/v1/markers", body=b'{"node_id":"node-b"}', cluster_id="cluster-a"
    )
    store.enqueue_processor_request(queued)
    assert holder.ordering_key() == queued.ordering_key() == "cluster-a"
    processor = _notified_coordinator(store, shard=0)

    assert _empty_fault_rounds(processor, 12) == pytest.approx([0.2] + [0.4] * 11), (
        "a held lane polls the routine row at most every 0.4 s"
    )
    assert store.lane_probes == 12
    assert store.backlog_reads == 12, "an empty fault backlog is read fresh"

    _complete_elsewhere(store, holder)
    _empty_fault_rounds(processor, 1)
    taken = store.get_processor_request(queued.request_id)
    assert (taken.status, taken.lease_owner) == (
        ProcessorRequestStatus.LEASED,
        "pod-a:1",
    ), "the next fault round claims the row once its lane is free"


def test_a_lane_blocked_fault_row_keeps_the_fault_busy_ceiling(
    sqlite_store: SqliteStore,
) -> None:
    store = _CountingStore(sqlite_store)
    xid = processor_request("/v1/gpu-events/xid", body=b'{"node_id":"node-a"}')
    _claim_elsewhere(store, xid, lease=timedelta(minutes=2))
    queued = processor_request("/v1/gpu-events/xid", body=b'{"node_id":"node-a"}')
    store.enqueue_processor_request(queued)
    processor = _notified_coordinator(store, shard=0)

    assert _empty_fault_rounds(processor, 12) == pytest.approx([0.1] * 12), (
        "a fault row behind a held lane is polled every 0.1 s"
    )
    assert store.backlog_reads == 1, "its PENDING row is believed for the hold"


def test_a_backlog_the_store_cannot_count_keeps_the_fault_stream_polling(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    sqlite_store: SqliteStore,
) -> None:
    for store_without_query in (_UncountedStore(sqlite_store), build_store()):
        uncounted = _notified_coordinator(store_without_query, shard=0)
        assert _empty_fault_rounds(uncounted, 12) == pytest.approx(
            UNSETTLED_INTERVALS + [0.5] * 5
        ), "a store without the backlog query keeps the fault idle ceiling"

    store = sqlite_store

    def broken(*, now: datetime) -> bool:
        raise RuntimeError("statement timeout")

    monkeypatch.setattr(store, "has_unleased_fault_backlog", broken)
    failing = _notified_coordinator(store, shard=0)
    with caplog.at_level(logging.ERROR, logger=telemetry_spool.LOGGER.name):
        intervals = _empty_fault_rounds(failing, 12)

    assert intervals == pytest.approx(UNSETTLED_INTERVALS + [0.5] * 5), (
        "a failed backlog read keeps the fault idle ceiling"
    )
    assert [
        record.getMessage()
        for record in caplog.records
        if record.name == telemetry_spool.LOGGER.name
    ] == ["processor fault backlog probe failed"] * 10


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
