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
"""

from __future__ import annotations

import json
import time
from concurrent.futures import Future
from datetime import timedelta
from threading import Event, Lock, Thread
from typing import Any, Callable

import pytest

from gpu_fault.processor import (
    ProcessorCoordinator,
    ProcessorLeaseSettings,
    ProcessorPoolSettings,
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
def test_the_fault_stream_never_stretches_to_the_notification_fallback(
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
    }
    assert empty_round_ceiling(ceilings, lane_blocked=lambda: False, **settled) == fault
    assert empty_round_ceiling(ceilings, lane_blocked=lambda: True, **settled) == 0.1


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
