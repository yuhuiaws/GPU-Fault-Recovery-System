"""The ``gpu-fault-processor-inbox`` consumer thread has a liveness signal and
does not die of an exception outside its claim ``try`` (B-6, 2026-09-08).

``/livez`` checks only the spool consumer; a consumer loop that exited left the
Pod Ready with nothing claiming. ``processor_consumer_running`` and
``consumer_last_cycle_age_seconds`` are the signal for ``/livez`` and an alert
("process up, claim rounds flat").
"""

from __future__ import annotations

import time
from threading import Thread

from gpu_fault.processor import (
    ProcessorCoordinator,
    ProcessorLeaseSettings,
    ProcessorPoolSettings,
)
from tests._builders import build_store


def _processor(store) -> ProcessorCoordinator:
    return ProcessorCoordinator(
        store,
        owner_id="pod-a:1",
        internal_token="processor-token",
        pools=ProcessorPoolSettings(fault_worker_count=1),
        lease=ProcessorLeaseSettings(poll_seconds=0.01, idle_backoff_max_seconds=0.02),
        active_consumers=True,
    )


def _wait_until(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def test_consumer_liveness_follows_the_run_processor_thread():
    processor = _processor(build_store())
    assert processor.processor_consumer_running is False
    assert processor.consumer_last_cycle_age_seconds is None
    thread = Thread(target=processor.run_processor, daemon=True)
    thread.start()
    try:
        assert _wait_until(lambda: processor.processor_consumer_running), (
            "run_processor must flag the consumer as running"
        )
        assert _wait_until(
            lambda: processor.metrics_snapshot()["consumer"]["cycles"] >= 2
        ), "the consumer must complete at least two cycles"
        snapshot = processor.metrics_snapshot()["consumer"]
        assert snapshot["running"] == 1
        assert 0.0 <= snapshot["last_cycle_age_seconds"] < 1.0
        assert processor.consumer_is_live(max_cycle_age_seconds=1.0), (
            "a consumer that cycled within 1 s is live"
        )
    finally:
        processor.stop()
        thread.join(timeout=2)
    assert not thread.is_alive(), "stop() must end the run_processor thread"
    assert processor.processor_consumer_running is False
    assert processor.metrics_snapshot()["consumer"]["running"] == 0
    assert not processor.consumer_is_live(max_cycle_age_seconds=60.0), (
        "a stopped consumer is never live"
    )


def test_consumer_loop_survives_an_exception_before_the_claim(monkeypatch):
    processor = _processor(build_store())
    calls: list[int] = []
    original = processor._check_execution_deadlines

    def flaky() -> None:
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("deadline sweep blew up")
        original()

    monkeypatch.setattr(processor, "_check_execution_deadlines", flaky)
    thread = Thread(target=processor.run_processor, daemon=True)
    thread.start()
    try:
        assert _wait_until(lambda: len(calls) >= 3), calls
        assert thread.is_alive(), "one failing sweep must not kill the loop"
        assert processor.processor_consumer_running, (
            "the consumer keeps running after a cycle error"
        )
        assert processor.metrics_snapshot()["consumer"]["cycle_errors"] == 1
    finally:
        processor.stop()
        thread.join(timeout=2)
    assert not thread.is_alive(), "stop() must end the run_processor thread"
