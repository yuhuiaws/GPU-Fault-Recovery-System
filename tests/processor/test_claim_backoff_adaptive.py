"""Empty claim rounds stop holding a busy replica at the busy ceiling.

92% of a perf run's claim rounds were empty (28k of 30k): while any pool had
a request in flight, every empty stream was capped at the 0.4 s busy ceiling
(0.1 s for faults) and scanned the queue hundreds of times a second for
nothing. With the edge-triggered notifications an idle stream wakes on the
next admitted row, so after ``busy_empty_rounds_before_idle`` consecutive
empty rounds the stream backs off like an idle one.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from gpu_fault.processor import (
    ProcessorCoordinator,
    ProcessorLeaseSettings,
    ProcessorPoolSettings,
)
from gpu_fault.processor.claim_backoff import BackoffCeilings, empty_round_ceiling
from tests._builders import build_store, processor_request

AVAILABLE = {"fault": 2, "observation": 2, "gpu": 2, "host": 2}
LEASE = timedelta(seconds=30)


def _processor(rounds: int = 3) -> ProcessorCoordinator:
    return ProcessorCoordinator(
        build_store(),
        owner_id="pod-a:1",
        internal_token="token-" + "x" * 32,
        pools=ProcessorPoolSettings(
            fault_worker_count=2,
            observation_worker_count=2,
            gpu_telemetry_worker_count=2,
            host_telemetry_worker_count=2,
        ),
        lease=ProcessorLeaseSettings(
            poll_seconds=0.1,
            idle_backoff_max_seconds=2.0,
            busy_backoff_max_seconds=0.4,
            fault_idle_backoff_max_seconds=0.5,
            fault_busy_backoff_max_seconds=0.1,
            busy_empty_rounds_before_idle=rounds,
        ),
        active_consumers=True,
    )


def _round(processor: ProcessorCoordinator) -> None:
    processor._stream_idle_until.clear()
    processor._claim_active_by_pool(AVAILABLE, lease_duration=LEASE)


def test_a_busy_replica_settles_an_empty_stream_onto_the_idle_ceiling() -> None:
    processor = _processor(rounds=3)
    processor._pools_busy = True

    _round(processor)
    _round(processor)
    assert processor._stream_empty_rounds["host-telemetry"] == 2
    assert processor._stream_idle_interval["host-telemetry"] == pytest.approx(0.4), (
        "below the threshold the busy ceiling holds"
    )
    assert processor._stream_idle_interval["fault"] == pytest.approx(0.1)
    assert processor.claim_settled_empty_total == 0

    _round(processor)
    assert processor._stream_empty_rounds["host-telemetry"] == 3
    assert processor._stream_idle_interval["host-telemetry"] == pytest.approx(0.8), (
        "the third empty round leaves the busy ceiling and keeps doubling"
    )
    assert processor._stream_idle_interval["fault"] == pytest.approx(0.2), (
        "the fault stream leaves its 0.1 s busy ceiling and doubles towards 0.5"
    )
    assert processor.claim_settled_empty_total > 0
    for _ in range(4):
        _round(processor)
    assert processor._stream_idle_interval["host-telemetry"] == pytest.approx(2.0)
    assert processor._stream_idle_interval["fault"] == pytest.approx(0.5)


def test_a_non_empty_claim_starts_the_count_over() -> None:
    processor = _processor(rounds=2)
    processor._pools_busy = True
    _round(processor)
    assert processor._stream_empty_rounds["fault"] == 1

    request = processor_request("/v1/gpu-events/xid", body=b'{"node_id":"node-a"}')
    processor.store.enqueue_processor_request(request)
    processor._stream_idle_until.clear()
    claimed = processor._claim_active_by_pool(AVAILABLE, lease_duration=LEASE)

    assert [item.request_id for item in claimed] == [request.request_id]
    assert "fault" not in processor._stream_empty_rounds
    assert "fault" not in processor._stream_idle_interval
    # The telemetry streams were empty again and keep counting.
    assert processor._stream_empty_rounds["host-telemetry"] == 2
    assert processor._stream_idle_interval["host-telemetry"] == pytest.approx(0.4), (
        "the round that reached the threshold doubles 0.2 to 0.4, still under the cap"
    )


def test_a_notification_for_a_stream_resets_its_empty_rounds() -> None:
    processor = _processor(rounds=2)
    processor._pools_busy = True
    _round(processor)
    _round(processor)
    assert processor._stream_empty_rounds["fault"] == 2

    with processor._state_lock:
        processor._notification_pending_streams.add("fault")
    processor._run_processor_cycle({}, {})

    assert "fault" not in processor._stream_empty_rounds
    assert processor._stream_empty_rounds["host-telemetry"] == 2, (
        "only the notified stream starts over"
    )
    with processor._state_lock:
        processor._notification_pending_streams.add("*")
    processor._run_processor_cycle({}, {})
    assert processor._stream_empty_rounds == {}


def test_ceiling_rule_covers_every_branch() -> None:
    quiet = BackoffCeilings(
        idle=2.0,
        busy=0.4,
        fault_idle=0.5,
        fault_busy=0.1,
        notification_fallback=5.0,
        notifications_enabled=False,
    )
    noisy = BackoffCeilings(**{**quiet.__dict__, "notifications_enabled": True})
    free = lambda: False  # noqa: E731
    blocked = lambda: True  # noqa: E731

    def unasked() -> bool:
        raise AssertionError("a poller never asks for the fault backlog")

    assert quiet.idle_ceiling("fault") == 0.5 and quiet.idle_ceiling("host") == 2.0
    # The fault stream's idle ceiling stays at fault idle: the queue trigger
    # announces only idle->busy edges. It stretches to the fallback only once
    # settled with no fault backlog (test_fault_stream_wake.py).
    assert noisy.idle_ceiling("fault") == 0.5 and noisy.idle_ceiling("host") == 5.0
    for stream, busy_ceiling in (("fault", 0.1), ("host", 0.4)):
        kwargs = {"stream": stream, "next_interval": 0.8}
        idle = quiet.idle_ceiling(stream)
        assert (
            empty_round_ceiling(
                quiet,
                settled_empty=False,
                pools_busy=True,
                lane_blocked=free,
                fault_backlog_empty=unasked,
                **kwargs,
            )
            == busy_ceiling
        ), "busy pools hold the busy ceiling until the stream settles empty"
        assert (
            empty_round_ceiling(
                quiet,
                settled_empty=True,
                pools_busy=True,
                lane_blocked=free,
                fault_backlog_empty=unasked,
                **kwargs,
            )
            == idle
        ), "settled empty: busy pools no longer hold it"
        assert (
            empty_round_ceiling(
                quiet,
                settled_empty=True,
                pools_busy=True,
                lane_blocked=blocked,
                fault_backlog_empty=unasked,
                **kwargs,
            )
            == busy_ceiling
        ), (
            "a lane-blocked backlog keeps the busy ceiling: no edge announces a freed lane"
        )
        assert (
            empty_round_ceiling(
                quiet,
                settled_empty=False,
                pools_busy=False,
                lane_blocked=free,
                fault_backlog_empty=unasked,
                **kwargs,
            )
            == idle
        )
    assert (
        empty_round_ceiling(
            quiet,
            stream="host",
            next_interval=0.2,
            settled_empty=False,
            pools_busy=True,
            lane_blocked=blocked,
            fault_backlog_empty=unasked,
        )
        == 2.0
    ), "an interval still under the busy ceiling is not capped by it"


def test_settings_validate_and_read_the_environment(monkeypatch) -> None:
    assert ProcessorLeaseSettings().busy_empty_rounds_before_idle == 8
    with pytest.raises(ValueError, match="busy empty rounds before idle"):
        ProcessorLeaseSettings(busy_empty_rounds_before_idle=0)
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_BUSY_EMPTY_ROUNDS_BEFORE_IDLE", "3")
    assert ProcessorLeaseSettings.from_environment().busy_empty_rounds_before_idle == 3
