"""Jitter on the node-side posting cadences, and the thresholds that must cover it.

The stable per-node phase spreads a fleet across each interval; the jitter
adds the random part (nodes sharing a phase stop posting in step, a fleet
restarted together does not put its first posts into the same 15 s) within
bounds the readiness thresholds are proven to cover.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.collector_kinds import (
    COLLECTOR_KINDS,
    collector_posting_cadences,
    collector_silent_thresholds,
    silent_threshold_shortfalls,
)
from gpu_fault.collectors.cadence import (
    DEFAULT_JITTER_RATIO,
    DEFAULT_STARTUP_JITTER_SECONDS,
    CadenceJitter,
    jittered_next_phase,
    worst_case_silence_seconds,
)
from gpu_fault.collectors.scheduling import next_stable_phase
from gpu_fault.collectors.sinks import HttpEventSink
from gpu_fault.telemetry import CollectorKind

NOW = datetime(2030, 1, 1, 12, 0, tzinfo=timezone.utc)
#: The collectors' deterministic start-up spread (GPU_FAULT_COLLECTOR_STARTUP_SPREAD_SECONDS).
STARTUP_SPREAD_SECONDS = 15.0


def test_defaults_and_environment_validation() -> None:
    jitter = CadenceJitter.from_environment({})
    assert jitter == CadenceJitter(
        ratio=DEFAULT_JITTER_RATIO, startup_seconds=DEFAULT_STARTUP_JITTER_SECONDS
    )
    assert (jitter.ratio, jitter.startup_seconds) == (0.1, 20.0)
    custom = CadenceJitter.from_environment(
        {
            "GPU_FAULT_COLLECTOR_CADENCE_JITTER_RATIO": "0.2",
            "GPU_FAULT_COLLECTOR_STARTUP_JITTER_SECONDS": "0",
        }
    )
    assert (custom.ratio, custom.startup_seconds) == (0.2, 0.0)
    for name, value in (
        ("GPU_FAULT_COLLECTOR_CADENCE_JITTER_RATIO", "-0.1"),
        ("GPU_FAULT_COLLECTOR_CADENCE_JITTER_RATIO", "0.6"),
        ("GPU_FAULT_COLLECTOR_STARTUP_JITTER_SECONDS", "-1"),
    ):
        with pytest.raises(ValueError):
            CadenceJitter.from_environment({name: value})


def test_intervals_and_offsets_stay_within_the_ratio() -> None:
    jitter = CadenceJitter(ratio=0.1, startup_seconds=20.0)
    for _ in range(500):
        wait = jitter.interval(300.0)
        assert 270.0 <= wait <= 330.0
        assert -30.0 <= jitter.offset(300.0) <= 30.0
        assert 0.0 <= jitter.startup_offset() <= 20.0
    assert jitter.max_interval(300.0) == pytest.approx(330.0)
    # The extremes of the uniform draw are honoured, nothing past them.
    assert jitter.interval(300.0, uniform=lambda low, high: high) == pytest.approx(
        330.0
    )
    assert jitter.interval(300.0, uniform=lambda low, high: low) == pytest.approx(270.0)
    assert jitter.interval(300.0, uniform=lambda low, high: 10 * high) == 330.0
    assert jitter.startup_offset(uniform=lambda low, high: -5.0) == 0.0


def test_zero_ratio_and_zero_startup_jitter_switch_the_randomness_off() -> None:
    still = CadenceJitter(ratio=0.0, startup_seconds=0.0)
    assert still.interval(300.0) == 300.0
    assert still.offset(300.0) == 0.0
    assert still.startup_offset() == 0.0
    scheduled = jittered_next_phase(
        NOW,
        cluster_id="c",
        node_id="n",
        channel="gpu-metrics",
        interval_seconds=300,
        jitter=still,
    )
    assert scheduled == next_stable_phase(
        NOW, cluster_id="c", node_id="n", channel="gpu-metrics", interval_seconds=300
    )


def test_a_jittered_phase_moves_within_the_ratio_and_never_into_the_past() -> None:
    jitter = CadenceJitter(ratio=0.1, startup_seconds=0.0)
    anchor = next_stable_phase(
        NOW, cluster_id="c", node_id="n", channel="gpu-metrics", interval_seconds=300
    )
    for _ in range(200):
        moved = jittered_next_phase(
            NOW,
            cluster_id="c",
            node_id="n",
            channel="gpu-metrics",
            interval_seconds=300,
            jitter=jitter,
        )
        assert abs((moved - anchor).total_seconds()) <= 30.0
        assert moved >= NOW
    # A phase ten seconds away with the maximum negative offset (-30 s) would
    # land in the past; it is clamped to the observation instead.
    late = anchor - timedelta(seconds=10)
    clamped = jittered_next_phase(
        late,
        cluster_id="c",
        node_id="n",
        channel="gpu-metrics",
        interval_seconds=300,
        jitter=jitter,
        uniform=lambda low, high: low,
    )
    assert clamped == late


def _retry_ladder_seconds() -> float:
    """The sink's worst case: every attempt times out, every backoff draws its maximum."""

    sink = HttpEventSink("http://control.test")
    attempts = sink.max_attempts
    return attempts * sink.timeout_seconds + sum(
        min(2 ** (attempt - 1), 8) for attempt in range(1, attempts)
    )


def test_every_silence_threshold_covers_the_jittered_cadence() -> None:
    """A node that is merely late must not be reported silent (readiness,
    collector-silent notifications): threshold >= one jittered interval at its
    longest + the start-up spread and its random part + the retry ladder."""

    jitter = CadenceJitter.from_environment({})
    ladder = _retry_ladder_seconds()
    assert ladder == pytest.approx(47.0), "4 x 10 s timeouts + 1 + 2 + 4 s backoff"
    shortfalls = silent_threshold_shortfalls(
        lambda cadence: worst_case_silence_seconds(
            cadence,
            jitter=jitter,
            startup_spread_seconds=STARTUP_SPREAD_SECONDS,
            delivery_ladder_seconds=ladder,
        )
    )
    assert shortfalls == {}, (
        "raise the GPU_FAULT_*_SILENT_AFTER_SECONDS default or lower the jitter: "
        f"{shortfalls}"
    )
    cadences = collector_posting_cadences()
    assert cadences[CollectorKind.GPU_INVENTORY] == 60.0
    assert all(
        cadences[kind] == 300.0
        for kind in (
            CollectorKind.GPU_METRICS,
            CollectorKind.HOST_TELEMETRY,
            CollectorKind.NODE_LOGS,
            CollectorKind.NVIDIA_KERNEL,
            CollectorKind.FABRIC_MANAGER_LOG,
        )
    )
    assert set(cadences) == set(collector_silent_thresholds()), (
        "every scheduled kind declares its cadence next to its threshold"
    )
    for kind, spec in COLLECTOR_KINDS.items():
        if spec.silent_threshold is not None:
            assert spec.cadence is not None, kind
            assert spec.cadence.env.startswith("GPU_FAULT_"), kind


def test_shortfalls_name_the_threshold_and_the_gap_it_misses() -> None:
    shortfalls = silent_threshold_shortfalls(
        lambda cadence: cadence * 3, thresholds={CollectorKind.GPU_INVENTORY: 100.0}
    )
    assert shortfalls == {CollectorKind.GPU_INVENTORY: (100.0, 180.0)}
    assert (
        silent_threshold_shortfalls(
            lambda cadence: cadence, thresholds={CollectorKind.GPU_INVENTORY: 100.0}
        )
        == {}
    )
