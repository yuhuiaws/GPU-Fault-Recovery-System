"""Jitter on the node-side posting cadences, and the thresholds that must cover it.

The stable per-node phase spreads a fleet across each interval; the jitter
adds the random part (nodes sharing a phase stop posting in step, a fleet
restarted together does not put its first posts into the same 15 s) within
bounds the readiness thresholds are proven to cover.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any

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
from gpu_fault.collectors.scheduling import next_stable_phase, stable_phase_seconds
from gpu_fault.collectors.sinks import HttpEventSink
from gpu_fault.telemetry import CollectorKind

from ._support import (
    DcgmMetricsCollector,
    HostTelemetryCollector,
    RecordingSink,
    context,
)

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


def _next_phase(
    observed_at: datetime,
    jitter: CadenceJitter,
    uniform: Callable[[float, float], float] = random.uniform,
    channel: str = "gpu-metrics",
) -> datetime:
    return jittered_next_phase(
        observed_at,
        cluster_id="c",
        node_id="n",
        channel=channel,
        interval_seconds=300,
        jitter=jitter,
        uniform=uniform,
    )


def test_a_jittered_phase_moves_by_half_the_ratio_and_never_into_the_past() -> None:
    jitter = CadenceJitter(ratio=0.1, startup_seconds=0.0)
    anchor = next_stable_phase(
        NOW, cluster_id="c", node_id="n", channel="gpu-metrics", interval_seconds=300
    )
    assert (anchor - NOW).total_seconds() > 15.0, "NOW serves no phase"
    for _ in range(200):
        moved = _next_phase(NOW, jitter)
        assert abs((moved - anchor).total_seconds()) <= 15.0
        assert moved > NOW
    # A post up to half a spread early, or late by a tick, served `anchor`; the
    # next one belongs to the following phase whichever way the draw goes.
    # Looking up from the post itself re-armed onto `anchor` (a few seconds
    # ahead, or clamped to the post) and summarised twice (COLLECT-001).
    following = anchor + timedelta(seconds=300)
    for lead in (15.0, 10.0, 0.5, 0.0, -30.0):
        served_at = anchor - timedelta(seconds=lead)
        for draw, offset in ((lambda low, high: low, -15.0), (max, 15.0)):
            moved = _next_phase(served_at, jitter, draw)
            assert (moved - following).total_seconds() == pytest.approx(offset)


@pytest.mark.parametrize("tick", [15.0, 15.4])
@pytest.mark.parametrize("draw", ["lowest", "highest", "seeded"])
def test_a_tick_loop_posts_once_per_stable_phase_within_the_spacing(
    draw: str, tick: float
) -> None:
    """Every phase is posted exactly once and consecutive posts stay within
    interval x (1 +- ratio) plus the tick that notices them, which is the bound
    the silence thresholds are proven against. Before the fix the lowest draw
    posted every phase twice and a seeded run doubled about every other one."""

    jitter = CadenceJitter(ratio=0.1, startup_seconds=0.0)
    uniform = {
        "lowest": lambda low, high: low,
        "highest": max,
        "seeded": random.Random(65).uniform,
    }[draw]
    phase = stable_phase_seconds("c", "n", "host-telemetry", 300)
    due: datetime | None = None
    posts: list[datetime] = []
    for index in range(int(3 * 3600 / tick)):
        observed = NOW + timedelta(seconds=index * tick)
        if due is None or observed >= due:
            posts.append(observed)
            due = _next_phase(observed, jitter, uniform, channel="host-telemetry")
    scheduled = posts[1:]
    served = [round((post.timestamp() - phase) / 300) for post in scheduled]
    assert served == list(range(served[0], served[0] + len(served)))
    gaps = [(b - a).total_seconds() for a, b in zip(scheduled, scheduled[1:])]
    assert all(270.0 - tick <= gap <= 330.0 + tick for gap in gaps), gaps


def _lowest_jitter_draw(monkeypatch: pytest.MonkeyPatch) -> None:
    """The collectors' own jitter at 0.1, every draw at its most negative: each
    summary fires before its stable phase, the case the re-arm doubled."""

    monkeypatch.setenv("GPU_FAULT_COLLECTOR_CADENCE_JITTER_RATIO", "0.1")
    offset = CadenceJitter.offset

    def lowest(
        self: CadenceJitter,
        interval_seconds: float,
        uniform: Callable[[float, float], float] = random.uniform,
    ) -> float:
        return offset(self, interval_seconds, lambda low, high: low)

    monkeypatch.setattr(CadenceJitter, "offset", lowest)


def _summary_gaps(requests: list[tuple[str, dict[str, Any]]]) -> list[float]:
    stamps = [
        datetime.fromisoformat(payload["observed_at"])
        for _, payload in requests
        if payload["edge_filter_reasons"] == ["health-summary"]
    ]
    return [(b - a).total_seconds() for a, b in zip(stamps, stamps[1:])]


def test_the_dcgm_edge_filter_summarises_once_per_phase_with_the_jitter_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """COLLECT-001 on the live site: a healthy node's GPU_METRICS went out in
    pairs one or two ticks apart because each early summary re-armed."""

    _lowest_jitter_draw(monkeypatch)
    sink = RecordingSink()
    ticks = iter(NOW + timedelta(seconds=15 * index) for index in range(80))
    collector = DcgmMetricsCollector(
        sink,
        context(),
        node_id="worker-1",
        now=lambda: next(ticks),
        health_summary_seconds=300,
    )
    text = (
        'DCGM_FI_DEV_GPU_TEMP{gpu="0",UUID="GPU-a"} 70\n'
        'DCGM_FI_DEV_ECC_DBE_VOL_TOTAL{gpu="0",UUID="GPU-a"} 0\n'
    )
    for _ in range(80):
        collector.collect_text(text)

    reasons = [payload["edge_filter_reasons"] for _, payload in sink.requests]
    assert reasons[0] == ["initial-baseline"]
    assert set(map(tuple, reasons[1:])) == {("health-summary",)}
    assert _summary_gaps(sink.requests) == [300.0] * (len(reasons) - 2)
    assert len(reasons) >= 4


def test_the_host_edge_filter_summarises_once_per_phase_with_the_jitter_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _lowest_jitter_draw(monkeypatch)
    sink = RecordingSink()
    ticks = iter(NOW + timedelta(seconds=15 * index) for index in range(80))
    collector = HostTelemetryCollector(
        sink,
        context(),
        node_id="worker-1",
        now=lambda: next(ticks),
        health_summary_seconds=300,
    )
    for name in HostTelemetryCollector.CONTRIBUTORS:
        monkeypatch.setattr(collector, name, lambda _observed_at: [])
    for _ in range(80):
        collector.collect_once()

    reasons = [payload["edge_filter_reasons"] for _, payload in sink.requests]
    assert reasons[0] == ["baseline"]
    assert set(map(tuple, reasons[1:])) == {("health-summary",)}
    assert _summary_gaps(sink.requests) == [300.0] * (len(reasons) - 2)
    assert len(reasons) >= 4


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
    ), "the health summaries post every 300 s by default"
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
