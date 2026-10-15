"""COLLECT-001's jitter-aware verdict, on delivery stamps the live case recorded.

The plain-EKS run that failed delivered nothing but pure health summaries (no raw
evidence in the window), yet both chains posted in bursts one or two collection
ticks apart: a summary that went out before its stable phase re-armed onto that
same phase. The verdict now names that burst instead of reading it as an
unsuppressed chain, and its window and gap bounds follow the cadence jitter.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import collector_cadence_evidence as cadence
from scripts.e2e.regional import run_collector_acceptance as acceptance

CHAINS = {
    "GPU_METRICS": {"interval": 15, "summary": 300},
    "HOST_TELEMETRY": {"interval": 15, "summary": 300},
}
# The failed run's in-window collector status stamps (stable phases :16 and :34).
LIVE_BURSTS = {
    "GPU_METRICS": [
        "2026-10-15T14:56:11.911382+00:00",
        "2026-10-15T14:56:43.200630+00:00",
        "2026-10-15T15:01:03.654445+00:00",
        "2026-10-15T15:01:18.689703+00:00",
    ],
    "HOST_TELEMETRY": [
        "2026-10-15T14:54:40.501802+00:00",
        "2026-10-15T14:59:16.620395+00:00",
        "2026-10-15T14:59:32.028315+00:00",
        "2026-10-15T14:59:47.428090+00:00",
        "2026-10-15T15:05:09.402224+00:00",
    ],
}
ORIGIN = datetime(2026, 10, 15, 14, 54, tzinfo=timezone.utc)


def _deliveries(offsets: list[float], prefix: str = "dcgm") -> dict[str, str]:
    return {
        (ORIGIN + timedelta(seconds=offset)).isoformat(): f"{prefix}-{index}"
        for index, offset in enumerate(offsets)
    }


def _verdict(
    deliveries: dict[str, str],
    *,
    ratio: float = 0.1,
    edge_reasons: dict[str, list[str]] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    return cadence.chain_verdict(
        "GPU_METRICS",
        deliveries,
        interval=15,
        summary=300,
        ratio=ratio,
        duration=cadence.observation_seconds(CHAINS, ratio),
        edge_reasons=edge_reasons or {},
    )


def test_ratio_zero_reduces_to_the_unjittered_window_and_gap() -> None:
    assert cadence.observation_seconds(CHAINS, 0.0) == 2 * 300 + 4 * 15
    assert cadence.summary_window(interval=15, summary=300, ratio=0.0) == 270
    observed, errors = _verdict(_deliveries([10, 310, 610]), ratio=0.0)
    assert errors == []
    assert observed["deliveries_allowed"] == 660 // 300 + 1


def test_the_collectors_default_jitter_sets_the_window_and_the_gap() -> None:
    ratio = cadence.jitter_ratio({"GPU_FAULT_METRICS_INTERVAL_SECONDS": "15"})
    assert ratio == 0.1, "collector.env never carries the ratio: the default"
    assert cadence.observation_seconds(CHAINS, ratio) == 2 * 330 + 4 * 15
    assert cadence.summary_window(
        interval=15, summary=300, ratio=ratio
    ) == pytest.approx(240.0)
    # The jittered extremes a healthy chain can produce: one summary 15 s early
    # and a tick late, the next 15 s early -- 255 s -- then 15 s late, 345 s.
    observed, errors = _verdict(_deliveries([30, 285, 630]))
    assert errors == [], errors
    assert observed["gaps_seconds"] == [255.0, 345.0]
    # Two deliveries 345 s apart in a 720 s window are still two.
    assert _verdict(_deliveries([5, 350]))[1] == []


def test_the_live_bursts_are_named_as_repeated_summaries() -> None:
    expected = {
        "GPU_METRICS": [
            "2026-10-15T14:56:43.200630+00:00 after 31.3 s",
            "2026-10-15T15:01:18.689703+00:00 after 15.0 s",
        ],
        "HOST_TELEMETRY": [
            "2026-10-15T14:59:32.028315+00:00 after 15.4 s",
            "2026-10-15T14:59:47.428090+00:00 after 15.4 s",
        ],
    }
    for kind, stamps in LIVE_BURSTS.items():
        deliveries = {stamp: f"{kind}-{index}" for index, stamp in enumerate(stamps)}
        _, errors = cadence.chain_verdict(
            kind,
            deliveries,
            interval=15,
            summary=300,
            ratio=0.1,
            duration=720,
            edge_reasons={},
        )
        assert (
            f"{kind} delivered its health summary again inside one 240 s summary "
            f"window: {expected[kind]}"
        ) in errors, errors


def test_an_edge_may_precede_a_summary_but_a_summary_may_not() -> None:
    deliveries = _deliveries([10, 300, 310])
    # An edge that was not itself the summary left the schedule alone, so the
    # summary it was followed by is the one that was due.
    edge = {"dcgm-1": ["candidate-confirmed"]}
    assert _verdict(deliveries, edge_reasons=edge)[1] == []
    # The edge rode on the summary: the next pure summary repeated it.
    riding = {"dcgm-1": ["candidate-confirmed", "health-summary"]}
    errors = _verdict(deliveries, edge_reasons=riding)[1]
    assert any("again inside one" in error for error in errors), errors
    # An edge right after a summary is the node's to report.
    assert _verdict(deliveries, edge_reasons={"dcgm-2": ["xid-changed"]})[1] == []


def test_the_case_hands_batch_ids_and_evidence_reasons_to_the_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Clock:
        now = 1000.0

        def monotonic(self) -> float:
            return self.now

        def sleep(self, seconds: float) -> None:
            self.now += seconds

    monkeypatch.setattr(acceptance, "time", Clock())
    start = datetime.now(timezone.utc)
    statuses = [
        {
            "collector": collector,
            "batch_id": f"{prefix}{offset}",
            "observed_at": (start + timedelta(seconds=offset)).isoformat(),
        }
        for collector, prefix in acceptance.COLLECT001_PRODUCERS.items()
        for offset in (60, 75, 360)
    ]
    evidence = [{"payload": {"batch_id": "host-75", "edge_filter_reasons": ["x"]}}]
    monkeypatch.setattr(acceptance, "collector_statuses", lambda *_, **__: statuses)
    monkeypatch.setattr(acceptance, "recent_evidence", lambda *_, **__: evidence)
    env = {
        "GPU_FAULT_METRICS_INTERVAL_SECONDS": "15",
        "GPU_FAULT_DCGM_HEALTH_SUMMARY_SECONDS": "300",
        "GPU_FAULT_HOST_INTERVAL_SECONDS": "15",
        "GPU_FAULT_HOST_HEALTH_SUMMARY_SECONDS": "300",
    }
    fixture = SimpleNamespace(
        snapshot=lambda: {"collector_env": env}, regional=object(), node="n"
    )

    result = acceptance.run_collect001(fixture, tmp_path)  # type: ignore[arg-type]

    assert result["observation_seconds"] == 720
    assert result["deliveries"]["GPU_METRICS"]["jitter_ratio"] == 0.1
    assert [error.split(" delivered")[0] for error in result["errors"]] == [
        "GPU_METRICS"
    ], "the host batch 15 s later was an edge (evidence), the GPU one a repeat"
