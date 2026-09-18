"""Marker reads widen the scan bound; markerless reads keep the exact bound."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from scripts.e2e.regional.kmsg_clock import (
    KMSG_CLOCK_SKEW_SECONDS,
    marker_observed_after,
)
from scripts.e2e.regional.regional_live_fixture import (
    RegionalLiveFixture,
    RegionalLiveSettings,
)


def fixture(tmp_path: Path) -> RegionalLiveFixture:
    cpu = tmp_path / "cpu.kubeconfig"
    gpu = tmp_path / "gpu.kubeconfig"
    cpu.write_text("apiVersion: v1\n", encoding="utf-8")
    gpu.write_text("apiVersion: v1\n", encoding="utf-8")
    return RegionalLiveFixture(
        RegionalLiveSettings(
            cpu_kubeconfig=cpu,
            gpu_kubeconfig=gpu,
            gpu_context="gpu-context",
            namespace="gpu-fault-system",
            cluster_id="cluster-a",
            region="us-west-2",
        )
    )


def capture_bound(
    regional: RegionalLiveFixture,
    monkeypatch: pytest.MonkeyPatch,
    marker: str,
    observed_after: datetime | None,
) -> str:
    seen: list[str] = []

    def cpu_python(_script: str, *arguments: str) -> dict[str, str]:
        seen.append(arguments[3])
        return {"release_id": "test-release"}

    monkeypatch.setattr(regional, "cpu_python", cpu_python)
    regional.store_snapshot(
        node="node-a", marker=marker, observed_after=observed_after, queue_attempts=1
    )
    assert len(seen) == 1, "one snapshot must read the Store once"
    return seen[0]


def test_marker_read_widens_observed_after_by_the_clock_skew(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    injected = datetime(2026, 9, 14, 22, 55, 11, tzinfo=timezone.utc)
    sent = capture_bound(fixture(tmp_path), monkeypatch, "marker-a", injected)
    expected = marker_observed_after("marker-a", injected)
    assert expected is not None
    assert sent == expected.isoformat(), "the shared marker clock contract applies"
    assert (injected - datetime.fromisoformat(sent)).total_seconds() == (
        KMSG_CLOCK_SKEW_SECONDS
    ), "the scan must widen exactly once, backwards"


def test_markerless_read_sends_the_exact_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    injected = datetime(2026, 9, 14, 22, 55, 11, tzinfo=timezone.utc)
    sent = capture_bound(fixture(tmp_path), monkeypatch, "", injected)
    assert sent == injected.isoformat(), "markerless time bounds remain hard bounds"


@pytest.mark.parametrize("marker", ["", "marker-a"])
def test_missing_time_bound_is_not_invented(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, marker: str
) -> None:
    assert capture_bound(fixture(tmp_path), monkeypatch, marker, None) == "", (
        "an omitted bound must keep the existing unbounded protocol"
    )
