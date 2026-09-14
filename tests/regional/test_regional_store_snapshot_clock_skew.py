"""The regional store read must widen its time bound by the kmsg clock skew.

A ``/dev/kmsg`` line is stamped on the kernel's boot-time base, so its
``observed_at`` lands a little before the ``datetime.now()`` a runner takes
just before the write. ``store.list_xid_events`` filters on that kernel
timestamp, so an exact ``observed_after`` cut drops the freshly injected event
and ``wait_for_workflow`` waits its whole budget for a decision the control
plane already recorded (COLLECT-016 a-restart, 2026-09-14). The collector
fixture already widens the bound with ``marker_observed_after``; the regional
fixture must do the same for a marker-tagged read.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from scripts.e2e.regional.kmsg_clock import (
    KMSG_CLOCK_SKEW_SECONDS,
    marker_observed_after,
)
from scripts.e2e.regional.regional_live_fixture import (
    RegionalLiveFixture,
    RegionalLiveSettings,
)


def _fixture(tmp_path: Path) -> RegionalLiveFixture:
    cpu = tmp_path / "cpu.kubeconfig"
    gpu = tmp_path / "gpu.kubeconfig"
    cpu.write_text("apiVersion: v1\n")
    gpu.write_text("apiVersion: v1\n")
    settings = RegionalLiveSettings(
        cpu_kubeconfig=cpu,
        gpu_kubeconfig=gpu,
        gpu_context="ctx",
        namespace="gpu-fault-system",
        cluster_id="hp-cluster",
        region="us-west-2",
    )
    return RegionalLiveFixture(settings)


def _capture_bound(fixture: RegionalLiveFixture, marker: str, observed_after: datetime):
    seen: dict[str, str] = {}

    def fake_cpu_python(script: str, *arguments: str) -> dict[str, str]:
        # arguments = (cluster_id, node, marker, observed_after, job_id, ...)
        seen["observed_after"] = arguments[3]
        return {"release_id": "test-release"}

    fixture.cpu_python = fake_cpu_python  # type: ignore[method-assign]
    fixture.store_snapshot(
        node="hyperpod-i-node",
        marker=marker,
        observed_after=observed_after,
        queue_attempts=1,
    )
    return seen["observed_after"]


def test_marker_read_widens_observed_after_by_the_clock_skew(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    injected = datetime(2026, 9, 14, 22, 55, 11, tzinfo=timezone.utc)
    sent = _capture_bound(fixture, "c016-a-1789426511", injected)
    expected = marker_observed_after("c016-a-1789426511", injected)
    assert expected is not None
    assert sent == expected.isoformat()
    # And the widening is exactly the documented skew, backwards in time.
    assert (injected - datetime.fromisoformat(sent)).total_seconds() == (
        KMSG_CLOCK_SKEW_SECONDS
    )


def test_markerless_read_sends_the_exact_bound(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    injected = datetime(2026, 9, 14, 22, 55, 11, tzinfo=timezone.utc)
    sent = _capture_bound(fixture, "", injected)
    # No marker means the bound is a hard cut, not a marker scan limit.
    assert sent == injected.isoformat()
