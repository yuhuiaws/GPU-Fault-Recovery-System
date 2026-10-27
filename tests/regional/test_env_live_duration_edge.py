"""GF-ENV-LIVE-006: the clock-cleared read must repeat the confirmed duration.

Live 2026-10-27 (the HyperPod fault node, release R13g) the ``hw_thermal``
attempt FAILed: both read-only diagnostic workflows ended FAILED
with VALIDATE_GPU "active_gpu_health_findings[thermal_violation_total_us
WARNING 3e+06>10000]" exactly GPU_FAULT_TEMPERATURE_WARNING_GRACE_SECONDS
(120 s) after their creation, and the product sent two diagnostic-inconclusive
notices -- its designed answer to a thermal WARNING still active at the grace
(``adapters/gpu_validation._validate_active_findings``,
``execution/terminal_state.terminal_decision``). The plan served four rising
reads (0, 1e6, 2e6, 3e6 us under 0x40) and a flat tail (3e6 under 0). The DCGM
edge filter delivered the third read (the clock reason confirmed on three
samples: 2e6), suppressed the fourth (a confirmed candidate is no new edge and
a violation counter is judged by duty cycle, not by change) and delivered the
first tail read as ``candidate-recovered`` carrying 3e6 -- +1e6 over the
delivered 2e6, which the ingest graded as a fresh WARNING
(``thermal_duration_decision``: delta >= 10000 us at 2e6 us/min). Nothing
could clear it before the grace: the flat tail is no edge, and the
pass-through real counter (0) is a reset the ingest refuses to grade (an
invalid delta keeps the finding); the health summary that landed on the
HyperPod node's stable phase (01:49:47Z) carried that reset. Plain EKS had
passed the same plan the day before, so one input graded two ways: the plan
may not leave the clearing to chance.

These tests drive both plan shapes through the real collector delivery filter
and the real ingest. With the shipped shape (one rising read per confirmation
sample, the tail at the confirmed read's duration) the cleared edge is a zero
delta and clears both findings, so VALIDATE_GPU's active-findings read finds
none; with the old shape the thermal finding is re-minted on the cleared edge
and survives the pass-through reset. No product code changes with this module.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.gpu_metrics import GpuMetricBatch, GpuMetricsService
from scripts.e2e.regional import env_live_plans as plans
from tests.collectors._support import DcgmMetricsCollector, RecordingSink, context

NOW = datetime(2026, 10, 27, 1, 47, tzinfo=timezone.utc)
#: The cluster of ``tests.collectors._support.context()``.
CLUSTER = "hp-cluster"
NODE = "worker-1"
GPU = "GPU-1a588e85-965e-fc68-40e4-225aaa3315b6"
BOOT = "3dad2d1c-6eb1-4958-9991-11b4f9c724b0"
CLOCK = "clock_throttle_reasons"
THERMAL = "thermal_violation_total_us"
#: The real exporter's GPU_IDLE bit: not a candidate, so pass-through reads
#: before the arm and after the plan are not edges.
IDLE = 1
PASSTHROUGH: list[tuple[int, float]] = [(IDLE, 0.0), (IDLE, 0.0)]


def exporter_text(clock: int, thermal_us: float) -> str:
    """One scrape of an idle H200 whose clock reasons and thermal-violation
    duration are the plan's. The exporter declares the duration a ``counter``
    in nanoseconds (the parser reads it back by its bare name in microseconds,
    as the live collector did)."""

    labels = f'gpu="0",UUID="{GPU}",modelName="NVIDIA H200"'
    return "\n".join(
        [
            f"DCGM_FI_DEV_GPU_TEMP{{{labels}}} 30",
            f"DCGM_FI_DEV_ECC_DBE_VOL_TOTAL{{{labels}}} 0",
            f"DCGM_FI_DEV_ROW_REMAP_FAILURE{{{labels}}} 0",
            f"DCGM_FI_DEV_ROW_REMAP_PENDING{{{labels}}} 0",
            f"DCGM_FI_DEV_PCIE_REPLAY_COUNTER{{{labels}}} 0",
            f"DCGM_FI_DEV_NVLINK_ERROR_DL_CRC{{{labels}}} 0",
            f"DCGM_FI_DEV_NVLINK_ERROR_DL_RECOVERY{{{labels}}} 0",
            f"DCGM_FI_DEV_NVLINK_ERROR_DL_REPLAY{{{labels}}} 0",
            f"DCGM_FI_DEV_CLOCKS_EVENT_REASONS{{{labels}}} {clock}",
            f"DCGM_FI_DEV_POWER_MGMT_LIMIT{{{labels}}} 700",
            "# TYPE DCGM_FI_DEV_THERMAL_VIOLATION counter",
            f"DCGM_FI_DEV_THERMAL_VIOLATION{{{labels}}} {thermal_us * 1000:.0f}",
            "",
        ]
    )


def plan_reads(plan: dict[str, Any]) -> list[tuple[int, float]]:
    """The (clock mask, duration us) the proxy serves per collector read."""

    return [
        (int(step["values"][CLOCK]), float(step["values"][THERMAL]))
        for step in plan["steps"]
        for _ in range(int(step["reads"]))
    ]


def deliver(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    reads: list[tuple[int, float]],
    *,
    times: list[int],
) -> list[GpuMetricBatch]:
    """What the real collector ships for ``reads`` scraped at ``times``."""

    path = tmp_path / "boot"
    path.write_text(BOOT + "\n", encoding="ascii")
    monkeypatch.setenv("GPU_FAULT_BOOT_ID_PATH", str(path))
    sink = RecordingSink()
    clock = iter(NOW + timedelta(seconds=at) for at in times)
    dcgm = DcgmMetricsCollector(
        sink,
        context(),
        node_id=NODE,
        now=lambda: next(clock),
        health_summary_seconds=300,
        edge_confirmation_samples=3,
    )
    for mask, thermal_us in reads:
        dcgm.collect_text(exporter_text(mask, thermal_us))
    return [GpuMetricBatch.model_validate(payload) for _path, payload in sink.requests]


def served(batch: GpuMetricBatch, name: str) -> float:
    return next(
        sample.value
        for sample in batch.samples
        if sample.canonical_name == name and sample.gpu_uuid == GPU
    )


def shipped(batches: list[GpuMetricBatch]) -> list[tuple[list[str], float, float]]:
    return [
        (batch.edge_filter_reasons, served(batch, CLOCK), served(batch, THERMAL))
        for batch in batches
    ]


def active(service: GpuMetricsService) -> dict[str, Any]:
    """What ``_validate_active_findings`` reads for the GPU on its next attempt."""

    return {
        finding.canonical_name: finding
        for finding in service.findings(CLUSTER, NODE)
        if finding.gpu_uuid == GPU
    }


def test_the_cleared_read_is_a_zero_delta_and_clears_both_findings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan = plans.thermal_duration_plan(confirmation_samples=3)["hw_thermal"]
    reads = [*PASSTHROUGH, *plan_reads(plan), (IDLE, 0.0), (IDLE, 0.0)]
    batches = deliver(
        monkeypatch, tmp_path, reads, times=[15 * index for index in range(len(reads))]
    )
    assert not any(batch.collection_errors for batch in batches), (
        "every scrape carries the core fields"
    )
    assert shipped(batches) == [
        (["initial-baseline"], IDLE, 0.0),
        (["candidate-confirmed"], plans.HW_THERMAL_BIT, 2_000_000.0),
        (["candidate-recovered"], 0, 2_000_000.0),
    ], "the filter ships the confirmation and the first cleared read, nothing else"

    service = GpuMetricsService()
    baseline, confirmed, cleared = [service.ingest(batch) for batch in batches]

    assert not baseline.findings, "the pass-through baseline breaches nothing"
    assert sorted(item.canonical_name for item in confirmed.new_findings) == [
        CLOCK,
        THERMAL,
    ], "the confirmed read mints the clock-reason and duration findings"
    (duration,) = [
        item for item in confirmed.findings if item.canonical_name == THERMAL
    ]
    assert duration.delta == 2_000_000.0 and duration.severity.value == "WARNING"
    assert not cleared.findings and not cleared.new_findings, (
        "a zero delta under a cleared reason breaches nothing"
    )
    assert active(service) == {}, (
        "both findings are cleared by the batch VALIDATE_GPU re-checks on"
    )


def test_a_rise_after_the_confirmation_surfaces_on_the_cleared_read_and_stays(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The live four-read ramp, then the health summary with the real counter."""

    live = {
        "steps": [
            *(
                {"reads": 1, "values": {THERMAL: index * 1_000_000.0, CLOCK: 0x40}}
                for index in range(4)
            ),
            {"reads": 3, "values": {THERMAL: 3_000_000.0, CLOCK: 0}},
        ]
    }
    reads = [*PASSTHROUGH, *plan_reads(live), (IDLE, 0.0)]
    # The last read is the pass-through real counter one health summary later
    # (``jittered_next_phase`` is at most 330 s past the last delivery).
    times = [15 * index for index in range(len(reads) - 1)]
    times.append(times[-1] + 420)
    batches = deliver(monkeypatch, tmp_path, reads, times=times)
    assert shipped(batches) == [
        (["initial-baseline"], IDLE, 0.0),
        (["candidate-confirmed"], 0x40, 2_000_000.0),
        (["candidate-recovered"], 0, 3_000_000.0),
        (["health-summary"], IDLE, 0.0),
    ], "the fourth rising read is no edge: its value rides on the cleared read"

    service = GpuMetricsService()
    service.ingest(batches[0])
    service.ingest(batches[1])
    recovered = service.ingest(batches[2])

    (duration,) = [
        item for item in recovered.findings if item.canonical_name == THERMAL
    ]
    assert duration.value == 3_000_000.0 and duration.delta == 1_000_000.0
    assert duration.severity.value == "WARNING" and 1.9e6 < duration.rate_per_minute
    assert not recovered.new_findings, (
        "same severity and action: the finding is replaced, no new incident"
    )
    assert set(active(service)) == {THERMAL}, (
        "the cleared read re-minted the duration finding the clock clear left"
    )

    summary = service.ingest(batches[3])

    assert not summary.findings and not summary.new_findings
    assert set(active(service)) == {THERMAL}, (
        "the pass-through counter is a reset: an invalid delta keeps the finding, "
        "so VALIDATE_GPU fails on it at the grace"
    )
