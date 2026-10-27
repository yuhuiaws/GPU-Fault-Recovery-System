"""GF-GENERAL-LIVE-024: the cleared read of both ramp plans is a zero delta.

Live 2026-10-26 (both clusters, R13g) ``thermal_warning`` and
``power_performance`` PASSed with the former plan shape -- the confirmation
samples plus three rising reads, the tail holding the last of them -- and the
product accepted exactly two breach batches each. The second batch is the
violation counter's own edge: at 1e6 us per 15 s read its duty cycle (0.067)
is over the collector's 0.05 threshold, so the counter confirmed one read after
the warning-band temperature / the draw-at-the-cap correlation (read 7, 4e6)
and the cleared read (read 10) carried 6e6 -- a rise of 2e6 on the
``candidate-recovered`` batch, the input GF-ENV-LIVE-006 FAILed on the HyperPod
fault node the next day (``thermal_duration_decision`` re-mints the WARNING,
the flat tail is no edge, the pass-through counter is a reset the ingest keeps
the finding through; plain EKS had cleared the identical delivery the day
before). The power plan's rise passed only because
``power_violation_decision`` reads the draw and utilization of the same batch
and the tail lets the idle GPU's real draw through.

These tests drive both shapes through the real collector delivery filter and
the real ingest with its composites. The shipped shape delivers the confirmed
read and one cleared read that is a zero delta: the component findings and the
composite clear on the batch VALIDATE_GPU re-checks on. The former shape
re-mints the thermal duration WARNING on the cleared read and keeps it through
the pass-through reset. No product code changes with this module.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.gpu_metrics import (
    GpuMetricBatch,
    GpuMetricSample,
    GpuMetricsService,
    GpuMetricsThresholds,
)
from gpu_fault.gpu_power_policy import power_violation_decision
from scripts.e2e.regional import env_live_plans as plans
from scripts.e2e.regional.env_live_expectations import served_reads
from tests.collectors._support import DcgmMetricsCollector, RecordingSink, context

#: Live plain-EKS ``thermal_warning`` arm 2026-10-26 21:51Z; the node's next
#: health-summary phase is 295 s (+-15 s jitter) past the first delivery, so
#: no summary lands inside a 16-read plan.
NOW = datetime(2026, 10, 26, 21, 46, 30, tzinfo=timezone.utc)
#: The cluster of ``tests.collectors._support.context()``.
CLUSTER = "hp-cluster"
NODE = "worker-1"
GPU = "GPU-053ab270-714e-0f16-46ad-f96f00134d8a"
BOOT = "b39c4e22-0873-4642-ba92-895dfd9e76ef"
TLIMIT = "gpu_tlimit_c"
CLOCK = "clock_throttle_reasons"
THERMAL = "thermal_violation_total_us"
POWER = "power_violation_total_us"
DRAW = "power_usage_w"
UTIL = "gpu_utilization_percent"
#: The real exporter's GPU_IDLE bit: not a candidate.
IDLE = 1
#: What the idle live H200 really reports (pass-through reads and the fields a
#: plan does not serve).
REAL: dict[str, float] = {
    TLIMIT: 59.0,
    CLOCK: IDLE,
    THERMAL: 0.0,
    DRAW: 70.0,
    UTIL: 0.0,
    POWER: 0.0,
}
# Live p5en H200 (2026-10-24): only the T.Limit scale is published.
H200_LIMITS = {
    "gpu_tlimit_c": 59.0,
    "gpu_shutdown_tlimit_c": -8.0,
    "gpu_slowdown_tlimit_c": -2.0,
    "gpu_max_operating_tlimit_c": 0.0,
    "memory_max_operating_tlimit_c": 0.0,
}
#: The cached ``nvidia-smi -q -x`` thresholds the collector adds to every scrape.
LIMITS = [
    GpuMetricSample(
        metric_name=f"nvidia_smi_{name}",
        canonical_name=name,
        value=value,
        unit="celsius",
        gpu_index="0",
        gpu_uuid=GPU,
        labels={"gpu": "0", "threshold_source": "nvidia-smi-xml", "UUID": GPU},
    )
    for name, value in H200_LIMITS.items()
    if name != "gpu_tlimit_c"
]
WARNING = "WARNING"
DIAGNOSTIC = "RUN_DIAGNOSTICS"


def exporter_text(read: dict[str, float]) -> str:
    """One scrape of the H200 with the read's values laid over the idle GPU.

    The exporter declares the violation durations ``counter`` in nanoseconds
    (the parser reads them back by their bare names in microseconds).
    """

    values = {**REAL, **read}
    labels = f'gpu="0",UUID="{GPU}",modelName="NVIDIA H200"'
    return "\n".join(
        [
            f"DCGM_FI_DEV_GPU_TEMP{{{labels}}} 30",
            f"DCGM_FI_DEV_GPU_TEMP_LIMIT{{{labels}}} {values[TLIMIT]}",
            f"DCGM_FI_DEV_ECC_DBE_VOL_TOTAL{{{labels}}} 0",
            f"DCGM_FI_DEV_ROW_REMAP_FAILURE{{{labels}}} 0",
            f"DCGM_FI_DEV_ROW_REMAP_PENDING{{{labels}}} 0",
            f"DCGM_FI_DEV_PCIE_REPLAY_COUNTER{{{labels}}} 0",
            f"DCGM_FI_DEV_NVLINK_ERROR_DL_CRC{{{labels}}} 0",
            f"DCGM_FI_DEV_NVLINK_ERROR_DL_RECOVERY{{{labels}}} 0",
            f"DCGM_FI_DEV_NVLINK_ERROR_DL_REPLAY{{{labels}}} 0",
            f"DCGM_FI_DEV_CLOCKS_EVENT_REASONS{{{labels}}} {int(values[CLOCK])}",
            f"DCGM_FI_DEV_SM_CLOCK{{{labels}}} 1980",
            f"DCGM_FI_DEV_POWER_USAGE{{{labels}}} {values[DRAW]}",
            f"DCGM_FI_DEV_POWER_MGMT_LIMIT{{{labels}}} 700",
            f"DCGM_FI_DEV_GPU_UTIL{{{labels}}} {values[UTIL]}",
            "# TYPE DCGM_FI_DEV_THERMAL_VIOLATION counter",
            f"DCGM_FI_DEV_THERMAL_VIOLATION{{{labels}}} {values[THERMAL] * 1000:.0f}",
            "# TYPE DCGM_FI_DEV_POWER_VIOLATION counter",
            f"DCGM_FI_DEV_POWER_VIOLATION{{{labels}}} {values[POWER] * 1000:.0f}",
            "",
        ]
    )


def armed_reads(plan: dict[str, Any]) -> list[dict[str, float]]:
    """What the proxy serves per collector read while the plan is armed: the
    plan's values over the real reading (an omitted field passes through)."""

    return [
        {name: value for name, value in read.items() if value is not None}
        for read in served_reads(plan)
    ]


def around(plan: dict[str, Any]) -> list[dict[str, float]]:
    """Two pass-through reads, the armed plan, two pass-through reads."""

    return [{}, {}, *armed_reads(plan), {}, {}]


def deliver(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, reads: list[dict[str, float]]
) -> list[GpuMetricBatch]:
    """What the real collector ships for ``reads`` scraped every 15 s."""

    path = tmp_path / "boot"
    path.write_text(BOOT + "\n", encoding="ascii")
    monkeypatch.setenv("GPU_FAULT_BOOT_ID_PATH", str(path))
    sink = RecordingSink()
    clock = iter(NOW + timedelta(seconds=15 * index) for index in range(len(reads)))
    dcgm = DcgmMetricsCollector(
        sink,
        context(),
        node_id=NODE,
        now=lambda: next(clock),
        health_summary_seconds=300,
        edge_confirmation_samples=3,
    )
    for read in reads:
        dcgm.collect_text(exporter_text(read), extra_samples=list(LIMITS))
    batches = [
        GpuMetricBatch.model_validate(payload) for _path, payload in sink.requests
    ]
    assert not any(batch.collection_errors for batch in batches), (
        "every scrape carries the required fields"
    )
    return batches


def served(batch: GpuMetricBatch, name: str) -> float:
    return next(
        sample.value
        for sample in batch.samples
        if sample.canonical_name == name and sample.gpu_uuid == GPU
    )


def shipped(
    batches: list[GpuMetricBatch], *names: str
) -> list[tuple[list[str], tuple[float, ...]]]:
    return [
        (batch.edge_filter_reasons, tuple(served(batch, name) for name in names))
        for batch in batches
    ]


def active(service: GpuMetricsService) -> set[str]:
    """What ``_validate_active_findings`` reads for the GPU on its next attempt."""

    return {
        finding.canonical_name
        for finding in service.findings(CLUSTER, NODE)
        if finding.gpu_uuid == GPU
    }


def graded(findings: list[Any]) -> dict[str, tuple[str, str | None]]:
    return {
        item.canonical_name: (item.severity.value, item.automatic_action)
        for item in findings
    }


def tlimit_decision() -> plans.VariantDecision:
    device = plans.build_device_plan(
        gpu_uuid=GPU, gpu_index="0", product="NVIDIA H200", raw_limits=H200_LIMITS
    )
    return device.decisions[plans.CORE_TLIMIT]


def power_plan(confirmation_samples: int = 3) -> dict[str, Any]:
    return plans.power_limit_throttling_plan(
        confirmation_samples=confirmation_samples,
        power_limit_w=700.0,
        power_limit_ratio=0.95,
        min_utilization_percent=80.0,
    )


def old_shape(plan: dict[str, Any], counter: str, marker: str) -> dict[str, Any]:
    """The live 2026-10-26 shape: three more rising reads, the tail at the last."""

    rising = [step for step in plan["steps"] if step["values"].get(marker)]
    step_us = rising[1]["values"][counter] - rising[0]["values"][counter]
    extra = [
        {
            "reads": 1,
            "values": {
                **rising[-1]["values"],
                counter: rising[-1]["values"][counter] + index * step_us,
            },
        }
        for index in range(1, 4)
    ]
    tail = dict(plan["steps"][-1])
    tail["values"] = {**tail["values"], counter: extra[-1]["values"][counter]}
    return {"steps": [plan["steps"][0], *rising, *extra, tail]}


# --------------------------------------------------------------------------- #
# thermal_warning
# --------------------------------------------------------------------------- #
def test_thermal_warning_clears_its_composite_on_the_zero_delta_cleared_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan = plans.thermal_stress_plan(tlimit_decision(), confirmation_samples=3)
    batches = deliver(monkeypatch, tmp_path, around(plan))
    assert shipped(batches, TLIMIT, CLOCK, THERMAL) == [
        (["initial-baseline"], (59.0, IDLE, 0.0)),
        (["candidate-confirmed"], (0.5, plans.SW_THERMAL_BIT, 3_000_000.0)),
        (["candidate-recovered"], (59.0, 0.0, 3_000_000.0)),
    ], "the filter ships the confirmation and the first cleared read, nothing else"

    service = GpuMetricsService()
    baseline = service.ingest(batches[0])
    assert not baseline.findings, "the pass-through baseline breaches nothing"

    confirmed = service.ingest(batches[1])
    assert graded(confirmed.findings) == {
        TLIMIT: (WARNING, DIAGNOSTIC),
        CLOCK: (WARNING, DIAGNOSTIC),
        THERMAL: (WARNING, DIAGNOSTIC),
    }, "the confirmed read mints the three components as read-only diagnostics"
    (composite,) = confirmed.new_composite_findings
    assert composite.correlation_rule_id == "THERMAL_STRESS"
    assert (composite.severity.value, composite.automatic_action) == (
        WARNING,
        DIAGNOSTIC,
    ), "a warning-band temperature keeps THERMAL_STRESS a WARNING diagnostic"
    assert set(composite.component_metrics) >= {TLIMIT, CLOCK, THERMAL}
    assert set(confirmed.suppressed_finding_ids) == {
        item.finding_id for item in confirmed.findings
    }, "every component is suppressed into the composite"
    assert [item.canonical_name for item in confirmed.new_findings] == [
        composite.canonical_name
    ], "one incident: the composite's"
    assert active(service) == {TLIMIT, CLOCK, THERMAL, composite.canonical_name}

    cleared = service.ingest(batches[2])
    assert not cleared.findings and not cleared.composite_findings, (
        "a zero delta under a cleared reason and the real temperature breaches nothing"
    )
    assert active(service) == set(), (
        "the components and the composite are cleared by the batch VALIDATE_GPU "
        "re-checks on"
    )


def test_the_former_thermal_tail_re_minted_the_duration_on_the_cleared_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The live 2026-10-26 shape, then the pass-through reset after the plan."""

    plan = old_shape(
        plans.thermal_stress_plan(tlimit_decision(), confirmation_samples=3),
        THERMAL,
        CLOCK,
    )
    batches = deliver(monkeypatch, tmp_path, around(plan))
    assert shipped(batches, TLIMIT, CLOCK, THERMAL) == [
        (["initial-baseline"], (59.0, IDLE, 0.0)),
        (["candidate-confirmed"], (0.5, plans.SW_THERMAL_BIT, 3_000_000.0)),
        # The counter's own duty-cycle edge, one read later: the second
        # accepted breach batch of every live run (accepted_injected_samples 2).
        (["candidate-confirmed"], (0.5, plans.SW_THERMAL_BIT, 4_000_000.0)),
        # The cleared read carries the last rising value, 2e6 over the delivered.
        (["candidate-recovered"], (59.0, 0.0, 6_000_000.0)),
        # The pass-through real counter: the counter's candidate recovers.
        (["candidate-recovered"], (59.0, IDLE, 0.0)),
    ], "the rising reads after the confirmation surface on the cleared read"

    service = GpuMetricsService()
    baseline = service.ingest(batches[0])
    assert not baseline.findings, "the pass-through baseline breaches nothing"
    confirmed = service.ingest(batches[1])
    assert len(confirmed.new_composite_findings) == 1
    sustained = service.ingest(batches[2])
    assert not sustained.new_findings, (
        "the sustained batch replaces the findings: one composite incident"
    )
    assert active(service) == {TLIMIT, CLOCK, THERMAL, "composite:THERMAL_STRESS"}

    cleared = service.ingest(batches[3])
    (duration,) = cleared.findings
    assert duration.canonical_name == THERMAL
    assert duration.value == 6_000_000.0 and duration.delta == 2_000_000.0
    assert duration.severity.value == WARNING and 2.6e6 < duration.rate_per_minute
    assert not cleared.composite_findings, (
        "the composite clears with the temperature, the component does not"
    )
    assert active(service) == {THERMAL}, (
        "the cleared read re-minted the duration finding the clock clear left"
    )

    reset = service.ingest(batches[4])
    assert not reset.findings, "a counter reset is graded as nothing"
    assert active(service) == {THERMAL}, (
        "the pass-through counter is a reset: an invalid delta keeps the finding, "
        "so VALIDATE_GPU fails on it at the grace"
    )


# --------------------------------------------------------------------------- #
# power_performance
# --------------------------------------------------------------------------- #
def test_power_performance_clears_its_composite_on_the_zero_delta_cleared_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    batches = deliver(monkeypatch, tmp_path, around(power_plan()))
    assert shipped(batches, DRAW, UTIL, POWER) == [
        (["initial-baseline"], (70.0, 0.0, 0.0)),
        (["candidate-confirmed"], (682.5, 90.0, 3_000_000.0)),
        (["candidate-recovered"], (70.0, 0.0, 3_000_000.0)),
    ], "the draw-at-the-cap correlation confirms once and recovers once"

    service = GpuMetricsService()
    baseline = service.ingest(batches[0])
    assert not baseline.findings, "the pass-through baseline breaches nothing"

    confirmed = service.ingest(batches[1])
    assert graded(confirmed.findings) == {POWER: (WARNING, DIAGNOSTIC)}
    (composite,) = confirmed.new_composite_findings
    assert composite.correlation_rule_id == "POWER_LIMIT_THROTTLING"
    assert (composite.severity.value, composite.automatic_action) == (
        WARNING,
        DIAGNOSTIC,
    )
    assert composite.component_metrics == [POWER, DRAW, "power_limit_w", UTIL]
    assert [item.canonical_name for item in confirmed.new_findings] == [
        composite.canonical_name
    ], "the violation component is suppressed into the composite"
    assert active(service) == {POWER, composite.canonical_name}

    cleared = service.ingest(batches[2])
    assert not cleared.findings and not cleared.composite_findings
    assert active(service) == set(), (
        "a zero delta clears the component whatever the draw"
    )


def test_the_former_power_tail_passed_only_through_the_idle_draw(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan = old_shape(power_plan(), POWER, DRAW)
    batches = deliver(monkeypatch, tmp_path, around(plan))
    assert shipped(batches, DRAW, UTIL, POWER) == [
        (["initial-baseline"], (70.0, 0.0, 0.0)),
        (["candidate-confirmed"], (682.5, 90.0, 3_000_000.0)),
        (["candidate-confirmed"], (682.5, 90.0, 4_000_000.0)),
        (["candidate-recovered"], (70.0, 0.0, 6_000_000.0)),
        (["candidate-recovered"], (70.0, 0.0, 0.0)),
    ], "the same two confirmations and the same rise on the cleared read"

    service = GpuMetricsService()
    service.ingest(batches[0])
    confirmed = service.ingest(batches[1])
    sustained = service.ingest(batches[2])
    assert len(confirmed.new_composite_findings) == 1 and not sustained.new_findings
    assert active(service) == {POWER, "composite:POWER_LIMIT_THROTTLING"}

    cleared = service.ingest(batches[3])
    assert not cleared.findings and active(service) == set(), (
        "the rule reads the draw of the same batch: the idle pass-through draw "
        "sits under the ratio, so the rise is not graded"
    )
    reset = service.ingest(batches[4])
    assert not reset.findings and active(service) == set()
    # The same rise graded with the draw still at the cap -- a GPU really
    # pinned there while the plan clears -- is the WARNING the thermal plan
    # got; only the flat tail removes that dependence.
    rise = next(s for s in batches[3].samples if s.canonical_name == POWER)
    decision = power_violation_decision(
        GpuMetricsThresholds(),
        batches[2],
        rise,
        2_000_000.0,
        GPU,
        2_000_000.0 * 60 / 45,
    )
    assert decision is not None and decision["severity"].value == WARNING, (
        "with the cap peers present the 2e6 rise is a fresh WARNING"
    )
