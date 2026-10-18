"""Grading on the T.Limit scale, as NVIDIA defines it (batch 134).

Driver 580.178.04 on H200 reports shutdown -8 C, slowdown -2 C and GPU/memory
max operating 0 C, all "relative to T.Limit ... instead of absolute
measurements" (nvidia-smi documentation, GPU ATTRIBUTES > Temperature). NVIDIA
documents no conversion to absolute degrees, so these thresholds grade the
current T.Limit reading, which counts down: a breach is at or below. The GPU
core's absolute reading is not graded against the configured fallback while
the T.Limit judgement exists. The reading covers the memory too (an HBM read
above the memory max operating temperature "produced a negative T.Limit
temperature reading", 535.129.03 release notes), so the memory threshold joins
the warning boundary; the absolute memory reading keeps its fallback.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from gpu_fault.adapters.gpu_validation import finding_summary, node_failure_text
from gpu_fault.gpu_metrics import (
    GpuMetricBatch,
    GpuMetricSample,
    GpuMetricSource,
    GpuMetricsService,
    GpuMetricsThresholds,
)
from gpu_fault.gpu_temperature_policy import (
    NVIDIA_NVML_TEMPERATURE_REFERENCE,
    NVIDIA_SMI_TLIMIT_REFERENCE,
    temperature_decision,
)
from gpu_fault.host_health import NodeHealthCategory, NodeHealthFinding
from gpu_fault.models import RecoveryAction, Severity
from tests._builders import gpu_metric_batch

NOW = datetime(2026, 9, 21, 6, 0, tzinfo=UTC)
THRESHOLDS = GpuMetricsThresholds()
#: What driver 580.178.04 reported for every H200 of the plain-EKS p5en.
H200_580 = {
    "gpu_shutdown_tlimit_c": -8.0,
    "gpu_slowdown_tlimit_c": -2.0,
    "gpu_max_operating_tlimit_c": 0.0,
    "memory_max_operating_tlimit_c": 0.0,
}


def sample(name: str, value: float) -> GpuMetricSample:
    return GpuMetricSample(
        metric_name=name,
        canonical_name=name,
        value=value,
        unit="celsius",
        gpu_index="0",
        gpu_uuid="GPU-a",
    )


def decide(name: str, value: float, limits: dict[str, float]):
    decision = temperature_decision(THRESHOLDS, sample(name, value), limits)
    if decision is None:
        return None
    return (
        decision["severity"],
        decision["automatic_action"],
        decision["threshold_value"],
        decision["threshold_source"],
    )


def batch(
    batch_id: str, samples: list[GpuMetricSample], *, seconds: int = 0
) -> GpuMetricBatch:
    return gpu_metric_batch(
        batch_id,
        NOW + timedelta(seconds=seconds),
        GpuMetricSource.DCGM_EXPORTER,
        samples,
        cluster_id="tlimit-cluster",
        node_id="tlimit-node",
        product="H200",
        driver_branch=580,
    )


def limits(values: dict[str, float] = H200_580) -> list[GpuMetricSample]:
    return [sample(name, value) for name, value in values.items()]


@pytest.mark.parametrize(
    "reading,expected",
    [
        (58.0, None),
        (4.0, None),
        (3.0, ("WARNING", "RUN_DIAGNOSTICS", 3.0)),
        (0.0, ("WARNING", "RUN_DIAGNOSTICS", 3.0)),
        (-1.0, ("WARNING", "RUN_DIAGNOSTICS", 3.0)),
        (-2.0, ("CRITICAL", "DRAIN", -2.0)),
        (-9.0, ("CRITICAL", "DRAIN", -2.0)),
    ],
    ids=["live-58", "4", "3", "0", "-1", "slowdown", "past-shutdown"],
)
def test_the_live_580_limits_grade_the_margin_at_or_below(
    reading: float, expected: tuple[str, str, float] | None
) -> None:
    grade = decide("gpu_tlimit_c", reading, {**H200_580, "gpu_tlimit_c": reading})

    assert grade == (None if expected is None else (*expected, "NVIDIA_DEVICE_TLIMIT"))


@pytest.mark.parametrize(
    "thresholds,critical,warning",
    [
        ({"gpu_max_operating_tlimit_c": 0.0}, 0.0, 5.0),
        ({"gpu_shutdown_tlimit_c": -8.0}, -5.0, 0.0),
        ({"gpu_slowdown_tlimit_c": -2.0, "gpu_max_operating_tlimit_c": 5.0}, -2, 5),
        ({"gpu_slowdown_tlimit_c": -2.0, "memory_max_operating_tlimit_c": 4.0}, -2, 4),
        ({"gpu_slowdown_tlimit_c": -2.0, "gpu_max_operating_tlimit_c": -150.5}, -2, 3),
    ],
    ids=[
        "max-operating-only",
        "shutdown-only",
        "max-operating-first",
        "memory-max-operating-first",
        "garbage",
    ],
)
def test_a_missing_tlimit_threshold_falls_back_like_the_absolute_rule(
    thresholds: dict[str, float], critical: float, warning: float
) -> None:
    """Slowdown, else max operating, else shutdown plus the shutdown margin.

    The warning sits the warning margin before critical, or at a max operating
    threshold (GPU or memory) when that one comes first; an implausible value
    is ignored like N/A.
    """

    def grade(reading: float):
        return decide("gpu_tlimit_c", reading, {**thresholds, "gpu_tlimit_c": reading})

    assert grade(critical) == ("CRITICAL", "DRAIN", critical, "NVIDIA_DEVICE_TLIMIT")
    assert grade(warning) == (
        "WARNING",
        "RUN_DIAGNOSTICS",
        warning,
        "NVIDIA_DEVICE_TLIMIT",
    )
    assert grade(warning + 1) is None


def test_the_absolute_reading_yields_to_the_margin_and_only_to_it() -> None:
    judged = {**H200_580, "gpu_tlimit_c": 40.0}

    assert decide("gpu_temperature_c", 99, judged) is None, (
        "with a T.Limit reading and T.Limit limits the absolute reading is not "
        "graded on the configured fallback"
    )
    for context in (
        H200_580,
        {"gpu_tlimit_c": 40.0},
        {"memory_max_operating_tlimit_c": 0.0, "gpu_tlimit_c": 40.0},
        {**H200_580, "gpu_tlimit_c": 4294967294.0},
    ):
        assert decide("gpu_temperature_c", 90, context) == (
            "CRITICAL",
            "DRAIN",
            90,
            "CONFIGURED_FALLBACK",
        ), f"no usable T.Limit judgement in {context}: the fallback still grades"
    assert decide("gpu_tlimit_c", -5, {"memory_max_operating_tlimit_c": 0.0}) is None


def test_gpu_memory_keeps_its_absolute_limit_or_the_fallback() -> None:
    assert decide("memory_temperature_c", 95, {**H200_580, "gpu_tlimit_c": -9}) == (
        "CRITICAL",
        "DRAIN",
        95,
        "CONFIGURED_FALLBACK",
    ), "the absolute memory reading keeps its safety net on a T.Limit GPU"
    assert decide(
        "memory_temperature_c", 90, {"memory_max_operating_temperature_c": 90}
    ) == ("CRITICAL", "DRAIN", 90, "NVIDIA_DEVICE_LIMIT")


def test_each_scale_names_its_own_vendor_reference() -> None:
    tlimit = temperature_decision(
        THRESHOLDS, sample("gpu_tlimit_c", -3), {**H200_580, "gpu_tlimit_c": -3}
    )
    absolute = temperature_decision(
        THRESHOLDS, sample("gpu_temperature_c", 93), {"gpu_slowdown_temperature_c": 92}
    )
    fallback = temperature_decision(THRESHOLDS, sample("gpu_temperature_c", 93), {})

    assert None not in (tlimit, absolute, fallback), "every case breaches"
    assert (tlimit["policy_source"], tlimit["policy_reference"]) == (
        "SITE_NVIDIA_DEVICE_LIMIT_DERIVED",
        NVIDIA_SMI_TLIMIT_REFERENCE,
    )
    assert (absolute["policy_source"], absolute["policy_reference"]) == (
        "SITE_NVIDIA_DEVICE_LIMIT_DERIVED",
        NVIDIA_NVML_TEMPERATURE_REFERENCE,
    )
    assert NVIDIA_NVML_TEMPERATURE_REFERENCE.endswith(
        "/nvml-api/latest/api/group__nvmlDeviceQueries.html"
    ), "the old nvml-api/group__nvmlDeviceQueries.html URL answers 404"
    assert "policy_reference" not in fallback


def test_ingestion_raises_one_tlimit_finding_and_no_fallback_one() -> None:
    service = GpuMetricsService()

    result = service.ingest(
        batch(
            "tlimit-critical",
            [
                *limits(),
                sample("gpu_tlimit_c", -3),
                sample("gpu_temperature_c", 91),
                sample("memory_temperature_c", 40),
            ],
        )
    )

    assert [
        (
            item.canonical_name,
            item.severity,
            item.automatic_action,
            item.value,
            item.threshold_value,
            item.threshold_source,
            item.policy_reference,
        )
        for item in result.new_findings
    ] == [
        (
            "gpu_tlimit_c",
            "CRITICAL",
            "DRAIN",
            -3.0,
            -2.0,
            "NVIDIA_DEVICE_TLIMIT",
            NVIDIA_SMI_TLIMIT_REFERENCE,
        )
    ]


def test_a_tlimit_finding_is_the_temperature_of_the_thermal_composite() -> None:
    service = GpuMetricsService()
    thermal = [
        *limits(),
        sample("gpu_tlimit_c", 1),
        sample("gpu_temperature_c", 86),
        sample("clock_throttle_reasons", 0x20),
        sample("sm_clock_mhz", 900),
    ]

    service.ingest(batch("thermal-warning", thermal))
    critical = service.ingest(batch("thermal-critical", thermal, seconds=15))

    (composite,) = critical.new_composite_findings
    assert (composite.correlation_rule_id, composite.severity) == (
        "THERMAL_STRESS",
        "CRITICAL",
    )
    assert {"gpu_tlimit_c", "clock_throttle_reasons"} <= set(
        composite.component_metrics
    )


@pytest.mark.parametrize("reading,composite", [(2.0, False), (40.0, True)])
def test_a_tlimit_finding_explains_a_power_violation(
    reading: float, composite: bool
) -> None:
    service = GpuMetricsService()
    service.ingest(batch("power-baseline", [sample("power_violation_total_us", 0)]))

    result = service.ingest(
        batch(
            "power-correlated",
            [
                *limits(),
                sample("gpu_tlimit_c", reading),
                sample("power_violation_total_us", 1),
                sample("power_usage_w", 680),
                sample("power_limit_w", 700),
                sample("gpu_utilization_percent", 95),
            ],
            seconds=15,
        )
    )

    rules = [item.correlation_rule_id for item in result.new_composite_findings]
    assert ("POWER_LIMIT_THROTTLING" in rules) is composite, (
        "a T.Limit temperature finding is a temperature finding for the power "
        f"rule too: {rules}"
    )


@pytest.mark.parametrize(
    "metric",
    [
        "gpu_tlimit_c",
        "gpu_slowdown_tlimit_c",
        "gpu_shutdown_tlimit_c",
        "gpu_max_operating_tlimit_c",
        "memory_max_operating_tlimit_c",
    ],
)
def test_tlimit_findings_are_thermal_faults(metric: str) -> None:
    finding = NodeHealthFinding(
        finding_id="tlimit-finding",
        event_id="tlimit-event",
        cluster_id="tlimit-cluster",
        node_id="tlimit-node",
        observed_at=NOW,
        category=NodeHealthCategory.GPU,
        severity=Severity.CRITICAL,
        recommended_action=RecoveryAction.DRAIN,
        reason="tlimit",
        metric_name=metric,
    )

    assert finding.fault_class() == "GPU_THERMAL"


def test_validation_text_shows_a_margin_breach_as_at_or_below() -> None:
    service = GpuMetricsService()
    (finding,) = service.ingest(
        batch("tlimit-text", [*limits(), sample("gpu_tlimit_c", -3)])
    ).new_findings

    text = node_failure_text(
        "tlimit-node",
        ["active_gpu_health_findings"],
        {"tlimit-node": [finding_summary(finding)]},
    )

    assert "gpu_tlimit_c CRITICAL -3<=-2 GPU-a" in text, text
