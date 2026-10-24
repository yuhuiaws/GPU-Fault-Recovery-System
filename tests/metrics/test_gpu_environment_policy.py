"""NVIDIA limit semantics and stateful environmental confirmation boundaries."""

from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.gpu_metrics import GpuMetricBatch, GpuMetricSample, GpuMetricsService
from gpu_fault.models import RecoveryAction
from gpu_fault.store import InMemoryStore

NOW = datetime(2026, 10, 23, tzinfo=timezone.utc)


def ingest(
    service, seconds, values, *, boot="boot-a", source="DCGM_EXPORTER", errors=()
):
    return service.ingest(
        GpuMetricBatch(
            batch_id=f"{seconds}-{boot}-{source}",
            cluster_id="cluster-a",
            node_id="node-a",
            product="H100",
            observed_at=NOW + timedelta(seconds=seconds),
            source=source,
            collection_errors=list(errors),
            samples=[
                GpuMetricSample(
                    metric_name=name,
                    canonical_name=name,
                    value=value,
                    gpu_uuid="GPU-a",
                    pci_bdf="0000:81:00.0",
                    labels={"source_boot_id": boot},
                )
                for name, value in values.items()
            ],
        )
    )


def grade(result, name):
    return next(item for item in result.findings if item.canonical_name == name)


@pytest.mark.parametrize("mask", [0, 0x01, 0x04, 0x08, 0x10, 0x100])
def test_normal_clock_reasons_never_authorize_hardware_recovery(mask):
    service = GpuMetricsService(store=InMemoryStore())
    for seconds in (0, 15, 30, 45):
        result = ingest(service, seconds, {"clock_throttle_reasons": mask})
        assert not result.findings, result


@pytest.mark.parametrize("mask", [0x80, 0x84, 0xC0, 0x180])
def test_power_brake_requires_three_valid_observations_and_survives_service_restart(
    mask,
):
    store = InMemoryStore()
    for index in range(3):
        result = ingest(
            GpuMetricsService(store=store), index * 15, {"clock_throttle_reasons": mask}
        )
        finding = grade(result, "clock_throttle_reasons")
        assert finding.confirmation_count == index + 1
        assert finding.automatic_action == (
            RecoveryAction.DRAIN.value
            if index == 2
            else RecoveryAction.RUN_DIAGNOSTICS.value
        )
        assert finding.threshold_source == "NVIDIA_HW_POWER_BRAKE"


@pytest.mark.parametrize("interruption", ["clear", "gap", "boot", "source", "unknown"])
def test_confirmation_cannot_bridge_incomparable_observations(interruption):
    service = GpuMetricsService(store=InMemoryStore())
    values = {"clock_throttle_reasons": 0x80}
    ingest(service, 0, values)
    ingest(service, 15, values)
    seconds, kwargs = 30, {}
    if interruption == "clear":
        ingest(service, 20, {"clock_throttle_reasons": 0})
    elif interruption == "gap":
        seconds = 120
    elif interruption == "boot":
        kwargs["boot"] = "boot-b"
    elif interruption == "source":
        kwargs["source"] = "NVIDIA_SMI"
    else:
        ingest(service, 20, {}, errors=["exporter unavailable"])
    result = ingest(service, seconds, values, **kwargs)
    assert grade(result, "clock_throttle_reasons").confirmation_count == 1
    assert grade(result, "clock_throttle_reasons").automatic_action == "RUN_DIAGNOSTICS"


@pytest.mark.parametrize("mask", [-1, 128.5, float(2**63 - 1), float("nan")])
def test_invalid_masks_do_not_activate_power_brake(mask):
    service = GpuMetricsService(store=InMemoryStore())
    for seconds in (0, 15, 30):
        result = ingest(service, seconds, {"clock_throttle_reasons": mask})
        assert all(
            item.canonical_name != "clock_throttle_reasons" for item in result.findings
        ), result


def test_warning_history_cannot_confirm_a_new_critical_temperature():
    service = GpuMetricsService(store=InMemoryStore())
    for seconds in (0, 15, 30):
        ingest(service, seconds, {"gpu_temperature_c": 86})
    result = ingest(service, 45, {"gpu_temperature_c": 95})
    assert grade(result, "gpu_temperature_c").confirmation_count == 1
    assert grade(result, "gpu_temperature_c").automatic_action == "RUN_DIAGNOSTICS"
    repeated = ingest(service, 45, {"gpu_temperature_c": 95})
    assert repeated.duplicate is True, repeated
    ingest(service, 60, {"gpu_temperature_c": 95})
    final = ingest(service, 75, {"gpu_temperature_c": 95})
    assert grade(final, "gpu_temperature_c").automatic_action == "DRAIN"


def test_absolute_shutdown_limit_is_immediate_even_with_healthy_tlimit():
    result = ingest(
        GpuMetricsService(store=InMemoryStore()),
        0,
        {
            "gpu_temperature_c": 106,
            "gpu_shutdown_temperature_c": 105,
            "gpu_tlimit_c": 40,
            "gpu_slowdown_tlimit_c": -2,
        },
    )
    assert grade(result, "gpu_temperature_c").automatic_action == "DRAIN"


@pytest.mark.parametrize(
    "values,name",
    [
        (
            {"gpu_temperature_c": 89, "gpu_max_operating_temperature_c": 88},
            "gpu_temperature_c",
        ),
        ({"gpu_tlimit_c": -1, "gpu_max_operating_tlimit_c": 0}, "gpu_tlimit_c"),
    ],
)
def test_software_operating_limit_alone_is_not_hardware_slowdown(values, name):
    service = GpuMetricsService(store=InMemoryStore())
    for seconds in (0, 15, 30, 45):
        result = ingest(service, seconds, values)
        assert grade(result, name).automatic_action == "RUN_DIAGNOSTICS"


def test_persistent_thermal_duration_and_warning_correlation_do_not_drain():
    service = GpuMetricsService(store=InMemoryStore())
    for index in range(5):
        result = ingest(
            service,
            index * 15,
            {
                "gpu_temperature_c": 86,
                "clock_throttle_reasons": 0x20,
                "thermal_violation_total_us": index * 1_000_000,
            },
        )
        assert all(
            item.automatic_action == "RUN_DIAGNOSTICS"
            for item in result.findings + result.composite_findings
        ), result


def test_a_thermal_warning_cannot_suppress_a_confirmed_power_brake():
    service = GpuMetricsService(store=InMemoryStore())
    for seconds in (0, 15, 30):
        result = ingest(
            service, seconds, {"gpu_temperature_c": 86, "clock_throttle_reasons": 0xC0}
        )
    assert any(
        item.threshold_source == "NVIDIA_HW_POWER_BRAKE"
        and item.automatic_action == "DRAIN"
        for item in result.new_findings
    ), result
