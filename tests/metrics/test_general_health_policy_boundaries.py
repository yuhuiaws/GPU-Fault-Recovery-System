from datetime import timedelta

import pytest

from gpu_fault.gpu_metrics import GpuMetricsService
from tests.metrics.test_gpu_metrics import NOW, batch, sample


@pytest.mark.parametrize(
    ("product", "metric", "action"),
    [
        ("H100", "row_remap_pending", "RESET_GPU"),
        ("V100", "retired_pages_pending", "RESET_GPU"),
        ("H100", "retired_pages_pending", "RUN_DIAGNOSTICS"),
        ("UNKNOWN", "row_remap_pending", "RUN_DIAGNOSTICS"),
        ("H100", "row_remap_failure", "DRAIN"),
        ("UNKNOWN", "row_remap_failure", "QUARANTINE"),
        ("H100", "ecc_dbe_volatile_total", "QUARANTINE"),
    ],
)
def test_memory_actions_require_the_matching_architecture(product, metric, action):
    value = sample(metric, 1).model_copy(update={"labels": {}})
    result = GpuMetricsService().ingest(
        batch("memory", [value]).model_copy(update={"product": product})
    )
    assert result.new_findings[0].automatic_action == action


@pytest.mark.parametrize("damage", ["reset", "boot", "identity", "negative"])
def test_counter_discontinuity_is_not_a_fresh_error_or_healthy_delta(damage):
    service = GpuMetricsService()
    service.ingest(batch("baseline", [sample("pcie_replay_total", 10)]))
    service.ingest(
        batch(
            "bad",
            [sample("pcie_replay_total", 20)],
            observed_at=NOW + timedelta(seconds=15),
        )
    )
    value = sample("pcie_replay_total", 0 if damage == "reset" else 30)
    if damage == "boot":
        # A baseline with a known boot and a new boot must not share a delta.
        service.ingest(
            batch(
                "boot-baseline",
                [value.model_copy(update={"labels": {"boot_id": "a"}})],
                observed_at=NOW + timedelta(seconds=30),
            )
        )
        value = value.model_copy(update={"value": 40, "labels": {"boot_id": "b"}})
    elif damage == "identity":
        value = value.model_copy(update={"pci_bdf": "0000:82:00.0"})
    elif damage == "negative":
        value = value.model_copy(update={"value": -1})
    result = service.ingest(
        batch("discontinuity", [value], observed_at=NOW + timedelta(seconds=45))
    )
    assert not any(
        item.canonical_name == "pcie_replay_total" for item in result.new_findings
    ), result


@pytest.mark.parametrize(
    "name", ["thermal_violation_total_us", "power_violation_total_us"]
)
def test_microsecond_noise_cannot_request_recovery(name):
    service = GpuMetricsService()
    peers = [
        sample("power_usage_w", 490),
        sample("power_limit_w", 500),
        sample("gpu_utilization_percent", 95),
    ]
    service.ingest(batch("zero", [sample(name, 0), *peers]))
    for index in range(1, 4):
        result = service.ingest(
            batch(
                f"noise-{index}",
                [sample(name, index), *peers],
                observed_at=NOW + timedelta(seconds=15 * index),
            )
        )
        assert result.new_findings == [], result


def test_duplicate_metric_slots_are_unknown_not_arbitrarily_selected():
    result = GpuMetricsService().ingest(
        batch(
            "duplicate",
            [sample("gpu_temperature_c", 40), sample("gpu_temperature_c", 99)],
        )
    )
    assert result.accepted_samples == 0, result
    assert [item.canonical_name for item in result.findings] == [
        "dcgm_field_completeness"
    ]
    assert result.new_findings == [], "unknown observation does not authorize a reset"


def test_unrelated_samples_do_not_advance_composite_consecutive_count():
    service = GpuMetricsService()
    names = ["ecc_sbe_volatile_total", "row_remap_correctable_total"]
    service.ingest(batch("base", [sample(name, 0) for name in names]))
    result = service.ingest(
        batch(
            "errors",
            [sample(name, 1) for name in names],
            observed_at=NOW + timedelta(seconds=15),
        )
    )
    assert len(result.new_composite_findings) == 1, result
    key = (
        "hp-cluster",
        "worker-1",
        "GPU-a",
        "composite:CORRECTABLE_MEMORY_DEGRADATION",
    )
    before = service.store.get_gpu_finding_states([key])[0]
    for index in (2, 3):
        service.ingest(
            batch(
                f"temperature-{index}",
                [sample("gpu_temperature_c", 40)],
                observed_at=NOW + timedelta(seconds=index * 15),
            )
        )
    after = service.store.get_gpu_finding_states([key])[0]
    assert after.consecutive_breaches == before.consecutive_breaches == 1
    assert after.finding.automatic_action == "RUN_DIAGNOSTICS"
