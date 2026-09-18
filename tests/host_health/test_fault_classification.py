from __future__ import annotations

from datetime import UTC, datetime

import pytest

from gpu_fault.host_health import (
    HostMetricSample,
    NodeHealthCategory,
    NodeHealthFinding,
    efa_inventory_rule,
)
from gpu_fault.models import RecoveryAction, Severity


def finding(
    metric, *, reason="finding", parameters=None, category=NodeHealthCategory.GPU
):
    return NodeHealthFinding(
        finding_id="classification-finding",
        event_id="classification-event",
        cluster_id="classification-cluster",
        node_id="classification-node",
        observed_at=datetime(2026, 9, 14, tzinfo=UTC),
        category=category,
        severity=Severity.WARNING,
        recommended_action=RecoveryAction.RUN_DIAGNOSTICS,
        reason=reason,
        metric_name=metric,
        diagnostic_parameters=parameters or {},
    )


@pytest.mark.parametrize(
    ("metric", "expected"),
    [
        ("gpu_temperature_c", "GPU_THERMAL"),
        ("gpu_temperature_celsius", "GPU_THERMAL"),
        ("memory_temperature_c", "GPU_THERMAL"),
        ("thermal_violation_total_us", "GPU_THERMAL"),
        ("power_violation_total_us", "GPU_THERMAL"),
        ("gpu_slowdown_temperature_c", "GPU_THERMAL"),
        ("memory_max_operating_temperature_c", "GPU_THERMAL"),
        ("power_usage_w", "GPU_THERMAL"),
        ("power_limit_w", "GPU_THERMAL"),
        ("ecc_sbe_volatile_total", "GPU_MEMORY"),
        ("ecc_dbe_volatile_total", "GPU_MEMORY"),
        ("ecc_sbe_aggregate_total", "GPU_MEMORY"),
        ("ecc_dbe_aggregate_total", "GPU_MEMORY"),
        ("dcgm_ecc_dbe_delta", "GPU_MEMORY"),
        ("retired_pages_sbe_total", "GPU_MEMORY"),
        ("retired_pages_dbe_total", "GPU_MEMORY"),
        ("retired_pages_pending", "GPU_MEMORY"),
        ("row_remap_correctable_total", "GPU_MEMORY"),
        ("row_remap_uncorrectable_total", "GPU_MEMORY"),
        ("row_remap_pending", "GPU_MEMORY"),
        ("row_remap_failure", "GPU_MEMORY"),
        ("nvlink_crc_flit_error_total", "GPU_FABRIC"),
        ("nvlink_crc_data_error_total", "GPU_FABRIC"),
        ("nvlink_replay_error_total", "GPU_FABRIC"),
        ("nvlink_recovery_error_total", "GPU_FABRIC"),
        ("nvlink_crc_aggregate_error_total", "GPU_FABRIC"),
        ("nvlink_recovery_aggregate_error_total", "GPU_FABRIC"),
        ("nvlink_replay_aggregate_error_total", "GPU_FABRIC"),
        ("gpu_inventory_mismatch", "GPU_INVENTORY"),
        ("gpu_inventory_identity_changed", "GPU_INVENTORY"),
        ("gpu_expected_count_unknown", "GPU_INVENTORY"),
        ("gpu_kubernetes_allocatable_mismatch", "GPU_INVENTORY"),
        ("pcie_replay_total", "GPU"),
        ("dcgm_field_completeness", "GPU"),
        ("host_gpu_utilization_percent", "GPU"),
    ],
)
def test_canonical_metric_owns_classification_not_incidental_reason(metric, expected):
    value = finding(
        metric,
        reason="NVLink ECC memory temperature power inventory missing",
        parameters={"diagnostic_reason": "unrelated GPU memory error"},
    )
    assert value.fault_class() == expected
    assert value.marker().fault_class == expected


@pytest.mark.parametrize(
    ("rule", "expected"),
    [
        ("THERMAL_STRESS", "GPU_THERMAL"),
        ("POWER_LIMIT_THROTTLING", "GPU_THERMAL"),
        ("GPU_MEMORY_DEGRADATION", "GPU_MEMORY"),
        ("CORRECTABLE_MEMORY_DEGRADATION", "GPU_MEMORY"),
        ("NVLINK_LINK_DEGRADATION", "GPU_FABRIC"),
        ("MULTI_GPU_NVLINK_FABRIC_FAILURE", "GPU_FABRIC"),
        ("PCIE_XID_LINK_FAILURE", "GPU"),
    ],
)
@pytest.mark.parametrize("structured_only", [False, True])
def test_composite_rule_classification_is_explicit(rule, expected, structured_only):
    value = finding(
        None if structured_only else f"composite:{rule}",
        reason="memory temperature NVLink missing device",
        parameters={"correlation_rule_id": rule},
    )
    assert value.fault_class() == expected


@pytest.mark.parametrize(
    "metric", [None, "unknown_metric", "nvlink_unrecognized", "composite:UNKNOWN"]
)
@pytest.mark.parametrize(
    "reason",
    [
        "ECC memory failure",
        "thermal power failure",
        "NVLink SXID failure",
        "GPU missing",
    ],
)
def test_prose_cannot_assign_an_unknown_gpu_signal_a_trusted_hardware_subclass(
    metric, reason
):
    assert (
        finding(
            metric, reason=reason, parameters={"diagnostic_reason": reason}
        ).fault_class()
        == "GPU"
    )


def test_memory_temperature_wins_over_conflicting_secondary_rule_metadata():
    assert (
        finding(
            "memory_temperature_c",
            reason="memory failure",
            parameters={"correlation_rule_id": "GPU_MEMORY_DEGRADATION"},
        ).fault_class()
        == "GPU_THERMAL"
    )


@pytest.mark.parametrize(
    ("category", "expected"),
    [
        (NodeHealthCategory.RDMA, "NETWORK_FABRIC"),
        (NodeHealthCategory.NETWORK, "NETWORK_FABRIC"),
        (NodeHealthCategory.NCCL, "NETWORK_FABRIC"),
        (NodeHealthCategory.MEMORY, "MEMORY"),
        (NodeHealthCategory.CPU, "CPU"),
    ],
)
def test_non_gpu_categories_do_not_inherit_gpu_metric_classification(
    category, expected
):
    assert (
        finding(
            "memory_temperature_c", category=category, reason="GPU memory ECC"
        ).fault_class()
        == expected
    )


@pytest.mark.parametrize(
    ("failure_mode", "expected_action"),
    [
        ("DRIVER_UNBOUND", RecoveryAction.REMEDIATE_EFA_DRIVER),
        ("LINK_INACTIVE", RecoveryAction.RUN_DIAGNOSTICS),
        ("EXCESS_DEVICE", RecoveryAction.RUN_DIAGNOSTICS),
        ("PCI_DEVICE_MISSING", RecoveryAction.REBOOT_NODE),
        ("UNRECOGNIZED", RecoveryAction.REBOOT_NODE),
        (None, RecoveryAction.REBOOT_NODE),
    ],
)
def test_efa_inventory_rule_resolves_without_mutating_the_sample(
    failure_mode, expected_action
):
    labels = {} if failure_mode is None else {"failure_mode": failure_mode}
    sample = HostMetricSample(name="efa_inventory_mismatch", value=1, labels=labels)
    before = sample.model_copy(deep=True)

    threshold, category, severity, action, reason = efa_inventory_rule(sample)

    assert (threshold, category, severity, action) == (
        1.0,
        NodeHealthCategory.RDMA,
        Severity.CRITICAL,
        expected_action,
    )
    assert reason, "a resolved recovery rule must retain its diagnostic reason"
    assert sample == before, "pure rule resolution must not change evidence or labels"
