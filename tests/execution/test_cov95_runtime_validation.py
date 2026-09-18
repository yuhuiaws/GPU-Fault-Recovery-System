from __future__ import annotations

from datetime import timedelta

import pytest

from gpu_fault.gpu_metrics import GpuMetricSource
from gpu_fault.models import WorkflowOperation, WorkflowStepStatus
from gpu_fault.telemetry import CollectorKind
from tests.execution._cov95_runtime_validation import ValidationHarness

GPU = WorkflowOperation.VALIDATE_GPU
HOST = WorkflowOperation.VALIDATE_HOST
FABRIC = WorkflowOperation.VALIDATE_FABRIC
REBOOT = WorkflowOperation.RESTART_NODE
GPU_PLUGIN = WorkflowOperation.RESTART_GPU_DEVICE_PLUGIN


@pytest.mark.parametrize(
    "field",
    [
        "post_action_max_sample_age",
        "temperature_warning_grace",
        "transient_warning_grace",
    ],
)
def test_validation_configuration_rejects_windows_shorter_than_supported_grace(
    field: str,
) -> None:
    with pytest.raises(ValueError, match="sample age|warning grace"):
        ValidationHarness(GPU, **{field: timedelta(seconds=1)})


def test_validation_adapter_matches_both_owner_and_registered_operation() -> None:
    h = ValidationHarness(GPU)
    assert h.adapter.supports(h.context.step), (
        "matching owner and validation operation must be supported"
    )
    assert not h.adapter.supports(
        h.context.step.model_copy(update={"execution_owner": "other-owner"})
    ), "another adapter's step must not be accepted"
    assert not h.adapter.supports(
        h.context.step.model_copy(update={"operation": REBOOT})
    ), "read-only validation must not claim node mutation operations"


@pytest.mark.parametrize(
    ("name", "threshold"),
    [
        ("cpu_usage_percent", 98),
        ("load1_per_cpu", 2),
        ("memory_used_percent", 95),
        ("swap_used_percent", 80),
        ("filesystem_used_percent", 98),
        ("shared_filesystem_unavailable", 1),
        ("shared_filesystem_used_percent", 98),
        ("disk_io_util_percent", 98),
        ("disk_io_await_ms", 100),
        ("smart_health_failed", 1),
        ("network_link_down", 1),
        ("bmc_critical_sensor", 1),
    ],
)
def test_host_validation_rejects_each_unsafe_threshold_at_the_boundary(
    name: str, threshold: float
) -> None:
    h = ValidationHarness(HOST)
    h.feed.host_metric("node-a", name, threshold - 0.01)
    before = h.execute()
    assert before.status is WorkflowStepStatus.SUCCEEDED, before
    h.feed.host_metric("node-a", name, threshold)
    result = h.execute()
    assert result.status is WorkflowStepStatus.FAILED, result
    assert result.details["node_failures"] == {"node-a": [name]}, result
    assert h.feed.gpu_reads == [], (
        "host validation must not substitute unrelated GPU readings"
    )


def test_missing_host_metric_waits_instead_of_treating_an_incomplete_sample_as_health() -> (
    None
):
    h = ValidationHarness(HOST)
    h.feed.host["node-a"] = [
        item for item in h.feed.host["node-a"] if item.name != "load1_per_cpu"
    ]
    result = h.execute()
    assert result.status is WorkflowStepStatus.WAITING, result
    assert result.details["node_pending"] == {
        "node-a": {"missing_recent_host_metrics": ["load1_per_cpu"]}
    }, result


@pytest.mark.parametrize(
    ("metric", "value"),
    [
        ("network_link_up", 0),
        ("rdma_link_down", 1),
        ("rdma_errors_delta", 1),
        ("efa_rnr_errors_delta", 1),
        ("network_drops_delta", 1),
    ],
)
def test_fabric_validation_rejects_link_and_transport_errors(
    metric: str, value: float
) -> None:
    h = ValidationHarness(FABRIC)
    h.feed.host_metric("node-a", metric, value)
    result = h.execute()
    assert result.status is WorkflowStepStatus.FAILED, result
    assert result.details["node_failures"] == {"node-a": [metric]}, result


@pytest.mark.parametrize("missing", ["link", "rdma"])
def test_fabric_validation_waits_for_required_network_evidence(missing: str) -> None:
    h = ValidationHarness(FABRIC, require_rdma=missing == "rdma")
    if missing == "link":
        h.feed.host["node-a"] = [
            item for item in h.feed.host["node-a"] if item.name != "network_link_up"
        ]
    result = h.execute()
    assert result.status is WorkflowStepStatus.WAITING, result
    expected = "network_link_up" if missing == "link" else "rdma_link_down"
    assert result.details["node_pending"]["node-a"][
        "missing_recent_fabric_metrics"
    ] == [expected], result
    if missing == "rdma":
        h.feed.host_metric("node-a", "rdma_link_down", 0)
        assert h.execute().status is WorkflowStepStatus.SUCCEEDED, (
            "healthy RDMA evidence should clear the observation hold"
        )


@pytest.mark.parametrize("action", [None, GPU_PLUGIN])
def test_inventory_requirement_requires_completed_action_and_post_action_sample(
    action,
) -> None:
    h = ValidationHarness(
        GPU,
        action=action,
        requirements={
            "node-a": {
                "active_metric": "gpu_inventory_active_count",
                "expected_count": 8,
            }
        },
    )
    first = h.execute()
    assert first.status is WorkflowStepStatus.WAITING, first
    pending = first.details["node_pending"]["node-a"]
    if action is None:
        assert "missing_inventory_action_completion" in pending, pending
    else:
        assert pending["missing_post_action_inventory_metric"] == [
            "gpu_inventory_active_count"
        ], pending
        h.feed.host_metric("node-a", "gpu_inventory_active_count", 7)
        mismatch = h.execute()
        assert mismatch.status is WorkflowStepStatus.FAILED, mismatch
        assert mismatch.details["node_failures"] == {
            "node-a": ["gpu_inventory_active_count=7,expected=8"]
        }, mismatch
        h.feed.host_metric("node-a", "gpu_inventory_active_count", 8)
        assert h.execute().status is WorkflowStepStatus.SUCCEEDED, (
            "the expected inventory observed after recovery must clear the guard"
        )


def test_reboot_validation_uses_historical_expected_inventory_but_requires_fresh_actual_count() -> (
    None
):
    h = ValidationHarness(GPU, action=REBOOT)
    h.feed.host_metric(
        "node-a",
        "gpu_inventory_expected_count",
        8,
        observed_at=h.action_at - timedelta(minutes=10),
    )
    h.feed.host_metric(
        "node-a", "gpu_inventory_active_count", 8, observed_at=h.action_at
    )
    first = h.execute()
    assert first.status is WorkflowStepStatus.WAITING, first
    assert first.details["node_pending"]["node-a"][
        "missing_post_action_inventory_metric"
    ] == ["gpu_inventory_active_count"], first
    h.feed.host_metric("node-a", "gpu_inventory_active_count", 8)
    second = h.execute()
    assert second.status is WorkflowStepStatus.SUCCEEDED, second


def test_nvidia_smi_fallback_requires_memory_health_fields_for_every_target_gpu() -> (
    None
):
    h = ValidationHarness(GPU)
    h.feed.gpu["node-a"] = []
    h.feed.gpu_metric("node-a", "gpu_temperature_c", source=GpuMetricSource.NVIDIA_SMI)
    first = h.execute()
    assert first.status is WorkflowStepStatus.WAITING, first
    required = ["ecc_dbe_volatile_total", "row_remap_failure", "row_remap_pending"]
    assert first.details["node_pending"]["node-a"]["missing_recent_metrics_by_gpu"] == {
        "GPU-a": required
    }, first
    for metric in required:
        h.feed.gpu_metric("node-a", metric, source=GpuMetricSource.NVIDIA_SMI)
    second = h.execute()
    assert second.status is WorkflowStepStatus.SUCCEEDED, second


def test_pending_node_keeps_observed_failures_visible_until_all_nodes_can_be_judged() -> (
    None
):
    h = ValidationHarness(HOST, nodes=["node-a", "node-b"])
    h.feed.host_metric("node-a", "smart_health_failed", 1)
    h.feed.statuses["node-b"] = []
    result = h.execute()
    assert result.status is WorkflowStepStatus.WAITING, result
    assert result.details["node_pending"]["node-b"] == {
        "missing_or_stale_collectors": [CollectorKind.HOST_TELEMETRY.value]
    }, result
    assert result.details["failed_nodes_observed"] == ["node-a"], result
    assert result.details["node_failures_observed"] == {
        "node-a": ["smart_health_failed"]
    }, result
