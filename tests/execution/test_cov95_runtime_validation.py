from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

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
        ("filesystem_inodes_used_percent", 98),
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


def test_inventory_requirement_requires_completed_action_and_post_action_sample() -> (
    None
):
    h = ValidationHarness(
        GPU,
        action=GPU_PLUGIN,
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
    assert pending["missing_post_action_inventory_metric"] == [
        "gpu_inventory_active_count"
    ], pending
    h.feed.host_metric("node-a", "gpu_inventory_active_count", 7)
    mismatch = h.execute()
    assert mismatch.status is WorkflowStepStatus.FAILED, mismatch
    assert mismatch.details["node_failures"] == {
        "node-a": ["gpu_inventory_active_count=7,expected=8"]
    }, mismatch
    assert (
        mismatch.details["inventory_evidence"]["node-a"]["gpu_inventory_active_count"][
            "value"
        ]
        == 7
    ), mismatch
    h.feed.host_metric("node-a", "gpu_inventory_active_count", 8)
    recovered = h.execute()
    assert recovered.status is WorkflowStepStatus.SUCCEEDED, (
        "the expected inventory observed after recovery must clear the guard"
    )
    assert (
        recovered.details["inventory_evidence"]["node-a"]["gpu_inventory_active_count"][
            "value"
        ]
        == 8
    ), recovered


def test_inventory_requirement_waits_for_a_planned_action_that_has_not_completed() -> (
    None
):
    h = ValidationHarness(
        GPU,
        action=GPU_PLUGIN,
        requirements={
            "node-a": {
                "active_metric": "gpu_inventory_active_count",
                "expected_count": 8,
            }
        },
    )
    # The plugin restart is in the plan ahead of the validation but has no
    # completed record: the completion it will produce is what the gate waits
    # for, and a fresh reading taken before it says nothing about the repair.
    h.context = replace(
        h.context,
        workflow=h.context.workflow.model_copy(update={"step_executions": []}),
    )
    h.feed.host_metric(
        "node-a",
        "gpu_inventory_active_count",
        8,
        observed_at=datetime.now(timezone.utc),
    )
    first = h.execute()
    assert first.status is WorkflowStepStatus.WAITING, first
    assert first.details["node_pending"]["node-a"][
        "missing_inventory_action_completion"
    ] == ["REPLACE_NODE", "RESTART_GPU_DEVICE_PLUGIN", "RESTART_NODE"], first


@pytest.mark.parametrize(
    ("operation", "metric", "expected"),
    [
        (FABRIC, "efa_inventory_active_count", 16),
        (GPU, "gpu_inventory_active_count", 8),
    ],
)
def test_diagnostic_inventory_validation_judges_fresh_readings_without_an_action(
    operation: WorkflowOperation, metric: str, expected: int
) -> None:
    """A diagnostic chain (RUN_DIAGNOSTICS: FREEZE_EVIDENCE -> VALIDATE_*) plans
    no recovery action, so the inventory gate has no completion to wait for; it
    judges the readings observed after the chain reached the validation. Live
    (GF-GENERAL-LIVE-013 link_inactive): the step waited for an action nothing
    would ever produce, failed at the 600 s cap, and the cap's failure rebooted a
    node whose port had been ACTIVE again for minutes."""

    h = ValidationHarness(
        operation,
        requirements={"node-a": {"active_metric": metric, "expected_count": expected}},
    )
    first = h.execute()
    assert first.status is WorkflowStepStatus.WAITING, first
    pending = first.details["node_pending"]["node-a"]
    assert "missing_inventory_action_completion" not in pending, pending
    assert pending["missing_fresh_inventory_metric"] == [metric], pending
    assert pending["observed_after"] == [h.context.workflow.created_at.isoformat()], (
        pending
    )
    # The detection's own reading predates the chain: not fresh, not judged.
    h.feed.host_metric(
        "node-a",
        metric,
        expected - 1,
        observed_at=h.context.workflow.created_at - timedelta(seconds=30),
    )
    stale = h.execute()
    assert stale.status is WorkflowStepStatus.WAITING, stale
    assert stale.details["node_pending"]["node-a"][
        "missing_fresh_inventory_metric"
    ] == [metric], stale
    h.feed.host_metric(
        "node-a", metric, expected, observed_at=datetime.now(timezone.utc)
    )
    recovered = h.execute()
    assert recovered.status is WorkflowStepStatus.SUCCEEDED, recovered
    evidence = recovered.details["inventory_evidence"]["node-a"][metric]
    assert (evidence["value"], evidence["expected"]) == (expected, expected), recovered
    assert evidence["observed_at"] > h.context.workflow.created_at.isoformat(), (
        recovered
    )
    h.feed.host_metric(
        "node-a", metric, expected - 1, observed_at=datetime.now(timezone.utc)
    )
    still_short = h.execute()
    assert still_short.status is WorkflowStepStatus.FAILED, still_short
    assert still_short.details["node_failures"] == {
        "node-a": [f"{metric}={expected - 1},expected={expected}"]
    }, still_short


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
