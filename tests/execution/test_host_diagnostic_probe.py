from dataclasses import replace

import pytest

from gpu_fault.host_diagnostics import (
    HOST_DIAGNOSTIC_FAILURE,
    HOST_DIAGNOSTIC_PROBE,
    diagnostic_probe,
    validate_diagnostic_probe,
)
from gpu_fault.host_health import HostMetricSample, NodeHealthFinding, NodeHealthPolicy
from gpu_fault.models import RecoveryAction, WorkflowStepStatus
from gpu_fault.models import WorkflowOperation as Op
from gpu_fault.orchestration.escalation import HardwareEscalationService
from tests._builders import (
    build_store,
    host_telemetry_batch,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)
from tests.execution._cov95_runtime_validation import ValidationHarness


def test_host_policy_retains_metrics_needed_for_actual_validation():
    store = build_store()
    harness = ValidationHarness(Op.VALIDATE_HOST)
    batch = host_telemetry_batch(
        "host-source",
        harness.feed.sampled_at,
        [
            HostMetricSample(name=name, value=1)
            for name in (
                "cpu_usage_percent",
                "swap_used_percent",
                "shared_filesystem_unavailable",
                "disk_io_await_ms",
                "smart_health_failed",
                "bmc_critical_sensor",
                "memory_available_percent",
                "training_rank_cpu_percent",
            )
        ],
    )
    NodeHealthPolicy(store).evaluate_metrics(batch)
    assert {
        item.name for item in store.list_telemetry_metrics_latest("cluster-a", "node-a")
    } == {sample.name for sample in batch.samples}


@pytest.mark.parametrize("comparison,value,healthy", [("gte", 99, 20), ("lte", 1, 20)])
def test_validation_requires_the_triggering_device_and_comparison(
    comparison, value, healthy
):
    h = ValidationHarness(Op.VALIDATE_HOST)
    name = "page_cache_percent" if comparison == "gte" else "memory_available_percent"
    probe = {
        "metric": name,
        "device": "target",
        "comparison": comparison,
        "threshold": 90 if comparison == "gte" else 5,
        "observed_at": h.action_at.isoformat(),
    }
    step = h.context.step.model_copy(
        update={"parameters": {HOST_DIAGNOSTIC_PROBE: probe}}
    )
    h.context = replace(h.context, step=step)
    assert h.execute().status is WorkflowStepStatus.WAITING
    h.feed.host_metric("node-a", name, healthy)
    h.feed.host["node-a"][-1].device = "other"
    assert h.execute().status is WorkflowStepStatus.WAITING
    h.feed.host_metric("node-a", name, value)
    h.feed.host["node-a"][-1].device = "target"
    assert h.execute().status is WorkflowStepStatus.FAILED
    h.feed.host_metric("node-a", name, healthy)
    h.feed.host["node-a"][-1].device = "target"
    assert h.execute().status is WorkflowStepStatus.SUCCEEDED


def test_low_gpu_validation_failure_does_not_authorize_hardware_recovery():
    h = ValidationHarness(Op.VALIDATE_GPU)
    finding = NodeHealthFinding(
        finding_id="low-gpu",
        event_id="low-gpu",
        cluster_id="cluster-a",
        node_id="node-a",
        observed_at=h.action_at,
        category="GPU",
        severity="warning",
        reason="low utilization",
        recommended_action=RecoveryAction.RUN_DIAGNOSTICS,
        metric_name="host_gpu_utilization_percent",
        device="GPU-a",
        value=0,
        diagnostic_parameters={"comparison": "lte", "threshold_percent": 5},
    )
    step = h.context.step.model_copy(
        update={"parameters": {HOST_DIAGNOSTIC_PROBE: diagnostic_probe(finding)}}
    )
    h.context = replace(h.context, step=step)
    h.feed.host_metric("node-a", finding.metric_name, 0)
    h.feed.host["node-a"][-1].device = "GPU-a"
    result = h.execute()
    assert result.status is WorkflowStepStatus.FAILED
    assert result.details[HOST_DIAGNOSTIC_FAILURE] is True
    workflow = workflow_request(
        "low-workflow",
        "incident",
        official_steps=[workflow_step(Op.RUN_DCGM_DIAGNOSTIC), step],
        step_executions=[
            workflow_step_execution(
                0, Op.RUN_DCGM_DIAGNOSTIC, WorkflowStepStatus.SUCCEEDED
            ),
            workflow_step_execution(
                1, Op.VALIDATE_GPU, WorkflowStepStatus.FAILED, details=result.details
            ),
        ],
    )
    assert HardwareEscalationService.classify(workflow) is None


def test_widened_validation_does_not_apply_the_reporters_probe_to_other_nodes():
    h = ValidationHarness(Op.VALIDATE_HOST, nodes=["node-a", "node-b"])
    probe = {
        "metric": "swap_used_percent",
        "device": None,
        "node_id": "node-a",
        "comparison": "gte",
        "threshold": 80,
        "observed_at": h.action_at.isoformat(),
    }
    step = h.context.step.model_copy(
        update={"parameters": {HOST_DIAGNOSTIC_PROBE: probe}}
    )
    h.context = replace(h.context, step=step)
    h.feed.host_metric("node-a", "swap_used_percent", 20)
    h.feed.host["node-a"][-1].device = None
    assert h.execute().status is WorkflowStepStatus.SUCCEEDED


@pytest.mark.parametrize("probe", [{}, [], 0], ids=["empty-object", "list", "number"])
def test_invalid_probe_does_not_skip_validation(probe):
    h = ValidationHarness(Op.VALIDATE_HOST)
    step = h.context.step.model_copy(
        update={"parameters": {HOST_DIAGNOSTIC_PROBE: probe}}
    )
    h.context = replace(h.context, step=step)
    result = h.execute()
    assert result.status is WorkflowStepStatus.WAITING, result
    assert result.details[HOST_DIAGNOSTIC_FAILURE] is True


@pytest.mark.parametrize(
    "action,metric",
    [
        (RecoveryAction.NO_ACTION, "swap_used_percent"),
        (RecoveryAction.RUN_DIAGNOSTICS, "efa_inventory_mismatch"),
        (RecoveryAction.RUN_DIAGNOSTICS, "unrecognized_metric"),
        (RecoveryAction.RUN_DIAGNOSTICS, "memory_available_percent"),
    ],
)
def test_only_known_complete_diagnostic_rules_produce_a_probe(action, metric):
    h = ValidationHarness(Op.VALIDATE_HOST)
    finding = NodeHealthFinding(
        finding_id="probe",
        event_id="probe",
        cluster_id="cluster-a",
        node_id="node-a",
        observed_at=h.action_at,
        category="MEMORY",
        severity="warning",
        reason="test",
        recommended_action=action,
        metric_name=metric,
        value=0,
    )
    assert diagnostic_probe(finding) is None


def test_nonfinite_probe_threshold_is_not_a_health_verdict():
    h = ValidationHarness(Op.VALIDATE_HOST)
    pending, failure = validate_diagnostic_probe(
        {
            "metric": "swap_used_percent",
            "comparison": "gte",
            "threshold": float("nan"),
            "observed_at": h.action_at.isoformat(),
        },
        [],
    )
    assert pending == {"invalid_host_diagnostic_probe": True}
    assert failure is None
