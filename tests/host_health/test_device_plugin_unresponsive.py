"""``*_device_plugin_unresponsive`` grades like the allocatable rule it
complements: CRITICAL, a device plugin restart, the probe's labels on the
finding (GF-GENERAL-LIVE-014 hung plugin)."""

from __future__ import annotations

import pytest

from gpu_fault.host_health import HostMetricSample, NodeHealthCategory, NodeHealthPolicy
from gpu_fault.models import RecoveryAction, Severity
from tests._builders import build_store, host_telemetry_batch
from tests.host_health._support import NOW

LABELS = {
    "resource": "GPU",
    "resource_name": "nvidia.com/gpu",
    "node_instance_type": "p5en.48xlarge",
    "sockets": "nvidia-gpu.sock",
    "probe_state": "TIMEOUT",
    "probe_detail": "nvidia-gpu.sock: no bytes from the plugin within 5s",
    "expected_count": "8",
    "failure_mode": "DEVICE_PLUGIN_UNRESPONSIVE",
}


@pytest.mark.parametrize(
    ("metric", "resource", "action", "category", "fault_class"),
    [
        (
            "gpu_device_plugin_unresponsive",
            "nvidia.com/gpu",
            RecoveryAction.RESTART_GPU_DEVICE_PLUGIN,
            NodeHealthCategory.GPU,
            "GPU_INVENTORY",
        ),
        (
            "efa_device_plugin_unresponsive",
            "vpc.amazonaws.com/efa",
            RecoveryAction.RESTART_EFA_DEVICE_PLUGIN,
            NodeHealthCategory.RDMA,
            "NETWORK_FABRIC",
        ),
    ],
)
def test_a_hung_device_plugin_is_a_critical_plugin_restart(
    metric, resource, action, category, fault_class
) -> None:
    policy = NodeHealthPolicy(build_store())
    labels = {**LABELS, "resource_name": resource}

    findings = policy.evaluate_metrics(
        host_telemetry_batch(
            "hung-plugin",
            NOW,
            [HostMetricSample(name=metric, value=1, labels=labels)],
            runtime_profile_version="simulated-v1",
        )
    )

    assert len(findings) == 1, findings
    finding = findings[0]
    assert finding.recommended_action is action
    assert finding.severity is Severity.CRITICAL
    assert finding.category is category
    assert finding.metric_name == metric
    assert finding.fault_class() == fault_class
    assert "does not answer" in finding.reason and resource in finding.reason
    assert finding.diagnostic_parameters == labels, (
        "the probe's evidence travels with the finding: the planner reads "
        "expected_count, the operator reads which socket and how long"
    )


def test_a_plugin_that_answers_raises_nothing() -> None:
    policy = NodeHealthPolicy(build_store())

    findings = policy.evaluate_metrics(
        host_telemetry_batch(
            "live-plugin",
            NOW,
            [
                HostMetricSample(
                    name="gpu_device_plugin_unresponsive",
                    value=0,
                    labels={
                        **LABELS,
                        "probe_state": "RESPONDED",
                        "failure_mode": "HEALTHY",
                    },
                )
            ],
            runtime_profile_version="simulated-v1",
        )
    )

    assert findings == []
