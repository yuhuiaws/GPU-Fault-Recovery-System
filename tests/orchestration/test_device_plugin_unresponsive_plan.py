"""A ``*_device_plugin_unresponsive`` finding plans the same plugin restart as
an allocatable loss, with one difference the restart step must be told about:
the hung plugin still advertises every device, so the step is stamped
``plugin_unresponsive`` and must not read the full allocatable as healthy."""

from __future__ import annotations

import pytest

from gpu_fault.device_plugin_liveness import (
    DEVICE_PLUGIN_UNRESPONSIVE_METRICS,
    NODE_DEVICE_RESOURCE_METRICS,
)
from gpu_fault.host_health import NodeHealthCategory
from gpu_fault.models import RecoveryAction, WorkflowOperation
from gpu_fault.orchestration.families.conflicts import (
    DEVICE_PLUGIN_UNAVAILABLE_OPERATIONS,
    INVENTORY_FINDING_VALIDATIONS,
    TRANSIENT_INVENTORY_FINDINGS,
)
from gpu_fault.orchestration.families.validation import ValidationOperationService
from tests.orchestration._cov95_orch_extra_health import health_service
from tests.orchestration._cov95_orch_extra_support import finding, memory_store

PARAMETERS = {
    "resource": "GPU",
    "node_instance_type": "p5en.48xlarge",
    "sockets": "nvidia-gpu.sock",
    "probe_state": "TIMEOUT",
    "expected_count": "8",
    "failure_mode": "DEVICE_PLUGIN_UNRESPONSIVE",
}


def _plan(metric: str, action: RecoveryAction, category: NodeHealthCategory):
    service, _absorbed = health_service(memory_store())
    _, workflow = service.ingest(
        finding(
            recommended_action=action,
            category=category,
            severity="critical",
            metric_name=metric,
            diagnostic_parameters=PARAMETERS,
        )
    )
    return workflow


@pytest.mark.parametrize(
    ("metric", "action", "category", "plugin", "validate"),
    [
        (
            "gpu_device_plugin_unresponsive",
            RecoveryAction.RESTART_GPU_DEVICE_PLUGIN,
            NodeHealthCategory.GPU,
            WorkflowOperation.RESTART_GPU_DEVICE_PLUGIN,
            WorkflowOperation.VALIDATE_GPU,
        ),
        (
            "efa_device_plugin_unresponsive",
            RecoveryAction.RESTART_EFA_DEVICE_PLUGIN,
            NodeHealthCategory.RDMA,
            WorkflowOperation.RESTART_EFA_DEVICE_PLUGIN,
            WorkflowOperation.VALIDATE_FABRIC,
        ),
    ],
)
def test_the_restart_step_is_told_the_plugin_is_hung_and_may_escalate_to_reboot(
    metric, action, category, plugin, validate
) -> None:
    workflow = _plan(metric, action, category)

    operations = [step.operation for step in workflow.official_steps]
    assert operations[-3:] == [
        plugin,
        WorkflowOperation.TRIGGER_HEALTH_SNAPSHOT,
        validate,
    ], operations
    restart = next(step for step in workflow.official_steps if step.operation is plugin)
    assert restart.parameters["expected_count"] == 8
    assert restart.parameters["plugin_unresponsive"] is True
    assert restart.parameters["failure_escalation_action"] == "REBOOT_NODE", (
        "a replacement Pod kubelet never starts (its sync loop blocked in the "
        "hung plugin's Allocate) ends in the reboot the allocatable rule also owns"
    )
    validation = next(
        step for step in workflow.official_steps if step.operation is validate
    )
    assert validation.parameters["failure_escalation_action"] == "REBOOT_NODE"


def test_an_allocatable_loss_is_not_stamped_as_a_hung_plugin() -> None:
    workflow = _plan(
        "gpu_kubernetes_allocatable_mismatch",
        RecoveryAction.RESTART_GPU_DEVICE_PLUGIN,
        NodeHealthCategory.GPU,
    )

    restart = next(
        step
        for step in workflow.official_steps
        if step.operation is WorkflowOperation.RESTART_GPU_DEVICE_PLUGIN
    )
    assert "plugin_unresponsive" not in restart.parameters
    assert restart.parameters["expected_count"] == 8


def test_the_validation_checks_the_inventory_the_hung_plugin_should_serve() -> None:
    gpu = finding(
        recommended_action=RecoveryAction.RESTART_GPU_DEVICE_PLUGIN,
        metric_name="gpu_device_plugin_unresponsive",
        diagnostic_parameters=PARAMETERS,
    )
    efa = finding(
        recommended_action=RecoveryAction.RESTART_EFA_DEVICE_PLUGIN,
        category=NodeHealthCategory.RDMA,
        metric_name="efa_device_plugin_unresponsive",
        diagnostic_parameters={**PARAMETERS, "resource": "EFA", "expected_count": "16"},
    )

    gpu_requirement = ValidationOperationService.inventory_parameters(gpu)[
        "inventory_requirements_by_node"
    ][gpu.node_id]
    assert gpu_requirement == {
        "metrics": {"gpu_inventory_active_count": 8},
        "node_instance_type": "p5en.48xlarge",
    }, "VALIDATE_GPU judges the GPUs the restarted plugin must serve again"
    efa_requirement = ValidationOperationService.inventory_parameters(efa)[
        "inventory_requirements_by_node"
    ][efa.node_id]
    assert efa_requirement["metrics"] == {"efa_inventory_active_count": 16}


def test_the_liveness_findings_share_the_device_resource_family() -> None:
    assert DEVICE_PLUGIN_UNRESPONSIVE_METRICS == {
        "gpu_device_plugin_unresponsive",
        "efa_device_plugin_unresponsive",
    }
    assert DEVICE_PLUGIN_UNRESPONSIVE_METRICS < NODE_DEVICE_RESOURCE_METRICS
    assert {
        "efa_inventory_mismatch",
        "efa_kubernetes_allocatable_mismatch",
        "gpu_kubernetes_allocatable_mismatch",
    } < NODE_DEVICE_RESOURCE_METRICS
    for metric in DEVICE_PLUGIN_UNRESPONSIVE_METRICS:
        assert (
            DEVICE_PLUGIN_UNAVAILABLE_OPERATIONS <= TRANSIENT_INVENTORY_FINDINGS[metric]
        ), (
            "a probe that times out while the chain stops, restores or replaces "
            "the plugin is the chain's own side effect"
        )
        assert WorkflowOperation.VALIDATE_GPU in INVENTORY_FINDING_VALIDATIONS[metric]
    assert INVENTORY_FINDING_VALIDATIONS["efa_device_plugin_unresponsive"] == {
        WorkflowOperation.VALIDATE_GPU,
        WorkflowOperation.VALIDATE_FABRIC,
    }
