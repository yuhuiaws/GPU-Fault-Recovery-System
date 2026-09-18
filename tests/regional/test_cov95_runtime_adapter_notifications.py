from __future__ import annotations

from dataclasses import replace

import pytest

from gpu_fault.adapters import NodeActionWorkflowAdapter
from gpu_fault.models import WorkflowOperation, WorkflowStepStatus
from gpu_fault.node_agent import NodeActionResult, NodeActionStatus, SignedNodeAction
from tests._builders import build_store, workflow_step_execution
from tests.execution.test_node_action_transport_retry import ENDPOINT, SECRET
from tests.regional._cov95_runtime_adapter import context_for
from tests.regional._cov95_runtime_support import offline_runtime as offline_runtime


def complete(endpoint: str, signed: SignedNodeAction) -> NodeActionResult:
    return NodeActionResult(
        command_id=signed.command.command_id,
        operation=signed.command.operation,
        status=NodeActionStatus.SUCCEEDED,
        details={"diagnostic_outcome": "PASS", "unit_evidence": "verified"},
    )


@pytest.mark.parametrize(
    "operation",
    [WorkflowOperation.RUN_DCGM_DIAGNOSTIC, WorkflowOperation.RESTART_FABRIC_MANAGER],
)
@pytest.mark.parametrize("mode", ["no-store", "persist", "notify"])
def test_success_notification_is_persisted_before_the_optional_delivery_callback(
    operation: WorkflowOperation, mode: str
) -> None:
    store = build_store()
    delivered = []

    def notify(identifier: str) -> None:
        stored = store.get_notification(identifier)
        assert stored.incident_id == "incident-a"
        delivered.append(identifier)

    adapter = NodeActionWorkflowAdapter(
        {"node-a": ENDPOINT},
        SECRET,
        sender=complete,
        store=None if mode == "no-store" else store,
        alert_sender=notify if mode == "notify" else None,
    )
    outcome = adapter.execute(context_for(adapter, operation, nodes=["node-a"]))
    assert outcome.status is WorkflowStepStatus.SUCCEEDED
    assert outcome.details["node_results"]["node-a"]["unit_evidence"] == "verified"
    if mode == "no-store":
        assert "notification_id" not in outcome.details
        assert store.list_notifications() == []
    else:
        (notification,) = store.list_notifications()
        assert outcome.details["notification_id"] == notification.notification_id
        assert notification.cluster_name == "cluster-a"
    assert delivered == (
        [outcome.details["notification_id"]] if mode == "notify" else []
    )


@pytest.mark.parametrize("prior_reset", [False, True])
def test_restore_retries_only_an_existing_completed_fabric_reset_notification(
    prior_reset: bool,
) -> None:
    store = build_store()
    delivered = []
    adapter = NodeActionWorkflowAdapter(
        {"node-a": ENDPOINT},
        SECRET,
        sender=complete,
        store=store,
        alert_sender=delivered.append,
    )
    reset_context = context_for(
        adapter, WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES, nodes=["node-a"]
    )
    reset = adapter.execute(reset_context)
    assert reset.status is WorkflowStepStatus.SUCCEEDED
    identifier = reset.details["notification_id"]
    restore_context = context_for(
        adapter, WorkflowOperation.RESTORE_GPU_SERVICES, nodes=["node-a"], step_index=1
    )
    if prior_reset:
        restore_context = replace(
            restore_context,
            workflow=restore_context.workflow.model_copy(
                update={
                    "step_executions": [
                        workflow_step_execution(
                            0,
                            WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES,
                            details=reset.details,
                        )
                    ]
                }
            ),
        )
    restored = adapter.execute(restore_context)
    assert restored.status is WorkflowStepStatus.SUCCEEDED
    assert delivered == [identifier] * (2 if prior_reset else 1)
    assert restored.details.get("fabric_reset_notification_id") == (
        identifier if prior_reset else None
    )
    assert len(store.list_notifications()) == 1
