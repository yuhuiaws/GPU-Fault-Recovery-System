from __future__ import annotations

from datetime import timedelta

import pytest

from gpu_fault.host_health import NodeHealthCategory
from gpu_fault.models import RecoveryAction, Severity, WorkflowOperation
from gpu_fault.notification_service import AdvisoryNotificationService
from gpu_fault.store import InMemoryStore
from tests._builders import (
    fault_incident,
    node_health_finding,
    workflow_request,
    workflow_step,
)
from tests.completion._support import NOW
from tests.notifications._support import RecordingNotifier


@pytest.mark.parametrize("with_workflow", [True, False])
@pytest.mark.parametrize("kind", ["efa", "inventory", "host"])
def test_preview_uses_real_finding_and_workflow_contracts_without_sending(
    kind, with_workflow
):
    store, notifier = InMemoryStore(), RecordingNotifier()
    workflow_id = "workflow-local" if with_workflow else None
    store.save_incident(
        fault_incident("incident-local", "event-local", workflow_request_id=workflow_id)
    )
    if with_workflow:
        store.save_workflow(
            workflow_request(
                workflow_id,
                "incident-local",
                official_steps=[workflow_step(WorkflowOperation.RUN_DCGM_DIAGNOSTIC)],
            )
        )
    finding = node_health_finding(
        "finding-local",
        "event-local",
        observed_at=NOW,
        category=NodeHealthCategory.RDMA if kind == "efa" else NodeHealthCategory.CPU,
        severity=Severity.WARNING,
        reason="unit sustained health risk",
        recommended_action=RecoveryAction.REBOOT_NODE,
        evidence_ref="s3://local/evidence",
        metric_name="efa_traffic_bytes_per_second"
        if kind == "efa"
        else "gpu_inventory_count",
        affected_workload_ids=["training/job/train"],
        diagnostic_parameters={
            "signal": "unit-signal",
            "expected_count": 8,
            "observed_count": 7,
        },
    )
    service = AdvisoryNotificationService(store, notifier)
    preview = {
        "efa": service.preview_efa_rdma_event,
        "inventory": service.preview_hardware_inventory_event,
        "host": service.preview_host_resource_event,
    }[kind]
    notification = preview("incident-local", finding)
    repeated = preview("incident-local", finding)
    assert repeated.notification_id == notification.notification_id
    assert notification.evidence_refs == ["s3://local/evidence"]
    assert notification.cluster_name == finding.cluster_id
    assert finding.node_id in notification.body_text
    if with_workflow:
        assert workflow_id in notification.body_text
        assert "RUN_DCGM_DIAGNOSTIC" in notification.body_text
    assert notifier.notifications == []
    assert len(store.list_notifications()) == 1
    if kind == "host":
        assert notification.category == "HEALTH_TREND"
        assert notification.priority == 200
        next_bucket = finding.model_copy(
            update={"event_id": "later-event", "observed_at": NOW + timedelta(hours=2)}
        )
        assert (
            preview("incident-local", next_bucket).notification_id
            != notification.notification_id
        )
    elif kind == "efa":
        assert notification.category == "FAULT_DETECTED"
        assert notification.priority == 0


def test_managed_action_preview_selects_priority_without_taking_over_execution():
    store, notifier = InMemoryStore(), RecordingNotifier()
    store.save_incident(
        fault_incident(
            "incident-local",
            "event-local",
            workflow_request_id="workflow-local",
            effective_action=RecoveryAction.REBOOT_NODE,
        )
    )
    store.save_workflow(
        workflow_request(
            "workflow-local",
            "incident-local",
            official_steps=[
                workflow_step(
                    WorkflowOperation.RESTART_WORKLOAD, "hyperpod-managed-workload"
                ),
                workflow_step(WorkflowOperation.RESTART_NODE, "hyperpod-managed-node"),
            ],
        )
    )
    service = AdvisoryNotificationService(store, notifier)
    notification = service.preview("incident-local")
    assert notification.incident_id == "incident-local"
    assert notification.cluster_name == "cluster-a"
    assert "REBOOT_NODE" in notification.body_text
    assert "hyperpod-managed-node" in notification.body_text
    assert (
        service.preview("incident-local").notification_id
        == notification.notification_id
    )
    assert notifier.notifications == []
    assert store.get_workflow("workflow-local").status.value == "PENDING"
