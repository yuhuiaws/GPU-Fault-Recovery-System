"""The control plane mails DCGM results and full-fabric resets for regional
executors too (ARCH-E E7).

The node-action adapter builds both notifications itself, but only when it has
a Store -- and the regional cluster executor constructs it with ``store=None``,
so on every regional deployment those two mails were never produced. The
control-plane side, ``dispatch_remote_completion``, covered RESTART_FABRIC_MANAGER
and RESET_GPU only. It now builds the other two from the persisted remote
command result, with the same deduplication keys the adapter would have used.
"""

from __future__ import annotations

from gpu_fault.models import (
    IncidentState,
    NotificationStatus,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepSpec,
)
from gpu_fault.notification_service import AdvisoryNotificationService
from gpu_fault.notifications.registry import NotificationKind
from gpu_fault.regional import RemoteActionCommand, RemoteCommandStatus
from tests._builders import build_store, fault_incident
from tests.notifications._support import RecordingNotifier


def _command(
    store,
    operation: WorkflowOperation,
    *,
    status: RemoteCommandStatus,
    result_details: dict,
    parameters: dict | None = None,
    error: str | None = None,
) -> RemoteActionCommand:
    step = WorkflowStepSpec(
        operation=operation,
        execution_owner="gpu-fault-node-agent",
        node_ids=["node-a"],
        workload_ids=["job-a"],
        parameters=parameters or {},
    )
    workflow = WorkflowRequest(
        request_id=f"workflow-{operation.value.lower()}",
        incident_id=f"incident-{operation.value.lower()}",
        status=WorkflowStatus.RUNNING,
        official_action="DIAGNOSE",
        fencing_token=1,
        official_steps=[step],
    )
    incident = fault_incident(
        workflow.incident_id,
        f"event-{operation.value.lower()}",
        state=IncidentState.ACTION_PENDING,
        workflow_request_id=workflow.request_id,
        official_action="DIAGNOSE",
        reasons=["XID 79"],
        fencing_token=1,
    )
    store.save_incident(incident)
    store.save_workflow(workflow)
    return RemoteActionCommand(
        command_id=f"remote-{operation.value.lower()}",
        cluster_id=incident.cluster_id,
        workflow_request_id=workflow.request_id,
        incident_id=incident.incident_id,
        step_index=0,
        fencing_token=1,
        idempotency_key=f"{workflow.request_id}/0/{operation.value}",
        step=step,
        workflow=workflow,
        incident=incident,
        status=status,
        result_details=result_details,
        error=error,
    )


DCGM_NODE_RESULTS = {
    "node-a": {
        "diagnostic_outcome": "FAIL",
        "diagnostic_findings": [
            {
                "test_name": "memory",
                "status": "FAIL",
                "entities": ["GPU-0"],
                "error_codes": ["DCGM_FR_ECC_UNCORRECTABLE"],
                "messages": ["uncorrectable ECC"],
            }
        ],
        "recommended_actions": [
            {
                "priority": "HIGH",
                "action_code": "REPLACE_GPU",
                "trigger_tests": ["memory"],
                "instruction": "open a support case",
            }
        ],
        "evidence_ref": "s3://evidence/dcgm-node-a.json",
    }
}


def _service(store) -> tuple[AdvisoryNotificationService, RecordingNotifier]:
    notifier = RecordingNotifier()
    return (
        AdvisoryNotificationService(store, notifier, async_delivery=False),
        notifier,
    )


def test_a_failed_regional_dcgm_diagnostic_is_mailed_from_the_persisted_result():
    store = build_store()
    service, notifier = _service(store)
    command = _command(
        store,
        WorkflowOperation.RUN_DCGM_DIAGNOSTIC,
        status=RemoteCommandStatus.FAILED,
        error="DCGM diagnostic failed or was inconclusive on node-a",
        result_details={
            "node_results": DCGM_NODE_RESULTS,
            "control_plane_action": "DRAIN_AND_QUARANTINE",
        },
    )

    results = service.dispatch_remote_completion(command)

    assert [item.status for item in results] == [NotificationStatus.SENT], results
    [notification] = store.list_notifications()
    assert "DCGM" in notification.subject, notification.subject
    assert "FAIL" in notification.subject, notification.subject
    assert "DRAIN_AND_QUARANTINE" in notification.body_text
    assert "REPLACE_GPU" in notification.body_text
    assert notification.evidence_refs == ["s3://evidence/dcgm-node-a.json"]
    assert notification.deduplication_key == (
        f"{command.cluster_id}/{command.incident_id}/dcgm-diagnostic/"
        f"{command.idempotency_key}"
    ), "the key must match the adapter's so the two paths never mail twice"
    assert len(notifier.notifications) == 1


def test_a_passed_regional_dcgm_diagnostic_is_mailed_once():
    store = build_store()
    service, notifier = _service(store)
    command = _command(
        store,
        WorkflowOperation.RUN_DCGM_DIAGNOSTIC,
        status=RemoteCommandStatus.SUCCEEDED,
        result_details={
            "node_results": {"node-a": {"diagnostic_outcome": "PASS"}},
            "control_plane_action": "COOLDOWN_AND_VALIDATE",
        },
    )

    first = service.dispatch_remote_completion(command)
    second = service.dispatch_remote_completion(command)

    assert [item.status for item in first] == [NotificationStatus.SENT], first
    assert [item.status for item in second] == [NotificationStatus.DUPLICATE], second
    [notification] = store.list_notifications()
    assert "PASS" in notification.subject, notification.subject
    assert "COOLDOWN_AND_VALIDATE" in notification.body_text
    assert len(notifier.notifications) == 1


def test_a_dcgm_result_without_node_results_is_not_mailed():
    """An executor-side rejection carries no diagnostic; there is nothing to say."""
    store = build_store()
    service, notifier = _service(store)
    command = _command(
        store,
        WorkflowOperation.RUN_DCGM_DIAGNOSTIC,
        status=RemoteCommandStatus.FAILED,
        error="ClusterExecutorError: no adapter",
        result_details={},
    )

    assert service.dispatch_remote_completion(command) == []
    assert store.list_notifications() == []
    assert notifier.notifications == []


def test_a_regional_full_fabric_reset_is_mailed_from_the_step_parameters():
    store = build_store()
    service, notifier = _service(store)
    command = _command(
        store,
        WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES,
        status=RemoteCommandStatus.SUCCEEDED,
        parameters={
            "fabric_partition": "cluster-a/node-a/local-nvswitch",
            "sxid": 10003,
        },
        result_details={
            "node_results": {"node-a": {"reset": True, "gpus": 8, "nvswitches": 4}}
        },
    )

    first = service.dispatch_remote_completion(command)
    second = service.dispatch_remote_completion(command)

    assert [item.status for item in first] == [NotificationStatus.SENT], first
    assert [item.status for item in second] == [NotificationStatus.DUPLICATE], second
    [notification] = store.list_notifications()
    assert "NVSwitch" in notification.subject, notification.subject
    assert "10003" in notification.body_text
    assert "cluster-a/node-a/local-nvswitch" in notification.body_text
    assert len(notifier.notifications) == 1


def test_an_adapter_produced_notification_is_sent_not_rebuilt():
    """A single-cluster executor already saved and named the notification."""
    store = build_store()
    service, notifier = _service(store)
    command = _command(
        store,
        WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES,
        status=RemoteCommandStatus.SUCCEEDED,
        parameters={"fabric_partition": "p", "sxid": 10003},
        result_details={"node_results": {"node-a": {"reset": True}}},
    )
    existing = service.builders.build(
        NotificationKind.FABRIC_RESET_COMPLETED,
        cluster_id=command.cluster_id,
        incident_id=command.incident.incident_id,
        workflow_id=command.workflow.request_id,
        event_id=command.incident.event_id,
        event_type=command.incident.event_type,
        policy_source=command.incident.policy_source,
        official_action=command.incident.official_action,
        reasons=command.incident.reasons,
        operation_id=command.idempotency_key,
        node_results=command.result_details["node_results"],
        workload_ids=command.step.workload_ids,
        fabric_partition="p",
        sxid=10003,
    )
    existing = store.save_notification_if_absent(existing)
    command = command.model_copy(
        update={
            "result_details": {
                **command.result_details,
                "notification_id": existing.notification_id,
            }
        }
    )

    results = service.dispatch_remote_completion(command)

    assert [item.status for item in results] == [NotificationStatus.SENT], results
    assert len(store.list_notifications()) == 1
    assert len(notifier.notifications) == 1
