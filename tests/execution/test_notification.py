from __future__ import annotations

from tests._builders import (
    build_store,
    copy_model,
    node_action_result,
    workflow_step_execution,
)

from ._support import (
    AdvisoryNotificationService,
    FakeCoreApi,
    KubernetesWorkflowAdapter,
    NodeActionWorkflowAdapter,
    NotificationResult,
    NotificationStatus,
    SupportEscalationAdapter,
    UnusedApi,
    WorkflowExecutionRequest,
    WorkflowOperation,
    WorkflowStepContext,
    WorkflowStepStatus,
    workflow_state,
)


def test_support_escalation_creates_fixed_ticket_notification() -> None:
    store = build_store()
    incident, workflow = workflow_state(store, [WorkflowOperation.ESCALATE_SUPPORT])
    incident = copy_model(
        incident,
        reasons=[
            "reset failed",
            "reboot failed",
            "healthy warm-spare replacement failed",
        ],
    )
    step = copy_model(
        workflow.official_steps[0],
        execution_owner="gpu-fault-support-escalation",
        node_ids=["node-a"],
        workload_ids=["training/pytorchjob/train"],
    )
    workflow = copy_model(workflow, official_steps=[step])
    sent = []
    adapter = SupportEscalationAdapter(store, alert_sender=sent.append)

    outcome = adapter.execute(
        WorkflowStepContext(
            workflow=workflow,
            incident=incident,
            step=step,
            step_index=0,
            request=WorkflowExecutionRequest(
                expected_fencing_token=workflow.fencing_token
            ),
            idempotency_key="support/incident-active",
        )
    )

    assert outcome.status is WorkflowStepStatus.SUCCEEDED
    assert outcome.details["hardware_disposition"] == ("OFFLINE_QUARANTINED")
    notification = store.list_notifications()[0]
    assert outcome.details["ticket_id"] in notification.body_text
    assert "实际失败的自动恢复步骤" in notification.body_text
    assert "NONE_RECORDED" in notification.body_text
    assert "内部厂商支持升级记录" in notification.body_text
    assert sent == [notification.notification_id]


def test_xid74_support_uses_nvlink_template() -> None:
    store = build_store()
    incident, workflow = workflow_state(store, [WorkflowOperation.ESCALATE_SUPPORT])
    incident = copy_model(
        incident,
        event_type="XID",
        official_action="WORKFLOW_NVLINK_ERR",
        reasons=["register4.bit18:fabric_reset_required"],
    )
    step = copy_model(
        workflow.official_steps[0],
        execution_owner="gpu-fault-support-escalation",
        node_ids=["node-a"],
    )
    workflow = copy_model(workflow, official_steps=[step])

    outcome = SupportEscalationAdapter(store).execute(
        WorkflowStepContext(
            workflow=workflow,
            incident=incident,
            step=step,
            step_index=0,
            request=WorkflowExecutionRequest(
                expected_fencing_token=workflow.fencing_token
            ),
            idempotency_key="support/xid74",
        )
    )

    assert outcome.status is WorkflowStepStatus.SUCCEEDED
    notification = store.list_notifications()[0]
    assert "XID 74 NVLink" in notification.subject
    assert "register4.bit18" in notification.body_text


def test_full_fabric_reset_sends_one_idempotent_email() -> None:
    store = build_store()
    incident, workflow = workflow_state(
        store, [WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES]
    )
    step = copy_model(
        workflow.official_steps[0],
        execution_owner="gpu-fault-node-agent",
        node_ids=["node-a"],
        gpu_uuids=["GPU-a", "GPU-b"],
        workload_ids=["training/pytorchjob/training-a"],
        parameters={
            "fabric_partition": "hp-cluster/node-a/local-nvswitch",
            "sxid": 10003,
        },
    )
    incident = copy_model(
        incident,
        event_id="kernel-log-sxid",
        event_type="SXID",
        policy_source="NVIDIA_FABRIC_MANAGER",
        official_action="RESET_ALL_GPUS_AND_NVSWITCHES",
        reasons=["fatal trunk-link SXID"],
    )
    workflow = copy_model(workflow, official_steps=[step])
    store.save_incident(incident)
    store.save_workflow(workflow)

    class RecordingNotifier:
        def __init__(self) -> None:
            self.deliveries = []

        def send(self, notification):
            self.deliveries.append(notification)
            return NotificationResult(
                notification_id=notification.notification_id,
                status=NotificationStatus.SENT,
                provider_message_id="ses-fabric-reset-1",
            )

    notifier = RecordingNotifier()
    service = AdvisoryNotificationService(store, notifier)

    def sender(_, envelope):
        return node_action_result(
            envelope.command.command_id,
            envelope.command.operation,
            details={
                "reset_scope": "ALL_LOCAL_GPUS_AND_NVSWITCHES",
                "reset_gpu_uuids": ["GPU-a", "GPU-b"],
                "verified_no_gpu_clients": True,
                "inventory_verified_after": True,
            },
        )

    adapter = NodeActionWorkflowAdapter(
        {"node-a": "http://node-a:9099"},
        "s" * 32,
        sender=sender,
        store=store,
        alert_sender=service.send,
    )
    context = WorkflowStepContext(
        workflow=workflow,
        incident=incident,
        step=step,
        step_index=0,
        request=WorkflowExecutionRequest(expected_fencing_token=3),
        idempotency_key=(f"{workflow.request_id}/0/RESET_ALL_GPUS_NVSWITCHES"),
    )

    first = adapter.execute(context)
    second = adapter.execute(context)

    assert first.status is WorkflowStepStatus.SUCCEEDED
    assert second.status is WorkflowStepStatus.SUCCEEDED
    assert len(store.list_notifications()) == 1
    assert len(notifier.deliveries) == 1
    assert "SXID：10003" in notifier.deliveries[0].body_text


def test_restore_retries_fabric_reset_email_after_network_recovers() -> None:
    store = build_store()
    incident, workflow = workflow_state(
        store,
        [
            WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES,
            WorkflowOperation.RESTORE_GPU_SERVICES,
        ],
    )
    reset_step = copy_model(
        workflow.official_steps[0],
        execution_owner="gpu-fault-node-agent",
        node_ids=["node-a"],
        gpu_uuids=["GPU-a"],
    )
    restore_step = copy_model(
        workflow.official_steps[1],
        execution_owner="gpu-fault-node-agent",
        node_ids=["node-a"],
        gpu_uuids=["GPU-a"],
    )
    workflow = copy_model(workflow, official_steps=[reset_step, restore_step])
    store.save_workflow(workflow)

    class RecoveringNotifier:
        def __init__(self) -> None:
            self.attempts = 0

        def send(self, notification):
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("network unavailable")
            return NotificationResult(
                notification_id=notification.notification_id,
                status=NotificationStatus.SENT,
                provider_message_id="ses-reset-retry-1",
            )

    notifier = RecoveringNotifier()
    service = AdvisoryNotificationService(store, notifier)

    def sender(_, envelope):
        return node_action_result(
            envelope.command.command_id,
            envelope.command.operation,
            details={"operation": envelope.command.operation.value},
        )

    adapter = NodeActionWorkflowAdapter(
        {"node-a": "http://node-a:9099"},
        "s" * 32,
        sender=sender,
        store=store,
        alert_sender=service.send,
    )
    request = WorkflowExecutionRequest(expected_fencing_token=3)
    reset = adapter.execute(
        WorkflowStepContext(
            workflow=workflow,
            incident=incident,
            step=reset_step,
            step_index=0,
            request=request,
            idempotency_key="workflow/reset",
        )
    )
    notification_id = reset.details["notification_id"]
    assert (
        store.get_notification_result(notification_id).status
        is NotificationStatus.FAILED
    )

    workflow = copy_model(
        workflow,
        step_executions=[
            workflow_step_execution(
                0,
                WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES,
                details={"notification_id": notification_id},
            )
        ],
    )
    restored = adapter.execute(
        WorkflowStepContext(
            workflow=workflow,
            incident=incident,
            step=restore_step,
            step_index=1,
            request=request,
            idempotency_key="workflow/restore",
        )
    )

    assert restored.status is WorkflowStepStatus.SUCCEEDED
    assert restored.details["fabric_reset_notification_id"] == (notification_id)
    assert notifier.attempts == 2
    assert (
        store.get_notification_result(notification_id).status is NotificationStatus.SENT
    )


def test_storeless_mechanical_inspection_uses_notification_sink() -> None:
    class NotificationSink:
        def __init__(self) -> None:
            self.notifications = []

        def save_notification_if_absent(self, notification):
            self.notifications.append(notification)
            return notification

    core = FakeCoreApi()
    sink = NotificationSink()
    adapter = KubernetesWorkflowAdapter(
        core_api=core,
        batch_api=UnusedApi(),
        custom_api=UnusedApi(),
        store=None,
        notification_sink=sink,
    )
    store = build_store()
    incident, workflow = workflow_state(store, [WorkflowOperation.CHECK_MECHANICALS])
    step = copy_model(workflow.official_steps[0], execution_owner=adapter.owner)

    waiting = adapter.execute(
        WorkflowStepContext(
            workflow=workflow,
            incident=incident,
            step=step,
            step_index=0,
            request=WorkflowExecutionRequest(
                expected_fencing_token=workflow.fencing_token
            ),
            idempotency_key="mechanical/storeless",
        )
    )

    assert waiting.status is WorkflowStepStatus.WAITING
    assert waiting.details["notification_id"] == (sink.notifications[0].notification_id)
    assert sink.notifications[0].cluster_name == incident.cluster_id
