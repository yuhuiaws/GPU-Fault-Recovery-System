from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from threading import Barrier

import httpx
import pytest

from gpu_fault.app import ApplicationContext, create_app
from gpu_fault.models import (
    AdvisoryNotification,
    IncidentState,
    NotificationStatus,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepSpec,
)
from gpu_fault.notification_service import AdvisoryNotificationService
from gpu_fault.notifications import (
    FABRIC_RESET_EMAIL_TEMPLATE,
    FABRIC_RESET_TEMPLATE_VERSION,
    GPU_COUNT_CHANGE_EMAIL_TEMPLATE,
    GPU_RESET_EMAIL_TEMPLATE,
    GPU_RESET_TEMPLATE_VERSION,
    NOT_APPLICABLE_EMAIL_TEMPLATE,
    NOT_APPLICABLE_TEMPLATE_VERSION,
    NVLINK74_SUPPORT_TEMPLATE_VERSION,
    RESTART_FABRIC_MANAGER_EMAIL_TEMPLATE,
    RESTART_FABRIC_MANAGER_TEMPLATE_VERSION,
    RESTART_GUARD_TEMPLATE_VERSION,
    RESTART_NODE_EMAIL_TEMPLATE,
    RESTART_NODE_TEMPLATE_VERSION,
    RESTART_WORKLOAD_EMAIL_TEMPLATE,
    RESTART_WORKLOAD_TEMPLATE_VERSION,
    SXID_EVENT_TEMPLATE_VERSION,
    XID_INVESTIGATORY_EMAIL_TEMPLATE,
    XID_INVESTIGATORY_TEMPLATE_VERSION,
    DisabledNotificationNotifier,
    HyperPodAdvisoryEmailBuilder,
    NotApplicableEmailBuilder,
    Nvlink74MechanicalEmailBuilder,
    Nvlink74SupportEmailBuilder,
    RestartGuardEmailBuilder,
    SesEmailNotifier,
    SesNotificationConfig,
    WarmSpareReplacementEmailBuilder,
    XidInvestigatoryEmailBuilder,
)
from gpu_fault.regional import RemoteActionCommand, RemoteCommandStatus
from gpu_fault.store import InMemoryStore, SqliteStore
from tests._builders import build_store, fault_incident
from tests.notifications._support import (
    FakeSesV2Client,
    PartiallyFailingNotifier,
    RecordingNotifier,
    _notification_marker,
    advisory,
)


def test_store_labels_drill_notification_from_incident() -> None:
    store = build_store()
    incident = fault_incident(
        "incident-drill",
        "event-drill",
        "TEST",
        policy_version="test",
        policy_source="test",
        drill_id="maintenance-42",
    )
    store.save_incident(incident)

    saved = store.save_notification_if_absent(
        AdvisoryNotification(
            deduplication_key="event-drill/notification",
            cluster_name="cluster-a",
            incident_id=incident.incident_id,
            subject="[GPU fault] reset completed",
            body_text="reset completed",
            support_case_draft="",
        )
    )

    assert saved.drill_id == "maintenance-42"
    assert saved.subject.startswith("[DRILL:maintenance-42]"), (
        'expected saved.subject.startswith("[DRILL:maintenance-42]") to be truthy'
    )
    assert "演练通知 / DRILL - 非真实故障" in saved.body_text
    assert "Drill ID：maintenance-42" in saved.body_text


def test_store_labels_performance_notification_without_incident() -> None:
    store = build_store()

    saved = store.save_notification_if_absent(
        AdvisoryNotification(
            deduplication_key="perf-cap-000/collector-silent",
            cluster_name="perf-cap-000",
            incident_id="incident-not-persisted",
            subject="[GPU fault] collector silent",
            body_text="collector silent",
            support_case_draft="",
        )
    )

    assert saved.drill_id == "perf-capacity"
    assert saved.subject.startswith("[DRILL:perf-capacity]"), (
        "performance notification subject was not labeled as a drill"
    )
    assert "演练通知 / DRILL - 非真实故障" in saved.body_text


def test_remote_fabric_manager_completion_creates_one_drill_notification() -> None:
    store = build_store()
    notifier = RecordingNotifier()
    service = AdvisoryNotificationService(store, notifier, async_delivery=False)
    step = WorkflowStepSpec(
        operation=WorkflowOperation.RESTART_FABRIC_MANAGER,
        execution_owner="gpu-fault-node-agent",
        node_ids=["node-a"],
        gpu_uuids=["GPU-a"],
    )
    workflow = WorkflowRequest(
        request_id="workflow-fm",
        incident_id="incident-fm",
        status=WorkflowStatus.SUCCEEDED,
        official_action="RESTART_FM",
        fencing_token=1,
        official_steps=[step],
    )
    incident = fault_incident(
        "incident-fm",
        "event-fm",
        state=IncidentState.RECOVERED,
        workflow_request_id=workflow.request_id,
        official_action="RESTART_FM",
        reasons=["solo XID 45"],
        fencing_token=1,
        drill_id="destr010-test",
    )
    store.save_incident(incident)
    store.save_workflow(workflow)
    command = RemoteActionCommand(
        command_id="remote-fm",
        cluster_id=incident.cluster_id,
        workflow_request_id=workflow.request_id,
        incident_id=incident.incident_id,
        step_index=0,
        fencing_token=1,
        idempotency_key="workflow-fm/0/RESTART_FABRIC_MANAGER",
        step=step,
        workflow=workflow,
        incident=incident,
        status=RemoteCommandStatus.SUCCEEDED,
        result_details={
            "node_results": {
                "node-a": {
                    "active": True,
                    "previous_main_pid": "101",
                    "current_main_pid": "202",
                    "service": "nvidia-fabricmanager",
                }
            }
        },
    )

    first = service.dispatch_remote_completion(command)
    second = service.dispatch_remote_completion(command)

    notifications = store.list_notifications()
    assert len(notifications) == 1
    assert notifications[0].drill_id == "destr010-test"
    assert "Fabric Manager" in notifications[0].subject
    assert notifications[0].category == "ACTION_COMPLETED"
    result = store.get_notification_result(notifications[0].notification_id)
    assert result is not None
    assert result.status is NotificationStatus.SKIPPED
    assert notifier.notifications == []
    assert [item.status for item in first + second] == [
        NotificationStatus.SKIPPED,
        NotificationStatus.SKIPPED,
    ]


def test_warm_spare_replacement_email_lists_rebinding() -> None:
    notification = WarmSpareReplacementEmailBuilder().build(
        cluster_id="cluster-a",
        incident_id="incident-a",
        workflow_id="workflow-a",
        event_id="event-a",
        policy_source="SITE_NODE_HEALTH",
        official_action="REPLACE_NODE",
        effective_action="REPLACE_NODE",
        reasons=["unrecoverable node fault"],
        operation_id="workflow-a/4/REPLACE_NODE",
        fault_node_ids=["node-old"],
        spare_node_ids=["node-spare"],
        node_rebindings={"node-old": "node-spare"},
        confirmation_source="healthy-running-warm-spare",
        provider_mutation_submitted=False,
    )

    assert "warm-spare替换成功" in notification.subject
    assert "node-old -> node-spare" in notification.body_text
    assert "Provider replacement API submitted：false" in (notification.body_text)
    assert "incident-a" in notification.body_text
    assert "workflow-a" in notification.body_text
    assert "healthy-running-warm-spare" in notification.body_text
    assert "GPU/Fabric validation" in notification.body_text
    assert "Workflow 最终状态为准" in notification.body_text
    assert notification.deduplication_key.endswith(
        "/warm-spare-replacement/workflow-a/4/REPLACE_NODE"
    ), (
        'expected notification.deduplication_key.endswith( "/warm-spare-replacement/workflow-a/4/REPLACE_NODE" ) to be truthy'
    )


def test_xid74_support_email_uses_fixed_decoded_template() -> None:
    notification = Nvlink74SupportEmailBuilder().build(
        cluster_id="hp-cluster",
        incident_id="incident-xid74",
        workflow_id="workflow-xid74",
        event_id="kmsg-xid74",
        node_ids=["worker-1"],
        workload_ids=["training/pytorchjob/train-a"],
        reasons=["register4.bit18:fabric_reset_required"],
        policy_source="NVIDIA_CATALOG",
        official_action="WORKFLOW_NVLINK_ERR",
        ticket_id="vendor-ticket-incident-xid74",
    )

    assert NVLINK74_SUPPORT_TEMPLATE_VERSION in notification.body_text
    assert "register4.bit18" in notification.body_text
    assert "reset、节点 reboot" not in notification.body_text
    assert "未执行 XID 74 单 GPU reset" in notification.body_text
    assert "unschedulable 和 quarantine" not in notification.body_text


def test_xid74_mechanical_email_uses_fixed_acknowledgement() -> None:
    notification = Nvlink74MechanicalEmailBuilder().build(
        cluster_id="hp-cluster",
        incident_id="incident-xid74",
        workflow_id="workflow-xid74",
        node_ids=["worker-1"],
        link_id=3,
        pci_bdf="0000:59:00.0",
        occurrence_counts={"register1.bit8": 1},
        annotation=("gpu-fault.io/mechanical-inspection-complete"),
        annotation_value="incident-xid74:2",
    )

    assert "xid74-mechanical-zh-v1" in notification.body_text
    assert "register1.bit8=1" in notification.body_text
    assert "incident-xid74:2" in notification.body_text
    assert "不会自动确认" in notification.body_text


def test_xid_investigatory_email_uses_fixed_field_template() -> None:
    notification = XidInvestigatoryEmailBuilder().build(
        cluster_id="hp-cluster",
        node_id="worker-1",
        incident_id="incident-xid11",
        workflow_id="workflow-xid11",
        event_id="kmsg-xid11",
        xid=11,
        product="H200",
        observed_at="2026-07-26T12:00:00+00:00",
        workload_state="ACTIVE",
        workload_ids=["training/pytorchjob/train-a"],
        disposition="EXECUTABLE",
        official_action="RESTART_APP",
        effective_action="RESTART_WORKLOAD",
        investigatory_action="CHECK_APP/CUDA",
        policy_source="NVIDIA_CATALOG",
        policy_version="nvidia-xid-catalog-610",
        reasons=["exact NVIDIA Catalog 610 Immediate Action"],
        evidence_refs=["kmsg://worker-1/boot-1/11"],
    )

    assert XID_INVESTIGATORY_TEMPLATE_VERSION in notification.body_text
    assert "NVIDIA Immediate Action：RESTART_APP" in notification.body_text
    assert "NVIDIA Investigatory Action：CHECK_APP/CUDA" in (notification.body_text)
    assert "不会自动执行 Investigatory Action" in notification.body_text
    assert "{investigatory_action}" in XID_INVESTIGATORY_EMAIL_TEMPLATE
    assert "prompt" not in XID_INVESTIGATORY_EMAIL_TEMPLATE.lower()


@pytest.mark.parametrize("durable", [False, True])
def test_marker_lookup_is_scoped_to_one_incident(tmp_path, durable: bool) -> None:
    store = SqliteStore(str(tmp_path / "marker-scope.db")) if durable else build_store()
    try:
        store.add_marker(_notification_marker("marker-a-late", "incident-a", seconds=2))
        store.add_marker(_notification_marker("marker-b", "incident-b", seconds=1))
        store.add_marker(
            _notification_marker("marker-a-early", "incident-a", seconds=0)
        )

        markers = store.list_markers_for_incident("incident-a")
    finally:
        if durable:
            store.close()

    assert [marker.marker_id for marker in markers] == [
        "marker-a-early",
        "marker-a-late",
    ]


def test_notification_evidence_does_not_scan_all_markers() -> None:
    class GuardStore(InMemoryStore):
        def list_markers(self):
            raise AssertionError("full marker scan is forbidden")

    store = GuardStore()
    store.add_marker(_notification_marker("marker-a", "incident-a", seconds=0))
    service = AdvisoryNotificationService(store, DisabledNotificationNotifier())

    refs = service._evidence_refs_for_incident(
        "incident-a", "s3://evidence/explicit.json"
    )

    assert refs == ["s3://evidence/explicit.json", "s3://evidence/marker-a.json"]


def test_unrelated_incidents_do_not_share_one_notification_lock() -> None:
    barrier = Barrier(2)

    class BarrierStore(InMemoryStore):
        def get_incident(self, incident_id):
            barrier.wait(timeout=5)
            raise RuntimeError(incident_id)

    service = AdvisoryNotificationService(
        BarrierStore(), DisabledNotificationNotifier()
    )
    first = "incident-0"
    second = next(
        f"incident-{index}"
        for index in range(1, 100)
        if service._incident_lock(f"incident-{index}")
        is not service._incident_lock(first)
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(service.preview, incident_id)
            for incident_id in (first, second)
        ]
        for future in futures:
            with pytest.raises(RuntimeError):
                future.result(timeout=10)

    assert not barrier.broken, "expected barrier.broken to be falsy"


def test_email_contains_context_and_optional_support_case_draft() -> None:
    notification = HyperPodAdvisoryEmailBuilder().build(
        advisory(),
        cluster_name="hp-cluster",
        incident_id="incident-42",
        node_ids=["worker-1"],
        issue_summary="Repeated GPU fallen-off-bus event",
        region_name="us-west-2",
    )

    assert "ADVISE_ONLY" in notification.body_text
    assert "No recovery action or AWS Support case" in (notification.body_text)
    assert "Decision owner: customer administrator" in (notification.body_text)
    assert "repeated XID 79" in notification.support_case_draft
    assert "worker-1" in notification.support_case_draft


def test_ses_delivery_is_disabled_by_default() -> None:
    client = FakeSesV2Client()
    notifier = SesEmailNotifier(
        SesNotificationConfig(
            sender="gpu@example.com", recipients=["admin@example.com"]
        ),
        client=client,
    )

    result = notifier.send(
        HyperPodAdvisoryEmailBuilder().build(
            advisory(),
            cluster_name="hp-cluster",
            incident_id="incident-42",
            node_ids=["worker-1"],
            issue_summary="GPU fault",
        )
    )

    assert result.status is NotificationStatus.SKIPPED
    assert not client.requests, "expected client.requests to be falsy"


def test_gpu_count_change_email_is_actionable() -> None:
    notification = RestartGuardEmailBuilder().build_gpu_count_change(
        cluster_id="cluster-a",
        incident_id="incident-1",
        job_id="training-a",
        attempt_id="training-a-a001",
        workload_ids=["training/pytorchjob/training-a"],
        source_gpu_count=24,
        target_gpu_count=16,
        approval_annotation="24:16",
    )

    assert "一、发生了什么" in notification.body_text
    assert "二、建议管理员做什么" in notification.body_text
    assert RESTART_GUARD_TEMPLATE_VERSION in notification.body_text
    assert "当前没有创建新的训练 workload" in notification.body_text
    assert "方案 A：恢复原资源" in notification.body_text
    assert "方案 B：接受 16 张 GPU" in notification.body_text
    assert "world size" in notification.body_text
    assert (
        'kubectl --context "${GPU_FAULT_KUBE_CONTEXT:'
        '?set GPU_FAULT_KUBE_CONTEXT for cluster cluster-a}" '
        "-n training annotate pytorchjob training-a "
        "gpu-fault.io/approve-gpu-count-change='24:16' --overwrite"
        in notification.body_text
    )


def test_restart_guard_uses_a_predefined_template() -> None:
    required_fields = {
        "{job_id}",
        "{attempt_id}",
        "{cluster_id}",
        "{source_gpu_count}",
        "{target_gpu_count}",
        "{approval_commands}",
        "{incident_id}",
        "{template_version}",
    }

    assert all(field in GPU_COUNT_CHANGE_EMAIL_TEMPLATE for field in required_fields), (
        "expected all(field in GPU_COUNT_CHANGE_EMAIL_TEMPLATE for field in required_fields) to be truthy"
    )
    assert "prompt" not in GPU_COUNT_CHANGE_EMAIL_TEMPLATE.lower()


def test_workload_restart_email_is_a_field_only_template() -> None:
    notification = RestartGuardEmailBuilder().build_workload_restarted(
        cluster_id="cluster-a",
        incident_id="incident-1",
        workflow_id="workflow-1",
        operation_id="workflow-1/2/RESTART_WORKLOAD",
        job_id="training-a",
        source_attempt_id="training-a-a001",
        restart_attempt_id="training-a-a001-r-abcd1234",
        workload_ids=["training/pytorchjob/training-a"],
        source_gpu_count=24,
        target_gpu_count=24,
        restart_count=1,
        restart_budget=2,
    )

    assert "系统动作：RESTART_WORKLOAD" in notification.body_text
    assert "原 attempt：training-a-a001" in notification.body_text
    assert "新 attempt：training-a-a001-r-abcd1234" in notification.body_text
    assert "原 GPU 数量：24" in notification.body_text
    assert "重启 GPU 数量：24" in notification.body_text
    assert "本任务已使用自动重启次数：1" in notification.body_text
    assert RESTART_WORKLOAD_TEMPLATE_VERSION in notification.body_text
    assert "建议" not in notification.body_text
    assert "prompt" not in RESTART_WORKLOAD_EMAIL_TEMPLATE.lower()


def test_gpu_reset_email_is_a_field_only_template() -> None:
    notification = RestartGuardEmailBuilder().build_gpu_reset_completed(
        cluster_id="hp-cluster",
        incident_id="incident-1",
        workflow_id="workflow-1",
        event_id="xid-95",
        event_type="XID",
        policy_source="NVIDIA_CATALOG",
        official_action="RESET_GPU",
        reasons=["XID 95 requires a GPU reset"],
        operation_id="workflow-1/6/RESET_GPU",
        node_ids=["worker-1"],
        gpu_uuids=["GPU-a"],
        node_results={
            "worker-1": {"status": "SUCCEEDED", "reset_gpu_uuids": ["GPU-a"]}
        },
        workload_ids=["training/pytorchjob/training-a"],
    )

    assert "系统动作：RESET_GPU" in notification.body_text
    assert "目标节点：worker-1" in notification.body_text
    assert "目标 GPU UUID：GPU-a" in notification.body_text
    assert "Status：SUCCEEDED" in notification.body_text
    assert GPU_RESET_TEMPLATE_VERSION in notification.body_text
    assert "prompt" not in GPU_RESET_EMAIL_TEMPLATE.lower()


def test_node_restart_email_is_a_field_only_template() -> None:
    notification = RestartGuardEmailBuilder().build_node_restarted(
        cluster_id="hp-cluster",
        incident_id="incident-1",
        workflow_id="workflow-1",
        event_id="kernel-log-1",
        event_type="XID",
        xid=79,
        policy_source="NVIDIA_CATALOG",
        official_action="RESTART_BM",
        effective_action="REBOOT_NODE",
        reasons=["exact NVIDIA Catalog 610 Immediate Action for XID 79"],
        operation_id="hyperpod-op-1",
        node_ids=["worker-1"],
        source_boot_id="boot-a",
        agent_baselines={
            "worker-1": {"boot_id": "boot-a", "agent_incarnation_id": "incarnation-a"}
        },
        agent_observations=[
            {
                "node_id": "worker-1",
                "boot_id": "boot-b",
                "agent_incarnation_id": "incarnation-b",
            }
        ],
        provider_observations=[
            {
                "node_id": "worker-1",
                "node_logical_id": "logical-1",
                "instance_id": "i-123",
                "status": "Running",
            }
        ],
        confirmation_source=("hyperpod-running-and-new-agent-incarnation"),
    )

    assert "系统动作：RESTART_NODE" in notification.body_text
    assert "故障标识：XID 79" in notification.body_text
    assert "Policy source：NVIDIA_CATALOG" in notification.body_text
    assert "策略动作：RESTART_BM" in notification.body_text
    assert "Effective action：REBOOT_NODE" in notification.body_text
    assert "Immediate Action for XID 79" in notification.body_text
    assert "重启前 boot ID：boot-a" in notification.body_text
    assert "重启后 boot ID：boot-b" in notification.body_text
    assert "InstanceId：i-123" in notification.body_text
    assert "HyperPod status：Running" in notification.body_text
    assert RESTART_NODE_TEMPLATE_VERSION in notification.body_text
    assert "建议" not in notification.body_text
    assert "prompt" not in RESTART_NODE_EMAIL_TEMPLATE.lower()


def test_fabric_manager_restart_email_is_a_field_only_template() -> None:
    notification = RestartGuardEmailBuilder().build_fabric_manager_restarted(
        cluster_id="hp-cluster",
        incident_id="incident-45",
        workflow_id="workflow-45",
        event_id="kernel-log-45",
        event_type="XID",
        policy_source="NVIDIA_CATALOG",
        official_action="RESTART_FM",
        reasons=["solo XID 45 requires Fabric Manager restart"],
        operation_id=("workflow-45/1/RESTART_FABRIC_MANAGER"),
        node_results={
            "worker-1": {
                "service": "nvidia-fabricmanager",
                "active": True,
                "previous_main_pid": "101",
                "current_main_pid": "202",
            }
        },
        workload_ids=["training/pytorchjob/training-a"],
    )

    assert "系统动作：RESTART_FABRIC_MANAGER" in notification.body_text
    assert "故障类型：XID" in notification.body_text
    assert "策略动作：RESTART_FM" in notification.body_text
    assert "重启前 MainPID：101" in notification.body_text
    assert "重启后 MainPID：202" in notification.body_text
    assert "服务状态：active" in notification.body_text
    assert RESTART_FABRIC_MANAGER_TEMPLATE_VERSION in notification.body_text
    assert notification.category == "ACTION_COMPLETED"
    assert "建议" not in notification.body_text
    assert "prompt" not in RESTART_FABRIC_MANAGER_EMAIL_TEMPLATE.lower()


def test_fabric_reset_email_is_a_field_only_template() -> None:
    notification = RestartGuardEmailBuilder().build_fabric_reset_completed(
        cluster_id="hp-cluster",
        incident_id="incident-sxid",
        workflow_id="workflow-sxid",
        event_id="kernel-sxid",
        event_type="SXID",
        policy_source="NVIDIA_FABRIC_MANAGER",
        official_action="RESET_ALL_GPUS_AND_NVSWITCHES",
        reasons=["fatal trunk-link SXID"],
        operation_id=("workflow-sxid/5/RESET_ALL_GPUS_NVSWITCHES"),
        node_results={
            "worker-1": {
                "reset_scope": ("ALL_LOCAL_GPUS_AND_NVSWITCHES"),
                "reset_gpu_uuids": ["GPU-a", "GPU-b"],
                "verified_no_gpu_clients": True,
                "inventory_verified_after": True,
            }
        },
        workload_ids=["training/pytorchjob/training-a"],
        fabric_partition="hp-cluster/worker-1/local-nvswitch",
        sxid=10003,
    )

    assert "系统动作：RESET_ALL_GPUS_NVSWITCHES" in (notification.body_text)
    assert "SXID：10003" in notification.body_text
    assert "GPU-a, GPU-b" in notification.body_text
    assert "Reset 后 inventory：True" in notification.body_text
    assert FABRIC_RESET_TEMPLATE_VERSION in notification.body_text
    assert "建议" not in notification.body_text
    assert "prompt" not in FABRIC_RESET_EMAIL_TEMPLATE.lower()


def test_node_restart_template_supports_non_xid_incidents() -> None:
    notification = RestartGuardEmailBuilder().build_node_restarted(
        cluster_id="hp-cluster",
        incident_id="incident-health-1",
        workflow_id="workflow-health-1",
        event_id="hma-node-health-worker-1",
        event_type="HYPERPOD_HMA",
        xid=None,
        policy_source="HYPERPOD_HMA",
        official_action="RESTART_BM",
        effective_action="REBOOT_NODE",
        reasons=["HyperPod HMA reported a node health failure"],
        operation_id="hyperpod-op-2",
        node_ids=["worker-1"],
        source_boot_id="boot-a",
        agent_baselines={},
        agent_observations=[],
        provider_observations=[],
        confirmation_source="explicit-operation-id",
    )

    assert "故障类型：HYPERPOD_HMA" in notification.body_text
    assert "故障标识：hma-node-health-worker-1" in notification.body_text
    assert "Policy source：HYPERPOD_HMA" in notification.body_text
    assert "XID：UNKNOWN" not in notification.body_text


def test_node_restart_template_maps_two_nodes_independently() -> None:
    notification = RestartGuardEmailBuilder().build_node_restarted(
        cluster_id="hp-cluster",
        incident_id="incident-2",
        workflow_id="workflow-2",
        event_id="distributed-xid-79",
        event_type="XID",
        xid=79,
        policy_source="NVIDIA_CATALOG",
        official_action="RESTART_BM",
        effective_action="REBOOT_NODE",
        reasons=["two nodes reported XID 79"],
        operation_id="hyperpod-op-2",
        node_ids=["worker-1", "worker-2"],
        source_boot_id=None,
        agent_baselines={
            "worker-1": {"boot_id": "boot-1-old"},
            "worker-2": {"boot_id": "boot-2-old"},
        },
        agent_observations=[
            {
                "node_id": "worker-1",
                "boot_id": "boot-1-new",
                "agent_incarnation_id": "incarnation-1-new",
            },
            {
                "node_id": "worker-2",
                "boot_id": "boot-2-new",
                "agent_incarnation_id": "incarnation-2-new",
            },
        ],
        provider_observations=[
            {
                "node_id": "worker-1",
                "node_logical_id": "logical-1",
                "instance_id": "i-111",
                "status": "Running",
            },
            {
                "node_id": "worker-2",
                "node_logical_id": "logical-2",
                "instance_id": "i-222",
                "status": "Running",
            },
        ],
        confirmation_source=("hyperpod-running-and-new-agent-incarnation"),
    )

    body = notification.body_text
    assert "worker-1; InstanceId：i-111" in body
    assert "重启前 boot ID：boot-1-old" in body
    assert "重启后 boot ID：boot-1-new" in body
    assert "worker-2; InstanceId：i-222" in body
    assert "重启前 boot ID：boot-2-old" in body
    assert "重启后 boot ID：boot-2-new" in body


def test_not_applicable_email_is_a_field_only_template() -> None:
    notification = NotApplicableEmailBuilder().build(
        cluster_id="hp-cluster",
        node_id="worker-1",
        incident_id="incident-1",
        event_id="xid-1",
        xid=1,
        product="H200",
        driver_branch=575,
        cuda_version="12.9",
        observed_at="2026-07-22T12:00:00+00:00",
        workload_state="ACTIVE",
        workload_ids=["training/pytorchjob/train-a"],
        official_action="CONTACT_SUPPORT",
        policy_source="NVIDIA_CATALOG",
        policy_version="catalog/version-1",
        reasons=["Catalog XID 1 is not applicable to product H200"],
        safety_action="QUARANTINE",
        workflow_id="workflow-1",
        workflow_status="SAFETY_PENDING",
        safety_steps=["FREEZE_EVIDENCE", "MARK_UNSCHEDULABLE", "QUARANTINE"],
        official_steps=["FREEZE_EVIDENCE"],
        evidence_refs=["kmsg://worker-1/boot-1/10"],
    )

    assert NOT_APPLICABLE_TEMPLATE_VERSION in notification.body_text
    assert "XID：1" in notification.body_text
    assert "GPU 产品：H200" in notification.body_text
    assert "Disposition：NOT_APPLICABLE" in notification.body_text
    assert "Safety action：QUARANTINE" in notification.body_text
    assert "建议" not in notification.body_text
    assert "prompt" not in NOT_APPLICABLE_EMAIL_TEMPLATE.lower()


def test_not_applicable_xid_creates_and_sends_one_email() -> None:
    notifier = RecordingNotifier()
    context = ApplicationContext(notification_notifier=notifier)
    now = datetime(2026, 7, 22, 12, 0, tzinfo=timezone.utc)

    async def scenario() -> None:
        transport = httpx.ASGITransport(app=create_app(context))
        payload = {
            "event_id": "not-applicable-xid-1",
            "cluster_id": "hp-cluster",
            "node_id": "worker-1",
            "observed_at": now.isoformat(),
            "xid": 1,
            "product": "H200",
            "driver_branch": 575,
            "cuda_version": "12.9",
            "runtime_profile_version": "simulated-v1",
            "workload_state": "ACTIVE",
            "affected_workload_ids": ["training/pytorchjob/train-a"],
            "evidence_ref": "kmsg://worker-1/boot-1/10",
        }
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            first = await client.post("/v1/gpu-events/xid", json=payload)
            second = await client.post("/v1/gpu-events/xid", json=payload)

        assert first.status_code == 200
        assert second.status_code == 200
        first_body = first.json()
        assert first_body["disposition"] == "NOT_APPLICABLE"
        assert first_body["advisory_notification_id"]
        assert (
            second.json()["advisory_notification_id"]
            == first_body["advisory_notification_id"]
        )

    asyncio.run(scenario())

    assert len(notifier.notifications) == 1
    notification = notifier.notifications[0]
    assert notification.incident_id.startswith("inc-"), (
        'expected notification.incident_id.startswith("inc-") to be truthy'
    )
    assert "Workload 状态：ACTIVE" in notification.body_text
    assert "官方恢复动作：未执行" in notification.body_text
    assert "NVIDIA Investigatory Action：NONE" in notification.body_text
    assert "建议" not in notification.body_text
    assert len(context.store.list_notifications()) == 1


def test_applicable_xid_sends_one_investigatory_email_per_event() -> None:
    notifier = RecordingNotifier()
    context = ApplicationContext(notification_notifier=notifier)
    now = datetime(2026, 7, 26, 12, 0, tzinfo=timezone.utc)

    async def scenario() -> None:
        transport = httpx.ASGITransport(app=create_app(context))
        payload = {
            "event_id": "investigatory-xid-11",
            "cluster_id": "hp-cluster",
            "node_id": "worker-1",
            "observed_at": now.isoformat(),
            "xid": 11,
            "gpu_uuid": "GPU-a",
            "product": "H100",
            "driver_branch": 575,
            "cuda_version": "12.9",
            "runtime_profile_version": "simulated-v1",
            "workload_state": "ACTIVE",
            "affected_workload_ids": ["training/pytorchjob/train-a"],
            "evidence_ref": "kmsg://worker-1/boot-1/11",
        }
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            first = await client.post("/v1/gpu-events/xid", json=payload)
            second = await client.post("/v1/gpu-events/xid", json=payload)

        assert first.status_code == 200
        assert second.status_code == 200
        first_id = first.json()["investigatory_notification_id"]
        assert first_id, "expected first_id to be truthy"
        assert second.json()["investigatory_notification_id"] == first_id

    asyncio.run(scenario())

    assert len(notifier.notifications) == 1
    notification = notifier.notifications[0]
    assert "XID：11" in notification.body_text
    assert "NVIDIA Immediate Action：RESTART_APP" in notification.body_text
    assert "NVIDIA Investigatory Action：CHECK_APP/CUDA" in (notification.body_text)


@pytest.mark.parametrize(
    ("sxid", "investigatory_action"),
    [
        (20012, "CHECK_LINK_MECHANICAL_CONNECTIONS"),
        (10004, "CHECK_SYSTEM_COOLING"),
        (10005, "VERIFY_THERMAL_EVENT_CLEARED"),
        (28006, "MONITOR_PROGRESS_AND_PERFORMANCE"),
    ],
)
def test_every_sxid_sends_one_fixed_event_email(
    sxid: int, investigatory_action: str
) -> None:
    notifier = RecordingNotifier()
    context = ApplicationContext(notification_notifier=notifier)
    now = datetime(2026, 7, 26, 12, 30, tzinfo=timezone.utc)

    async def scenario() -> None:
        transport = httpx.ASGITransport(app=create_app(context))
        payload = {
            "event_id": f"sxid-email-{sxid}",
            "cluster_id": "hp-cluster",
            "node_id": "worker-1",
            "observed_at": now.isoformat(),
            "event_source": "KERNEL_LOG",
            "sxid": sxid,
            "classification": "NON_FATAL",
            "classification_source": ("NVIDIA_FABRIC_MANAGER_CATALOG"),
            "product": "H200",
            "switch_id": "3",
            "port": "46",
            "pci_bdf": "0000:c1:00.0",
            "runtime_profile_version": "simulated-v1",
            "workload_state": "ACTIVE",
            "affected_workload_ids": ["training/pytorchjob/train-a"],
            "evidence_ref": (f"kmsg://worker-1/boot-1/{sxid}"),
        }
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            first = await client.post("/v1/gpu-events/sxid", json=payload)
            second = await client.post("/v1/gpu-events/sxid", json=payload)

        assert first.status_code == 200, first.text
        assert second.status_code == 200, second.text
        first_id = first.json()["investigatory_notification_id"]
        assert first_id, "expected first_id to be truthy"
        assert second.json()["investigatory_notification_id"] == first_id

    asyncio.run(scenario())

    assert len(notifier.notifications) == 1
    notification = notifier.notifications[0]
    assert f"SXID：{sxid}" in notification.body_text
    assert (
        f"NVIDIA Investigatory Action：{investigatory_action}" in notification.body_text
    )
    assert SXID_EVENT_TEMPLATE_VERSION in notification.body_text
    assert "建议" not in notification.body_text


def test_disabled_preview_does_not_consume_delivery_key() -> None:
    client = FakeSesV2Client()
    config = SesNotificationConfig(
        sender="gpu@example.com", recipients=["admin@example.com"]
    )
    notifier = SesEmailNotifier(config, client=client)
    notification = HyperPodAdvisoryEmailBuilder().build(
        advisory(),
        cluster_name="hp-cluster",
        incident_id="incident-42",
        node_ids=["worker-1"],
        issue_summary="GPU fault",
    )

    assert notifier.send(notification).status is NotificationStatus.SKIPPED
    config.execution_enabled = True
    assert notifier.send(notification).status is NotificationStatus.SENT
    assert len(client.requests) == 1


def test_ses_delivery_is_idempotent() -> None:
    client = FakeSesV2Client()
    notifier = SesEmailNotifier(
        SesNotificationConfig(
            sender="gpu@example.com",
            recipients=["admin@example.com"],
            execution_enabled=True,
        ),
        client=client,
    )
    notification = HyperPodAdvisoryEmailBuilder().build(
        advisory(),
        cluster_name="hp-cluster",
        incident_id="incident-42",
        node_ids=["worker-1"],
        issue_summary="GPU fault",
    )

    first = notifier.send(notification)
    second = notifier.send(notification)

    assert first.status is NotificationStatus.SENT
    assert first.provider_message_id == "ses-message-1"
    assert second.status is NotificationStatus.DUPLICATE
    assert len(client.requests) == 1
    assert client.requests[0]["Destination"]["ToAddresses"] == ["admin@example.com"]


def test_ses_delivery_adds_declared_site_context() -> None:
    client = FakeSesV2Client()
    notifier = SesEmailNotifier(
        SesNotificationConfig(
            sender="sender@example.com",
            recipients=["ops@example.com", "oncall@example.com"],
            region_name="us-west-2",
            site_id="site-a",
            account_id="123456789012",
            subject_prefix="[PROD]",
            execution_enabled=True,
        ),
        client=client,
    )
    notification = HyperPodAdvisoryEmailBuilder().build(
        advisory(),
        cluster_name="cluster-a",
        incident_id="incident-a",
        node_ids=["runtime-node-9"],
        issue_summary="GPU fault",
    )

    assert notifier.send(notification).status is NotificationStatus.SENT
    request = client.requests[0]["Content"]["Simple"]
    assert request["Subject"]["Data"].startswith(
        "[PROD] [site:site-a] [region:us-west-2] [account:123456789012]"
    ), "SES subject omitted the declared site context"
    body = request["Body"]["Text"]["Data"]
    assert "- Site: site-a" in body
    assert "- AWS Account: 123456789012" in body
    assert "- Region: us-west-2" in body
    assert "- Cluster: cluster-a" in body
    assert "runtime-node-9" in body


def test_dispatch_continues_after_one_delivery_failure() -> None:
    store = build_store()
    builder = HyperPodAdvisoryEmailBuilder()
    failed = builder.build(
        advisory(),
        cluster_name="hp-cluster",
        incident_id="incident-fail",
        node_ids=["worker-1"],
        issue_summary="GPU fault",
    )
    successful = builder.build(
        advisory(),
        cluster_name="hp-cluster",
        incident_id="incident-ok",
        node_ids=["worker-2"],
        issue_summary="GPU fault",
    )
    store.save_notification_if_absent(failed)
    store.save_notification_if_absent(successful)
    service = AdvisoryNotificationService(store, PartiallyFailingNotifier())

    report = service.dispatch_pending()

    assert report.attempted == 2
    assert report.sent == 1
    assert report.failed == 1
    assert {item.status for item in report.results} == {
        NotificationStatus.SENT,
        NotificationStatus.FAILED,
    }
