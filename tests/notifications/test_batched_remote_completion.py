"""A batched step's completion is mailed when its carrier finishes.

Since the compound node command (性能 C) the RESET_GPU step of a reset chain
rides inside a carrier headed by QUIESCE_GPU_SERVICES. ``dispatch_remote_completion``
keyed only on the head step, so a real batched reset never produced
GPU_RESET_COMPLETED: the live DESTR-001, HA-003 and HA-004 resets went unmailed
while NOTIFY-001's standalone drills did mail. The contract pinned here: every
batched step with a SUCCEEDED verdict is mailed exactly as its standalone
command would have been -- same kind, ``operation_id`` = the step's own
idempotency key (the standalone deduplication key), ``node_results`` from the
step's own entry -- once the carrier is terminal, and never for the head
QUIESCE/RESTORE, a WAITING or FAILED entry, or an open carrier.
"""

from __future__ import annotations

from typing import Any

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
from gpu_fault.remote_command_models import BATCHED_RESULTS_KEY, BatchedStep
from tests._builders import build_store, fault_incident
from tests.notifications._support import RecordingNotifier

CHAIN = (
    WorkflowOperation.FREEZE_EVIDENCE,
    WorkflowOperation.MARK_UNSCHEDULABLE,
    WorkflowOperation.QUIESCE_GPU_SERVICES,
    WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
    WorkflowOperation.RESET_GPU,
    WorkflowOperation.RESTORE_GPU_SERVICES,
    WorkflowOperation.VALIDATE_GPU,
    WorkflowOperation.RESTORE_SCHEDULING,
)
HEAD_INDEX = 2
RESET_INDEX = 4
RESET_NODE_RESULTS = {
    "node-a": {
        "node_action_command_id": "workflow-reset/4/RESET_GPU/node-a/agent-6",
        "reset_gpu_uuids": ["GPU-reset-a"],
        "status": "SUCCEEDED",
    }
}
QUIESCE_NODE_RESULTS = {
    "node-a": {"active_services": ["nvidia-dcgm", "kubelet"], "failsafe_seconds": 420}
}


def _entry(
    status: RemoteCommandStatus,
    node_results: dict[str, Any] | None = None,
    *,
    error: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One ``batched_results`` entry as ``BatchedStepResult.model_dump`` writes it."""

    details: dict[str, Any] = {}
    if node_results is not None:
        details["node_results"] = node_results
    details.update(extra or {})
    return {
        "status": status.value,
        "status_source": None,
        "details": details,
        "error": error,
    }


def _step(
    operation: WorkflowOperation, parameters: dict[str, Any] | None = None
) -> WorkflowStepSpec:
    return WorkflowStepSpec(
        operation=operation,
        execution_owner="gpu-fault-node-agent",
        node_ids=["node-a"],
        gpu_uuids=["GPU-a"],
        workload_ids=["training/pytorchjob/job-a"],
        parameters=parameters or {},
    )


def _compound_command(
    store: Any,
    *,
    status: RemoteCommandStatus = RemoteCommandStatus.SUCCEEDED,
    batched_results: dict[int, dict[str, Any]],
    operations: tuple[WorkflowOperation, ...] = CHAIN,
    head_index: int = HEAD_INDEX,
    batched_indexes: tuple[int, ...] = (3, 4, 5),
    parameters: dict[int, dict[str, Any]] | None = None,
    error: str | None = None,
    extra_details: dict[str, Any] | None = None,
    drill_id: str | None = None,
) -> RemoteActionCommand:
    """A carrier the way the regional adapter mints and the executor reports it."""

    steps = [_step(op, (parameters or {}).get(i)) for i, op in enumerate(operations)]
    workflow = WorkflowRequest(
        request_id="workflow-reset",
        incident_id="incident-reset",
        status=WorkflowStatus.RUNNING,
        official_action="RESET_GPU",
        fencing_token=1,
        official_steps=steps,
    )
    incident = fault_incident(
        workflow.incident_id,
        "event-reset",
        state=IncidentState.ACTION_PENDING,
        workflow_request_id=workflow.request_id,
        official_action="RESET_GPU",
        reasons=["XID 46"],
        fencing_token=1,
        **({"drill_id": drill_id} if drill_id else {}),
    )
    store.save_incident(incident)
    store.save_workflow(workflow)

    def key(index: int) -> str:
        return f"{workflow.request_id}/{index}/{operations[index].value}"

    return RemoteActionCommand(
        command_id="remote-carrier",
        cluster_id=incident.cluster_id,
        workflow_request_id=workflow.request_id,
        incident_id=incident.incident_id,
        step_index=head_index,
        fencing_token=1,
        idempotency_key=key(head_index),
        step=steps[head_index],
        workflow=workflow,
        incident=incident,
        batched_steps=[
            BatchedStep(step_index=i, step=steps[i], idempotency_key=key(i))
            for i in batched_indexes
        ],
        status=status,
        error=error,
        result_details={
            BATCHED_RESULTS_KEY: {str(i): v for i, v in batched_results.items()},
            "batched_step_indexes": [head_index, *batched_indexes],
            **(extra_details or {}),
        },
    )


def _service(store: Any) -> tuple[AdvisoryNotificationService, RecordingNotifier]:
    notifier = RecordingNotifier()
    return (
        AdvisoryNotificationService(store, notifier, async_delivery=False),
        notifier,
    )


def _full_reset_results() -> dict[int, dict[str, Any]]:
    return {
        2: _entry(RemoteCommandStatus.SUCCEEDED, QUIESCE_NODE_RESULTS),
        3: _entry(
            RemoteCommandStatus.SUCCEEDED, {"node-a": {"verified_no_gpu_clients": True}}
        ),
        4: _entry(RemoteCommandStatus.SUCCEEDED, RESET_NODE_RESULTS),
        5: _entry(RemoteCommandStatus.SUCCEEDED, {"node-a": {"restored": True}}),
    }


def test_a_batched_reset_is_mailed_once_keyed_by_its_own_step() -> None:
    store = build_store()
    service, notifier = _service(store)
    command = _compound_command(store, batched_results=_full_reset_results())

    results = service.dispatch_remote_completion(command)

    assert [item.status for item in results] == [NotificationStatus.SENT], results
    [notification] = store.list_notifications()
    assert notification.category == "ACTION_COMPLETED"
    assert "GPU 已自动重置" in notification.subject, notification.subject
    assert notification.deduplication_key == (
        f"{command.cluster_id}/{command.incident_id}/gpu-reset/"
        "workflow-reset/4/RESET_GPU"
    ), "the mail must be keyed by the batched step's own idempotency key"
    assert "workflow-reset/2/QUIESCE_GPU_SERVICES" not in notification.body_text, (
        "the head's key must not name the reset"
    )
    assert "GPU-reset-a" in notification.body_text, (
        "node_results must come from the batched step's own entry"
    )
    assert "nvidia-dcgm" not in notification.body_text, (
        "the head's node_results must not leak into the reset mail"
    )
    assert len(notifier.notifications) == 1


def test_the_head_quiesce_and_restore_never_mail() -> None:
    store = build_store()
    service, notifier = _service(store)
    command = _compound_command(
        store,
        batched_results={
            2: _entry(RemoteCommandStatus.SUCCEEDED, QUIESCE_NODE_RESULTS),
            3: _entry(RemoteCommandStatus.SUCCEEDED, {"node-a": {}}),
            5: _entry(RemoteCommandStatus.SUCCEEDED, {"node-a": {"restored": True}}),
        },
        batched_indexes=(3, 5),
    )

    assert service.dispatch_remote_completion(command) == []
    assert store.list_notifications() == []
    assert notifier.notifications == []


def test_waiting_and_failed_batched_results_never_mail() -> None:
    store = build_store()
    service, notifier = _service(store)
    command = _compound_command(
        store,
        status=RemoteCommandStatus.FAILED,
        error="node agent node-a: reset refused",
        batched_results={
            2: _entry(RemoteCommandStatus.SUCCEEDED, QUIESCE_NODE_RESULTS),
            3: _entry(RemoteCommandStatus.WAITING, {"node-a": {"pending": True}}),
            4: _entry(
                RemoteCommandStatus.FAILED,
                RESET_NODE_RESULTS,
                error="node agent node-a: reset refused",
            ),
        },
    )

    assert service.dispatch_remote_completion(command) == []
    assert store.list_notifications() == []
    assert notifier.notifications == []


def test_an_open_carrier_does_not_mail_a_finished_passenger_yet() -> None:
    """A WAITING report (RESTORE still pending on the node) may already carry
    the reset's SUCCEEDED entry; the mail lands with the terminal report."""

    store = build_store()
    service, notifier = _service(store)
    results = _full_reset_results()
    results[5] = _entry(RemoteCommandStatus.WAITING, {"node-a": {"pending": True}})
    command = _compound_command(
        store, status=RemoteCommandStatus.WAITING, batched_results=results
    )

    assert service.dispatch_remote_completion(command) == []
    assert store.list_notifications() == []
    assert notifier.notifications == []


def test_a_failed_restore_after_a_completed_reset_still_mails_the_reset() -> None:
    store = build_store()
    service, notifier = _service(store)
    results = _full_reset_results()
    results[5] = _entry(
        RemoteCommandStatus.FAILED,
        {"node-a": {"restored": False}},
        error="RESTORE_GPU_SERVICES: nvidia-dcgm did not return active",
    )
    command = _compound_command(
        store,
        status=RemoteCommandStatus.FAILED,
        error="RESTORE_GPU_SERVICES: nvidia-dcgm did not return active",
        batched_results=results,
        extra_details={
            "batched_step_index": 5,
            "batched_operation": "RESTORE_GPU_SERVICES",
        },
    )

    results_sent = service.dispatch_remote_completion(command)

    assert [item.status for item in results_sent] == [NotificationStatus.SENT]
    [notification] = store.list_notifications()
    assert notification.deduplication_key.endswith(
        "/gpu-reset/workflow-reset/4/RESET_GPU"
    ), "the mail must be keyed by the RESET_GPU step's own idempotency key"
    assert len(notifier.notifications) == 1


def test_a_second_dispatch_of_the_same_carrier_sends_nothing_new() -> None:
    store = build_store()
    service, notifier = _service(store)
    command = _compound_command(store, batched_results=_full_reset_results())

    first = service.dispatch_remote_completion(command)
    second = service.dispatch_remote_completion(command)

    assert [item.status for item in first] == [NotificationStatus.SENT]
    assert [item.status for item in second] == [NotificationStatus.DUPLICATE]
    assert len(store.list_notifications()) == 1
    assert len(notifier.notifications) == 1


def test_the_heads_existing_notification_does_not_suppress_a_batched_mail() -> None:
    """``result_details["notification_id"]`` names the single-cluster adapter's
    notification for the head step; it must not swallow the passengers."""

    store = build_store()
    service, notifier = _service(store)
    command = _compound_command(store, batched_results=_full_reset_results())
    head_notification = service.builders.build(
        NotificationKind.FABRIC_MANAGER_RESTARTED,
        cluster_id=command.cluster_id,
        incident_id=command.incident.incident_id,
        workflow_id=command.workflow.request_id,
        event_id=command.incident.event_id,
        event_type=command.incident.event_type,
        policy_source=command.incident.policy_source,
        official_action=command.incident.official_action,
        reasons=command.incident.reasons,
        operation_id=command.idempotency_key,
        node_results={"node-a": {"active": True}},
        workload_ids=command.step.workload_ids,
    )
    head_notification = store.save_notification_if_absent(head_notification)
    command = command.model_copy(
        update={
            "result_details": {
                **command.result_details,
                "notification_id": head_notification.notification_id,
            }
        }
    )

    results = service.dispatch_remote_completion(command)

    assert [item.status for item in results] == [
        NotificationStatus.SENT,
        NotificationStatus.SENT,
    ], results
    keys = sorted(item.deduplication_key for item in store.list_notifications())
    assert any(key.endswith("/gpu-reset/workflow-reset/4/RESET_GPU") for key in keys), (
        keys
    )
    assert len(notifier.notifications) == 2


def test_a_carrier_headed_by_the_reset_mails_the_reset_exactly_once() -> None:
    """QUIESCE already resolved: the head is RESET_GPU, RESTORE the passenger."""

    store = build_store()
    service, notifier = _service(store)
    command = _compound_command(
        store,
        head_index=RESET_INDEX,
        batched_indexes=(5,),
        batched_results={
            4: _entry(RemoteCommandStatus.SUCCEEDED, RESET_NODE_RESULTS),
            5: _entry(RemoteCommandStatus.SUCCEEDED, {"node-a": {"restored": True}}),
        },
        extra_details={"node_results": RESET_NODE_RESULTS},
    )

    results = service.dispatch_remote_completion(command)

    assert [item.status for item in results] == [NotificationStatus.SENT], results
    [notification] = store.list_notifications()
    assert notification.deduplication_key.endswith(
        "/gpu-reset/workflow-reset/4/RESET_GPU"
    ), "the mail must be keyed by the RESET_GPU step's own idempotency key"
    assert len(notifier.notifications) == 1


def test_a_batched_full_fabric_reset_mails_its_own_kind() -> None:
    store = build_store()
    service, notifier = _service(store)
    operations = (
        WorkflowOperation.QUIESCE_GPU_SERVICES,
        WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES,
        WorkflowOperation.RESTORE_GPU_SERVICES,
    )
    command = _compound_command(
        store,
        operations=operations,
        head_index=0,
        batched_indexes=(1, 2),
        parameters={
            1: {"fabric_partition": "cluster-a/node-a/local-nvswitch", "sxid": 10003}
        },
        batched_results={
            0: _entry(RemoteCommandStatus.SUCCEEDED, QUIESCE_NODE_RESULTS),
            1: _entry(
                RemoteCommandStatus.SUCCEEDED,
                {"node-a": {"reset": True, "gpus": 8, "nvswitches": 4}},
            ),
            2: _entry(RemoteCommandStatus.SUCCEEDED, {"node-a": {"restored": True}}),
        },
    )

    results = service.dispatch_remote_completion(command)

    assert [item.status for item in results] == [NotificationStatus.SENT], results
    [notification] = store.list_notifications()
    assert "NVSwitch" in notification.subject, notification.subject
    assert "10003" in notification.body_text
    assert notification.deduplication_key.endswith(
        "/workflow-reset/1/RESET_ALL_GPUS_NVSWITCHES"
    ), notification.deduplication_key
    assert len(notifier.notifications) == 1


def test_a_batched_dcgm_diagnostic_mails_its_verdict_even_when_it_failed() -> None:
    """The standalone rule carries over: a failed diagnostic is the case the
    mail exists for, so a FAILED DCGM entry with node results still mails."""

    store = build_store()
    service, notifier = _service(store)
    operations = (
        WorkflowOperation.QUIESCE_GPU_SERVICES,
        WorkflowOperation.RUN_DCGM_DIAGNOSTIC,
        WorkflowOperation.RESTORE_GPU_SERVICES,
    )
    verdict = {"node-a": {"diagnostic_outcome": "FAIL", "diagnostic_findings": []}}
    command = _compound_command(
        store,
        operations=operations,
        head_index=0,
        batched_indexes=(1, 2),
        status=RemoteCommandStatus.FAILED,
        error="DCGM diagnostic failed on node-a",
        batched_results={
            0: _entry(RemoteCommandStatus.SUCCEEDED, QUIESCE_NODE_RESULTS),
            1: _entry(
                RemoteCommandStatus.FAILED,
                verdict,
                error="DCGM diagnostic failed on node-a",
                extra={"control_plane_action": "DRAIN_AND_QUARANTINE"},
            ),
        },
    )

    results = service.dispatch_remote_completion(command)

    assert [item.status for item in results] == [NotificationStatus.SENT], results
    [notification] = store.list_notifications()
    assert "DCGM" in notification.subject, notification.subject
    assert "DRAIN_AND_QUARANTINE" in notification.body_text
    assert notification.deduplication_key.endswith(
        "/workflow-reset/1/RUN_DCGM_DIAGNOSTIC"
    ), notification.deduplication_key
    assert len(notifier.notifications) == 1


def test_a_standalone_reset_command_is_mailed_as_before() -> None:
    """No batched steps: the head path alone, keyed by the command's own key,
    which is the same ``<workflow>/<index>/RESET_GPU`` a passenger gets."""

    store = build_store()
    service, notifier = _service(store)
    command = _compound_command(
        store, head_index=RESET_INDEX, batched_indexes=(), batched_results={}
    )
    command = command.model_copy(
        update={"result_details": {"node_results": RESET_NODE_RESULTS}}
    )
    assert command.batched_steps == []

    first = service.dispatch_remote_completion(command)
    second = service.dispatch_remote_completion(command)

    assert [item.status for item in first] == [NotificationStatus.SENT], first
    assert [item.status for item in second] == [NotificationStatus.DUPLICATE], second
    [notification] = store.list_notifications()
    assert notification.deduplication_key == (
        f"{command.cluster_id}/{command.incident_id}/gpu-reset/"
        "workflow-reset/4/RESET_GPU"
    )
    assert len(notifier.notifications) == 1


def test_a_drill_carrier_records_but_never_mails_the_batched_reset() -> None:
    store = build_store()
    service, notifier = _service(store)
    command = _compound_command(
        store, batched_results=_full_reset_results(), drill_id="destr001-unit"
    )

    results = service.dispatch_remote_completion(command)

    assert [item.status for item in results] == [NotificationStatus.SKIPPED], results
    [notification] = store.list_notifications()
    assert notification.drill_id == "destr001-unit"
    assert notification.subject.startswith("[DRILL:destr001-unit]"), (
        notification.subject
    )
    assert notifier.notifications == []
