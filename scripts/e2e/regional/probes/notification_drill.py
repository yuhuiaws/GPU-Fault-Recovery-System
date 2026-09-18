from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

from gpu_fault.app.routes.regional import (
    RegionalRouterDependencies,
    complete_remote_command,
)
from gpu_fault.async_store import AsyncStoreExecutor
from gpu_fault.models import (
    AdvisoryNotification,
    FaultIncident,
    IncidentState,
    NotificationResult,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepSpec,
)
from gpu_fault.notification_service import AdvisoryNotificationService
from gpu_fault.notifications import (
    RestartGuardEmailBuilder,
    notification_notifier_from_environment,
)
from gpu_fault.store import InMemoryStore
from gpu_fault.regional import (
    RemoteActionCommand,
    RemoteCommandResult,
    RemoteCommandStatus,
)
from gpu_fault.regional_compatibility import RegionalExecutorCompatibilityPolicy

INJECTION_PATH = "gpu_fault.app.routes.regional.complete_remote_command"


class DrillNotifier:
    def __init__(
        self, inner: Any, drill_id: str, deadline: datetime | None = None
    ) -> None:
        self.inner = inner
        self.drill_id = drill_id
        self.calls = 0
        self.deadline = deadline

    def send(self, notification: AdvisoryNotification) -> NotificationResult:
        if self.deadline is not None and datetime.now(timezone.utc) >= self.deadline:
            raise RuntimeError("notification maintenance window ended")
        if (
            notification.drill_id != self.drill_id
            or not notification.subject.startswith(f"[DRILL:{self.drill_id}]")
            or not notification.body_text.startswith("DRILL /")
        ):
            raise RuntimeError("refusing an unlabelled acceptance email")
        self.calls += 1
        return self.inner.send(notification)


def label_drill(
    notification: AdvisoryNotification, drill_id: str
) -> AdvisoryNotification:
    return notification.model_copy(
        update={
            "drill_id": drill_id,
            "subject": f"[DRILL:{drill_id}] {notification.subject}",
            "body_text": "DRILL / no GPU or workload action occurred.\n\n"
            + notification.body_text,
        }
    )


class DrillEmailBuilder(RestartGuardEmailBuilder):
    def __init__(self, drill_id: str) -> None:
        self.drill_id = drill_id

    def build_gpu_reset_completed(self, **values: Any) -> AdvisoryNotification:
        return label_drill(super().build_gpu_reset_completed(**values), self.drill_id)

    def build_workload_restarted(self, **values: Any) -> AdvisoryNotification:
        return label_drill(super().build_workload_restarted(**values), self.drill_id)


class CompletionRecorder(AdvisoryNotificationService):
    def __init__(
        self, store: InMemoryStore, notifier: DrillNotifier, drill_id: str
    ) -> None:
        super().__init__(
            store,
            notifier,
            restart_email_builder=DrillEmailBuilder(drill_id),
            async_delivery=False,
            deliver_drills=True,
        )
        self.results: list[NotificationResult] = []

    def dispatch_remote_completion(
        self, command: RemoteActionCommand
    ) -> list[NotificationResult]:
        results = super().dispatch_remote_completion(command)
        self.results.extend(results)
        return results


def build_notification(
    kind: str,
    drill_id: str,
    cluster_id: str,
) -> AdvisoryNotification:
    builder = RestartGuardEmailBuilder()
    if kind == "gpu-reset":
        notification = builder.build_gpu_reset_completed(
            cluster_id=cluster_id,
            incident_id=f"incident-{drill_id}",
            workflow_id=f"workflow-{drill_id}",
            event_id=f"event-{drill_id}",
            event_type="NOTIFY_ACCEPTANCE_DRILL",
            policy_source="ACCEPTANCE_DRILL",
            official_action="RESET_GPU",
            reasons=["notification path acceptance drill; no GPU action occurred"],
            operation_id=f"operation-{drill_id}",
            node_ids=[f"node-{drill_id}"],
            gpu_uuids=[f"GPU-{drill_id}"],
            node_results={
                f"node-{drill_id}": {
                    "status": "SUCCEEDED",
                    "reset_gpu_uuids": [f"GPU-{drill_id}"],
                }
            },
            workload_ids=[],
        )
    else:
        notification = builder.build_workload_restarted(
            cluster_id=cluster_id,
            incident_id=f"incident-{drill_id}",
            workflow_id=f"workflow-{drill_id}",
            operation_id=f"operation-{drill_id}",
            job_id=f"job-{drill_id}",
            source_attempt_id=f"attempt-{drill_id}-source",
            restart_attempt_id=f"attempt-{drill_id}-target",
            workload_ids=[f"training/pytorchjob/job-{drill_id}"],
            source_gpu_count=24,
            target_gpu_count=24,
            restart_count=1,
            restart_budget=1,
        )
    return notification.model_copy(
        update={
            "drill_id": drill_id,
            "subject": f"[DRILL:{drill_id}] {notification.subject}",
            "body_text": (
                "DRILL / 验收演练：未执行 GPU reset、任务重启或节点变更。\n\n"
                + notification.body_text
            ),
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kind", choices=("gpu-reset", "workload-restart"), required=True
    )
    parser.add_argument("--drill-id", required=True)
    parser.add_argument("--cluster-id", required=True)
    parser.add_argument("--maintenance-window-end", default="")
    parser.add_argument("--duplicate-delay-seconds", type=int, default=0)
    arguments = parser.parse_args()
    deadline = (
        datetime.fromisoformat(arguments.maintenance_window_end.replace("Z", "+00:00"))
        if arguments.maintenance_window_end
        else None
    )
    if deadline is not None and deadline.tzinfo is None:
        raise ValueError("notification maintenance window needs a timezone")
    result = replay_completion(
        arguments.kind,
        arguments.drill_id,
        arguments.cluster_id,
        notification_notifier_from_environment(),
        deadline=deadline,
        duplicate_delay_seconds=arguments.duplicate_delay_seconds,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


def replay_completion(
    kind: str,
    drill_id: str,
    cluster_id: str,
    inner: Any,
    *,
    deadline: datetime | None = None,
    duplicate_delay_seconds: int = 0,
) -> dict[str, Any]:
    if (
        type(duplicate_delay_seconds) is not int
        or not 0 <= duplicate_delay_seconds <= 90
    ):
        raise ValueError("duplicate delay must be an integer within 0..90 seconds")
    if deadline is not None and datetime.now(timezone.utc) >= deadline:
        raise RuntimeError("notification maintenance window ended")
    if (
        deadline is not None
        and datetime.now(timezone.utc) + timedelta(seconds=duplicate_delay_seconds)
        >= deadline
    ):
        raise RuntimeError("notification duplicate window exceeds maintenance deadline")
    store = InMemoryStore()
    operation = {
        "gpu-reset": WorkflowOperation.RESET_GPU,
        "workload-restart": WorkflowOperation.RESTART_WORKLOAD,
    }[kind]
    step = WorkflowStepSpec(
        operation=operation,
        execution_owner="notification-drill",
        node_ids=[f"node-{drill_id}"],
        gpu_uuids=[f"GPU-{drill_id}"],
        workload_ids=[f"training/pytorchjob/job-{drill_id}"],
    )
    incident = FaultIncident(
        incident_id=f"incident-{drill_id}",
        event_id=f"event-{drill_id}",
        event_type="NOTIFY_ACCEPTANCE_DRILL",
        cluster_id=cluster_id,
        node_ids=step.node_ids,
        policy_version="notification-drill/v1",
        policy_source="ACCEPTANCE_DRILL",
        official_action=operation.value,
        state=IncidentState.ACTION_PENDING,
        fencing_token=1,
        drill_id=drill_id,
    )
    workflow = WorkflowRequest(
        request_id=f"workflow-{drill_id}",
        incident_id=incident.incident_id,
        status=WorkflowStatus.BLOCKED,
        official_action=operation.value,
        official_steps=[step],
        fencing_token=1,
        blocked_reasons=["isolated result-handler replay; no action is dispatched"],
    )
    store.save_incident(incident)
    store.save_workflow(workflow)
    store.ensure_remote_command(
        RemoteActionCommand(
            command_id=f"remote-{drill_id}",
            cluster_id=cluster_id,
            workflow_request_id=workflow.request_id,
            incident_id=incident.incident_id,
            step_index=0,
            fencing_token=1,
            idempotency_key=f"{workflow.request_id}/0/{operation.value}",
            step=step,
            workflow=workflow,
            incident=incident,
        )
    )
    command = store.claim_remote_commands(
        cluster_id,
        "isolated-executor",
        limit=1,
        lease_seconds=180,
        execution_owners={"notification-drill"},
    )[0]
    payload = RemoteCommandResult(
        lease_token=command.lease_token,
        status=RemoteCommandStatus.SUCCEEDED,
        details={
            "node_results": {
                step.node_ids[0]: {
                    "status": "SUCCEEDED",
                    "reset_gpu_uuids": step.gpu_uuids,
                }
            },
            "notification_context": {
                "job_id": f"job-{drill_id}",
                "source_attempt_id": f"source-{drill_id}",
                "restart_attempt_id": f"target-{drill_id}",
                "source_gpu_count": 8,
                "target_gpu_count": 8,
                "restart_count": 1,
                "restart_budget": 1,
            },
        },
    )
    notifier = DrillNotifier(inner, drill_id, deadline)
    service = CompletionRecorder(store, notifier, drill_id)
    wakes: list[bool] = []
    store_io = AsyncStoreExecutor(
        workers=1, max_in_flight=1, admission_timeout_seconds=1
    )
    dependencies = RegionalRouterDependencies(
        context=SimpleNamespace(
            store=store,
            advisory_notifications=service,
            dispatcher=SimpleNamespace(wake=lambda: wakes.append(True)),
        ),
        store_io=store_io,
        auth_registry={},
        max_unclaimed_seconds=180,
        max_claim_age_seconds=180,
        executor_compatibility=RegionalExecutorCompatibilityPolicy.from_mapping({}),
    )
    executed_at = datetime.now(timezone.utc).isoformat()
    windows: dict[str, str] = {}

    async def replay() -> list[str]:
        statuses = []
        for index in range(4):
            if deadline is not None and datetime.now(timezone.utc) >= deadline:
                raise RuntimeError("notification maintenance window ended")
            if index == 1:
                if duplicate_delay_seconds:
                    time.sleep(duplicate_delay_seconds)
                    if deadline is not None and datetime.now(timezone.utc) >= deadline:
                        raise RuntimeError("notification maintenance window ended")
                windows["duplicate_window_start"] = datetime.now(
                    timezone.utc
                ).isoformat()
            completed = await complete_remote_command(
                command.command_id, payload, cluster_id, dependencies
            )
            statuses.append(completed.status.value)
            if index == 0:
                windows["initial_completed_at"] = datetime.now(timezone.utc).isoformat()
        windows["duplicate_window_end"] = datetime.now(timezone.utc).isoformat()
        return statuses

    try:
        command_statuses = asyncio.run(replay())
    finally:
        store_io.close()
    provider_ids = [item.provider_message_id for item in service.results]
    notifications = store.list_notifications()
    return {
        "kind": kind,
        "drill_id": drill_id,
        "executed_at": executed_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        **windows,
        "duplicate_delay_seconds": duplicate_delay_seconds,
        "notification_id": notifications[0].notification_id
        if len(notifications) == 1
        else None,
        "notification_count": len(notifications),
        "notifier_calls": notifier.calls,
        "injection_path": INJECTION_PATH,
        "completion_calls": len(wakes),
        "command_statuses": command_statuses,
        "statuses": [item.status.value for item in service.results],
        "provider_message_id_present": bool(provider_ids and provider_ids[0]),
        "provider_message_id_stable": bool(provider_ids and provider_ids[0])
        and all(item == provider_ids[0] for item in provider_ids),
        "http_authorization_exercised": False,
    }


if __name__ == "__main__":
    raise SystemExit(main())
