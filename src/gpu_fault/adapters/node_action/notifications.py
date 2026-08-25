from __future__ import annotations

from typing import Any, Callable

from gpu_fault.execution import (
    WorkflowStepContext,
)
from gpu_fault.models import (
    WorkflowOperation,
    WorkflowStepStatus,
)


class NodeActionNotificationMixin:
    # Attributes supplied by the composed concrete implementation.
    alert_sender: Callable[..., Any]
    restart_email_builder: Any
    store: Any

    def _add_fabric_reset_notification(
        self,
        context: WorkflowStepContext,
        node_results: dict[str, dict[str, Any]],
        details: dict[str, Any],
    ) -> None:
        if self.store is None:
            return
        parameters = context.step.parameters
        notification = self.restart_email_builder.build_fabric_reset_completed(
            cluster_id=context.incident.cluster_id,
            incident_id=context.incident.incident_id,
            workflow_id=context.workflow.request_id,
            event_id=context.incident.event_id,
            event_type=context.incident.event_type,
            policy_source=context.incident.policy_source,
            official_action=context.incident.official_action,
            reasons=context.incident.reasons,
            operation_id=context.idempotency_key,
            node_results=node_results,
            workload_ids=context.step.workload_ids,
            fabric_partition=parameters.get(
                "fabric_partition",
                parameters.get("fabric_partitions_by_node"),
            ),
            sxid=parameters.get("sxid", parameters.get("sxids_by_node")),
        )
        notification = self.store.save_notification_if_absent(notification)
        details["notification_id"] = notification.notification_id
        if self.alert_sender is not None:
            self.alert_sender(notification.notification_id)

    def _retry_fabric_reset_notification(
        self, context: WorkflowStepContext
    ) -> str | None:
        if self.store is None or self.alert_sender is None:
            return None
        reset_execution = next(
            (
                item
                for item in reversed(context.workflow.step_executions)
                if item.step_index < context.step_index
                and item.operation is WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES
                and item.status is WorkflowStepStatus.SUCCEEDED
                and isinstance(item.details.get("notification_id"), str)
            ),
            None,
        )
        if reset_execution is None:
            return None
        notification_id = reset_execution.details["notification_id"]
        self.alert_sender(notification_id)
        return notification_id
