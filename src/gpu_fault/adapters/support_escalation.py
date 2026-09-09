from __future__ import annotations

from gpu_fault.execution import (
    WorkflowStepContext,
    WorkflowStepOutcome,
)
from gpu_fault.models import (
    WorkflowStepSpec,
    WorkflowStepStatus,
)
from gpu_fault.notifications import (
    HardwareEscalationEmailBuilder,
    Nvlink74SupportEmailBuilder,
)
from gpu_fault.operation_registry import (
    OperationAdapter,
    operations_for_adapter,
)


class SupportEscalationAdapter:
    """Creates a durable vendor-ticket record and sends its fixed email."""

    OPERATIONS = operations_for_adapter(OperationAdapter.SUPPORT)

    def __init__(
        self,
        store,
        *,
        owner: str = "gpu-fault-support-escalation",
        alert_sender=None,
    ) -> None:
        self.store = store
        self.owner = owner
        self.alert_sender = alert_sender
        self.builder = HardwareEscalationEmailBuilder()
        self.nvlink74_builder = Nvlink74SupportEmailBuilder()

    def supports(self, step: WorkflowStepSpec) -> bool:
        return step.execution_owner == self.owner and step.operation in self.OPERATIONS

    def execute(self, context: WorkflowStepContext) -> WorkflowStepOutcome:
        ticket_id = f"vendor-ticket-{context.incident.incident_id}"
        builder = (
            self.nvlink74_builder
            if context.incident.official_action == "WORKFLOW_NVLINK_ERR"
            else self.builder
        )
        notification = builder.build(
            cluster_id=context.incident.cluster_id,
            incident_id=context.incident.incident_id,
            workflow_id=context.workflow.request_id,
            event_id=context.incident.event_id,
            node_ids=context.step.node_ids,
            workload_ids=context.step.workload_ids,
            reasons=context.incident.reasons,
            policy_source=context.incident.policy_source,
            official_action=context.incident.official_action,
            ticket_id=ticket_id,
            **(
                {
                    "event_type": context.incident.event_type,
                    "failed_operations": sorted(
                        {
                            execution.operation.value
                            for execution in context.workflow.step_executions
                            if execution.status is WorkflowStepStatus.FAILED
                        }
                    ),
                }
                if builder is self.builder
                else {}
            ),
        )
        notification = self.store.save_notification_if_absent(notification)
        if self.alert_sender is not None:
            self.alert_sender(notification.notification_id)
        return WorkflowStepOutcome.succeeded(
            operation_id=context.idempotency_key,
            details={
                "ticket_id": ticket_id,
                "notification_id": notification.notification_id,
                "hardware_disposition": "OFFLINE_QUARANTINED",
            },
        )
