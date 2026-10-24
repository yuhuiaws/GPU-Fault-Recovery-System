from __future__ import annotations

from typing import Any

from gpu_fault.execution import (
    WorkflowStepContext,
    WorkflowStepOutcome,
)
from gpu_fault.execution.hung_handoff import hung_handoff_workflow
from gpu_fault.execution.hung_handoff_evidence import handoff_evidence
from gpu_fault.models import (
    WorkflowRequest,
    WorkflowStepSpec,
    WorkflowStepStatus,
)
from gpu_fault.notifications import (
    HardwareEscalationEmailBuilder,
    Nvlink74SupportEmailBuilder,
)
from gpu_fault.notifications.hung_support import (
    HUNG_SUPPORT_DISPOSITION,
    HUNG_SUPPORT_HANDOFF_TEMPLATE_VERSION,
    HungSupportHandoffEmailBuilder,
)
from gpu_fault.operation_registry import (
    OperationAdapter,
    operations_for_adapter,
)
from gpu_fault.orchestration.escalation import escalation_origin
from gpu_fault.store.shared.errors import NotFoundError
from gpu_fault.suspended_workloads import suspended_workload_line, suspended_workloads

#: How far the hand-off email walks the escalation chain for suspended jobs.
LINEAGE_LIMIT = 8


def escalation_lineage(
    store: Any, context: WorkflowStepContext
) -> list[WorkflowRequest]:
    """The hand-off workflow and the workflows it escalated from, newest first.

    ``<rung>-after-<workflow>`` event ids (``escalation_origin``) name each
    predecessor; an unreadable record ends the walk.
    """

    lineage = [context.workflow]
    seen = {context.workflow.request_id}
    event_id = context.incident.event_id
    for _ in range(LINEAGE_LIMIT):
        origin = escalation_origin(event_id)
        if origin is None or origin[1] in seen:
            break
        try:
            workflow = store.get_workflow(origin[1])
            event_id = store.get_incident(workflow.incident_id).event_id
        except (KeyError, NotFoundError):
            break
        seen.add(workflow.request_id)
        lineage.append(workflow)
    return lineage


class SupportEscalationAdapter:
    """Creates a durable vendor-ticket record and sends its fixed email.

    The read-only hung DAG's hand-off (``execution/hung_handoff``) sends the
    hung notice instead of the hardware escalation: that diagnostic isolated
    nothing, so neither its mail nor its step details may say otherwise.
    """

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
        self.hung_builder = HungSupportHandoffEmailBuilder()

    def supports(self, step: WorkflowStepSpec) -> bool:
        return step.execution_owner == self.owner and step.operation in self.OPERATIONS

    def execute(self, context: WorkflowStepContext) -> WorkflowStepOutcome:
        if hung_handoff_workflow(context.workflow):
            return self._hung_handoff(context)
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
                    # The jobs the remediation left suspended: the operator's
                    # restore does not restart them (G7, batch 139).
                    "suspended_workloads": [
                        suspended_workload_line(item)
                        for item in suspended_workloads(
                            escalation_lineage(self.store, context)
                        )
                    ],
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

    def _hung_handoff(self, context: WorkflowStepContext) -> WorkflowStepOutcome:
        """The read-only hung diagnostic's notice: what each evidence step
        found, the task-scoped fabric verdict and the operator's next steps.
        Nothing was cordoned, quarantined, stopped or reset, so the details
        carry no hardware disposition."""

        incident = context.incident
        ticket_id = f"vendor-ticket-{incident.incident_id}"
        evidence = handoff_evidence(context.workflow, context.step_index)
        notification = self.hung_builder.build(
            cluster_id=incident.cluster_id,
            incident_id=incident.incident_id,
            workflow_id=context.workflow.request_id,
            event_id=incident.event_id,
            job_id=incident.job_id or evidence.job_id,
            attempt_id=incident.attempt_id or evidence.attempt_id,
            node_ids=context.step.node_ids or incident.node_ids,
            workload_ids=context.step.workload_ids,
            reasons=incident.reasons,
            policy_source=incident.policy_source,
            ticket_id=ticket_id,
            evidence=evidence,
        )
        notification = self.store.save_notification_if_absent(notification)
        if self.alert_sender is not None:
            self.alert_sender(notification.notification_id)
        return WorkflowStepOutcome.succeeded(
            operation_id=context.idempotency_key,
            details={
                "ticket_id": ticket_id,
                "notification_id": notification.notification_id,
                "notification_template": HUNG_SUPPORT_HANDOFF_TEMPLATE_VERSION,
                "support_disposition": HUNG_SUPPORT_DISPOSITION,
                "nodes_isolated": False,
                "workloads_stopped": False,
                "fabric_verdict": evidence.fabric.verdict,
            },
        )
