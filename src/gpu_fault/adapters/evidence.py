from __future__ import annotations

from datetime import datetime, timezone

from gpu_fault.execution import (
    WorkflowStepContext,
    WorkflowStepOutcome,
)
from gpu_fault.models import (
    WorkflowStepSpec,
)
from gpu_fault.operation_registry import (
    OperationAdapter,
    operations_for_adapter,
)


class ControlPlaneEvidenceAdapter:
    """Persists the immutable incident/workflow references as evidence."""

    OPERATIONS = operations_for_adapter(OperationAdapter.CONTROL_PLANE)

    def __init__(self, owner: str = "gpu-fault-control-plane") -> None:
        self.owner = owner

    def supports(self, step: WorkflowStepSpec) -> bool:
        return step.execution_owner == self.owner and step.operation in self.OPERATIONS

    def execute(self, context: WorkflowStepContext) -> WorkflowStepOutcome:
        return WorkflowStepOutcome.succeeded(
            operation_id=context.idempotency_key,
            details={
                "incident_id": context.incident.incident_id,
                "event_id": context.incident.event_id,
                "policy_version": context.incident.policy_version,
                "official_action": context.incident.official_action,
                "captured_at": datetime.now(timezone.utc).isoformat(),
            },
        )
