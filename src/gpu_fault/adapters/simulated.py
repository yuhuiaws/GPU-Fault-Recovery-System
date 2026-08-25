from __future__ import annotations

from uuid import uuid4

from gpu_fault.models import (
    OperationResult,
    PlanStatus,
    RecoveryPlan,
)
from gpu_fault.store.contracts import ControlPlaneStore


class SimulatedRecoveryExecutor:
    """Safe phase-one executor that records success without host changes."""

    def __init__(self, store: ControlPlaneStore) -> None:
        self.store = store
        self.operations: dict[str, OperationResult] = {}

    def execute(self, plan: RecoveryPlan) -> OperationResult:
        for step in plan.steps:
            required_state = step.parameters.get("requires_incident_state")
            incident_id = step.parameters.get("incident_id")
            if required_state and incident_id:
                incident = self.store.get_incident(incident_id)
                if incident.state.value != required_state:
                    operation = OperationResult(
                        operation_id=f"op-{uuid4()}",
                        plan_id=plan.plan_id,
                        status=PlanStatus.FAILED,
                        error=(
                            f"incident {incident_id} is "
                            f"{incident.state.value}; requires "
                            f"{required_state}"
                        ),
                    )
                    self.operations[operation.operation_id] = operation
                    self.store.save_plan(
                        plan.model_copy(update={"status": PlanStatus.FAILED})
                    )
                    return operation

        operation = OperationResult(
            operation_id=f"op-{uuid4()}",
            plan_id=plan.plan_id,
            status=PlanStatus.SUCCEEDED,
            completed_steps=len(plan.steps),
        )
        self.operations[operation.operation_id] = operation
        self.store.save_plan(plan.model_copy(update={"status": PlanStatus.SUCCEEDED}))
        return operation
