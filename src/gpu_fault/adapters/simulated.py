from __future__ import annotations

from uuid import uuid4

from gpu_fault.models import (
    IncidentState,
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
        """Simulate the plan, honouring its one execution premise.

        The premise is read from ``plan.restart_after_incident_id`` and not
        from the plan's step parameters: the planner no longer writes
        ``requires_incident_state`` onto a step (the compiler derives it for
        the workflow engine), so reading the step here would have turned the
        gate into a no-op and reported SUCCEEDED for a restart whose incident
        was still being repaired (F-G5).
        """

        premise_id = plan.restart_after_incident_id
        if premise_id is not None:
            incident = self.store.get_incident(premise_id)
            if incident.state is not IncidentState.RECOVERED:
                operation = OperationResult(
                    operation_id=f"op-{uuid4()}",
                    plan_id=plan.plan_id,
                    status=PlanStatus.FAILED,
                    error=(
                        f"incident {premise_id} is "
                        f"{incident.state.value}; requires "
                        f"{IncidentState.RECOVERED.value}"
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
