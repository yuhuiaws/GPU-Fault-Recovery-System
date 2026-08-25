from __future__ import annotations


from gpu_fault.execution import (
    WorkflowStepContext,
    WorkflowStepOutcome,
)
from gpu_fault.managed_recovery import (
    HyperPodManagedRecoveryObserver,
)
from gpu_fault.models import (
    WorkflowOperation,
    WorkflowStepSpec,
)
from gpu_fault.operation_registry import (
    OperationAdapter,
    operations_for_adapter,
)


class ManagedRecoveryObserverAdapter:
    """Observes delegated recovery without becoming a second writer."""

    OPERATIONS = operations_for_adapter(OperationAdapter.MANAGED_RECOVERY)

    def __init__(
        self,
        owners: set[str],
        observer: HyperPodManagedRecoveryObserver | None = None,
    ) -> None:
        self.owners = owners
        self.observer = observer

    def supports(self, step: WorkflowStepSpec) -> bool:
        return step.execution_owner in self.owners and step.operation in self.OPERATIONS

    def execute(self, context: WorkflowStepContext) -> WorkflowStepOutcome:
        if (
            context.step.operation is WorkflowOperation.REPLACE_NODE
            and context.step.parameters.get("replacement_strategy")
            == "HEALTHY_WARM_SPARE_ONLY"
        ):
            return WorkflowStepOutcome.failed(
                "healthy warm-spare replacement cannot be delegated "
                "to managed/provider node recovery"
            )
        if self.observer is not None and context.step.operation in {
            WorkflowOperation.RESTART_NODE,
            WorkflowOperation.REPLACE_NODE,
        }:
            return self.observer.observe(context)
        operation_id = (
            f"delegated/{context.workflow.request_id}/"
            f"{context.step_index}/{context.step.operation.value}"
        )
        if operation_id in context.request.confirmed_adapter_operation_ids:
            return WorkflowStepOutcome.succeeded(
                operation_id=operation_id,
                details={
                    "delegated_owner": (context.step.execution_owner),
                    "externally_confirmed": True,
                },
            )
        return WorkflowStepOutcome.waiting(
            operation_id=operation_id,
            details={
                "delegated_owner": context.step.execution_owner,
                "mutation_submitted_by_control_plane": False,
                "requires_external_confirmation": True,
            },
        )
