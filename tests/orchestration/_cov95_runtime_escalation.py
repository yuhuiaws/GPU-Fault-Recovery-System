from __future__ import annotations

from typing import Any

from gpu_fault.app import default_simulated_profile
from gpu_fault.models import (
    IncidentState,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
)
from gpu_fault.orchestration.escalation import HardwareEscalationService
from gpu_fault.store import InMemoryStore
from tests._builders import (
    fault_incident,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)
from tests.orchestration._cov95_runtime_builder import builder

RESTART_CONTEXT = {
    "cluster_id": "cluster-a",
    "job_id": "job-a",
    "source_attempt_id": "attempt-a",
    "source_gpu_count": 2,
    "restart_budget": 1,
}


class EscalationHarness:
    def __init__(
        self,
        operations: list[WorkflowOperation],
        *,
        failed_index: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.store = InMemoryStore()
        self.store.save_profile(default_simulated_profile())
        failed = len(operations) - 1 if failed_index is None else failed_index
        steps = [
            workflow_step(operation, node_ids=["node-a"], gpu_uuids=["GPU-a"])
            for operation in operations
        ]
        self.source = fault_incident(
            "source-incident",
            "source-event",
            cluster_id="cluster-a",
            job_id="job-a",
            attempt_id="attempt-a",
            state=IncidentState.ESCALATED,
            node_ids=["node-a", "node-b"],
            gpu_uuids=["GPU-a"],
            workflow_request_id="source-workflow",
        )
        self.workflow = workflow_request(
            "source-workflow",
            self.source.incident_id,
            WorkflowStatus.FAILED,
            runtime_profile_version="simulated-v1",
            official_steps=steps,
            completed_step_indexes=list(range(failed)),
            completed_operations=operations[:failed],
            step_executions=[
                *[
                    workflow_step_execution(
                        index, operation, WorkflowStepStatus.SUCCEEDED
                    )
                    for index, operation in enumerate(operations[:failed])
                ],
                workflow_step_execution(
                    failed,
                    operations[failed],
                    WorkflowStepStatus.FAILED,
                    error="unit recovery failure",
                    details=details or {},
                ),
            ],
        )
        self.store.save_incident_and_workflow(self.source, self.workflow)
        self.source = self.store.get_incident(self.source.incident_id)
        self.workflow = self.store.get_workflow(self.workflow.request_id)
        self.service = HardwareEscalationService(self.store, builder())

    def amend(self, **updates: Any) -> None:
        current = self.store.get_workflow(self.workflow.request_id)
        self.store.save_workflow(current.model_copy(update=updates), expected=current)
        self.workflow = self.store.get_workflow(current.request_id)

    def escalate(self):
        return self.service.escalate(self.workflow)
