from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

from gpu_fault.execution import (
    WorkflowExecutionRequest,
    WorkflowStepContext,
    WorkflowStepOutcome,
)
from gpu_fault.models import (
    WorkflowExecutionResult,
    WorkflowOperation,
    WorkflowStepSpec,
)
from gpu_fault.store import InMemoryStore
from tests._builders import active_workflow_executor
from tests.execution._support import workflow_state


class RecordingAdapter:
    def __init__(self) -> None:
        self.calls: list[WorkflowStepContext] = []
        self.outcomes: dict[WorkflowOperation, WorkflowStepOutcome] = {}
        self.before: Callable[[WorkflowStepContext], None] | None = None

    def supports(self, step: WorkflowStepSpec) -> bool:
        return step.execution_owner == "owner-a"

    def execute(self, context: WorkflowStepContext) -> WorkflowStepOutcome:
        self.calls.append(deepcopy(context))
        if self.before is not None:
            self.before(context)
        return self.outcomes.get(
            context.step.operation,
            WorkflowStepOutcome.succeeded(operation_id=context.idempotency_key),
        )


class FlowHarness:
    def __init__(self, operations: list[WorkflowOperation]) -> None:
        self.store = InMemoryStore()
        self.incident, self.workflow = workflow_state(self.store, operations)
        now = datetime.now(timezone.utc)
        self.incident = self.incident.model_copy(
            update={"created_at": now, "updated_at": now}
        )
        self.workflow = self.workflow.model_copy(
            update={"created_at": now, "updated_at": now}
        )
        self.store.save_incident_and_workflow(self.incident, self.workflow)
        self.workflow = self.store.get_workflow(self.workflow.request_id)
        self.incident = self.store.get_incident(self.incident.incident_id)
        self.adapter = RecordingAdapter()
        self.executor = active_workflow_executor(self.store, [self.adapter], operations)

    def amend(self, **updates: Any) -> None:
        current = self.store.get_workflow(self.workflow.request_id)
        self.store.save_workflow(current.model_copy(update=updates), expected=current)
        self.workflow = self.store.get_workflow(current.request_id)

    def execute(
        self, request: WorkflowExecutionRequest | None = None
    ) -> WorkflowExecutionResult:
        return self.executor.execute(
            self.workflow.request_id,
            request
            or WorkflowExecutionRequest(
                expected_fencing_token=self.workflow.fencing_token
            ),
        )
