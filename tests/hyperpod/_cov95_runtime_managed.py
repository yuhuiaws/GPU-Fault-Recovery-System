from __future__ import annotations

from gpu_fault.execution import WorkflowStepContext, WorkflowStepOutcome
from gpu_fault.models import WorkflowOperation
from tests.hyperpod.test_managed_recovery import (
    agent,
    context_with_previous,
    node,
    observer_fixture,
)


class ManagedHarness:
    def __init__(
        self, operation: WorkflowOperation = WorkflowOperation.REPLACE_NODE
    ) -> None:
        (
            self.store,
            self.lifecycle,
            self.identities,
            self.observer,
            self.isolation,
            self.incident,
            self.workflow,
            self.step,
            self.request,
        ) = observer_fixture(operation)
        self.store.save_agent(agent("k8s-old", "i-old", "boot-old"))
        self.context = WorkflowStepContext(
            workflow=self.workflow,
            incident=self.incident,
            step=self.step,
            step_index=0,
            request=self.request,
            idempotency_key="workflow-managed/0/managed",
        )

    def start(self) -> WorkflowStepOutcome:
        return self.observer.observe(self.context)

    def follow(self, previous: WorkflowStepOutcome) -> WorkflowStepOutcome:
        return self.observer.observe(
            context_with_previous(
                self.incident, self.workflow, self.step, self.request, previous
            )
        )

    def replacement(self) -> None:
        self.lifecycle.node = node("i-new", "k8s-new")
        self.store.save_agent(agent("k8s-new", "i-new", "boot-new"))
