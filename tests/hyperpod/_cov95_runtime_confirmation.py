from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

from gpu_fault.adapters import HyperPodLifecycleStepAdapter
from gpu_fault.execution import (
    WorkflowExecutionRequest,
    WorkflowStepContext,
    WorkflowStepOutcome,
)
from gpu_fault.hyperpod import HyperPodNode
from gpu_fault.models import WorkflowOperation, WorkflowStatus, WorkflowStepStatus
from gpu_fault.store import InMemoryStore
from tests._builders import workflow_step_execution
from tests.execution._support import isolated_kubernetes_adapter, workflow_state
from tests.hyperpod.test_managed_recovery import agent

NOW = datetime(2026, 9, 12, 12, tzinfo=timezone.utc)


class ConfirmationProvider:
    def __init__(self, nodes: list[HyperPodNode]) -> None:
        self.nodes = nodes
        self.reads: list[list[str]] = []
        self.fail: Exception | None = None
        self.submissions: list[Any] = []

    def resolve_nodes(
        self, identifiers: list[str], **kwargs: Any
    ) -> list[HyperPodNode]:
        self.reads.append(identifiers)
        if self.fail is not None:
            raise self.fail
        return self.nodes

    def execute_step(self, *args: Any, **kwargs: Any) -> Any:
        self.submissions.append((args, kwargs))
        raise AssertionError("confirmation polling must not submit a provider action")


class ConfirmationHarness:
    def __init__(self, operation: WorkflowOperation, *, stabilization: int = 0) -> None:
        self.store = InMemoryStore()
        self.incident, self.workflow = workflow_state(self.store, [operation])
        self.clock = NOW
        self.ready = True
        self.agent_name = (
            "node-new" if operation is WorkflowOperation.REPLACE_NODE else "node-a"
        )
        self.record = agent(
            self.agent_name, "i-new", "inc-after", generation=2
        ).model_copy(
            update={
                "cluster_id": "cluster-a",
                "boot_id": "boot-after",
                "retired_incarnation_ids": ["boot-before"],
                "first_seen_at": NOW,
                "last_seen_at": NOW,
                "lease_expires_at": NOW + timedelta(minutes=5),
            }
        )
        self.store.save_agent(self.record)
        self.registry = SimpleNamespace(
            store=self.store,
            now=lambda: self.clock,
            readiness=lambda *_args: SimpleNamespace(ready=self.ready),
        )
        node = HyperPodNode(
            node_logical_id="logical",
            instance_id="i-new",
            instance_group_name="workers",
            instance_type="ml.p5.48xlarge",
            status="Running",
            kubernetes_labels={"kubernetes.io/hostname": self.agent_name},
        )
        self.provider = ConfirmationProvider([node])
        self.scheduler = isolated_kubernetes_adapter()
        self.adapter = HyperPodLifecycleStepAdapter(
            self.provider,
            registry=self.registry,
            kubernetes_adapter=self.scheduler,
            post_reboot_stabilization_seconds=stabilization,
        )
        step = self.workflow.official_steps[0].model_copy(
            update={"execution_owner": self.adapter.owner}
        )
        self.workflow = self.workflow.model_copy(
            update={"official_steps": [step], "status": WorkflowStatus.RUNNING}
        )
        self.details: dict[str, Any] = {
            "agent_baselines": {
                "node-a": {
                    "boot_id": "boot-before",
                    "agent_incarnation_id": "inc-before",
                }
            },
            "provider_baselines": {
                "node-a": {
                    "node_logical_id": "logical",
                    "instance_id": "i-old",
                    "kubernetes_node_name": "node-a",
                }
            },
        }
        self.context = WorkflowStepContext(
            workflow=self.workflow,
            incident=self.incident,
            step=step,
            step_index=0,
            request=WorkflowExecutionRequest(expected_fencing_token=3),
            idempotency_key="unit-confirmation",
        )

    def advance(self, seconds: int) -> None:
        self.clock += timedelta(seconds=seconds)

    def execute(self) -> WorkflowStepOutcome:
        record = workflow_step_execution(
            0,
            self.context.step.operation,
            WorkflowStepStatus.WAITING,
            adapter_operation_id="provider-owned",
            details=deepcopy(self.details),
        )
        context = replace(
            self.context,
            workflow=self.workflow.model_copy(update={"step_executions": [record]}),
        )
        outcome = self.adapter.execute(context)
        self.details = deepcopy(outcome.details)
        return outcome
