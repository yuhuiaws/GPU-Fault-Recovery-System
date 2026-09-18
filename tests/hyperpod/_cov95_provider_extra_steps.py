from __future__ import annotations

from gpu_fault.adapters import HyperPodLifecycleStepAdapter
from gpu_fault.execution import WorkflowExecutionRequest, WorkflowStepContext
from gpu_fault.hyperpod import HyperPodNode, HyperPodPreflight
from gpu_fault.models import WorkflowOperation
from gpu_fault.store import InMemoryStore
from tests.execution._support import (
    FakeHyperPodLifecycle,
    isolated_kubernetes_adapter,
    workflow_state,
)


class RecordingLifecycle(FakeHyperPodLifecycle):
    def __init__(self):
        super().__init__()
        self.preflights = []
        self.resolutions = []
        self.failures = []
        self.node = HyperPodNode(
            node_logical_id="logical-node-a",
            instance_id="i-unit",
            status="Running",
            kubernetes_labels={"kubernetes.io/hostname": "node-a"},
        )

    def resolve_nodes(self, identifiers, **kwargs):
        self.resolutions.append(list(identifiers))
        return [self.node]

    def preflight(self, action, identifiers, **kwargs):
        self.preflights.append((action, list(identifiers), kwargs))
        return HyperPodPreflight(
            action=action,
            cluster_name="hp-cluster",
            cluster_status="InService",
            node_recovery="None",
            targets=[self.node],
            execution_enabled=True,
            safe_to_submit=True,
        )

    def execute_step(self, step, **kwargs):
        result = super().execute_step(step, **kwargs)
        if self.failures:
            return result.model_copy(
                update={"successful_node_logical_ids": [], "failures": self.failures}
            )
        return result


class InitialStep:
    def __init__(self):
        self.store = InMemoryStore()
        self.incident, self.workflow = workflow_state(
            self.store, [WorkflowOperation.RESTART_NODE]
        )
        self.provider = RecordingLifecycle()
        self.scheduler = isolated_kubernetes_adapter()
        self.adapter = HyperPodLifecycleStepAdapter(
            self.provider, kubernetes_adapter=self.scheduler, store=self.store
        )
        step = self.workflow.official_steps[0].model_copy(
            update={"execution_owner": self.adapter.owner}
        )
        self.workflow = self.workflow.model_copy(update={"official_steps": [step]})
        self.context = WorkflowStepContext(
            workflow=self.workflow,
            incident=self.incident,
            step=step,
            step_index=0,
            request=WorkflowExecutionRequest(
                expected_fencing_token=self.workflow.fencing_token,
                confirm_cluster_name="hp-cluster",
            ),
            idempotency_key="unit-provider-extra/reboot",
        )

    def execute(self):
        return self.adapter.execute(self.context)
