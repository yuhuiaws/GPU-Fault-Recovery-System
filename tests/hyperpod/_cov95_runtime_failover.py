from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

from gpu_fault.adapters import HyperPodLifecycleStepAdapter, KubernetesWorkflowAdapter
from gpu_fault.adapters.common import (
    ANNOTATION_FENCING,
    ANNOTATION_INCIDENT,
    QUARANTINE_TAINT,
    quarantine_taint_value,
)
from gpu_fault.execution import (
    WorkflowExecutionRequest,
    WorkflowStepContext,
    WorkflowStepOutcome,
)
from gpu_fault.hyperpod_spares import SpareAllocation
from gpu_fault.models import WorkflowOperation, WorkflowStatus, WorkflowStepStatus
from gpu_fault.store import InMemoryStore
from tests._builders import workflow_step_execution
from tests.execution._cov95_runtime_restart import ApiError
from tests.execution._support import (
    FakeSpareCoordinator,
    RecordingNodeActionAdapter,
    workflow_state,
)
from tests.execution.test_restart_safety import UnusedApi
from tests.hyperpod.test_hyperpod_spares import FakeCore, hyperpod_node, kubernetes_node


class ReadOnlyProvider:
    def __init__(self) -> None:
        self.preflights = []
        self.submissions = []
        self.node = hyperpod_node("node-a", "i-old")

    def preflight(self, *args, **kwargs):
        self.preflights.append((args, kwargs))
        return SimpleNamespace(
            safe_to_submit=True, gate_failures=[], node_recovery="None"
        )

    def resolve_nodes(self, identifiers, **kwargs):
        assert identifiers == ["node-a"], identifiers
        return [self.node]

    def execute_step(self, *args, **kwargs):
        self.submissions.append((args, kwargs))
        raise AssertionError(
            "warm-spare handling must never submit a provider mutation"
        )


class SnapshotAdapter(RecordingNodeActionAdapter):
    def __init__(self) -> None:
        super().__init__({"spare-a": "http://unit-agent"})
        self.outcomes = {}

    def execute(self, context):
        self.contexts.append(deepcopy(context))
        return self.outcomes.get(
            context.step.operation,
            WorkflowStepOutcome.succeeded(details={"snapshot_triggered": True}),
        )


class FailoverCore(FakeCore):
    def __init__(self, nodes) -> None:
        super().__init__(nodes)
        self.conflict = False
        self.read_error = None

    def patch_node(self, node_id, body):
        if self.conflict and node_id == "spare-a":
            raise ApiError(409)
        return super().patch_node(node_id, body)

    def read_node(self, node_id):
        if self.read_error is not None:
            raise self.read_error
        return super().read_node(node_id)


class CheckingSpares(FakeSpareCoordinator):
    def __init__(self) -> None:
        super().__init__(
            SpareAllocation(
                applicable=True,
                sufficient=True,
                required=1,
                selected_node_ids=("spare-a",),
            )
        )
        self.check_clients = False

    def allocate(self, **kwargs):
        result = super().allocate(**kwargs)
        checker = kwargs.get("gpu_client_checker")
        if self.check_clients and checker is not None:
            reasons = checker(
                hyperpod_node("spare-logical", "i-spare", spare=True),
                "spare-a",
                "reserve",
            )
            if reasons:
                return replace(
                    result,
                    sufficient=False,
                    selected_node_ids=(),
                    reason="; ".join(reasons),
                )
        return result


class FailoverHarness:
    def __init__(self) -> None:
        self.store = InMemoryStore()
        incident, workflow = workflow_state(
            self.store, [WorkflowOperation.REPLACE_NODE]
        )
        old = kubernetes_node()
        old["metadata"]["annotations"].update(
            {
                ANNOTATION_INCIDENT: incident.incident_id,
                ANNOTATION_FENCING: str(workflow.fencing_token),
            }
        )
        old["spec"]["taints"] = [
            {
                "key": QUARANTINE_TAINT,
                "value": quarantine_taint_value(incident.incident_id),
                "effect": "NoSchedule",
            }
        ]
        self.core = FailoverCore(
            {
                "node-a": old,
                "hyperpod-i-old": old,
                "spare-a": kubernetes_node(unschedulable=False),
            }
        )
        self.provider = ReadOnlyProvider()
        self.spares = CheckingSpares()
        self.actions = SnapshotAdapter()
        self.scheduler = KubernetesWorkflowAdapter(
            core_api=self.core,
            batch_api=UnusedApi(),
            custom_api=UnusedApi(),
            store=self.store,
        )
        self.adapter = HyperPodLifecycleStepAdapter(
            self.provider,
            spare_coordinator=self.spares,
            kubernetes_adapter=self.scheduler,
            node_action_adapter=self.actions,
        )
        step = workflow.official_steps[0].model_copy(
            update={
                "execution_owner": self.adapter.owner,
                "parameters": {"replacement_strategy": "HEALTHY_WARM_SPARE_ONLY"},
            }
        )
        self.context = WorkflowStepContext(
            workflow=workflow.model_copy(update={"official_steps": [step]}),
            incident=incident,
            step=step,
            step_index=0,
            request=WorkflowExecutionRequest(
                expected_fencing_token=workflow.fencing_token,
                confirm_cluster_name="hp-cluster",
            ),
            idempotency_key="unit-failover",
        )

    def execute(self):
        return self.adapter.execute(self.context)

    def follow(self, result) -> None:
        record = workflow_step_execution(
            0,
            WorkflowOperation.REPLACE_NODE,
            WorkflowStepStatus.WAITING,
            adapter_operation_id=result.adapter_operation_id,
            details=deepcopy(result.details),
        )
        self.context = replace(
            self.context,
            workflow=self.context.workflow.model_copy(
                update={"status": WorkflowStatus.RUNNING, "step_executions": [record]}
            ),
        )
