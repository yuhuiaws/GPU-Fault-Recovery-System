from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Any

from kubernetes import client

from gpu_fault.adapters import KubernetesWorkflowAdapter
from gpu_fault.execution import WorkflowExecutionRequest, WorkflowStepContext
from gpu_fault.models import WorkflowOperation
from gpu_fault.store import InMemoryStore
from tests.execution._cov95_runtime_restart import ApiError
from tests.execution._support import workflow_state
from tests.execution.test_restart_safety import UnusedApi
from tests.hyperpod.test_hyperpod_spares import FakeCore


class PluginCore(FakeCore):
    def __init__(self, resource: str, *, typed: bool) -> None:
        super().__init__(
            {
                "node-a": {
                    "metadata": {"resourceVersion": "1", "annotations": {}},
                    "spec": {},
                    "status": {"allocatable": {resource: "0"}},
                }
            }
        )
        self.typed = typed
        self.pods: list[Any] = [
            {
                "metadata": {"name": "plugin-old", "uid": "pod-old"},
                "status": {"conditions": [{"type": "Ready", "status": "True"}]},
            }
        ]
        self.deleted: list[tuple[str, str]] = []
        self.queries: list[tuple[str, str, str]] = []
        self.delete_error: Exception | None = None
        self.clear_error = False
        self.conflicts = 0
        self.patch_attempts = 0

    def read_node(self, node_id: str) -> Any:
        value = deepcopy(super().read_node(node_id))
        if not self.typed:
            return value
        return client.V1Node(
            metadata=client.V1ObjectMeta(
                resource_version=value["metadata"]["resourceVersion"],
                annotations=value["metadata"]["annotations"],
            ),
            spec=client.V1NodeSpec(),
            status=client.V1NodeStatus(allocatable=value["status"]["allocatable"]),
        )

    def patch_node(self, node_id: str, body: dict[str, Any]) -> None:
        self.patch_attempts += 1
        if self.conflicts:
            self.conflicts -= 1
            raise ApiError(409)
        if self.clear_error and any(
            value is None for value in body["metadata"]["annotations"].values()
        ):
            raise RuntimeError("fake annotation cleanup failure")
        return super().patch_node(node_id, body)

    def list_namespaced_pod(
        self, namespace: str, *, label_selector: str, field_selector: str
    ) -> Any:
        self.queries.append((namespace, label_selector, field_selector))
        return SimpleNamespace(items=deepcopy(self.pods))

    def delete_namespaced_pod(self, name: str, namespace: str, **kwargs: Any) -> None:
        assert kwargs == {"grace_period_seconds": 0}, kwargs
        self.deleted.append((namespace, name))
        if self.delete_error is not None:
            raise self.delete_error


class PluginHarness:
    def __init__(
        self, family: str = "gpu", *, typed: bool = False, **parameters: Any
    ) -> None:
        self.family = family
        self.resource = "vpc.amazonaws.com/efa" if family == "efa" else "nvidia.com/gpu"
        operation = (
            WorkflowOperation.RESTART_EFA_DEVICE_PLUGIN
            if family == "efa"
            else WorkflowOperation.RESTART_GPU_DEVICE_PLUGIN
        )
        self.core = PluginCore(self.resource, typed=typed)
        self.node = self.core.nodes["node-a"]
        self.annotations = self.node["metadata"]["annotations"]
        store = InMemoryStore()
        incident, workflow = workflow_state(store, [operation])
        self.adapter = KubernetesWorkflowAdapter(
            core_api=self.core, batch_api=UnusedApi(), custom_api=UnusedApi()
        )
        step = workflow.official_steps[0].model_copy(
            update={
                "execution_owner": self.adapter.owner,
                "parameters": {
                    "expected_count": 8,
                    "restart_timeout_seconds": 60,
                    **parameters,
                },
            }
        )
        workflow = workflow.model_copy(update={"official_steps": [step]})
        self.context = WorkflowStepContext(
            workflow=workflow,
            incident=incident,
            step=step,
            step_index=0,
            request=WorkflowExecutionRequest(
                expected_fencing_token=workflow.fencing_token
            ),
            idempotency_key=f"{workflow.request_id}/0/{operation.value}",
        )

    def key(self, field: str) -> str:
        return f"gpu-fault.io/{self.family}-plugin-restart-{field}"

    def execute(self):
        return self.adapter.execute(self.context)
