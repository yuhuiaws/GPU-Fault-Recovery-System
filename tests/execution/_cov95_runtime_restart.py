from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any

from gpu_fault.adapters import KubernetesWorkflowAdapter
from gpu_fault.adapters.common import LABEL_ATTEMPT_ID, LABEL_JOB_ID
from gpu_fault.execution import WorkflowStepContext, WorkflowStepOutcome
from gpu_fault.models import WorkflowOperation, WorkflowStepStatus
from gpu_fault.store import InMemoryStore
from tests._builders import workflow_step, workflow_step_execution
from tests.execution.test_restart_safety import UnusedApi, restart_context

COUNTS = {"job": 4, "pytorchjob": 6, "jobset": 12}


class ApiError(RuntimeError):
    def __init__(self, status: int) -> None:
        super().__init__(f"fake Kubernetes status {status}")
        self.status = status


def template() -> dict[str, Any]:
    return {
        "metadata": {
            "labels": {
                LABEL_ATTEMPT_ID: "attempt-a",
                "batch.kubernetes.io/controller-uid": "old-controller",
            },
            "annotations": {"keep": "template"},
        },
        "spec": {
            "restartPolicy": "Never",
            "containers": [
                {"name": "trainer", "resources": {"limits": {"nvidia.com/gpu": "2"}}}
            ],
        },
    }


def source_workload(kind: str, *, terminal: bool) -> dict[str, Any]:
    if kind == "job":
        spec = {"completions": 2, "suspend": True, "template": template()}
        status: dict[str, Any] = {"active": 0, "failed": int(terminal)}
    elif kind == "pytorchjob":
        spec = {
            "runPolicy": {"suspend": True},
            "pytorchReplicaSpecs": {
                "Master": {"replicas": 1, "template": template()},
                "Worker": {"replicas": 2, "template": template()},
            },
        }
        status = {
            "conditions": [
                {"type": "Failed" if terminal else "Suspended", "status": "True"}
            ]
        }
    else:
        spec = {
            "suspend": True,
            "replicatedJobs": [
                {
                    "name": "workers",
                    "replicas": 2,
                    "template": {"spec": {"completions": 3, "template": template()}},
                }
            ],
        }
        status = {
            "conditions": [
                {"type": "Failed" if terminal else "Suspended", "status": "True"}
            ]
        }
    return {
        "metadata": {
            "name": "training-job",
            "namespace": "training",
            "uid": "source-uid",
            "resourceVersion": "5",
            "generation": 2,
            "creationTimestamp": "2026-09-12T12:00:00Z",
            "managedFields": [],
            "labels": {LABEL_JOB_ID: "train-1", LABEL_ATTEMPT_ID: "attempt-a"},
            "annotations": {"keep": "source"},
        },
        "spec": spec,
        "status": status,
    }


def pod_templates(body: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    if kind == "job":
        return [body["spec"]["template"]]
    if kind == "pytorchjob":
        return [row["template"] for row in body["spec"]["pytorchReplicaSpecs"].values()]
    return [
        row["template"]["spec"]["template"] for row in body["spec"]["replicatedJobs"]
    ]


class WorkloadApi:
    """Private in-memory Kubernetes objects; every external method is a fake."""

    def __init__(self, kind: str, *, terminal: bool) -> None:
        self.source: Any = source_workload(kind, terminal=terminal)
        self.created: dict[str, dict[str, Any]] = {}
        self.patches: list[dict[str, Any]] = []
        self.reads: list[tuple[str, str]] = []

    def read(self, name: str, namespace: str) -> Any:
        assert namespace == "training", namespace
        self.reads.append((namespace, name))
        if name == "training-job":
            return deepcopy(self.source)
        if name in self.created:
            return deepcopy(self.created[name])
        raise ApiError(404)

    def read_namespaced_job(self, name: str, namespace: str) -> Any:
        return self.read(name, namespace)

    def patch_namespaced_job(
        self, name: str, namespace: str, body: dict[str, Any]
    ) -> None:
        assert (namespace, name) == ("training", "training-job"), (namespace, name)
        self.patches.append(deepcopy(body))

    def get_namespaced_custom_object(
        self, group: str, version: str, namespace: str, plural: str, name: str
    ) -> Any:
        return self.read(name, namespace)

    def patch_namespaced_custom_object(
        self,
        group: str,
        version: str,
        namespace: str,
        plural: str,
        name: str,
        body: dict[str, Any],
    ) -> None:
        assert (namespace, name) == ("training", "training-job"), (namespace, name)
        self.patches.append(deepcopy(body))

    def create_namespaced_job(self, namespace: str, body: dict[str, Any]) -> None:
        assert namespace == "training", namespace
        self.created[body["metadata"]["name"]] = deepcopy(body)

    def create_namespaced_custom_object(
        self,
        group: str,
        version: str,
        namespace: str,
        plural: str,
        body: dict[str, Any],
    ) -> None:
        self.create_namespaced_job(namespace, body)


class RestartHarness:
    def __init__(
        self, kind: str, *, terminal: bool = True, source_gpu_count: int | None = None
    ) -> None:
        self.kind = kind
        self.api = WorkloadApi(kind, terminal=terminal)
        self.store = InMemoryStore()
        self.adapter = KubernetesWorkflowAdapter(
            core_api=UnusedApi(),
            batch_api=self.api,
            custom_api=self.api,
            store=self.store,
        )
        self.context = restart_context(
            f"shape-{kind}",
            source_gpu_count=COUNTS[kind]
            if source_gpu_count is None
            else source_gpu_count,
            restart_budget=1,
            workload_id=f"training/{kind}/training-job",
            source_attempt_id="attempt-a",
        )

    def execute(self) -> WorkflowStepOutcome:
        return self.adapter.execute(self.context)

    def bind_replacement(self, mapping: Any, *, succeeded: bool = True) -> None:
        current = self.context
        replacement = workflow_step(
            WorkflowOperation.REPLACE_NODE,
            "gpu-fault-hyperpod-adapter",
            node_ids=["node-old"],
        )
        execution = workflow_step_execution(
            0,
            WorkflowOperation.REPLACE_NODE,
            WorkflowStepStatus.SUCCEEDED if succeeded else WorkflowStepStatus.FAILED,
            details={"node_rebindings": mapping},
        )
        workflow = current.workflow.model_copy(
            update={
                "official_steps": [replacement, current.step],
                "step_executions": [execution],
            }
        )
        key = f"{workflow.request_id}/1/RESTART_WORKLOAD"
        authorization = current.request.restart_authorization
        assert authorization is not None, current.request
        request = current.request.model_copy(
            update={
                "restart_authorization": authorization.model_copy(
                    update={"reservation_id": key}
                )
            }
        )
        self.context = replace(
            current,
            workflow=workflow,
            request=request,
            step_index=1,
            idempotency_key=key,
        )


def changed_parameters(
    context: WorkflowStepContext, **parameters: Any
) -> WorkflowStepContext:
    step = context.step.model_copy(
        update={"parameters": {**context.step.parameters, **parameters}}
    )
    return replace(
        context,
        step=step,
        workflow=context.workflow.model_copy(update={"official_steps": [step]}),
    )
