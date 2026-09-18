from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

from gpu_fault.adapters import KubernetesWorkflowAdapter
from gpu_fault.execution import WorkflowStepOutcome
from gpu_fault.models import WorkflowOperation, WorkflowStepStatus
from tests._builders import workflow_step_execution
from tests.execution._cov95_runtime_restart import RestartHarness


class LogCore:
    def __init__(self) -> None:
        self.pods: list[Any] = [
            {
                "metadata": {"name": "worker-0", "uid": "pod-0", "annotations": {}},
                "spec": {"nodeName": "node-a", "containers": [{"name": "trainer"}]},
            }
        ]
        self.value: bytes | str = b"first\nlast\n"
        self.log_error: Exception | None = None
        self.patch_error: Exception | None = None
        self.delete_error: Exception | None = None
        self.events: list[tuple[str, Any]] = []
        self.mapping_response = False

    def list_namespaced_pod(self, namespace: str, *, label_selector: str) -> Any:
        assert namespace == "training", namespace
        assert (
            label_selector
            == "gpu-fault.io/managed=true,gpu-fault.io/attempt-id=attempt-a"
        ), label_selector
        pods = deepcopy(self.pods)
        return {"items": pods} if self.mapping_response else SimpleNamespace(items=pods)

    def read_namespaced_pod_log(
        self, name: str, namespace: str, **kwargs: Any
    ) -> bytes | str:
        self.events.append(("read", (namespace, name, kwargs)))
        if self.log_error is not None:
            raise self.log_error
        return self.value

    def patch_namespaced_pod(
        self, name: str, namespace: str, body: dict[str, Any]
    ) -> None:
        self.events.append(("patch", (namespace, name, deepcopy(body))))
        if self.patch_error is not None:
            raise self.patch_error

    def delete_namespaced_pod(self, name: str, namespace: str, **kwargs: Any) -> None:
        assert kwargs == {"grace_period_seconds": 0}, kwargs
        self.events.append(("delete", (namespace, name)))
        if self.delete_error is not None:
            raise self.delete_error
        self.pods = []


class EvidenceSink:
    def __init__(self) -> None:
        self.requests: list[Any] = []
        self.error: Exception | None = None

    def capture_evidence(self, request: Any) -> Any:
        self.requests.append(deepcopy(request))
        if self.error is not None:
            raise self.error
        return SimpleNamespace(record_id=request.record_id)


class LogHarness:
    def __init__(self, *, sink: str = "remote", **options: Any) -> None:
        source = RestartHarness("job", terminal=False)
        self.api = source.api
        self.store = source.store
        self.core = LogCore()
        self.sink = EvidenceSink()
        self.adapter = KubernetesWorkflowAdapter(
            core_api=self.core,
            batch_api=self.api,
            custom_api=self.api,
            evidence_sink=self.sink if sink == "remote" else None,
            store=self.store if sink == "local" else None,
            workload_log_max_bytes=4096,
            workload_log_s3_max_bytes=8192,
            workload_log_tail_lines=2,
            request_timeout_seconds=7,
            **options,
        )
        step = source.context.step.model_copy(
            update={
                "operation": WorkflowOperation.STOP_WORKLOADS,
                "parameters": {
                    "termination_initiator_incident_id": source.context.incident.incident_id
                },
            }
        )
        self.context = replace(
            source.context,
            step=step,
            workflow=source.context.workflow.model_copy(
                update={"official_steps": [step]}
            ),
            idempotency_key="workflow-log/0/STOP_WORKLOADS",
        )

    def execute(self) -> WorkflowStepOutcome:
        return self.adapter.execute(self.context)

    def follow(self) -> None:
        record = workflow_step_execution(
            self.context.step_index,
            WorkflowOperation.STOP_WORKLOADS,
            WorkflowStepStatus.WAITING,
        )
        self.context = replace(
            self.context,
            workflow=self.context.workflow.model_copy(
                update={"step_executions": [record]}
            ),
        )
