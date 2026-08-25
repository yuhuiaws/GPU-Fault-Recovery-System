"""Shared test builders for repeated control-plane setup."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from typing import Any

import httpx

from gpu_fault.app import ApplicationContext, create_app
from gpu_fault.execution import ProductionExecutorConfig, ProductionWorkflowExecutor
from gpu_fault.gpu_metrics import GpuMetricBatch
from gpu_fault.host_health import HostTelemetryBatch, NodeHealthFinding
from gpu_fault.models import (
    Environment,
    FaultIncident,
    WorkflowExecutionRequest,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepExecution,
    WorkflowStepSpec,
    WorkflowStepStatus,
)
from gpu_fault.node_agent import NodeActionResult, NodeActionStatus
from gpu_fault.policy import SxidEvent
from gpu_fault.processor import ProcessorRequest
from gpu_fault.store import InMemoryStore
from gpu_fault.watcher import AttemptObservation, ContainerObservation, WorkloadPhase


def build_store(**overrides: Any) -> InMemoryStore:
    store = InMemoryStore()
    for name, value in overrides.items():
        setattr(store, name, value)
    return store


def build_context(**overrides: Any) -> ApplicationContext:
    return ApplicationContext(**overrides)


def copy_model(model: Any, **updates: Any) -> Any:
    return model.model_copy(update=updates)


def workflow_request(
    request_id: str,
    incident_id: str,
    status: Any = WorkflowStatus.PENDING,
    fencing_token: int = 3,
    **values: Any,
) -> WorkflowRequest:
    return WorkflowRequest(
        request_id=request_id,
        incident_id=incident_id,
        status=status,
        fencing_token=fencing_token,
        **values,
    )


def fault_incident(
    incident_id: str,
    event_id: str,
    event_type: str = "XID",
    cluster_id: str = "cluster-a",
    *,
    node_ids: list[str] | None = None,
    policy_version: str = "610",
    policy_source: str = "NVIDIA",
    **values: Any,
) -> FaultIncident:
    return FaultIncident(
        incident_id=incident_id,
        event_id=event_id,
        event_type=event_type,
        cluster_id=cluster_id,
        node_ids=["node-a"] if node_ids is None else node_ids,
        policy_version=policy_version,
        policy_source=policy_source,
        **values,
    )


def attempt_observation(
    job_id: str,
    attempt_id: str,
    observed_at: Any,
    *,
    cluster_id: str = "cluster-a",
    environment: Any = Environment.HYPERPOD_EKS,
    workload_phase: Any = WorkloadPhase.RUNNING,
    expected_critical_ranks: int = 1,
    runtime_profile_version: str = "simulated-v1",
    **values: Any,
) -> AttemptObservation:
    return AttemptObservation(
        cluster_id=cluster_id,
        environment=environment,
        job_id=job_id,
        attempt_id=attempt_id,
        workload_phase=workload_phase,
        observed_at=observed_at,
        expected_critical_ranks=expected_critical_ranks,
        runtime_profile_version=runtime_profile_version,
        **values,
    )


def node_health_finding(
    finding_id: str,
    event_id: str,
    *,
    cluster_id: str = "cluster-a",
    node_id: str = "node-a",
    **values: Any,
) -> NodeHealthFinding:
    return NodeHealthFinding(
        finding_id=finding_id,
        event_id=event_id,
        cluster_id=cluster_id,
        node_id=node_id,
        **values,
    )


def workflow_step(
    operation: WorkflowOperation,
    execution_owner: str = "owner-a",
    *,
    node_ids: list[str] | None = None,
    **values: Any,
) -> WorkflowStepSpec:
    return WorkflowStepSpec(
        operation=operation,
        execution_owner=execution_owner,
        node_ids=["node-a"] if node_ids is None else node_ids,
        **values,
    )


def workflow_step_execution(
    step_index: int,
    operation: WorkflowOperation,
    status: Any = WorkflowStepStatus.SUCCEEDED,
    **values: Any,
) -> WorkflowStepExecution:
    return WorkflowStepExecution(
        step_index=step_index, operation=operation, status=status, **values
    )


def node_action_result(
    command_id: str,
    operation: WorkflowOperation,
    status: Any = NodeActionStatus.SUCCEEDED,
    **values: Any,
) -> NodeActionResult:
    return NodeActionResult(
        command_id=command_id, operation=operation, status=status, **values
    )


def container_observation(
    pod_uid: str,
    pod_name: str,
    rank: int,
    node_id: str,
    *,
    container_name: str = "trainer",
    role: str = "worker",
    **values: Any,
) -> ContainerObservation:
    return ContainerObservation(
        pod_uid=pod_uid,
        pod_name=pod_name,
        container_name=container_name,
        role=role,
        rank=rank,
        node_id=node_id,
        **values,
    )


def host_telemetry_batch(
    batch_id: str,
    observed_at: Any,
    samples: list[Any],
    *,
    cluster_id: str = "cluster-a",
    node_id: str = "node-a",
    **values: Any,
) -> HostTelemetryBatch:
    return HostTelemetryBatch(
        batch_id=batch_id,
        cluster_id=cluster_id,
        node_id=node_id,
        observed_at=observed_at,
        samples=samples,
        **values,
    )


def gpu_metric_batch(
    batch_id: str,
    observed_at: Any,
    source: Any,
    samples: list[Any],
    *,
    cluster_id: str = "cluster-a",
    node_id: str = "node-a",
    **values: Any,
) -> GpuMetricBatch:
    return GpuMetricBatch(
        batch_id=batch_id,
        cluster_id=cluster_id,
        node_id=node_id,
        observed_at=observed_at,
        source=source,
        samples=samples,
        **values,
    )


def build_sxid_event(
    event_id: str,
    observed_at: Any,
    sxid: int,
    classification: Any,
    classification_source: str,
    *,
    cluster_id: str = "cluster-a",
    node_id: str = "node-a",
    **values: Any,
) -> SxidEvent:
    return SxidEvent(
        event_id=event_id,
        cluster_id=cluster_id,
        node_id=node_id,
        observed_at=observed_at,
        sxid=sxid,
        classification=classification,
        classification_source=classification_source,
        **values,
    )


def active_workflow_executor(
    store: Any,
    adapters: Iterable[Any],
    operations: Iterable[WorkflowOperation],
    *,
    executor_id: str = "executor-a",
    lease_duration_seconds: int = 180,
    workflow_execution_timeout_seconds: int = 3600,
    workflow_preemption_enabled: bool = True,
    notification_sender: Any = None,
) -> ProductionWorkflowExecutor:
    return ProductionWorkflowExecutor(
        store,
        list(adapters),
        ProductionExecutorConfig(
            enabled=True,
            executor_id=executor_id,
            allowed_operations=frozenset(operations),
            lease_duration_seconds=lease_duration_seconds,
            workflow_execution_timeout_seconds=(workflow_execution_timeout_seconds),
            workflow_preemption_enabled=workflow_preemption_enabled,
        ),
        notification_sender=notification_sender,
    )


def execute_workflow(
    executor: ProductionWorkflowExecutor,
    request_id: str,
    *,
    expected_fencing_token: int = 3,
    **request_overrides: Any,
) -> Any:
    return executor.execute(
        request_id,
        WorkflowExecutionRequest(
            expected_fencing_token=expected_fencing_token, **request_overrides
        ),
    )


def processor_request(
    path: str,
    *,
    method: str = "POST",
    query: str = "",
    body: bytes = b"{}",
    content_type: str | None = "application/json",
    cluster_id: str | None = "cluster-a",
    execution_authorized: bool = False,
    parsed_payload: dict[str, Any] | None = None,
) -> ProcessorRequest:
    return ProcessorRequest.from_http(
        method=method,
        path=path,
        query=query,
        body=body,
        content_type=content_type,
        cluster_id=cluster_id,
        execution_authorized=execution_authorized,
        parsed_payload=parsed_payload,
    )


@asynccontextmanager
async def asgi_client(
    target: ApplicationContext | Any,
) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(target) if isinstance(target, ApplicationContext) else target
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client
