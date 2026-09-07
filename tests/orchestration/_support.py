# ruff: noqa: F401
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.app import ApplicationContext
from gpu_fault.fleet import AgentLifecycleState, AgentRecord
from gpu_fault.host_health import NodeHealthCategory, NodeHealthFinding
from gpu_fault.models import (
    CapabilityMode,
    CapabilityName,
    Environment,
    FaultIncident,
    IncidentState,
    PlanStatus,
    RecoveryAction,
    TerminalEvent,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepExecution,
    WorkflowStepSpec,
    WorkflowStepStatus,
    WorkloadState,
)
from gpu_fault.operation_registry import (
    CONTAINMENT_ONLY_OPERATIONS,
    DESTRUCTIVE_OPERATIONS,
    NODE_MUTATING_OPERATIONS,
)
from gpu_fault.orchestration import (
    DagBrancher,
    IncidentOrchestrator,
    RecoveryArbiter,
    WorkflowFencingError,
)
from gpu_fault.policy import (
    ActionDisposition,
    SxidClassification,
    SxidEvent,
    SxidLinkScope,
    XidEvent,
)
from gpu_fault.watcher import AttemptObservation, ContainerObservation, WorkloadPhase
from tests._builders import (
    attempt_observation,
    container_observation,
    node_health_finding,
)

NOW = datetime(2026, 7, 20, 8, 0, tzinfo=timezone.utc)


def event(
    code: int,
    *,
    event_id: str,
    workload_state: WorkloadState = WorkloadState.IDLE,
    affected_workload_ids: list[str] | None = None,
) -> XidEvent:
    return XidEvent(
        event_id=event_id,
        cluster_id="cluster-a",
        node_id="node-a",
        observed_at=NOW,
        xid=code,
        gpu_uuid="GPU-a",
        product="H100",
        driver_branch=575,
        cuda_version="12.9",
        runtime_profile_version="simulated-v1",
        workload_state=workload_state,
        affected_workload_ids=affected_workload_ids or [],
    )


def ingest(context: ApplicationContext, xid_event: XidEvent):
    decision = context.policy.evaluate_xid(xid_event)
    return (decision, *context.orchestrator.ingest(xid_event, decision))


def _active_agent(*, boot_id: str) -> AgentRecord:
    return AgentRecord(
        cluster_id="cluster-a",
        node_id="node-a",
        endpoint="http://node-a:9099",
        agent_protocol_version=3,
        agent_version="0.10.0",
        artifact_sha256="a" * 64,
        policy_version="policy-v1",
        runtime_profile_version="simulated-v1",
        config_digest="b" * 64,
        allowed_operations=[
            WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
            WorkflowOperation.RESET_GPU,
        ],
        boot_id=boot_id,
        node_instance_id="instance-a",
        agent_incarnation_id=boot_id,
        first_seen_at=NOW - timedelta(minutes=1),
        last_seen_at=NOW,
        lease_expires_at=NOW + timedelta(minutes=1),
        lifecycle_state=AgentLifecycleState.ACTIVE,
    )


def _node_event(
    code: int, *, event_id: str, gpu_uuid: str, node_id: str = "node-a"
) -> XidEvent:
    return event(code, event_id=event_id).model_copy(
        update={"gpu_uuid": gpu_uuid, "node_id": node_id}
    )


def _claims_the_node(workflow: WorkflowRequest) -> bool:
    return bool(
        {step.operation for step in workflow.official_steps}
        & {
            WorkflowOperation.QUIESCE_GPU_SERVICES,
            WorkflowOperation.RESET_GPU,
            WorkflowOperation.RESTART_NODE,
            WorkflowOperation.REPLACE_NODE,
            WorkflowOperation.RESTORE_SCHEDULING,
        }
    )


def _save_attempt(context: ApplicationContext, node_ids: tuple[str, ...]) -> None:
    context.store.save_attempt_observation(
        attempt_observation(
            "job-a",
            "attempt-a",
            NOW,
            expected_critical_ranks=len(node_ids),
            containers=[
                container_observation(
                    f"pod-{index}",
                    f"worker-{index}",
                    index,
                    node_id,
                    gpu_uuids=[f"GPU-{node_id[-1]}"],
                )
                for index, node_id in enumerate(node_ids)
            ],
            workload_ids=["training/job/job-a"],
            restart_budget=2,
        )
    )


def _inventory_mismatch_finding(
    *,
    event_id: str,
    workload_state: WorkloadState = WorkloadState.IDLE,
    affected_workload_ids: list[str] | None = None,
) -> NodeHealthFinding:
    return node_health_finding(
        f"finding-{event_id}",
        event_id,
        observed_at=NOW,
        category=NodeHealthCategory.GPU,
        severity="critical",
        metric_name="gpu_inventory_mismatch",
        reason="active GPU inventory does not match the configured node invariant",
        recommended_action=RecoveryAction.REBOOT_NODE,
        runtime_profile_version="simulated-v1",
        workload_state=workload_state,
        affected_workload_ids=affected_workload_ids or [],
    )


def _device_resource_finding(
    *, event_id: str, metric_name: str, action: RecoveryAction
) -> NodeHealthFinding:
    return node_health_finding(
        f"finding-{event_id}",
        event_id,
        observed_at=NOW,
        category=NodeHealthCategory.GPU
        if metric_name.startswith("gpu_")
        else NodeHealthCategory.RDMA,
        severity="critical",
        metric_name=metric_name,
        reason=metric_name,
        recommended_action=action,
        runtime_profile_version="simulated-v1",
        workload_state=WorkloadState.IDLE,
        diagnostic_parameters={
            "expected_count": "16",
            "failure_mode": "DRIVER_UNBOUND"
            if action is RecoveryAction.REMEDIATE_EFA_DRIVER
            else "KUBERNETES_RESOURCE_MISSING",
        },
    )


__all__ = [name for name in globals() if not name.startswith("__")]
