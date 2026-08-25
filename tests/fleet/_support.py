"""Shared fixtures and builders for split test shards."""

from __future__ import annotations

from datetime import datetime, timezone

from gpu_fault.fleet import (
    AgentHeartbeat,
    FleetCompatibilityPolicy,
    FleetRegistry,
    SignedAgentHeartbeat,
    sign_agent_heartbeat,
)
from gpu_fault.models import (
    IncidentState,
    RecoveryAction,
    WorkflowOperation,
    WorkflowRequest,
)
from gpu_fault.store import InMemoryStore
from tests._builders import (
    build_store,
    copy_model,
    fault_incident,
    workflow_request,
    workflow_step,
)

NOW = datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc)

SECRET = "fleet-secret-" + "x" * 32

ARTIFACT = "a" * 64

CONFIG = "c" * 64


def heartbeat(
    node_id: str,
    *,
    version: str = "0.9.0",
    observed_at: datetime = NOW,
    boot_id: str = "boot-a",
    node_instance_id: str = "instance-a",
    agent_incarnation_id: str | None = None,
    collector_services: dict | None = None,
    installed_unit_report=None,
) -> AgentHeartbeat:
    return AgentHeartbeat(
        cluster_id="cluster-a",
        node_id=node_id,
        endpoint=f"http://{node_id}:9099",
        agent_protocol_version=3,
        agent_version=version,
        artifact_sha256=ARTIFACT,
        policy_version="catalog-a",
        runtime_profile_version="profile-a",
        config_digest=CONFIG,
        allowed_operations=[
            WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
            WorkflowOperation.RESET_GPU,
        ],
        collector_services=collector_services or {},
        installed_unit_report=installed_unit_report,
        boot_id=boot_id,
        node_instance_id=node_instance_id,
        agent_incarnation_id=(agent_incarnation_id or boot_id),
        observed_at=observed_at,
    )


def signed(value: AgentHeartbeat) -> SignedAgentHeartbeat:
    return SignedAgentHeartbeat(
        heartbeat=value, signature=sign_agent_heartbeat(value, SECRET)
    )


def registry(store=None, *, now=None) -> FleetRegistry:
    return FleetRegistry(
        store or build_store(),
        SECRET,
        FleetCompatibilityPolicy(
            required_agent_version="0.9.0",
            required_artifact_sha256=ARTIFACT,
            required_policy_version="catalog-a",
            required_runtime_profile_version="profile-a",
            required_config_digest=CONFIG,
        ),
        now=now or (lambda: NOW),
    )


def workflow_state(store: InMemoryStore):
    incident = fault_incident(
        "incident-a",
        "event-a",
        node_ids=["node-a", "node-b"],
        gpu_uuids=["GPU-a", "GPU-b"],
        policy_version="catalog-a",
        policy_source="NVIDIA_CATALOG",
        official_action="RESET_GPU",
        effective_action=RecoveryAction.RESET_GPU,
        state=IncidentState.ACTION_PENDING,
        workflow_request_id="workflow-a",
        fencing_token=3,
    )
    workflow = workflow_request(
        "workflow-a",
        incident.incident_id,
        runtime_profile_version="profile-a",
        official_action="RESET_GPU",
        official_steps=[
            workflow_step(
                WorkflowOperation.RESET_GPU,
                "gpu-fault-node-agent",
                node_ids=["node-a", "node-b"],
                gpu_uuids=["GPU-a", "GPU-b"],
                parameters={
                    "gpu_uuids_by_node": {"node-a": ["GPU-a"], "node-b": ["GPU-b"]}
                },
            )
        ],
    )
    store.save_incident(incident)
    store.save_workflow(workflow)
    return workflow


def quiesce_then_full_reset_workflow(store: InMemoryStore) -> WorkflowRequest:
    workflow = copy_model(
        workflow_state(store),
        official_action="RESET_ALL_GPUS_AND_NVSWITCHES",
        official_steps=[
            workflow_step(
                WorkflowOperation.QUIESCE_GPU_SERVICES,
                "gpu-fault-node-agent",
                node_ids=["node-a", "node-b"],
                gpu_uuids=["GPU-a", "GPU-b"],
                parameters={
                    "gpu_uuids_by_node": {"node-a": ["GPU-a"], "node-b": ["GPU-b"]}
                },
            ),
            workflow_step(
                WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES,
                "gpu-fault-node-agent",
                node_ids=["node-a", "node-b"],
                gpu_uuids=["GPU-a", "GPU-b"],
                parameters={
                    "gpu_uuids_by_node": {"node-a": ["GPU-a"], "node-b": ["GPU-b"]}
                },
            ),
        ],
    )
    store.save_workflow(workflow)
    return workflow
