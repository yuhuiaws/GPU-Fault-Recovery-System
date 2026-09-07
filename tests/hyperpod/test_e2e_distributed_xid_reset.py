from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from gpu_fault.adapters import NodeActionWorkflowAdapter
from gpu_fault.app import ApplicationContext, default_simulated_profile
from gpu_fault.execution import WorkflowStepContext, WorkflowStepOutcome
from gpu_fault.fleet import (
    AgentHeartbeat,
    BarrierCoordinator,
    BarrierState,
    FleetCompatibilityPolicy,
    FleetRegistry,
    SignedAgentHeartbeat,
    sign_agent_heartbeat,
)
from gpu_fault.models import CapabilityName, WorkflowOperation, WorkflowStatus
from gpu_fault.store import InMemoryStore
from tests._builders import (
    active_workflow_executor,
    asgi_client,
    build_store,
    copy_model,
    execute_workflow,
    node_action_result,
)

NOW = datetime(2026, 7, 20, 13, 0, tzinfo=timezone.utc)
SECRET = "distributed-reset-secret-" + "x" * 32
ARTIFACT = "a" * 64
CONFIG = "c" * 64
CLUSTER = "three-node-training-cluster"
NODES = ["worker-0", "worker-1", "worker-2"]
FAULT_NODES = NODES[:2]


class WorkloadAdapter:
    def __init__(self) -> None:
        self.calls: list[tuple[WorkflowOperation, list[str]]] = []
        self.stop_operation_id = "stop/distributed-training"

    def supports(self, step) -> bool:
        return step.operation not in {
            WorkflowOperation.QUIESCE_GPU_SERVICES,
            WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
            WorkflowOperation.RESET_GPU,
            WorkflowOperation.RESTORE_GPU_SERVICES,
        }

    def execute(self, context: WorkflowStepContext) -> WorkflowStepOutcome:
        self.calls.append((context.step.operation, context.step.node_ids))
        if (
            context.step.operation is WorkflowOperation.STOP_WORKLOADS
            and self.stop_operation_id
            not in context.request.confirmed_adapter_operation_ids
        ):
            return WorkflowStepOutcome.waiting(operation_id=self.stop_operation_id)
        return WorkflowStepOutcome.succeeded(operation_id=context.idempotency_key)


def register_agents(store: InMemoryStore) -> FleetRegistry:
    registry = FleetRegistry(
        store,
        SECRET,
        FleetCompatibilityPolicy(
            required_agent_version="0.9.0",
            required_artifact_sha256=ARTIFACT,
            required_policy_version="catalog-a",
            required_runtime_profile_version="profile-a",
            required_config_digest=CONFIG,
        ),
        now=lambda: NOW,
    )
    for node_id in NODES:
        heartbeat = AgentHeartbeat(
            cluster_id=CLUSTER,
            node_id=node_id,
            endpoint=f"http://{node_id}:9099",
            agent_protocol_version=3,
            agent_version="0.9.0",
            artifact_sha256=ARTIFACT,
            policy_version="catalog-a",
            runtime_profile_version="profile-a",
            config_digest=CONFIG,
            allowed_operations=[
                WorkflowOperation.QUIESCE_GPU_SERVICES,
                WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
                WorkflowOperation.RESET_GPU,
                WorkflowOperation.RESTORE_GPU_SERVICES,
            ],
            boot_id=f"boot-{node_id}",
            node_instance_id=f"instance-{node_id}",
            agent_incarnation_id=f"incarnation-{node_id}",
            observed_at=NOW,
        )
        registry.register(
            SignedAgentHeartbeat(
                heartbeat=heartbeat, signature=sign_agent_heartbeat(heartbeat, SECRET)
            )
        )
    return registry


def test_three_node_job_stops_before_two_fault_gpu_resets() -> None:
    store = build_store()
    default_profile = default_simulated_profile()
    profile = copy_model(
        default_profile,
        cluster_id=CLUSTER,
        profile_version="profile-a",
        capabilities=[
            copy_model(item, owner="gpu-fault-node-agent")
            if item.capability
            in {CapabilityName.GPU_RESET, CapabilityName.DEEP_DIAGNOSTICS}
            else item
            for item in default_profile.capabilities
        ],
    )
    store.save_profile(profile)
    registry = register_agents(store)
    barriers = BarrierCoordinator(store, now=lambda: NOW)
    context = ApplicationContext(
        store=store, fleet_registry=registry, barrier_coordinator=barriers
    )

    async def detect_faults() -> str:
        async with asgi_client(context) as client:
            response = await client.post(
                "/v1/gpu-events/xid/distributed",
                json={
                    "batch_id": "xid-batch-training-1",
                    "job_id": "training-job-1",
                    "attempt_id": "training-job-1-a001",
                    "restart_budget": 2,
                    "affected_workload_ids": ["training/pytorchjob/training-job-1"],
                    "allocation": [
                        {"node_id": node_id, "rank": rank, "gpu_uuids": [f"GPU-{rank}"]}
                        for rank, node_id in enumerate(NODES)
                    ],
                    "events": [
                        {
                            "event_id": f"xid95-{node_id}",
                            "cluster_id": CLUSTER,
                            "node_id": node_id,
                            "observed_at": NOW.isoformat(),
                            "xid": 95,
                            "gpu_uuid": f"GPU-{rank}",
                            "product": "H200",
                            "driver_branch": 575,
                            "cuda_version": "12.9",
                            "job_id": "training-job-1",
                            "runtime_profile_version": "profile-a",
                            "workload_state": "ACTIVE",
                            "affected_workload_ids": [
                                "training/pytorchjob/training-job-1"
                            ],
                        }
                        for rank, node_id in enumerate(FAULT_NODES)
                    ],
                },
            )
            assert response.status_code == 200, response.text
            body = response.json()
            assert {item["official_action"] for item in body["decisions"]} == {
                "RESET_GPU"
            }
            return body["workflow"]["request_id"]

    workflow_id = asyncio.run(detect_faults())
    workflow = store.get_workflow(workflow_id)
    by_operation = {step.operation: step for step in workflow.official_steps}
    assert by_operation[WorkflowOperation.STOP_WORKLOADS].node_ids == NODES
    assert by_operation[WorkflowOperation.STOP_WORKLOADS].workload_ids == [
        "training/pytorchjob/training-job-1"
    ]
    assert by_operation[WorkflowOperation.RESTART_WORKLOAD].node_ids == NODES
    assert by_operation[WorkflowOperation.RESTART_WORKLOAD].parameters == {
        "cluster_id": CLUSTER,
        "job_id": "training-job-1",
        "source_attempt_id": "training-job-1-a001",
        "source_gpu_count": 3,
        "restart_budget": 2,
    }
    assert by_operation[WorkflowOperation.RESET_GPU].node_ids == FAULT_NODES
    assert by_operation[WorkflowOperation.RESET_GPU].parameters[
        "gpu_uuids_by_node"
    ] == {"worker-0": ["GPU-0"], "worker-1": ["GPU-1"]}
    assert by_operation[WorkflowOperation.VERIFY_NO_GPU_CLIENTS].parameters[
        "gpu_uuids_by_node"
    ] == {"worker-0": ["GPU-0"], "worker-1": ["GPU-1"]}

    sent = []

    def sender(endpoint, envelope):
        sent.append((endpoint, envelope.command))
        return node_action_result(
            envelope.command.command_id,
            envelope.command.operation,
            details={"node_id": envelope.command.node_id},
        )

    workload = WorkloadAdapter()
    node_actions = NodeActionWorkflowAdapter(
        {}, SECRET, registry=registry, barriers=barriers, sender=sender
    )
    executor = active_workflow_executor(
        store,
        [workload, node_actions],
        (step.operation for step in workflow.official_steps),
    )

    stopped = execute_workflow(executor, workflow_id, expected_fencing_token=1)
    assert stopped.status is WorkflowStatus.RUNNING
    assert sent == []
    assert (WorkflowOperation.STOP_WORKLOADS, NODES) in workload.calls

    prepared = execute_workflow(
        executor,
        workflow_id,
        expected_fencing_token=1,
        confirmed_adapter_operation_ids=[workload.stop_operation_id],
    )
    assert prepared.status is WorkflowStatus.RUNNING
    assert not any(
        command.operation is WorkflowOperation.RESET_GPU for _, command in sent
    )
    assert [
        (command.node_id, command.gpu_uuids)
        for _, command in sent
        if command.operation is WorkflowOperation.VERIFY_NO_GPU_CLIENTS
    ] == [
        ("worker-0", ["GPU-0"]),
        ("worker-1", ["GPU-1"]),
        ("worker-0", ["GPU-0"]),
        ("worker-1", ["GPU-1"]),
    ]
    assert next(iter(store.list_barriers())).state is (BarrierState.PREPARED)

    completed = execute_workflow(executor, workflow_id, expected_fencing_token=1)
    assert completed.status is WorkflowStatus.SUCCEEDED
    reset_commands = [
        command
        for _, command in sent
        if command.operation is WorkflowOperation.RESET_GPU
    ]
    assert [(item.node_id, item.gpu_uuids) for item in reset_commands] == [
        ("worker-0", ["GPU-0"]),
        ("worker-1", ["GPU-1"]),
    ]
    assert not any(command.node_id == "worker-2" for _, command in sent)
    assert (WorkflowOperation.RESTART_WORKLOAD, NODES) in workload.calls
    barrier = next(iter(store.list_barriers()))
    assert barrier.state is BarrierState.COMMITTED
