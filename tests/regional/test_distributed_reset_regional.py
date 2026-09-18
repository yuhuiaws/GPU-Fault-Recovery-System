"""Distributed XIDs through the actual CPU and storeless regional executor loops."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from threading import RLock

import pytest

from gpu_fault.adapters import KubernetesWorkflowAdapter, NodeActionWorkflowAdapter
from gpu_fault.adapters.kubernetes.stop_ownership import (
    KubernetesStopOwnershipValidator,
)
from gpu_fault.app import default_simulated_profile
from gpu_fault.cluster_executor import ClusterActionExecutor, RegionalFleetRegistry
from gpu_fault.execution import WorkflowStepOutcome
from gpu_fault.fleet import (
    CURRENT_AGENT_PROTOCOL_VERSION,
    AgentHeartbeat,
    FleetRegistry,
    SignedAgentHeartbeat,
    sign_agent_heartbeat,
)
from gpu_fault.models import CapabilityName, WorkflowOperation, WorkflowStatus
from gpu_fault.node_agent import NodeActionResult, NodeActionStatus
from gpu_fault.operation_registry import MULTI_NODE_BARRIER_OPERATIONS
from gpu_fault.orchestration.families.reset import ResetOperationService
from gpu_fault.policy import (
    DistributedXidBatch,
    GpuFaultPolicyEngine,
    load_sxid_policy,
    load_xid_policy,
)
from gpu_fault.regional import RegionalRemoteWorkflowAdapter, RemoteCommandStatus
from tests._builders import active_workflow_executor, execute_workflow
from tests.orchestration._cov95_runtime_builder import builder, xid
from tests.regional._batching_support import ACTIVE_POLICY, NODE_OWNER, NODE_SIDE
from tests.regional._late_ownership_runtime import WORKLOAD, KubernetesState
from tests.regional._regional_support import TOKEN_A, registration
from tests.regional.test_remote_step_batching_e2e import StoreClient

CLUSTER = "cluster-a"
NODES = ["node-a", "node-b", "node-c"]
LOCAL_OWNER = "local-workloads"
KEY = "distributed-local-test-" + "x" * 32


class FleetClient(StoreClient):
    """Only the HTTP boundary is replaced; fleet verdicts use the real CPU registry."""

    def __init__(self, store, registry):
        super().__init__(store)
        self.registry = registry

    def _get(self, path):
        if path == "/v1/regional/executors/fleet-rollout-fence":
            return {"cluster_id": CLUSTER, "fencing_deployment_ids": []}
        for node in NODES:
            if path == f"/v1/fleet/agents/{CLUSTER}/{node}":
                return self.store.get_agent(CLUSTER, node).model_dump(mode="json")
        raise AssertionError(f"unexpected local fleet read: {path}")

    def _post(self, path, payload):
        assert path == "/v1/fleet/readiness"
        assert payload["cluster_id"] == CLUSTER
        return self.registry.readiness(CLUSTER, payload["node_ids"]).model_dump(
            mode="json"
        )


def distributed_plan():
    compiler = builder()
    profile = default_simulated_profile()
    compiler.store.save_profile(
        profile.model_copy(
            update={
                "capabilities": [
                    item.model_copy(
                        update={
                            "owner": NODE_OWNER
                            if item.capability is CapabilityName.GPU_RESET
                            else LOCAL_OWNER
                        }
                    )
                    for item in profile.capabilities
                ]
            }
        )
    )
    batch = DistributedXidBatch(
        batch_id="regional-distributed-reset",
        job_id="job",
        attempt_id="attempt",
        restart_budget=1,
        affected_workload_ids=[WORKLOAD],
        allocation=[
            {"node_id": node, "rank": index, "gpu_uuids": [f"GPU-{node}"]}
            for index, node in enumerate(NODES)
        ],
        events=[
            xid(
                event_id=f"regional-xid-{node}",
                node_id=node,
                gpu_uuid=f"GPU-{node}",
                xid=95,
                product="H200",
                runtime_profile_version=profile.profile_version,
                job_id="job",
                attempt_id="attempt",
                workload_state="ACTIVE",
                affected_workload_ids=[WORKLOAD],
            )
            for node in NODES[:2]
        ],
    )
    policy = GpuFaultPolicyEngine(load_xid_policy(), load_sxid_policy())
    service = ResetOperationService(compiler.store, compiler, RLock())
    incident, workflow = service.ingest_distributed_xids(
        batch, [policy.evaluate_xid(event) for event in batch.events]
    )
    return compiler.store, incident, workflow


@pytest.mark.parametrize(
    "batched", [False, True], ids=["individual-commands", "batched-commands"]
)
@pytest.mark.parametrize(
    "staggered", [False, True], ids=["ordered-quiesce", "late-quiesce-a"]
)
def test_distributed_reset_completes_in_storeless_regional_topology(
    tmp_path, batched, staggered
):
    store, _incident, workflow = distributed_plan()
    assert workflow.status is WorkflowStatus.PENDING
    assert workflow.dag_enabled, "regional multi-node recovery requires node branches"
    assert all(
        len(step.node_ids) == 1
        for step in workflow.official_steps
        if step.operation in MULTI_NODE_BARRIER_OPERATIONS
    ), "a storeless regional Executor must never receive a multi-node reset step"
    stop_steps = [
        step
        for step in workflow.official_steps
        if step.operation is WorkflowOperation.STOP_WORKLOADS
    ]
    restart_steps = [
        step
        for step in workflow.official_steps
        if step.operation is WorkflowOperation.RESTART_WORKLOAD
    ]
    assert len(stop_steps) == len(restart_steps) == 1
    assert stop_steps[0].node_ids == restart_steps[0].node_ids == NODES
    store.save_regional_cluster(registration(CLUSTER, TOKEN_A))
    cpu_fleet = FleetRegistry(store, KEY)
    for node in NODES:
        heartbeat = AgentHeartbeat(
            cluster_id=CLUSTER,
            node_id=node,
            endpoint=f"http://{node}:9099",
            agent_protocol_version=CURRENT_AGENT_PROTOCOL_VERSION,
            agent_version="0.10.0",
            artifact_sha256="a" * 64,
            policy_version="610",
            runtime_profile_version=workflow.runtime_profile_version,
            config_digest="c" * 64,
            allowed_operations=list(NODE_SIDE),
            boot_id=f"{node}-boot",
            node_instance_id=f"instance-{node}",
            agent_incarnation_id=f"incarnation-{node}",
            observed_at=datetime.now(timezone.utc),
        )
        cpu_fleet.register(
            SignedAgentHeartbeat(
                heartbeat=heartbeat, signature=sign_agent_heartbeat(heartbeat, KEY)
            )
        )
    state = KubernetesState()
    state.nodes["node-c"] = deepcopy(state.nodes["node-b"])
    state.nodes["node-c"]["metadata"].update(name="node-c", uid="node-c-uid")
    state.nodes["node-c"]["status"]["nodeInfo"]["bootID"] = "node-c-boot"
    state.pods.append(state.pod("node-c", "pod-node-c"))
    kubernetes = KubernetesWorkflowAdapter(
        owner=LOCAL_OWNER, core_api=state, batch_api=state, custom_api=state, store=None
    )
    calls = []

    class Workloads:
        owner = LOCAL_OWNER

        def supports(self, step):
            return step.execution_owner == self.owner

        def execute(self, context):
            calls.append((context.step.operation, context.step.node_ids))
            if context.step.operation is WorkflowOperation.STOP_WORKLOADS:
                return kubernetes.execute(context)
            return WorkflowStepOutcome.succeeded(operation_id=context.idempotency_key)

    allow_second = False
    allow_first_quiesce = not staggered
    sent = []

    def send(_endpoint, envelope):
        command = envelope.command
        assert not state.pods, "all three original participant Pods must be stopped"
        sent.append(command)
        if (
            command.node_id == "node-a"
            and command.operation is WorkflowOperation.QUIESCE_GPU_SERVICES
            and not allow_first_quiesce
        ):
            return NodeActionResult(
                command_id=command.command_id,
                operation=command.operation,
                status=NodeActionStatus.FAILED,
                retryable=True,
                error="quiesce preflight probe is temporarily unavailable",
            )
        if (
            command.node_id == "node-b"
            and command.operation is WorkflowOperation.VERIFY_NO_GPU_CLIENTS
            and not allow_second
        ):
            return NodeActionResult(
                command_id=command.command_id,
                operation=command.operation,
                status=NodeActionStatus.FAILED,
                error="GPU clients are still active",
            )
        return NodeActionResult(
            command_id=command.command_id,
            operation=command.operation,
            status=NodeActionStatus.SUCCEEDED,
            details={"failsafe_seconds": 420, "reset_gpu_uuids": command.gpu_uuids},
        )

    client = FleetClient(store, cpu_fleet)
    fleet = RegionalFleetRegistry(client)
    node_actions = NodeActionWorkflowAdapter({}, KEY, registry=fleet, sender=send)
    regional = ClusterActionExecutor(
        client,
        [Workloads(), node_actions],
        executor_id="local-regional",
        allowed_namespaces={"training"},
        claim_state_path=str(tmp_path / "claim.json"),
        liveness_state_path=str(tmp_path / "liveness.json"),
        sleep=lambda _: None,
    )
    regional.stop_ownership_validator = KubernetesStopOwnershipValidator.from_adapter(
        kubernetes, cluster_id=CLUSTER, allowed_namespaces=frozenset({"training"})
    )
    control = active_workflow_executor(
        store,
        [
            RegionalRemoteWorkflowAdapter(
                store,
                owners={LOCAL_OWNER, NODE_OWNER},
                step_batching=ACTIVE_POLICY if batched else None,
            )
        ],
        [step.operation for step in workflow.official_steps],
    )
    assert node_actions.barriers is None and node_actions.store is None
    assert kubernetes.store is None
    assert regional.fleet_registry is fleet and fleet.store is fleet
    if staggered:
        for _ in range(16):
            execute_workflow(control, workflow.request_id, expected_fencing_token=1)
            regional.run_once()
            if any(
                command.node_id == "node-b"
                and command.operation is WorkflowOperation.VERIFY_NO_GPU_CLIENTS
                for command in sent
            ):
                break
        assert any(
            command.node_id == "node-b"
            and command.operation is WorkflowOperation.VERIFY_NO_GPU_CLIENTS
            for command in sent
        ), "the second branch must progress while the first quiesce is waiting"
        allow_first_quiesce = True
    for _ in range(24):
        execute_workflow(control, workflow.request_id, expected_fencing_token=1)
        regional.run_once()
        if (WorkflowOperation.RESTORE_SCHEDULING, ["node-a"]) in calls:
            break
    assert (WorkflowOperation.RESTORE_SCHEDULING, ["node-a"]) in calls
    assert not any(
        operation is WorkflowOperation.RESTART_WORKLOAD for operation, _ in calls
    ), "the workload must remain stopped until every live node branch completes"
    assert [
        command.node_id
        for command in sent
        if command.operation is WorkflowOperation.RESET_GPU
    ] == ["node-a"]
    allow_second = True
    for _ in range(32):
        result = execute_workflow(
            control, workflow.request_id, expected_fencing_token=1
        )
        if result.status is WorkflowStatus.SUCCEEDED:
            break
        regional.run_once()
    final = store.get_workflow(workflow.request_id)
    assert final.status is WorkflowStatus.SUCCEEDED, final
    assert [
        command.node_id
        for command in sent
        if command.operation is WorkflowOperation.RESET_GPU
    ] == NODES[:2]
    assert all(command.gpu_uuids == [f"GPU-{command.node_id}"] for command in sent), (
        "each command must retain its node-local GPU scope"
    )
    assert not any(command.node_id == "node-c" for command in sent), (
        "an unaffected allocation participant must not receive a node reset"
    )
    assert calls.count((WorkflowOperation.STOP_WORKLOADS, NODES)) == 1
    assert calls.count((WorkflowOperation.RESTART_WORKLOAD, NODES)) == 1
    assert regional.barrier_unavailable_holds_total == 0
    assert store.list_barriers() == []
    assert all(
        command.status is RemoteCommandStatus.SUCCEEDED
        for command in store.list_remote_commands()
    ), "successful distributed recovery must leave no unfinished remote command"
