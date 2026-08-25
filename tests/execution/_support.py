# ruff: noqa: F401
from __future__ import annotations

import gzip
import io
import json
import threading
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from urllib.error import HTTPError

import httpx
import pytest

import gpu_fault.app.context as context_module
from gpu_fault.app import ApplicationContext, create_app
from gpu_fault.execution import (
    ProductionExecutorConfig,
    ProductionWorkflowExecutor,
    WorkflowDispatcher,
    WorkflowDispatcherConfig,
    WorkflowExecutionError,
    WorkflowStepContext,
    WorkflowStepOutcome,
)
from gpu_fault.fleet import (
    NODE_ACTION_KEY_VERSION_DERIVED,
    AgentHeartbeat,
    AgentLifecycleState,
    FleetNodeReadiness,
    FleetReadinessReport,
    FleetRegistry,
    SignedAgentHeartbeat,
    derive_node_action_secret,
    sign_agent_heartbeat,
)
from gpu_fault.hyperpod import HyperPodAction, HyperPodNode, HyperPodSubmissionResult
from gpu_fault.hyperpod_spares import SpareAllocation
from gpu_fault.models import (
    Environment,
    FaultIncident,
    IncidentState,
    NotificationResult,
    NotificationStatus,
    WorkflowExecutionRequest,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepExecution,
    WorkflowStepSpec,
    WorkflowStepStatus,
)
from gpu_fault.node_agent import NodeActionResult, NodeActionStatus
from gpu_fault.notification_service import AdvisoryNotificationService
from gpu_fault.processor import ProcessorRequest, ProcessorRequestStatus
from gpu_fault.regional import (
    RemoteActionCommand,
    RemoteCommandResult,
    RemoteCommandStatus,
)
from gpu_fault.runtime_adapters import (
    ANNOTATION_MECHANICAL_INSPECTION_COMPLETE,
    GpuValidationAdapter,
    HyperPodLifecycleStepAdapter,
    KubernetesWorkflowAdapter,
    ManagedRecoveryObserverAdapter,
    NodeActionPending,
    NodeActionWorkflowAdapter,
    SupportEscalationAdapter,
    quarantine_taint_value,
)
from gpu_fault.store import (
    InMemoryStore,
    NotFoundError,
    SqliteStore,
    WorkflowLeaseError,
)
from gpu_fault.telemetry import CollectorKind, CollectorStatus, EvidenceKind
from gpu_fault.watcher import AttemptObservation, ContainerObservation, WorkloadPhase
from tests._builders import (
    active_workflow_executor,
    build_store,
    copy_model,
    execute_workflow,
    fault_incident,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)

NOW = datetime(2026, 7, 20, 10, 0, tzinfo=timezone.utc)


class FakeAdapter:
    def __init__(self, outcomes: dict[WorkflowOperation, WorkflowStepOutcome]) -> None:
        self.outcomes = outcomes
        self.calls: list[str] = []

    def supports(self, step: WorkflowStepSpec) -> bool:
        return step.execution_owner == "owner-a" and step.operation in self.outcomes

    def execute(self, context: WorkflowStepContext) -> WorkflowStepOutcome:
        self.calls.append(context.idempotency_key)
        outcome = self.outcomes[context.step.operation]
        if (
            outcome.status is WorkflowStepStatus.WAITING
            and outcome.adapter_operation_id
            in context.request.confirmed_adapter_operation_ids
        ):
            return WorkflowStepOutcome.succeeded(
                operation_id=outcome.adapter_operation_id
            )
        return outcome


def workflow_state(
    store: InMemoryStore, operations: list[WorkflowOperation]
) -> tuple[FaultIncident, WorkflowRequest]:
    incident = fault_incident(
        "incident-active",
        "event-active",
        official_action="RESTART_BM",
        state=IncidentState.ACTION_PENDING,
        fencing_token=3,
        created_at=NOW,
        updated_at=NOW,
    )
    workflow = workflow_request(
        "workflow-active",
        incident.incident_id,
        runtime_profile_version="active-v1",
        official_action="RESTART_BM",
        official_steps=[
            workflow_step(
                operation,
                parameters=dict(RESTART_PARAMETERS)
                if operation is WorkflowOperation.RESTART_WORKLOAD
                else {},
            )
            for operation in operations
        ],
        created_at=NOW,
        updated_at=NOW,
    )
    incident = copy_model(incident, workflow_request_id=workflow.request_id)
    store.save_incident(incident)
    store.save_workflow(workflow)
    return incident, workflow


def executor(
    store: InMemoryStore, adapter: FakeAdapter, operations: list[WorkflowOperation]
) -> ProductionWorkflowExecutor:
    return active_workflow_executor(store, [adapter], operations)


def _preempting_successor(
    store: InMemoryStore,
    incident: FaultIncident,
    predecessor: WorkflowRequest,
    operation: WorkflowOperation = WorkflowOperation.RESTART_NODE,
) -> WorkflowRequest:
    successor = workflow_request(
        "workflow-successor",
        incident.incident_id,
        fencing_token=predecessor.fencing_token,
        predecessor_workflow_id=predecessor.request_id,
        preempt_predecessor=True,
        preemption_reason="strictly stronger recovery action",
        runtime_profile_version="active-v1",
        official_action=operation.value,
        official_steps=[workflow_step(operation)],
    )
    store.save_workflow(successor)
    store.save_incident(copy_model(incident, workflow_request_id=successor.request_id))
    return successor


class FakeHyperPodLifecycle:
    def __init__(self) -> None:
        self.calls = 0
        self.preflight_kwargs = {}

    def preflight(self, *_args, **_kwargs):
        from types import SimpleNamespace

        self.preflight_kwargs = _kwargs
        return SimpleNamespace(safe_to_submit=True, gate_failures=[])

    def execute_step(self, step, **kwargs):
        self.calls += 1
        assert kwargs["confirm_cluster_name"] == "hp-cluster"
        assert kwargs["isolation_verified_nodes"] == ["node-a"]
        return HyperPodSubmissionResult(
            operation_id="hyperpod-operation-1",
            idempotency_key=kwargs["idempotency_key"],
            action=HyperPodAction.REBOOT,
            cluster_name="hp-cluster",
            requested_node_logical_ids=step.node_ids,
            successful_node_logical_ids=step.node_ids,
        )


class FakeSpareCoordinator:
    def __init__(self, allocation):
        self.allocation = allocation
        self.calls = []
        self.releases = []

    def allocate(self, **kwargs):
        self.calls.append(kwargs)
        return self.allocation

    def release(self, node_ids, incident_id):
        self.releases.append((node_ids, incident_id))


class RecordingNodeActionAdapter:
    owner = "gpu-fault-node-agent"

    def __init__(self, endpoints=None):
        self.contexts = []
        self.endpoints = endpoints or {}

    def knows_node(self, cluster_id, node_id):
        return node_id in self.endpoints

    def execute(self, context):
        self.contexts.append(context)
        return WorkflowStepOutcome.succeeded(
            operation_id=context.idempotency_key,
            details={
                "snapshot_triggered": (
                    context.step.operation is WorkflowOperation.TRIGGER_HEALTH_SNAPSHOT
                )
            },
        )


class StubFleetRegistry:
    """Minimal endpoint/maintenance_endpoint provider.

    Models the contract NodeActionWorkflowAdapter needs so addressing
    can be driven by fleet heartbeats instead of a static endpoint map.
    """

    def __init__(
        self,
        endpoints,
        *,
        generation=7,
        ready=True,
        node_action_key_version=1,
        required_node_action_key_version=None,
    ):
        self._endpoints = endpoints
        self._generation = generation
        self._ready = ready
        self._node_action_key_version = node_action_key_version
        self.policy = SimpleNamespace(
            required_node_action_key_version=(required_node_action_key_version)
        )
        self.store = self
        self.endpoint_calls = []
        self.maintenance_calls = []

    def endpoint(self, cluster_id, node_id):
        self.endpoint_calls.append((cluster_id, node_id))
        if not self._ready or node_id not in self._endpoints:
            raise ValueError(f"node {node_id} is not fleet-ready: heartbeat is stale")
        return self._endpoints[node_id], self._generation

    def maintenance_endpoint(self, cluster_id, node_id, expected_generation):
        self.maintenance_calls.append((cluster_id, node_id, expected_generation))
        if node_id not in self._endpoints:
            raise NotFoundError(f"{cluster_id}/{node_id}")
        if expected_generation != self._generation:
            raise ValueError(
                f"agent generation changed from "
                f"{expected_generation} to {self._generation}"
            )
        return self._endpoints[node_id]

    def get_agent(self, cluster_id, node_id):
        if node_id not in self._endpoints:
            raise NotFoundError(f"{cluster_id}/{node_id}")
        return SimpleNamespace(
            node_id=node_id,
            endpoint=self._endpoints[node_id],
            generation=self._generation,
            node_action_key_version=(self._node_action_key_version),
            lifecycle_state=AgentLifecycleState.ACTIVE,
        )

    def readiness(self, cluster_id, node_ids):
        nodes = [
            FleetNodeReadiness(
                node_id=node_id,
                ready=self._ready and node_id in self._endpoints,
                endpoint=self._endpoints.get(node_id),
                generation=(self._generation if node_id in self._endpoints else None),
                reasons=(
                    []
                    if self._ready and node_id in self._endpoints
                    else ["heartbeat is stale"]
                ),
            )
            for node_id in node_ids
        ]
        return FleetReadinessReport(
            cluster_id=cluster_id,
            ready=all(node.ready for node in nodes),
            evaluated_at=datetime(2026, 8, 6, tzinfo=timezone.utc),
            nodes=nodes,
        )


def _real_flight_signal(rank: int, *, seq: int, state: str) -> dict[str, object]:
    """Per-rank signal as the Agent reports a real torch 2.10 dump."""

    return {
        "rank": rank,
        "pid": 1000 + rank,
        "node_id": f"node-{rank // 8}",
        "flight_recorder": {
            "status": "dumped",
            "entry_count": 21,
            "last_entry": {
                "pg_name": "0:default_pg",
                "collective_seq_id": seq,
                "state": state,
                "profiling_name": "nccl:all_reduce",
                # PyTorch leaves this None unless its watchdog caught the
                # collective mid-flight.
                "time_discovered_started": None,
                "time_discovered_completed": None,
            },
        },
        "python_stack": {"signature": "aaaa", "collective_frames": True},
        "proc": {
            "cpu_ticks_delta": 200,
            "thread_states": {"R": 2, "S": 7},
            "voluntary_ctxt_switches_delta": 0,
            "wchan_unchanged": True,
        },
        "gpu": {"utilization_gpu_percent": 100.0},
    }


def _hung_signal_without_dump(rank: int, *, spinning: bool) -> dict[str, object]:
    """Shape of a real 24-rank hang whose flight recorder never dumped."""

    return {
        "rank": rank,
        "pid": 1000 + rank,
        "node_id": f"node-{rank // 8}",
        "gpu_uuid": f"GPU-{rank}",
        "flight_recorder": {"status": "dump_missing"},
        "python_stack": {"signature": "aaaa", "collective_frames": True},
        "proc": {
            "cpu_ticks_delta": 40 if spinning else 0,
            "thread_states": {"S": 4},
            "voluntary_ctxt_switches_delta": 0,
            "wchan_unchanged": True,
        },
        "gpu": {"utilization_gpu_percent": 100.0 if spinning else 0.0},
    }


class FakeCoreApi:
    def __init__(self) -> None:
        self.node = {
            "metadata": {"resourceVersion": "1", "annotations": {}},
            "spec": {
                "unschedulable": False,
                "taints": [
                    {
                        "key": "sagemaker.amazonaws.com/node-health-status",
                        "value": "Unschedulable",
                        "effect": "NoSchedule",
                    }
                ],
            },
        }

    def read_node(self, _):
        return self.node

    def patch_node(self, _, body):
        self.node["metadata"]["annotations"].update(
            {
                key: value
                for key, value in body["metadata"]["annotations"].items()
                if value is not None
            }
        )
        for key, value in body["metadata"]["annotations"].items():
            if value is None:
                self.node["metadata"]["annotations"].pop(key, None)
        self.node["spec"].update(body["spec"])


class UnusedApi:
    pass


class FakeBatchApi:
    def __init__(self) -> None:
        self.active = 1
        self.terminating = 0
        self.failed = 0
        self.succeeded = 0
        self.suspend_patches: list[bool] = []
        self.annotations = {}
        self.created = {}
        self.job_spec_extra = {}

    def patch_namespaced_job(self, _, __, body):
        self.suspend_patches.append(body["spec"]["suspend"])
        self.annotations.update(body["metadata"].get("annotations", {}))

    def read_namespaced_job(self, name, _namespace):
        if name in self.created:
            return self.created[name]
        if name != "training-job":
            error = KeyError(name)
            error.status = 404
            raise error
        return {
            "metadata": {
                "name": name,
                "resourceVersion": "1",
                "annotations": self.annotations,
            },
            "spec": {
                **self.job_spec_extra,
                "template": {
                    "metadata": {
                        "labels": {
                            "gpu-fault.io/managed": "true",
                            "gpu-fault.io/attempt-id": "attempt-a",
                        },
                        "annotations": {},
                    },
                    "spec": {
                        "restartPolicy": "Never",
                        "containers": [
                            {
                                "name": "trainer",
                                "resources": {"limits": {"nvidia.com/gpu": "1"}},
                            }
                        ],
                    },
                },
            },
            "status": {
                "active": self.active,
                "terminating": self.terminating,
                "failed": self.failed,
                "succeeded": self.succeeded,
            },
        }

    def create_namespaced_job(self, _namespace, body):
        self.created[body["metadata"]["name"]] = body


class RecordingOwnershipProvider:
    """Stand-in for RegionalIncidentOwnershipProvider."""

    def __init__(self, terminal: dict[str, bool], *, error=None) -> None:
        self.terminal = terminal
        self.error = error
        self.queried: list[str] = []

    def incident_workflow_is_terminal(self, incident_id: str) -> bool:
        self.queried.append(incident_id)
        if self.error is not None:
            raise self.error
        return self.terminal.get(incident_id, False)


def _storeless_isolation_context(core, provider):
    store = build_store()
    incident, workflow = workflow_state(store, [WorkflowOperation.MARK_UNSCHEDULABLE])
    core.node["metadata"]["annotations"] = {
        "gpu-fault.io/incident-id": "incident-dead",
        "gpu-fault.io/fencing-token": "1",
        "gpu-fault.io/previous-unschedulable": "false",
    }
    core.node["spec"]["taints"].append(
        {
            "key": "gpu-fault.io/quarantined",
            "value": quarantine_taint_value("incident-dead"),
            "effect": "NoSchedule",
        }
    )
    adapter = KubernetesWorkflowAdapter(
        core_api=core,
        batch_api=UnusedApi(),
        custom_api=UnusedApi(),
        store=None,
        ownership_provider=provider,
    )
    step = copy_model(workflow.official_steps[0], execution_owner=adapter.owner)
    context = WorkflowStepContext(
        workflow=workflow,
        incident=incident,
        step=step,
        step_index=0,
        request=WorkflowExecutionRequest(expected_fencing_token=workflow.fencing_token),
        idempotency_key="isolate/incident-active",
    )
    return adapter, context


def _managed_job_recovery_workflow(
    store: InMemoryStore,
    operation: WorkflowOperation,
    adapter: KubernetesWorkflowAdapter,
    *,
    workload_ids: list[str],
    parameters: dict[str, object] | None = None,
) -> WorkflowRequest:
    _, workflow = workflow_state(store, [operation])
    step = copy_model(
        workflow.official_steps[0],
        execution_owner=adapter.owner,
        workload_ids=workload_ids,
        parameters=parameters or {},
    )
    workflow = copy_model(workflow, official_steps=[step])
    store.save_workflow(workflow)
    return workflow


def _run_managed_job_recovery_step(
    store: InMemoryStore,
    workflow: WorkflowRequest,
    adapter: KubernetesWorkflowAdapter,
    operation: WorkflowOperation,
):
    return execute_workflow(
        active_workflow_executor(store, [adapter], {operation}), workflow.request_id
    )


RESTART_PARAMETERS = {
    "cluster_id": "cluster-a",
    "job_id": "training-job",
    "source_attempt_id": "attempt-a",
    "source_gpu_count": 1,
    "restart_budget": 1,
}

__all__ = [name for name in globals() if not name.startswith("__")]


class _FleetPreflightRegistry:
    def __init__(self, *, ready: bool) -> None:
        self.ready = ready
        self.calls = []

    def readiness(self, cluster_id, node_ids):
        self.calls.append((cluster_id, node_ids))
        return FleetReadinessReport(
            cluster_id=cluster_id,
            ready=self.ready,
            evaluated_at=NOW,
            nodes=[
                FleetNodeReadiness(
                    node_id=node_id,
                    ready=self.ready,
                    generation=7,
                    endpoint=f"https://{node_id}:9099",
                    reasons=(
                        []
                        if self.ready
                        else ["agent protocol version mismatch: expected 3, got 2"]
                    ),
                )
                for node_id in node_ids
            ],
        )


def _run_chained_preemptions(chain_length: int, *, incremental_arrival: bool):
    operations = [
        WorkflowOperation.RESET_GPU,
        WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES,
        WorkflowOperation.RESTART_NODE,
        WorkflowOperation.QUARANTINE,
        WorkflowOperation.ESCALATE_SUPPORT,
    ][:chain_length]
    store = build_store()
    incident, first = workflow_state(store, [operations[0]])
    first = copy_model(first, status=WorkflowStatus.RUNNING)
    store.save_workflow(first)
    workflows = [first]
    adapter = FakeAdapter(
        {operation: WorkflowStepOutcome.succeeded() for operation in operations}
    )
    dispatcher = WorkflowDispatcher(
        store,
        active_workflow_executor(store, [adapter], operations),
        WorkflowDispatcherConfig(enabled=True, batch_size=100),
    )

    predecessor = first
    for index, operation in enumerate(operations[1:], start=1):
        successor = workflow_request(
            f"workflow-chain-{index}",
            incident.incident_id,
            fencing_token=first.fencing_token,
            predecessor_workflow_id=predecessor.request_id,
            preempt_predecessor=True,
            preemption_reason=f"chain escalation {index}",
            runtime_profile_version="active-v1",
            official_action=operation.value,
            official_steps=[workflow_step(operation)],
        )
        store.save_workflow(successor)
        store.save_incident(
            copy_model(incident, workflow_request_id=successor.request_id)
        )
        workflows.append(successor)
        if incremental_arrival:
            dispatcher.run_once()
            if index < chain_length - 1:
                store.save_workflow(
                    copy_model(successor, status=WorkflowStatus.RUNNING)
                )
        predecessor = successor

    passes = 1 if incremental_arrival else chain_length + 1
    for _ in range(passes):
        dispatcher.run_once()

    return store, incident, workflows, operations, adapter


def _assert_chained_preemption_result(result, chain_length: int) -> None:
    store, incident, workflows, operations, adapter = result
    assert [store.get_workflow(item.request_id).status for item in workflows[:-1]] == [
        WorkflowStatus.SUPERSEDED
    ] * (chain_length - 1)
    assert (
        store.get_workflow(workflows[-1].request_id).status is WorkflowStatus.SUCCEEDED
    )
    assert adapter.calls == [f"{workflows[-1].request_id}/0/{operations[-1].value}"]
    assert [
        store.get_workflow(item.request_id).step_executions for item in workflows[:-1]
    ] == [[] for _ in workflows[:-1]]
    assert [item.predecessor_workflow_id for item in workflows[1:]] == [
        item.request_id for item in workflows[:-1]
    ]
    assert len({item.request_id for item in workflows}) == chain_length
    current_incident = store.get_incident(incident.incident_id)
    assert current_incident.workflow_request_id == workflows[-1].request_id
    assert current_incident.fencing_token == workflows[-1].fencing_token


def _remote_waiting_state(
    store: InMemoryStore, operation: WorkflowOperation
) -> tuple[FaultIncident, WorkflowRequest, str]:
    incident, workflow = workflow_state(store, [operation])
    command_id = f"remote-waiting-{operation.value.lower()}"
    command = RemoteActionCommand(
        command_id=command_id,
        cluster_id=incident.cluster_id,
        workflow_request_id=workflow.request_id,
        incident_id=incident.incident_id,
        step_index=0,
        fencing_token=workflow.fencing_token,
        idempotency_key=(f"{workflow.request_id}/0/{operation.value}"),
        step=workflow.official_steps[0],
        workflow=workflow,
        incident=incident,
    )
    store.ensure_remote_command(command)
    claimed = store.claim_remote_commands(
        incident.cluster_id, "cluster-executor-a", limit=1, lease_seconds=60
    )[0]
    store.complete_remote_command(
        incident.cluster_id,
        command_id,
        RemoteCommandResult(
            lease_token=claimed.lease_token, status=RemoteCommandStatus.WAITING
        ),
    )
    workflow = copy_model(
        workflow,
        status=WorkflowStatus.RUNNING,
        step_executions=[
            workflow_step_execution(
                0,
                operation,
                WorkflowStepStatus.WAITING,
                adapter_operation_id=f"remote/{command_id}",
                details={"remote_status": "WAITING", "remote_command_id": command_id},
            )
        ],
    )
    store.save_workflow(workflow)
    return incident, workflow, command_id
