"""A second GPU merged once the compound reset command is minted (batch 136).

With remote step batching the control plane mints the window's QUIESCE as the
head of one compound command whose VERIFY, RESET and RESTORE passengers carry
the scope they had at mint time, and the data plane runs the whole window under
one lease. A merge that then widened the plan's pending passengers changed only
the plan: their per-step results still answer the plan's steps by index, so the
record said RESET_GPU SUCCEEDED for a GPU the node never reset.

End to end, in process (the harness of ``test_remote_step_batching_e2e``): the
real control-plane executor and ``RegionalRemoteWorkflowAdapter``, the real
``ClusterActionExecutor``, the real merge service and disposition applier. The
node-side adapter answers each step with the GPUs its own command carried.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from gpu_fault.cluster_executor import ClusterActionExecutor
from gpu_fault.execution.models import WorkflowStepOutcome
from gpu_fault.execution.step_bounds import previous_execution
from gpu_fault.models import (
    FaultIncident,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepSpec,
    WorkflowStepStatus,
)
from gpu_fault.operation_registry import (
    NODE_EXCLUSIVE_OPERATIONS,
    WORKLOAD_SCOPED_OPERATIONS,
)
from gpu_fault.orchestration.arbitration import RecoveryArbiter
from gpu_fault.orchestration.dag_branching import DagBrancher
from gpu_fault.orchestration.disposition import Disposition, DispositionApplier
from gpu_fault.orchestration.families.conflicts import NodeConflictService
from gpu_fault.orchestration.workflow_merge import WorkflowMergeService
from gpu_fault.regional import RegionalRemoteWorkflowAdapter
from tests._builders import (
    active_workflow_executor,
    build_store,
    copy_model,
    execute_workflow,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)
from tests.regional._batching_support import (
    ACTIVE_POLICY,
    ALL_OWNERS,
    KUBERNETES_OWNER,
    NODE_OWNER,
    VALIDATION_OWNER,
    chain_state,
    quiesce_details,
    reset_chain,
    step_context,
)
from tests.regional._regional_support import TOKEN_A, registration
from tests.regional.test_remote_step_batching_e2e import LocalAdapter, StoreClient

CLUSTER = "cluster-a"
NOW = datetime(2026, 7, 28, 0, 0, tzinfo=timezone.utc)
RESET = WorkflowOperation.RESET_GPU


@pytest.fixture(autouse=True)
def claim_state_outside_shared_tmp(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(
        "GPU_FAULT_CLUSTER_EXECUTOR_CLAIM_STATE_PATH", str(tmp_path / "claim.json")
    )


class NodeSide:
    """The node-action adapter: every result names the GPUs its command carried."""

    owner = NODE_OWNER

    def __init__(self) -> None:
        self.resets: list[list[str]] = []

    def supports(self, step) -> bool:
        return step.execution_owner == self.owner and step.operation in {
            WorkflowOperation.QUIESCE_GPU_SERVICES,
            WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
            RESET,
            WorkflowOperation.RESTORE_GPU_SERVICES,
        }

    def execute(self, context) -> WorkflowStepOutcome:
        step = context.step
        mapping = step.parameters.get("gpu_uuids_by_node")
        gpus = list(mapping["node-a"]) if mapping else list(step.gpu_uuids)
        if step.operation is WorkflowOperation.QUIESCE_GPU_SERVICES:
            return WorkflowStepOutcome.succeeded(details=quiesce_details())
        if step.operation is RESET:
            self.resets.append(gpus)
            result: dict[str, Any] = {"reset_gpu_uuids": gpus}
        else:
            result = {"gpu_uuids": gpus}
        return WorkflowStepOutcome.succeeded(
            details={"node_results": {"node-a": result}}
        )


def data_plane(store) -> tuple[ClusterActionExecutor, NodeSide]:
    node = NodeSide()
    adapters = [
        LocalAdapter(
            KUBERNETES_OWNER,
            {
                WorkflowOperation.MARK_UNSCHEDULABLE: WorkflowStepOutcome.succeeded(
                    details={"cordoned": ["node-a"]}
                ),
                WorkflowOperation.RESTORE_SCHEDULING: WorkflowStepOutcome.succeeded(
                    details={"uncordoned": ["node-a"]}
                ),
            },
        ),
        node,
        LocalAdapter(
            VALIDATION_OWNER,
            {WorkflowOperation.VALIDATE_GPU: WorkflowStepOutcome.succeeded(details={})},
        ),
    ]
    executor = ClusterActionExecutor(
        StoreClient(store),
        adapters,
        executor_id="executor-a",
        allowed_namespaces={"training"},
    )
    return executor, node


def merge_service() -> tuple[WorkflowMergeService, DispositionApplier]:
    arbiter = RecoveryArbiter()
    brancher = DagBrancher(arbiter)
    merger = WorkflowMergeService(
        arbiter,
        brancher,
        preemption_enabled=True,
        workload_scoped_operations=set(WORKLOAD_SCOPED_OPERATIONS),
        node_exclusive_operations=set(NODE_EXCLUSIVE_OPERATIONS),
        workflow_resource_claims_by_node=(
            NodeConflictService.workflow_resource_claims_by_node
        ),
    )
    applier = DispositionApplier(
        arbiter=arbiter,
        brancher=brancher,
        aggregation_deadlines=lambda now, _workflow: (now, now),
        prepare_preempting_successor=merger.prepare_preempting_successor,
        preempt_parallel_job_branch=merger.preempt_parallel_branch,
        workflow_preemption_enabled=True,
    )
    return merger, applier


def second_gpu_candidate(incident: FaultIncident) -> WorkflowRequest:
    """What the planner compiles for an XID on GPU-b of the same node."""

    steps = [
        step.model_copy(
            update={
                "gpu_uuids": ["GPU-b"],
                "parameters": {"gpu_uuids_by_node": {"node-a": ["GPU-b"]}},
            }
        )
        if step.execution_owner == NODE_OWNER
        else step
        for step in reset_chain()
    ]
    return workflow_request(
        "workflow-gpu-b",
        incident.incident_id,
        runtime_profile_version="active-v1",
        official_action="RESTART_BM",
        official_steps=steps,
        created_at=NOW,
        updated_at=NOW,
    )


def merge_second_gpu(
    store, incident: FaultIncident, request_id: str
) -> WorkflowRequest:
    """The second fault's merge, as a family applies it, saved like the store's
    merge transaction saves it."""

    merger, applier = merge_service()
    existing = store.get_workflow(request_id)
    candidate = second_gpu_candidate(incident)
    disposition = merger.disposition(existing, candidate, "node-a", {"GPU-b"})
    merged, _ = applier.apply(
        disposition,
        node_id="node-a",
        candidate=copy_model(incident, gpu_uuids=["GPU-b"]),
        candidate_workflow=candidate,
        existing_incident=incident,
        existing_workflow=existing,
        gpu_uuids={"GPU-b"},
        mutable=False,
        # The merge's event follows the ones the executor already wrote.
        now=datetime.now(timezone.utc),
    )
    store.save_workflow(merged)
    return merged


def drive(store, control_plane, executor, request_id: str, *, rounds: int = 40):
    for _ in range(rounds):
        execute_workflow(control_plane, request_id)
        if store.get_workflow(request_id).status is WorkflowStatus.SUCCEEDED:
            return store.get_workflow(request_id)
        executor.run_once()
    final = store.get_workflow(request_id)
    raise AssertionError((final.status, final.blocked_reasons))


def reset_record(workflow: WorkflowRequest) -> tuple[list[str], list[str]]:
    """The plan's RESET_GPU targets, and the GPUs its recorded result reset."""

    index, step = next(
        (index, step)
        for index, step in enumerate(workflow.official_steps)
        if step.operation is RESET and not step.branch_id
    )
    execution = previous_execution(workflow, step, index)
    assert execution is not None, "the reset has a record"
    return (
        list(step.gpu_uuids),
        list(execution.details["node_results"]["node-a"]["reset_gpu_uuids"]),
    )


def test_a_gpu_merged_after_the_compound_is_minted_is_reset_in_its_own_window() -> None:
    store = build_store()
    store.save_regional_cluster(registration(CLUSTER, TOKEN_A))
    steps = reset_chain()
    incident, workflow = chain_state(store, steps)
    control_plane = active_workflow_executor(
        store,
        [
            RegionalRemoteWorkflowAdapter(
                store, owners=ALL_OWNERS, step_batching=ACTIVE_POLICY
            )
        ],
        [step.operation for step in steps],
    )
    executor, node = data_plane(store)
    for _ in range(10):
        execute_workflow(control_plane, workflow.request_id)
        if any(item.batched_steps for item in store.list_remote_commands()):
            break
        executor.run_once()
    (compound,) = [item for item in store.list_remote_commands() if item.batched_steps]
    assert [item.step.gpu_uuids for item in compound.batched_steps] == [["GPU-a"]] * 3

    merged = merge_second_gpu(store, incident, workflow.request_id)

    first = drive(store, control_plane, executor, workflow.request_id)
    planned, performed = reset_record(first)
    assert planned == performed == ["GPU-a"], (
        f"the record claims a reset of {planned}, the node reset {performed}"
    )
    assert merged.request_id != workflow.request_id, "no join: a successor workflow"
    assert merged.predecessor_workflow_id == workflow.request_id
    second = drive(store, control_plane, executor, merged.request_id)
    assert reset_record(second) == (["GPU-b"], ["GPU-b"])
    assert node.resets == [["GPU-a"], ["GPU-b"]], "each GPU reset once, by its window"
    second_compound = [
        item
        for item in store.list_remote_commands()
        if item.batched_steps and item.workflow_request_id == merged.request_id
    ]
    assert len(second_compound) == 1, "the second window is one compound of its own"


def test_the_merge_reads_the_minted_compound_as_a_dispatched_window() -> None:
    store = build_store()
    store.save_regional_cluster(registration(CLUSTER, TOKEN_A))
    steps = reset_chain()
    incident, workflow = chain_state(store, steps)
    control_plane = active_workflow_executor(
        store,
        [
            RegionalRemoteWorkflowAdapter(
                store, owners=ALL_OWNERS, step_batching=ACTIVE_POLICY
            )
        ],
        [step.operation for step in steps],
    )
    executor, _ = data_plane(store)
    for _ in range(10):
        execute_workflow(control_plane, workflow.request_id)
        if any(item.batched_steps for item in store.list_remote_commands()):
            break
        executor.run_once()
    merger, _ = merge_service()

    verdict = merger.disposition(
        store.get_workflow(workflow.request_id),
        second_gpu_candidate(incident),
        "node-a",
        {"GPU-b"},
    )

    assert verdict is Disposition.QUEUE_SUCCESSOR, verdict


JOB = "training/pytorchjob/train"


def job_chain(gpu: str) -> list[WorkflowStepSpec]:
    """A job's linear reset plan on node-a: the job is stopped and restarted
    whole, the node-side window names ``gpu``."""

    chain = [
        step.model_copy(update={"gpu_uuids": [gpu]})
        if step.execution_owner == NODE_OWNER
        else step
        for step in reset_chain()
    ]
    whole_job = {"node_ids": ["node-a", "node-b"], "workload_ids": [JOB]}
    return [
        chain[0],
        workflow_step(WorkflowOperation.STOP_WORKLOADS, KUBERNETES_OWNER, **whole_job),
        *chain[1:],
        workflow_step(
            WorkflowOperation.RESTART_WORKLOAD, KUBERNETES_OWNER, **whole_job
        ),
    ]


def test_a_job_plan_grown_into_a_dag_still_reads_its_window_from_the_compound() -> None:
    """Re-send stability across the merge: the record becomes a DAG with a
    successor branch while the compound command is in flight, and every step of
    the dispatched window is still answered by that command -- no step mints a
    command of its own, and the second GPU's QUIESCE is a new command."""

    store = build_store()
    store.save_regional_cluster(registration(CLUSTER, TOKEN_A))
    incident, workflow = chain_state(
        store, job_chain("GPU-a"), completed_step_indexes=[0, 1]
    )
    adapter = RegionalRemoteWorkflowAdapter(
        store, owners=ALL_OWNERS, step_batching=ACTIVE_POLICY
    )
    head = adapter.execute(step_context(workflow, incident, 2))
    (compound,) = store.list_remote_commands()
    assert [item.step_index for item in compound.batched_steps] == [3, 4, 5]
    running = copy_model(
        workflow,
        status=WorkflowStatus.RUNNING,
        execution_owner_id="executor-a",
        step_executions=[
            workflow_step_execution(
                2,
                WorkflowOperation.QUIESCE_GPU_SERVICES,
                WorkflowStepStatus.WAITING,
                adapter_operation_id=head.adapter_operation_id,
                details=head.details,
            )
        ],
    )
    store.save_workflow(running)
    merger, applier = merge_service()
    candidate = workflow_request(
        "workflow-gpu-b",
        incident.incident_id,
        runtime_profile_version="active-v1",
        official_action="RESTART_BM",
        official_steps=job_chain("GPU-b"),
        created_at=NOW,
        updated_at=NOW,
    )

    verdict = merger.disposition(running, candidate, "node-a", {"GPU-b"})
    grown, _ = applier.apply(
        verdict,
        node_id="node-a",
        candidate=copy_model(incident, gpu_uuids=["GPU-b"]),
        candidate_workflow=candidate,
        existing_incident=incident,
        existing_workflow=running,
        gpu_uuids={"GPU-b"},
        mutable=False,
        now=datetime.now(timezone.utc),
    )
    store.save_workflow(grown)
    executor, node = data_plane(store)
    executor.run_once()
    minted = len(store.list_remote_commands())
    outcomes = [adapter.execute(step_context(grown, incident, i)) for i in (2, 3, 4, 5)]

    assert verdict is Disposition.QUEUE_BRANCH_SUCCESSOR, verdict
    assert grown.dag_enabled is True
    assert [item.status for item in outcomes] == [WorkflowStepStatus.SUCCEEDED] * 4
    assert len(store.list_remote_commands()) == minted, (
        "a step of the dispatched window minted a command of its own"
    )
    assert outcomes[2].details["node_results"]["node-a"]["reset_gpu_uuids"] == ["GPU-a"]
    assert grown.official_steps[4].gpu_uuids == ["GPU-a"], (
        "the plan names what the command reset"
    )
    assert node.resets == [["GPU-a"]]
    (second_quiesce,) = [
        index
        for index, step in enumerate(grown.official_steps)
        if step.operation is WorkflowOperation.QUIESCE_GPU_SERVICES
        and step.gpu_uuids == ["GPU-b"]
    ]
    adapter.execute(step_context(grown, incident, second_quiesce))
    (own,) = [
        item
        for item in store.list_remote_commands()
        if item.step_index == second_quiesce
    ]
    assert own.command_id != compound.command_id
    assert own.batched_steps == [], "a DAG step is dispatched on its own"
    assert own.step.gpu_uuids == ["GPU-b"]
