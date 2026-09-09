"""A reset chain end to end, in process, with and without compound commands.

The real control-plane executor loop drives ``RegionalRemoteWorkflowAdapter``
against the memory store; the real ``ClusterActionExecutor`` claims, runs and
reports against the same store through a client that stands in for the HTTP
routes. Batching must leave the workflow's record of the chain unchanged --
the same operations SUCCEEDED with the same detail keys -- and only shrink the
number of remote commands it took.
"""

from __future__ import annotations

from typing import Any

import pytest

from gpu_fault.cluster_executor import ClusterActionExecutor
from gpu_fault.execution.models import WorkflowStepOutcome
from gpu_fault.execution.step_bounds import previous_execution
from gpu_fault.models import WorkflowOperation, WorkflowStatus, WorkflowStepStatus
from gpu_fault.regional import RegionalRemoteWorkflowAdapter
from gpu_fault.remote_step_batching import RemoteStepBatchingPolicy
from tests._builders import active_workflow_executor, build_store, execute_workflow
from tests.regional._batching_support import (
    ACTIVE_POLICY,
    ALL_OWNERS,
    KUBERNETES_OWNER,
    NODE_OWNER,
    NODE_SIDE,
    VALIDATION_OWNER,
    chain_state,
    quiesce_details,
    reset_chain,
)
from tests.regional._regional_support import TOKEN_A, registration

CLUSTER = "cluster-a"


@pytest.fixture(autouse=True)
def claim_state_outside_shared_tmp(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(
        "GPU_FAULT_CLUSTER_EXECUTOR_CLAIM_STATE_PATH", str(tmp_path / "claim.json")
    )


class StoreClient:
    """The regional routes, minus HTTP: each call is the store method the
    route would run for an authenticated ``cluster-a`` executor."""

    cluster_id = CLUSTER

    def __init__(self, store) -> None:
        self.store = store
        self.progress_posts = 0

    def claim(self, executor_id, *, execution_owners, max_commands, lease_seconds, **_):
        return self.store.claim_remote_commands(
            CLUSTER,
            executor_id,
            limit=max_commands,
            lease_seconds=lease_seconds,
            execution_owners=set(execution_owners),
            accept_batched_steps=True,
        )

    def complete(self, command, result):
        return self.store.complete_remote_command(CLUSTER, command.command_id, result)

    def renew(self, command, executor_id, lease_seconds):
        return self.store.renew_remote_command_lease(
            CLUSTER,
            command.command_id,
            executor_id,
            command.lease_token,
            lease_seconds=lease_seconds,
        )

    def progress(self, command, executor_id, batched_results):
        self.progress_posts += 1
        return self.store.record_remote_command_progress(
            CLUSTER,
            command.command_id,
            executor_id,
            command.lease_token,
            batched_results=batched_results,
        )


class LocalAdapter:
    def __init__(self, owner: str, outcomes: dict[WorkflowOperation, Any]) -> None:
        self.owner = owner
        self.outcomes = outcomes
        self.calls: list[str] = []

    def supports(self, step) -> bool:
        return step.execution_owner == self.owner and step.operation in self.outcomes

    def execute(self, context) -> WorkflowStepOutcome:
        self.calls.append(context.idempotency_key)
        return self.outcomes[context.step.operation]


def data_plane(store) -> tuple[ClusterActionExecutor, StoreClient, list[LocalAdapter]]:
    client = StoreClient(store)
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
        LocalAdapter(
            NODE_OWNER,
            {
                WorkflowOperation.QUIESCE_GPU_SERVICES: WorkflowStepOutcome.succeeded(
                    details=quiesce_details()
                ),
                WorkflowOperation.VERIFY_NO_GPU_CLIENTS: WorkflowStepOutcome.succeeded(
                    details={"gpu_client_quiesce_attempt": 1, "node_results": {}}
                ),
                WorkflowOperation.RESET_GPU: WorkflowStepOutcome.succeeded(
                    details={"node_results": {"node-a": {"reset_gpu_uuids": ["GPU-a"]}}}
                ),
                WorkflowOperation.RESTORE_GPU_SERVICES: WorkflowStepOutcome.succeeded(
                    details={"node_results": {"node-a": {"status": "SUCCEEDED"}}}
                ),
            },
        ),
        LocalAdapter(
            VALIDATION_OWNER,
            {WorkflowOperation.VALIDATE_GPU: WorkflowStepOutcome.succeeded(details={})},
        ),
    ]
    return (
        ClusterActionExecutor(
            client, adapters, executor_id="executor-a", allowed_namespaces={"training"}
        ),
        client,
        adapters,
    )


def run_chain(policy: RemoteStepBatchingPolicy | None):
    store = build_store()
    store.save_regional_cluster(registration(CLUSTER, TOKEN_A))
    steps = reset_chain()
    incident, workflow = chain_state(store, steps)
    control_plane = active_workflow_executor(
        store,
        [RegionalRemoteWorkflowAdapter(store, owners=ALL_OWNERS, step_batching=policy)],
        [step.operation for step in steps],
    )
    executor, client, adapters = data_plane(store)
    for _ in range(40):
        execute_workflow(control_plane, workflow.request_id)
        if store.get_workflow(workflow.request_id).status is WorkflowStatus.SUCCEEDED:
            break
        executor.run_once()
    final = store.get_workflow(workflow.request_id)
    assert final.status is WorkflowStatus.SUCCEEDED, (
        final.status,
        final.blocked_reasons,
    )
    return store, final, client, adapters


def test_the_reset_chain_completes_the_same_way_with_fewer_remote_commands() -> None:
    store_plain, plain, client_plain, adapters_plain = run_chain(None)
    store_batched, batched, client_batched, adapters_batched = run_chain(ACTIVE_POLICY)

    assert (
        plain.completed_step_indexes == batched.completed_step_indexes == list(range(7))
    )
    assert plain.completed_operations == batched.completed_operations
    for index, step in enumerate(plain.official_steps):
        before = previous_execution(plain, step, index)
        after = previous_execution(batched, step, index)
        assert before is not None and after is not None, step.operation
        assert before.status is after.status is WorkflowStepStatus.SUCCEEDED
        assert sorted(before.details) == sorted(after.details), step.operation
        if step.operation in NODE_SIDE:
            assert before.details == after.details, step.operation
    # The node saw the same actions under the same idempotency keys.
    assert [adapter.calls for adapter in adapters_plain] == [
        adapter.calls for adapter in adapters_batched
    ]
    assert len(store_plain.list_remote_commands()) == 7
    assert len(store_batched.list_remote_commands()) == 4, (
        "cordon, one compound command for the four node-side steps, validate, release"
    )
    compound = next(
        item for item in store_batched.list_remote_commands() if item.batched_steps
    )
    assert [item.step_index for item in compound.batched_steps] == [2, 3, 4]
    assert client_plain.progress_posts == 0
    assert client_batched.progress_posts == 4
