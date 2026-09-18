from __future__ import annotations

from typing import Any

import pytest

from gpu_fault.adapters import NodeActionWorkflowAdapter
from gpu_fault.adapters.common import NodeActionPending
from gpu_fault.fleet import BarrierState
from gpu_fault.models import WorkflowOperation, WorkflowStepStatus
from gpu_fault.node_agent import NodeActionResult, NodeActionStatus, SignedNodeAction
from tests.execution.test_node_action_transport_retry import ENDPOINT, SECRET
from tests.regional._cov95_runtime_adapter import context_for, fleet_adapter
from tests.regional._cov95_runtime_support import offline_runtime as offline_runtime


def success(signed: SignedNodeAction) -> NodeActionResult:
    return NodeActionResult(
        command_id=signed.command.command_id,
        operation=signed.command.operation,
        status=NodeActionStatus.SUCCEEDED,
        details={"owned_verification": signed.command.node_id},
    )


def test_multi_node_reset_without_a_coordinator_fails_before_dispatch() -> None:
    calls = []
    adapter = NodeActionWorkflowAdapter(
        {"node-a": ENDPOINT, "node-b": "http://node-b:9099"},
        SECRET,
        sender=lambda *args: calls.append(args),
    )
    outcome = adapter.execute(context_for(adapter))
    assert outcome.status is WorkflowStepStatus.FAILED
    assert "barrier coordinator" in outcome.error
    assert calls == []


@pytest.mark.parametrize("mapping", [[], {}, {"node-a": ["GPU-a"], "node-b": []}])
def test_multi_node_gpu_scope_is_validated_before_creating_a_barrier(
    mapping: Any,
) -> None:
    calls = []
    adapter = fleet_adapter(sender=lambda *args: calls.append(args))
    context = context_for(adapter, parameters={"gpu_uuids_by_node": mapping})
    outcome = adapter.execute(context)
    assert outcome.status is WorkflowStepStatus.FAILED
    assert "GPU UUIDs" in outcome.error or "gpu_uuids_by_node" in outcome.error
    assert calls == []
    with pytest.raises(KeyError):
        adapter.barriers.store.get_barrier(context.idempotency_key)


@pytest.mark.parametrize("failure", ["pending", "failed"])
def test_unproved_preparation_aborts_and_cannot_dispatch_a_reset_on_replay(
    failure: str,
) -> None:
    calls = []

    def send(endpoint: str, signed: SignedNodeAction) -> NodeActionResult:
        item = signed.command
        calls.append((item.operation, item.node_id))
        if failure == "pending":
            raise NodeActionPending(item.command_id)
        return NodeActionResult(
            command_id=item.command_id,
            operation=item.operation,
            status=NodeActionStatus.FAILED,
        )

    adapter = fleet_adapter(sender=send)
    context = context_for(adapter)
    first = adapter.execute(context)
    assert first.status is WorkflowStepStatus.FAILED
    reason = "prepare request failed" if failure == "pending" else "prepare failed"
    assert reason in first.error
    barrier = adapter.barriers.store.get_barrier(context.idempotency_key)
    assert barrier.state is BarrierState.ABORTED
    again = adapter.execute(context)
    assert again.status is WorkflowStepStatus.FAILED
    assert calls == [(WorkflowOperation.VERIFY_NO_GPU_CLIENTS, "node-a")]


@pytest.mark.parametrize("failure", ["request", "busy-exhausted"])
def test_failed_commit_preserves_the_other_nodes_completed_evidence(
    failure: str,
) -> None:
    calls = []

    def send(endpoint: str, signed: SignedNodeAction) -> NodeActionResult:
        item = signed.command
        calls.append((item.operation, item.node_id))
        if item.operation is WorkflowOperation.RESET_GPU and item.node_id == "node-b":
            if failure == "request":
                raise RuntimeError("synthetic reset request refusal")
            return NodeActionResult(
                command_id=item.command_id,
                operation=item.operation,
                status=NodeActionStatus.FAILED,
                error="GPU compute clients are still active",
            )
        return success(signed)

    adapter = fleet_adapter(sender=send, verify_max_attempts=1)
    context = context_for(adapter)
    assert adapter.execute(context).status is WorkflowStepStatus.WAITING
    result = adapter.execute(context)
    assert result.status is WorkflowStepStatus.FAILED
    barrier = adapter.barriers.store.get_barrier(context.idempotency_key)
    assert barrier.state is BarrierState.FAILED
    a, b = barrier.participants
    assert a.commit_details == {"owned_verification": "node-a"}
    assert b.error is not None
    assert calls == [
        (WorkflowOperation.VERIFY_NO_GPU_CLIENTS, "node-a"),
        (WorkflowOperation.VERIFY_NO_GPU_CLIENTS, "node-b"),
        (WorkflowOperation.RESET_GPU, "node-a"),
        (WorkflowOperation.RESET_GPU, "node-b"),
    ]


def test_partial_prepare_and_completed_replay_never_repeat_a_proved_action() -> None:
    calls = []

    def send(endpoint: str, signed: SignedNodeAction) -> NodeActionResult:
        calls.append((signed.command.operation, signed.command.node_id))
        return success(signed)

    adapter = fleet_adapter(sender=send)
    context = context_for(adapter)
    barrier = adapter.barriers.create(
        barrier_id=context.idempotency_key,
        cluster_id=context.incident.cluster_id,
        workflow_request_id=context.workflow.request_id,
        incident_id=context.incident.incident_id,
        fencing_token=context.workflow.fencing_token,
        operation=context.step.operation,
        generations={"node-a": 1, "node-b": 1},
    )
    adapter.barriers.record_prepare(
        barrier.barrier_id, "node-a", details={"owned_verification": "node-a"}
    )
    prepared = adapter.execute(context)
    assert prepared.status is WorkflowStepStatus.WAITING
    assert prepared.details["prepared_nodes"] == ["node-a", "node-b"]
    committed = adapter.execute(context)
    assert committed.status is WorkflowStepStatus.SUCCEEDED
    before = list(calls)
    replay = adapter.execute(context)
    assert replay.status is WorkflowStepStatus.SUCCEEDED
    assert replay.details["barrier_state"] == "COMMITTED"
    assert (
        calls
        == before
        == [
            (WorkflowOperation.VERIFY_NO_GPU_CLIENTS, "node-b"),
            (WorkflowOperation.RESET_GPU, "node-a"),
            (WorkflowOperation.RESET_GPU, "node-b"),
        ]
    )
