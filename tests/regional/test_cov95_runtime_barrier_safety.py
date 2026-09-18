from __future__ import annotations

import pytest

from gpu_fault.execution import WorkflowStepOutcome
from gpu_fault.fleet import BarrierParticipantState, BarrierState
from gpu_fault.models import WorkflowOperation, WorkflowStepStatus
from gpu_fault.node_agent import NodeActionResult, NodeActionStatus, SignedNodeAction
from tests.regional._cov95_runtime_adapter import context_for, fleet_adapter
from tests.regional._cov95_runtime_support import offline_runtime as offline_runtime


@pytest.mark.parametrize("phase", ["prepare", "commit"])
@pytest.mark.parametrize("target_node", ["node-a", "node-b"])
@pytest.mark.parametrize(
    "failure",
    ["interrupted", "unknown", "transport-manual", "contradictory", "no-error"],
)
def test_barrier_preserves_unknown_outcome_and_never_commits_an_unproved_node(
    phase: str, target_node: str, failure: str
) -> None:
    calls = []

    def send(
        endpoint: str, envelope: SignedNodeAction
    ) -> NodeActionResult | WorkflowStepOutcome:
        command = envelope.command
        calls.append((command.operation, command.node_id))
        target_operation = (
            WorkflowOperation.VERIFY_NO_GPU_CLIENTS
            if phase == "prepare"
            else WorkflowOperation.RESET_GPU
        )
        if command.operation is target_operation and command.node_id == target_node:
            if failure == "transport-manual":
                return WorkflowStepOutcome.failed(
                    "synthetic command identity conflict",
                    details={"manual_confirmation_required": True},
                )
            return NodeActionResult(
                command_id=command.command_id,
                operation=command.operation,
                status=(
                    NodeActionStatus.INTERRUPTED
                    if failure in {"interrupted", "no-error"}
                    else NodeActionStatus.SUCCEEDED
                    if failure == "contradictory"
                    else NodeActionStatus.FAILED
                ),
                error=None
                if failure == "no-error"
                else "synthetic action outcome unproved",
                retryable=False,
                details={"outcome_unknown": True}
                if failure in {"unknown", "contradictory"}
                else {},
            )
        return NodeActionResult(
            command_id=command.command_id,
            operation=command.operation,
            status=NodeActionStatus.SUCCEEDED,
            details={"verified": True},
        )

    adapter = fleet_adapter(sender=send)
    context = context_for(adapter)
    if phase == "commit":
        assert adapter.execute(context).status is WorkflowStepStatus.WAITING
    result = adapter.execute(context)
    assert result.status is WorkflowStepStatus.FAILED
    assert result.details["manual_confirmation_required"] is True
    assert result.details["failed_nodes"] == [target_node]
    assert target_node in result.details["node_failures"]
    barrier = adapter.barriers.store.get_barrier(context.idempotency_key)
    assert barrier.state is (
        BarrierState.ABORTED if phase == "prepare" else BarrierState.FAILED
    )
    nodes = ["node-a", "node-b"]
    target_index = nodes.index(target_node)
    expected_states = [
        BarrierParticipantState.FAILED
        if index == target_index
        else BarrierParticipantState.PREPARED
        if phase == "prepare" and index < target_index
        else BarrierParticipantState.PENDING
        if phase == "prepare"
        else BarrierParticipantState.COMMITTED
        if index < target_index
        else BarrierParticipantState.PREPARED
        for index in range(2)
    ]
    assert [item.state for item in barrier.participants] == expected_states
    expected_calls = (
        [
            (WorkflowOperation.VERIFY_NO_GPU_CLIENTS, node)
            for node in nodes[: target_index + 1]
        ]
        if phase == "prepare"
        else [(WorkflowOperation.VERIFY_NO_GPU_CLIENTS, node) for node in nodes]
        + [(WorkflowOperation.RESET_GPU, node) for node in nodes[: target_index + 1]]
    )
    assert calls == expected_calls
    completed_nodes = nodes[:target_index] if phase == "commit" else []
    assert result.details["completed_nodes"] == completed_nodes
    assert result.details["node_results"] == {
        node: {"verified": True} for node in completed_nodes
    }
    previous_calls = list(calls)
    replay = adapter.execute(context)
    assert replay.status is WorkflowStepStatus.FAILED
    assert replay.details["manual_confirmation_required"] is True
    assert calls == previous_calls
