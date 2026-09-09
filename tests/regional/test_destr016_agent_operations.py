"""DESTR-016 asks the Node Agent only for node-action operations."""

from __future__ import annotations

from scripts.e2e.regional import destr016_verdicts as verdicts


def test_agent_operations_only_name_node_action_operations() -> None:
    """VALIDATE_GPU runs through the GPU_VALIDATION adapter; requiring it in the
    Agent's allowed_operations refused every live node (2026-09-08)."""
    from gpu_fault.models import WorkflowOperation
    from gpu_fault.operation_registry import OperationAdapter, operations_for_adapter

    node_actions = {
        item.value for item in operations_for_adapter(OperationAdapter.NODE_ACTION)
    }
    assert set(verdicts.AGENT_OPERATIONS) <= node_actions, verdicts.AGENT_OPERATIONS
    assert WorkflowOperation.VALIDATE_GPU.value not in verdicts.AGENT_OPERATIONS
