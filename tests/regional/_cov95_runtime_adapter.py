from __future__ import annotations

from dataclasses import replace
from typing import Any

from gpu_fault.adapters import NodeActionWorkflowAdapter
from gpu_fault.execution import WorkflowStepContext
from gpu_fault.fleet import BarrierCoordinator
from gpu_fault.models import WorkflowOperation
from tests._builders import workflow_step
from tests.execution.test_node_action_transport_retry import step_context
from tests.fleet._support import SECRET, heartbeat, registry, signed


def fleet_adapter(**kwargs: Any) -> NodeActionWorkflowAdapter:
    fleet = registry()
    for node in ("node-a", "node-b"):
        fleet.register(signed(heartbeat(node)))
    return NodeActionWorkflowAdapter(
        {},
        SECRET,
        registry=fleet,
        barriers=BarrierCoordinator(fleet.store, now=fleet.now),
        **kwargs,
    )


def context_for(
    adapter: NodeActionWorkflowAdapter,
    operation: WorkflowOperation = WorkflowOperation.RESET_GPU,
    *,
    nodes: list[str] | None = None,
    parameters: dict[str, Any] | None = None,
    **changes: Any,
) -> WorkflowStepContext:
    context = step_context(adapter, operation)
    step = workflow_step(
        operation,
        adapter.owner,
        node_ids=["node-a", "node-b"] if nodes is None else nodes,
        gpu_uuids=["GPU-a", "GPU-b"],
        parameters=(
            {"gpu_uuids_by_node": {"node-a": ["GPU-a"], "node-b": ["GPU-b"]}}
            if parameters is None
            else parameters
        ),
    )
    return replace(
        context,
        step=step,
        workflow=context.workflow.model_copy(update={"official_steps": [step]}),
        **changes,
    )
