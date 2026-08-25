from __future__ import annotations

from collections.abc import Sequence
import logging
from typing import Any

from gpu_fault.models import (
    FaultIncident,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStepSpec,
)
from gpu_fault.operation_registry import (
    DESTRUCTIVE_OPERATIONS,
    OperationAdapter,
    operations_for_adapter,
)


_PREFLIGHTED_NODE_ACTION_OPERATIONS = operations_for_adapter(
    OperationAdapter.NODE_ACTION
) - {WorkflowOperation.COLLECT_HUNG_TRIAGE}
LOGGER = logging.getLogger(__name__)


def fleet_preflight_reason(
    registry: Any,
    workflow: WorkflowRequest,
    incident: FaultIncident,
    steps: Sequence[WorkflowStepSpec],
) -> str | None:
    """Explain why a workflow cannot safely enter its destructive phase."""

    completed = set(workflow.completed_step_indexes)
    if any(
        index in completed and step.operation in DESTRUCTIVE_OPERATIONS
        for index, step in enumerate(steps)
    ):
        # Do not strand compensation after a mutation. Remaining node
        # actions still enforce their normal generation/readiness gate.
        return None

    pending = [step for index, step in enumerate(steps) if index not in completed]
    if not any(step.operation in DESTRUCTIVE_OPERATIONS for step in pending):
        return None
    node_action_steps = [
        step
        for step in pending
        if step.operation in _PREFLIGHTED_NODE_ACTION_OPERATIONS
    ]
    if not node_action_steps:
        return None
    node_ids = list(
        dict.fromkeys(
            node_id for step in node_action_steps for node_id in step.node_ids
        )
    )
    if not node_ids:
        return (
            "fleet compatibility preflight blocked destructive workflow: "
            "a pending node action has no explicit node target"
        )
    try:
        report = registry.readiness(incident.cluster_id, node_ids)
    except Exception as exc:  # noqa: BLE001 - fail closed before mutation
        return (
            "fleet compatibility preflight is unavailable before "
            f"destructive workflow steps: {type(exc).__name__}: {exc}"
        )
    if report.ready:
        return None
    reasons = [
        f"{node.node_id}: {reason}" for node in report.nodes for reason in node.reasons
    ]
    return "fleet compatibility preflight blocked destructive workflow: " + (
        "; ".join(reasons) or "fleet readiness returned false"
    )


def command_requires_fleet_preflight(
    operation: WorkflowOperation,
) -> bool:
    """Return true only at a remote destructive-action boundary."""

    return operation in DESTRUCTIVE_OPERATIONS


def held_workflow_result(
    executor: Any,
    workflow: WorkflowRequest,
    incident: FaultIncident,
    steps: Sequence[WorkflowStepSpec],
) -> Any | None:
    registry = getattr(executor, "fleet_registry", None)
    if registry is None:
        return None
    reason = fleet_preflight_reason(
        registry,
        workflow,
        incident,
        steps,
    )
    if reason is None:
        return None
    LOGGER.warning(
        "workflow held before destructive steps: "
        "workflow=%s incident=%s cluster=%s reason=%s",
        workflow.request_id,
        incident.incident_id,
        incident.cluster_id,
        reason,
    )
    return executor._result(workflow, incident, error=reason)
