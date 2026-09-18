"""Pure terminal decisions; persistence, leases and hooks belong to the executor."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from gpu_fault.execution.node_action_uncertainty import (
    UNRESOLVED_NODE_ACTION,
    has_unresolved_node_action,
)
from gpu_fault.models import (
    BlockedKind,
    FaultIncident,
    IncidentState,
    WorkflowEventCode,
    WorkflowEventKind,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    bounded_reasons,
    record_workflow_event,
)
from gpu_fault.operation_registry import (
    DESTRUCTIVE_OPERATIONS,
    NODE_WIDE_RECOVERY_OPERATIONS,
)
from gpu_fault.workflow_quarantine import has_unrecovered_quarantine

DIAGNOSTIC_INCONCLUSIVE_REASON = "diagnostic inconclusive"


def inconclusive_reason(error: str | None) -> str:
    return (
        f"{DIAGNOSTIC_INCONCLUSIVE_REASON}: {error}"
        if error
        else DIAGNOSTIC_INCONCLUSIVE_REASON
    )


def diagnostic_only(workflow: WorkflowRequest) -> bool:
    """A validation before a planned mutation is a gate, not a pure diagnostic."""

    steps = (
        workflow.safety_steps
        if workflow.executes_safety_steps
        else workflow.official_steps
    )
    operations = (
        {step.operation for step in steps}
        | set(workflow.completed_operations)
        | {item.operation for item in workflow.step_executions}
    )
    return bool(operations) and not (
        operations & (DESTRUCTIVE_OPERATIONS | NODE_WIDE_RECOVERY_OPERATIONS)
    )


def failure_incident_state(workflow: WorkflowRequest) -> IncidentState:
    if {
        WorkflowOperation.MARK_UNSCHEDULABLE,
        WorkflowOperation.QUARANTINE,
    }.intersection(workflow.completed_operations):
        return IncidentState.QUARANTINED
    if diagnostic_only(workflow):
        return IncidentState.RECOVERED
    return IncidentState.ESCALATED


def terminal_incident_state(
    workflow: WorkflowRequest, status: WorkflowStatus
) -> IncidentState:
    if status is WorkflowStatus.FAILED:
        return failure_incident_state(workflow)
    if status is WorkflowStatus.BLOCKED:
        return IncidentState.QUARANTINED
    completed = workflow.completed_operations
    if (
        status is WorkflowStatus.SUCCEEDED
        and WorkflowOperation.ESCALATE_SUPPORT in completed
    ):
        return IncidentState.ESCALATED
    isolated = has_unrecovered_quarantine(
        workflow.model_copy(update={"status": status})
    )
    return IncidentState.QUARANTINED if isolated else IncidentState.RECOVERED


@dataclass(frozen=True)
class TerminalDecision:
    workflow: WorkflowRequest
    incident: FaultIncident | None
    error: str | None
    diagnostic_inconclusive: bool


def terminal_decision(
    workflow: WorkflowRequest,
    incident: FaultIncident | None,
    status: WorkflowStatus,
    execution_epoch: int,
    *,
    now: datetime,
    actor: str,
    reason: str | None = None,
    incident_state: IncidentState | None = None,
    updates: Mapping[str, object] | None = None,
) -> TerminalDecision:
    """Build new records without changing the input records or writing anything."""

    if has_unresolved_node_action(workflow):
        status = WorkflowStatus.BLOCKED
        updates = {
            **(dict(updates) if updates else {}),
            "blocked_kind": BlockedKind.NEEDS_OPERATOR,
        }
        state = incident_state or failure_incident_state(workflow)
        incident_state = (
            IncidentState.QUARANTINED
            if state is IncidentState.QUARANTINED
            else IncidentState.ESCALATED
        )
        reason = (
            f"{reason}; {UNRESOLVED_NODE_ACTION}" if reason else UNRESOLVED_NODE_ACTION
        )
    ended = workflow.model_copy(
        update={
            **(dict(updates) if updates else {}),
            "status": status,
            "execution_owner_id": None,
            "execution_lease_expires_at": None,
            "updated_at": now,
        }
    )
    derived_state = (
        incident_state
        if incident_state is not None
        else terminal_incident_state(ended, status)
    )
    inconclusive = (
        status is WorkflowStatus.FAILED
        and incident_state is None
        and diagnostic_only(ended)
    )
    if incident is not None:
        update: dict[str, Any] = {"state": derived_state, "updated_at": now}
        if inconclusive:
            update["reasons"] = bounded_reasons(
                [*incident.reasons, inconclusive_reason(reason)]
            )
        incident = incident.model_copy(update=update)
    details: dict[str, Any] = {
        "execution_epoch": execution_epoch,
        "incident_state": derived_state.value,
    }
    if inconclusive:
        details["diagnostic_inconclusive"] = True
    ended = record_workflow_event(
        ended,
        WorkflowEventKind.TERMINAL,
        code=WorkflowEventCode.TERMINALIZED.value,
        reason=inconclusive_reason(reason) if inconclusive else reason,
        actor=actor,
        status=status.value,
        details=details,
        at=now,
    )
    return TerminalDecision(ended, incident, reason, inconclusive)
