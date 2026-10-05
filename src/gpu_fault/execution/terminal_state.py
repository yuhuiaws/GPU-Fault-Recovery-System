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
    WorkflowStepStatus,
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


_ISOLATION_OPERATIONS = frozenset(
    {WorkflowOperation.MARK_UNSCHEDULABLE, WorkflowOperation.QUARANTINE}
)


def failure_incident_state(workflow: WorkflowRequest) -> IncidentState:
    if _ISOLATION_OPERATIONS.intersection(workflow.completed_operations):
        return IncidentState.QUARANTINED
    if any(
        execution.operation in _ISOLATION_OPERATIONS
        and execution.status is WorkflowStepStatus.FAILED
        and execution.details.get("isolated_nodes")
        for execution in workflow.step_executions
    ):
        # An isolation that failed after cordoning some of its nodes still
        # holds those nodes: the record says QUARANTINED so the operator exit
        # is the validated restore, not a close that orphans the isolation.
        return IncidentState.QUARANTINED
    if diagnostic_only(workflow):
        return IncidentState.RECOVERED
    return IncidentState.ESCALATED


def _completed_nodes(
    workflow: WorkflowRequest, operations: frozenset[WorkflowOperation]
) -> set[str]:
    steps = (
        workflow.safety_steps
        if workflow.executes_safety_steps
        else workflow.official_steps
    )
    completed = set(workflow.completed_step_indexes)
    return {
        node
        for index, step in enumerate(steps)
        if index in completed and step.operation in operations
        for node in step.node_ids
    }


_RELEASE_OPERATIONS = frozenset({WorkflowOperation.RESTORE_SCHEDULING})


def predecessor_holds_isolation(
    predecessor: WorkflowRequest | None, workflow: WorkflowRequest
) -> bool:
    """Nodes the incident's predecessor workflow cordoned or quarantined that
    neither it nor this successor released with a completed RESTORE_SCHEDULING.

    A successor queued on the incident takes its pointer; when the predecessor
    then ends FAILED its QUARANTINED verdict is not written (``_save_terminal``
    keeps a stale snapshot from stealing the pointer back), and the successor's
    own verdict -- RECOVERED for a plan with no containment -- would call a
    node that is still cordoned recovered (live 2026-10-05, gpu-eks-cluster: a
    RESTART_GPU_DEVICE_PLUGIN successor of a failed RESET_GPU). The isolation is
    the incident's, so the successor's verdict carries it.
    """

    if predecessor is None or predecessor.incident_id != workflow.incident_id:
        return False
    isolated = _completed_nodes(predecessor, _ISOLATION_OPERATIONS)
    released = _completed_nodes(predecessor, _RELEASE_OPERATIONS) | _completed_nodes(
        workflow, _RELEASE_OPERATIONS
    )
    return bool(isolated - released)


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
    predecessor: WorkflowRequest | None = None,
) -> TerminalDecision:
    """Build new records without changing the input records or writing anything.

    ``predecessor`` is the incident's previous workflow when this record names
    one: isolation it left on the nodes keeps the incident QUARANTINED whatever
    this record's own steps say (``predecessor_holds_isolation``).
    """

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
    if (
        incident_state is None
        and derived_state in {IncidentState.RECOVERED, IncidentState.ESCALATED}
        and predecessor_holds_isolation(predecessor, ended)
    ):
        derived_state = IncidentState.QUARANTINED
    inconclusive = (
        status is WorkflowStatus.FAILED
        and incident_state is None
        and diagnostic_only(ended)
    )
    if status is WorkflowStatus.FAILED and reason and not ended.terminal_failure_reason:
        # The reason is the record's, not only the audit event's: the API, the
        # acceptance verdicts and the failure handler read the field (live
        # 2026-10-05 both records of a reset-to-reboot chain ended FAILED with
        # the field None and the reason only in the TERMINAL event).
        ended = ended.model_copy(
            update={
                "terminal_failure_reason": (
                    inconclusive_reason(reason) if inconclusive else reason
                )
            }
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
