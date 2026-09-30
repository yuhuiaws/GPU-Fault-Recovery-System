"""Resume a NEEDS_OPERATOR record whose open question a remote receipt answered.

``terminal_state.terminal_decision`` parks a workflow ``BLOCKED / NEEDS_OPERATOR``
when a node action's outcome is unknown at the moment it ends (F-A4): nothing
automatic may treat the node as free while a reset or reboot may still be
running. The question is normally answered by an operator confirmation
(``submit-remediation --disposition confirm-node-action``), and the incident
close settles the record afterwards.

One shape needs no operator: the executor itself answers. A workflow deadline
only *requests* the cancellation of a LEASED remote command; the executor that
holds the lease reports what ran -- or, when the cancellation was observed at
admission, that nothing did (``executor-cancelled-before-start``,
``node_action_not_started``). That receipt can land milliseconds after the
engine, in the same tick as the cancellation, already parked the record.
Live 2026-09-28 (GF-REGIONAL-DESTR-018): the receipt resolved every unresolved
action, but nothing read it -- the record stayed NEEDS_OPERATOR, the
compensation restore it still owed was never run, the incident never reached
the support hand-off, and the confirm verb refused because a never-started
reset has no agent ``node_results`` to confirm.

The dispatcher's sweep therefore re-opens such a record: when a fresh read of
its remote commands shows no unresolved node action left, no operator
confirmation involved, no open command, and the incident still pointing at
this record, the record goes back to PENDING with its failure parked
(``pending_failure_*``) so the ordinary executor path runs the compensation
it owes and ends it FAILED -- the shape the record would have had if the
receipt had arrived one tick earlier. Fail-closed stays intact: a record whose
action is still unknown, or that an operator has confirmed, is not touched.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from gpu_fault.execution.node_action_uncertainty import (
    UNRESOLVED_NODE_ACTION,
    has_unresolved_node_action,
    node_actions_operator_confirmed,
    refresh_remote_action_state,
)
from gpu_fault.models import (
    BlockedKind,
    IncidentState,
    WorkflowEventKind,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepStatus,
    record_operator_event,
)
from gpu_fault.remote_command_models import RemoteCommandStatus
from gpu_fault.store import NotFoundError

LOGGER = logging.getLogger(__name__)

DISPATCHER_ACTOR = "dispatcher"
RESUME_MARKER = (
    "resumed NEEDS_OPERATOR workflow after remote receipts resolved its node actions"
)
OPEN_REMOTE_STATUSES = frozenset(
    {
        RemoteCommandStatus.PENDING,
        RemoteCommandStatus.LEASED,
        RemoteCommandStatus.WAITING,
    }
)
SWEEP_LIMIT = 1000


def parked_for_unresolved_action(workflow: WorkflowRequest) -> bool:
    """Whether ``terminal_decision`` parked this record for an unknown outcome."""

    terminal = [
        item for item in workflow.events if item.kind is WorkflowEventKind.TERMINAL
    ]
    if terminal and UNRESOLVED_NODE_ACTION in str(terminal[-1].reason or ""):
        return True
    return any(
        item.operation is WorkflowOperation.RESTORE_GPU_SERVICES
        and item.status is WorkflowStepStatus.FAILED
        and item.details.get("restore_gpu_services_withheld") is True
        for item in workflow.step_executions
    )


def receipt_settled_resume_reasons(
    store: Any, workflow: WorkflowRequest, *, now: datetime
) -> tuple[list[str], WorkflowRequest]:
    """Why the record must stay parked; empty when the receipts settled it.

    Returns the refreshed record alongside so the caller writes what it judged.
    """

    reasons: list[str] = []
    if (
        workflow.status is not WorkflowStatus.BLOCKED
        or workflow.blocked_kind is not BlockedKind.NEEDS_OPERATOR
    ):
        return ["workflow is not a NEEDS_OPERATOR record"], workflow
    if not parked_for_unresolved_action(workflow):
        reasons.append("record was not parked for an unresolved node action")
    if node_actions_operator_confirmed(workflow):
        reasons.append(
            "an operator confirmation answers the record; the incident close settles it"
        )
    refreshed = refresh_remote_action_state(store, workflow)
    if has_unresolved_node_action(refreshed):
        reasons.append("a node action is still unresolved")
    open_commands = sorted(
        str(command.command_id)
        for command in store.list_remote_commands(
            workflow_request_ids=[workflow.request_id]
        )
        if command.status in OPEN_REMOTE_STATUSES
    )
    if open_commands:
        reasons.append(f"remote command(s) still open: {', '.join(open_commands)}")
    try:
        incident = store.get_incident(workflow.incident_id)
    except (KeyError, NotFoundError):
        incident = None
    if incident is None:
        reasons.append("incident is missing")
    else:
        if incident.workflow_request_id != workflow.request_id:
            reasons.append("the incident has moved on to another workflow")
        if incident.state is IncidentState.RECOVERED:
            reasons.append(
                "incident already RECOVERED; the settled close ends the record"
            )
    if (
        workflow.execution_lease_expires_at is not None
        and workflow.execution_lease_expires_at > now
    ):
        reasons.append("workflow execution lease has not expired")
    return reasons, refreshed


def _parked_failure(workflow: WorkflowRequest) -> tuple[int | None, str]:
    """The step whose failure the record carried into the park, and its error."""

    failed = [
        item
        for item in workflow.step_executions
        if item.status is WorkflowStepStatus.FAILED
        and item.operation is not WorkflowOperation.RESTORE_GPU_SERVICES
    ]
    index = failed[-1].step_index if failed else workflow.pending_failure_step_index
    terminal = [
        item for item in workflow.events if item.kind is WorkflowEventKind.TERMINAL
    ]
    reason = str(terminal[-1].reason or "") if terminal else ""
    reason = reason.replace(f"; {UNRESOLVED_NODE_ACTION}", "").replace(
        UNRESOLVED_NODE_ACTION, ""
    )
    error = (
        workflow.pending_failure_error
        or (failed[-1].error if failed and failed[-1].error else None)
        or reason.strip()
        or "workflow step failed"
    )
    return index, error


def resumed_record(
    workflow: WorkflowRequest,
    refreshed: WorkflowRequest,
    *,
    now: datetime,
    actor: str = DISPATCHER_ACTOR,
) -> WorkflowRequest:
    """The PENDING form of a parked record whose receipts arrived."""

    index, error = _parked_failure(refreshed)
    resumed: WorkflowRequest = refreshed.model_copy(
        update={
            "status": WorkflowStatus.PENDING,
            "blocked_kind": None,
            "pending_failure_step_index": index,
            "pending_failure_error": error,
            "execution_owner_id": None,
            "execution_lease_expires_at": None,
            "updated_at": now,
        }
    )
    return record_operator_event(
        resumed,
        WorkflowEventKind.OPERATOR_RECONCILED,
        actor=actor,
        reference=None,
        previous_status=workflow.status,
        at=now,
        details={
            "resumption": RESUME_MARKER,
            "pending_failure_step_index": index,
            "pending_failure_error": error,
        },
    )


def resume_receipt_settled_workflows(
    store: Any,
    *,
    now: datetime,
    actor: str = DISPATCHER_ACTOR,
    limit: int = SWEEP_LIMIT,
) -> list[str]:
    """The dispatcher's sweep: re-open every parked record its receipts settled.

    One compare-and-set write per record with the audit event in that write;
    a failure on one record is logged and leaves it for the next tick.
    """

    resumed_ids: list[str] = []
    for workflow in store.list_workflows(
        {WorkflowStatus.BLOCKED}, limit=limit, newest_first=True
    ):
        if workflow.blocked_kind is not BlockedKind.NEEDS_OPERATOR:
            continue
        try:
            reasons, refreshed = receipt_settled_resume_reasons(
                store, workflow, now=now
            )
            if reasons:
                continue
            store.save_workflow(
                resumed_record(workflow, refreshed, now=now, actor=actor),
                expected=workflow,
            )
        except Exception:  # noqa: BLE001 - keep sweeping, leave the record as read
            LOGGER.exception(
                "receipt-settled resume failed, workflow left parked: %s",
                workflow.request_id,
            )
            continue
        resumed_ids.append(workflow.request_id)
        LOGGER.warning(
            "NEEDS_OPERATOR workflow resumed by %s: remote receipts resolved its node "
            "actions: workflow=%s incident=%s",
            actor,
            workflow.request_id,
            workflow.incident_id,
        )
    return resumed_ids
