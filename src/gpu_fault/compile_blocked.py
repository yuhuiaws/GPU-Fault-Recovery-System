"""Close a workflow that BLOCKED at compile time and never reached a node.

The compiler refuses a workflow before any step runs when the Runtime Profile
has no executable owner for one of its capabilities (``blocked_reasons`` says
which). Such a record is BLOCKED, was never claimed, holds no lease, has no
step executions, no remote commands and -- because it was never plan-driven --
no source recovery plan. Nothing else closes it: the validated restore its
incident's operator runs finds no isolation from this incident to undo, so no
verified restore successor ever appears, and the restore reconcile refuses it by
design (no source plan to carry the audit).

Meanwhile the release preflight (``workflow_safety``) counted it as an active
destructive workflow, so the record blocked exactly the release that would add
the missing owner. Observed live on a regional site: a ``REMEDIATE_EFA_DRIVER``
workflow BLOCKED with ``no executable owner for efaDriverRemediation``. Until
2026-09-08 the close was an operator's ``workflow-reconcile --mode
compile-blocked``; every live use was to unblock a release preflight, and every
condition it checked is a Store predicate, so the dispatcher now closes the
shape itself on its periodic sweep (``WorkflowDispatcher.sweep_stuck_records``)
and the preflight probe sets the shape aside with the same inlined predicate
(``tests/regional/test_release_workflow_safety.py`` pins the two copies).

The verdict never keys on age. It is re-derived from a fresh read every tick,
written through a compare-and-set ``save_workflow`` so a record that moved
since the read is refused rather than overwritten, ends the workflow
``SUPERSEDED`` with the original ``blocked_reasons`` in ``preemption_reason``,
and appends an ``OPERATOR_RECONCILED`` event with actor ``dispatcher`` in the
same write. The incident is not written: a BLOCKED-at-compile incident owns no
node state to release, and a blind incident overwrite is what the two-step
write must avoid. Nothing is deleted.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Iterable, Mapping

from gpu_fault.models import (
    IncidentState,
    WorkflowEventKind,
    WorkflowRequest,
    WorkflowStatus,
    record_operator_event,
)
from gpu_fault.remote_command_models import RemoteCommandStatus
from gpu_fault.store import NotFoundError

LOGGER = logging.getLogger(__name__)

DISPATCHER_ACTOR = "dispatcher"
CLOSE_MARKER = "closed compile-time BLOCKED workflow"
#: The second shape the sweep closes: a BLOCKED record -- dispatched or not --
#: whose incident an operator (or a validated restore) has since closed
#: RECOVERED. Nothing waits on it any more (``close_incident`` refuses while a
#: workflow is still open, so a BLOCKED row it left behind is settled paperwork)
#: and nothing else ends it, yet the release preflight counted it as active
#: destructive work: an Always-Fatal SXID's BLOCKED ``RESTART_BM`` blocked the
#: release carrying the fix for the replayed log line that opened it
#: (2026-09-11).
SETTLED_CLOSE_MARKER = "closed BLOCKED workflow of a RECOVERED incident"
SETTLED_INCIDENT_STATES = frozenset({IncidentState.RECOVERED, IncidentState.ESCALATED})
OPEN_REMOTE_STATUSES = frozenset(
    {
        RemoteCommandStatus.PENDING,
        RemoteCommandStatus.LEASED,
        RemoteCommandStatus.WAITING,
    }
)
# Newest first: a record that just blocked at compile time is the one about to
# hold a release, and the BLOCKED history behind it only grows.
SWEEP_LIMIT = 1000


def already_closed(workflow: WorkflowRequest) -> bool:
    """Whether an earlier sweep (or the retired operator mode) closed this record.

    Recognised by the marker in ``preemption_reason`` rather than by status
    alone, so a workflow superseded for any other reason is still reported with
    its real status instead of silently skipped.
    """

    return workflow.status is WorkflowStatus.SUPERSEDED and CLOSE_MARKER in str(
        workflow.preemption_reason or ""
    )


def never_dispatched_reasons(workflow: WorkflowRequest) -> list[str]:
    """Every reason the row *alone* is not a compile-time BLOCKED no-op.

    This half needs no other read, so the sweep asks it of every BLOCKED row
    first and pays for the incident and command reads only on the survivors.
    ``execution_epoch`` is the guard that separates a compile-time refusal from
    a dispatch-time one: a claim bumps the epoch, so a record the dispatcher
    itself BLOCKED for an internal error (``BlockedKind.INTERNAL_ERROR``) has
    been claimed and is left for the operator the alert names.
    """

    reasons: list[str] = []
    if workflow.status is not WorkflowStatus.BLOCKED:
        reasons.append(f"workflow is {workflow.status.value}, not BLOCKED")
    if not workflow.blocked_reasons:
        reasons.append(
            "workflow carries no blocked_reasons, so it did not block at compile time"
        )
    if workflow.execution_epoch:
        reasons.append(
            "workflow was claimed by an executor "
            f"(execution_epoch {workflow.execution_epoch}), "
            "so it did not block at compile time"
        )
    if workflow.step_executions:
        reasons.append("workflow has step executions, so it was dispatched")
    if workflow.completed_operations or workflow.completed_step_indexes:
        reasons.append("workflow completed operations, so it changed state")
    if workflow.execution_owner_id:
        reasons.append("workflow still has an execution owner")
    if workflow.source_plan_id:
        reasons.append(
            "workflow has a source recovery plan; the restore reconcile owns it"
        )
    if workflow.remediation_budget_claims:
        reasons.append("workflow holds remediation budget claims")
    return reasons


def compile_blocked_reasons(
    workflow: WorkflowRequest,
    incident: Any | None,
    open_commands: Iterable[str],
) -> list[str]:
    """Every reason this record is not a compile-time BLOCKED no-op.

    Each condition is required; together they leave no reading under which the
    workflow did, or still could, change a node, or under which anyone is still
    waiting on it.
    """

    reasons = never_dispatched_reasons(workflow)
    commands = sorted(str(item) for item in open_commands)
    if commands:
        reasons.append("workflow has open remote commands: " + ", ".join(commands))
    if incident is None:
        reasons.append("incident is missing")
    elif incident.state not in SETTLED_INCIDENT_STATES:
        reasons.append(
            f"incident is {incident.state.value}, still waiting on a workflow"
        )
    return reasons


def closed_compile_blocked_record(
    workflow: WorkflowRequest,
    *,
    reconciled_at: datetime,
    actor: str,
    reference: str | None = None,
    details: Mapping[str, Any] | None = None,
) -> WorkflowRequest:
    """The SUPERSEDED form of ``workflow``, or a refusal if it is not the shape.

    ``actor`` and ``reference`` go on the ``OPERATOR_RECONCILED`` event the
    returned copy carries, so the audit lands in the same write as the close.
    """

    if workflow.status is not WorkflowStatus.BLOCKED:
        raise ValueError(f"workflow is {workflow.status.value}, not BLOCKED")
    if workflow.step_executions or workflow.completed_operations:
        raise ValueError("workflow was dispatched, so it is not a compile-time record")
    attribution = f"{actor} reconciliation" + (f" {reference}" if reference else "")
    closed: WorkflowRequest = workflow.model_copy(
        update={
            "status": WorkflowStatus.SUPERSEDED,
            "preemption_reason": (
                f"{attribution}: {CLOSE_MARKER} that never changed a node; "
                "blocked_reasons="
                + json.dumps(list(workflow.blocked_reasons), sort_keys=True)
            ),
            "superseded_at": reconciled_at,
            "updated_at": reconciled_at,
        }
    )
    recorded: WorkflowRequest = record_operator_event(
        closed,
        WorkflowEventKind.OPERATOR_RECONCILED,
        actor=actor,
        reference=reference,
        previous_status=workflow.status,
        at=reconciled_at,
        details={
            "terminalization": CLOSE_MARKER,
            "blocked_reasons": list(workflow.blocked_reasons),
            **dict(details or {}),
        },
    )
    return recorded


def close_compile_blocked_workflow(
    store: Any,
    workflow: WorkflowRequest,
    *,
    now: datetime,
) -> bool:
    """Close one BLOCKED record if every condition holds; ``False`` otherwise.

    The incident and the command list are read fresh here, and the write is a
    compare-and-set against the row the caller listed: a record another writer
    moved in between raises ``StaleWriteError`` up to the sweep, which logs it
    and re-reads next tick instead of overwriting.
    """

    incident: Any | None
    try:
        incident = store.get_incident(workflow.incident_id)
    except (KeyError, NotFoundError):
        incident = None
    open_commands = [
        str(command.command_id)
        for command in store.list_remote_commands(
            workflow_request_ids=[workflow.request_id]
        )
        if command.status in OPEN_REMOTE_STATUSES
    ]
    if compile_blocked_reasons(workflow, incident, open_commands):
        return False
    closed = closed_compile_blocked_record(
        workflow, reconciled_at=now, actor=DISPATCHER_ACTOR
    )
    store.save_workflow(closed, expected=workflow)
    LOGGER.warning(
        "compile-time BLOCKED workflow closed by the dispatcher: workflow=%s "
        "incident=%s incident_state=%s blocked_reasons=%s",
        workflow.request_id,
        workflow.incident_id,
        incident.state.value if incident is not None else None,
        json.dumps(list(workflow.blocked_reasons), sort_keys=True),
    )
    return True


def settled_incident_blocked_reasons(
    workflow: WorkflowRequest,
    incident: Any | None,
    open_commands: Iterable[str],
) -> list[str]:
    """Why ``workflow`` is not a BLOCKED record of a RECOVERED incident.

    Fewer guards than the compile-time shape on purpose: this record may have
    run steps. What makes it safe to end is that its incident is closed --
    RECOVERED, not ESCALATED, which still awaits an operator who may act on the
    record -- and that nothing on it is live: no execution owner, no budget
    claim, no open remote command.
    """

    reasons: list[str] = []
    if workflow.status is not WorkflowStatus.BLOCKED:
        reasons.append(f"workflow is {workflow.status.value}, not BLOCKED")
    if workflow.execution_owner_id:
        reasons.append("workflow still has an execution owner")
    if workflow.remediation_budget_claims:
        reasons.append("workflow holds remediation budget claims")
    if workflow.source_plan_id:
        # Plan-driven: the restore reconcile owns it and carries its audit.
        reasons.append(
            "workflow has a source recovery plan; the restore reconcile owns it"
        )
    commands = list(open_commands)
    if commands:
        reasons.append("workflow has open remote commands: " + ", ".join(commands))
    if incident is None:
        reasons.append("incident is missing")
    elif incident.state is not IncidentState.RECOVERED:
        reasons.append(f"incident is {incident.state.value}, not RECOVERED")
    return reasons


def closed_settled_incident_record(
    workflow: WorkflowRequest,
    *,
    reconciled_at: datetime,
    actor: str,
    reference: str | None = None,
) -> WorkflowRequest:
    """The SUPERSEDED form of a BLOCKED record whose incident is RECOVERED."""

    if workflow.status is not WorkflowStatus.BLOCKED:
        raise ValueError(f"workflow is {workflow.status.value}, not BLOCKED")
    attribution = f"{actor} reconciliation" + (f" {reference}" if reference else "")
    closed: WorkflowRequest = workflow.model_copy(
        update={
            "status": WorkflowStatus.SUPERSEDED,
            "preemption_reason": (
                f"{attribution}: {SETTLED_CLOSE_MARKER}; blocked_reasons="
                + json.dumps(list(workflow.blocked_reasons), sort_keys=True)
            ),
            "superseded_at": reconciled_at,
            "updated_at": reconciled_at,
        }
    )
    return record_operator_event(
        closed,
        WorkflowEventKind.OPERATOR_RECONCILED,
        actor=actor,
        reference=reference,
        previous_status=workflow.status,
        at=reconciled_at,
        details={
            "terminalization": SETTLED_CLOSE_MARKER,
            "blocked_reasons": list(workflow.blocked_reasons),
            "completed_operations": [
                str(getattr(item, "value", item))
                for item in workflow.completed_operations
            ],
        },
    )


def close_settled_incident_blocked_workflow(
    store: Any,
    workflow: WorkflowRequest,
    *,
    now: datetime,
    actor: str = DISPATCHER_ACTOR,
    reference: str | None = None,
) -> bool:
    """Close one BLOCKED record of a RECOVERED incident; ``False`` otherwise.

    Same discipline as the compile-time close: incident and commands read
    fresh, one compare-and-set write, the audit event in that write.
    """

    incident: Any | None
    try:
        incident = store.get_incident(workflow.incident_id)
    except (KeyError, NotFoundError):
        incident = None
    open_commands = [
        str(command.command_id)
        for command in store.list_remote_commands(
            workflow_request_ids=[workflow.request_id]
        )
        if command.status in OPEN_REMOTE_STATUSES
    ]
    if settled_incident_blocked_reasons(workflow, incident, open_commands):
        return False
    closed = closed_settled_incident_record(
        workflow, reconciled_at=now, actor=actor, reference=reference
    )
    store.save_workflow(closed, expected=workflow)
    LOGGER.warning(
        "BLOCKED workflow of a RECOVERED incident closed by %s: workflow=%s "
        "incident=%s blocked_reasons=%s",
        actor,
        workflow.request_id,
        workflow.incident_id,
        json.dumps(list(workflow.blocked_reasons), sort_keys=True),
    )
    return True


def close_compile_blocked_workflows(
    store: Any,
    *,
    now: datetime,
    limit: int = SWEEP_LIMIT,
) -> list[str]:
    """The dispatcher's sweep: close every BLOCKED record nothing waits on.

    Two shapes: the compile-time no-op (never dispatched, incident settled) and
    the BLOCKED record of an incident already RECOVERED. One isolated write per
    record; a failure on one is logged and leaves that record for the next tick
    without stopping the others. Returns the ids closed this tick. Nothing is
    deleted.
    """

    closed_ids: list[str] = []
    for workflow in store.list_workflows(
        {WorkflowStatus.BLOCKED}, limit=limit, newest_first=True
    ):
        try:
            if never_dispatched_reasons(workflow):
                closed = close_settled_incident_blocked_workflow(
                    store, workflow, now=now
                )
            else:
                closed = close_compile_blocked_workflow(
                    store, workflow, now=now
                ) or close_settled_incident_blocked_workflow(store, workflow, now=now)
        except Exception:  # noqa: BLE001 - keep sweeping, leave the record as read
            LOGGER.exception(
                "compile-time BLOCKED close failed, workflow left as it was: %s",
                workflow.request_id,
            )
            continue
        if closed:
            closed_ids.append(workflow.request_id)
    return closed_ids
