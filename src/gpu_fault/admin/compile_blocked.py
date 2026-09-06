"""Close a workflow that BLOCKED at compile time and never reached a node.

The compiler refuses a workflow before any step runs when the Runtime Profile
has no executable owner for one of its capabilities (``blocked_reasons`` says
which). Such a record is BLOCKED, holds no lease, has no step executions, no
remote commands and -- because it was never plan-driven -- no source recovery
plan. Nothing closes it:

* the validated restore its incident's operator runs ends at
  ``RESTORE_SCHEDULING`` on a node that carries no isolation from this incident
  (the containment pre-actions never ran either), so no verified restore
  successor ever appears;
* ``--mode restore`` refuses it by design (no source plan to carry the audit),
  and ``--mode retired-generation`` only handles ``PENDING``/``RUNNING`` records
  an incident re-planned away from.

Meanwhile the release preflight (``workflow_safety``) counts it as an active
destructive workflow and the Runtime Profile transition gate counts it as
old-profile activity, so the record blocks exactly the release that would add
the missing owner. Observed live on a regional site: a ``REMEDIATE_EFA_DRIVER``
workflow BLOCKED with ``no executable owner for efaDriverRemediation``.

This module is the operator's audited close for that one shape. It is shipped
to the CPU ingress Pod as *source* and run against the deployed runtime -- the
same arrangement as ``retired_generation`` and for the same reason: the record
has to be closed before the release that carries this code -- so it imports
only long-standing modules and writes through ``save_workflow`` after
re-deriving every condition from a fresh read. The workflow ends ``SUPERSEDED``
with the operator's reference in ``preemption_reason``; the incident is not
written (a BLOCKED-at-compile incident owns no node state to release, and a
blind incident overwrite is what the two-step write must avoid).

The admin side adds the one condition the runtime cannot see: the GPU node
carries no gpu-fault isolation at all (schedulable, no quarantine taint, no
ownership annotations), checked through ``kubectl`` before the plan is written.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Iterable

from gpu_fault.models import IncidentState, WorkflowRequest, WorkflowStatus
from gpu_fault.remote_command_models import RemoteCommandStatus
from gpu_fault.store import NotFoundError

PLAN_MODE = "compile-blocked-plan"
APPLY_MODE = "compile-blocked-apply"
CLOSE_MARKER = "closed compile-time BLOCKED workflow"
SETTLED_INCIDENT_STATES = frozenset({IncidentState.RECOVERED, IncidentState.ESCALATED})
OPEN_REMOTE_STATUSES = frozenset(
    {
        RemoteCommandStatus.PENDING,
        RemoteCommandStatus.LEASED,
        RemoteCommandStatus.WAITING,
    }
)
# Restamped by writes that change nothing the verdict reads; hashing it made a
# reviewed plan unappliable, so it is shown to the operator but not bound.
DIGEST_EXCLUDED_ITEM_FIELDS = frozenset({"workflow_updated_at"})
MAX_WORKFLOW_IDS = 1000


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def requested_workflow_ids(requested: Iterable[str]) -> list[str]:
    values = sorted({str(item).strip() for item in requested if str(item).strip()})
    if not values:
        raise ValueError("compile-blocked reconcile requires explicit workflow IDs")
    if len(values) > MAX_WORKFLOW_IDS:
        raise ValueError(
            f"compile-blocked reconcile accepts at most {MAX_WORKFLOW_IDS} workflow IDs"
        )
    return values


def plan_digest_items(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            key: value
            for key, value in item.items()
            if key not in DIGEST_EXCLUDED_ITEM_FIELDS
        }
        for item in items
    ]


def already_closed(workflow: WorkflowRequest) -> bool:
    """Whether an earlier apply already closed this record.

    Recognised by the marker in ``preemption_reason`` rather than by status
    alone, so a workflow superseded for any other reason is still reported as
    ineligible with its real status instead of silently skipped.
    """

    return workflow.status is WorkflowStatus.SUPERSEDED and CLOSE_MARKER in str(
        workflow.preemption_reason or ""
    )


def compile_blocked_reasons(
    workflow: WorkflowRequest,
    incident: Any | None,
    open_commands: list[str],
) -> list[str]:
    """Every reason this record is not a compile-time BLOCKED no-op.

    Each condition is required; together they leave no reading under which the
    workflow did, or still could, change a node.
    """

    reasons: list[str] = []
    if workflow.status is not WorkflowStatus.BLOCKED:
        reasons.append(f"workflow is {workflow.status.value}, not BLOCKED")
    if not workflow.blocked_reasons:
        reasons.append(
            "workflow carries no blocked_reasons, so it did not block at compile time"
        )
    if workflow.step_executions:
        reasons.append("workflow has step executions, so it was dispatched")
    if workflow.completed_operations or workflow.completed_step_indexes:
        reasons.append("workflow completed operations, so it changed state")
    if getattr(workflow, "execution_owner_id", None):
        reasons.append("workflow still has an execution owner")
    if workflow.source_plan_id:
        reasons.append("workflow has a source recovery plan; use --mode restore")
    if getattr(workflow, "remediation_budget_claims", None):
        reasons.append("workflow holds remediation budget claims")
    if open_commands:
        reasons.append("workflow has open remote commands: " + ", ".join(open_commands))
    if incident is None:
        reasons.append("incident is missing")
    elif incident.state not in SETTLED_INCIDENT_STATES:
        reasons.append(
            f"incident is {incident.state.value}, still waiting on a workflow"
        )
    return reasons


def _plan_item(
    store: Any,
    request_id: str,
    commands: list[Any],
) -> dict[str, Any]:
    try:
        workflow: WorkflowRequest = store.get_workflow(request_id)
    except (KeyError, NotFoundError):
        return {
            "request_id": request_id,
            "eligible": False,
            "already_closed": False,
            "reasons": ["workflow does not exist"],
        }
    incident: Any | None
    try:
        incident = store.get_incident(workflow.incident_id)
    except (KeyError, NotFoundError):
        incident = None
    open_commands = sorted(
        str(item.command_id)
        for item in commands
        if item.workflow_request_id == request_id
        and item.status in OPEN_REMOTE_STATUSES
    )
    closed = already_closed(workflow)
    reasons = (
        [] if closed else compile_blocked_reasons(workflow, incident, open_commands)
    )
    return {
        "request_id": request_id,
        "incident_id": workflow.incident_id,
        "cluster_id": incident.cluster_id if incident is not None else None,
        "node_ids": sorted(incident.node_ids) if incident is not None else [],
        "incident_state": incident.state.value if incident is not None else None,
        "incident_workflow_id": (
            incident.workflow_request_id if incident is not None else None
        ),
        "status": workflow.status.value,
        "official_action": workflow.official_action,
        "blocked_reasons": list(workflow.blocked_reasons),
        "fencing_token": workflow.fencing_token,
        "execution_epoch": getattr(workflow, "execution_epoch", 0),
        "workflow_updated_at": workflow.updated_at.isoformat(),
        "source_plan_id": workflow.source_plan_id,
        "open_remote_commands": open_commands,
        "step_execution_count": len(workflow.step_executions),
        "already_closed": closed,
        "eligible": not closed and not reasons,
        "reasons": reasons,
    }


def build_compile_blocked_plan(
    store: Any,
    workflow_ids: Iterable[str],
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    evaluated_at = now or datetime.now(timezone.utc)
    request_ids = requested_workflow_ids(workflow_ids)
    commands = store.list_remote_commands(workflow_request_ids=request_ids)
    items = [_plan_item(store, request_id, commands) for request_id in request_ids]
    plan = {
        "schema_version": 1,
        "mode": PLAN_MODE,
        "evaluated_at": evaluated_at.isoformat(),
        "items": items,
    }
    plan["plan_sha256"] = _canonical_sha256(
        {
            "schema_version": plan["schema_version"],
            "mode": plan["mode"],
            "items": plan_digest_items(items),
        }
    )
    return plan


def closed_compile_blocked_record(
    workflow: WorkflowRequest,
    *,
    expected_fencing_token: int,
    expected_execution_epoch: int,
    reference: str,
    reconciled_at: datetime,
) -> WorkflowRequest:
    """The SUPERSEDED form of ``workflow``, or a refusal if it moved.

    Re-derived from a fresh read at apply time: the approval bound the fencing
    token and epoch the operator saw, so a record that was claimed or re-planned
    since is refused rather than closed under a stale reading.
    """

    if workflow.status is not WorkflowStatus.BLOCKED:
        raise ValueError(f"workflow is {workflow.status.value}, not BLOCKED")
    if workflow.fencing_token != expected_fencing_token:
        raise ValueError("workflow fencing token changed since the plan was approved")
    if getattr(workflow, "execution_epoch", 0) != expected_execution_epoch:
        raise ValueError("workflow execution epoch changed since the plan was approved")
    if workflow.step_executions or workflow.completed_operations:
        raise ValueError("workflow was dispatched since the plan was approved")
    return workflow.model_copy(
        update={
            "status": WorkflowStatus.SUPERSEDED,
            "preemption_reason": (
                f"operator reconciliation {reference}: {CLOSE_MARKER} that never "
                "changed a node; blocked_reasons="
                + json.dumps(list(workflow.blocked_reasons), sort_keys=True)
            ),
            "superseded_at": reconciled_at,
            "updated_at": reconciled_at,
        }
    )


def apply_compile_blocked_plan(
    store: Any,
    *,
    workflow_ids: Iterable[str],
    expected_plan_sha256: str,
    reference: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Apply an approved plan, one isolated write per item, always returning.

    The plan is rebuilt and its digest compared with the approval; the digest
    covers every field the verdict reads, so a match means the record handed to
    ``save_workflow`` is the one the operator reviewed. A record an earlier,
    partial apply already closed is skipped and named, so rerunning the same
    command is safe.
    """

    applied_at = now or datetime.now(timezone.utc)
    requested = requested_workflow_ids(workflow_ids)
    plan = build_compile_blocked_plan(store, requested, now=applied_at)
    if plan["plan_sha256"] != expected_plan_sha256:
        raise ValueError("compile-blocked reconcile plan changed before apply")
    blocked = [
        f"{item['request_id']}: " + "; ".join(item["reasons"])
        for item in plan["items"]
        if not item["eligible"] and not item["already_closed"]
    ]
    if blocked:
        raise ValueError(
            "compile-blocked reconcile plan contains ineligible records: "
            + " | ".join(blocked)
        )
    applied: list[str] = []
    skipped: list[str] = []
    failures: dict[str, str] = {}
    for item in plan["items"]:
        request_id = str(item["request_id"])
        if item["already_closed"]:
            skipped.append(request_id)
            continue
        try:
            workflow = store.get_workflow(request_id)
            closed = closed_compile_blocked_record(
                workflow,
                expected_fencing_token=int(item["fencing_token"]),
                expected_execution_epoch=int(item["execution_epoch"]),
                reference=reference,
                reconciled_at=applied_at,
            )
            store.save_workflow(closed)
        except Exception as exc:  # noqa: BLE001 -- per-item isolation, reported
            failures[request_id] = f"{type(exc).__name__}: {exc}"
            continue
        applied.append(request_id)
    return {
        "schema_version": 1,
        "mode": APPLY_MODE,
        "applied_at": applied_at.isoformat(),
        "reference": reference,
        "settled_plan_sha256": plan["plan_sha256"],
        "applied_workflow_ids": applied,
        "already_closed_workflow_ids": skipped,
        "failed_workflow_ids": sorted(failures),
        "failures": failures,
        "records_deleted": 0,
    }
