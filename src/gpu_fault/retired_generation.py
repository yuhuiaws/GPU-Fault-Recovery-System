"""Decide when an incident has retired one of its own workflow generations.

An incident points at exactly one workflow. When a family re-plans, it bumps the
incident's fencing token and moves that pointer, and the workflow it moved away
from is left holding everything a live workflow holds: an execution owner, a
renewing lease, remediation budget claims and possibly an unsettled destructive
remote command. Usually the re-plan also links the two records, and the executor
supersedes the predecessor at a step boundary. The re-plan paths that do not
link them leave a record that nothing will ever close, and because the
dispatcher admits one workflow per incident, that record also starves the
successor its incident is now waiting on.

This module is the single copy of that decision, for callers that cannot share
an import:

* ``WorkflowDispatcher._revoke_retired_generations`` -- the self-healing sweep,
  which is the only path that revokes a retired generation now.
* ``TransactionalWorkflowMixin.reconcile_retired_generation_workflow`` -- the
  audited Store write the sweep goes through.
* ``build_retired_generation_plan`` -- the read-only report the DESTR-017
  acceptance runner records, and the digest rule (``plan_digest_items``) the
  administrator's ``workflow-reconcile`` shares.

The administrator's retired-generation mode, which shipped this file's source
into the running Pod, was folded into the dispatcher sweep; the imports below
stay narrow (``models``, ``operation_registry``, ``remote_command_models`` and
``store`` only) so nothing here pulls ``gpu_fault.execution`` back into the
store layer through ``workflow_resolution``.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, NamedTuple

import gpu_fault.models as _models
from gpu_fault.models import (
    FaultIncident,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepStatus,
    resolved_step_indexes,
)
from gpu_fault.operation_registry import (
    CONTAINMENT_ONLY_OPERATIONS,
    DESTRUCTIVE_OPERATIONS,
    NODE_MUTATING_OPERATIONS,
)
from gpu_fault.remote_command_models import RemoteCommandStatus
from gpu_fault.store import NotFoundError

# The attributed audit event (I1) is resolved as a capability of the deployed
# ``gpu_fault.models``, not imported by name: this file's source is shipped
# into the running Pod, and the image there may predate the event kind. When it
# does, the revocation still happens and the string audit below is all there is.
_RECORD_OPERATOR_EVENT: Any = getattr(_models, "record_operator_event", None)
_OPERATOR_RETIRED_GENERATION: Any = getattr(
    getattr(_models, "WorkflowEventKind", None), "OPERATOR_RETIRED_GENERATION", None
)

OPEN_REMOTE_STATUSES = {
    RemoteCommandStatus.PENDING,
    RemoteCommandStatus.WAITING,
    RemoteCommandStatus.LEASED,
}
RETIRED_GENERATION_STATUSES = {
    WorkflowStatus.PENDING,
    WorkflowStatus.SAFETY_PENDING,
    WorkflowStatus.RUNNING,
    WorkflowStatus.BLOCKED,
}
# How many open workflows one discovery pass reads before it stops and says so.
# The candidate status set is every open workflow in the fleet, so the old
# ``> 1000 -> refuse`` ceiling made discovery unavailable on any busy fleet --
# and the only way to learn the ids to pass explicitly was the discovery that
# had just refused (P1-72G). A truncated scan is reported, never raised.
DISCOVERY_SCAN_LIMIT = 10_000

# Blocker codes. ``cancellable`` used to be decided by matching the *prose* of
# a reason ("workflow has open remote ..."), so any rewording silently turned
# the two-pass apply into "everything needs an operator" (P2-72H). The code is
# the contract; the message is for the operator.
STATUS_NOT_OPEN_CODE = "status_not_open"
UNSETTLED_LOCAL_STEPS_CODE = "unsettled_local_steps"
COMPLETED_NODE_MUTATING_CODE = "completed_destructive_operations"
OPEN_REMOTE_COMMANDS_CODE = "open_remote_commands"
GENERATION_NOT_RETIRED_CODE = "generation_not_retired"
ALREADY_REVOKED_CODE = "already_revoked"


class RetiredGenerationBlocker(NamedTuple):
    """One reason a retired generation cannot be revoked, as code and prose."""

    code: str
    message: str


def retired_generation_successor(
    store: Any,
    workflow: WorkflowRequest,
) -> WorkflowRequest | None:
    """The workflow the incident moved to, if it retired ``workflow``'s generation.

    This is the *pointer* half of the test and says nothing about whether
    ``workflow`` is safe to close -- ``retired_generation_reasons`` answers that.
    It answers one question: has the incident replaced this workflow with a
    strictly later generation of itself?

    A legitimate preemption does not look like this. ``families/faults.py`` gives
    the successor the predecessor's token unchanged, and ``grouped_health.py``
    reuses the predecessor's ``request_id`` when it replaces steps in place, so
    in both shapes the strict inequality below is false. Only a re-plan that
    bumped the generation *and* minted a new record *without* linking the old one
    lands here.

    The last condition is what keeps the third shape out. When the successor
    carries ``predecessor_workflow_id`` pointing back at ``workflow``, somebody is
    already responsible for that record and it must be left alone -- with
    ``preempt_predecessor`` the executor supersedes it at a step boundary, and
    only that path compensates an unrestored ``QUIESCE_GPU_SERVICES``; without
    it the successor is queued *behind* the predecessor and the predecessor is
    meant to run to completion. Revoking either shape here would drop real work,
    and refusing to dispatch either one would deadlock the pair.
    """

    try:
        incident = store.get_incident(workflow.incident_id)
    except (KeyError, NotFoundError):
        return None
    successor_id = incident.workflow_request_id
    if (
        not successor_id
        or successor_id == workflow.request_id
        or workflow.fencing_token >= incident.fencing_token
    ):
        return None
    try:
        successor: WorkflowRequest = store.get_workflow(successor_id)
    except (KeyError, NotFoundError):
        return None
    if (
        successor.incident_id != workflow.incident_id
        or successor.fencing_token != incident.fencing_token
        or successor.predecessor_workflow_id == workflow.request_id
    ):
        return None
    return successor


def open_remote_command_ids(
    workflow: WorkflowRequest,
    remote_commands: Iterable[Any],
) -> list[str]:
    return sorted(
        item.command_id
        for item in remote_commands
        if item.workflow_request_id == workflow.request_id
        and item.status in OPEN_REMOTE_STATUSES
    )


def unsettled_local_step_indexes(workflow: WorkflowRequest) -> list[int]:
    """Steps handed to a non-remote adapter that never reported back.

    ``WorkflowStepStatus`` has no "running" value, so a ``WAITING`` execution is
    the only shape in-flight work takes. When it is remote-backed its
    ``adapter_operation_id`` names the command, and the command's own status --
    checked separately, and cancellable -- is the authority on it. When it is not,
    nothing in the Store says whether the adapter's action landed, which is the
    same "unknown provider action" the restore reconcile refuses on. The executor
    reads it the same way in ``_supersede_if_safe``: a remote ``WAITING`` step may
    be cancelled, a local one blocks preemption outright.
    """

    return sorted(
        item.step_index
        for item in workflow.step_executions
        if item.status is WorkflowStepStatus.WAITING
        and not str(item.adapter_operation_id or "").startswith("remote/")
    )


def completed_destructive_operations(workflow: WorkflowRequest) -> list[str]:
    """Node-mutating operations the workflow has actually carried out.

    Judged against ``NODE_MUTATING_OPERATIONS``, not ``DESTRUCTIVE_OPERATIONS``
    (F-B4 (4)). A cordon or a quarantine taint is destructive in the audit sense
    but it changes only what the scheduler may place on the node; the successor
    generation of the same incident owns the isolation and lifts it with its own
    ``RESTORE_SCHEDULING``. Refusing to revoke a record because it *isolated* a
    node kept the retired generation -- and the starvation it causes -- alive for
    the sake of a step the successor was going to redo anyway.
    """

    return sorted(
        operation.value
        for operation in set(workflow.completed_operations) & NODE_MUTATING_OPERATIONS
    )


def completed_containment_operations(workflow: WorkflowRequest) -> list[str]:
    """Isolation the workflow left in place. Reported, not a blocker."""

    return sorted(
        operation.value
        for operation in set(workflow.completed_operations)
        & CONTAINMENT_ONLY_OPERATIONS
    )


def pending_destructive_operations(workflow: WorkflowRequest) -> list[str]:
    """What revoking this record prevents. Evidence only -- it gates nothing.

    Named separately from ``completed_destructive_operations`` because the two
    read almost the same and mean opposite things: a *completed* destructive
    operation forbids revocation, while a *pending* one is the reason to revoke.
    """

    # The step set follows ``executes_safety_steps`` -- the explicit
    # ``safety_only`` flag, or SAFETY_PENDING for records written before the
    # flag existed -- and never ``bool(blocked_reasons)``: a warning appended
    # to that list must not flip which steps count (F-B4 (4)). A superseded step
    # is not pending (F-C2).
    steps = (
        workflow.safety_steps
        if workflow.executes_safety_steps
        else workflow.official_steps
    )
    resolved = resolved_step_indexes(workflow)
    return sorted(
        {
            step.operation.value
            for index, step in enumerate(steps)
            if index not in resolved and step.operation in DESTRUCTIVE_OPERATIONS
        }
    )


def retired_generation_reasons(
    workflow: WorkflowRequest,
    successor: WorkflowRequest | None,
    remote_commands: Iterable[Any],
) -> list[str]:
    """Why ``workflow`` cannot be revoked as a retired generation, if it cannot.

    Deliberately not a lease or owner test. A retired generation may hold both --
    on 2026-09-04 one held an execution owner, a renewing lease and six
    remediation budget claims -- and breaking a lease is the whole point of
    revoking it. What it must not hold is *effect*:

    * Its status is still open. Anything terminal needs nothing.
    * It has completed no node-mutating operation. This is the load-bearing
      condition. ``completed_operations`` is what the workflow actually did to
      the fleet, and every member of ``NODE_MUTATING_OPERATIONS`` -- a restart,
      a stopped workload, ``QUIESCE_GPU_SERVICES`` -- is something a later
      workflow would have to compensate for, so such a record is not revocable
      by supersession and must reach an operator. Containment-only operations
      (a cordon, a quarantine taint) do not count (F-B4 (4)): they change what
      the scheduler may place, not the node, and the successor generation's
      ``RESTORE_SCHEDULING`` lifts them; they are reported separately.
    * Every remote command it owns is settled. An open command is effect that
      has not landed yet, and it outlives the workflow row: closing the workflow
      around it would leave a destructive command that the next fence change
      releases onto real nodes. Callers cancel first and revoke once the store
      shows the commands terminal.
    * No step is ``WAITING`` on a local adapter. See
      ``unsettled_local_step_indexes``: that is in-flight effect no Store row can
      account for, so unlike an open remote command it cannot be cancelled and
      has to reach an operator.

    Reasons are returned rather than a bool so the operator plan can print
    exactly which of them held. ``retired_generation_blockers`` is the
    structured form; this keeps the prose list for callers that only print it.
    """

    return [
        blocker.message
        for blocker in retired_generation_blockers(workflow, successor, remote_commands)
    ]


def retired_generation_blockers(
    workflow: WorkflowRequest,
    successor: WorkflowRequest | None,
    remote_commands: Iterable[Any],
) -> list[RetiredGenerationBlocker]:
    """``retired_generation_reasons`` with a machine-readable code per reason."""

    blockers: list[RetiredGenerationBlocker] = []
    if workflow.status not in RETIRED_GENERATION_STATUSES:
        blockers.append(
            RetiredGenerationBlocker(
                STATUS_NOT_OPEN_CODE,
                f"workflow status is {workflow.status.value}, not open",
            )
        )
    unsettled = unsettled_local_step_indexes(workflow)
    if unsettled:
        blockers.append(
            RetiredGenerationBlocker(
                UNSETTLED_LOCAL_STEPS_CODE,
                "workflow has unsettled adapter actions on steps: "
                + ", ".join(str(index) for index in unsettled),
            )
        )
    executed = completed_destructive_operations(workflow)
    if executed:
        blockers.append(
            RetiredGenerationBlocker(
                COMPLETED_NODE_MUTATING_CODE,
                "workflow already completed destructive operations: "
                + ", ".join(executed),
            )
        )
    open_commands = open_remote_command_ids(workflow, remote_commands)
    if open_commands:
        blockers.append(
            RetiredGenerationBlocker(
                OPEN_REMOTE_COMMANDS_CODE,
                "workflow has open remote commands: " + ", ".join(open_commands),
            )
        )
    if successor is None:
        blockers.append(
            RetiredGenerationBlocker(
                GENERATION_NOT_RETIRED_CODE,
                "workflow generation has not been retired by its incident",
            )
        )
    return blockers


def already_revoked(
    workflow: WorkflowRequest, successor: WorkflowRequest | None
) -> bool:
    """Whether an earlier apply already closed ``workflow`` in favour of ``successor``.

    A partial apply leaves the records it did revoke SUPERSEDED, and the natural
    next move is to run the same command again. That rerun used to refuse the
    whole plan because those records were "not open" (P1-60F); they are the
    finished half of this very job, so the plan names them and the apply skips
    them.
    """

    return (
        workflow.status is WorkflowStatus.SUPERSEDED
        and successor is not None
        and workflow.preempted_by_workflow_id == successor.request_id
    )


def retired_generation_audit(
    workflow: WorkflowRequest,
    incident: FaultIncident,
    successor: WorkflowRequest,
    *,
    reference: str | None = None,
) -> str:
    prefix = "" if reference is None else f"operator reconciliation {reference}: "
    return (
        f"{prefix}revoked retired generation {workflow.request_id} at generation "
        f"{workflow.fencing_token} in favour of {successor.request_id}: incident "
        f"{incident.incident_id} advanced to generation {incident.fencing_token}"
    )


def retired_generation_records(
    workflow: WorkflowRequest,
    incident: FaultIncident,
    successor: WorkflowRequest,
    remote_commands: Iterable[Any],
    *,
    expected_fencing_token: int,
    reference: str | None,
    reconciled_at: datetime,
    actor: str | None = None,
    approval: Mapping[str, Any] | None = None,
) -> tuple[WorkflowRequest, FaultIncident]:
    """Terminalize a retired generation, re-verifying every condition first.

    An operator write (one with a ``reference``) also appends an
    ``OPERATOR_RETIRED_GENERATION`` event naming ``actor`` (the STS ARN), the
    reference, the plan digests in ``approval`` and the status transition (I1),
    when the running ``gpu_fault.models`` has the event kind. The dispatcher's
    sweep passes no reference and leaves no operator event.

    Every caller re-derives the verdict here rather than trusting the one it
    computed earlier, so a record that became live between plan and apply is
    rejected instead of revoked. The safety of this write comes from that
    re-derivation inside the caller's transaction, not from holding the
    workflow's execution lease -- which the record's own dispatcher is still
    renewing, and which is released here on purpose.

    ``blocked_reasons`` is deliberately not appended to. It selects which step
    set later readers see (``safety_steps`` when it is non-empty), so an audit
    line there would silently change what
    ``release_unattempted_restart_reservations`` releases. The audit goes on the
    incident and in ``preemption_reason``.
    """

    reasons = retired_generation_reasons(workflow, successor, remote_commands)
    if workflow.fencing_token != expected_fencing_token:
        reasons.append(
            "workflow fencing token changed: expected "
            f"{expected_fencing_token}, found {workflow.fencing_token}"
        )
    if incident.workflow_request_id != successor.request_id:
        reasons.append("incident no longer names the successor")
    if successor.incident_id != workflow.incident_id:
        reasons.append("successor belongs to another incident")
    if successor.fencing_token != incident.fencing_token:
        reasons.append("successor is not at the incident generation")
    if workflow.fencing_token >= incident.fencing_token:
        reasons.append("workflow generation is not behind its incident")
    if reasons:
        raise ValueError("retired generation reconcile rejected: " + "; ".join(reasons))
    audit = retired_generation_audit(
        workflow,
        incident,
        successor,
        reference=reference,
    )
    updated_workflow = workflow.model_copy(
        update={
            "status": WorkflowStatus.SUPERSEDED,
            "preempted_by_workflow_id": successor.request_id,
            "preemption_reason": audit,
            "superseded_at": reconciled_at,
            "execution_owner_id": None,
            "execution_lease_expires_at": None,
            "updated_at": reconciled_at,
        }
    )
    if (
        reference is not None
        and _RECORD_OPERATOR_EVENT is not None
        and _OPERATOR_RETIRED_GENERATION is not None
    ):
        updated_workflow = _RECORD_OPERATOR_EVENT(
            updated_workflow,
            _OPERATOR_RETIRED_GENERATION,
            actor=actor,
            reference=reference,
            previous_status=workflow.status,
            at=reconciled_at,
            details={
                "successor_workflow_id": successor.request_id,
                "expected_fencing_token": expected_fencing_token,
                "incident_fencing_token": incident.fencing_token,
                **dict(approval or {}),
            },
        )
    updated_incident = incident.model_copy(
        update={
            "reasons": list(dict.fromkeys([*incident.reasons, audit])),
            "updated_at": reconciled_at,
        }
    )
    return updated_workflow, updated_incident


def retired_generation_plan_item(
    store: Any,
    request_id: str,
    remote_commands: Iterable[Any],
) -> dict[str, Any]:
    """Evidence for one candidate, eligible or not.

    ``cancellable`` is the two-pass hinge: open remote commands are the one
    blocker an apply may clear on its own, because cancelling a command of a
    retired generation is correct under every reading. Any other reason means
    the record needs an operator, not a retry. It is decided by blocker *code*,
    never by the wording of a reason (P2-72H).
    """

    try:
        workflow = store.get_workflow(request_id)
    except (KeyError, NotFoundError):
        return {
            "request_id": request_id,
            "eligible": False,
            "cancellable": False,
            "already_revoked": False,
            "blocker_codes": ["missing"],
            "reasons": ["workflow does not exist"],
        }
    commands = list(remote_commands)
    successor = retired_generation_successor(store, workflow)
    try:
        incident: FaultIncident | None = store.get_incident(workflow.incident_id)
    except (KeyError, NotFoundError):
        incident = None
    blockers = retired_generation_blockers(workflow, successor, commands)
    codes = [blocker.code for blocker in blockers]
    open_commands = open_remote_command_ids(workflow, commands)
    return {
        "request_id": request_id,
        "incident_id": workflow.incident_id,
        "cluster_id": incident.cluster_id if incident is not None else None,
        "node_ids": sorted(incident.node_ids) if incident is not None else [],
        "incident_state": incident.state.value if incident is not None else None,
        "incident_fencing_token": (
            incident.fencing_token if incident is not None else None
        ),
        "workflow_status": workflow.status.value,
        "fencing_token": workflow.fencing_token,
        "workflow_updated_at": workflow.updated_at.isoformat(),
        "successor_workflow_id": (
            successor.request_id if successor is not None else None
        ),
        "successor_fencing_token": (
            successor.fencing_token if successor is not None else None
        ),
        "completed_destructive_operations": completed_destructive_operations(workflow),
        "completed_containment_operations": completed_containment_operations(workflow),
        "pending_destructive_operations": pending_destructive_operations(workflow),
        "unsettled_local_steps": unsettled_local_step_indexes(workflow),
        "remediation_budget_claims": sorted(workflow.remediation_budget_claims),
        "open_remote_commands": open_commands,
        "eligible": not blockers,
        "cancellable": bool(open_commands)
        and all(code == OPEN_REMOTE_COMMANDS_CODE for code in codes),
        "already_revoked": already_revoked(workflow, successor),
        "blocker_codes": codes,
        "reasons": [blocker.message for blocker in blockers],
    }


def discover_open_workflows(
    store: Any,
    statuses: set[WorkflowStatus],
    *,
    scan_limit: int = DISCOVERY_SCAN_LIMIT,
) -> tuple[list[WorkflowRequest], bool]:
    """Up to ``scan_limit`` open workflows, oldest first, and whether more exist.

    One bounded read rather than a paginated walk. ``list_workflows`` only has
    a keyset cursor (``after``) in its ``dispatchable_at`` mode, and that mode
    pushes the dispatcher's filters below the LIMIT: rows behind an open
    predecessor or a future ``not_before`` are dropped in the store -- exactly
    the rows a discovery must not lose. Outside that mode there is no cursor,
    and paging through ``exclude_request_ids`` is not neutral on the memory and
    SQLite backends either. The bound is reported to the operator as
    ``scan_truncated`` instead of being raised.
    """

    workflows = store.list_workflows(statuses=statuses, limit=scan_limit + 1)
    return list(workflows[:scan_limit]), len(workflows) > scan_limit


def discovery_report(
    *, scanned: int, selected: int, candidates: int, scan_truncated: bool
) -> dict[str, Any]:
    return {
        "scanned": scanned,
        "selected": selected,
        "remaining": max(0, candidates - selected),
        "scan_truncated": scan_truncated,
    }


def requested_workflow_ids(requested: Iterable[str], *, limit: int = 1000) -> list[str]:
    values = sorted({str(item).strip() for item in requested if str(item).strip()})
    if len(values) > limit:
        raise ValueError(f"reconcile accepts at most {limit} workflow IDs")
    return values


def retired_generation_discovery(
    store: Any,
    requested: Iterable[str] | None,
    *,
    max_items: int | None = None,
    scan_limit: int = DISCOVERY_SCAN_LIMIT,
) -> tuple[list[str], dict[str, Any] | None]:
    """The candidate ids, and a discovery report when they were discovered.

    Explicit ids are taken as given (the report is ``None``). Discovery scans the
    open workflows once, keeps the ones whose incident has retired them, and
    returns the oldest ``max_items`` of those so repeated batches walk the
    backlog; the report says how many were scanned and how many candidates the
    batch left behind.
    """

    if requested:
        return requested_workflow_ids(requested), None
    workflows, truncated = discover_open_workflows(
        store, RETIRED_GENERATION_STATUSES, scan_limit=scan_limit
    )
    candidates = [
        item.request_id
        for item in workflows
        if retired_generation_successor(store, item) is not None
    ]
    selected = candidates if max_items is None else candidates[:max_items]
    return sorted(selected), discovery_report(
        scanned=len(workflows),
        selected=len(selected),
        candidates=len(candidates),
        scan_truncated=truncated,
    )


def retired_generation_candidates(
    store: Any,
    requested: Iterable[str] | None,
) -> list[str]:
    return retired_generation_discovery(store, requested)[0]


def retired_generation_plan_items(
    store: Any,
    workflow_ids: Iterable[str] | None,
    *,
    max_items: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    request_ids, report = retired_generation_discovery(
        store, workflow_ids, max_items=max_items
    )
    if not request_ids:
        return [], report
    # Scoped to the workflows under reconciliation rather than the whole command
    # history: every use filters on ``workflow_request_id``, and the candidate
    # list is bounded, so the narrowed read answers the same question against a
    # bounded number of rows.
    commands = store.list_remote_commands(workflow_request_ids=request_ids)
    return [
        retired_generation_plan_item(store, request_id, commands)
        for request_id in request_ids
    ], report


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


# Reported for the operator to read, but kept out of the plan digest.
#
# A retired generation is being dispatched on a loop -- that is the whole
# complaint against it -- and every tick renews a lease and stamps
# ``updated_at``. Including it made the digest change roughly once a minute, so
# the plan an operator had just reviewed could never be applied: on 2026-09-04
# the apply refused ``workflow-45c6b6b7`` with "plan changed before apply" when
# the only difference between the two evaluations was 19:25:59 -> 19:26:54.
#
# Dropping it costs nothing, because it never carried a decision. Everything the
# revocation depends on -- status, both fencing tokens, incident state, the
# successor's identity and generation, completed and pending destructive
# operations, unsettled local steps, budget claims, open commands -- stays in the
# digest, so any material change still refuses. And the write itself is guarded
# where a timestamp could not guard it anyway: the Store's retired-generation
# reconcile compares the fencing token inside the transaction. ``updated_at`` is
# still printed,
# because "this record was touched seconds ago" is exactly what tells an operator
# the wedge is live rather than historical.
DIGEST_EXCLUDED_ITEM_FIELDS = frozenset({"workflow_updated_at"})


def plan_digest_items(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """The plan items reduced to the fields the operator's approval binds."""

    return [
        {
            key: value
            for key, value in item.items()
            if key not in DIGEST_EXCLUDED_ITEM_FIELDS
        }
        for item in items
    ]


def build_retired_generation_plan(
    store: Any,
    workflow_ids: Iterable[str] | None = None,
    *,
    max_items: int | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    evaluated_at = now or datetime.now(timezone.utc)
    items, report = retired_generation_plan_items(
        store, workflow_ids, max_items=max_items
    )
    plan = {
        "schema_version": 1,
        "mode": "retired-generation-plan",
        "evaluated_at": evaluated_at.isoformat(),
        # Outside the digest on purpose: the backlog behind this batch moves
        # while the operator reads, and the approval binds the batch, not it.
        "discovery": report,
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
