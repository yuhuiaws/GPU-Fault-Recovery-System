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

This module is the single copy of that decision, for three callers that cannot
share an import:

* ``WorkflowDispatcher._revoke_retired_generations`` -- the self-healing sweep.
* ``TransactionalWorkflowMixin.reconcile_retired_generation_workflow`` -- the
  audited operator write.
* ``gpu_fault.admin.workflow_reconcile`` -- which ships *this file's source* into
  the running Pod, because an operator has to be able to close such a record
  using the image that is already deployed, including the release that carries
  this module for the first time.

That third caller is why the imports below are deliberately narrow and
long-standing: ``models``, ``operation_registry``, ``remote_command_models`` and
``store`` only. Nothing here may import ``gpu_fault.workflow_resolution``,
``gpu_fault.execution`` or anything else that has changed in the same release,
or the shipped copy stops running against the deployed image.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, NamedTuple

from gpu_fault.models import (
    resolved_step_indexes,
    FaultIncident,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepStatus,
)
from gpu_fault.operation_registry import (
    CONTAINMENT_ONLY_OPERATIONS,
    DESTRUCTIVE_OPERATIONS,
    NODE_MUTATING_OPERATIONS,
)
from gpu_fault.remote_command_models import RemoteCommandStatus
from gpu_fault.store import NotFoundError

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
) -> tuple[WorkflowRequest, FaultIncident]:
    """Terminalize a retired generation, re-verifying every condition first.

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
# where a timestamp could not guard it anyway: ``_revoke_planned_item`` compares
# the fencing token inside the transaction. ``updated_at`` is still printed,
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


def _revoke_planned_item(
    store: Any,
    item: dict[str, Any],
    *,
    reference: str,
    applied_at: datetime,
    waiting_ttl: timedelta | None = None,
) -> tuple[str, str | None]:
    transactional = getattr(store, "reconcile_retired_generation_workflow", None)
    if transactional is not None:
        revoked, _ = transactional(
            item["request_id"],
            item["successor_workflow_id"],
            expected_fencing_token=int(item["fencing_token"]),
            reference=reference,
            reconciled_at=applied_at,
        )
    else:
        # The release that first carries this module has to be able to close a
        # retired generation *before* it deploys -- that record is exactly what
        # the release preflight refuses to roll past -- and the Store it talks to
        # is the previously deployed one, which has no transactional form of this
        # write. Resolved as a capability rather than by catching an
        # ``AttributeError``: a missing Store method is a static property of the
        # deployment, not a transient failure. The window this opens is one
        # re-read: ``retired_generation_records`` re-derives every condition from
        # rows read here, and the only concurrent writer is the dispatcher
        # renewing a lease whose fields this write clears anyway.
        workflow = store.get_workflow(item["request_id"])
        incident = store.get_incident(workflow.incident_id)
        successor = store.get_workflow(item["successor_workflow_id"])
        revoked, updated_incident = retired_generation_records(
            workflow,
            incident,
            successor,
            store.list_remote_commands(workflow_request_ids=[workflow.request_id]),
            expected_fencing_token=int(item["fencing_token"]),
            reference=reference,
            reconciled_at=applied_at,
        )
        store.save_workflow(revoked)
        store.save_incident(updated_incident)
    return revoked.request_id, _release_restart_reservations(
        store, revoked, waiting_ttl=waiting_ttl
    )


def _release_restart_reservations(
    store: Any,
    workflow: WorkflowRequest,
    *,
    waiting_ttl: timedelta | None = None,
) -> str | None:
    """Release restart reservations the revoked workflow will never attempt.

    Imported here rather than at module scope: ``gpu_fault.execution`` reaches
    this module through ``workflow_resolution``, so a top-level import would
    close an import cycle, and the shipped copy of this file must not depend on
    the execution package having the shape it has in this release.

    Which is also why a shortfall is returned rather than raised. This runs
    *after* the workflow row is already terminalized, so letting an import or
    attribute error out of here would hand the operator a traceback for a write
    that in fact succeeded -- at the one moment they have no other way to clear
    the release blocker, and with no way to tell from the output whether to run
    the command again. The cost of the degraded path is bounded and nameable: a
    restart reservation stays held, so a later legitimate restart of that job can
    be refused for budget. Reported, so the operator can release it deliberately
    rather than discover it as an unexplained refusal weeks later.
    """

    try:
        from gpu_fault.execution import restart_budget_preflight

        release = restart_budget_preflight.release_unattempted_restart_reservations
    except (ImportError, AttributeError) as exc:
        return (
            f"{workflow.request_id}: revoked, but restart reservations were not "
            f"released -- the deployed image cannot do it ({exc}). Release them "
            "with the release that carries this fix."
        )
    try:
        # Inside the same guard as the import, for the same reason: the release
        # reads and writes the Store, and a transient storage error here would
        # otherwise turn a committed revocation into a traceback (P1-72F).
        # ``waiting_ttl`` is passed only when given: the previously deployed
        # image's release may not know the keyword, and the default call has
        # to stay the one it accepts.
        if waiting_ttl is None:
            release(store, workflow)
        else:
            release(store, workflow, waiting_ttl=waiting_ttl)
    except Exception as exc:  # noqa: BLE001 -- reported to the operator, not raised
        return (
            f"{workflow.request_id}: revoked, but releasing its restart "
            f"reservations failed ({type(exc).__name__}: {exc}). Release them "
            "deliberately, or run the command again for this workflow."
        )
    return None


# Item fields the cancel pass is allowed to change between the approved plan and
# the settled re-plan. Everything else must read back identical, or the second
# pass is describing a record the operator never approved (P1-72D).
SECOND_PASS_MUTABLE_FIELDS = frozenset(
    {
        "open_remote_commands",
        "cancellable",
        "eligible",
        "reasons",
        "blocker_codes",
        "workflow_updated_at",
    }
)


def second_pass_drift(
    approved: dict[str, Any],
    settled: dict[str, Any],
) -> list[str]:
    """Field-level differences the cancel pass cannot account for.

    Cancelling commands may only empty ``open_remote_commands`` and drop the one
    blocker that named them; a change to any other field -- the generation, the
    step list's destructive content, the successor, the incident's state -- names
    the field and both values so the operator can tell a re-plan from a tick.
    """

    request_id = str(approved.get("request_id"))
    drift: list[str] = []
    for field in sorted(set(approved) | set(settled)):
        if field in SECOND_PASS_MUTABLE_FIELDS:
            continue
        before, after = approved.get(field), settled.get(field)
        if before != after:
            drift.append(f"{request_id}: {field} {before!r} -> {after!r}")
    still_open = set(settled.get("open_remote_commands") or [])
    if not still_open <= set(approved.get("open_remote_commands") or []):
        drift.append(
            f"{request_id}: open_remote_commands gained "
            f"{sorted(still_open - set(approved.get('open_remote_commands') or []))}"
        )
    remaining = [
        code
        for code in settled.get("blocker_codes") or []
        if code != OPEN_REMOTE_COMMANDS_CODE
    ]
    approved_codes = [
        code
        for code in approved.get("blocker_codes") or []
        if code != OPEN_REMOTE_COMMANDS_CODE
    ]
    if remaining != approved_codes:
        drift.append(f"{request_id}: blocker_codes {approved_codes!r} -> {remaining!r}")
    return drift


def apply_retired_generation_plan(
    store: Any,
    *,
    workflow_ids: Iterable[str],
    expected_plan_sha256: str,
    reference: str,
    now: datetime | None = None,
    waiting_ttl: timedelta | None = None,
) -> dict[str, Any]:
    """Cancel the open commands of a retired generation, then terminalize it.

    Both passes happen here rather than asking the operator to run the command
    twice, because cancelling a ``PENDING`` or ``WAITING`` command settles it in
    the same Store call: the second plan below usually finds nothing left to
    wait for. A ``LEASED`` command does not settle -- only the executor holding
    the lease can report it -- so that case stops with the reasons printed
    instead of revoking a workflow whose effect is still in flight.

    The approval binds the write. The compare-and-set value and the successor
    handed to the Store come from the plan whose digest the operator approved,
    and the settled re-plan is compared with it field by field: the cancel pass
    may only have emptied ``open_remote_commands``. Each revocation is isolated,
    so one failing row leaves a result naming what was applied, what failed and
    why; a rerun finds the applied rows as ``already_revoked`` and skips them.

    ``waiting_ttl`` is how long a ``RESTART_WORKLOAD`` may have been WAITING
    before its restart reservation counts as never used (F-C9). This module has
    no executor config to read the cap from, so the caller supplies it; without
    it a WAITING record keeps its reservation, as before.
    """

    applied_at = now or datetime.now(timezone.utc)
    requested = requested_workflow_ids(workflow_ids)
    plan = build_retired_generation_plan(store, requested, now=applied_at)
    if plan["plan_sha256"] != expected_plan_sha256:
        raise ValueError("retired generation reconcile plan changed before apply")
    if not plan["items"]:
        raise ValueError("retired generation reconcile plan has no workflows")
    skipped = [item for item in plan["items"] if item["already_revoked"]]
    blocked = [
        f"{item['request_id']}: " + "; ".join(item["reasons"])
        for item in plan["items"]
        if not item["eligible"]
        and not item["cancellable"]
        and not item["already_revoked"]
    ]
    if blocked:
        raise ValueError(
            "retired generation reconcile plan contains ineligible records: "
            + " | ".join(blocked)
        )
    cancelled: dict[str, dict[str, int]] = {}
    for item in plan["items"]:
        if not item["cancellable"]:
            continue
        cancelled[item["request_id"]] = store.cancel_remote_commands_for_workflow(
            item["request_id"],
            reason=(
                f"operator reconciliation {reference}: cancelled before revoking "
                f"retired generation {item['request_id']} at generation "
                f"{item['fencing_token']}"
            ),
        )
    settled = build_retired_generation_plan(store, requested, now=applied_at)
    settled_by_id = {str(item["request_id"]): item for item in settled["items"]}
    drift = [
        line
        for item in plan["items"]
        if not item["already_revoked"]
        for line in second_pass_drift(item, settled_by_id.get(item["request_id"], {}))
    ]
    if drift:
        raise ValueError(
            "retired generation reconcile plan changed between cancelling remote "
            "commands and revoking, nothing was revoked: " + " | ".join(drift)
        )
    unsettled = [
        f"{item['request_id']}: " + "; ".join(item["reasons"])
        for item in settled["items"]
        if not item["eligible"] and not item["already_revoked"]
    ]
    if unsettled:
        raise ValueError(
            "retired generation reconcile stopped after cancelling remote "
            "commands, records are not settled yet: " + " | ".join(unsettled)
        )
    applied: list[str] = []
    failures: dict[str, str] = {}
    warnings: list[str] = []
    for item in plan["items"]:
        if item["already_revoked"]:
            continue
        try:
            request_id, warning = _revoke_planned_item(
                store,
                item,
                reference=reference,
                applied_at=applied_at,
                waiting_ttl=waiting_ttl,
            )
        except Exception as exc:  # noqa: BLE001 -- per-item isolation, reported
            # The rows already revoked above are committed; the operator has to
            # see them in the result, not lose them to a traceback (P1-60F).
            failures[str(item["request_id"])] = f"{type(exc).__name__}: {exc}"
            continue
        applied.append(request_id)
        if warning is not None:
            warnings.append(warning)
    return {
        "schema_version": 1,
        "mode": "retired-generation-apply",
        "plan_sha256": expected_plan_sha256,
        "settled_plan_sha256": settled["plan_sha256"],
        "reference": reference,
        "applied_at": applied_at.isoformat(),
        "applied_workflow_ids": sorted(applied),
        "already_revoked_workflow_ids": sorted(
            str(item["request_id"]) for item in skipped
        ),
        "failed_workflow_ids": sorted(failures),
        "failures": dict(sorted(failures.items())),
        "restart_reservation_warnings": sorted(warnings),
        "cancelled_remote_commands": {
            request_id: {
                "cancelled": int(counts.get("cancelled", 0)),
                "cancellation_requested": int(counts.get("cancellation_requested", 0)),
            }
            for request_id, counts in sorted(cancelled.items())
        },
        "archive_eligible_incident_ids": sorted(
            {
                str(item["incident_id"])
                for item in settled["items"]
                if item.get("incident_id")
            }
        ),
        "records_deleted": 0,
    }
