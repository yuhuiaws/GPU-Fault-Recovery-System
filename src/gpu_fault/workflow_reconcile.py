from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

from gpu_fault.models import (
    BlockedKind,
    FaultIncident,
    RecoveryPlan,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepStatus,
)
from gpu_fault.remote_command_models import RemoteCommandStatus
from gpu_fault.retired_generation import (
    DISCOVERY_SCAN_LIMIT,
    discover_open_workflows,
    discovery_report,
    plan_digest_items,
    requested_workflow_ids,
)
from gpu_fault.store import NotFoundError
from gpu_fault.store.contracts import ControlPlaneStore
from gpu_fault.workflow_resolution import (
    completed_containment_operations,
    reconciled_restore_records,
    restore_reconciliation_reasons,
    verified_restore_successor,
    workflow_never_changed_a_node,
)
from gpu_fault.execution.restart_budget_preflight import (
    release_unattempted_restart_reservations,
)

# How an eligible item will be terminalized. Both end SUPERSEDED; the first
# names the successor that restored the node, the second has none to name.
VERIFIED_RESTORE = "verified-restore"
NEVER_CHANGED = "never-changed"


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _blocked_kind_values(blocked_kinds: Iterable[BlockedKind | str] | None) -> set[str]:
    return {
        item.value if isinstance(item, BlockedKind) else str(item).strip()
        for item in blocked_kinds or ()
    }


def _discover_blocked(
    store: ControlPlaneStore,
    *,
    incident_ids: Iterable[str] | None,
    blocked_kinds: Iterable[BlockedKind | str] | None,
    max_items: int | None,
    scan_limit: int,
) -> tuple[list[str], dict[str, Any]]:
    """The BLOCKED backlog, batched by incident and by kind, oldest first.

    Discovery used to refuse outright past a thousand BLOCKED rows -- while the
    only way to learn which ids to pass explicitly was this same discovery
    (P1-58G). It now scans a bounded window, applies the operator's batch
    filters, and reports what the batch left behind instead of raising.
    """

    workflows, truncated = discover_open_workflows(
        store, {WorkflowStatus.BLOCKED}, scan_limit=scan_limit
    )
    wanted_incidents = {str(item).strip() for item in incident_ids or () if item}
    wanted_kinds = _blocked_kind_values(blocked_kinds)
    candidates = [
        item.request_id
        for item in workflows
        if (not wanted_incidents or item.incident_id in wanted_incidents)
        and (
            not wanted_kinds
            or (
                item.blocked_kind is not None
                and item.blocked_kind.value in wanted_kinds
            )
        )
    ]
    selected = candidates if max_items is None else candidates[:max_items]
    return sorted(selected), discovery_report(
        scanned=len(workflows),
        selected=len(selected),
        candidates=len(candidates),
        scan_truncated=truncated,
    )


def _plan_item(
    store: ControlPlaneStore,
    request_id: str,
    commands: list[Any],
    *,
    evaluated_at: datetime,
) -> dict[str, Any]:
    try:
        workflow: WorkflowRequest = store.get_workflow(request_id)
    except (KeyError, NotFoundError):
        return {
            "request_id": request_id,
            "eligible": False,
            "reasons": ["workflow does not exist"],
        }
    open_commands = sorted(
        item.command_id
        for item in commands
        if item.workflow_request_id == request_id
        and item.status
        in {
            RemoteCommandStatus.PENDING,
            RemoteCommandStatus.WAITING,
            RemoteCommandStatus.LEASED,
        }
    )
    waiting_steps = sorted(
        item.step_index
        for item in workflow.step_executions
        if item.status is WorkflowStepStatus.WAITING
    )
    successor = verified_restore_successor(store, workflow)
    source_plan: RecoveryPlan | None = None
    if workflow.source_plan_id:
        try:
            source_plan = store.get_plan(workflow.source_plan_id)
        except (KeyError, NotFoundError):
            source_plan = None
    incident: FaultIncident | None
    try:
        incident = store.get_incident(workflow.incident_id)
    except (KeyError, NotFoundError):
        incident = None
    reasons = restore_reconciliation_reasons(
        workflow,
        incident,
        successor,
        source_plan,
        commands,
        evaluated_at=evaluated_at,
    )
    return {
        "request_id": request_id,
        "incident_id": workflow.incident_id,
        "cluster_id": incident.cluster_id if incident is not None else None,
        "node_ids": sorted(incident.node_ids) if incident is not None else [],
        "incident_state": incident.state.value if incident is not None else None,
        "blocked_kind": (
            workflow.blocked_kind.value if workflow.blocked_kind is not None else None
        ),
        "fencing_token": workflow.fencing_token,
        # A claim moves the epoch and a re-plan moves the token; a merge moves
        # neither. Both are in the digest, so an approval binds exactly the
        # record it described (P0-72A (2)).
        "execution_epoch": workflow.execution_epoch,
        "workflow_updated_at": workflow.updated_at.isoformat(),
        "successor_workflow_id": (
            successor.request_id if successor is not None else None
        ),
        "source_plan_id": workflow.source_plan_id,
        "source_plan_status": (
            source_plan.status.value if source_plan is not None else None
        ),
        "open_remote_commands": open_commands,
        "waiting_step_indexes": waiting_steps,
        "never_changed_a_node": workflow_never_changed_a_node(workflow),
        "completed_containment_operations": completed_containment_operations(workflow),
        "terminalization": (
            VERIFIED_RESTORE if successor is not None else NEVER_CHANGED
        ),
        "eligible": not reasons,
        "reasons": reasons,
    }


def build_workflow_reconcile_plan(
    store: ControlPlaneStore,
    workflow_ids: Iterable[str] | None = None,
    *,
    incident_ids: Iterable[str] | None = None,
    blocked_kinds: Iterable[BlockedKind | str] | None = None,
    max_items: int | None = None,
    scan_limit: int = DISCOVERY_SCAN_LIMIT,
    now: datetime | None = None,
) -> dict[str, Any]:
    evaluated_at = now or datetime.now(timezone.utc)
    requested = [str(item) for item in workflow_ids or () if str(item).strip()]
    report: dict[str, Any] | None
    if requested:
        request_ids, report = requested_workflow_ids(requested), None
    else:
        request_ids, report = _discover_blocked(
            store,
            incident_ids=incident_ids,
            blocked_kinds=blocked_kinds,
            max_items=max_items,
            scan_limit=scan_limit,
        )
    # Scoped to the workflows under reconciliation rather than reading the whole
    # command history. Every use of this list -- here and inside
    # ``restore_reconciliation_reasons`` -- filters on
    # ``workflow_request_id == workflow.request_id``, and the id list is bounded,
    # so the narrowed read answers exactly the same question against a bounded
    # number of rows.
    commands = (
        store.list_remote_commands(workflow_request_ids=request_ids)
        if request_ids
        else []
    )
    items = [
        _plan_item(store, request_id, commands, evaluated_at=evaluated_at)
        for request_id in request_ids
    ]
    plan = {
        "schema_version": 1,
        "mode": "workflow-reconcile-plan",
        "evaluated_at": evaluated_at.isoformat(),
        # Outside the digest: the backlog behind this batch keeps moving while
        # the operator reads, and the approval binds the batch, not it.
        "discovery": report,
        "items": items,
    }
    plan["plan_sha256"] = _canonical_sha256(
        {
            "schema_version": plan["schema_version"],
            "mode": plan["mode"],
            # Same trimming as the retired-generation digest, for the same
            # reason: ``workflow_updated_at`` is restamped by writes that change
            # nothing the verdict reads, and hashing it made the plan an
            # operator had just reviewed unappliable (P0-72A (1)).
            "items": plan_digest_items(items),
        }
    )
    return plan


def _close_never_changed(
    store: ControlPlaneStore,
    item: dict[str, Any],
    *,
    reference: str,
    applied_at: datetime,
) -> tuple[WorkflowRequest, FaultIncident, RecoveryPlan]:
    """Terminalize a BLOCKED record that never changed a node (F-B4 (4)).

    There is no restore successor to name, and ``reconcile_restored_workflow``
    requires one, so this path re-derives every condition itself and writes the
    workflow through ``amend_workflow`` -- the out-of-lease write that bumps
    ``merge_revision`` -- and the plan through ``save_plan``. The incident is
    deliberately not written: without a transaction that would be a blind
    overwrite of a row new events may be merging into, and the audit is carried
    by ``preemption_reason`` and the plan's ``reconciliation_reference``. A
    BLOCKED row itself is quiescent (merges refuse it since F-B4 (2), the
    dispatcher never claims it), which is what makes this two-step write
    acceptable until the Store grows a transactional form.
    """

    request_id = str(item["request_id"])
    workflow = store.get_workflow(request_id)
    incident = store.get_incident(workflow.incident_id)
    if not workflow.source_plan_id:
        raise ValueError("workflow has no source recovery plan")
    source_plan = store.get_plan(workflow.source_plan_id)
    updated_workflow, updated_incident, updated_plan = reconciled_restore_records(
        workflow,
        incident,
        None,
        source_plan,
        store.list_remote_commands(workflow_request_ids=[request_id]),
        expected_fencing_token=int(item["fencing_token"]),
        # The keys the approval digest covers (P0-72A (2)); ``updated_at``
        # is not one of them and a heartbeat may have moved it since the
        # plan was rebuilt a moment ago (F-K1).
        expected_execution_epoch=int(item["execution_epoch"]),
        expected_workflow_updated_at=None,
        reference=reference,
        reconciled_at=applied_at,
    )
    amended = store.amend_workflow(
        request_id,
        {
            "status": updated_workflow.status,
            "preempted_by_workflow_id": updated_workflow.preempted_by_workflow_id,
            "preemption_reason": updated_workflow.preemption_reason,
            "superseded_at": updated_workflow.superseded_at,
        },
    )
    store.save_plan(updated_plan)
    return amended, updated_incident, updated_plan


def apply_workflow_reconcile_plan(
    store: ControlPlaneStore,
    *,
    workflow_ids: Iterable[str],
    expected_plan_sha256: str,
    reference: str,
    now: datetime | None = None,
    waiting_ttl: timedelta | None = None,
) -> dict[str, Any]:
    """Apply an approved plan, one isolated write per item, always returning.

    The plan is rebuilt and its digest compared with the approval; the digest
    covers every field the verdict reads, so a match means the values handed to
    the Store below are the ones the operator approved. Each item is written on
    its own: a failure on one is recorded under ``failures`` and the result still
    names what was applied, so the operator never has to reconstruct a partial
    apply from a traceback.

    ``waiting_ttl`` is the executor's RESTART_WORKLOAD waiting cap: a reservation
    whose only record is a wait older than it is released too. This module has
    no executor config, so the caller supplies it; without it WAITING
    reservations are kept (the helper's own default).
    """

    applied_at = now or datetime.now(timezone.utc)
    plan = build_workflow_reconcile_plan(
        store,
        workflow_ids,
        now=applied_at,
    )
    if plan["plan_sha256"] != expected_plan_sha256:
        raise ValueError("workflow reconcile plan changed before apply")
    rejected = [item for item in plan["items"] if not item["eligible"]]
    if rejected:
        raise ValueError("workflow reconcile plan contains ineligible records")
    applied: list[str] = []
    failures: dict[str, str] = {}
    warnings: list[str] = []
    resolved_plan_ids: list[str] = []
    incident_ids: set[str] = set()
    for item in plan["items"]:
        try:
            if item["terminalization"] == NEVER_CHANGED:
                updated_workflow, incident, source_plan = _close_never_changed(
                    store, item, reference=reference, applied_at=applied_at
                )
            else:
                updated_workflow, incident, source_plan = (
                    store.reconcile_restored_workflow(
                        item["request_id"],
                        item["successor_workflow_id"],
                        expected_fencing_token=int(item["fencing_token"]),
                        # Compare on the epoch the approval bound, not on a
                        # re-read ``updated_at`` a heartbeat restamps (F-K1).
                        expected_execution_epoch=int(item["execution_epoch"]),
                        expected_workflow_updated_at=None,
                        reference=reference,
                        reconciled_at=applied_at,
                    )
                )
        except Exception as exc:  # noqa: BLE001 -- per-item isolation, reported
            failures[str(item["request_id"])] = f"{type(exc).__name__}: {exc}"
            continue
        incident_ids.add(incident.incident_id)
        applied.append(updated_workflow.request_id)
        resolved_plan_ids.append(source_plan.plan_id)
        try:
            # The fifth terminalization path (F-C9): the superseded predecessor
            # may still hold a restart reservation planning took for it. The row
            # is already terminal here, so a failure is a warning, not a failure
            # of the item (the same rule as the retired-generation sibling).
            release_unattempted_restart_reservations(
                store, updated_workflow, waiting_ttl=waiting_ttl
            )
        except Exception as exc:  # noqa: BLE001 -- reported, the write succeeded
            warnings.append(
                f"{updated_workflow.request_id}: closed, but releasing its restart "
                f"reservations failed ({type(exc).__name__}: {exc})"
            )
    return {
        "schema_version": 1,
        "mode": "workflow-reconcile-apply",
        "plan_sha256": expected_plan_sha256,
        "reference": reference,
        "applied_at": applied_at.isoformat(),
        "applied_workflow_ids": applied,
        "failed_workflow_ids": sorted(failures),
        "failures": dict(sorted(failures.items())),
        "restart_reservation_warnings": sorted(warnings),
        "resolved_plan_ids": resolved_plan_ids,
        "archive_eligible_incident_ids": sorted(incident_ids),
        "records_deleted": 0,
    }
