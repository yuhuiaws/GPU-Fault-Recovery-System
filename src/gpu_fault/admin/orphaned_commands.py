"""Cancel remote commands a terminal workflow left open.

A remote command is the GPU-side executor's unit of work for one workflow
step. When the step's workflow reaches a terminal state the command should be
settled too, and the current executor does that: a step that hits its waiting
cap cancels its remote commands before the step fails. An older release did
not, so a step failed by the per-step cap could leave its command ``WAITING``
forever -- observed live: ``CHECK_MECHANICALS`` on a workflow already
``FAILED``, thirteen hours old.

Nothing settles such a command afterwards. The executor never revisits a
terminal workflow, ``--mode retired-generation`` cancels only for generations an
incident re-planned away from, and the operator has no cancel entry point. Yet
the release upgrade refuses to start while any command is ``PENDING``,
``LEASED`` or ``WAITING`` -- so the orphan blocks exactly the release that
stops orphans from forming.

This module is the audited cancel for that one shape: every open command whose
workflow is terminal (``FAILED``, ``SUCCEEDED``, ``BLOCKED``, ``SUPERSEDED``).
A command whose workflow is still ``PENDING``/``RUNNING``/``SAFETY_PENDING``
is never touched -- an open command there is live work, and an executor holding
its lease is the only party allowed to settle it. Cancelling is done through
the Store's own ``cancel_remote_commands_for_workflow`` (long-standing, also
what the retired-generation close uses), which marks the request; the agent
side sees the cancellation and stops. Shipped to the ingress Pod as source like
``retired_generation`` so it can run against the image that left the orphan.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

import gpu_fault.models as _models
from gpu_fault.models import WorkflowStatus
from gpu_fault.remote_command_models import RemoteCommandStatus
from gpu_fault.store import NotFoundError

# Probed rather than imported: this source runs against the deployed image,
# whose ``models`` may predate the attributed operator event (I1). Without it
# the cancel still happens and the audit stays in the cancellation reason.
_BUILD_OPERATOR_EVENT: Any = getattr(_models, "build_operator_event", None)
_OPERATOR_RECONCILED: Any = getattr(
    getattr(_models, "WorkflowEventKind", None), "OPERATOR_RECONCILED", None
)
AUDIT_ACTION = "cancelled orphaned remote commands"

PLAN_MODE = "orphaned-commands-plan"
APPLY_MODE = "orphaned-commands-apply"
TERMINAL_WORKFLOW_STATUSES = frozenset(
    {
        WorkflowStatus.FAILED,
        WorkflowStatus.SUCCEEDED,
        WorkflowStatus.BLOCKED,
        WorkflowStatus.SUPERSEDED,
    }
)
OPEN_REMOTE_STATUSES = frozenset(
    {
        RemoteCommandStatus.PENDING,
        RemoteCommandStatus.LEASED,
        RemoteCommandStatus.WAITING,
    }
)
MAX_WORKFLOW_IDS = 1000
# An orphaned command is not static: the executor keeps re-leasing a WAITING
# CHECK_MECHANICALS to poll for an acknowledgement that will never come, so its
# status flips WAITING <-> LEASED and the lease owner comes and goes between the
# operator's review and the apply. Both are shown, neither is bound; the
# approval binds which commands (by id, step and operation) of which terminal
# workflow are cancelled.
DIGEST_EXCLUDED_COMMAND_FIELDS = frozenset({"status", "lease_owner"})


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def requested_workflow_ids(requested: Iterable[str]) -> list[str]:
    values = sorted({str(item).strip() for item in requested if str(item).strip()})
    if not values:
        raise ValueError("orphaned-commands reconcile requires explicit workflow IDs")
    if len(values) > MAX_WORKFLOW_IDS:
        raise ValueError(
            f"orphaned-commands reconcile accepts at most {MAX_WORKFLOW_IDS} workflow IDs"
        )
    return values


def plan_digest_items(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """The plan items reduced to the fields the operator's approval binds."""

    digest_items = []
    for item in items:
        reduced = dict(item)
        reduced["open_commands"] = [
            {
                key: value
                for key, value in command.items()
                if key not in DIGEST_EXCLUDED_COMMAND_FIELDS
            }
            for command in item.get("open_commands") or []
        ]
        digest_items.append(reduced)
    return digest_items


def _plan_item(store: Any, request_id: str, commands: list[Any]) -> dict[str, Any]:
    open_commands = sorted(
        (
            {
                "command_id": str(item.command_id),
                "status": item.status.value,
                "operation": item.step.operation.value,
                "step_index": item.step_index,
                "lease_owner": item.lease_owner,
            }
            for item in commands
            if item.workflow_request_id == request_id
            and item.status in OPEN_REMOTE_STATUSES
        ),
        key=lambda item: item["command_id"],
    )
    try:
        workflow = store.get_workflow(request_id)
    except (KeyError, NotFoundError):
        return {
            "request_id": request_id,
            "workflow_status": None,
            "open_commands": open_commands,
            "eligible": False,
            "reasons": ["workflow does not exist"],
        }
    reasons: list[str] = []
    if workflow.status not in TERMINAL_WORKFLOW_STATUSES:
        reasons.append(
            f"workflow is {workflow.status.value}; its open commands are live work"
        )
    if getattr(workflow, "execution_owner_id", None) and workflow.status not in (
        TERMINAL_WORKFLOW_STATUSES
    ):
        reasons.append("workflow still has an execution owner")
    if not open_commands:
        reasons.append("workflow has no open remote commands")
    return {
        "request_id": request_id,
        "incident_id": workflow.incident_id,
        "workflow_status": workflow.status.value,
        "official_action": workflow.official_action,
        "fencing_token": workflow.fencing_token,
        "open_commands": open_commands,
        "eligible": not reasons,
        "reasons": reasons,
    }


def build_orphaned_commands_plan(
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
        {"schema_version": 1, "mode": PLAN_MODE, "items": plan_digest_items(items)}
    )
    return plan


def _record_cancellation(
    store: Any,
    request_id: str,
    cancelled: Mapping[str, int],
    *,
    reference: str,
    applied_at: datetime,
    actor: str | None,
    approval: Mapping[str, Any],
) -> None:
    """Append the operator event for a cancel to the (unchanged) workflow.

    The workflow is terminal and stays as it was; the event is the one place its
    history says who cancelled the commands it left open, and under which
    approval (I1). It goes through ``amend_workflow`` -- the out-of-lease write
    -- only when the deployed Store's signature takes an ``event``: this source
    runs against the previously deployed image, whose Store may not.
    """

    amend = getattr(store, "amend_workflow", None)
    if (
        amend is None
        or _BUILD_OPERATOR_EVENT is None
        or _OPERATOR_RECONCILED is None
        or "event" not in inspect.signature(amend).parameters
    ):
        return
    workflow = store.get_workflow(request_id)
    amend(
        request_id,
        {},
        event=_BUILD_OPERATOR_EVENT(
            workflow,
            _OPERATOR_RECONCILED,
            actor=actor,
            reference=reference,
            previous_status=workflow.status,
            at=applied_at,
            details={
                "action": AUDIT_ACTION,
                "cancelled_remote_commands": dict(cancelled),
                **dict(approval),
            },
        ),
    )


def apply_orphaned_commands_plan(
    store: Any,
    *,
    workflow_ids: Iterable[str],
    expected_plan_sha256: str,
    reference: str,
    now: datetime | None = None,
    actor: str | None = None,
    admin_plan_sha256: str | None = None,
) -> dict[str, Any]:
    """Cancel the approved orphans, one workflow at a time, always returning.

    The plan is rebuilt and compared with the approval, so a command that was
    settled or a workflow that was revived since the review refuses the apply
    instead of being acted on under a stale reading. ``actor`` and
    ``admin_plan_sha256`` go on the operator event each cancel leaves on its
    workflow; a cancel whose event could not be written is still a cancel, and
    is named under ``audit_warnings`` rather than ``failures``.
    """

    applied_at = now or datetime.now(timezone.utc)
    approval = {
        key: value
        for key, value in (
            ("plan_sha256", expected_plan_sha256),
            ("admin_plan_sha256", admin_plan_sha256),
        )
        if value is not None
    }
    requested = requested_workflow_ids(workflow_ids)
    plan = build_orphaned_commands_plan(store, requested, now=applied_at)
    if plan["plan_sha256"] != expected_plan_sha256:
        raise ValueError("orphaned-commands reconcile plan changed before apply")
    blocked = [
        f"{item['request_id']}: " + "; ".join(item["reasons"])
        for item in plan["items"]
        if not item["eligible"]
    ]
    if blocked:
        raise ValueError(
            "orphaned-commands reconcile plan contains ineligible records: "
            + " | ".join(blocked)
        )
    cancelled: dict[str, dict[str, int]] = {}
    failures: dict[str, str] = {}
    audit_warnings: dict[str, str] = {}
    for item in plan["items"]:
        request_id = str(item["request_id"])
        try:
            cancelled[request_id] = dict(
                store.cancel_remote_commands_for_workflow(
                    request_id,
                    reason=(
                        f"operator reconciliation {reference}: cancelled remote "
                        f"commands orphaned by {item['workflow_status']} workflow "
                        f"{request_id}"
                    ),
                )
            )
        except Exception as exc:  # noqa: BLE001 -- per-item isolation, reported
            failures[request_id] = f"{type(exc).__name__}: {exc}"
            continue
        try:
            _record_cancellation(
                store,
                request_id,
                cancelled[request_id],
                reference=reference,
                applied_at=applied_at,
                actor=actor,
                approval=approval,
            )
        except Exception as exc:  # noqa: BLE001 -- reported, the cancel happened
            audit_warnings[request_id] = f"{type(exc).__name__}: {exc}"
    return {
        "schema_version": 1,
        "mode": APPLY_MODE,
        "applied_at": applied_at.isoformat(),
        "reference": reference,
        "actor": actor,
        "settled_plan_sha256": plan["plan_sha256"],
        "applied_workflow_ids": sorted(cancelled),
        "cancelled_remote_commands": cancelled,
        "failed_workflow_ids": sorted(failures),
        "failures": failures,
        "audit_warnings": audit_warnings,
        "records_deleted": 0,
    }
