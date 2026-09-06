"""Scan-side filters shared by the Python-filtered store backends."""

from __future__ import annotations

from datetime import datetime
from typing import Callable, Collection

from gpu_fault.models import WorkflowRequest, WorkflowStatus, workflow_is_open
from gpu_fault.store.shared.time import utc_text

OPEN_WORKFLOW_STATUSES = frozenset(
    {
        WorkflowStatus.PENDING,
        WorkflowStatus.SAFETY_PENDING,
        WorkflowStatus.RUNNING,
    }
)


def held_reason(
    workflow: WorkflowRequest,
    *,
    dispatchable_at: datetime,
    exclude_request_ids: Collection[str],
    lookup: Callable[[str], WorkflowRequest | None],
) -> str | None:
    """The pushdown of ``list_workflows`` for the Python-filtered backends.

    Mirrors the Postgres clauses: a retired id, a future ``not_before``, or a
    predecessor that is still open (PENDING / SAFETY_PENDING / RUNNING). A
    missing predecessor is *not* a reason; the dispatcher reports that itself.
    """

    if workflow.request_id in exclude_request_ids:
        return "retired"
    if workflow.not_before is not None and workflow.not_before > dispatchable_at:
        return "not_before"
    if workflow.predecessor_workflow_id is not None:
        predecessor = lookup(workflow.predecessor_workflow_id)
        if predecessor is not None and workflow_is_open(
            predecessor.status, predecessor.blocked_kind
        ):
            return "predecessor"
    return None


def dispatch_eligible_at(workflow: WorkflowRequest) -> datetime:
    """When the row became dispatchable: the later of ``created_at`` and
    ``not_before`` (F-A2a).

    This is the dispatch-mode sort key. ``updated_at`` was the key before, and
    every merge into the row (ABSORB, WIDEN_IN_PLACE, a budget refusal) bumped
    it, so the node with the densest faults sorted last. Nothing but the two
    fields below moves this value, and a merge rewrites neither.
    """

    if workflow.not_before is not None and workflow.not_before > workflow.created_at:
        return workflow.not_before
    return workflow.created_at


def dispatch_order_key(workflow: WorkflowRequest) -> tuple[str, str]:
    """The ``(dispatch_eligible_at, request_id)`` key as the stores compare it.

    Text on purpose: Postgres orders ``GREATEST(payload->>'created_at',
    payload->>'not_before')`` as text so an expression index can serve it, and
    the Python-filtered backends must page identically, cursor included.
    """

    return utc_text(dispatch_eligible_at(workflow)), workflow.request_id
