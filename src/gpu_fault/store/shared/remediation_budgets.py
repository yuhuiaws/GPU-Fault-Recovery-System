"""Remediation budget occupancy, admission and hold (the store side of §9.5).

A workflow's claims are held from its admission until it leaves RUNNING for a
terminal status. Occupancy of a scope therefore counts:

* every RUNNING workflow that lists the scope, whatever its lease says: a
  lease is execution ownership, the hold is a remediation in flight. A lease
  lapses on every dispatcher hand-over, for a row past the batch cut and after
  an internal-error release, while the reset or reboot it drove goes on; the
  deadline watchdog bounds how long a RUNNING row can stay RUNNING;
* a SUPERSEDED predecessor while the successor that preempted it has not
  claimed yet: its hold is reserved for that successor, so nobody else takes
  the slot in between and the successor is not refused for it (it is excluded
  from the successor's own count), and the reservation ends with the
  successor's claim or its end -- exactly one hold through the handoff.

A BLOCKED workflow holds nothing: it no longer acts; the node it occupies is
still fenced by the predecessor and node-busy gates.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from datetime import datetime

from gpu_fault.models import WorkflowRequest, WorkflowStatus
from gpu_fault.store.shared.errors import RemediationBudgetError

_SUCCESSOR_WAITING = frozenset({WorkflowStatus.PENDING, WorkflowStatus.SAFETY_PENDING})


def reserved_for(
    predecessor: WorkflowRequest, successor: WorkflowRequest | None
) -> bool:
    """Whether ``predecessor``'s hold is reserved for ``successor`` right now."""

    return (
        successor is not None
        and predecessor.status is WorkflowStatus.SUPERSEDED
        and predecessor.preempted_by_workflow_id == successor.request_id
        and successor.preempt_predecessor
        and successor.predecessor_workflow_id == predecessor.request_id
        and successor.status in _SUCCESSOR_WAITING
    )


def budget_holds(
    workflows: Iterable[WorkflowRequest],
) -> list[tuple[WorkflowRequest, str | None]]:
    """Every row whose claims count, with the successor a reservation is for.

    ``workflows`` must contain the successor of any SUPERSEDED row that is to
    count (the stores pass every row, or the PostgreSQL candidate set that
    includes them).
    """

    rows = list(workflows)
    by_id = {row.request_id: row for row in rows}
    holds: list[tuple[WorkflowRequest, str | None]] = []
    for row in rows:
        if not row.remediation_budget_claims:
            continue
        if row.status is WorkflowStatus.RUNNING:
            holds.append((row, None))
            continue
        successor = by_id.get(row.preempted_by_workflow_id or "")
        if successor is not None and reserved_for(row, successor):
            holds.append((row, successor.request_id))
    return holds


def budget_occupancy(
    workflows: Iterable[WorkflowRequest], *, claimant: str | None = None
) -> Counter[str]:
    """Holders per scope, leaving out ``claimant``'s own hold and the one
    reserved for it."""

    occupancy: Counter[str] = Counter()
    for holder, successor in budget_holds(workflows):
        if claimant is not None and claimant in (holder.request_id, successor):
            continue
        occupancy.update(set(holder.remediation_budget_claims))
    return occupancy


def apply_remediation_budget(
    workflow: WorkflowRequest,
    active_workflows: Iterable[WorkflowRequest],
    claims: dict[str, int],
    *,
    hold: bool = False,
) -> WorkflowRequest:
    """Admit ``workflow`` for ``claims`` or raise; with ``hold`` keep its claims.

    Admission checks every scope against ``budget_occupancy``. A hold is the
    claim of a workflow whose budgeted work has begun: it keeps every scope it
    holds, adds ``claims`` and is never refused -- a refusal would strand a
    reset, a reboot or the compensation it owes (§9.5).
    """

    normalized = {
        str(scope): int(limit)
        for scope, limit in claims.items()
        if str(scope) and int(limit) > 0
    }
    if hold:
        normalized = {
            **{
                scope: limit
                for scope, limit in workflow.remediation_budget_limits.items()
                if scope in workflow.remediation_budget_claims
            },
            **normalized,
        }
        scopes = sorted(set(workflow.remediation_budget_claims) | set(normalized))
    else:
        occupancy = budget_occupancy(active_workflows, claimant=workflow.request_id)
        for scope, limit in sorted(normalized.items()):
            active = occupancy[scope]
            if active < limit:
                continue
            raise RemediationBudgetError(
                "remediation concurrency budget is full: "
                f"scope={scope} active={active} limit={limit}",
                scope=scope,
            )
        scopes = sorted(normalized)
    return workflow.model_copy(
        update={
            "remediation_budget_claims": scopes,
            "remediation_budget_limits": normalized,
            "remediation_budget_last_blocked_reason": None,
            "remediation_budget_last_blocked_scope": None,
        }
    )


def extend_remediation_budget(
    workflow: WorkflowRequest,
    active_workflows: Iterable[WorkflowRequest],
    claims: dict[str, int],
) -> WorkflowRequest:
    """Add ``claims`` to a leased workflow's held budget, or raise.

    Same counting rule as :func:`apply_remediation_budget`; the scopes the
    workflow already holds stay held, so a refusal changes nothing.
    """
    extended = apply_remediation_budget(workflow, active_workflows, claims)
    return workflow.model_copy(
        update={
            "remediation_budget_claims": sorted(
                set(workflow.remediation_budget_claims)
                | set(extended.remediation_budget_claims)
            ),
            "remediation_budget_limits": {
                **workflow.remediation_budget_limits,
                **extended.remediation_budget_limits,
            },
        }
    )


def blocked_by_remediation_budget(
    workflow: WorkflowRequest,
    reason: str,
    *,
    scope: str | None = None,
    now: datetime,
) -> WorkflowRequest:
    """Record a refused claim on the workflow.

    ``scope`` is the refusing budget scope (``RemediationBudgetError.scope``);
    an empty string means the caller did not know it and is stored as None.
    """
    return workflow.model_copy(
        update={
            "execution_owner_id": None,
            "execution_lease_expires_at": None,
            "remediation_budget_wait_count": (
                workflow.remediation_budget_wait_count + 1
            ),
            "remediation_budget_last_blocked_reason": reason,
            "remediation_budget_last_blocked_scope": scope or None,
            "updated_at": now,
        }
    )
