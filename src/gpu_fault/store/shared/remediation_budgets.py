"""Remediation budget occupancy, admission and hold (the store side of §9.5).

A workflow's claims are held from its admission until it ends -- or, parked
BLOCKED, until nothing it started can still be running. Occupancy of a scope
therefore counts:

* every RUNNING workflow that lists the scope, whatever its lease says: a
  lease is execution ownership, the hold is a remediation in flight. A lease
  lapses on every dispatcher hand-over, for a row past the batch cut and after
  an internal-error release, while the reset or reboot it drove goes on; the
  deadline watchdog bounds how long a RUNNING row can stay RUNNING;
* a SUPERSEDED predecessor while the successor that preempted it has not
  claimed yet: its hold is reserved for that successor, so nobody else takes
  the slot in between and the successor is not refused for it (it is excluded
  from the successor's own count), and the reservation ends with the
  successor's claim or its end -- exactly one hold through the handoff;
* a BLOCKED workflow that still occupies its node (``workflow_is_open``:
  NEEDS_OPERATOR, INTERNAL_ERROR) with a node action that may still be
  running -- an unresolved outcome, or a provider or agent action still
  WAITING: the slot stays taken until an operator answers it
  (``confirm-node-action``), a reconcile ends the record, the receipt-settled
  sweep re-opens it to claim again, or the action is older than the longest
  a node action can live (``parked_hold_expires_at``), when the dispatcher
  releases the claims and leaves the record BLOCKED for the operator.

Every other BLOCKED record holds nothing: a settled safety plan ran to its
end, a legacy record without a kind no longer occupies its node, and a parked
record whose action is answered holds its node only as paperwork -- that node
stays fenced by the predecessor and node-busy gates.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import datetime, timedelta

from gpu_fault.execution.node_action_uncertainty import (
    latest_node_action_executions,
    operator_confirmation,
    unresolved_node_action,
)
from gpu_fault.models import (
    WorkflowEventCode,
    WorkflowEventKind,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepExecution,
    WorkflowStepStatus,
    record_workflow_event,
    workflow_is_open,
)
from gpu_fault.store.shared.errors import RemediationBudgetError

_SUCCESSOR_WAITING = frozenset({WorkflowStatus.PENDING, WorkflowStatus.SAFETY_PENDING})
# The longest a node action can live, and so how long a parked record's hold
# can matter (§9.5). The action's step waits at most its ceiling, and every
# ceiling stays inside the node workflow lifetime
# (``execution.config.validate_timing_relationships``;
# GPU_FAULT_NODE_WORKFLOW_MAX_LIFETIME_SECONDS, 3600 s by default). A step that
# gives up cancels its remote commands and the dispatcher cancels what a
# terminal record left open (``orphaned_commands``), so nothing of it starts
# afterwards; a command an executor already holds runs at most the cluster
# executor's execution cap (``cluster_executor.lease.
# DEFAULT_MAX_EXECUTION_SECONDS``, 1800 s, "longer than the slowest legitimate
# action") plus one command lease (``ClusterActionExecutor``'s default, 120 s)
# before its verdict lands or the lease lapses.
NODE_ACTION_SETTLE_SECONDS = 1800 + 120
NODE_ACTION_LONGEST_LIFE_SECONDS = 3600 + NODE_ACTION_SETTLE_SECONDS


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


def parked_node_actions(workflow: WorkflowRequest) -> list[WorkflowStepExecution]:
    """The record's own node actions that may still be running: an unresolved
    outcome, or a provider or agent action still WAITING that no operator
    answered (the newest record of each step answers)."""

    return [
        execution
        for execution in latest_node_action_executions(workflow)
        if unresolved_node_action(execution)
        or (
            execution.status is WorkflowStepStatus.WAITING
            and operator_confirmation(execution.details) is None
        )
    ]


def held_while_blocked(workflow: WorkflowRequest) -> bool:
    """Whether a BLOCKED record still holds its claims: it occupies its node
    and a node action of its own may still be running."""

    return (
        workflow.status is WorkflowStatus.BLOCKED
        and workflow_is_open(workflow.status, workflow.blocked_kind)
        and bool(parked_node_actions(workflow))
    )


def parked_hold_expires_at(workflow: WorkflowRequest) -> datetime | None:
    """When none of a parked record's node actions can still be running.

    The newest action's start plus the longest a node action can live, and
    never before the record's own hard lifetime (``lifetime_deadline_at``,
    which a configured lifetime or an operator-acknowledgement floor moves)
    plus what a held command can still run past it. ``None`` without one.
    """

    actions = parked_node_actions(workflow)
    if not actions:
        return None
    expires = max(item.started_at for item in actions) + timedelta(
        seconds=NODE_ACTION_LONGEST_LIFE_SECONDS
    )
    if workflow.lifetime_deadline_at is not None:
        expires = max(
            expires,
            workflow.lifetime_deadline_at
            + timedelta(seconds=NODE_ACTION_SETTLE_SECONDS),
        )
    return expires


def released_parked_hold(
    workflow: WorkflowRequest, *, now: datetime, actor: str
) -> WorkflowRequest:
    """``workflow`` with its claims released and the audit event appended
    (the action's age and the bound it passed); the record stays BLOCKED."""

    expires = parked_hold_expires_at(workflow)
    if expires is None or now < expires:
        raise ValueError(f"{workflow.request_id} still holds a node action in flight")
    actions = parked_node_actions(workflow)
    started = max(item.started_at for item in actions)
    age = int((now - started).total_seconds())
    bound = int((expires - started).total_seconds())
    released = workflow.model_copy(
        update={"remediation_budget_claims": [], "updated_at": now}
    )
    return record_workflow_event(
        released,
        WorkflowEventKind.OPERATOR_RECONCILED,
        code=WorkflowEventCode.PARKED_BUDGET_RELEASED.value,
        actor=actor,
        status=workflow.status.value,
        reason=(
            f"remediation budget released: the node action started {age}s ago, "
            f"past the {bound}s it can live; the record stays BLOCKED"
        ),
        details={
            "released_budget_claims": sorted(workflow.remediation_budget_claims),
            "node_actions": [
                f"{item.step_index}/{item.operation.value}" for item in actions
            ],
            "action_started_at": started.isoformat(),
            "action_age_seconds": age,
            "action_bound_seconds": bound,
            "hold_expired_at": expires.isoformat(),
        },
        at=now,
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
        if row.status is WorkflowStatus.RUNNING or held_while_blocked(row):
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


def budget_full_reason(scope: str, active: int, limit: int) -> str:
    """The refusal a full scope records on a waiter (store and dispatcher)."""

    return (
        "remediation concurrency budget is full: "
        f"scope={scope} active={active} limit={limit}"
    )


def apply_remediation_budget(
    workflow: WorkflowRequest,
    active_workflows: Iterable[WorkflowRequest],
    claims: dict[str, int],
    *,
    hold: bool = False,
) -> WorkflowRequest:
    """Admit ``workflow`` for ``claims`` or raise; with ``hold`` keep its claims.

    Admission checks every scope against ``budget_occupancy`` of
    ``active_workflows`` (every row that may hold one; the in-memory and SQLite
    stores pass them all). A hold is the claim of a workflow whose budgeted
    work has begun: it keeps every scope it holds, adds ``claims`` and is never
    refused -- a refusal would strand a reset, a reboot or the compensation it
    owes (§9.5).
    """

    occupancy: Mapping[str, int] = (
        {} if hold else budget_occupancy(active_workflows, claimant=workflow.request_id)
    )
    return admit_against_occupancy(workflow, occupancy, claims, hold=hold)


def admit_against_occupancy(
    workflow: WorkflowRequest,
    occupancy: Mapping[str, int],
    claims: dict[str, int],
    *,
    hold: bool = False,
) -> WorkflowRequest:
    """``apply_remediation_budget`` on holders already counted per scope, the
    claimant's own hold left out (PostgreSQL counts its RUNNING holders in SQL)."""

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
        for scope, limit in sorted(normalized.items()):
            active = occupancy.get(scope, 0)
            if active < limit:
                continue
            raise RemediationBudgetError(
                budget_full_reason(scope, active, limit), scope=scope
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
    return extend_against_occupancy(
        workflow,
        budget_occupancy(active_workflows, claimant=workflow.request_id),
        claims,
    )


def extend_against_occupancy(
    workflow: WorkflowRequest,
    occupancy: Mapping[str, int],
    claims: dict[str, int],
) -> WorkflowRequest:
    """``extend_remediation_budget`` on holders already counted per scope."""

    extended = admit_against_occupancy(workflow, occupancy, claims)
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
