"""The dispatcher's starvation guard over held remediation budgets (§9.5).

A workflow holds its budget claims until it ends, so a storm that fills one
scope leaves its refused workflows PENDING at the head of the dispatcher's
selection -- the oldest ``batch_size`` dispatchable rows. Tried again every
tick they would fill that selection, refused again every tick, and starve
every other cluster behind them. A PENDING / SAFETY_PENDING row whose
recorded refusing scope still has no free slot is therefore passed over;
once the scope has room its waiters are candidates again, in eligibility
order. The guard only skips and never reorders, so FIFO within a scope is
what it was.

Deliberately no per-tick quota ("k free slots let k waiters through"): the
first waiter of a scope may still not claim -- a destructive workflow the
fleet-rollout fence holds returns without claiming -- and a quota spent on it
would starve the waiters behind it for as long as the fence holds.

The guard only ever passes rows over, never admits one: admission stays the
store's, under its budget locks. Never passed over: RUNNING rows (their
leases are renewed by being dispatched), a hold (``budget_work_begun``, never
refused), a preempting successor (part of the occupancy it would be judged
against is the hold reserved for it), a row without a recorded scope (never
refused, or refused before the scope was recorded) and a scope of a kind the
policy does not know. The occupancy is read at most once per tick, and only
when a row with a recorded scope comes up.
"""

from __future__ import annotations

from collections.abc import Callable

from gpu_fault.execution.remediation_budget import (
    RemediationBudgetPolicy,
    budget_work_begun,
    scope_limit,
)
from gpu_fault.models import WorkflowRequest, WorkflowStatus
from gpu_fault.store.contracts import ControlPlaneStore

SKIP_REASON = "remediation_budget"


class BudgetGuard:
    """One dispatch tick's view of which budget scopes are full."""

    def __init__(
        self, store: ControlPlaneStore, policy: Callable[[], RemediationBudgetPolicy]
    ) -> None:
        self._store = store
        self._policy = policy
        self._occupancy: dict[str, int] | None = None

    def passes_over(self, workflow: WorkflowRequest) -> bool:
        """Whether the dispatch scan should skip ``workflow`` this tick."""

        scope = workflow.remediation_budget_last_blocked_scope
        if (
            not scope
            or workflow.status is WorkflowStatus.RUNNING
            or workflow.preempt_predecessor
        ):
            return False
        limit = scope_limit(self._policy(), scope)
        steps = (
            workflow.safety_steps
            if workflow.executes_safety_steps
            else workflow.official_steps
        )
        if limit is None or budget_work_begun(workflow, steps):
            return False
        if self._occupancy is None:
            self._occupancy = self._store.remediation_budget_occupancy()
        return self._occupancy.get(scope, 0) >= limit
