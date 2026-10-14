"""The dispatcher's view of held remediation budgets (§9.5).

A workflow holds its budget claims until it ends, so a storm that fills one
scope leaves its refused workflows PENDING at the head of the dispatcher's
selection -- the oldest ``batch_size`` dispatchable rows. Tried again every
tick they would fill that selection, refused again every tick, and starve
every other cluster behind them. A PENDING / SAFETY_PENDING row whose
recorded refusing scope still has no free slot is therefore passed over;
once the scope has room its waiters are candidates again, in eligibility
order. The guard only skips and never reorders, so FIFO within a scope is
what it was.

A row never refused has no recorded scope. Tried once each, the first ticks of
a storm of N rows filled the selection with them and another cluster waited
about N / batch_size ticks. When the tick's occupancy shows a full scope, the
guard therefore works out the scopes such a row would ask for (its incident's
cluster, cached for the dispatcher's lifetime) and passes it over when one of
them is full, recording the refusal on the row as the store would (a
compare-and-set write, once per row), so the waiting gauges count it. At most
``FRESH_WAITER_STORE_CALLS`` incident reads and writes a tick; a row past
that is left to the store.

Deliberately no per-tick quota ("k free slots let k waiters through"): the
first waiter of a scope may still not claim -- a destructive workflow the
fleet-rollout fence holds returns without claiming -- and a quota spent on it
would starve the waiters behind it for as long as the fence holds.

The guard only ever passes rows over, never admits one: admission stays the
store's, under its budget locks. Never passed over: RUNNING rows (their
leases are renewed by being dispatched), a hold (``budget_work_begun``, never
refused), a preempting successor (part of the occupancy it would be judged
against is the hold reserved for it), a row with an execution owner, and a
scope of a kind the policy does not know.

Over-limit scopes. A hold is never refused, so a scope can hold more than its
limit: workflows an older release left RUNNING without their claims re-take
them when this release dispatches them, an older dispatcher alternating with
this one during a roll drops them again and admits on its own count, and a
lowered limit leaves the earlier holders in place. The occupancy is read once
per tick; ``BudgetWatch`` keeps the scopes above their limit for /metrics and
logs every change. Nothing new is admitted into a full or over-limit scope --
the store refuses it, the guard passes it over -- until it drains below its
limit.

Parked holds. A BLOCKED record whose node action may still be running keeps
its claims (``store.shared.remediation_budgets.held_while_blocked``) -- but
only for as long as that action can live: ``release_lapsed_parked_holds``
releases them once ``parked_hold_expires_at`` has passed, in one
compare-and-set write with its audit event, counted and logged once. The
record itself stays BLOCKED for the operator.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from datetime import datetime, timedelta, timezone

from gpu_fault.execution.config import ProductionExecutorConfig
from gpu_fault.execution.remediation_budget import (
    RemediationBudgetPolicy,
    budget_work_begun,
    claims_in_cluster,
    scope_limit,
)
from gpu_fault.execution.transient_errors import transient_store_error
from gpu_fault.models import (
    EXECUTABLE_WORKFLOW_STATUSES,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepSpec,
)
from gpu_fault.store import NotFoundError
from gpu_fault.store.contracts import ControlPlaneStore
from gpu_fault.store.shared.errors import StaleWriteError
from gpu_fault.store.shared.remediation_budgets import (
    blocked_by_remediation_budget,
    budget_full_reason,
    held_while_blocked,
    parked_hold_expires_at,
    released_parked_hold,
)

LOGGER = logging.getLogger(__name__)

SKIP_REASON = "remediation_budget"
# Incident reads and refusal writes the guard may spend on never-refused rows
# in one tick (about a second of PostgreSQL time); the next tick goes on.
FRESH_WAITER_STORE_CALLS = 1000
# incident id -> cluster id is immutable; the map is dropped when it grows
# past this many incidents rather than kept in age order.
CLUSTER_CACHE_LIMIT = 50_000
# Every scope kind ``scope_limit`` knows, exported at zero when not over.
SCOPE_TYPES = ("class", "cluster", "domain", "node", "region")
# The parked-hold sweep: the bound it applies is over an hour, so a minute
# between looks costs nothing; the newest BLOCKED rows, as the sibling
# BLOCKED sweeps read them.
PARKED_SWEEP_INTERVAL = timedelta(seconds=60)
PARKED_SWEEP_LIMIT = 1000
DISPATCHER_ACTOR = "dispatcher"


class BudgetWatch:
    """What the dispatcher keeps across ticks: incident clusters and the
    scopes its last tick found above their limit."""

    def __init__(self) -> None:
        self.clusters: dict[str, str] = {}
        # scope -> (holders, configured limit), from the last tick's read.
        self.over_limit: dict[str, tuple[int, int]] = {}
        self.over_limit_last_seen_timestamp_seconds = 0.0
        self.parked_holds_released_total = 0
        self.parked_hold_release_last_seen_timestamp_seconds = 0.0
        self.next_parked_sweep_at = datetime.min.replace(tzinfo=timezone.utc)

    def guard(
        self, store: ControlPlaneStore, config: ProductionExecutorConfig
    ) -> BudgetGuard:
        """A new tick's guard over this watch."""

        return BudgetGuard(store, config, self)

    def over_limit_by_scope_type(self) -> dict[str, int]:
        return dict(Counter(scope.split(":", 1)[0] for scope in self.over_limit))

    def observe(self, over_limit: dict[str, tuple[int, int]]) -> None:
        if over_limit:
            self.over_limit_last_seen_timestamp_seconds = time.time()
        if over_limit == self.over_limit:
            return
        if over_limit:
            LOGGER.warning(
                "remediation budget scope(s) above their limit: %s; holds are "
                "never refused (in-flight work re-taken after an upgrade from a "
                "release that dropped its claims, an older dispatcher still in "
                "the roll, or a lowered limit), and nothing new is admitted "
                "into these scopes until they drain below the limit",
                "; ".join(
                    f"{scope} holders={holders} limit={limit}"
                    for scope, (holders, limit) in sorted(over_limit.items())
                ),
            )
        else:
            LOGGER.info("remediation budget scopes drained back to their limits")
        self.over_limit = dict(over_limit)

    def idle(self) -> None:
        """This process stopped dispatching: another one holds the lease and
        reports the scopes."""

        self.over_limit = {}


class BudgetGuard:
    """One dispatch tick's view of which budget scopes are full."""

    def __init__(
        self,
        store: ControlPlaneStore,
        config: ProductionExecutorConfig,
        watch: BudgetWatch | None = None,
    ) -> None:
        self._store = store
        self._config = config
        self._watch = watch if watch is not None else BudgetWatch()
        self._occupancy: dict[str, int] | None = None
        self._any_full: bool | None = None
        self._calls = 0

    @property
    def _policy(self) -> RemediationBudgetPolicy:
        # Read when a row needs it: a tick without holders or waiters never does.
        return self._config.remediation_budget

    def occupancy(self) -> dict[str, int]:
        """The tick's one occupancy read."""

        if self._occupancy is None:
            self._occupancy = self._store.remediation_budget_occupancy()
        return self._occupancy

    def passes_over(self, workflow: WorkflowRequest) -> bool:
        """Whether the dispatch scan should skip ``workflow`` this tick."""

        if workflow.status is WorkflowStatus.RUNNING or workflow.preempt_predecessor:
            return False
        steps = (
            workflow.safety_steps
            if workflow.executes_safety_steps
            else workflow.official_steps
        )
        if budget_work_begun(workflow, steps):
            return False
        scope = workflow.remediation_budget_last_blocked_scope
        if not scope:
            return self._fresh_waiter(workflow, steps)
        limit = scope_limit(self._policy, scope)
        return limit is not None and self.occupancy().get(scope, 0) >= limit

    def observe(self) -> None:
        """Hand the scopes above their limit to the watch; reads the
        occupancy if no row needed it this tick."""

        over_limit: dict[str, tuple[int, int]] = {}
        for scope, holders in sorted(self.occupancy().items()):
            limit = scope_limit(self._policy, scope)
            if limit is not None and holders > limit:
                over_limit[scope] = (holders, limit)
        self._watch.observe(over_limit)

    def _fresh_waiter(
        self, workflow: WorkflowRequest, steps: list[WorkflowStepSpec]
    ) -> bool:
        """Pass a never-refused row over when a scope it asks for is full."""

        occupancy = self.occupancy()
        if self._any_full is None:
            self._any_full = any(
                holders >= limit
                for scope, holders in occupancy.items()
                if (limit := scope_limit(self._policy, scope)) is not None
            )
        if workflow.execution_owner_id is not None or not self._any_full:
            return False
        cluster = self._cluster_of(workflow.incident_id)
        if cluster is None:
            return False
        claims = claims_in_cluster(self._policy, workflow, cluster, steps)
        full = [
            scope
            for scope in sorted(claims)
            if occupancy.get(scope, 0) >= claims[scope]
        ]
        if not full or self._calls >= FRESH_WAITER_STORE_CALLS:
            return False
        refusing = full[0]
        self._calls += 1
        reason = budget_full_reason(refusing, occupancy[refusing], claims[refusing])
        try:
            self._store.save_workflow(
                blocked_by_remediation_budget(
                    workflow, reason, scope=refusing, now=datetime.now(timezone.utc)
                ),
                expected=workflow,
            )
        except (StaleWriteError, NotFoundError):
            return False  # the row moved since the scan: the store decides
        except Exception as exc:  # noqa: BLE001 - one row, not the tick (D-4)
            if transient_store_error(exc):
                raise
            LOGGER.exception(
                "budget waiter %s could not be recorded; left to the store",
                workflow.request_id,
            )
            return False
        return True

    def _cluster_of(self, incident_id: str) -> str | None:
        clusters = self._watch.clusters
        cluster = clusters.get(incident_id)
        if cluster is not None or self._calls >= FRESH_WAITER_STORE_CALLS:
            return cluster
        self._calls += 1
        try:
            cluster = self._store.get_incident(incident_id).cluster_id
        except NotFoundError:
            return None
        if len(clusters) >= CLUSTER_CACHE_LIMIT:
            clusters.clear()
        clusters[incident_id] = cluster
        return cluster


def released_status(
    scanned: WorkflowRequest, current: WorkflowRequest
) -> WorkflowStatus:
    """The status an internal-error release hands a row back in.

    The status the scan saw, so a row the error hit before it did anything
    looks untouched -- unless its own budgeted work began in this dispatch: it
    stays RUNNING, and so keeps its hold through the backoff while the node
    is mid-remediation (a PENDING row holds nothing). RUNNING stays RUNNING
    either way, so a resumed workflow keeps its progress.
    """

    steps = (
        current.safety_steps
        if scanned.executes_safety_steps
        else current.official_steps
    )
    if current.status is WorkflowStatus.RUNNING and budget_work_begun(current, steps):
        return WorkflowStatus.RUNNING
    if scanned.status in EXECUTABLE_WORKFLOW_STATUSES:
        return scanned.status
    return current.status


def release_lapsed_parked_holds(
    store: ControlPlaneStore, watch: BudgetWatch, *, now: datetime
) -> list[str]:
    """Release the claims of parked records past the longest their node
    action can live (§9.5); returns the ids released. At most one look per
    ``PARKED_SWEEP_INTERVAL``; one isolated compare-and-set write per record."""

    if now < watch.next_parked_sweep_at:
        return []
    watch.next_parked_sweep_at = now + PARKED_SWEEP_INTERVAL
    released: list[str] = []
    for workflow in store.list_workflows(
        {WorkflowStatus.BLOCKED}, limit=PARKED_SWEEP_LIMIT, newest_first=True
    ):
        expires = parked_hold_expires_at(workflow)
        if (
            not workflow.remediation_budget_claims
            or not held_while_blocked(workflow)
            or expires is None
            or now < expires
        ):
            continue
        try:
            record = released_parked_hold(workflow, now=now, actor=DISPATCHER_ACTOR)
            store.save_workflow(record, expected=workflow)
        except (StaleWriteError, NotFoundError):
            continue  # the row moved since the read: the next look decides
        except Exception as exc:  # noqa: BLE001 - one row, not the sweep (D-4)
            if transient_store_error(exc):
                raise
            LOGGER.exception(
                "parked hold of %s could not be released; left for the next look",
                workflow.request_id,
            )
            continue
        watch.parked_holds_released_total += 1
        watch.parked_hold_release_last_seen_timestamp_seconds = time.time()
        released.append(workflow.request_id)
        details = record.events[-1].details
        LOGGER.warning(
            "parked workflow %s (%s): remediation budget released, its node "
            "action started %ss ago and can live at most %ss; released claims "
            "%s; the record stays BLOCKED for the operator",
            workflow.request_id,
            workflow.blocked_kind.value if workflow.blocked_kind else None,
            details["action_age_seconds"],
            details["action_bound_seconds"],
            details["released_budget_claims"],
        )
    return released
