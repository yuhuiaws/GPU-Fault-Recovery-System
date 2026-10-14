"""A BLOCKED record whose node action may still be running keeps its budget.

``terminal_state`` parks a workflow BLOCKED / NEEDS_OPERATOR when a node
action's outcome is unknown as it ends -- a reboot whose result never came back:
nothing automatic may treat the node as free. Batch 89 released the budget hold
at every BLOCKED, so with ``cluster_limit=1`` a second workflow started in the
cluster while the first one's reboot might still be running. The hold now lasts
while the record still occupies its node with an unresolved node action, until
an operator answers it (``confirm-node-action``), a reconcile ends the record,
or its receipts settle it; every other BLOCKED kind releases at once.
"""

from __future__ import annotations

import inspect
import logging
from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.cluster_executor.executor import ClusterActionExecutor
from gpu_fault.cluster_executor.lease import DEFAULT_MAX_EXECUTION_SECONDS
from gpu_fault.execution.budget_guard import BudgetWatch, release_lapsed_parked_holds
from gpu_fault.execution.config import ProductionExecutorConfig
from gpu_fault.execution.remediation_budget import RemediationBudgetPolicy
from gpu_fault.models import BlockedKind, WorkflowStatus, WorkflowStepStatus
from gpu_fault.store import InMemoryStore, RemediationBudgetError, SqliteStore
from gpu_fault.store.shared.remediation_budgets import (
    NODE_ACTION_LONGEST_LIFE_SECONDS,
    NODE_ACTION_SETTLE_SECONDS,
    budget_occupancy,
    parked_hold_expires_at,
)
from tests._builders import (
    copy_model,
    execute_workflow,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)
from tests.execution._budget_hold_support import (
    REBOOT,
    REBOOT_PLAN,
    STOP,
    UNKNOWN_OUTCOME,
    UnknownOutcomeAdapter,
    budget_executor,
    holders,
    operator_confirmed,
    seed,
)

NOW = datetime(2026, 7, 20, 10, 0, tzinfo=timezone.utc)
CLUSTER = "cluster:cluster-a"
ONE_PER_CLUSTER = RemediationBudgetPolicy(region_limit=20, cluster_limit=1)


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    if request.param == "memory":
        yield InMemoryStore()
        return
    sqlite = SqliteStore(str(tmp_path / "blocked-hold.db"))
    try:
        yield sqlite
    finally:
        sqlite.close()


def _park_a(store):
    """A stops its workload and reboots; the reboot's outcome never comes back."""

    a = seed(store, "a", created_at=NOW)
    b = seed(store, "b", created_at=NOW + timedelta(seconds=1))
    adapter = UnknownOutcomeAdapter(REBOOT_PLAN, {a.request_id})
    executor = budget_executor(store, adapter, ONE_PER_CLUSTER)
    assert execute_workflow(executor, a.request_id).status is WorkflowStatus.BLOCKED
    parked = store.get_workflow(a.request_id)
    assert parked.blocked_kind is BlockedKind.NEEDS_OPERATOR
    assert CLUSTER in parked.remediation_budget_claims
    return a, b, adapter, executor


def test_a_reboot_of_unknown_outcome_holds_the_cluster_until_an_operator_confirms(
    store,
):
    a, b, adapter, executor = _park_a(store)

    assert store.remediation_budget_occupancy().get(CLUSTER) == 1, (
        "a parked record whose reboot may still be running released its budget"
    )
    with pytest.raises(RemediationBudgetError) as refused:
        execute_workflow(executor, b.request_id)
    assert refused.value.scope == CLUSTER
    assert b.request_id not in adapter.started(), adapter.calls

    # confirm-node-action answers the question on the step it parked on; the
    # record stays BLOCKED (the reconcile or the incident close ends it) but
    # its node is no longer mid-action, so the hold ends here.
    parked = store.get_workflow(a.request_id)
    store.save_workflow(
        operator_confirmed(parked, now=NOW, node_id="node-a"), expected=parked
    )
    assert store.get_workflow(a.request_id).status is WorkflowStatus.BLOCKED
    assert store.remediation_budget_occupancy().get(CLUSTER, 0) == 0

    assert execute_workflow(executor, b.request_id).status is WorkflowStatus.RUNNING
    assert holders([store.get_workflow(b.request_id)], CLUSTER) == [b.request_id]


def test_a_reconcile_that_ends_the_parked_record_releases_its_budget(store):
    """workflow-reconcile ends the record SUPERSEDED with its claims cleared."""

    a, b, _, executor = _park_a(store)
    parked = store.get_workflow(a.request_id)
    store.save_workflow(
        copy_model(
            parked,
            status=WorkflowStatus.SUPERSEDED,
            superseded_at=NOW,
            remediation_budget_claims=[],
        ),
        expected=parked,
    )

    assert store.remediation_budget_occupancy().get(CLUSTER, 0) == 0
    assert execute_workflow(executor, b.request_id).status is WorkflowStatus.RUNNING


# A provider reboot still WAITING carries no outcome flag: the HyperPod
# lifecycle adapter's record of a submitted reboot it is still polling.
PROVIDER_WAITING = {"submitted_nodes": ["node"], "activated_spare_nodes": []}


def _blocked(
    name: str, kind: BlockedKind | None, *, unresolved: bool, waiting: bool = False
):
    reboot = workflow_step_execution(
        1,
        REBOOT,
        WorkflowStepStatus.WAITING if waiting else WorkflowStepStatus.FAILED,
        details=(
            dict(PROVIDER_WAITING)
            if waiting
            else dict(UNKNOWN_OUTCOME)
            if unresolved
            else {"node_action_outcome_known": True}
        ),
        started_at=NOW,
    )
    return workflow_request(
        f"workflow-{name}",
        f"incident-{name}",
        status=WorkflowStatus.BLOCKED,
        blocked_kind=kind,
        official_steps=[
            workflow_step(STOP, node_ids=[f"node-{name}"]),
            workflow_step(REBOOT, node_ids=[f"node-{name}"]),
        ],
        completed_step_indexes=[0],
        step_executions=[workflow_step_execution(0, STOP), reboot],
        remediation_budget_claims=[CLUSTER, "region", f"node:cluster-a:node-{name}"],
    )


def test_each_blocked_kind_holds_only_while_its_node_may_still_be_mid_action():
    """One row per BLOCKED kind, all listing claims.

    NEEDS_OPERATOR with an unresolved node action -- the record ``terminal_state``
    parks -- holds; once an operator confirmed the action, or when it was never
    unresolved, its node holds only paperwork: released. INTERNAL_ERROR also
    occupies its node and the shared predicate keeps one with an unresolved
    action (the terminal funnel parks such a record NEEDS_OPERATOR, so only an
    older writer leaves one) and, failing closed, one whose provider reboot is
    still WAITING (no outcome flag, nothing polls it any more); without either
    it holds nothing. SAFETY_SETTLED ran its safety plan to the end and a
    legacy BLOCKED without a kind no longer occupies its node: released
    whatever the action says.
    """

    parked = _blocked("parked", BlockedKind.NEEDS_OPERATOR, unresolved=True)
    rows = [
        parked,
        operator_confirmed(
            _blocked("confirmed", BlockedKind.NEEDS_OPERATOR, unresolved=True),
            now=NOW,
            node_id="node-confirmed",
        ),
        _blocked("answered", BlockedKind.NEEDS_OPERATOR, unresolved=False),
        _blocked("internal-unknown", BlockedKind.INTERNAL_ERROR, unresolved=True),
        _blocked("internal", BlockedKind.INTERNAL_ERROR, unresolved=False),
        _blocked(
            "internal-waiting",
            BlockedKind.INTERNAL_ERROR,
            unresolved=False,
            waiting=True,
        ),
        _blocked("settled", BlockedKind.SAFETY_SETTLED, unresolved=True),
        _blocked("legacy", None, unresolved=True),
    ]

    assert budget_occupancy(rows) == {
        CLUSTER: 3,
        "region": 3,
        "node:cluster-a:node-parked": 1,
        "node:cluster-a:node-internal-unknown": 1,
        "node:cluster-a:node-internal-waiting": 1,
    }
    assert budget_occupancy(rows, claimant=parked.request_id)[CLUSTER] == 2


# ------------------------------------------------- the longest an action lives


def test_a_parked_hold_lasts_the_longest_a_node_action_can_live():
    """The bound is derived from the limits the code already enforces: the node
    workflow lifetime every step ceiling stays inside, the cluster executor's
    execution cap and one command lease -- and never ends before the record's
    own lifetime plus what a held command can still run past it."""

    lease = inspect.signature(ClusterActionExecutor).parameters["lease_seconds"]
    assert NODE_ACTION_SETTLE_SECONDS == DEFAULT_MAX_EXECUTION_SECONDS + lease.default
    lifetime = ProductionExecutorConfig.from_mapping({}).node_workflow_lifetime_seconds
    assert NODE_ACTION_LONGEST_LIFE_SECONDS == lifetime + NODE_ACTION_SETTLE_SECONDS

    parked = _blocked("parked", BlockedKind.NEEDS_OPERATOR, unresolved=True)
    longest = timedelta(seconds=NODE_ACTION_LONGEST_LIFE_SECONDS)
    assert parked_hold_expires_at(parked) == NOW + longest
    floored = copy_model(parked, lifetime_deadline_at=NOW + timedelta(days=1))
    assert parked_hold_expires_at(floored) == NOW + timedelta(
        days=1, seconds=NODE_ACTION_SETTLE_SECONDS
    ), "an acknowledgement-floored lifetime keeps the hold to its own end"
    assert (
        parked_hold_expires_at(
            _blocked("answered", BlockedKind.NEEDS_OPERATOR, unresolved=False)
        )
        is None
    )


def _released_events(row):
    return [event for event in row.events if event.code == "PARKED_BUDGET_RELEASED"]


def test_a_lapsed_parked_hold_is_released_once_and_the_record_stays_blocked(
    store, caplog
):
    a, b, _, executor = _park_a(store)
    parked = store.get_workflow(a.request_id)
    expires = parked_hold_expires_at(parked)
    assert expires is not None
    watch = BudgetWatch()

    assert (
        release_lapsed_parked_holds(store, watch, now=expires - timedelta(seconds=1))
        == []
    )
    assert store.remediation_budget_occupancy().get(CLUSTER) == 1, (
        "a hold released before its action can have ended"
    )

    caplog.set_level(logging.WARNING, logger="gpu_fault.execution.budget_guard")
    later = expires + timedelta(minutes=5)
    assert release_lapsed_parked_holds(store, watch, now=later) == [a.request_id]

    row = store.get_workflow(a.request_id)
    assert row.status is WorkflowStatus.BLOCKED
    assert row.blocked_kind is BlockedKind.NEEDS_OPERATOR
    assert row.remediation_budget_claims == []
    (event,) = _released_events(row)
    assert event.actor == "dispatcher"
    assert event.details["released_budget_claims"] == sorted(
        parked.remediation_budget_claims
    )
    assert event.details["action_bound_seconds"] == NODE_ACTION_LONGEST_LIFE_SECONDS
    assert event.details["action_age_seconds"] >= NODE_ACTION_LONGEST_LIFE_SECONDS
    assert watch.parked_holds_released_total == 1
    assert watch.parked_hold_release_last_seen_timestamp_seconds > 0
    assert store.remediation_budget_occupancy().get(CLUSTER, 0) == 0
    assert execute_workflow(executor, b.request_id).status is WorkflowStatus.RUNNING

    # Logged and audited once: the record holds nothing for a later look.
    assert (
        release_lapsed_parked_holds(store, watch, now=later + timedelta(hours=1)) == []
    )
    assert len(_released_events(store.get_workflow(a.request_id))) == 1
    released_logs = [
        record for record in caplog.records if a.request_id in record.getMessage()
    ]
    assert len(released_logs) == 1, [record.getMessage() for record in released_logs]


def test_an_internal_error_record_with_a_waiting_provider_reboot_holds_until_the_bound(
    store,
):
    """Fail closed: the dispatcher blocked the record while a HyperPod reboot
    was still WAITING; nothing polls it any more, so its slot stays taken until
    the reboot can no longer be running."""

    holder = copy_model(
        _blocked("a", BlockedKind.INTERNAL_ERROR, unresolved=False, waiting=True),
        incident_id="incident-a",
        execution_epoch=1,
    )
    store.save_workflow(holder)
    waiter = seed(store, "b", created_at=NOW + timedelta(seconds=1))
    executor = budget_executor(
        store, UnknownOutcomeAdapter(REBOOT_PLAN, set()), ONE_PER_CLUSTER
    )

    with pytest.raises(RemediationBudgetError) as refused:
        execute_workflow(executor, waiter.request_id)
    assert refused.value.scope == CLUSTER

    expires = parked_hold_expires_at(store.get_workflow(holder.request_id))
    assert expires == NOW + timedelta(seconds=NODE_ACTION_LONGEST_LIFE_SECONDS)
    watch = BudgetWatch()
    assert release_lapsed_parked_holds(store, watch, now=expires) == [holder.request_id]
    assert store.get_workflow(holder.request_id).status is WorkflowStatus.BLOCKED
    assert execute_workflow(executor, waiter.request_id).status is (
        WorkflowStatus.RUNNING
    )
