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

from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.execution.remediation_budget import RemediationBudgetPolicy
from gpu_fault.models import BlockedKind, WorkflowStatus, WorkflowStepStatus
from gpu_fault.store import InMemoryStore, RemediationBudgetError, SqliteStore
from gpu_fault.store.shared.remediation_budgets import budget_occupancy
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


def _blocked(name: str, kind: BlockedKind | None, *, unresolved: bool):
    reboot = workflow_step_execution(
        1,
        REBOOT,
        WorkflowStepStatus.FAILED,
        details=(
            dict(UNKNOWN_OUTCOME) if unresolved else {"node_action_outcome_known": True}
        ),
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
    older writer leaves one); without one it holds nothing. SAFETY_SETTLED ran
    its safety plan to the end and a legacy BLOCKED without a kind no longer
    occupies its node: released whatever the action says.
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
        _blocked("settled", BlockedKind.SAFETY_SETTLED, unresolved=True),
        _blocked("legacy", None, unresolved=True),
    ]

    assert budget_occupancy(rows) == {
        CLUSTER: 2,
        "region": 2,
        "node:cluster-a:node-parked": 1,
        "node:cluster-a:node-internal-unknown": 1,
    }
    assert budget_occupancy(rows, claimant=parked.request_id)[CLUSTER] == 1
