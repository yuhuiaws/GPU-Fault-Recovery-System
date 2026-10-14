"""Held remediation budgets on PostgreSQL, in each workflow storage mode.

The rule (``gpu_fault.store.shared.remediation_budgets``) is decided on the
rows ``remediation_budget_holders.budget_candidates`` reads under the scope
advisory locks; ``remediation_budget_occupancy`` is the same rule as one
aggregate for the dispatcher. Both have to see a legacy row, a dual row and a
dedicated row alike.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from threading import Barrier

import pytest

from gpu_fault.execution.remediation_budget import (
    RemediationBudgetPolicy,
    remediation_budget_claims,
)
from gpu_fault.models import BlockedKind, WorkflowStatus, WorkflowStepStatus
from gpu_fault.store import PostgresStore
from gpu_fault.store.shared.errors import RemediationBudgetError
from gpu_fault.store.shared.remediation_budgets import budget_occupancy
from tests._builders import (
    copy_model,
    execute_workflow,
    workflow_request,
    workflow_step_execution,
)
from tests.execution._budget_hold_support import (
    REBOOT,
    REBOOT_PLAN,
    STOP,
    UNKNOWN_OUTCOME,
    ScriptedAdapter,
    UnknownOutcomeAdapter,
    budget_dispatcher,
    budget_executor,
    operator_confirmed,
    seed,
)
from tests.store import test_postgres_state_tables as state_tables
from tests.store.test_postgres_workflow_state_tables import select_mode

database = state_tables.database
migration_database = state_tables.migration_database
POSTGRES_URL = os.getenv("GPU_FAULT_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(
    not POSTGRES_URL, reason="GPU_FAULT_TEST_POSTGRES_URL is not configured"
)
NOW = datetime(2026, 7, 20, 10, 0, tzinfo=UTC)
CLUSTER = "cluster:cluster-a"
TWO_PER_CLUSTER = RemediationBudgetPolicy(cluster_limit=2, resource_class_limit=10)
ONE_PER_CLUSTER = RemediationBudgetPolicy(cluster_limit=1, resource_class_limit=10)


@pytest.fixture(params=["legacy", "dual", "dedicated"])
def store(migration_database, request):
    select_mode(migration_database, "workflow", request.param)
    assert POSTGRES_URL is not None, "budget hold tests require local PostgreSQL"
    instance = PostgresStore(POSTGRES_URL, initialize_schema=False)
    try:
        yield instance
    finally:
        instance.close()


def _end(store, request_id: str) -> None:
    row = store.get_workflow(request_id)
    store.save_workflow_if_leased(
        row.model_copy(
            update={
                "status": WorkflowStatus.SUCCEEDED,
                "execution_owner_id": None,
                "execution_lease_expires_at": None,
            }
        ),
        row.execution_owner_id or "",
        row.execution_epoch,
    )


def test_two_dispatch_attempts_race_for_the_last_slot(store):
    holder = seed(store, "holder", created_at=NOW)
    left = seed(store, "left", created_at=NOW + timedelta(seconds=1))
    right = seed(store, "right", created_at=NOW + timedelta(seconds=2))
    adapter = ScriptedAdapter(REBOOT_PLAN)
    execute_workflow(
        budget_executor(store, adapter, TWO_PER_CLUSTER), holder.request_id
    )
    assert POSTGRES_URL is not None, "a second replica needs the database"
    replica = PostgresStore(POSTGRES_URL, initialize_schema=False)
    executors = {
        left.request_id: budget_executor(store, adapter, TWO_PER_CLUSTER),
        right.request_id: budget_executor(
            replica, adapter, TWO_PER_CLUSTER, executor_id="executor-b"
        ),
    }
    ready = Barrier(2)

    def dispatch(request_id: str) -> str:
        ready.wait(timeout=10)
        try:
            execute_workflow(executors[request_id], request_id)
        except RemediationBudgetError:
            return "refused"
        return "admitted"

    try:
        with ThreadPoolExecutor(max_workers=2) as workers:
            results = dict(
                zip(
                    (left.request_id, right.request_id),
                    workers.map(dispatch, (left.request_id, right.request_id)),
                )
            )
    finally:
        replica.close()

    assert sorted(results.values()) == ["admitted", "refused"], results
    winner = next(key for key, value in results.items() if value == "admitted")
    loser = next(key for key, value in results.items() if value == "refused")
    assert store.remediation_budget_occupancy().get(CLUSTER) == 2
    assert store.get_workflow(loser).status is WorkflowStatus.PENDING
    assert adapter.started() == {holder.request_id, winner}, adapter.calls


def test_a_lapsed_lease_still_holds_the_scope(store):
    holder = seed(store, "holder", created_at=NOW)
    waiter = seed(store, "waiter", created_at=NOW + timedelta(seconds=1))
    past = datetime.now(UTC) - timedelta(minutes=10)
    claims = remediation_budget_claims(
        ONE_PER_CLUSTER,
        holder,
        store.get_incident(holder.incident_id),
        holder.official_steps,
    )
    leased = store.claim_workflow(
        holder.request_id,
        "executor-gone",
        holder.fencing_token,
        now=past,
        lease_duration=timedelta(seconds=30),
        remediation_budget_claims=claims,
    )
    assert leased.execution_lease_expires_at is not None
    assert leased.execution_lease_expires_at < datetime.now(UTC)

    with pytest.raises(RemediationBudgetError) as refused:
        store.claim_workflow(
            waiter.request_id,
            "executor-b",
            waiter.fencing_token,
            remediation_budget_claims=claims,
        )
    assert refused.value.scope == CLUSTER
    assert store.remediation_budget_occupancy().get(CLUSTER) == 1


def test_a_parked_record_of_unknown_outcome_holds_until_an_operator_confirms(store):
    """The BLOCKED rows a claim and the occupancy read are decided by the
    shared rule in every storage mode."""

    parked = seed(store, "a", created_at=NOW)
    waiter = seed(store, "b", created_at=NOW + timedelta(seconds=1))
    adapter = UnknownOutcomeAdapter(REBOOT_PLAN, {parked.request_id})
    executor = budget_executor(store, adapter, ONE_PER_CLUSTER)
    assert execute_workflow(executor, parked.request_id).status is (
        WorkflowStatus.BLOCKED
    )
    assert store.get_workflow(parked.request_id).blocked_kind is (
        BlockedKind.NEEDS_OPERATOR
    )

    assert store.remediation_budget_occupancy().get(CLUSTER) == 1
    with pytest.raises(RemediationBudgetError) as refused:
        execute_workflow(executor, waiter.request_id)
    assert refused.value.scope == CLUSTER

    row = store.get_workflow(parked.request_id)
    store.save_workflow(
        operator_confirmed(row, now=NOW, node_id="node-a"), expected=row
    )
    assert store.remediation_budget_occupancy().get(CLUSTER, 0) == 0
    assert execute_workflow(executor, waiter.request_id).status is (
        WorkflowStatus.RUNNING
    )
    assert waiter.request_id in adapter.started(), adapter.calls


def test_the_handoff_reservation_keeps_exactly_one_hold(store):
    predecessor = seed(store, "a", created_at=NOW)
    adapter = ScriptedAdapter(REBOOT_PLAN)
    executor = budget_executor(store, adapter, ONE_PER_CLUSTER)
    execute_workflow(executor, predecessor.request_id)
    held = store.get_workflow(predecessor.request_id)
    successor = workflow_request(
        "workflow-s",
        predecessor.incident_id,
        fencing_token=predecessor.fencing_token,
        predecessor_workflow_id=predecessor.request_id,
        preempt_predecessor=True,
        preemption_reason="strictly stronger recovery action",
        runtime_profile_version="active-v1",
        official_steps=[step.model_copy() for step in held.official_steps],
        created_at=NOW + timedelta(seconds=2),
        updated_at=NOW + timedelta(seconds=2),
    )
    store.save_workflow(successor)
    other = seed(store, "b", created_at=NOW + timedelta(seconds=1))
    store.save_workflow_if_leased(
        held.model_copy(
            update={
                "status": WorkflowStatus.SUPERSEDED,
                "preempted_by_workflow_id": successor.request_id,
                "preemption_reason": "strictly stronger recovery action",
                "execution_owner_id": None,
                "execution_lease_expires_at": None,
            }
        ),
        held.execution_owner_id or "",
        held.execution_epoch,
    )

    assert store.remediation_budget_occupancy().get(CLUSTER) == 1
    other_claims = remediation_budget_claims(
        ONE_PER_CLUSTER,
        other,
        store.get_incident(other.incident_id),
        other.official_steps,
    )
    with pytest.raises(RemediationBudgetError):
        store.claim_workflow(
            other.request_id,
            "executor-b",
            other.fencing_token,
            remediation_budget_claims=other_claims,
        )

    claimed = store.claim_workflow(
        successor.request_id,
        "executor-a",
        successor.fencing_token,
        remediation_budget_claims=remediation_budget_claims(
            ONE_PER_CLUSTER,
            successor,
            store.get_incident(successor.incident_id),
            successor.official_steps,
        ),
    )
    assert claimed.status is WorkflowStatus.RUNNING
    assert store.remediation_budget_occupancy().get(CLUSTER) == 1, (
        "the handoff counted the predecessor and the successor at once"
    )
    with pytest.raises(RemediationBudgetError):
        store.claim_workflow(
            other.request_id,
            "executor-b",
            other.fencing_token,
            remediation_budget_claims=other_claims,
        )
    _end(store, successor.request_id)
    assert store.remediation_budget_occupancy().get(CLUSTER, 0) == 0
    admitted = store.claim_workflow(
        other.request_id,
        "executor-b",
        other.fencing_token,
        remediation_budget_claims=other_claims,
    )
    assert CLUSTER in admitted.remediation_budget_claims


def _blocked_shape(kind: BlockedKind, details: dict[str, bool]) -> dict[str, object]:
    """A BLOCKED row of ``kind`` whose reboot failed with ``details``."""

    return {
        "status": WorkflowStatus.BLOCKED,
        "blocked_kind": kind,
        "completed_step_indexes": [0],
        "step_executions": [
            workflow_step_execution(0, STOP),
            workflow_step_execution(
                1, REBOOT, WorkflowStepStatus.FAILED, details=dict(details)
            ),
        ],
    }


def test_occupancy_matches_the_shared_rule_on_every_row_shape(store):
    shapes = {
        "running": {"status": WorkflowStatus.RUNNING},
        "lapsed": {
            "status": WorkflowStatus.RUNNING,
            "execution_owner_id": "executor-gone",
            "execution_lease_expires_at": NOW,
        },
        "blocked": {"status": WorkflowStatus.BLOCKED},
        "blocked-parked": _blocked_shape(BlockedKind.NEEDS_OPERATOR, UNKNOWN_OUTCOME),
        "blocked-answered": _blocked_shape(BlockedKind.NEEDS_OPERATOR, {}),
        "blocked-settled": _blocked_shape(BlockedKind.SAFETY_SETTLED, UNKNOWN_OUTCOME),
        "blocked-internal": _blocked_shape(BlockedKind.INTERNAL_ERROR, {}),
        "failed": {"status": WorkflowStatus.FAILED},
        "pending-stale": {"status": WorkflowStatus.PENDING},
        "superseded-alone": {"status": WorkflowStatus.SUPERSEDED},
    }
    for index, (name, values) in enumerate(shapes.items()):
        workflow = seed(store, name, created_at=NOW + timedelta(seconds=index))
        store.save_workflow(
            copy_model(
                workflow,
                remediation_budget_claims=[CLUSTER, "region", f"node:{name}"],
                **values,
            )
        )
    predecessor = seed(store, "pred", created_at=NOW + timedelta(seconds=10))
    store.save_workflow(
        copy_model(
            predecessor,
            status=WorkflowStatus.SUPERSEDED,
            preempted_by_workflow_id="workflow-succ",
            remediation_budget_claims=[CLUSTER, "node:pred"],
        )
    )
    store.save_workflow(
        workflow_request(
            "workflow-succ",
            predecessor.incident_id,
            predecessor_workflow_id=predecessor.request_id,
            preempt_predecessor=True,
            official_steps=list(predecessor.official_steps),
            created_at=NOW + timedelta(seconds=11),
            updated_at=NOW + timedelta(seconds=11),
        )
    )

    occupancy = store.remediation_budget_occupancy()

    assert occupancy == dict(budget_occupancy(store.list_workflows(None, limit=1000)))
    assert occupancy == {
        CLUSTER: 4,
        "region": 3,
        "node:running": 1,
        "node:lapsed": 1,
        "node:blocked-parked": 1,
        "node:pred": 1,
    }


def test_a_hold_is_never_refused_and_keeps_what_it_holds(store):
    holder = seed(store, "holder", created_at=NOW)
    started = seed(store, "started", created_at=NOW + timedelta(seconds=1))
    execute_workflow(
        budget_executor(store, ScriptedAdapter(REBOOT_PLAN), ONE_PER_CLUSTER),
        holder.request_id,
    )
    store.save_workflow(
        copy_model(
            started,
            status=WorkflowStatus.RUNNING,
            completed_step_indexes=[0],
            completed_operations=[STOP],
            remediation_budget_claims=["domain:cluster-a:rack-1"],
            remediation_budget_limits={"domain:cluster-a:rack-1": 1},
        )
    )

    claimed = store.claim_workflow(
        started.request_id,
        "executor-a",
        started.fencing_token,
        remediation_budget_claims={CLUSTER: 1, "region": 20},
        remediation_budget_hold=True,
    )

    assert claimed.remediation_budget_claims == [
        CLUSTER,
        "domain:cluster-a:rack-1",
        "region",
    ]
    assert claimed.remediation_budget_limits == {
        CLUSTER: 1,
        "domain:cluster-a:rack-1": 1,
        "region": 20,
    }
    assert store.remediation_budget_occupancy().get(CLUSTER) == 2


def test_a_full_scope_storm_does_not_starve_another_cluster(store):
    """The dispatcher's one occupancy aggregate per tick, on PostgreSQL."""

    storm = [
        seed(store, f"storm-{index}", created_at=NOW + timedelta(milliseconds=index))
        for index in range(8)
    ]
    adapter = ScriptedAdapter(REBOOT_PLAN)
    dispatcher = budget_dispatcher(store, adapter, TWO_PER_CLUSTER, batch_size=4)
    for _ in range(3):
        dispatcher.run_once()
    waiting = [store.get_workflow(item.request_id) for item in storm]
    assert sum(item.status is WorkflowStatus.RUNNING for item in waiting) == 2
    assert {
        item.remediation_budget_last_blocked_scope
        for item in waiting
        if item.status is WorkflowStatus.PENDING
    } == {CLUSTER}

    late = seed(
        store, "late", cluster_id="cluster-b", created_at=NOW + timedelta(hours=1)
    )
    report = dispatcher.run_once()

    assert store.get_workflow(late.request_id).status is WorkflowStatus.RUNNING
    assert report.filtered.get("remediation_budget") == 6, report.filtered
    assert f"{late.request_id}/1/{REBOOT.value}" in adapter.calls
