"""The remediation storm breaker on PostgreSQL, in each workflow storage mode.

A claim judges its cluster's breaker inside its own transaction, under the
``remediation_storm/<cluster>`` advisory lock that sorts last in the budget
lock batch (F-B2 order), and an operator's release replaces the record by
compare-and-set under the same key (``store/postgres/workflows.py`` and
``store/shared/remediation_storm_records.py``). The memory and SQLite halves
are ``tests/execution/test_remediation_storm_admission.py``; these cases run
the same decisions on the PostgreSQL store and add what only a shared
database shows: two replicas reaching the tripping admission at once, and the
lock a claim waits for while it already holds its node's.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from threading import Barrier

import pytest

from gpu_fault.execution import storm_admission
from gpu_fault.execution.remediation_budget import RemediationBudgetPolicy
from gpu_fault.fleet import AgentLifecycleState, AgentRecord
from gpu_fault.models import WorkflowEventCode, WorkflowOperation, WorkflowStatus
from gpu_fault.remediation_storm import StormBreakerState, released_breaker, storm_scope
from gpu_fault.store import PostgresStore
from gpu_fault.store.shared.errors import RemediationBudgetError, StaleWriteError
from gpu_fault.store.shared.remediation_storm_records import storm_lock_key
from tests._builders import (
    execute_workflow,
    fault_incident,
    workflow_request,
    workflow_step,
)
from tests.execution._budget_hold_support import (
    REBOOT_PLAN,
    ScriptedAdapter,
    budget_dispatcher,
    budget_executor,
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
# The shipped policy: no fixed count caps a cluster, the breaker bounds it.
POLICY = RemediationBudgetPolicy()
CLUSTER = "cluster-a"
SCOPE = storm_scope(CLUSTER)
# The advisory locks on one key text as the store hashes it
# (hashtextextended(key, 0)); a bigint key shows its high half in classid.
_ADVISORY_LOCKS = """
    SELECT l.pid, l.granted
    FROM pg_locks AS l
    CROSS JOIN (SELECT hashtextextended(%s, 0) AS key) AS k
    WHERE l.locktype = 'advisory'
      AND l.objsubid = 1
      AND l.classid::bigint = ((k.key >> 32) & 4294967295)
      AND l.objid::bigint = (k.key & 4294967295)
"""


@pytest.fixture(params=["legacy", "dual", "dedicated"])
def store(migration_database, request):
    select_mode(migration_database, "workflow", request.param)
    assert POSTGRES_URL is not None, "storm breaker tests require local PostgreSQL"
    instance = PostgresStore(POSTGRES_URL, initialize_schema=False)
    try:
        yield instance
    finally:
        instance.close()


def _now() -> datetime:
    return datetime.now(UTC)


def _seed_idle(store, count: int, *, offset: int = 0, cluster_id: str = CLUSTER):
    """``count`` idle GPU nodes, each its own incident (domain) and node."""

    start = _now() - timedelta(seconds=30)
    return [
        seed(
            store,
            f"idle-{index}",
            cluster_id=cluster_id,
            created_at=start + timedelta(milliseconds=index),
        ).request_id
        for index in range(offset, offset + count)
    ]


def _codes(workflow) -> list[str | None]:
    return [event.code for event in workflow.events]


def _trip(store, adapter: ScriptedAdapter) -> list[str]:
    """Three idle nodes, admitted one by one: the third trips the breaker."""

    ids = _seed_idle(store, 3)
    budget_dispatcher(store, adapter, POLICY, max_workers=1).run_once()
    return ids


def _locks(connection, key: str) -> list[tuple[int, bool]]:
    return [tuple(row) for row in connection.execute(_ADVISORY_LOCKS, (key,))]


def test_idle_nodes_trip_it_and_the_tripping_admission_is_held(store):
    adapter = ScriptedAdapter(REBOOT_PLAN)

    ids = _trip(store, adapter)

    rows = [store.get_workflow(request_id) for request_id in ids]
    assert [row.status for row in rows] == [
        WorkflowStatus.RUNNING,
        WorkflowStatus.RUNNING,
        WorkflowStatus.PENDING,
    ]
    held = rows[2]
    assert held.remediation_budget_last_blocked_scope == SCOPE
    assert WorkflowEventCode.REMEDIATION_STORM_TRIPPED.value in _codes(held)
    assert WorkflowEventCode.REMEDIATION_BUDGET_WAIT.value in _codes(held)
    assert held.remediation_budget_claims == [], "a held admission claims nothing"
    breaker = store.get_remediation_storm_breaker(CLUSTER)
    assert breaker is not None and breaker.state is StormBreakerState.TRIPPED
    assert breaker.trip_node_ids == ["node-idle-0", "node-idle-1", "node-idle-2"]
    assert breaker.trip_threshold == 3, "no fleet registered: the floor"
    assert breaker.trip_workflow_id == ids[2]
    assert adapter.started() == set(ids[:2]), adapter.calls
    assert store.list_remediation_storm_breakers() == [breaker]


def test_release_by_compare_and_set_admits_the_held_row_and_re_arms_it(store):
    adapter = ScriptedAdapter(REBOOT_PLAN)
    ids = _trip(store, adapter)
    tripped = store.get_remediation_storm_breaker(CLUSTER)
    assert tripped is not None and tripped.open, "the third admission tripped it"
    released = released_breaker(
        tripped,
        operator="alice",
        reason="driver rollout reverted",
        reference="CHG-1",
        now=_now(),
    )

    with pytest.raises(StaleWriteError):
        store.replace_remediation_storm_breaker(
            released, expected=tripped.model_copy(update={"window": []})
        )
    assert store.get_remediation_storm_breaker(CLUSTER) == tripped
    store.replace_remediation_storm_breaker(released, expected=tripped)
    budget_dispatcher(store, adapter, POLICY, max_workers=1).run_once()

    assert store.get_workflow(ids[2]).status is WorkflowStatus.RUNNING
    after = store.get_remediation_storm_breaker(CLUSTER)
    assert after is not None and after.state is StormBreakerState.CLOSED
    assert after.window == [], "the held row is admitted without counting again"
    assert after.released_by == "alice"


def test_one_job_on_many_nodes_never_trips_it(store):
    nodes = [f"node-{index}" for index in range(5)]
    incident = fault_incident(
        "incident-job",
        "event-job",
        cluster_id=CLUSTER,
        node_ids=nodes,
        job_id="job-7",
        attempt_id="a-1",
        official_action="RESTART_BM",
        fencing_token=3,
    )
    workflow = workflow_request(
        "workflow-job",
        incident.incident_id,
        runtime_profile_version="active-v1",
        official_steps=[
            workflow_step(WorkflowOperation.STOP_WORKLOADS, node_ids=nodes),
            *[
                workflow_step(WorkflowOperation.RESTART_NODE, node_ids=[node])
                for node in nodes
            ],
        ],
    )
    store.save_incident_and_workflow(
        incident.model_copy(update={"workflow_request_id": workflow.request_id}),
        workflow,
    )

    execute_workflow(
        budget_executor(store, ScriptedAdapter(REBOOT_PLAN), POLICY),
        workflow.request_id,
    )

    assert store.get_workflow(workflow.request_id).status is WorkflowStatus.RUNNING
    breaker = store.get_remediation_storm_breaker(CLUSTER)
    assert breaker is not None and breaker.state is StormBreakerState.CLOSED
    assert sorted({entry.node_id for entry in breaker.window}) == nodes
    assert {entry.domain for entry in breaker.window} == {"job:job-7"}


def test_two_replicas_reaching_the_tripping_admission_trip_it_once(store):
    """The third and fourth idle node are claimed at once by two replicas:
    the breaker's lock orders them, so one trips it and the other finds it
    open -- one trip, both held, nothing started."""

    adapter = ScriptedAdapter(REBOOT_PLAN)
    admitted = _seed_idle(store, 2)
    budget_dispatcher(store, adapter, POLICY, max_workers=1).run_once()
    racing = _seed_idle(store, 2, offset=2)
    assert POSTGRES_URL is not None, "a second replica needs the database"
    replica = PostgresStore(POSTGRES_URL, initialize_schema=False)
    executors = {
        racing[0]: budget_executor(store, adapter, POLICY),
        racing[1]: budget_executor(replica, adapter, POLICY, executor_id="executor-b"),
    }
    ready = Barrier(2)

    def dispatch(request_id: str) -> str:
        ready.wait(timeout=10)
        try:
            execute_workflow(executors[request_id], request_id)
        except RemediationBudgetError as refused:
            return refused.scope or "refused"
        return "admitted"

    try:
        with ThreadPoolExecutor(max_workers=2) as workers:
            results = list(workers.map(dispatch, racing))
    finally:
        replica.close()

    assert results == [SCOPE, SCOPE], results
    rows = [store.get_workflow(request_id) for request_id in racing]
    assert [row.status for row in rows] == [WorkflowStatus.PENDING] * 2
    trips = [
        row.request_id
        for row in rows
        if WorkflowEventCode.REMEDIATION_STORM_TRIPPED.value in _codes(row)
    ]
    assert len(trips) == 1, [_codes(row) for row in rows]
    breaker = store.get_remediation_storm_breaker(CLUSTER)
    assert breaker is not None and breaker.state is StormBreakerState.TRIPPED
    assert breaker.trip_workflow_id == trips[0]
    tripping_node = "node-" + trips[0].removeprefix("workflow-")
    assert breaker.trip_node_ids == sorted(
        ["node-idle-0", "node-idle-1", tripping_node]
    )
    assert adapter.started() == set(admitted), adapter.calls


def test_a_claim_waits_for_the_breaker_lock_holding_its_node_lock(store, database):
    """An operator write holds ``remediation_storm/<cluster>``: a first
    admission in that cluster waits for it with its node's budget lock already
    granted (the breaker's lock sorts last), and a claim in another cluster
    does not wait at all."""

    import psycopg

    held_here = _seed_idle(store, 1)[0]
    elsewhere = _seed_idle(store, 1, offset=1, cluster_id="cluster-b")[0]
    node_key = f"remediation_budget/node:{CLUSTER}:node-idle-0"
    storm_key = storm_lock_key(CLUSTER)
    adapter = ScriptedAdapter(REBOOT_PLAN)
    assert POSTGRES_URL is not None, "the operator's session needs the database"
    operator = psycopg.connect(POSTGRES_URL)
    try:
        operator.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (storm_key,)
        )
        with ThreadPoolExecutor(max_workers=1) as workers:
            waiting = workers.submit(
                execute_workflow, budget_executor(store, adapter, POLICY), held_here
            )
            deadline = time.monotonic() + 20
            while not any(not granted for _pid, granted in _locks(database, storm_key)):
                assert time.monotonic() < deadline, "the claim never queued on the lock"
                assert not waiting.done(), "the claim did not wait for the breaker"
                time.sleep(0.05)
            (claimer,) = [
                pid for pid, granted in _locks(database, storm_key) if not granted
            ]
            assert (claimer, True) in _locks(database, node_key), (
                "the node's budget lock is taken before the breaker's"
            )

            other = execute_workflow(budget_executor(store, adapter, POLICY), elsewhere)
            assert other.status is WorkflowStatus.RUNNING, "cluster-b has its own lock"
            assert not waiting.done(), "cluster-a's claim still waits"

            operator.rollback()
            assert waiting.result(timeout=30).status is WorkflowStatus.RUNNING
    finally:
        operator.close()

    breaker = store.get_remediation_storm_breaker(CLUSTER)
    assert breaker is not None and breaker.state is StormBreakerState.CLOSED
    assert [entry.node_id for entry in breaker.window] == ["node-idle-0"]


def _agent(
    node_id: str, state: AgentLifecycleState, *, cluster_id: str = CLUSTER
) -> AgentRecord:
    now = _now()
    return AgentRecord(
        cluster_id=cluster_id,
        node_id=node_id,
        endpoint="https://10.0.0.7:9099",
        agent_version="1.0.0",
        artifact_sha256="a" * 64,
        policy_version="610",
        runtime_profile_version="active-v1",
        config_digest="b" * 64,
        allowed_operations=[],
        first_seen_at=now,
        last_seen_at=now,
        lifecycle_state=state,
    )


def test_the_breaker_counts_the_clusters_gpu_nodes_in_the_database(
    store, database, monkeypatch
):
    """``count_agents`` answers what listing the cluster and counting would
    (batch 246): its ACTIVE and DRAINING Agents, never a REVOKED one or another
    cluster's, and a payload without ``lifecycle_state`` as the ACTIVE it
    decodes to. The breaker's GPU-node count asks it and never lists (and so
    never decodes) the cluster's Agents."""

    active, draining, revoked = (
        AgentLifecycleState.ACTIVE,
        AgentLifecycleState.DRAINING,
        AgentLifecycleState.REVOKED,
    )
    for index, state in enumerate([active] * 3 + [draining] * 2 + [revoked] * 2):
        store.save_agent(_agent(f"node-{index}", state))
    store.save_agent(_agent("node-0", active, cluster_id="cluster-b"))
    database.execute(
        "UPDATE gpu_fault_objects SET payload = payload - 'lifecycle_state' "
        "WHERE kind='agent' AND key=%s",
        (f"{CLUSTER}/node-0",),
    )
    counted = {active, draining}
    listed = [
        agent
        for agent in store.list_agents(CLUSTER)
        if agent.lifecycle_state in counted
    ]

    assert store.count_agents(CLUSTER, counted) == len(listed) == 5
    assert store.count_agents(CLUSTER, [revoked]) == 2
    assert store.count_agents(CLUSTER, []) == 0
    assert store.count_agents("cluster-none", counted) == 0

    def listing(*_args, **_kwargs):
        raise AssertionError("the breaker's count listed the cluster's Agents")

    monkeypatch.setattr(store, "list_agents", listing)
    assert storm_admission.cluster_gpu_nodes(store, CLUSTER, now=time.monotonic()) == 5
