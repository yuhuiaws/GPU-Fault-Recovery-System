"""The remediation storm breaker through the claim path (design §9.5).

The executor's claim judges a first admission of budgeted work against the
cluster's breaker in the store's own transaction: many unrelated nodes trip
it and the tripping admission is held, one job on many nodes never does,
work in flight goes on, the trip is persisted (restarts and replicas see it),
one operator notice is sent per trip, the dispatcher passes held rows over,
and only an operator's release lets them run.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.execution import storm_admission
from gpu_fault.execution.remediation_budget import RemediationBudgetPolicy
from gpu_fault.models import (
    IncidentState,
    WorkflowEventCode,
    WorkflowOperation,
    WorkflowStatus,
)
from gpu_fault.remediation_storm import StormBreakerState, released_breaker, storm_scope
from gpu_fault.store import InMemoryStore, SqliteStore
from tests._builders import copy_model, fault_incident, workflow_request, workflow_step
from tests.execution._budget_hold_support import (
    REBOOT_PLAN,
    ScriptedAdapter,
    all_workflows,
    budget_dispatcher,
    seed,
)

# The shipped policy: no fixed count caps a cluster, the breaker bounds it.
POLICY = RemediationBudgetPolicy()
SCOPE = storm_scope("cluster-a")


def _store(tmp_path, durable: bool):
    return SqliteStore(str(tmp_path / "storm.db")) if durable else InMemoryStore()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _seed_idle(store, count: int, *, offset: int = 0) -> list[str]:
    start = _now() - timedelta(seconds=30)
    return [
        seed(
            store, f"idle-{index}", created_at=start + timedelta(milliseconds=index)
        ).request_id
        for index in range(offset, offset + count)
    ]


def _codes(workflow) -> list[str | None]:
    return [event.code for event in workflow.events]


@pytest.mark.parametrize("durable", [False, True])
def test_many_idle_nodes_trip_it_and_the_tripping_admission_is_held(
    tmp_path, durable: bool
) -> None:
    store = _store(tmp_path, durable)
    ids = _seed_idle(store, 3)
    adapter = ScriptedAdapter(REBOOT_PLAN)
    dispatcher = budget_dispatcher(store, adapter, POLICY, max_workers=1)

    dispatcher.run_once()

    rows = {row.request_id: row for row in all_workflows(store)}
    assert [rows[i].status for i in ids] == [
        WorkflowStatus.RUNNING,
        WorkflowStatus.RUNNING,
        WorkflowStatus.PENDING,
    ]
    held = rows[ids[2]]
    assert held.remediation_budget_last_blocked_scope == SCOPE
    assert WorkflowEventCode.REMEDIATION_STORM_TRIPPED.value in _codes(held)
    assert WorkflowEventCode.REMEDIATION_BUDGET_WAIT.value in _codes(held)
    breaker = store.get_remediation_storm_breaker("cluster-a")
    assert breaker is not None and breaker.state is StormBreakerState.TRIPPED
    assert breaker.trip_node_ids == ["node-idle-0", "node-idle-1", "node-idle-2"]
    assert breaker.trip_threshold == 3, "no fleet registered: the floor"
    assert adapter.started() == set(ids[:2])


def test_one_job_on_many_nodes_never_trips_it() -> None:
    store = InMemoryStore()
    nodes = [f"node-{index}" for index in range(6)]
    incident = fault_incident(
        "incident-job",
        "event-job",
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

    budget_dispatcher(store, ScriptedAdapter(REBOOT_PLAN), POLICY).run_once()

    assert store.get_workflow(workflow.request_id).status is WorkflowStatus.RUNNING
    breaker = store.get_remediation_storm_breaker("cluster-a")
    assert breaker is not None and breaker.state is StormBreakerState.CLOSED
    assert sorted({entry.node_id for entry in breaker.window}) == nodes


def test_two_nodes_never_trip_it() -> None:
    store = InMemoryStore()
    _seed_idle(store, 2)

    budget_dispatcher(store, ScriptedAdapter(REBOOT_PLAN), POLICY).run_once()

    breaker = store.get_remediation_storm_breaker("cluster-a")
    assert breaker is not None and breaker.state is StormBreakerState.CLOSED


def test_a_large_cluster_trips_at_twenty_percent(monkeypatch) -> None:
    monkeypatch.setattr(storm_admission, "cluster_gpu_nodes", lambda _s, _c: 100)
    store = InMemoryStore()
    ids = _seed_idle(store, 21)

    budget_dispatcher(
        store, ScriptedAdapter(REBOOT_PLAN), POLICY, max_workers=1
    ).run_once()

    statuses = [store.get_workflow(i).status for i in ids]
    assert statuses.count(WorkflowStatus.RUNNING) == 19
    breaker = store.get_remediation_storm_breaker("cluster-a")
    assert breaker is not None and breaker.trip_threshold == 20


def test_work_in_flight_goes_on_while_it_is_open() -> None:
    store = InMemoryStore()
    ids = _seed_idle(store, 3)
    adapter = ScriptedAdapter(REBOOT_PLAN)
    dispatcher = budget_dispatcher(store, adapter, POLICY, max_workers=1)
    dispatcher.run_once()
    assert store.get_remediation_storm_breaker("cluster-a").open, "tripped"

    # The two admitted reboots complete under the open breaker.
    adapter.released.update(ids[:2])
    dispatcher.run_once()

    assert [store.get_workflow(i).status for i in ids] == [
        WorkflowStatus.SUCCEEDED,
        WorkflowStatus.SUCCEEDED,
        WorkflowStatus.PENDING,
    ]


def test_the_dispatcher_passes_held_rows_over_and_records_new_ones() -> None:
    store = InMemoryStore()
    ids = _seed_idle(store, 3)
    adapter = ScriptedAdapter(REBOOT_PLAN)
    dispatcher = budget_dispatcher(store, adapter, POLICY, max_workers=1)
    dispatcher.run_once()
    fresh = _seed_idle(store, 1, offset=10)[0]

    dispatcher.run_once()

    assert dispatcher.filtered_total.get("remediation_budget", 0) >= 2
    recorded = store.get_workflow(fresh)
    assert recorded.status is WorkflowStatus.PENDING
    assert recorded.remediation_budget_last_blocked_scope == SCOPE
    assert recorded.execution_epoch == 0, "recorded by the guard, never claimed"
    assert store.get_workflow(ids[2]).remediation_budget_wait_count == 1
    assert adapter.started() == set(ids[:2]), "only the admitted two ran steps"


def test_one_operator_notice_per_trip() -> None:
    store = InMemoryStore()
    _seed_idle(store, 3)
    dispatcher = budget_dispatcher(
        store, ScriptedAdapter(REBOOT_PLAN), POLICY, max_workers=1
    )
    sent: list[str] = []
    dispatcher.executor.notification_sender = sent.append

    dispatcher.run_once()  # trips it
    dispatcher.run_once()  # the sweep sends the notice
    dispatcher.run_once()  # and never again

    breaker = store.get_remediation_storm_breaker("cluster-a")
    assert breaker.notified_trip_id == breaker.trip_id
    assert len(sent) == 1
    notice = store.get_notification(sent[0])
    assert notice.deduplication_key == (
        f"cluster-a/remediation-storm-tripped/{breaker.trip_id}"
    )
    assert "remediation-storm release" in notice.body_text


def test_release_admits_the_held_rows_and_re_arms_it() -> None:
    store = InMemoryStore()
    ids = _seed_idle(store, 3)
    adapter = ScriptedAdapter(REBOOT_PLAN)
    dispatcher = budget_dispatcher(store, adapter, POLICY, max_workers=1)
    dispatcher.run_once()
    breaker = store.get_remediation_storm_breaker("cluster-a")

    store.replace_remediation_storm_breaker(
        released_breaker(
            breaker,
            operator="alice",
            reason="driver rollout reverted",
            reference="CHG-1",
            now=_now(),
        ),
        expected=breaker,
    )
    dispatcher.run_once()

    assert store.get_workflow(ids[2]).status is WorkflowStatus.RUNNING
    after = store.get_remediation_storm_breaker("cluster-a")
    assert after.state is StormBreakerState.CLOSED
    assert after.window == [], "the held row is admitted without counting again"


def test_an_open_breaker_survives_a_restart(tmp_path) -> None:
    path = str(tmp_path / "storm.db")
    store = SqliteStore(path)
    _seed_idle(store, 3)
    budget_dispatcher(
        store, ScriptedAdapter(REBOOT_PLAN), POLICY, max_workers=1
    ).run_once()

    restarted = SqliteStore(path)
    late = _seed_idle(restarted, 1, offset=20)[0]
    budget_dispatcher(restarted, ScriptedAdapter(REBOOT_PLAN), POLICY).run_once()

    breaker = restarted.get_remediation_storm_breaker("cluster-a")
    assert breaker is not None and breaker.state is StormBreakerState.TRIPPED
    assert restarted.get_workflow(late).status is WorkflowStatus.PENDING


def test_only_a_first_admission_is_judged() -> None:
    incident = fault_incident("incident-x", "event-x", job_id="job-7")
    claims = {"node:cluster-a:node-a": 1, "domain:cluster-a:partition-1": 1}
    steps = [workflow_step(WorkflowOperation.RESET_GPU, node_ids=["node-a"])]
    fresh = workflow_request("workflow-x", "incident-x", official_steps=steps)
    running = copy_model(
        fresh, status=WorkflowStatus.RUNNING, remediation_budget_claims=sorted(claims)
    )
    successor = copy_model(fresh, preempt_predecessor=True)
    store = InMemoryStore()

    candidate = storm_admission.storm_candidate(
        store, fresh, incident, claims, steps, hold=False
    )
    assert candidate is not None
    assert candidate.node_ids == frozenset({"node-a"})
    assert candidate.domain == "job:job-7"
    for workflow, hold in ((fresh, True), (running, False), (successor, False)):
        assert (
            storm_admission.storm_candidate(
                store, workflow, incident, claims, steps, hold=hold
            )
            is None
        ), (workflow.status, workflow.preempt_predecessor, hold)
    assert (
        storm_admission.storm_candidate(store, fresh, incident, {}, steps, hold=False)
        is None
    )
    # A claim whose only budgeted step is a workload stop/restart brings no node
    # into destructive NODE remediation, so the breaker does not judge it (b223).
    stop_only = [
        workflow_step(WorkflowOperation.STOP_WORKLOADS, node_ids=["node-a", "node-b"])
    ]
    stop_workflow = workflow_request(
        "workflow-stop", "incident-x", official_steps=stop_only
    )
    assert (
        storm_admission.storm_candidate(
            store, stop_workflow, incident, claims, stop_only, hold=False
        )
        is None
    )


# --------------------------------------------------------------------------- #
# A job's workload-stop/restart peers are not counted (b223)
# --------------------------------------------------------------------------- #
def _seed_job(
    store,
    suffix: str,
    nodes: list[str],
    *,
    created_at: datetime,
    faulted: str | None = None,
):
    """A job workflow as the job family compiles it for a reboot-class fault on
    one node: STOP_WORKLOADS / RESTART_WORKLOAD across the whole allocation, a
    RESTART_NODE (reboot) on the faulted node only."""

    faulted = faulted or nodes[0]
    workload = f"default/pytorchjob/{suffix}"
    incident = fault_incident(
        f"incident-{suffix}",
        f"event-{suffix}",
        cluster_id="cluster-a",
        node_ids=[faulted],
        job_id=f"job-{suffix}",
        attempt_id=f"{suffix}-a1",
        official_action="RESTART_BM",
        state=IncidentState.ACTION_PENDING,
        fencing_token=3,
        created_at=created_at,
        updated_at=created_at,
    )
    workflow = workflow_request(
        f"workflow-{suffix}",
        incident.incident_id,
        runtime_profile_version="active-v1",
        official_action="RESTART_BM",
        official_steps=[
            workflow_step(
                WorkflowOperation.STOP_WORKLOADS,
                node_ids=list(nodes),
                workload_ids=[workload],
            ),
            workflow_step(WorkflowOperation.RESTART_NODE, node_ids=[faulted]),
            workflow_step(
                WorkflowOperation.RESTART_WORKLOAD,
                node_ids=list(nodes),
                workload_ids=[workload],
                parameters={
                    "cluster_id": "cluster-a",
                    "job_id": f"job-{suffix}",
                    "source_attempt_id": f"{suffix}-a1",
                    "source_gpu_count": 8 * len(nodes),
                    "restart_budget": 1,
                },
            ),
        ],
        created_at=created_at,
        updated_at=created_at,
    )
    store.save_incident_and_workflow(
        incident.model_copy(update={"workflow_request_id": workflow.request_id}),
        workflow,
    )
    return workflow


def _window_nodes(store) -> list[str]:
    breaker = store.get_remediation_storm_breaker("cluster-a")
    assert breaker is not None
    return sorted(entry.node_id for entry in breaker.window)


def test_a_two_node_job_counts_only_its_faulted_node() -> None:
    store = InMemoryStore()
    _seed_job(store, "j", ["node-0", "node-1"], faulted="node-0", created_at=_now())

    budget_dispatcher(store, ScriptedAdapter(REBOOT_PLAN), POLICY).run_once()

    breaker = store.get_remediation_storm_breaker("cluster-a")
    assert breaker is not None and breaker.state is StormBreakerState.CLOSED
    # node-1 gets STOP_WORKLOADS/RESTART_WORKLOAD only (the job is stopped as a
    # whole), so it is not counted; only the rebooted node-0 is.
    assert _window_nodes(store) == ["node-0"]
    assert store.get_workflow("workflow-j").status is WorkflowStatus.RUNNING


def test_two_two_node_jobs_stay_below_the_floor() -> None:
    store = InMemoryStore()
    start = _now() - timedelta(seconds=30)
    _seed_job(store, "a", ["a0", "a1"], faulted="a0", created_at=start)
    _seed_job(
        store, "b", ["b0", "b1"], faulted="b0", created_at=start + timedelta(seconds=1)
    )

    budget_dispatcher(
        store, ScriptedAdapter(REBOOT_PLAN), POLICY, max_workers=1
    ).run_once()

    breaker = store.get_remediation_storm_breaker("cluster-a")
    assert breaker is not None and breaker.state is StormBreakerState.CLOSED
    # Two faulted nodes across two job domains: 2 < the floor of 3, so both run
    # in parallel. Before b223 each job's healthy peer counted too (4 >= 3) and
    # the second job was held.
    assert _window_nodes(store) == ["a0", "b0"]
    assert [
        store.get_workflow(item).status for item in ("workflow-a", "workflow-b")
    ] == [WorkflowStatus.RUNNING, WorkflowStatus.RUNNING]


def test_two_sixty_four_node_jobs_do_not_trip_at_the_scale_threshold(
    monkeypatch,
) -> None:
    # A ~500 GPU-node cluster: threshold = ceil(0.2 * 500) = 100.
    monkeypatch.setattr(storm_admission, "cluster_gpu_nodes", lambda _s, _c: 500)
    store = InMemoryStore()
    start = _now() - timedelta(seconds=30)
    _seed_job(store, "a", [f"a{i}" for i in range(64)], faulted="a0", created_at=start)
    _seed_job(
        store,
        "b",
        [f"b{i}" for i in range(64)],
        faulted="b0",
        created_at=start + timedelta(seconds=1),
    )

    budget_dispatcher(
        store, ScriptedAdapter(REBOOT_PLAN), POLICY, max_workers=1
    ).run_once()

    breaker = store.get_remediation_storm_breaker("cluster-a")
    assert breaker is not None and breaker.state is StormBreakerState.CLOSED
    # One bad node per 64-node job: 2 counted, far below 100. Before b223 the
    # two whole allocations counted 128 >= 100 and false-tripped the breaker.
    assert _window_nodes(store) == ["a0", "b0"]
    assert breaker.threshold == 100


def test_three_one_faulted_node_jobs_still_trip_the_floor() -> None:
    store = InMemoryStore()
    start = _now() - timedelta(seconds=30)
    for index, suffix in enumerate("abc"):
        _seed_job(
            store,
            suffix,
            [f"{suffix}-n0", f"{suffix}-n1"],
            faulted=f"{suffix}-n0",
            created_at=start + timedelta(seconds=index),
        )

    budget_dispatcher(
        store, ScriptedAdapter(REBOOT_PLAN), POLICY, max_workers=1
    ).run_once()

    breaker = store.get_remediation_storm_breaker("cluster-a")
    assert breaker is not None and breaker.state is StormBreakerState.TRIPPED
    # One rebooted node per job, three job domains: a genuine correlated storm
    # (bad image/policy) still reaches the floor of 3 and holds the third.
    assert (breaker.trip_node_ids, breaker.trip_domains) == (
        ["a-n0", "b-n0", "c-n0"],
        3,
    )
    assert breaker.trip_threshold == 3
    assert store.get_workflow("workflow-c").status is WorkflowStatus.PENDING


def test_a_quarantine_still_counts_toward_the_breaker() -> None:
    # F-C5: a false-positive isolation cascade must still be bounded. QUARANTINE
    # is containment, not a node-hardware mutation, but the breaker counts its
    # first admission (remediation_budget._budgeted); b223 does not change that.
    incident = fault_incident("incident-q", "event-q", node_ids=["node-q"])
    steps = [workflow_step(WorkflowOperation.QUARANTINE, node_ids=["node-q"])]
    workflow = workflow_request("workflow-q", "incident-q", official_steps=steps)

    candidate = storm_admission.storm_candidate(
        InMemoryStore(),
        workflow,
        incident,
        {"node:cluster-a:node-q": 1},
        steps,
        hold=False,
    )

    assert candidate is not None and candidate.node_ids == frozenset({"node-q"})
