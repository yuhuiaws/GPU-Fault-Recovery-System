"""A perf run's synthetic healthy fleet (perf plan §8.5).

Since the fixed remediation limits were retired, a GPU cluster's remediation
storm breaker trips once max(3, ceil(20% of its ACTIVE/DRAINING Agents))
distinct nodes of two or more workload domains newly enter destructive
remediation within 600 s. A perf cluster that registers only the model's own
nodes sits at the floor of 3, so every action-bearing suite registers a healthy
fleet beside them: the cluster registers ``--fleet-nodes-per-cluster`` GPU
nodes (256 by default, threshold 52), never fewer than 5 x the model's + 1.
The fleet heartbeats for the whole run by compare-and-set, never becomes a
workflow node, and is purged with the run's clusters.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.app import ApplicationContext
from gpu_fault.execution import storm_admission
from gpu_fault.fleet import AgentLifecycleState
from gpu_fault.remediation_storm import (
    RemediationStormBreaker,
    StormBreakerState,
    StormCandidate,
    storm_threshold,
)
from gpu_fault.remediation_storm import storm_admission as admit
from scripts.perf import regional_storm_breaker as storm
from scripts.perf import regional_synthetic_fleet as fleet
from scripts.perf import seed_regional_action_workflows as seed
from scripts.perf import storm_breaker_pod, synthetic_fleet_pod
from scripts.perf.regional_validation_telemetry import session_input
from tests._builders import build_store
from tests.regional.test_regional_action_capacity import release_agent_identity
from tests.regional.test_validation_telemetry_watcher import FakeSession, Launcher

ROOT = Path(__file__).resolve().parents[2]
NOW = datetime(2026, 10, 29, 12, 0, tzinfo=timezone.utc)
RUN = "run-a"


def save_fleet(context: ApplicationContext, *, clusters: int, per_cluster: int) -> int:
    return seed.seed_fleet_agents(
        context,
        run_id=RUN,
        clusters=clusters,
        per_cluster=per_cluster,
        template=None,
        identity=release_agent_identity(),
        now=NOW,
    )


def model_agent(context: ApplicationContext, cluster_id: str, node_id: str) -> None:
    seed.save_synthetic_agent(
        context,
        template=None,
        identity=release_agent_identity(),
        cluster_id=cluster_id,
        node_id=node_id,
        run_id=RUN,
        now=NOW,
    )


@pytest.mark.parametrize(
    ("suite", "model_nodes"),
    [("action 32x4x4", 16), ("integrated", 4), ("correlated", 2)],
)
def test_the_default_fleet_puts_every_suite_far_below_a_threshold_of_52(
    suite: str, model_nodes: int
) -> None:
    plan = fleet.fleet_plan(
        clusters=32, model_nodes_per_cluster=model_nodes, nodes_per_cluster=256
    )

    assert plan["threshold_nodes"] == 52, suite
    assert plan["fleet_agents_per_cluster"] == 256 - model_nodes
    assert plan["fleet_agents"] == 32 * (256 - model_nodes)
    assert plan["model_margin_nodes"] == 52 - model_nodes > 0, suite
    assert "threshold 52 nodes in 600 s" in fleet.plan_line(plan)


def test_a_fleet_below_five_times_the_model_plus_one_is_refused() -> None:
    assert fleet.minimum_nodes(16) == 81
    fleet.fleet_plan(clusters=1, model_nodes_per_cluster=16, nodes_per_cluster=81)

    with pytest.raises(ValueError, match=r"below 81 = 5 x 16 model nodes \+ 1"):
        fleet.fleet_plan(clusters=1, model_nodes_per_cluster=16, nodes_per_cluster=80)


def test_the_storm_step_may_cross_the_threshold_but_not_the_cluster() -> None:
    plan = fleet.fleet_plan(
        clusters=1, model_nodes_per_cluster=64, nodes_per_cluster=256, storm=True
    )

    assert plan["fleet_agents_per_cluster"] == 192
    assert "cross it on purpose" in fleet.plan_line(plan)
    with pytest.raises(ValueError, match="the breaker step's own workflow nodes"):
        fleet.fleet_plan(
            clusters=1, model_nodes_per_cluster=64, nodes_per_cluster=63, storm=True
        )


def judge_model(
    *, workflows: int, nodes_per_workflow: int, cluster_nodes: int | None
) -> list[bool]:
    """The product's own decision over a model of job-less workflows."""

    record: RemediationStormBreaker | None = None
    admitted = []
    for index in range(workflows):
        record, refusal = admit(
            record,
            StormCandidate(
                cluster_id="perf-cap-000",
                workflow_id=f"workflow-{index}",
                node_ids=frozenset(
                    f"node-{index}-{node}" for node in range(nodes_per_workflow)
                ),
                domain=f"incident:{index}",
            ),
            now=NOW + timedelta(seconds=index),
            cluster_nodes=cluster_nodes,
        )
        admitted.append(refusal is None)
    return admitted


@pytest.mark.parametrize(
    ("workflows", "nodes_per_workflow"), [(4, 4), (4, 1), (2, 1)], ids=str
)
def test_the_product_admits_every_model_workflow_on_a_fleet_cluster(
    workflows: int, nodes_per_workflow: int
) -> None:
    model_nodes = workflows * nodes_per_workflow
    plan = fleet.fleet_plan(
        clusters=1, model_nodes_per_cluster=model_nodes, nodes_per_cluster=256
    )

    assert all(
        judge_model(
            workflows=workflows,
            nodes_per_workflow=nodes_per_workflow,
            cluster_nodes=plan["nodes_per_cluster"],
        )
    ), "the model must stay below the breaker of a production-sized cluster"


def test_without_the_fleet_the_action_and_integrated_models_trip_at_the_floor() -> None:
    """What the b220 lane measured: only the model's nodes registered."""

    action = judge_model(workflows=4, nodes_per_workflow=4, cluster_nodes=16)
    integrated = judge_model(workflows=4, nodes_per_workflow=1, cluster_nodes=4)
    correlated = judge_model(workflows=2, nodes_per_workflow=1, cluster_nodes=2)

    assert action == [True, False, False, False]
    assert integrated == [True, True, False, False]
    assert correlated == [True, True], "two nodes never reach the floor of 3"


def test_fleet_agents_carry_the_run_and_the_model_identity() -> None:
    context = ApplicationContext(store=build_store())

    created = save_fleet(context, clusters=2, per_cluster=3)

    assert created == 6
    agents = context.store.list_agents("perf-cap-001")
    assert [agent.node_id for agent in agents] == [
        f"fleet-node-{RUN}-c001-n{index:04d}" for index in range(3)
    ]
    assert {agent.lifecycle_state.value for agent in agents} == {"ACTIVE"}
    assert {agent.artifact_sha256 for agent in agents} == {"a" * 64}
    assert agents[0].lease_expires_at == NOW + timedelta(seconds=1800), (
        "the lease follows ACTION_AGENT_LEASE_SECONDS (default 1800)"
    )
    assert agents[0].boot_id == f"boot-{RUN}-{agents[0].node_id}"


def test_the_breaker_counts_the_fleet_beside_the_model() -> None:
    context = ApplicationContext(store=build_store())
    save_fleet(context, clusters=1, per_cluster=252)
    for index in range(4):
        model_agent(context, "perf-cap-000", seed.integrated_node_id(RUN, 0, index))

    nodes = storm_admission.cluster_gpu_nodes(context.store, "perf-cap-000")

    assert nodes == 256
    assert storm_threshold(nodes) == 52


def test_the_model_refresh_and_telemetry_skip_the_fleet() -> None:
    context = ApplicationContext(store=build_store())
    save_fleet(context, clusters=1, per_cluster=5)
    model_agent(context, "perf-cap-000", f"corr-node-{RUN}-c000-preempt")

    agents = seed.run_agents(context, run_id=RUN, clusters=1)
    refreshed = seed.refresh_run_agent_heartbeats(
        context,
        run_id=RUN,
        clusters=1,
        lease_seconds=60,
        publish_telemetry=True,
        clock=lambda: NOW + timedelta(minutes=5),
    )

    assert [record.node_id for _, record in agents] == [f"corr-node-{RUN}-c000-preempt"]
    assert refreshed == 1
    fleet_node = seed.fleet_node_id(RUN, 0, 0)
    assert context.store.get_agent("perf-cap-000", fleet_node).last_seen_at == NOW
    assert not context.store.list_collector_statuses("perf-cap-000", fleet_node), (
        "the fleet publishes no telemetry"
    )


class RecordingStore:
    """The seed's writes, in order, over a memory store: each workflow with the
    Agents its cluster had, and the stamps it carried beside the wall clock."""

    def __init__(self) -> None:
        self.inner = build_store()
        self.writes: list[tuple[str, int]] = []
        self.agents_saved_at: list[datetime] = []
        # (workflow.created_at, incident.created_at, when it was saved)
        self.stamps: list[tuple[datetime, datetime, datetime]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def save_agent(self, agent: Any) -> None:
        self.inner.save_agent(agent)
        self.agents_saved_at.append(datetime.now(timezone.utc))

    def save_incident_and_workflow(self, incident: Any, workflow: Any) -> None:
        self.writes.append(
            (workflow.request_id, len(self.inner.list_agents(incident.cluster_id)))
        )
        self.stamps.append(
            (workflow.created_at, incident.created_at, datetime.now(timezone.utc))
        )
        self.inner.save_incident_and_workflow(incident, workflow)


def run_seed_main(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    store: Any,
    environment: dict[str, str],
) -> dict[str, Any]:
    monkeypatch.setattr(
        seed,
        "ApplicationContext",
        SimpleNamespace(from_environment=lambda: ApplicationContext(store=store)),
    )
    monkeypatch.setattr(
        seed,
        "provider_and_pin_facts",
        lambda *_args, **_kwargs: {"synthetic_provider": "hyperpod"},
    )
    for name, value in {
        "ACTION_RUN_ID": RUN,
        "ACTION_CLUSTERS": "1",
        "ACTION_AGENT_IDENTITY_JSON": json.dumps(release_agent_identity()),
        **environment,
    }.items():
        monkeypatch.setenv(name, value)
    seed.main()
    return json.loads(capsys.readouterr().out.strip().splitlines()[-1])


def test_the_seed_saves_every_agent_before_any_workflow(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The breaker counts the cluster at its first admission and keeps the
    count for a minute: a workflow claimed while later workflows' Agents were
    still being saved would be judged against a smaller cluster."""

    store = RecordingStore()

    summary = run_seed_main(
        monkeypatch,
        capsys,
        store,
        {
            "ACTION_SEED_MODE": "workflows",
            "ACTION_WORKFLOWS_PER_CLUSTER": "3",
            "ACTION_NODES_PER_WORKFLOW": "4",
        },
    )

    assert summary["workflows_created"] == 3 and summary["agents_created"] == 12
    assert [count for _, count in store.writes] == [12, 12, 12], (
        "every workflow must be saved after all twelve Agents"
    )
    assert [
        request_id for request_id, _ in store.writes
    ] == storm_breaker_pod.step_workflow_ids(RUN, 3), (
        "the breaker step reads the seeded workflows by these ids"
    )


def test_each_seeded_workflow_is_stamped_when_it_is_saved(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The suite's first action and workflow duration run from a workflow's
    created_at. Stamped with the seed's start they also held the Agent template
    read over the registered fleet and every Agent save before the first
    workflow -- harness time (batch 246). Each workflow and its incident carry
    their own save time now, in save order, and the summary names the basis."""

    store = RecordingStore()

    summary = run_seed_main(
        monkeypatch,
        capsys,
        store,
        {
            "ACTION_SEED_MODE": "workflows",
            "ACTION_WORKFLOWS_PER_CLUSTER": "3",
            "ACTION_NODES_PER_WORKFLOW": "4",
        },
    )

    last_agent = max(store.agents_saved_at)
    assert len(store.stamps) == 3
    for created, incident_created, saved in store.stamps:
        assert last_agent <= created <= saved, (
            "a workflow carries its own save time, not the seed's start"
        )
        assert incident_created == created, "the incident is stamped with it"
    created = [created for created, _, _ in store.stamps]
    assert created == sorted(created), (
        "the dispatch order (created_at, key) stays the save order"
    )
    assert summary["first_action_basis"] == seed.FIRST_ACTION_BASIS == "workflow-saved"


def test_the_fleet_mode_registers_per_cluster_agents(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    store = build_store()

    summary = run_seed_main(
        monkeypatch,
        capsys,
        store,
        {
            "ACTION_SEED_MODE": seed.FLEET_AGENTS_MODE,
            "ACTION_FLEET_AGENTS_PER_CLUSTER": "7",
            "ACTION_AGENT_LEASE_SECONDS": "900",
        },
    )

    assert summary["agents_created"] == 7
    assert summary["node_id_prefix"] == f"fleet-node-{RUN}-"
    assert summary["lease_seconds"] == 900
    assert summary["agent_identity_source"] == "release-state"
    assert len(store.list_agents("perf-cap-000")) == 7


def test_the_pod_programs_spell_the_seed_ids() -> None:
    assert synthetic_fleet_pod.FLEET_NODE_PREFIX == seed.FLEET_NODE_PREFIX
    assert synthetic_fleet_pod.HEARTBEAT_MODE == fleet.HEARTBEAT_MODE
    assert synthetic_fleet_pod.CENSUS_MODE == fleet.CENSUS_MODE
    assert storm_breaker_pod.step_workflow_ids(RUN, 2) == [
        f"workflow-{seed.action_scope(RUN, 0, index)}" for index in range(2)
    ]


class ScriptedStop:
    """``threading.Event`` as the heartbeat loop uses it: each wait runs the
    next scripted action (``None``: keep going) and stops after the script."""

    def __init__(self, actions: list[Callable[[], None] | None]) -> None:
        self.actions = list(actions)
        self.waits: list[float] = []
        self.stopped = False

    def is_set(self) -> bool:
        return self.stopped

    def wait(self, timeout: float | None = None) -> bool:
        self.waits.append(float(timeout or 0.0))
        if not self.actions:
            self.stopped = True
            return True
        action = self.actions.pop(0)
        if action is not None:
            action()
        return False


class Ticks:
    def __init__(self) -> None:
        self.now = NOW

    def __call__(self) -> datetime:
        self.now += timedelta(seconds=1)
        return self.now


def heartbeat(store: Any, stop: Any, clock: Ticks, **overrides: Any) -> dict[str, Any]:
    options: dict[str, Any] = {
        "run_id": RUN,
        "clusters": 2,
        "lease_seconds": 900,
        "interval_seconds": 300,
        "deadline_seconds": 3600,
        "stop": stop,
        "clock": clock,
        "monotonic": lambda: 0.0,
        **overrides,
    }
    return synthetic_fleet_pod.watch_heartbeats(store, **options)


def test_the_heartbeat_renews_every_fleet_agent_by_compare_and_set() -> None:
    context = ApplicationContext(store=build_store())
    save_fleet(context, clusters=2, per_cluster=3)
    model_agent(context, "perf-cap-000", seed.integrated_node_id(RUN, 0, 0))
    clock = Ticks()
    stop = ScriptedStop([None, None])

    summary = heartbeat(context.store, stop, clock)

    assert summary["ended_by"] == "stopped"
    assert summary["passes"] == 1, "one whole pass, then the stop in the next"
    assert summary["renewed"] == 9, (
        "two passes over the first cluster, one over the second"
    )
    assert summary["not_renewed"] == 0
    assert stop.waits[:2] == [150.0, 150.0], "each cluster gets interval / clusters"
    fleet_agent = context.store.get_agent("perf-cap-001", seed.fleet_node_id(RUN, 1, 2))
    assert fleet_agent.last_seen_at > NOW
    assert fleet_agent.lease_expires_at == fleet_agent.last_seen_at + timedelta(
        seconds=900
    )
    model = context.store.get_agent("perf-cap-000", seed.integrated_node_id(RUN, 0, 0))
    assert model.last_seen_at == NOW, "the model's own Agents are not the fleet's"
    assert 0 < summary["longest_renewal_gap_seconds"] < 900


def test_a_purged_fleet_is_never_written_back_and_ends_the_session() -> None:
    context = ApplicationContext(store=build_store())
    save_fleet(context, clusters=2, per_cluster=2)

    def purge() -> None:
        for cluster_id in ("perf-cap-000", "perf-cap-001"):
            context.store.delete_regional_cluster(cluster_id)

    summary = heartbeat(context.store, ScriptedStop([None, purge, None, None]), Ticks())

    assert summary["ended_by"] == "fleet-gone"
    assert summary["passes"] == 2
    assert summary["last_pass"] == {}, "the second pass found no fleet Agent"
    assert context.store.list_agents() == [], "a purged Agent must stay purged"


def test_a_moved_agent_is_left_as_the_product_moved_it() -> None:
    context = ApplicationContext(store=build_store())
    save_fleet(context, clusters=1, per_cluster=2)
    node = seed.fleet_node_id(RUN, 0, 1)
    draining = context.store.get_agent("perf-cap-000", node).model_copy(
        update={"lifecycle_state": AgentLifecycleState.DRAINING}
    )
    context.store.save_agent(draining)

    summary = heartbeat(context.store, ScriptedStop([]), Ticks(), clusters=1)

    assert summary["renewed"] == 1 and summary["not_renewed"] == 1
    assert context.store.get_agent("perf-cap-000", node) == draining


def test_the_heartbeat_ends_at_its_deadline() -> None:
    context = ApplicationContext(store=build_store())
    save_fleet(context, clusters=1, per_cluster=1)
    clock = iter([0.0, 0.0, 10.0, 10.0, 10.0])

    summary = heartbeat(
        context.store,
        ScriptedStop([None, None]),
        Ticks(),
        clusters=1,
        deadline_seconds=5,
        monotonic=lambda: next(clock),
    )

    assert summary["ended_by"] == "deadline"


def breaker_record(**values: Any) -> RemediationStormBreaker:
    return RemediationStormBreaker(cluster_id="perf-cap-000", updated_at=NOW, **values)


def test_the_census_reads_the_breakers_own_count_and_record() -> None:
    context = ApplicationContext(store=build_store())
    save_fleet(context, clusters=2, per_cluster=20)
    model_agent(context, "perf-cap-000", seed.integrated_node_id(RUN, 0, 0))
    context.store.replace_remediation_storm_breaker(
        breaker_record(state=StormBreakerState.CLOSED, threshold=5), expected=None
    )

    census = synthetic_fleet_pod.census(
        context.store, run_id=RUN, clusters=2, now=NOW + timedelta(minutes=1)
    )

    assert census["product_breaker"] is True
    first = census["clusters"]["perf-cap-000"]
    assert (
        first["agents"]
        == first["counted_nodes"]
        == first["breaker_cluster_nodes"]
        == 21
    )
    assert first["threshold"] == 5
    assert first["fleet_agents"] == first["fleet_current"] == 20
    assert first["model_agents"] == 1
    assert first["breaker"]["state"] == "CLOSED" and first["breaker"]["open"] is False
    assert census["clusters"]["perf-cap-001"]["breaker"] is None


def census_of(**breaker: Any) -> dict[str, Any]:
    item = {
        "agents": 256,
        "fleet_agents": 252,
        "fleet_current": 252,
        "breaker": {
            "state": "CLOSED",
            "open": False,
            "threshold": 52,
            "trip_id": None,
            "trip_nodes": 0,
            "trip_threshold": 0,
            **breaker,
        },
    }
    return {"product_breaker": True, "clusters": {"perf-cap-000": item}}


HEARTBEAT_OK = {
    "stop_confirmed": True,
    "longest_renewal_gap_seconds": 300.0,
    "sessions": [{"ended_by": "stopped"}],
}
PLAN = fleet.fleet_plan(clusters=1, model_nodes_per_cluster=4, nodes_per_cluster=256)


def test_a_clean_fleet_passes() -> None:
    assert (
        fleet.fleet_errors(PLAN, census_of(), HEARTBEAT_OK, expected_trips=frozenset())
        == []
    )


@pytest.mark.parametrize(
    ("census", "heartbeat", "message"),
    [
        (census_of(threshold=3), HEARTBEAT_OK, "the fleet was not counted"),
        (
            census_of(trip_id="perf-cap-000@t", trip_nodes=3, trip_threshold=3),
            HEARTBEAT_OK,
            "tripped during the run",
        ),
        ({"error": "exec failed"}, HEARTBEAT_OK, "census failed"),
        (census_of(), {**HEARTBEAT_OK, "stop_confirmed": False}, "did not stop"),
        (
            census_of(),
            {**HEARTBEAT_OK, "longest_renewal_gap_seconds": 900.0},
            "a whole lease",
        ),
        (
            census_of(),
            {**HEARTBEAT_OK, "sessions": [{"ended_by": "deadline"}]},
            "ended deadline before the run did",
        ),
    ],
)
def test_a_fleet_that_did_not_hold_the_threshold_fails(
    census: dict[str, Any], heartbeat: dict[str, Any], message: str
) -> None:
    errors = fleet.fleet_errors(PLAN, census, heartbeat, expected_trips=frozenset())

    assert any(message in error for error in errors), errors


def test_a_threshold_other_than_planned_but_above_the_model_is_a_note() -> None:
    """A model Agent the product retired leaves 255 counted nodes: 51, not 52."""

    census = census_of(threshold=51)

    assert (
        fleet.fleet_errors(PLAN, census, HEARTBEAT_OK, expected_trips=frozenset()) == []
    )
    assert fleet.fleet_notes(PLAN, census) == [
        "the breaker last judged perf-cap-000 against 51 nodes, the plan's threshold is 52"
    ]


def test_a_shrunken_or_lapsed_fleet_fails() -> None:
    census = census_of()
    census["clusters"]["perf-cap-000"].update(fleet_agents=250, fleet_current=249)

    errors = fleet.fleet_errors(PLAN, census, HEARTBEAT_OK, expected_trips=frozenset())

    assert any(
        "250 fleet Agents, the run registered 252" in error for error in errors
    ), errors
    assert any("1 fleet Agents are not ACTIVE" in error for error in errors), errors


def test_the_storm_steps_own_trip_is_expected() -> None:
    census = census_of(trip_id="perf-cap-000@t", trip_nodes=52, trip_threshold=52)

    assert (
        fleet.fleet_errors(
            PLAN, census, HEARTBEAT_OK, expected_trips=frozenset({"perf-cap-000"})
        )
        == []
    )


def test_an_open_breaker_is_handed_over_then_released(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []

    def action(cluster_id: str, name: str, **options: Any) -> dict[str, Any]:
        calls.append((cluster_id, name))
        assert options["reference"] == "PERF-STORM-run-a"
        return {"status": "released" if name == "release" else "aborted"}

    monkeypatch.setattr(storm, "storm_operator_action", action)
    census = {
        "clusters": {
            "perf-cap-000": {"breaker": {"open": True, "state": "TRIPPED"}},
            "perf-cap-001": {"breaker": {"open": True, "state": "ABORTED"}},
            "perf-cap-002": {"breaker": {"open": False, "state": "CLOSED"}},
            "perf-cap-003": {"breaker": None},
        }
    }

    exits = fleet.close_open_breakers(census, run_id=RUN)

    assert calls == [
        ("perf-cap-000", "abort"),
        ("perf-cap-000", "release"),
        ("perf-cap-001", "release"),
    ]
    assert [item["cluster_id"] for item in exits] == ["perf-cap-000", "perf-cap-001"]


class FakeHeartbeat:
    def __init__(self, **options: Any) -> None:
        self.options = options
        self.started = False
        self.sessions = [{"ended_by": "stopped", "passes": 2, "renewed": 10}]

    def start(self) -> None:
        self.started = True

    def stop(self) -> dict[str, Any]:
        return {"stop_confirmed": True}

    def heartbeat_summary(self, *, stop_confirmed: bool) -> dict[str, Any]:
        return {
            "stop_confirmed": stop_confirmed,
            "sessions": self.sessions,
            "longest_renewal_gap_seconds": 300.0,
        }


def fleet_wire(monkeypatch: pytest.MonkeyPatch, census: dict[str, Any]) -> list[str]:
    events: list[str] = []
    monkeypatch.setattr(
        fleet,
        "register_fleet",
        lambda **kwargs: events.append("register") or {"agents_created": 252},
    )
    monkeypatch.setattr(
        fleet,
        "FleetHeartbeat",
        lambda **options: events.append("heartbeat") or FakeHeartbeat(**options),
    )
    monkeypatch.setattr(
        fleet, "fleet_census", lambda **_kwargs: events.append("census") or census
    )
    monkeypatch.setattr(
        fleet,
        "close_open_breakers",
        lambda *_args, **_kwargs: events.append("close") or [],
    )
    return events


def test_the_fleet_block_registers_first_and_proves_the_threshold_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = fleet_wire(monkeypatch, census_of())

    with fleet.synthetic_fleet(
        tmp_path, run_id=RUN, plan=PLAN, agent_identity=None, deadline_seconds=3600
    ) as document:
        events.append("run")
        assert document["registration"] == {"agents_created": 252}

    assert events == ["register", "heartbeat", "run", "census", "close"]
    written = json.loads((tmp_path / fleet.FLEET_ARTIFACT).read_text())
    assert written["verdict"] == "PASS"
    assert written["plan"]["threshold_nodes"] == 52


def test_the_fleet_block_fails_a_run_whose_breaker_tripped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fleet_wire(monkeypatch, census_of(trip_id="t", trip_nodes=3, trip_threshold=3))

    with pytest.raises(RuntimeError, match="synthetic fleet: .*tripped during the run"):
        with fleet.synthetic_fleet(
            tmp_path, run_id=RUN, plan=PLAN, agent_identity=None, deadline_seconds=60
        ):
            pass

    assert (
        json.loads((tmp_path / fleet.FLEET_ARTIFACT).read_text())["verdict"] == "FAIL"
    )


def test_the_fleet_block_never_masks_the_runs_own_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = fleet_wire(monkeypatch, {"error": "exec failed"})

    with pytest.raises(KeyError, match="the run's own"):
        with fleet.synthetic_fleet(
            tmp_path, run_id=RUN, plan=PLAN, agent_identity=None, deadline_seconds=60
        ):
            raise KeyError("the run's own")

    assert events[-2:] == ["census", "close"], "the fleet still closes on a failure"


def test_the_cleanup_proof_refuses_any_agent_or_breaker_left(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clean = {"clusters": {"perf-cap-000": {"agents": 0, "breaker": None}}}
    left = {"clusters": {"perf-cap-000": {"agents": 3, "breaker": {"state": "CLOSED"}}}}
    monkeypatch.setattr(fleet, "fleet_census", lambda **_kwargs: clean)
    fleet.verify_fleet_cleanup(tmp_path, run_id=RUN, clusters=1)
    assert (
        json.loads((tmp_path / fleet.FLEET_CLEANUP_ARTIFACT).read_text())["residue"]
        == {}
    )

    monkeypatch.setattr(fleet, "fleet_census", lambda **_kwargs: left)
    with pytest.raises(
        RuntimeError, match="perf-cap-000: 3 Agents and a breaker record"
    ):
        fleet.verify_fleet_cleanup(tmp_path, run_id=RUN, clusters=1)


def control_for(
    answers: dict[str, str], seen: list[tuple[tuple[str, ...], dict[str, Any]]]
):
    def control(*args: str, **kwargs: Any) -> str:
        seen.append((args, kwargs))
        if args[:2] == ("get", "pod"):
            return "api-a\n"
        for marker, answer in answers.items():
            if marker in args:
                return answer
        raise AssertionError(f"unexpected control call {args}")

    return control


def test_the_registration_runs_the_seed_fleet_mode_in_an_api_pod() -> None:
    seen: list[tuple[tuple[str, ...], dict[str, Any]]] = []
    control = control_for(
        {
            f"ACTION_SEED_MODE={fleet.FLEET_AGENTS_MODE}": json.dumps(
                {"agents_created": 252}
            )
        },
        seen,
    )

    fleet.register_fleet(run_id=RUN, plan=PLAN, agent_identity=None, control_fn=control)

    args, kwargs = seen[-1]
    assert args[:4] == ("exec", "-i", "api-a", "--")
    assert "ACTION_FLEET_AGENTS_PER_CLUSTER=252" in args
    assert "ACTION_AGENT_LEASE_SECONDS=900" in args
    assert kwargs["stdin"] == fleet.SEED_SCRIPT.read_bytes()
    with pytest.raises(RuntimeError, match="created 251 Agents, planned 252"):
        fleet.register_fleet(
            run_id=RUN,
            plan=PLAN,
            agent_identity=None,
            control_fn=control_for(
                {
                    "ACTION_FLEET_AGENTS_PER_CLUSTER=252": json.dumps(
                        {"agents_created": 251}
                    )
                },
                seen,
            ),
        )


def test_the_heartbeat_session_runs_the_fleet_program(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fleet.registry, "CONTROL_KUBECONFIG", "/tmp/cpu.kubeconfig")
    monkeypatch.setattr(fleet.registry, "CONTROL_NAMESPACE", "gpu-fault-system")
    session = FakeSession(
        {
            "mode": fleet.HEARTBEAT_MODE,
            "ended_by": "stopped",
            "renewed": 5,
            "passes": 1,
        },
        exits_early=False,
    )
    launcher = Launcher([session])
    running = fleet.FleetHeartbeat(
        run_id=RUN,
        clusters=32,
        deadline_seconds=3600,
        popen=launcher,
        control_fn=control_for({}, []),
    )
    running.start()

    stopped = running.stop()
    summary = running.heartbeat_summary(stop_confirmed=stopped["stop_confirmed"])

    (argv,) = launcher.argv
    assert argv[-3:] == [
        "/opt/gpu-fault/control-plane/bin/python",
        "-c",
        fleet.FLEET_BOOTSTRAP,
    ]
    assert {
        f"FLEET_RUN_ID={RUN}",
        "FLEET_CLUSTERS=32",
        f"FLEET_MODE={fleet.HEARTBEAT_MODE}",
        "FLEET_LEASE_SECONDS=900",
        "FLEET_INTERVAL_SECONDS=300",
        "FLEET_DEADLINE_SECONDS=3600",
    } <= set(argv)
    program = fleet.FLEET_POD_SCRIPT.read_bytes()
    assert session.stdin.data == str(len(program)).encode() + b"\n" + program
    assert summary["renewed"] == 5 and summary["stop_confirmed"] is True


def test_a_session_that_found_no_fleet_left_is_not_relaunched() -> None:
    gone = FakeSession(
        {"mode": fleet.HEARTBEAT_MODE, "ended_by": "fleet-gone"}, exits_early=False
    )
    gone.closed.set()
    launcher = Launcher([gone])
    running = fleet.FleetHeartbeat(
        run_id=RUN,
        clusters=1,
        deadline_seconds=60,
        popen=launcher,
        control_fn=control_for({}, []),
    )
    running.start()
    assert gone.wait(timeout=10) == 0
    running.stop()

    assert len(launcher.argv) == 1, "fleet-gone is a final ending"


def pod_environment(tmp_path: Path, **values: str) -> dict[str, str]:
    return {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": os.pathsep.join((str(ROOT / "src"), str(ROOT))),
        "HOME": str(tmp_path),
        "FLEET_RUN_ID": RUN,
        "FLEET_CLUSTERS": "1",
        **values,
    }


def test_the_real_fleet_program_censuses_an_empty_store(tmp_path: Path) -> None:
    program = fleet.FLEET_POD_SCRIPT.read_bytes()
    completed = subprocess.run(
        [sys.executable, "-"],
        input=program,
        capture_output=True,
        cwd=tmp_path,
        env=pod_environment(tmp_path, FLEET_MODE=fleet.CENSUS_MODE),
        timeout=120,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr.decode()
    census = json.loads(completed.stdout.decode().splitlines()[-1])
    assert census["mode"] == fleet.CENSUS_MODE
    assert census["clusters"]["perf-cap-000"]["agents"] == 0


def test_the_real_bootstrap_ends_a_heartbeat_with_no_fleet(tmp_path: Path) -> None:
    process = subprocess.Popen(
        [sys.executable, "-c", fleet.FLEET_BOOTSTRAP],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=tmp_path,
        env=pod_environment(
            tmp_path,
            FLEET_MODE=fleet.HEARTBEAT_MODE,
            FLEET_LEASE_SECONDS="900",
            FLEET_INTERVAL_SECONDS="0.05",
            FLEET_DEADLINE_SECONDS="60",
        ),
    )
    stdin = process.stdin
    assert stdin is not None, "the session needs a stdin pipe"
    stdin.write(session_input(fleet.FLEET_POD_SCRIPT.read_bytes()))
    stdin.flush()
    try:
        # stdin stays open: the session must end on its own, not on EOF.
        process.wait(timeout=120)
    finally:
        process.kill()
        stdin.close()
    assert process.stdout is not None and process.stderr is not None, "pipes"
    output, errors = process.stdout.read(), process.stderr.read()

    assert process.returncode == 0, errors.decode()
    summary = json.loads(output.decode().strip().splitlines()[-1])
    assert summary["mode"] == fleet.HEARTBEAT_MODE
    assert summary["ended_by"] == "fleet-gone", "an empty store has no fleet to keep"


def test_the_real_bootstrap_stops_when_the_runner_closes_stdin(tmp_path: Path) -> None:
    """A long slot ends at once on EOF: the runner's stop, or a runner that died."""

    process = subprocess.Popen(
        [sys.executable, "-c", fleet.FLEET_BOOTSTRAP],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=tmp_path,
        env=pod_environment(
            tmp_path,
            FLEET_MODE=fleet.HEARTBEAT_MODE,
            FLEET_LEASE_SECONDS="900",
            FLEET_INTERVAL_SECONDS="600",
            FLEET_DEADLINE_SECONDS="3600",
        ),
    )
    try:
        output, errors = process.communicate(
            session_input(fleet.FLEET_POD_SCRIPT.read_bytes()), timeout=120
        )
    finally:
        process.kill()

    assert process.returncode == 0, errors.decode()
    summary = json.loads(output.decode().strip().splitlines()[-1])
    assert summary["ended_by"] == "stopped", "stdin EOF is the stop signal"
