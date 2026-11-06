"""The remediation storm breaker step of the perf suite (perf plan §8.5).

One synthetic cluster registers a production-sized fleet and crosses its
breaker on purpose: the admitted workflows run on while the rest are held, the
critical alert fires, the operator exit releases the breaker and the held work
completes without counting again. The step's plan is checked against the
product's own decision, its status program and verdict against the product's
real claim path in a memory store, and its phases, timeouts and cleanup with a
scripted product.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.app import ApplicationContext
from gpu_fault.app.metrics import METRIC_CONTRIBUTORS
from gpu_fault.app.remediation_budget_metrics import (
    budget_tally,
    remediation_storm_lines,
)
from gpu_fault.execution.remediation_budget import RemediationBudgetPolicy
from gpu_fault.models import IncidentState
from gpu_fault.remediation_storm import RemediationStormBreaker, StormCandidate
from gpu_fault.remediation_storm import storm_admission as admit
from gpu_fault.store import InMemoryStore
from gpu_fault.workflow_reconcile import remediation_storm_action
from scripts.e2e.regional.regional_live_fixture import component_python
from scripts.perf import regional_capacity_registry as capacity_registry
from scripts.perf import regional_storm_breaker as storm
from scripts.perf import regional_storm_breaker_suite as suite
from scripts.perf import seed_regional_action_workflows as seed
from scripts.perf import storm_breaker_pod
from tests._builders import copy_model, fault_incident, workflow_request, workflow_step
from tests.execution._budget_hold_support import (
    REBOOT_PLAN,
    ScriptedAdapter,
    budget_dispatcher,
)
from tests.regional._perf_caller_support import (
    default_signal_dispositions,
    deliver_signal,
)
from tests.regional.test_regional_action_capacity import release_agent_identity

ROOT = Path(__file__).resolve().parents[2]
RUN = "storm-a"
CLUSTER = storm.STORM_CLUSTER
NOW = datetime(2026, 10, 29, 12, 0, tzinfo=timezone.utc)


def judge(plan: dict[str, Any]) -> tuple[list[bool], RemediationStormBreaker | None]:
    record: RemediationStormBreaker | None = None
    admitted = []
    for index in range(plan["workflows"]):
        record, refusal = admit(
            record,
            StormCandidate(
                cluster_id=CLUSTER,
                workflow_id=f"workflow-{index}",
                node_ids=frozenset(
                    f"w{index}-n{node}" for node in range(plan["nodes_per_workflow"])
                ),
                domain=f"incident:{index}",
            ),
            now=NOW + timedelta(seconds=index),
            cluster_nodes=plan["nodes_per_cluster"],
        )
        admitted.append(refusal is None)
    return admitted, record


@pytest.mark.parametrize(
    ("nodes", "per_workflow", "held"),
    [(256, 4, 4), (256, 1, 3), (100, 4, 2), (20, 8, 1), (10, 1, 1), (512, 4, 2)],
    ids=str,
)
def test_the_plan_trips_where_the_product_does(
    nodes: int, per_workflow: int, held: int
) -> None:
    plan = storm.breaker_plan(
        nodes_per_cluster=nodes, nodes_per_workflow=per_workflow, held_workflows=held
    )

    admitted, record = judge(plan)

    assert admitted == [True] * plan["admitted_workflows"] + [False] * held
    assert record is not None and record.state.value == "TRIPPED"
    assert len(record.trip_node_ids) == plan["trip_nodes"]
    assert record.trip_domains == plan["trip_domains"]
    assert record.trip_threshold == plan["threshold_nodes"]
    assert record.trip_workflow_id == f"workflow-{plan['admitted_workflows']}"


def test_the_default_step_holds_four_of_sixteen_workflows() -> None:
    plan = storm.breaker_plan(
        nodes_per_cluster=256, nodes_per_workflow=4, held_workflows=4
    )

    assert plan["threshold_nodes"] == 52
    assert (plan["admitted_workflows"], plan["workflows"]) == (12, 16)
    assert plan["model_nodes"] == 64 and plan["commands"] == 160
    assert "the next trips it at 52 nodes across 13 domains" in storm.plan_line(plan)


def test_a_step_larger_than_its_cluster_is_refused() -> None:
    with pytest.raises(ValueError, match="more than the 40 GPU nodes"):
        storm.breaker_plan(nodes_per_cluster=40, nodes_per_workflow=9, held_workflows=4)
    with pytest.raises(ValueError, match=">= 1 held"):
        storm.breaker_plan(
            nodes_per_cluster=256, nodes_per_workflow=4, held_workflows=0
        )


def seeded_store(plan: dict[str, Any]) -> tuple[InMemoryStore, list[str]]:
    """The step's cluster as the seed leaves it: the fleet, the model's Agents,
    one job-less workflow (one incident: one domain) per scope, REBOOT_PLAN-owned
    so the product's scripted adapter runs them."""

    store = InMemoryStore()
    context = ApplicationContext(store=store)
    now = datetime.now(timezone.utc)
    seed.seed_fleet_agents(
        context,
        run_id=RUN,
        clusters=1,
        per_cluster=plan["nodes_per_cluster"] - plan["model_nodes"],
        template=None,
        identity=release_agent_identity(),
        now=now,
    )
    ids = []
    for index in range(plan["workflows"]):
        scope = seed.action_scope(RUN, 0, index)
        nodes = seed.action_nodes(scope, plan["nodes_per_workflow"])
        for node in nodes:
            seed.save_synthetic_agent(
                context,
                template=None,
                identity=release_agent_identity(),
                cluster_id=CLUSTER,
                node_id=node,
                run_id=RUN,
                now=now,
            )
        created = now - timedelta(seconds=30) + timedelta(milliseconds=index)
        incident = fault_incident(
            f"incident-{scope}",
            f"event-{scope}",
            cluster_id=CLUSTER,
            node_ids=nodes,
            official_action="RESTART_BM",
            state=IncidentState.ACTION_PENDING,
            fencing_token=3,
            created_at=created,
            updated_at=created,
        )
        workflow = workflow_request(
            f"workflow-{scope}",
            incident.incident_id,
            runtime_profile_version="active-v1",
            official_action="RESTART_BM",
            official_steps=[workflow_step(op, node_ids=nodes) for op in REBOOT_PLAN],
            created_at=created,
            updated_at=created,
        )
        store.save_incident(
            copy_model(incident, workflow_request_id=workflow.request_id)
        )
        store.save_workflow(workflow)
        ids.append(workflow.request_id)
    return store, ids


def read(store: InMemoryStore, plan: dict[str, Any]) -> tuple[dict, dict]:
    status = json.loads(
        json.dumps(
            storm_breaker_pod.status(
                store, run_id=RUN, cluster_id=CLUSTER, workflows=plan["workflows"]
            ),
            default=str,
        )
    )
    return status, storm.classify(status)


def gauges(store: InMemoryStore) -> dict[str, float]:
    lines = remediation_storm_lines(
        store.list_remediation_storm_breakers(),
        budget_tally(
            store.list_workflows(None, limit=1000), datetime.now(timezone.utc)
        ),
    )
    return storm.parse_storm_gauges("\n".join(lines), CLUSTER)


def test_the_step_judges_the_products_own_breaker_cycle() -> None:
    """The product's real claim path, dispatcher and operator release over the
    step's seeded cluster, read through the step's own status program."""

    plan = storm.breaker_plan(
        nodes_per_cluster=256, nodes_per_workflow=4, held_workflows=4
    )
    store, ids = seeded_store(plan)
    admitted, held = (
        ids[: plan["admitted_workflows"]],
        ids[plan["admitted_workflows"] :],
    )
    adapter = ScriptedAdapter(REBOOT_PLAN)
    dispatcher = budget_dispatcher(
        store, adapter, RemediationBudgetPolicy(), max_workers=1
    )

    dispatcher.run_once()
    tripped, snapshot = read(store, plan)

    breaker = tripped["breaker"]
    assert snapshot["breaker_state"] == "TRIPPED"
    assert breaker["trip_cluster_nodes"] == 256, "the fleet is counted"
    assert breaker["trip_threshold"] == 52
    assert len(breaker["trip_node_ids"]) == plan["trip_nodes"]
    assert breaker["trip_domains"] == plan["trip_domains"]
    assert snapshot["product_held"] == snapshot["held"] == held
    assert snapshot["tripped_by"] == [held[0]] == [breaker["trip_workflow_id"]]
    assert storm.held_violations(snapshot, snapshot["product_held"]) == []
    read_back = gauges(store)
    # The tripped timestamp is rendered with "%g" (six significant digits), so
    # the product's own gauge reads it only to within ~5000 s; not compared.
    read_back.pop("tripped_timestamp_seconds")
    assert read_back == {
        "open": 1.0,
        "window_nodes": 52.0,
        "threshold_nodes": 52.0,
        "held_workflows": 4.0,
    }

    adapter.released.update(admitted)
    dispatcher.run_once()
    dispatcher.run_once()
    in_flight, snapshot = read(store, plan)
    assert snapshot["breaker_state"] == "TRIPPED"
    assert set(admitted) <= set(snapshot["succeeded"]), "in-flight work goes on"
    assert snapshot["product_held"] == held, "new work stays held"
    assert in_flight["breaker"]["notified_trip_id"] == breaker["trip_id"]

    released = remediation_storm_action(
        store,
        CLUSTER,
        "release",
        actor="perf-test",
        reason="perf storm breaker step",
        reference=f"PERF-STORM-{RUN}",
        now=datetime.now(timezone.utc),
    )
    assert released["status"] == "released"
    dispatcher.run_once()
    adapter.released.update(held)
    dispatcher.run_once()
    dispatcher.run_once()
    final, snapshot = read(store, plan)

    assert snapshot["succeeded"] == sorted(ids)
    assert final["breaker"]["state"] == "CLOSED"
    assert final["breaker"]["window_nodes"] == 0, "held work does not count again"
    assert final["breaker"]["trip_id"] == breaker["trip_id"]
    assert gauges(store)["open"] == 0.0
    summary = {
        "plan": plan,
        "trip": breaker,
        "phases": {"drain": {"outcome": "done"}, "resolve": {"outcome": "done"}},
        "final_breaker": final["breaker"],
        "executor": {"completed_commands": plan["commands"]},
        "held_workflows": held,
        "admitted_succeeded_before_release": admitted,
        "violations": [],
        "release": released,
        "alert": {"firing_at": "t"},
        "gauges": {"open_matched_at": "t"},
    }
    status, checks = suite.verdict(summary)
    assert status == "PASS", checks


def test_the_gauges_are_read_on_the_tier_that_renders_them() -> None:
    """The breaker gauges are a fleet-level contributor's ("closed-loop"); the
    ingress tier (api-ha, no background services) skips those, so the b223
    perf run read no gauge on api-ha for its whole hold and resolve phases."""

    assert "closed-loop" in METRIC_CONTRIBUTORS.fleet_level_names, (
        "the breaker gauges moved off the fleet-level contributors"
    )
    body = "\n".join(
        [
            f'{storm.GAUGE_PREFIX}open{{cluster_id="{CLUSTER}"}} 1',
            f'{storm.GAUGE_PREFIX}held_workflows{{cluster_id="{CLUSTER}"}} 4',
            f'{storm.GAUGE_PREFIX}open{{cluster_id="other"}} 0',
        ]
    )
    calls: list[tuple[str, ...]] = []

    def control(*args: str, **_options: Any) -> str:
        calls.append(args)
        if args[0] == "get":
            return "gpu-fault-control-worker-7d9-x2"
        return body

    values = storm.storm_gauges(CLUSTER, control_fn=control)

    assert values == {"open": 1.0, "held_workflows": 4.0}
    selector, read = calls
    assert "app=gpu-fault-control-worker" in selector, selector
    assert read[1] == "gpu-fault-control-worker-7d9-x2", read
    assert "http://127.0.0.1:8081/metrics" in read[-1], read
    with pytest.raises(RuntimeError, match="no Running gpu-fault-control-worker"):
        storm.storm_gauges(CLUSTER, control_fn=lambda *args, **options: "")


def command_status(created: str, updated: str) -> dict[str, Any]:
    return {"count": 2, "first_created_at": created, "last_updated_at": updated}


def test_the_drain_is_timed_from_the_release() -> None:
    released = NOW
    status = {
        "workflows": {
            "a": {"status": "SUCCEEDED", "updated_at": "2026-10-29T11:58:00+00:00"},
            "h1": {"status": "SUCCEEDED", "updated_at": "2026-10-29T12:02:40+00:00"},
            "h2": {"status": "SUCCEEDED", "updated_at": "2026-10-29T12:03:00+00:00"},
        },
        "commands": {
            "a": command_status(
                "2026-10-29T11:55:00+00:00", "2026-10-29T11:58:00+00:00"
            ),
            "h1": command_status(
                "2026-10-29T12:00:05+00:00", "2026-10-29T12:02:40+00:00"
            ),
            "h2": command_status(
                "2026-10-29T12:00:20+00:00", "2026-10-29T12:03:00+00:00"
            ),
        },
    }

    measured = storm.drain_measurements(
        status, released_at=released, held_ids=["h1", "h2"], admitted_ids=["a"]
    )

    assert measured["release_to_first_held_command_seconds"] == 5.0
    assert measured["release_to_last_held_command_seconds"] == 20.0
    assert measured["release_to_held_completion_seconds"] == 180.0
    assert measured["held_completed"] == 2
    assert measured["workflow_execution_seconds"]["held"] == {
        "count": 2,
        "p50": 157.5,
        "max": 160.0,
    }
    assert measured["workflow_execution_seconds"]["admitted"]["max"] == 180.0


def test_an_unfinished_drain_has_no_completion_time() -> None:
    status = {
        "workflows": {"h1": {"status": "RUNNING", "updated_at": NOW.isoformat()}},
        "commands": {"h1": command_status(NOW.isoformat(), NOW.isoformat())},
    }

    measured = storm.drain_measurements(
        status, released_at=NOW, held_ids=["h1"], admitted_ids=[]
    )

    assert measured["release_to_held_completion_seconds"] is None
    assert measured["held_completed"] == 0


def test_a_held_workflow_that_ran_is_a_violation() -> None:
    snapshot = {"with_commands": ["h1"], "running": ["h2"], "terminal": []}

    assert storm.held_violations(snapshot, ["h1", "h2", "h3"]) == [
        "h1 has remote commands while held",
        "h2 left PENDING while held",
    ]


def test_the_alert_state_is_read_per_cluster_from_amp() -> None:
    alerts = {
        "data": {
            "alerts": [
                {
                    "labels": {"alertname": storm.ALERT_NAME, "cluster_id": CLUSTER},
                    "state": "firing",
                },
                {
                    "labels": {
                        "alertname": storm.ALERT_NAME,
                        "cluster_id": "perf-cap-001",
                    },
                    "state": "pending",
                },
                {
                    "labels": {"alertname": "Other", "cluster_id": CLUSTER},
                    "state": "firing",
                },
            ]
        }
    }
    rules = {"data": {"groups": [{"rules": [{"name": storm.ALERT_NAME}]}]}}

    assert storm.alert_states(CLUSTER, fetch=lambda _path: alerts) == ["firing"]
    assert storm.alert_states("perf-cap-009", fetch=lambda _path: alerts) == []
    assert storm.alert_rule_loaded(fetch=lambda _path: rules) is True
    assert (
        storm.alert_rule_loaded(fetch=lambda _path: {"data": {"groups": []}}) is False
    )
    with pytest.raises(RuntimeError, match="alert inventory is missing"):
        storm.alert_states(CLUSTER, fetch=lambda _path: {"data": {}})


def local_pod(home: Path) -> Any:
    """``kubectl exec`` into an api-ha Pod, run here on the checkout instead:
    the product's own programs against an empty memory store."""

    def control(
        *args: str, stdin: bytes | None = None, check: bool = True, **_kwargs: Any
    ) -> str:
        if args[:2] == ("get", "pod"):
            return "api-a\n"
        assert args[:4] == ("exec", "-i", "api-a", "--"), args
        command = [
            sys.executable if item == component_python("cpu") else item
            for item in args[4:]
        ]
        completed = subprocess.run(
            command,
            input=stdin,
            capture_output=True,
            env={
                "PATH": os.environ.get("PATH", ""),
                "PYTHONPATH": os.pathsep.join((str(ROOT / "src"), str(ROOT))),
                "HOME": str(home),
            },
            timeout=120,
            check=False,
        )
        if check and completed.returncode:
            raise RuntimeError(completed.stderr.decode())
        return completed.stdout.decode()

    return control


def test_the_operator_exit_runs_the_verbs_own_pod_program(tmp_path: Path) -> None:
    released = storm.storm_operator_action(
        CLUSTER,
        "release",
        reason="perf storm breaker step",
        reference=f"PERF-STORM-{RUN}",
        control_fn=local_pod(tmp_path),
        actor="perf-test",
    )

    assert released["status"] == "already-closed", "a closed breaker is a no-op"
    assert released["operator"] == "perf-test"
    assert released["reference"] == f"PERF-STORM-{RUN}"
    with pytest.raises(RuntimeError, match="abort of perf-cap-000: .*not open"):
        storm.storm_operator_action(
            CLUSTER,
            "abort",
            reason="perf storm breaker step",
            reference=f"PERF-STORM-{RUN}",
            control_fn=local_pod(tmp_path),
            actor="perf-test",
        )


def test_the_status_program_runs_in_the_pod_as_shipped(tmp_path: Path) -> None:
    status = storm.breaker_status(
        run_id=RUN, workflows=2, control_fn=local_pod(tmp_path)
    )

    assert status["breaker"] is None
    assert status["held_workflows"] == []
    assert sorted(status["workflows"]) == storm_breaker_pod.step_workflow_ids(RUN, 2)
    assert set(status["workflows"].values()) == {None}


class Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class ScriptedProduct:
    """The product as the step reads it, one transition per poll.

    ``alert`` lists the AMP states the hold phase sees, poll by poll.
    """

    def __init__(self, plan: dict[str, Any], *, alert: list[list[str]]) -> None:
        self.plan = plan
        self.ids = storm_breaker_pod.step_workflow_ids(RUN, plan["workflows"])
        self.admitted = self.ids[: plan["admitted_workflows"]]
        self.held = self.ids[plan["admitted_workflows"] :]
        self.polls = 0
        self.alert = list(alert)
        self.state = "seeded"
        self.actions: list[str] = []
        self.breaker: dict[str, Any] | None = None

    def status(self, **_kwargs: Any) -> dict[str, Any]:
        self.polls += 1
        if self.state == "seeded":
            self.state = "tripped"
            self.breaker = {
                "state": "TRIPPED",
                "open": True,
                "trip_id": "perf-cap-000@t",
                "tripped_at": NOW.isoformat(),
                "trip_node_ids": [
                    f"n{index}" for index in range(self.plan["trip_nodes"])
                ],
                "trip_domains": self.plan["trip_domains"],
                "trip_threshold": self.plan["threshold_nodes"],
                "trip_cluster_nodes": self.plan["nodes_per_cluster"],
                "trip_workflow_id": self.held[0],
                "notified_trip_id": None,
                "window_nodes": self.plan["trip_nodes"],
            }
        elif self.state == "tripped":
            self.state = "admitted-done"
            assert self.breaker is not None, "tripped first"
            self.breaker["notified_trip_id"] = "perf-cap-000@t"
        elif self.state == "released":
            self.state = "drained"
        workflows: dict[str, Any] = {}
        commands: dict[str, Any] = {}
        for request_id in self.ids:
            held = request_id in self.held
            done = (not held and self.state != "tripped") or self.state == "drained"
            workflows[request_id] = {
                "status": "SUCCEEDED" if done else ("PENDING" if held else "RUNNING"),
                "last_blocked_scope": "storm:perf-cap-000" if held else None,
                "tripped_breaker": request_id == self.held[0],
                "updated_at": (NOW + timedelta(minutes=5)).isoformat(),
            }
            if not held or self.state == "drained":
                started = NOW + (
                    timedelta(minutes=4) if held else timedelta(seconds=30)
                )
                commands[request_id] = command_status(
                    started.isoformat(), (NOW + timedelta(minutes=5)).isoformat()
                )
        held_now = [] if self.state in {"released", "drained"} else list(self.held)
        if self.state in {"released", "drained"}:
            for request_id in self.held:
                workflows[request_id]["last_blocked_scope"] = None
        return {
            "cluster_id": CLUSTER,
            "breaker": self.breaker,
            "held_workflows": held_now,
            "workflows": workflows,
            "commands": commands,
        }

    def alert_states(self, _cluster: str) -> list[str]:
        if self.state in {"released", "drained"}:
            return []
        return self.alert.pop(0) if self.alert else []

    def gauges(self, _cluster: str) -> dict[str, float]:
        if self.breaker is None or not self.breaker["open"]:
            return {"open": 0.0}
        return {
            "open": 1.0,
            "threshold_nodes": float(self.plan["threshold_nodes"]),
            "window_nodes": float(self.plan["trip_nodes"]),
            "held_workflows": float(self.plan["held_workflows"]),
        }

    def operator(self, cluster_id: str, action: str, **_kwargs: Any) -> dict[str, Any]:
        self.actions.append(action)
        if action == "release" and self.breaker is not None:
            self.breaker.update(state="CLOSED", open=False, window_nodes=0)
            self.state = "released" if self.state != "drained" else self.state
        return {
            "status": "released" if action == "release" else "aborted",
            "recorded_at": (NOW + timedelta(minutes=3)).isoformat(),
        }


def step_args(**overrides: Any) -> SimpleNamespace:
    values = {
        "executor_workers": 16,
        "delay_scale": 1.0,
        "lease_seconds": 120,
        "poll_seconds": 10.0,
        "trip_timeout_seconds": 300,
        "hold_timeout_seconds": 900,
        "drain_timeout_seconds": 900,
        "alert_timeout_seconds": 600,
        **overrides,
    }
    return SimpleNamespace(**values)


@pytest.fixture
def product(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Any]:
    plan = storm.breaker_plan(
        nodes_per_cluster=256, nodes_per_workflow=4, held_workflows=4
    )
    events: list[str] = []

    def build(alert: list[list[str]], **overrides: Any) -> tuple[Any, ScriptedProduct]:
        fake = ScriptedProduct(plan, alert=alert)
        monkeypatch.setattr(storm, "breaker_status", fake.status)
        monkeypatch.setattr(storm, "alert_states", fake.alert_states)
        monkeypatch.setattr(storm, "storm_gauges", fake.gauges)
        monkeypatch.setattr(storm, "storm_operator_action", fake.operator)
        clock = Clock()
        step = suite.StormBreakerStep(
            step_args(**overrides),
            plan=plan,
            run_id=RUN,
            artifacts=tmp_path,
            resources=SimpleNamespace(create=lambda manifest: events.append("create")),
            registry_scope="live",
            monotonic=clock,
            sleep=clock.sleep,
        )
        return step, fake

    monkeypatch.setattr(
        suite,
        "run_agent_heartbeat_refresher",
        lambda **_kwargs: lambda: events.append("beat"),
    )
    monkeypatch.setattr(suite, "upsert_configmap", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(suite, "executor_scripts", lambda: {})
    monkeypatch.setattr(
        suite,
        "executor_job",
        lambda **kwargs: events.append("job") or {"kwargs": kwargs},
    )
    monkeypatch.setattr(suite, "executor_identity", lambda **_kwargs: {})
    monkeypatch.setattr(
        suite, "ensure_load_service_account", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(suite, "run_manifest", lambda manifest, _run_id: manifest)
    monkeypatch.setattr(suite, "executor_job_failure", lambda *_args: None)
    monkeypatch.setattr(
        suite,
        "dataplane",
        lambda *args, **_kwargs: "pod/executor-0" if "name" in args else "1|",
    )
    monkeypatch.setattr(
        suite,
        "seed",
        lambda **kwargs: events.append("seed") or {"workflows_created": 16},
    )
    monkeypatch.setattr(suite, "collect_executor_logs", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        suite,
        "aggregate_executor_documents",
        lambda _documents: {"completed_commands": plan["commands"]},
    )
    yield SimpleNamespace(build=build, events=events, plan=plan)


def test_the_step_runs_every_phase_and_passes(product: Any) -> None:
    step, fake = product.build([[], ["pending"], ["firing"]])

    summary = step.run(fleet_ready_at=0.0)
    status, checks = suite.verdict(summary)

    assert status == "PASS", checks
    assert list(summary["phases"]) == ["trip", "hold", "drain", "resolve"]
    assert all(item["outcome"] == "done" for item in summary["phases"].values()), (
        summary["phases"]
    )
    assert fake.actions == ["release"]
    assert product.events.index("seed") < product.events.index("beat"), (
        "the model's heartbeats start with its Agents"
    )
    assert summary["held_workflows"] == fake.held
    assert summary["admitted_succeeded_before_release"] == fake.admitted
    assert summary["measurements"]["release_to_held_completion_seconds"] == 120.0
    assert summary["alert"]["firing_at"] is not None


def test_the_step_waits_for_the_breaker_count_to_settle(product: Any) -> None:
    """A count a process cached for the cluster id before the fleet was complete
    lives a minute: the workflows are seeded no sooner than that."""

    step, _fake = product.build([["firing"]])

    step.run(fleet_ready_at=-20.0)

    assert step.monotonic.sleeps[0] == suite.COUNT_SETTLE_SECONDS - 20.0


def test_a_missing_alert_fails_the_step_but_still_releases_the_held_work(
    product: Any,
) -> None:
    step, fake = product.build([], hold_timeout_seconds=60)

    summary = step.run(fleet_ready_at=0.0)
    status, checks = suite.verdict(summary)

    assert status == "FAIL"
    assert checks["alert_firing"] is False
    assert summary["phases"]["hold"]["outcome"] == "timeout"
    assert fake.actions == ["release"], "the held work must not be stranded"
    assert checks["held_completed_after_release"] is True


def test_a_breaker_that_never_trips_ends_the_step_without_a_release(
    product: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    step, fake = product.build([], trip_timeout_seconds=30)
    monkeypatch.setattr(
        storm,
        "breaker_status",
        lambda **_kwargs: {
            "cluster_id": CLUSTER,
            "breaker": None,
            "held_workflows": [],
            "workflows": {},
            "commands": {},
        },
    )

    summary = step.run(fleet_ready_at=0.0)
    status, checks = suite.verdict(summary)

    assert status == "FAIL" and checks["fleet_counted"] is False
    assert list(summary["phases"]) == ["trip"]
    assert summary["phases"]["trip"]["outcome"] == "timeout"
    assert fake.actions == [] and summary["release"] is None


def test_the_cleanup_hands_over_and_releases_an_open_breaker(product: Any) -> None:
    step, fake = product.build([])
    fake.status()  # tripped, four held

    record = step.close_breaker()

    assert fake.actions == ["abort", "release"]
    assert record["state_before"] == "TRIPPED"
    assert record["state_after"] == "CLOSED"
    written = json.loads((step.artifacts / "storm-breaker-exit.json").read_text())
    assert written == record


def test_the_cleanup_of_a_closed_breaker_changes_nothing(
    product: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    step, fake = product.build([])
    closed = {"state": "CLOSED", "open": False}
    monkeypatch.setattr(
        storm,
        "breaker_status",
        lambda **_kwargs: {
            "cluster_id": CLUSTER,
            "breaker": closed,
            "held_workflows": [],
            "workflows": {},
            "commands": {},
        },
    )

    record = step.close_breaker()

    assert fake.actions == []
    assert record["state_before"] == record["state_after"] == "CLOSED"


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--hold-timeout-seconds", "1400"], "must stay within 1500 s"),
        (["--synthetic-ttl-seconds", "1800"], "must outlast every phase"),
        (["--nodes-per-workflow", "100"], "more than the 256 GPU nodes"),
        (
            ["--fleet-nodes-per-cluster", "40", "--nodes-per-workflow", "16"],
            "more than the 40",
        ),
    ],
)
def test_the_step_refuses_a_plan_it_cannot_run_safely(
    arguments: list[str],
    message: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(suite, "validate_registry_target", lambda **_kwargs: "live")
    monkeypatch.setattr(capacity_registry, "DATAPLANE_CONTEXT", "fake-context")

    with pytest.raises(SystemExit):
        suite.main(arguments)

    assert message in capsys.readouterr().err


def test_a_dropped_status_read_is_survived_but_not_three_in_a_row(
    product: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    step, fake = product.build([["firing"]])
    failures = iter([True, True, False])

    def flaky(**kwargs: Any) -> dict[str, Any]:
        if next(failures, False):
            raise RuntimeError("exec stream dropped")
        return fake.status(**kwargs)

    monkeypatch.setattr(storm, "breaker_status", flaky)
    summary = step.run(fleet_ready_at=0.0)

    assert summary["phases"]["trip"]["read_errors"] == 2
    assert suite.verdict(summary)[0] == "PASS"

    step, _fake = product.build([])

    def lost(**_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("exec stream dropped")

    monkeypatch.setattr(storm, "breaker_status", lost)
    with pytest.raises(RuntimeError, match="exec stream dropped"):
        step.run(fleet_ready_at=0.0)
    assert step.phases["trip"]["read_errors"] == suite.MAX_FAILED_READS


class FakeStep:
    """The step as ``main`` drives it: run, then the cleanup whatever happened."""

    failure: BaseException | None = None

    def __init__(self, events: list[str], *_args: Any, **_kwargs: Any) -> None:
        self.events = events
        self.history = [{"phase": "trip"}]

    def run(self, *, fleet_ready_at: float) -> dict[str, Any]:
        self.events.append("step-run")
        if self.failure is not None:
            raise self.failure
        return {"plan": {}, "phases": {}}

    def close_breaker(self) -> dict[str, Any]:
        self.events.append("close-breaker")
        return {"state_before": "CLOSED", "state_after": "CLOSED"}


@pytest.fixture
def orchestration(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> SimpleNamespace:
    events: list[str] = []
    wire = SimpleNamespace(events=events, rule_loaded=True, fleet_kwargs={})

    def register(count: int, artifacts: Path, **kwargs: Any) -> None:
        assert count == 1, "the step registers one synthetic cluster"
        events.append("register")
        (artifacts / "registry-registration-intent.json").write_text(
            json.dumps({"run_id": kwargs["run_id"], "cluster_ids": [CLUSTER]})
        )

    @contextmanager
    def synthetic_fleet(artifacts: Path, **kwargs: Any) -> Iterator[dict]:
        wire.fleet_kwargs = kwargs
        events.append("fleet-enter")
        try:
            yield {"plan": kwargs["plan"]}
        finally:
            events.append("fleet-exit")

    monkeypatch.setattr(suite, "validate_registry_target", lambda **_kwargs: "live")
    monkeypatch.setattr(capacity_registry, "DATAPLANE_CONTEXT", "fake-context")
    monkeypatch.setattr(suite, "release_identity", lambda: {"release_id": "rel-a"})
    monkeypatch.setattr(suite, "artifact_dir", lambda *_args: tmp_path)
    monkeypatch.setattr(
        storm,
        "alert_rule_loaded",
        lambda: events.append("alert-rule") or wire.rule_loaded,
    )
    monkeypatch.setattr(suite, "register", register)
    monkeypatch.setattr(suite, "validate_registered_resources", lambda *_args: None)
    monkeypatch.setattr(suite, "RunResources", lambda *_args: SimpleNamespace())
    monkeypatch.setattr(suite, "synthetic_fleet", synthetic_fleet)
    monkeypatch.setattr(
        suite,
        "StormBreakerStep",
        lambda *args, **kwargs: FakeStep(events, *args, **kwargs),
    )
    monkeypatch.setattr(suite, "teardown", lambda **_kwargs: events.append("teardown"))
    monkeypatch.setattr(
        suite,
        "verify_fleet_cleanup",
        lambda *_args, **_kwargs: events.append("fleet-cleanup"),
    )
    monkeypatch.setattr(
        suite, "verdict", lambda _summary: ("PASS", {"fleet_counted": True})
    )
    wire.artifacts = tmp_path
    return wire


ARGS = ["--suite-id", RUN, "--allow-live-registry", "--confirm-live-registry", "x"]


def test_main_runs_the_step_inside_its_fleet_and_always_cleans_up(
    orchestration: SimpleNamespace,
) -> None:
    assert suite.main(ARGS) == 0

    assert orchestration.events == [
        "alert-rule",
        "register",
        "fleet-enter",
        "step-run",
        "close-breaker",
        "fleet-exit",
        "teardown",
        "fleet-cleanup",
    ]
    assert orchestration.fleet_kwargs["expected_trips"] == frozenset({CLUSTER})
    assert orchestration.fleet_kwargs["plan"]["fleet_agents_per_cluster"] == 192
    run = json.loads((orchestration.artifacts / "run.json").read_text())
    assert run["storm_breaker_plan"]["held_workflows"] == 4
    assert run["suite_id"] == RUN, "the recovery command adopts the run by it"
    summary = json.loads((orchestration.artifacts / "summary.json").read_text())
    assert summary["status"] == "PASS"
    assert summary["breaker_exit"]["state_after"] == "CLOSED"
    status = json.loads((orchestration.artifacts / "status.json").read_text())
    assert status["status"] == "ok"


def test_a_site_without_the_alert_rule_registers_nothing(
    orchestration: SimpleNamespace,
) -> None:
    orchestration.rule_loaded = False

    with pytest.raises(
        RuntimeError, match="carry no GpuFaultRemediationStormBreakerOpen"
    ):
        suite.main(ARGS)

    assert orchestration.events == ["alert-rule"]
    status = json.loads((orchestration.artifacts / "status.json").read_text())
    assert status["status"] == "aborted"


def test_a_step_that_raises_still_closes_the_breaker_and_tears_down(
    orchestration: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(FakeStep, "failure", KeyboardInterrupt())

    with pytest.raises(KeyboardInterrupt):
        suite.main(ARGS)

    assert orchestration.events[-5:] == [
        "step-run",
        "close-breaker",
        "fleet-exit",
        "teardown",
        "fleet-cleanup",
    ]
    history = json.loads(
        (orchestration.artifacts / "storm-breaker-history.json").read_text()
    )
    assert history == [{"phase": "trip"}], "the polls survive the failure"
    status = json.loads((orchestration.artifacts / "status.json").read_text())
    assert status["status"] == "aborted"


def test_a_signal_during_the_step_closes_the_breaker_and_tears_down(
    orchestration: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    """A SIGTERM used to end the step with the breaker open, its held work
    parked and the synthetic cluster registered; now it ends through the
    step's cleanup and the teardown, and a second signal there is only
    recorded."""

    def signalled(step: FakeStep, *, fleet_ready_at: float) -> dict[str, Any]:
        step.events.append("step-run")
        deliver_signal(signal.SIGTERM)
        return {}

    def close_breaker(step: FakeStep) -> dict[str, Any]:
        deliver_signal(signal.SIGINT)  # must not cut the hand-over short
        step.events.append("close-breaker")
        return {"state_before": "OPEN", "state_after": "CLOSED"}

    monkeypatch.setattr(FakeStep, "run", signalled)
    monkeypatch.setattr(FakeStep, "close_breaker", close_breaker)
    default_signal_dispositions(request)

    assert suite.main(ARGS) == 128 + signal.SIGTERM, "the exit status names SIGTERM"

    assert orchestration.events[-5:] == [
        "step-run",
        "close-breaker",
        "fleet-exit",
        "teardown",
        "fleet-cleanup",
    ], "the breaker hand-over, the fleet exit and the teardown all ran"
    status = json.loads((orchestration.artifacts / "status.json").read_text())
    assert status["status"] == "aborted", status
    assert status["reason"] == "RunTerminated: received SIGTERM", status
