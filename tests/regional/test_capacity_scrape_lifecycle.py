from __future__ import annotations

import copy
import io
import json
import re
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
import httpx
import pytest
import yaml  # type: ignore[import-untyped]
from botocore.credentials import Credentials

from scripts.e2e.regional import capacity_acceptance_base as base
from scripts.e2e.regional import capacity_acceptance_cases as cases
from tests.regional._cov95_capacity_support import capacity_fixture as capacity_fixture


class Clock:
    def __init__(self) -> None:
        self.now = 2_000_000_000.0
        self.sleeps: list[float] = []

    def time(self) -> float:
        return self.now

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class LifecycleHarness(cases.CapacityAcceptanceCases):
    def __init__(self, root: Path, clock: Clock) -> None:
        self.run_dir = root
        self.run_id = "cap-scrape-unit"
        self.case_id = "GF-REGIONAL-CAP-002"
        self.secret_name = "unit-capacity-registry"
        self.configmap_name = "unit-capacity-scripts"
        self.scrape_source_binding = {"source_sha256": "a" * 64}
        self.maintenance_deadline = datetime.fromtimestamp(
            clock.time() + 1800, timezone.utc
        )
        self.cap002_scrape_stopped = True
        self.active_probe = None
        self.clock = clock
        self.events: list[tuple[Any, ...]] = []
        self.resources = {
            ("secret", self.secret_name),
            ("configmap", self.configmap_name),
        }
        self.holds: set[str] = set()
        self.ready_problem = ""
        self.up_source_problem = ""
        self.resolve_values: list[tuple[str, float]] = []
        self.resolve_states: list[list[str]] = []
        self.start_error: BaseException | None = None
        self.stop_error: BaseException | None = None
        self.stop_proof: dict[str, Any] = {
            "cleanup_complete": True,
            "process_termination_proven": True,
            "pod_absent": True,
            "configmap_absent": True,
        }
        self.behavior_error: BaseException | None = None
        self.release_error = False
        self.refused_delete = ""
        self.scraper_running = False
        self.resolving = False
        self.released_at: float | None = None
        self.probe = base.Probe(
            case="CAP002",
            deployment="unit-capacity-probe",
            service="unit-capacity-probe",
            database="gpu_fault_unit_cap002",
            pod="unit-capacity-probe-pod",
            local_port=12345,
            url="http://127.0.0.1:12345",
            port_forward=None,
        )

    def deploy_probe(self, case: str, overrides: Any) -> base.Probe:
        self.events.append(("deploy", case))
        self.active_probe = self.probe
        self.resources.update(
            {("deployment", self.probe.deployment), ("service", self.probe.service)}
        )
        return self.probe

    def kubectl(self, *args: str, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        action, kind = args[:2]
        output = ""
        if action == "delete":
            self.events.append(("delete", kind))
            if kind == self.refused_delete:
                raise OSError("synthetic delete failure")
            self.resources.discard((kind, args[2]))
        elif args[:2] == ("get", "pods"):
            items = (
                [{"metadata": {"name": self.probe.pod}}]
                if ("deployment", self.probe.deployment) in self.resources
                else []
            )
            output = json.dumps({"items": items})
        else:
            assert action == "get", "the cleanup fake permits no other commands"
            output = "{}" if (kind, args[2]) in self.resources else ""
        return subprocess.CompletedProcess(args, 0, output, "")

    def drop_database_fallback(self, database: str) -> None:
        self.events.append(("database", database))

    def _cap002_behavior(self, probe: Any, case_dir: Any) -> dict[str, Any]:
        self.events.append(("behavior",))
        if self.behavior_error is not None:
            raise self.behavior_error
        return {
            "passed": True,
            "initial_status_counts": {503: 20},
            "retry_after_values": ["2"],
            "retry_status_counts": {200: 1},
            "store_io_rejections": 3,
        }

    def probe_control(self, probe: Any, path: str, payload: Any) -> dict[str, Any]:
        tag = payload["tag"]
        self.events.append((path, tag))
        if path.endswith("/hold"):
            self.holds.add(tag)
        else:
            self.holds.discard(tag)
            if tag == "alert":
                self.released_at = self.clock.time()
            if self.release_error:
                raise OSError("synthetic private cleanup diagnostic")
        return {"tag": tag}

    def metrics(self, url: str) -> list[tuple[str, dict[str, str], float]]:
        return [
            ("gpu_fault_store_io_in_flight", {}, 4.0),
            ("gpu_fault_store_io_max_in_flight", {}, 4.0),
            ("gpu_fault_store_io_rejections_total", {"reason": "capacity"}, 3.0),
        ]

    def vector(self, value: str, *, age: float = 0) -> dict[str, Any]:
        return {
            "status": "success",
            "data": {
                "resultType": "vector",
                "result": [
                    {
                        "metric": {"pod": self.probe.pod, "capacity_run": self.run_id},
                        "value": [self.clock.time() - age, value],
                    }
                ],
            },
        }

    def amp_request(self, method: str, path: str, params: Any = None) -> dict[str, Any]:
        assert self.scraper_running, "target samples require a running companion"
        assert (method, path) == ("POST", "/api/v1/query"), (
            "the fake only serves instant AMP queries"
        )
        query = params["query"]
        self.events.append(("query", query, params.get("time")))
        if query.startswith("timestamp(up{"):
            timestamp = self.clock.time()
            if self.up_source_problem == "stale":
                timestamp -= 91
            elif self.up_source_problem == "future":
                timestamp += 1
            elif self.up_source_problem == "before-start":
                timestamp = 1_999_999_999
            elif self.up_source_problem == "nan":
                timestamp = float("nan")
            return self.vector(str(timestamp))
        if not query.startswith("up{"):
            if "alert" in self.holds:
                return self.vector("1")
            value, age = self.resolve_values.pop(0) if self.resolve_values else ("0", 0)
            self.resolving = True
            self.events.append(("resolved-ratio", value, age))
            return self.vector(value, age=age)
        response = self.vector("1")
        row = response["data"]["result"][0]
        problem = self.ready_problem
        if problem == "status":
            response["status"] = "error"
        elif problem == "shape":
            response["data"]["resultType"] = "matrix"
        elif problem == "missing":
            response["data"]["result"] = []
        elif problem == "multiple":
            response["data"]["result"].append(copy.deepcopy(row))
        elif problem in {"pod", "capacity_run"}:
            row["metric"][problem] = "unrelated-target"
        elif problem == "down":
            row["value"][1] = "0"
        elif problem == "nan":
            row["value"][1] = "NaN"
        elif problem == "stale":
            row["value"][0] -= 91
        elif problem == "before-start":
            row["value"][0] = 1_999_999_999
        elif problem == "future":
            row["value"][0] += 1
        elif problem == "boolean":
            row["value"][1] = True
        return response

    def alert_states(self, alert_name: str) -> list[str]:
        if "alert" in self.holds:
            states = ["firing"]
        elif self.resolving and self.resolve_states:
            states = self.resolve_states.pop(0)
        else:
            states = []
        self.events.append(("states", list(states)))
        return states


class FakeCompanion:
    def __init__(
        self,
        harness: LifecycleHarness,
        probe: base.Probe,
        *,
        expected_source: dict[str, Any],
        deadline: datetime,
    ) -> None:
        assert expected_source == harness.scrape_source_binding, (
            "the planned source binding must reach the companion unchanged"
        )
        assert deadline == harness.maintenance_deadline, (
            "the companion must not receive a renewed maintenance deadline"
        )
        self.harness = harness
        self.probe = probe

    @property
    def selector(self) -> str:
        return f'pod="{self.probe.pod}",capacity_run="{self.harness.run_id}"'

    def start(self) -> dict[str, Any]:
        self.harness.events.append(("scrape-start",))
        self.harness.scraper_running = True
        if self.harness.start_error is not None:
            raise self.harness.start_error
        return {"started": True}

    def stop(self) -> dict[str, Any]:
        self.harness.events.append(("scrape-stop",))
        summary = json.loads(
            (self.harness.run_dir / "CAP-002/summary.json").read_text()
        )
        assert summary["status"] != "PASS", (
            "case evidence cannot claim PASS before companion shutdown"
        )
        if self.harness.stop_error is not None:
            raise self.harness.stop_error
        self.harness.scraper_running = False
        return dict(self.harness.stop_proof)


@pytest.fixture
def lifecycle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> LifecycleHarness:
    clock = Clock()
    harness = LifecycleHarness(tmp_path, clock)
    monkeypatch.setattr(cases, "time", clock)
    monkeypatch.setattr(cases, "ScrapeCompanion", FakeCompanion)

    def external(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("lifecycle tests must not reach a real command or network")

    monkeypatch.setattr(base, "run_fixture_command", external)
    monkeypatch.setattr(boto3, "Session", external)
    monkeypatch.setattr(boto3, "client", external)
    monkeypatch.setattr(subprocess, "Popen", external)
    monkeypatch.setattr(httpx, "get", external)
    return harness


@pytest.mark.parametrize(
    "problem",
    [
        "status",
        "shape",
        "missing",
        "multiple",
        "pod",
        "capacity_run",
        "down",
        "nan",
        "stale",
        "before-start",
        "future",
        "boolean",
    ],
)
def test_unproven_scrape_cannot_start_behavior_or_holds(
    lifecycle: LifecycleHarness, problem: str
) -> None:
    lifecycle.ready_problem = problem
    with pytest.raises(base.CapError, match="scrape readiness"):
        lifecycle.case_002_v2()
    assert ("behavior",) not in lifecycle.events, "unready AMP forbids claim traffic"
    assert not any(event[0] == "/__cap__/hold" for event in lifecycle.events), (
        "unready AMP forbids saturation holds"
    )
    ready_queries = [
        event
        for event in lifecycle.events
        if event[0] == "query" and event[1].startswith("up{")
    ]
    assert len(ready_queries) == 28, "readiness must retain the finite retry budget"
    assert lifecycle.clock.sleeps == [15] * 27, "polling stays at fifteen seconds"
    assert lifecycle.cleanup_all() == [], (
        "a stopped companion permits dependency cleanup"
    )
    assert not lifecycle.resources, "owned dependencies must not leak after proven stop"


def test_fresh_zero_and_global_resolution_precede_stop_and_dependency_cleanup(
    lifecycle: LifecycleHarness,
) -> None:
    lifecycle.resolve_values = [("1", 0), ("0", 91), ("0", 0), ("0", 0)]
    lifecycle.resolve_states = [[], [], ["pending"], []]
    result = lifecycle.case_002_v2()
    assert result["status"] == "PASS", "all lifecycle proofs are required for PASS"
    assert result["scrape_ready"] and result["scrape_stopped"], (
        "readiness and shutdown are separate proofs"
    )
    assert lifecycle.cleanup_all() == [], "common cleanup follows case cleanup"
    assert not lifecycle.resources, "all owned dependencies must be absent"
    events = lifecycle.events
    stop = events.index(("scrape-stop",))
    assert events[stop - 2 : stop] == [("resolved-ratio", "0", 0), ("states", [])], (
        "only a fresh zero and globally empty alert state can finish resolution"
    )
    assert [event[1:] for event in events if event[0] == "resolved-ratio"] == [
        ("1", 0),
        ("0", 91),
        ("0", 0),
        ("0", 0),
    ], "saturated, stale and globally pending samples cannot finish resolution"
    assert (
        events.index(("/__cap__/release", "alert"))
        < stop
        < events.index(("delete", "deployment"))
        < events.index(("database", lifecycle.probe.database))
        < events.index(("delete", "secret"))
    ), "release, resolution, scraper, probe, database and Secret cleanup stay ordered"
    queries = [event[1] for event in events if event[0] == "query"]
    selector = f'pod="{lifecycle.probe.pod}",capacity_run="{lifecycle.run_id}"'
    assert queries[0] == f"up{{{selector}}}", "readiness selects only this run's target"
    ready_queries = [event for event in events if event[0] == "query"][:2]
    assert queries[1] == f"timestamp(up{{{selector}}})", (
        "source age comes from timestamp(up), not the query evaluation timestamp"
    )
    assert ready_queries[0][2] == ready_queries[1][2] is not None, (
        "the up value and its source timestamp must share an evaluation instant"
    )
    assert all(
        f"gpu_fault_store_io_in_flight{{{selector}}}" in query
        and f"gpu_fault_store_io_max_in_flight{{{selector}}}" in query
        for query in queries[2:]
    ), "production targets must never substitute for the companion"
    cutoff = lifecycle.released_at
    assert cutoff is not None
    assert all(
        f"timestamp(gpu_fault_store_io_{name}{{{selector}}}) >= {cutoff}" in queries[-1]
        and f"time() - timestamp(gpu_fault_store_io_{name}{{{selector}}}) <= 90"
        in queries[-1]
        and f"time() - timestamp(gpu_fault_store_io_{name}{{{selector}}}) >= 0"
        in queries[-1]
        for name in ("in_flight", "max_in_flight")
    ), "computed ratio timestamps cannot substitute for post-release source samples"


@pytest.mark.parametrize("problem", ["stale", "future", "before-start", "nan"])
def test_fresh_amp_evaluation_does_not_make_an_old_up_source_ready(
    lifecycle: LifecycleHarness, problem: str
) -> None:
    lifecycle.up_source_problem = problem
    with pytest.raises(base.CapError, match="scrape readiness"):
        lifecycle.case_002_v2()
    polls = json.loads(
        (lifecycle.run_dir / "CAP-002/scrape-ready-poll.json").read_text()
    )
    assert len(polls) == 28 and all(row["target_up"] == 1 for row in polls), (
        "a fresh evaluation containing up=1 does not establish source freshness"
    )
    assert all(row["ready"] is False for row in polls), (
        "the underlying timestamp(up) value must independently pass freshness checks"
    )
    assert ("behavior",) not in lifecycle.events, "stale source data forbids traffic"
    assert not any(event[0] == "/__cap__/hold" for event in lifecycle.events), (
        "stale source data forbids saturation even when evaluation timestamps are fresh"
    )


def test_empty_alert_inventory_does_not_make_a_saturated_target_pass(
    lifecycle: LifecycleHarness,
) -> None:
    lifecycle.resolve_values = [("1", 0)] * 28
    with pytest.raises(base.CapError, match="CAP-002 failed"):
        lifecycle.case_002_v2()
    summary = json.loads((lifecycle.run_dir / "CAP-002/summary.json").read_text())
    assert summary["status"] == "FAIL", "a stale last saturated sample cannot pass"
    assert summary["alert_resolved_after_release"] is False, (
        "empty global alert state alone is insufficient"
    )
    assert (
        len([event for event in lifecycle.events if event[0] == "resolved-ratio"]) == 28
    ), "unresolved saturation must exhaust only the existing bounded retry count"


@pytest.mark.parametrize("start_failed", [False, True])
def test_unproven_stop_retains_probe_database_secret_and_configmap(
    lifecycle: LifecycleHarness, start_failed: bool
) -> None:
    primary = OSError("synthetic original failure")
    if start_failed:
        lifecycle.start_error = primary
    else:
        lifecycle.behavior_error = primary
    lifecycle.stop_error = OSError("synthetic private cleanup diagnostic")
    with pytest.raises(OSError) as caught:
        lifecycle.case_002_v2()
    assert caught.value is primary, "cleanup must preserve the original failure object"
    assert lifecycle.cleanup_all() == [
        "CAP002 scrape shutdown is unverified; resources retained"
    ], "outer cleanup cannot discard a running companion's dependencies"
    with pytest.raises(base.CapError, match="scrape shutdown"):
        lifecycle.cleanup_probe(lifecycle.probe)
    with pytest.raises(base.CapError, match="scrape shutdown"):
        lifecycle.cleanup_common()
    assert lifecycle.active_probe is lifecycle.probe, (
        "recovery retains the probe tracker"
    )
    assert not any(event[0] in {"delete", "database"} for event in lifecycle.events), (
        "unknown shutdown permits no dependency deletion"
    )
    summary = json.loads((lifecycle.run_dir / "CAP-002/summary.json").read_text())
    assert summary["status"] == "FAIL" and summary["scrape_stopped"] is False
    assert summary["cleanup_errors"] == ["scrape companion stop: OSError"], (
        "cleanup evidence includes types, never private exception messages"
    )
    assert "synthetic private cleanup diagnostic" not in json.dumps(summary)


@pytest.mark.parametrize("flag", ["cleanup_complete", "process_termination_proven"])
@pytest.mark.parametrize("value", [None, False, 1], ids=["missing", "false", "integer"])
def test_invalid_stop_proof_retains_probe_database_secret_and_configmap(
    lifecycle: LifecycleHarness, flag: str, value: Any
) -> None:
    if value is None:
        lifecycle.stop_proof.pop(flag)
    else:
        lifecycle.stop_proof[flag] = value
    with pytest.raises(base.CapError, match="CAP-002 failed"):
        lifecycle.case_002_v2()
    assert lifecycle.cap002_scrape_stopped is False, (
        "only two explicit True flags can establish companion shutdown"
    )
    assert lifecycle.cleanup_all() == [
        "CAP002 scrape shutdown is unverified; resources retained"
    ], "outer cleanup cannot infer termination from invalid proof"
    assert lifecycle.active_probe is lifecycle.probe, (
        "the dependent probe remains tracked for recovery"
    )
    assert lifecycle.resources == {
        ("deployment", lifecycle.probe.deployment),
        ("service", lifecycle.probe.service),
        ("secret", lifecycle.secret_name),
        ("configmap", lifecycle.configmap_name),
    }, "invalid stop proof must retain all dependent Kubernetes resources"
    assert not any(event[0] in {"delete", "database"} for event in lifecycle.events), (
        "an invalid proof cannot authorize probe or database deletion"
    )
    summary = json.loads((lifecycle.run_dir / "CAP-002/summary.json").read_text())
    assert summary["status"] == "FAIL" and summary["scrape_stopped"] is False, (
        "invalid stop proof must not produce a passing cleanup verdict"
    )
    assert summary["cleanup_errors"] == ["scrape companion stop: CapError"], (
        "proof rejection remains a safe, explicit cleanup error"
    )


def test_partial_start_is_stopped_before_dependencies_are_removed(
    lifecycle: LifecycleHarness,
) -> None:
    primary = OSError("synthetic partial start")
    lifecycle.start_error = primary
    with pytest.raises(OSError) as caught:
        lifecycle.case_002_v2()
    assert caught.value is primary, "start failure must remain the primary failure"
    assert lifecycle.cleanup_all() == []
    assert lifecycle.events.index(("scrape-stop",)) < lifecycle.events.index(
        ("delete", "deployment")
    ), "partial start still requires owned companion cleanup first"
    assert not any(
        event[0] in {"query", "behavior", "/__cap__/hold"} for event in lifecycle.events
    ), "a failed start cannot authorize target queries or work"


def test_stop_failure_cannot_turn_successful_behavior_into_a_pass(
    lifecycle: LifecycleHarness,
) -> None:
    lifecycle.stop_error = OSError("synthetic private cleanup diagnostic")
    with pytest.raises(base.CapError, match="CAP-002 failed"):
        lifecycle.case_002_v2()
    summary = json.loads((lifecycle.run_dir / "CAP-002/summary.json").read_text())
    assert summary["alert_fired"] and summary["alert_resolved_after_release"], (
        "the refusal must come after successful behavior and alert resolution"
    )
    assert summary["status"] == "FAIL" and summary["scrape_stopped"] is False, (
        "unproven companion shutdown independently forbids PASS"
    )
    assert lifecycle.cleanup_all(), "outer cleanup must retain dependent resources"
    assert not any(event[0] == "delete" for event in lifecycle.events), (
        "a successful behavior phase does not authorize unproven cleanup"
    )


def test_probe_cleanup_failure_retains_common_dependencies(
    lifecycle: LifecycleHarness,
) -> None:
    lifecycle.refused_delete = "deployment"
    with pytest.raises(base.CapError, match="CAP-002 failed"):
        lifecycle.case_002_v2()
    errors = lifecycle.cleanup_all()
    assert errors and lifecycle.cap002_scrape_stopped, (
        "scraper closure does not prove probe cleanup"
    )
    assert ("secret", lifecycle.secret_name) in lifecycle.resources
    assert ("configmap", lifecycle.configmap_name) in lifecycle.resources
    assert not any(event[0] == "database" for event in lifecycle.events), (
        "a retained probe must not lose its database"
    )


def test_cleanup_errors_accumulate_without_disclosing_or_replacing_primary_failure(
    lifecycle: LifecycleHarness,
) -> None:
    primary = RuntimeError("synthetic behavior failure")
    lifecycle.behavior_error = primary
    lifecycle.release_error = True
    lifecycle.stop_error = OSError("synthetic private cleanup diagnostic")
    with pytest.raises(RuntimeError) as caught:
        lifecycle.case_002_v2()
    assert caught.value is primary
    assert primary.__notes__ == [
        "release behavior: OSError",
        "release alert: OSError",
        "scrape companion stop: OSError",
    ], "all cleanup failures are safe notes on the original exception"


def test_cleanup_interruption_preserves_primary_failure_and_retains_dependencies(
    lifecycle: LifecycleHarness,
) -> None:
    primary = OSError("synthetic behavior failure")
    lifecycle.behavior_error = primary
    lifecycle.stop_error = KeyboardInterrupt()
    with pytest.raises(OSError) as caught:
        lifecycle.case_002_v2()
    assert caught.value is primary, (
        "a cleanup interruption cannot replace the case failure"
    )
    assert primary.__notes__ == ["scrape companion stop: KeyboardInterrupt"]
    assert lifecycle.cleanup_all(), (
        "unknown stop must leave recovery dependencies intact"
    )
    assert not any(event[0] == "delete" for event in lifecycle.events), (
        "interrupted scraper shutdown must not delete dependent resources"
    )


def test_cleanup_evidence_failure_cannot_replace_the_original_exception(
    lifecycle: LifecycleHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    primary = OSError("synthetic behavior failure")
    lifecycle.behavior_error = primary
    original = base.write_json

    def write(path: Path, value: Any) -> None:
        if path.name == "cleanup.json":
            raise OSError("synthetic private evidence diagnostic")
        original(path, value)

    monkeypatch.setattr(cases, "write_json", write)
    with pytest.raises(OSError) as caught:
        lifecycle.case_002_v2()
    assert caught.value is primary, "failed evidence writing must preserve case failure"
    assert primary.__notes__ == ["cleanup evidence: OSError"], (
        "evidence failures are retained without copying their private text"
    )
    assert lifecycle.cleanup_all() == [], "proven process shutdown still allows cleanup"


@pytest.mark.parametrize(
    "invalid", ["missing-source", "missing-deadline", "naive", "expired"]
)
def test_missing_plan_or_maintenance_binding_cannot_deploy(
    lifecycle: LifecycleHarness, invalid: str
) -> None:
    if invalid == "missing-source":
        lifecycle.scrape_source_binding = None
    elif invalid == "missing-deadline":
        lifecycle.maintenance_deadline = None
    elif invalid == "naive":
        lifecycle.maintenance_deadline = datetime(2033, 1, 1)
    else:
        lifecycle.maintenance_deadline = datetime.fromtimestamp(
            lifecycle.clock.time(), timezone.utc
        )
    with pytest.raises(base.CapError, match="plan-bound|deadline"):
        lifecycle.case_002_v2()
    assert lifecycle.events == [], "missing bindings fail before any remote work"


def test_readiness_wait_cannot_extend_the_maintenance_deadline(
    lifecycle: LifecycleHarness,
) -> None:
    lifecycle.ready_problem = "down"
    lifecycle.maintenance_deadline = datetime.fromtimestamp(
        lifecycle.clock.time() + 7, timezone.utc
    )
    with pytest.raises(base.CapError, match="deadline"):
        lifecycle.case_002_v2()
    assert lifecycle.clock.sleeps == [7], "poll sleeps use only the remaining window"
    assert ("behavior",) not in lifecycle.events, "an expired window authorizes no load"
    assert lifecycle.cap002_scrape_stopped, "expiry must still attempt owned cleanup"


@pytest.mark.parametrize("case", ["CAP001", "CAP002", "CAP003", "CAP004"])
def test_private_probe_label_and_credentials_never_match_production_scraping(
    capacity: Any, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    harness, api = capacity
    monkeypatch.setattr(
        harness, "_wait_for_probe", lambda **kwargs: harness.active_probe
    )
    harness.deploy_probe(case, {})
    deployment = next(item for item in api.applied if item["kind"] == "Deployment")
    template = deployment["spec"]["template"]
    app = template["metadata"]["labels"]["app"]
    assert app == "gpu-fault-capacity-probe", "private probes have their own app label"
    assert app not in base.PRODUCTION_APPS, "private probes are not production roles"
    manifest = (
        Path(__file__).resolve().parents[2]
        / "deploy/observability/adot-control-plane.yaml"
    )
    configmap = next(
        item
        for item in yaml.safe_load_all(manifest.read_text())
        if item["kind"] == "ConfigMap"
    )
    config = yaml.safe_load(configmap["data"]["collector.yaml"])
    job = config["receivers"]["prometheus"]["config"]["scrape_configs"][0]
    keep = next(
        item
        for item in job["relabel_configs"]
        if item.get("action") == "keep"
        and item["source_labels"] == ["__meta_kubernetes_pod_label_app"]
    )
    assert re.fullmatch(keep["regex"], app) is None, (
        "production ADOT must not send its execution token to any private probe"
    )
    env = {item["name"]: item for item in template["spec"]["containers"][0]["env"]}
    assert env["GPU_FAULT_EXECUTION_TOKEN"] == {
        "name": "GPU_FAULT_EXECUTION_TOKEN",
        "valueFrom": {
            "secretKeyRef": {"name": harness.secret_name, "key": "execution-token"}
        },
    }, "the probe still requires its private execution credential"
    assert job["authorization"] == {
        "type": "Bearer",
        "credentials_file": "/etc/gpu-fault/execution-token",
    }, "production scraper authentication must remain enabled"


def test_constructor_keeps_optional_binding_without_requiring_it_for_other_cases(
    capacity: Any,
) -> None:
    harness, api = capacity
    assert (
        harness.scrape_source_binding is None and harness.maintenance_deadline is None
    )
    before = len(api.calls)
    binding = {"source_sha256": "b" * 64}
    deadline = datetime(2033, 1, 1, tzinfo=timezone.utc)
    harness.__init__(
        site_path=harness.site_path,
        run_dir=harness.root_run_dir,
        case_id="GF-REGIONAL-CAP-002",
        predecessor={"valid": True},
        scrape_source_binding=binding,
        maintenance_deadline=deadline,
    )
    assert harness.scrape_source_binding == binding
    assert harness.maintenance_deadline is deadline
    assert len(api.calls) - before == 2, (
        "construction adds no scrape reads or mutations"
    )


def test_cap002_amp_http_timeout_uses_only_remaining_maintenance_time(
    capacity: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    harness, _api = capacity
    now = datetime.now(timezone.utc).timestamp()
    harness.case_id = "GF-REGIONAL-CAP-002"
    harness.maintenance_deadline = datetime.fromtimestamp(now + 7, timezone.utc)
    monkeypatch.setattr(time, "time", lambda: now)
    monkeypatch.setattr(
        boto3,
        "Session",
        lambda: SimpleNamespace(
            get_credentials=lambda: Credentials("unit-access", "unit-signing-key")
        ),
    )
    timeouts: list[float] = []

    def open_request(request: Any, *, timeout: float) -> io.BytesIO:
        assert request.full_url.startswith("https://aps-workspaces."), (
            "AMP requests retain verified HTTPS"
        )
        assert request.get_header("Authorization"), "AMP requests remain SigV4 signed"
        timeouts.append(timeout)
        return io.BytesIO(b'{"status":"success"}')

    monkeypatch.setattr(urllib.request, "urlopen", open_request)
    assert harness.amp_request("GET", "/api/v1/alerts") == {"status": "success"}
    assert timeouts == [7], "HTTP cannot outlive the remaining maintenance window"
