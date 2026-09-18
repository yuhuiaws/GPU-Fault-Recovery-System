from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from scripts.e2e.regional import capacity_acceptance_cases as cases
from tests.execution.test_cluster_executor_lease_and_report import remote_command
from tests.regional._cov95_cap_cases import CapacityModel, Clock, InlinePool
from tests.regional.test_capacity_acceptance_safety import storm


@pytest.fixture
def model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    def build(failure: str = "") -> CapacityModel:
        clock = Clock()
        value = CapacityModel(tmp_path, clock, failure=failure)
        value.scrape_source_binding = {"source_sha256": "a" * 64}
        value.maintenance_deadline = datetime.fromtimestamp(
            clock.time() + 1800, timezone.utc
        )
        value.cap002_scrape_stopped = True
        monkeypatch.setattr(cases, "time", clock)
        monkeypatch.setattr(cases, "ThreadPoolExecutor", InlinePool)
        monkeypatch.setattr(cases, "httpx", SimpleNamespace(Client=value.client))
        monkeypatch.setattr(
            "gpu_fault.cluster_executor.regional_client.urlopen", value.urlopen
        )
        monkeypatch.setattr(
            cases,
            "ScrapeCompanion",
            lambda _harness, probe, **_kw: SimpleNamespace(
                selector=f'pod="{probe.pod}",capacity_run="{value.run_id}"',
                start=lambda: {"started": True},
                stop=lambda: {
                    "cleanup_complete": True,
                    "process_termination_proven": True,
                },
            ),
        )
        original_amp = value.amp_request

        def amp(method: str, path: str, params: Any) -> dict[str, Any]:
            response = original_amp(method, path, params)
            source_timestamp = params["query"].startswith("timestamp(up{")
            ready = source_timestamp or params["query"].startswith("up{")
            response["data"]["resultType"] = "vector"
            response["data"]["result"] = (
                []
                if failure == "alert" and "alert" in value.holds and not ready
                else [
                    {
                        "metric": {"pod": "cap-probe", "capacity_run": value.run_id},
                        "value": [
                            clock.time(),
                            str(clock.time())
                            if source_timestamp
                            else "1"
                            if ready or "alert" in value.holds
                            else "0",
                        ],
                    }
                ]
            )
            return response

        monkeypatch.setattr(value, "amp_request", amp)
        return value

    return build


@pytest.mark.parametrize(
    "field,value,problem",
    [
        ("a_status_counts", {202: 3000}, "cluster A was never rejected with 429"),
        ("a_retry_after_values", ["1"], "cluster A Retry-After is not exactly 2"),
        (
            "b_baseline_status_counts",
            {202: 14},
            "cluster B baseline was not accepted end to end",
        ),
        (
            "b_status_counts",
            {202: 59},
            "cluster B was not accepted end to end during the storm",
        ),
        ("a_rejections", 0.0, "no admission rejections were counted for cluster A"),
        ("b_rejections", 1.0, "cluster B was rejected"),
        ("queue_drained", False, "the processor queue did not drain"),
    ],
)
def test_storm_verdict_requires_each_independent_capacity_promise(
    field: str, value: Any, problem: str
) -> None:
    result = storm()
    assert cases.cap001_failures(result) == []
    if field.endswith("_rejections"):
        result["metric_maxima"][field] = value
    else:
        result[field] = value
    assert cases.cap001_failures(result) == [problem]


@pytest.mark.parametrize("failure", ["", "drain", "latency"])
def test_storm_case_accounts_for_both_streams_and_keeps_failed_verdict(
    model: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: str
) -> None:
    value = model(failure)
    value.clock.sleep_scale = 60 if failure == "drain" else 1
    sent = []

    def send(
        _harness: Any,
        _client: Any,
        phase: str,
        cluster: int,
        sequence: int,
        scheduled: float,
    ) -> dict[str, Any]:
        sent.append((phase, cluster, sequence, scheduled))
        rejected = cluster == 0 and sequence >= 1000
        return {
            "phase": phase,
            "cluster": cluster,
            "status": 429 if rejected else 202,
            "retry_after": "2" if rejected else None,
            "transport_retries": 0,
            "latency_ms": 0
            if failure == "latency"
            else (1 if phase == "baseline" else 1.5),
        }

    def monitor(
        _harness: Any, _probe: Any, _stop: Any, maxima: Any, samples: Any, _errors: Any
    ) -> None:
        maxima.update(queue_depth=20, a_rejections=2000, b_rejections=0)
        samples.append({"queue_depth": 20})

    monkeypatch.setattr(cases, "cap001_send", send)
    monkeypatch.setattr(cases, "cap001_monitor", monitor)
    if failure:
        with pytest.raises(cases.CapError, match="CAP-001 failed"):
            value.case_001()
    else:
        result = value.case_001()
        assert result["status"] == "PASS"
    summary = json.loads((tmp_path / "CAP-001/summary.json").read_text())
    assert summary["status"] == ("FAIL" if failure else "PASS")
    assert summary["a_status_counts"] == {"202": 1000, "429": 2000}
    assert summary["b_status_counts"] == {"202": 60}
    assert summary["b_baseline_status_counts"] == {"202": 15}
    assert summary["queue_drained"] is (failure != "drain")
    assert summary["b_p95_degradation_ratio"] == (None if failure == "latency" else 1.5)
    assert Counter((phase, cluster) for phase, cluster, *_rest in sent) == {
        ("baseline", 1): 15,
        ("storm", 0): 3000,
        ("storm", 1): 60,
    }
    assert [
        sequence
        for phase, _cluster, sequence, _scheduled in sent
        if phase == "baseline"
    ] == list(range(15))
    assert value.events[-1] == ("cleanup", "cap-probe")
    assert all(client.closed for client in value.clients), (
        "storm case leaked its client"
    )


@pytest.mark.parametrize(
    "failure",
    [
        "",
        "behavior",
        "saturation",
        "alert",
        "release",
        "resolution-read",
        "unresolved",
        "cleanup",
    ],
)
def test_saturation_case_requires_behavior_alert_resolution_and_cleanup(
    model: Any, tmp_path: Path, failure: str
) -> None:
    value = model(failure)
    if failure:
        with pytest.raises(cases.CapError):
            value.case_002_v2()
    else:
        result = value.case_002_v2()
        assert result["status"] == "PASS"
        assert result["initial_status_counts"] == {503: 20}
        assert result["retry_status_counts"] == {200: 1}, (
            "recovery must come from one successful product Executor retry"
        )
        assert result["alert_fired"] is True
    behavior = json.loads((tmp_path / "CAP-002/behavior.json").read_text())
    proof = behavior["executor_retry"]
    assert behavior["passed"] is (failure != "behavior"), (
        "only the behavior control may fail before the alert lifecycle"
    )
    assert behavior["initial_status_counts"] == {
        ("403" if failure == "behavior" else "503"): 20
    }, "all initial saturation requests must still be observed"
    assert proof["execution_path"] == (
        "production-ClusterActionExecutor.run/claim-backoff"
    ), "the case must execute the product claim loop"
    assert proof["adapter_count"] == proof["commands_executed"] == 0, (
        "capacity retry proof cannot execute a workload or node action"
    )
    assert proof["passed"] is (failure != "behavior"), (
        "a nonretryable response cannot become a successful retry proof"
    )
    assert [item["status"] for item in proof["attempts"]] == (
        [403] if failure == "behavior" else [503, 200]
    ), "the product must recover from 503 but stop on a nonretryable response"
    if failure == "behavior":
        assert proof["retry_delays_seconds"] == [], (
            "a nonretryable refusal cannot schedule another claim"
        )
    else:
        assert len(proof["retry_delays_seconds"]) == 1, (
            "exactly one retry must recover the isolated claim"
        )
        assert proof["retry_delays_seconds"][0] >= proof["poll_seconds"] == 2, (
            "recovery must retain the real Executor's configured claim backoff"
        )
        assert proof["attempts"][-1]["commands"] == 0, (
            "an empty database must not return executable work"
        )
    assert len(value.requests) == 20, "the initial saturation batch cannot shrink"
    assert len(value.wire_requests) == len(proof["attempts"]), (
        "every reported retry attempt must correspond to a serialized HTTP request"
    )
    assert all(
        item["payload"] == value.wire_requests[0]["payload"]
        and item["payload"]["executor_id"] == "cap002-real-executor"
        and item["headers"]["x-gpu-fault-cluster-id"] == "cap-cluster-000"
        and item["payload"]["execution_owners"] == []
        and item["payload"]["max_commands"] == 1
        for item in value.wire_requests
    ), "retries must preserve the real Executor's identity and empty adapter scope"
    assert all(response.closed for response in value.wire_responses), (
        "successful wire responses must be closed before probe cleanup"
    )
    summary = json.loads((tmp_path / "CAP-002/summary.json").read_text())
    cleanup = json.loads((tmp_path / "CAP-002/cleanup.json").read_text())
    assert summary["status"] == ("FAIL" if failure else "PASS")
    assert summary["alert_resolved_after_release"] is (
        failure not in {"resolution-read", "unresolved"}
    )
    assert cleanup["errors"] == summary["cleanup_errors"]
    expected_errors = {
        "release": ["release behavior: OSError", "release alert: OSError"],
        "resolution-read": ["alert resolution: OSError"],
        "cleanup": ["probe cleanup: OSError"],
    }
    assert cleanup["errors"] == expected_errors.get(failure, []), (
        "each cleanup failure must be observed at its intended lifecycle boundary"
    )
    if failure not in {"behavior", "saturation"}:
        assert summary["alert_fired"] is (failure != "alert"), (
            "cleanup failure injection must not mask the alert firing phase"
        )
    if failure == "cleanup":
        assert cleanup["passed"] is False
        assert "database_dropped" not in cleanup
    assert value.events[-1] == ("cleanup", "cap-probe")
    assert value.holds == set()
    expected_phases = ([] if failure == "behavior" else ["retry"]) + [
        "proof-finally",
        "cleanup",
    ]
    assert value.behavior_release_phases == expected_phases, (
        "retry release, proof finalization and case cleanup are distinct phases"
    )
    assert value.releases["behavior"] == len(expected_phases), (
        "every required behavior-hold release must still run"
    )
    assert all(client.closed for client in value.clients), (
        "claim client survived cleanup"
    )
    for path, request in value.requests:
        assert path == "/v1/regional/executors/claim"
        cluster_id = request["headers"]["X-GPU-Fault-Cluster-ID"]
        assert request["headers"]["Authorization"] == (
            f"Bearer synthetic-{int(cluster_id.rsplit('-', 1)[-1]):03d}"
        )


def test_saturation_preflight_refuses_an_already_firing_alert(model: Any) -> None:
    value = model("preactive")
    with pytest.raises(cases.CapError, match="already active"):
        value.case_002_v2()
    assert value.events == []
    assert value.clients == []


def test_saturation_transport_error_still_releases_and_cleans(
    model: Any, tmp_path: Path
) -> None:
    value = model("request")
    with pytest.raises(httpx.ReadTimeout):
        value.case_002_v2()
    assert value.events[-1] == ("cleanup", "cap-probe")
    assert value.holds == set()
    assert all(client.closed for client in value.clients), (
        "failed request leaked its client"
    )
    assert (
        json.loads((tmp_path / "CAP-002/summary.json").read_text())["status"] == "FAIL"
    )


def unexpected_claim(executor_id: str) -> dict[str, Any]:
    return {
        **remote_command("unexpected-capacity-command").model_dump(mode="json"),
        "cluster_id": "cap-cluster-000",
        "status": "LEASED",
        "lease_owner": executor_id,
        "lease_expires_at": (
            datetime.now(timezone.utc) + timedelta(seconds=120)
        ).isoformat(),
    }


@pytest.mark.parametrize("failure", ["retry-exhaustion", "unexpected-command"])
def test_real_capacity_retry_stops_without_executing_unproven_recovery(
    model: Any, tmp_path: Path, failure: str
) -> None:
    value = model(failure)
    probe = value.deploy_probe("CAP002", {})
    value.probe_control(probe, "/__cap__/hold", {"tag": "behavior"})
    if failure == "unexpected-command":
        value.wire_commands = [unexpected_claim("cap002-real-executor")]
    proof = cases.run_claim_retry_proof(
        probe.url,
        value.tokens[0],
        tmp_path,
        lambda: value.probe_control(probe, "/__cap__/release", {"tag": "behavior"}),
        poll_seconds=0.001,
    )
    statuses = [503] * 5 if failure == "retry-exhaustion" else [503, 200]
    assert [item["status"] for item in proof["attempts"]] == statuses, (
        "persistent rejection must stop at five wire attempts; nonempty 200 must stop"
    )
    assert len(value.wire_requests) == len(statuses), (
        "the bounded retry must not issue an unrecorded sixth HTTP request"
    )
    assert proof["passed"] is False, "neither refusal path proves empty-claim recovery"
    assert proof["adapter_count"] == proof["commands_executed"] == 0, (
        "even an authoritative unexpected claim must not reach an adapter"
    )
    assert len(proof["retry_delays_seconds"]) == len(statuses) - 1, (
        "every wire retry must pass through the product backoff"
    )
    assert all(
        delay >= 0.001 * 2**index
        for index, delay in enumerate(proof["retry_delays_seconds"])
    ), "bounded retries must preserve the configured exponential delays"
    assert value.releases["behavior"] == 1 and value.holds == set(), (
        "the first actual 503 must release the isolated hold exactly once"
    )
    if failure == "unexpected-command":
        assert proof["attempts"][-1]["commands"] == 1, (
            "the nonempty refusal must follow successful command deserialization"
        )
    assert all(response.closed for response in value.wire_responses), (
        "even a rejected nonempty response must close its HTTP stream"
    )


@pytest.mark.parametrize("failure", ["", "capacity"])
def test_claim_capacity_case_measures_every_cluster_count_before_deciding(
    model: Any, tmp_path: Path, failure: str
) -> None:
    value = model(failure)
    if failure:
        with pytest.raises(cases.CapError, match="CAP-003 failed"):
            value.case_003()
    else:
        result = value.case_003()
        assert result["status"] == "PASS"
    summary = json.loads((tmp_path / "CAP-003/summary.json").read_text())
    assert summary["status"] == ("FAIL" if failure else "PASS")
    assert [row["cluster_count"] for row in summary["scenarios"]] == [1, 5, 10, 20]
    assert [row["requests"] for row in summary["scenarios"]] == [60, 300, 600, 1200]
    assert len(value.requests) == 2160
    long_poll = summary["production_long_poll"]
    assert [row["cluster_count"] for row in long_poll] == [1, 5, 10, 20], (
        "the production long-poll phase must measure every cluster scale"
    )
    assert [row["requests"] for row in long_poll] == [6, 30, 60, 120], (
        "each cluster must have two Executors making three held claims each"
    )
    assert all(
        row["passed"] is True
        and row["wait_seconds"] == 20
        and row["effective_wait_seconds"] == 12
        and row["executor_replicas_per_cluster"] == 2
        and row["status_counts"] == {"200": row["requests"]}
        and row["hold_overhead_ms"]["p95"] < cases.CAP003_KNEE_P95_MS
        and all(sample["elapsed_seconds"] >= 12 for sample in row["samples"])
        for row in long_poll
    ), "a short-poll capacity failure must not replace the separate held-claim proof"
    assert len(value.wire_requests) == 217, (
        "one warmup and 216 held claims must traverse the real regional client"
    )
    assert value.wire_requests[0]["payload"]["executor_id"] == "cap003-warmup", (
        "the lazy listener must be started before taking its baseline"
    )
    assert all(
        item["payload"]["wait_seconds"] == 20
        and item["payload"]["max_commands"] == 1
        and item["payload"]["lease_seconds"] == 10
        and item["payload"]["execution_owners"] == ["gpu-fault-kubernetes-adapter"]
        and item["timeout"] == 35
        for item in value.wire_requests
    ), "the real wire request must preserve production wait, lease and HTTP budgets"
    assert Counter(
        item["headers"]["x-gpu-fault-cluster-id"] for item in value.wire_requests[1:]
    ) == {
        f"cap-cluster-{index:03d}": 6 * sum(index < scale for scale in (1, 5, 10, 20))
        for index in range(20)
    }, "every held claim must retain the intended cluster binding at every scale"
    assert all(response.closed for response in value.wire_responses), (
        "held-claim responses must be closed before cleanup"
    )
    assert value.events[-1] == ("cleanup", "cap-probe")
    assert all(client.closed for client in value.clients), (
        "capacity case leaked a client"
    )
    if failure:
        assert summary["knee"]["cluster_count"] == 1
        assert summary["recommendations"]["executor_poll_seconds"] is None
    else:
        assert summary["knee"] is None
        assert summary["recommendations"]["sustained_cluster_count_at_1rps"] == 20


def test_nonempty_long_poll_warmup_fails_before_listener_measurement(
    model: Any, tmp_path: Path
) -> None:
    value = model()
    value.wire_commands = [unexpected_claim("cap003-warmup")]
    with pytest.raises(cases.CapError, match="warmup found commands"):
        value.case_003()
    assert len(value.requests) == 2160, "all short-poll scales must still execute"
    assert len(value.wire_requests) == 1, "a failed warmup cannot start held-claim load"
    assert value.wire_requests[0]["payload"]["executor_id"] == "cap003-warmup", (
        "the refusal must come from the real serialized warmup"
    )
    assert ("/__cap__/claim-wakeups", "cap003") not in value.events, (
        "a nonempty warmup cannot authorize listener measurement"
    )
    assert value.events[-1] == ("cleanup", "cap-probe"), (
        "warmup refusal must still clean its isolated probe"
    )
    assert all(client.closed for client in value.clients), (
        "warmup refusal must not leak the short-poll client"
    )
    assert all(response.closed for response in value.wire_responses), (
        "the rejected warmup response must close"
    )
    assert not (tmp_path / "CAP-003/summary.json").exists(), (
        "an invalid warmup must not produce a capacity summary"
    )


@pytest.mark.parametrize("failure", ["claim-shape", "cpu", "connections"])
def test_claim_capacity_rejects_unknown_commands_or_invalid_measurements(
    model: Any, tmp_path: Path, failure: str
) -> None:
    value = model(failure)
    with pytest.raises(cases.CapError, match="claim|measurements"):
        value.case_003()
    assert all(client.closed for client in value.clients), (
        "measurement error leaked a client"
    )
    assert not (tmp_path / "CAP-003/summary.json").exists(), (
        "incomplete measurements produced a misleading capacity summary"
    )
