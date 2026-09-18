from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from scripts.e2e.regional import capacity_acceptance_base as base
from scripts.e2e.regional import capacity_acceptance_cases as cases
from scripts.e2e.regional import capacity_acceptance_traffic as traffic


def storm() -> dict[str, Any]:
    return {
        "a_status_counts": {202: 1000, 429: 2000},
        "a_retry_after_values": ["2"],
        "b_baseline_status_counts": {202: cases.CAP001_BASELINE_REQUESTS},
        "b_status_counts": {202: cases.CAP001_STORM_B_REQUESTS},
        "b_baseline_latency_ms": {"p95": 40.0},
        "b_latency_ms": {"p95": 70.0},
        "b_latency_factor": 2.0,
        "queue_depth_bound": cases.CAP001_A_QUEUE_CAP + cases.CAP001_QUEUE_SLACK,
        "metric_maxima": {
            "queue_depth": 21.0,
            "a_rejections": 2000.0,
            "b_rejections": 0.0,
        },
        "queue_drained": True,
    }


@pytest.mark.parametrize("bad", ["transport", "nan", "metrics", "sampling", "count"])
def test_cap001_rejects_incomplete_or_nonfinite_evidence(bad: str) -> None:
    result = storm()
    assert cases.cap001_failures(result) == [], "the complete control must pass"
    if bad == "transport":
        result["a_status_counts"] = {429: 2999, "transport-error:ConnectError": 1}
    elif bad == "nan":
        result["b_latency_ms"]["p95"] = float("nan")
    elif bad == "metrics":
        del result["metric_maxima"]["queue_depth"]
    elif bad == "sampling":
        result["metric_errors"] = ["ReadTimeout"]
    else:
        result["a_status_counts"] = {429: 1}

    assert cases.cap001_failures(result), (
        "unknown transport/measurement state must fail"
    )


@pytest.mark.parametrize("value", ["", "queue NaN\n", "queue +Inf\n", "queue -1\n"])
def test_required_metrics_cannot_default_to_zero(value: str) -> None:
    samples = base.CapHarnessBase.parse_metrics(value)

    with pytest.raises(base.CapError, match="missing or invalid"):
        base.CapHarnessBase.metric_value(samples, "queue")


def test_real_prometheus_label_escaping_is_parsed() -> None:
    samples = base.CapHarnessBase.parse_metrics('queue{cluster_id="a\\nb"} 0\n')

    assert samples == [("queue", {"cluster_id": "a\nb"}, 0.0)], (
        "metrics parsing must preserve Prometheus label semantics"
    )


def test_cap001_transport_retries_are_bounded_and_preserve_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[httpx.Request] = []

    def refuse(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        raise httpx.ConnectError("mocked connection failure", request=request)

    harness = SimpleNamespace(
        tokens=["synthetic-a", "synthetic-b"],
        cluster_headers=base.CapHarnessBase.cluster_headers,
    )
    monkeypatch.setattr(traffic.time, "sleep", lambda _seconds: None)
    with httpx.Client(
        base_url="http://probe.invalid", transport=httpx.MockTransport(refuse)
    ) as client:
        result = traffic.cap001_send(harness, client, "storm", 1, 7, 0)

    assert result["status"] == "transport-error:ConnectError", (
        "transport is not an HTTP success"
    )
    assert result["transport_retries"] == len(requests) == 3, (
        "retry count must stay bounded"
    )
    assert len({request.content for request in requests}) == 1, (
        "retries must be idempotent"
    )
    assert json.loads(requests[0].content)["cluster_id"] == "cap-cluster-001", (
        "the B stream must retain its own bound identity"
    )


def harness(tmp_path: Path) -> cases.CapacityAcceptanceCases:
    value = cases.CapacityAcceptanceCases.__new__(cases.CapacityAcceptanceCases)
    value.run_dir = tmp_path
    value.tokens = ["synthetic"] * 20
    value.run_id = "cap-unit"
    value.scrape_source_binding = {"source_sha256": "a" * 64}
    value.maintenance_deadline = datetime.now(timezone.utc) + timedelta(hours=1)
    value.cap002_scrape_stopped = True
    return value


@pytest.mark.parametrize("ratio", [None, "0.5", "NaN", "1"])
def test_cap002_requires_a_real_target_saturation_vector(
    ratio: str | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = harness(tmp_path)
    monkeypatch.setattr(value, "probe_control", lambda *_a: {})
    monkeypatch.setattr(
        value,
        "metrics",
        lambda _url: [
            ("gpu_fault_store_io_in_flight", {}, 4.0),
            ("gpu_fault_store_io_max_in_flight", {}, 4.0),
            ("gpu_fault_store_io_rejections_total", {"reason": "capacity"}, 3.0),
        ],
    )
    monkeypatch.setattr(
        value,
        "amp_request",
        lambda *_a, **_k: {
            "status": "success",
            "data": {
                "resultType": "vector",
                "result": []
                if ratio is None
                else [
                    {
                        "metric": {"pod": "probe", "capacity_run": value.run_id},
                        "value": [time.time(), ratio],
                    }
                ],
            },
        },
    )
    monkeypatch.setattr(value, "alert_states", lambda _name: ["firing"])
    monkeypatch.setattr(cases.time, "sleep", lambda _seconds: None)
    behavior = {
        "passed": True,
        "initial_status_counts": {503: 20},
        "retry_after_values": ["2"],
        "retry_status_counts": {200: 20},
        "store_io_rejections": 3.0,
    }

    result, passed = value.cap002_alert(
        SimpleNamespace(url="http://probe.invalid", pod="probe"),
        tmp_path,
        behavior,
        selector=f'pod="probe",capacity_run="{value.run_id}"',
    )

    assert passed is (ratio == "1"), (
        "a global firing alert cannot substitute for the target"
    )
    assert result["status"] == ("PASS" if ratio == "1" else "FAIL"), (
        "the recorded verdict must agree with target evidence"
    )


@pytest.mark.parametrize("states", [[], ["pending"], ["unknown"]])
def test_cap002_failure_still_releases_resolves_and_deletes(
    states: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = harness(tmp_path)
    events: list[str] = []
    polls: list[list[str]] = []
    probe = SimpleNamespace(url="http://probe.invalid", pod="probe")
    monkeypatch.setattr(value, "deploy_probe", lambda *_a: probe)
    monkeypatch.setattr(cases.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        cases,
        "ScrapeCompanion",
        lambda *_a, **_kw: SimpleNamespace(
            selector=f'pod="{probe.pod}",capacity_run="{value.run_id}"',
            start=lambda: {"started": True},
            stop=lambda: {"cleanup_complete": True, "process_termination_proven": True},
        ),
    )
    monkeypatch.setattr(
        value,
        "amp_request",
        lambda _method, _path, params: {
            "status": "success",
            "data": {
                "resultType": "vector",
                "result": [
                    {
                        "metric": {"pod": probe.pod, "capacity_run": value.run_id},
                        "value": [
                            time.time(),
                            str(time.time())
                            if params["query"].startswith("timestamp(up{")
                            else "1"
                            if params["query"].startswith("up{")
                            else "0",
                        ],
                    }
                ],
            },
        },
    )

    def alerts(_name: str) -> list[str]:
        result = states if polls else []
        polls.append(result)
        if len(polls) > 1:
            events.append("resolve")
        return result

    def control(_probe: Any, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        events.append(path + ":" + payload["tag"])
        if path.endswith("hold"):
            raise OSError("hold acknowledgement lost")
        return {}

    monkeypatch.setattr(value, "alert_states", alerts)
    monkeypatch.setattr(value, "probe_control", control)
    monkeypatch.setattr(
        value,
        "cleanup_probe",
        lambda _probe: events.append("delete") or {"database_dropped": True},
    )

    with pytest.raises(OSError, match="acknowledgement"):
        value.case_002_v2()

    assert (
        events.index("/__cap__/release:alert")
        < events.index("resolve")
        < events.index("delete")
    ), "failed behavior still requires release and resolution while the target exists"
    result = json.loads((tmp_path / "CAP-002/summary.json").read_text(encoding="utf-8"))
    assert result["status"] == "FAIL", "a failed injection must not be recorded as PASS"
    assert result["alert_resolved_after_release"] is (states == []), (
        "pending or unknown alerts are not resolved"
    )


@pytest.mark.parametrize(
    "bad", ["missing", "duplicate", "unchanged", "naive", "unknown"]
)
def test_cap004_requires_complete_unique_advancing_leases(bad: str) -> None:
    commands = [{"command_id": f"command-{index}"} for index in range(25)]
    progressions = [
        {
            "command_id": item["command_id"],
            "initial": "2026-09-12T00:00:00Z",
            "latest": "2026-09-12T00:00:10Z",
            "renewal_count": 1,
        }
        for item in commands
    ]
    assert cases.cap004_progression_errors(commands, progressions) == [], (
        "the complete advancing control must pass"
    )
    if bad == "missing":
        progressions.clear()
    elif bad == "duplicate":
        progressions[-1] = progressions[0]
    elif bad == "unchanged":
        progressions[0]["latest"] = progressions[0]["initial"]
    elif bad == "naive":
        progressions[0]["latest"] = "2026-09-12T00:00:10"
    else:
        progressions[0]["latest"] = None

    assert cases.cap004_progression_errors(commands, progressions), (
        "incomplete or non-advancing lease evidence cannot prove renewal"
    )


def test_capacity_http_clients_close_on_request_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = harness(tmp_path)
    probe = SimpleNamespace(url="http://probe.invalid", pod="probe")
    clients: list[httpx.Client] = []
    cleanup_calls: list[tuple[Any, list[bool]]] = []
    original = httpx.Client

    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("mocked request failure", request=request)

    def client(**kwargs: Any) -> httpx.Client:
        result = original(transport=httpx.MockTransport(fail), **kwargs)
        clients.append(result)
        return result

    def cleanup(current: Any) -> dict[str, Any]:
        cleanup_calls.append((current, [client.is_closed for client in clients]))
        return {"database_dropped": True, "residual_probe_pods": []}

    monkeypatch.setattr(cases.httpx, "Client", client)
    monkeypatch.setattr(value, "deploy_probe", lambda *_a: probe)
    monkeypatch.setattr(value, "cleanup_probe", cleanup)
    monkeypatch.setattr(value, "pod_cpu_usage_usec", lambda _pod: 1)
    monkeypatch.setattr(value, "isolated_db_connections", lambda _pod: 1)
    monkeypatch.setattr(value, "kubectl", lambda *_a: SimpleNamespace(stdout="{}"))
    monkeypatch.setattr(cases.time, "sleep", lambda _seconds: None)

    with pytest.raises(httpx.ReadTimeout, match="mocked request failure"):
        value.case_003()

    assert clients and all(client.is_closed for client in clients), (
        "a failed load request must not retain its client pool"
    )
    assert cleanup_calls == [(probe, [True] * len(clients))], (
        "the same probe must be cleaned exactly once after its HTTP clients close"
    )


def test_cap003_missing_cloudwatch_measurements_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = harness(tmp_path)
    value.region = "us-west-2"
    value.aurora_cluster_id = "synthetic"
    monkeypatch.setattr(
        base.boto3,
        "client",
        lambda *_a, **_k: SimpleNamespace(
            get_metric_statistics=lambda **_kw: {"Datapoints": []}
        ),
    )
    now = datetime.now(timezone.utc)

    with pytest.raises(base.CapError, match="CloudWatch.*missing"):
        value.cloudwatch_window(now, now)
