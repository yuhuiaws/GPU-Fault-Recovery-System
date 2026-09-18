"""Exercise legacy harness inputs and verdicts against real in-process runtime."""

from __future__ import annotations

import io
from copy import deepcopy
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from anyio.from_thread import start_blocking_portal

from gpu_fault.app import create_app
from gpu_fault.channel_registry import (
    COLLECTOR_HEALTH_PATH,
    FABRIC_MANAGER_PATH,
    GPU_METRICS_PATH,
)
from gpu_fault.collectors.sinks import HttpEventSink
from gpu_fault.host_health import NodeHealthFinding
from gpu_fault.models import IncidentState
from gpu_fault.store import InMemoryStore, SqliteStore
from scripts.e2e.hyperpod import run_hyperpod_dcgm_metrics_e2e as dcgm
from scripts.e2e.hyperpod import run_hyperpod_efa_traffic_e2e as efa
from scripts.e2e.hyperpod import run_hyperpod_three_source_fault_e2e as three
from tests._builders import build_context


@pytest.fixture
def api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request):
    database = tmp_path / "control.db"
    memory = getattr(request, "param", "sqlite") == "memory"
    store = InMemoryStore() if memory else SqliteStore(str(database))
    context = build_context(store=store)
    # One event loop, like the deployed API, without starting background services.
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app(context)),
        base_url="http://isolated.test",
    )

    def request_json(path, *, method="GET", payload=None):
        response = portal.call(partial(client.request, method, path, json=payload))
        response.raise_for_status()
        return response.json()

    monkeypatch.setattr(
        HttpEventSink,
        "post",
        lambda _sink, path, payload: request_json(path, method="POST", payload=payload),
    )
    for module in (dcgm, three, efa):
        monkeypatch.setattr(module, "API_URL", "http://isolated.test")
        monkeypatch.setattr(module, "DB_PATH", database)
    monkeypatch.setattr(three, "request_json", request_json)
    monkeypatch.setattr(efa, "request_json", request_json)
    monkeypatch.setattr(dcgm, "get_json", request_json)
    monkeypatch.setattr(
        dcgm,
        "urlopen",
        lambda *_args, **_kwargs: io.BytesIO(dcgm.MetricsHandler.metrics.encode()),
    )
    monkeypatch.setattr(
        three,
        "urlopen",
        lambda *_args, **_kwargs: io.BytesIO(
            three.metric_text(three.MetricsHandler.value).encode()
        ),
    )
    monkeypatch.setattr(three, "FM_LOG_PATH", tmp_path / "private-fm.log")
    monkeypatch.setattr(three, "FM_STATE_PATH", tmp_path / "private-fm-state.json")
    with start_blocking_portal() as portal:
        try:
            yield SimpleNamespace(
                context=context, client=client, request=request_json, memory=memory
            )
        finally:
            portal.call(client.aclose)
            if not memory:
                store.close()


def test_dcgm_harness_all_policy_and_persistent_edge_cases_use_current_runtime(
    api,
) -> None:
    cases = dcgm.run_cases()
    assert [case["id"] for case in cases] == [
        f"DCGM-E2E-{index:03d}" for index in range(1, 14)
    ]
    assert all(case["status"] == "PASSED" for case in cases), (
        "every DCGM harness case must pass against the current in-process runtime"
    )
    power = cases[2]["result"]
    assert any(
        item["rule"] == "POWER_LIMIT_THROTTLING" and item["severity"] == "WARNING"
        for item in power["composites"]
    ), "the nanosecond input must exceed the microsecond policy boundary"
    duration = next(
        item
        for item in api.context.gpu_metrics.latest(dcgm.CLUSTER_ID, "dcgm-e2e-power")
        if item.sample.canonical_name == "power_violation_total_us"
    )
    assert duration.sample.value == 1_000_000
    assert cases[-1]["edge_confirmation_samples"] == 3
    assert [
        transition["findings"][0]["action"] for transition in cases[-1]["transitions"]
    ] == ["RUN_DIAGNOSTICS", "DRAIN"]
    assert all(
        workflow.step_executions == []
        for workflow in api.context.store.list_workflows()
    ), "policy harness cases must not execute workflow actions"


def test_dcgm_harness_cannot_reuse_a_previous_response_for_a_suppressed_sample(
    api,
) -> None:
    series = dcgm.MetricSeries("dcgm-persistent-negative")
    baseline = series.collect({})
    assert baseline is not None
    assert series.collect({"DCGM_FI_DEV_GPU_TEMP": 86}) is None
    assert series.collect({"DCGM_FI_DEV_GPU_TEMP": 86}) is None
    confirmed = series.collect({"DCGM_FI_DEV_GPU_TEMP": 86})
    assert confirmed is not None
    assert confirmed["batch_id"] != baseline["batch_id"]


def test_dcgm_harness_rejects_a_receipt_from_another_batch(api, monkeypatch) -> None:
    monkeypatch.setattr(
        HttpEventSink, "post", lambda *_args: {"batch_id": "another-batch"}
    )
    with pytest.raises(AssertionError, match="different sample"):
        dcgm.MetricSeries("dcgm-misbound-receipt").collect({})


@pytest.mark.parametrize("api", ["sqlite", "memory"], indirect=True)
def test_three_source_harness_has_full_baseline_and_appends_after_fm_eof(api) -> None:
    api.request("/v1/runtime-profiles", method="POST", payload=three.runtime_profile())
    api.request(
        "/v1/workload-observations", method="POST", payload=three.workload_observation()
    )
    sink = three.RecordingSink()
    collector = three.dcgm_collector(sink)
    three.MetricsHandler.value = 0
    baseline = three.scrape(collector)
    assert baseline.collection_errors == []
    assert sink.single_response(GPU_METRICS_PATH)["response"]["new_findings"] == []
    concurrent = three.run_concurrent_collectors(collector, sink)
    if api.memory:
        responses = concurrent["responses"]
        events = [
            "gpu-" + responses["dcgm"]["response"]["new_findings"][0]["finding_id"],
            responses["kernel"]["response"]["normalized"]["xid_events"][0]["event_id"],
            responses["fabric_manager"]["response"]["normalized"]["sxid_events"][0][
                "event_id"
            ],
        ]
        store = api.context.store
        state = {
            "objects": {
                "incident": [
                    {"payload": item.model_dump(mode="json")}
                    for item in store.list_incidents_by_state(
                        three.CLUSTER_ID, list(IncidentState)
                    )
                ],
                "workflow": [
                    {"payload": item.model_dump(mode="json")}
                    for item in store.list_workflows()
                ],
            },
            "links": [
                {
                    "kind": "incident_by_event",
                    "key": event_id,
                    "value": store.get_incident_by_event(event_id).incident_id,
                }
                for event_id in events
            ],
        }
    else:
        state = three.load_state()
    report = three.validate(concurrent, state)
    assert concurrent["collector_results"]["fabric_manager"]["delivered"] == 1
    assert report["workflow_step_execution_count"] == 0
    assert (
        report["event_incident_links"]["dcgm"] == report["event_incident_links"]["sxid"]
    )
    assert "RESTART_NODE" in report["workflow_operations"]


def test_three_source_response_selection_ignores_later_health_summaries(
    monkeypatch,
) -> None:
    monkeypatch.setattr(HttpEventSink, "post", lambda _sink, _path, payload: payload)
    sink = three.RecordingSink()
    sink.post(FABRIC_MANAGER_PATH, {"record_id": "private-record"})
    sink.post(COLLECTOR_HEALTH_PATH, {"summary_id": "private-summary"})
    assert sink.single_response(FABRIC_MANAGER_PATH)["response"] == {
        "record_id": "private-record"
    }
    sink.post(FABRIC_MANAGER_PATH, {"record_id": "unexpected-second-record"})
    with pytest.raises(AssertionError, match="exactly one"):
        sink.single_response(FABRIC_MANAGER_PATH)


def test_three_source_refuses_existing_fm_fixture_without_overwriting_it(
    api, monkeypatch
) -> None:
    three.FM_LOG_PATH.write_text("preexisting private data")
    collector = three.dcgm_collector(three.RecordingSink())
    with pytest.raises(RuntimeError, match="fresh private FM"):
        three.run_concurrent_collectors(collector, three.RecordingSink())
    assert three.FM_LOG_PATH.read_text() == "preexisting private data"


def hung_workflow(api) -> dict:
    api.request("/v1/runtime-profiles", method="POST", payload=efa.runtime_profile())
    finding = NodeHealthFinding(
        finding_id="private-hung-finding",
        event_id="private-hung-event",
        cluster_id=efa.CLUSTER_ID,
        node_id=efa.NODE_ID,
        observed_at=datetime.now(timezone.utc),
        category="RDMA",
        severity="critical",
        reason="private active-workload zero traffic",
        metric_name="efa_traffic_bytes_per_second",
        recommended_action="RUN_DIAGNOSTICS",
        runtime_profile_version=efa.PROFILE,
        workload_state="ACTIVE",
        job_id=efa.JOB_ID,
        attempt_id=efa.ATTEMPT_ID,
        affected_workload_ids=[efa.WORKLOAD_ID],
        diagnostic_parameters={
            "diagnostic_reason": "EFA_TRAFFIC_HUNG_SUSPECTED",
            "capture_process_state": True,
            "attempt_node_ids": [efa.NODE_ID],
        },
    )
    _, workflow = api.context.orchestrator.ingest_node_health(finding)
    return workflow.model_dump(mode="json")


def test_efa_harness_accepts_actual_active_hung_triage_dag(api) -> None:
    workflow = hung_workflow(api)
    efa.validate_hung_workflow(workflow)
    assert workflow["official_steps"][1]["operation"] == "COLLECT_HUNG_TRIAGE"
    assert workflow["step_executions"] == []


@pytest.mark.parametrize(
    "damage",
    ["old-sequence", "dependency", "blocked", "foreign-node", "recapture", "executed"],
)
def test_efa_harness_rejects_missing_triage_or_unsafe_dag_evidence(api, damage) -> None:
    workflow = deepcopy(hung_workflow(api))
    if damage == "old-sequence":
        workflow["official_steps"].pop(1)
    elif damage == "dependency":
        workflow["official_steps"][2]["depends_on_step_indexes"] = [0]
    elif damage == "blocked":
        workflow["blocked_reasons"] = ["required capability not owned"]
    elif damage == "foreign-node":
        workflow["official_steps"][1]["node_ids"] = ["foreign-node"]
    elif damage == "recapture":
        workflow["official_steps"][2]["parameters"]["capture_process_state"] = True
    else:
        workflow["step_executions"] = [{"operation": "COLLECT_HUNG_TRIAGE"}]
    with pytest.raises(AssertionError):
        efa.validate_hung_workflow(workflow)


def test_hyperpod_harnesses_do_not_query_host_gpus_for_synthetic_metrics(
    api, monkeypatch
) -> None:
    from gpu_fault.collectors.gpu.dcgm import DcgmMetricsCollector

    forbidden = Mock(side_effect=AssertionError("host collector I/O is not a fixture"))
    monkeypatch.setattr(DcgmMetricsCollector, "collect_once", forbidden)
    assert dcgm.MetricSeries("dcgm-no-host-query").collect({}) is not None
    three.MetricsHandler.value = 0
    three.scrape(three.dcgm_collector(three.RecordingSink()))
    forbidden.assert_not_called()


def test_efa_harness_cleanup_precedes_report_io_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = SimpleNamespace(url="http://isolated.invalid")
    stopped = Mock()
    operations = set()
    report = tmp_path / "efa-report.json"
    monkeypatch.setattr(efa, "DB_PATH", tmp_path / "efa-control.db")
    monkeypatch.setattr(efa, "REPORT_PATH", report)

    def launch(environment):
        operations.update(environment["GPU_FAULT_ALLOWED_OPERATIONS"].split(","))
        return api

    def fail_report(path, *_args, **_kwargs):
        assert path == report
        stopped.assert_called_once_with(api)
        raise OSError("private report write refused")

    monkeypatch.setattr(efa, "launch_isolated_api", launch)
    monkeypatch.setattr(
        efa,
        "wait_for_isolated_api",
        Mock(side_effect=RuntimeError("private preflight failed")),
    )
    monkeypatch.setattr(efa, "stop_isolated_api", stopped)
    monkeypatch.setattr(Path, "write_text", fail_report)
    with pytest.raises(OSError, match="private report write refused"):
        efa.main()
    stopped.assert_called_once_with(api)
    assert operations == {
        "FREEZE_EVIDENCE",
        "COLLECT_HUNG_TRIAGE",
        "COLLECT_DIAGNOSTIC_BUNDLE",
        "VALIDATE_FABRIC",
    }
