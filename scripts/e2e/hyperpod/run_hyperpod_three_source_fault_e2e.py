from __future__ import annotations

import json
import os
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

if __package__:
    from scripts.e2e.isolated_api import (
        launch_isolated_api,
        random_execution_token,
        require_assertions_enabled,
        stop_isolated_api,
        wait_for_isolated_api,
    )
else:
    from isolated_api import (
        launch_isolated_api,
        random_execution_token,
        require_assertions_enabled,
        stop_isolated_api,
        wait_for_isolated_api,
    )
from gpu_fault.collectors import (
    CollectorContext,
    DcgmMetricsCollector,
    FabricManagerLogCollector,
    HttpEventSink,
    KernelLogCollector,
)
from gpu_fault.models import CapabilityName


CLUSTER_ID = os.getenv(
    "GPU_FAULT_CLUSTER_ID",
    "three-source-e2e-cluster",
)
NODE_ID = "three-source-e2e-node"
JOB_ID = "three-source-training"
ATTEMPT_ID = "three-source-training-a001"
WORKLOAD_ID = "gpu-fault-system/pytorchjob/three-source-training"
GPU_UUID = "GPU-THREE-SOURCE-E2E-0"
GPU_BDF = "0000:01:00.0"
API_URL = ""
MOCK_URL = "http://127.0.0.1:9401/metrics"
DB_PATH = Path(
    os.getenv(
        "GPU_FAULT_THREE_SOURCE_E2E_DB",
        "/state/three-source-control-plane.db",
    )
)
REPORT_PATH = Path(
    os.getenv(
        "GPU_FAULT_THREE_SOURCE_E2E_REPORT",
        "/report/three-source-e2e.json",
    )
)
FM_LOG_PATH = DB_PATH.with_name(f"{DB_PATH.stem}-fabricmanager.log")
FM_STATE_PATH = DB_PATH.with_name(f"{DB_PATH.stem}-fabricmanager-state.json")


class MetricsHandler(BaseHTTPRequestHandler):
    value = 0.0
    lock = threading.Lock()

    def do_GET(self) -> None:
        if self.path != "/metrics":
            self.send_error(404)
            return
        with self.lock:
            value = self.value
        body = (
            "# HELP DCGM_FI_DEV_PCIE_REPLAY_COUNTER synthetic counter\n"
            "# TYPE DCGM_FI_DEV_PCIE_REPLAY_COUNTER counter\n"
            "DCGM_FI_DEV_PCIE_REPLAY_COUNTER"
            f'{{gpu="0",UUID="{GPU_UUID}",'
            f'pci_bus_id="{GPU_BDF}"}} {value}\n'
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *_args: Any) -> None:
        return


class RecordingSink(HttpEventSink):
    def __init__(self) -> None:
        super().__init__(API_URL)
        self.responses: list[dict[str, Any]] = []

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = super().post(path, payload)
        self.responses.append({"path": path, "payload": payload, "response": response})
        return response


def request_json(
    path: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
) -> Any:
    body = json.dumps(payload).encode() if payload is not None else None
    request = Request(
        f"{API_URL}{path}",
        data=body,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=15) as response:
        return json.load(response)


def workload_observation() -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    return {
        "cluster_id": CLUSTER_ID,
        "environment": "hyperpod-eks",
        "job_id": JOB_ID,
        "attempt_id": ATTEMPT_ID,
        "workload_phase": "RUNNING",
        "observed_at": now.isoformat(),
        "started_at": now.isoformat(),
        "expected_critical_ranks": 1,
        "containers": [
            {
                "pod_uid": "three-source-trainer-pod-uid",
                "pod_name": "three-source-training-master-0",
                "container_name": "trainer",
                "role": "master",
                "rank": 0,
                "node_id": NODE_ID,
                "gpu_uuids": [GPU_UUID],
            }
        ],
        "workload_ids": [WORKLOAD_ID],
        "runtime_profile_version": "dcgm-three-source-e2e-v1",
        "restart_budget": 3,
    }


def runtime_profile() -> dict[str, Any]:
    capabilities = [item.value for item in CapabilityName]
    return {
        "cluster_id": CLUSTER_ID,
        "environment": "hyperpod-eks",
        "profile_version": "dcgm-three-source-e2e-v1",
        "claims": [
            {
                "capability": capability,
                "mode": "OWN",
                "owner": "three-source-e2e-runtime",
                "adapter": "disabled-e2e-adapter",
            }
            for capability in capabilities
        ],
        "observed": [
            {
                "capability": capability,
                "owner": "three-source-e2e-runtime",
                "available": True,
                "version": "e2e-v1",
            }
            for capability in capabilities
        ],
    }


def collector_context() -> CollectorContext:
    return CollectorContext(
        cluster_id=CLUSTER_ID,
        runtime_profile_version="dcgm-three-source-e2e-v1",
        product="H200",
        driver_branch=570,
        cuda_version="12.8",
        workload_state="ACTIVE",
        affected_workload_ids=[WORKLOAD_ID],
    )


def dcgm_collector(sink: RecordingSink) -> DcgmMetricsCollector:
    collector = DcgmMetricsCollector(
        sink,
        collector_context(),
        node_id=NODE_ID,
        metrics_url=MOCK_URL,
    )
    # The test endpoint validates exporter parsing; the bounded temperature-limit
    # probe runs against the host's nvidia-smi and yields no samples without it.
    return collector


def run_concurrent_collectors() -> dict[str, Any]:
    dcgm_sink = RecordingSink()
    kernel_sink = RecordingSink()
    fabric_sink = RecordingSink()
    dcgm = dcgm_collector(dcgm_sink)
    kernel = KernelLogCollector(
        kernel_sink,
        collector_context(),
        node_id=NODE_ID,
        boot_id="three-source-e2e-boot",
    )
    FM_LOG_PATH.write_text(
        "nvidia-nvswitch3: "
        "SXid (PCI:0000:c1:00.0): 12020, Fatal, "
        "Link 46 egress sequence ID error\n"
    )
    fabric = FabricManagerLogCollector(
        fabric_sink,
        collector_context(),
        node_id=NODE_ID,
        journal_enabled=False,
        log_paths=[str(FM_LOG_PATH)],
        state_path=str(FM_STATE_PATH),
    )
    with MetricsHandler.lock:
        MetricsHandler.value = 100
    kmsg = (
        "6,1001,123456789,-;"
        f"NVRM: Xid (PCI:{GPU_BDF.removesuffix('.0')}): 11, "
        "Ch 00000001, Invalid or corrupted push buffer stream\n"
    )
    barrier = threading.Barrier(3)

    def collect_dcgm() -> Any:
        barrier.wait()
        return dcgm.collect_once().model_dump(mode="json")

    def collect_kernel() -> Any:
        barrier.wait()
        return kernel.collect_lines([kmsg]).model_dump(mode="json")

    def collect_fabric() -> Any:
        barrier.wait()
        return fabric.collect_once().model_dump(mode="json")

    started_at = datetime.now(timezone.utc)
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            "dcgm": executor.submit(collect_dcgm),
            "kernel": executor.submit(collect_kernel),
            "fabric_manager": executor.submit(collect_fabric),
        }
        collector_results = {name: future.result() for name, future in futures.items()}
    finished_at = datetime.now(timezone.utc)
    responses = {
        "dcgm": dcgm_sink.responses[-1],
        "kernel": kernel_sink.responses[-1],
        "fabric_manager": fabric_sink.responses[-1],
    }
    return {
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_ms": (finished_at - started_at).total_seconds() * 1000,
        "collector_results": collector_results,
        "responses": responses,
    }


def load_state() -> dict[str, Any]:
    connection = sqlite3.connect(DB_PATH)
    try:
        objects = connection.execute(
            "SELECT kind, key, payload FROM objects"
        ).fetchall()
        by_kind: dict[str, list[dict[str, Any]]] = {}
        for kind, key, payload in objects:
            by_kind.setdefault(kind, []).append(
                {"key": key, "payload": json.loads(payload)}
            )
        links = [
            {"kind": kind, "key": key, "value": value}
            for kind, key, value in connection.execute(
                "SELECT kind, key, value FROM links ORDER BY kind, key"
            )
        ]
        return {"objects": by_kind, "links": links}
    finally:
        connection.close()


def validate(
    concurrent: dict[str, Any],
    state: dict[str, Any],
) -> dict[str, Any]:
    responses = concurrent["responses"]
    metric = responses["dcgm"]["response"]
    kernel = responses["kernel"]["response"]
    fabric = responses["fabric_manager"]["response"]

    assert responses["dcgm"]["path"] == ("/v1/collector-events/gpu-metrics")
    assert responses["kernel"]["path"] == ("/v1/collector-events/nvidia-kernel")
    assert responses["fabric_manager"]["path"] == (
        "/v1/collector-events/fabric-manager"
    )
    assert metric["new_findings"][0]["canonical_name"] == ("pcie_replay_total")
    assert metric["new_findings"][0]["automatic_action"] == ("RUN_DIAGNOSTICS")
    xid_event = kernel["normalized"]["xid_events"][0]
    sxid_event = fabric["normalized"]["sxid_events"][0]
    assert xid_event["xid"] == 11
    assert xid_event["event_source"] == "KERNEL_LOG"
    assert sxid_event["sxid"] == 12020
    assert sxid_event["event_source"] == "FABRIC_MANAGER_LOG"
    assert kernel["decisions"][0]["action"] == "RESTART_WORKLOAD"
    assert fabric["decisions"][0]["action"] == "REBOOT_NODE"

    incidents = state["objects"].get("incident", [])
    workflows = state["objects"].get("workflow", [])
    assert len(incidents) == 1, incidents
    assert len(workflows) == 1, workflows
    incident = incidents[0]["payload"]
    workflow = workflows[0]["payload"]
    assert incident["job_id"] == JOB_ID
    assert incident["attempt_id"] == ATTEMPT_ID
    assert incident["workload_identity_source"] == ("SOLE_ACTIVE_ATTEMPT_ON_NODE")
    assert kernel["decisions"][0]["incident_id"] == (incident["incident_id"])
    assert fabric["decisions"][0]["incident_id"] == (incident["incident_id"])

    metric_event_id = "gpu-" + metric["new_findings"][0]["finding_id"]
    incident_links = {
        item["key"]: item["value"]
        for item in state["links"]
        if item["kind"] == "incident_by_event"
    }
    assert incident_links[metric_event_id] == incident["incident_id"]
    assert incident_links[xid_event["event_id"]] == incident["incident_id"]
    assert incident_links[sxid_event["event_id"]] == incident["incident_id"]

    operations = [step["operation"] for step in workflow["official_steps"]]
    assert "RESTART_NODE" in operations
    assert operations.count("RESTART_WORKLOAD") == 1
    assert "RESET_ALL_GPUS_NVSWITCHES" not in operations
    restart = next(
        step
        for step in workflow["official_steps"]
        if step["operation"] == "RESTART_WORKLOAD"
    )
    assert restart["parameters"]["job_id"] == JOB_ID
    assert restart["parameters"]["source_attempt_id"] == ATTEMPT_ID
    assert workflow["step_executions"] == []
    assert not state["objects"].get("workflow_execution")

    reasons = " ".join(incident["reasons"])
    assert "PCIe replay rate exceeded" in reasons
    assert "XID 11" in reasons
    assert "SXID 12020" in reasons
    return {
        "shared_incident_id": incident["incident_id"],
        "shared_workflow_id": workflow["request_id"],
        "job_id": incident["job_id"],
        "attempt_id": incident["attempt_id"],
        "workload_identity_source": (incident["workload_identity_source"]),
        "workflow_status": workflow["status"],
        "workflow_not_before": workflow["not_before"],
        "workflow_operations": operations,
        "workflow_step_execution_count": len(workflow["step_executions"]),
        "event_incident_links": {
            "dcgm": incident_links[metric_event_id],
            "xid": incident_links[xid_event["event_id"]],
            "sxid": incident_links[sxid_event["event_id"]],
        },
        "decisions": {
            "dcgm": {
                "metric": metric["new_findings"][0]["canonical_name"],
                "action": metric["new_findings"][0]["automatic_action"],
            },
            "xid": {
                "code": xid_event["xid"],
                "source": xid_event["event_source"],
                "action": kernel["decisions"][0]["action"],
            },
            "sxid": {
                "code": sxid_event["sxid"],
                "source": sxid_event["event_source"],
                "action": fabric["decisions"][0]["action"],
            },
            "arbitrated_action": "RESTART_NODE",
        },
        "incident_reasons": incident["reasons"],
    }


def main() -> int:
    global API_URL, MOCK_URL
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", 0), MetricsHandler)
    MOCK_URL = f"http://127.0.0.1:{server.server_port}/metrics"
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    env = os.environ.copy()
    env.update(
        {
            "GPU_FAULT_EXECUTOR_MODE": "active",
            "GPU_FAULT_STORE_URL": f"sqlite:///{DB_PATH}",
            "GPU_FAULT_EXECUTION_TOKEN": random_execution_token(),
            "GPU_FAULT_ALLOWED_OPERATIONS": (
                "FREEZE_EVIDENCE,COLLECT_DIAGNOSTIC_BUNDLE,"
                "MARK_UNSCHEDULABLE,QUARANTINE,STOP_WORKLOADS,"
                "RESTART_WORKLOAD,QUIESCE_GPU_SERVICES,"
                "VERIFY_NO_GPU_CLIENTS,RESET_ALL_GPUS_NVSWITCHES,"
                "RESTORE_GPU_SERVICES,VALIDATE_GPU,VALIDATE_FABRIC,"
                "RESTORE_SCHEDULING"
            ),
            "GPU_FAULT_ENABLE_WORKFLOW_DISPATCHER": "false",
            "GPU_FAULT_ENABLE_KUBERNETES_ADAPTER": "false",
            "GPU_FAULT_ENABLE_NODE_ACTION_ADAPTER": "false",
            "GPU_FAULT_ENABLE_AGENT_REGISTRY": "false",
            "GPU_FAULT_ENABLE_QUICK_DIAGNOSTICS": "false",
            "GPU_FAULT_ALLOW_EMAIL": "false",
            "GPU_FAULT_MULTI_NODE_AGGREGATION_WINDOW_SECONDS": "5",
        }
    )
    api = launch_isolated_api(env)
    API_URL = api.url
    report: dict[str, Any] = {
        "schema_version": 2,
        "report_type": "fault-e2e",
        "test_case_id": "HYPERPOD-DCGM-XID-SXID-CONCURRENT-E2E",
        "cluster_id": CLUSTER_ID,
        "node_id": NODE_ID,
        "job_id": JOB_ID,
        "attempt_id": ATTEMPT_ID,
        "dispatcher_enabled": False,
        "verdict": "FAIL",
        "status": "FAILED",
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        require_assertions_enabled()
        wait_for_isolated_api(api, expected_executor="active")
        request_json(
            "/v1/runtime-profiles",
            method="POST",
            payload=runtime_profile(),
        )
        request_json(
            "/v1/workload-observations",
            method="POST",
            payload=workload_observation(),
        )
        baseline_sink = RecordingSink()
        baseline = dcgm_collector(baseline_sink)
        with MetricsHandler.lock:
            MetricsHandler.value = 0
        baseline.collect_once()
        assert not baseline_sink.responses[-1]["response"]["new_findings"]

        concurrent = run_concurrent_collectors()
        stop_isolated_api(api)
        state = load_state()
        report["concurrent_collection"] = {
            "started_at": concurrent["started_at"],
            "finished_at": concurrent["finished_at"],
            "duration_ms": concurrent["duration_ms"],
            "collector_results": concurrent["collector_results"],
        }
        report["validation"] = validate(concurrent, state)
        report["object_counts"] = {
            kind: len(items) for kind, items in sorted(state["objects"].items())
        }
        report["verdict"] = "PASS"
        report["status"] = "PASSED"
        return_code = 0
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        return_code = 1
    finally:
        stop_isolated_api(api)
        server.shutdown()
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, indent=2, sort_keys=True))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
