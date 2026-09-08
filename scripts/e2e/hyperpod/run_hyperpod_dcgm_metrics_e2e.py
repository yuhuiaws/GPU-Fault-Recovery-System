from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.request import urlopen

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
    DCGM_METRICS,
    CollectorContext,
    DcgmMetricsCollector,
    HttpEventSink,
)


CLUSTER_ID = os.getenv("GPU_FAULT_CLUSTER_ID", "dcgm-e2e-cluster")
API_URL = ""
MOCK_URL = "http://127.0.0.1:9401/metrics"
REPORT_PATH = Path(os.getenv("GPU_FAULT_DCGM_E2E_REPORT", "/report/dcgm-e2e.json"))
DB_PATH = Path(os.getenv("GPU_FAULT_DCGM_E2E_DB", "/state/control-plane.db"))
GPU0 = {"gpu": "0", "UUID": "GPU-DCGM-E2E-0", "pci_bus_id": "0000:01:00.0"}
GPU1 = {"gpu": "1", "UUID": "GPU-DCGM-E2E-1", "pci_bus_id": "0000:02:00.0"}

DEFAULTS = {
    "DCGM_FI_DEV_GPU_TEMP": 45,
    "DCGM_FI_DEV_MEMORY_TEMP": 50,
    "DCGM_FI_DEV_POWER_USAGE": 300,
    "DCGM_FI_DEV_POWER_MGMT_LIMIT": 700,
    "DCGM_FI_DEV_GPU_UTIL": 60,
    "DCGM_FI_DEV_MEM_COPY_UTIL": 40,
    "DCGM_FI_DEV_FB_USED": 1024,
    "DCGM_FI_DEV_FB_FREE": 140000,
    "DCGM_FI_DEV_SM_CLOCK": 1500,
    "DCGM_FI_DEV_MEM_CLOCK": 2400,
    "DCGM_FI_DEV_CLOCK_THROTTLE_REASONS": 0,
    "DCGM_FI_DEV_XID_ERRORS": 0,
}

COUNTERS = {
    name
    for name in DCGM_METRICS
    if name not in DEFAULTS
    and name
    not in {
        "DCGM_FI_DEV_RETIRED_PENDING",
        "DCGM_FI_DEV_ROW_REMAP_FAILURE",
        "DCGM_FI_DEV_ROW_REMAP_PENDING",
    }
}


class MetricsHandler(BaseHTTPRequestHandler):
    metrics = ""
    lock = threading.Lock()

    def do_GET(self) -> None:
        if self.path != "/metrics":
            self.send_error(404)
            return
        with self.lock:
            body = self.metrics.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *_args: Any) -> None:
        return


class RecordingSink(HttpEventSink):
    def __init__(self, base_url: str) -> None:
        super().__init__(base_url)
        self.last_response: dict[str, Any] = {}

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.last_response = super().post(path, payload)
        return self.last_response


def metric_text(values: dict[str, float], gpu: dict[str, str] = GPU0) -> str:
    labels = ",".join(f'{key}="{value}"' for key, value in gpu.items())
    lines = []
    for name in DCGM_METRICS:
        value = values.get(name, DEFAULTS.get(name, 0))
        metric_type = "counter" if name in COUNTERS else "gauge"
        lines.extend(
            [
                f"# HELP {name} synthetic DCGM E2E metric",
                f"# TYPE {name} {metric_type}",
                f"{name}{{{labels}}} {value}",
            ]
        )
    return "\n".join(lines) + "\n"


def set_metrics(*payloads: tuple[dict[str, float], dict[str, str]]) -> None:
    text = "".join(metric_text(values, gpu) for values, gpu in payloads)
    with MetricsHandler.lock:
        MetricsHandler.metrics = text


def get_json(path: str) -> Any:
    with urlopen(f"{API_URL}{path}", timeout=10) as response:
        return json.load(response)


def collect(
    node_id: str,
    values: dict[str, float],
    *,
    second_gpu: dict[str, float] | None = None,
) -> dict[str, Any]:
    payloads = [(values, GPU0)]
    if second_gpu is not None:
        payloads.append((second_gpu, GPU1))
    set_metrics(*payloads)
    sink = RecordingSink(API_URL)
    collector = DcgmMetricsCollector(
        sink,
        CollectorContext(
            cluster_id=CLUSTER_ID,
            runtime_profile_version="dcgm-e2e-v1",
            product="H200",
            driver_branch=570,
            cuda_version="12.8",
        ),
        node_id=node_id,
        metrics_url=MOCK_URL,
    )
    # The test endpoint validates exporter parsing; device limits require
    # nvidia-smi and are covered by node-side integration separately.
    collector._temperature_limit_samples = []
    batch = collector.collect_once()
    result = sink.last_response
    result["collector_batch_id"] = batch.batch_id
    result["collector_sample_count"] = len(batch.samples)
    return result


def assert_rule(result: dict[str, Any], rule: str, severity: str) -> None:
    matches = [
        item
        for item in result["composite_findings"]
        if item["correlation_rule_id"] == rule
    ]
    assert matches, f"missing composite rule {rule}: {result}"
    assert matches[0]["severity"] == severity, matches[0]


def summarize(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "batch_id": result["batch_id"],
        "accepted_samples": result["accepted_samples"],
        "collector_sample_count": result["collector_sample_count"],
        "findings": [
            {
                "metric": item["canonical_name"],
                "severity": item["severity"],
                "action": item["automatic_action"],
                "delta": item.get("delta"),
            }
            for item in result["findings"]
        ],
        "composites": [
            {
                "rule": item["correlation_rule_id"],
                "severity": item["severity"],
                "action": item["automatic_action"],
                "metrics": item["component_metrics"],
            }
            for item in result["composite_findings"]
        ],
        "xid_decisions": [
            {
                "event_id": item["event_id"],
                "official_action": item["official_action"],
                "effective_action": item["action"],
            }
            for item in result["decisions"]
        ],
    }


def _baseline_and_limit_cases(cases: list[dict[str, Any]]) -> str:
    baseline = collect("dcgm-e2e-all-metrics", {})
    expected = {canonical for canonical, _unit in DCGM_METRICS.values()}
    latest = get_json(f"/v1/gpu-metrics/{CLUSTER_ID}/dcgm-e2e-all-metrics/latest")
    observed = {item["sample"]["canonical_name"] for item in latest}
    assert expected == observed, sorted(expected - observed)
    assert baseline["accepted_samples"] == len(DCGM_METRICS)
    assert not baseline["new_findings"]
    cases.append(
        {
            "id": "DCGM-E2E-001",
            "name": "all supported metrics baseline",
            "status": "PASSED",
            "expected_metric_count": len(expected),
            "observed_metric_count": len(observed),
            "metrics": sorted(observed),
            "result": summarize(baseline),
        }
    )

    thermal_node = "dcgm-e2e-thermal"
    collect(thermal_node, {})
    thermal_values = {
        "DCGM_FI_DEV_GPU_TEMP": 86,
        "DCGM_FI_DEV_MEMORY_TEMP": 91,
        "DCGM_FI_DEV_CLOCK_THROTTLE_REASONS": 0x20,
        "DCGM_FI_DEV_THERMAL_VIOLATION": 1,
        "DCGM_FI_DEV_SM_CLOCK": 900,
        "DCGM_FI_DEV_MEM_CLOCK": 1600,
    }
    first = collect(thermal_node, thermal_values)
    thermal_values["DCGM_FI_DEV_THERMAL_VIOLATION"] = 2
    second = collect(thermal_node, thermal_values)
    assert_rule(first, "THERMAL_STRESS", "WARNING")
    assert_rule(second, "THERMAL_STRESS", "CRITICAL")
    cases.append(
        {
            "id": "DCGM-E2E-002",
            "name": "thermal violation and thermal throttle correlation",
            "status": "PASSED",
            "first": summarize(first),
            "second": summarize(second),
        }
    )

    power_node = "dcgm-e2e-power"
    collect(power_node, {})
    power = collect(
        power_node,
        {
            "DCGM_FI_DEV_POWER_USAGE": 690,
            "DCGM_FI_DEV_POWER_MGMT_LIMIT": 700,
            "DCGM_FI_DEV_GPU_UTIL": 95,
            "DCGM_FI_DEV_POWER_VIOLATION": 100,
        },
    )
    assert_rule(power, "POWER_LIMIT_THROTTLING", "WARNING")
    cases.append(
        {
            "id": "DCGM-E2E-003",
            "name": "power limit throttling correlation",
            "status": "PASSED",
            "result": summarize(power),
        }
    )
    return thermal_node


def _memory_cases(cases: list[dict[str, Any]]) -> None:
    corrected_node = "dcgm-e2e-corrected-memory"
    collect(corrected_node, {})
    corrected_results = []
    for value in (1, 2, 3):
        result = collect(
            corrected_node,
            {
                "DCGM_FI_DEV_ECC_SBE_VOL_TOTAL": value,
                "DCGM_FI_DEV_RETIRED_SBE": value,
                "DCGM_FI_DEV_CORRECTABLE_REMAPPED_ROWS": value,
            },
        )
        corrected_results.append(result)
    assert_rule(corrected_results[0], "CORRECTABLE_MEMORY_DEGRADATION", "WARNING")
    assert_rule(corrected_results[2], "CORRECTABLE_MEMORY_DEGRADATION", "CRITICAL")
    cases.append(
        {
            "id": "DCGM-E2E-004",
            "name": "correctable memory degradation trend",
            "status": "PASSED",
            "samples": [summarize(item) for item in corrected_results],
        }
    )

    retired = "dcgm-e2e-retired-dbe"
    collect(retired, {})
    retired_result = collect(retired, {"DCGM_FI_DEV_RETIRED_DBE": 1})
    assert any(
        item["canonical_name"] == "retired_pages_dbe_total"
        and item["severity"] == "CRITICAL"
        for item in retired_result["findings"]
    )
    cases.append(
        {
            "id": "DCGM-E2E-005",
            "name": "uncorrectable retired page increment",
            "status": "PASSED",
            "result": summarize(retired_result),
        }
    )

    memory_node = "dcgm-e2e-memory-failure"
    collect(memory_node, {})
    memory = collect(
        memory_node,
        {
            "DCGM_FI_DEV_ECC_DBE_VOL_TOTAL": 1,
            "DCGM_FI_DEV_ROW_REMAP_FAILURE": 1,
        },
    )
    assert_rule(memory, "GPU_MEMORY_DEGRADATION", "CRITICAL")
    cases.append(
        {
            "id": "DCGM-E2E-006",
            "name": "uncorrectable ECC and row-remap failure",
            "status": "PASSED",
            "result": summarize(memory),
        }
    )

    repair_node = "dcgm-e2e-memory-repair"
    collect(repair_node, {})
    repair = collect(
        repair_node,
        {
            "DCGM_FI_DEV_RETIRED_PENDING": 1,
            "DCGM_FI_DEV_ROW_REMAP_PENDING": 1,
            "DCGM_FI_DEV_UNCORRECTABLE_REMAPPED_ROWS": 1,
            "DCGM_FI_DEV_ECC_DBE_AGG_TOTAL": 1,
        },
    )
    metrics = {item["canonical_name"] for item in repair["findings"]}
    assert {
        "retired_pages_pending",
        "row_remap_pending",
        "row_remap_uncorrectable_total",
        "ecc_dbe_aggregate_total",
    } <= metrics
    cases.append(
        {
            "id": "DCGM-E2E-007",
            "name": "pending repair and aggregate memory counters",
            "status": "PASSED",
            "result": summarize(repair),
        }
    )


def _link_cases(cases: list[dict[str, Any]]) -> None:
    pcie_node = "dcgm-e2e-pcie-xid"
    collect(pcie_node, {})
    time.sleep(1)
    pcie = collect(
        pcie_node,
        {
            "DCGM_FI_DEV_PCIE_REPLAY_COUNTER": 100,
            "DCGM_FI_DEV_XID_ERRORS": 79,
        },
    )
    assert_rule(pcie, "PCIE_XID_LINK_FAILURE", "CRITICAL")
    assert pcie["decisions"], pcie
    cases.append(
        {
            "id": "DCGM-E2E-008",
            "name": "PCIe replay and XID 79 correlation",
            "status": "PASSED",
            "result": summarize(pcie),
        }
    )

    nvlink_node = "dcgm-e2e-nvlink-single"
    collect(nvlink_node, {})
    nvlink = collect(
        nvlink_node,
        {
            "DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL": 1,
            "DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_TOTAL": 1,
        },
    )
    assert_rule(nvlink, "NVLINK_LINK_DEGRADATION", "CRITICAL")
    cases.append(
        {
            "id": "DCGM-E2E-009",
            "name": "single-GPU multi-class NVLink degradation",
            "status": "PASSED",
            "result": summarize(nvlink),
        }
    )

    fabric_node = "dcgm-e2e-nvlink-fabric"
    collect(fabric_node, {}, second_gpu={})
    fabric = collect(
        fabric_node,
        {"DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_TOTAL": 1},
        second_gpu={"DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_TOTAL": 1},
    )
    assert_rule(fabric, "MULTI_GPU_NVLINK_FABRIC_FAILURE", "CRITICAL")
    cases.append(
        {
            "id": "DCGM-E2E-010",
            "name": "multi-GPU NVLink fabric failure",
            "status": "PASSED",
            "result": summarize(fabric),
        }
    )

    aggregate_node = "dcgm-e2e-nvlink-aggregate"
    collect(aggregate_node, {})
    aggregate = collect(
        aggregate_node,
        {
            "DCGM_FI_DEV_NVLINK_ERROR_DL_CRC": 1,
            "DCGM_FI_DEV_NVLINK_ERROR_DL_RECOVERY": 1,
            "DCGM_FI_DEV_NVLINK_ERROR_DL_REPLAY": 1,
        },
    )
    assert_rule(aggregate, "NVLINK_LINK_DEGRADATION", "CRITICAL")
    cases.append(
        {
            "id": "DCGM-E2E-011",
            "name": "aggregate NVLink counters",
            "status": "PASSED",
            "result": summarize(aggregate),
        }
    )


def _recovery_case(
    cases: list[dict[str, Any]],
    thermal_node: str,
) -> None:
    recovered = collect(thermal_node, {})
    active = get_json(f"/v1/gpu-health-findings/{CLUSTER_ID}/{thermal_node}")
    assert not active, active
    cases.append(
        {
            "id": "DCGM-E2E-012",
            "name": "active finding recovery and clear",
            "status": "PASSED",
            "active_findings_after_recovery": len(active),
            "result": summarize(recovered),
        }
    )


def run_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    thermal_node = _baseline_and_limit_cases(cases)
    _memory_cases(cases)
    _link_cases(cases)
    _recovery_case(cases, thermal_node)
    return cases


def sqlite_counts() -> dict[str, Any]:
    connection = sqlite3.connect(DB_PATH)
    try:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        result: dict[str, Any] = {}
        for table in sorted(tables):
            if table.startswith("sqlite_"):
                continue
            result[table] = connection.execute(
                f'SELECT count(*) FROM "{table}"'
            ).fetchone()[0]
        if "objects" in tables:
            result["object_kinds"] = {
                kind: count
                for kind, count in connection.execute(
                    "SELECT kind, count(*) FROM objects GROUP BY kind ORDER BY kind"
                )
            }
        return result
    finally:
        connection.close()


def main() -> int:
    global API_URL, MOCK_URL
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", 0), MetricsHandler)
    MOCK_URL = f"http://127.0.0.1:{server.server_port}/metrics"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    env = os.environ.copy()
    env.update(
        {
            "GPU_FAULT_EXECUTOR_MODE": "active",
            "GPU_FAULT_STORE_URL": f"sqlite:///{DB_PATH}",
            "GPU_FAULT_EXECUTION_TOKEN": random_execution_token(),
            "GPU_FAULT_ALLOWED_OPERATIONS": (
                "FREEZE_EVIDENCE,MARK_UNSCHEDULABLE,STOP_WORKLOADS,"
                "QUARANTINE,COLLECT_DIAGNOSTIC_BUNDLE,"
                "RUN_DCGM_DIAGNOSTIC,RUN_FIELD_DIAGNOSTIC,"
                "RESET_GPU,VALIDATE_GPU"
            ),
            "GPU_FAULT_ENABLE_WORKFLOW_DISPATCHER": "false",
            "GPU_FAULT_ENABLE_KUBERNETES_ADAPTER": "false",
            "GPU_FAULT_ENABLE_NODE_ACTION_ADAPTER": "false",
            "GPU_FAULT_ENABLE_AGENT_REGISTRY": "false",
            "GPU_FAULT_ALLOW_EMAIL": "false",
        }
    )
    api = launch_isolated_api(env)
    API_URL = api.url
    started_at = datetime.now(timezone.utc)
    report: dict[str, Any] = {
        "schema_version": 2,
        "report_type": "fault-e2e",
        "test_suite": "HyperPod DCGM Metrics Collector E2E",
        "started_at": started_at.isoformat(),
        "cluster_id": CLUSTER_ID,
        "mock_endpoint": MOCK_URL,
        "control_plane": API_URL,
        "dispatcher_enabled": False,
        "supported_exporter_metric_count": len(DCGM_METRICS),
        "verdict": "FAIL",
        "status": "FAILED",
    }
    try:
        require_assertions_enabled()
        wait_for_isolated_api(api, expected_executor="active")
        report["test_cases"] = run_cases()
        report["verdict"] = "PASS"
        report["status"] = "PASSED"
        return_code = 0
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        return_code = 1
    finally:
        stop_isolated_api(api)
        server.shutdown()
        if DB_PATH.exists():
            report["sqlite_table_counts"] = sqlite_counts()
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, indent=2, sort_keys=True))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
