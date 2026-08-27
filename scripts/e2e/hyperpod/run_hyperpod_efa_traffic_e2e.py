from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
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
    HostTelemetryCollector,
    HttpEventSink,
)
from gpu_fault.host_health import HostTelemetryBatch
from gpu_fault.models import CapabilityName
from gpu_fault.store import SqliteStore


CLUSTER_ID = os.getenv("GPU_FAULT_CLUSTER_ID", "hyperpod-efa-traffic-e2e")
NODE_ID = "efa-traffic-e2e-node"
JOB_ID = "efa-traffic-training"
ATTEMPT_ID = "efa-traffic-training-a001"
WORKLOAD_ID = "gpu-fault-system/pytorchjob/efa-traffic-training"
PROFILE = "efa-traffic-e2e-v1"
API_URL = ""
DB_PATH = Path(
    os.getenv(
        "GPU_FAULT_EFA_E2E_DB",
        "/state/efa-traffic-control-plane.db",
    )
)
REPORT_PATH = Path(
    os.getenv(
        "GPU_FAULT_EFA_E2E_REPORT",
        "/report/efa-traffic-e2e.json",
    )
)


class RecordingSink(HttpEventSink):
    def __init__(self) -> None:
        super().__init__(API_URL)
        self.last_response: dict[str, Any] = {}

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.last_response = super().post(path, payload)
        return self.last_response


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


def runtime_profile() -> dict[str, Any]:
    capabilities = [item.value for item in CapabilityName]
    return {
        "cluster_id": CLUSTER_ID,
        "environment": "hyperpod-eks",
        "profile_version": PROFILE,
        "claims": [
            {
                "capability": capability,
                "mode": "OWN",
                "owner": "efa-traffic-e2e",
                "adapter": "disabled-e2e-adapter",
            }
            for capability in capabilities
        ],
        "observed": [
            {
                "capability": capability,
                "owner": "efa-traffic-e2e",
                "available": True,
                "version": "e2e-v1",
            }
            for capability in capabilities
        ],
    }


def observe_attempt(observed_at: datetime) -> None:
    request_json(
        "/v1/workload-observations",
        method="POST",
        payload={
            "cluster_id": CLUSTER_ID,
            "environment": "hyperpod-eks",
            "job_id": JOB_ID,
            "attempt_id": ATTEMPT_ID,
            "workload_phase": "RUNNING",
            "observed_at": observed_at.isoformat(),
            "started_at": observed_at.isoformat(),
            "expected_critical_ranks": 1,
            "containers": [
                {
                    "pod_uid": "efa-traffic-worker-0",
                    "pod_name": "efa-traffic-worker-0",
                    "container_name": "trainer",
                    "role": "worker",
                    "rank": 0,
                    "node_id": NODE_ID,
                    "gpu_uuids": ["GPU-EFA-E2E-0"],
                }
            ],
            "workload_ids": [WORKLOAD_ID],
            "runtime_profile_version": PROFILE,
            "restart_budget": 1,
        },
    )


def main() -> int:
    global API_URL
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "GPU_FAULT_EXECUTOR_MODE": "active",
            "GPU_FAULT_STORE_URL": f"sqlite:///{DB_PATH}",
            "GPU_FAULT_EXECUTION_TOKEN": random_execution_token(),
            "GPU_FAULT_ALLOWED_OPERATIONS": (
                "FREEZE_EVIDENCE,COLLECT_DIAGNOSTIC_BUNDLE,VALIDATE_FABRIC"
            ),
            "GPU_FAULT_ENABLE_WORKFLOW_DISPATCHER": "false",
            "GPU_FAULT_ENABLE_KUBERNETES_ADAPTER": "false",
            "GPU_FAULT_ENABLE_NODE_ACTION_ADAPTER": "false",
            "GPU_FAULT_ENABLE_AGENT_REGISTRY": "false",
            "GPU_FAULT_ALLOW_EMAIL": "false",
            "GPU_FAULT_EFA_TRAFFIC_MIN_ACTIVE_BPS": "100",
            "GPU_FAULT_EFA_TRAFFIC_SPIKE_RATIO": "2",
            "GPU_FAULT_EFA_TRAFFIC_DROP_RATIO": "0.5",
            "GPU_FAULT_EFA_TRAFFIC_ZERO_BPS": "1",
            "GPU_FAULT_EFA_TRAFFIC_ZERO_WARNING_SECONDS": "15",
            "GPU_FAULT_EFA_TRAFFIC_ZERO_HUNG_SECONDS": "30",
            "GPU_FAULT_EFA_TRAFFIC_STARTUP_GRACE_SECONDS": "0",
        }
    )
    api = launch_isolated_api(env)
    API_URL = api.url
    report: dict[str, Any] = {
        "schema_version": 2,
        "report_type": "fault-e2e",
        "test_case_id": "HYPERPOD-EFA-TRAFFIC-HUNG-E2E",
        "cluster_id": CLUSTER_ID,
        "dispatcher_enabled": False,
        "verdict": "FAIL",
        "status": "FAILED",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "samples": [],
    }
    try:
        require_assertions_enabled()
        wait_for_isolated_api(api, expected_executor="active")
        request_json(
            "/v1/runtime-profiles",
            method="POST",
            payload=runtime_profile(),
        )
        with tempfile.TemporaryDirectory() as root:
            counters = Path(root) / "rdmap0" / "ports" / "1" / "hw_counters"
            counters.mkdir(parents=True)
            (counters / "rx_bytes").write_text("0")
            (counters / "tx_bytes").write_text("0")
            sink = RecordingSink()
            collector = HostTelemetryCollector(
                sink,
                CollectorContext(
                    cluster_id=CLUSTER_ID,
                    runtime_profile_version=PROFILE,
                    workload_state="ACTIVE",
                    affected_workload_ids=[WORKLOAD_ID],
                ),
                node_id=NODE_ID,
                infiniband_root=root,
            )
            started = datetime.now(timezone.utc)
            rx = tx = 0
            collector.collect_rdma_samples(started)
            sequence = [
                ("baseline", 1_000),
                ("spike", 3_000),
                ("normal-after-spike", 1_000),
                ("drop", 100),
                ("normal-after-drop", 1_000),
                ("zero-pending", 0),
                ("zero-warning", 0),
                ("hung-suspected", 0),
                ("recovered", 1_000),
            ]
            for index, (name, bytes_per_second) in enumerate(sequence, start=1):
                observed_at = started + timedelta(seconds=index * 15)
                increment = int(bytes_per_second * 15 / 2)
                rx += increment
                tx += increment
                (counters / "rx_bytes").write_text(str(rx))
                (counters / "tx_bytes").write_text(str(tx))
                observe_attempt(observed_at)
                samples = collector.collect_rdma_samples(observed_at)
                batch = HostTelemetryBatch(
                    batch_id=f"efa-e2e-{index}-{name}",
                    cluster_id=CLUSTER_ID,
                    node_id=NODE_ID,
                    observed_at=observed_at,
                    samples=samples,
                    runtime_profile_version=PROFILE,
                    workload_state="ACTIVE",
                    affected_workload_ids=[WORKLOAD_ID],
                    evidence_ref=f"efa-e2e://{name}",
                )
                sink.post(
                    "/v1/collector-events/host-telemetry",
                    batch.model_dump(mode="json"),
                )
                report["samples"].append(
                    {
                        "name": name,
                        "bytes_per_second": bytes_per_second,
                        "findings": sink.last_response["findings"],
                    }
                )

        by_name = {item["name"]: item["findings"] for item in report["samples"]}
        assert "increased abruptly" in by_name["spike"][0]["reason"]
        assert "dropped abruptly" in by_name["drop"][0]["reason"]
        assert by_name["zero-warning"][0]["severity"] == "warning"
        hung = by_name["hung-suspected"][0]
        assert hung["severity"] == "critical"
        assert hung["diagnostic_parameters"]["capture_process_state"]
        assert not by_name["recovered"]

        store = SqliteStore(str(DB_PATH))
        workflows = [item.model_dump(mode="json") for item in store.list_workflows()]
        hung_workflow = next(
            item
            for item in workflows
            if item["incident_id"] == f"inc-{hung['event_id']}"
        )
        operations = [step["operation"] for step in hung_workflow["official_steps"]]
        assert operations == [
            "FREEZE_EVIDENCE",
            "COLLECT_DIAGNOSTIC_BUNDLE",
            "VALIDATE_FABRIC",
        ]
        assert (
            hung_workflow["official_steps"][1]["parameters"]["diagnostic_reason"]
            == "EFA_TRAFFIC_HUNG_SUSPECTED"
        )
        report["hung_workflow"] = hung_workflow
        state_key = store.efa_traffic_state_key(
            CLUSTER_ID,
            NODE_ID,
            JOB_ID,
            ATTEMPT_ID,
        )
        report["efa_traffic_state"] = [
            store.get_efa_traffic_state(state_key).model_dump(mode="json")
        ]
        report["verdict"] = "PASS"
        report["status"] = "PASSED"
        return 0
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        return 1
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        rendered_report = json.dumps(report, indent=2, sort_keys=True)
        REPORT_PATH.write_text(rendered_report)
        stop_isolated_api(api)
        print("EFA_E2E_REPORT_BEGIN")
        print(rendered_report)
        print("EFA_E2E_REPORT_END")


if __name__ == "__main__":
    raise SystemExit(main())
