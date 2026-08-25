from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import gpu_fault.collectors as collectors
from gpu_fault.collectors import (
    CollectorError,
    CollectorContext,
    DcgmMetricsCollector,
    FabricManagerLogCollector,
    HostTelemetryCollector,
    HttpEventSink,
)
from gpu_fault.models import WorkloadState
from gpu_fault.host_health import (
    NodeHealthPolicy,
    NodeLogBatch,
    NodeLogEntry,
)
from gpu_fault.store import InMemoryStore


class Sink:
    def __init__(self) -> None:
        self.requests = []

    def post(self, _path, _payload):
        self.requests.append((_path, _payload))
        return {}


def pod_probe() -> None:
    requests: list[dict[str, str]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append(dict(self.headers.items()))
            if len(requests) == 1:
                self.send_response(429)
                self.send_header("Retry-After", "2")
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"accepted":true}')

        def log_message(self, *_args):
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    started = time.monotonic()
    try:
        result = HttpEventSink(
            f"http://127.0.0.1:{server.server_port}",
            jitter=lambda _low, _high: 0,
        ).post("/probe", {"cluster_id": "audit-collector"})
    finally:
        server.shutdown()
        thread.join(timeout=5)
    elapsed = time.monotonic() - started
    assert result == {"accepted": True}
    assert elapsed >= 2
    assert requests[0]["User-Agent"] == ("gpu-fault-collector/0.10.0")

    policy = NodeHealthPolicy(InMemoryStore())
    findings = policy.evaluate_logs(
        NodeLogBatch(
            batch_id="audit-log-priority",
            cluster_id="audit-cluster",
            node_id="audit-node",
            collected_at=datetime.now(timezone.utc),
            entries=[
                NodeLogEntry(
                    entry_id="audit-entry",
                    source="audit",
                    observed_at=datetime.now(timezone.utc),
                    message=(
                        "out of memory while EFA RDMA link down reported fatal error"
                    ),
                )
            ],
        )
    )
    assert len(findings) == 1
    assert findings[0].category.value == "RDMA"
    assert findings[0].severity.value == "critical"

    original_which = collectors.shutil.which
    collectors.shutil.which = lambda command: (
        f"/usr/bin/{command}" if command in {"ipmitool", "smartctl"} else None
    )
    try:

        def runner(argv, **_kwargs):
            if argv == ["ipmitool", "sensor"]:
                return subprocess.CompletedProcess(
                    argv,
                    0,
                    stdout="Temp | 32 | nc\nFan | 0 | cr\n",
                    stderr="",
                )
            if argv == ["smartctl", "--scan-open"]:
                return subprocess.CompletedProcess(
                    argv,
                    0,
                    stdout="\n/dev/nvme0 -d nvme\n\n",
                    stderr="",
                )
            return subprocess.CompletedProcess(
                argv,
                0,
                stdout=json.dumps({"smart_status": {"passed": True}}),
                stderr="",
            )

        collector = HostTelemetryCollector(
            Sink(),
            CollectorContext(cluster_id="audit-cluster"),
            node_id="audit-node",
            runner=runner,
        )
        assert collector._bmc(datetime.now(timezone.utc))[0].value == 1
        smart = collector._smart(datetime.now(timezone.utc))
        assert len(smart) == 1 and smart[0].value == 0
    finally:
        collectors.shutil.which = original_which

    print(
        "PASS",
        {
            "retry_elapsed_seconds": round(elapsed, 3),
            "user_agent": requests[0]["User-Agent"],
            "log_category": findings[0].category.value,
            "log_severity": findings[0].severity.value,
            "bmc_critical_count": 1,
            "smart_blank_line_safe": True,
        },
    )


def host_rdma_probe(root: str) -> None:
    collector = HostTelemetryCollector(
        Sink(),
        CollectorContext(cluster_id="audit-cluster"),
        node_id="audit-node",
        infiniband_root=root,
    )
    samples = collector._rdma(datetime.now(timezone.utc))
    devices = sorted({item.device.split("/", 1)[0] for item in samples if item.device})
    assert devices
    for device in devices:
        assert collector._is_efa_device(collectors.Path(root) / device)

    with tempfile.TemporaryDirectory() as directory:
        workdir = Path(directory)
        log = workdir / "fabricmanager.log"
        state = workdir / "state.json"
        messages = [
            "[2026-08-13T12:34:56Z] nvidia-nvswitch0: "
            "SXid (PCI:0000:ab:00.0): 22013, Non-fatal, "
            "Link 12 first",
            "[2026-08-13T12:35:56Z] nvidia-nvswitch0: "
            "SXid (PCI:0000:ab:00.0): 22014, Non-fatal, "
            "Link 13 second",
        ]
        log.write_text("\n".join(messages) + "\n")

        class FailSecond:
            def __init__(self):
                self.calls = 0

            def post(self, *_args, **_kwargs):
                self.calls += 1
                if self.calls == 2:
                    raise CollectorError("second delivery failed")
                return {}

        fm = FabricManagerLogCollector(
            FailSecond(),
            CollectorContext(cluster_id="audit-cluster"),
            node_id="audit-node",
            journal_enabled=False,
            log_paths=[str(log)],
            state_path=str(state),
        )
        try:
            fm.collect_once()
        except CollectorError:
            pass
        else:
            raise AssertionError("second FM delivery must fail")
        first_offset = json.loads(state.read_text())["files"][str(log)]["offset"]
        assert first_offset == len(messages[0]) + 1
        retry_sink = Sink()
        retry = FabricManagerLogCollector(
            retry_sink,
            CollectorContext(cluster_id="audit-cluster"),
            node_id="audit-node",
            journal_enabled=False,
            log_paths=[str(log)],
            state_path=str(state),
        )
        assert retry.collect_once().delivered == 1
        assert retry_sink.requests[0][1]["message"] == messages[1]
        assert retry_sink.requests[0][1]["observed_at"] == ("2026-08-13T12:35:56+00:00")
        assert fm._is_fabric_manager("nvidia-fabricmanager@0.service", "")

    dcgm_sink = Sink()
    dcgm = DcgmMetricsCollector(
        dcgm_sink,
        CollectorContext(cluster_id="audit-cluster"),
        node_id="audit-node",
        edge_confirmation_samples=2,
    )
    dcgm.collect_text('DCGM_FI_DEV_GPU_TEMP{gpu="0",UUID="GPU-a"} 70\n')
    dcgm.collect_text('DCGM_FI_DEV_GPU_TEMP{gpu="0",UUID="GPU-a"} 91\n')
    dcgm.collect_text('DCGM_FI_DEV_GPU_TEMP{gpu="0",UUID="GPU-a"} 70\n')
    assert len(dcgm_sink.requests) == 1

    host_sink = Sink()
    times = iter(
        [
            datetime(2026, 8, 13, 12, 0, tzinfo=timezone.utc),
            datetime(2026, 8, 13, 12, 0, 15, tzinfo=timezone.utc),
            datetime(2026, 8, 13, 12, 0, 30, tzinfo=timezone.utc),
            datetime(2026, 8, 13, 12, 5, 15, tzinfo=timezone.utc),
        ]
    )
    values = iter([20.0, 99.0, 99.0, 99.0])
    host = HostTelemetryCollector(
        host_sink,
        CollectorContext(
            cluster_id="audit-cluster",
            workload_state=WorkloadState.ACTIVE,
        ),
        node_id="audit-node",
        now=lambda: next(times),
        health_summary_seconds=300,
    )
    for name in (
        "_cpu",
        "_memory",
        "_filesystems",
        "_shared_filesystems",
        "_lustre",
        "_diskstats",
        "_network",
        "_tcp",
        "_gpu_utilization",
        "_gpu_inventory",
        "_efa_inventory",
        "_rdma",
        "_efa_network",
        "_nvswitch_topology",
        "_smart",
        "_bmc",
    ):
        setattr(host, name, lambda _observed_at: [])
    host._cpu = lambda _observed_at: [
        host._sample("cpu_usage_percent", next(values), "percent")
    ]
    for _ in range(4):
        host.collect_once()
    assert [payload["edge_filter_reasons"] for _, payload in host_sink.requests] == [
        ["baseline"],
        ["threshold:cpu_usage_percent"],
        ["health-summary"],
    ]

    print(
        "PASS",
        {
            "rdma_devices": devices,
            "sample_count": len(samples),
            "all_devices_are_efa": True,
            "fm_retry_delivered": 1,
            "fm_timestamp_preserved": True,
            "dcgm_transient_deliveries": len(dcgm_sink.requests),
            "host_edge_deliveries": len(host_sink.requests),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("pod", "host-rdma"))
    parser.add_argument(
        "--infiniband-root",
        default="/sys/class/infiniband",
    )
    args = parser.parse_args()
    if args.mode == "pod":
        pod_probe()
    else:
        host_rdma_probe(args.infiniband_root)


if __name__ == "__main__":
    main()
