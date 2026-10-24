"""Real host collector/API/executor; only OS and Kubernetes I/O are fixtures."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from subprocess import CompletedProcess
from types import SimpleNamespace

from gpu_fault.adapters import ControlPlaneEvidenceAdapter
from gpu_fault.adapters.gpu_validation import GpuValidationAdapter
from gpu_fault.adapters.kubernetes.adapter import KubernetesWorkflowAdapter
from gpu_fault.app import ApplicationContext, default_simulated_profile
from gpu_fault.collectors import CollectorContext, HostTelemetryCollector
from gpu_fault.collectors.host import system_metrics
from gpu_fault.execution import ProductionExecutorConfig
from gpu_fault.gpu_metrics import GpuMetricsService
from gpu_fault.host_health import HostMetricSample
from gpu_fault.models import WorkflowExecutionRequest
from gpu_fault.models import WorkflowOperation as Op
from gpu_fault.store import SqliteStore
from tests.execution._support import FakeCoreApi, UnusedApi

OPERATIONS = frozenset(
    {
        Op.FREEZE_EVIDENCE,
        Op.VALIDATE_HOST,
        Op.VALIDATE_FABRIC,
        Op.MARK_UNSCHEDULABLE,
        Op.QUARANTINE,
    }
)
CONTRIBUTORS = {
    "cpu": "_cpu",
    "memory": "_memory",
    "swap": "_memory",
    "filesystem": "_filesystems",
    "inode": "_filesystems",
    "smart": "_smart",
    "smart-query": "_smart",
    "smart-scan": "_smart",
    "gpu-unknown": "_gpu_inventory",
    "bmc": "_bmc",
    "bmc-query": "_bmc",
    "net-missing": "_network",
    "net-down": "_network",
    "net-unknown": "_network",
    "page-cache": "_memory",
    "shared": "_shared_filesystems",
    "disk-io": "_diskstats",
    "net-errors": "_network",
    "rdma-errors": "_rdma",
}


class HostFaultRig:
    def __init__(self, directory, monkeypatch, kind):
        self.directory = directory
        self.kind = kind
        self.evidence = []
        self.broken = False
        self.clock = datetime.now(timezone.utc) - timedelta(
            seconds=180 if kind in {"cpu", "memory"} else 60
        )
        self.proc = directory / "proc"
        self.proc.mkdir()
        self.net = directory / "net"
        self.net.mkdir()
        (self.proc / "loadavg").write_text("0.1 0.1 0.1 1/10 123\n")
        (self.proc / "mountinfo").write_text("1 0 8:1 / / rw - ext4 /dev/test rw\n")
        self.set_fault(False)
        if kind == "shared":
            (self.proc / "mountinfo").write_text(
                "1 0 8:1 / /shared rw - nfs4 server:/test rw\n"
            )

        def fixture_path(value):
            paths = {
                "/proc/stat": self.proc / "stat",
                "/proc/meminfo": self.proc / "meminfo",
                "/proc/loadavg": self.proc / "loadavg",
                "/proc/self/mountinfo": self.proc / "mountinfo",
                "/proc/diskstats": self.proc / "diskstats",
            }
            return paths.get(str(value), Path(value))

        monkeypatch.setattr(system_metrics, "Path", fixture_path)
        monkeypatch.setattr(system_metrics.shutil, "which", lambda name: name)
        monkeypatch.setattr(
            HostTelemetryCollector, "CONTRIBUTORS", (CONTRIBUTORS[kind],)
        )
        self.collector = HostTelemetryCollector(
            SimpleNamespace(post=lambda *_: {}),
            CollectorContext(
                cluster_id="cluster-a", runtime_profile_version="simulated-v1"
            ),
            node_id="node-a",
            filesystems=["/"],
            net_class_root=str(self.net),
            proc_root=str(self.proc),
            required_interfaces=["ens5"] if kind.startswith("net-") else [],
            infiniband_root=str(directory / "ib"),
            runner=self.command,
            statvfs=lambda _: SimpleNamespace(
                f_blocks=100,
                f_frsize=4096,
                f_bavail=1 if self.broken and kind != "inode" else 80,
                f_files=1000,
                f_ffree=10 if self.broken and kind == "inode" else 800,
            ),
            # Only the GPU inventory case configures an invariant; its
            # ``/proc/driver/nvidia/gpus`` does not exist under ``proc``.
            expected_gpu_count=8 if kind == "gpu-unknown" else None,
            edge_filter_enabled=False,
            now=lambda: self.clock,
        )
        self.store = SqliteStore(str(directory / "control.db"))
        self.core = FakeCoreApi()
        owner = "simulated-runtime"
        self.context = ApplicationContext(
            store=self.store,
            production_executor_config=ProductionExecutorConfig(
                enabled=True,
                executor_id="general-host-e2e",
                allowed_operations=OPERATIONS,
            ),
            production_adapters=[
                ControlPlaneEvidenceAdapter(owner=owner),
                GpuValidationAdapter(
                    GpuMetricsService(store=self.store), store=self.store, owner=owner
                ),
                KubernetesWorkflowAdapter(
                    owner=owner,
                    core_api=self.core,
                    batch_api=UnusedApi(),
                    custom_api=UnusedApi(),
                    store=self.store,
                ),
            ],
            execution_token="general-host-e2e-" + "x" * 32,
        )
        self.store.save_profile(default_simulated_profile())

    def set_fault(self, broken):
        self.broken = broken
        available = 1 if broken and self.kind == "memory" else 800
        swap_free = 100 if broken and self.kind == "swap" else 1000
        cached = 950 if broken and self.kind == "page-cache" else 0
        free = available
        if cached:
            free = 20
        (self.proc / "meminfo").write_text(
            f"MemTotal: 1000 kB\nMemAvailable: {available} kB\n"
            f"MemFree: {free} kB\nCached: {cached} kB\n"
            f"SwapTotal: 1000 kB\nSwapFree: {swap_free} kB\n"
        )
        (self.proc / "stat").write_text(
            "cpu 199 0 0 801 0 0 0 0\n" if broken else "cpu 100 0 0 800 0 0 0 0\n"
        )
        self.counter = getattr(self, "counter", 0) + (5 if broken else 1)
        elapsed = max(
            15, (self.clock - getattr(self, "disk_at", self.clock)).total_seconds()
        )
        self.disk_at = self.clock
        self.disk_ms = getattr(self, "disk_ms", 0) + int(
            elapsed * (990 if broken else 10)
        )
        self.disk_ios = getattr(self, "disk_ios", 0) + int(elapsed * 10)
        (self.proc / "diskstats").write_text(
            f"8 0 nvme0n1 {self.disk_ios} 0 0 0 0 0 0 0 0 {self.disk_ms} {self.disk_ios * 2}\n"
        )
        if self.kind in {"net-down", "net-unknown", "net-errors"}:
            interface = self.net / "ens5"
            stats = interface / "statistics"
            stats.mkdir(parents=True, exist_ok=True)
            state = "unknown" if self.kind == "net-unknown" else "down"
            if self.kind == "net-errors":
                state = "up"
            (interface / "operstate").write_text(state if broken else "up")
            for counter in ("rx_errors", "tx_errors", "rx_dropped", "tx_dropped"):
                (stats / counter).write_text(
                    str(self.counter)
                    if counter == "rx_errors" and self.kind == "net-errors"
                    else "0"
                )
        if self.kind == "rdma-errors":
            port = self.directory / "ib" / "efa_0" / "ports" / "1"
            counters = port / "counters"
            counters.mkdir(parents=True, exist_ok=True)
            device = port.parents[1] / "device"
            device.mkdir(exist_ok=True)
            (device / "uevent").write_text("DRIVER=efa\n")
            (port / "state").write_text("4: ACTIVE\n")
            (port / "phys_state").write_text("5: LinkUp\n")
            (counters / "port_rcv_errors").write_text(str(self.counter))

    def command(self, argv, **_):
        if argv == ["smartctl", "--scan-open"]:
            if self.kind == "smart-scan":
                # smartctl reports a failed platform scan on stdout, exit 0.
                return CompletedProcess(
                    argv, 0, "# scan_smart_devices: glob(3) aborted\n", ""
                )
            return CompletedProcess(argv, 0, "/dev/test -d scsi\n", "")
        if argv[0] == "nvidia-smi" and self.kind == "gpu-unknown":
            return CompletedProcess(
                argv, 255, "", "Failed to initialize NVML: Unknown Error"
            )
        if argv == ["smartctl", "-H", "-j", "/dev/test"]:
            if self.kind == "smart-query":
                return CompletedProcess(argv, 2, "not-json", "query failed")
            return CompletedProcess(
                argv, 0, json.dumps({"smart_status": {"passed": not self.broken}}), ""
            )
        if argv == ["ipmitool", "sensor"]:
            if self.kind == "bmc-query":
                return CompletedProcess(argv, 1, "", "query failed")
            status = "cr" if self.broken else "ok"
            return CompletedProcess(
                argv, 0, f"CPU Temp | 95.000 | degrees C | {status} | na\n", ""
            )
        raise AssertionError(f"unexpected external command: {argv}")

    async def publish(self, client):
        batch = self.collector.collect_once().model_copy(
            update={"received_at": self.clock}
        )
        present = {sample.name for sample in batch.samples}
        for name, value in (
            ("load1_per_cpu", 0.1),
            ("memory_used_percent", 20),
            ("filesystem_used_percent", 20),
            ("rdma_link_down", 0),
            ("rdma_errors_delta", 0),
            ("network_link_up", 1),
        ):
            if name not in present:
                batch.samples.append(HostMetricSample(name=name, value=value))
        response = await client.post(
            "/v1/collector-events/host-telemetry", json=batch.model_dump(mode="json")
        )
        assert response.status_code == 200, response.text
        self.evidence.append(
            {
                "raw_files": {
                    str(path.relative_to(self.directory)): path.read_text()
                    for root in (self.proc, self.net, self.directory / "ib")
                    for path in root.rglob("*")
                    if path.is_file()
                },
                "simulated_statvfs_bavail": (
                    1 if self.broken and self.kind != "inode" else 80
                ),
                "simulated_statvfs_ffree": (
                    10 if self.broken and self.kind == "inode" else 800
                ),
                "injected_kind": self.kind,
                "batch": batch.model_dump(mode="json"),
                "decision": response.json(),
            }
        )
        return batch, response.json()

    def execute(self, request_id):
        return self.context.workflow_executor.execute(
            request_id, WorkflowExecutionRequest(expected_fencing_token=1)
        )

    def close(self):
        evidence = {
            "samples": self.evidence,
            "workflows": [
                item.model_dump(mode="json") for item in self.store.list_workflows()
            ],
            "simulated_node_final_state": self.core.node,
            "live_actions_executed": False,
        }
        self.set_fault(False)
        self.store.close()
        evidence["cleanup"] = (
            "fixture fault cleared and database closed; no live resources created"
        )
        print("GENERAL_FAULT_EVIDENCE=" + json.dumps(evidence, default=str))
