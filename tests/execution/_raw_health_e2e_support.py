"""Raw exporter/journal/file injection into the real health control-plane rig."""

import json
import subprocess

from gpu_fault.collectors import CollectorContext
from gpu_fault.collectors.gpu.dcgm import (
    DCGM_METRICS,
    DCGM_NANOSECOND_DURATION_FIELDS,
    DcgmMetricsCollector,
)
from gpu_fault.collectors.gpu.discovery import query_nvidia_temperature_limits
from gpu_fault.collectors.logs.node import NodeLogCollector
from tests.execution._health_policy_e2e_support import JOB, HealthRig


class RawHealthRig(HealthRig):
    def __init__(self, path, monkeypatch):
        super().__init__(path, monkeypatch)
        self.path = path
        self.raw_inputs = []
        self.responses = []
        self.raw_context = CollectorContext(
            cluster_id="cluster-a",
            runtime_profile_version="simulated-v1",
            product="H100",
            workload_state="ACTIVE",
            affected_workload_ids=[JOB],
        )
        rig = self

        class Sink:
            def post(self, route, payload):
                response = rig.post(route, payload)
                rig.responses.append(response)
                return response

        self.sink = Sink()
        self.exporter = DcgmMetricsCollector(
            self.sink,
            self.raw_context,
            node_id="node-a",
            edge_filter_enabled=False,
            now=lambda: self.clock,
        )
        monkeypatch.setattr(
            "gpu_fault.collectors.logs.node.shutil.which",
            lambda _: "/usr/bin/journalctl",
        )

    def gpu(self, values, *, errors=None, temperature_xml=None):
        defaults = {
            "gpu_temperature_c": 40,
            "ecc_dbe_volatile_total": 0,
            "row_remap_failure": 0,
            "row_remap_pending": 0,
            "pcie_replay_total": 0,
            "clock_throttle_reasons": 0,
            "power_limit_w": 500,
            "nvlink_crc_flit_error_total": 0,
            "nvlink_recovery_error_total": 0,
            "nvlink_replay_error_total": 0,
        }
        defaults.update(values)
        if errors:
            defaults.pop("row_remap_failure", None)
        names = {canonical: name for name, (canonical, _) in DCGM_METRICS.items()}
        text = "".join(
            f'{names[name]}{{UUID="GPU-a",pci_bus_id="0000:81:00.0",gpu="0"}} '
            f"{value * 1000 if name in DCGM_NANOSECOND_DURATION_FIELDS else value}\n"
            for name, value in defaults.items()
        )
        self.raw_inputs.append({"source": "dcgm-exporter", "text": text})
        extra = []
        if temperature_xml is not None:
            self.raw_inputs.append(
                {"source": "nvidia-smi-xml", "text": temperature_xml}
            )
            extra = query_nvidia_temperature_limits(
                lambda argv, **_: subprocess.CompletedProcess(
                    argv, 0, temperature_xml, ""
                )
            )
        self.exporter.collect_text(text, observed_at=self.clock, extra_samples=extra)
        return self.responses[-1]

    def log(self, text, *, source="dmesg", fields=None):
        self.sequence += 1
        paths = []
        records = []
        if source == "training-log":
            path = (
                self.path
                / "pods"
                / "ns_train_pod-a"
                / "trainer"
                / f"{self.sequence}.log"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"{self.clock.isoformat()} stdout F {text}\n")
            paths.append(str(path))
        else:
            records.append(
                {
                    "__CURSOR": f"cursor-{self.sequence}",
                    "__REALTIME_TIMESTAMP": str(
                        int(self.clock.timestamp() * 1_000_000)
                    ),
                    "_TRANSPORT": "kernel" if source == "dmesg" else "stdout",
                    "_BOOT_ID": "boot-current",
                    "MESSAGE": text,
                    **(fields or {}),
                }
            )
        raw = "\n".join(json.dumps(record) for record in records)
        self.raw_inputs.append(
            {"source": source, "text": raw if records else path.read_text()}
        )
        collector = NodeLogCollector(
            self.sink,
            self.raw_context,
            node_id="node-a",
            boot_id="boot-current",
            now=lambda: self.clock,
            training_log_paths=paths,
            initial_tail_bytes=65536,
            pod_log_root=str(self.path / "pods"),
            context_lines=0,
            runner=lambda argv, **_: subprocess.CompletedProcess(argv, 0, raw, ""),
        )
        collector.collect_once()
        return self.responses[-1]
