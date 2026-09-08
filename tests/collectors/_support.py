# ruff: noqa: F401
from __future__ import annotations

import base64
import gzip
import io
import json
import logging
import os
import subprocess
import threading
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from gpu_fault import collectors_cli
from gpu_fault.app import ApplicationContext, create_app
from gpu_fault.collectors import (
    CONNECTION_POOL,
    NVIDIA_SMI_CORE_FIELDS,
    NVIDIA_SMI_REMAP_FIELDS,
    CloudWatchHmaCollector,
    CollectorContext,
    CollectorError,
    DcgmMetricsCollector,
    FabricManagerLogCollector,
    HostTelemetryCollector,
    HttpEventSink,
    KernelLogCollector,
    KubernetesHmaNodeCollector,
    KubernetesNodeResourceCollector,
    NodeLogCollector,
    NvidiaSmiMetricsCollector,
    SqsEventSink,
    SqsHmaConsumer,
    TrainingProgressCollector,
    context_from_environment,
    discover_gpu_product,
    discover_gpu_software_versions,
    next_stable_phase,
    normalize_gpu_product,
    query_nvidia_temperature_limits,
    sink_from_environment,
    stable_phase_seconds,
)
from gpu_fault.gpu_metrics import GpuMetricSample
from gpu_fault.hma import HMA_FAULT_DETAILS, HMA_FAULT_REASONS, HMA_HEALTH_STATUS

NOW = datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc)


class _LocalControlPlane:
    """Loopback HTTP/1.1 server that records connections and bodies."""

    def __init__(
        self, *, status: int = 200, drop_after_request: int | None = None
    ) -> None:
        self.connections: list[tuple[str, int]] = []
        self.requests: list[tuple[str | None, dict[str, Any]]] = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self) -> None:  # noqa: N802
                raw = self.rfile.read(int(self.headers["Content-Length"]))
                encoding = self.headers.get("Content-Encoding")
                if encoding == "gzip":
                    raw = gzip.decompress(raw)
                owner.requests.append((encoding, json.loads(raw)))
                body = b'{"ok":true}'
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                if (
                    drop_after_request is not None
                    and len(owner.requests) == drop_after_request
                ):
                    # Close without announcing it, which is exactly how
                    # an idle keep-alive connection goes stale.
                    self.close_connection = True

            def log_message(self, *_args) -> None:
                return

        class Server(ThreadingHTTPServer):
            daemon_threads = True

            def process_request(self, request, client_address):
                owner.connections.append(client_address)
                super().process_request(request, client_address)

        self._server = Server(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self) -> _LocalControlPlane:
        self._thread.start()
        return self

    def __exit__(self, *_args) -> bool:
        CONNECTION_POOL.close()
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)
        return False


class RecordingSink:
    def __init__(self) -> None:
        self.requests: list[tuple[str, dict[str, Any]]] = []

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.requests.append((path, payload))
        return {"accepted": True}


class BufferingSink:
    """A sink whose durable outbox took every record: nothing went live.

    ``HttpEventSink.post`` raises after the outbox accepted the record, so a
    collector that reads the exception as "the event is lost" re-reads its
    source and re-posts the same payload every tick (ARCH-G3).
    """

    def __init__(self) -> None:
        self.requests: list[tuple[str, dict[str, Any]]] = []

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.requests.append((path, payload))
        raise CollectorError("network unavailable", buffered=True, replayable=True)


class RejectingSink:
    """A sink whose control plane rejected the record: it went nowhere."""

    def __init__(self, *, status_code: int = 422) -> None:
        self.status_code = status_code
        self.requests: list[tuple[str, dict[str, Any]]] = []

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.requests.append((path, payload))
        raise CollectorError(
            f"rejected ({self.status_code})", status_code=self.status_code
        )


class StopTheLoop(BaseException):
    """Breaks a collector's ``while True`` without being caught by it.

    Every collector run loop catches ``Exception`` on purpose, so a test that
    drives ``run()`` needs a non-``Exception`` signal to stop it.
    """


def context() -> CollectorContext:
    return CollectorContext(
        cluster_id="hp-cluster",
        runtime_profile_version="simulated-v1",
        product="H100",
        driver_branch=575,
        cuda_version="12.9",
    )


def completed_nvidia_smi(
    stdout: str, *, returncode: int = 0, stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["nvidia-smi"], returncode=returncode, stdout=stdout, stderr=stderr
    )


def cloudwatch_envelope(
    log_events: list[dict[str, Any]],
    *,
    stream: str = "worker-1/SagemakerHealthMonitoringAgent",
) -> dict[str, Any]:
    payload = json.dumps(
        {
            "messageType": "DATA_MESSAGE",
            "owner": "123456789012",
            "logGroup": "/aws/sagemaker/Clusters/hp-cluster",
            "logStream": stream,
            "logEvents": log_events,
        }
    ).encode()
    return {"awslogs": {"data": base64.b64encode(gzip.compress(payload)).decode()}}


def write_fake_rank(
    root, pid: int, *, cpu_ticks: int, write_bytes: int, wchar: int
) -> None:
    directory = root / str(pid)
    directory.mkdir(parents=True, exist_ok=True)
    # proc(5) numbers ``state`` as field 3, so this list covers fields
    # 3 to 22: utime at index 11, stime at 12, starttime at 19.
    fields = ["0"] * 20
    fields[0] = "S"
    fields[11] = str(cpu_ticks)
    fields[19] = "980"
    (directory / "stat").write_text(
        f"{pid} (pt_main_thread) " + " ".join(fields) + "\n"
    )
    (directory / "io").write_text(
        "rchar: 0\n"
        f"wchar: {wchar}\n"
        "syscr: 0\n"
        "syscw: 7\n"
        "read_bytes: 0\n"
        f"write_bytes: {write_bytes}\n"
        "cancelled_write_bytes: 0\n"
    )
    (directory / "status").write_text(
        "Name:\tpt_main_thread\n"
        "voluntary_ctxt_switches:\t11\n"
        "nonvoluntary_ctxt_switches:\t3\n"
    )


def rank_liveness_collector(tmp_path, *, gpu_utilization: int):
    def runner(argv, **_kwargs):
        stdout = (
            "1234\n5678\n"
            if any("compute-apps" in str(item) for item in argv)
            else f"GPU-a, {gpu_utilization}\nGPU-b, {gpu_utilization}\n"
        )
        return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")

    return HostTelemetryCollector(
        RecordingSink(),
        context(),
        node_id="worker-1",
        proc_root=str(tmp_path),
        runner=runner,
    )


def rank_liveness_cycle(collector, observed_at):
    collector._gpu_utilization(observed_at)
    return {item.name: item.value for item in collector._rank_liveness(observed_at)}


__all__ = [name for name in globals() if not name.startswith("__")]
