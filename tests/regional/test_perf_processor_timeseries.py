"""The processor's second-by-second series and the drain from the load stop.

The burst-50 analysis had only before/after snapshots: whether the workers were
busy while the backlog lasted could not be answered, and the drain probe began
after the after-load collection, about 45 s after the load stopped. Every
control-plane Pod now samples its processor families once per second through
one in-Pod session, and the drain probe runs from the load stop beside the
collection; the old placement's readings stay as ``*_after_collection``.
"""

from __future__ import annotations

import http.server
import json
import os
import select
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from scripts.perf import regional_capacity_registry as registry
from scripts.perf import regional_capacity_suite as capacity
from scripts.perf import regional_capacity_timeseries as timeseries

ROOT = Path(__file__).resolve().parents[2]


def sample(t: float, **values: float) -> dict[str, Any]:
    return {"t": t, "v": values}


def test_the_series_summary_reads_in_flight_against_capacity_while_backlog_lasts() -> (
    None
):
    cycles = "gpu_fault_processor_consumer_cycles_total"
    samples = [
        sample(
            0.0,
            gpu_fault_processor_queue_depth=0,
            gpu_fault_processor_in_flight=0,
            gpu_fault_processor_workers=16,
        ),
        {"t": 1.0, "error": "URLError"},
        sample(
            2.0,
            gpu_fault_processor_queue_depth=900,
            gpu_fault_processor_in_flight=12,
            gpu_fault_processor_workers=16,
            **{f'{cycles}{{process="0"}}': 10, f'{cycles}{{process="1"}}': 10},
        ),
        sample(
            4.0,
            gpu_fault_processor_queue_depth=400,
            gpu_fault_processor_in_flight=4,
            gpu_fault_processor_workers=16,
            **{f'{cycles}{{process="0"}}': 30, f'{cycles}{{process="1"}}': 30},
        ),
    ]

    summary = timeseries.series_summary(samples)

    assert summary == {
        "samples": 4,
        "errors": 1,
        "backlog_samples": 2,
        "capacity_workers": 16,
        "mean_in_flight_while_backlog": 8.0,
        "max_in_flight_while_backlog": 12.0,
        "utilization_while_backlog": 0.5,
        "consumer_cycles_per_second_while_backlog": 20.0,
    }, "only backlog samples count; cycles add up over the Pod's processes"


class FakeSession:
    """A remote sampler: samples until its stdin (the stop channel) closes."""

    def __init__(self, values: list[dict[str, Any]]) -> None:
        self.closed = threading.Event()
        self.written = b""
        self.stdin = self
        self.stdout = self._lines(values)
        self.killed = False

    def write(self, data: bytes) -> int:
        self.written += data
        return len(data)

    def flush(self) -> None:
        return None

    def close(self) -> None:
        self.closed.set()

    def kill(self) -> None:
        self.killed = True

    def _lines(self, values: list[dict[str, Any]]) -> Iterator[bytes]:
        for item in values:
            yield (json.dumps(item) + "\n").encode()
        self.closed.wait(timeout=30)


def test_every_pod_runs_one_sampler_session_until_the_run_stops_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(registry, "CONTROL_KUBECONFIG", "/tmp/cpu.kubeconfig")
    monkeypatch.setattr(registry, "CONTROL_NAMESPACE", "gpu-fault-system")
    pods = [
        "gpu-fault-api-ha-a",
        "gpu-fault-control-worker-b",
        "gpu-fault-telemetry-spool-worker-c",
    ]
    sessions: dict[str, FakeSession] = {}
    argvs: list[list[str]] = []

    def popen(argv: list[str], **kwargs: Any) -> FakeSession:
        assert (
            kwargs["stdin"] is subprocess.PIPE and kwargs["stdout"] is subprocess.PIPE
        )
        argvs.append(argv)
        pod = argv[argv.index("-i") + 1]
        sessions[pod] = FakeSession(
            [
                sample(
                    1.0,
                    gpu_fault_processor_queue_depth=5,
                    gpu_fault_processor_workers=8,
                )
            ]
            * 3
        )
        return sessions[pod]

    series = timeseries.ProcessorTimeseries(pods, limit=2, popen=popen).start()
    deadline = time.monotonic() + 10
    while (
        any(len(item.samples) < 2 for item in series.sessions.values())
        and time.monotonic() < deadline
    ):
        time.sleep(0.01)

    series.stop()

    ports = {argv[argv.index("-i") + 1]: argv for argv in argvs}
    assert "SAMPLE_PORT=8080" in ports["gpu-fault-api-ha-a"]
    assert "SAMPLE_PORT=8081" in ports["gpu-fault-control-worker-b"]
    assert "SAMPLE_PORT=8082" in ports["gpu-fault-telemetry-spool-worker-c"]
    assert all(
        argv[-3:] == ["python3", "-c", timeseries.BOOTSTRAP] for argv in argvs
    ), "every session runs the bootstrap that reads its program from stdin"
    assert all(
        item.written == timeseries.session_input(timeseries.SAMPLE_PROGRAM)
        and item.closed.is_set()
        for item in sessions.values()
    ), "the program arrives length-prefixed and stop() closes the stop channel"
    document = series.document()
    assert all(
        len(entry["samples"]) == 2 and entry["truncated"]
        for entry in document["pods"].values()
    ), "a Pod keeps at most the sample limit"
    assert series.summary()["consumers"] == {
        "pods": 3,
        "capacity_workers": 24,
        "mean_in_flight_while_backlog": 0.0,
    }


class MetricsHandler(http.server.BaseHTTPRequestHandler):
    BODY = "\n".join(
        [
            "# TYPE gpu_fault_processor_in_flight gauge",
            "gpu_fault_processor_in_flight 7",
            'gpu_fault_processor_in_flight_by_phase{phase="replay"} 7',
            "gpu_fault_processor_workers 16",
            "gpu_fault_processor_queue_depth 120",
            'gpu_fault_processor_consumer_cycles_total{process="0"} 41',
            "gpu_fault_ingress_lane_in_flight 3",
        ]
    ).encode()

    def do_GET(self) -> None:  # noqa: N802 - the http.server hook
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(self.BODY)

    def log_message(self, *_args: Any) -> None:
        return None


def test_the_in_pod_program_samples_its_own_metrics_until_stdin_closes(
    tmp_path: Path,
) -> None:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), MetricsHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        process = subprocess.Popen(
            [sys.executable, "-c", timeseries.BOOTSTRAP],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=tmp_path,
            env={
                "PATH": os.environ.get("PATH", ""),
                "SAMPLE_PORT": str(server.server_address[1]),
                "SAMPLE_INTERVAL": "0.05",
                "SAMPLE_LIMIT": "100",
                "SAMPLE_FAMILIES": ",".join(timeseries.SAMPLED_FAMILIES),
            },
        )
        assert process.stdin is not None and process.stdout is not None
        process.stdin.write(timeseries.session_input(timeseries.SAMPLE_PROGRAM))
        process.stdin.flush()
        # The 0.5 s of sampling starts with the first sample: under a loaded
        # release gate the interpreter's start-up alone can outlast a fixed
        # sleep, and stdin's EOF would then stop the program before it sampled.
        ready, _, _ = select.select([process.stdout], [], [], 60)
        assert ready, "the sampler printed no sample within 60 s"
        first = process.stdout.readline()
        time.sleep(0.5)
        assert process.poll() is None, "the sampler stopped while stdin was open"
        output, errors = process.communicate(timeout=60)  # closes stdin: EOF
    finally:
        server.shutdown()

    assert process.returncode == 0, errors.decode()
    lines = [json.loads(line) for line in (first + output).decode().splitlines()]
    assert len(lines) >= 3, "about one sample per interval"
    assert lines[0]["v"] == {
        "gpu_fault_processor_in_flight": 7.0,
        "gpu_fault_processor_workers": 16.0,
        "gpu_fault_processor_queue_depth": 120.0,
        'gpu_fault_processor_consumer_cycles_total{process="0"}': 41.0,
    }, "the sampled families exactly, labelled per process; nothing else"


def test_the_drain_reference_reads_the_old_placement_off_the_new_history() -> None:
    drain = {
        "drain_seconds": 50.0,
        "depth_at_load_stop": 4000.0,
        "samples": [
            {"epoch": 100.0, "queue_depth": 4000.0},
            {"epoch": 130.0, "queue_depth": 2500.0},
            {"epoch": 146.0, "queue_depth": 1200.0},
            {"epoch": 150.0, "queue_depth": 1.0},
        ],
    }

    reference = timeseries.drain_reference(drain, collected_at=145.0)

    assert reference["depth_at_load_stop"] == 4000.0, "measured from the load stop"
    assert reference["depth_after_collection"] == 1200.0, (
        "the first sample once the collection had finished"
    )
    assert reference["drain_seconds_after_collection"] == 5.0
    assert timeseries.drain_reference({"samples": []}, 1.0) == {"samples": []}


def test_the_drain_starts_at_the_load_stop_beside_the_after_load_collection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    started = threading.Event()

    def drain(pods: list[str], **targets: float) -> dict[str, Any]:
        events.append("drain-start")
        started.set()
        return {
            "samples": [{"epoch": time.time(), "queue_depth": 0.0}],
            "targets": targets,
        }

    def after(name: str, value: Any) -> Any:
        def call(*_args: Any, **_kwargs: Any) -> Any:
            assert started.wait(timeout=10), "the drain must already be sampling"
            events.append(name)
            return value

        return call

    monkeypatch.setattr(capacity, "queue_drain", drain)
    monkeypatch.setattr(capacity, "scrape_metrics", after("metrics-after", {}))
    monkeypatch.setattr(capacity, "scrape_cgroup", after("cgroup-after", {}))
    monkeypatch.setattr(capacity, "postgres_counters", after("postgres-after", {}))
    monkeypatch.setattr(capacity, "collect_logs", after("logs", []))

    _cgroup, _pg, _documents, reference = capacity.collect_after_load(
        ["api"], tmp_path, "job", targets=(5.0, 1.0)
    )

    assert events == [
        "drain-start",
        "metrics-after",
        "cgroup-after",
        "postgres-after",
        "logs",
    ]
    assert reference["targets"] == {
        "target_queue_depth": 5.0,
        "target_spool_depth": 1.0,
    }
    assert "drain_seconds_after_collection" in reference
    assert (tmp_path / "metrics-after.json").is_file(), "the after-load scrape is kept"


def test_a_background_call_returns_its_value_or_raises_its_error() -> None:
    ok = timeseries.Background(lambda value: value * 2, 21)
    ok.start()
    failing = timeseries.Background(lambda: 1 / 0)
    failing.start()

    assert ok.result(timeout=10) == 42
    with pytest.raises(ZeroDivisionError):
        failing.result(timeout=10)


def test_the_suite_keeps_the_processor_families_it_samples() -> None:
    for family in (
        'gpu_fault_processor_consumer_cycles_total{process="0"} 1',
        "gpu_fault_processor_request_processing_seconds_count 9",
        "gpu_fault_processor_fault_pressure_active 1",
        "gpu_fault_telemetry_spool_fault_pressure_workers 2",
    ):
        assert capacity.METRIC_PATTERN.search(family), family
