"""``gpu_fault.app.serve``: one SO_REUSEPORT service socket per worker.

Why: under ``uvicorn --workers`` every worker accepts from the one socket the
supervisor bound, so a cold-connect burst waits in a single kernel accept
queue drained at the slowest worker's pace (4,250 connections against four
workers kept 3,600-3,900 waiting; ``scripts/perf/ingress_event_loop_burst.py``).
With a socket per worker the kernel hashes each connection to one of the
workers' own queues. These tests drive the real launcher as a subprocess on a
tiny ASGI app (``tests/app/_reuseport_app.py``) and read the kernel's own view
(``/proc/net/tcp``) of the listening sockets.
"""

from __future__ import annotations

import http.client
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.app import listener as listener_module
from gpu_fault.app import process_health, serve
from gpu_fault.app.listener import ServiceListener
from gpu_fault.app.metric_aggregation import STRATEGIES, Strategy

ROOT = Path(__file__).resolve().parents[2]
LISTEN = "0A"


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _get(port: int, path: str) -> tuple[int, dict[str, Any]]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        connection.request("GET", path)
        response = connection.getresponse()
        return response.status, json.loads(response.read() or b"{}")
    finally:
        connection.close()


def _listening_inodes(port: int) -> set[str]:
    """Inodes of every LISTEN socket on ``port`` in this network namespace."""

    suffix = f":{port:04X}"
    inodes: set[str] = set()
    with open("/proc/net/tcp", encoding="ascii") as handle:
        next(handle)
        for line in handle:
            fields = line.split()
            if fields[1].endswith(suffix) and fields[3] == LISTEN:
                inodes.add(fields[9])
    return inodes


def _launch(port: int, *extra: str, app: str = "tests.app._reuseport_app:app"):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "src"), str(ROOT), env.get("PYTHONPATH", "")]
    )
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "gpu_fault.app.serve",
            app,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--timeout-worker-healthcheck",
            "5",
            "--timeout-graceful-shutdown",
            "3",
            "--no-access-log",
            "--no-proxy-headers",
            *extra,
        ],
        env=env,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def _wait_for(predicate, timeout: float, what: str) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.1)
    pytest.fail(f"timed out waiting for {what}")


def _stop(process: subprocess.Popen) -> str:
    if process.poll() is None:
        process.send_signal(signal.SIGTERM)
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    output = process.stdout.read().decode(errors="replace") if process.stdout else ""
    return output


@pytest.fixture
def launched():
    processes: list[subprocess.Popen] = []

    def start(port: int, *extra: str, **kwargs: Any) -> subprocess.Popen:
        process = _launch(port, *extra, **kwargs)
        processes.append(process)
        return process

    yield start
    for process in processes:
        _stop(process)


def test_every_worker_binds_its_own_listening_socket_and_accepts(launched) -> None:
    port = _free_port()
    process = launched(port, "--workers", "3")
    _wait_for(lambda: len(_listening_inodes(port)) == 3, 30, "three LISTEN sockets")
    assert len(_listening_inodes(port)) == 3, (
        "each worker must own a SO_REUSEPORT socket; uvicorn shares one"
    )
    pids: dict[int, int] = {}
    for _ in range(90):
        status, payload = _get(port, "/pid")
        assert status == 200
        pids[payload["pid"]] = pids.get(payload["pid"], 0) + 1
    assert len(pids) == 3, (
        f"the kernel must spread connections over all workers: {pids}"
    )
    assert process.pid not in pids, "the supervisor itself never serves"
    status, payload = _get(port, "/listener")
    facts = payload["listener"]
    assert facts["reuse_port"] is True
    assert facts["port"] == port
    assert facts["worker_index"] in {0, 1, 2}
    # 90 /pid connections plus this one, spread over the workers; the
    # worker that answered counted at least its own share plus this request.
    assert facts["accepted_total"] >= 1 + min(pids.values())
    assert facts["open_connections"] >= 1, "this connection is open while it answers"


def test_sigterm_stops_every_worker_and_releases_the_port(launched) -> None:
    port = _free_port()
    process = launched(port, "--workers", "2")
    _wait_for(lambda: len(_listening_inodes(port)) == 2, 30, "two LISTEN sockets")
    process.send_signal(signal.SIGTERM)
    assert process.wait(timeout=30) == 0, "a graceful stop exits 0"
    output = _stop(process)
    assert "Stopping parent process" in output
    assert not _listening_inodes(port), "no worker may survive the supervisor"


def test_a_worker_that_dies_after_serving_is_replaced(launched) -> None:
    port = _free_port()
    launched(port, "--workers", "2")
    _wait_for(lambda: len(_listening_inodes(port)) == 2, 30, "two LISTEN sockets")
    before: set[int] = set()
    for _ in range(40):
        before.add(_get(port, "/pid")[1]["pid"])
    assert len(before) == 2
    victim = sorted(before)[0]
    os.kill(victim, signal.SIGKILL)

    def replaced() -> bool:
        if len(_listening_inodes(port)) != 2:
            return False
        seen: set[int] = set()
        for _ in range(20):
            try:
                seen.add(_get(port, "/pid")[1]["pid"])
            except OSError:
                # A connection hashed to the dying worker's socket is reset
                # until the kernel drops that socket from the SO_REUSEPORT
                # group (no tcp_migrate_req in a Pod netns); the clients'
                # retry ladder covers that window.
                return False
        return victim not in seen and len(seen) == 2

    _wait_for(replaced, 30, "a replacement worker")


def test_a_worker_that_never_serves_stops_the_parent_with_startup_failure(
    launched,
) -> None:
    port = _free_port()
    process = launched(
        port, "--workers", "2", "--factory", app="tests.app._reuseport_app:broken"
    )
    assert process.wait(timeout=60) == serve.STARTUP_FAILURE, (
        "a broken application must not be crash-looped one child at a time"
    )
    output = _stop(process)
    assert "cannot build the application" in output
    assert "failed to start; stopping the parent" in output
    assert not _listening_inodes(port), "no worker socket may outlive the parent"


def test_the_command_line_is_uvicorns_subset_and_is_validated() -> None:
    settings = serve.parse_args(
        [
            "gpu_fault.app:create_app",
            "--factory",
            "--host",
            "0.0.0.0",
            "--port",
            "8080",
            "--workers",
            "4",
            "--loop",
            "uvloop",
            "--http",
            "httptools",
            "--backlog",
            "8192",
            "--timeout-keep-alive",
            "5",
            "--no-proxy-headers",
            "--timeout-graceful-shutdown",
            "60",
            "--timeout-worker-healthcheck",
            "20",
            "--no-access-log",
            "--limit-concurrency",
            "4096",
        ]
    )
    assert settings == serve.ServerSettings(
        app="gpu_fault.app:create_app",
        factory=True,
        host="0.0.0.0",
        port=8080,
        workers=4,
        loop="uvloop",
        http="httptools",
        backlog=8192,
        timeout_keep_alive=5,
        timeout_graceful_shutdown=60,
        timeout_worker_healthcheck=20,
        limit_concurrency=4096,
        proxy_headers=False,
        access_log=False,
    )
    config = settings.uvicorn_config()
    assert config.workers == 1, "each worker runs one Server; the supervisor fans out"
    assert (config.backlog, config.timeout_keep_alive, config.limit_concurrency) == (
        8192,
        5,
        4096,
    )
    assert config.proxy_headers is False and config.access_log is False
    for argv in (
        ["app:app", "--workers", "0"],
        ["app:app", "--port", "70000"],
        ["app:app", "--backlog", "0"],
        ["app:app", "--unknown-flag"],
    ):
        with pytest.raises(SystemExit):
            serve.parse_args(argv)


def test_two_sockets_share_a_port_only_with_reuse_port() -> None:
    first = serve.bind_service_socket("127.0.0.1", 0, 16)
    port = int(first.getsockname()[1])
    try:
        second = serve.bind_service_socket("127.0.0.1", port, 16)
        second.close()
        with pytest.raises(OSError):
            serve.bind_service_socket("127.0.0.1", port, 16, reuse_port=False)
        assert first.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT) == 1
    finally:
        first.close()


def test_the_listener_facts_render_as_per_worker_metrics(monkeypatch) -> None:
    monkeypatch.setattr(listener_module, "_LISTENER", None)
    lines: list[str] = []
    process_health.render_service_listener_metrics(lines)
    assert lines == [], "nothing is rendered behind the plain uvicorn supervisor"

    listener = listener_module.register_service_listener(
        ServiceListener(port=8080, reuse_port=True, worker_index=2)
    )
    listener.connection_accepted()
    listener.connection_accepted()
    listener.open_connections = lambda: 7
    process_health.render_service_listener_metrics(lines)
    samples = [line for line in lines if not line.startswith("#")]
    assert samples == [
        "gpu_fault_listen_reuse_port 1",
        'gpu_fault_listen_accepted_connections_total{port="8080"} 2',
        'gpu_fault_listen_open_connections{port="8080"} 7',
    ]
    listener.open_connections = lambda: 1 // 0
    assert listener.snapshot()["open_connections"] == 0, "a metrics read never raises"
    listener_module.reset_service_listener()
    assert listener_module.service_listener() is None


def test_listen_families_aggregate_per_worker() -> None:
    """Each worker's queue and accept counter are its own; the Pod sums open
    connections; reuse_port is one answer per Pod."""

    assert (
        STRATEGIES["gpu_fault_listen_accept_queue_connections"] is Strategy.PER_PROCESS
    )
    assert (
        STRATEGIES["gpu_fault_listen_accepted_connections_total"]
        is Strategy.PER_PROCESS
    )
    assert STRATEGIES["gpu_fault_listen_open_connections"] is Strategy.SUM
    assert STRATEGIES["gpu_fault_listen_reuse_port"] is Strategy.ANY
