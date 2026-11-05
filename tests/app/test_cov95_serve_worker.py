"""``gpu_fault.app.serve`` workers, in this process.

``tests/app/test_serve_reuse_port.py`` launches the real supervisor as a
subprocess, so what runs inside a spawned worker never shows up in its
coverage. Here a worker runs in the test process with uvicorn's event loop
replaced by ``ScriptedServer`` (``tests/app/_cov95_serve_fakes.py``): the
socket it binds, the listener it registers for ``/metrics``, the protocol
wrapper that counts accepted connections, the start-up failure exits and the
health pipe the supervisor pings are all real.
"""

from __future__ import annotations

import errno
import logging
import multiprocessing
import os
import runpy
import signal
import socket
import subprocess
import sys
import threading
from collections.abc import Iterator
from typing import Any

import pytest

from gpu_fault.app import listener as listener_module
from gpu_fault.app import serve
from tests.app._cov95_serve_fakes import (
    RECORDING_PROTOCOL,
    TRANSPORT,
    FailingConnection,
    FakeProcess,
    LogRecords,
    RecordingProtocol,
    ScriptedServer,
    ServerScript,
    health_threads_stopped,
    record_launcher_logs,
    service_listener_restored,
    uvicorn_logging_restored,
)

APP = "tests.app._reuseport_app:app"


@pytest.fixture(autouse=True)
def launcher_globals() -> Iterator[None]:
    with uvicorn_logging_restored(), service_listener_restored():
        yield


@pytest.fixture
def script(monkeypatch: pytest.MonkeyPatch) -> ServerScript:
    scripted = ServerScript()
    monkeypatch.setattr(serve.uvicorn, "Server", scripted.server)
    return scripted


@pytest.fixture
def logs(monkeypatch: pytest.MonkeyPatch) -> LogRecords:
    return record_launcher_logs(monkeypatch)


def _settings(**overrides: Any) -> serve.ServerSettings:
    values: dict[str, Any] = {
        "app": APP,
        "port": 0,
        "http": RECORDING_PROTOCOL,
        "access_log": False,
    }
    values.update(overrides)
    return serve.ServerSettings(**values)


def _ping(connection: Any) -> Any:
    connection.send(b"ping")
    assert connection.poll(10), "the worker's health thread must answer a ping"
    return connection.recv()


def _close_worker_pipes(worker: serve.WorkerProcess) -> bool:
    """Close the supervisor's end first so the health thread sees EOF and ends."""

    worker.parent_conn.close()
    stopped = health_threads_stopped()
    worker.child_conn.close()
    return stopped


def test_a_worker_hands_uvicorn_a_bound_socket_and_counts_what_it_accepts(
    script: ServerScript, logs: LogRecords
) -> None:
    assert serve.serve_worker(_settings(backlog=64), 3) is True, (
        "a worker whose server started reports that it served"
    )
    (server,) = script.servers
    assert server.accepting == [0], (
        "uvicorn gets a bound socket that does not listen yet: listen() comes "
        "after the factory and the lifespan start-up"
    )
    (port,) = server.ports
    assert port > 0
    listener = listener_module.service_listener()
    assert listener is not None, "the worker registers its socket for /metrics"
    assert (listener.port, listener.reuse_port, listener.worker_index) == (
        port,
        True,
        3,
    )
    snapshot = listener.snapshot()
    assert (snapshot["accepted_total"], snapshot["open_connections"]) == (1, 1), (
        "the accepted connection is counted and read from uvicorn's live set"
    )
    protocol_class = server.config.http_protocol_class
    assert issubclass(protocol_class, RecordingProtocol), protocol_class
    assert protocol_class is not RecordingProtocol, "the protocol is wrapped"
    assert (protocol_class.__name__, protocol_class.__qualname__) == (
        RecordingProtocol.__name__,
        RecordingProtocol.__qualname__,
    ), "the counting wrapper keeps uvicorn's protocol name in tracebacks and logs"
    (protocol,) = server.protocols
    assert protocol.transport is TRANSPORT, (
        "uvicorn's protocol still sees the connection"
    )
    assert logs.messages() == [
        f"worker 3 pid {os.getpid()} bound 127.0.0.1:{port} on its own "
        "SO_REUSEPORT socket (backlog 64); it listens once the application has "
        "started"
    ]


def test_a_worker_that_cannot_bind_its_port_exits_with_startup_failure(
    script: ServerScript, logs: LogRecords
) -> None:
    occupant = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        occupant.bind(("127.0.0.1", 0))
        occupant.listen(1)
        port = int(occupant.getsockname()[1])
        with pytest.raises(SystemExit) as stopped:
            serve.serve_worker(_settings(port=port), 1)
    finally:
        occupant.close()
    assert stopped.value.code == serve.STARTUP_FAILURE, (
        "a port held without SO_REUSEPORT is a start-up failure, not a crash"
    )
    (server,) = script.servers
    assert server.ran is False, "nothing is served without the socket"
    assert listener_module.service_listener() is None, "no listener is registered"
    (message,) = logs.messages(logging.ERROR)
    assert message.startswith(f"worker 1 cannot bind 127.0.0.1:{port}: "), message
    assert f"[Errno {errno.EADDRINUSE}]" in message, message


def test_an_application_factory_that_raises_exits_with_startup_failure(
    script: ServerScript, logs: LogRecords
) -> None:
    settings = _settings(app="tests.app._reuseport_app:broken", factory=True)
    with pytest.raises(SystemExit) as stopped:
        serve.serve_worker(settings, 2)
    assert stopped.value.code == serve.STARTUP_FAILURE, (
        "a factory that raises must stop the worker with uvicorn's start-up code "
        "so the supervisor does not restart it for ever"
    )
    (server,) = script.servers
    assert server.started is False
    assert server.config.http_protocol_class is RecordingProtocol, (
        "an application that never loaded gets no counting protocol"
    )
    assert server.accepting == [0], "the worker bound its socket before loading"
    failure = logs.records[-1]
    assert failure.getMessage() == "worker 2 cannot build the application"
    assert failure.exc_info is not None, "the factory's traceback is logged"
    assert isinstance(failure.exc_info[1], RuntimeError), failure.exc_info


def test_an_application_that_cannot_be_imported_keeps_uvicorns_own_exit(
    script: ServerScript, logs: LogRecords
) -> None:
    settings = _settings(app="tests.app._cov95_no_such_module:app")
    with pytest.raises(SystemExit) as stopped:
        serve.serve_worker(settings, 0)
    assert stopped.value.code == serve.STARTUP_FAILURE, (
        "uvicorn exits STARTUP_FAILURE on an import error and the worker keeps it"
    )
    assert "worker 0 cannot build the application" not in logs.messages(), (
        "uvicorn's own exit passes through; it is not re-reported as a factory failure"
    )
    (server,) = script.servers
    assert server.started is False


def test_the_health_pipe_answers_whether_the_worker_serves_and_ends_on_eof(
    script: ServerScript,
) -> None:
    supervisor_end, worker_end = multiprocessing.Pipe()
    answers: list[Any] = []

    def ping_while_starting(server: ScriptedServer) -> None:
        answers.append(_ping(supervisor_end))

    script.while_starting = ping_while_starting
    try:
        assert serve.serve_worker(_settings(), 0, worker_end) is True
        answers.append(_ping(supervisor_end))
    finally:
        supervisor_end.close()
        stopped = health_threads_stopped()
        worker_end.close()
    assert answers == [False, True], (
        "a worker still starting answers 'not started' (so it is not replaced) "
        "and 'started' once it serves"
    )
    assert stopped, "the health thread ends once the supervisor's end closes"


def test_the_worker_process_runs_the_worker_on_its_own_health_pipe(
    script: ServerScript,
) -> None:
    worker = serve.WorkerProcess(_settings(), 0)
    assert worker.process.name == "gpu-fault-serve-worker-0"
    assert (worker.pid, worker.exitcode) == (None, None), "not started yet"
    answers: list[Any] = []

    def ping_while_starting(server: ScriptedServer) -> None:
        answers.append(_ping(worker.parent_conn))

    script.while_starting = ping_while_starting
    try:
        # What the spawned child runs: the process target, in this process.
        assert worker.process.run() is None
    finally:
        stopped = _close_worker_pipes(worker)
    assert answers == [False], "the child answers on the pipe the supervisor pings"
    assert script.servers[0].started is True
    assert stopped, "the health thread ends with the pipe"


def test_a_worker_process_that_never_served_exits_with_startup_failure(
    script: ServerScript,
) -> None:
    script.start = False
    worker = serve.WorkerProcess(_settings(), 1)
    try:
        with pytest.raises(SystemExit) as stopped:
            worker.process.run()
    finally:
        threads_stopped = _close_worker_pipes(worker)
    assert stopped.value.code == serve.STARTUP_FAILURE, (
        "a child whose server never started tells the supervisor so by its exit"
    )
    assert threads_stopped, "the health thread ends with the pipe"


def test_a_worker_process_interrupted_by_ctrl_c_exits_quietly(
    script: ServerScript,
) -> None:
    script.interrupt = True
    worker = serve.WorkerProcess(_settings(), 0)
    try:
        assert worker.process.run() is None, "Ctrl-C is a stop, not a failure"
    finally:
        threads_stopped = _close_worker_pipes(worker)
    assert threads_stopped, "the health thread ends with the pipe"
    (server,) = script.servers
    assert (server.ran, server.started) == (True, False)


def test_a_worker_is_alive_while_its_process_runs_and_answers_pings() -> None:
    worker = serve.WorkerProcess(_settings(), 0)
    worker.process = FakeProcess()
    try:
        worker.start()
        assert worker.process.calls == ["start"]
        worker.child_conn.send(True)
        assert worker.is_alive(timeout=5) is True, "a serving worker is alive"
        worker.child_conn.send(False)
        assert worker.is_alive(timeout=5) is True, (
            "a worker still starting answers too, so it is alive"
        )
        assert worker.is_alive(timeout=0.05) is False, (
            "no answer within the health-check timeout is a hung worker"
        )
        assert [worker.child_conn.recv() for _ in range(3)] == [b"ping"] * 3
        worker.child_conn.send_bytes(b"not a pickle")
        assert worker.is_alive(timeout=5) is False, (
            "an answer that does not unpickle is no answer"
        )
        assert worker.child_conn.recv() == b"ping"
        worker.process.alive = False
        assert worker.is_alive(timeout=5) is False, "a dead process is not alive"
        assert worker.child_conn.poll(0) is False, "a dead process is not pinged"
        worker.process.alive = True
        worker.child_conn.close()
        assert worker.is_alive(timeout=5) is False, "a broken pipe is no answer"
    finally:
        worker.close()


def test_waiting_for_a_new_worker_pings_until_it_serves() -> None:
    worker = serve.WorkerProcess(_settings(), 1)
    worker.process = FakeProcess()
    try:
        worker.child_conn.send(False)
        worker.child_conn.send(True)
        assert worker.wait_until_ready(30, threading.Event()) is True, (
            "a worker that comes up within the timeout is ready"
        )
        assert [worker.child_conn.recv() for _ in range(2)] == [b"ping", b"ping"]
        assert worker.child_conn.poll(0) is False, "no ping after it served"
    finally:
        worker.close()


def test_waiting_for_a_new_worker_gives_up_on_death_stop_or_deadline() -> None:
    worker = serve.WorkerProcess(_settings(), 1)
    worker.process = FakeProcess(alive=False)
    try:
        assert worker.wait_until_ready(30, threading.Event()) is False, (
            "a child that died while starting is never ready"
        )
        worker.process.alive = True
        stopping = threading.Event()
        stopping.set()
        assert worker.wait_until_ready(30, stopping) is False, (
            "a stopping supervisor does not wait for a new worker"
        )
        assert worker.child_conn.poll(0) is False, "neither case pings the child"
        for _ in range(20):
            worker.child_conn.send(False)
        assert worker.wait_until_ready(0.3, threading.Event()) is False, (
            "a worker still starting at the deadline is not ready"
        )
        pings = 0
        while worker.child_conn.poll(0):
            if worker.child_conn.recv() == b"ping":
                pings += 1
        assert 1 <= pings < 20, f"it kept asking until the deadline: {pings} pings"
    finally:
        worker.close()


def test_terminate_stops_a_running_child_with_sigterm(logs: LogRecords) -> None:
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    worker = serve.WorkerProcess(_settings(), 0)
    worker.process = FakeProcess(pid=child.pid)
    try:
        worker.terminate()
        assert child.wait(timeout=30) == -signal.SIGTERM, (
            "the child gets SIGTERM, uvicorn's graceful stop"
        )
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=30)
        worker.close()
    assert logs.messages() == [f"Terminated child process [{child.pid}]"]


def test_terminate_leaves_exited_unstarted_and_vanished_children_alone(
    monkeypatch: pytest.MonkeyPatch, logs: LogRecords
) -> None:
    sent: list[tuple[int, int]] = []

    def kill(pid: int, signum: int) -> None:
        sent.append((pid, signum))
        raise ProcessLookupError(errno.ESRCH, "No such process")

    monkeypatch.setattr(serve.os, "kill", kill)
    worker = serve.WorkerProcess(_settings(), 0)
    try:
        worker.process = FakeProcess(pid=4242, exitcode=0)
        worker.terminate()
        worker.process = FakeProcess(pid=None)
        worker.terminate()
        assert sent == [], "an exited or never-started child is not signalled"
        worker.process = FakeProcess(pid=4243)
        worker.terminate()
    finally:
        worker.close()
    assert sent == [(4243, signal.SIGTERM)]
    assert logs.messages() == ["Terminated child process [4243]"], (
        "a child that vanished before the signal is not an error"
    )


def test_join_waits_for_the_child_and_closes_both_pipe_ends() -> None:
    worker = serve.WorkerProcess(_settings(), 0)
    worker.process = FakeProcess(exitcode=-9)
    worker.kill()
    worker.join(timeout=2.5)
    assert worker.process.calls == ["kill", ("join", 2.5)]
    assert (worker.parent_conn.closed, worker.child_conn.closed) == (True, True)
    assert worker.exitcode == -9
    worker.close()
    assert worker.parent_conn.closed, "closing twice is harmless"


def test_closing_the_pipes_survives_an_end_that_fails_to_close() -> None:
    worker = serve.WorkerProcess(_settings(), 0)
    original = worker.parent_conn
    failing = FailingConnection()
    worker.parent_conn = failing  # type: ignore[assignment]
    try:
        worker.close()
    finally:
        original.close()
    assert failing.close_attempts == 1
    assert worker.child_conn.closed, "the other end is still closed"


def test_one_worker_serves_in_process_and_exits_zero(script: ServerScript) -> None:
    argv = [APP, "--port", "0", "--http", RECORDING_PROTOCOL, "--no-access-log"]
    assert serve.main(argv) == 0, "a single worker that served exits 0"
    (server,) = script.servers
    assert server.started is True
    listener = listener_module.service_listener()
    assert listener is not None and listener.worker_index == 0, listener


def test_one_worker_that_never_serves_exits_with_startup_failure(
    script: ServerScript,
) -> None:
    script.start = False
    argv = [APP, "--port", "0", "--http", RECORDING_PROTOCOL, "--no-access-log"]
    assert serve.main(argv) == serve.STARTUP_FAILURE, (
        "--workers 1 runs in-process and still exits STARTUP_FAILURE"
    )
    assert len(script.servers) == 1, "one in-process server, no supervisor"


def test_the_module_runs_as_a_command(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["serve", "--help"])
    # A fresh execution of the module as ``__main__``, as ``python -m`` runs it.
    monkeypatch.delitem(sys.modules, "gpu_fault.app.serve")
    with pytest.raises(SystemExit) as stopped:
        runpy.run_module("gpu_fault.app.serve", run_name="__main__")
    assert stopped.value.code == 0
    usage = capsys.readouterr().out
    assert usage.startswith("usage: python -m gpu_fault.app.serve"), usage
    assert "--timeout-worker-healthcheck" in usage, usage
