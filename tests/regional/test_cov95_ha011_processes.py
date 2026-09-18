from __future__ import annotations

import time
from pathlib import Path
from threading import Thread
from types import SimpleNamespace

import pytest

from scripts.e2e.regional.ha011_contracts import ProofError
from scripts.e2e.regional.probes import ha011_processes as processes
from tests.regional._cov95_ha011_support import (
    blocked_external_transports as blocked_external_transports,
)


class Connection:
    def __init__(self) -> None:
        self.response = {"kind": "ready", "pid": 321}
        self.available = True
        self.closed = False

    def poll(self, _timeout):
        return self.available

    def recv(self):
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response

    def close(self):
        self.closed = True


class Process:
    def __init__(self, *, args, **_kwargs) -> None:
        self.stop = args[3]
        self.pid = 321
        self.exitcode = None
        self.running = False
        self.killed = False
        self.hang = False
        self.unkillable = False
        self.wrong_wait = False
        self.start_error = False

    def start(self):
        if self.start_error:
            raise OSError("controlled spawn failure")
        self.running = True

    def is_alive(self):
        return self.running

    def kill(self):
        self.killed = True
        if not self.unkillable:
            self.running = False
            self.exitcode = 1 if self.wrong_wait else -9

    def join(self, timeout):
        if self.running and self.stop.is_set() and not self.hang:
            self.running = False
            self.exitcode = 0


@pytest.fixture
def context(monkeypatch):
    first, second = Connection(), Connection()
    created = []

    def spawn(**kwargs):
        process = Process(**kwargs)
        created.append(process)
        return process

    context = SimpleNamespace(Pipe=lambda: (first, second), Process=spawn)
    monkeypatch.setattr(processes.multiprocessing, "get_context", lambda mode: context)
    return first, second, created, context


def worker():
    return processes.WorkerProcess(
        lambda *_args: None, role="spool", request_id="owned"
    )


def test_file_signals_wait_without_cross_process_locks(tmp_path: Path) -> None:
    signal = processes.FileSignal(tmp_path / "signal")
    assert signal.wait(0) is False, "unset control flags must obey finite waits"
    results = []
    thread = Thread(target=lambda: results.append(signal.wait()))
    thread.start()
    signal.set()
    thread.join(timeout=2)
    assert not thread.is_alive() and results == [True], (
        "owned file flags must wake without a process-owned condition lock"
    )
    signal.set()
    assert signal.wait(0), "flag writes must be idempotent"


def test_spawn_failure_closes_pipe_and_removes_owned_signals(context) -> None:
    first, second, created, factory = context
    original = factory.Process

    def cannot_start(**kwargs):
        process = original(**kwargs)
        process.start_error = True
        return process

    factory.Process = cannot_start
    with pytest.raises(OSError, match="spawn failure"):
        worker()
    assert first.closed and second.closed, (
        "failed spawn must close both owned pipe ends"
    )
    assert not created[0].stop.path.parent.exists(), (
        "a child that never started must leave no control directory"
    )


@pytest.mark.parametrize("failure", ["deadline", "exit", "eof", "pid", "type", "error"])
def test_observation_failures_are_bounded_and_never_accepted(
    context, failure: str
) -> None:
    first, _second, created, _factory = context
    current = worker()
    try:
        deadline = time.monotonic() + 1
        if failure == "deadline":
            deadline = 0
        elif failure == "exit":
            first.available = False
            created[0].running = False
            created[0].exitcode = 0
        elif failure == "eof":
            first.response = EOFError()
        elif failure == "pid":
            first.response["pid"] = 999
        elif failure == "type":
            first.response = ["not-an-observation"]
        else:
            first.response["kind"] = "error"
        with pytest.raises(ProofError):
            current.receive(deadline)
    finally:
        current.close()
    assert first.closed, "observation failures must still close their owned pipe"


def test_alive_poll_and_matching_request_wait(context) -> None:
    first, _second, _created, _factory = context
    current = worker()
    try:
        first.available = False
        current.receive(time.monotonic() + 1)
        assert current.events == [], "an empty poll must not manufacture an observation"
        first.available = True
        first.response = {"kind": "claim", "pid": 321, "request_id": "owned"}
        assert (
            current.wait("claim", time.monotonic() + 1, "owned")["request_id"]
            == "owned"
        ), "wait must return the matching request observation"
    finally:
        current.close()


@pytest.mark.parametrize("failure", ["already-dead", "wrong-status", "unkillable"])
def test_crash_requires_a_confirmed_owned_wait_status(context, failure: str) -> None:
    _first, _second, created, _factory = context
    current = worker()
    process = created[0]
    if failure == "already-dead":
        process.running = False
        process.exitcode = 0
    elif failure == "wrong-status":
        process.wrong_wait = True
    else:
        process.unkillable = True
    try:
        with pytest.raises(ProofError):
            current.crash()
    finally:
        process.running = False
        process.exitcode = 0
        current.close()


def test_grace_timeout_is_not_reported_as_a_clean_shutdown(context) -> None:
    _first, _second, created, _factory = context
    current = worker()
    process = created[0]
    process.hang = True
    with pytest.raises(ProofError, match="grace"):
        current.finish()
    assert process.killed and not process.running, (
        "grace failure must still drain only the owned child"
    )
    with pytest.raises(ProofError, match="unsuccessfully"):
        current.close()
    assert not current.stop.path.parent.exists(), (
        "confirmed dead child control files must be removed"
    )


def test_unconfirmed_process_preserves_owned_control_files(context) -> None:
    first, _second, created, _factory = context
    current = worker()
    created[0].hang = created[0].unkillable = True
    with pytest.raises(ProofError, match="grace"):
        current.close()
    assert first.closed and current.stop.path.parent.exists(), (
        "unconfirmed process death must preserve control files for recovery"
    )
    created[0].running = False
    created[0].exitcode = 0
    current.close()
    assert not current.stop.path.parent.exists(), (
        "the fixture must remove its files once fake death is confirmed"
    )
