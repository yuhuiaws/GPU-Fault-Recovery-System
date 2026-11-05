"""Fakes for driving ``gpu_fault.app.serve`` in this process.

In the real launch (``tests/app/test_serve_reuse_port.py`` drives it end to
end as a subprocess) the worker code runs inside spawned children. These fakes
replace the two parts a test cannot run here: uvicorn's event loop
(``ScriptedServer``, installed as ``uvicorn.Server``) and the spawned child
(``ScriptedWorker``, installed as ``serve.WorkerProcess``). Everything else
stays real: the sockets, the health pipes, the signal handlers the supervisor
installs and the uvicorn ``Config`` that loads the application.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import socket
import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import pytest
from uvicorn.server import ServerState

from gpu_fault.app import listener as listener_module
from gpu_fault.app import serve

#: ``--http`` takes an import string as well as a protocol name.
RECORDING_PROTOCOL = "tests.app._cov95_serve_fakes:RecordingProtocol"
HEALTH_THREAD = "gpu-fault-serve-health"
SUPERVISED_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
UVICORN_LOGGERS = ("uvicorn", "uvicorn.error", "uvicorn.access", "uvicorn.asgi")


class FakeTransport:
    """The transport an accepted connection hands its protocol."""


TRANSPORT = FakeTransport()


class RecordingProtocol(asyncio.Protocol):
    """An HTTP protocol class with uvicorn's constructor signature.

    Like uvicorn's own protocols it joins the server's live connection set when
    a connection is made, which is what the worker's listener reads as open
    connections.
    """

    def __init__(
        self,
        config: Any,
        server_state: ServerState,
        app_state: dict[str, Any],
        _loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self.config = config
        self.server_state = server_state
        self.app_state = app_state
        self.transport: Any = None

    def connection_made(self, transport: Any) -> None:  # type: ignore[override]
        self.transport = transport
        self.server_state.connections.add(self)  # type: ignore[arg-type]


@dataclass
class ServerScript:
    """How every ``ScriptedServer`` built during a test behaves."""

    start: bool = True
    interrupt: bool = False
    while_starting: Callable[[ScriptedServer], None] | None = None
    servers: list[ScriptedServer] = field(default_factory=list)

    def server(self, config: Any) -> ScriptedServer:
        """Stands in for ``uvicorn.Server(config)``."""

        built = ScriptedServer(config, self)
        self.servers.append(built)
        return built


class ScriptedServer:
    """``uvicorn.Server`` without the event loop.

    ``run(sockets)`` records what it was handed, then does what uvicorn does
    before it listens: loads the application through the config (the
    launcher's wrapped ``load``), builds one HTTP protocol the way uvicorn's
    ``create_protocol`` does and gives it a connection. It reports started per
    the script and closes the sockets, as uvicorn's shutdown does.
    """

    def __init__(self, config: Any, script: ServerScript) -> None:
        self.config = config
        self.script = script
        self.started = False
        self.ran = False
        self.server_state = ServerState()
        self.accepting: list[int] = []
        self.ports: list[int] = []
        self.protocols: list[Any] = []

    def run(self, sockets: Sequence[socket.socket] | None = None) -> None:
        self.ran = True
        handed = list(sockets or [])
        try:
            for sock in handed:
                self.accepting.append(
                    sock.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN)
                )
                self.ports.append(int(sock.getsockname()[1]))
            if self.script.interrupt:
                raise KeyboardInterrupt
            self.config.load()
            protocol = self.config.http_protocol_class(
                config=self.config, server_state=self.server_state, app_state={}
            )
            protocol.connection_made(TRANSPORT)
            self.protocols.append(protocol)
            if self.script.while_starting is not None:
                self.script.while_starting(self)
            self.started = self.script.start
        finally:
            for sock in handed:
                sock.close()


class LogRecords(logging.Handler):
    """Every record the launcher logged during one test."""

    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def messages(self, level: int | None = None) -> list[str]:
        return [
            record.getMessage()
            for record in self.records
            if level is None or record.levelno == level
        ]


def record_launcher_logs(monkeypatch: pytest.MonkeyPatch) -> LogRecords:
    """Give the launcher a logger uvicorn's ``dictConfig`` cannot rewire.

    Building a ``uvicorn.Config`` reconfigures the ``uvicorn`` loggers (it
    replaces their handlers and stops propagation), so neither ``caplog`` nor a
    handler attached beforehand would see the launcher's lines.
    """

    records = LogRecords()
    logger = logging.Logger("tests.app.serve-launcher", logging.DEBUG)
    logger.propagate = False
    logger.addHandler(records)
    monkeypatch.setattr(serve, "LOGGER", logger)
    return records


@contextmanager
def uvicorn_logging_restored() -> Iterator[None]:
    """Undo the logging configuration every ``uvicorn.Config`` applies."""

    saved = []
    for name in UVICORN_LOGGERS:
        logger = logging.getLogger(name)
        saved.append((logger, list(logger.handlers), logger.level, logger.propagate))
    try:
        yield
    finally:
        for logger, handlers, level, propagate in saved:
            for handler in logger.handlers:
                if handler not in handlers:
                    handler.close()
            logger.handlers = handlers
            logger.setLevel(level)
            logger.propagate = propagate


@contextmanager
def service_listener_restored() -> Iterator[None]:
    """Start without a registered listener; put back whatever was there."""

    previous = listener_module.service_listener()
    listener_module.reset_service_listener()
    try:
        yield
    finally:
        if previous is None:
            listener_module.reset_service_listener()
        else:
            listener_module.register_service_listener(previous)


def health_threads_stopped(timeout: float = 10.0) -> bool:
    """Join the workers' health threads; ``True`` when none is left running."""

    for thread in threading.enumerate():
        if thread.name == HEALTH_THREAD:
            thread.join(timeout)
    return not any(
        thread.name == HEALTH_THREAD and thread.is_alive()
        for thread in threading.enumerate()
    )


class FakeProcess:
    """The spawned child behind a ``serve.WorkerProcess``, never started."""

    def __init__(
        self, *, alive: bool = True, pid: int | None = 4242, exitcode: int | None = None
    ) -> None:
        self.alive = alive
        self.pid = pid
        self.exitcode = exitcode
        self.calls: list[Any] = []

    def start(self) -> None:
        self.calls.append("start")

    def is_alive(self) -> bool:
        return self.alive

    def kill(self) -> None:
        self.calls.append("kill")

    def join(self, timeout: float | None = None) -> None:
        self.calls.append(("join", timeout))


class FailingConnection:
    """A pipe end whose close fails, as a socket already torn down can."""

    def __init__(self) -> None:
        self.close_attempts = 0

    def close(self) -> None:
        self.close_attempts += 1
        raise OSError("bad file descriptor")


@contextmanager
def signal_handlers_restored() -> Iterator[None]:
    """``Supervisor.run`` installs process-wide handlers; put the old ones back."""

    saved = {signum: signal.getsignal(signum) for signum in SUPERVISED_SIGNALS}
    try:
        yield
    finally:
        for signum, handler in saved.items():
            if handler is not None:
                signal.signal(signum, handler)


def deliver(signum: int, *, handler_of: int | None = None) -> None:
    """Run the Python handler installed for ``handler_of`` (default ``signum``)
    with ``signum``, the way the interpreter runs it when the signal arrives."""

    handler = signal.getsignal(signum if handler_of is None else handler_of)
    assert callable(handler), f"no Python handler is installed for {signum}"
    handler(signum, None)


def signals_on_call(
    call: int, *signums: int, handler_of: int | None = None
) -> Callable[[int], None]:
    """An ``on_alive`` hook delivering ``signums`` on the ``call``-th health check."""

    def hook(count: int) -> None:
        if count == call:
            for signum in signums:
                deliver(signum, handler_of=handler_of)

    return hook


@dataclass
class WorkerPlan:
    """How one spawned ``ScriptedWorker`` behaves.

    ``alive`` answers successive health checks (the last answer repeats);
    ``exitcode`` is what the worker exited with once killed; the hooks run
    inside the matching call.
    """

    alive: Sequence[bool] = (True,)
    ready: bool = True
    exitcode: int = -signal.SIGKILL
    on_alive: Callable[[int], None] | None = None
    on_join: Callable[[], None] | None = None
    on_wait: Callable[[threading.Event], None] | None = None


class ScriptedWorker:
    """Stands in for ``serve.WorkerProcess``: no child, scripted answers."""

    def __init__(
        self, fleet: Fleet, settings: serve.ServerSettings, index: int, serial: int
    ) -> None:
        self.fleet = fleet
        self.settings = settings
        self.index = index
        self.plan = fleet.plans.pop(0) if fleet.plans else WorkerPlan()
        self.name = f"w{serial}"
        self.pid = 1000 + serial
        self.exitcode: int | None = None
        self.alive_calls = 0
        self.health_timeouts: list[float] = []
        self.ready_args: tuple[float, threading.Event] | None = None

    def start(self) -> None:
        self.fleet.events.append(("start", self.name))

    def is_alive(self, timeout: float) -> bool:
        self.alive_calls += 1
        self.health_timeouts.append(timeout)
        self.fleet.count_health_check()
        answers = self.plan.alive
        answer = answers[min(self.alive_calls, len(answers)) - 1]
        if self.plan.on_alive is not None:
            self.plan.on_alive(self.alive_calls)
        return answer

    def wait_until_ready(self, timeout: float, should_exit: threading.Event) -> bool:
        self.fleet.events.append(("wait", self.name))
        self.ready_args = (timeout, should_exit)
        if self.plan.on_wait is not None:
            self.plan.on_wait(should_exit)
        return self.plan.ready

    def terminate(self) -> None:
        self.fleet.events.append(("terminate", self.name))

    def kill(self) -> None:
        self.fleet.events.append(("kill", self.name))
        self.exitcode = self.plan.exitcode

    def join(self, timeout: float | None = None) -> None:
        self.fleet.events.append(("join", self.name))
        if self.plan.on_join is not None:
            self.plan.on_join()


class Fleet:
    """The workers a ``Supervisor`` spawns, built from per-spawn plans.

    Install ``fleet.worker`` as ``serve.WorkerProcess``. ``events`` is the
    order in which workers were started, waited for, terminated, killed and
    joined. A scripted stop that never reaches the supervisor would leave its
    loop running; past ``RUNAWAY_CHECKS`` health checks the fleet delivers
    SIGTERM itself and sets ``runaway`` for the test to fail on.
    """

    RUNAWAY_CHECKS = 500

    def __init__(self, *plans: WorkerPlan) -> None:
        self.plans = list(plans)
        self.spawned: list[ScriptedWorker] = []
        self.events: list[tuple[str, str]] = []
        self.health_checks = 0
        self.runaway = False
        self.supervisor: serve.Supervisor | None = None

    def worker(self, settings: serve.ServerSettings, index: int) -> ScriptedWorker:
        spawned = ScriptedWorker(self, settings, index, serial=len(self.spawned))
        self.spawned.append(spawned)
        return spawned

    def count_health_check(self) -> None:
        self.health_checks += 1
        if self.health_checks > self.RUNAWAY_CHECKS and not self.runaway:
            self.runaway = True
            deliver(signal.SIGTERM)

    def stop_supervisor(self) -> None:
        """Ask the supervisor to stop from inside a worker call."""

        assert self.supervisor is not None, (
            "the test must hand the fleet its supervisor"
        )
        self.supervisor.should_exit.set()
