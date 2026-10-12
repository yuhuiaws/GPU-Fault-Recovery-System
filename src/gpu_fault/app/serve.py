"""Run the control-plane application with one service socket per worker.

``uvicorn --workers N`` binds the service port once, in the supervisor, and
hands that socket to every worker: all N processes ``accept()`` from one
kernel accept queue. Under the 50-cluster cold-connect burst that queue is
where the storm lives - 4,250 connections against four workers kept
3,600-3,900 of them waiting in the shared FIFO while the processes held
550-740 each (``scripts/perf/ingress_event_loop_burst.py``) - and a queue
shared by four processes is drained at the pace of whichever worker is
slowest to come back to ``accept()``. The liveness probe already moved to a
listener of its own (``gpu_fault.app.liveness``); this module does the same
for the service port.

Every worker binds the service port itself with ``SO_REUSEPORT``: the kernel
hashes each incoming connection's 4-tuple to one of the N sockets, so the
Pod has N independent accept queues (each bounded by ``somaxconn`` - 4096 in
a Pod network namespace - so the Pod as a whole can hold N times as many
completed connections before SYNs are dropped) and a worker stalled in a GC
pause or a long flush delays only the connections hashed to it. Measured in
the same reproduction with ``--launcher uvicorn|serve``: at 4,250 concurrent
cold connections the two are within noise (1,141/1,135 vs 1,110/1,097 req/s,
client p99 4.18/4.03 vs 4.32/4.51 s, every worker 24.7-25.2 % of the
requests); at 5,500 - above ``somaxconn``, where the live 50-cluster burst's
4,266 connections per Pod sit - the shared socket pins at 4,097 queued and
drops SYNs (961 req/s, 73 503s, client p99 27.9 s, max 31.6 s) while four
sockets hold 5,089 (1,051-1,186 each) at 1,117 req/s, 0 non-2xx, p99 5.35 s.
The known costs are the same hashing: a connection is pinned to its worker for its
lifetime, so long-lived client connections do not rebalance until they
reconnect; the 4-tuple hash gives a per-worker share with the variance of a
random assignment; and a connection still in a worker's accept queue when
that worker's socket closes (a crash, or the stop at the end of a roll) is
reset rather than handed to a sibling, because a Pod network namespace
cannot set ``net.ipv4.tcp_migrate_req`` - the clients' idempotent retry
ladder covers that window. ``/metrics`` exports what each worker accepted and
holds open (``gpu_fault_listen_accepted_connections_total``,
``gpu_fault_listen_open_connections``) and the depth of its own accept queue
(``gpu_fault_listen_accept_queue_connections``), so the imbalance is a
number, not a guess.

The command line is uvicorn's own subset that the manifests use::

    python -m gpu_fault.app.serve gpu_fault.app:create_app --factory \\
        --host 0.0.0.0 --port 8080 --workers 4 --loop uvloop --http httptools \\
        --backlog 8192 --timeout-keep-alive 5 --no-proxy-headers \\
        --timeout-graceful-shutdown 60 --timeout-worker-healthcheck 20 \\
        --no-access-log --limit-concurrency 4096

Everything after the socket is uvicorn: each worker runs ``uvicorn.Server``
on its socket with the configured protocol, loop, keep-alive, graceful
shutdown and concurrency limit, and the application is unchanged. The
supervisor keeps uvicorn's contract - SIGTERM/SIGINT stop every worker
gracefully, SIGHUP restarts them one at a time, a worker that stops
answering the health-check pipe for ``--timeout-worker-healthcheck`` seconds
or dies is replaced, and a worker that exits with uvicorn's
``STARTUP_FAILURE`` (3: the bind, the import, the application factory or the
lifespan start-up failed) stops the parent with the same code, so the Pod
restarts and backs off instead of crash-looping one child. ``--workers 1``
runs in-process, as uvicorn does, still on an ``SO_REUSEPORT`` socket of its
own.
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import os
import pickle
import signal
import socket
import sys
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from multiprocessing.connection import Connection
from typing import Any

import uvicorn
from uvicorn.config import STARTUP_FAILURE

from gpu_fault.app.listener import ServiceListener, register_service_listener

# The supervisor logs through uvicorn's own logger so its lines sit beside
# "Started server process" in the same format.
LOGGER = logging.getLogger("uvicorn.error")

multiprocessing.allow_connection_pickling()
SPAWN = multiprocessing.get_context("spawn")

#: How often the supervisor looks at its workers and its signal queue.
SUPERVISOR_TICK_SECONDS = 0.5
#: Signals the supervisor acts on; anything else keeps its default.
_SIGNALS = {signal.SIGINT: "int", signal.SIGTERM: "term", signal.SIGHUP: "hup"}


@dataclass(frozen=True)
class ServerSettings:
    """The command line, as plain values a spawned worker can unpickle."""

    app: str
    factory: bool = False
    host: str = "127.0.0.1"
    port: int = 8000
    workers: int = 1
    loop: str = "auto"
    http: str = "auto"
    backlog: int = 2048
    timeout_keep_alive: int = 5
    timeout_graceful_shutdown: int | None = None
    timeout_worker_healthcheck: int = 5
    limit_concurrency: int | None = None
    proxy_headers: bool = True
    access_log: bool = True
    log_level: str | None = None

    def uvicorn_config(self) -> uvicorn.Config:
        return uvicorn.Config(
            self.app,
            factory=self.factory,
            host=self.host,
            port=self.port,
            # One process per Server: the supervisor here owns the fan-out.
            workers=1,
            loop=self.loop,
            http=self.http,
            backlog=self.backlog,
            timeout_keep_alive=self.timeout_keep_alive,
            timeout_graceful_shutdown=self.timeout_graceful_shutdown,
            timeout_worker_healthcheck=self.timeout_worker_healthcheck,
            limit_concurrency=self.limit_concurrency,
            proxy_headers=self.proxy_headers,
            access_log=self.access_log,
            log_level=self.log_level,
        )


def parse_args(argv: Sequence[str] | None = None) -> ServerSettings:
    parser = argparse.ArgumentParser(
        prog="python -m gpu_fault.app.serve",
        description=(
            "Serve an ASGI application with uvicorn, one SO_REUSEPORT service "
            "socket per worker process."
        ),
    )
    parser.add_argument("app", help="ASGI application or factory, module:attribute")
    parser.add_argument("--factory", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--loop", default="auto")
    parser.add_argument("--http", default="auto")
    parser.add_argument("--backlog", type=int, default=2048)
    parser.add_argument("--timeout-keep-alive", type=int, default=5)
    parser.add_argument("--timeout-graceful-shutdown", type=int, default=None)
    parser.add_argument("--timeout-worker-healthcheck", type=int, default=5)
    parser.add_argument("--limit-concurrency", type=int, default=None)
    parser.add_argument(
        "--no-proxy-headers", dest="proxy_headers", action="store_false"
    )
    parser.add_argument("--no-access-log", dest="access_log", action="store_false")
    parser.add_argument("--log-level", default=None)
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if not 0 <= args.port <= 65535:
        parser.error("--port must be a TCP port")
    if args.backlog < 1 or args.timeout_worker_healthcheck < 1:
        parser.error("--backlog and --timeout-worker-healthcheck must be positive")
    return ServerSettings(
        app=args.app,
        factory=args.factory,
        host=args.host,
        port=args.port,
        workers=args.workers,
        loop=args.loop,
        http=args.http,
        backlog=args.backlog,
        timeout_keep_alive=args.timeout_keep_alive,
        timeout_graceful_shutdown=args.timeout_graceful_shutdown,
        timeout_worker_healthcheck=args.timeout_worker_healthcheck,
        limit_concurrency=args.limit_concurrency,
        proxy_headers=args.proxy_headers,
        access_log=args.access_log,
        log_level=args.log_level,
    )


def bind_service_socket(
    host: str, port: int, backlog: int, *, reuse_port: bool = True
) -> socket.socket:
    """A listening socket this process owns; ``SO_REUSEPORT`` lets siblings share the port."""

    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if reuse_port:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        sock.bind((host, port))
        sock.listen(backlog)
    except OSError:
        sock.close()
        raise
    return sock


def _counting_protocol(base: type[Any], listener: ServiceListener) -> type[Any]:
    """uvicorn's HTTP protocol class, counting every accepted connection."""

    class CountingProtocol(base):  # type: ignore[misc]
        def connection_made(self, transport: Any) -> None:
            listener.connection_accepted()
            super().connection_made(transport)

    CountingProtocol.__name__ = base.__name__
    CountingProtocol.__qualname__ = base.__qualname__
    return CountingProtocol


def _answer_health_pings(connection: Connection, server: uvicorn.Server) -> None:
    """Reply to the supervisor's pings with whether this worker is serving."""

    while True:
        try:
            connection.recv()
            connection.send(bool(server.started))
        except (EOFError, OSError):
            return


def serve_worker(
    settings: ServerSettings,
    worker_index: int,
    health: Connection | None = None,
) -> bool:
    """Bind this worker's socket and run uvicorn on it; ``True`` once it served."""

    config = settings.uvicorn_config()
    server = uvicorn.Server(config)
    if health is not None:
        threading.Thread(
            target=_answer_health_pings,
            args=(health, server),
            name="gpu-fault-serve-health",
            daemon=True,
        ).start()
    try:
        sock = bind_service_socket(settings.host, settings.port, settings.backlog)
    except OSError as exc:
        LOGGER.error(
            "worker %s cannot bind %s:%s: %s",
            worker_index,
            settings.host,
            settings.port,
            exc,
        )
        sys.exit(STARTUP_FAILURE)
    listener = register_service_listener(
        ServiceListener(
            port=int(sock.getsockname()[1]), reuse_port=True, worker_index=worker_index
        )
    )
    listener.open_connections = lambda: len(server.server_state.connections)
    original_load = config.load

    def load_and_count() -> None:
        # Runs where uvicorn runs it: inside ``Server._serve`` on the worker's
        # event loop, so the application factory sees the same loop as under
        # the uvicorn command. uvicorn exits STARTUP_FAILURE on an import
        # error; a factory that raises would otherwise die with a traceback
        # and exit 1, which the supervisor could only read as a crash to
        # restart for ever.
        try:
            original_load()
        except SystemExit:
            raise
        except Exception:
            LOGGER.exception("worker %s cannot build the application", worker_index)
            sys.exit(STARTUP_FAILURE)
        config.http_protocol_class = _counting_protocol(
            config.http_protocol_class, listener
        )

    config.load = load_and_count  # type: ignore[method-assign]
    LOGGER.info(
        "worker %s pid %s serves %s:%s on its own SO_REUSEPORT socket (backlog %s)",
        worker_index,
        os.getpid(),
        settings.host,
        listener.port,
        settings.backlog,
    )
    server.run(sockets=[sock])
    return bool(server.started)


def _worker_main(
    settings: ServerSettings, worker_index: int, health: Connection
) -> None:
    """Spawn entry point: exit ``STARTUP_FAILURE`` if the server never served."""

    try:
        started = serve_worker(settings, worker_index, health)
    except KeyboardInterrupt:
        return
    if not started:
        sys.exit(STARTUP_FAILURE)


class WorkerProcess:
    """One spawned worker and the pipe the supervisor pings it on."""

    def __init__(self, settings: ServerSettings, index: int) -> None:
        self.settings = settings
        self.index = index
        self.parent_conn, self.child_conn = SPAWN.Pipe()
        self.process = SPAWN.Process(
            target=_worker_main,
            args=(settings, index, self.child_conn),
            name=f"gpu-fault-serve-worker-{index}",
        )

    def start(self) -> None:
        self.process.start()

    def _healthcheck(self, timeout: float) -> bool | None:
        try:
            self.parent_conn.send(b"ping")
            if self.parent_conn.poll(timeout):
                return bool(self.parent_conn.recv())
            return None
        except (OSError, EOFError, pickle.UnpicklingError):
            return None

    def is_alive(self, timeout: float) -> bool:
        return self.process.is_alive() and self._healthcheck(timeout) is not None

    def wait_until_ready(self, timeout: float, should_exit: threading.Event) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not should_exit.is_set():
            if not self.process.is_alive():
                return False
            if self._healthcheck(timeout=1.0) is True:
                return True
            time.sleep(0.1)
        return False

    def terminate(self) -> None:
        if self.process.exitcode is None and self.process.pid is not None:
            try:
                os.kill(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            LOGGER.info("Terminated child process [%s]", self.process.pid)

    def kill(self) -> None:
        self.process.kill()

    def join(self, timeout: float | None = None) -> None:
        self.process.join(timeout)
        self.close()

    def close(self) -> None:
        for connection in (self.parent_conn, self.child_conn):
            try:
                connection.close()
            except OSError:
                pass

    @property
    def pid(self) -> int | None:
        return self.process.pid

    @property
    def exitcode(self) -> int | None:
        return self.process.exitcode


class Supervisor:
    """uvicorn's multiprocess contract over :func:`serve_worker` children."""

    def __init__(self, settings: ServerSettings) -> None:
        self.settings = settings
        self.workers: list[WorkerProcess] = []
        self.should_exit = threading.Event()
        self.startup_failed = False
        self._signal_queue: list[int] = []

    def _install_signal_handlers(self) -> None:
        for sig in _SIGNALS:
            signal.signal(sig, lambda signum, _frame: self._signal_queue.append(signum))

    def _spawn(self, index: int) -> WorkerProcess:
        worker = WorkerProcess(self.settings, index)
        worker.start()
        return worker

    def run(self) -> int:
        LOGGER.info(
            "Started parent process [%s]: %d workers, one SO_REUSEPORT socket each on %s:%s",
            os.getpid(),
            self.settings.workers,
            self.settings.host,
            self.settings.port,
        )
        self._install_signal_handlers()
        self.workers = [self._spawn(index) for index in range(self.settings.workers)]
        while not self.should_exit.wait(SUPERVISOR_TICK_SECONDS):
            self._handle_signals()
            self._keep_workers_alive()
        for worker in self.workers:
            worker.terminate()
        for worker in self.workers:
            worker.join()
        LOGGER.info("Stopping parent process [%s]", os.getpid())
        return STARTUP_FAILURE if self.startup_failed else 0

    def _handle_signals(self) -> None:
        while self._signal_queue:
            signum = self._signal_queue.pop(0)
            name = _SIGNALS.get(signal.Signals(signum), "")
            if name in {"int", "term"}:
                LOGGER.info("Received SIG%s, exiting.", name.upper())
                self.should_exit.set()
            elif name == "hup":
                LOGGER.info("Received SIGHUP, restarting workers one at a time.")
                self._restart_all()

    def _keep_workers_alive(self) -> None:
        if self.should_exit.is_set():
            return
        for index, worker in enumerate(self.workers):
            if worker.is_alive(timeout=self.settings.timeout_worker_healthcheck):
                continue
            worker.kill()
            worker.join()
            if worker.exitcode == STARTUP_FAILURE:
                # The bind, the import, the factory or the lifespan start-up
                # is broken and would fail the same way on every restart; stop
                # the parent so the Pod restarts (and backs off) instead of
                # crash-looping one child.
                LOGGER.error(
                    "Child process [%s] failed to start; stopping the parent.",
                    worker.pid,
                )
                self.startup_failed = True
                self.should_exit.set()
                return
            if self.should_exit.is_set():
                return
            LOGGER.info(
                "Child process [%s] died (exit %s)", worker.pid, worker.exitcode
            )
            self.workers[index] = self._spawn(index)

    def _restart_all(self) -> None:
        """Replace each worker, the new one serving before the old one stops."""

        for index, old in enumerate(self.workers):
            if self.should_exit.is_set():
                return
            new = self._spawn(index)
            if not new.wait_until_ready(
                self.settings.timeout_worker_healthcheck, self.should_exit
            ):
                new.kill()
                new.join()
                if not self.should_exit.is_set():
                    LOGGER.error(
                        "New child process [%s] was not ready in time; keeping "
                        "worker [%s] and aborting the restart.",
                        new.pid,
                        old.pid,
                    )
                return
            old.terminate()
            old.join()
            self.workers[index] = new


def main(argv: Sequence[str] | None = None) -> int:
    settings = parse_args(argv)
    # Building a Config configures uvicorn's loggers; the supervisor's own
    # lines would otherwise go nowhere until the first worker is spawned.
    settings.uvicorn_config()
    if settings.workers == 1:
        # In-process, as uvicorn runs a single worker; still on a socket of
        # its own so the metrics read the same for every topology.
        return 0 if serve_worker(settings, 0) else STARTUP_FAILURE
    return Supervisor(settings).run()


if __name__ == "__main__":
    sys.exit(main())
