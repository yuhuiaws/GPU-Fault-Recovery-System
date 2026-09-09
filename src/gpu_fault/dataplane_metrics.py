"""A shared ``/metrics`` + ``/healthz`` server for data-plane poll loops.

A component declares one :class:`MetricFamily` (prefix + counters, gauges,
timestamps) and moves values; this module owns the text exposition, following
``completion_metrics_server``: HELP and TYPE for every series, timestamps as
integer Unix seconds with ``0`` for "never", port ``0`` disabled, a bind failure
one ERROR line -- the owning loop keeps running. ``/healthz`` answers a
caller-supplied predicate (200 / 503) so liveness can leave the breadcrumb file
later; a predicate that raises reports healthy. Standard library only.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections.abc import Callable, Sequence
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

LOGGER = logging.getLogger(__name__)

BIND_ADDRESS = "0.0.0.0"
_PREFIX = re.compile(r"^gpu_fault_[a-z][a-z0-9_]*_$")
_SUFFIX = re.compile(r"^[a-z][a-z0-9_]*$")
MetricSpec = tuple[str, str]  # (name without the family prefix, HELP text)
HealthPredicate = Callable[[], bool]


class MetricFamily:
    """One component's series under one prefix, safe to move from any thread.

    Counters only go up, gauges are set, timestamps are marked (``datetime``,
    epoch seconds or ``None`` = now). Every name is checked: a typo fails the
    first test pass instead of rendering a series nobody alerts on.
    """

    def __init__(
        self,
        prefix: str,
        *,
        counters: Sequence[MetricSpec],
        gauges: Sequence[MetricSpec] = (),
        timestamps: Sequence[MetricSpec] = (),
    ) -> None:
        if not _PREFIX.match(prefix):
            raise ValueError(f"metric prefix {prefix!r} must match {_PREFIX.pattern}")
        self.prefix = prefix
        kinds = (("counter", counters), ("gauge", gauges), ("timestamp", timestamps))
        self._specs = tuple((k, n, h) for k, specs in kinds for n, h in specs)
        self._kinds: dict[str, str] = {}
        for kind, name, _help in self._specs:
            if not _SUFFIX.match(name) or name in self._kinds:
                raise ValueError(f"metric name {name!r} is invalid or repeated")
            self._kinds[name] = kind
        self._values: dict[str, float] = dict.fromkeys(self._kinds, 0.0)
        self._lock = threading.Lock()

    def names(self) -> tuple[str, ...]:
        return tuple(self.prefix + name for _kind, name, _help in self._specs)

    def _check(self, name: str, kind: str) -> None:
        if self._kinds.get(name) != kind:
            raise KeyError(f"{name!r} is not a {kind} of {self.prefix}")

    def inc(self, name: str, amount: int = 1) -> None:
        self._check(name, "counter")
        if amount < 0:
            raise ValueError(f"counter {name!r} cannot go down (amount {amount})")
        with self._lock:
            self._values[name] += amount

    def set(self, name: str, value: float) -> None:
        self._check(name, "gauge")
        with self._lock:
            self._values[name] = float(value)

    def mark(self, name: str, when: datetime | float | None = None) -> None:
        self._check(name, "timestamp")
        if when is None:
            when = time.time()
        elif isinstance(when, datetime):
            when = when.timestamp()
        with self._lock:
            self._values[name] = float(when)

    def render(self) -> str:  # HELP, TYPE and one sample per series
        with self._lock:
            values = dict(self._values)
        lines: list[str] = []
        for kind, name, help_text in self._specs:
            full, value = self.prefix + name, values[name]
            sample = int(value) if value.is_integer() else repr(value)
            lines.append(f"# HELP {full} {help_text}")
            lines.append(f"# TYPE {full} {'gauge' if kind == 'timestamp' else kind}")
            lines.append(f"{full} {sample}")
        return "\n".join(lines) + "\n"


class _HTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    family: MetricFamily
    health: HealthPredicate | None


class _Handler(BaseHTTPRequestHandler):
    server: _HTTPServer

    def do_GET(self) -> None:  # noqa: N802 - http.server naming
        path = self.path.split("?", 1)[0]
        if path == "/metrics":
            self._respond(200, self.server.family.render())
        elif path == "/healthz":
            self._respond(*self._health())
        else:
            self.send_error(404, "Not Found")

    def _health(self) -> tuple[int, str]:
        predicate = self.server.health
        if predicate is None:
            return 200, "ok\n"
        try:
            healthy = bool(predicate())
        except Exception:
            LOGGER.exception("%s health check raised", self.server.family.prefix)
            return 200, "health unknown: the check itself failed\n"
        return (200, "ok\n") if healthy else (503, "unhealthy\n")

    def _respond(self, status: int, body_text: str) -> None:
        body = body_text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        LOGGER.debug("metrics request: " + format, *args)


class MetricsServer:
    """Serve one family's ``/metrics`` and ``/healthz`` on a daemon thread."""

    def __init__(
        self, family: MetricFamily, *, port: int, health: HealthPredicate | None = None
    ) -> None:
        self.family, self.health, self.requested_port = family, health, port
        self._server: _HTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:  # the bound port; differs from a requested 0
        if self._server is None:
            return self.requested_port
        return int(self._server.server_address[1])

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:  # raises OSError when the bind fails
        server = _HTTPServer((BIND_ADDRESS, self.requested_port), _Handler)
        server.family, server.health = self.family, self.health
        thread = threading.Thread(
            target=server.serve_forever, name="gpu-fault-dataplane-metrics", daemon=True
        )
        self._server, self._thread = server, thread
        thread.start()
        LOGGER.info("%s metrics listening on port %s", self.family.prefix, self.port)

    def stop(self) -> None:
        server, thread = self._server, self._thread
        if server is None:
            return
        server.shutdown()
        server.server_close()
        if thread is not None:
            thread.join(timeout=5)
        self._server = self._thread = None


def start_metrics_server(
    family: MetricFamily, *, port: int, health: HealthPredicate | None = None
) -> MetricsServer | None:
    """Best-effort start; ``None`` (and one log line) when ``port`` is 0 or the
    bind failed. The owning loop must keep running whatever happens here."""
    prefix = family.prefix
    if port <= 0:
        LOGGER.info("%s metrics disabled (port %s)", prefix, port)
        return None
    if port > 65535:
        # socket.bind raises OverflowError, not OSError, for this; a typo in
        # the port variable must not take the component down at start-up.
        LOGGER.error("%s metrics disabled: port %s is out of range", prefix, port)
        return None
    server = MetricsServer(family, port=port, health=health)
    try:
        server.start()
    except (OSError, OverflowError) as exc:
        LOGGER.error("%s metrics disabled: cannot bind port %s (%s)", prefix, port, exc)
        return None
    return server
