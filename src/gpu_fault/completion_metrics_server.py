"""A minimal ``/metrics`` endpoint for the standalone Completion Watcher.

The completion controller is a watch/poll loop with no HTTP server of its own,
so the counters it keeps (``reconcile_failures_total``,
``evicted_attempts_total``, ``restore_skipped_total``) were only reachable from
a debugger. This module exposes them in Prometheus text exposition on a
daemon thread so the Pod can carry the same ``prometheus.io/scrape``
annotations as every other GPU-fault workload.

The server is strictly best-effort: a port that cannot be bound is logged at
ERROR and the controller keeps running without metrics. Nothing here may
raise into the watch loop.
"""

from __future__ import annotations

import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

LOGGER = logging.getLogger(__name__)

METRICS_PORT_ENV = "GPU_FAULT_COMPLETION_WATCHER_METRICS_PORT"
DEFAULT_METRICS_PORT = 9109
BIND_ADDRESS = "0.0.0.0"

# (metric name, controller attribute, help text). Every entry is a counter that
# only ever increases for the lifetime of the process; a restart resets it,
# which is what Prometheus expects of the type.
COUNTERS: tuple[tuple[str, str, str], ...] = (
    (
        "gpu_fault_completion_controller_reconcile_failures_total",
        "reconcile_failures_total",
        "Attempts whose reconcile raised and was logged instead of aborting "
        "the pass for every other attempt.",
    ),
    (
        "gpu_fault_completion_controller_evicted_attempts_total",
        "evicted_attempts_total",
        "Attempts whose controller-side state was dropped because the watcher "
        "core pruned them and no Pod of theirs is left.",
    ),
    (
        "gpu_fault_completion_controller_restore_skipped_total",
        "restore_skipped_total",
        "Persisted attempt records skipped at start-up because they could not "
        "be restored.",
    ),
)


def metrics_port_from_environment() -> int:
    """The TCP port to serve on; ``0`` disables the server entirely."""

    port = int(os.getenv("GPU_FAULT_COMPLETION_WATCHER_METRICS_PORT", "9109"))
    if port < 0:
        raise ValueError(f"{METRICS_PORT_ENV} must be 0 (disabled) or a TCP port")
    return port


def render_completion_metrics(controller: Any) -> str:
    """Prometheus text exposition of the controller's counters."""

    lines: list[str] = []
    for name, attribute, help_text in COUNTERS:
        value = getattr(controller, attribute, 0)
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} counter")
        lines.append(f"{name} {int(value)}")
    return "\n".join(lines) + "\n"


class _MetricsHandler(BaseHTTPRequestHandler):
    server: CompletionMetricsHTTPServer

    def do_GET(self) -> None:  # noqa: N802 - http.server naming
        if self.path.split("?", 1)[0] != "/metrics":
            self.send_error(404, "Not Found")
            return
        body = render_completion_metrics(self.server.controller).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        # Scrapes every few seconds would otherwise flood the watcher's log.
        LOGGER.debug("metrics request: " + format, *args)


class CompletionMetricsHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], controller: Any) -> None:
        self.controller = controller
        super().__init__(address, _MetricsHandler)


class CompletionMetricsServer:
    """Serve ``/metrics`` for one controller on a daemon thread."""

    def __init__(self, controller: Any, *, port: int) -> None:
        self.controller = controller
        self.requested_port = port
        self._server: CompletionMetricsHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        """The bound port (differs from the request when it was ``0``)."""

        if self._server is None:
            return self.requested_port
        return int(self._server.server_address[1])

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def daemon_thread(self) -> bool:
        return self._thread is not None and self._thread.daemon

    def start(self) -> None:
        """Bind and start serving; raises ``OSError`` when the bind fails."""

        server = CompletionMetricsHTTPServer(
            (BIND_ADDRESS, self.requested_port), self.controller
        )
        thread = threading.Thread(
            target=server.serve_forever,
            name="gpu-fault-completion-metrics",
            daemon=True,
        )
        self._server = server
        self._thread = thread
        thread.start()
        LOGGER.info("completion watcher metrics listening on port %s", self.port)

    def stop(self) -> None:
        server, thread = self._server, self._thread
        if server is None:
            return
        server.shutdown()
        server.server_close()
        if thread is not None:
            thread.join(timeout=5)
        self._server = None
        self._thread = None


def start_completion_metrics_server(
    controller: Any, *, port: int | None = None
) -> CompletionMetricsServer | None:
    """Best-effort start; ``None`` when disabled or when the bind failed.

    The controller must keep running whatever happens here, so every failure
    is logged and swallowed. ``port`` defaults to the environment setting.
    """

    try:
        if port is None:
            port = metrics_port_from_environment()
    except ValueError:
        LOGGER.exception("completion watcher metrics disabled: bad port setting")
        return None
    if port == 0:
        LOGGER.info("completion watcher metrics disabled (%s=0)", METRICS_PORT_ENV)
        return None
    server = CompletionMetricsServer(controller, port=port)
    try:
        server.start()
    except OSError as exc:
        LOGGER.error(
            "completion watcher metrics disabled: cannot bind %s:%s (%s)",
            BIND_ADDRESS,
            port,
            exc,
        )
        return None
    return server
