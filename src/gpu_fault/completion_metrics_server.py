"""``/metrics`` and ``/healthz`` for the standalone Completion Watcher.

The completion controller is a watch/poll loop with no HTTP server of its own,
so the counters it keeps (``reconcile_failures_total``,
``evicted_attempts_total``, ``restore_skipped_total``,
``outbox_append_failures_total``, ``reconcile_runs_total``,
``metadata_takeovers_total``), the outbox depth gauges (``outbox_depth``,
``outbox_quarantined_depth``) and the timestamp of the last completed full
reconcile pass were only reachable from a debugger. This module exposes them in
Prometheus text exposition on a daemon thread so the Pod can carry the same
``prometheus.io/scrape`` annotations as every other GPU-fault workload.

``/healthz`` turns that last value into the Pod's liveness signal (F3). A watch
stream the API server drops without an RST used to park the only thread that
relists Pods: no relist, no reconcile, no observations, and ten minutes later
every node reads UNKNOWN so every node-mutating plan is BLOCKED. The probe
therefore fails only on a stuck *loop* -- three watch timeouts with no
completed full pass -- never on an idle cluster, which posts nothing at all yet
keeps completing a pass on every relist.

The server is strictly best-effort: a port that cannot be bound is logged at
ERROR and the controller keeps running without metrics. Nothing here may
raise into the watch loop.
"""

from __future__ import annotations

import logging
import os
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

LOGGER = logging.getLogger(__name__)

METRICS_PORT_ENV = "GPU_FAULT_COMPLETION_WATCHER_METRICS_PORT"
DEFAULT_METRICS_PORT = 9109
BIND_ADDRESS = "0.0.0.0"
METRICS_PATH = "/metrics"
HEALTH_PATH = "/healthz"
# A stuck watch is one that has missed three relists. The window has to be a
# multiple of the watch timeout, not a constant: an operator who raises
# GPU_FAULT_WATCHER_WATCH_TIMEOUT_SECONDS must not turn the probe into a
# restart loop.
HEALTH_STALE_CYCLE_MULTIPLIER = 3
# Before the first pass completes there is nothing to compare against, so the
# probe grants exactly one watch timeout of grace from process start: enough
# for the initial list plus a full reconcile of every attempt on a large
# cluster, short enough that a list that never returns is caught.
HEALTH_STARTUP_GRACE_CYCLES = 1
DEFAULT_WATCH_TIMEOUT_SECONDS = 30.0

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
    (
        "gpu_fault_completion_controller_reconcile_runs_total",
        "reconcile_runs_total",
        "Reconcile passes the loop has completed, full and debounced. A flat "
        "line for longer than the watch timeout means the loop is not turning.",
    ),
    (
        "gpu_fault_completion_controller_metadata_takeovers_total",
        "metadata_takeovers_total",
        "Attempts whose spec was re-derived because their Pods disagreed and "
        "then agreed again on a new generation.",
    ),
    (
        "gpu_fault_completion_outbox_append_failures_total",
        "outbox_append_failures_total",
        "Critical completion events whose write-ahead ConfigMap copy could "
        "not be written or cleared; delivery went ahead anyway, so any value "
        "above zero means a restart can lose an event or replay a delivered "
        "one.",
    ),
)

# (metric name, controller attribute, help text) for values that go up and
# down. Both are refreshed by the outbox replay that runs at the top of every
# reconcile pass, so they lag a newly buffered record by at most one pass.
GAUGES: tuple[tuple[str, str, str], ...] = (
    (
        "gpu_fault_completion_outbox_depth",
        "outbox_depth",
        "Completion records buffered in the write-ahead ConfigMap: critical "
        "events the control plane has not accepted yet, plus the latest "
        "undelivered workload observation per attempt.",
    ),
    (
        "gpu_fault_completion_outbox_quarantined_depth",
        "outbox_quarantined_depth",
        "Buffered records the replay has given up on: it skips them for ever, "
        "so they are delivered only by a later live POST or by an operator "
        "replay. Anything above zero needs a look.",
    ),
)

LAST_CYCLE_METRIC = "gpu_fault_completion_watcher_last_cycle_completed_timestamp"
LAST_CYCLE_HELP = (
    "Unix seconds at which the last *full* reconcile pass completed; 0 before "
    "the first one. Only a pass that relisted every Pod moves it, so "
    "time() - this is the watch loop's staleness even on a cluster with no "
    "managed job. This is the value /healthz and the liveness probe read."
)


def metrics_port_from_environment() -> int:
    """The TCP port to serve on; ``0`` disables the server entirely."""

    port = int(os.getenv("GPU_FAULT_COMPLETION_WATCHER_METRICS_PORT", "9109"))
    if port < 0:
        raise ValueError(f"{METRICS_PORT_ENV} must be 0 (disabled) or a TCP port")
    return port


def _epoch_seconds(value: Any) -> int:
    """``datetime`` (or epoch number, or ``None``) as integer Unix seconds.

    ``0`` means "no full pass has completed yet", which is what Prometheus
    readers expect of an absent timestamp. Nothing here may raise: a scrape
    must never depend on the shape of a controller attribute.
    """

    if value is None:
        return 0
    if isinstance(value, datetime):
        return int(value.timestamp())
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def render_completion_metrics(controller: Any) -> str:
    """Prometheus text exposition of the controller's counters."""

    lines: list[str] = []
    for metrics, metric_type in ((COUNTERS, "counter"), (GAUGES, "gauge")):
        for name, attribute, help_text in metrics:
            value = getattr(controller, attribute, 0)
            lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} {metric_type}")
            lines.append(f"{name} {int(value)}")
    lines.append(f"# HELP {LAST_CYCLE_METRIC} {LAST_CYCLE_HELP}")
    lines.append(f"# TYPE {LAST_CYCLE_METRIC} gauge")
    lines.append(
        f"{LAST_CYCLE_METRIC} "
        f"{_epoch_seconds(getattr(controller, 'last_cycle_completed_at', None))}"
    )
    return "\n".join(lines) + "\n"


def _controller_now(controller: Any) -> datetime | None:
    """The controller's own clock, so a test clock drives the health check."""

    clock = getattr(controller, "now", None)
    if callable(clock):
        value = clock()
        if isinstance(value, datetime):
            return value
        return None
    return datetime.now(timezone.utc)


def evaluate_completion_health(controller: Any) -> tuple[int, str]:
    """``(status, body)`` for ``/healthz``: 503 only when the loop is stuck.

    The reconcile pass is the liveness signal, never the traffic: a cluster
    with no managed Pod posts nothing at all and must stay healthy, while a
    watch stream that hangs stops the 30 s relist and is the one failure this
    probe exists to end. Before the first pass the check falls back to a single
    watch timeout of startup grace.

    An unreadable controller (a clock that is not a datetime, a missing
    attribute) reports healthy: a bug in this evaluator must not restart a
    working watcher.
    """

    watch_timeout = DEFAULT_WATCH_TIMEOUT_SECONDS
    configured = getattr(controller, "watch_timeout_seconds", None)
    try:
        if configured is not None and float(configured) > 0:
            watch_timeout = float(configured)
    except (TypeError, ValueError):
        pass

    now = _controller_now(controller)
    last_cycle = getattr(controller, "last_cycle_completed_at", None)
    started_at = getattr(controller, "started_at", None)
    if now is None:
        return 200, "health unknown: the controller exposes no usable clock\n"

    if last_cycle is None:
        if not isinstance(started_at, datetime):
            return 200, "health unknown: no start time and no completed pass\n"
        grace = HEALTH_STARTUP_GRACE_CYCLES * watch_timeout
        age = (now - started_at).total_seconds()
        if age <= grace:
            return 200, f"starting: {age:.1f}s of {grace:.1f}s startup grace used\n"
        return (
            503,
            f"stuck: no reconcile pass completed {age:.1f}s after start "
            f"(grace {grace:.1f}s)\n",
        )

    if not isinstance(last_cycle, datetime):
        return 200, "health unknown: last_cycle_completed_at is not a timestamp\n"
    limit = HEALTH_STALE_CYCLE_MULTIPLIER * watch_timeout
    age = (now - last_cycle).total_seconds()
    if age <= limit:
        return (
            200,
            f"ok: last full reconcile pass {age:.1f}s ago (limit {limit:.1f}s)\n",
        )
    return (
        503,
        f"stuck: last full reconcile pass {age:.1f}s ago, over the "
        f"{limit:.1f}s limit of {HEALTH_STALE_CYCLE_MULTIPLIER} watch timeouts\n",
    )


class _MetricsHandler(BaseHTTPRequestHandler):
    server: CompletionMetricsHTTPServer

    def do_GET(self) -> None:  # noqa: N802 - http.server naming
        path = self.path.split("?", 1)[0]
        if path == METRICS_PATH:
            self._respond(200, render_completion_metrics(self.server.controller))
            return
        if path == HEALTH_PATH:
            # Reads two timestamps off the controller and nothing else: the
            # probe never takes the reconcile lock, calls the API server or
            # touches the outbox, so a scrape cannot slow the loop it watches.
            try:
                status, body = evaluate_completion_health(self.server.controller)
            except Exception:
                LOGGER.exception("completion watcher health check failed")
                status, body = 200, "health unknown: the check itself failed\n"
            self._respond(status, body)
            return
        self.send_error(404, "Not Found")

    def _respond(self, status: int, body_text: str) -> None:
        body = body_text.encode("utf-8")
        self.send_response(status)
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
