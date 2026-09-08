"""``/metrics`` and ``/healthz`` for the standalone Completion Watcher.

The completion controller is a watch/poll loop with no HTTP server of its own,
so the counters it keeps (``reconcile_failures_total``,
``evicted_attempts_total``, ``restore_skipped_total``,
``outbox_append_failures_total``, ``reconcile_runs_total``,
``metadata_takeovers_total``, ``resumed_attempts_total``), the outbox depth
gauges (``outbox_depth``, ``outbox_quarantined_depth``) and the timestamp of
the last completed full
reconcile pass were only reachable from a debugger. This module exposes them in
Prometheus text exposition on a daemon thread so the Pod can carry the same
``prometheus.io/scrape`` annotations as every other GPU-fault workload.

``/healthz`` turns the loop's *progress* timestamp into the Pod's liveness
signal (F3). A watch stream the API server drops without an RST used to park
the only thread that relists Pods: no relist, no reconcile, no observations,
and ten minutes later every node reads UNKNOWN so every node-mutating plan is
BLOCKED. The probe fails only when the loop has finished nothing at all for a
whole delivery budget (``progress_stall_budget_seconds``). It stays green for
an idle cluster, which posts nothing yet relists every watch timeout, for a
slow pass that keeps finishing attempts, and for an API-server outage the loop
keeps retrying -- none of those is a stuck loop, and restarting the single
Completion Watcher there would only add a cold start to an incident.

Nothing here may raise into the watch loop, and an unreadable controller
reports healthy. But the server is no longer optional: the liveness probe reads
this endpoint, so a port that cannot be bound (or ``PORT=0``) now means kubelet
keeps killing the Pod instead of running it without metrics. Both are logged at
ERROR/INFO and the loop still starts, which is what makes the failure visible
in the Pod's restart count rather than silent.
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
# The controller derives the real window from its delivery timeouts and
# publishes it as ``progress_stall_budget_seconds``. These two only cover a
# controller that does not expose it: a stuck watch has also missed three
# relists, and the fallback stays a multiple of the watch timeout so an
# operator who raises GPU_FAULT_WATCHER_WATCH_TIMEOUT_SECONDS cannot turn the
# probe into a restart loop.
HEALTH_STALE_CYCLE_MULTIPLIER = 3
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
        "gpu_fault_completion_controller_resumed_attempts_total",
        "resumed_attempts_total",
        "Attempts whose missing-Pod tombstone was withdrawn because Pods of "
        "that attempt-id were listed again. Anything above zero means a STOPPED "
        "terminal was published for an attempt that was still alive.",
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

# (metric name, controller attribute, help text) for the two timestamps.
TIMESTAMPS: tuple[tuple[str, str, str], ...] = (
    (
        "gpu_fault_completion_watcher_last_cycle_completed_timestamp",
        "last_cycle_completed_at",
        "Unix seconds at which the last *full* reconcile pass completed; 0 "
        "before the first one. Only a pass that relisted every Pod moves it. "
        "This is the value humans alert on -- it can legitimately lag by more "
        "than one watch timeout, so it is not what /healthz judges.",
    ),
    (
        "gpu_fault_completion_watcher_last_progress_timestamp",
        "last_progress_at",
        "Unix seconds at which the loop last finished a step: a relist, a "
        "watch event, one attempt's reconcile, or a failed cycle going back "
        "to its retry sleep. This is the value /healthz and the liveness "
        "probe read, so time() - this is what kubelet acts on.",
    ),
)


def metrics_port_from_environment() -> int:
    """The TCP port to serve on.

    ``0`` disables the server, which is only safe where nothing probes it: the
    shipped Deployment points its startup, liveness and readiness probes at
    ``/healthz``, so disabling the server there means kubelet keeps killing the
    Pod. ``tests/regional/test_production_safety_config.py`` pins that.
    """

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
    for name, attribute, help_text in TIMESTAMPS:
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} gauge")
        lines.append(f"{name} {_epoch_seconds(getattr(controller, attribute, None))}")
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


def _stall_budget_seconds(controller: Any) -> float:
    """The liveness window: the controller's own budget, or three relists.

    The controller derives it from the delivery timeouts it actually uses, so
    the probe cannot drift away from what one legal blocking step costs. The
    fallback exists for a controller (or a test double) that predates it.
    """

    budget = getattr(controller, "progress_stall_budget_seconds", None)
    try:
        if budget is not None and float(budget) > 0:
            return float(budget)
    except (TypeError, ValueError):
        pass
    watch_timeout = DEFAULT_WATCH_TIMEOUT_SECONDS
    configured = getattr(controller, "watch_timeout_seconds", None)
    try:
        if configured is not None and float(configured) > 0:
            watch_timeout = float(configured)
    except (TypeError, ValueError):
        pass
    return HEALTH_STALE_CYCLE_MULTIPLIER * watch_timeout


def evaluate_completion_health(controller: Any) -> tuple[int, str]:
    """``(status, body)`` for ``/healthz``: 503 only when the loop is stuck.

    Progress is the signal, never traffic and never pass completion: a cluster
    with no managed Pod posts nothing, a fault storm can spend minutes inside
    one pass, and an API-server outage can fail every list -- all three are the
    loop working, and all three keep this at 200. What the probe catches is a
    step that never returns (a watch stream the API server dropped without an
    RST, a parked LIST), which finishes nothing and so stops the clock.

    Cold start is *not* handled here: the Deployment's ``startupProbe`` owns
    it, so this stays a single criterion.

    An unreadable controller (a clock that is not a datetime, a missing
    attribute) reports healthy: a bug in this evaluator must not restart a
    working watcher.
    """

    now = _controller_now(controller)
    if now is None:
        return 200, "health unknown: the controller exposes no usable clock\n"
    last_progress = getattr(controller, "last_progress_at", None)
    if not isinstance(last_progress, datetime):
        return 200, "health unknown: last_progress_at is not a timestamp\n"

    limit = _stall_budget_seconds(controller)
    age = (now - last_progress).total_seconds()
    if age <= limit:
        return (
            200,
            f"ok: the loop last made progress {age:.1f}s ago (limit {limit:.1f}s)\n",
        )
    return (
        503,
        f"stuck: the loop has finished nothing for {age:.1f}s, over its "
        f"{limit:.1f}s budget\n",
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
