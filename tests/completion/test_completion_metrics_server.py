"""The completion watcher exports its counters over a minimal ``/metrics``.

The controller is a standalone watch loop with no HTTP surface, so its
``reconcile_failures_total`` / ``evicted_attempts_total`` /
``restore_skipped_total`` counters were only ever visible in a debugger. The
server here must never take the controller down with it: a port that is
already bound is logged and the watch loop keeps running without metrics.
"""

from __future__ import annotations

import logging
import socket
from datetime import datetime, timedelta, timezone
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from gpu_fault import completion_metrics_server as metrics_module
from gpu_fault.completion_metrics_server import (
    CompletionMetricsServer,
    metrics_port_from_environment,
    render_completion_metrics,
    start_completion_metrics_server,
)

NOW = datetime(2026, 9, 8, 12, 0, tzinfo=timezone.utc)


class FakeController:
    reconcile_failures_total = 3
    evicted_attempts_total = 5
    restore_skipped_total = 7
    outbox_append_failures_total = 9
    outbox_depth = 4
    outbox_quarantined_depth = 2
    reconcile_runs_total = 21
    metadata_takeovers_total = 6
    resumed_attempts_total = 2
    watch_timeout_seconds = 30
    # The value the shipped Deployment derives (120 s receipt poll + 4 x 10 s
    # HTTP + 3 x 30 s Retry-After + 60 s margin).
    progress_stall_budget_seconds = 310.0

    def __init__(
        self,
        *,
        cycle_age_seconds: float | None = 5.0,
        progress_age_seconds: float | None = 5.0,
        uptime_seconds: float = 12.0,
    ) -> None:
        self.started_at = NOW - timedelta(seconds=uptime_seconds)
        self.last_cycle_completed_at = (
            None
            if cycle_age_seconds is None
            else NOW - timedelta(seconds=cycle_age_seconds)
        )
        self.last_progress_at = (
            None
            if progress_age_seconds is None
            else NOW - timedelta(seconds=progress_age_seconds)
        )

    def now(self) -> datetime:
        """The controller's injected clock; the health check reads it."""

        return NOW


def _ephemeral(controller: object) -> CompletionMetricsServer:
    """Bind port 0 directly: the start helper treats 0 as *disabled*."""

    server = CompletionMetricsServer(controller, port=0)
    server.start()
    return server


def _get(url: str) -> tuple[int, str, str]:
    with urlopen(url, timeout=5) as response:
        return (
            response.status,
            response.headers.get("Content-Type", ""),
            response.read().decode("utf-8"),
        )


def _status(url: str) -> tuple[int, str]:
    """The status code of a GET, treating 5xx as a value rather than a raise."""

    try:
        status, _content_type, body = _get(url)
    except HTTPError as error:
        return error.code, error.read().decode("utf-8")
    return status, body


def test_metrics_endpoint_exposes_the_three_controller_counters() -> None:
    server = _ephemeral(FakeController())
    try:
        status, content_type, body = _get(f"http://127.0.0.1:{server.port}/metrics")
    finally:
        server.stop()

    assert status == 200, body
    assert content_type.startswith("text/plain"), content_type
    lines = body.splitlines()
    for name, value in (
        ("gpu_fault_completion_controller_reconcile_failures_total", 3),
        ("gpu_fault_completion_controller_evicted_attempts_total", 5),
        ("gpu_fault_completion_controller_restore_skipped_total", 7),
    ):
        assert f"# HELP {name} " in body, f"{name} has no HELP line"
        assert f"# TYPE {name} counter" in lines, f"{name} is not typed as a counter"
        assert f"{name} {value}" in lines, f"{name} sample missing from {body!r}"


def test_metrics_endpoint_exposes_the_outbox_write_ahead_failure_counter() -> None:
    """F1(b)/F12: a WAL write that fails no longer vetoes delivery, so the
    only way an operator learns the outbox is unwritable is this counter."""

    server = _ephemeral(FakeController())
    try:
        _, _, body = _get(f"http://127.0.0.1:{server.port}/metrics")
    finally:
        server.stop()

    name = "gpu_fault_completion_outbox_append_failures_total"
    lines = body.splitlines()
    assert f"# HELP {name} " in body, f"{name} has no HELP line: {body!r}"
    assert f"# TYPE {name} counter" in lines, f"{name} is not typed as a counter"
    assert f"{name} 9" in lines, f"{name} sample missing from {body!r}"


def test_metrics_endpoint_exposes_the_outbox_depth_gauges() -> None:
    """A quarantined record is one nobody replays: it has to be visible.

    ``replay`` skips quarantined records for ever and no production caller
    passes ``include_quarantined``, so a non-zero quarantined depth is an
    operator action item, not a statistic.
    """

    server = _ephemeral(FakeController())
    try:
        _, _, body = _get(f"http://127.0.0.1:{server.port}/metrics")
    finally:
        server.stop()

    lines = body.splitlines()
    for name, value in (
        ("gpu_fault_completion_outbox_depth", 4),
        ("gpu_fault_completion_outbox_quarantined_depth", 2),
    ):
        assert f"# HELP {name} " in body, f"{name} has no HELP line: {body!r}"
        assert f"# TYPE {name} gauge" in lines, f"{name} is not typed as a gauge"
        assert f"{name} {value}" in lines, f"{name} sample missing from {body!r}"


def test_metrics_endpoint_reads_live_counter_values_on_each_scrape() -> None:
    controller = FakeController()
    server = _ephemeral(controller)
    try:
        controller.reconcile_failures_total = 11
        _, _, body = _get(f"http://127.0.0.1:{server.port}/metrics")
    finally:
        server.stop()

    assert "gpu_fault_completion_controller_reconcile_failures_total 11" in body, body


def test_unknown_paths_return_404() -> None:
    server = _ephemeral(FakeController())
    try:
        with pytest.raises(HTTPError) as raised:
            _get(f"http://127.0.0.1:{server.port}/other")
    finally:
        server.stop()

    assert raised.value.code == 404, raised.value.code


def test_stop_releases_the_port() -> None:
    server = _ephemeral(FakeController())
    port = server.port
    server.stop()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("0.0.0.0", port))
    assert not server.is_running, "server thread is still alive after stop()"


def test_port_zero_disables_the_server(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GPU_FAULT_COMPLETION_WATCHER_METRICS_PORT", "0")

    assert metrics_port_from_environment() == 0, "0 must be read as disabled"
    assert start_completion_metrics_server(FakeController()) is None, (
        "a disabled port must not start a server"
    )
    assert start_completion_metrics_server(FakeController(), port=0) is None, (
        "an explicit port 0 must not start a server either"
    )


def test_bind_failure_is_logged_and_does_not_raise(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
        occupied.bind(("0.0.0.0", 0))
        occupied.listen(1)
        port = occupied.getsockname()[1]
        with caplog.at_level(logging.ERROR, logger=metrics_module.LOGGER.name):
            server = start_completion_metrics_server(FakeController(), port=port)

    assert server is None, "a bind failure must return None, not a half-started server"
    errors = [record for record in caplog.records if record.levelno == logging.ERROR]
    assert errors, "bind failure must be logged at ERROR"
    assert str(port) in errors[0].getMessage(), errors[0].getMessage()


def test_default_port_is_9109(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GPU_FAULT_COMPLETION_WATCHER_METRICS_PORT", raising=False)

    assert metrics_port_from_environment() == 9109, "default must be 9109"


@pytest.mark.parametrize("value", ["-1", "-9109"])
def test_negative_port_is_rejected(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("GPU_FAULT_COMPLETION_WATCHER_METRICS_PORT", value)

    with pytest.raises(ValueError, match="GPU_FAULT_COMPLETION_WATCHER_METRICS_PORT"):
        metrics_port_from_environment()


def test_render_handles_a_controller_without_restore_counter() -> None:
    class Bare:
        reconcile_failures_total = 1
        evicted_attempts_total = 2

    body = render_completion_metrics(Bare())

    assert "gpu_fault_completion_controller_restore_skipped_total 0" in body, body
    assert "gpu_fault_completion_watcher_last_cycle_completed_timestamp 0" in body, (
        f"a controller with no completed pass must report 0, not raise: {body!r}"
    )
    assert "gpu_fault_completion_watcher_last_progress_timestamp 0" in body, (
        f"a controller with no progress stamp must report 0, not raise: {body!r}"
    )


def test_metrics_export_cycle_age_and_outbox_stats() -> None:
    """F3/F12: only a completed full pass moves this timestamp.

    A watch stream that hangs for ever kept every counter frozen and exposed
    nothing an alert could read, so ten minutes of silence looked identical to
    an idle cluster.
    """
    controller = FakeController(cycle_age_seconds=5.0)
    server = _ephemeral(controller)
    try:
        _, _, body = _get(f"http://127.0.0.1:{server.port}/metrics")
    finally:
        server.stop()

    lines = body.splitlines()
    stamp = "gpu_fault_completion_watcher_last_cycle_completed_timestamp"
    assert f"# TYPE {stamp} gauge" in lines, (
        f"{stamp} is not typed as a gauge: {body!r}"
    )
    expected = int((NOW - timedelta(seconds=5)).timestamp())
    assert f"{stamp} {expected}" in lines, f"{stamp} sample missing from {body!r}"
    progress = "gpu_fault_completion_watcher_last_progress_timestamp"
    assert f"# TYPE {progress} gauge" in lines, (
        f"{progress} is not typed as a gauge: {body!r}"
    )
    assert f"{progress} {expected}" in lines, (
        f"the value /healthz judges must be scrapable too: {body!r}"
    )
    for name, value in (
        ("gpu_fault_completion_controller_reconcile_runs_total", 21),
        ("gpu_fault_completion_controller_metadata_takeovers_total", 6),
        # A withdrawn missing-Pod tombstone means a STOPPED terminal was
        # published for an attempt that was still alive (F2), so it has to be
        # scrapable, not just logged.
        ("gpu_fault_completion_controller_resumed_attempts_total", 2),
    ):
        assert f"# HELP {name} " in body, f"{name} has no HELP line: {body!r}"
        assert f"# TYPE {name} counter" in lines, f"{name} is not typed as a counter"
        assert f"{name} {value}" in lines, f"{name} sample missing from {body!r}"


def test_healthz_passes_while_the_loop_keeps_making_progress() -> None:
    """A healthy idle cluster still relists every watch timeout.

    Liveness must key off the loop advancing, never off traffic: a cluster with
    no managed Pod posts nothing and must stay healthy.
    """
    server = _ephemeral(FakeController(progress_age_seconds=309.0))
    try:
        status, body = _status(f"http://127.0.0.1:{server.port}/healthz")
    finally:
        server.stop()

    assert status == 200, f"progress inside the budget is healthy: {status} {body!r}"


def test_healthz_fails_when_progress_stops() -> None:
    """C1: a whole budget with no step forward is the restart signal."""
    server = _ephemeral(FakeController(progress_age_seconds=311.0))
    try:
        status, body = _status(f"http://127.0.0.1:{server.port}/healthz")
    finally:
        server.stop()

    assert status == 503, f"311 s exceeds the 310 s budget: {status} {body!r}"
    assert "311" in body, f"the body must name the progress age: {body!r}"


def test_healthz_ignores_a_first_pass_that_has_not_finished() -> None:
    """C1/I3: cold-start tolerance belongs to the startupProbe, not here.

    A pass that is still running is making progress, so /healthz says 200 even
    though no full pass has completed and the process has been up far longer
    than one watch timeout. The old dual criterion (cycle timestamp plus a one
    watch-timeout grace from start) failed a slow first pass on a large
    cluster.
    """
    server = _ephemeral(
        FakeController(
            cycle_age_seconds=None, progress_age_seconds=1.0, uptime_seconds=600.0
        )
    )
    try:
        status, body = _status(f"http://127.0.0.1:{server.port}/healthz")
    finally:
        server.stop()

    assert status == 200, f"an unfinished but advancing pass is alive: {body!r}"


def test_healthz_falls_back_to_three_watch_timeouts_without_a_budget() -> None:
    """An older controller exposes no budget; the probe still has to work."""

    class NoBudget(FakeController):
        progress_stall_budget_seconds = None

    for age, expected in ((89.0, 200), (91.0, 503)):
        server = _ephemeral(NoBudget(progress_age_seconds=age))
        try:
            status, body = _status(f"http://127.0.0.1:{server.port}/healthz")
        finally:
            server.stop()

        assert status == expected, f"age={age}s vs 3 x 30 s: {status} {body!r}"


def test_healthz_is_unavailable_not_failing_for_a_controller_without_a_clock() -> None:
    """A health check that raises must not restart a working watcher."""

    class Bare:
        reconcile_failures_total = 1

    server = _ephemeral(Bare())
    try:
        status, body = _status(f"http://127.0.0.1:{server.port}/healthz")
    finally:
        server.stop()

    assert status == 200, f"an unknown cycle age is not a stuck loop: {status} {body!r}"


def test_server_thread_is_a_daemon() -> None:
    server = _ephemeral(FakeController())
    try:
        assert server.is_running, "server must be serving after start()"
        assert server.daemon_thread, "the serving thread must not block exit"
    finally:
        server.stop()


def test_metrics_export_the_liveness_budget_next_to_the_progress_timestamp() -> None:
    """Final review M2: an alert needs the limit next to the age it compares.

    ``/healthz`` compares ``time() - last_progress_timestamp`` to the budget the
    controller derives from its delivery timeouts; without the budget on the
    same scrape an alert has to hard-code a number that a configuration change
    silently invalidates.
    """

    body = render_completion_metrics(FakeController())

    lines = body.splitlines()
    budget = "gpu_fault_completion_watcher_progress_stall_budget_seconds"
    assert f"# TYPE {budget} gauge" in lines, f"{budget} is not typed as a gauge"
    assert f"{budget} 310" in lines, f"{budget} sample missing from {body!r}"
    assert "gpu_fault_completion_watcher_last_progress_timestamp" in body, (
        "the budget is only useful next to the timestamp it bounds"
    )


def test_metrics_export_the_outbox_expiry_counter() -> None:
    """Final review C1: a retry record that aged out is a counted loss."""

    class ExpiringController(FakeController):
        outbox_expired_total = 3

    body = render_completion_metrics(ExpiringController())

    name = "gpu_fault_completion_outbox_expired_total"
    lines = body.splitlines()
    assert f"# HELP {name} " in body, f"{name} has no HELP line: {body!r}"
    assert f"# TYPE {name} counter" in lines, f"{name} is not typed as a counter"
    assert f"{name} 3" in lines, f"{name} sample missing from {body!r}"
