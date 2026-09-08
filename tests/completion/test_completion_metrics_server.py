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


class FakeController:
    reconcile_failures_total = 3
    evicted_attempts_total = 5
    restore_skipped_total = 7
    outbox_append_failures_total = 9


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


def test_server_thread_is_a_daemon() -> None:
    server = _ephemeral(FakeController())
    try:
        assert server.is_running, "server must be serving after start()"
        assert server.daemon_thread, "the serving thread must not block exit"
    finally:
        server.stop()
