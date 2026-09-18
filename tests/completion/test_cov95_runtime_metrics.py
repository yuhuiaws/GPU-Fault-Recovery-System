from __future__ import annotations

from datetime import timedelta
from http.client import HTTPResponse
from types import SimpleNamespace

import pytest

from gpu_fault import completion_metrics_server as metrics
from tests.completion._cov95_runtime_metrics import (
    FakeHTTPServer,
    FakeThread,
    MemorySocket,
)
from tests.completion._cov95_runtime_support import make_controller
from tests.completion._support import NOW, FakeSink
from tests.completion.test_completion_metrics_server import FakeController


@pytest.mark.parametrize(
    "budget,watch_timeout,expected",
    [("invalid", "invalid", 90), (0, 0, 90), (-1, 40, 120), (None, None, 90)],
)
def test_unusable_liveness_configuration_falls_back_to_bounded_relists(
    budget, watch_timeout, expected
):
    controller = SimpleNamespace(
        now=lambda: NOW,
        last_progress_at=NOW - timedelta(seconds=expected + 1),
        progress_stall_budget_seconds=budget,
        watch_timeout_seconds=watch_timeout,
    )
    status, body = metrics.evaluate_completion_health(controller)
    assert status == 503
    assert f"{expected:.1f}s budget" in body


@pytest.mark.parametrize(
    "clock,progress", [(lambda: "invalid", NOW), (lambda: NOW, None)]
)
def test_unreadable_progress_is_explicitly_unknown_not_false_stall(clock, progress):
    status, body = metrics.evaluate_completion_health(
        SimpleNamespace(now=clock, last_progress_at=progress)
    )
    assert status == 200
    assert body.startswith("health unknown:"), (
        "unreadable health needs an explicit verdict"
    )


@pytest.mark.parametrize(
    "stamp,expected", [("123.5", 123), ("invalid", 0), (object(), 0)]
)
def test_metrics_timestamp_conversion_never_uses_unparseable_values(stamp, expected):
    controller = SimpleNamespace(last_cycle_completed_at=stamp)
    body = metrics.render_completion_metrics(controller)
    name = "gpu_fault_completion_watcher_last_cycle_completed_timestamp"
    lines = [line for line in body.splitlines() if line.startswith(f"{name} ")]
    assert lines == [f"{name} {expected}"]


@pytest.mark.parametrize("path", ["/metrics?local=1", "/healthz", "/unknown"])
def test_http_handlers_use_real_http_framing_over_an_in_memory_socket(
    monkeypatch, path
):
    def initialize(server, address, handler):
        server.server_address = address
        server.RequestHandlerClass = handler

    monkeypatch.setattr(metrics.ThreadingHTTPServer, "__init__", initialize)
    controller = FakeController()
    if path == "/healthz":

        def broken_clock():
            raise RuntimeError("fake clock failure")

        controller.now = broken_clock
    server = metrics.CompletionMetricsHTTPServer(("memory", 1), controller)
    connection = MemorySocket(f"GET {path} HTTP/1.0\r\nHost: memory\r\n\r\n".encode())
    server.RequestHandlerClass(connection, ("memory", 1), server)
    response = HTTPResponse(MemorySocket(bytes(connection.outgoing)))
    response.begin()
    body = response.read().decode()
    assert response.status == (404 if path == "/unknown" else 200)
    if path.startswith("/metrics"):
        assert response.getheader("Content-Length") == str(len(body.encode()))
        assert "text/plain" in response.getheader("Content-Type")
        assert "gpu_fault_completion_controller_reconcile_failures_total 3" in body
    elif path == "/healthz":
        assert body == "health unknown: the check itself failed\n"


def test_metrics_server_start_and_idempotent_stop_use_bounded_fake_thread(monkeypatch):
    servers, threads = [], []

    def server(address, controller):
        created = FakeHTTPServer(address, controller)
        servers.append(created)
        return created

    def thread(**kwargs):
        created = FakeThread(**kwargs)
        threads.append(created)
        return created

    monkeypatch.setattr(metrics, "CompletionMetricsHTTPServer", server)
    monkeypatch.setattr(metrics, "threading", SimpleNamespace(Thread=thread))
    controller = FakeController()
    idle = metrics.CompletionMetricsServer(controller, port=9111)
    assert idle.port == 9111
    assert idle.is_running is False
    idle.stop()
    active = metrics.start_completion_metrics_server(controller, port=9111)
    assert active is not None
    assert active.port == 9111
    assert active.is_running is True
    assert active.daemon_thread is True
    active.stop()
    active.stop()
    assert active.is_running is False
    assert servers[0].calls == ["serve", "shutdown", "close"]
    assert threads[0].joins == [5]
    assert threads[0].name == "gpu-fault-completion-metrics"


def test_bad_port_setting_does_not_start_transport(monkeypatch, caplog) -> None:
    monkeypatch.setenv(metrics.METRICS_PORT_ENV, "not-a-port")
    assert metrics.start_completion_metrics_server(FakeController()) is None
    assert "bad port setting" in caplog.text


@pytest.mark.parametrize("value", ["invalid", 0, -1])
def test_sink_timing_fallback_preserves_controller_liveness_budget(value) -> None:
    sink = FakeSink()
    sink.processor_receipt_timeout_seconds = value
    sink.timeout_seconds = value
    sink.max_attempts = value
    assert make_controller(sink=sink).progress_stall_budget_seconds == 310


def test_read_only_inner_sink_still_stamps_live_delivery_progress(caplog) -> None:
    inner = FakeSink()

    class ReadOnlySink:
        @property
        def sink(self):
            return inner

        def post(self, path, payload):
            return inner.post(path, payload)

    controller = make_controller(sink=ReadOnlySink())
    assert controller.progress_stall_budget_seconds == 310
    controller.sink.post("/local", {"accepted": True})
    assert controller.last_progress_at == NOW
    assert inner.posts == [("/local", {"accepted": True})]
    assert "cannot stamp progress on the buffered-record sink" in caplog.text
