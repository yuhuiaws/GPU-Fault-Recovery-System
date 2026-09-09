"""The shared data-plane ``/metrics`` + ``/healthz`` server (F8).

The node-installer reconciler is the first user; the cluster-action executor
is next. Both are loops with no HTTP surface of their own, so the module has
to stay dependency-free and must never take the loop down with it: port 0 is
"disabled", a bind failure is one ERROR line and the process carries on.
"""

from __future__ import annotations

import logging
import re
import socket
import threading
from datetime import datetime, timezone
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from gpu_fault import dataplane_metrics as metrics_module
from gpu_fault.dataplane_metrics import (
    MetricFamily,
    MetricsServer,
    start_metrics_server,
)

NOW = datetime(2026, 9, 8, 12, 0, tzinfo=timezone.utc)
SAMPLE = re.compile(r"^(?P<name>[a-z_]+) (?P<value>-?\d+(?:\.\d+)?)$")


def _family() -> MetricFamily:
    return MetricFamily(
        "gpu_fault_example_",
        counters=(("passes_total", "Passes completed."),),
        gauges=(("depth", "Items waiting."),),
        timestamps=(("last_pass_completed_timestamp", "Unix seconds of last pass."),),
    )


def _get(url: str) -> tuple[int, str, str]:
    try:
        with urlopen(url, timeout=5) as response:
            return (
                response.status,
                response.headers.get("Content-Type", ""),
                response.read().decode("utf-8"),
            )
    except HTTPError as error:
        return error.code, "", error.read().decode("utf-8")


def _ephemeral(
    family: MetricFamily, health: metrics_module.HealthPredicate | None = None
) -> MetricsServer:
    """Bind port 0 directly: the start helper treats 0 as *disabled*."""

    server = MetricsServer(family, port=0, health=health)
    server.start()
    return server


def test_render_emits_help_type_and_one_sample_per_metric() -> None:
    family = _family()
    family.inc("passes_total")
    family.inc("passes_total", 2)
    family.set("depth", 4)
    family.mark("last_pass_completed_timestamp", NOW)

    body = family.render()

    lines = body.splitlines()
    assert body.endswith("\n"), "exposition must end with a newline"
    samples = {
        match.group("name"): match.group("value")
        for match in map(SAMPLE.match, lines)
        if match is not None
    }
    assert samples == {
        "gpu_fault_example_passes_total": "3",
        "gpu_fault_example_depth": "4",
        "gpu_fault_example_last_pass_completed_timestamp": str(int(NOW.timestamp())),
    }, body
    for name, metric_type in (
        ("gpu_fault_example_passes_total", "counter"),
        ("gpu_fault_example_depth", "gauge"),
        ("gpu_fault_example_last_pass_completed_timestamp", "gauge"),
    ):
        assert f"# HELP {name} " in body, f"{name} has no HELP line: {body!r}"
        assert f"# TYPE {name} {metric_type}" in lines, (
            f"{name} is not typed as a {metric_type}: {body!r}"
        )
    assert family.names() == tuple(samples), "names() must list every series"


def test_fresh_family_renders_zero_for_everything() -> None:
    body = _family().render()

    assert "gpu_fault_example_passes_total 0" in body.splitlines(), body
    assert "gpu_fault_example_last_pass_completed_timestamp 0" in body.splitlines(), (
        "an absent timestamp must read 0, which is what Prometheus readers expect"
    )


def test_counters_only_go_up_and_names_are_checked() -> None:
    family = _family()
    with pytest.raises(ValueError, match="passes_total"):
        family.inc("passes_total", -1)
    with pytest.raises(KeyError, match="unknown_total"):
        family.inc("unknown_total")
    with pytest.raises(KeyError, match="depth"):
        family.inc("depth")  # a gauge is not a counter
    with pytest.raises(KeyError, match="passes_total"):
        family.set("passes_total", 5)  # a counter is not a gauge
    with pytest.raises(ValueError, match="prefix"):
        MetricFamily("GPU-Fault-", counters=(("x_total", "x"),))
    with pytest.raises(ValueError, match="name"):
        MetricFamily("gpu_fault_example_", counters=(("Bad-Name", "x"),))


def test_metrics_endpoint_serves_the_family_live() -> None:
    family = _family()
    server = _ephemeral(family)
    try:
        family.inc("passes_total", 7)
        status, content_type, body = _get(f"http://127.0.0.1:{server.port}/metrics")
        family.inc("passes_total")
        _status, _content_type, later = _get(
            f"http://127.0.0.1:{server.port}/metrics?x=1"
        )
        not_found, _ct, _body = _get(f"http://127.0.0.1:{server.port}/other")
    finally:
        server.stop()

    assert status == 200, body
    assert content_type.startswith("text/plain"), content_type
    assert "gpu_fault_example_passes_total 7" in body.splitlines(), body
    assert "gpu_fault_example_passes_total 8" in later.splitlines(), (
        "each scrape must read the live value"
    )
    assert not_found == 404
    assert not server.is_running, "server thread is still alive after stop()"


@pytest.mark.parametrize(("healthy", "expected"), [(True, 200), (False, 503)])
def test_healthz_follows_the_callers_predicate(healthy: bool, expected: int) -> None:
    server = _ephemeral(_family(), health=lambda: healthy)
    try:
        status, _content_type, body = _get(f"http://127.0.0.1:{server.port}/healthz")
    finally:
        server.stop()

    assert status == expected, body


def test_healthz_without_a_predicate_is_200_and_a_raising_one_reports_unknown(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A bug in the caller's check must not get a working loop restarted."""

    def broken() -> bool:
        raise RuntimeError("clock missing")

    bare = _ephemeral(_family())
    try:
        status, _ct, _body = _get(f"http://127.0.0.1:{bare.port}/healthz")
    finally:
        bare.stop()
    assert status == 200

    server = _ephemeral(_family(), health=broken)
    try:
        with caplog.at_level(logging.ERROR, logger=metrics_module.LOGGER.name):
            status, _ct, body = _get(f"http://127.0.0.1:{server.port}/healthz")
    finally:
        server.stop()
    assert status == 200, body
    assert "unknown" in body, body
    assert any(record.levelno == logging.ERROR for record in caplog.records), (
        "a raising predicate must be logged, not silently mapped to healthy"
    )


def test_port_zero_disables_the_server(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger=metrics_module.LOGGER.name):
        assert start_metrics_server(_family(), port=0) is None, (
            "port 0 must not start a server"
        )
    assert any("disabled" in record.getMessage() for record in caplog.records)


def test_bind_failure_is_logged_and_does_not_raise(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
        occupied.bind(("0.0.0.0", 0))
        occupied.listen(1)
        port = occupied.getsockname()[1]
        with caplog.at_level(logging.ERROR, logger=metrics_module.LOGGER.name):
            server = start_metrics_server(_family(), port=port)

    assert server is None, "a bind failure must return None, not a half-started server"
    errors = [record for record in caplog.records if record.levelno == logging.ERROR]
    assert errors, "bind failure must be logged at ERROR"
    assert str(port) in errors[0].getMessage(), errors[0].getMessage()


def test_started_server_runs_on_a_daemon_thread() -> None:
    live = _ephemeral(_family())
    try:
        assert live.is_running
        thread = next(
            t for t in threading.enumerate() if t.name == "gpu-fault-dataplane-metrics"
        )
        assert thread.daemon, "a non-daemon thread would block process exit"
    finally:
        live.stop()
