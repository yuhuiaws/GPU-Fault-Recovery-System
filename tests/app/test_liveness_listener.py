"""The per-process liveness listener and the ``/livez`` verdict it shares.

Why a listener of its own: under the 50-cluster cold-connect burst the
kubelet's probe connection sat in the service socket's accept queue behind
~3,900 storm connections (3.6 s locally at 1,090 req/s, over 5 s live) while
the request itself took 0.1 s in-app, and all three api-ha Pods were
restarted although every request was answered 202. The listener binds its
own port with SO_REUSEPORT in every uvicorn worker and answers from the
worker's event loop, so a stalled loop still fails the probe and a busy one
answers within an iteration.
"""

from __future__ import annotations

import asyncio
import gc
import http.client
import json
import socket
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.app import ApplicationContext, create_app
from gpu_fault.app.liveness import (
    LIVENESS_PORT_ENV,
    LivenessListener,
    liveness_port_from_environment,
    liveness_report,
)
from tests._builders import build_store


def _get(port: int, path: str, method: str = "GET") -> tuple[int, bytes]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        connection.request(method, path)
        response = connection.getresponse()
        return response.status, response.read()
    finally:
        connection.close()


def _raw(port: int, payload: bytes) -> bytes:
    with socket.create_connection(("127.0.0.1", port), timeout=5) as raw:
        raw.sendall(payload)
        return raw.recv(512)


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def test_the_listener_answers_the_probe_from_the_event_loop() -> None:
    verdicts: list[tuple[int, dict[str, Any]]] = [(200, {"status": "alive"})]

    async def scenario() -> None:
        listener = LivenessListener(
            port=0, report=lambda: verdicts[0], host="127.0.0.1"
        )
        port = await listener.start()
        assert listener.bound_port == port
        try:
            status, body = await asyncio.to_thread(_get, port, "/livez")
            assert (status, json.loads(body)) == (200, {"status": "alive"})
            verdicts[0] = (503, {"status": "dead", "dead_threads": ["x"]})
            status, body = await asyncio.to_thread(_get, port, "/livez?probe=1")
            assert status == 503 and json.loads(body)["dead_threads"] == ["x"]
            status, _ = await asyncio.to_thread(_get, port, "/healthz")
            assert status == 404, "only the liveness path is served here"
            status, _ = await asyncio.to_thread(_get, port, "/livez", "POST")
            assert status == 405
            status, body = await asyncio.to_thread(_get, port, "/livez", "HEAD")
            assert (status, body) == (503, b""), "HEAD carries no body"
            first_line = (
                await asyncio.to_thread(_raw, port, b"garbage\r\n\r\n")
            ).split(b"\r\n")[0]
            assert first_line == b"HTTP/1.1 400 Bad Request"
            assert listener.requests_total == 6
            assert listener.failures_total == 2, "the two 503s"
        finally:
            await listener.close()
        with pytest.raises(OSError):
            _get(port, "/livez")

    asyncio.run(scenario())


def test_two_processes_worth_of_listeners_share_one_port() -> None:
    """SO_REUSEPORT: every uvicorn worker binds the same port and the kernel
    spreads the kubelet's connections over them, as the shared service
    socket already did - a probe may land on any worker."""

    async def scenario() -> set[int]:
        first = LivenessListener(
            port=0, report=lambda: (200, {"status": "alive"}), host="127.0.0.1"
        )
        port = await first.start()
        second = LivenessListener(
            port=port, report=lambda: (503, {"status": "dead"}), host="127.0.0.1"
        )
        await second.start()
        try:
            statuses = set()
            for _ in range(40):
                status, _ = await asyncio.to_thread(_get, port, "/livez")
                statuses.add(status)
            return statuses
        finally:
            await first.close()
            await second.close()

    assert asyncio.run(scenario()) == {200, 503}


def test_the_port_setting_is_validated_and_off_by_default() -> None:
    assert liveness_port_from_environment({}) == 0
    assert liveness_port_from_environment({LIVENESS_PORT_ENV: "8090"}) == 8090
    with pytest.raises(ValueError, match="TCP port"):
        liveness_port_from_environment({LIVENESS_PORT_ENV: "70000"})
    with pytest.raises(ValueError):
        liveness_port_from_environment({LIVENESS_PORT_ENV: "eight"})
    with pytest.raises(ValueError, match="TCP port"):
        LivenessListener(port=-1, report=lambda: (200, {}))


def test_the_verdict_names_the_dead_thread_of_the_role() -> None:
    environment = {"GPU_FAULT_SERVICE_ROLE": "spool-worker"}
    status, payload = liveness_report(
        service_role="spool-worker",
        processor=SimpleNamespace(spool_consumer_running=False),
        environment=environment,
    )
    assert (status, payload["status"]) == (503, "dead")
    assert payload["dead_threads"] == ["telemetry-spool-consumer"]
    assert payload["service_role"] == "spool-worker"

    dead_consumer = SimpleNamespace(
        active_consumers=True,
        processor_consumer_running=False,
        consumer_last_cycle_age_seconds=12.0,
    )
    status, payload = liveness_report(
        service_role="worker", processor=dead_consumer, environment={}
    )
    assert status == 503 and payload["dead_threads"] == ["processor-consumer"]
    assert payload["service_role"] == "combined", "no role in the environment"

    status, payload = liveness_report(
        service_role="ingress", processor=None, environment=environment
    )
    assert (status, payload) == (
        200,
        {"status": "alive", "service_role": "spool-worker", "dead_threads": []},
    )


def _ingress_app(monkeypatch: pytest.MonkeyPatch, port: int) -> Any:
    monkeypatch.setenv("GPU_FAULT_SERVICE_ROLE", "ingress")
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_MODE", "active-active")
    monkeypatch.setenv("POD_UID", "pod-liveness-listener")
    monkeypatch.setenv("GPU_FAULT_PROCESS_METRICS_DIR", "off")
    monkeypatch.setenv(LIVENESS_PORT_ENV, str(port))
    return create_app(
        ApplicationContext(
            store=build_store(), execution_token="liveness-listener-token-" + "x" * 32
        )
    )


def test_the_lifespan_starts_the_listener_and_closes_it(monkeypatch) -> None:
    port = _free_port()
    app = _ingress_app(monkeypatch, port)
    listener = app.state.liveness_listener
    assert listener is not None and listener.port == port

    async def scenario() -> tuple[int, dict[str, Any]]:
        async with app.router.lifespan_context(app):
            assert listener.bound_port == port
            status, body = await asyncio.to_thread(_get, port, "/livez")
            return status, json.loads(body)

    status, payload = asyncio.run(scenario())
    assert status == 200
    assert payload == {"status": "alive", "service_role": "ingress", "dead_threads": []}
    with pytest.raises(OSError):
        _get(port, "/livez")


def test_without_the_port_setting_there_is_no_listener(monkeypatch) -> None:
    monkeypatch.delenv(LIVENESS_PORT_ENV, raising=False)
    monkeypatch.setenv("GPU_FAULT_SERVICE_ROLE", "ingress")
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_MODE", "active-active")
    monkeypatch.setenv("POD_UID", "pod-liveness-listener-off")
    app = create_app(
        ApplicationContext(
            store=build_store(), execution_token="liveness-listener-token-" + "y" * 32
        )
    )
    assert app.state.liveness_listener is None


def test_the_startup_gc_policy_freezes_the_heap_when_asked(monkeypatch) -> None:
    monkeypatch.setenv("GPU_FAULT_GC_FREEZE_AFTER_STARTUP", "true")
    monkeypatch.delenv(LIVENESS_PORT_ENV, raising=False)
    monkeypatch.setenv("GPU_FAULT_SERVICE_ROLE", "ingress")
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_MODE", "active-active")
    monkeypatch.setenv("POD_UID", "pod-liveness-gc-policy")
    app = create_app(
        ApplicationContext(
            store=build_store(), execution_token="liveness-listener-token-" + "z" * 32
        )
    )
    before = gc.get_freeze_count()

    async def scenario() -> int:
        async with app.router.lifespan_context(app):
            return gc.get_freeze_count()

    try:
        assert asyncio.run(scenario()) > before
    finally:
        # The frozen objects belong to this test process; put them back.
        gc.unfreeze()
