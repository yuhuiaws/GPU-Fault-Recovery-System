from __future__ import annotations

import asyncio
import socket
import time
from threading import Thread

import httpx
import pytest
import uvicorn

from gpu_fault.app import ApplicationContext, create_app
from tests._builders import asgi_client, build_context, processor_request


def test_unhealthy_processor_fails_health_check_and_exports_metrics(
    monkeypatch,
) -> None:
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_MODE", "active-active")
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_EXIT_ON_DEADLINE", "false")
    monkeypatch.setenv("POD_UID", "pod-deadline")
    token = "processor-deadline-token-" + "x" * 32
    app = create_app(ApplicationContext(execution_token=token))
    request = processor_request("/v1/workload-observations")
    # One strike suffices here: the test is about the health surface, not
    # the F-D7 threshold (covered in test_deadline_responsibility.py).
    app.state.processor.deadline_exceeded_process_threshold = 1
    app.state.processor._mark_execution_deadline_exceeded(request)

    async def scenario():
        async with asgi_client(app) as client:
            return await client.get("/healthz"), await client.get("/metrics")

    health, metrics = asyncio.run(scenario())

    assert health.status_code == 503
    assert health.json()["status"] == "unhealthy"
    assert (
        health.json()["processor_unhealthy_reason"]
        == "processor request execution deadline exceeded"
    )
    assert "gpu_fault_processor_healthy 0" in metrics.text
    assert "gpu_fault_processor_deadline_exceeded_total 1" in metrics.text


def test_processor_api_backpressure_and_metrics(monkeypatch) -> None:
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_MODE", "active-active")
    monkeypatch.setenv("POD_UID", "pod-capacity")
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_MAX_QUEUE_DEPTH", "1")
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_MAX_CLUSTER_QUEUE_DEPTH", "1")
    token = "processor-capacity-token-" + "x" * 32
    context = ApplicationContext(execution_token=token)
    context.store.enqueue_processor_request(processor_request("/v1/gpu-events/xid"))
    app = create_app(context)

    async def scenario():
        async with asgi_client(app) as client:
            rejected = await client.post(
                "/v1/gpu-events/xid",
                headers={"X-GPU-Fault-Cluster-ID": "cluster-b"},
                json={},
            )
            metrics = await client.get("/metrics")
            return rejected, metrics

    rejected, metrics = asyncio.run(scenario())

    assert rejected.status_code == 429
    assert rejected.headers["Retry-After"] == "2"
    assert rejected.json()["scope"] == "global"
    assert metrics.status_code == 200
    assert "gpu_fault_processor_queue_depth 1" in metrics.text
    assert (
        'gpu_fault_processor_cluster_queue_depth{cluster_id="cluster-a"} 1'
    ) in metrics.text
    assert "gpu_fault_processor_request_processing_seconds_bucket" in metrics.text


def test_regional_in_memory_metrics_render_remote_command_stats(monkeypatch) -> None:
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_MODE", "direct")
    context = build_context()
    context.regional_mode = True
    app = create_app(context)

    async def scenario():
        async with asgi_client(app) as client:
            return await client.get("/metrics")

    metrics = asyncio.run(scenario())

    assert metrics.status_code == 200
    assert "gpu_fault_remote_command_oldest_unclaimed_seconds" in metrics.text


def test_processor_api_fault_reserve_rejects_observation_not_fault(monkeypatch) -> None:
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_MODE", "active-active")
    monkeypatch.setenv("POD_UID", "pod-fault-reserve")
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_MAX_QUEUE_DEPTH", "10")
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_MAX_CLUSTER_QUEUE_DEPTH", "5")
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_FAULT_RESERVED_QUEUE_DEPTH", "0")
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_FAULT_RESERVED_CLUSTER_DEPTH", "2")
    token = "processor-reserve-token-" + "x" * 32
    context = ApplicationContext(execution_token=token)
    for _ in range(3):
        context.store.enqueue_processor_request(
            processor_request("/v1/workload-observations")
        )
    app = create_app(context)

    async def scenario():
        async with asgi_client(app) as client:
            observation = await client.post(
                "/v1/workload-observations",
                headers={"X-GPU-Fault-Cluster-ID": "cluster-a"},
                json={},
            )
            fault = await client.post(
                "/v1/collector-events/nvidia-kernel",
                headers={"X-GPU-Fault-Cluster-ID": "cluster-a"},
                json={},
            )
            metrics = await client.get("/metrics")
            return observation, fault, metrics

    observation, fault, metrics = asyncio.run(scenario())

    assert observation.status_code == 429
    assert observation.json()["scope"] == "cluster_reserved"
    assert fault.status_code == 202
    assert (
        'gpu_fault_processor_admission_rejections_total{scope="cluster_reserved"} 1'
    ) in metrics.text
    assert (
        "gpu_fault_processor_admission_rejections_by_cluster_total"
        '{cluster_id="cluster-a",scope="cluster_reserved"} 1'
    ) in metrics.text


def test_processor_rejects_oversize_body_before_queue_write(monkeypatch) -> None:
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_MODE", "active-active")
    monkeypatch.setenv("POD_UID", "pod-request-size")
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_MAX_REQUEST_BYTES", "64")
    token = "processor-size-token-" + "x" * 32
    context = ApplicationContext(execution_token=token)
    app = create_app(context)

    async def scenario():
        async with asgi_client(app) as client:
            rejected = await client.post(
                "/v1/collector-events/node-logs",
                content=b"x" * 65,
                headers={
                    "Content-Type": "application/json",
                    "X-GPU-Fault-Cluster-ID": "cluster-a",
                },
            )
            metrics = await client.get("/metrics")
            return rejected, metrics

    rejected, metrics = asyncio.run(scenario())

    assert rejected.status_code == 413
    assert rejected.json()["max_bytes"] == 64
    assert context.store.processor_queue_stats()["depth"] == 0
    assert "gpu_fault_processor_oversize_rejections_total 1" in metrics.text


@pytest.mark.parametrize(
    ("processor_mode", "expected_role"), [("active-active", "active-consumer")]
)
def test_queued_http_request_is_replayed(
    monkeypatch, processor_mode, expected_role
) -> None:
    try:
        probe = socket.socket()
    except PermissionError:
        pytest.skip("sandbox does not permit local sockets")
    with probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    token = "processor-test-token-" + "x" * 32
    monkeypatch.setenv("POD_UID", "pod-http-leader")
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_MODE", processor_mode)
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_LOCAL_URL", f"http://127.0.0.1:{port}")
    context = ApplicationContext(execution_token=token)
    server = uvicorn.Server(
        uvicorn.Config(
            create_app(context), host="127.0.0.1", port=port, log_level="error"
        )
    )
    thread = Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                health = httpx.get(f"http://127.0.0.1:{port}/healthz", timeout=1)
                if (
                    health.status_code == 200
                    and health.json()["processor_role"] == expected_role
                ):
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.05)
        else:
            pytest.fail("processor leader did not become ready")

        denied = httpx.post(f"http://127.0.0.1:{port}/v1/workflows/dispatch", timeout=5)
        accepted = httpx.post(
            f"http://127.0.0.1:{port}/v1/workflows/dispatch",
            headers={"X-GPU-Fault-Execution-Token": token},
            timeout=5,
        )

        assert denied.status_code == 403
        assert accepted.status_code == 200
        assert accepted.json()["scanned"] == 0
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        assert not thread.is_alive()
