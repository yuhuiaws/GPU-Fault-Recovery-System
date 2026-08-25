from __future__ import annotations

import asyncio
import base64
import json
from datetime import datetime, timedelta, timezone
from threading import Event

import pytest

from gpu_fault.app import ApplicationContext, create_app
from gpu_fault.processor import ProcessorRequest, ProcessorRequestStatus
from tests._builders import asgi_client, copy_model, processor_request
from tests.processor._leadership_support import NOW, REQUEST_LEASE, _telemetry


def test_telemetry_uses_node_lanes_and_faults_have_priority(stores) -> None:
    first, second = stores
    now = datetime.now(timezone.utc)
    leadership = first.acquire_processor_leadership(
        "pod-a", now=now, lease_duration=timedelta(minutes=1)
    )

    def telemetry(node_id: str) -> ProcessorRequest:
        return processor_request(
            "/v1/collector-events/gpu-metrics",
            body=f'{{"node_id":"{node_id}"}}'.encode(),
        )

    node_a_first = telemetry("node-a")
    node_a_second = telemetry("node-a")
    node_b = telemetry("node-b")
    fault = processor_request("/v1/gpu-events/xid")
    for item in (node_a_first, node_a_second, node_b, fault):
        first.enqueue_processor_request(item)

    claimed = second.claim_processor_requests(
        "pod-a", leadership.epoch, now=now, lease_duration=REQUEST_LEASE, limit=16
    )

    assert claimed[0].request_id == fault.request_id
    assert {item.request_id for item in claimed[1:]} == {
        node_a_first.request_id,
        node_b.request_id,
    }
    assert node_a_second.request_id not in {item.request_id for item in claimed}


def test_healthy_telemetry_channels_use_independent_node_sublanes() -> None:
    def request(path: str, body: bytes) -> ProcessorRequest:
        return processor_request(path, body=body)

    inventory = request("/v1/collector-events/gpu-inventory", b'{"node_id":"node-a"}')
    gpu_summary = request(
        "/v1/collector-events/gpu-metrics",
        b'{"node_id":"node-a","edge_filter_reasons":["health-summary"]}',
    )
    host_summary = request(
        "/v1/collector-events/host-telemetry",
        b'{"node_id":"node-a",'
        b'"edge_filter_reasons":["health-summary"],'
        b'"collection_errors":[]}',
    )
    gpu_edge = request(
        "/v1/collector-events/gpu-metrics",
        b'{"node_id":"node-a","edge_filter_reasons":["threshold:temperature"]}',
    )

    assert (
        len(
            {
                inventory.ordering_key(),
                gpu_summary.ordering_key(),
                host_summary.ordering_key(),
            }
        )
        == 3
    )
    assert gpu_edge.ordering_key() == "cluster-a:node:node-a"


def test_active_claim_can_supply_worker_pools_independently(stores) -> None:
    first, second = stores
    now = datetime.now(timezone.utc)
    requests = {
        path: processor_request(
            path,
            body=b'{"attempt_id":"attempt-a"}'
            if path == "/v1/workload-observations"
            else f'{{"node_id":"node-{index}"}}'.encode(),
        )
        for index, path in enumerate(
            (
                "/v1/workload-observations",
                "/v1/collector-events/gpu-inventory",
                "/v1/collector-events/gpu-metrics",
                "/v1/collector-events/host-telemetry",
                "/v1/gpu-events/xid",
            )
        )
    }
    for request in requests.values():
        first.enqueue_processor_request(request)

    claimed = second.claim_active_processor_requests(
        "pod-a",
        now=now,
        lease_duration=REQUEST_LEASE,
        limit=4,
        include_paths={
            "/v1/collector-events/gpu-inventory",
            "/v1/collector-events/gpu-metrics",
        },
    )

    assert {item.path for item in claimed} == {
        "/v1/collector-events/gpu-inventory",
        "/v1/collector-events/gpu-metrics",
    }
    assert all(
        item.path
        not in {
            "/v1/collector-events/gpu-inventory",
            "/v1/collector-events/gpu-metrics",
        }
        for item in first.claim_active_processor_requests(
            "pod-b",
            now=now,
            lease_duration=REQUEST_LEASE,
            limit=4,
            exclude_paths={
                "/v1/collector-events/gpu-inventory",
                "/v1/collector-events/gpu-metrics",
            },
        )
    ), (
        'expected all( item.path not in { "/v1/collector-events/gpu-inventory", "/v1/collector-events/gpu-metrics", } for item in first.claim_active_proces... to be truthy'
    )


def test_processor_queue_admission_is_global_and_cluster_scoped(stores) -> None:
    first, second = stores

    def request(cluster_id: str) -> ProcessorRequest:
        return processor_request("/v1/gpu-events/xid", cluster_id=cluster_id)

    accepted, reason = first.try_enqueue_processor_request(
        request("cluster-a"), max_depth=2, max_cluster_depth=1
    )
    assert accepted is not None
    assert reason is None
    rejected, reason = second.try_enqueue_processor_request(
        request("cluster-a"), max_depth=2, max_cluster_depth=1
    )
    assert rejected is None
    assert reason == "cluster"
    accepted, reason = second.try_enqueue_processor_request(
        request("cluster-b"), max_depth=2, max_cluster_depth=1
    )
    assert accepted is not None
    assert reason is None
    rejected, reason = first.try_enqueue_processor_request(
        request("cluster-c"), max_depth=2, max_cluster_depth=1
    )
    assert rejected is None
    assert reason == "global"

    stats = second.processor_queue_stats()
    assert stats["depth"] == 2
    assert stats["by_cluster"] == {"cluster-a": 1, "cluster-b": 1}
    assert stats["oldest_age_seconds"] >= 0


def test_processor_batch_admission_preserves_coalescing_and_limits(stores) -> None:
    first, _second = stores

    def telemetry(node_id: str, sequence: int) -> ProcessorRequest:
        return processor_request(
            "/v1/collector-events/gpu-inventory",
            body=json.dumps(
                {
                    "cluster_id": "cluster-a",
                    "node_id": node_id,
                    "observed_at": (NOW + timedelta(seconds=sequence)).isoformat(),
                }
            ).encode(),
        )

    first_batch = first.try_enqueue_processor_requests_batch(
        [telemetry("node-a", 0), telemetry("node-a", 1), telemetry("node-b", 0)],
        max_depth=10,
        max_cluster_depth=2,
    )

    assert first_batch[0][0] is not None
    assert first_batch[1][0].request_id == first_batch[0][0].request_id
    assert first_batch[1][1] == "coalesced"
    assert first_batch[2][0] is not None
    assert first.processor_queue_stats()["depth"] == 2

    rejected = first.try_enqueue_processor_requests_batch(
        [telemetry("node-c", 0)], max_depth=10, max_cluster_depth=2
    )
    assert rejected == [(None, "cluster")]


def test_processor_http_telemetry_uses_admission_microbatch(monkeypatch) -> None:
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_MODE", "active-active")
    monkeypatch.setenv("POD_UID", "pod-admission-batch")
    context = ApplicationContext(execution_token="batch-token-" + "x" * 32)
    original = context.store.try_enqueue_processor_requests_batch
    batch_sizes = []
    batch_clusters = []

    def record_batch(requests, **kwargs):
        batch_sizes.append(len(requests))
        batch_clusters.append({request.cluster_id for request in requests})
        return original(requests, **kwargs)

    monkeypatch.setattr(
        context.store, "try_enqueue_processor_requests_batch", record_batch
    )
    app = create_app(context)

    async def scenario():
        async with asgi_client(app) as client:
            responses = await asyncio.gather(
                *[
                    client.post(
                        "/v1/collector-events/gpu-inventory",
                        headers={
                            "X-GPU-Fault-Cluster-ID": (
                                "cluster-a" if index < 8 else "cluster-b"
                            )
                        },
                        json={
                            "cluster_id": ("cluster-a" if index < 8 else "cluster-b"),
                            "node_id": f"node-{index}",
                            "observed_at": NOW.isoformat(),
                        },
                    )
                    for index in range(16)
                ]
            )
        return responses

    responses = asyncio.run(scenario())
    app.state.store_io.close()

    assert all(response.status_code == 202 for response in responses), (
        "expected all(response.status_code == 202 for response in responses) to be truthy"
    )
    assert sum(batch_sizes) == 16
    assert max(batch_sizes) > 1
    assert all(len(clusters) == 1 for clusters in batch_clusters), (
        "expected all(len(clusters) == 1 for clusters in batch_clusters) to be truthy"
    )


def test_active_standby_mode_is_rejected() -> None:
    from gpu_fault.app.processor_factory import ProcessorFactory

    with pytest.raises(RuntimeError, match="has been removed"):
        ProcessorFactory(
            ApplicationContext(execution_token="processor-token-" + "x" * 32),
            mode="active-standby",
            exit_grace_seconds=5,
        ).build()


def test_processor_http_faults_use_cluster_microbatch(monkeypatch) -> None:
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_MODE", "active-active")
    monkeypatch.setenv("GPU_FAULT_SERVICE_ROLE", "ingress")
    monkeypatch.setenv("POD_UID", "pod-fault-admission-batch")
    context = ApplicationContext(execution_token="fault-batch-token-" + "x" * 32)
    original = context.store.try_enqueue_processor_requests_batch
    batch_sizes = []

    def record_batch(requests, **kwargs):
        if requests[0].queue_priority() == 0:
            batch_sizes.append(len(requests))
        return original(requests, **kwargs)

    monkeypatch.setattr(
        context.store, "try_enqueue_processor_requests_batch", record_batch
    )
    app = create_app(context)

    async def scenario():
        async with asgi_client(app) as client:
            return await asyncio.gather(
                *[
                    client.post(
                        "/v1/collector-events/nvidia-kernel",
                        headers={"X-GPU-Fault-Cluster-ID": "cluster-a"},
                        json={"cluster_id": "cluster-a", "node_id": f"node-{index}"},
                    )
                    for index in range(16)
                ]
            )

    responses = asyncio.run(scenario())
    app.state.store_io.close()
    app.state.fault_store_io.close()

    assert all(response.status_code == 202 for response in responses), (
        "expected all(response.status_code == 202 for response in responses) to be truthy"
    )
    assert sum(batch_sizes) == 16
    assert max(batch_sizes) > 1
    assert all(
        "gpu_fault_admission;dur=" in response.headers["Server-Timing"]
        for response in responses
    ), (
        'expected all( "gpu_fault_admission;dur=" in response.headers["Server-Timing"] for response in responses ) to be truthy'
    )


def test_processor_http_evidence_uses_cluster_microbatch(monkeypatch) -> None:
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_MODE", "active-active")
    monkeypatch.setenv("GPU_FAULT_SERVICE_ROLE", "ingress")
    monkeypatch.setenv("POD_UID", "pod-evidence-admission-batch")
    context = ApplicationContext(execution_token="evidence-batch-token-" + "x" * 32)
    original = context.store.try_enqueue_processor_requests_batch
    batch_sizes = []

    def record_batch(requests, **kwargs):
        if requests[0].queue_priority() == 50:
            batch_sizes.append(len(requests))
        return original(requests, **kwargs)

    monkeypatch.setattr(
        context.store, "try_enqueue_processor_requests_batch", record_batch
    )
    app = create_app(context)

    async def scenario():
        async with asgi_client(app) as client:
            return await asyncio.gather(
                *[
                    client.post(
                        "/v1/collector-events/gpu-metrics",
                        headers={"X-GPU-Fault-Cluster-ID": "cluster-a"},
                        json={
                            "cluster_id": "cluster-a",
                            "node_id": f"node-{index}",
                            "observed_at": NOW.isoformat(),
                            "source": "DCGM_EXPORTER",
                            "samples": [],
                            "edge_filter_reasons": ["threshold:synthetic-priority-50"],
                        },
                    )
                    for index in range(16)
                ]
            )

    responses = asyncio.run(scenario())
    app.state.store_io.close()
    app.state.fault_store_io.close()

    assert all(response.status_code == 202 for response in responses), (
        "expected all(response.status_code == 202 for response in responses) to be truthy"
    )
    assert sum(batch_sizes) == 16
    assert max(batch_sizes) > 1


def test_fault_ingress_has_its_own_store_io_lane(monkeypatch) -> None:
    """A saturated telemetry lane must not answer for fault ingress.

    Fault requests already had a reserved ingress semaphore and reserved
    queue depth, then queued for the same store I/O threads as the
    telemetry burst - which is where fault p99 went to 20s while its p50
    stayed under a second.
    """
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_MODE", "active-active")
    monkeypatch.setenv("GPU_FAULT_SERVICE_ROLE", "ingress")
    monkeypatch.setenv("POD_UID", "pod-fault-lane")
    monkeypatch.setenv("GPU_FAULT_STORE_IO_WORKERS", "1")
    monkeypatch.setenv("GPU_FAULT_STORE_IO_MAX_IN_FLIGHT", "1")
    monkeypatch.setenv("GPU_FAULT_STORE_IO_ADMISSION_TIMEOUT_SECONDS", "0.05")
    monkeypatch.setenv("GPU_FAULT_FAULT_STORE_IO_WORKERS", "1")
    monkeypatch.setenv("GPU_FAULT_FAULT_STORE_IO_MAX_IN_FLIGHT", "1")
    context = ApplicationContext(execution_token="fault-lane-token-" + "x" * 32)
    original = context.store.try_enqueue_processor_requests_batch
    entered = Event()
    release = Event()

    def slow_batch(requests, **kwargs):
        if requests[0].queue_priority() == 100:
            entered.set()
            assert release.wait(timeout=5), (
                "expected release.wait(timeout=5) to be truthy"
            )
        return original(requests, **kwargs)

    monkeypatch.setattr(
        context.store, "try_enqueue_processor_requests_batch", slow_batch
    )
    app = create_app(context)

    async def scenario():
        async with asgi_client(app) as client:
            telemetry = asyncio.create_task(
                client.post(
                    "/v1/collector-events/gpu-inventory",
                    headers={"X-GPU-Fault-Cluster-ID": "cluster-a"},
                    json={
                        "cluster_id": "cluster-a",
                        "node_id": "node-a",
                        "observed_at": NOW.isoformat(),
                    },
                )
            )
            assert await asyncio.to_thread(entered.wait, 2)
            fault = await client.post(
                "/v1/collector-events/nvidia-kernel",
                headers={"X-GPU-Fault-Cluster-ID": "cluster-a"},
                json={},
            )
            release.set()
            accepted = await telemetry
            # /metrics reads the store too, so scrape it after the
            # single telemetry thread is free again.
            metrics = await client.get("/metrics")
            return accepted, fault, metrics

    accepted, fault, metrics = asyncio.run(scenario())
    app.state.store_io.close()

    assert accepted.status_code == 202
    # The only store I/O thread was held by telemetry for the whole call.
    assert fault.status_code == 202
    assert float(fault.headers["X-GPU-Fault-Server-Duration-Ms"]) >= 0
    assert "gpu_fault_total;dur=" in fault.headers["Server-Timing"]
    assert "gpu_fault_admission;dur=" in fault.headers["Server-Timing"]
    assert 'gpu_fault_ingress_lane_workers{lane="fault-store"} 1' in metrics.text
    assert (
        'gpu_fault_ingress_lane_rejections_total{lane="fault-store"} 0' in metrics.text
    )


def test_ingress_backpressure_reserves_fault_capacity(monkeypatch) -> None:
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_MODE", "active-active")
    monkeypatch.setenv("GPU_FAULT_SERVICE_ROLE", "ingress")
    monkeypatch.setenv("POD_UID", "pod-ingress-backpressure")
    monkeypatch.setenv("GPU_FAULT_INGRESS_NORMAL_CONCURRENCY", "1")
    monkeypatch.setenv("GPU_FAULT_INGRESS_NORMAL_WAIT_SECONDS", "0.01")
    monkeypatch.setenv("GPU_FAULT_INGRESS_FAULT_CONCURRENCY", "1")
    context = ApplicationContext(execution_token="backpressure-token-" + "x" * 32)
    original = context.store.try_enqueue_processor_requests_batch
    entered = Event()
    release = Event()

    def slow_batch(requests, **kwargs):
        if requests[0].queue_priority() == 100:
            entered.set()
            assert release.wait(timeout=2), (
                "expected release.wait(timeout=2) to be truthy"
            )
        return original(requests, **kwargs)

    monkeypatch.setattr(
        context.store, "try_enqueue_processor_requests_batch", slow_batch
    )
    app = create_app(context)

    async def scenario():
        async with asgi_client(app) as client:
            first = asyncio.create_task(
                client.post(
                    "/v1/collector-events/gpu-inventory",
                    headers={"X-GPU-Fault-Cluster-ID": "cluster-a"},
                    json={
                        "cluster_id": "cluster-a",
                        "node_id": "node-a",
                        "observed_at": NOW.isoformat(),
                    },
                )
            )
            assert await asyncio.to_thread(entered.wait, 1)
            rejected = await client.post(
                "/v1/collector-events/gpu-inventory",
                headers={"X-GPU-Fault-Cluster-ID": "cluster-a"},
                json={
                    "cluster_id": "cluster-a",
                    "node_id": "node-b",
                    "observed_at": NOW.isoformat(),
                },
            )
            fault = await client.post(
                "/v1/collector-events/nvidia-kernel",
                headers={"X-GPU-Fault-Cluster-ID": "cluster-a"},
                json={},
            )
            release.set()
            accepted = await first
            metrics = await client.get("/metrics")
            return accepted, rejected, fault, metrics

    accepted, rejected, fault, metrics = asyncio.run(scenario())
    app.state.store_io.close()

    assert accepted.status_code == 202
    assert rejected.status_code == 503
    assert rejected.json()["scope"] == "normal"
    assert fault.status_code == 202
    assert (
        'gpu_fault_ingress_backpressure_rejections_total{scope="normal"} 1'
    ) in metrics.text


def test_processor_queue_reserves_global_capacity_for_faults(stores) -> None:
    first, second = stores

    def request(path: str, cluster_id: str) -> ProcessorRequest:
        return processor_request(path, cluster_id=cluster_id)

    for cluster_id in ("cluster-a", "cluster-b", "cluster-c"):
        accepted, reason = first.try_enqueue_processor_request(
            request("/v1/workload-observations", cluster_id),
            max_depth=5,
            max_cluster_depth=5,
            reserved_fault_depth=2,
            reserved_cluster_fault_depth=0,
        )
        assert accepted is not None
        assert reason is None

    rejected, reason = second.try_enqueue_processor_request(
        request("/v1/workload-observations", "cluster-d"),
        max_depth=5,
        max_cluster_depth=5,
        reserved_fault_depth=2,
        reserved_cluster_fault_depth=0,
    )
    assert rejected is None
    assert reason == "global_reserved"

    for cluster_id in ("cluster-d", "cluster-e"):
        accepted, reason = second.try_enqueue_processor_request(
            request("/v1/gpu-events/xid", cluster_id),
            max_depth=5,
            max_cluster_depth=5,
            reserved_fault_depth=2,
            reserved_cluster_fault_depth=0,
        )
        assert accepted is not None
        assert reason is None

    rejected, reason = first.try_enqueue_processor_request(
        request("/v1/gpu-events/xid", "cluster-f"),
        max_depth=5,
        max_cluster_depth=5,
        reserved_fault_depth=2,
        reserved_cluster_fault_depth=0,
    )
    assert rejected is None
    assert reason == "global"


def test_processor_queue_reserves_cluster_capacity_for_faults(stores) -> None:
    first, second = stores

    def request(path: str) -> ProcessorRequest:
        return processor_request(path)

    for _ in range(3):
        accepted, reason = first.try_enqueue_processor_request(
            request("/v1/workload-observations"),
            max_depth=10,
            max_cluster_depth=5,
            reserved_fault_depth=0,
            reserved_cluster_fault_depth=2,
        )
        assert accepted is not None
        assert reason is None

    rejected, reason = second.try_enqueue_processor_request(
        request("/v1/workload-observations"),
        max_depth=10,
        max_cluster_depth=5,
        reserved_fault_depth=0,
        reserved_cluster_fault_depth=2,
    )
    assert rejected is None
    assert reason == "cluster_reserved"

    for _ in range(2):
        accepted, reason = second.try_enqueue_processor_request(
            request("/v1/gpu-events/xid"),
            max_depth=10,
            max_cluster_depth=5,
            reserved_fault_depth=0,
            reserved_cluster_fault_depth=2,
        )
        assert accepted is not None
        assert reason is None

    rejected, reason = first.try_enqueue_processor_request(
        request("/v1/gpu-events/xid"),
        max_depth=10,
        max_cluster_depth=5,
        reserved_fault_depth=0,
        reserved_cluster_fault_depth=2,
    )
    assert rejected is None
    assert reason == "cluster"


def test_completed_processor_requests_are_cleaned_in_batches(stores) -> None:
    first, second = stores
    old = NOW - timedelta(minutes=20)
    requests = [
        copy_model(
            processor_request("/v1/collector-events/gpu-metrics"),
            status=ProcessorRequestStatus.COMPLETED,
            updated_at=old + timedelta(seconds=index),
        )
        for index in range(3)
    ]
    for item in requests:
        first.enqueue_processor_request(item)

    assert (
        second.cleanup_completed_processor_requests(
            older_than=NOW - timedelta(minutes=10), limit=2
        )
        == 2
    )
    with pytest.raises(KeyError):
        second.get_processor_request(requests[0].request_id)
    assert (
        second.cleanup_completed_processor_requests(
            older_than=NOW - timedelta(minutes=10), limit=2
        )
        == 1
    )


def test_pending_telemetry_is_coalesced_latest_wins(stores) -> None:
    first, second = stores

    def telemetry(node_id: str, value: int) -> ProcessorRequest:
        # A health summary, so the batch lands on its own latest-wins lane
        # rather than the node lane every channel shares.
        return _telemetry(node_id, value, reasons=["health-summary"])

    original, result = first.try_enqueue_processor_request(
        telemetry("node-a", 1), max_depth=10, max_cluster_depth=10
    )
    replacement, result = second.try_enqueue_processor_request(
        telemetry("node-a", 2), max_depth=10, max_cluster_depth=10
    )
    other_node, other_result = second.try_enqueue_processor_request(
        telemetry("node-b", 3), max_depth=10, max_cluster_depth=10
    )

    assert replacement.request_id == original.request_id
    assert result == "coalesced"
    assert b'"value":2' in replacement.body()
    assert other_result is None
    assert other_node.request_id != original.request_id
    assert second.processor_queue_stats()["depth"] == 2


def test_evidence_does_not_overwrite_what_shares_its_node_lane(stores) -> None:
    """The node lane carries every channel that resolves to one node.

    A batch that reports a threshold breach, and a node log batch from the
    same node, both key on ``{cluster}:node:{node}``. Coalescing on the
    tier instead of the lane replaced whichever of them was pending -- the
    ingress had already answered 202 with the pending request's id, so the
    caller waited on a receipt for a payload that no longer existed.
    """

    first, second = stores
    logs, logs_result = first.try_enqueue_processor_request(
        _telemetry("node-a", 1, path="/v1/collector-events/node-logs"),
        max_depth=10,
        max_cluster_depth=10,
    )
    breach, breach_result = second.try_enqueue_processor_request(
        _telemetry("node-a", 2, reasons=["threshold:gpu_ecc_uncorrectable"]),
        max_depth=10,
        max_cluster_depth=10,
    )

    assert logs_result is None
    assert breach_result is None
    assert breach.request_id != logs.request_id
    assert second.processor_queue_stats()["depth"] == 2
    assert (
        second.get_processor_request(logs.request_id).path
        == "/v1/collector-events/node-logs"
    )


def test_consecutive_breaches_are_never_coalesced_away(stores) -> None:
    """The detectors count confirmations, so an intermediate sample is
    evidence in its own right and not a restatement of the last one."""

    first, second = stores
    reasons = ["sustained:memory_available_percent"]
    older, older_result = first.try_enqueue_processor_request(
        _telemetry(
            "node-a", 1, path="/v1/collector-events/host-telemetry", reasons=reasons
        ),
        max_depth=10,
        max_cluster_depth=10,
    )
    newer, newer_result = second.try_enqueue_processor_request(
        _telemetry(
            "node-a", 2, path="/v1/collector-events/host-telemetry", reasons=reasons
        ),
        max_depth=10,
        max_cluster_depth=10,
    )

    assert older_result is None
    assert newer_result is None
    assert newer.request_id != older.request_id
    assert second.processor_queue_stats()["depth"] == 2


def test_collector_request_returns_after_durable_enqueue(monkeypatch) -> None:
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_MODE", "active-active")
    monkeypatch.setenv("POD_UID", "pod-collector-accept")
    token = "processor-collector-token-" + "x" * 32
    context = ApplicationContext(execution_token=token)
    app = create_app(context)

    async def scenario():
        async with asgi_client(app) as client:
            return await client.post(
                "/v1/collector-events/gpu-metrics",
                headers={"X-GPU-Fault-Cluster-ID": "cluster-a"},
                json={"node_id": "node-a"},
            )

    response = asyncio.run(scenario())

    assert response.status_code == 202
    assert response.json()["accepted"] is True
    assert response.json()["processor_request_id"].startswith("processor-"), (
        'expected response.json()["processor_request_id"].startswith("processor-") to be truthy'
    )
    assert context.store.processor_queue_stats()["depth"] == 1


@pytest.mark.parametrize(
    "path",
    [
        "/v1/workload-observations",
        "/v1/training-progress",
        "/v1/attempts/failure-detected",
        "/v1/attempts/terminal",
    ],
)
def test_high_volume_processor_paths_return_receipt(monkeypatch, path) -> None:
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_MODE", "active-active")
    monkeypatch.setenv("POD_UID", f"pod-async-{path.rsplit('/', 1)[-1]}")
    token = "processor-async-token-" + "x" * 32
    context = ApplicationContext(execution_token=token)
    app = create_app(context)

    async def scenario():
        async with asgi_client(app) as client:
            return await client.post(
                path, headers={"X-GPU-Fault-Cluster-ID": "cluster-a"}, json={}
            )

    response = asyncio.run(scenario())

    assert response.status_code == 202
    body = response.json()
    assert body["accepted"] is True
    assert body["status_url"] == (
        f"/v1/processor/requests/{body['processor_request_id']}"
    )


def test_processor_receipt_returns_pending_then_final_response(monkeypatch) -> None:
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_MODE", "active-active")
    monkeypatch.setenv("POD_UID", "pod-receipt")
    token = "processor-receipt-token-" + "x" * 32
    context = ApplicationContext(execution_token=token)
    app = create_app(context)

    async def enqueue_and_read_pending():
        async with asgi_client(app) as client:
            accepted = await client.post(
                "/v1/workload-observations",
                headers={"X-GPU-Fault-Cluster-ID": "cluster-a"},
                json={},
            )
            pending = await client.get(accepted.json()["status_url"])
            return accepted, pending

    accepted, pending = asyncio.run(enqueue_and_read_pending())
    assert accepted.status_code == 202
    assert pending.status_code == 202
    request_id = accepted.json()["processor_request_id"]
    claimed_at = datetime.now(timezone.utc)
    leadership = context.store.acquire_processor_leadership(
        "pod-a", now=claimed_at, lease_duration=timedelta(minutes=1)
    )
    claimed = context.store.claim_processor_requests(
        "pod-a", leadership.epoch, now=claimed_at, lease_duration=REQUEST_LEASE, limit=1
    )[0]
    context.store.complete_processor_request(
        request_id,
        "pod-a",
        leadership.epoch,
        claimed.lease_token,
        response_status=200,
        response_content_type="application/json",
        response_body_base64=base64.b64encode(b'{"stored":true}').decode("ascii"),
    )

    async def read_final():
        async with asgi_client(app) as client:
            return await client.get(accepted.json()["status_url"])

    final = asyncio.run(read_final())
    assert final.status_code == 200
    assert final.json() == {"stored": True}


def test_active_active_api_enqueues_for_lane_consumers(monkeypatch) -> None:
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_MODE", "active-active")
    monkeypatch.setenv("POD_UID", "pod-active-api")
    token = "processor-active-token-" + "x" * 32
    context = ApplicationContext(execution_token=token)
    app = create_app(context)

    async def scenario():
        async with asgi_client(app) as client:
            return await client.post(
                "/v1/collector-events/gpu-metrics",
                headers={"X-GPU-Fault-Cluster-ID": "cluster-a"},
                json={"node_id": "node-a"},
            )

    response = asyncio.run(scenario())
    request = context.store.get_processor_request(
        response.json()["processor_request_id"]
    )

    assert response.status_code == 202
    assert request.ordering_key().startswith("cluster-a:"), (
        'expected request.ordering_key().startswith("cluster-a:") to be truthy'
    )
    assert app.state.processor.active_consumers, (
        "expected app.state.processor.active_consumers to be truthy"
    )


def test_active_consumer_replay_rejects_stale_lane_lease(monkeypatch) -> None:
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_MODE", "active-active")
    monkeypatch.setenv("POD_UID", "pod-lane-replay")
    token = "processor-lane-replay-" + "x" * 32
    context = ApplicationContext(execution_token=token)
    app = create_app(context)
    request = context.store.enqueue_processor_request(
        processor_request("/v1/collector-events/gpu-metrics")
    )
    claimed = context.store.claim_active_processor_requests(
        app.state.processor.owner_id,
        now=datetime.now(timezone.utc),
        lease_duration=REQUEST_LEASE,
        limit=1,
    )[0]

    async def scenario():
        async with asgi_client(app) as client:
            headers = {
                "X-GPU-Fault-Processor-Replay": (
                    "test-processor-replay-secret-" + "r" * 32
                ),
                "X-GPU-Fault-Processor-Owner-ID": (app.state.processor.owner_id),
                "X-GPU-Fault-Processor-Request-ID": (claimed.request_id),
                "X-GPU-Fault-Processor-Lane-Epoch": str(claimed.leader_epoch),
                "X-GPU-Fault-Processor-Lane-Token": (claimed.lease_token),
                "X-GPU-Fault-Processor-Lane-Key": "Y2x1c3Rlci1h",
                "X-GPU-Fault-Cluster-ID": "cluster-a",
                "Content-Type": "application/json",
            }
            headers["X-GPU-Fault-Processor-Lane-Token"] = "stale"
            return await client.post(
                "/v1/collector-events/gpu-metrics", headers=headers, json={}
            )

    stale = asyncio.run(scenario())

    assert request.request_id == claimed.request_id
    assert stale.status_code == 409
    assert stale.headers["X-GPU-Fault-Processor-Retry"] == "lane-lease-changed"
