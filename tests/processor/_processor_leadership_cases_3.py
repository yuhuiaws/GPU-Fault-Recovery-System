from __future__ import annotations

import asyncio
import base64
import json
import time
from datetime import datetime, timedelta, timezone
from threading import Event, Thread

import httpx
import pytest

from gpu_fault.app import ApplicationContext, create_app
from gpu_fault.processor import (
    ProcessorCoordinator,
    ProcessorLanePolicy,
    ProcessorLeaseSettings,
    ProcessorPoolSettings,
    ProcessorRequest,
    ProcessorRequestStatus,
    ProcessorStaleSettings,
    processor_partition_id,
)
from gpu_fault.training_health import TrainingProgressHeartbeat
from tests._builders import (
    asgi_client,
    attempt_observation,
    build_context,
    build_store,
    copy_model,
    processor_request,
)
from tests.processor._leadership_support import NOW, REQUEST_LEASE


def test_active_consumer_replay_accepts_the_claiming_process_owner(monkeypatch) -> None:
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_MODE", "active-active")
    token = "processor-cross-worker-" + "x" * 32
    store = build_store()
    monkeypatch.setenv("POD_UID", "pod-worker-a")
    app_a = create_app(ApplicationContext(store=store, execution_token=token))
    monkeypatch.setenv("POD_UID", "pod-worker-b")
    app_b = create_app(ApplicationContext(store=store, execution_token=token))
    assert app_a.state.processor.owner_id != app_b.state.processor.owner_id
    payload = {
        "snapshot_id": "snapshot-cross-worker",
        "cluster_id": "cluster-a",
        "node_id": "node-a",
        "observed_at": NOW.isoformat(),
        "source": "NVIDIA_SMI",
        "source_boot_id": "boot-a",
        "devices": [
            {
                "gpu_index": 0,
                "gpu_uuid": "GPU-node-a-0",
                "pci_bdf": "00000000:59:00.0",
                "product": "H200",
            }
        ],
    }
    request = store.enqueue_processor_request(
        processor_request(
            "/v1/collector-events/gpu-inventory", body=json.dumps(payload).encode()
        )
    )
    claimed = store.claim_active_processor_requests(
        app_a.state.processor.owner_id,
        now=datetime.now(timezone.utc),
        lease_duration=REQUEST_LEASE,
        limit=1,
    )[0]

    async def scenario():
        async with asgi_client(app_b) as client:
            return await client.post(
                request.path,
                headers={
                    "X-GPU-Fault-Processor-Replay": (
                        "test-processor-replay-secret-" + "r" * 32
                    ),
                    "X-GPU-Fault-Processor-Owner-ID": (app_a.state.processor.owner_id),
                    "X-GPU-Fault-Processor-Request-ID": (claimed.request_id),
                    "X-GPU-Fault-Processor-Lane-Epoch": str(claimed.leader_epoch),
                    "X-GPU-Fault-Processor-Lane-Token": (claimed.lease_token),
                    "X-GPU-Fault-Processor-Lane-Key": (
                        base64.urlsafe_b64encode(
                            request.ordering_key().encode()
                        ).decode()
                    ),
                    "X-GPU-Fault-Cluster-ID": "cluster-a",
                    "Content-Type": "application/json",
                },
                content=request.body(),
            )

    response = asyncio.run(scenario())
    app_a.state.store_io.close()
    app_a.state.telemetry_spool_store_io.close()
    app_b.state.store_io.close()
    app_b.state.telemetry_spool_store_io.close()

    assert response.status_code == 200
    assert store.get_gpu_inventory_snapshot("cluster-a", "node-a") is not None


def test_replay_tracker_exposes_the_receiving_handler_phase(monkeypatch) -> None:
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_MODE", "active-active")
    monkeypatch.setenv("POD_UID", "pod-handler-phase")
    token = "processor-handler-phase-" + "x" * 32
    context = ApplicationContext(execution_token=token)
    app = create_app(context)
    payload = {
        "cluster_id": "cluster-a",
        "node_id": "node-a",
        "record_id": "kernel-phase-79",
        "observed_at": NOW.isoformat(),
        "message": "NVRM: Xid (PCI:0000:b9:00): 79",
    }
    request = context.store.enqueue_processor_request(
        processor_request(
            "/v1/collector-events/nvidia-kernel", body=json.dumps(payload).encode()
        )
    )
    claimed = context.store.claim_active_processor_requests(
        app.state.processor.owner_id,
        now=datetime.now(timezone.utc),
        lease_duration=REQUEST_LEASE,
        limit=1,
    )[0]
    entered = Event()
    release = Event()
    normalize = context.hma.normalize_kernel

    def blocked_normalize(event):
        entered.set()
        release.wait(5)
        return normalize(event)

    monkeypatch.setattr(context.hma, "normalize_kernel", blocked_normalize)

    async def scenario():
        async with asgi_client(app) as client:
            response_task = asyncio.create_task(
                client.post(
                    request.path,
                    headers={
                        "X-GPU-Fault-Processor-Replay": (
                            "test-processor-replay-secret-" + "r" * 32
                        ),
                        "X-GPU-Fault-Processor-Owner-ID": (
                            app.state.processor.owner_id
                        ),
                        "X-GPU-Fault-Processor-Request-ID": (claimed.request_id),
                        "X-GPU-Fault-Processor-Lane-Epoch": str(claimed.leader_epoch),
                        "X-GPU-Fault-Processor-Lane-Token": (claimed.lease_token),
                        "X-GPU-Fault-Processor-Lane-Key": (
                            base64.urlsafe_b64encode(
                                request.ordering_key().encode()
                            ).decode()
                        ),
                        "X-GPU-Fault-Cluster-ID": "cluster-a",
                        "Content-Type": "application/json",
                    },
                    content=request.body(),
                )
            )
            assert await asyncio.to_thread(entered.wait, 5)
            snapshot = app.state.processor_replay_tracker.snapshot()
            release.set()
            response = await response_task
            return snapshot, response

    try:
        snapshot, response = asyncio.run(scenario())
    finally:
        release.set()
        app.state.store_io.close()
        app.state.telemetry_spool_store_io.close()

    assert response.status_code == 200
    assert snapshot[0]["request_id"] == claimed.request_id
    assert snapshot[0]["phase"] == "kernel_normalize"
    assert snapshot[0]["native_thread_id"] > 0


def test_queued_processor_requires_independent_replay_secret(monkeypatch) -> None:
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_MODE", "active-active")
    monkeypatch.setenv("POD_UID", "pod-replay-secret")
    execution_token = "processor-execution-token-" + "x" * 32
    context = ApplicationContext(execution_token=execution_token)
    monkeypatch.delenv("GPU_FAULT_PROCESSOR_REPLAY_SECRET", raising=False)

    with pytest.raises(RuntimeError, match="PROCESSOR_REPLAY_SECRET"):
        create_app(context)

    monkeypatch.setenv("GPU_FAULT_PROCESSOR_REPLAY_SECRET", execution_token)
    with pytest.raises(RuntimeError, match="must differ"):
        create_app(context)


def test_processor_replay_rejects_nlb_source_and_execution_token(monkeypatch) -> None:
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_MODE", "active-active")
    monkeypatch.setenv("POD_UID", "pod-replay-source")
    execution_token = "processor-execution-token-" + "x" * 32
    replay_secret = "processor-replay-secret-" + "r" * 32
    context = ApplicationContext(
        execution_token=execution_token, processor_replay_secret=replay_secret
    )
    app = create_app(context)

    async def scenario():
        external = httpx.ASGITransport(app=app, client=("198.51.100.10", 443))
        loopback = httpx.ASGITransport(app=app, client=("127.0.0.1", 12345))
        async with httpx.AsyncClient(
            transport=external, base_url="http://test"
        ) as client:
            from_nlb = await client.post(
                "/v1/collector-events/gpu-metrics",
                headers={"X-GPU-Fault-Processor-Replay": replay_secret},
                json={},
            )
        async with httpx.AsyncClient(
            transport=loopback, base_url="http://test"
        ) as client:
            leaked_execution_token = await client.post(
                "/v1/collector-events/gpu-metrics",
                headers={"X-GPU-Fault-Processor-Replay": execution_token},
                json={},
            )
        return from_nlb, leaked_execution_token

    from_nlb, leaked_execution_token = asyncio.run(scenario())

    assert from_nlb.status_code == 403
    assert leaked_execution_token.status_code == 403
    assert "loopback" in from_nlb.json()["detail"]


def test_worker_shutdown_budget_covers_processor_deadline(monkeypatch) -> None:
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_MODE", "active-active")
    monkeypatch.setenv("GPU_FAULT_SERVICE_ROLE", "worker")
    monkeypatch.setenv("POD_UID", "pod-shutdown-budget")
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_REQUEST_MAX_EXECUTION_SECONDS", "120")
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_EXIT_GRACE_SECONDS", "5")
    monkeypatch.setenv("GPU_FAULT_LIFESPAN_SHUTDOWN_MAX_SECONDS", "124")
    token = "processor-shutdown-token-" + "x" * 32
    context = ApplicationContext(
        execution_token=token, processor_replay_secret="shutdown-replay-" + "r" * 32
    )

    with pytest.raises(RuntimeError, match="LIFESPAN_SHUTDOWN_MAX_SECONDS"):
        create_app(context)


def test_processor_abandons_in_flight_request_lease() -> None:
    store = build_store()
    pending = store.enqueue_processor_request(
        processor_request("/v1/workload-observations")
    )
    claimed = store.claim_active_processor_requests(
        "pod-a:1", now=datetime.now(timezone.utc), lease_duration=REQUEST_LEASE, limit=1
    )[0]
    processor = ProcessorCoordinator(
        store,
        owner_id="pod-a:1",
        internal_token="processor-replay-token",
        active_consumers=True,
    )
    processor._request_started(claimed)

    report = processor.abandon_in_flight()

    assert report == {"in_flight": 1, "released": 1, "failed": 0}
    current = store.get_processor_request(pending.request_id)
    assert current.status is ProcessorRequestStatus.PENDING
    assert current.lease_owner is None


def test_processor_discards_result_after_hard_deadline(monkeypatch) -> None:
    store = build_store()
    request = store.enqueue_processor_request(
        processor_request("/v1/workload-observations")
    )
    claimed = store.claim_active_processor_requests(
        "pod-a:1",
        now=datetime.now(timezone.utc),
        lease_duration=timedelta(seconds=1),
        limit=1,
    )[0]
    unhealthy_reasons = []
    processor = ProcessorCoordinator(
        store,
        owner_id="pod-a:1",
        internal_token="processor-token",
        lease=ProcessorLeaseSettings(
            request_lease_seconds=1,
            request_renew_seconds=0.005,
            request_max_execution_seconds=0.02,
        ),
        pools=ProcessorPoolSettings(worker_count=1),
        active_consumers=True,
        on_unhealthy=unhealthy_reasons.append,
    )

    class SlowResponse:
        status = 200
        headers = {"Content-Type": "application/json"}

        def __enter__(self):
            time.sleep(0.05)
            return self

        def __exit__(self, *_args):
            return False

        def read(self) -> bytes:
            return b"{}"

    monkeypatch.setattr(
        "gpu_fault.processor.coordinator.urlopen",
        lambda *_args, **_kwargs: SlowResponse(),
    )

    processor._process(claimed)

    current = store.get_processor_request(request.request_id)
    metrics = processor.metrics_snapshot()
    # F-D7: the deadline is charged to the request (released with a retry
    # count and a backoff); one timeout does not make the process unhealthy.
    assert current.status is ProcessorRequestStatus.PENDING
    assert current.retry_count == 1
    assert current.not_before is not None
    assert processor.is_healthy()
    assert metrics["in_flight"] == 0
    assert metrics["deadline_exceeded_total"] == 1
    assert metrics["healthy"] == 1
    assert unhealthy_reasons == []


def test_processor_retries_completion_without_replaying_handler(monkeypatch) -> None:
    store = build_store()
    request = store.enqueue_processor_request(
        processor_request("/v1/workflows/dispatch")
    )
    claimed = store.claim_active_processor_requests(
        "pod-a:1", now=datetime.now(timezone.utc), lease_duration=REQUEST_LEASE, limit=1
    )[0]
    processor = ProcessorCoordinator(
        store,
        owner_id="pod-a:1",
        internal_token="processor-token",
        active_consumers=True,
    )
    original = store.complete_active_processor_request
    calls = 0

    def flaky_completion(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls < 3:
            raise RuntimeError("transient completion failure")
        return original(*args, **kwargs)

    class Response:
        status = 200
        headers = {"Content-Type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b"{}"

    monkeypatch.setattr(store, "complete_active_processor_request", flaky_completion)
    monkeypatch.setattr(
        "gpu_fault.processor.coordinator.urlopen", lambda *_args, **_kwargs: Response()
    )

    processor._execute(claimed, deadline=time.monotonic() + 10)
    runtime = processor.metrics_snapshot()

    assert calls == 3
    assert (
        store.get_processor_request(request.request_id).status
        is ProcessorRequestStatus.COMPLETED
    )
    assert runtime["completion_retries_total"] == 2
    assert runtime["completion_failures_total"] == 0


@pytest.mark.parametrize(
    ("age_seconds", "expected_status"),
    [(0, ProcessorRequestStatus.PENDING), (301, ProcessorRequestStatus.COMPLETED)],
)
def test_processor_retries_fresh_5xx_but_bounds_old_requests(
    monkeypatch, age_seconds, expected_status
) -> None:
    store = build_store()
    request = copy_model(
        processor_request("/v1/collector-events/gpu-metrics"),
        created_at=datetime.now(timezone.utc) - timedelta(seconds=age_seconds),
    )
    store.enqueue_processor_request(request)
    claimed = store.claim_active_processor_requests(
        "pod-a:1", now=datetime.now(timezone.utc), lease_duration=REQUEST_LEASE, limit=1
    )[0]
    processor = ProcessorCoordinator(
        store,
        owner_id="pod-a:1",
        internal_token="processor-token",
        active_consumers=True,
        lease=ProcessorLeaseSettings(
            retryable_response_max_age_seconds=300,
            retry_backoff_seconds=2,
            retry_backoff_max_seconds=10,
        ),
    )

    class Response:
        status = 503
        headers = {"Content-Type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self) -> bytes:
            return b'{"detail":"writer is read-only"}'

    monkeypatch.setattr(
        "gpu_fault.processor.coordinator.urlopen", lambda *_args, **_kwargs: Response()
    )

    processor._process(claimed)

    current = store.get_processor_request(request.request_id)
    assert current.status is expected_status
    if expected_status is ProcessorRequestStatus.PENDING:
        assert current.response_status is None
        assert current.retry_count == 1
        assert current.not_before is not None
        assert current.not_before > datetime.now(timezone.utc)
        assert current.lane_policy is ProcessorLanePolicy.REORDERABLE
        assert (
            store.claim_active_processor_requests(
                "pod-b:1",
                now=datetime.now(timezone.utc),
                lease_duration=REQUEST_LEASE,
                limit=1,
            )
            == []
        )
        retried = store.claim_active_processor_requests(
            "pod-b:1",
            now=current.not_before + timedelta(milliseconds=1),
            lease_duration=REQUEST_LEASE,
            limit=1,
        )
        assert [item.request_id for item in retried] == [request.request_id]
        metrics = processor.metrics_snapshot()
        assert metrics["retry_rescheduled_total"] == 1
        assert metrics["retry_delay_seconds_max"] == pytest.approx(2)
    else:
        assert current.response_status == 503


def test_deferred_strict_retry_blocks_lane_but_reorderable_retry_does_not(
    stores,
) -> None:
    first, _ = stores
    now = datetime.now(timezone.utc)
    strict = copy_model(
        processor_request(
            "/v1/collector-events/host-telemetry",
            body=b'{"node_id":"node-a","edge_filter_reasons":["threshold:cpu"]}',
        ),
        not_before=now + timedelta(seconds=30),
        lane_policy=ProcessorLanePolicy.STRICT,
    )
    later = processor_request(
        "/v1/collector-events/host-telemetry",
        body=b'{"node_id":"node-a","edge_filter_reasons":["recovered"]}',
    )
    first.enqueue_processor_request(strict)
    first.enqueue_processor_request(later)

    assert (
        first.claim_active_processor_requests(
            "pod-a:1", now=now, lease_duration=REQUEST_LEASE, limit=2
        )
        == []
    )

    first = build_store()
    reorderable = copy_model(
        strict,
        request_id="processor-reorderable",
        lane_policy=ProcessorLanePolicy.REORDERABLE,
    )
    first.enqueue_processor_request(reorderable)
    first.enqueue_processor_request(later)
    claimed = first.claim_active_processor_requests(
        "pod-a:1", now=now, lease_duration=REQUEST_LEASE, limit=2
    )
    assert [item.request_id for item in claimed] == [later.request_id]


def test_graceful_shutdown_releases_claimed_not_started_request() -> None:
    store = build_store()
    request = store.enqueue_processor_request(
        processor_request("/v1/collector-events/host-telemetry")
    )
    claimed = store.claim_active_processor_requests(
        "pod-a:1", now=datetime.now(timezone.utc), lease_duration=REQUEST_LEASE, limit=1
    )[0]
    processor = ProcessorCoordinator(
        store,
        owner_id="pod-a:1",
        internal_token="processor-token",
        active_consumers=True,
    )

    processor.register_claimed_requests([claimed])
    processor.release_unstarted_claims()

    current = store.get_processor_request(request.request_id)
    assert current.status is ProcessorRequestStatus.PENDING
    metrics = processor.metrics_snapshot()
    assert metrics["claimed_not_started"] == 0
    assert metrics["claimed_not_started_released_total"] == 1


def test_processor_releases_after_completion_retries_fail(monkeypatch) -> None:
    store = build_store()
    request = store.enqueue_processor_request(
        processor_request("/v1/workflows/dispatch")
    )
    claimed = store.claim_active_processor_requests(
        "pod-a:1", now=datetime.now(timezone.utc), lease_duration=REQUEST_LEASE, limit=1
    )[0]
    processor = ProcessorCoordinator(
        store,
        owner_id="pod-a:1",
        internal_token="processor-token",
        active_consumers=True,
    )

    class Response:
        status = 200
        headers = {"Content-Type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b"{}"

    monkeypatch.setattr(
        store,
        "complete_active_processor_request",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("persistent completion failure")
        ),
    )
    monkeypatch.setattr(
        "gpu_fault.processor.coordinator.urlopen", lambda *_args, **_kwargs: Response()
    )

    with pytest.raises(RuntimeError, match="persistent completion failure"):
        processor._execute(claimed, deadline=time.monotonic() + 10)
    runtime = processor.metrics_snapshot()
    current = store.get_processor_request(request.request_id)

    assert current.status is ProcessorRequestStatus.PENDING
    assert current.lease_owner is None
    assert runtime["completion_retries_total"] == 2
    assert runtime["completion_failures_total"] == 1


def test_processor_reports_in_flight_phase() -> None:
    store = build_store()
    request = store.enqueue_processor_request(
        processor_request("/v1/gpu-events/xid", body=b'{"node_id":"node-a","xid":79}')
    )
    claimed = store.claim_active_processor_requests(
        "pod-a:1", now=datetime.now(timezone.utc), lease_duration=REQUEST_LEASE, limit=1
    )[0]
    processor = ProcessorCoordinator(
        store,
        owner_id="pod-a:1",
        internal_token="processor-token",
        active_consumers=True,
    )
    processor._request_started(claimed)
    processor._set_request_phase(claimed.request_id, "replay_http")

    snapshot = processor.in_flight_snapshot()
    runtime = processor.metrics_snapshot()
    processor._request_finished(claimed.request_id)
    completed_runtime = processor.metrics_snapshot()

    assert snapshot[0]["request_id"] == request.request_id
    assert snapshot[0]["phase"] == "replay_http"
    assert snapshot[0]["phase_elapsed_seconds"] >= 0
    assert runtime["in_flight_by_phase"]["replay_http"]["count"] == 1
    assert completed_runtime["lane_holder_by_path"]["/v1/gpu-events/xid"]["count"] == 1


def test_idle_streams_back_off_with_a_short_fault_ceiling() -> None:
    """An empty queue must not keep costing one claim per stream.

    Six worker replicas polling seven streams every 100ms committed
    about 500 transactions a second against an empty queue, over half
    of what the Aurora writer sustains at 16 ACU. Telemetry streams
    that come back empty are skipped for a doubling interval. Faults use
    a shorter 500ms ceiling rather than busy-polling every 100ms.
    """
    store = build_store()
    processor = ProcessorCoordinator(
        store,
        owner_id="pod-a:1",
        internal_token="token-" + "x" * 32,
        pools=ProcessorPoolSettings(
            fault_worker_count=2,
            observation_worker_count=2,
            gpu_telemetry_worker_count=2,
            host_telemetry_worker_count=2,
        ),
        lease=ProcessorLeaseSettings(
            poll_seconds=0.1,
            idle_backoff_max_seconds=2.0,
            fault_idle_backoff_max_seconds=0.5,
        ),
        active_consumers=True,
    )
    calls: list[frozenset[str] | None] = []
    original = store.claim_active_processor_requests

    def record(owner_id, **kwargs):
        include = kwargs.get("include_paths")
        calls.append(frozenset(include) if include else None)
        return original(owner_id, **kwargs)

    store.claim_active_processor_requests = record
    available = {"fault": 2, "observation": 2, "gpu": 2, "host": 2}
    processor._claim_active_by_pool(available, lease_duration=timedelta(seconds=30))
    first_round = len(calls)
    assert first_round > 1
    calls.clear()

    # Straight back in: every empty stream, including fault, is backed off.
    processor._claim_active_by_pool(available, lease_duration=timedelta(seconds=30))
    assert calls == []

    # Once the short fault backoff expires, the new fault is claimed.
    request = processor_request("/v1/gpu-events/xid", body=b'{"node_id":"node-a"}')
    store.enqueue_processor_request(request)
    processor._stream_idle_until.pop("fault", None)
    claimed = processor._claim_active_by_pool(
        available, lease_duration=timedelta(seconds=30)
    )
    assert [item.request_id for item in claimed] == [request.request_id]

    # The interval doubles while the stream stays empty, and is capped.
    processor._stream_idle_until.clear()
    for _ in range(8):
        processor._claim_active_by_pool(available, lease_duration=timedelta(seconds=30))
        processor._stream_idle_until.clear()
    assert processor._stream_idle_interval["host-telemetry"] == pytest.approx(2.0)
    assert processor._stream_idle_interval["fault"] == pytest.approx(0.5)
    processor._pools_busy = True
    processor._stream_idle_until.clear()
    processor._claim_active_by_pool(available, lease_duration=timedelta(seconds=30))
    assert processor._stream_idle_interval["fault"] == pytest.approx(0.1)


def test_processor_notifications_wake_claims_and_extend_fallback(monkeypatch) -> None:
    store = build_store()
    ready = Event()
    request_id = "request-notification-test"
    shard = processor_partition_id(request_id, 8)

    def listen(stop, owner_id, shard_count, notify, on_state, **_kwargs):
        assert owner_id == "pod-a:1"
        assert shard_count == 8
        on_state(True, shard)
        ready.set()
        notify(
            f'{{"priority":0,"path":"/v1/gpu-events/xid","request_id":"{request_id}"}}'
        )
        stop.wait()
        on_state(False, None)

    monkeypatch.setattr(
        store, "listen_processor_queue_notifications", listen, raising=False
    )
    processor = ProcessorCoordinator(
        store,
        owner_id="pod-a:1",
        internal_token="token-" + "x" * 32,
        pools=ProcessorPoolSettings(fault_worker_count=1),
        lease=ProcessorLeaseSettings(
            poll_seconds=0.1,
            fault_idle_backoff_max_seconds=0.5,
            processor_notification_fallback_seconds=5.0,
        ),
        active_consumers=True,
    )
    thread = Thread(target=processor.run_queue_notifications)
    thread.start()
    try:
        assert ready.wait(timeout=2)
        assert processor.metrics_snapshot()["notifications_enabled"] == 1
        assert processor.metrics_snapshot()["notifications_received_total"] == 1
        processor._notification_pending_streams.clear()
        available = {"fault": 1, "observation": 0, "gpu": 0, "host": 0}
        processor._stream_idle_interval["fault"] = 4.0
        processor._claim_active_by_pool(available, lease_duration=timedelta(seconds=30))
        assert processor._stream_idle_interval["fault"] == pytest.approx(5.0)
        processor.notify_work_available(
            '{"partition":0,"priority":0,'
            '"path":"/v1/gpu-events/xid",'
            f'"request_id":"{request_id}"'
            "}"
        )
        assert processor._notification_pending_streams == {"fault"}
    finally:
        processor.stop()
        thread.join(timeout=2)

    assert not thread.is_alive()
    processor._notification_pending_streams.clear()
    processor._set_notification_state(True, (shard + 1) % 8)
    filtered_before = processor.metrics_snapshot()["notifications_filtered_total"]
    processor.notify_work_available(
        '{"partition":0,"priority":0,'
        '"path":"/v1/gpu-events/xid",'
        f'"request_id":"{request_id}"'
        "}"
    )
    assert not processor._notification_pending_streams
    assert (
        processor.metrics_snapshot()["notifications_filtered_total"]
        == filtered_before + 1
    )


def test_claimed_stream_resets_its_idle_backoff() -> None:
    store = build_store()
    processor = ProcessorCoordinator(
        store,
        owner_id="pod-a:1",
        internal_token="token-" + "x" * 32,
        pools=ProcessorPoolSettings(
            fault_worker_count=1, host_telemetry_worker_count=1
        ),
        lease=ProcessorLeaseSettings(poll_seconds=0.1, idle_backoff_max_seconds=2.0),
        active_consumers=True,
    )
    available = {"fault": 1, "observation": 0, "gpu": 0, "host": 1}
    processor._claim_active_by_pool(available, lease_duration=timedelta(seconds=30))
    assert "host-telemetry" in processor._stream_idle_until

    store.enqueue_processor_request(
        processor_request(
            "/v1/collector-events/host-telemetry", body=b'{"node_id":"node-a"}'
        )
    )
    # Pretend the backoff elapsed rather than sleeping through it.
    processor._stream_idle_until.clear()
    claimed = processor._claim_active_by_pool(
        available, lease_duration=timedelta(seconds=30)
    )
    assert [item.path for item in claimed] == ["/v1/collector-events/host-telemetry"]
    assert "host-telemetry" not in processor._stream_idle_until
    assert "host-telemetry" not in processor._stream_idle_interval


def test_lane_blocked_backlog_caps_the_claim_backoff() -> None:
    """A backlog behind a leased lane must not read as an idle queue.

    The claim skips every request whose lane is leased, so a tail piled
    on a handful of lanes returns nothing - and the idle backoff then
    grew to two seconds, pinning drain throughput at rows-per-claim /
    2s (measured: ~7 rows/s against ~165 accepted/s during the burst).
    While the backlog is only lane-blocked the ceiling is the much
    shorter busy interval.
    """

    store = build_store()
    processor = ProcessorCoordinator(
        store,
        owner_id="pod-a:1",
        internal_token="token-" + "x" * 32,
        pools=ProcessorPoolSettings(
            fault_worker_count=1, host_telemetry_worker_count=1
        ),
        lease=ProcessorLeaseSettings(
            poll_seconds=0.1, idle_backoff_max_seconds=2.0, busy_backoff_max_seconds=0.4
        ),
        active_consumers=True,
    )
    for index in range(3):
        store.enqueue_processor_request(
            processor_request(
                "/v1/collector-events/host-telemetry",
                body=json.dumps({"node_id": "node-a", "sequence": index}).encode(),
            )
        )
    # A sibling replica holds the only lane these three share.
    sibling = store.claim_active_processor_requests(
        "pod-b:1",
        now=datetime.now(timezone.utc),
        lease_duration=timedelta(seconds=30),
        limit=10,
    )
    assert len(sibling) == 1

    available = {"fault": 1, "observation": 0, "gpu": 0, "host": 1}
    for _ in range(8):
        assert (
            processor._claim_active_by_pool(
                available, lease_duration=timedelta(seconds=30)
            )
            == []
        )
        # Pretend each backoff elapsed rather than sleeping through it.
        processor._stream_idle_until.clear()
    assert processor._stream_idle_interval["host-telemetry"] == pytest.approx(0.4)
    claim = processor.metrics_snapshot()["claim"]
    assert claim["lane_blocked"] > 0
    assert claim["empty"] == claim["rounds"]

    # Once the lane frees, the stream is claimable again and resets.
    store.complete_active_processor_request(
        sibling[0].request_id,
        "pod-b:1",
        sibling[0].leader_epoch,
        sibling[0].lease_token,
        response_status=200,
        response_content_type="application/json",
        response_body_base64="",
    )
    claimed = processor._claim_active_by_pool(
        available, lease_duration=timedelta(seconds=30)
    )
    assert len(claimed) == 1
    assert "host-telemetry" not in processor._stream_idle_interval


def test_idle_queue_keeps_the_full_claim_backoff() -> None:
    """The probe must not undo the idle-poll saving on an empty queue."""

    store = build_store()
    processor = ProcessorCoordinator(
        store,
        owner_id="pod-a:1",
        internal_token="token-" + "x" * 32,
        pools=ProcessorPoolSettings(
            fault_worker_count=1, host_telemetry_worker_count=1
        ),
        lease=ProcessorLeaseSettings(
            poll_seconds=0.1, idle_backoff_max_seconds=2.0, busy_backoff_max_seconds=0.4
        ),
        active_consumers=True,
    )
    available = {"fault": 1, "observation": 0, "gpu": 0, "host": 1}
    for _ in range(8):
        processor._claim_active_by_pool(available, lease_duration=timedelta(seconds=30))
        processor._stream_idle_until.clear()
    assert processor._stream_idle_interval["host-telemetry"] == pytest.approx(2.0)
    assert processor.metrics_snapshot()["claim"]["lane_blocked"] == 0


def test_processor_completes_expired_inventory_as_stale() -> None:
    store = build_store()
    processor = ProcessorCoordinator(
        store,
        owner_id="processor-a",
        internal_token="token-" + "x" * 32,
        active_consumers=True,
        stale=ProcessorStaleSettings(gpu_inventory_stale_seconds=60),
    )
    observed_at = datetime.now(timezone.utc) - timedelta(minutes=5)
    request = processor_request(
        "/v1/collector-events/gpu-inventory",
        body=json.dumps(
            {
                "cluster_id": "cluster-a",
                "node_id": "node-a",
                "observed_at": observed_at.isoformat(),
            }
        ).encode(),
    )
    accepted, reason = store.try_enqueue_processor_request(
        request, max_depth=10, max_cluster_depth=10
    )
    assert accepted is not None
    assert reason is None
    claimed = store.claim_active_processor_requests(
        "processor-a",
        now=datetime.now(timezone.utc),
        lease_duration=timedelta(seconds=30),
        limit=1,
    )
    assert len(claimed) == 1

    assert processor._complete_if_stale(claimed[0])

    completed = store.get_processor_request(request.request_id)
    assert completed.status is ProcessorRequestStatus.COMPLETED
    assert completed.response_status == 200
    assert json.loads(completed.response_body())["status"] == ("STALE_SUPERSEDED")
    metrics = processor.metrics_snapshot()
    assert metrics["stale_superseded_total"] == 1
    assert metrics["stale_superseded_by_path"] == {
        "/v1/collector-events/gpu-inventory": 1
    }


def test_processor_only_expires_healthy_telemetry_summaries() -> None:
    processor = ProcessorCoordinator(
        build_store(),
        owner_id="processor-a",
        internal_token="token-" + "x" * 32,
        active_consumers=False,
        stale=ProcessorStaleSettings(health_summary_stale_seconds=60),
    )
    observed_at = datetime.now(timezone.utc) - timedelta(minutes=5)

    def request(path: str, payload: dict) -> ProcessorRequest:
        return processor_request(
            path,
            body=json.dumps(
                {
                    "cluster_id": "cluster-a",
                    "node_id": "node-a",
                    "observed_at": observed_at.isoformat(),
                    **payload,
                }
            ).encode(),
        )

    gpu_summary = request(
        "/v1/collector-events/gpu-metrics", {"edge_filter_reasons": ["health-summary"]}
    )
    gpu_edge = request(
        "/v1/collector-events/gpu-metrics",
        {"edge_filter_reasons": ["threshold:gpu_temperature"]},
    )
    host_summary = request(
        "/v1/collector-events/host-telemetry",
        {"edge_filter_reasons": ["health-summary"], "collection_errors": []},
    )
    host_error = request(
        "/v1/collector-events/host-telemetry",
        {
            "edge_filter_reasons": ["health-summary"],
            "collection_errors": ["rdma inventory failed"],
        },
    )
    xid = request("/v1/gpu-events/xid", {})
    sxid = request("/v1/gpu-events/sxid", {})

    assert processor._stale_disposition(gpu_summary) is not None
    assert processor._stale_disposition(host_summary) is not None
    assert processor._stale_disposition(gpu_edge) is None
    assert processor._stale_disposition(host_error) is None
    assert processor._stale_disposition(xid) is None
    assert processor._stale_disposition(sxid) is None


def test_processor_expires_context_only_after_newer_state_exists() -> None:
    store = build_store()
    processor = ProcessorCoordinator(
        store,
        owner_id="processor-a",
        internal_token="token-" + "x" * 32,
        active_consumers=False,
        stale=ProcessorStaleSettings(
            observation_stale_seconds=60, training_progress_stale_seconds=60
        ),
    )
    observed_at = datetime.now(timezone.utc) - timedelta(minutes=5)
    observation = attempt_observation(
        "job-a", "attempt-a", observed_at, runtime_profile_version="hyperpod-v1"
    )
    observation_request = processor_request(
        "/v1/workload-observations", body=observation.model_dump_json().encode()
    )
    assert processor._stale_disposition(observation_request) is None

    store.save_attempt_observation(
        copy_model(observation, observed_at=datetime.now(timezone.utc))
    )
    assert processor._stale_disposition(observation_request) is not None

    heartbeat = TrainingProgressHeartbeat(
        cluster_id="cluster-a",
        attempt_id="attempt-a",
        rank=0,
        observed_at=observed_at,
        step=1,
    )
    heartbeat_request = processor_request(
        "/v1/training-progress", body=heartbeat.model_dump_json().encode()
    )
    assert processor._stale_disposition(heartbeat_request) is None
    store.observe_training_progress(
        copy_model(heartbeat, observed_at=datetime.now(timezone.utc), step=2)
    )
    assert processor._stale_disposition(heartbeat_request) is not None


def test_event_loop_lag_monitor_advances(monkeypatch) -> None:
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_MODE", "direct")
    monkeypatch.setenv("GPU_FAULT_SERVICE_ROLE", "ingress")
    app = create_app(build_context())

    async def scenario() -> str:
        async with app.router.lifespan_context(app):
            await asyncio.sleep(0.65)
            async with asgi_client(app) as client:
                response = await client.get("/metrics")
                return response.text

    metrics = asyncio.run(scenario())
    line = next(
        item
        for item in metrics.splitlines()
        if item.startswith("gpu_fault_event_loop_lag_seconds_count ")
    )
    assert int(line.rsplit(" ", 1)[1]) >= 1
