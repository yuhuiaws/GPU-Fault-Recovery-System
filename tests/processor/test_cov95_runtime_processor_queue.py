"""Queue ownership, wakeups and replay errors through public consumer loops."""

from __future__ import annotations

import io
import json
from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest

from gpu_fault.channel_registry import (
    GPU_INVENTORY_PATH,
    GPU_METRICS_PATH,
    HOST_TELEMETRY_PATH,
    NODE_LOG_PATH,
    NVIDIA_KERNEL_PATH,
    WORKLOAD_OBSERVATIONS_PATH,
)
from gpu_fault.processor import ProcessorLeaseSettings, ProcessorPoolSettings
from gpu_fault.processor.models import ProcessorLeadership, processor_partition_id
from tests.collectors import _cov95_runtime_collect as common
from tests.processor import _cov95_runtime_processor as support

isolated_runtime = common.isolated_runtime
runtime = support.runtime


@pytest.mark.parametrize(
    ("notification", "stream"),
    [
        ({"path": WORKLOAD_OBSERVATIONS_PATH}, "observation"),
        ({"path": GPU_INVENTORY_PATH}, "gpu-inventory"),
        ({"path": GPU_METRICS_PATH}, "gpu-metrics"),
        ({"path": HOST_TELEMETRY_PATH}, "host-telemetry"),
        ({"path": NODE_LOG_PATH}, "node-log"),
        ({"path": WORKLOAD_OBSERVATIONS_PATH, "priority": 0}, "fault"),
        ({"path": "/unrecognized"}, "fault"),
        ({"path": []}, "fault"),
        ("not-json", "*"),
    ],
)
def test_owned_notification_wakes_only_its_backed_off_stream(
    runtime, monkeypatch, notification, stream
):
    processor = runtime.make_processor(
        lease=ProcessorLeaseSettings(
            poll_seconds=0.01, processor_notification_shard_count=1
        )
    )
    payload = (
        notification if isinstance(notification, str) else json.dumps(notification)
    )

    def listen(stop, owner, shards, notify, state, *, on_progress):
        assert shards == 1
        state(True, 0)

    def wake(event, count):
        if count == 1:
            runtime.clock.seconds = 0.001
            processor.notify_work_available(payload)
        else:
            processor.stop()

    monkeypatch.setattr(
        processor.store, "listen_processor_queue_notifications", listen, raising=False
    )
    runtime.on_wait = wake
    processor.run_queue_notifications()
    processor.run_processor()

    metrics = processor.metrics_snapshot()
    rounds = metrics["claim"]["rounds_by_stream"]
    assert set(rounds) == {
        "fault",
        "observation",
        "gpu-inventory",
        "gpu-metrics",
        "host-telemetry",
        "node-log",
    }
    for name, count in rounds.items():
        assert count == (2 if stream in {name, "*"} else 1), (
            f"notification {notification!r} incorrectly woke {name}: {rounds}"
        )
    assert metrics["notifications_received_total"] == 1
    assert metrics["notifications_filtered_total"] == 0


def test_foreign_shard_notification_is_filtered_before_a_listener_disconnect(
    runtime, monkeypatch, caplog
):
    processor = runtime.make_processor(
        lease=ProcessorLeaseSettings(processor_notification_shard_count=2)
    )
    request_id = next(
        f"request-{index}"
        for index in range(20)
        if processor_partition_id(f"request-{index}", 2) == 1
    )

    def listen(stop, owner, shards, notify, state, *, on_progress):
        state(True, 0)
        notify(json.dumps({"request_id": request_id, "path": GPU_METRICS_PATH}))
        raise OSError("listener disconnected")

    monkeypatch.setattr(
        processor.store, "listen_processor_queue_notifications", listen, raising=False
    )
    processor.run_queue_notifications()

    metrics = processor.metrics_snapshot()
    assert metrics["notifications_filtered_total"] == 1
    assert metrics["notifications_received_total"] == 0
    assert metrics["notification_reconnects_total"] == 1
    assert metrics["notification_listener_connected"] == 0
    assert "listener disconnected" in caplog.text


@pytest.mark.parametrize("owner", ["runtime-owner", "another-owner", "unavailable"])
def test_leadership_cycle_observes_owner_and_read_failures(runtime, owner, caplog):
    calls = []

    def acquire(requested_owner, *, now, lease_duration):
        calls.append((requested_owner, lease_duration))
        if owner == "unavailable":
            raise OSError("leadership unavailable")
        return ProcessorLeadership(
            owner_id=owner,
            epoch=4,
            updated_at=now,
            lease_expires_at=now + lease_duration,
        )

    processor = runtime.make_processor(
        SimpleNamespace(acquire_processor_leadership=acquire), active_consumers=False
    )
    runtime.wait_limit = 1
    processor.run_leadership()

    assert calls == [("runtime-owner", timedelta(seconds=15))]
    assert processor.is_leader() is (owner == "runtime-owner")
    if owner == "runtime-owner":
        assert processor.leadership.epoch == 4
        runtime.clock.sleep(15)
        assert not processor.is_leader(), (
            "an expired leadership cannot authorize claims"
        )
    else:
        assert processor.leadership is None
    if owner == "unavailable":
        assert "leadership renewal failed" in caplog.text


def test_passive_consumer_without_leadership_never_claims(runtime):
    processor = runtime.make_processor(active_consumers=False)
    item = support.enqueue(processor, NVIDIA_KERNEL_PATH, {"node_id": "node-a"})

    processor.run_processor()

    assert processor.store.get_processor_request(item.request_id).status == "PENDING"
    assert runtime.submitted == []
    assert processor.metrics_snapshot()["consumer"]["cycles"] == 2


@pytest.mark.parametrize("defer", [False, True])
def test_unstarted_claim_is_released_when_pool_cannot_execute(runtime, defer, caplog):
    processor = runtime.make_processor(
        pools=ProcessorPoolSettings(
            fault_worker_count=1,
            observation_worker_count=0,
            gpu_telemetry_worker_count=0,
            host_telemetry_worker_count=0,
        )
    )
    item = support.enqueue(processor, GPU_METRICS_PATH, {"node_id": "node-a"})
    runtime.defer = defer
    runtime.wait_limit = 1
    if not defer:
        runtime.submit_error = RuntimeError("pool cannot submit")

    processor.run_processor()

    state = processor.store.get_processor_request(item.request_id)
    assert state.status == "PENDING"
    assert state.retry_count == 0
    assert processor.metrics_snapshot()["claimed_not_started_released_total"] == 1
    assert processor.in_flight_snapshot() == []
    if defer:
        assert runtime.submitted[0][2].cancelled(), (
            "shutdown must cancel work that was never started"
        )
    else:
        assert "pool cannot submit" in caplog.text


@pytest.mark.parametrize(
    ("mode", "attempts", "expected_status", "retry_count"),
    [
        ("recover", 3, "COMPLETED", 0),
        ("exhaust", 3, "PENDING", 1),
        ("late-error", 1, "PENDING", 1),
        ("late-success", 1, "PENDING", 1),
        ("permanent", 1, "COMPLETED", 0),
        ("retryable", 1, "PENDING", 1),
        ("lane-changed", 1, "PENDING", 1),
    ],
)
def test_replay_transport_outcomes_have_one_durable_disposition(
    runtime, monkeypatch, mode, attempts, expected_status, retry_count
):
    unhealthy = []
    processor = runtime.make_processor(
        lease=ProcessorLeaseSettings(
            poll_seconds=0.01, deadline_exceeded_process_threshold=1
        ),
        on_unhealthy=unhealthy.append,
    )
    item = support.enqueue(processor, NVIDIA_KERNEL_PATH, {"node_id": "node-a"})
    calls = []

    def send(request, *, timeout):
        calls.append(request.full_url)
        if mode.startswith("late"):
            runtime.clock.sleep(31)
        if mode in {"exhaust", "late-error"} or (mode == "recover" and len(calls) < 3):
            raise OSError("temporary transport failure")
        if mode in {"permanent", "retryable"}:
            raise HTTPError(
                request.full_url,
                422 if mode == "permanent" else 503,
                "replay",
                {"Content-Type": "application/json"},
                io.BytesIO(b'{"detail":"refused"}'),
            )
        headers = {"Content-Type": "application/json"}
        if mode == "lane-changed":
            headers["X-GPU-Fault-Processor-Retry"] = "lane-lease-changed"
        return common.Response({"handled": True}, headers=headers)

    monkeypatch.setattr("gpu_fault.processor.coordinator.urlopen", send)
    processor.run_processor()

    state = processor.store.get_processor_request(item.request_id)
    assert len(calls) == attempts
    assert state.status == expected_status
    assert state.retry_count == retry_count
    assert processor.in_flight_snapshot() == []
    metrics = processor.metrics_snapshot()
    assert sum(metrics["processed"].values()) == 1
    if mode.startswith("late"):
        assert len(unhealthy) == 1
        assert metrics["deadline_exceeded_total"] == 1
        assert not processor.is_healthy(), "deadline threshold must latch unhealthy"
        runtime.clock.sleep(301)
        assert processor.is_healthy(), "expired unhealthy latch must recover"
        assert processor.unhealthy_reason is None
    else:
        assert unhealthy == []
    if mode == "permanent":
        assert metrics["fault_rejections_total"] == 1
        assert state.response_status == 422


@pytest.mark.parametrize("authorized", [False, True])
def test_replay_carries_scoped_headers_and_query(runtime, monkeypatch, authorized):
    processor = runtime.make_processor(execution_token="test-execution")
    item = support.enqueue(
        processor,
        NVIDIA_KERNEL_PATH,
        {"node_id": "node-a"},
        query="cursor=42",
        execution_authorized=authorized,
        content_type=None,
    )
    sent = []

    def send(request, *, timeout):
        sent.append((request.full_url, dict(request.header_items())))
        return common.Response()

    monkeypatch.setattr("gpu_fault.processor.coordinator.urlopen", send)
    processor.run_processor()

    assert processor.store.get_processor_request(item.request_id).status == "COMPLETED"
    url, raw_headers = sent[0]
    headers = {key.lower(): value for key, value in raw_headers.items()}
    assert url.endswith(NVIDIA_KERNEL_PATH + "?cursor=42"), url
    assert headers["x-gpu-fault-processor-owner-id"] == "runtime-owner"
    assert headers["x-gpu-fault-processor-request-id"] == item.request_id
    assert headers["x-gpu-fault-cluster-id"] == "cluster-a"
    assert "x-gpu-fault-processor-lane-token" in headers
    assert ("x-gpu-fault-execution-token" in headers) is authorized
    assert "content-type" not in headers


def test_execution_authorization_without_token_releases_before_transport(runtime):
    processor = runtime.make_processor()
    item = support.enqueue(
        processor, "/v1/workflows/dispatch", execution_authorized=True
    )

    processor.run_processor()

    state = processor.store.get_processor_request(item.request_id)
    assert state.status == "PENDING"
    assert state.retry_count == 1
    assert processor.metrics_snapshot()["processed"] == {"success": 0, "error": 1}
    assert processor.in_flight_snapshot() == []


@pytest.mark.parametrize("failure", ["interlock", "lane-blocked"])
def test_claim_probe_failure_keeps_consumer_live(runtime, monkeypatch, failure, caplog):
    processor = runtime.make_processor(
        lease=ProcessorLeaseSettings(poll_seconds=0.01, busy_backoff_max_seconds=0.01)
    )

    def fail(**kwargs):
        raise OSError(f"{failure} probe unavailable")

    name = (
        "count_fault_rows_blocked_by_observation"
        if failure == "interlock"
        else "active_backlog_is_lane_blocked"
    )
    monkeypatch.setattr(processor.store, name, fail)
    processor.run_processor()

    assert processor.metrics_snapshot()["consumer"]["cycles"] >= 2
    assert processor.metrics_snapshot()["healthy"] == 1
    assert f"{failure} probe unavailable" in caplog.text


def test_invalid_settings_cannot_enable_a_zero_retry_delay(runtime):
    settings = replace(ProcessorLeaseSettings(), retry_backoff_seconds=2)
    processor = runtime.make_processor(lease=settings)
    assert processor.retry_backoff_seconds == 2
    with pytest.raises(ValueError, match="retry backoff"):
        replace(settings, retry_backoff_max_seconds=1)
