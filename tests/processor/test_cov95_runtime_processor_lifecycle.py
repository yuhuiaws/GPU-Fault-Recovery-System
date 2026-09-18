"""Real queue replay must leave every claim and renewal with a disposition."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from gpu_fault.channel_registry import GPU_METRICS_PATH, NVIDIA_KERNEL_PATH
from gpu_fault.processor import ProcessorCoordinator
from tests.collectors import _cov95_runtime_collect as collect_support
from tests.processor import _cov95_runtime_processor as support

isolated_runtime = collect_support.isolated_runtime
runtime = support.runtime


@pytest.mark.parametrize("payload", ["null", "[]", '"scalar"', "4", "true"])
def test_non_object_notification_does_not_disconnect_listener(payload):
    callbacks = []

    def listen(stop, owner, shards, notify, state, *, on_progress):
        state(True, 0)
        notify(payload)
        callbacks.append(on_progress())
        notify("not-json")

    processor = ProcessorCoordinator(
        SimpleNamespace(listen_processor_queue_notifications=listen),
        owner_id="runtime-owner",
        internal_token="test-replay",
        active_consumers=True,
    )
    processor.run_queue_notifications()

    metrics = processor.metrics_snapshot()
    assert callbacks == [0], "malformed notification must not abort the listener"
    assert metrics["notification_listener_connected"] == 1
    assert metrics["notifications_received_total"] == 2
    assert metrics["notification_reconnects_total"] == 0


@pytest.mark.parametrize("body", [b"{", b"\xff"])
def test_invalid_batch_json_releases_claim_and_stops_renewal(runtime, body, caplog):
    processor = runtime.make_processor()
    item = support.enqueue(processor, GPU_METRICS_PATH, body=body)

    processor.run_processor()

    state = processor.store.get_processor_request(item.request_id)
    assert state.status == "PENDING"
    assert state.retry_count == 1
    assert processor.in_flight_snapshot() == []
    assert processor.metrics_snapshot()["processed"]["error"] == 1
    assert runtime.threads, "the batch must have entered its renewal lifecycle"
    assert all(thread.joined for thread in runtime.threads), (
        "failed body decoding must join every started renewal thread"
    )
    assert all(thread.args[1].is_set() for thread in runtime.threads), (
        "failed body decoding must stop renewal before releasing the lane"
    )
    assert "batch execution raised" in caplog.text


@pytest.mark.parametrize("path", [NVIDIA_KERNEL_PATH, GPU_METRICS_PATH])
def test_renewal_start_failure_releases_started_request(runtime, path):
    processor = runtime.make_processor()
    item = support.enqueue(processor, path, {"node_id": "node-a"})
    runtime.thread_start_error = RuntimeError("thread capacity exhausted")

    processor.run_processor()

    state = processor.store.get_processor_request(item.request_id)
    assert state.status == "PENDING"
    assert state.retry_count == 1
    assert processor.in_flight_snapshot() == []
    assert processor.metrics_snapshot()["processed"]["error"] == 1
    assert all(pool.closed for pool in runtime.pools), (
        "consumer shutdown must drain every executor even after a start failure"
    )


def test_valid_batch_replay_completes_and_signals_without_host_calls(
    runtime, monkeypatch
):
    processor = runtime.make_processor()
    item = support.enqueue(
        processor,
        GPU_METRICS_PATH,
        {"node_id": "node-a", "edge_filter_reasons": ["metric-threshold"]},
    )
    requests = []

    def reply(request, *, timeout):
        payload = json.loads(request.data)
        requests.append((request.full_url, payload, timeout))
        return collect_support.Response(
            {"results": [{"request_id": item.request_id, "status": 200, "body": {}}]}
        )

    monkeypatch.setattr("gpu_fault.processor.coordinator.urlopen", reply)
    processor.run_processor()

    state = processor.store.get_processor_request(item.request_id)
    assert state.status == "COMPLETED"
    assert state.response_status == 200
    assert len(requests) == 1
    assert requests[0][1]["items"][0]["path"] == GPU_METRICS_PATH
    assert processor.in_flight_snapshot() == []
    assert processor.metrics_snapshot()["processed"] == {"success": 1, "error": 0}
