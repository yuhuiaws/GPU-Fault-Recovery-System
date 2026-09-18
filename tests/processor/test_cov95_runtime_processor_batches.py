"""Batch failures, stale evidence and competing owners through consumer entrypoints."""

from __future__ import annotations

import json
from datetime import timedelta

import pytest

from gpu_fault.channel_registry import (
    GPU_INVENTORY_PATH,
    GPU_METRICS_PATH,
    NVIDIA_KERNEL_PATH,
    WORKLOAD_OBSERVATIONS_PATH,
)
from tests._builders import attempt_observation, container_observation
from tests.collectors import _cov95_runtime_collect as common
from tests.processor import _cov95_runtime_processor as support

isolated_runtime = common.isolated_runtime
runtime = support.runtime


def observation_payload():
    return attempt_observation(
        "job-a",
        "attempt-a",
        common.NOW,
        containers=[
            container_observation("pod-a", "pod-a", 0, "node-a", gpu_uuids=["GPU-a"])
        ],
    ).model_dump(mode="json")


@pytest.mark.parametrize("phase", ["save", "renew-start"])
@pytest.mark.parametrize("release_failure", [False, True])
def test_observation_batch_startup_and_write_errors_finalize_every_local_claim(
    runtime, monkeypatch, phase, release_failure, caplog
):
    processor = runtime.make_processor()
    item = support.enqueue(processor, WORKLOAD_OBSERVATIONS_PATH, observation_payload())

    def fail_save(*args):
        raise OSError("fake observation write unavailable")

    def fail_release(*args, **kwargs):
        raise OSError("fake release unavailable")

    if phase == "save":
        monkeypatch.setattr(
            processor.store, "save_attempt_observations_batch", fail_save
        )
    else:
        runtime.thread_start_error = RuntimeError("fake renewal unavailable")
    if release_failure:
        monkeypatch.setattr(
            processor.store, "release_active_processor_request", fail_release
        )

    processor.run_processor()

    state = processor.store.get_processor_request(item.request_id)
    assert state.status == ("LEASED" if release_failure else "PENDING")
    assert processor.metrics_snapshot()["processed"] == {"success": 0, "error": 1}
    assert processor.in_flight_snapshot() == []
    assert processor.store.list_attempt_observation_states("cluster-a") == []
    if release_failure:
        assert "fake release unavailable" in caplog.text


@pytest.mark.parametrize("path", [WORKLOAD_OBSERVATIONS_PATH, GPU_METRICS_PATH])
def test_batch_completion_cannot_release_a_competing_owner_lane(
    runtime, monkeypatch, path
):
    processor = runtime.make_processor()
    payload = (
        observation_payload()
        if path == WORKLOAD_OBSERVATIONS_PATH
        else {"node_id": "node-a"}
    )
    item = support.enqueue(processor, path, payload)
    complete = processor.store.complete_active_processor_requests_batch
    competing = []

    def race(completions):
        row = completions[0]
        processor.store.release_active_processor_request(
            row["request_id"], row["owner_id"], row["lane_epoch"], row["lease_token"]
        )
        competing.extend(
            processor.store.claim_active_processor_requests(
                "competing-owner",
                now=runtime.clock.now(),
                lease_duration=timedelta(seconds=30),
                limit=1,
            )
        )
        return complete(completions)

    monkeypatch.setattr(
        processor.store, "complete_active_processor_requests_batch", race
    )
    monkeypatch.setattr(
        "gpu_fault.processor.coordinator.urlopen",
        lambda *args, **kwargs: common.Response(
            {"results": [{"request_id": item.request_id, "status": 200, "body": {}}]}
        ),
    )
    processor.run_processor()

    state = processor.store.get_processor_request(item.request_id)
    assert state.status == "LEASED"
    assert state.lease_owner == "competing-owner"
    assert state.lease_token == competing[0].lease_token
    assert processor.metrics_snapshot()["processed"]["error"] == 1
    assert processor.in_flight_snapshot() == []


@pytest.mark.parametrize("payload", [{}, {"unexpected": True}])
def test_invalid_observation_is_replayed_to_its_validator_not_saved_as_observation(
    runtime, monkeypatch, payload
):
    processor = runtime.make_processor()
    item = support.enqueue(processor, WORKLOAD_OBSERVATIONS_PATH, payload)
    paths = []

    def send(request, **kwargs):
        paths.append(request.full_url)
        return common.Response({"detail": "invalid observation"}, status=422)

    monkeypatch.setattr("gpu_fault.processor.coordinator.urlopen", send)
    processor.run_processor()
    state = processor.store.get_processor_request(item.request_id)
    assert state.response_status == 422
    assert paths == [processor.local_url + WORKLOAD_OBSERVATIONS_PATH]
    assert processor.store.list_attempt_observation_states("cluster-a") == []


@pytest.mark.parametrize(
    "body",
    [
        b"[]",
        b'{"observed_at":null}',
        b'{"observed_at":""}',
        b'{"observed_at":"invalid"}',
    ],
)
def test_unusable_sample_time_does_not_silently_retire_the_request(
    runtime, monkeypatch, body
):
    processor = runtime.make_processor()
    item = support.enqueue(processor, GPU_METRICS_PATH, body=body)
    requests = []

    def send(request, **kwargs):
        requests.append(json.loads(request.data))
        return common.Response(
            {"results": [{"request_id": item.request_id, "status": 422, "body": {}}]}
        )

    monkeypatch.setattr("gpu_fault.processor.coordinator.urlopen", send)
    processor.run_processor()
    assert processor.store.get_processor_request(item.request_id).response_status == 422
    assert len(requests) == 1
    assert processor.metrics_snapshot()["stale_superseded_total"] == 0


@pytest.mark.parametrize("fenced", [False, True])
@pytest.mark.parametrize("release_error", [False, True])
def test_stale_inventory_is_retired_only_after_a_successful_fenced_completion(
    runtime, monkeypatch, fenced, release_error, caplog
):
    processor = runtime.make_processor()
    item = support.enqueue(
        processor,
        GPU_INVENTORY_PATH,
        {
            "node_id": "node-a",
            "observed_at": (common.NOW - timedelta(hours=1))
            .replace(tzinfo=None)
            .isoformat(),
        },
    )
    if fenced:
        monkeypatch.setattr(
            processor.store,
            "complete_active_processor_request",
            lambda *args, **kwargs: None,
        )
    if release_error:

        def fail_release(*args, **kwargs):
            raise OSError("fake stale release unavailable")

        monkeypatch.setattr(
            processor.store, "release_active_processor_request", fail_release
        )

    processor.run_processor()

    state = processor.store.get_processor_request(item.request_id)
    metrics = processor.metrics_snapshot()
    if not fenced:
        assert state.status == "COMPLETED"
        assert json.loads(state.response_body())["status"] == "STALE_SUPERSEDED"
        assert metrics["stale_superseded_total"] == 1
    else:
        assert state.status == ("LEASED" if release_error else "PENDING")
        assert metrics["stale_superseded_total"] == 0
        assert metrics["processed"]["error"] == 1
        if release_error:
            assert "fake stale release unavailable" in caplog.text
    assert processor.in_flight_snapshot() == []


@pytest.mark.parametrize(
    "response",
    [{}, {"results": []}, {"results": [{"request_id": "unrelated", "status": 200}]}],
)
def test_batch_reply_omitting_its_claim_releases_instead_of_claiming_success(
    runtime, monkeypatch, response
):
    processor = runtime.make_processor()
    item = support.enqueue(processor, GPU_METRICS_PATH, {"node_id": "node-a"})
    monkeypatch.setattr(
        "gpu_fault.processor.coordinator.urlopen",
        lambda *args, **kwargs: common.Response(response),
    )
    processor.run_processor()
    state = processor.store.get_processor_request(item.request_id)
    assert state.status == "PENDING"
    assert state.retry_count == 1
    assert processor.metrics_snapshot()["processed"]["success"] == 0
    assert processor.in_flight_snapshot() == []


def test_legacy_leader_replay_uses_cluster_lease_without_active_lane_headers(
    runtime, monkeypatch
):
    processor = runtime.make_processor(active_consumers=False)
    item = support.enqueue(processor, NVIDIA_KERNEL_PATH, {"node_id": "node-a"})

    def acquired(event, count):
        raise common.StopLoop

    runtime.on_wait = acquired
    with pytest.raises(common.StopLoop):
        processor.run_leadership()
    runtime.on_wait = None
    runtime.waits = 0
    requests = []

    def send(request, **kwargs):
        requests.append({name.lower(): value for name, value in request.header_items()})
        return common.Response({"handled": True})

    monkeypatch.setattr("gpu_fault.processor.coordinator.urlopen", send)
    processor.run_processor()
    assert processor.store.get_processor_request(item.request_id).status == "COMPLETED"
    assert "x-gpu-fault-processor-lane-token" not in requests[0]
    assert runtime.threads == []
