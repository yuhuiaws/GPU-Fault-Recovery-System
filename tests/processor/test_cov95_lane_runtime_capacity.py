"""Claimed rows without a free worker go back to PENDING untouched.

The legacy leader claims across every pool at once (one ``limit`` for the
whole cycle), so a claim can return more rows of one pool than that pool has
free workers -- fault rows beyond the fault pool, observation batches beyond
the observation pool, telemetry batches beyond the GPU pool. Those rows are
handed back through the leader's own release (no retry charged); a row past
its retry horizon is completed through the leader's completion. The harness is
the cooperative scheduler of ``tests/processor/_cov95_runtime_processor.py``
around the real coordinator and in-memory store; its pools hold every
submitted worker (``defer``) so the dispatched rows are still claimed, never
started, when the processor stops, and the shutdown hands them back too.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any
from urllib.error import URLError

import pytest

from gpu_fault.channel_registry import (
    GPU_METRICS_PATH,
    NVIDIA_KERNEL_PATH,
    WORKLOAD_OBSERVATIONS_PATH,
)
from gpu_fault.processor import ProcessorPoolSettings
from tests._builders import attempt_observation, container_observation
from tests.collectors import _cov95_runtime_collect as common
from tests.processor import _cov95_runtime_processor as support

isolated_runtime = common.isolated_runtime
runtime = support.runtime


def _pools(**counts: int) -> ProcessorPoolSettings:
    values = {
        "fault_worker_count": 1,
        "observation_worker_count": 1,
        "gpu_telemetry_worker_count": 1,
        "host_telemetry_worker_count": 1,
    }
    values.update(counts)
    return ProcessorPoolSettings(**values)


def _legacy_leader(runtime, **kwargs: Any):
    """A legacy (non active-consumer) processor that holds the leadership."""

    processor = runtime.make_processor(active_consumers=False, **kwargs)

    def acquired(event, count):
        raise common.StopLoop

    runtime.on_wait = acquired
    with pytest.raises(common.StopLoop):
        processor.run_leadership()
    runtime.on_wait = None
    runtime.waits = 0
    runtime.wait_limit = 1
    return processor


def _dispatched(runtime, name: str) -> set[str]:
    """The request ids of the one worker the claim cycle submitted."""

    submitted = [function for function, _args, _future in runtime.submitted]
    assert submitted == [name], submitted
    (_function, (work,), _future) = runtime.submitted[0]
    batch = work if isinstance(work, list) else [work]
    return {item.request_id for item in batch}


def _released_in_order(
    processor, released: list[str], over_capacity: set[str], dispatched: set[str]
) -> None:
    """Over-capacity rows go back at dispatch, the never-started ones at stop."""

    assert set(released[: len(over_capacity)]) == over_capacity, released
    assert set(released[len(over_capacity) :]) == dispatched, released
    assert processor.metrics_snapshot()["claimed_not_started_released_total"] == len(
        dispatched
    ), "only the shutdown release counts as an unstarted claim"
    for request_id in released:
        _untouched(processor, request_id)


def _record(store, name: str, calls: list[str], monkeypatch) -> None:
    method = getattr(store, name)

    def recorded(request_id: str, *args: Any, **kwargs: Any) -> Any:
        calls.append(request_id)
        return method(request_id, *args, **kwargs)

    monkeypatch.setattr(store, name, recorded)


def _untouched(processor, request_id: str) -> None:
    state = processor.store.get_processor_request(request_id)
    assert state.status == "PENDING", state
    assert state.retry_count == 0, "a capacity release charges no retry"


def test_a_fault_row_beyond_the_fault_pool_goes_back_through_the_leader(
    runtime, monkeypatch, caplog
) -> None:
    processor = _legacy_leader(runtime, pools=_pools())
    rows = [
        support.enqueue(processor, NVIDIA_KERNEL_PATH, {"node_id": node})
        for node in ("node-a", "node-b")
    ]
    released: list[str] = []
    _record(processor.store, "release_processor_request", released, monkeypatch)
    runtime.defer = True

    with caplog.at_level(logging.WARNING, logger="gpu_fault.processor.lane_runtime"):
        processor.run_processor()

    dispatched = _dispatched(runtime, "_process")
    (over_capacity,) = {row.request_id for row in rows} - dispatched
    _released_in_order(processor, released, {over_capacity}, dispatched)
    assert f"processor request released request_id={over_capacity}" in caplog.text


def _observation(index: int) -> dict[str, Any]:
    return attempt_observation(
        f"job-{index}",
        f"attempt-{index}",
        common.NOW,
        containers=[
            container_observation(
                f"pod-{index}", f"pod-{index}", 0, f"node-{index}", gpu_uuids=["G"]
            )
        ],
    ).model_dump(mode="json")


def test_an_observation_batch_beyond_the_observation_pool_goes_back_untouched(
    runtime, monkeypatch
) -> None:
    processor = _legacy_leader(runtime, pools=_pools(fault_worker_count=40))
    rows = [
        support.enqueue(processor, WORKLOAD_OBSERVATIONS_PATH, _observation(index))
        for index in range(36)
    ]
    released: list[str] = []
    _record(processor.store, "release_processor_request", released, monkeypatch)
    runtime.defer = True

    processor.run_processor()

    dispatched = _dispatched(runtime, "_process_observation_batch")
    over_capacity = {row.request_id for row in rows} - dispatched
    assert len(over_capacity) > len(dispatched), (
        "two more observation batches were claimed than the one worker takes"
    )
    _released_in_order(processor, released, over_capacity, dispatched)


def _gpu_sample(index: int) -> dict[str, Any]:
    return {"cluster_id": "cluster-a", "node_id": f"node-{index}", "samples": []}


def test_a_telemetry_batch_beyond_the_gpu_pool_goes_back_untouched(
    runtime, monkeypatch
) -> None:
    processor = _legacy_leader(runtime, pools=_pools(fault_worker_count=24))
    rows = [
        support.enqueue(processor, GPU_METRICS_PATH, _gpu_sample(index))
        for index in range(10)
    ]
    released: list[str] = []
    _record(processor.store, "release_processor_request", released, monkeypatch)
    runtime.defer = True

    processor.run_processor()

    dispatched = _dispatched(runtime, "_process_telemetry_batch")
    over_capacity = {row.request_id for row in rows} - dispatched
    assert over_capacity, "more GPU rows were claimed than one batch holds"
    _released_in_order(processor, released, over_capacity, dispatched)


def test_a_legacy_row_failing_past_its_retry_horizon_is_completed_by_the_leader(
    runtime, monkeypatch
) -> None:
    processor = _legacy_leader(runtime, pools=_pools())
    item = support.enqueue(
        processor,
        NVIDIA_KERNEL_PATH,
        {"node_id": "node-a"},
        created_at=common.NOW - timedelta(days=2),
    )
    completed: list[str] = []
    _record(processor.store, "complete_processor_request", completed, monkeypatch)

    def unreachable(*args: Any, **kwargs: Any) -> Any:
        raise URLError("api unreachable")

    monkeypatch.setattr("gpu_fault.processor.coordinator.urlopen", unreachable)

    processor.run_processor()

    assert completed == [item.request_id], "the leader's own completion is used"
    state = processor.store.get_processor_request(item.request_id)
    assert state.status == "COMPLETED", state
    assert state.response_status == 503, state
    assert processor.metrics_snapshot()["retry_horizon_failures_total"] == 1


def test_a_shutdown_release_that_fails_is_logged_and_the_rest_continue(
    runtime, caplog
) -> None:
    processor = runtime.make_processor()
    unleased = support.enqueue(processor, NVIDIA_KERNEL_PATH, {"node_id": "node-a"})
    processor.register_claimed_requests([unleased])

    with caplog.at_level(logging.ERROR, logger="gpu_fault.processor.lane_runtime"):
        processor.release_unstarted_claims()

    assert (
        f"processor shutdown release failed request_id={unleased.request_id}"
        in caplog.text
    ), caplog.text
    assert "claimed processor request has no fencing token" in caplog.text
    assert processor.metrics_snapshot()["claimed_not_started_released_total"] == 0
    state = processor.store.get_processor_request(unleased.request_id)
    assert (state.status, state.retry_count) == ("PENDING", 0), (
        "a release that failed leaves the row as it was"
    )
