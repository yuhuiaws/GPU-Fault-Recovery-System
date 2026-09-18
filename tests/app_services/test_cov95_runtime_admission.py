from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException

from gpu_fault.app import admission_runtime
from gpu_fault.app.admission_runtime import AdmissionRuntimeFactory, NulInRequestBody
from gpu_fault.async_store import (
    REQUEST_DEADLINE,
    RequestDeadlineExceeded,
    StoreIoCapacityExceeded,
)
from tests._builders import build_context
from tests.app_services._cov95_runtime_admission import Admission, item
from tests.app_services._cov95_runtime_admission import (
    admission_fixture as admission_fixture,
)
from tests.regional._cov95_runtime_support import offline_runtime as offline_runtime


@pytest.mark.parametrize(
    ("environment", "message"),
    [
        ({"GPU_FAULT_CAPACITY_MANAGED_NODE_COUNT": "-1"}, "cannot be negative"),
        ({"GPU_FAULT_CAPACITY_LARGEST_CLUSTER_NODE_COUNT": "-1"}, "cannot be negative"),
        (
            {
                "GPU_FAULT_TELEMETRY_SPOOL": "true",
                "GPU_FAULT_TELEMETRY_SPOOL_ADMISSION_PARTITIONS": "0",
            },
            "partition count",
        ),
        (
            {
                "GPU_FAULT_TELEMETRY_SPOOL": "true",
                "GPU_FAULT_TELEMETRY_SPOOL_MAX_ITEM_BYTES": "0",
            },
            "item byte limit",
        ),
    ],
)
def test_invalid_capacity_and_spool_limits_are_refused_before_runtime_creation(
    monkeypatch: pytest.MonkeyPatch, environment: dict[str, str], message: str
) -> None:
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    with pytest.raises((RuntimeError, ValueError), match=message):
        AdmissionRuntimeFactory(build_context(), None).limits()


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("GPU_FAULT_PROCESSOR_ADMISSION_BATCH_SIZE", "0", "batch size"),
        ("GPU_FAULT_PROCESSOR_ADMISSION_BATCH_GROUPS", "0", "flush groups"),
        ("GPU_FAULT_PROCESSOR_ADMISSION_BATCH_DELAY_SECONDS", "-1", "delay"),
        ("GPU_FAULT_PROCESSOR_ADMISSION_PROJECTION_MARGIN", "-1", "projection margin"),
        ("GPU_FAULT_TELEMETRY_SPOOL_BATCH_SIZE", "0", "batch size"),
        ("GPU_FAULT_TELEMETRY_SPOOL_ADMISSION_PARTITIONS", "0", "partitions"),
    ],
)
def test_invalid_batch_schedule_is_refused_without_starting_store_work(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str, message: str
) -> None:
    executors = []
    original = admission_runtime.AsyncStoreExecutor

    def create(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        executors.append(result)
        return result

    monkeypatch.setattr(admission_runtime, "AsyncStoreExecutor", create)
    monkeypatch.setenv(name, value)
    try:
        with pytest.raises(ValueError, match=message):
            AdmissionRuntimeFactory(build_context(), None).build()
    finally:
        for executor in executors:
            executor.close()


def test_spool_envelope_must_fit_the_consumer_replay_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GPU_FAULT_TELEMETRY_SPOOL", "true")
    processor = SimpleNamespace(telemetry_spool_replay_batch_max_bytes=1)
    with pytest.raises(ValueError, match="replay batch byte limit"):
        AdmissionRuntimeFactory(build_context(), processor).limits()


@pytest.mark.parametrize("backend_error", [False, True])
def test_cancelled_cluster_waiter_does_not_starve_an_independent_cluster(
    monkeypatch: pytest.MonkeyPatch, admission: Admission, backend_error: bool
) -> None:
    runtime = admission.runtime

    async def scenario() -> None:
        first_started, release = asyncio.Event(), asyncio.Event()
        first, second = item("cluster-a"), item("cluster-b")
        calls = []

        async def run(function: Any, items: list[Any]) -> Any:
            cluster = items[0].cluster_id
            calls.append(cluster)
            if cluster == "cluster-a":
                first_started.set()
                await release.wait()
                if backend_error:
                    raise RuntimeError("synthetic cancelled batch failure")
            return function(items)

        monkeypatch.setattr(runtime.store_io, "run", run)
        cancelled = asyncio.create_task(runtime.admission_batcher.submit(first))
        try:
            await asyncio.wait_for(first_started.wait(), timeout=2)
            cancelled.cancel()
            with pytest.raises(asyncio.CancelledError):
                await cancelled
            queued, _reason = await asyncio.wait_for(
                runtime.admission_batcher.submit(second), timeout=2
            )
            assert queued.request_id == second.request_id
            assert calls == ["cluster-a", "cluster-b"]
            assert runtime.admission_batcher.in_flight >= 1
        finally:
            release.set()
            await admission.close_batchers()
        assert runtime.admission_batcher.in_flight == 0
        assert runtime.admission_batcher.pending_depth == 0
        assert runtime.admission_batcher.submitted_total == 2

    asyncio.run(scenario())


def test_wrong_result_count_fails_every_batch_waiter_without_leaving_pending_work(
    monkeypatch: pytest.MonkeyPatch, admission: Admission
) -> None:
    runtime = admission.runtime
    observed = []

    def malformed(items: list[Any], **kwargs: Any) -> list:
        observed.extend(items)
        return []

    monkeypatch.setattr(
        admission.context.store, "try_enqueue_processor_requests_batch", malformed
    )

    async def scenario() -> None:
        try:
            first, second = item(), item()
            results = await asyncio.wait_for(
                asyncio.gather(
                    runtime.admission_batcher.submit(first),
                    runtime.admission_batcher.submit(second),
                    return_exceptions=True,
                ),
                timeout=2,
            )
            assert len(results) == 2
            assert all(
                isinstance(result, RuntimeError) and "wrong number" in str(result)
                for result in results
            ), results
            assert observed == [first, second]
        finally:
            await admission.close_batchers()
        assert runtime.admission_batcher.pending_depth == 0
        assert runtime.admission_batcher.in_flight == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("unbounded", [False, True])
def test_batch_uses_the_last_waiters_deadline_and_restores_the_callers_context(
    monkeypatch: pytest.MonkeyPatch, admission: Admission, unbounded: bool
) -> None:
    deadline = time.monotonic() + 60
    seen = []

    async def run(function: Any, items: list[Any]) -> Any:
        seen.append(REQUEST_DEADLINE.get())
        return function(items)

    monkeypatch.setattr(admission.runtime.store_io, "run", run)

    async def submit(value: Any, bound: float | None) -> Any:
        token = REQUEST_DEADLINE.set(bound)
        try:
            return await admission.runtime.admission_batcher.submit(value)
        finally:
            REQUEST_DEADLINE.reset(token)

    async def scenario() -> None:
        try:
            results = await asyncio.gather(
                submit(item(), deadline - 1),
                submit(item(), None if unbounded else deadline),
            )
            assert len(results) == 2
        finally:
            await admission.close_batchers()
        assert REQUEST_DEADLINE.get() is None

    asyncio.run(scenario())
    assert seen == [None if unbounded else deadline]


@pytest.mark.parametrize(
    "error",
    [
        StoreIoCapacityExceeded("unit capacity"),
        RequestDeadlineExceeded("unit deadline"),
    ],
)
def test_bounded_endpoint_maps_io_failures_to_specific_retryable_http_responses(
    monkeypatch: pytest.MonkeyPatch, admission: Admission, error: Exception
) -> None:
    async def fail(*args: Any, **kwargs: Any) -> Any:
        raise error

    monkeypatch.setattr(admission.runtime.store_io, "run", fail)
    calls = []
    endpoint = admission.runtime.bounded_io_endpoint(lambda: calls.append("handler"))
    with pytest.raises(HTTPException) as raised:
        asyncio.run(endpoint())
    assert raised.value.status_code == 503
    assert raised.value.headers == {"Retry-After": "2"}
    assert raised.value.detail == (
        "request deadline exceeded"
        if isinstance(error, RequestDeadlineExceeded)
        else "store I/O capacity exceeded"
    )
    assert calls == []


@pytest.mark.parametrize(
    "body", [b'{"\\u0000":"value"}', b'{"nested":[1,false,{"message":"\\u0000"}]}']
)
def test_decoder_rejects_nul_in_keys_and_nested_values(
    admission: Admission, body: bytes
) -> None:
    with pytest.raises(NulInRequestBody, match="NUL"):
        admission.runtime.decode_json_body(body, "")


def test_decoder_empty_request_retains_the_empty_body(admission: Admission) -> None:
    assert admission.runtime.decode_json_body(b"", "") == (b"", {})
