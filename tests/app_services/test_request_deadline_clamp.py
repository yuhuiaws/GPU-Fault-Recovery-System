"""The synchronous wait honours the request's own deadline, and an expired
deadline is reported as what it is.

FINAL-建议汇总 F-E1 (P0-74A, P1-74C, P2-74E). ``_poll_for_response`` computed
its own 115 s horizon from ``processor_response_timeout_seconds`` and ignored
``REQUEST_DEADLINE`` (15/30 s), so the configured timeout could never take
effect and the client got a store-capacity 503 without a request id instead.
``_try_spool`` *replaced* the deadline with its own budget rather than
tightening it.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import time
from types import SimpleNamespace

from gpu_fault.app.middleware import dispatch
from gpu_fault.async_store import (
    REQUEST_DEADLINE,
    AsyncStoreExecutor,
    RequestDeadlineExceeded,
    StoreIoCapacityExceeded,
)
from gpu_fault.processor import ProcessorRequestStatus
from tests._builders import processor_request
from tests.processor.test_processor_response_wait import _dependencies


def _run(coroutine_factory, *, deadline_in: float | None):
    async def scenario():
        token = None
        if deadline_in is not None:
            token = REQUEST_DEADLINE.set(time.monotonic() + deadline_in)
        try:
            return await coroutine_factory()
        finally:
            if token is not None:
                REQUEST_DEADLINE.reset(token)

    return asyncio.run(scenario())


def test_the_synchronous_wait_is_clamped_to_the_request_deadline():
    store = SimpleNamespace(
        get_processor_request=lambda _request_id: SimpleNamespace(
            status=ProcessorRequestStatus.PENDING
        )
    )
    dependencies = _dependencies(store, timeout_seconds=5.0)
    item = processor_request("/v1/gpu-events/xid")
    started = time.monotonic()

    response = _run(
        lambda: dispatch._poll_for_response(item, dependencies, None), deadline_in=0.05
    )

    assert time.monotonic() - started < 1.0
    assert response.status_code == 503
    body = json.loads(response.body)
    assert body["detail"] == "processor response timed out"
    assert body["processor_request_id"] == item.request_id


def test_an_expired_deadline_is_not_reported_as_store_capacity():
    def get_processor_request(_request_id):
        raise StoreIoCapacityExceeded(
            "request deadline exceeded before store I/O admission"
        )

    store = SimpleNamespace(get_processor_request=get_processor_request)
    dependencies = _dependencies(store, timeout_seconds=5.0)
    item = processor_request("/v1/gpu-events/xid")

    response = _run(
        lambda: dispatch._poll_for_response(item, dependencies, None), deadline_in=-0.01
    )

    assert response.status_code == 503
    body = json.loads(response.body)
    assert body["detail"] == "processor response timed out"
    assert body["processor_request_id"] == item.request_id


def test_the_spool_tightens_the_deadline_instead_of_replacing_it():
    seen: list[float | None] = []

    class RecordingBatcher:
        async def submit(self, item):
            seen.append(REQUEST_DEADLINE.get())
            return item, "queued"

    store = SimpleNamespace()
    dependencies = dataclasses.replace(
        _dependencies(store, timeout_seconds=5.0),
        telemetry_spool_enabled=True,
        telemetry_spool_batcher=RecordingBatcher(),
        telemetry_request_budget_seconds=30.0,
    )
    item = processor_request("/v1/collector-events/host-telemetry")
    prepared = dispatch.PreparedProcessorRequest(
        item=item, body=b"{}", store_pool=None, server_timing={}
    )
    before = time.monotonic()

    _run(lambda: dispatch._try_spool(prepared, dependencies), deadline_in=0.5)

    assert seen and seen[0] is not None
    assert seen[0] <= before + 0.6, "the 30 s spool budget replaced a 0.5 s deadline"


def test_the_store_executor_names_an_expired_deadline():
    executor = AsyncStoreExecutor(
        workers=1, max_in_flight=1, admission_timeout_seconds=1.0
    )

    async def scenario():
        token = REQUEST_DEADLINE.set(time.monotonic() - 0.01)
        try:
            await executor.run(lambda: None)
        finally:
            REQUEST_DEADLINE.reset(token)

    try:
        asyncio.run(scenario())
    except RequestDeadlineExceeded:
        pass
    else:  # pragma: no cover - the assertion below reports it
        raise AssertionError("an expired deadline was admitted")

    assert executor.rejected_total == 1
    assert executor.rejected_by_reason["deadline"] == 1
    assert executor.rejected_by_reason["capacity"] == 0
