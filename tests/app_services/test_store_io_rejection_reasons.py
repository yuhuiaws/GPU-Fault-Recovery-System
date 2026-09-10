"""Every Store I/O rejection lands in exactly one ``reason`` and the Pod-level
merge keeps the reasons apart.

``gpu_fault_store_io_rejections_total`` used to count two different things
under one series: the bounded executor refusing work because its lane was
full, and any retryable PostgreSQL failure (lost connection, serialization
failure, deadlock) the request layer answers as 503 + Retry-After instead of
500 (F-E3). ``GpuFaultStoreIoRejected`` read the sum and paged critical for
three 503s during one Aurora failover. The counter now carries ``reason`` from
the closed set in :data:`gpu_fault.async_store.STORE_IO_REJECTION_REASONS`,
and the alert rules select on it.
"""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace

import pytest

from gpu_fault.app import ApplicationContext, create_app, process_metrics
from gpu_fault.async_store import (
    REQUEST_DEADLINE,
    STORE_IO_REJECTION_REASONS,
    AsyncStoreExecutor,
    RequestDeadlineExceeded,
    StoreIoCapacityExceeded,
)
from tests._builders import asgi_client, build_store
from tests.store.test_async_store import _admission_batcher

FAMILY = "gpu_fault_store_io_rejections_total"


def _zeroed(**counts: int) -> dict[str, int]:
    expected = {reason: 0 for reason in STORE_IO_REJECTION_REASONS}
    expected.update(counts)
    return expected


def test_the_reason_set_is_closed_and_every_reason_starts_at_zero() -> None:
    executor = AsyncStoreExecutor(
        workers=1, max_in_flight=1, admission_timeout_seconds=1
    )
    try:
        assert STORE_IO_REJECTION_REASONS == (
            "backend_unavailable",
            "capacity",
            "deadline",
        )
        assert executor.rejected_by_reason == _zeroed()
        with pytest.raises(ValueError, match="unknown store I/O rejection reason"):
            executor.record_rejection("writer_unavailable")
        assert executor.rejected_total == 0, "a refused reason must not count"
    finally:
        executor.close()


def test_a_full_lane_past_the_admission_timeout_is_capacity() -> None:
    executor = AsyncStoreExecutor(
        workers=1, max_in_flight=1, admission_timeout_seconds=0.05
    )
    release = threading.Event()

    def hold_the_slot() -> None:
        release.wait(5)

    async def scenario() -> None:
        holder = asyncio.ensure_future(executor.run(hold_the_slot))
        for _ in range(200):
            if executor.in_flight == 1:
                break
            await asyncio.sleep(0.005)
        assert executor.in_flight == 1
        with pytest.raises(StoreIoCapacityExceeded) as info:
            await executor.run(lambda: None)
        assert not isinstance(info.value, RequestDeadlineExceeded), (
            "a store failure must not be reported as a deadline rejection"
        )
        release.set()
        await holder

    try:
        asyncio.run(scenario())
        assert executor.rejected_by_reason == _zeroed(capacity=1)
        assert executor.rejected_total == 1
    finally:
        release.set()
        executor.close()


def test_a_caller_whose_budget_already_passed_is_deadline() -> None:
    executor = AsyncStoreExecutor(
        workers=1, max_in_flight=1, admission_timeout_seconds=1
    )

    async def scenario() -> None:
        token = REQUEST_DEADLINE.set(time.monotonic() - 1)
        try:
            with pytest.raises(RequestDeadlineExceeded):
                await executor.run(lambda: None)
        finally:
            REQUEST_DEADLINE.reset(token)

    try:
        asyncio.run(scenario())
        assert executor.rejected_by_reason == _zeroed(deadline=1)
        assert executor.rejected_total == 1
    finally:
        executor.close()


def test_a_writer_outage_seen_by_the_executor_is_backend_unavailable() -> None:
    executor = AsyncStoreExecutor(
        workers=1, max_in_flight=1, admission_timeout_seconds=1
    )
    operational_error = type(
        "OperationalError", (Exception,), {"__module__": "psycopg", "sqlstate": None}
    )

    def lost_connection() -> None:
        raise operational_error("server closed the connection unexpectedly")

    async def scenario() -> None:
        with pytest.raises(
            StoreIoCapacityExceeded, match="writer is temporarily unavailable"
        ):
            await executor.run(lost_connection)

    try:
        asyncio.run(scenario())
        assert executor.rejected_by_reason == _zeroed(backend_unavailable=1)
        assert executor.rejected_total == 1
    finally:
        executor.close()


def test_a_retryable_error_wrapped_by_the_admission_batch_is_backend_unavailable() -> (
    None
):
    """The F-E3 wrap runs after the executor returned the batch's exception,
    so the executor never counted it; the batcher has to, and under the reason
    the alert rules treat as the writer answering, not the lane being full."""
    psycopg = pytest.importorskip("psycopg")
    executor = AsyncStoreExecutor(
        workers=1, max_in_flight=2, admission_timeout_seconds=2
    )

    def admit_batch(items, **_kwargs):
        raise psycopg.errors.lookup("40P01")("deadlock detected")

    batcher = _admission_batcher(
        SimpleNamespace(), executor, flush_delay_seconds=0, admit_batch=admit_batch
    )

    async def scenario() -> None:
        with pytest.raises(StoreIoCapacityExceeded) as info:
            await asyncio.wait_for(
                batcher.submit(SimpleNamespace(cluster_id="a", path="/v1/x")),
                timeout=2.0,
            )
        assert not isinstance(info.value, RequestDeadlineExceeded), (
            "a store failure must not be reported as a deadline rejection"
        )

    try:
        asyncio.run(scenario())
        assert executor.rejected_by_reason == _zeroed(backend_unavailable=1)
        assert executor.rejected_total == 1
    finally:
        executor.close()


def test_the_scrape_exports_one_series_per_reason_and_no_process_label() -> None:
    app = create_app(ApplicationContext(store=build_store()))
    app.state.store_io.record_rejection("capacity")
    app.state.store_io.record_rejection("backend_unavailable")
    app.state.store_io.record_rejection("backend_unavailable")

    async def fetch() -> str:
        async with asgi_client(app) as client:
            response = await client.get("/metrics")
            assert response.status_code == 200, response.text
            return response.text

    metrics = asyncio.run(fetch())
    series = [line for line in metrics.splitlines() if line.startswith(FAMILY)]

    assert series == [
        f'{FAMILY}{{reason="backend_unavailable"}} 2',
        f'{FAMILY}{{reason="capacity"}} 1',
        f'{FAMILY}{{reason="deadline"}} 0',
    ], series
    assert "process_id=" not in metrics
    assert metrics.count(f"# TYPE {FAMILY} counter") == 1


def test_the_pod_merge_adds_each_reason_across_processes_separately() -> None:
    """Four uvicorn processes each keep their own counts; the answering process
    sums them per reason, so the alert sees one Pod-wide series per reason and
    a reason only one process ever hit still shows up."""
    header = [
        f"# HELP {FAMILY} Store calls rejected, by reason.",
        f"# TYPE {FAMILY} counter",
    ]
    local = process_metrics.parse_lines(
        header
        + [
            f'{FAMILY}{{reason="backend_unavailable"}} 2',
            f'{FAMILY}{{reason="capacity"}} 0',
            f'{FAMILY}{{reason="deadline"}} 0',
        ]
    )
    sibling = process_metrics.parse_lines(
        header
        + [
            f'{FAMILY}{{reason="backend_unavailable"}} 1',
            f'{FAMILY}{{reason="capacity"}} 4',
            f'{FAMILY}{{reason="deadline"}} 0',
        ]
    )
    sibling.slot = 1
    sibling.pid = 4243

    lines = process_metrics.aggregate(local, [sibling])
    series = [line for line in lines if line.startswith(FAMILY)]

    assert series == [
        f'{FAMILY}{{reason="backend_unavailable"}} 3',
        f'{FAMILY}{{reason="capacity"}} 4',
        f'{FAMILY}{{reason="deadline"}} 0',
    ], series
    assert lines.count(f"# TYPE {FAMILY} counter") == 1
