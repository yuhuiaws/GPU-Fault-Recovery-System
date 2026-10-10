"""The ingress admission batchers take a floor and a ceiling of in-flight
flush slots from site-wide values divided among the ingress processes, move
their live limit between the two on the flush-latency EWMA and their backlog,
and stretch their coalescing window while the store is contended.

Before, every process carried its own literal (16 routine + 8 fault + 4
evidence); on 3 api-ha Pods x 4 uvicorn workers that was up to 336 concurrent
admission INSERT transactions on one table, the perf suite's load-start
convoy. A fixed site-wide cap of 24 then shed telemetry instead (4,601
requests expired in the queue while every process sat at 2 slots), a fixed 96
passed (1,051 req/s, no 503s, 81 ms per flush), and an adaptive 48..192 with
1.0 / 2.0 s thresholds brought the convoy back (~250 concurrent flushes, 1.08 s
per flush, never closing). The release defaults are therefore the proven fixed
caps with the floors equal to them (evidence keeps 12..24), and the thresholds
sit at ~2x and ~4x the uncontended flush. The fault tier keeps a range of its
own so a telemetry burst never takes a fault slot.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.app.admission import _ProcessorAdmissionBatcher
from gpu_fault.app.admission_runtime import AdmissionRuntimeFactory
from gpu_fault.async_store import AsyncStoreExecutor
from tests._builders import build_context
from tests.app_services._cov95_runtime_admission import BATCHERS, POOLS, item

SITE_RANGES = {
    "GPU_FAULT_INGRESS_WRITER_PROCESSES": "12",
    "GPU_FAULT_PROCESSOR_ADMISSION_FLUSH_CONCURRENCY": "96",
    "GPU_FAULT_PROCESSOR_ADMISSION_FLUSH_CONCURRENCY_FLOOR": "96",
    "GPU_FAULT_PROCESSOR_FAULT_ADMISSION_FLUSH_CONCURRENCY": "48",
    "GPU_FAULT_PROCESSOR_FAULT_ADMISSION_FLUSH_CONCURRENCY_FLOOR": "48",
    "GPU_FAULT_PROCESSOR_EVIDENCE_ADMISSION_FLUSH_CONCURRENCY": "24",
    "GPU_FAULT_PROCESSOR_EVIDENCE_ADMISSION_FLUSH_CONCURRENCY_FLOOR": "12",
}


def _runtime(monkeypatch: pytest.MonkeyPatch, environment: dict[str, str]):
    for name in (
        "GPU_FAULT_STORE_IO_WORKERS",
        "GPU_FAULT_INGRESS_DECODE_WORKERS",
        "GPU_FAULT_FAULT_STORE_IO_WORKERS",
        "GPU_FAULT_EVIDENCE_STORE_IO_WORKERS",
        "GPU_FAULT_FAULT_DECODE_WORKERS",
        "GPU_FAULT_TELEMETRY_SPOOL_STORE_IO_WORKERS",
    ):
        monkeypatch.setenv(name, "1")
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    return AdmissionRuntimeFactory(build_context(), None).build()


def _ranges(runtime) -> dict[str, tuple[int, int, int]]:
    """(floor, live limit, ceiling) per tier."""

    result = {}
    for tier, name in (
        ("routine", "admission_batcher"),
        ("fault", "fault_batcher"),
        ("evidence", "evidence_batcher"),
    ):
        batcher = getattr(runtime, name)
        result[tier] = (
            batcher.concurrency.floor,
            batcher.max_in_flight_flushes,
            batcher.concurrency.ceiling,
        )
    return result


async def _close(runtime) -> None:
    for name in BATCHERS:
        await getattr(runtime, name).close()
    for name in POOLS:
        getattr(runtime, name).close()


def test_each_process_takes_its_share_of_the_site_wide_ranges(monkeypatch) -> None:
    runtime = _runtime(monkeypatch, SITE_RANGES)
    try:
        assert _ranges(runtime) == {
            "routine": (8, 8, 8),
            "fault": (4, 4, 4),
            "evidence": (1, 1, 2),
        }, "routine and fault are the proven fixed caps; evidence keeps a range"
        assert runtime.admission_batcher.concurrency.adaptive is False
        assert runtime.fault_batcher.concurrency.adaptive is False
        assert runtime.evidence_batcher.concurrency.adaptive is True
        assert runtime.admission_batcher.flush_delay_max_seconds == 0.05
        assert runtime.fault_batcher.flush_delay_max_seconds == 0.05
        for name in ("admission_batcher", "fault_batcher", "evidence_batcher"):
            controller = getattr(runtime, name).concurrency
            assert controller.open_below_seconds == 0.15
            assert controller.close_above_seconds == 0.3
    finally:
        asyncio.run(_close(runtime))


def test_a_value_that_does_not_divide_rounds_the_fleet_total_up(monkeypatch) -> None:
    runtime = _runtime(
        monkeypatch,
        {
            **SITE_RANGES,
            "GPU_FAULT_PROCESSOR_ADMISSION_FLUSH_CONCURRENCY": "30",
            "GPU_FAULT_PROCESSOR_ADMISSION_FLUSH_CONCURRENCY_FLOOR": "13",
            "GPU_FAULT_PROCESSOR_FAULT_ADMISSION_FLUSH_CONCURRENCY": "1",
            "GPU_FAULT_PROCESSOR_FAULT_ADMISSION_FLUSH_CONCURRENCY_FLOOR": "1",
        },
    )
    try:
        assert _ranges(runtime) == {
            "routine": (2, 2, 3),
            "fault": (1, 1, 1),
            "evidence": (1, 1, 2),
        }
        assert runtime.fault_batcher.concurrency.adaptive is False
    finally:
        asyncio.run(_close(runtime))


def test_a_ceiling_without_a_floor_is_a_fixed_limit(monkeypatch) -> None:
    """A ConfigMap rendered before the floor existed keeps its fixed cap."""

    runtime = _runtime(
        monkeypatch,
        {
            "GPU_FAULT_INGRESS_WRITER_PROCESSES": "12",
            "GPU_FAULT_PROCESSOR_ADMISSION_FLUSH_CONCURRENCY": "24",
            "GPU_FAULT_PROCESSOR_FAULT_ADMISSION_FLUSH_CONCURRENCY": "24",
            "GPU_FAULT_PROCESSOR_EVIDENCE_ADMISSION_FLUSH_CONCURRENCY": "12",
        },
    )
    try:
        assert _ranges(runtime) == {
            "routine": (2, 2, 2),
            "fault": (2, 2, 2),
            "evidence": (1, 1, 1),
        }
        assert runtime.admission_batcher.concurrency.adaptive is False
    finally:
        asyncio.run(_close(runtime))


def test_an_explicit_per_process_value_pins_both_ends(monkeypatch) -> None:
    runtime = _runtime(
        monkeypatch,
        {
            **SITE_RANGES,
            "GPU_FAULT_PROCESSOR_ADMISSION_BATCH_GROUPS": "5",
            "GPU_FAULT_PROCESSOR_EVIDENCE_ADMISSION_BATCH_GROUPS": "3",
            "GPU_FAULT_PROCESSOR_ADMISSION_BATCH_DELAY_MAX_SECONDS": "0.2",
        },
    )
    try:
        assert _ranges(runtime) == {
            "routine": (5, 5, 5),
            "fault": (4, 4, 4),
            "evidence": (3, 3, 3),
        }
        assert runtime.admission_batcher.flush_delay_max_seconds == 0.2
    finally:
        asyncio.run(_close(runtime))


def test_the_thresholds_are_read_per_tier(monkeypatch) -> None:
    runtime = _runtime(
        monkeypatch,
        {
            **SITE_RANGES,
            "GPU_FAULT_PROCESSOR_ADMISSION_FLUSH_CONCURRENCY_OPEN_SECONDS": "0.4",
            "GPU_FAULT_PROCESSOR_ADMISSION_FLUSH_CONCURRENCY_CLOSE_SECONDS": "0.9",
            "GPU_FAULT_PROCESSOR_FAULT_ADMISSION_FLUSH_CONCURRENCY_CLOSE_SECONDS": "3",
        },
    )
    try:
        routine = runtime.admission_batcher.concurrency
        assert (routine.open_below_seconds, routine.close_above_seconds) == (0.4, 0.9)
        fault = runtime.fault_batcher.concurrency
        assert (fault.open_below_seconds, fault.close_above_seconds) == (0.15, 3.0)
    finally:
        asyncio.run(_close(runtime))


def test_a_base_delay_above_the_cap_leaves_the_window_fixed(monkeypatch) -> None:
    runtime = _runtime(
        monkeypatch,
        {
            "GPU_FAULT_PROCESSOR_ADMISSION_BATCH_DELAY_SECONDS": "0.1",
            "GPU_FAULT_PROCESSOR_ADMISSION_BATCH_DELAY_MAX_SECONDS": "0.05",
        },
    )
    try:
        batcher = runtime.admission_batcher
        assert batcher.flush_delay_seconds == 0.1
        assert batcher.flush_delay_max_seconds == 0.1
        batcher.round_seconds_ewma = 10.0
        assert batcher.effective_flush_delay_seconds() == 0.1
    finally:
        asyncio.run(_close(runtime))


def test_without_site_values_a_bare_process_keeps_the_historical_defaults(
    monkeypatch,
) -> None:
    runtime = _runtime(monkeypatch, {})
    try:
        # Fault groups default to the fault store I/O worker count (1 here).
        assert _ranges(runtime) == {
            "routine": (16, 16, 16),
            "fault": (1, 1, 1),
            "evidence": (4, 4, 4),
        }
    finally:
        asyncio.run(_close(runtime))


@pytest.mark.parametrize(
    ("environment", "message"),
    [
        (
            {**SITE_RANGES, "GPU_FAULT_INGRESS_WRITER_PROCESSES": "0"},
            "writer process count must be positive",
        ),
        (
            {
                **SITE_RANGES,
                "GPU_FAULT_PROCESSOR_FAULT_ADMISSION_FLUSH_CONCURRENCY": "0",
                "GPU_FAULT_PROCESSOR_FAULT_ADMISSION_FLUSH_CONCURRENCY_FLOOR": "0",
            },
            "flush concurrency must be positive",
        ),
        (
            {
                **SITE_RANGES,
                "GPU_FAULT_PROCESSOR_ADMISSION_FLUSH_CONCURRENCY_FLOOR": "97",
            },
            "FLUSH_CONCURRENCY_FLOOR cannot exceed",
        ),
        (
            {
                "GPU_FAULT_INGRESS_WRITER_PROCESSES": "12",
                "GPU_FAULT_PROCESSOR_ADMISSION_FLUSH_CONCURRENCY_FLOOR": "12",
            },
            "FLUSH_CONCURRENCY_FLOOR requires",
        ),
        (
            {
                **SITE_RANGES,
                "GPU_FAULT_PROCESSOR_ADMISSION_FLUSH_CONCURRENCY_OPEN_SECONDS": "2",
                "GPU_FAULT_PROCESSOR_ADMISSION_FLUSH_CONCURRENCY_CLOSE_SECONDS": "1",
            },
            "close threshold cannot be below the open threshold",
        ),
    ],
)
def test_an_invalid_fleet_setting_refuses_to_start(
    monkeypatch, environment: dict[str, str], message: str
) -> None:
    with pytest.raises((RuntimeError, ValueError), match=message):
        _runtime(monkeypatch, environment)


def _batcher(store: Any, executor: AsyncStoreExecutor, **kwargs: Any):
    return _ProcessorAdmissionBatcher(
        store,
        executor,
        max_depth=100_000,
        max_cluster_depth=100_000,
        reserved_fault_depth=0,
        reserved_cluster_fault_depth=0,
        global_admission_guard=0,
        **kwargs,
    )


def test_the_coalescing_window_follows_the_flush_ewma() -> None:
    executor = AsyncStoreExecutor(
        workers=1, max_in_flight=1, admission_timeout_seconds=1
    )
    try:
        batcher = _batcher(
            SimpleNamespace(),
            executor,
            flush_delay_seconds=0.002,
            flush_delay_max_seconds=0.05,
            contention_flush_seconds=0.1,
        )
        expectations = {
            0.0: 0.002,
            0.05: 0.002,
            0.1: 0.002,
            0.15: 0.026,
            0.2: 0.05,
            5.0: 0.05,
            0.01: 0.002,
        }
        for ewma, expected in expectations.items():
            batcher.round_seconds_ewma = ewma
            assert batcher.effective_flush_delay_seconds() == pytest.approx(expected), (
                f"flush EWMA {ewma}s"
            )
        # Equal base and cap: the adaptation is off whatever the store does.
        fixed = _batcher(SimpleNamespace(), executor, flush_delay_seconds=0.002)
        fixed.round_seconds_ewma = 30.0
        assert fixed.effective_flush_delay_seconds() == 0.002
        assert fixed.flush_delay_max_seconds == 0.002
    finally:
        executor.close()


def test_the_window_bounds_are_validated() -> None:
    executor = AsyncStoreExecutor(
        workers=1, max_in_flight=1, admission_timeout_seconds=1
    )
    try:
        with pytest.raises(ValueError, match="delay cap cannot be below the base"):
            _batcher(
                SimpleNamespace(),
                executor,
                flush_delay_seconds=0.05,
                flush_delay_max_seconds=0.002,
            )
        with pytest.raises(ValueError, match="contention threshold must be positive"):
            _batcher(SimpleNamespace(), executor, contention_flush_seconds=0)
        with pytest.raises(ValueError, match="ceiling cannot be below the floor"):
            _batcher(
                SimpleNamespace(), executor, max_flush_groups=2, min_flush_groups=3
            )
    finally:
        executor.close()


def test_a_slow_store_stretches_the_next_window_and_it_is_exported() -> None:
    """One slow flush (0.25 s against a 0.1 s threshold) is enough: the next
    dispatch pass waits the capped window, and the counters say so."""

    class SlowStore:
        def try_enqueue_processor_requests_batch(self, items, **_kwargs):
            time.sleep(0.25)
            return [(entry, None) for entry in items]

    executor = AsyncStoreExecutor(
        workers=2, max_in_flight=4, admission_timeout_seconds=5
    )

    async def scenario() -> None:
        batcher = _batcher(
            SlowStore(),
            executor,
            flush_delay_seconds=0.0,
            flush_delay_max_seconds=0.05,
            contention_flush_seconds=0.1,
        )
        assert batcher.flush_delay_current_seconds == 0.0
        first = await batcher.submit(item("cluster-a"))
        assert first[1] is None
        assert batcher.round_seconds_ewma >= 0.2
        second = await batcher.submit(item("cluster-b"))
        assert second[1] is None
        await batcher.close()
        assert batcher.delay_stretched_rounds_total >= 1
        assert batcher.flush_delay_current_seconds == pytest.approx(0.05)

    try:
        asyncio.run(scenario())
    finally:
        executor.close()
