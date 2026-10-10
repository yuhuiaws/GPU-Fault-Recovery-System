"""The adaptive in-flight flush limit: open fast on a saturated backlog while
the store answers, hold in the hysteresis band, close multiplicatively when
flush latency climbs, decay to the floor when the limit stops binding - and
the deadline projection counts the slots the controller is about to open.

Numbers behind the design (live integrated-32 suite): a fixed cap of 2 per
process expired 4,601 requests with ~800 queued per process and 0.59 s per
flush; 8 per process admitted the same burst at 1,051 req/s, 81 ms per flush,
and was still saturated at the ceiling for part of it.
"""

from __future__ import annotations

import asyncio
import time
from threading import Event
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.app.admission import _ProcessorAdmissionBatcher
from gpu_fault.app.admission_concurrency import FlushConcurrencyController
from gpu_fault.async_store import (
    REQUEST_DEADLINE,
    AsyncStoreExecutor,
    RequestDeadlineExceeded,
)
from tests.app_services._cov95_runtime_admission import item


def _controller(**kwargs: Any) -> FlushConcurrencyController:
    defaults = {"floor": 4, "ceiling": 16}
    return FlushConcurrencyController(**{**defaults, **kwargs})


def _flushes(controller: FlushConcurrencyController, count: int) -> None:
    for _ in range(count):
        controller.record_flush()


def test_a_process_starts_at_the_floor_and_a_fixed_range_never_moves() -> None:
    controller = _controller()
    assert controller.limit == 4
    assert controller.adaptive is True
    fixed = _controller(floor=5, ceiling=5)
    assert fixed.adaptive is False
    _flushes(fixed, 50)
    for ewma in (0.0, 0.5, 5.0):
        assert fixed.observe(saturated=True, flush_ewma=ewma) == 5
        assert fixed.observe(saturated=False, flush_ewma=ewma) == 5
    assert (fixed.raised_total, fixed.lowered_total) == (0, 0)


def test_saturation_opens_one_slot_per_completed_flush_up_to_the_ceiling() -> None:
    controller = _controller()
    # No flush has completed at this level yet: nothing to judge, no change.
    assert controller.observe(saturated=True, flush_ewma=0.08) == 4
    for expected in range(5, 17):
        controller.record_flush()
        assert controller.observe(saturated=True, flush_ewma=0.08) == expected
        # A second pass before another completion holds the new level.
        assert controller.observe(saturated=True, flush_ewma=0.08) == expected
    controller.record_flush()
    assert controller.observe(saturated=True, flush_ewma=0.08) == 16, "the ceiling"
    assert controller.raised_total == 12
    assert controller.lowered_total == 0


def test_an_unsaturated_batcher_never_opens() -> None:
    controller = _controller()
    _flushes(controller, 10)
    assert controller.observe(saturated=False, flush_ewma=0.05) == 4
    assert controller.raised_total == 0


def test_the_band_between_the_thresholds_holds() -> None:
    controller = _controller(open_below_seconds=1.0, close_above_seconds=2.0)
    controller.record_flush()
    assert controller.observe(saturated=True, flush_ewma=0.99) == 5
    _flushes(controller, 20)
    for ewma in (1.0, 1.5, 2.0):
        assert controller.observe(saturated=True, flush_ewma=ewma) == 5, ewma
    assert (controller.raised_total, controller.lowered_total) == (1, 0)
    # The band holds a saturated process; one that stopped binding still
    # decays - that rule is about demand, not latency.
    assert controller.observe(saturated=False, flush_ewma=1.5) == 4
    assert controller.lowered_total == 1


def test_latency_above_the_close_threshold_halves_the_headroom_once_settled() -> None:
    controller = _controller(open_below_seconds=1.0, close_above_seconds=2.0)
    for _ in range(12):
        controller.record_flush()
        controller.observe(saturated=True, flush_ewma=0.1)
    assert controller.limit == 16
    # One slow sample is not a verdict: the level must settle first (a
    # generation of flushes), and saturation does not keep it open.
    assert controller.observe(saturated=True, flush_ewma=2.5) == 16
    _flushes(controller, controller.settle_flushes - 1)
    assert controller.observe(saturated=True, flush_ewma=2.5) == 16
    controller.record_flush()
    assert controller.observe(saturated=True, flush_ewma=2.5) == 10, "headroom 12 -> 6"
    _flushes(controller, controller.settle_flushes)
    assert controller.observe(saturated=True, flush_ewma=2.5) == 7, "headroom 6 -> 3"
    _flushes(controller, controller.settle_flushes)
    assert controller.observe(saturated=True, flush_ewma=2.5) == 5
    _flushes(controller, controller.settle_flushes)
    assert controller.observe(saturated=True, flush_ewma=2.5) == 4, "the floor"
    _flushes(controller, controller.settle_flushes)
    assert controller.observe(saturated=True, flush_ewma=2.5) == 4
    assert controller.lowered_total == 4
    # Hysteresis: back under the open threshold it climbs again.
    controller.record_flush()
    assert controller.observe(saturated=True, flush_ewma=0.3) == 5


def test_a_limit_that_stops_binding_decays_one_slot_per_generation() -> None:
    controller = _controller()
    for _ in range(4):
        controller.record_flush()
        controller.observe(saturated=True, flush_ewma=0.1)
    assert controller.limit == 8
    assert controller.observe(saturated=False, flush_ewma=0.1) == 8, "not settled"
    _flushes(controller, controller.settle_flushes)
    assert controller.observe(saturated=False, flush_ewma=0.1) == 7
    for expected in (6, 5, 4):
        _flushes(controller, controller.settle_flushes)
        assert controller.observe(saturated=False, flush_ewma=0.1) == expected
    _flushes(controller, controller.settle_flushes)
    assert controller.observe(saturated=False, flush_ewma=0.1) == 4, "the floor holds"
    assert controller.lowered_total == 4


def test_settling_takes_a_generation_of_flushes_at_the_current_level() -> None:
    controller = _controller(settle_flushes_min=3)
    assert controller.settle_flushes == 4, "max(3, limit 4)"
    for _ in range(8):
        controller.record_flush()
        controller.observe(saturated=True, flush_ewma=0.1)
    assert controller.limit == 12
    assert controller.settle_flushes == 12
    assert _controller(floor=1, ceiling=4).settle_flushes == 3


def test_may_open_names_the_regime_the_projection_divides_by() -> None:
    controller = _controller(open_below_seconds=1.0, close_above_seconds=2.0)
    assert controller.may_open(0.08) is True
    assert controller.may_open(0.99) is True
    assert controller.may_open(1.0) is False
    assert controller.may_open(3.0) is False
    assert _controller(floor=4, ceiling=4).may_open(0.0) is False


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"floor": 0}, "floor must be positive"),
        ({"floor": 5, "ceiling": 4}, "ceiling cannot be below the floor"),
        ({"open_below_seconds": 0}, "thresholds must be positive"),
        ({"close_above_seconds": -1}, "thresholds must be positive"),
        (
            {"open_below_seconds": 2.0, "close_above_seconds": 1.0},
            "close threshold cannot be below the open threshold",
        ),
        ({"settle_flushes_min": 0}, "settle count must be positive"),
    ],
)
def test_the_range_and_thresholds_are_validated(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _controller(**kwargs)


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


def test_the_delay_stretch_and_the_concurrency_controller_act_in_order() -> None:
    """The coalescing window stretches first (EWMA 0.1-0.2 s), the limit keeps
    opening up to the open threshold (1 s), and only a flush EWMA above the
    close threshold (2 s) closes slots - at which point the window is already
    at its cap, so a contended store sees fewer and wider statements."""

    executor = AsyncStoreExecutor(
        workers=1, max_in_flight=1, admission_timeout_seconds=1
    )
    try:
        batcher = _batcher(
            SimpleNamespace(),
            executor,
            max_flush_groups=8,
            min_flush_groups=2,
            flush_delay_seconds=0.002,
            flush_delay_max_seconds=0.05,
            contention_flush_seconds=0.1,
        )
        controller = batcher.concurrency
        assert batcher.max_in_flight_flushes == 2
        # Quick store: base window, limit opens to the ceiling.
        batcher.round_seconds_ewma = 0.08
        assert batcher.effective_flush_delay_seconds() == 0.002
        assert batcher.projection_in_flight_limit() == 8
        # Slower store: the window stretches while slots still open.
        batcher.round_seconds_ewma = 0.15
        assert batcher.effective_flush_delay_seconds() == pytest.approx(0.026)
        assert controller.may_open(0.15) is True
        for _ in range(6):
            controller.record_flush()
            controller.observe(saturated=True, flush_ewma=0.15)
        assert batcher.max_in_flight_flushes == 8
        # Under-capacity regime seen live at the fixed cap (0.59 s): window at
        # its cap, limit still permitted to open - this must not close.
        batcher.round_seconds_ewma = 0.59
        assert batcher.effective_flush_delay_seconds() == 0.05
        assert controller.may_open(0.59) is True
        # Convoy regime: the window is at its cap and the limit closes.
        batcher.round_seconds_ewma = 2.5
        assert batcher.effective_flush_delay_seconds() == 0.05
        assert controller.may_open(2.5) is False
        for _ in range(controller.settle_flushes):
            controller.record_flush()
        assert controller.observe(saturated=True, flush_ewma=2.5) == 5
        assert batcher.max_in_flight_flushes == 5
        assert batcher.projection_in_flight_limit() == 5
        assert batcher.in_flight_limit_raised_total == 6
        assert batcher.in_flight_limit_lowered_total == 1
    finally:
        executor.close()


def test_a_saturated_burst_opens_slots_and_the_counters_say_so() -> None:
    """Twelve clusters arrive at once against a floor of one slot and a 20 ms
    store: the controller opens a slot per completed flush, the burst runs
    more than one flush at a time, and the limit is exported."""

    class Store:
        def try_enqueue_processor_requests_batch(self, items, **_kwargs):
            time.sleep(0.02)
            return [(entry, None) for entry in items]

    executor = AsyncStoreExecutor(
        workers=8, max_in_flight=16, admission_timeout_seconds=5
    )

    async def scenario() -> None:
        batcher = _batcher(
            Store(),
            executor,
            max_flush_groups=6,
            min_flush_groups=1,
            flush_delay_seconds=0.0,
        )
        results = await asyncio.gather(
            *(batcher.submit(item(f"cluster-{index:02d}")) for index in range(12))
        )
        assert all(result[1] is None for result in results), "every entry admitted"
        assert batcher.in_flight_limit_raised_total >= 1
        assert batcher.in_flight_max >= 2, "the burst ran wider than the floor"
        assert batcher.max_in_flight_flushes <= 6
        await batcher.close()

    try:
        asyncio.run(scenario())
    finally:
        executor.close()


def test_the_projection_counts_the_slots_the_controller_can_still_open() -> None:
    """Sixteen scopes queue behind one stalled flush on a floor of one slot.

    While the flush EWMA is under the open threshold the controller will run
    the backlog at the ceiling, so a request with 2 s of budget is admitted
    (projected ~1.5 s at 8 slots); once the EWMA says the store is contended
    the live limit is the honest divisor and the same request is shed on
    arrival (projected ~24 s at 1 slot) instead of expiring in the queue.
    """

    release = Event()
    stalled = Event()

    class Store:
        def try_enqueue_processor_requests_batch(self, items, **_kwargs):
            stalled.set()
            release.wait(10)
            return [(entry, None) for entry in items]

    executor = AsyncStoreExecutor(
        workers=4, max_in_flight=32, admission_timeout_seconds=10
    )

    async def scenario() -> None:
        batcher = _batcher(
            Store(),
            executor,
            max_flush_groups=8,
            min_flush_groups=1,
            flush_delay_seconds=0.0,
            projection_margin=1.0,
        )
        queued = [
            asyncio.create_task(batcher.submit(item(f"cluster-{index:02d}")))
            for index in range(17)
        ]
        while not stalled.is_set():
            await asyncio.sleep(0.001)
        await asyncio.sleep(0.02)
        assert batcher.in_flight == 1
        assert batcher.scopes_pending == 16
        token = REQUEST_DEADLINE.set(time.monotonic() + 2.0)
        try:
            batcher.round_seconds_ewma = 0.5
            assert batcher.projection_in_flight_limit() == 8
            admitted = asyncio.create_task(batcher.submit(item("cluster-new-a")))
            await asyncio.sleep(0.01)
            assert not admitted.done(), "queued, not shed: capacity is about to open"
            batcher.round_seconds_ewma = 1.5
            assert batcher.projection_in_flight_limit() == 1
            with pytest.raises(RequestDeadlineExceeded, match="cannot reach"):
                await batcher.submit(item("cluster-new-b"))
            assert batcher.shed_total == 1
        finally:
            REQUEST_DEADLINE.reset(token)
        release.set()
        results = await asyncio.gather(*queued, admitted)
        assert all(result[1] is None for result in results), "every entry admitted"
        await batcher.close()

    try:
        asyncio.run(scenario())
    finally:
        executor.close()
