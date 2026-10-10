"""The adaptive in-flight flush limit: open on a saturated backlog while the
store answers, hold in the hysteresis band, close multiplicatively when flush
latency climbs, decay to the floor when the limit stops binding - one change
per settled generation of flushes - and the deadline projection counts the
slots the controller is about to open.

Numbers behind the design (live integrated-32 suite, schema v22, 12 ingress
processes): a fixed cap of 2 per process expired 4,601 requests with ~800
queued per process and 0.59 s per flush; a fixed 8 per process admitted the
same burst at 1,051 req/s with no 503s and 81 ms per flush; an adaptive 4..16
per process with 1.0 / 2.0 s thresholds let the site reach ~250 concurrent
flushes, averaged 1.08 s per flush without ever closing, flapped (81 raises,
51 lowers in 14 minutes) and shed 35.8 % of telemetry.
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


def _settle(controller: FlushConcurrencyController) -> None:
    """Complete one generation of flushes at the current level."""

    _flushes(controller, controller.settle_flushes)


def _ramp(
    controller: FlushConcurrencyController, level: int, ewma: float = 0.08
) -> None:
    """Saturated and quick: one slot per settled generation up to ``level``."""

    while controller.limit < level:
        _settle(controller)
        controller.observe(saturated=True, flush_ewma=ewma)


def test_the_default_thresholds_are_multiples_of_the_uncontended_flush() -> None:
    controller = FlushConcurrencyController(floor=1, ceiling=2)
    assert controller.open_below_seconds == 0.15, "~2x the live 81 ms flush"
    assert controller.close_above_seconds == 0.3, "~4x; the failed run averaged 1.08 s"
    assert controller.settle_flushes_min == 3


def test_a_process_starts_at_the_floor_and_a_fixed_range_never_moves() -> None:
    controller = _controller()
    assert controller.limit == 4
    assert controller.adaptive is True
    # The release default for the routine and fault tiers: floor == ceiling,
    # the proven fixed cap of 8 per process, whatever latency does.
    fixed = _controller(floor=8, ceiling=8)
    assert fixed.adaptive is False
    _flushes(fixed, 50)
    for ewma in (0.0, 0.08, 0.5, 5.0):
        assert fixed.observe(saturated=True, flush_ewma=ewma) == 8
        assert fixed.observe(saturated=False, flush_ewma=ewma) == 8
    assert (fixed.raised_total, fixed.lowered_total) == (0, 0)
    assert fixed.may_open(0.0) is False


def test_saturation_opens_one_slot_per_settled_generation_up_to_the_ceiling() -> None:
    controller = _controller()
    # Not settled: nothing to judge yet, no change - however saturated.
    assert controller.observe(saturated=True, flush_ewma=0.08) == 4
    _flushes(controller, 3)
    assert controller.observe(saturated=True, flush_ewma=0.08) == 4, "3 of 4"
    controller.record_flush()
    assert controller.observe(saturated=True, flush_ewma=0.08) == 5
    # One completion is no longer enough for the next raise: a raise waits
    # for a generation at the new level, exactly as a close does.
    controller.record_flush()
    assert controller.observe(saturated=True, flush_ewma=0.08) == 5
    for expected in range(6, 17):
        _settle(controller)
        assert controller.observe(saturated=True, flush_ewma=0.08) == expected
    _settle(controller)
    assert controller.observe(saturated=True, flush_ewma=0.08) == 16, "the ceiling"
    assert controller.raised_total == 12
    assert controller.lowered_total == 0


def test_an_unsaturated_batcher_never_opens() -> None:
    controller = _controller()
    _flushes(controller, 10)
    assert controller.observe(saturated=False, flush_ewma=0.05) == 4
    assert controller.raised_total == 0


def test_the_band_between_the_thresholds_holds() -> None:
    controller = _controller()
    _settle(controller)
    assert controller.observe(saturated=True, flush_ewma=0.149) == 5
    _flushes(controller, 20)
    for ewma in (0.15, 0.2, 0.3):
        assert controller.observe(saturated=True, flush_ewma=ewma) == 5, ewma
    assert (controller.raised_total, controller.lowered_total) == (1, 0)
    # The band holds a saturated process; one that stopped binding still
    # decays - that rule is about demand, not latency.
    assert controller.observe(saturated=False, flush_ewma=0.2) == 4
    assert controller.lowered_total == 1


def test_latency_above_the_close_threshold_halves_the_headroom_once_settled() -> None:
    controller = _controller()
    _ramp(controller, 16)
    assert controller.limit == 16
    # One slow sample is not a verdict: the level must settle first (a
    # generation of flushes), and saturation does not keep it open.
    assert controller.observe(saturated=True, flush_ewma=0.5) == 16
    _flushes(controller, controller.settle_flushes - 1)
    assert controller.observe(saturated=True, flush_ewma=0.5) == 16
    controller.record_flush()
    assert controller.observe(saturated=True, flush_ewma=0.5) == 10, "headroom 12 -> 6"
    _settle(controller)
    assert controller.observe(saturated=True, flush_ewma=0.5) == 7, "headroom 6 -> 3"
    _settle(controller)
    assert controller.observe(saturated=True, flush_ewma=0.5) == 5
    _settle(controller)
    assert controller.observe(saturated=True, flush_ewma=0.5) == 4, "the floor"
    _settle(controller)
    assert controller.observe(saturated=True, flush_ewma=0.5) == 4
    assert controller.lowered_total == 4
    # Hysteresis: back under the open threshold a settled floor climbs again.
    assert controller.observe(saturated=True, flush_ewma=0.1) == 5


def test_a_limit_that_stops_binding_decays_one_slot_per_generation() -> None:
    controller = _controller()
    _ramp(controller, 8)
    assert controller.limit == 8
    assert controller.observe(saturated=False, flush_ewma=0.1) == 8, "not settled"
    _settle(controller)
    assert controller.observe(saturated=False, flush_ewma=0.1) == 7
    for expected in (6, 5, 4):
        _settle(controller)
        assert controller.observe(saturated=False, flush_ewma=0.1) == expected
    _settle(controller)
    assert controller.observe(saturated=False, flush_ewma=0.1) == 4, "the floor holds"
    assert controller.lowered_total == 4


def test_settling_takes_a_generation_of_flushes_at_the_current_level() -> None:
    controller = _controller(settle_flushes_min=3)
    assert controller.settle_flushes == 4, "max(3, limit 4)"
    _ramp(controller, 12)
    assert controller.settle_flushes == 12
    assert _controller(floor=1, ceiling=4).settle_flushes == 3


def test_may_open_names_the_regime_the_projection_divides_by() -> None:
    controller = _controller()
    assert controller.may_open(0.08) is True
    assert controller.may_open(0.149) is True
    assert controller.may_open(0.15) is False
    assert controller.may_open(1.08) is False
    assert _controller(floor=4, ceiling=4).may_open(0.0) is False


def test_the_live_sequence_closes_within_one_generation_of_the_latency_rising() -> None:
    """The range that failed live (4..16 per process), replayed: saturated at
    81 ms the limit opens to the ceiling; the flush EWMA then climbs
    0.08 -> 1.0 s as the store convoys. Under that release's 2.0 s close
    threshold the limit never moved (the run averaged 1.08 s per flush); with
    the retuned thresholds the first generation that settles above 0.30 s
    closes it, and three generations later it is at the floor."""

    controller = _controller(floor=4, ceiling=16)
    _ramp(controller, 16, ewma=0.08)
    assert controller.limit == 16
    changes = controller.raised_total + controller.lowered_total
    rising = [0.1, 0.14, 0.2, 0.26, 0.31, 0.4, 0.55, 0.7, 0.85, 1.0]
    limits = []
    for ewma in rising:
        controller.record_flush()
        limits.append(controller.observe(saturated=True, flush_ewma=ewma))
    assert limits == [16] * len(rising), "inside one generation nothing moves"
    _flushes(controller, controller.settle_flushes - len(rising))
    assert controller.observe(saturated=True, flush_ewma=1.0) == 10
    for expected in (7, 5, 4):
        _settle(controller)
        assert controller.observe(saturated=True, flush_ewma=1.0) == expected
    assert controller.raised_total + controller.lowered_total - changes == 4
    # The previous release's thresholds would have held the slots open.
    legacy = _controller(open_below_seconds=1.0, close_above_seconds=2.0)
    _ramp(legacy, 16, ewma=0.08)
    _settle(legacy)
    assert legacy.observe(saturated=True, flush_ewma=1.08) == 16
    assert legacy.lowered_total == 0


def test_alternating_latency_changes_the_limit_at_most_once_per_generation() -> None:
    """81 raises and 51 lowers in 14 minutes live: a raise needed one completed
    flush, so the limit flapped with every sample. A saturated controller fed
    a flush EWMA that crosses both thresholds on alternate flushes now changes
    its limit no more often than once per settled generation."""

    controller = _controller(floor=4, ceiling=16)
    history = []
    for index in range(400):
        controller.record_flush()
        history.append(
            controller.observe(saturated=True, flush_ewma=0.08 if index % 2 else 0.5)
        )
    change_points = [i for i in range(1, len(history)) if history[i] != history[i - 1]]
    assert len(change_points) >= 2, "it moved: the point is the cadence"
    gaps = [later - earlier for earlier, later in zip(change_points, change_points[1:])]
    assert min(gaps) >= controller.settle_flushes_min
    assert len(change_points) <= 400 // controller.settle_flushes_min
    assert controller.floor <= controller.limit <= controller.ceiling


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"floor": 0}, "floor must be positive"),
        ({"floor": 5, "ceiling": 4}, "ceiling cannot be below the floor"),
        ({"open_below_seconds": 0}, "thresholds must be positive"),
        ({"close_above_seconds": -1}, "thresholds must be positive"),
        (
            {"open_below_seconds": 0.3, "close_above_seconds": 0.15},
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
    """Window first, then the limit: the coalescing window stretches from a
    flush EWMA of 0.1 s and is at its cap by 0.2 s; the limit keeps opening
    below 0.15 s, holds between 0.15 and 0.30 s and closes above 0.30 s - by
    which point the window is already at its cap, so a contended store sees
    fewer and wider statements before it sees fewer concurrent ones."""

    executor = AsyncStoreExecutor(
        workers=1, max_in_flight=1, admission_timeout_seconds=1
    )
    try:
        batcher = _batcher(
            SimpleNamespace(),
            executor,
            max_flush_groups=8,
            min_flush_groups=4,
            flush_delay_seconds=0.002,
            flush_delay_max_seconds=0.05,
            contention_flush_seconds=0.1,
        )
        controller = batcher.concurrency
        assert batcher.max_in_flight_flushes == 4
        assert (controller.open_below_seconds, controller.close_above_seconds) == (
            0.15,
            0.3,
        ), "the batcher's defaults are the controller's"
        # The proven 81 ms store: base window, and the limit may open.
        batcher.round_seconds_ewma = 0.08
        assert batcher.effective_flush_delay_seconds() == 0.002
        assert batcher.projection_in_flight_limit() == 8
        _ramp(controller, 8, ewma=0.08)
        assert batcher.max_in_flight_flushes == 8
        # A slower store: the window stretches while the limit still may open.
        batcher.round_seconds_ewma = 0.12
        assert batcher.effective_flush_delay_seconds() == pytest.approx(0.0116)
        assert controller.may_open(0.12) is True
        # The band: window at its cap, limit held.
        batcher.round_seconds_ewma = 0.2
        assert batcher.effective_flush_delay_seconds() == 0.05
        assert controller.may_open(0.2) is False
        _settle(controller)
        assert controller.observe(saturated=True, flush_ewma=0.2) == 8, "the band"
        # The failed live run's average flush: the window is at its cap and
        # the limit closes toward the floor.
        batcher.round_seconds_ewma = 1.08
        assert batcher.effective_flush_delay_seconds() == 0.05
        assert controller.observe(saturated=True, flush_ewma=1.08) == 6
        assert batcher.max_in_flight_flushes == 6
        assert batcher.projection_in_flight_limit() == 6
        assert batcher.in_flight_limit_raised_total == 4
        assert batcher.in_flight_limit_lowered_total == 1
    finally:
        executor.close()


def test_a_saturated_burst_opens_slots_and_the_counters_say_so() -> None:
    """Sixteen clusters arrive at once against a floor of one slot and a 20 ms
    store: the controller opens a slot per settled generation, the burst runs
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
            *(batcher.submit(item(f"cluster-{index:02d}")) for index in range(16))
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


def test_a_burst_at_the_floor_is_not_shed_for_capacity_about_to_open() -> None:
    """Sixteen scopes queue behind one stalled flush on a floor of one slot.

    While the flush EWMA is under the open threshold (0.12 s) the controller
    will run the backlog at the ceiling, so a request with 2 s of budget is
    admitted (projected ~0.36 s at 8 slots, ~2 s at the floor); once the EWMA
    is in the band (0.2 s) the live limit is the honest divisor and the same
    request is shed on arrival (projected ~3.4 s at 1 slot) instead of
    expiring in the queue.
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
            batcher.round_seconds_ewma = 0.12
            assert batcher.projection_in_flight_limit() == 8
            admitted = asyncio.create_task(batcher.submit(item("cluster-new-a")))
            await asyncio.sleep(0.01)
            assert not admitted.done(), "queued, not shed: capacity is about to open"
            batcher.round_seconds_ewma = 0.2
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
