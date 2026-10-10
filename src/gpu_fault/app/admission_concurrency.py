"""Adaptive in-flight limit for the admission batchers' flushes.

Batch 48 replaced the per-process flush literals with site-wide caps that the
ingress processes divide among themselves, and the live perf suite showed the
cap it chose to be an order of magnitude too low: at 2 flushes per process (24
site-wide) every process ran at its cap for the whole load start, the backlog
per process grew to ~800 entries, 4,601 telemetry requests expired in the queue
(18.7 % reserved 503s against a 2 % budget) and a flush took 0.59 s on average.
The same site with 8 per process (96) admitted the same burst at 1,051 req/s
with no expiry at all, 81 ms per flush, 60-69 entries queued per process at the
peak and the store's BufferContent waits at 1 % (30 % before): the 0.59 s was
queueing - 16-item batches and a 32 s storm - not the store, and the fixed cap
was the bottleneck. Yet 8 per process was still saturated for part of the
burst, and a static cap wide enough for every burst holds that many
transactions open against a store that may one day be the slow part.

So the cap is a range. Every process starts at the floor and the
:class:`FlushConcurrencyController` decides once per dispatch pass:

* **open** one more slot while the batcher is *saturated* (every slot busy
  and work still queued) and the flush-latency EWMA is below
  ``open_below_seconds`` - the store is answering, more parallel flushes buy
  throughput. A raise needs only one flush completed since the last change,
  so a saturated process climbs from the floor to the ceiling in a few
  hundred milliseconds: the live burst lasts seconds, and a controller that
  opened one slot per generation would shed like the fixed cap did;
* **hold** between the two thresholds - hysteresis, so a store that is slower
  but coping is neither pushed further nor starved;
* **close** toward the floor when the EWMA passes ``close_above_seconds`` -
  the store is contended, and fewer concurrent transactions is the only lever
  that helps it; the step halves the headroom above the floor (additive
  increase, multiplicative decrease), so a contended store sheds load in a
  few passes;
* **decay** one slot at a time when the limit is not binding, so a burst that
  has passed leaves the process back at the floor and the next one ramps
  again from there, and an idle site never holds more flushes open than the
  floor allows.

Closing and decay wait for the level to *settle* - at least
``max(settle_flushes_min, limit)`` flushes completed since the last change,
one generation of in-flight flushes - so the EWMA reflects the level being
judged and one slow sample cannot cascade a wide limit down to the floor.
The thresholds sit above the under-capacity regime, which must open (0.59 s
per flush at the fixed cap), and at the convoy regime, which must close
(2.2-3.6 s per flush before schema v22): 12x and 25x the uncontended 81 ms.
``floor == ceiling`` switches the adaptation off and the controller is a
constant.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FlushConcurrencyController:
    floor: int
    ceiling: int
    open_below_seconds: float = 1.0
    close_above_seconds: float = 2.0
    settle_flushes_min: int = 3
    limit: int = field(init=False)
    raised_total: int = field(default=0, init=False)
    lowered_total: int = field(default=0, init=False)
    _flushes_since_change: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.floor < 1:
            raise ValueError("admission flush concurrency floor must be positive")
        if self.ceiling < self.floor:
            raise ValueError(
                "admission flush concurrency ceiling cannot be below the floor"
            )
        if self.open_below_seconds <= 0 or self.close_above_seconds <= 0:
            raise ValueError("admission flush concurrency thresholds must be positive")
        if self.close_above_seconds < self.open_below_seconds:
            raise ValueError(
                "admission flush concurrency close threshold cannot be below "
                "the open threshold"
            )
        if self.settle_flushes_min < 1:
            raise ValueError(
                "admission flush concurrency settle count must be positive"
            )
        self.limit = self.floor

    @property
    def adaptive(self) -> bool:
        return self.floor < self.ceiling

    @property
    def settle_flushes(self) -> int:
        """Flushes that must complete at a level before it may be lowered."""

        return max(self.settle_flushes_min, self.limit)

    @property
    def settled(self) -> bool:
        return self._flushes_since_change >= self.settle_flushes

    def may_open(self, flush_ewma: float) -> bool:
        """Whether a saturated batcher would be given more slots right now.

        The deadline projection uses this to count the capacity the
        controller is about to open as present rather than missing: while
        the store answers below the open threshold, a backlog is served at
        the ceiling, not at whatever the limit happens to be mid-ramp.
        """

        return self.adaptive and flush_ewma < self.open_below_seconds

    def record_flush(self) -> None:
        self._flushes_since_change += 1

    def observe(self, *, saturated: bool, flush_ewma: float) -> int:
        """One dispatch pass's verdict; returns the limit the next pass uses.

        ``saturated`` is the dispatcher's own observation that every slot was
        busy and work still waited; ``flush_ewma`` is the batcher's flush
        latency EWMA, the same signal that stretches the coalescing window.
        """

        if not self.adaptive:
            return self.limit
        if flush_ewma > self.close_above_seconds:
            if self.limit > self.floor and self.settled:
                self._lower(max(1, (self.limit - self.floor + 1) // 2))
        elif saturated and flush_ewma < self.open_below_seconds:
            if self.limit < self.ceiling and self._flushes_since_change > 0:
                self._raise()
        elif not saturated and self.limit > self.floor and self.settled:
            self._lower(1)
        return self.limit

    def _raise(self) -> None:
        self.limit += 1
        self.raised_total += 1
        self._flushes_since_change = 0

    def _lower(self, step: int) -> None:
        self.limit = max(self.floor, self.limit - step)
        self.lowered_total += 1
        self._flushes_since_change = 0
