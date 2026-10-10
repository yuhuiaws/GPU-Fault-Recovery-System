"""Adaptive in-flight limit for the admission batchers' flushes.

Batch 48 replaced the per-process flush literals with site-wide caps that the
ingress processes divide among themselves; the live perf suite then measured
three points on the same schema (v22), the same 32-cluster burst and the same
12 ingress processes:

* a fixed cap of 24 site-wide (2 per process) shed telemetry: every process
  sat at its cap for the whole load start, ~800 entries queued per process,
  4,601 requests expired in the queue (18.7 % reserved 503s against a 2 %
  budget) and a 16-item flush took 0.59 s - queueing, not the store;
* a fixed cap of 96 (8 per process; fault 48) passed: 1,051 req/s, no 503 of
  any kind, 81 ms per flush, 60-69 entries queued per process at the peak,
  the store at 113 active statements for ~5 s with BufferContent waits at
  1 %;
* an adaptive range of 48..192 (4..16 per process; fault 24..96) with open /
  close thresholds of 1.0 / 2.0 s failed: the controller let the site reach
  ~250 concurrent flushes, the page-lock convoy on the queue table came back
  (a 69 s storm, 142 active statements, BufferContent 36 %), a flush took
  1.08 s on average - below the 2.0 s close threshold, so the slots stayed
  open while 35.8 % of telemetry was shed - and the limit oscillated (81
  raises, 51 lowers in 14 minutes). Starting at 4 per process also shed
  during the first second of the burst.

So the store on this schema tolerates ~100 concurrent admission flushes and
convoys above that, and the ceiling is the proven fixed cap. The range is
kept for the one job a fixed cap cannot do - give slots back when flush
latency says the store is contended - and the defaults make the controller
*behave* as the fixed cap unless latency says otherwise: the floor carries
the first second of a burst without shedding, which with 8 per process means
``floor == ceiling`` for the routine and fault tiers (a constant; the
adaptation is off), and the evidence tier keeps 1..2 per process.

Where a range is open, the :class:`FlushConcurrencyController` decides once
per dispatch pass:

* **open** one more slot while the batcher is *saturated* (every slot busy
  and work still queued) and the flush-latency EWMA is below
  ``open_below_seconds`` - the store is answering, more parallel flushes buy
  throughput;
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

Every change - a raise as much as a close or a decay - waits for the level to
*settle*: at least ``max(settle_flushes_min, limit)`` flushes completed since
the last change, one generation of in-flight flushes, so the EWMA reflects
the level being judged, one sample cannot cascade a wide limit down to the
floor, and a limit cannot be raised and lowered on alternate passes (the 81
raises and 51 lowers above were one flush apart). The thresholds are
multiples of the uncontended 81 ms flush: open below ~2x (0.15 s), close
above ~4x (0.30 s) - the 1.08 s average of the failed run is far past the
close threshold, so a controller with these thresholds would have been
closing hard instead of holding.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FlushConcurrencyController:
    floor: int
    ceiling: int
    open_below_seconds: float = 0.15
    close_above_seconds: float = 0.3
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
        """Flushes that must complete at a level before it may be changed."""

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
        Nothing moves until the level has settled, whichever way it would go.
        """

        if not self.adaptive or not self.settled:
            return self.limit
        if flush_ewma > self.close_above_seconds:
            if self.limit > self.floor:
                self._lower(max(1, (self.limit - self.floor + 1) // 2))
        elif saturated and flush_ewma < self.open_below_seconds:
            if self.limit < self.ceiling:
                self._raise()
        elif not saturated and self.limit > self.floor:
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
