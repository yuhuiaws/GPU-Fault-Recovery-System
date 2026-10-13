"""The ceiling an empty claim round backs a stream off to.

Six worker replicas polling seven streams every 100 ms cost about 500 commits
a second against an empty queue, so an empty stream is skipped for a doubling
interval up to a ceiling. The ceiling depends on what the empty result means:
while this replica's pools are busy (or the store says the backlog is lane-
blocked) an empty claim means "every eligible lane is leased" and the stream
is capped at a short busy ceiling so it drains at the rate lanes free up. But
a stream that comes back empty round after round while this replica's pools
are busy is not blocked work about to free up; it is idle, and with the
edge-triggered notifications of schema v21 it is woken the moment a row
arrives. After ``busy_empty_rounds_before_idle`` consecutive empty rounds the
"pools busy" heuristic therefore stops holding it at the busy ceiling and it
backs off like an idle stream (92% of the claim rounds of a perf run were
empty while the busy ceiling held them at 0.4 s). A backlog the store reports
as lane-blocked keeps the busy ceiling regardless: its rows are already
PENDING, so no notification edge announces the lane freeing up, and polling
is the only way to drain that tail at the rate lanes free up.

The fault stream stretches to the notification fallback only while no
fault-tier row is active. The queue trigger notifies only an idle->busy (path,
priority) edge, on the assumption that a woken claim loop polls until its
streams are empty; a priority-0 row admitted while another one of its path is
PENDING (lane-blocked, deferred, held by the observation interlock) is never
announced. With every control-worker backed off to the 5 s fallback such rows
waited in 5 s steps. The invariant the coordinator keeps instead: while a
fault worker is free the fault stream is claimed at least every
``fault_idle`` (rounded up to the poll), one poll after a claim that returned
rows without filling its limit, and at once when a fault worker finishes
(``FaultSlotWake``). A silent priority-0 row therefore waits at most about
``fault_idle``; the routine streams keep the fallback, their documented
polling contract.

That cadence made 97% of a perf run's fault claim rounds empty polls of an
idle queue, and it is only needed while a row can be silent. A silent row
needs another PENDING row of its path and priority, hence of the fault tier,
and ``processor_fault_backlog_depth`` counts every fault-tier row PENDING or
LEASED (so a lease that expires without an update counts too). Once the
stream has settled empty with notifications on, the coordinator asks for that
depth instead of the lane probe, and with none the stream backs off to the
fallback: the next fault-tier row finds its key idle and notifies, and the
edge wakes every shard owner and clears the stream's backoff. A fault backlog
keeps the cadence above. The depth is read after the empty claim, so it
misses only a row that commits later without an edge -- one whose edge went
to a concurrent first writer that rolled back, or one admitted behind a
PENDING row that completed before it committed -- and such a row waits for
the stream's next poll, at most the fallback. Meanwhile the stream's
routine-tier rows (collector health, admin and fleet paths) get the routine
contract: the fallback, and no lane probe while no fault-tier row is active.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class BackoffCeilings:
    """The configured ceilings, all already clamped to at least the poll interval."""

    idle: float
    busy: float
    fault_idle: float
    fault_busy: float
    notification_fallback: float
    notifications_enabled: bool

    def idle_ceiling(self, stream: str) -> float:
        if stream == "fault":
            # Notifications never announce a row behind a PENDING one.
            if self.notifications_enabled:
                return min(self.fault_idle, self.notification_fallback)
            return self.fault_idle
        return self.notification_fallback if self.notifications_enabled else self.idle


def empty_round_ceiling(
    ceilings: BackoffCeilings,
    *,
    stream: str,
    next_interval: float,
    settled_empty: bool,
    pools_busy: bool,
    lane_blocked: Callable[[], bool],
    fault_backlog_empty: Callable[[], bool],
) -> float:
    """The ceiling for ``stream`` after an empty claim round.

    ``settled_empty`` is the adaptive part: the stream has come back empty
    ``busy_empty_rounds_before_idle`` times in a row, so ``pools_busy`` no
    longer holds it at the busy ceiling. ``lane_blocked`` -- the store's own
    verdict on the backlog -- always does, and is consulted lazily because it
    is a round trip the idle path should not pay. ``fault_backlog_empty`` --
    no fault-tier row is PENDING or LEASED -- is asked only by a settled fault
    stream with notifications on, and then in place of ``lane_blocked``.
    """

    idle = ceilings.idle_ceiling(stream)

    def held() -> bool:
        return (pools_busy and not settled_empty) or lane_blocked()

    if stream == "fault":
        if ceilings.notifications_enabled and settled_empty and fault_backlog_empty():
            return ceilings.notification_fallback
        return ceilings.fault_busy if held() else idle
    if next_interval > ceilings.busy and held():
        return ceilings.busy
    return idle
