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
        if self.notifications_enabled:
            return self.notification_fallback
        return self.fault_idle if stream == "fault" else self.idle


def empty_round_ceiling(
    ceilings: BackoffCeilings,
    *,
    stream: str,
    next_interval: float,
    settled_empty: bool,
    pools_busy: bool,
    lane_blocked: Callable[[], bool],
) -> float:
    """The ceiling for ``stream`` after an empty claim round.

    ``settled_empty`` is the adaptive part: the stream has come back empty
    ``busy_empty_rounds_before_idle`` times in a row, so ``pools_busy`` no
    longer holds it at the busy ceiling. ``lane_blocked`` -- the store's own
    verdict on the backlog -- always does, and is consulted lazily because it
    is a round trip the idle path should not pay.
    """

    idle = ceilings.idle_ceiling(stream)

    def held() -> bool:
        return (pools_busy and not settled_empty) or lane_blocked()

    if stream == "fault":
        return ceilings.fault_busy if held() else idle
    if next_interval > ceilings.busy and held():
        return ceilings.busy
    return idle
