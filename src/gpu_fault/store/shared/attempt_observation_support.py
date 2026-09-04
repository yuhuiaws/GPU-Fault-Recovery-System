from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Protocol

from gpu_fault.models import TerminalEvent

if TYPE_CHECKING:

    class AttemptObservationTerminalSupport(Protocol):
        def _terminalize_attempt_observation(self, event: TerminalEvent) -> bool: ...

        def _reconcile_terminal_attempt_observations(self, limit: int) -> int: ...

    class MemoryAttemptEventState(Protocol):
        _attempt_event_keys: dict[tuple[str, str], str]
        _events: dict[str, TerminalEvent]

else:

    class AttemptObservationTerminalSupport:
        pass

    class MemoryAttemptEventState:
        pass


def terminalize_attempt_observation(
    store: AttemptObservationTerminalSupport,
    event: TerminalEvent,
) -> bool:
    return store._terminalize_attempt_observation(event)


def reconcile_terminal_attempt_observations(
    store: AttemptObservationTerminalSupport,
    limit: int,
) -> int:
    return store._reconcile_terminal_attempt_observations(limit)


def memory_attempt_terminal_event(
    store: MemoryAttemptEventState,
    cluster_id: str,
    attempt_id: str,
) -> TerminalEvent | None:
    event_key = store._attempt_event_keys.get((cluster_id, attempt_id))
    return store._events.get(event_key) if event_key is not None else None


def memory_terminal_events(
    store: MemoryAttemptEventState,
) -> Iterable[TerminalEvent]:
    return store._events.values()


def bound_attempt_observation_states(
    states: list[Any],
    *,
    limit: int | None,
    newest_first: bool,
) -> list[Any]:
    """Order and truncate a read of the attempt-observation table.

    Attempt observations are retained for
    ``GPU_FAULT_ATTEMPT_OBSERVATION_MAX_AGE_SECONDS`` (7 days by default), so on a
    busy fleet the table holds every training attempt from the last week. The
    ``/metrics`` ownership family read all of it on every scrape; this is how it
    asks for a bounded newest-first slice instead, the same shape
    ``list_workflows`` already offers.

    Sorting is skipped entirely when the caller asks for neither a bound nor an
    order. Six callers read this method for their own filtering and have always
    got the storage order; imposing one on them would be a silent behaviour
    change that buys nothing.
    """

    if limit is not None and limit < 0:
        raise ValueError("attempt observation scan limit must not be negative")
    if limit is None and not newest_first:
        return states
    ordered = sorted(
        states,
        key=lambda item: (
            item.observation.observed_at,
            item.observation.attempt_id,
        ),
        reverse=newest_first,
    )
    return ordered if limit is None else ordered[:limit]
