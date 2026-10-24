"""Bounded application phase hints; a missing heartbeat is never a pause proof."""

from __future__ import annotations

from datetime import datetime
from typing import Any

PAUSE_BUDGETS = {
    "INITIALIZING": 300,
    "CHECKPOINT_LOAD": 1800,
    "CHECKPOINT_SAVE": 1800,
    "RESTORING": 1800,
}


def phase(heartbeat: Any) -> str:
    value = heartbeat.labels.get("phase", "TRAINING").upper()
    return value if value in {*PAUSE_BUDGETS, "TRAINING"} else "UNKNOWN"


def phase_key(heartbeat: Any) -> str:
    return (
        f"training-phase/{heartbeat.cluster_id}/{heartbeat.attempt_id}/{heartbeat.rank}"
    )


def observe_phase(store: Any, heartbeat: Any, *, progressed: bool = False) -> None:
    key = phase_key(heartbeat)
    active = phase(heartbeat) in PAUSE_BUDGETS
    if active or progressed:
        previous = store.get_health_signal_state(key)
        marker = (
            f"progress:{heartbeat.step}"
            if progressed
            else (previous.semantic_fingerprint if previous is not None else None)
        )
        store.claim_health_signal_transition(
            key,
            active,
            heartbeat.observed_at,
            semantic_fingerprint=marker,
        )
    # A phase label flapping through TRAINING without a new step cannot
    # replenish the pause budget. paused() still rejects non-pause phases.


def paused(store: Any, heartbeat: Any, now: datetime) -> bool:
    budget = PAUSE_BUDGETS.get(phase(heartbeat))
    state = store.get_health_signal_state(phase_key(heartbeat))
    if (
        budget is None
        or state is None
        or not state.active
        or state.active_since is None
    ):
        return False
    since = max(state.active_since, state.semantic_since or state.active_since)
    return bool(0 <= (now - since).total_seconds() <= budget)
