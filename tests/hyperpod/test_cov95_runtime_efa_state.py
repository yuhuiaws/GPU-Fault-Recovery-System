from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from gpu_fault.efa_traffic_state import (
    apply_efa_traffic_admin_action,
    efa_traffic_admin_decision_id,
    efa_traffic_state_key,
    next_efa_traffic_state,
)
from gpu_fault.models import EfaTrafficAdminAction, EfaTrafficSignal, EfaTrafficState
from gpu_fault.store.shared.errors import EfaTrafficAdminConflict

NOW = datetime(2026, 9, 12, 12, tzinfo=timezone.utc)
KEY = "cluster-a/node-a/job-a/attempt-a"


def sample(
    previous: EfaTrafficState | None = None,
    *,
    seconds: int = 0,
    bps: float = 1000,
    event_id: str = "spike-first",
    **overrides: Any,
) -> tuple[EfaTrafficState, bool]:
    values = {
        "state_key": KEY,
        "cluster_id": "cluster-a",
        "node_id": "node-a",
        "job_id": "job-a",
        "attempt_id": "attempt-a",
        "observed_at": NOW + timedelta(seconds=seconds),
        "bytes_per_second": bps,
        "minimum_active_bps": 100,
        "spike_ratio": 2,
        "drop_ratio": 0.5,
        "zero_bps": 0,
        "zero_warning_seconds": 20,
        "zero_hung_seconds": 60,
        "baseline_alpha": 0.25,
        "startup_grace_seconds": 30,
        "spike_event_id": event_id,
        **overrides,
    }
    return next_efa_traffic_state(previous, **values)


def test_traffic_baseline_requires_active_sample_and_ignores_warmup_for_alerting() -> (
    None
):
    low, emit_low = sample(bps=50)
    assert low.signal is EfaTrafficSignal.WARMUP and emit_low is False, low
    assert low.baseline_bytes_per_second is None and low.had_active_traffic is False, (
        low
    )
    active, emit_active = sample(low, seconds=10, bps=1000)
    assert active.baseline_bytes_per_second == 1000 and active.had_active_traffic, (
        active
    )
    assert active.signal is EfaTrafficSignal.WARMUP and emit_active is False, active
    normal, emit_normal = sample(active, seconds=20, bps=1200)
    assert normal.signal is EfaTrafficSignal.NORMAL and emit_normal is False, normal
    assert normal.baseline_bytes_per_second == 1050, normal
    assert normal.first_observed_at == NOW, normal
    assert efa_traffic_state_key("cluster-a", "node-a", "job-a", "attempt-a") == KEY, (
        normal
    )


@pytest.mark.parametrize("had_active", [False, True])
def test_zero_traffic_cannot_signal_hang_before_active_history_and_startup_grace(
    had_active: bool,
) -> None:
    initial, _ = sample(bps=1000 if had_active else 0)
    state, emit = sample(initial, seconds=10 if had_active else 1000, bps=0)
    assert state.signal is EfaTrafficSignal.WARMUP and emit is False, state
    assert state.zero_since is None and state.zero_first_seen_at is None, state


def test_zero_traffic_moves_through_thresholds_and_recovery_once() -> None:
    state, _ = sample()
    expected = [
        (30, 0, EfaTrafficSignal.ZERO_PENDING, False, 1),
        (50, 0, EfaTrafficSignal.ZERO_WARNING, True, 2),
        (51, 0, EfaTrafficSignal.ZERO_WARNING, False, 3),
        (90, 0, EfaTrafficSignal.HUNG_SUSPECTED, True, 4),
        (91, 1000, EfaTrafficSignal.RECOVERED, True, 0),
        (92, 1000, EfaTrafficSignal.NORMAL, False, 0),
    ]
    for seconds, bps, signal, expected_emit, count in expected:
        state, emit = sample(state, seconds=seconds, bps=bps)
        assert (state.signal, emit, state.consecutive_zero_samples) == (
            signal,
            expected_emit,
            count,
        ), state
    assert state.zero_since is state.zero_first_seen_at is None, state
    assert state.baseline_bytes_per_second == 1000, state


@pytest.mark.parametrize(
    ("seconds", "progress_seconds", "cap", "expected_since", "signal"),
    [
        (40, 45, None, 40, EfaTrafficSignal.ZERO_PENDING),
        (110, 200, 20, 50, EfaTrafficSignal.HUNG_SUSPECTED),
        (50, 25, None, 30, EfaTrafficSignal.ZERO_WARNING),
    ],
)
def test_progress_suppression_is_capped_by_observed_time_and_first_zero_sample(
    seconds: int,
    progress_seconds: int,
    cap: int | None,
    expected_since: int,
    signal: EfaTrafficSignal,
) -> None:
    initial, _ = sample()
    zero, _ = sample(initial, seconds=30, bps=0)
    result, _ = sample(
        zero,
        seconds=seconds,
        bps=0,
        progress_observed_at=NOW + timedelta(seconds=progress_seconds),
        progress_suppression_max_seconds=cap,
    )
    assert result.signal is signal, result
    assert result.zero_since == NOW + timedelta(seconds=expected_since), result
    assert result.zero_first_seen_at == NOW + timedelta(seconds=30), result
    assert result.consecutive_zero_samples == 2, result


@pytest.mark.parametrize(
    ("bps", "signal", "baseline", "emitted"),
    [
        (2000, EfaTrafficSignal.SPIKE, 1000, True),
        (500, EfaTrafficSignal.DROP, 1000, True),
        (1500, EfaTrafficSignal.NORMAL, 1125, False),
    ],
)
def test_spike_and_drop_do_not_pollute_the_learned_baseline(
    bps: float, signal: EfaTrafficSignal, baseline: float, emitted: bool
) -> None:
    initial, _ = sample()
    result, emit = sample(initial, seconds=10, bps=bps)
    assert result.signal is signal and emit is emitted, result
    assert result.baseline_bytes_per_second == baseline, result


@pytest.mark.parametrize("seconds", [9, 10])
def test_stale_or_duplicate_samples_preserve_state_and_emit_nothing(
    seconds: int,
) -> None:
    initial, _ = sample()
    spike, _ = sample(initial, seconds=10, bps=3000)
    result, emit = sample(spike, seconds=seconds, bps=0, event_id="not-a-new-event")
    assert result is spike and emit is False, result
    assert result.active_spike_event_id == "spike-first", result


def test_spike_acknowledgement_survives_same_transition_and_clears_on_recovery() -> (
    None
):
    initial, _ = sample()
    spike, _ = sample(initial, seconds=10, bps=3000)
    acknowledged, decision = apply_efa_traffic_admin_action(
        spike,
        event_id="spike-first",
        action=EfaTrafficAdminAction.ACKNOWLEDGE_TRANSIENT,
        operator="unit-operator",
        reason="expected data transfer",
        decided_at=NOW + timedelta(seconds=11),
    )
    repeated, emit = sample(
        acknowledged, seconds=12, bps=3500, event_id="ignored-event"
    )
    assert repeated.signal is EfaTrafficSignal.SPIKE and emit is False, repeated
    assert repeated.active_spike_event_id == "spike-first", repeated
    assert repeated.spike_acknowledged_at == NOW + timedelta(seconds=11), repeated
    assert repeated.spike_acknowledged_by == "unit-operator", repeated
    assert decision.accepted_sample_bytes_per_second is None, decision
    recovered, _ = sample(repeated, seconds=13, bps=1000)
    assert recovered.signal is EfaTrafficSignal.NORMAL, recovered
    assert recovered.active_spike_event_id is recovered.spike_acknowledged_at is None, (
        recovered
    )
    assert (
        recovered.spike_acknowledged_by
        is recovered.spike_acknowledgement_reason
        is None
    ), recovered


def test_accepting_baseline_records_scope_and_the_sample_that_changed_it() -> None:
    initial, _ = sample()
    spike, _ = sample(initial, seconds=10, bps=5000)
    updated, decision = apply_efa_traffic_admin_action(
        spike,
        event_id="spike-first",
        action=EfaTrafficAdminAction.ACCEPT_NEW_BASELINE,
        operator="unit-operator",
        reason="approved workload scale",
        decided_at=NOW + timedelta(seconds=11),
    )
    assert (
        updated.signal is EfaTrafficSignal.NORMAL
        and updated.baseline_bytes_per_second == 5000
    ), updated
    assert decision.previous_baseline_bytes_per_second == 1000, decision
    assert (
        decision.resulting_baseline_bytes_per_second
        == decision.accepted_sample_bytes_per_second
        == 5000
    ), decision
    assert (
        decision.cluster_id,
        decision.node_id,
        decision.job_id,
        decision.attempt_id,
    ) == ("cluster-a", "node-a", "job-a", "attempt-a"), decision
    assert decision.decision_id == efa_traffic_admin_decision_id(
        "spike-first", EfaTrafficAdminAction.ACCEPT_NEW_BASELINE
    ), decision
    assert updated.active_spike_event_id is None, updated


@pytest.mark.parametrize("wrong_event", [False, True])
def test_admin_decision_requires_the_current_spike_transition(
    wrong_event: bool,
) -> None:
    state, _ = sample()
    if wrong_event:
        state, _ = sample(state, seconds=10, bps=3000)
    before = state.model_dump()
    with pytest.raises(EfaTrafficAdminConflict, match="active SPIKE|currently SPIKE"):
        apply_efa_traffic_admin_action(
            state,
            event_id="foreign-event",
            action=EfaTrafficAdminAction.ACKNOWLEDGE_TRANSIENT,
            operator="unit-operator",
            reason="unit reason",
            decided_at=NOW + timedelta(seconds=11),
        )
    assert state.model_dump() == before, state
