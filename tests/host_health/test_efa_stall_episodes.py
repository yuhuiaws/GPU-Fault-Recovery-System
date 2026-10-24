from datetime import timedelta

import pytest

from gpu_fault.efa_stall_episode import EfaRecoveryPolicy
from gpu_fault.efa_traffic_state import next_efa_traffic_state
from gpu_fault.models import EfaTrafficSignal
from tests.host_health._support import NOW


def sample(previous, offset, rate, **extra):
    return next_efa_traffic_state(
        previous,
        state_key="cluster-a/node-a",
        cluster_id="cluster-a",
        node_id="node-a",
        job_id="job-a",
        attempt_id=extra.pop("attempt_id", "attempt-a"),
        observed_at=NOW + timedelta(seconds=offset),
        bytes_per_second=rate,
        minimum_active_bps=1048576,
        spike_ratio=4,
        drop_ratio=0.25,
        zero_bps=1,
        zero_warning_seconds=60,
        zero_hung_seconds=600,
        baseline_alpha=0.2,
        startup_grace_seconds=120,
        spike_event_id=f"spike-{offset}",
        **extra,
    )


def stalled():
    state = None
    for offset, rate in [
        (0, 16_000_000),
        (30, 16_000_000),
        (60, 16_000_000),
        *[(offset, 0) for offset in range(130, 731, 60)],
    ]:
        state, _ = sample(state, offset, rate)
    return state


def test_noise_or_single_spike_breaks_zero_timer_but_not_stall_episode():
    state = stalled()
    episode = state.stall_episode
    for offset, rate in [
        (740, 2),
        (750, 16_000_000),
        (780, 0),
        *[(offset, 0) for offset in range(840, 1381, 60)],
    ]:
        state, emit = sample(state, offset, rate)
        assert state.stall_episode.episode_id == episode.episode_id, state
        assert state.baseline_bytes_per_second == 16_000_000, state
        assert state.last_stable_traffic_at is None, state
        assert not (emit and state.signal is EfaTrafficSignal.HUNG_SUSPECTED), state
    assert state.zero_since == NOW + timedelta(seconds=780), state


def test_sustained_partial_baseline_recovery_rearms_new_episode_only():
    state = stalled()
    first = state.stall_episode.episode_id
    assert state.stall_episode.recovery_threshold_bps == 4_000_000, state
    for offset in (740, 800):
        state, _ = sample(state, offset, 4_000_000)
        assert state.stall_episode is not None, state
    state, emit = sample(state, 860, 4_000_000)
    assert emit and state.signal is EfaTrafficSignal.RECOVERED, state
    assert (
        state.stall_episode is None
        and state.last_stable_traffic_at == NOW + timedelta(seconds=860)
    ), state
    for offset in range(870, 1471, 60):
        state, emit = sample(state, offset, 0)
    assert emit and state.signal is EfaTrafficSignal.HUNG_SUSPECTED, state
    assert state.stall_episode.episode_id != first, state


@pytest.mark.parametrize("invalid", ["missing", "gap", "below"])
def test_incomplete_recovery_window_is_not_a_rearm(invalid):
    state = stalled()
    state, _ = sample(state, 740, 16_000_000)
    state, _ = sample(state, 800, 16_000_000)
    if invalid == "missing":
        state, _ = sample(state, 820, 0, observation_valid=False)
    elif invalid == "below":
        state, _ = sample(state, 820, 2)
    state, _ = sample(state, 900 if invalid == "gap" else 860, 16_000_000)
    assert state.stall_episode is not None and state.last_stable_traffic_at is None, (
        state
    )
    assert state.stall_episode.recovery_samples == 1, state


def test_untrusted_history_uses_fallback_and_never_active_is_protected():
    state, _ = sample(None, 0, 2_000_000)
    state, _ = sample(state, 130, 0)
    assert state.stall_episode.recovery_threshold_source == "ABSOLUTE_FALLBACK", state
    assert state.stall_episode.recovery_threshold_bps == 1048576, state
    state = None
    for offset in range(0, 1801, 60):
        state, emit = sample(state, offset, 0)
        assert not emit and state.stall_episode is None, state
    assert state.signal is EfaTrafficSignal.WARMUP, state


def test_attempt_change_resets_baseline_episode_and_rearm():
    state = stalled()
    for offset in (740, 800, 860):
        state, _ = sample(state, offset, 16_000_000)
    state, _ = sample(state, 870, 0, attempt_id="attempt-b")
    assert state.stall_episode is None and state.last_stable_traffic_at is None, state
    assert (
        state.baseline_bytes_per_second is None and state.active_baseline_samples == 0
    ), state
    assert not state.had_active_traffic, state


def test_upgrade_of_legacy_stalled_state_does_not_rearm_on_one_sample():
    legacy = stalled().model_copy(
        update={"stall_episode": None, "active_baseline_samples": 0}
    )
    state, _ = sample(legacy, 740, 16_000_000)
    assert state.stall_episode is not None and state.stall_episode.hung_reported, state
    assert state.stall_episode.recovery_threshold_source == "ABSOLUTE_FALLBACK", state
    assert (
        state.signal is not EfaTrafficSignal.RECOVERED
        and state.last_stable_traffic_at is None
    ), state


@pytest.mark.parametrize("value", ["nan", "inf", "0", "2"])
def test_invalid_recovery_ratio_is_rejected(monkeypatch, value):
    monkeypatch.setenv("GPU_FAULT_EFA_TRAFFIC_RECOVERY_RATIO", value)
    with pytest.raises(ValueError):
        EfaRecoveryPolicy.from_env()
