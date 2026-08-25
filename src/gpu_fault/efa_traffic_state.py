from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from gpu_fault.models import (
    EfaTrafficAdminAction,
    EfaTrafficAdminDecision,
    EfaTrafficSignal,
    EfaTrafficState,
)
from gpu_fault.store.shared.errors import EfaTrafficAdminConflict


@dataclass
class _EfaWorkingState:
    first_observed_at: datetime
    baseline: float | None
    had_active: bool
    zero_since: datetime | None
    zero_first_seen_at: datetime | None
    zero_samples: int
    old_signal: EfaTrafficSignal
    active_spike_event_id: str | None
    spike_acknowledged_at: datetime | None
    spike_acknowledged_by: str | None
    spike_acknowledgement_reason: str | None

    @classmethod
    def from_previous(
        cls,
        previous: EfaTrafficState | None,
        observed_at: datetime,
    ) -> _EfaWorkingState:
        return cls(
            first_observed_at=(
                previous.first_observed_at if previous is not None else observed_at
            ),
            baseline=(
                previous.baseline_bytes_per_second if previous is not None else None
            ),
            had_active=(previous.had_active_traffic if previous is not None else False),
            zero_since=(previous.zero_since if previous is not None else None),
            zero_first_seen_at=(
                previous.zero_first_seen_at if previous is not None else None
            ),
            zero_samples=(
                previous.consecutive_zero_samples if previous is not None else 0
            ),
            old_signal=(
                previous.signal if previous is not None else EfaTrafficSignal.WARMUP
            ),
            active_spike_event_id=(
                previous.active_spike_event_id if previous is not None else None
            ),
            spike_acknowledged_at=(
                previous.spike_acknowledged_at if previous is not None else None
            ),
            spike_acknowledged_by=(
                previous.spike_acknowledged_by if previous is not None else None
            ),
            spike_acknowledgement_reason=(
                previous.spike_acknowledgement_reason if previous is not None else None
            ),
        )


def _zero_traffic_signal(
    state: _EfaWorkingState,
    *,
    observed_at: datetime,
    startup_grace_seconds: float,
    progress_observed_at: datetime | None,
    progress_suppression_max_seconds: float | None,
    zero_warning_seconds: float,
    zero_hung_seconds: float,
) -> EfaTrafficSignal:
    state.zero_samples += 1
    elapsed = (observed_at - state.first_observed_at).total_seconds()
    if not (state.had_active and elapsed >= startup_grace_seconds):
        return EfaTrafficSignal.WARMUP
    state.zero_since = state.zero_since or observed_at
    state.zero_first_seen_at = state.zero_first_seen_at or state.zero_since
    if progress_observed_at is not None and progress_observed_at > state.zero_since:
        restart_at = min(progress_observed_at, observed_at)
        if progress_suppression_max_seconds is not None:
            restart_at = min(
                restart_at,
                state.zero_first_seen_at
                + timedelta(seconds=progress_suppression_max_seconds),
            )
        state.zero_since = max(state.zero_since, restart_at)
    zero_duration = (observed_at - state.zero_since).total_seconds()
    if zero_duration >= zero_hung_seconds:
        return EfaTrafficSignal.HUNG_SUSPECTED
    if zero_duration >= zero_warning_seconds:
        return EfaTrafficSignal.ZERO_WARNING
    return EfaTrafficSignal.ZERO_PENDING


def _active_traffic_signal(
    state: _EfaWorkingState,
    *,
    bytes_per_second: float,
    minimum_active_bps: float,
    spike_ratio: float,
    drop_ratio: float,
    baseline_alpha: float,
) -> EfaTrafficSignal:
    recovering = (
        state.old_signal
        in {
            EfaTrafficSignal.ZERO_PENDING,
            EfaTrafficSignal.ZERO_WARNING,
            EfaTrafficSignal.HUNG_SUSPECTED,
        }
        and state.had_active
    )
    state.zero_since = None
    state.zero_first_seen_at = None
    state.zero_samples = 0
    if state.baseline is None:
        if bytes_per_second >= minimum_active_bps:
            state.baseline = bytes_per_second
            state.had_active = True
        return EfaTrafficSignal.WARMUP
    ratio = bytes_per_second / max(state.baseline, 1.0)
    if ratio >= spike_ratio:
        signal = EfaTrafficSignal.SPIKE
    elif ratio <= drop_ratio:
        signal = EfaTrafficSignal.DROP
    elif recovering:
        signal = EfaTrafficSignal.RECOVERED
    else:
        signal = EfaTrafficSignal.NORMAL
    if signal in {
        EfaTrafficSignal.NORMAL,
        EfaTrafficSignal.RECOVERED,
    }:
        state.baseline = (
            state.baseline * (1.0 - baseline_alpha) + bytes_per_second * baseline_alpha
        )
    state.had_active = state.had_active or bytes_per_second >= minimum_active_bps
    return signal


def next_efa_traffic_state(
    previous: EfaTrafficState | None,
    *,
    state_key: str,
    cluster_id: str,
    node_id: str,
    job_id: str,
    attempt_id: str,
    observed_at: datetime,
    bytes_per_second: float,
    minimum_active_bps: float,
    spike_ratio: float,
    drop_ratio: float,
    zero_bps: float,
    zero_warning_seconds: float,
    zero_hung_seconds: float,
    baseline_alpha: float,
    startup_grace_seconds: float,
    spike_event_id: str,
    progress_observed_at: datetime | None = None,
    progress_suppression_max_seconds: float | None = None,
) -> tuple[EfaTrafficState, bool]:
    if previous is not None and observed_at <= previous.observed_at:
        return previous, False
    working = _EfaWorkingState.from_previous(previous, observed_at)
    if bytes_per_second <= zero_bps:
        signal = _zero_traffic_signal(
            working,
            observed_at=observed_at,
            startup_grace_seconds=startup_grace_seconds,
            progress_observed_at=progress_observed_at,
            progress_suppression_max_seconds=(progress_suppression_max_seconds),
            zero_warning_seconds=zero_warning_seconds,
            zero_hung_seconds=zero_hung_seconds,
        )
    else:
        signal = _active_traffic_signal(
            working,
            bytes_per_second=bytes_per_second,
            minimum_active_bps=minimum_active_bps,
            spike_ratio=spike_ratio,
            drop_ratio=drop_ratio,
            baseline_alpha=baseline_alpha,
        )
    if (
        signal is EfaTrafficSignal.SPIKE
        and working.old_signal is not EfaTrafficSignal.SPIKE
    ):
        working.active_spike_event_id = spike_event_id
        working.spike_acknowledged_at = None
        working.spike_acknowledged_by = None
        working.spike_acknowledgement_reason = None
    elif signal is not EfaTrafficSignal.SPIKE:
        working.active_spike_event_id = None
        working.spike_acknowledged_at = None
        working.spike_acknowledged_by = None
        working.spike_acknowledgement_reason = None
    state = EfaTrafficState(
        state_key=state_key,
        cluster_id=cluster_id,
        node_id=node_id,
        job_id=job_id,
        attempt_id=attempt_id,
        first_observed_at=working.first_observed_at,
        observed_at=observed_at,
        baseline_bytes_per_second=working.baseline,
        last_bytes_per_second=bytes_per_second,
        had_active_traffic=working.had_active,
        zero_since=working.zero_since,
        zero_first_seen_at=working.zero_first_seen_at,
        consecutive_zero_samples=working.zero_samples,
        signal=signal,
        active_spike_event_id=working.active_spike_event_id,
        spike_acknowledged_at=working.spike_acknowledged_at,
        spike_acknowledged_by=working.spike_acknowledged_by,
        spike_acknowledgement_reason=(working.spike_acknowledgement_reason),
    )
    emit = (
        signal
        in {
            EfaTrafficSignal.SPIKE,
            EfaTrafficSignal.DROP,
            EfaTrafficSignal.ZERO_WARNING,
            EfaTrafficSignal.HUNG_SUSPECTED,
            EfaTrafficSignal.RECOVERED,
        }
        and signal != working.old_signal
    )
    return state, emit


def efa_traffic_state_key(
    cluster_id: str,
    node_id: str,
    job_id: str,
    attempt_id: str,
) -> str:
    return "/".join([cluster_id, node_id, job_id, attempt_id])


def efa_traffic_admin_decision_id(event_id: str, action: EfaTrafficAdminAction) -> str:
    return f"efa-traffic-admin/{event_id}/{action.value}"


def apply_efa_traffic_admin_action(
    state: EfaTrafficState,
    *,
    event_id: str,
    action: EfaTrafficAdminAction,
    operator: str,
    reason: str,
    decided_at: datetime,
) -> tuple[EfaTrafficState, EfaTrafficAdminDecision]:
    if state.signal is not EfaTrafficSignal.SPIKE:
        raise EfaTrafficAdminConflict("EFA traffic state is not currently SPIKE")
    if state.active_spike_event_id != event_id:
        raise EfaTrafficAdminConflict(
            "event_id does not identify the active SPIKE transition"
        )
    previous_baseline = state.baseline_bytes_per_second
    if action is EfaTrafficAdminAction.ACCEPT_NEW_BASELINE:
        updated = state.model_copy(
            update={
                "baseline_bytes_per_second": (state.last_bytes_per_second),
                "signal": EfaTrafficSignal.NORMAL,
                "active_spike_event_id": None,
                "spike_acknowledged_at": None,
                "spike_acknowledged_by": None,
                "spike_acknowledgement_reason": None,
            }
        )
    else:
        updated = state.model_copy(
            update={
                "spike_acknowledged_at": decided_at,
                "spike_acknowledged_by": operator,
                "spike_acknowledgement_reason": reason,
            }
        )
    decision = EfaTrafficAdminDecision(
        decision_id=efa_traffic_admin_decision_id(event_id, action),
        cluster_id=state.cluster_id,
        node_id=state.node_id,
        job_id=state.job_id,
        attempt_id=state.attempt_id,
        event_id=event_id,
        action=action,
        operator=operator,
        reason=reason,
        decided_at=decided_at,
        previous_signal=state.signal,
        resulting_signal=updated.signal,
        previous_baseline_bytes_per_second=previous_baseline,
        resulting_baseline_bytes_per_second=(updated.baseline_bytes_per_second),
        accepted_sample_bytes_per_second=(
            state.last_bytes_per_second
            if action is EfaTrafficAdminAction.ACCEPT_NEW_BASELINE
            else None
        ),
    )
    return updated, decision
