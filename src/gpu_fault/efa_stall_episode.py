"""Traffic rearm is distinct from both a near-zero streak and job recovery."""

from __future__ import annotations

import hashlib
import os
from typing import Any

from pydantic import Field

from gpu_fault.models import (
    EfaStallEpisode,
    EfaTrafficSignal,
    EfaTrafficState,
    StrictModel,
)


class EfaRecoveryPolicy(StrictModel):
    ratio: float = Field(default=0.25, gt=0, le=1, allow_inf_nan=False)
    noise_floor_bps: float = Field(default=65536, gt=0, allow_inf_nan=False)
    fallback_bps: float = Field(default=1048576, gt=0, allow_inf_nan=False)
    sustained_seconds: float = Field(default=120, ge=60, le=3600, allow_inf_nan=False)
    minimum_samples: int = Field(default=3, ge=3, le=120)
    max_gap_seconds: float = Field(default=60, ge=1, le=360, allow_inf_nan=False)

    @classmethod
    def from_env(cls) -> EfaRecoveryPolicy:
        return cls(
            ratio=float(os.getenv("GPU_FAULT_EFA_TRAFFIC_RECOVERY_RATIO", "0.25")),
            noise_floor_bps=float(
                os.getenv("GPU_FAULT_EFA_TRAFFIC_RECOVERY_FLOOR_BPS", "65536")
            ),
            fallback_bps=float(
                os.getenv("GPU_FAULT_EFA_TRAFFIC_RECOVERY_FALLBACK_BPS", "1048576")
            ),
            sustained_seconds=float(
                os.getenv("GPU_FAULT_EFA_TRAFFIC_RECOVERY_SECONDS", "120")
            ),
            minimum_samples=int(
                os.getenv("GPU_FAULT_EFA_TRAFFIC_RECOVERY_MIN_SAMPLES", "3")
            ),
            max_gap_seconds=float(
                os.getenv("GPU_FAULT_EFA_TRAFFIC_RECOVERY_MAX_GAP_SECONDS", "60")
            ),
        )


def episode_parameters(state: EfaTrafficState) -> dict[str, Any]:
    return {
        "efa_stall_episode": state.stall_episode.model_dump(mode="json")
        if state.stall_episode
        else None,
        "efa_last_stable_traffic_at": state.last_stable_traffic_at.isoformat()
        if state.last_stable_traffic_at
        else None,
        "traffic_rearm_is_training_recovery": False,
    }


def advance_episode(
    previous: EfaTrafficState | None,
    state: EfaTrafficState,
    policy: EfaRecoveryPolicy,
    *,
    valid: bool,
    minimum_active_bps: float,
    zero_bps: float,
) -> tuple[EfaTrafficState, bool]:
    episode = (
        previous.stall_episode.model_copy()
        if previous and previous.stall_episode
        else None
    )
    rearmed = previous.last_stable_traffic_at if previous else None
    count = previous.active_baseline_samples if previous else 0
    low_signals = {
        EfaTrafficSignal.ZERO_PENDING,
        EfaTrafficSignal.ZERO_WARNING,
        EfaTrafficSignal.HUNG_SUSPECTED,
    }
    legacy_stall = previous is not None and previous.signal in low_signals
    if episode is None and (state.signal in low_signals or legacy_stall):
        baseline = previous.baseline_bytes_per_second if previous else None
        trusted = baseline is not None and baseline >= minimum_active_bps and count >= 3
        start = (
            previous.zero_first_seen_at or previous.observed_at
            if legacy_stall and previous
            else state.observed_at
        )
        identity = (
            f"{state.state_key}/{state.job_id}/{state.attempt_id}/{start.isoformat()}"
        )
        episode = EfaStallEpisode(
            episode_id=hashlib.sha256(identity.encode()).hexdigest()[:32],
            started_at=start,
            baseline_bytes_per_second=baseline if trusted else None,
            recovery_threshold_bps=max(
                (baseline or 0) * policy.ratio if trusted else policy.fallback_bps,
                policy.noise_floor_bps,
                zero_bps * 2,
            ),
            recovery_threshold_source="PRE_STALL_BASELINE"
            if trusted
            else "ABSOLUTE_FALLBACK",
            hung_reported=bool(
                legacy_stall
                and previous
                and previous.signal is EfaTrafficSignal.HUNG_SUSPECTED
            ),
        )
    suppress_hung = False
    if episode is not None:
        # Do not train the baseline on noise, low traffic or a recovery spike.
        if previous is not None:
            state = state.model_copy(
                update={"baseline_bytes_per_second": previous.baseline_bytes_per_second}
            )
        gap = (
            (state.observed_at - previous.observed_at).total_seconds()
            if previous
            else 0
        )
        if (
            not valid
            or gap > policy.max_gap_seconds
            or state.last_bytes_per_second < episode.recovery_threshold_bps
        ):
            episode.recovery_since = None
            episode.recovery_samples = 0
        if valid and state.last_bytes_per_second >= episode.recovery_threshold_bps:
            episode.recovery_since = episode.recovery_since or state.observed_at
            episode.recovery_samples += 1
            if (
                episode.recovery_samples >= policy.minimum_samples
                and (state.observed_at - episode.recovery_since).total_seconds()
                >= policy.sustained_seconds
            ):
                episode = None
                rearmed = state.observed_at
                state = state.model_copy(update={"signal": EfaTrafficSignal.RECOVERED})
        if episode is not None:
            if state.signal is EfaTrafficSignal.RECOVERED:
                state = state.model_copy(update={"signal": EfaTrafficSignal.NORMAL})
            if state.signal is EfaTrafficSignal.HUNG_SUSPECTED:
                suppress_hung = episode.hung_reported
                episode.hung_reported = True
    elif (
        valid
        and state.last_bytes_per_second >= minimum_active_bps
        and state.signal
        in {
            EfaTrafficSignal.WARMUP,
            EfaTrafficSignal.NORMAL,
            EfaTrafficSignal.RECOVERED,
        }
    ):
        count = min(count + 1, 1_000_000)
    return state.model_copy(
        update={
            "stall_episode": episode,
            "active_baseline_samples": count,
            "last_stable_traffic_at": rearmed,
        }
    ), suppress_hung
