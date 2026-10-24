"""A short delivery window prevents edge coalescing from hiding RDMA recovery."""

from datetime import datetime, timedelta

from gpu_fault.efa_stall_episode import EfaRecoveryPolicy
from gpu_fault.efa_traffic_state import traffic_observation
from gpu_fault.host_health import HostTelemetryBatch
from gpu_fault.models import WorkloadState


class EfaRecoverySampling:
    def __init__(self) -> None:
        self.policy = EfaRecoveryPolicy.from_env()
        self.below_floor = False
        self.until: datetime | None = None

    def due(self, batch: HostTelemetryBatch, delivered_at: datetime | None) -> bool:
        if batch.workload_state is WorkloadState.IDLE:
            self.below_floor, self.until = False, None
            return False
        observation = traffic_observation(batch.samples)
        if observation is None or not observation[1]:
            return False
        rate = observation[0]
        spacing = min(30.0, self.policy.max_gap_seconds / 2)
        if self.below_floor and rate >= self.policy.noise_floor_bps:
            duration = max(
                self.policy.sustained_seconds + 2 * spacing,
                self.policy.minimum_samples * spacing,
            )
            self.until = batch.observed_at + timedelta(seconds=duration)
        self.below_floor = rate < self.policy.noise_floor_bps
        return bool(
            self.until is not None
            and batch.observed_at <= self.until
            and (
                delivered_at is None
                or (batch.observed_at - delivered_at).total_seconds() >= spacing
            )
        )
