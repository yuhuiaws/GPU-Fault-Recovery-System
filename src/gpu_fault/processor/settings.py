"""Typed settings groups for :class:`ProcessorCoordinator`.

The coordinator used to take 44 keyword parameters and its factory fed them
through four untyped ``dict``s, so a renamed setting surfaced only when the
process started (review item S15). Each group below keeps the field names and
defaults the constructor had, owns the validation that involves only its own
fields, and reads its environment variables in ``from_environment`` with the
same ``os.getenv("NAME", "default")`` calls the factory used to make, so the
environment reference keeps classifying them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# One claim, one business transaction and one fenced completion may carry up
# to this many routine samples, bounded independently by the replay envelope.
TELEMETRY_SPOOL_REPLAY_BATCH_MAX_ITEMS = 64

_STALE_LIMITS_MESSAGE = "processor stale limits must be positive: "


@dataclass(frozen=True)
class ProcessorLeaseSettings:
    """Leader lease, request lease, retry and claim-cadence settings."""

    lease_seconds: int = 15
    renew_seconds: float = 3
    request_lease_seconds: int = 120
    request_renew_seconds: float = 5
    request_max_execution_seconds: float = 30
    deadline_exceeded_process_threshold: int = 3
    unhealthy_ttl_seconds: float = 300
    retryable_response_max_age_seconds: float = 300
    retry_backoff_seconds: float = 1
    retry_backoff_max_seconds: float = 30
    poll_seconds: float = 0.1
    idle_backoff_max_seconds: float = 2.0
    busy_backoff_max_seconds: float = 0.4
    fault_idle_backoff_max_seconds: float = 0.5
    fault_busy_backoff_max_seconds: float = 0.1
    processor_notification_fallback_seconds: float = 5.0
    processor_notification_shard_count: int = 8

    def __post_init__(self) -> None:
        if self.lease_seconds < 5:
            raise ValueError("processor leader lease must be at least 5s")
        if self.renew_seconds <= 0 or self.renew_seconds >= self.lease_seconds:
            raise ValueError("processor renew interval must be below its lease")
        if (
            self.request_renew_seconds <= 0
            or self.request_renew_seconds >= self.request_lease_seconds
        ):
            raise ValueError("processor request renew interval must be below its lease")
        if self.request_max_execution_seconds <= 0:
            raise ValueError(
                "processor request maximum execution time must be positive"
            )
        if self.retryable_response_max_age_seconds <= 0:
            raise ValueError(_STALE_LIMITS_MESSAGE + "retryable response")
        if (
            self.retry_backoff_seconds <= 0
            or self.retry_backoff_max_seconds < self.retry_backoff_seconds
        ):
            raise ValueError(
                "processor retry backoff must be positive and not exceed its maximum"
            )
        if self.processor_notification_fallback_seconds <= 0:
            raise ValueError("processor notification fallback must be positive")
        if self.processor_notification_shard_count <= 0:
            raise ValueError("processor notification shard count must be positive")

    @classmethod
    def from_environment(cls) -> ProcessorLeaseSettings:
        return cls(
            lease_seconds=int(
                os.getenv("GPU_FAULT_PROCESSOR_LEADER_LEASE_SECONDS", "15")
            ),
            renew_seconds=float(
                os.getenv("GPU_FAULT_PROCESSOR_LEADER_RENEW_SECONDS", "3")
            ),
            request_lease_seconds=int(
                os.getenv("GPU_FAULT_PROCESSOR_REQUEST_LEASE_SECONDS", "120")
            ),
            request_renew_seconds=float(
                os.getenv("GPU_FAULT_PROCESSOR_REQUEST_RENEW_SECONDS", "5")
            ),
            request_max_execution_seconds=float(
                os.getenv(
                    "GPU_FAULT_PROCESSOR_REQUEST_MAX_EXECUTION_SECONDS",
                    "30",
                )
            ),
            deadline_exceeded_process_threshold=int(
                os.getenv(
                    "GPU_FAULT_PROCESSOR_DEADLINE_EXCEEDED_PROCESS_THRESHOLD",
                    "3",
                )
            ),
            unhealthy_ttl_seconds=float(
                os.getenv("GPU_FAULT_PROCESSOR_UNHEALTHY_TTL_SECONDS", "300")
            ),
            retryable_response_max_age_seconds=float(
                os.getenv(
                    "GPU_FAULT_PROCESSOR_RETRYABLE_RESPONSE_MAX_AGE_SECONDS",
                    "300",
                )
            ),
            retry_backoff_seconds=float(
                os.getenv(
                    "GPU_FAULT_PROCESSOR_RETRY_BACKOFF_SECONDS",
                    "1",
                )
            ),
            retry_backoff_max_seconds=float(
                os.getenv(
                    "GPU_FAULT_PROCESSOR_RETRY_BACKOFF_MAX_SECONDS",
                    "30",
                )
            ),
            idle_backoff_max_seconds=float(
                os.getenv("GPU_FAULT_PROCESSOR_IDLE_BACKOFF_MAX_SECONDS", "2")
            ),
            busy_backoff_max_seconds=float(
                os.getenv(
                    "GPU_FAULT_PROCESSOR_BUSY_BACKOFF_MAX_SECONDS",
                    "0.4",
                )
            ),
            fault_idle_backoff_max_seconds=float(
                os.getenv(
                    "GPU_FAULT_PROCESSOR_FAULT_IDLE_BACKOFF_MAX_SECONDS",
                    "0.5",
                )
            ),
            fault_busy_backoff_max_seconds=float(
                os.getenv(
                    "GPU_FAULT_PROCESSOR_FAULT_BUSY_BACKOFF_MAX_SECONDS",
                    "0.1",
                )
            ),
            processor_notification_fallback_seconds=float(
                os.getenv(
                    "GPU_FAULT_PROCESSOR_NOTIFICATION_FALLBACK_SECONDS",
                    "5",
                )
            ),
            processor_notification_shard_count=_notification_shard_count(),
        )


def _notification_shard_count() -> int:
    """Shards must cover the consumer processes, or the losers are pollers.

    The deployment computes ``GPU_FAULT_PROCESSOR_NOTIFICATION_SHARDS`` as
    replicas x uvicorn workers; the code default of 8 knew nothing of
    that and left 16 of 24 processes without a shard (F-D11). With
    ``GPU_FAULT_PROCESSOR_CONSUMER_PROCESSES`` set the shard count is
    derived from it, and an explicit shard count below it is refused at
    startup rather than discovered as a warning per process at run time.
    """

    shards_raw = os.getenv("GPU_FAULT_PROCESSOR_NOTIFICATION_SHARDS")
    processes_raw = os.getenv("GPU_FAULT_PROCESSOR_CONSUMER_PROCESSES")
    if processes_raw is None:
        return int(shards_raw if shards_raw is not None else "8")
    processes = int(processes_raw)
    if processes <= 0:
        raise RuntimeError(
            "GPU_FAULT_PROCESSOR_CONSUMER_PROCESSES must be a positive integer"
        )
    if shards_raw is None:
        return processes
    shards = int(shards_raw)
    if shards < processes:
        raise RuntimeError(
            "GPU_FAULT_PROCESSOR_NOTIFICATION_SHARDS "
            f"({shards}) is below GPU_FAULT_PROCESSOR_CONSUMER_PROCESSES "
            f"({processes}); every consumer process needs a shard or it "
            "never receives a notification"
        )
    return shards


@dataclass(frozen=True)
class ProcessorPoolSettings:
    """Worker pool sizes and the fault-pressure scheduling bounds."""

    worker_count: int = 4
    fault_worker_count: int | None = None
    observation_worker_count: int | None = None
    gpu_telemetry_worker_count: int | None = None
    host_telemetry_worker_count: int | None = None
    fault_pressure_evidence_workers: int = 1
    routine_starvation_seconds: float = 30.0

    def __post_init__(self) -> None:
        if self.worker_count <= 0:
            raise ValueError("processor worker count must be positive")
        if self.routine_starvation_seconds <= 0:
            raise ValueError("processor routine starvation bound must be positive")
        counts = self.worker_counts()
        if counts["fault"] <= 0 or any(value < 0 for value in counts.values()):
            raise ValueError(
                "processor fault workers must be positive and "
                "other pool workers cannot be negative"
            )
        evidence_pool_sizes = [
            counts[name] for name in ("gpu", "host") if counts[name] > 0
        ]
        if self.fault_pressure_evidence_workers <= 0 or (
            evidence_pool_sizes
            and self.fault_pressure_evidence_workers > min(evidence_pool_sizes)
        ):
            raise ValueError(
                "processor fault-pressure evidence workers must be "
                "positive and not exceed a dedicated evidence pool"
            )

    @property
    def default_pool(self) -> int:
        """The per-pool size a worker budget splits into when none is explicit."""

        return max(1, self.worker_count // 4)

    def worker_counts(self) -> dict[str, int]:
        """Workers per pool: the explicit split, or every worker on faults."""

        explicit_pools = any(
            value is not None
            for value in (
                self.fault_worker_count,
                self.observation_worker_count,
                self.gpu_telemetry_worker_count,
                self.host_telemetry_worker_count,
            )
        )
        if explicit_pools:
            return {
                "fault": self.fault_worker_count or 0,
                "observation": self.observation_worker_count or 0,
                "gpu": self.gpu_telemetry_worker_count or 0,
                "host": self.host_telemetry_worker_count or 0,
            }
        return {
            "fault": self.worker_count,
            "observation": 0,
            "gpu": 0,
            "host": 0,
        }

    @classmethod
    def from_environment(cls) -> ProcessorPoolSettings:
        worker_count = int(os.getenv("GPU_FAULT_PROCESSOR_WORKERS", "4"))
        default_pool = max(1, worker_count // 4)
        return cls(
            worker_count=worker_count,
            fault_worker_count=int(
                os.getenv(
                    "GPU_FAULT_PROCESSOR_FAULT_WORKERS",
                    str(default_pool),
                )
            ),
            observation_worker_count=int(
                os.getenv(
                    "GPU_FAULT_PROCESSOR_OBSERVATION_WORKERS",
                    str(default_pool),
                )
            ),
            gpu_telemetry_worker_count=int(
                os.getenv(
                    "GPU_FAULT_PROCESSOR_GPU_TELEMETRY_WORKERS",
                    str(default_pool),
                )
            ),
            host_telemetry_worker_count=int(
                os.getenv(
                    "GPU_FAULT_PROCESSOR_HOST_TELEMETRY_WORKERS",
                    str(default_pool),
                )
            ),
            fault_pressure_evidence_workers=int(
                os.getenv(
                    "GPU_FAULT_PROCESSOR_FAULT_PRESSURE_EVIDENCE_WORKERS",
                    "1",
                )
            ),
            routine_starvation_seconds=float(
                os.getenv(
                    "GPU_FAULT_PROCESSOR_ROUTINE_STARVATION_SECONDS",
                    "30",
                )
            ),
        )


@dataclass(frozen=True)
class ProcessorStaleSettings:
    """How old each derived view may get before the processor rebuilds it."""

    gpu_inventory_stale_seconds: float = 180
    health_summary_stale_seconds: float = 420
    observation_stale_seconds: float = 120
    training_progress_stale_seconds: float = 120

    def __post_init__(self) -> None:
        stale_limits = {
            "GPU inventory": self.gpu_inventory_stale_seconds,
            "health summary": self.health_summary_stale_seconds,
            "workload observation": self.observation_stale_seconds,
            "training progress": self.training_progress_stale_seconds,
        }
        invalid_stale_limits = [
            name for name, value in stale_limits.items() if value <= 0
        ]
        if invalid_stale_limits:
            raise ValueError(_STALE_LIMITS_MESSAGE + ", ".join(invalid_stale_limits))

    @classmethod
    def from_environment(cls) -> ProcessorStaleSettings:
        return cls(
            gpu_inventory_stale_seconds=float(
                os.getenv("GPU_FAULT_PROCESSOR_GPU_INVENTORY_STALE_SECONDS", "180")
            ),
            health_summary_stale_seconds=float(
                os.getenv("GPU_FAULT_PROCESSOR_HEALTH_SUMMARY_STALE_SECONDS", "420")
            ),
            observation_stale_seconds=float(
                os.getenv("GPU_FAULT_PROCESSOR_OBSERVATION_STALE_SECONDS", "120")
            ),
            training_progress_stale_seconds=float(
                os.getenv("GPU_FAULT_PROCESSOR_TRAINING_PROGRESS_STALE_SECONDS", "120")
            ),
        )


@dataclass(frozen=True)
class ProcessorSpoolSettings:
    """Telemetry spool replay pool, lease and byte budgets."""

    telemetry_spool_enabled: bool = False
    telemetry_spool_workers: int = 4
    telemetry_spool_lease_seconds: float = 60
    telemetry_spool_retry_backoff_seconds: float = 1.0
    telemetry_spool_notification_fallback_seconds: float = 5.0
    telemetry_spool_fault_pressure_workers: int = 1
    telemetry_spool_fault_pressure_poll_seconds: float = 0.5
    telemetry_spool_max_in_flight_bytes: int = 64 * 1024 * 1024
    telemetry_spool_replay_batch_max_items: int = TELEMETRY_SPOOL_REPLAY_BATCH_MAX_ITEMS
    telemetry_spool_replay_batch_max_bytes: int = 8 * 1024 * 1024

    def __post_init__(self) -> None:
        if self.telemetry_spool_enabled and self.telemetry_spool_workers <= 0:
            raise ValueError(
                "telemetry spool workers must be positive when the spool is enabled"
            )
        if not 2 <= self.telemetry_spool_notification_fallback_seconds <= 5:
            raise ValueError(
                "telemetry spool notification fallback must be between 2 and 5 seconds"
            )
        if (
            self.telemetry_spool_fault_pressure_workers <= 0
            or self.telemetry_spool_fault_pressure_workers
            > self.telemetry_spool_workers
        ):
            raise ValueError(
                "telemetry spool fault-pressure workers must be "
                "positive and not exceed normal spool workers"
            )
        if self.telemetry_spool_fault_pressure_poll_seconds <= 0:
            raise ValueError(
                "telemetry spool fault-pressure poll interval must be positive"
            )
        if self.telemetry_spool_max_in_flight_bytes <= 0:
            raise ValueError("telemetry spool in-flight byte limit must be positive")
        if not (
            1
            <= self.telemetry_spool_replay_batch_max_items
            <= TELEMETRY_SPOOL_REPLAY_BATCH_MAX_ITEMS
        ):
            raise ValueError(
                "telemetry spool replay batch item limit must be "
                f"between 1 and {TELEMETRY_SPOOL_REPLAY_BATCH_MAX_ITEMS}"
            )
        if (
            self.telemetry_spool_replay_batch_max_bytes <= 0
            or self.telemetry_spool_replay_batch_max_bytes
            > self.telemetry_spool_max_in_flight_bytes
        ):
            raise ValueError(
                "telemetry spool replay batch byte limit must be "
                "positive and not exceed the in-flight byte limit"
            )
        # Every replay slot must be able to hold a full batch at once. With
        # ``workers x batch_max > max_in_flight`` the last free slot's byte
        # budget is smaller than the head row, and the claim -- which always
        # takes the head row -- is abandoned and retaken in a loop (E-6 /
        # F-D8). Production runs 8 x 8 MiB = 64 MiB exactly; only checked
        # when the spool is on, because the worker role derives 12 workers
        # from its processor budget and never starts the consumer.
        if (
            self.telemetry_spool_enabled
            and self.telemetry_spool_workers
            * self.telemetry_spool_replay_batch_max_bytes
            > self.telemetry_spool_max_in_flight_bytes
        ):
            raise ValueError(
                "telemetry spool workers x replay batch byte limit "
                f"({self.telemetry_spool_workers} x "
                f"{self.telemetry_spool_replay_batch_max_bytes}) must fit "
                "within the in-flight byte limit "
                f"({self.telemetry_spool_max_in_flight_bytes})"
            )

    @classmethod
    def from_environment(cls, *, default_pool: int) -> ProcessorSpoolSettings:
        max_in_flight_bytes = int(
            os.getenv(
                "GPU_FAULT_TELEMETRY_SPOOL_MAX_IN_FLIGHT_BYTES",
                str(64 * 1024 * 1024),
            )
        )
        replay_batch_max_bytes = int(
            os.getenv(
                "GPU_FAULT_TELEMETRY_SPOOL_REPLAY_BATCH_MAX_BYTES",
                str(8 * 1024 * 1024),
            )
        )
        # The derived worker count must respect the in-flight budget (E-6):
        # a worker role derives 12 from its processor budget while shipping
        # the spool-worker's 64 MiB / 8 MiB limits, and must not fail to
        # start for a consumer it never runs.
        default_workers = max(
            1,
            min(
                default_pool * 2, max_in_flight_bytes // max(1, replay_batch_max_bytes)
            ),
        )
        return cls(
            telemetry_spool_enabled=os.getenv("GPU_FAULT_TELEMETRY_SPOOL", "0")
            .strip()
            .lower()
            in {"1", "true", "yes"},
            telemetry_spool_workers=int(
                os.getenv(
                    "GPU_FAULT_TELEMETRY_SPOOL_WORKERS",
                    str(default_workers),
                )
            ),
            telemetry_spool_lease_seconds=float(
                os.getenv("GPU_FAULT_TELEMETRY_SPOOL_LEASE_SECONDS", "60")
            ),
            telemetry_spool_retry_backoff_seconds=float(
                os.getenv(
                    "GPU_FAULT_TELEMETRY_SPOOL_RETRY_BACKOFF_SECONDS",
                    "1",
                )
            ),
            telemetry_spool_notification_fallback_seconds=float(
                os.getenv(
                    "GPU_FAULT_TELEMETRY_SPOOL_NOTIFICATION_FALLBACK_SECONDS",
                    "5",
                )
            ),
            telemetry_spool_fault_pressure_workers=int(
                os.getenv(
                    "GPU_FAULT_TELEMETRY_SPOOL_FAULT_PRESSURE_WORKERS",
                    "1",
                )
            ),
            telemetry_spool_fault_pressure_poll_seconds=float(
                os.getenv(
                    "GPU_FAULT_TELEMETRY_SPOOL_FAULT_PRESSURE_POLL_SECONDS",
                    "0.5",
                )
            ),
            telemetry_spool_max_in_flight_bytes=max_in_flight_bytes,
            telemetry_spool_replay_batch_max_items=int(
                os.getenv(
                    "GPU_FAULT_TELEMETRY_SPOOL_REPLAY_BATCH_MAX_ITEMS",
                    "64",
                )
            ),
            telemetry_spool_replay_batch_max_bytes=replay_batch_max_bytes,
        )
