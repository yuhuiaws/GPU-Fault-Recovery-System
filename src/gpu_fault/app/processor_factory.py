from __future__ import annotations

import logging
import os
import secrets
import time
from threading import Thread
from urllib.parse import urlsplit

from gpu_fault.app.identity import pod_process_owner
from gpu_fault.processor import ProcessorCoordinator


LOGGER = logging.getLogger(__name__)


class ProcessorFactory:
    def __init__(
        self,
        context,
        *,
        mode: str,
        exit_grace_seconds: float,
    ) -> None:
        self.context = context
        self.mode = mode
        self.exit_grace_seconds = exit_grace_seconds
        self.processor: ProcessorCoordinator | None = None

    def build(self) -> ProcessorCoordinator | None:
        if self.mode not in {
            "direct",
            "active-active",
        }:
            raise RuntimeError(
                "GPU_FAULT_PROCESSOR_MODE must be direct or "
                "active-active; active-standby has been removed"
            )
        if self.exit_grace_seconds < 0:
            raise ValueError("processor exit grace seconds cannot be negative")
        if self.mode == "direct":
            return None
        owner_id, replay_secret, local_url = self._identity()
        worker_count = int(os.getenv("GPU_FAULT_PROCESSOR_WORKERS", "4"))
        default_pool = max(1, worker_count // 4)
        kwargs = {
            **self._lease_settings(),
            **self._pool_settings(worker_count, default_pool),
            **self._stale_settings(),
            **self._spool_settings(default_pool),
        }
        self.processor = ProcessorCoordinator(
            self.context.store,
            owner_id=owner_id,
            internal_token=replay_secret,
            execution_token=self.context.execution_token,
            local_url=local_url,
            active_consumers=self.mode == "active-active",
            on_unhealthy=self._recycle,
            **kwargs,
        )
        return self.processor

    def _identity(self) -> tuple[str, str, str]:
        owner_id = pod_process_owner()
        if not owner_id:
            raise RuntimeError("queued processor mode requires POD_UID")
        if not self.context.execution_token:
            raise RuntimeError("queued processor mode requires the execution token")
        replay_secret = self.context.processor_replay_secret or os.getenv(
            "GPU_FAULT_PROCESSOR_REPLAY_SECRET", ""
        )
        if len(replay_secret) < 32:
            raise RuntimeError(
                "queued processor mode requires an independent "
                "GPU_FAULT_PROCESSOR_REPLAY_SECRET of at least "
                "32 characters"
            )
        if secrets.compare_digest(
            replay_secret,
            self.context.execution_token,
        ):
            raise RuntimeError(
                "GPU_FAULT_PROCESSOR_REPLAY_SECRET must differ from "
                "GPU_FAULT_EXECUTION_TOKEN"
            )
        local_url = os.getenv(
            "GPU_FAULT_PROCESSOR_LOCAL_URL",
            "http://127.0.0.1:8080",
        )
        host = urlsplit(local_url).hostname or ""
        if host not in {"127.0.0.1", "::1", "localhost"}:
            raise RuntimeError(
                "GPU_FAULT_PROCESSOR_LOCAL_URL must use a loopback "
                "host so replay traffic cannot traverse the NLB"
            )
        return owner_id, replay_secret, local_url

    @staticmethod
    def _lease_settings() -> dict:
        return {
            "lease_seconds": int(
                os.getenv("GPU_FAULT_PROCESSOR_LEADER_LEASE_SECONDS", "15")
            ),
            "renew_seconds": float(
                os.getenv("GPU_FAULT_PROCESSOR_LEADER_RENEW_SECONDS", "3")
            ),
            "request_lease_seconds": int(
                os.getenv("GPU_FAULT_PROCESSOR_REQUEST_LEASE_SECONDS", "120")
            ),
            "request_renew_seconds": float(
                os.getenv("GPU_FAULT_PROCESSOR_REQUEST_RENEW_SECONDS", "5")
            ),
            "request_max_execution_seconds": float(
                os.getenv(
                    "GPU_FAULT_PROCESSOR_REQUEST_MAX_EXECUTION_SECONDS",
                    "30",
                )
            ),
            "retryable_response_max_age_seconds": float(
                os.getenv(
                    "GPU_FAULT_PROCESSOR_RETRYABLE_RESPONSE_MAX_AGE_SECONDS",
                    "300",
                )
            ),
            "idle_backoff_max_seconds": float(
                os.getenv("GPU_FAULT_PROCESSOR_IDLE_BACKOFF_MAX_SECONDS", "2")
            ),
            "busy_backoff_max_seconds": float(
                os.getenv(
                    "GPU_FAULT_PROCESSOR_BUSY_BACKOFF_MAX_SECONDS",
                    "0.4",
                )
            ),
            "fault_idle_backoff_max_seconds": float(
                os.getenv(
                    "GPU_FAULT_PROCESSOR_FAULT_IDLE_BACKOFF_MAX_SECONDS",
                    "0.5",
                )
            ),
            "fault_busy_backoff_max_seconds": float(
                os.getenv(
                    "GPU_FAULT_PROCESSOR_FAULT_BUSY_BACKOFF_MAX_SECONDS",
                    "0.1",
                )
            ),
            "processor_notification_fallback_seconds": float(
                os.getenv(
                    "GPU_FAULT_PROCESSOR_NOTIFICATION_FALLBACK_SECONDS",
                    "5",
                )
            ),
            "processor_notification_shard_count": int(
                os.getenv("GPU_FAULT_PROCESSOR_NOTIFICATION_SHARDS", "8")
            ),
            "routine_starvation_seconds": float(
                os.getenv(
                    "GPU_FAULT_PROCESSOR_ROUTINE_STARVATION_SECONDS",
                    "30",
                )
            ),
        }

    @staticmethod
    def _pool_settings(
        worker_count: int,
        default_pool: int,
    ) -> dict:
        return {
            "fault_pressure_evidence_workers": int(
                os.getenv(
                    "GPU_FAULT_PROCESSOR_FAULT_PRESSURE_EVIDENCE_WORKERS",
                    "1",
                )
            ),
            "worker_count": worker_count,
            "fault_worker_count": int(
                os.getenv(
                    "GPU_FAULT_PROCESSOR_FAULT_WORKERS",
                    str(default_pool),
                )
            ),
            "observation_worker_count": int(
                os.getenv(
                    "GPU_FAULT_PROCESSOR_OBSERVATION_WORKERS",
                    str(default_pool),
                )
            ),
            "gpu_telemetry_worker_count": int(
                os.getenv(
                    "GPU_FAULT_PROCESSOR_GPU_TELEMETRY_WORKERS",
                    str(default_pool),
                )
            ),
            "host_telemetry_worker_count": int(
                os.getenv(
                    "GPU_FAULT_PROCESSOR_HOST_TELEMETRY_WORKERS",
                    str(default_pool),
                )
            ),
        }

    @staticmethod
    def _stale_settings() -> dict:
        names = {
            "gpu_inventory_stale_seconds": (
                "GPU_FAULT_PROCESSOR_GPU_INVENTORY_STALE_SECONDS",
                "180",
            ),
            "health_summary_stale_seconds": (
                "GPU_FAULT_PROCESSOR_HEALTH_SUMMARY_STALE_SECONDS",
                "420",
            ),
            "observation_stale_seconds": (
                "GPU_FAULT_PROCESSOR_OBSERVATION_STALE_SECONDS",
                "120",
            ),
            "training_progress_stale_seconds": (
                "GPU_FAULT_PROCESSOR_TRAINING_PROGRESS_STALE_SECONDS",
                "120",
            ),
        }
        return {
            key: float(os.getenv(env_name, default))
            for key, (env_name, default) in names.items()
        }

    @staticmethod
    def _spool_settings(default_pool: int) -> dict:
        return {
            "telemetry_spool_enabled": os.getenv("GPU_FAULT_TELEMETRY_SPOOL", "0")
            .strip()
            .lower()
            in {"1", "true", "yes"},
            "telemetry_spool_workers": int(
                os.getenv(
                    "GPU_FAULT_TELEMETRY_SPOOL_WORKERS",
                    str(default_pool * 2),
                )
            ),
            "telemetry_spool_lease_seconds": float(
                os.getenv("GPU_FAULT_TELEMETRY_SPOOL_LEASE_SECONDS", "60")
            ),
            "telemetry_spool_retry_backoff_seconds": float(
                os.getenv(
                    "GPU_FAULT_TELEMETRY_SPOOL_RETRY_BACKOFF_SECONDS",
                    "1",
                )
            ),
            "telemetry_spool_notification_fallback_seconds": float(
                os.getenv(
                    "GPU_FAULT_TELEMETRY_SPOOL_NOTIFICATION_FALLBACK_SECONDS",
                    "5",
                )
            ),
            "telemetry_spool_fault_pressure_workers": int(
                os.getenv(
                    "GPU_FAULT_TELEMETRY_SPOOL_FAULT_PRESSURE_WORKERS",
                    "1",
                )
            ),
            "telemetry_spool_fault_pressure_poll_seconds": float(
                os.getenv(
                    "GPU_FAULT_TELEMETRY_SPOOL_FAULT_PRESSURE_POLL_SECONDS",
                    "0.5",
                )
            ),
            "telemetry_spool_max_in_flight_bytes": int(
                os.getenv(
                    "GPU_FAULT_TELEMETRY_SPOOL_MAX_IN_FLIGHT_BYTES",
                    str(64 * 1024 * 1024),
                )
            ),
            "telemetry_spool_replay_batch_max_items": int(
                os.getenv(
                    "GPU_FAULT_TELEMETRY_SPOOL_REPLAY_BATCH_MAX_ITEMS",
                    "64",
                )
            ),
            "telemetry_spool_replay_batch_max_bytes": int(
                os.getenv(
                    "GPU_FAULT_TELEMETRY_SPOOL_REPLAY_BATCH_MAX_BYTES",
                    str(8 * 1024 * 1024),
                )
            ),
        }

    def _recycle(self, reason: str) -> None:
        if os.getenv("GPU_FAULT_PROCESSOR_EXIT_ON_DEADLINE", "false").lower() != "true":
            return

        def exit_process() -> None:
            LOGGER.critical(
                "processor requires process restart reason=%s exit_grace_seconds=%s",
                reason,
                self.exit_grace_seconds,
            )
            if self.exit_grace_seconds > 0:
                time.sleep(self.exit_grace_seconds)
            abandon = self.processor.abandon_in_flight()
            LOGGER.critical(
                "terminating unhealthy processor process "
                "in_flight=%s released=%s failed=%s",
                abandon["in_flight"],
                abandon["released"],
                abandon["failed"],
            )
            os._exit(70)

        Thread(
            target=exit_process,
            name="gpu-fault-processor-fatal-exit",
            daemon=True,
        ).start()
