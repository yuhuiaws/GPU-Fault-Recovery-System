from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import logging
import os
from threading import Event
import time
from typing import Callable

from gpu_fault.telemetry import CollectorKind
from gpu_fault.collector_requirements import (
    collector_silent_thresholds,
)


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PeriodicServiceConfig:
    training_interval: float
    spare_interval: float
    identity_interval: float
    cleanup_interval: float
    completed_retention: float
    cleanup_batch_size: int
    cleanup_budget_seconds: float
    batch_retention: float
    terminal_retention: float
    latest_retention: float
    finding_retention: float
    observation_max_age: float
    remote_retention: float
    remote_claim_deadline: float
    lane_retention: float
    deployment_retention: float
    archive_interval: float
    silence_interval: float
    silent_after: dict[CollectorKind, float]
    silent_alert_interval: float

    @classmethod
    def from_environment(cls) -> PeriodicServiceConfig:
        value = os.getenv
        return cls(
            training_interval=float(
                value("GPU_FAULT_TRAINING_HEALTH_SCAN_SECONDS", "15")
            ),
            spare_interval=float(value("GPU_FAULT_SPARE_HEALTH_SCAN_SECONDS", "30")),
            identity_interval=float(
                value("GPU_FAULT_HYPERPOD_IDENTITY_REFRESH_SECONDS", "20")
            ),
            cleanup_interval=float(
                value("GPU_FAULT_PROCESSOR_CLEANUP_INTERVAL_SECONDS", "60")
            ),
            completed_retention=float(
                value(
                    "GPU_FAULT_PROCESSOR_COMPLETED_RETENTION_SECONDS",
                    "600",
                )
            ),
            cleanup_batch_size=int(
                value("GPU_FAULT_PROCESSOR_CLEANUP_BATCH_SIZE", "1000")
            ),
            cleanup_budget_seconds=float(
                value("GPU_FAULT_PROCESSOR_CLEANUP_BUDGET_SECONDS", "20")
            ),
            batch_retention=float(
                value(
                    "GPU_FAULT_GPU_METRICS_BATCH_RETENTION_SECONDS",
                    "86400",
                )
            ),
            terminal_retention=float(
                value(
                    "GPU_FAULT_HOT_STATE_TERMINAL_RETENTION_SECONDS",
                    "2592000",
                )
            ),
            latest_retention=float(
                value(
                    "GPU_FAULT_HOT_STATE_LATEST_RETENTION_SECONDS",
                    "2592000",
                )
            ),
            finding_retention=float(
                value(
                    "GPU_FAULT_GPU_FINDING_HISTORY_RETENTION_SECONDS",
                    "2592000",
                )
            ),
            observation_max_age=float(
                value(
                    "GPU_FAULT_ATTEMPT_OBSERVATION_MAX_AGE_SECONDS",
                    "604800",
                )
            ),
            remote_retention=float(
                value(
                    "GPU_FAULT_REMOTE_COMMAND_RETENTION_SECONDS",
                    "86400",
                )
            ),
            remote_claim_deadline=float(
                value(
                    "GPU_FAULT_REMOTE_COMMAND_CLAIM_DEADLINE_SECONDS",
                    "900",
                )
            ),
            lane_retention=float(
                value(
                    "GPU_FAULT_PROCESSOR_LANE_RETENTION_SECONDS",
                    "86400",
                )
            ),
            deployment_retention=float(
                value(
                    "GPU_FAULT_FLEET_DEPLOYMENT_RETENTION_SECONDS",
                    "604800",
                )
            ),
            archive_interval=float(
                value(
                    "GPU_FAULT_CONTROL_RECORD_ARCHIVE_INTERVAL_SECONDS",
                    "3600",
                )
            ),
            silence_interval=float(
                value("GPU_FAULT_COLLECTOR_SILENCE_SCAN_SECONDS", "60")
            ),
            silent_after=collector_silent_thresholds(),
            silent_alert_interval=float(
                value("GPU_FAULT_COLLECTOR_SILENT_ALERT_SECONDS", "3600")
            ),
        )


class PeriodicServiceRunner:
    def __init__(
        self,
        *,
        context,
        processor,
        stop: Event,
        identity_registries: list,
        ingest_node_health_findings: Callable,
        notify_silent_collectors: Callable,
        config: PeriodicServiceConfig | None = None,
    ) -> None:
        self.context = context
        self.processor = processor
        self.stop = stop
        self.identity_registries = identity_registries
        self.ingest_node_health_findings = ingest_node_health_findings
        self.notify_silent_collectors = notify_silent_collectors
        self.config = config or PeriodicServiceConfig.from_environment()
        self.next_run: dict[str, float] = {}
        self.next_lease_attempt: dict[str, float] = {}

    def run(self) -> None:
        while not self.stop.wait(0.2):
            if not self._active():
                continue
            now = time.monotonic()
            self._run_training(now)
            self._run_spare(now)
            self._run_identity(now)
            self._run_cleanup(now)
            self._run_archive(now)
            self._run_silence(now)

    def _active(self) -> bool:
        return self.processor.is_healthy() and (
            self.processor.active_consumers or self.processor.is_leader()
        )

    def _due(self, key: str, now: float, interval: float) -> bool:
        if now < self.next_run.get(key, 0.0):
            return False
        if not self._owns_task_lease(key, interval):
            return False
        self.next_run[key] = now + interval
        return True

    def _owns_task_lease(self, key: str, interval: float) -> bool:
        if not self.processor.active_consumers:
            return self.processor.is_leader()
        observed = time.monotonic()
        if observed < self.next_lease_attempt.get(key, 0.0):
            return False
        self.next_lease_attempt[key] = observed + max(1.0, min(3.0, interval))
        lease = self.context.store.acquire_periodic_task_lease(
            key,
            self.processor.owner_id,
            now=datetime.now(timezone.utc),
            lease_duration=timedelta(seconds=max(5.0, interval * 2)),
        )
        return lease.owner_id == self.processor.owner_id

    def _run_training(self, now: float) -> None:
        if os.getenv(
            "GPU_FAULT_ENABLE_TRAINING_HEALTH_MONITOR", "false"
        ).lower() != "true" or not self._due(
            "training-health",
            now,
            self.config.training_interval,
        ):
            return
        try:
            result = self.context.training_health.scan_all()
            if result.findings:
                self.ingest_node_health_findings(
                    "training-health-monitor",
                    result.findings,
                )
        except Exception:
            LOGGER.exception("training health scan failed")

    def _run_spare(self, now: float) -> None:
        controller = self.context.spare_health_controller
        if controller is None or not self._due(
            "spare-health", now, self.config.spare_interval
        ):
            return
        try:
            controller.scan()
        except Exception:
            LOGGER.exception("spare health scan failed")

    def _run_identity(self, now: float) -> None:
        if not self.identity_registries or not self._due(
            "identity-refresh", now, self.config.identity_interval
        ):
            return
        try:
            for registry in self.identity_registries:
                registry.refresh()
        except Exception:
            LOGGER.exception("HyperPod identity refresh failed")

    def _run_cleanup(self, now: float) -> None:
        if not self._due("processor-cleanup", now, self.config.cleanup_interval):
            return
        deadline = time.monotonic() + self.config.cleanup_budget_seconds
        self._cleanup_local(deadline)
        if self.context.regional_mode:
            self._cleanup_regional(deadline)

    def _cleanup_local(self, deadline: float) -> None:
        cfg = self.config
        now = datetime.now(timezone.utc)
        jobs = [
            (
                "processor cleanup",
                lambda: self.context.store.cleanup_completed_processor_requests(
                    older_than=now - timedelta(seconds=cfg.completed_retention),
                    limit=cfg.cleanup_batch_size,
                ),
            ),
            (
                "raw evidence cleanup",
                lambda: self.context.store.cleanup_expired_raw_evidence(
                    now=datetime.now(timezone.utc),
                    limit=cfg.cleanup_batch_size,
                ),
            ),
            (
                "hot state cleanup",
                lambda: self.context.store.cleanup_hot_state(
                    now=datetime.now(timezone.utc),
                    batch_retention=timedelta(seconds=cfg.batch_retention),
                    terminal_retention=timedelta(seconds=cfg.terminal_retention),
                    latest_retention=timedelta(seconds=cfg.latest_retention),
                    finding_history_retention=timedelta(seconds=cfg.finding_retention),
                    attempt_observation_max_age=(
                        timedelta(seconds=cfg.observation_max_age)
                        if cfg.observation_max_age > 0
                        else None
                    ),
                    limit=cfg.cleanup_batch_size,
                ),
            ),
            (
                "processor lane cleanup",
                lambda: self.context.store.cleanup_processor_lanes(
                    older_than=datetime.now(timezone.utc)
                    - timedelta(seconds=cfg.lane_retention),
                    limit=cfg.cleanup_batch_size,
                ),
            ),
        ]
        for label, job in jobs:
            deleted = self._drain_cleanup(label, job, deadline)
            if deleted:
                LOGGER.info("%s removed %s records", label, deleted)

    def _cleanup_regional(self, deadline: float) -> None:
        cfg = self.config
        jobs = [
            (
                "remote command cleanup",
                lambda: self.context.store.cleanup_terminal_remote_commands(
                    older_than=datetime.now(timezone.utc)
                    - timedelta(seconds=cfg.remote_retention),
                    limit=cfg.cleanup_batch_size,
                ),
            ),
            (
                "fleet deployment cleanup",
                lambda: self.context.store.cleanup_terminal_fleet_deployments(
                    older_than=datetime.now(timezone.utc)
                    - timedelta(seconds=cfg.deployment_retention),
                    limit=cfg.cleanup_batch_size,
                ),
            ),
        ]
        if cfg.remote_claim_deadline > 0:
            jobs.insert(
                1,
                (
                    "unclaimed remote command expiry",
                    lambda: self.context.store.expire_unclaimed_remote_commands(
                        older_than=datetime.now(timezone.utc)
                        - timedelta(seconds=cfg.remote_claim_deadline),
                        limit=cfg.cleanup_batch_size,
                    ),
                ),
            )
        for label, job in jobs:
            deleted = self._drain_cleanup(label, job, deadline)
            if deleted:
                log = (
                    LOGGER.error
                    if label == "unclaimed remote command expiry"
                    else LOGGER.info
                )
                log("%s affected %s records", label, deleted)

    def _drain_cleanup(
        self,
        label: str,
        run: Callable,
        deadline: float,
    ) -> int:
        total = 0
        while True:
            try:
                deleted = run()
            except Exception:
                LOGGER.exception("%s failed", label)
                return total
            if isinstance(deleted, dict):
                count = sum(deleted.values())
                saturated = any(
                    value >= self.config.cleanup_batch_size
                    for value in deleted.values()
                )
            else:
                count = int(deleted)
                saturated = count >= self.config.cleanup_batch_size
            total += count
            if not saturated or self.stop.is_set() or time.monotonic() >= deadline:
                if saturated and time.monotonic() >= deadline:
                    LOGGER.warning(
                        "%s stopped at cleanup budget after %s rows",
                        label,
                        total,
                    )
                return total

    def _run_archive(self, now: float) -> None:
        archiver = self.context.control_record_archiver
        if archiver is None or not self._due(
            "control-record-archive",
            now,
            self.config.archive_interval,
        ):
            return
        try:
            archived = archiver.run_once()
            if archived:
                LOGGER.info(
                    "archived %s terminal incident audit bundles",
                    len(archived),
                )
        except Exception:
            LOGGER.exception("control record archive failed")

    def _run_silence(self, now: float) -> None:
        if not self.context.regional_mode or not self._due(
            "collector-silence",
            now,
            self.config.silence_interval,
        ):
            return
        try:
            self.notify_silent_collectors(
                self.context,
                observed_at=datetime.now(timezone.utc),
                silent_after_seconds=self.config.silent_after,
                alert_interval_seconds=(self.config.silent_alert_interval),
            )
        except Exception:
            LOGGER.exception("collector silence scan failed")
