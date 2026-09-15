from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from threading import Event
from typing import Any, Callable

from gpu_fault.collector_requirements import (
    collector_silent_thresholds,
)
from gpu_fault.env_validation import training_health_monitor_enabled
from gpu_fault.telemetry import CollectorKind

LOGGER = logging.getLogger(__name__)

# What one cleanup call reports: a row count, or per-table counts.
CleanupResult = int | dict[str, int]
CleanupJob = tuple[str, Callable[[], CleanupResult]]


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
    # F-D5: how often LEASED processor rows whose lease lapsed go back to
    # PENDING outside the claim window.
    lease_reclaim_interval: float = 30.0
    # F-D10 (P1-75F): how often the processor counter table is compared with
    # the queue rows it summarises. A full-table count, so never per scrape.
    counter_drift_interval: float = 60.0
    # Control-plane review 2026-09-08, F-8: incidents one archive round may
    # take, and retention for the kinds that had no cleanup path at all --
    # inactive markers, terminal notifications (with their delivery, result
    # and dedup link), completion records (decision + event) and registry
    # heartbeat rows of processes that are gone. A non-positive retention
    # switches that sweep off.
    archive_batch_size: int = 200
    marker_retention: float = 2592000.0
    notification_retention: float = 2592000.0
    completion_record_retention: float = 2592000.0
    registry_member_retention: float = 86400.0
    # Control-plane review 2026-09-08, D-9: a LEASED remote command whose
    # lease lapsed this long ago while its workflow moved to another
    # fencing_token is failed with status_source="stale-fence".
    remote_stale_fence_grace: float = 600.0

    def __post_init__(self) -> None:
        # The env reference generator infers a variable's kind from the
        # coercion at its read site, so the reads stay inline in
        # ``from_environment`` and the positivity rule lives on the field.
        if self.lease_reclaim_interval <= 0:
            raise ValueError(
                "GPU_FAULT_PROCESSOR_LEASE_RECLAIM_SECONDS must be positive"
            )
        if self.counter_drift_interval <= 0:
            raise ValueError(
                "GPU_FAULT_PROCESSOR_COUNTER_DRIFT_SCAN_SECONDS must be positive"
            )
        if self.archive_batch_size <= 0:
            raise ValueError(
                "GPU_FAULT_CONTROL_RECORD_ARCHIVE_BATCH_SIZE must be positive"
            )

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
                    # 24 h kept the lane table at millions of rows that every
                    # claim window LEFT JOINed; the row is only needed while
                    # a stale fencing token could still be presented (F-F2).
                    "3600",
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
                    # 25 incidents an hour could never drain a backlog; 200
                    # every 10 min can (F-8).
                    "600",
                )
            ),
            silence_interval=float(
                value("GPU_FAULT_COLLECTOR_SILENCE_SCAN_SECONDS", "60")
            ),
            silent_after=collector_silent_thresholds(),
            silent_alert_interval=float(
                value("GPU_FAULT_COLLECTOR_SILENT_ALERT_SECONDS", "3600")
            ),
            lease_reclaim_interval=float(
                os.getenv("GPU_FAULT_PROCESSOR_LEASE_RECLAIM_SECONDS", "30")
            ),
            counter_drift_interval=float(
                os.getenv("GPU_FAULT_PROCESSOR_COUNTER_DRIFT_SCAN_SECONDS", "60")
            ),
            archive_batch_size=int(
                os.getenv("GPU_FAULT_CONTROL_RECORD_ARCHIVE_BATCH_SIZE", "200")
            ),
            marker_retention=float(
                os.getenv("GPU_FAULT_MARKER_RETENTION_SECONDS", "2592000")
            ),
            notification_retention=float(
                os.getenv("GPU_FAULT_NOTIFICATION_RETENTION_SECONDS", "2592000")
            ),
            completion_record_retention=float(
                os.getenv("GPU_FAULT_COMPLETION_RECORD_RETENTION_SECONDS", "2592000")
            ),
            registry_member_retention=float(
                os.getenv("GPU_FAULT_REGISTRY_MEMBER_RETENTION_SECONDS", "86400")
            ),
            remote_stale_fence_grace=float(
                os.getenv("GPU_FAULT_REMOTE_COMMAND_STALE_FENCE_GRACE_SECONDS", "600")
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
        # F-F1: failures are counted per job and never end the runner thread.
        self.periodic_lease_errors_total = 0
        self.periodic_job_errors_total: dict[str, int] = {}
        # F-F2: each cleanup job is accounted for by name, so a job squeezed to
        # one batch a minute is visible instead of silent.
        self.cleanup_rows_total: dict[str, int] = {}
        self.cleanup_budget_exhausted_total: dict[str, int] = {}
        self.cleanup_job_errors_total: dict[str, int] = {}
        self._cleanup_rotation = 0
        # F-D5: LEASED rows whose lease lapsed and were handed back to PENDING.
        self.processor_expired_leases_reclaimed_total: int = 0
        # F-D10 (P1-75F): the last measured gap between queue rows and the
        # counter table, and the clusters whose counters disagree. Gauges
        # refreshed by the drift job; a scrape only reads them.
        self.processor_counter_drift_abs: int = 0
        self.processor_counter_mismatched_clusters: int = 0
        # ARCH-E E3: the runner's own heartbeat, and each job's last run. The
        # counters above are written by the loop, so a dead thread leaves
        # every one of them at a healthy-looking value.
        self.last_cycle_timestamp_seconds = 0.0
        self.job_last_run_timestamp_seconds: dict[str, float] = {}
        # Newest error behind each error counter, as a Unix time: a scrape of a
        # multi-process Pod samples one process, so an alert reads max() of
        # these rather than increase() of the counters (ARCH-E E4).
        self.lease_error_last_seen_timestamp_seconds = 0.0
        self.job_error_last_seen_timestamp_seconds: dict[str, float] = {}

    def run(self) -> None:
        while not self.stop.wait(0.2):
            self.last_cycle_timestamp_seconds = time.time()
            if not self._active():
                continue
            self.run_all_due(time.monotonic())

    _JOBS = (
        "training",
        "spare",
        "identity",
        "cleanup",
        "archive",
        "silence",
        "lease_reclaim",
        "counter_drift",
    )

    def run_all_due(self, now: float) -> None:
        """One tick: every job gets its turn even if another one raised.

        The loop body used to call every job in ``_JOBS`` bare; the one
        statement they all share without a guard -- the task-lease write in
        ``_due`` -- ended the thread on its first store error, and with it
        every periodic service for the life of the process (F-F1).

        ``now`` is refreshed after a job actually ran: a 20 s cleanup used to
        leave the jobs behind it judging their schedule by a clock taken
        before it started (F-F2).
        """

        for name in self._JOBS:
            try:
                ran = getattr(self, f"_run_{name}")(now)
            except Exception:
                # The job body raised inside ``_run_scheduled`` (which logged
                # the traceback and rescheduled it) or ``_run_<name>`` itself
                # did. Either way it is a failure of this job: counted and
                # stamped here, and ``job_last_run`` is deliberately *not*
                # refreshed -- that gauge is what the stall alert reads, and a
                # job failing every round must look stalled, not freshly run
                # (control-plane review 2026-09-08, F-1).
                self.periodic_job_errors_total[name] = (
                    self.periodic_job_errors_total.get(name, 0) + 1
                )
                self.job_error_last_seen_timestamp_seconds[name] = time.time()
                LOGGER.warning("periodic service %s failed; continuing", name)
                now = time.monotonic()
                continue
            if ran:
                now = time.monotonic()
                self.job_last_run_timestamp_seconds[name] = time.time()

    def metrics_snapshot(self) -> dict[str, Any]:
        return {
            "periodic_lease_errors_total": self.periodic_lease_errors_total,
            "periodic_job_errors_total": dict(self.periodic_job_errors_total),
            "cleanup_rows_total": dict(self.cleanup_rows_total),
            "cleanup_budget_exhausted_total": dict(self.cleanup_budget_exhausted_total),
            "cleanup_job_errors_total": dict(self.cleanup_job_errors_total),
            "processor_expired_leases_reclaimed_total": (
                self.processor_expired_leases_reclaimed_total
            ),
            "processor_counter_drift_abs": self.processor_counter_drift_abs,
            "processor_counter_mismatched_clusters": (
                self.processor_counter_mismatched_clusters
            ),
            "last_cycle_timestamp_seconds": self.last_cycle_timestamp_seconds,
            "job_last_run_timestamp_seconds": dict(self.job_last_run_timestamp_seconds),
            "lease_error_last_seen_timestamp_seconds": (
                self.lease_error_last_seen_timestamp_seconds
            ),
            "job_error_last_seen_timestamp_seconds": dict(
                self.job_error_last_seen_timestamp_seconds
            ),
        }

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

    def _run_scheduled(
        self,
        key: str,
        now: float,
        interval: float,
        body: Callable[[], None],
        failure_message: str,
    ) -> bool:
        """Run ``body`` if ``key`` is due; schedule the next run from its end.

        ``_due`` advanced ``next_run`` before the job from a ``now`` taken at
        the start of the tick, so a job that ran for 20 s was due again 20 s
        early -- the interval is the gap between runs, not between starts
        (F-F2). Returns whether the job ran.

        A body that raises is logged with its traceback, rescheduled a full
        interval out, and the exception is re-raised so ``run_all_due`` counts
        it against the job. Swallowing it here made every real failure mode
        -- store outages inside the reclaim, watchdog and drift jobs --
        invisible to ``periodic_job_errors_total`` and its alert, while the
        ``ran=True`` return refreshed ``job_last_run`` as if the job had
        succeeded (control-plane review 2026-09-08, F-1).
        """

        if not self._due(key, now, interval):
            return False
        try:
            body()
        except Exception:
            LOGGER.exception(failure_message)
            raise
        finally:
            self.next_run[key] = time.monotonic() + interval
        return True

    def _owns_task_lease(self, key: str, interval: float) -> bool:
        if not self.processor.active_consumers:
            return self.processor.is_leader()
        observed = time.monotonic()
        if observed < self.next_lease_attempt.get(key, 0.0):
            return False
        self.next_lease_attempt[key] = observed + max(1.0, min(3.0, interval))
        now = datetime.now(timezone.utc)
        lease_duration = timedelta(seconds=max(5.0, interval * 2))
        try:
            lease = self.context.store.acquire_periodic_task_lease(
                key,
                self.processor.owner_id,
                now=now,
                lease_duration=lease_duration,
            )
        except Exception:
            # A store hiccup means "not this tick", never "never again".
            self.periodic_lease_errors_total += 1
            self.lease_error_last_seen_timestamp_seconds = time.time()
            LOGGER.exception("periodic task lease %s could not be taken", key)
            return False
        if lease.owner_id == self.processor.owner_id:
            return True
        # Not ours: the store told us when it stops being theirs. Asking again
        # every 3 s made 24 processes take six advisory locks each, ~48 times a
        # second, to hear "no" (F-F2). Nothing can be won before expiry anyway.
        expires_at = getattr(lease, "lease_expires_at", None)
        if expires_at is not None:
            remaining = (expires_at - now).total_seconds()
            self.next_lease_attempt[key] = observed + max(
                1.0, min(remaining, lease_duration.total_seconds())
            )
        return False

    def _run_training(self, now: float) -> bool:
        if not training_health_monitor_enabled():
            return False

        def scan() -> None:
            training_health = self.context.training_health
            result = training_health.scan_all()
            if result.findings:
                self.ingest_node_health_findings(
                    "training-health-monitor",
                    result.findings,
                )
                training_health.mark_notified(result.findings)

        return self._run_scheduled(
            "training-health",
            now,
            self.config.training_interval,
            scan,
            "training health scan failed",
        )

    def _run_spare(self, now: float) -> bool:
        controller = self.context.spare_health_controller
        if controller is None:
            return False
        return self._run_scheduled(
            "spare-health",
            now,
            self.config.spare_interval,
            controller.scan,
            "spare health scan failed",
        )

    def _run_identity(self, now: float) -> bool:
        if not self.identity_registries:
            return False

        def refresh() -> None:
            for registry in self.identity_registries:
                registry.refresh()

        return self._run_scheduled(
            "identity-refresh",
            now,
            self.config.identity_interval,
            refresh,
            "HyperPod identity refresh failed",
        )

    def _run_cleanup(self, now: float) -> bool:
        return self._run_scheduled(
            "processor-cleanup",
            now,
            self.config.cleanup_interval,
            self._cleanup_round,
            "processor cleanup failed",
        )

    def _cleanup_round(self) -> None:
        """Run every cleanup job once, each with its own slice of the budget.

        One deadline used to be handed to all seven jobs in a fixed order, so
        the first (COMPLETED queue rows, the largest table) could spend the
        whole 20 s and processor-lane cleanup, last in line, got one batch a
        minute with nothing logged (F-F2). Now the remaining budget is split
        evenly among the jobs still to run -- a job that finishes early hands
        its slack on -- and the starting job rotates every round so no table
        is always the one working from leftovers. ``_drain_cleanup`` always
        runs at least one batch, so a job is never skipped outright.
        """

        jobs = self._cleanup_jobs()
        if not jobs:
            return
        start = self._cleanup_rotation % len(jobs)
        self._cleanup_rotation += 1
        ordered = jobs[start:] + jobs[:start]
        deadline = time.monotonic() + self.config.cleanup_budget_seconds
        for index, (job, run) in enumerate(ordered):
            remaining = max(0.0, deadline - time.monotonic())
            share = remaining / (len(ordered) - index)
            affected = self._drain_cleanup(job, run, time.monotonic() + share)
            if affected:
                log = (
                    LOGGER.error if job == "unclaimed_remote_commands" else LOGGER.info
                )
                log("%s cleanup affected %s records", job, affected)

    def _cleanup_jobs(self) -> list[CleanupJob]:
        jobs = self._local_cleanup_jobs()
        if self.context.regional_mode:
            jobs.extend(self._regional_cleanup_jobs())
        return jobs

    def _local_cleanup_jobs(self) -> list[CleanupJob]:
        cfg = self.config
        return [
            (
                "completed_requests",
                lambda: self.context.store.cleanup_completed_processor_requests(
                    older_than=datetime.now(timezone.utc)
                    - timedelta(seconds=cfg.completed_retention),
                    limit=cfg.cleanup_batch_size,
                ),
            ),
            (
                "raw_evidence",
                lambda: self.context.store.cleanup_expired_raw_evidence(
                    now=datetime.now(timezone.utc),
                    limit=cfg.cleanup_batch_size,
                ),
            ),
            (
                "hot_state",
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
                "processor_lanes",
                lambda: self.context.store.cleanup_processor_lanes(
                    older_than=datetime.now(timezone.utc)
                    - timedelta(seconds=cfg.lane_retention),
                    limit=cfg.cleanup_batch_size,
                ),
            ),
        ] + self._unbounded_kind_cleanup_jobs()

    def _unbounded_kind_cleanup_jobs(self) -> list[CleanupJob]:
        """Retention for the kinds that had none (F-8 / G-9).

        Each sweep keeps anything an existing incident still references --
        the archiver bundles those with the incident (F-I1) -- and takes only
        rows in a terminal state: inactive markers, notifications whose
        delivery is SENT/DEAD (never PENDING/RETRY/LEASED) and completion
        decisions whose attempt event aged out.
        """

        cfg = self.config
        jobs: list[CleanupJob] = []
        if cfg.marker_retention > 0:
            jobs.append(
                (
                    "inactive_markers",
                    lambda: self.context.store.cleanup_inactive_markers(
                        older_than=datetime.now(timezone.utc)
                        - timedelta(seconds=cfg.marker_retention),
                        limit=cfg.cleanup_batch_size,
                    ),
                )
            )
        if cfg.notification_retention > 0:
            jobs.append(
                (
                    "notifications",
                    lambda: self.context.store.cleanup_terminal_notifications(
                        older_than=datetime.now(timezone.utc)
                        - timedelta(seconds=cfg.notification_retention),
                        limit=cfg.cleanup_batch_size,
                    ),
                )
            )
        if cfg.completion_record_retention > 0:
            jobs.append(
                (
                    "completion_records",
                    lambda: self.context.store.cleanup_completion_records(
                        older_than=datetime.now(timezone.utc)
                        - timedelta(seconds=cfg.completion_record_retention),
                        limit=cfg.cleanup_batch_size,
                    ),
                )
            )
        return jobs

    def _regional_cleanup_jobs(self) -> list[CleanupJob]:
        cfg = self.config
        jobs: list[CleanupJob] = [
            (
                "remote_commands",
                lambda: self.context.store.cleanup_terminal_remote_commands(
                    older_than=datetime.now(timezone.utc)
                    - timedelta(seconds=cfg.remote_retention),
                    limit=cfg.cleanup_batch_size,
                ),
            ),
            (
                "fleet_deployments",
                lambda: self.context.store.cleanup_terminal_fleet_deployments(
                    older_than=datetime.now(timezone.utc)
                    - timedelta(seconds=cfg.deployment_retention),
                    limit=cfg.cleanup_batch_size,
                ),
            ),
        ]
        if cfg.registry_member_retention > 0:
            # One heartbeat row per process (POD_UID:pid); every rolling
            # update left the old ones behind for good, and the readiness
            # convergence check listed the whole kind (F-5 / F-8).
            jobs.append(
                (
                    "registry_members",
                    lambda: self.context.store.cleanup_stale_regional_registry_members(
                        older_than=datetime.now(timezone.utc)
                        - timedelta(seconds=cfg.registry_member_retention),
                        limit=cfg.cleanup_batch_size,
                    ),
                )
            )
        if cfg.remote_stale_fence_grace > 0:
            jobs.append(
                (
                    "stale_fence_remote_commands",
                    lambda: self.context.store.expire_stale_fenced_remote_commands(
                        lease_expired_before=datetime.now(timezone.utc)
                        - timedelta(seconds=cfg.remote_stale_fence_grace),
                        limit=cfg.cleanup_batch_size,
                    ),
                )
            )
        if cfg.remote_claim_deadline > 0:
            jobs.insert(
                1,
                (
                    "unclaimed_remote_commands",
                    lambda: self.context.store.expire_unclaimed_remote_commands(
                        older_than=datetime.now(timezone.utc)
                        - timedelta(seconds=cfg.remote_claim_deadline),
                        limit=cfg.cleanup_batch_size,
                    ),
                ),
            )
        return jobs

    def _drain_cleanup(
        self,
        job: str,
        run: Callable[[], CleanupResult],
        deadline: float,
    ) -> int:
        """Repeat ``run`` while it returns full batches and ``deadline`` holds.

        The first call is unconditional, so every job gets at least one batch
        per round however little budget reached it. Rows, errors and budget
        stops are counted by job (F-F2).
        """

        total = 0
        while True:
            try:
                deleted = run()
            except Exception:
                self.cleanup_job_errors_total[job] = (
                    self.cleanup_job_errors_total.get(job, 0) + 1
                )
                LOGGER.exception("%s cleanup failed", job)
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
            self.cleanup_rows_total[job] = self.cleanup_rows_total.get(job, 0) + count
            if not saturated or self.stop.is_set() or time.monotonic() >= deadline:
                if saturated and time.monotonic() >= deadline:
                    self.cleanup_budget_exhausted_total[job] = (
                        self.cleanup_budget_exhausted_total.get(job, 0) + 1
                    )
                    LOGGER.warning(
                        "%s cleanup stopped at its budget after %s rows",
                        job,
                        total,
                    )
                return total

    def _run_archive(self, now: float) -> bool:
        archiver = self.context.control_record_archiver
        if archiver is None:
            return False

        def archive() -> None:
            archived = archiver.run_once(limit=self.config.archive_batch_size)
            if archived:
                LOGGER.info(
                    "archived %s terminal incident audit bundles",
                    len(archived),
                )

        return self._run_scheduled(
            "control-record-archive",
            now,
            self.config.archive_interval,
            archive,
            "control record archive failed",
        )

    def _run_silence(self, now: float) -> bool:
        if not self.context.regional_mode:
            return False

        def notify() -> None:
            self.notify_silent_collectors(
                self.context,
                observed_at=datetime.now(timezone.utc),
                silent_after_seconds=self.config.silent_after,
                alert_interval_seconds=(self.config.silent_alert_interval),
            )

        return self._run_scheduled(
            "collector-silence",
            now,
            self.config.silence_interval,
            notify,
            "collector silence scan failed",
        )

    # One reclaim batch; large enough that a whole crashed pool's leases go
    # back in a round or two, small enough to hold no lock for long (F-D5).
    LEASE_RECLAIM_BATCH = 256

    def _run_lease_reclaim(self, now: float) -> bool:
        """Hand LEASED processor rows whose lease lapsed back to PENDING (F-D5).

        The claim window reclaims such rows too, but only inside its retry
        horizon and only on a priority somebody is claiming; a row behind the
        horizon stayed LEASED for good, with nothing counting it.
        """

        def reclaim() -> None:
            reclaimed = self.context.store.reclaim_expired_processor_leases(
                now=datetime.now(timezone.utc),
                limit=self.LEASE_RECLAIM_BATCH,
            )
            if reclaimed:
                self.processor_expired_leases_reclaimed_total += reclaimed
                LOGGER.warning(
                    "reclaimed %s processor requests whose lease had expired",
                    reclaimed,
                )

        return self._run_scheduled(
            "processor-lease-reclaim",
            now,
            self.config.lease_reclaim_interval,
            reclaim,
            "processor lease reclaim failed",
        )

    def _run_counter_drift(self, now: float) -> bool:
        """Compare the processor counter table with the rows it summarises
        (F-D10, P1-75F).

        The counter table decides how many rows a claim window may take per
        cluster; a counter that drifted low starves that cluster silently.
        The comparison is a full count of the queue table, so it runs here on
        its own interval and ``/metrics`` only reads the last answer.
        """

        def measure() -> None:
            status = self.context.store.processor_queue_count_status()
            drift = abs(int(status["expected_total"]) - int(status["counter_total"]))
            mismatched = int(status["mismatched_clusters"])
            self.processor_counter_drift_abs = drift
            self.processor_counter_mismatched_clusters = mismatched
            if drift or mismatched:
                LOGGER.warning(
                    "processor counter drift: rows=%s counters=%s "
                    "mismatched_clusters=%s",
                    status["expected_total"],
                    status["counter_total"],
                    mismatched,
                )

        return self._run_scheduled(
            "processor-counter-drift",
            now,
            self.config.counter_drift_interval,
            measure,
            "processor counter drift scan failed",
        )
