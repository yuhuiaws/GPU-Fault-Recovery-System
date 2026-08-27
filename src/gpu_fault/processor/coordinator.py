from __future__ import annotations

import base64
import json
import logging
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from threading import Event, RLock, Thread
from urllib import error as urllib_error
from urllib import request as urllib_request

from gpu_fault.channel_registry import (
    BATCHABLE_CHANNEL_PATHS,
    GPU_INVENTORY_PATH,
    GPU_METRICS_PATH,
    HOST_TELEMETRY_PATH,
    NODE_LOG_PATH,
    TELEMETRY_SPOOL_PATH_SCHEDULE,
    TRAINING_PROGRESS_PATH,
    WORKLOAD_OBSERVATIONS_PATH,
    ChannelPool,
    channel_for_path,
    paths_for_pool,
)
from gpu_fault.transport.http_client import urlopen
from gpu_fault.processor.metrics import ProcessorMetricsMixin
from gpu_fault.processor.batching import (
    GPU_INVENTORY_BATCH_SIZE,
    GPU_METRICS_BATCH_SIZE,
    HOST_TELEMETRY_BATCH_SIZE,
    telemetry_batch_size,
)
from gpu_fault.processor.lane_runtime import ProcessorLaneRuntimeMixin
from gpu_fault.processor.replay_completion import (
    finalize_replay_response,
)
from gpu_fault.processor.telemetry_spool import TelemetrySpoolCoordinatorMixin
from gpu_fault.processor.models import (
    ProcessorLeadership,
    ProcessorRequest,
    processor_partition_id,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class _InFlightRequest:
    item: ProcessorRequest
    started: float
    deadline: float
    phase: str
    phase_started: float


class ProcessorCoordinator(
    ProcessorLaneRuntimeMixin,
    ProcessorMetricsMixin,
    TelemetrySpoolCoordinatorMixin,
):
    _OBSERVATION_PATHS = paths_for_pool(ChannelPool.OBSERVATION)
    _GPU_INVENTORY_PATH = GPU_INVENTORY_PATH
    _GPU_METRICS_PATH = GPU_METRICS_PATH
    _HOST_TELEMETRY_PATH = HOST_TELEMETRY_PATH
    _NODE_LOG_PATH = NODE_LOG_PATH
    _OBSERVATION_BATCH_SIZE = 16
    # One claim, one business transaction and one fenced completion may
    # carry up to 64 routine samples, bounded independently by the 8 MiB
    # replay envelope.
    _TELEMETRY_BATCH_MAX = 64
    _TELEMETRY_BATCH_ENVELOPE_BYTES = 8192
    _TELEMETRY_SPOOL_PATH_SCHEDULE = TELEMETRY_SPOOL_PATH_SCHEDULE

    def __init__(
        self,
        store,
        *,
        owner_id: str,
        internal_token: str,
        execution_token: str | None = None,
        local_url: str = "http://127.0.0.1:8080",
        lease_seconds: int = 15,
        renew_seconds: float = 3,
        request_lease_seconds: int = 120,
        request_renew_seconds: float = 5,
        request_max_execution_seconds: float = 30,
        retryable_response_max_age_seconds: float = 300,
        retry_backoff_seconds: float = 1,
        retry_backoff_max_seconds: float = 30,
        poll_seconds: float = 0.1,
        idle_backoff_max_seconds: float = 2.0,
        busy_backoff_max_seconds: float = 0.4,
        fault_idle_backoff_max_seconds: float = 0.5,
        fault_busy_backoff_max_seconds: float = 0.1,
        processor_notification_fallback_seconds: float = 5.0,
        processor_notification_shard_count: int = 8,
        routine_starvation_seconds: float = 30.0,
        fault_pressure_evidence_workers: int = 1,
        worker_count: int = 4,
        fault_worker_count: int | None = None,
        observation_worker_count: int | None = None,
        gpu_telemetry_worker_count: int | None = None,
        host_telemetry_worker_count: int | None = None,
        gpu_inventory_stale_seconds: float = 180,
        health_summary_stale_seconds: float = 420,
        observation_stale_seconds: float = 120,
        training_progress_stale_seconds: float = 120,
        active_consumers: bool = False,
        telemetry_spool_enabled: bool = False,
        telemetry_spool_workers: int = 4,
        telemetry_spool_lease_seconds: float = 60,
        telemetry_spool_retry_backoff_seconds: float = 1.0,
        telemetry_spool_notification_fallback_seconds: float = 5.0,
        telemetry_spool_fault_pressure_workers: int = 1,
        telemetry_spool_fault_pressure_poll_seconds: float = 0.5,
        telemetry_spool_max_in_flight_bytes: int = (64 * 1024 * 1024),
        telemetry_spool_replay_batch_max_items: int = 64,
        telemetry_spool_replay_batch_max_bytes: int = (8 * 1024 * 1024),
        on_unhealthy: Callable[[str], None] | None = None,
    ) -> None:
        if lease_seconds < 5:
            raise ValueError("processor leader lease must be at least 5s")
        if renew_seconds <= 0 or renew_seconds >= lease_seconds:
            raise ValueError("processor renew interval must be below its lease")
        if worker_count <= 0:
            raise ValueError("processor worker count must be positive")
        if routine_starvation_seconds <= 0:
            raise ValueError("processor routine starvation bound must be positive")
        self.routine_starvation_seconds = routine_starvation_seconds
        if request_renew_seconds <= 0 or request_renew_seconds >= request_lease_seconds:
            raise ValueError("processor request renew interval must be below its lease")
        if request_max_execution_seconds <= 0:
            raise ValueError(
                "processor request maximum execution time must be positive"
            )
        stale_limits = {
            "GPU inventory": gpu_inventory_stale_seconds,
            "health summary": health_summary_stale_seconds,
            "workload observation": observation_stale_seconds,
            "training progress": training_progress_stale_seconds,
            "retryable response": retryable_response_max_age_seconds,
        }
        invalid_stale_limits = [
            name for name, value in stale_limits.items() if value <= 0
        ]
        if invalid_stale_limits:
            raise ValueError(
                "processor stale limits must be positive: "
                + ", ".join(invalid_stale_limits)
            )
        self.store = store
        self.owner_id = owner_id
        self.internal_token = internal_token
        self.execution_token = execution_token
        self.local_url = local_url.rstrip("/")
        self.lease_seconds = lease_seconds
        self.renew_seconds = renew_seconds
        self.request_lease_seconds = request_lease_seconds
        self.request_renew_seconds = request_renew_seconds
        self.request_max_execution_seconds = request_max_execution_seconds
        self._retry_max_age = retryable_response_max_age_seconds
        self._configure_retry_backoff(
            retry_backoff_seconds,
            retry_backoff_max_seconds,
        )
        self.gpu_inventory_stale_seconds = gpu_inventory_stale_seconds
        self.health_summary_stale_seconds = health_summary_stale_seconds
        self.observation_stale_seconds = observation_stale_seconds
        self.training_progress_stale_seconds = training_progress_stale_seconds
        self.poll_seconds = poll_seconds
        # Each claim is its own transaction. Six worker replicas
        # polling seven streams every 100ms cost about 500 commits a
        # second with an empty queue, which measured out at more than
        # half of what the Aurora writer can commit at 16 ACU: the
        # idle control plane was consuming the capacity the burst
        # needed. Streams that come back empty are therefore skipped
        # for a doubling interval, capped here. The fault stream is
        # exempt (see _claim_active_by_pool) so escalation latency is
        # unchanged.
        self.idle_backoff_max_seconds = max(poll_seconds, idle_backoff_max_seconds)
        # "Empty claim" and "idle queue" are not the same thing. A claim
        # comes back empty whenever every eligible lane is already
        # leased, which is exactly what a backlog on few lanes looks
        # like - and backing off to two seconds there is the difference
        # between draining a burst tail at 7 rows/s and draining it at
        # the rate lanes free up. While this replica still has requests
        # in flight it is demonstrably not idle: those requests release
        # their lanes within seconds, so the backoff is capped until the
        # pools go quiet. A replica whose own pools are empty because a
        # sibling holds every lane asks the store instead
        # (_backlog_is_lane_blocked).
        self.busy_backoff_max_seconds = max(
            poll_seconds,
            min(busy_backoff_max_seconds, idle_backoff_max_seconds),
        )
        self.fault_idle_backoff_max_seconds = max(
            poll_seconds, fault_idle_backoff_max_seconds
        )
        self.fault_busy_backoff_max_seconds = max(
            poll_seconds,
            min(
                fault_busy_backoff_max_seconds,
                fault_idle_backoff_max_seconds,
            ),
        )
        if processor_notification_fallback_seconds <= 0:
            raise ValueError("processor notification fallback must be positive")
        self.processor_notification_fallback_seconds = max(
            poll_seconds, processor_notification_fallback_seconds
        )
        if processor_notification_shard_count <= 0:
            raise ValueError("processor notification shard count must be positive")
        self.processor_notification_shard_count = processor_notification_shard_count
        self._stream_idle_until: dict[str, float] = {}
        self._stream_idle_interval: dict[str, float] = {}
        self._initialize_claim_metrics()
        explicit_pools = any(
            value is not None
            for value in (
                fault_worker_count,
                observation_worker_count,
                gpu_telemetry_worker_count,
                host_telemetry_worker_count,
            )
        )
        self.worker_counts = (
            {
                "fault": fault_worker_count or 0,
                "observation": observation_worker_count or 0,
                "gpu": gpu_telemetry_worker_count or 0,
                "host": host_telemetry_worker_count or 0,
            }
            if explicit_pools
            else {
                "fault": worker_count,
                "observation": 0,
                "gpu": 0,
                "host": 0,
            }
        )
        if self.worker_counts["fault"] <= 0 or any(
            value < 0 for value in self.worker_counts.values()
        ):
            raise ValueError(
                "processor fault workers must be positive and "
                "other pool workers cannot be negative"
            )
        evidence_pool_sizes = [
            self.worker_counts[name]
            for name in ("gpu", "host")
            if self.worker_counts[name] > 0
        ]
        if fault_pressure_evidence_workers <= 0 or (
            evidence_pool_sizes
            and fault_pressure_evidence_workers > min(evidence_pool_sizes)
        ):
            raise ValueError(
                "processor fault-pressure evidence workers must be "
                "positive and not exceed a dedicated evidence pool"
            )
        self.fault_pressure_evidence_workers = fault_pressure_evidence_workers
        self.worker_count = sum(self.worker_counts.values())
        self.active_consumers = active_consumers
        self.telemetry_spool_enabled = telemetry_spool_enabled
        if telemetry_spool_enabled and telemetry_spool_workers <= 0:
            raise ValueError(
                "telemetry spool workers must be positive when the spool is enabled"
            )
        self.telemetry_spool_workers = telemetry_spool_workers
        self.telemetry_spool_lease_seconds = telemetry_spool_lease_seconds
        self.telemetry_spool_retry_backoff_seconds = (
            telemetry_spool_retry_backoff_seconds
        )
        if not 2 <= telemetry_spool_notification_fallback_seconds <= 5:
            raise ValueError(
                "telemetry spool notification fallback must be between 2 and 5 seconds"
            )
        self.telemetry_spool_notification_fallback_seconds = (
            telemetry_spool_notification_fallback_seconds
        )
        if (
            telemetry_spool_fault_pressure_workers <= 0
            or telemetry_spool_fault_pressure_workers > telemetry_spool_workers
        ):
            raise ValueError(
                "telemetry spool fault-pressure workers must be "
                "positive and not exceed normal spool workers"
            )
        if telemetry_spool_fault_pressure_poll_seconds <= 0:
            raise ValueError(
                "telemetry spool fault-pressure poll interval must be positive"
            )
        self.telemetry_spool_fault_pressure_workers = (
            telemetry_spool_fault_pressure_workers
        )
        self.telemetry_spool_fault_pressure_poll_seconds = (
            telemetry_spool_fault_pressure_poll_seconds
        )
        if telemetry_spool_max_in_flight_bytes <= 0:
            raise ValueError("telemetry spool in-flight byte limit must be positive")
        if not (
            1 <= telemetry_spool_replay_batch_max_items <= self._TELEMETRY_BATCH_MAX
        ):
            raise ValueError(
                "telemetry spool replay batch item limit must be "
                f"between 1 and {self._TELEMETRY_BATCH_MAX}"
            )
        if (
            telemetry_spool_replay_batch_max_bytes <= 0
            or telemetry_spool_replay_batch_max_bytes
            > telemetry_spool_max_in_flight_bytes
        ):
            raise ValueError(
                "telemetry spool replay batch byte limit must be "
                "positive and not exceed the in-flight byte limit"
            )
        self.telemetry_spool_max_in_flight_bytes = telemetry_spool_max_in_flight_bytes
        self.telemetry_spool_replay_batch_max_items = (
            telemetry_spool_replay_batch_max_items
        )
        self.telemetry_spool_replay_batch_max_bytes = (
            telemetry_spool_replay_batch_max_bytes
        )
        self._initialize_spool_metrics()
        self._initialize_runtime_state(on_unhealthy)

    def _initialize_claim_metrics(self) -> None:
        # Set while at least one worker pool has an unfinished future.
        self._pools_busy = False
        self.claim_rounds_total = 0
        self.claim_rows_total = 0
        self.claim_empty_total = 0
        self.claim_rounds_by_stream: dict[str, int] = {}
        self.claim_rows_by_stream: dict[str, int] = {}
        self.claim_empty_by_stream: dict[str, int] = {}
        self.claim_backoff_skips_total = 0
        self.claim_probes_total = 0
        self.claim_lane_blocked_total = 0
        self.claim_seconds_sum = 0.0
        self.claim_seconds_max = 0.0
        # Set by every claim round: True while at least one claim came
        # back full, which is the only case where skipping the poll
        # interval is justified.
        self._claim_saturated = False

    def _initialize_spool_metrics(self) -> None:
        self.spool_claim_rounds_total = 0
        self.spool_claim_rows_total = 0
        self.spool_claim_rows_by_path: dict[str, int] = {}
        self.spool_claim_empty_total = 0
        self.spool_completed_total = 0
        self.spool_superseded_total = 0
        self.spool_released_total = 0
        self.spool_dropped_total = 0
        self.spool_abandoned_total = 0
        self.spool_errors_total = 0
        self.spool_direct_replay_total = 0
        self.spool_http_replay_total = 0
        self.spool_replay_seconds_sum = 0.0
        self.spool_replay_seconds_max = 0.0
        self._spool_in_flight_bytes = 0
        self._spool_in_flight_max_bytes = 0
        self._spool_consumer_running = False
        self._spool_work_available = Event()
        self._spool_notifications_enabled = False
        self._spool_notifications_received_total = 0
        self._spool_notification_reconnects_total = 0
        self._spool_fallback_polls_total = 0
        self._spool_fault_backlog_depth = 0
        self._spool_fault_pressure_active = False
        self._processor_fault_pressure_active = False
        self._processor_fault_pressure_activations_total = 0
        self.telemetry_spool_replay_handler = None

    def _initialize_runtime_state(self, on_unhealthy) -> None:
        self.on_unhealthy = on_unhealthy
        self._state_lock = RLock()
        self._leadership: ProcessorLeadership | None = None
        self._stop = Event()
        self._work_available = Event()
        self._notifications_enabled = False
        self._notification_shard: int | None = None
        self._notification_pending_streams: set[str] = set()
        self._notifications_received_total = 0
        self._notifications_filtered_total = 0
        self._notification_reconnects_total = 0
        self._processed = {"success": 0, "error": 0}
        self._duration_buckets = (
            0.1,
            0.25,
            0.5,
            1.0,
            2.5,
            5.0,
            10.0,
            30.0,
            60.0,
            120.0,
        )
        self._duration_counts = [0] * len(self._duration_buckets)
        self._duration_count = 0
        self._duration_sum = 0.0
        self._lane_wait = {
            scope: {"count": 0, "sum": 0.0, "max": 0.0}
            for scope in ("attempt", "node", "cluster")
        }
        self._in_flight: dict[str, _InFlightRequest] = {}
        self._initialize_lane_runtime_state()
        self._deadline_exceeded_requests: set[str] = set()
        self._deadline_exceeded_total = 0
        self._completion_retries_total = 0
        self._completion_failures_total = 0
        self._stale_superseded_total = 0
        self._stale_superseded_by_path: dict[str, int] = {}
        self._unhealthy_reason: str | None = None
        self._unhealthy_since: datetime | None = None

    @property
    def leadership(self) -> ProcessorLeadership | None:
        with self._state_lock:
            return self._leadership

    def is_leader(self) -> bool:
        leadership = self.leadership
        return (
            leadership is not None
            and leadership.owner_id == self.owner_id
            and leadership.lease_expires_at > datetime.now(timezone.utc)
        )

    def is_healthy(self) -> bool:
        with self._state_lock:
            return self._unhealthy_reason is None

    @property
    def spool_consumer_running(self) -> bool:
        with self._state_lock:
            return self._spool_consumer_running

    @property
    def unhealthy_reason(self) -> str | None:
        with self._state_lock:
            return self._unhealthy_reason

    def stop(self) -> None:
        self._stop.set()
        self._work_available.set()
        self._spool_work_available.set()

    def _set_notification_state(self, enabled: bool, shard: int | None) -> None:
        with self._state_lock:
            if self._notifications_enabled and not enabled:
                self._notification_reconnects_total += 1
            self._notifications_enabled = enabled
            self._notification_shard = shard
        if enabled:
            self._work_available.set()

    def _set_processor_fault_pressure(self, active: bool) -> None:
        with self._state_lock:
            if active and not self._processor_fault_pressure_active:
                self._processor_fault_pressure_activations_total += 1
            self._processor_fault_pressure_active = active

    def notify_work_available(self, payload: str) -> None:
        try:
            notification = json.loads(payload)
        except (TypeError, json.JSONDecodeError):
            stream = "*"
        else:
            request_id = notification.get("request_id")
            target_shard = processor_partition_id(
                request_id if isinstance(request_id, str) else payload,
                self.processor_notification_shard_count,
            )
            with self._state_lock:
                owned_shard = self._notification_shard
            if owned_shard != target_shard:
                with self._state_lock:
                    self._notifications_filtered_total += 1
                return
            path = notification.get("path")
            if notification.get("priority") == 0:
                stream = "fault"
            elif path in self._OBSERVATION_PATHS:
                stream = "observation"
            elif path == self._GPU_INVENTORY_PATH:
                stream = "gpu-inventory"
            elif path == self._GPU_METRICS_PATH:
                stream = "gpu-metrics"
            elif path == self._HOST_TELEMETRY_PATH:
                stream = "host-telemetry"
            elif path == self._NODE_LOG_PATH:
                stream = "node-log"
            else:
                stream = "fault"
        with self._state_lock:
            self._notifications_received_total += 1
            self._notification_pending_streams.add(stream)
        self._work_available.set()

    def run_queue_notifications(self) -> None:
        listener = getattr(
            self.store,
            "listen_processor_queue_notifications",
            None,
        )
        if listener is None:
            return
        try:
            listener(
                self._stop,
                self.owner_id,
                self.processor_notification_shard_count,
                self.notify_work_available,
                self._set_notification_state,
            )
        except Exception:
            self._set_notification_state(False, None)
            LOGGER.exception("processor queue notification listener failed")

    def _set_spool_notification_state(self, enabled: bool) -> None:
        with self._state_lock:
            if self._spool_notifications_enabled and not enabled:
                self._spool_notification_reconnects_total += 1
            self._spool_notifications_enabled = enabled
        if enabled:
            self._spool_work_available.set()

    def notify_telemetry_spool_work_available(self, _payload: str) -> None:
        with self._state_lock:
            self._spool_notifications_received_total += 1
        self._spool_work_available.set()

    def run_telemetry_spool_notifications(self) -> None:
        listener = getattr(
            self.store,
            "listen_telemetry_spool_notifications",
            None,
        )
        if listener is None:
            return
        try:
            listener(
                self._stop,
                self.notify_telemetry_spool_work_available,
                self._set_spool_notification_state,
            )
        except Exception:
            self._set_spool_notification_state(False)
            LOGGER.exception("telemetry spool notification listener failed")

    def _wait_for_work(self, timeout: float) -> None:
        self._work_available.wait(timeout)
        self._work_available.clear()

    def _next_claim_wait_seconds(self) -> float:
        now = time.monotonic()
        deadlines = [
            deadline - now
            for deadline in self._stream_idle_until.values()
            if deadline > now
        ]
        if not deadlines:
            return self.poll_seconds
        return max(self.poll_seconds, min(deadlines))

    def abandon_in_flight(self) -> dict[str, int]:
        """Stop claiming and release every request lease we still hold."""
        self._stop.set()
        with self._state_lock:
            items = [state.item for state in self._in_flight.values()]
        released = 0
        failed = 0
        for item in items:
            try:
                self._release(item)
                released += 1
            except Exception:
                failed += 1
                LOGGER.exception(
                    "processor in-flight abandon failed request_id=%s path=%s lane=%s",
                    item.request_id,
                    item.path,
                    item.ordering_key(),
                )
        return {
            "in_flight": len(items),
            "released": released,
            "failed": failed,
        }

    def run_leadership(self) -> None:
        while not self._stop.is_set():
            if not self.is_healthy():
                with self._state_lock:
                    self._leadership = None
                self._stop.wait(self.renew_seconds)
                continue
            now = datetime.now(timezone.utc)
            try:
                leadership = self.store.acquire_processor_leadership(
                    self.owner_id,
                    now=now,
                    lease_duration=timedelta(seconds=self.lease_seconds),
                )
                with self._state_lock:
                    self._leadership = (
                        leadership if leadership.owner_id == self.owner_id else None
                    )
            except Exception:
                with self._state_lock:
                    self._leadership = None
                LOGGER.exception("processor leadership renewal failed")
            self._stop.wait(self.renew_seconds)

    def run_processor(self) -> None:
        pools = {
            name: ThreadPoolExecutor(
                max_workers=count,
                thread_name_prefix=f"gpu-fault-{name}",
            )
            for name, count in self.worker_counts.items()
            if count > 0
        }
        futures: dict[str, set[Future]] = {name: set() for name in pools}
        try:
            while not self._stop.is_set():
                with self._state_lock:
                    notification_streams = set(self._notification_pending_streams)
                    self._notification_pending_streams.clear()
                if "*" in notification_streams:
                    self._stream_idle_until.clear()
                    self._stream_idle_interval.clear()
                else:
                    for stream in notification_streams:
                        self._stream_idle_until.pop(stream, None)
                        self._stream_idle_interval.pop(stream, None)
                self._check_execution_deadlines()
                for name, pool_futures in futures.items():
                    completed = {future for future in pool_futures if future.done()}
                    for future in completed:
                        try:
                            future.result()
                        except Exception:
                            LOGGER.exception(
                                "processor %s request execution failed",
                                name,
                            )
                    pool_futures.difference_update(completed)
                self._set_processor_fault_pressure(bool(futures.get("fault")))
                if not self.is_healthy():
                    self._stop.wait(self.poll_seconds)
                    continue
                leadership = self.leadership
                if not self.active_consumers and (
                    leadership is None or not self.is_leader()
                ):
                    self._stop.wait(self.poll_seconds)
                    continue
                available = {
                    name: self.worker_counts[name] - len(pool_futures)
                    for name, pool_futures in futures.items()
                }
                total_available = sum(available.values())
                # An empty claim while workers are still executing earlier
                # rows means "every eligible lane is leased", not "the queue
                # is idle" - so the idle backoff must not grow to seconds.
                self._pools_busy = any(
                    pool_futures for pool_futures in futures.values()
                )
                if total_available <= 0:
                    self._stop.wait(self.poll_seconds)
                    continue
                try:
                    lease_seconds = min(
                        self.request_lease_seconds,
                        self.request_max_execution_seconds,
                    )
                    if self.active_consumers:
                        requests = self._claim_active_by_pool(
                            available,
                            lease_duration=timedelta(seconds=lease_seconds),
                        )
                    else:
                        requests = self.store.claim_processor_requests(
                            self.owner_id,
                            leadership.epoch,
                            now=datetime.now(timezone.utc),
                            lease_duration=timedelta(seconds=lease_seconds),
                            limit=total_available,
                        )
                        self._claim_saturated = len(requests) >= total_available
                    self.register_claimed_requests(requests)
                    observation_requests = []
                    telemetry_requests: dict[
                        tuple[str, str], list[ProcessorRequest]
                    ] = {}
                    for item in requests:
                        pool_name = self._pool_for_request(item)
                        if pool_name == "observation":
                            observation_requests.append(item)
                            continue
                        if (
                            pool_name in {"gpu", "host"}
                            and item.path in BATCHABLE_CHANNEL_PATHS
                        ):
                            telemetry_requests.setdefault(
                                (pool_name, item.path), []
                            ).append(item)
                            continue
                        if available[pool_name] <= 0:
                            self._release(item)
                            continue
                        futures[pool_name].add(
                            pools[pool_name].submit(self._process, item)
                        )
                        available[pool_name] -= 1
                    while observation_requests:
                        batch = observation_requests[: self._OBSERVATION_BATCH_SIZE]
                        del observation_requests[: len(batch)]
                        if available["observation"] <= 0:
                            for item in batch:
                                self._release(item)
                            continue
                        futures["observation"].add(
                            pools["observation"].submit(
                                self._process_observation_batch,
                                batch,
                            )
                        )
                        available["observation"] -= 1
                    for (
                        pool_name,
                        path,
                    ), pending in telemetry_requests.items():
                        batch_size = telemetry_batch_size(path)
                        while pending:
                            batch = pending[:batch_size]
                            del pending[: len(batch)]
                            if available[pool_name] <= 0:
                                for item in batch:
                                    self._release(item)
                                continue
                            futures[pool_name].add(
                                pools[pool_name].submit(
                                    self._process_telemetry_batch,
                                    batch,
                                )
                            )
                            available[pool_name] -= 1
                    for item in requests:
                        LOGGER.info(
                            "processor request claimed "
                            "request_id=%s path=%s lane=%s owner=%s "
                            "epoch=%s lease_expires_at=%s",
                            item.request_id,
                            item.path,
                            item.ordering_key(),
                            self.owner_id,
                            item.leader_epoch,
                            item.lease_expires_at.isoformat()
                            if item.lease_expires_at
                            else None,
                        )
                    if not requests or not self._claim_saturated:
                        # Nothing claimed, or no claim filled its limit:
                        # either way the next round would re-run the
                        # claim query against the same drained or
                        # lane-blocked backlog. Sleep one poll interval
                        # so the query rate stays bounded; a consumer
                        # that is keeping up never gets here.
                        self._wait_for_work(self._next_claim_wait_seconds())
                except Exception:
                    LOGGER.exception("processor request cycle failed")
                    self._stop.wait(self.poll_seconds)
        finally:
            for pool in pools.values():
                pool.shutdown(wait=True, cancel_futures=True)
            self.release_unstarted_claims()

    def _telemetry_spool_batch_bytes(self, items: list) -> int:
        return 16 + sum(self._telemetry_spool_item_bytes(item) + 1 for item in items)

    def _claim_active_by_pool(
        self,
        available: dict[str, int],
        *,
        lease_duration: timedelta,
    ) -> list[ProcessorRequest]:
        now = datetime.now(timezone.utc)
        claimed: list[ProcessorRequest] = []
        dedicated_paths = set()
        if self.worker_counts.get("observation", 0) > 0:
            dedicated_paths.update(self._OBSERVATION_PATHS)
        if self.worker_counts.get("gpu", 0) > 0:
            dedicated_paths.update(
                {
                    self._GPU_INVENTORY_PATH,
                    self._GPU_METRICS_PATH,
                }
            )
        if self.worker_counts.get("host", 0) > 0:
            dedicated_paths.update(
                {
                    self._HOST_TELEMETRY_PATH,
                    self._NODE_LOG_PATH,
                }
            )

        self._claim_saturated = False

        def claim(
            *,
            limit: int,
            stream: str,
            include_paths: set[str] | None = None,
            exclude_paths: set[str] | None = None,
        ) -> list[ProcessorRequest]:
            if limit <= 0:
                return []
            backoff_eligible = True
            monotonic = time.monotonic()
            if backoff_eligible and monotonic < self._stream_idle_until.get(
                stream, 0.0
            ):
                self.claim_backoff_skips_total += 1
                return []
            claim_started = time.monotonic()
            rows = self.store.claim_active_processor_requests(
                self.owner_id,
                now=now,
                lease_duration=lease_duration,
                limit=limit,
                include_paths=include_paths,
                exclude_paths=exclude_paths,
                routine_starvation_seconds=(self.routine_starvation_seconds),
            )
            claim_elapsed = time.monotonic() - claim_started
            self.claim_rounds_total += 1
            self.claim_rows_total += len(rows)
            self.claim_rounds_by_stream[stream] = (
                self.claim_rounds_by_stream.get(stream, 0) + 1
            )
            self.claim_rows_by_stream[stream] = self.claim_rows_by_stream.get(
                stream, 0
            ) + len(rows)
            self.claim_seconds_sum += claim_elapsed
            self.claim_seconds_max = max(self.claim_seconds_max, claim_elapsed)
            if not rows:
                self.claim_empty_total += 1
                self.claim_empty_by_stream[stream] = (
                    self.claim_empty_by_stream.get(stream, 0) + 1
                )
            if backoff_eligible:
                if rows:
                    self._stream_idle_until.pop(stream, None)
                    self._stream_idle_interval.pop(stream, None)
                else:
                    next_interval = (
                        self._stream_idle_interval.get(stream, self.poll_seconds) * 2
                    )
                    ceiling = (
                        (
                            self.processor_notification_fallback_seconds
                            if self._notifications_enabled
                            else self.fault_idle_backoff_max_seconds
                        )
                        if stream == "fault"
                        else (
                            self.processor_notification_fallback_seconds
                            if self._notifications_enabled
                            else self.idle_backoff_max_seconds
                        )
                    )
                    if stream == "fault":
                        if self._pools_busy or self._backlog_is_lane_blocked(
                            now=now,
                            include_paths=include_paths,
                            exclude_paths=exclude_paths,
                        ):
                            ceiling = self.fault_busy_backoff_max_seconds
                    elif next_interval > self.busy_backoff_max_seconds and (
                        self._pools_busy
                        or self._backlog_is_lane_blocked(
                            now=now,
                            include_paths=include_paths,
                            exclude_paths=exclude_paths,
                        )
                    ):
                        ceiling = self.busy_backoff_max_seconds
                    interval = min(ceiling, next_interval)
                    self._stream_idle_interval[stream] = interval
                    self._stream_idle_until[stream] = monotonic + interval
            # A sub-claim that filled its limit means there is more
            # behind it, so the caller should come straight back. One
            # that did not is either drained or lane-blocked, and the
            # claim query is the most expensive statement in the
            # system - re-issuing it with no wait burns a core and an
            # Aurora connection for the same handful of rows.
            if len(rows) >= limit:
                self._claim_saturated = True
            return rows

        fault_requests = claim(
            limit=available.get("fault", 0),
            stream="fault",
            exclude_paths=dedicated_paths or None,
        )
        claimed.extend(fault_requests)
        fault_pressure_active = self._processor_fault_pressure_active or bool(
            fault_requests
        )
        self._set_processor_fault_pressure(fault_pressure_active)

        def pressure_limited_slots(pool_name: str) -> int:
            slots = available.get(pool_name, 0)
            if not fault_pressure_active:
                return slots
            active = self.worker_counts.get(pool_name, 0) - slots
            return max(
                0,
                min(
                    slots,
                    self.fault_pressure_evidence_workers - active,
                ),
            )

        observation_slots = available.get("observation", 0)
        claimed.extend(
            claim(
                limit=(observation_slots * self._OBSERVATION_BATCH_SIZE),
                stream="observation",
                include_paths=self._OBSERVATION_PATHS,
            )
        )

        gpu_slots = pressure_limited_slots("gpu")
        if gpu_slots > 0:
            # Inventory batches amortize one worker across up to 16
            # nodes. Reserve one slot for that stream and leave the
            # remaining workers to the materially heavier DCGM rule
            # evaluation path.
            initial_inventory_slots = 1
            inventory = claim(
                limit=(initial_inventory_slots * GPU_INVENTORY_BATCH_SIZE),
                stream="gpu-inventory",
                include_paths={self._GPU_INVENTORY_PATH},
            )
            claimed.extend(inventory)
            used_inventory_slots = (
                len(inventory) + GPU_INVENTORY_BATCH_SIZE - 1
            ) // GPU_INVENTORY_BATCH_SIZE
            remaining_gpu_slots = max(0, gpu_slots - used_inventory_slots)
            metrics = claim(
                limit=(remaining_gpu_slots * GPU_METRICS_BATCH_SIZE),
                stream="gpu-metrics",
                include_paths={self._GPU_METRICS_PATH},
            )
            claimed.extend(metrics)
            remaining_gpu_slots -= (
                len(metrics) + GPU_METRICS_BATCH_SIZE - 1
            ) // GPU_METRICS_BATCH_SIZE
            if remaining_gpu_slots > 0:
                claimed.extend(
                    claim(
                        limit=(remaining_gpu_slots * GPU_INVENTORY_BATCH_SIZE),
                        stream="gpu-inventory",
                        include_paths={self._GPU_INVENTORY_PATH},
                    )
                )

        host_slots = pressure_limited_slots("host")
        if host_slots > 0:
            node_logs = claim(
                limit=1,
                stream="node-log",
                include_paths={self._NODE_LOG_PATH},
            )
            claimed.extend(node_logs)
            remaining_host_slots = host_slots - len(node_logs)
            host_telemetry = claim(
                limit=(remaining_host_slots * HOST_TELEMETRY_BATCH_SIZE),
                stream="host-telemetry",
                include_paths={self._HOST_TELEMETRY_PATH},
            )
            claimed.extend(host_telemetry)
            remaining_host_slots -= (
                len(host_telemetry) + HOST_TELEMETRY_BATCH_SIZE - 1
            ) // HOST_TELEMETRY_BATCH_SIZE
            if remaining_host_slots > 0:
                claimed.extend(
                    claim(
                        limit=remaining_host_slots,
                        stream="node-log",
                        include_paths={self._NODE_LOG_PATH},
                    )
                )
        return claimed

    def _pool_for_request(self, item: ProcessorRequest) -> str:
        if item.queue_priority() == 0:
            return "fault"
        channel = channel_for_path(item.path)
        selected = channel.pool.value if channel is not None else "fault"
        return selected if self.worker_counts.get(selected, 0) > 0 else "fault"

    def _process(self, item: ProcessorRequest) -> None:
        if self._complete_if_stale(item):
            return
        deadline = self._request_started(item)
        renewal_stop = Event()
        renewal_thread = None
        if self.active_consumers:
            renewal_thread = Thread(
                target=self._renew_active_request,
                args=(item, renewal_stop, deadline),
                name=f"gpu-fault-processor-renew-{item.request_id}",
                daemon=True,
            )
            renewal_thread.start()
        LOGGER.info(
            "processor request started request_id=%s path=%s lane=%s owner=%s epoch=%s",
            item.request_id,
            item.path,
            item.ordering_key(),
            self.owner_id,
            item.leader_epoch,
        )
        try:
            self._execute(item, deadline=deadline)
        finally:
            renewal_stop.set()
            if renewal_thread is not None:
                renewal_thread.join(timeout=self.request_renew_seconds + 1)
            self._request_finished(item.request_id)

    def _process_observation_batch(self, items: list[ProcessorRequest]) -> None:
        from gpu_fault.watcher import AttemptObservation

        items = [item for item in items if not self._complete_if_stale(item)]
        if not items:
            return
        parsed = []
        invalid = []
        for item in items:
            try:
                parsed.append(
                    (
                        item,
                        AttemptObservation.model_validate_json(item.body()),
                    )
                )
            except Exception:
                invalid.append(item)
        for item in invalid:
            self._process(item)
        if not parsed:
            return
        started = {item.request_id: time.monotonic() for item, _ in parsed}
        for item, _ in parsed:
            self._request_started(item)
        try:
            self.store.save_attempt_observations_batch(
                [observation for _, observation in parsed]
            )
            completions = [
                {
                    "request_id": item.request_id,
                    "owner_id": self.owner_id,
                    "lane_epoch": item.leader_epoch,
                    "lease_token": item.lease_token,
                    "response_status": 200,
                    "response_content_type": "application/json",
                    "response_body_base64": base64.b64encode(
                        observation.model_dump_json().encode()
                    ).decode("ascii"),
                }
                for item, observation in parsed
            ]
            results = self.store.complete_active_processor_requests_batch(completions)
            for (item, _), result in zip(parsed, results, strict=True):
                if result is None:
                    raise ValueError("stale processor lane fencing token")
                self._observe_processing(
                    "success",
                    time.monotonic() - started[item.request_id],
                )
        except Exception:
            for item, _ in parsed:
                try:
                    self._release(item)
                except Exception:
                    LOGGER.exception(
                        "observation batch release failed request_id=%s",
                        item.request_id,
                    )
                self._observe_processing(
                    "error",
                    time.monotonic() - started[item.request_id],
                )
            raise
        finally:
            for item, _ in parsed:
                self._request_finished(item.request_id)

    def _process_telemetry_batch(self, items: list[ProcessorRequest]) -> None:
        items = [item for item in items if not self._complete_if_stale(item)]
        if not items:
            return
        started = {item.request_id: time.monotonic() for item in items}
        for item in items:
            self._request_started(item)
        request = urllib_request.Request(
            self.local_url + "/v1/internal/processor/telemetry-batch",
            data=json.dumps(
                {
                    "items": [
                        {
                            "request_id": item.request_id,
                            "path": item.path,
                            "payload": json.loads(item.body()),
                        }
                        for item in items
                    ]
                },
                separators=(",", ":"),
            ).encode(),
            headers={
                "Content-Type": "application/json",
                "X-GPU-Fault-Processor-Replay": self.internal_token,
                "Idempotency-Key": ",".join(item.request_id for item in items),
            },
            method="POST",
        )
        try:
            with urlopen(
                request,
                timeout=self.request_max_execution_seconds,
            ) as response:
                body = json.loads(response.read())
            by_id = {item["request_id"]: item for item in body.get("results", [])}
            completions = []
            for item in items:
                result = by_id.get(item.request_id)
                if result is None:
                    raise ValueError("telemetry batch response omitted request")
                completions.append(
                    {
                        "request_id": item.request_id,
                        "owner_id": self.owner_id,
                        "lane_epoch": item.leader_epoch,
                        "lease_token": item.lease_token,
                        "response_status": result["status"],
                        "response_content_type": "application/json",
                        "response_body_base64": base64.b64encode(
                            json.dumps(
                                result["body"],
                                separators=(",", ":"),
                            ).encode()
                        ).decode("ascii"),
                    }
                )
            completed = self.store.complete_active_processor_requests_batch(completions)
            for item, result in zip(items, completed, strict=True):
                if result is None:
                    raise ValueError("stale processor lane fencing token")
                self._observe_processing(
                    "success",
                    time.monotonic() - started[item.request_id],
                )
        except Exception:
            for item in items:
                try:
                    self._release(item)
                except Exception:
                    LOGGER.exception(
                        "telemetry batch release failed request_id=%s",
                        item.request_id,
                    )
                self._observe_processing(
                    "error",
                    time.monotonic() - started[item.request_id],
                )
            raise
        finally:
            for item in items:
                self._request_finished(item.request_id)

    @staticmethod
    def _payload_observed_at(payload: dict) -> datetime | None:
        value = payload.get("observed_at")
        if not isinstance(value, str) or not value:
            return None
        try:
            observed_at = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        if observed_at.tzinfo is None:
            observed_at = observed_at.replace(tzinfo=timezone.utc)
        return observed_at.astimezone(timezone.utc)

    def _stale_disposition(
        self,
        item: ProcessorRequest,
        *,
        now: datetime | None = None,
    ) -> tuple[str, datetime, float, float] | None:
        try:
            payload = json.loads(item.body())
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        observed_at = self._payload_observed_at(payload)
        if observed_at is None:
            return None
        observed_now = now or datetime.now(timezone.utc)
        age_seconds = max(0.0, (observed_now - observed_at).total_seconds())

        if item.path == self._GPU_INVENTORY_PATH:
            limit = self.gpu_inventory_stale_seconds
            if age_seconds > limit:
                return (
                    "expired GPU inventory snapshot",
                    observed_at,
                    age_seconds,
                    limit,
                )
            return None

        if item.path == self._GPU_METRICS_PATH:
            reasons = payload.get("edge_filter_reasons") or []
            if (
                set(reasons) == {"health-summary"}
                and age_seconds > self.health_summary_stale_seconds
            ):
                return (
                    "expired healthy GPU metrics summary",
                    observed_at,
                    age_seconds,
                    self.health_summary_stale_seconds,
                )
            return None

        if item.path == self._HOST_TELEMETRY_PATH:
            reasons = payload.get("edge_filter_reasons") or []
            collection_errors = payload.get("collection_errors") or []
            if (
                set(reasons) == {"health-summary"}
                and not collection_errors
                and age_seconds > self.health_summary_stale_seconds
            ):
                return (
                    "expired healthy host telemetry summary",
                    observed_at,
                    age_seconds,
                    self.health_summary_stale_seconds,
                )
            return None

        if (
            item.path == WORKLOAD_OBSERVATIONS_PATH
            and age_seconds > self.observation_stale_seconds
        ):
            cluster_id = payload.get("cluster_id")
            attempt_id = payload.get("attempt_id")
            if (
                isinstance(cluster_id, str)
                and cluster_id
                and isinstance(attempt_id, str)
                and attempt_id
                and hasattr(self.store, "list_attempt_observation_states")
            ):
                states = self.store.list_attempt_observation_states(cluster_id)
                if any(
                    state.observation.attempt_id == attempt_id
                    and state.observation.observed_at >= observed_at
                    for state in states
                ):
                    return (
                        "superseded workload observation",
                        observed_at,
                        age_seconds,
                        self.observation_stale_seconds,
                    )
            return None

        if (
            item.path == TRAINING_PROGRESS_PATH
            and age_seconds > self.training_progress_stale_seconds
        ):
            cluster_id = payload.get("cluster_id")
            attempt_id = payload.get("attempt_id")
            rank = payload.get("rank")
            if (
                isinstance(cluster_id, str)
                and cluster_id
                and isinstance(attempt_id, str)
                and attempt_id
                and isinstance(rank, int)
                and hasattr(self.store, "list_training_progress_states")
            ):
                states = self.store.list_training_progress_states(
                    cluster_id, attempt_id
                )
                if any(
                    state.heartbeat.rank == rank
                    and state.heartbeat.observed_at >= observed_at
                    for state in states
                ):
                    return (
                        "superseded training progress heartbeat",
                        observed_at,
                        age_seconds,
                        self.training_progress_stale_seconds,
                    )
        return None

    def _complete_if_stale(self, item: ProcessorRequest) -> bool:
        disposition = self._stale_disposition(item)
        if disposition is None:
            return False
        reason, observed_at, age_seconds, limit_seconds = disposition
        started = time.monotonic()
        self._request_started(item)
        response = json.dumps(
            {
                "status": "STALE_SUPERSEDED",
                "reason": reason,
                "observed_at": observed_at.isoformat(),
                "age_seconds": round(age_seconds, 6),
                "limit_seconds": limit_seconds,
            },
            separators=(",", ":"),
        ).encode()
        try:
            if self.active_consumers:
                completed = self.store.complete_active_processor_request(
                    item.request_id,
                    self.owner_id,
                    item.leader_epoch,
                    item.lease_token,
                    response_status=200,
                    response_content_type="application/json",
                    response_body_base64=base64.b64encode(response).decode("ascii"),
                    path=item.path,
                )
            else:
                completed = self.store.complete_processor_request(
                    item.request_id,
                    self.owner_id,
                    item.leader_epoch,
                    item.lease_token,
                    response_status=200,
                    response_content_type="application/json",
                    response_body_base64=base64.b64encode(response).decode("ascii"),
                )
            if completed is None:
                raise ValueError("stale processor lane fencing token")
            with self._state_lock:
                self._stale_superseded_total += 1
                self._stale_superseded_by_path[item.path] = (
                    self._stale_superseded_by_path.get(item.path, 0) + 1
                )
            self._observe_processing("success", time.monotonic() - started)
            LOGGER.info(
                "processor request completed as stale superseded "
                "request_id=%s path=%s observed_at=%s age_seconds=%.3f "
                "limit_seconds=%.3f reason=%s",
                item.request_id,
                item.path,
                observed_at.isoformat(),
                age_seconds,
                limit_seconds,
                reason,
            )
            return True
        except Exception:
            try:
                self._release(item)
            except Exception:
                LOGGER.exception(
                    "stale processor request release failed request_id=%s",
                    item.request_id,
                )
            self._observe_processing("error", time.monotonic() - started)
            raise
        finally:
            self._request_finished(item.request_id)

    def _renew_active_request(
        self,
        item: ProcessorRequest,
        stop: Event,
        deadline: float,
    ) -> None:
        while not stop.wait(self.request_renew_seconds):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._mark_execution_deadline_exceeded(item)
                return
            try:
                renewed = self.store.renew_active_processor_request(
                    item.request_id,
                    self.owner_id,
                    item.leader_epoch,
                    item.lease_token,
                    lease_duration=timedelta(
                        seconds=min(
                            self.request_lease_seconds,
                            remaining,
                        )
                    ),
                )
            except Exception:
                LOGGER.exception(
                    "processor request renewal failed "
                    "request_id=%s path=%s lane=%s owner=%s epoch=%s",
                    item.request_id,
                    item.path,
                    item.ordering_key(),
                    self.owner_id,
                    item.leader_epoch,
                )
                return
            if not renewed:
                LOGGER.warning(
                    "processor request renewal fenced "
                    "request_id=%s path=%s lane=%s owner=%s epoch=%s",
                    item.request_id,
                    item.path,
                    item.ordering_key(),
                    self.owner_id,
                    item.leader_epoch,
                )
                return

    def _execute(self, item: ProcessorRequest, *, deadline: float) -> None:
        started = time.monotonic()
        outcome = "error"
        if not item.lease_token or item.leader_epoch is None:
            self._observe_processing(outcome, time.monotonic() - started)
            raise RuntimeError("claimed processor request has no lease")
        url = self.local_url + item.path
        if item.query:
            url += "?" + item.query
        headers = {
            "X-GPU-Fault-Processor-Replay": self.internal_token,
            "X-GPU-Fault-Processor-Owner-ID": self.owner_id,
            "X-GPU-Fault-Processor-Request-ID": item.request_id,
        }
        if self.active_consumers:
            headers["X-GPU-Fault-Processor-Lane-Epoch"] = str(item.leader_epoch)
            headers["X-GPU-Fault-Processor-Lane-Token"] = item.lease_token
            headers["X-GPU-Fault-Processor-Lane-Key"] = base64.urlsafe_b64encode(
                item.ordering_key().encode("utf-8")
            ).decode("ascii")
        if item.execution_authorized:
            if not self.execution_token:
                self._release(item)
                self._observe_processing(outcome, time.monotonic() - started)
                raise RuntimeError("processor request requires an execution token")
            headers["X-GPU-Fault-Execution-Token"] = self.execution_token
        if item.content_type:
            headers["Content-Type"] = item.content_type
        if item.cluster_id:
            headers["X-GPU-Fault-Cluster-ID"] = item.cluster_id
        request = urllib_request.Request(
            url,
            data=item.body(),
            headers=headers,
            method=item.method,
        )
        self._set_request_phase(item.request_id, "replay_http")
        for attempt in range(3):
            try:
                with urlopen(
                    request,
                    timeout=max(
                        0.1,
                        min(
                            self.request_lease_seconds,
                            self.request_max_execution_seconds,
                        ),
                    ),
                ) as response:
                    status = response.status
                    content_type = response.headers.get("Content-Type")
                    retry_partition = response.headers.get(
                        "X-GPU-Fault-Processor-Retry"
                    )
                    body = response.read()
                break
            except urllib_error.HTTPError as exc:
                status = exc.code
                content_type = exc.headers.get("Content-Type")
                retry_partition = exc.headers.get("X-GPU-Fault-Processor-Retry")
                body = exc.read()
                break
            except Exception:
                if attempt < 2 and time.monotonic() < deadline:
                    time.sleep(0.05 * (2**attempt))
                    continue
                if time.monotonic() >= deadline:
                    self._mark_execution_deadline_exceeded(item)
                    self._observe_processing(outcome, time.monotonic() - started)
                    return
                self._release(item)
                self._observe_processing(outcome, time.monotonic() - started)
                raise
        if time.monotonic() >= deadline:
            self._mark_execution_deadline_exceeded(item)
            LOGGER.error(
                "processor request result discarded after deadline "
                "request_id=%s path=%s lane=%s owner=%s epoch=%s "
                "duration_seconds=%.3f",
                item.request_id,
                item.path,
                item.ordering_key(),
                self.owner_id,
                item.leader_epoch,
                time.monotonic() - started,
            )
            self._observe_processing(outcome, time.monotonic() - started)
            return
        finalize_replay_response(
            self,
            item,
            status=status,
            content_type=content_type,
            retry_partition=retry_partition,
            body=body,
            started=started,
        )

    def _request_started(self, item: ProcessorRequest) -> float:
        started = time.monotonic()
        deadline = started + self.request_max_execution_seconds
        wait_seconds = max(
            0.0,
            (datetime.now(timezone.utc) - item.created_at).total_seconds(),
        )
        ordering_key = item.ordering_key()
        scope = (
            "attempt"
            if ":attempt:" in ordering_key
            else "node"
            if ":node:" in ordering_key
            else "cluster"
        )
        with self._state_lock:
            self._mark_request_started(item.request_id)
            lane = self._lane_wait[scope]
            lane["count"] += 1
            lane["sum"] += wait_seconds
            lane["max"] = max(lane["max"], wait_seconds)
            self._in_flight[item.request_id] = _InFlightRequest(
                item=item,
                started=started,
                deadline=deadline,
                phase="claimed",
                phase_started=started,
            )
        return deadline

    def _set_request_phase(self, request_id: str, phase: str) -> None:
        observed = time.monotonic()
        with self._state_lock:
            state = self._in_flight.get(request_id)
            if state is None:
                return
            self._in_flight[request_id] = replace(
                state,
                phase=phase,
                phase_started=observed,
            )

    def in_flight_snapshot(self) -> list[dict]:
        observed = time.monotonic()
        with self._state_lock:
            states = list(self._in_flight.values())
        return [
            {
                "request_id": state.item.request_id,
                "path": state.item.path,
                "cluster_id": state.item.cluster_id,
                "ordering_key": state.item.ordering_key(),
                "phase": state.phase,
                "elapsed_seconds": max(0.0, observed - state.started),
                "phase_elapsed_seconds": max(0.0, observed - state.phase_started),
                "deadline_remaining_seconds": (state.deadline - observed),
                "lane_epoch": state.item.leader_epoch,
            }
            for state in states
        ]

    def _check_execution_deadlines(self) -> None:
        observed = time.monotonic()
        with self._state_lock:
            expired = [
                state.item
                for state in self._in_flight.values()
                if observed >= state.deadline
            ]
        for item in expired:
            self._mark_execution_deadline_exceeded(item)

    def _mark_execution_deadline_exceeded(self, item: ProcessorRequest) -> None:
        notify_unhealthy = False
        with self._state_lock:
            if item.request_id in self._deadline_exceeded_requests:
                return
            self._deadline_exceeded_requests.add(item.request_id)
            self._deadline_exceeded_total += 1
            if self._unhealthy_reason is None:
                self._unhealthy_reason = "processor request execution deadline exceeded"
                self._unhealthy_since = datetime.now(timezone.utc)
                notify_unhealthy = True
        LOGGER.error(
            "processor request execution deadline exceeded; "
            "stopping claims and requiring pod restart "
            "request_id=%s path=%s lane=%s owner=%s epoch=%s "
            "max_execution_seconds=%s",
            item.request_id,
            item.path,
            item.ordering_key(),
            self.owner_id,
            item.leader_epoch,
            self.request_max_execution_seconds,
        )
        if notify_unhealthy and self.on_unhealthy is not None:
            try:
                self.on_unhealthy("processor request execution deadline exceeded")
            except Exception:
                LOGGER.exception("processor unhealthy callback failed")
