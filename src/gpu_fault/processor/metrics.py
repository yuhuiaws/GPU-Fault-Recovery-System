from __future__ import annotations

from typing import Any

import time
from threading import RLock

from gpu_fault.processor.models import ProcessorRequest


class ProcessorMetricsMixin:
    # Attributes supplied by the composed concrete implementation.
    _completion_failures_total: Any
    _completion_retries_total: Any
    _completion_failure_releases_total: int
    _completions_by_path_status: dict[str, dict[str, int]]
    _fault_rejections_total: int
    _retry_horizon_failures_total: int
    _renewal_errors_total: int
    _renewal_fenced_total: int
    _claimed_not_started: dict[str, ProcessorRequest]
    _claimed_not_started_released_total: int
    _deadline_exceeded_total: Any
    _duration_buckets: tuple[float, ...]
    _duration_count: int
    _duration_counts: list[int]
    _duration_sum: float
    _in_flight: Any
    _lane_wait: dict[str, dict[str, float | int]]
    _lane_holder_by_path: dict[str, dict[str, float | int]]
    _notification_reconnects_total: int
    _notification_shard: int | None
    _notification_listener_connected: bool
    _notification_shardless_episodes_total: int
    _notifications_enabled: bool
    _notifications_filtered_total: int
    _notifications_received_total: int
    _fault_rows_skipped_by_observation_total: int
    _fault_rows_blocked_by_observation: int
    _interlock_probes_total: int
    _processed: dict[str, int]
    _retry_delay_seconds_max: float
    _retry_rescheduled_by_path: dict[str, int]
    _retry_rescheduled_total: int
    _processor_fault_pressure_activations_total: Any
    _processor_fault_pressure_active: Any
    _spool_consumer_running: Any
    _spool_fallback_polls_total: Any
    _spool_fault_backlog_depth: Any
    _spool_fault_pressure_active: Any
    _spool_in_flight_bytes: Any
    _spool_in_flight_max_bytes: Any
    _spool_notification_reconnects_total: Any
    _spool_notifications_enabled: Any
    _spool_notifications_received_total: Any
    _stale_superseded_by_path: Any
    _stale_superseded_total: Any
    _state_lock: RLock
    _stream_idle_interval: Any
    _unhealthy_reason: Any
    _unhealthy_since: Any
    active_consumers: Any
    claim_backoff_skips_total: Any
    claim_empty_by_stream: Any
    claim_empty_total: Any
    claim_lane_blocked_total: Any
    claim_probes_total: Any
    claim_rounds_by_stream: Any
    claim_rounds_total: Any
    claim_rows_by_stream: Any
    claim_rows_total: Any
    claim_seconds_max: Any
    claim_seconds_sum: Any
    fault_pressure_evidence_workers: Any
    processor_notification_shard_count: Any
    spool_abandoned_total: Any
    spool_claim_empty_total: Any
    spool_claim_rounds_total: Any
    spool_claim_rows_by_path: Any
    spool_claim_rows_total: Any
    spool_completed_total: Any
    spool_direct_replay_total: Any
    spool_dropped_total: Any
    spool_errors_total: Any
    spool_http_replay_total: Any
    spool_released_total: Any
    spool_replay_seconds_max: Any
    spool_replay_seconds_sum: Any
    spool_superseded_total: Any
    telemetry_spool_enabled: Any
    telemetry_spool_fault_pressure_workers: Any
    telemetry_spool_max_in_flight_bytes: Any
    telemetry_spool_notification_fallback_seconds: Any
    telemetry_spool_replay_batch_max_items: Any
    worker_count: Any

    def _observe_processing(self, outcome: str, duration_seconds: float) -> None:
        with self._state_lock:
            self._processed[outcome] += 1
            self._duration_count += 1
            self._duration_sum += duration_seconds
            for index, boundary in enumerate(self._duration_buckets):
                if duration_seconds <= boundary:
                    self._duration_counts[index] += 1

    def metrics_snapshot(self) -> dict:
        with self._state_lock:
            observed = time.monotonic()
            oldest_in_flight_seconds = max(
                (observed - state.started for state in self._in_flight.values()),
                default=0.0,
            )
            in_flight_by_phase: dict[str, dict[str, float | int]] = {}
            for state in self._in_flight.values():
                phase = in_flight_by_phase.setdefault(
                    state.phase,
                    {"count": 0, "oldest_seconds": 0.0},
                )
                phase["count"] += 1
                phase["oldest_seconds"] = max(
                    float(phase["oldest_seconds"]),
                    observed - state.phase_started,
                )
            return {
                "processed": dict(self._processed),
                "duration_buckets": list(
                    zip(
                        self._duration_buckets,
                        self._duration_counts,
                        strict=True,
                    )
                ),
                "duration_count": self._duration_count,
                "duration_sum": self._duration_sum,
                "worker_count": self.worker_count,
                "active_consumer": int(self.active_consumers),
                "in_flight": len(self._in_flight),
                "claimed_not_started": len(self._claimed_not_started),
                "claimed_not_started_released_total": (
                    self._claimed_not_started_released_total
                ),
                "oldest_in_flight_seconds": (oldest_in_flight_seconds),
                "in_flight_by_phase": in_flight_by_phase,
                "deadline_exceeded_total": (self._deadline_exceeded_total),
                "completion_retries_total": (self._completion_retries_total),
                "completion_failures_total": (self._completion_failures_total),
                "completion_failure_releases_total": (
                    self._completion_failure_releases_total
                ),
                "retry_horizon_failures_total": (self._retry_horizon_failures_total),
                # G1: a 4xx after the collector's 202 is the handler's real
                # verdict; counted per path and status class so a shape
                # drift on one channel is visible, and the fault-layer
                # subset separately because those are thrown-away faults.
                "completions_by_path_status": {
                    path: dict(values)
                    for path, values in self._completions_by_path_status.items()
                },
                "fault_rejections_total": self._fault_rejections_total,
                "renewal_errors_total": self._renewal_errors_total,
                "renewal_fenced_total": self._renewal_fenced_total,
                "retry_rescheduled_total": self._retry_rescheduled_total,
                "retry_rescheduled_by_path": dict(self._retry_rescheduled_by_path),
                "retry_delay_seconds_max": self._retry_delay_seconds_max,
                "notifications_enabled": int(self._notifications_enabled),
                "notifications_received_total": (self._notifications_received_total),
                "notification_reconnects_total": (self._notification_reconnects_total),
                "notification_shard": (
                    self._notification_shard
                    if self._notification_shard is not None
                    else -1
                ),
                "notification_shard_count": (self.processor_notification_shard_count),
                # F-D11: connected and enabled are different facts once
                # there are more consumer processes than shards.
                "notification_listener_connected": int(
                    self._notification_listener_connected
                ),
                "notification_shardless_episodes_total": (
                    self._notification_shardless_episodes_total
                ),
                # F-D3: liveness of the fault <-> observation interlock.
                "fault_rows_skipped_by_observation_total": (
                    self._fault_rows_skipped_by_observation_total
                ),
                "fault_rows_blocked_by_observation": (
                    self._fault_rows_blocked_by_observation
                ),
                "interlock_probes_total": self._interlock_probes_total,
                "fault_pressure": {
                    "active": int(self._processor_fault_pressure_active),
                    "evidence_workers": (self.fault_pressure_evidence_workers),
                    "activations_total": (
                        self._processor_fault_pressure_activations_total
                    ),
                },
                "notifications_filtered_total": (self._notifications_filtered_total),
                "stale_superseded_total": (self._stale_superseded_total),
                "stale_superseded_by_path": dict(self._stale_superseded_by_path),
                "lane_wait": {
                    scope: dict(values) for scope, values in self._lane_wait.items()
                },
                "lane_holder_by_path": {
                    path: dict(values)
                    for path, values in self._lane_holder_by_path.items()
                },
                # Written from the single consumer loop, read here - the
                # claim counters are what tell a drain tail ("empty claims
                # while lanes are blocked") from an idle queue.
                "claim": {
                    "rounds": self.claim_rounds_total,
                    "rows": self.claim_rows_total,
                    "empty": self.claim_empty_total,
                    "rounds_by_stream": dict(self.claim_rounds_by_stream),
                    "rows_by_stream": dict(self.claim_rows_by_stream),
                    "empty_by_stream": dict(self.claim_empty_by_stream),
                    "backoff_skips": self.claim_backoff_skips_total,
                    "probes": self.claim_probes_total,
                    "lane_blocked": self.claim_lane_blocked_total,
                    "seconds_sum": self.claim_seconds_sum,
                    "seconds_max": self.claim_seconds_max,
                    "backoff_seconds": max(
                        self._stream_idle_interval.values(),
                        default=0.0,
                    ),
                },
                # Kept apart from the claim counters above so an A/B can
                # attribute work to a tier: with the spool on, the queue's
                # telemetry streams go quiet and these move instead.
                "spool": {
                    "enabled": int(self.telemetry_spool_enabled),
                    "rounds": self.spool_claim_rounds_total,
                    "rows": self.spool_claim_rows_total,
                    "rows_by_path": dict(self.spool_claim_rows_by_path),
                    "empty": self.spool_claim_empty_total,
                    "completed": self.spool_completed_total,
                    "superseded": self.spool_superseded_total,
                    "released": self.spool_released_total,
                    "dropped": self.spool_dropped_total,
                    "abandoned": self.spool_abandoned_total,
                    "errors": self.spool_errors_total,
                    "direct_replay": self.spool_direct_replay_total,
                    "http_replay": self.spool_http_replay_total,
                    "replay_seconds_sum": (self.spool_replay_seconds_sum),
                    "replay_seconds_max": (self.spool_replay_seconds_max),
                    "max_in_flight_bytes": (self.telemetry_spool_max_in_flight_bytes),
                    "max_batch_items": (self.telemetry_spool_replay_batch_max_items),
                    "in_flight_bytes": (self._spool_in_flight_bytes),
                    "in_flight_bytes_max": (self._spool_in_flight_max_bytes),
                    "consumer_running": int(self._spool_consumer_running),
                    "fault_pressure_active": int(self._spool_fault_pressure_active),
                    "fault_backlog_depth": (self._spool_fault_backlog_depth),
                    "fault_pressure_workers": (
                        self.telemetry_spool_fault_pressure_workers
                    ),
                    "notifications_enabled": int(self._spool_notifications_enabled),
                    "notifications_received": (
                        self._spool_notifications_received_total
                    ),
                    "notification_reconnects": (
                        self._spool_notification_reconnects_total
                    ),
                    "fallback_polls": (self._spool_fallback_polls_total),
                    "notification_fallback_seconds": (
                        self.telemetry_spool_notification_fallback_seconds
                    ),
                },
                "healthy": int(self._unhealthy_reason is None),
                "unhealthy_reason": self._unhealthy_reason,
                "unhealthy_since": (
                    self._unhealthy_since.isoformat()
                    if self._unhealthy_since is not None
                    else None
                ),
            }
