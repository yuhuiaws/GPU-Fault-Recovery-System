"""How each exported ``/metrics`` family is aggregated over a Pod's processes.

The ingress and control-worker Pods run four uvicorn processes behind one
port and ADOT scrapes the Pod, so :mod:`gpu_fault.app.process_metrics` merges
every live process's render before answering. Merging needs a rule per
family, and the rule is a fact about the family's meaning, not its type:

``SUM``
    Counters, and gauges that add up across processes: queue depth held by
    this process, in-flight calls, pool connections, worker threads,
    rejections. Summary/histogram families are SUM as a whole; their
    ``_max`` sample takes the maximum (a high-water mark does not add).
``MAX``
    ``*_timestamp_seconds`` (the newest event anywhere in the Pod), ages and
    durations read as "worst process", high-water marks, and "any process
    is in this state" flags (fault pressure active, registry drift).
``MIN``
    "Healthy/running/connected" flags where any unhealthy process makes the
    Pod unhealthy (``gpu_fault_processor_healthy``), and freshness ages where
    the Pod is fresh if any process is (collector snapshot age).
``ANY``
    Values identical on every process: configuration constants
    (``*_max_size``, ``*_limit``, ``*_enabled`` read from the environment)
    and every store-derived, fleet-level family (the same table read on
    every process; the local render wins, then the lowest slot).
``PER_PROCESS``
    Facts that cannot be combined at all, exported once per process with a
    ``process="<slot>"`` label (the notification shard a process owns).

Every family the built-in contributors can render must be listed here;
``tests/metrics/test_metric_aggregation_registry.py`` renders each role and
fails on a family without a strategy, and on a registry entry no source file
still declares. Plugin contributors register theirs with :func:`register`.
"""

from __future__ import annotations

from enum import Enum

PROCESS_LABEL = "process"
PROCESSES_METRIC = "gpu_fault_metrics_aggregation_processes"
DEGRADED_METRIC = "gpu_fault_metrics_aggregation_degraded"


class Strategy(str, Enum):
    SUM = "sum"
    MAX = "max"
    MIN = "min"
    ANY = "any"
    PER_PROCESS = "per_process"


SUM, MAX, MIN, ANY, PER_PROCESS = (
    Strategy.SUM,
    Strategy.MAX,
    Strategy.MIN,
    Strategy.ANY,
    Strategy.PER_PROCESS,
)

# Families the merger itself emits after aggregation; listed so the drift test
# accepts them, never looked up by the engine.
MERGER_FAMILIES: dict[str, Strategy] = {
    PROCESSES_METRIC: ANY,
    DEGRADED_METRIC: ANY,
}

# Store-derived and fleet-level families: every process reads the same table,
# so the local (freshest) render wins. Configuration constants read from the
# environment sit here too.
_ANY = (
    # capacity / configuration constants
    "gpu_fault_capacity_largest_cluster_node_count",
    "gpu_fault_capacity_managed_node_count",
    "gpu_fault_processor_admission_batch_in_flight_limit",
    "gpu_fault_processor_notification_shard_count",
    "gpu_fault_processor_queue_bypass_enabled",
    "gpu_fault_telemetry_spool_enabled",
    "gpu_fault_telemetry_spool_notification_fallback_seconds",
    "gpu_fault_telemetry_spool_replay_batch_max_items",
    "gpu_fault_postgres_pool_max_size",
    "gpu_fault_postgres_pool_demand_connections",
    "gpu_fault_postgres_pool_oversubscription_ratio",
    "gpu_fault_postgres_pool_headroom_connections",
    "gpu_fault_postgres_unpooled_connections",
    # processor queue view (store)
    "gpu_fault_processor_queue_depth",
    "gpu_fault_processor_queue_oldest_age_seconds",
    "gpu_fault_processor_cluster_queue_depth",
    "gpu_fault_processor_cluster_queue_oldest_age_seconds",
    "gpu_fault_processor_counter_mode",
    # telemetry spool table view (store)
    "gpu_fault_telemetry_spool_depth",
    "gpu_fault_telemetry_spool_leased",
    "gpu_fault_telemetry_spool_oldest_age_seconds",
    "gpu_fault_telemetry_spool_bytes",
    "gpu_fault_telemetry_spool_leased_bytes",
    "gpu_fault_telemetry_spool_cluster_depth",
    # collector census (per-process snapshot of the same agent table)
    "gpu_fault_collector_silent_nodes",
    "gpu_fault_collector_erroring_nodes",
    # Aurora credential refresher status file (one file per Pod)
    "gpu_fault_aurora_credential_refresh_status_unreadable",
    "gpu_fault_aurora_credential_refresh_last_run_age_seconds",
    "gpu_fault_aurora_credential_refresh_last_run_ok",
    "gpu_fault_aurora_credential_refresh_last_success_age_seconds",
    # fleet-level contributors (worker role only)
    "gpu_fault_remote_command_total",
    "gpu_fault_remote_command_by_cluster_total",
    "gpu_fault_remote_command_oldest_unclaimed_seconds",
    "gpu_fault_remote_command_executor_internal_errors_total",
    "gpu_fault_remote_command_executor_internal_error_last_seen_timestamp_seconds",
    "gpu_fault_remote_command_unclaimed_expired",
    "gpu_fault_fleet_rollout_fence_age_seconds",
    "gpu_fault_fleet_rollout_never_started",
    "gpu_fault_fleet_pin_drift_nodes",
    "gpu_fault_workflow_total",
    "gpu_fault_workflow_step_total",
    "gpu_fault_workflow_scan_limit",
    "gpu_fault_workflow_scan_window_seconds",
    "gpu_fault_workflow_scan_size",
    "gpu_fault_workflow_scan_truncated",
    "gpu_fault_workflow_duration_seconds",
    "gpu_fault_workflow_overdue_seconds",
    "gpu_fault_workflow_step_waiting_seconds",
    "gpu_fault_workflow_step_waiting_warning_seconds",
    "gpu_fault_workflow_blocked_unreconciled",
    "gpu_fault_orphan_workflows",
    "gpu_fault_incident_dangling_workflow_pointers",
    "gpu_fault_remediation_budget_active_claims",
    "gpu_fault_remediation_budget_cluster_active_claims",
    "gpu_fault_remediation_budget_cluster_limit",
    "gpu_fault_remediation_budget_cluster_waiting_workflows",
    "gpu_fault_remediation_budget_wait_total",
    "gpu_fault_remediation_budget_waiting_workflows",
    "gpu_fault_remediation_budget_waiting_workflows_by_scope",
    "gpu_fault_ambiguous_attempt_ownership_current",
    "gpu_fault_attempt_observation_scan_limit",
    "gpu_fault_attempt_observation_scan_size",
    "gpu_fault_attempt_observation_scan_truncated",
    "gpu_fault_stale_attempt_observations",
    "gpu_fault_stale_agents",
    "gpu_fault_notification_total",
    "gpu_fault_notification_delivery_total",
    "gpu_fault_notification_outbox_depth",
    "gpu_fault_notification_oldest_pending_age_seconds",
    "gpu_fault_closed_loop_milestone_seconds",
    "gpu_fault_completion_decisions",
    "gpu_fault_completion_events_without_decision",
    "gpu_fault_incidents_by_state",
)

# Timestamps (newest event anywhere in the Pod), ages and durations read as
# the worst process, high-water marks, and "any process is in this state".
_MAX = (
    "gpu_fault_notification_delivery_error_last_seen_timestamp_seconds",
    "gpu_fault_notification_dispatch_last_cycle_timestamp_seconds",
    "gpu_fault_notification_expired_last_seen_timestamp_seconds",
    "gpu_fault_periodic_job_error_last_seen_timestamp_seconds",
    "gpu_fault_periodic_job_last_run_timestamp_seconds",
    "gpu_fault_periodic_last_cycle_timestamp_seconds",
    "gpu_fault_periodic_lease_error_last_seen_timestamp_seconds",
    "gpu_fault_processor_claim_last_round_timestamp_seconds",
    "gpu_fault_telemetry_spool_consumer_last_cycle_timestamp_seconds",
    "gpu_fault_workflow_dispatch_failure_handling_abandoned_last_seen_timestamp_seconds",
    "gpu_fault_workflow_dispatch_internal_error_last_seen_timestamp_seconds",
    "gpu_fault_workflow_dispatch_last_cycle_timestamp_seconds",
    "gpu_fault_workflow_dispatch_wakeup_last_seen_timestamp_seconds",
    "gpu_fault_processor_claim_backoff_seconds",
    "gpu_fault_processor_claim_seconds_max",
    "gpu_fault_processor_oldest_in_flight_seconds",
    "gpu_fault_processor_phase_oldest_seconds",
    "gpu_fault_processor_replay_inbound_phase_oldest_seconds",
    "gpu_fault_processor_retry_delay_seconds_max",
    "gpu_fault_processor_admission_batch_pending_max",
    "gpu_fault_processor_admission_batch_in_flight_max",
    "gpu_fault_processor_admission_batch_round_seconds",
    "gpu_fault_processor_admission_batch_scope_flush_seconds_max",
    "gpu_fault_processor_admission_batch_scope_in_flight_seconds",
    "gpu_fault_processor_admission_batch_scope_wait_seconds",
    "gpu_fault_processor_consumer_last_cycle_age_seconds",
    "gpu_fault_processor_fault_pressure_active",
    "gpu_fault_processor_fault_rows_blocked_by_observation",
    "gpu_fault_processor_counter_drift_abs",
    "gpu_fault_processor_counter_mismatched_clusters",
    "gpu_fault_telemetry_spool_in_flight_bytes_max",
    "gpu_fault_telemetry_spool_fault_pressure_active",
    "gpu_fault_telemetry_spool_fault_backlog_depth",
    "gpu_fault_workflow_pending_age_seconds_max",
    "gpu_fault_workflow_retired_generation_awaiting_operator",
    "gpu_fault_collector_last_success_age_seconds_max",
    "gpu_fault_regional_registry_refresh_error",
    "gpu_fault_regional_registry_secret_drift",
    "gpu_fault_spare_reservations_active",
)

# "Healthy/running/connected" flags: any unhealthy process makes the Pod
# unhealthy. Freshness ages where the Pod is fresh if any process is, and
# throttled ceilings where the most throttled process is the Pod's.
_MIN = (
    "gpu_fault_processor_healthy",
    "gpu_fault_processor_consumer_running",
    "gpu_fault_processor_notifications_enabled",
    "gpu_fault_processor_notification_listener_connected",
    "gpu_fault_workflow_dispatch_wakeup_listener_connected",
    "gpu_fault_telemetry_spool_consumer_running",
    "gpu_fault_telemetry_spool_notifications_enabled",
    "gpu_fault_collector_metrics_snapshot_age_seconds",
    "gpu_fault_postgres_credential_source_file",
    "gpu_fault_processor_fault_pressure_evidence_workers",
    "gpu_fault_telemetry_spool_fault_pressure_workers",
)

# A fact about one process that no arithmetic combines.
_PER_PROCESS = ("gpu_fault_processor_notification_shard",)

# Counters, summaries/histograms, and gauges that add up across processes.
_SUM = (
    "gpu_fault_ambiguous_attempt_ownership_total",
    "gpu_fault_control_record_archive_archived_total",
    "gpu_fault_control_record_archive_errors_total",
    "gpu_fault_control_record_archive_withheld_total",
    "gpu_fault_event_loop_lag_seconds",
    "gpu_fault_gpu_findings_without_incident_total",
    "gpu_fault_hardware_escalation_chain_terminated_total",
    "gpu_fault_hardware_escalation_containment_refused_total",
    "gpu_fault_health_signal_clock_regressions_total",
    "gpu_fault_incident_auto_closed_by_restore_total",
    "gpu_fault_incident_operator_closed_total",
    "gpu_fault_ingest_stale_event_link_repairs_total",
    "gpu_fault_ingest_unresolved_fault_signals_total",
    "gpu_fault_ingress_backpressure_rejections_total",
    "gpu_fault_ingress_decode_rejections_total",
    "gpu_fault_ingress_lane_in_flight",
    "gpu_fault_ingress_lane_rejections_total",
    "gpu_fault_ingress_lane_wait_seconds",
    "gpu_fault_ingress_lane_workers",
    "gpu_fault_metrics_contributor_errors_total",
    "gpu_fault_notification_dead_lettered_total",
    "gpu_fault_notification_delivery_errors_total",
    "gpu_fault_notification_expired_total",
    "gpu_fault_notification_suppressed_drills_total",
    "gpu_fault_periodic_cleanup_budget_exhausted_total",
    "gpu_fault_periodic_cleanup_job_errors_total",
    "gpu_fault_periodic_cleanup_rows_total",
    "gpu_fault_periodic_job_errors_total",
    "gpu_fault_periodic_lease_errors_total",
    "gpu_fault_policy_unknown_product_total",
    "gpu_fault_postgres_pool_checkout_wait_seconds",
    "gpu_fault_postgres_pool_size",
    "gpu_fault_postgres_pool_available",
    "gpu_fault_postgres_pool_requests_waiting",
    "gpu_fault_postgres_pool_requests_errors_total",
    "gpu_fault_postgres_pool_connections_errors_total",
    "gpu_fault_postgres_pool_connections_lost_total",
    "gpu_fault_postgres_credential_rotations_total",
    "gpu_fault_postgres_credential_read_failures_total",
    "gpu_fault_postgres_credential_authentication_failures_total",
    "gpu_fault_postgres_credential_reconnect_failures_total",
    "gpu_fault_processor_active_consumer",
    "gpu_fault_processor_admission_batch_cap_waits_total",
    "gpu_fault_processor_admission_batch_deferred_total",
    "gpu_fault_processor_admission_batch_expired_total",
    "gpu_fault_processor_admission_batch_flush_seconds",
    "gpu_fault_processor_admission_batch_groups_total",
    "gpu_fault_processor_admission_batch_in_flight",
    "gpu_fault_processor_admission_batch_items_total",
    "gpu_fault_processor_admission_batch_pending",
    "gpu_fault_processor_admission_batch_rounds_total",
    "gpu_fault_processor_admission_batch_scope_busy_defers_total",
    "gpu_fault_processor_admission_batch_scope_pending",
    "gpu_fault_processor_admission_batch_scopes_pending",
    "gpu_fault_processor_admission_batch_shed_total",
    "gpu_fault_processor_admission_batch_submitted_total",
    "gpu_fault_processor_admission_batch_wait_seconds",
    "gpu_fault_processor_admission_rejections_by_cluster_total",
    "gpu_fault_processor_admission_rejections_by_path_total",
    "gpu_fault_processor_admission_rejections_total",
    "gpu_fault_processor_claim_backoff_skips_total",
    "gpu_fault_processor_claim_empty_by_stream_total",
    "gpu_fault_processor_claim_empty_total",
    "gpu_fault_processor_claim_lane_blocked_total",
    "gpu_fault_processor_claim_probes_total",
    "gpu_fault_processor_claim_rounds_by_stream_total",
    "gpu_fault_processor_claim_rounds_total",
    "gpu_fault_processor_claim_rows_by_stream_total",
    "gpu_fault_processor_claim_rows_total",
    "gpu_fault_processor_claim_seconds_sum",
    "gpu_fault_processor_claimed_not_started",
    "gpu_fault_processor_claimed_not_started_released_total",
    "gpu_fault_processor_completion_failure_releases_total",
    "gpu_fault_processor_completion_failures_total",
    "gpu_fault_processor_completion_retries_total",
    "gpu_fault_processor_completions_by_path_status_total",
    "gpu_fault_processor_consumer_cycle_errors_total",
    "gpu_fault_processor_consumer_cycles_total",
    "gpu_fault_processor_deadline_exceeded_total",
    "gpu_fault_processor_evidence_admission_batch_expired_total",
    "gpu_fault_processor_evidence_admission_batch_flush_seconds",
    "gpu_fault_processor_evidence_admission_batch_in_flight",
    "gpu_fault_processor_evidence_admission_batch_items_total",
    "gpu_fault_processor_evidence_admission_batch_pending",
    "gpu_fault_processor_evidence_admission_batch_shed_total",
    "gpu_fault_processor_evidence_admission_batch_submitted_total",
    "gpu_fault_processor_evidence_admission_batch_wait_seconds",
    "gpu_fault_processor_expired_leases_reclaimed_total",
    "gpu_fault_processor_fault_admission_batch_expired_total",
    "gpu_fault_processor_fault_admission_batch_flush_seconds",
    "gpu_fault_processor_fault_admission_batch_in_flight",
    "gpu_fault_processor_fault_admission_batch_items_total",
    "gpu_fault_processor_fault_admission_batch_pending",
    "gpu_fault_processor_fault_admission_batch_shed_total",
    "gpu_fault_processor_fault_admission_batch_submitted_total",
    "gpu_fault_processor_fault_admission_batch_wait_seconds",
    "gpu_fault_processor_fault_pressure_activations_total",
    "gpu_fault_processor_fault_rejections_total",
    "gpu_fault_processor_fault_rows_skipped_by_observation_total",
    "gpu_fault_processor_in_flight",
    "gpu_fault_processor_in_flight_by_phase",
    "gpu_fault_processor_interlock_probes_total",
    "gpu_fault_processor_lane_holder_seconds",
    "gpu_fault_processor_lane_wait_seconds",
    "gpu_fault_processor_notification_reconnects_total",
    "gpu_fault_processor_notification_shardless_episodes_total",
    "gpu_fault_processor_notifications_filtered_total",
    "gpu_fault_processor_notifications_received_total",
    "gpu_fault_processor_oversize_rejections_total",
    "gpu_fault_processor_queue_bypass_total",
    "gpu_fault_processor_renewal_errors_total",
    "gpu_fault_processor_renewal_fenced_total",
    "gpu_fault_processor_replay_inbound_by_phase",
    "gpu_fault_processor_request_processing_seconds",
    "gpu_fault_processor_requests_processed_total",
    "gpu_fault_processor_retry_horizon_failures_total",
    "gpu_fault_processor_retry_rescheduled_by_path_total",
    "gpu_fault_processor_retry_rescheduled_total",
    "gpu_fault_processor_stale_superseded_by_path_total",
    "gpu_fault_processor_stale_superseded_total",
    "gpu_fault_processor_telemetry_coalesced_total",
    "gpu_fault_processor_workers",
    "gpu_fault_remote_command_batched_commands_total",
    "gpu_fault_remote_command_batched_steps_total",
    "gpu_fault_remote_command_open_sibling_holds_total",
    "gpu_fault_request_decode_admission_wait_seconds",
    "gpu_fault_request_decode_in_flight",
    "gpu_fault_request_decode_max_in_flight",
    "gpu_fault_request_decode_rejections_total",
    "gpu_fault_spare_reservations_reclaimed_total",
    "gpu_fault_store_io_admission_wait_seconds",
    "gpu_fault_store_io_in_flight",
    "gpu_fault_store_io_max_in_flight",
    "gpu_fault_store_io_rejections_total",
    "gpu_fault_telemetry_spool_abandoned_total",
    "gpu_fault_telemetry_spool_admission_batch_seconds",
    "gpu_fault_telemetry_spool_admission_items_total",
    "gpu_fault_telemetry_spool_admission_pending",
    "gpu_fault_telemetry_spool_admission_shed_total",
    "gpu_fault_telemetry_spool_admitted_by_path_total",
    "gpu_fault_telemetry_spool_admitted_total",
    "gpu_fault_telemetry_spool_claim_rounds_total",
    "gpu_fault_telemetry_spool_claim_rows_by_path_total",
    "gpu_fault_telemetry_spool_claim_rows_total",
    "gpu_fault_telemetry_spool_coalesced_total",
    "gpu_fault_telemetry_spool_dropped_total",
    "gpu_fault_telemetry_spool_errors_total",
    "gpu_fault_telemetry_spool_fallback_polls_total",
    "gpu_fault_telemetry_spool_in_flight_bytes",
    "gpu_fault_telemetry_spool_in_flight_bytes_limit",
    "gpu_fault_telemetry_spool_notification_reconnects_total",
    "gpu_fault_telemetry_spool_notifications_received_total",
    "gpu_fault_telemetry_spool_rejected_total",
    "gpu_fault_telemetry_spool_released_total",
    "gpu_fault_telemetry_spool_replay_seconds",
    "gpu_fault_telemetry_spool_replay_transport_total",
    "gpu_fault_telemetry_spool_replayed_total",
    "gpu_fault_telemetry_spool_stale_completed_total",
    "gpu_fault_telemetry_spool_superseded_total",
    "gpu_fault_workflow_branch_escalation_budget_refusals_total",
    "gpu_fault_workflow_dispatch_deferred_total",
    "gpu_fault_workflow_dispatch_failure_handling_abandoned_total",
    "gpu_fault_workflow_dispatch_filtered_total",
    "gpu_fault_workflow_dispatch_internal_errors_total",
    "gpu_fault_workflow_dispatch_node_busy_timeouts_total",
    "gpu_fault_workflow_dispatch_plan_sync_misses_total",
    "gpu_fault_workflow_dispatch_preemption_pending_seen_total",
    "gpu_fault_workflow_dispatch_sweep_errors_total",
    "gpu_fault_workflow_dispatch_wakeups_total",
    "gpu_fault_workflow_lifetime_exceeded_total",
    "gpu_fault_workflow_merge_record_only_total",
    "gpu_fault_workflow_pending_age_warnings_total",
    "gpu_fault_workflow_placement_holds_dissolved_total",
    "gpu_fault_workflow_placement_holds_failed_total",
    "gpu_fault_workflow_placement_holds_opened_total",
)


def _build() -> dict[str, Strategy]:
    table: dict[str, Strategy] = dict(MERGER_FAMILIES)
    for names, strategy in (
        (_ANY, ANY),
        (_MAX, MAX),
        (_MIN, MIN),
        (_PER_PROCESS, PER_PROCESS),
        (_SUM, SUM),
    ):
        for name in names:
            if name in table:
                raise RuntimeError(f"metric family {name} listed twice")
            table[name] = strategy
    return table


STRATEGIES: dict[str, Strategy] = _build()


def register(name: str, strategy: Strategy) -> None:
    """Declare a family's strategy (plugin contributors call this at import)."""

    if not name:
        raise ValueError("metric family name is required")
    existing = STRATEGIES.get(name)
    if existing is not None and existing is not strategy:
        raise RuntimeError(
            f"metric family {name} already registered as {existing.value}, "
            f"not {strategy.value}"
        )
    STRATEGIES[name] = strategy


def strategy_for(name: str) -> Strategy | None:
    return STRATEGIES.get(name)
