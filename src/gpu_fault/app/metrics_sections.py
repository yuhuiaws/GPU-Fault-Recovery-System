from __future__ import annotations

import os


def metrics_label_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


_metrics_label_value = metrics_label_value


def render_processor_metrics_1(
    lines,
    runtime,
    claim,
    processor_worker_threads,
    processor_admission_batcher,
    processor_oversize_rejections,
    processor_telemetry_coalesced,
    store_io,
    decode_io,
) -> None:
    lines.extend(
        [
            "# HELP gpu_fault_processor_workers Processor worker threads on this replica; 0 where the queue consumer does not run.",
            "# TYPE gpu_fault_processor_workers gauge",
            f"gpu_fault_processor_workers {processor_worker_threads}",
            "# HELP gpu_fault_processor_active_consumer Whether this replica actively consumes the queue.",
            "# TYPE gpu_fault_processor_active_consumer gauge",
            f"gpu_fault_processor_active_consumer {runtime['active_consumer']}",
            "# HELP gpu_fault_processor_notifications_enabled Whether PostgreSQL LISTEN wakeups are connected.",
            "# TYPE gpu_fault_processor_notifications_enabled gauge",
            f"gpu_fault_processor_notifications_enabled {runtime['notifications_enabled']}",
            "# HELP gpu_fault_processor_notifications_received_total Coalesced queue notification batches received.",
            "# TYPE gpu_fault_processor_notifications_received_total counter",
            f"gpu_fault_processor_notifications_received_total {runtime['notifications_received_total']}",
            "# HELP gpu_fault_processor_notifications_filtered_total Notifications ignored by non-owning shards.",
            "# TYPE gpu_fault_processor_notifications_filtered_total counter",
            f"gpu_fault_processor_notifications_filtered_total {runtime['notifications_filtered_total']}",
            "# HELP gpu_fault_processor_notification_reconnects_total LISTEN connections lost after becoming ready.",
            "# TYPE gpu_fault_processor_notification_reconnects_total counter",
            f"gpu_fault_processor_notification_reconnects_total {runtime['notification_reconnects_total']}",
            "# HELP gpu_fault_processor_notification_shard Advisory-lock notification shard owned by this process.",
            "# TYPE gpu_fault_processor_notification_shard gauge",
            f"gpu_fault_processor_notification_shard {runtime['notification_shard']}",
            "# HELP gpu_fault_processor_notification_shard_count Configured notification shard count.",
            "# TYPE gpu_fault_processor_notification_shard_count gauge",
            f"gpu_fault_processor_notification_shard_count {runtime['notification_shard_count']}",
            "# HELP gpu_fault_processor_in_flight Processor replay calls currently executing.",
            "# TYPE gpu_fault_processor_in_flight gauge",
            f"gpu_fault_processor_in_flight {runtime['in_flight']}",
            "# HELP gpu_fault_processor_claimed_not_started Claimed processor requests waiting for an execution worker.",
            "# TYPE gpu_fault_processor_claimed_not_started gauge",
            f"gpu_fault_processor_claimed_not_started {runtime.get('claimed_not_started', 0)}",
            "# HELP gpu_fault_processor_claimed_not_started_released_total Claimed requests released during graceful processor shutdown before execution began.",
            "# TYPE gpu_fault_processor_claimed_not_started_released_total counter",
            "gpu_fault_processor_claimed_not_started_released_total "
            f"{runtime.get('claimed_not_started_released_total', 0)}",
            "# HELP gpu_fault_processor_oldest_in_flight_seconds Elapsed time of the oldest executing replay call.",
            "# TYPE gpu_fault_processor_oldest_in_flight_seconds gauge",
            f"gpu_fault_processor_oldest_in_flight_seconds {runtime['oldest_in_flight_seconds']:.6f}",
            "# HELP gpu_fault_processor_claim_rounds_total Claim queries issued by this replica.",
            "# TYPE gpu_fault_processor_claim_rounds_total counter",
            f"gpu_fault_processor_claim_rounds_total {claim.get('rounds', 0)}",
            "# HELP gpu_fault_processor_claim_rows_total Requests returned by claim queries.",
            "# TYPE gpu_fault_processor_claim_rows_total counter",
            f"gpu_fault_processor_claim_rows_total {claim.get('rows', 0)}",
            "# HELP gpu_fault_processor_claim_empty_total Claim queries that returned nothing.",
            "# TYPE gpu_fault_processor_claim_empty_total counter",
            f"gpu_fault_processor_claim_empty_total {claim.get('empty', 0)}",
            "# HELP gpu_fault_processor_claim_backoff_skips_total Claim rounds skipped by the idle backoff.",
            "# TYPE gpu_fault_processor_claim_backoff_skips_total counter",
            f"gpu_fault_processor_claim_backoff_skips_total {claim.get('backoff_skips', 0)}",
            "# HELP gpu_fault_processor_claim_probes_total Lane-block probes issued before backing off.",
            "# TYPE gpu_fault_processor_claim_probes_total counter",
            f"gpu_fault_processor_claim_probes_total {claim.get('probes', 0)}",
            "# HELP gpu_fault_processor_claim_lane_blocked_total Empty claims explained by lanes leased elsewhere.",
            "# TYPE gpu_fault_processor_claim_lane_blocked_total counter",
            f"gpu_fault_processor_claim_lane_blocked_total {claim.get('lane_blocked', 0)}",
            "# HELP gpu_fault_processor_claim_seconds_sum Time spent inside claim queries.",
            "# TYPE gpu_fault_processor_claim_seconds_sum counter",
            f"gpu_fault_processor_claim_seconds_sum {claim.get('seconds_sum', 0.0):.6f}",
            "# HELP gpu_fault_processor_claim_seconds_max Slowest claim query on this replica.",
            "# TYPE gpu_fault_processor_claim_seconds_max gauge",
            f"gpu_fault_processor_claim_seconds_max {claim.get('seconds_max', 0.0):.6f}",
            "# HELP gpu_fault_processor_claim_backoff_seconds Current largest per-stream claim backoff interval.",
            "# TYPE gpu_fault_processor_claim_backoff_seconds gauge",
            f"gpu_fault_processor_claim_backoff_seconds {claim.get('backoff_seconds', 0.0):.6f}",
            "# HELP gpu_fault_processor_healthy Whether this replica can claim processor requests.",
            "# TYPE gpu_fault_processor_healthy gauge",
            f"gpu_fault_processor_healthy {runtime['healthy']}",
        ]
    )


def render_processor_metrics_2(
    lines,
    runtime,
    claim,
    processor_worker_threads,
    processor_admission_batcher,
    processor_oversize_rejections,
    processor_telemetry_coalesced,
    store_io,
    decode_io,
) -> None:
    lines.extend(
        [
            "# HELP gpu_fault_processor_deadline_exceeded_total Replay calls that exceeded the hard execution deadline.",
            "# TYPE gpu_fault_processor_deadline_exceeded_total counter",
            f"gpu_fault_processor_deadline_exceeded_total {runtime['deadline_exceeded_total']}",
            "# HELP gpu_fault_processor_completion_retries_total Retries of a completed handler's queue transition.",
            "# TYPE gpu_fault_processor_completion_retries_total counter",
            f"gpu_fault_processor_completion_retries_total {runtime['completion_retries_total']}",
            "# HELP gpu_fault_processor_completion_failures_total Queue completions released after exhausting retries.",
            "# TYPE gpu_fault_processor_completion_failures_total counter",
            f"gpu_fault_processor_completion_failures_total {runtime['completion_failures_total']}",
            "# HELP gpu_fault_processor_completion_failure_releases_total Claimed requests released back with a retry backoff after a failure (F-D4).",
            "# TYPE gpu_fault_processor_completion_failure_releases_total counter",
            f"gpu_fault_processor_completion_failure_releases_total {runtime.get('completion_failure_releases_total', 0)}",
            "# HELP gpu_fault_processor_retry_horizon_failures_total Requests completed as failed because they kept failing past the retry horizon (F-D4).",
            "# TYPE gpu_fault_processor_retry_horizon_failures_total counter",
            f"gpu_fault_processor_retry_horizon_failures_total {runtime.get('retry_horizon_failures_total', 0)}",
            "# HELP gpu_fault_processor_renewal_errors_total Lease renewals that hit a store error and were retried (F-D5).",
            "# TYPE gpu_fault_processor_renewal_errors_total counter",
            f"gpu_fault_processor_renewal_errors_total {runtime.get('renewal_errors_total', 0)}",
            "# HELP gpu_fault_processor_renewal_fenced_total Lease renewals refused because another owner holds the lane (F-D5).",
            "# TYPE gpu_fault_processor_renewal_fenced_total counter",
            f"gpu_fault_processor_renewal_fenced_total {runtime.get('renewal_fenced_total', 0)}",
            "# HELP gpu_fault_processor_fault_rows_skipped_by_observation_total Fault rows a claim round left behind because an incomplete observation on their scope held them back (F-D3).",
            "# TYPE gpu_fault_processor_fault_rows_skipped_by_observation_total counter",
            f"gpu_fault_processor_fault_rows_skipped_by_observation_total {runtime.get('fault_rows_skipped_by_observation_total', 0)}",
            "# HELP gpu_fault_processor_fault_rows_blocked_by_observation Fault rows held back by an observation at the last interlock probe (F-D3).",
            "# TYPE gpu_fault_processor_fault_rows_blocked_by_observation gauge",
            f"gpu_fault_processor_fault_rows_blocked_by_observation {runtime.get('fault_rows_blocked_by_observation', 0)}",
            "# HELP gpu_fault_processor_interlock_probes_total Times the fault stream asked the store how many rows the observation interlock holds (F-D3).",
            "# TYPE gpu_fault_processor_interlock_probes_total counter",
            f"gpu_fault_processor_interlock_probes_total {runtime.get('interlock_probes_total', 0)}",
            "# HELP gpu_fault_processor_notification_listener_connected Whether this process holds a NOTIFY listener connection, shard or not (F-D11).",
            "# TYPE gpu_fault_processor_notification_listener_connected gauge",
            f"gpu_fault_processor_notification_listener_connected {int(bool(runtime.get('notification_listener_connected', 0)))}",
            "# HELP gpu_fault_processor_notification_shardless_episodes_total Times this process connected its listener but won no shard and fell back to polling (F-D11).",
            "# TYPE gpu_fault_processor_notification_shardless_episodes_total counter",
            f"gpu_fault_processor_notification_shardless_episodes_total {runtime.get('notification_shardless_episodes_total', 0)}",
            "# HELP gpu_fault_processor_retry_rescheduled_total Retryable responses rescheduled with a future not-before time.",
            "# TYPE gpu_fault_processor_retry_rescheduled_total counter",
            f"gpu_fault_processor_retry_rescheduled_total {runtime.get('retry_rescheduled_total', 0)}",
            "# HELP gpu_fault_processor_retry_delay_seconds_max Longest retry delay scheduled by this processor.",
            "# TYPE gpu_fault_processor_retry_delay_seconds_max gauge",
            f"gpu_fault_processor_retry_delay_seconds_max {runtime.get('retry_delay_seconds_max', 0.0):.6f}",
            "# HELP gpu_fault_processor_retry_rescheduled_by_path_total Retryable responses rescheduled by normalized request path.",
            "# TYPE gpu_fault_processor_retry_rescheduled_by_path_total counter",
            "# HELP gpu_fault_processor_stale_superseded_total Stale periodic context requests completed without executing their business handler.",
            "# TYPE gpu_fault_processor_stale_superseded_total counter",
            f"gpu_fault_processor_stale_superseded_total {runtime['stale_superseded_total']}",
            "# HELP gpu_fault_processor_telemetry_coalesced_total Pending telemetry requests replaced by a newer batch.",
            "# TYPE gpu_fault_processor_telemetry_coalesced_total counter",
            f"gpu_fault_processor_telemetry_coalesced_total {processor_telemetry_coalesced}",
            "# HELP gpu_fault_processor_oversize_rejections_total Processor requests rejected before queue persistence.",
            "# TYPE gpu_fault_processor_oversize_rejections_total counter",
            f"gpu_fault_processor_oversize_rejections_total {processor_oversize_rejections}",
            "# HELP gpu_fault_processor_admission_rejections_total Processor requests rejected before enqueue.",
            "# TYPE gpu_fault_processor_admission_rejections_total counter",
            "# HELP gpu_fault_processor_admission_rejections_by_cluster_total Processor requests rejected by a per-cluster queue limit.",
            "# TYPE gpu_fault_processor_admission_rejections_by_cluster_total counter",
            "# HELP gpu_fault_ingress_backpressure_rejections_total Ingress requests rejected before decode/enqueue.",
            "# TYPE gpu_fault_ingress_backpressure_rejections_total counter",
            "# HELP gpu_fault_store_io_in_flight Store calls executing or waiting for a worker.",
            "# TYPE gpu_fault_store_io_in_flight gauge",
            f"gpu_fault_store_io_in_flight {store_io.in_flight}",
            "# HELP gpu_fault_store_io_max_in_flight Configured per-process Store I/O admission capacity.",
            "# TYPE gpu_fault_store_io_max_in_flight gauge",
            f"gpu_fault_store_io_max_in_flight {store_io.max_in_flight}",
            "# HELP gpu_fault_store_io_rejections_total Store calls rejected because I/O capacity was exhausted.",
            "# TYPE gpu_fault_store_io_rejections_total counter",
            "gpu_fault_store_io_rejections_total"
            f'{{process_id="{os.getpid()}"}} {store_io.rejected_total}',
            "# HELP gpu_fault_store_io_rejections_by_reason_total Store calls rejected, by reason: the caller's deadline passed, capacity was exhausted, or the writer was unavailable (F-E1).",
            "# TYPE gpu_fault_store_io_rejections_by_reason_total counter",
            *(
                f'gpu_fault_store_io_rejections_by_reason_total{{reason="{reason}"}} {count}'
                for reason, count in sorted(
                    getattr(store_io, "rejected_by_reason", {}).items()
                )
            ),
            "# HELP gpu_fault_store_io_admission_wait_seconds Time waiting for a Store I/O admission slot.",
            "# TYPE gpu_fault_store_io_admission_wait_seconds summary",
            f"gpu_fault_store_io_admission_wait_seconds_sum {store_io.admission_wait_sum_seconds:.6f}",
            f"gpu_fault_store_io_admission_wait_seconds_count {store_io.admission_wait_count}",
            f"gpu_fault_store_io_admission_wait_seconds_max {store_io.admission_wait_max_seconds:.6f}",
            "# HELP gpu_fault_request_decode_in_flight Request decodes executing or waiting for a worker.",
            "# TYPE gpu_fault_request_decode_in_flight gauge",
            f"gpu_fault_request_decode_in_flight {decode_io.in_flight}",
            "# HELP gpu_fault_request_decode_max_in_flight Configured per-process request decode capacity.",
            "# TYPE gpu_fault_request_decode_max_in_flight gauge",
            f"gpu_fault_request_decode_max_in_flight {decode_io.max_in_flight}",
            "# HELP gpu_fault_request_decode_rejections_total Requests rejected because decode capacity was exhausted.",
            "# TYPE gpu_fault_request_decode_rejections_total counter",
            f"gpu_fault_request_decode_rejections_total {decode_io.rejected_total}",
            "# HELP gpu_fault_request_decode_admission_wait_seconds Time waiting for a request decode slot.",
            "# TYPE gpu_fault_request_decode_admission_wait_seconds summary",
            f"gpu_fault_request_decode_admission_wait_seconds_sum {decode_io.admission_wait_sum_seconds:.6f}",
            f"gpu_fault_request_decode_admission_wait_seconds_count {decode_io.admission_wait_count}",
            f"gpu_fault_request_decode_admission_wait_seconds_max {decode_io.admission_wait_max_seconds:.6f}",
            "# HELP gpu_fault_processor_admission_batch_pending Telemetry requests waiting for an admission batch.",
            "# TYPE gpu_fault_processor_admission_batch_pending gauge",
            f"gpu_fault_processor_admission_batch_pending {processor_admission_batcher.pending_depth}",
            "# HELP gpu_fault_processor_admission_batch_pending_max High-water mark of the admission batch queue.",
            "# TYPE gpu_fault_processor_admission_batch_pending_max gauge",
            f"gpu_fault_processor_admission_batch_pending_max {processor_admission_batcher.pending_depth_max}",
            "# HELP gpu_fault_processor_admission_batch_submitted_total Telemetry requests handed to the admission batcher.",
            "# TYPE gpu_fault_processor_admission_batch_submitted_total counter",
            f"gpu_fault_processor_admission_batch_submitted_total {processor_admission_batcher.submitted_total}",
            "# HELP gpu_fault_processor_admission_batch_shed_total Requests rejected on arrival because the projected batch wait exceeded their deadline.",
        ]
    )
    for path, count in sorted(runtime.get("retry_rescheduled_by_path", {}).items()):
        lines.append(
            "gpu_fault_processor_retry_rescheduled_by_path_total"
            f'{{path="{_metrics_label_value(path)}"}} {count}'
        )
    lines.extend(
        [
            "# HELP gpu_fault_processor_lane_holder_seconds Time processor requests held an ordering lane, by normalized path.",
            "# TYPE gpu_fault_processor_lane_holder_seconds summary",
        ]
    )
    for path, values in sorted(runtime.get("lane_holder_by_path", {}).items()):
        escaped_path = _metrics_label_value(path)
        lines.append(
            "gpu_fault_processor_lane_holder_seconds_count"
            f'{{path="{escaped_path}"}} {values["count"]}'
        )
        lines.append(
            "gpu_fault_processor_lane_holder_seconds_sum"
            f'{{path="{escaped_path}"}} {values["sum"]:.6f}'
        )
        lines.append(
            "gpu_fault_processor_lane_holder_seconds_max"
            f'{{path="{escaped_path}"}} {values["max"]:.6f}'
        )


def render_processor_metrics_3(
    lines,
    runtime,
    claim,
    processor_worker_threads,
    processor_admission_batcher,
    processor_oversize_rejections,
    processor_telemetry_coalesced,
    store_io,
    decode_io,
) -> None:
    lines.extend(
        [
            "# TYPE gpu_fault_processor_admission_batch_shed_total counter",
            f"gpu_fault_processor_admission_batch_shed_total {processor_admission_batcher.shed_total}",
            "# HELP gpu_fault_processor_admission_batch_round_seconds Exponentially weighted mean flush round duration.",
            "# TYPE gpu_fault_processor_admission_batch_round_seconds gauge",
            f"gpu_fault_processor_admission_batch_round_seconds {processor_admission_batcher.round_seconds_ewma:.6f}",
            "# HELP gpu_fault_processor_admission_batch_expired_total Requests whose budget lapsed while queued for a batch.",
            "# TYPE gpu_fault_processor_admission_batch_expired_total counter",
            f"gpu_fault_processor_admission_batch_expired_total {processor_admission_batcher.expired_total}",
            "# HELP gpu_fault_processor_admission_batch_rounds_total Flush rounds executed by the admission batcher.",
            "# TYPE gpu_fault_processor_admission_batch_rounds_total counter",
            f"gpu_fault_processor_admission_batch_rounds_total {processor_admission_batcher.rounds_total}",
            "# HELP gpu_fault_processor_admission_batch_groups_total Per-cluster groups flushed by the admission batcher.",
            "# TYPE gpu_fault_processor_admission_batch_groups_total counter",
            f"gpu_fault_processor_admission_batch_groups_total {processor_admission_batcher.groups_total}",
            "# HELP gpu_fault_processor_admission_batch_items_total Requests included in a flushed group.",
            "# TYPE gpu_fault_processor_admission_batch_items_total counter",
            f"gpu_fault_processor_admission_batch_items_total {processor_admission_batcher.items_total}",
            "# HELP gpu_fault_processor_admission_batch_deferred_total Requests held back because a round was already full.",
            "# TYPE gpu_fault_processor_admission_batch_deferred_total counter",
            f"gpu_fault_processor_admission_batch_deferred_total {processor_admission_batcher.deferred_total}",
            "# HELP gpu_fault_processor_admission_batch_wait_seconds Time a request waited to be picked into a batch.",
            "# TYPE gpu_fault_processor_admission_batch_wait_seconds summary",
            f"gpu_fault_processor_admission_batch_wait_seconds_sum {processor_admission_batcher.queue_wait_sum_seconds:.6f}",
            f"gpu_fault_processor_admission_batch_wait_seconds_count {processor_admission_batcher.queue_wait_count}",
            f"gpu_fault_processor_admission_batch_wait_seconds_max {processor_admission_batcher.queue_wait_max_seconds:.6f}",
            "# HELP gpu_fault_processor_admission_batch_flush_seconds Store I/O admission plus transaction time per group.",
            "# TYPE gpu_fault_processor_admission_batch_flush_seconds summary",
            f"gpu_fault_processor_admission_batch_flush_seconds_sum {processor_admission_batcher.flush_sum_seconds:.6f}",
            f"gpu_fault_processor_admission_batch_flush_seconds_count {processor_admission_batcher.flush_count}",
            f"gpu_fault_processor_admission_batch_flush_seconds_max {processor_admission_batcher.flush_max_seconds:.6f}",
            "# HELP gpu_fault_processor_admission_batch_in_flight Per-scope admission flushes currently executing.",
            "# TYPE gpu_fault_processor_admission_batch_in_flight gauge",
            f"gpu_fault_processor_admission_batch_in_flight {processor_admission_batcher.in_flight}",
            "# HELP gpu_fault_processor_admission_batch_in_flight_max High-water mark of concurrent admission flushes.",
            "# TYPE gpu_fault_processor_admission_batch_in_flight_max gauge",
            f"gpu_fault_processor_admission_batch_in_flight_max {processor_admission_batcher.in_flight_max}",
            "# HELP gpu_fault_processor_admission_batch_in_flight_limit Configured cap on concurrent admission flushes.",
            "# TYPE gpu_fault_processor_admission_batch_in_flight_limit gauge",
            f"gpu_fault_processor_admission_batch_in_flight_limit {processor_admission_batcher.max_in_flight_flushes}",
            "# HELP gpu_fault_processor_admission_batch_scopes_pending Cluster scopes with entries waiting for a flush.",
            "# TYPE gpu_fault_processor_admission_batch_scopes_pending gauge",
            f"gpu_fault_processor_admission_batch_scopes_pending {processor_admission_batcher.scopes_pending}",
            "# HELP gpu_fault_processor_admission_batch_cap_waits_total Dispatch passes that left work queued because the in-flight cap was full.",
            "# TYPE gpu_fault_processor_admission_batch_cap_waits_total counter",
            f"gpu_fault_processor_admission_batch_cap_waits_total {processor_admission_batcher.cap_waits_total}",
            "# HELP gpu_fault_processor_admission_batch_scope_busy_defers_total Entries left queued because their own cluster was already flushing.",
            "# TYPE gpu_fault_processor_admission_batch_scope_busy_defers_total counter",
            f"gpu_fault_processor_admission_batch_scope_busy_defers_total {processor_admission_batcher.scope_busy_defers_total}",
            "# HELP gpu_fault_processor_admission_batch_scope_flush_seconds_max Longest single flush observed, labelled by the cluster that owned it.",
            "# TYPE gpu_fault_processor_admission_batch_scope_flush_seconds_max gauge",
            "# HELP gpu_fault_processor_admission_batch_scope_in_flight_seconds Age of the currently executing flush per cluster, worst first.",
            "# TYPE gpu_fault_processor_admission_batch_scope_in_flight_seconds gauge",
            "# HELP gpu_fault_processor_admission_batch_scope_pending Entries queued per cluster, oldest queue first.",
            "# TYPE gpu_fault_processor_admission_batch_scope_pending gauge",
            "# HELP gpu_fault_processor_admission_batch_scope_wait_seconds Age of the oldest queued entry per cluster.",
            "# TYPE gpu_fault_processor_admission_batch_scope_wait_seconds gauge",
            "# HELP gpu_fault_processor_requests_processed_total Processor requests completed by outcome.",
            "# TYPE gpu_fault_processor_requests_processed_total counter",
        ]
    )


def render_admission_metrics(
    lines, fault_admission_batcher, evidence_admission_batcher
):
    lines.extend(
        [
            "# HELP "
            "gpu_fault_processor_fault_admission_batch_pending "
            "Priority-zero requests waiting for a cluster batch.",
            "# TYPE gpu_fault_processor_fault_admission_batch_pending gauge",
            "gpu_fault_processor_fault_admission_batch_pending "
            f"{fault_admission_batcher.pending_depth}",
            "# HELP "
            "gpu_fault_processor_fault_admission_batch_submitted_total "
            "Priority-zero requests submitted to the batcher.",
            "# TYPE gpu_fault_processor_fault_admission_batch_submitted_total counter",
            "gpu_fault_processor_fault_admission_batch_submitted_total "
            f"{fault_admission_batcher.submitted_total}",
            "# HELP "
            "gpu_fault_processor_fault_admission_batch_items_total "
            "Priority-zero requests included in flushed batches.",
            "# TYPE gpu_fault_processor_fault_admission_batch_items_total counter",
            "gpu_fault_processor_fault_admission_batch_items_total "
            f"{fault_admission_batcher.items_total}",
            "# HELP "
            "gpu_fault_processor_fault_admission_batch_wait_seconds "
            "Time priority-zero requests waited for a batch.",
            "# TYPE gpu_fault_processor_fault_admission_batch_wait_seconds summary",
            "gpu_fault_processor_fault_admission_batch_wait_seconds_sum "
            f"{fault_admission_batcher.queue_wait_sum_seconds:.6f}",
            "gpu_fault_processor_fault_admission_batch_wait_seconds_count "
            f"{fault_admission_batcher.queue_wait_count}",
            "gpu_fault_processor_fault_admission_batch_wait_seconds_max "
            f"{fault_admission_batcher.queue_wait_max_seconds:.6f}",
            "# HELP "
            "gpu_fault_processor_fault_admission_batch_flush_seconds "
            "Store transaction time for a priority-zero cluster batch.",
            "# TYPE gpu_fault_processor_fault_admission_batch_flush_seconds summary",
            "gpu_fault_processor_fault_admission_batch_flush_seconds_sum "
            f"{fault_admission_batcher.flush_sum_seconds:.6f}",
            "gpu_fault_processor_fault_admission_batch_flush_seconds_count "
            f"{fault_admission_batcher.flush_count}",
            "gpu_fault_processor_fault_admission_batch_flush_seconds_max "
            f"{fault_admission_batcher.flush_max_seconds:.6f}",
            "# HELP "
            "gpu_fault_processor_fault_admission_batch_in_flight "
            "Priority-zero cluster batches currently executing.",
            "# TYPE gpu_fault_processor_fault_admission_batch_in_flight gauge",
            "gpu_fault_processor_fault_admission_batch_in_flight "
            f"{fault_admission_batcher.in_flight}",
            "# HELP "
            "gpu_fault_processor_fault_admission_batch_shed_total "
            "Priority-zero requests rejected by wait projection.",
            "# TYPE gpu_fault_processor_fault_admission_batch_shed_total counter",
            "gpu_fault_processor_fault_admission_batch_shed_total "
            f"{fault_admission_batcher.shed_total}",
            "# HELP "
            "gpu_fault_processor_fault_admission_batch_expired_total "
            "Priority-zero requests whose deadline expired in a batch.",
            "# TYPE gpu_fault_processor_fault_admission_batch_expired_total counter",
            "gpu_fault_processor_fault_admission_batch_expired_total "
            f"{fault_admission_batcher.expired_total}",
        ]
    )
    lines.extend(
        [
            "# HELP "
            "gpu_fault_processor_evidence_admission_batch_pending "
            "Priority-50 requests waiting for a cluster batch.",
            "# TYPE gpu_fault_processor_evidence_admission_batch_pending gauge",
            "gpu_fault_processor_evidence_admission_batch_pending "
            f"{evidence_admission_batcher.pending_depth}",
            "# HELP "
            "gpu_fault_processor_evidence_admission_batch_submitted_total "
            "Priority-50 requests submitted to the batcher.",
            "# TYPE "
            "gpu_fault_processor_evidence_admission_batch_submitted_total "
            "counter",
            "gpu_fault_processor_evidence_admission_batch_submitted_total "
            f"{evidence_admission_batcher.submitted_total}",
            "# HELP "
            "gpu_fault_processor_evidence_admission_batch_items_total "
            "Priority-50 requests included in flushed batches.",
            "# TYPE gpu_fault_processor_evidence_admission_batch_items_total counter",
            "gpu_fault_processor_evidence_admission_batch_items_total "
            f"{evidence_admission_batcher.items_total}",
            "# HELP "
            "gpu_fault_processor_evidence_admission_batch_wait_seconds "
            "Time priority-50 requests waited for a batch.",
            "# TYPE gpu_fault_processor_evidence_admission_batch_wait_seconds summary",
            "gpu_fault_processor_evidence_admission_batch_wait_seconds_sum "
            f"{evidence_admission_batcher.queue_wait_sum_seconds:.6f}",
            "gpu_fault_processor_evidence_admission_batch_wait_seconds_count "
            f"{evidence_admission_batcher.queue_wait_count}",
            "gpu_fault_processor_evidence_admission_batch_wait_seconds_max "
            f"{evidence_admission_batcher.queue_wait_max_seconds:.6f}",
            "# HELP "
            "gpu_fault_processor_evidence_admission_batch_flush_seconds "
            "Store transaction time for a priority-50 cluster batch.",
            "# TYPE gpu_fault_processor_evidence_admission_batch_flush_seconds summary",
            "gpu_fault_processor_evidence_admission_batch_flush_seconds_sum "
            f"{evidence_admission_batcher.flush_sum_seconds:.6f}",
            "gpu_fault_processor_evidence_admission_batch_flush_seconds_count "
            f"{evidence_admission_batcher.flush_count}",
            "gpu_fault_processor_evidence_admission_batch_flush_seconds_max "
            f"{evidence_admission_batcher.flush_max_seconds:.6f}",
            "# HELP "
            "gpu_fault_processor_evidence_admission_batch_in_flight "
            "Priority-50 cluster batches currently executing.",
            "# TYPE gpu_fault_processor_evidence_admission_batch_in_flight gauge",
            "gpu_fault_processor_evidence_admission_batch_in_flight "
            f"{evidence_admission_batcher.in_flight}",
            "# HELP "
            "gpu_fault_processor_evidence_admission_batch_shed_total "
            "Priority-50 requests rejected by wait projection.",
            "# TYPE gpu_fault_processor_evidence_admission_batch_shed_total counter",
            "gpu_fault_processor_evidence_admission_batch_shed_total "
            f"{evidence_admission_batcher.shed_total}",
            "# HELP "
            "gpu_fault_processor_evidence_admission_batch_expired_total "
            "Priority-50 requests whose deadline expired in a batch.",
            "# TYPE gpu_fault_processor_evidence_admission_batch_expired_total counter",
            "gpu_fault_processor_evidence_admission_batch_expired_total "
            f"{evidence_admission_batcher.expired_total}",
        ]
    )


def render_runtime_metrics_one(
    lines,
    runtime,
    claim,
    processor_replay_tracker,
    processor_admission_batcher,
    fault_store_io,
    evidence_store_io,
    fault_decode_io,
    telemetry_spool_store_io,
    processor_admission_rejections,
    ingress_backpressure_rejections,
    processor_admission_rejections_by_path,
    processor_queue_bypasses_by_path,
    processor_queue_bypass_paths,
    processor_queue_bypass_enabled,
    telemetry_spool_enabled,
):
    lines.extend(
        [
            "# HELP gpu_fault_processor_in_flight_by_phase "
            "Executing processor requests by current phase.",
            "# TYPE gpu_fault_processor_in_flight_by_phase gauge",
            "# HELP "
            "gpu_fault_processor_phase_oldest_seconds "
            "Longest elapsed time in a processor phase.",
            "# TYPE gpu_fault_processor_phase_oldest_seconds gauge",
        ]
    )
    for phase, values in sorted(runtime.get("in_flight_by_phase", {}).items()):
        escaped_phase = _metrics_label_value(phase)
        lines.append(
            "gpu_fault_processor_in_flight_by_phase"
            f'{{phase="{escaped_phase}"}} {values["count"]}'
        )
        lines.append(
            "gpu_fault_processor_phase_oldest_seconds"
            f'{{phase="{escaped_phase}"}} '
            f"{values['oldest_seconds']:.6f}"
        )
    inbound_phases = processor_replay_tracker.phases()
    lines.extend(
        [
            "# HELP "
            "gpu_fault_processor_replay_inbound_by_phase "
            "Inbound replay handlers by current phase.",
            "# TYPE gpu_fault_processor_replay_inbound_by_phase gauge",
            "# HELP "
            "gpu_fault_processor_replay_inbound_phase_oldest_seconds "
            "Longest elapsed time in an inbound replay phase.",
            "# TYPE gpu_fault_processor_replay_inbound_phase_oldest_seconds gauge",
        ]
    )
    for phase, values in sorted(inbound_phases.items()):
        escaped_phase = _metrics_label_value(phase)
        lines.append(
            "gpu_fault_processor_replay_inbound_by_phase"
            f'{{phase="{escaped_phase}"}} {values["count"]}'
        )
        lines.append(
            "gpu_fault_processor_replay_inbound_phase_oldest_seconds"
            f'{{phase="{escaped_phase}"}} '
            f"{values['oldest_seconds']:.6f}"
        )
    fault_pressure = runtime.get("fault_pressure", {})
    lines.extend(
        [
            "# HELP gpu_fault_processor_fault_pressure_active "
            "Whether local priority-zero work is throttling "
            "dedicated evidence pools.",
            "# TYPE gpu_fault_processor_fault_pressure_active gauge",
            "gpu_fault_processor_fault_pressure_active "
            f"{fault_pressure.get('active', 0)}",
            "# HELP "
            "gpu_fault_processor_fault_pressure_evidence_workers "
            "Per GPU/Host pool worker ceiling under fault pressure.",
            "# TYPE gpu_fault_processor_fault_pressure_evidence_workers gauge",
            "gpu_fault_processor_fault_pressure_evidence_workers "
            f"{fault_pressure.get('evidence_workers', 0)}",
            "# HELP "
            "gpu_fault_processor_fault_pressure_activations_total "
            "Transitions into local priority-zero pressure.",
            "# TYPE gpu_fault_processor_fault_pressure_activations_total counter",
            "gpu_fault_processor_fault_pressure_activations_total "
            f"{fault_pressure.get('activations_total', 0)}",
        ]
    )
    claim_streams = sorted(
        set(claim.get("rounds_by_stream", {}))
        | set(claim.get("rows_by_stream", {}))
        | set(claim.get("empty_by_stream", {}))
    )
    lines.extend(
        [
            "# HELP "
            "gpu_fault_processor_claim_rounds_by_stream_total "
            "Claim queries issued per logical stream.",
            "# TYPE gpu_fault_processor_claim_rounds_by_stream_total counter",
            "# HELP gpu_fault_processor_claim_rows_by_stream_total "
            "Rows returned per logical claim stream.",
            "# TYPE gpu_fault_processor_claim_rows_by_stream_total counter",
            "# HELP gpu_fault_processor_claim_empty_by_stream_total "
            "Empty claim queries per logical stream.",
            "# TYPE gpu_fault_processor_claim_empty_by_stream_total counter",
        ]
    )
    for stream in claim_streams:
        escaped_stream = _metrics_label_value(stream)
        lines.extend(
            [
                "gpu_fault_processor_claim_rounds_by_stream_total"
                f'{{stream="{escaped_stream}"}} '
                f"{claim.get('rounds_by_stream', {}).get(stream, 0)}",
                "gpu_fault_processor_claim_rows_by_stream_total"
                f'{{stream="{escaped_stream}"}} '
                f"{claim.get('rows_by_stream', {}).get(stream, 0)}",
                "gpu_fault_processor_claim_empty_by_stream_total"
                f'{{stream="{escaped_stream}"}} '
                f"{claim.get('empty_by_stream', {}).get(stream, 0)}",
            ]
        )
    if processor_admission_batcher.scope_flush_max_scope:
        escaped = _metrics_label_value(
            processor_admission_batcher.scope_flush_max_scope
        )
        lines.append(
            "gpu_fault_processor_admission_batch_scope"
            "_flush_seconds_max"
            f'{{cluster_id="{escaped}"}} '
            f"{processor_admission_batcher.scope_flush_max_seconds:.6f}"
        )


def render_runtime_metrics_two(
    lines,
    ctx,
    telemetry_spool_enabled,
    processor_queue_bypass_enabled,
    processor_queue_bypass_paths,
    processor_queue_bypasses_by_path,
):
    for path in sorted(processor_queue_bypass_paths):
        escaped_path = path.replace("\\", "\\\\").replace('"', '\\"')
        lines.append(
            "gpu_fault_processor_queue_bypass_total"
            f'{{path="{escaped_path}"}} '
            f"{processor_queue_bypasses_by_path.get(path, 0)}"
        )
    lines.append(
        "gpu_fault_processor_queue_bypass_enabled "
        f"{1 if processor_queue_bypass_enabled else 0}"
    )
    # Same reasoning: emitted in both arms so the diff has a baseline.
    # The depth is only read from the store when the spool is on -
    # a scrape should not pay for a table nothing is writing to.
    spool = (
        ctx.store.telemetry_spool_stats()
        if telemetry_spool_enabled
        else {
            "depth": 0,
            "leased": 0,
            "oldest_age_seconds": 0.0,
            "by_cluster": {},
            "payload_bytes": 0,
            "leased_bytes": 0,
        }
    )
    return spool


def render_spool_metrics_one(
    lines,
    spool,
    runtime,
    telemetry_spool_enabled,
    telemetry_spool_batcher,
    telemetry_spool_rejections,
    telemetry_spool_admitted_total,
    telemetry_spool_coalesced_total,
):
    spool_runtime = runtime.get("spool", {})
    lines.extend(
        [
            "# HELP gpu_fault_telemetry_spool_enabled "
            "Whether telemetry bypasses the processor queue.",
            "# TYPE gpu_fault_telemetry_spool_enabled gauge",
            f"gpu_fault_telemetry_spool_enabled {1 if telemetry_spool_enabled else 0}",
            "# HELP gpu_fault_telemetry_spool_depth "
            "Telemetry samples waiting to be replayed.",
            "# TYPE gpu_fault_telemetry_spool_depth gauge",
            f"gpu_fault_telemetry_spool_depth {spool['depth']}",
            "# HELP gpu_fault_telemetry_spool_leased "
            "Spooled samples currently claimed by a consumer.",
            "# TYPE gpu_fault_telemetry_spool_leased gauge",
            f"gpu_fault_telemetry_spool_leased {spool['leased']}",
            "# HELP gpu_fault_telemetry_spool_oldest_age_seconds "
            "Age of the oldest spooled telemetry sample.",
            "# TYPE gpu_fault_telemetry_spool_oldest_age_seconds gauge",
            "gpu_fault_telemetry_spool_oldest_age_seconds "
            f"{spool['oldest_age_seconds']:.6f}",
            "# HELP gpu_fault_telemetry_spool_bytes "
            "Serialized JSON bytes currently held by the spool.",
            "# TYPE gpu_fault_telemetry_spool_bytes gauge",
            f"gpu_fault_telemetry_spool_bytes {spool['payload_bytes']}",
            "# HELP gpu_fault_telemetry_spool_leased_bytes "
            "Serialized JSON bytes currently leased.",
            "# TYPE gpu_fault_telemetry_spool_leased_bytes gauge",
            f"gpu_fault_telemetry_spool_leased_bytes {spool['leased_bytes']}",
            "# HELP gpu_fault_telemetry_spool_in_flight_bytes "
            "Payload bytes held by replay futures in this process.",
            "# TYPE gpu_fault_telemetry_spool_in_flight_bytes gauge",
            "gpu_fault_telemetry_spool_in_flight_bytes "
            f"{spool_runtime.get('in_flight_bytes', 0)}",
            "# HELP gpu_fault_telemetry_spool_in_flight_bytes_max "
            "Peak replay payload bytes held by this process.",
            "# TYPE gpu_fault_telemetry_spool_in_flight_bytes_max gauge",
            "gpu_fault_telemetry_spool_in_flight_bytes_max "
            f"{spool_runtime.get('in_flight_bytes_max', 0)}",
            "# HELP "
            "gpu_fault_telemetry_spool_in_flight_bytes_limit "
            "Configured replay payload byte ceiling per process.",
            "# TYPE gpu_fault_telemetry_spool_in_flight_bytes_limit gauge",
            "gpu_fault_telemetry_spool_in_flight_bytes_limit "
            f"{spool_runtime.get('max_in_flight_bytes', 0)}",
            "# HELP "
            "gpu_fault_telemetry_spool_replay_batch_max_items "
            "Maximum samples claimed into one replay transaction.",
            "# TYPE gpu_fault_telemetry_spool_replay_batch_max_items gauge",
            "gpu_fault_telemetry_spool_replay_batch_max_items "
            f"{spool_runtime.get('max_batch_items', 0)}",
            "# HELP gpu_fault_telemetry_spool_consumer_running "
            "Whether this process has a live spool consumer loop.",
            "# TYPE gpu_fault_telemetry_spool_consumer_running gauge",
            "gpu_fault_telemetry_spool_consumer_running "
            f"{spool_runtime.get('consumer_running', 0)}",
            "# HELP "
            "gpu_fault_telemetry_spool_notifications_enabled "
            "Whether the dedicated PostgreSQL listener is connected.",
            "# TYPE gpu_fault_telemetry_spool_notifications_enabled gauge",
            "gpu_fault_telemetry_spool_notifications_enabled "
            f"{spool_runtime.get('notifications_enabled', 0)}",
            "# HELP "
            "gpu_fault_telemetry_spool_notifications_received_total "
            "Wake notifications received by this spool consumer.",
            "# TYPE gpu_fault_telemetry_spool_notifications_received_total counter",
            "gpu_fault_telemetry_spool_notifications_received_total "
            f"{spool_runtime.get('notifications_received', 0)}",
            "# HELP "
            "gpu_fault_telemetry_spool_notification_reconnects_total "
            "Dedicated spool listener reconnect transitions.",
            "# TYPE gpu_fault_telemetry_spool_notification_reconnects_total counter",
            "gpu_fault_telemetry_spool_notification_reconnects_total "
            f"{spool_runtime.get('notification_reconnects', 0)}",
            "# HELP "
            "gpu_fault_telemetry_spool_fallback_polls_total "
            "Low-frequency empty-spool fallback waits.",
            "# TYPE gpu_fault_telemetry_spool_fallback_polls_total counter",
            "gpu_fault_telemetry_spool_fallback_polls_total "
            f"{spool_runtime.get('fallback_polls', 0)}",
            "# HELP "
            "gpu_fault_telemetry_spool_notification_fallback_seconds "
            "Maximum wake delay when a spool notification is missed.",
            "# TYPE gpu_fault_telemetry_spool_notification_fallback_seconds gauge",
            "gpu_fault_telemetry_spool_notification_fallback_seconds "
            f"{spool_runtime.get('notification_fallback_seconds', 0)}",
            "# HELP "
            "gpu_fault_telemetry_spool_fault_pressure_active "
            "Whether priority-zero backlog is throttling replay.",
            "# TYPE gpu_fault_telemetry_spool_fault_pressure_active gauge",
            "gpu_fault_telemetry_spool_fault_pressure_active "
            f"{spool_runtime.get('fault_pressure_active', 0)}",
            "# HELP "
            "gpu_fault_telemetry_spool_fault_backlog_depth "
            "Priority-zero processor requests seen by this consumer.",
            "# TYPE gpu_fault_telemetry_spool_fault_backlog_depth gauge",
            "gpu_fault_telemetry_spool_fault_backlog_depth "
            f"{spool_runtime.get('fault_backlog_depth', 0)}",
            "# HELP "
            "gpu_fault_telemetry_spool_fault_pressure_workers "
            "Replay worker ceiling while fault pressure is active.",
            "# TYPE gpu_fault_telemetry_spool_fault_pressure_workers gauge",
            "gpu_fault_telemetry_spool_fault_pressure_workers "
            f"{spool_runtime.get('fault_pressure_workers', 0)}",
            "# HELP gpu_fault_telemetry_spool_admitted_total "
            "Telemetry requests accepted into the spool.",
            "# TYPE gpu_fault_telemetry_spool_admitted_total counter",
            "gpu_fault_telemetry_spool_admitted_total "
            f"{telemetry_spool_admitted_total}",
            "# HELP gpu_fault_telemetry_spool_coalesced_total "
            "Spool admissions that superseded a pending sample.",
            "# TYPE gpu_fault_telemetry_spool_coalesced_total counter",
            "gpu_fault_telemetry_spool_coalesced_total "
            f"{telemetry_spool_coalesced_total}",
            "# HELP gpu_fault_telemetry_spool_rejected_total "
            "Spool admissions rejected by a depth cap.",
            "# TYPE gpu_fault_telemetry_spool_rejected_total counter",
        ]
    )
    for scope in sorted(telemetry_spool_rejections):
        lines.append(
            "gpu_fault_telemetry_spool_rejected_total"
            f'{{scope="{scope}"}} '
            f"{telemetry_spool_rejections[scope]}"
        )
    return spool_runtime


def render_spool_metrics_two(
    lines,
    ctx,
    runtime,
    spool,
    spool_runtime,
    telemetry_spool_enabled,
    telemetry_spool_batcher,
    telemetry_spool_admitted_by_path,
):
    lines.extend(
        [
            "# HELP gpu_fault_telemetry_spool_replayed_total "
            "Spooled samples this replica has replayed and "
            "deleted.",
            "# TYPE gpu_fault_telemetry_spool_replayed_total counter",
            "gpu_fault_telemetry_spool_replayed_total "
            f"{spool_runtime.get('completed', 0)}",
            "# HELP gpu_fault_telemetry_spool_superseded_total "
            "Replays whose row a newer sample had already taken "
            "over.",
            "# TYPE gpu_fault_telemetry_spool_superseded_total counter",
            "gpu_fault_telemetry_spool_superseded_total "
            f"{spool_runtime.get('superseded', 0)}",
            "# HELP gpu_fault_telemetry_spool_released_total "
            "Spooled samples returned for another attempt.",
            "# TYPE gpu_fault_telemetry_spool_released_total counter",
            "gpu_fault_telemetry_spool_released_total "
            f"{spool_runtime.get('released', 0)}",
            "# HELP gpu_fault_telemetry_spool_abandoned_total "
            "Claims returned without consuming replay attempts.",
            "# TYPE gpu_fault_telemetry_spool_abandoned_total counter",
            "gpu_fault_telemetry_spool_abandoned_total "
            f"{spool_runtime.get('abandoned', 0)}",
            "# HELP gpu_fault_telemetry_spool_dropped_total "
            "Spooled samples abandoned after exhausting their "
            "replay attempts.",
            "# TYPE gpu_fault_telemetry_spool_dropped_total counter",
            "gpu_fault_telemetry_spool_dropped_total "
            f"{spool_runtime.get('dropped', 0)}",
            "# HELP gpu_fault_telemetry_spool_errors_total "
            "Replay batches that raised an internal exception.",
            "# TYPE gpu_fault_telemetry_spool_errors_total counter",
            f"gpu_fault_telemetry_spool_errors_total {spool_runtime.get('errors', 0)}",
            "# HELP gpu_fault_telemetry_spool_claim_rounds_total "
            "Spool claim queries issued by this replica.",
            "# TYPE gpu_fault_telemetry_spool_claim_rounds_total counter",
            "gpu_fault_telemetry_spool_claim_rounds_total "
            f"{spool_runtime.get('rounds', 0)}",
            "# HELP gpu_fault_telemetry_spool_claim_rows_total "
            "Rows returned by those claims.",
            "# TYPE gpu_fault_telemetry_spool_claim_rows_total counter",
            "gpu_fault_telemetry_spool_claim_rows_total "
            f"{spool_runtime.get('rows', 0)}",
            "# HELP "
            "gpu_fault_telemetry_spool_claim_rows_by_path_total "
            "Rows claimed by the weighted telemetry path scheduler.",
            "# TYPE gpu_fault_telemetry_spool_claim_rows_by_path_total counter",
            "# HELP gpu_fault_telemetry_spool_replay_seconds "
            "Time spent replaying spooled batches.",
            "# TYPE gpu_fault_telemetry_spool_replay_seconds summary",
            "gpu_fault_telemetry_spool_replay_seconds_sum "
            f"{spool_runtime.get('replay_seconds_sum', 0.0):.6f}",
            "gpu_fault_telemetry_spool_replay_seconds_max "
            f"{spool_runtime.get('replay_seconds_max', 0.0):.6f}",
            "# HELP "
            "gpu_fault_telemetry_spool_replay_transport_total "
            "Replay batches by in-process or loopback transport.",
            "# TYPE gpu_fault_telemetry_spool_replay_transport_total counter",
            "gpu_fault_telemetry_spool_replay_transport_total"
            '{transport="direct"} '
            f"{spool_runtime.get('direct_replay', 0)}",
            "gpu_fault_telemetry_spool_replay_transport_total"
            '{transport="http"} '
            f"{spool_runtime.get('http_replay', 0)}",
            "# HELP gpu_fault_telemetry_spool_admission_pending "
            "Requests waiting for a spool admission batch.",
            "# TYPE gpu_fault_telemetry_spool_admission_pending gauge",
            "gpu_fault_telemetry_spool_admission_pending "
            f"{telemetry_spool_batcher.pending_depth}",
            "# HELP "
            "gpu_fault_telemetry_spool_admission_batch_seconds "
            "Time one spool admission batch spent in the store.",
            "# TYPE gpu_fault_telemetry_spool_admission_batch_seconds summary",
            "gpu_fault_telemetry_spool_admission_batch_seconds_sum "
            f"{telemetry_spool_batcher.flush_sum_seconds:.6f}",
            "gpu_fault_telemetry_spool_admission_batch_seconds_"
            "count "
            f"{telemetry_spool_batcher.flush_count}",
            "gpu_fault_telemetry_spool_admission_batch_seconds_max "
            f"{telemetry_spool_batcher.flush_max_seconds:.6f}",
            "# HELP gpu_fault_telemetry_spool_admission_items_total "
            "Requests carried by spool admission batches.",
            "# TYPE gpu_fault_telemetry_spool_admission_items_total counter",
            "gpu_fault_telemetry_spool_admission_items_total "
            f"{telemetry_spool_batcher.items_total}",
            "# HELP gpu_fault_telemetry_spool_admission_shed_total "
            "Requests rejected on arrival because the batcher "
            "could not reach them inside their deadline.",
            "# TYPE gpu_fault_telemetry_spool_admission_shed_total counter",
            "gpu_fault_telemetry_spool_admission_shed_total "
            f"{telemetry_spool_batcher.shed_total}",
        ]
    )
    for path, count in sorted(spool_runtime.get("rows_by_path", {}).items()):
        lines.append(
            "gpu_fault_telemetry_spool_claim_rows_by_path_total"
            f'{{path="{_metrics_label_value(path)}"}} {count}'
        )
    lines.extend(
        [
            "# HELP "
            "gpu_fault_telemetry_spool_admitted_by_path_total "
            "Spool admissions by ingestion path.",
            "# TYPE gpu_fault_telemetry_spool_admitted_by_path_total counter",
        ]
    )
    for path in sorted(telemetry_spool_admitted_by_path):
        escaped_path = path.replace("\\", "\\\\").replace('"', '\\"')
        lines.append(
            "gpu_fault_telemetry_spool_admitted_by_path_total"
            f'{{path="{escaped_path}"}} '
            f"{telemetry_spool_admitted_by_path[path]}"
        )
    lines.extend(
        [
            "# HELP gpu_fault_telemetry_spool_cluster_depth "
            "Spooled telemetry samples by cluster.",
            "# TYPE gpu_fault_telemetry_spool_cluster_depth gauge",
        ]
    )
    for cluster_id, depth in sorted(spool["by_cluster"].items()):
        escaped = (
            cluster_id.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        )
        lines.append(
            f'gpu_fault_telemetry_spool_cluster_depth{{cluster_id="{escaped}"}} {depth}'
        )
    pool_metrics = (
        ctx.store.pool_metrics()
        if hasattr(ctx.store, "pool_metrics")
        else {
            "checkout_count": 0,
            "checkout_sum_seconds": 0.0,
            "checkout_max_seconds": 0.0,
        }
    )
    return pool_metrics


def render_capacity_metrics(lines: list[str]) -> None:
    """The declared fleet topology, so a scrape can size the fault reserve.

    Site-wide facts shipped by the release renderer; 0 means the release did
    not declare them. No labels: never a node id.
    """

    largest: int = int(os.getenv("GPU_FAULT_CAPACITY_LARGEST_CLUSTER_NODE_COUNT", "0"))
    managed: int = int(os.getenv("GPU_FAULT_CAPACITY_MANAGED_NODE_COUNT", "0"))
    lines.extend(
        [
            "# HELP gpu_fault_capacity_largest_cluster_node_count "
            "Declared node count of the largest managed GPU cluster.",
            "# TYPE gpu_fault_capacity_largest_cluster_node_count gauge",
            f"gpu_fault_capacity_largest_cluster_node_count {largest}",
            "# HELP gpu_fault_capacity_managed_node_count "
            "Declared node count across every managed GPU cluster.",
            "# TYPE gpu_fault_capacity_managed_node_count gauge",
            f"gpu_fault_capacity_managed_node_count {managed}",
        ]
    )


def render_pool_metrics(lines, pool_metrics, runtime):
    lines.extend(
        [
            "# HELP gpu_fault_postgres_pool_checkout_wait_seconds "
            "PostgreSQL pool checkout wait time.",
            "# TYPE gpu_fault_postgres_pool_checkout_wait_seconds summary",
            "gpu_fault_postgres_pool_checkout_wait_seconds_sum "
            f"{pool_metrics['checkout_sum_seconds']:.6f}",
            "gpu_fault_postgres_pool_checkout_wait_seconds_count "
            f"{pool_metrics['checkout_count']}",
            "gpu_fault_postgres_pool_checkout_wait_seconds_max "
            f"{pool_metrics['checkout_max_seconds']:.6f}",
        ]
    )
    for outcome, count in runtime["processed"].items():
        lines.append(
            "gpu_fault_processor_requests_processed_total"
            f'{{outcome="{outcome}"}} {count}'
        )
    lines.extend(
        [
            "# HELP gpu_fault_processor_lane_wait_seconds "
            "Queue wait before a processor lane begins execution.",
            "# TYPE gpu_fault_processor_lane_wait_seconds summary",
        ]
    )
    for scope, values in runtime["lane_wait"].items():
        lines.append(
            "gpu_fault_processor_lane_wait_seconds_sum"
            f'{{scope="{scope}"}} {values["sum"]:.6f}'
        )
        lines.append(
            "gpu_fault_processor_lane_wait_seconds_count"
            f'{{scope="{scope}"}} {values["count"]}'
        )
        lines.append(
            "gpu_fault_processor_lane_wait_seconds_max"
            f'{{scope="{scope}"}} {values["max"]:.6f}'
        )
    lines.extend(
        [
            "# HELP gpu_fault_processor_request_processing_seconds "
            "Processor request replay duration.",
            "# TYPE gpu_fault_processor_request_processing_seconds histogram",
        ]
    )
    for boundary, count in runtime["duration_buckets"]:
        lines.append(
            "gpu_fault_processor_request_processing_seconds_bucket"
            f'{{le="{boundary:g}"}} {count}'
        )
    lines.extend(
        [
            "gpu_fault_processor_request_processing_seconds_bucket"
            f'{{le="+Inf"}} {runtime["duration_count"]}',
            "gpu_fault_processor_request_processing_seconds_sum "
            f"{runtime['duration_sum']:.6f}",
            "gpu_fault_processor_request_processing_seconds_count "
            f"{runtime['duration_count']}",
        ]
    )


def render_runtime_metrics_details(
    lines,
    runtime,
    processor_admission_batcher,
    fault_store_io,
    evidence_store_io,
    fault_decode_io,
    telemetry_spool_store_io,
    processor_admission_rejections,
    ingress_backpressure_rejections,
    processor_admission_rejections_by_path,
) -> None:
    (
        batch_in_flight_scopes,
        batch_pending_scopes,
    ) = processor_admission_batcher.scope_queue_snapshot()
    for scope, elapsed in batch_in_flight_scopes:
        escaped = _metrics_label_value(scope)
        lines.append(
            "gpu_fault_processor_admission_batch_scope"
            "_in_flight_seconds"
            f'{{cluster_id="{escaped}"}} {elapsed:.6f}'
        )
    for scope, depth, oldest in batch_pending_scopes:
        escaped = _metrics_label_value(scope)
        lines.append(
            "gpu_fault_processor_admission_batch_scope_pending"
            f'{{cluster_id="{escaped}"}} {depth}'
        )
        lines.append(
            "gpu_fault_processor_admission_batch_scope"
            "_wait_seconds"
            f'{{cluster_id="{escaped}"}} {oldest:.6f}'
        )
    # The reserved fault lane is only worth having if you can see
    # whether it is the one that filled up.
    lines.extend(
        [
            "# HELP gpu_fault_ingress_lane_in_flight "
            "Calls executing or waiting per reserved ingress lane.",
            "# TYPE gpu_fault_ingress_lane_in_flight gauge",
            "# HELP gpu_fault_ingress_lane_workers "
            "Threads configured per reserved ingress lane.",
            "# TYPE gpu_fault_ingress_lane_workers gauge",
            "# HELP gpu_fault_ingress_lane_rejections_total "
            "Calls rejected per reserved ingress lane.",
            "# TYPE gpu_fault_ingress_lane_rejections_total counter",
            "# HELP gpu_fault_ingress_lane_wait_seconds "
            "Admission wait per reserved ingress lane.",
            "# TYPE gpu_fault_ingress_lane_wait_seconds summary",
        ]
    )
    for lane, executor in (
        ("fault-store", fault_store_io),
        ("evidence-store", evidence_store_io),
        ("fault-decode", fault_decode_io),
        ("telemetry-spool-admission", telemetry_spool_store_io),
    ):
        lines.extend(
            [
                "gpu_fault_ingress_lane_in_flight"
                f'{{lane="{lane}"}} {executor.in_flight}',
                f'gpu_fault_ingress_lane_workers{{lane="{lane}"}} {executor.workers}',
                "gpu_fault_ingress_lane_rejections_total"
                f'{{lane="{lane}"}} {executor.rejected_total}',
                "gpu_fault_ingress_lane_wait_seconds_sum"
                f'{{lane="{lane}"}} '
                f"{executor.admission_wait_sum_seconds:.6f}",
                "gpu_fault_ingress_lane_wait_seconds_count"
                f'{{lane="{lane}"}} '
                f"{executor.admission_wait_count}",
                "gpu_fault_ingress_lane_wait_seconds_max"
                f'{{lane="{lane}"}} '
                f"{executor.admission_wait_max_seconds:.6f}",
            ]
        )
    for scope, count in processor_admission_rejections.items():
        if "\x1f" in scope:
            continue
        lines.append(
            f'gpu_fault_processor_admission_rejections_total{{scope="{scope}"}} {count}'
        )
    for key, count in processor_admission_rejections.items():
        if "\x1f" not in key:
            continue
        scope, cluster_id = key.split("\x1f", 1)
        escaped_cluster = cluster_id.replace("\\", "\\\\").replace('"', '\\"')
        lines.append(
            "gpu_fault_processor_admission_rejections_by_cluster_total"
            f'{{cluster_id="{escaped_cluster}",scope="{scope}"}} '
            f"{count}"
        )
    for scope, count in ingress_backpressure_rejections.items():
        lines.append(
            "gpu_fault_ingress_backpressure_rejections_total"
            f'{{scope="{scope}"}} {count}'
        )
    for path, count in sorted(processor_admission_rejections_by_path.items()):
        escaped_path = path.replace("\\", "\\\\").replace('"', '\\"')
        lines.append(
            "gpu_fault_processor_admission_rejections_by_path_total"
            f'{{path="{escaped_path}"}} {count}'
        )
    for path, count in sorted(runtime["stale_superseded_by_path"].items()):
        escaped_path = path.replace("\\", "\\\\").replace('"', '\\"')
        lines.append(
            "gpu_fault_processor_stale_superseded_by_path_total"
            f'{{path="{escaped_path}"}} {count}'
        )
    # Emitted even when the cut is switched off, so the control arm of
    # an A/B reads as an explicit zero rather than as a missing
    # series.
    lines.extend(
        [
            "# HELP gpu_fault_processor_queue_bypass_total "
            "Requests executed inline instead of through the "
            "processor queue.",
            "# TYPE gpu_fault_processor_queue_bypass_total counter",
        ]
    )
