from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import Response

import gpu_fault.app.process_metrics as process_metrics
from gpu_fault.app.aurora_refresh_metrics import (
    aurora_credential_refresh_metric_lines,
)
from gpu_fault.app.authorization import authorization_bucket
from gpu_fault.app.builtin_metric_contributors import (
    closed_loop_metric_lines,
    completion_state_metric_lines,
    control_loop_metric_lines,
    fleet_pin_drift_metric_lines,
    fleet_rollout_metric_lines,
    orchestration_metric_lines,
    policy_metric_lines,
    postgres_pool_metric_lines,
    regional_registry_metric_lines,
    remote_command_metric_lines,
    spare_reservation_metric_lines,
)
from gpu_fault.app.metric_contributors import (
    MetricContributorRegistry,
)
from gpu_fault.app.metric_scan_cache import metric_scan_cache
from gpu_fault.app.metrics_sections import (
    render_admission_metrics,
    render_capacity_metrics,
    render_pool_metrics,
    render_processor_metrics_1,
    render_processor_metrics_2,
    render_processor_metrics_3,
    render_runtime_metrics_details,
    render_runtime_metrics_one,
    render_runtime_metrics_two,
    render_spool_metrics_one,
    render_spool_metrics_two,
)
from gpu_fault.app.runtime import AppRuntime
from gpu_fault.store.contracts import ProcessorQueueStats


def get_app_runtime() -> AppRuntime:
    raise RuntimeError("application runtime is not configured")


router = APIRouter(tags=["metrics"])


def _metrics_label_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def collector_silence_lines(runtime: AppRuntime) -> list[str]:
    return runtime.collector_metrics_snapshot.lines()


def _escape_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _processor_cluster_queue_lines(queue: ProcessorQueueStats) -> list[str]:
    """Render the per-cluster processor queue families.

    Depth says how much of one cluster's quota is in use; the oldest age (S6)
    says how long that cluster has actually been waiting. The claim window is
    FIFO across the whole region, so a storm in one cluster delays every other
    cluster's requests, and only the per-cluster age shows who is paying.
    """

    lines = [
        "# HELP gpu_fault_processor_cluster_queue_depth "
        "Incomplete processor requests by cluster.",
        "# TYPE gpu_fault_processor_cluster_queue_depth gauge",
    ]
    for cluster_id, depth in sorted(queue["by_cluster"].items()):
        lines.append(
            "gpu_fault_processor_cluster_queue_depth"
            f'{{cluster_id="{_escape_label(cluster_id)}"}} {depth}'
        )
    lines.extend(
        [
            "# HELP gpu_fault_processor_cluster_queue_oldest_age_seconds "
            "Age of the oldest incomplete processor request by cluster.",
            "# TYPE gpu_fault_processor_cluster_queue_oldest_age_seconds gauge",
        ]
    )
    for cluster_id, age in sorted(queue["oldest_age_by_cluster"].items()):
        lines.append(
            "gpu_fault_processor_cluster_queue_oldest_age_seconds"
            f'{{cluster_id="{_escape_label(cluster_id)}"}} {age:.6f}'
        )
    return lines


def render_prometheus_metrics(app_runtime: AppRuntime) -> list[str]:
    """Render this replica's own /metrics families, then the contributors.

    The families rendered inline here are read as per-replica facts by the
    alert rules (queue view, IO backpressure, admission counters, spool state)
    and ``scripts/verify-regional-alerting.py`` leaves them out of its
    aggregation check. Store-derived, cluster-level families belong in
    ``builtin_metric_contributors``, where every alert must ``max by``.
    """

    ctx = app_runtime.context
    processor = app_runtime.processor
    processor_replay_tracker = app_runtime.processor_replay_tracker
    store_io, decode_io = app_runtime.store_io, app_runtime.decode_io
    fault_store_io = app_runtime.fault_store_io
    evidence_store_io = app_runtime.evidence_store_io
    fault_decode_io = app_runtime.fault_decode_io
    telemetry_spool_store_io = app_runtime.telemetry_spool_store_io
    processor_admission_batcher = app_runtime.processor_admission_batcher
    fault_admission_batcher = app_runtime.fault_admission_batcher
    evidence_admission_batcher = app_runtime.evidence_admission_batcher
    telemetry_spool_batcher = app_runtime.telemetry_spool_batcher
    background_services_enabled = app_runtime.background_services_enabled
    telemetry_spool_enabled = app_runtime.telemetry_spool_enabled
    processor_queue_bypass_enabled = app_runtime.processor_queue_bypass_enabled
    processor_queue_bypass_paths = app_runtime.processor_queue_bypass_paths
    processor_admission_rejections = app_runtime.processor_admission_rejections
    processor_admission_rejections_by_path = (
        app_runtime.processor_admission_rejections_by_path
    )
    processor_oversize_rejections = app_runtime.dispatch_state.oversize_rejections
    processor_queue_bypasses_by_path = app_runtime.processor_queue_bypasses_by_path
    processor_telemetry_coalesced = app_runtime.dispatch_state.telemetry_coalesced
    telemetry_spool_admitted_by_path = app_runtime.telemetry_spool_admitted_by_path
    telemetry_spool_rejections = app_runtime.telemetry_spool_rejections
    ingress_backpressure_rejections = app_runtime.ingress_backpressure_rejections
    (
        event_loop_lag_count,
        event_loop_lag_sum,
        event_loop_lag_max,
    ) = app_runtime.event_loop_lag_snapshot()
    telemetry_spool_admitted_total = app_runtime.dispatch_state.telemetry_spool_admitted
    telemetry_spool_coalesced_total = (
        app_runtime.dispatch_state.telemetry_spool_coalesced
    )
    queue = ctx.store.processor_queue_stats()
    lines = [
        "# HELP gpu_fault_processor_queue_depth Incomplete processor requests.",
        "# TYPE gpu_fault_processor_queue_depth gauge",
        f"gpu_fault_processor_queue_depth {queue['depth']}",
        "# HELP gpu_fault_event_loop_lag_seconds Observed event-loop scheduling lag.",
        "# TYPE gpu_fault_event_loop_lag_seconds summary",
        f"gpu_fault_event_loop_lag_seconds_sum {event_loop_lag_sum:.6f}",
        f"gpu_fault_event_loop_lag_seconds_count {event_loop_lag_count}",
        f"gpu_fault_event_loop_lag_seconds_max {event_loop_lag_max:.6f}",
        "# HELP gpu_fault_processor_queue_oldest_age_seconds "
        "Age of the oldest incomplete processor request.",
        "# TYPE gpu_fault_processor_queue_oldest_age_seconds gauge",
        "gpu_fault_processor_queue_oldest_age_seconds "
        f"{queue['oldest_age_seconds']:.6f}",
    ]
    lines.extend(_processor_cluster_queue_lines(queue))
    runtime = (
        processor.metrics_snapshot()
        if processor is not None
        else {
            "processed": {"success": 0, "error": 0},
            "duration_buckets": [],
            "duration_count": 0,
            "duration_sum": 0.0,
            "worker_count": 0,
            "active_consumer": 0,
            "in_flight": 0,
            "oldest_in_flight_seconds": 0.0,
            "in_flight_by_phase": {},
            "deadline_exceeded_total": 0,
            "completion_retries_total": 0,
            "completion_failures_total": 0,
            "completions_by_path_status": {},
            "fault_rejections_total": 0,
            "notifications_enabled": 0,
            "notifications_received_total": 0,
            "notifications_filtered_total": 0,
            "notification_reconnects_total": 0,
            "notification_shard": -1,
            "notification_shard_count": 0,
            "notification_listener_connected": 0,
            "notification_shardless_episodes_total": 0,
            "fault_rows_skipped_by_observation_total": 0,
            "fault_rows_blocked_by_observation": 0,
            "interlock_probes_total": 0,
            "stale_superseded_total": 0,
            "stale_superseded_by_path": {},
            "lane_wait": {
                scope: {
                    "count": 0,
                    "sum": 0.0,
                    "max": 0.0,
                }
                for scope in ("attempt", "node", "cluster")
            },
            "claim": {
                "rounds": 0,
                "rows": 0,
                "empty": 0,
                "rounds_by_stream": {},
                "rows_by_stream": {},
                "empty_by_stream": {},
                "backoff_skips": 0,
                "probes": 0,
                "lane_blocked": 0,
                "seconds_sum": 0.0,
                "seconds_max": 0.0,
                "backoff_seconds": 0.0,
            },
            "healthy": 1,
            "unhealthy_reason": None,
            "unhealthy_since": None,
        }
    )
    services_active = background_services_enabled
    runtime["active_consumer"] = int(services_active and runtime["active_consumer"])
    claim = runtime.get("claim") or {}
    # Threads that exist, not env that was set. The pools are created
    # by run_processor, which does not start when background services
    # are off (GPU_FAULT_SERVICE_ROLE=ingress), so a configured count
    # there is inert. Reporting it anyway made a sum over replicas
    # overstate the fleet's consumer capacity by the whole ingress
    # tier.
    processor_worker_threads = runtime["worker_count"] if services_active else 0
    render_processor_metrics_1(
        lines,
        runtime,
        claim,
        processor_worker_threads,
        processor_admission_batcher,
        processor_oversize_rejections,
        processor_telemetry_coalesced,
        store_io,
        decode_io,
    )
    render_processor_metrics_2(
        lines,
        runtime,
        claim,
        processor_worker_threads,
        processor_admission_batcher,
        processor_oversize_rejections,
        processor_telemetry_coalesced,
        store_io,
        decode_io,
    )
    render_processor_metrics_3(
        lines,
        runtime,
        claim,
        processor_worker_threads,
        processor_admission_batcher,
        processor_oversize_rejections,
        processor_telemetry_coalesced,
        store_io,
        decode_io,
    )
    render_admission_metrics(lines, fault_admission_batcher, evidence_admission_batcher)
    render_runtime_metrics_one(
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
    )
    render_runtime_metrics_details(
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
    )
    spool = render_runtime_metrics_two(
        lines,
        ctx,
        telemetry_spool_enabled,
        processor_queue_bypass_enabled,
        processor_queue_bypass_paths,
        processor_queue_bypasses_by_path,
        scan_cache=metric_scan_cache(app_runtime),
    )
    spool_runtime = render_spool_metrics_one(
        lines,
        spool,
        runtime,
        telemetry_spool_enabled,
        telemetry_spool_batcher,
        telemetry_spool_rejections,
        telemetry_spool_admitted_total,
        telemetry_spool_coalesced_total,
    )
    pool_metrics = render_spool_metrics_two(
        lines,
        ctx,
        runtime,
        spool,
        spool_runtime,
        telemetry_spool_enabled,
        telemetry_spool_batcher,
        telemetry_spool_admitted_by_path,
    )
    render_pool_metrics(lines, pool_metrics, runtime, app_runtime)
    return lines


def capacity_metric_lines(_runtime: AppRuntime) -> list[str]:
    lines: list[str] = []
    render_capacity_metrics(lines)
    return lines


# ``fleet_level=True`` marks the contributors that read a cluster-level fact
# out of the store; every replica publishes the same number and the alerts
# ``max by`` over them, so roles without background services (ingress,
# spool-worker) skip them instead of repeating the worker tier's table scans
# (A-6). Everything else is a per-replica fact and renders on every role.
METRIC_CONTRIBUTORS = MetricContributorRegistry()
METRIC_CONTRIBUTORS.register("core", render_prometheus_metrics)
METRIC_CONTRIBUTORS.register("capacity", capacity_metric_lines)
METRIC_CONTRIBUTORS.register(
    "remote-command",
    remote_command_metric_lines,
    fleet_level=True,
)
METRIC_CONTRIBUTORS.register(
    "fleet-rollout",
    fleet_rollout_metric_lines,
    fleet_level=True,
)
METRIC_CONTRIBUTORS.register("fleet-pin-drift", fleet_pin_drift_metric_lines)
METRIC_CONTRIBUTORS.register("spare-reservations", spare_reservation_metric_lines)
METRIC_CONTRIBUTORS.register("regional-registry", regional_registry_metric_lines)
METRIC_CONTRIBUTORS.register("policy", policy_metric_lines)
METRIC_CONTRIBUTORS.register(
    "orchestration",
    orchestration_metric_lines,
    fleet_level=True,
)
METRIC_CONTRIBUTORS.register("closed-loop", closed_loop_metric_lines, fleet_level=True)
METRIC_CONTRIBUTORS.register("control-loop", control_loop_metric_lines)
METRIC_CONTRIBUTORS.register(
    "completion-state", completion_state_metric_lines, fleet_level=True
)
METRIC_CONTRIBUTORS.register("postgres-pool", postgres_pool_metric_lines)
METRIC_CONTRIBUTORS.register(
    "aurora-credential-refresh", aurora_credential_refresh_metric_lines
)
METRIC_CONTRIBUTORS.register(
    "collector-silence",
    collector_silence_lines,
)


def render_metric_response(app_runtime: AppRuntime) -> Response:
    # The Pod runs several uvicorn processes behind one port; this process's
    # render is merged with every live sibling's published render before it
    # is answered, so the scrape reads the Pod whichever process accepted it.
    lines = process_metrics.pod_coherent_lines(METRIC_CONTRIBUTORS.render(app_runtime))
    return Response(
        content="\n".join(lines) + "\n",
        media_type="text/plain; version=0.0.4",
    )


def process_local_metric_lines(app: Any) -> list[str]:
    """What the process-metrics publisher shares every few seconds: this
    process's full render. The fleet-level families cost one store read per
    scan-cache TTL per process however often they render, and some of those
    contributors also carry process-local counters that would otherwise never
    leave the process that counted them."""

    return METRIC_CONTRIBUTORS.render(app.state.runtime)


# The scrape renders on its own thread, never through ``runtime.store_io``
# (H2-2). That executor's admission semaphore is exactly the resource the
# store_io saturation alerts watch, and a scrape queued behind saturated
# request work timed out at ADOT's deadline -- so the saturation the alert
# existed for was the moment it lost its samples. One thread per process: a
# second scrape arriving mid-render waits for the first rather than doubling
# the store reads, and the pool's own timeout still bounds the render.
_RENDER_EXECUTOR: ThreadPoolExecutor | None = None
_RENDER_EXECUTOR_LOCK = Lock()


def _render_executor() -> ThreadPoolExecutor:
    global _RENDER_EXECUTOR
    with _RENDER_EXECUTOR_LOCK:
        if _RENDER_EXECUTOR is None:
            _RENDER_EXECUTOR = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="gpu-fault-metrics"
            )
        return _RENDER_EXECUTOR


@router.get("/metrics", include_in_schema=False)
@authorization_bucket("metrics")
async def prometheus_metrics(
    runtime: AppRuntime = Depends(get_app_runtime),
) -> Response:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _render_executor(), render_metric_response, runtime
    )


@router.get("/v1/internal/metrics/collector-silence")
@authorization_bucket("execution-token")
async def collector_silence_details(
    runtime: AppRuntime = Depends(get_app_runtime),
) -> dict[str, Any]:
    return {
        "nodes": runtime.collector_metrics_snapshot.details(),
    }
