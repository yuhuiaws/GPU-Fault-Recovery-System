from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response

from gpu_fault.app.authorization import authorization_bucket
from gpu_fault.app.builtin_metric_contributors import (
    closed_loop_metric_lines,
    orchestration_metric_lines,
    policy_metric_lines,
    remote_command_metric_lines,
)
from gpu_fault.app.metric_contributors import (
    MetricContributorRegistry,
)
from gpu_fault.app.metrics_sections import (
    render_admission_metrics,
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
from gpu_fault.async_store import StoreIoCapacityExceeded


def get_app_runtime() -> AppRuntime:
    raise RuntimeError("application runtime is not configured")


router = APIRouter(tags=["metrics"])


def _metrics_label_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def collector_silence_lines(runtime: AppRuntime) -> list[str]:
    return runtime.collector_metrics_snapshot.lines()


def render_prometheus_metrics(app_runtime: AppRuntime) -> list[str]:
    ctx = app_runtime.context
    processor = app_runtime.processor
    processor_replay_tracker = app_runtime.processor_replay_tracker
    store_io = app_runtime.store_io
    decode_io = app_runtime.decode_io
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
        "# HELP gpu_fault_processor_cluster_queue_depth "
        "Incomplete processor requests by cluster.",
        "# TYPE gpu_fault_processor_cluster_queue_depth gauge",
    ]
    for cluster_id, depth in sorted(queue["by_cluster"].items()):
        escaped = (
            cluster_id.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        )
        lines.append(
            f'gpu_fault_processor_cluster_queue_depth{{cluster_id="{escaped}"}} {depth}'
        )
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
            "notifications_enabled": 0,
            "notifications_received_total": 0,
            "notifications_filtered_total": 0,
            "notification_reconnects_total": 0,
            "notification_shard": -1,
            "notification_shard_count": 0,
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
    render_pool_metrics(lines, pool_metrics, runtime)
    return lines


METRIC_CONTRIBUTORS = MetricContributorRegistry()
METRIC_CONTRIBUTORS.register("core", render_prometheus_metrics)
METRIC_CONTRIBUTORS.register(
    "remote-command",
    remote_command_metric_lines,
)
METRIC_CONTRIBUTORS.register("policy", policy_metric_lines)
METRIC_CONTRIBUTORS.register(
    "orchestration",
    orchestration_metric_lines,
)
METRIC_CONTRIBUTORS.register("closed-loop", closed_loop_metric_lines)
METRIC_CONTRIBUTORS.register(
    "collector-silence",
    collector_silence_lines,
)


def render_metric_response(app_runtime: AppRuntime) -> Response:
    lines = METRIC_CONTRIBUTORS.render(app_runtime)
    return Response(
        content="\n".join(lines) + "\n",
        media_type="text/plain; version=0.0.4",
    )


@router.get("/metrics", include_in_schema=False)
@authorization_bucket("metrics")
async def prometheus_metrics(
    runtime: AppRuntime = Depends(get_app_runtime),
) -> Response:
    try:
        return await runtime.store_io.run(render_metric_response, runtime)
    except StoreIoCapacityExceeded as exc:
        raise HTTPException(
            status_code=503,
            detail="store I/O capacity exceeded",
            headers={"Retry-After": "2"},
        ) from exc


@router.get("/v1/internal/metrics/collector-silence")
@authorization_bucket("execution-token")
async def collector_silence_details(
    runtime: AppRuntime = Depends(get_app_runtime),
) -> dict[str, Any]:
    return {
        "nodes": runtime.collector_metrics_snapshot.details(),
    }
