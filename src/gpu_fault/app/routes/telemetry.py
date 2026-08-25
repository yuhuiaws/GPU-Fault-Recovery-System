from __future__ import annotations

from gpu_fault.app.authorization import authorization_bucket

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from fastapi import APIRouter, Depends, Header, HTTPException

from gpu_fault.channel_registry import TRAINING_PROGRESS_PATH
from gpu_fault.collector_requirements import (
    COLLECTOR_SYSTEMD_UNITS,
    collector_silent_thresholds,
    required_collectors_for_agent,
)
from gpu_fault.async_store import (
    AsyncStoreExecutor,
    StoreIoCapacityExceeded,
)
from gpu_fault.gpu_metrics import (
    GpuHealthFinding,
    GpuMetricLatest,
)
from gpu_fault.telemetry import (
    CollectorKind,
    CollectorStatus,
    EvidenceKind,
    RawEvidenceRecord,
)
from gpu_fault.training_health import (
    TrainingHealthResult,
    TrainingProgressHeartbeat,
)


@dataclass(frozen=True)
class TelemetryRouterDependencies:
    context: Any
    store_io: AsyncStoreExecutor
    ingest_node_health_findings: Callable


def get_telemetry_dependencies() -> TelemetryRouterDependencies:
    raise RuntimeError("telemetry router dependencies are not configured")


router = APIRouter(tags=["telemetry"])


async def _store_call(
    dependencies: TelemetryRouterDependencies,
    function: Callable,
    /,
    *args,
    **kwargs,
):
    try:
        return await dependencies.store_io.run(function, *args, **kwargs)
    except StoreIoCapacityExceeded as exc:
        raise HTTPException(
            status_code=503,
            detail="store I/O capacity exceeded",
            headers={"Retry-After": "2"},
        ) from exc


@router.get("/v1/collector-readiness/{cluster_id}")
@authorization_bucket("execution-token")
async def collector_readiness(
    cluster_id: str,
    dependencies: TelemetryRouterDependencies = Depends(get_telemetry_dependencies),
) -> dict[str, Any]:
    def read() -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        store = dependencies.context.store
        statuses = {
            (item.node_id, item.collector): item
            for item in store.list_collector_statuses(cluster_id)
        }
        thresholds = _collector_readiness_thresholds()
        nodes = []
        for agent in store.list_agents(cluster_id):
            collectors = {}
            for kind in required_collectors_for_agent(agent):
                unit = COLLECTOR_SYSTEMD_UNITS[kind]
                status = statuses.get((agent.node_id, kind))
                last = status.last_success_at if status else None
                age = (now - last).total_seconds() if last is not None else None
                service_state = agent.collector_services.get(unit)
                unit_state = (
                    service_state.active.value
                    if service_state is not None
                    else "unknown"
                )
                unit_enabled = (
                    service_state.enabled.value
                    if service_state is not None
                    else "unknown"
                )
                ready = (
                    age is not None
                    and age <= thresholds[kind]
                    and unit_state in {"active", "unknown"}
                )
                collectors[kind.value] = {
                    "unit": unit,
                    "unit_state": unit_state,
                    "unit_enabled": unit_enabled,
                    "last_success_at": (last.isoformat() if last else None),
                    "age_seconds": age,
                    "ready": ready,
                }
            nodes.append(
                {
                    "node_id": agent.node_id,
                    "collectors": collectors,
                    "ready": all(item["ready"] for item in collectors.values()),
                }
            )
        return {
            "cluster_id": cluster_id,
            "ready": bool(nodes) and all(node["ready"] for node in nodes),
            "nodes": nodes,
        }

    return await _store_call(dependencies, read)


def _collector_readiness_thresholds() -> dict[CollectorKind, float]:
    return {**collector_silent_thresholds()}


@router.get(
    "/v1/collector-status/{cluster_id}",
    response_model=list[CollectorStatus],
)
@authorization_bucket("execution-token")
async def collector_status(
    cluster_id: str,
    node_id: str | None = None,
    dependencies: TelemetryRouterDependencies = Depends(get_telemetry_dependencies),
) -> list[CollectorStatus]:
    return await _store_call(
        dependencies,
        dependencies.context.store.list_collector_statuses,
        cluster_id,
        node_id,
    )


@router.get(
    "/v1/evidence/{cluster_id}",
    response_model=list[RawEvidenceRecord],
)
@authorization_bucket("execution-token")
async def raw_evidence(
    cluster_id: str,
    node_id: str | None = None,
    attempt_id: str | None = None,
    kind: EvidenceKind | None = None,
    limit: int = 100,
    dependencies: TelemetryRouterDependencies = Depends(get_telemetry_dependencies),
) -> list[RawEvidenceRecord]:
    if limit < 1 or limit > 1000:
        raise HTTPException(
            status_code=422,
            detail="limit must be between 1 and 1000",
        )
    return await _store_call(
        dependencies,
        dependencies.context.store.list_raw_evidence,
        cluster_id,
        node_id=node_id,
        attempt_id=attempt_id,
        kind=kind,
        limit=limit,
    )


@router.post(
    TRAINING_PROGRESS_PATH,
    response_model=TrainingHealthResult,
)
@authorization_bucket("cluster-token")
async def ingest_training_progress(
    heartbeat: TrainingProgressHeartbeat,
    dependencies: TelemetryRouterDependencies = Depends(get_telemetry_dependencies),
) -> TrainingHealthResult:
    ctx = dependencies.context

    def ingest() -> TrainingHealthResult:
        try:
            result = ctx.training_health.ingest(heartbeat)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if result.findings:
            dependencies.ingest_node_health_findings(
                heartbeat.heartbeat_id, result.findings
            )
        ctx.evidence.capture(
            record_id=(f"training-progress/{heartbeat.heartbeat_id}"),
            cluster_id=heartbeat.cluster_id,
            node_id=heartbeat.node_id or "UNKNOWN",
            kind=EvidenceKind.TRAINING_PROGRESS,
            observed_at=heartbeat.observed_at,
            attempt_ids=[heartbeat.attempt_id],
            payload=heartbeat.model_dump(mode="json"),
        )
        return result

    return await _store_call(dependencies, ingest)


@router.post(
    "/v1/training-health/{cluster_id}/scan",
    response_model=TrainingHealthResult,
)
@authorization_bucket("execution-token")
async def scan_training_health(
    cluster_id: str,
    dependencies: TelemetryRouterDependencies = Depends(get_telemetry_dependencies),
) -> TrainingHealthResult:
    def scan() -> TrainingHealthResult:
        result = dependencies.context.training_health.scan(cluster_id)
        if result.findings:
            dependencies.ingest_node_health_findings(
                "training-health-scan", result.findings
            )
        return result

    return await _store_call(dependencies, scan)


@router.get(
    "/v1/training-progress/{cluster_id}/{attempt_id}",
    response_model=list[TrainingProgressHeartbeat],
)
@authorization_bucket("cluster-token")
async def latest_training_progress(
    cluster_id: str,
    attempt_id: str,
    dependencies: TelemetryRouterDependencies = Depends(get_telemetry_dependencies),
) -> list[TrainingProgressHeartbeat]:
    return await _store_call(
        dependencies,
        dependencies.context.store.list_training_progress,
        cluster_id,
        attempt_id,
    )


@router.get(
    "/v1/gpu-metrics/{cluster_id}/{node_id}/latest",
    response_model=list[GpuMetricLatest],
)
@authorization_bucket("dual-credential")
async def latest_gpu_metrics(
    cluster_id: str,
    node_id: str,
    authenticated_cluster: str | None = Header(
        default=None,
        alias="X-GPU-Fault-Cluster-ID",
    ),
    dependencies: TelemetryRouterDependencies = Depends(get_telemetry_dependencies),
) -> list[GpuMetricLatest]:
    ctx = dependencies.context
    if (
        ctx.regional_mode
        and authenticated_cluster
        and cluster_id != authenticated_cluster
    ):
        raise HTTPException(
            status_code=403,
            detail=(
                "authenticated cluster cannot read GPU metrics for another cluster"
            ),
        )
    return await _store_call(
        dependencies,
        ctx.gpu_metrics.latest,
        cluster_id,
        node_id,
    )


@router.get(
    "/v1/gpu-health-findings/{cluster_id}/{node_id}",
    response_model=list[GpuHealthFinding],
)
@authorization_bucket("execution-token")
async def gpu_health_findings(
    cluster_id: str,
    node_id: str,
    active_only: bool = True,
    dependencies: TelemetryRouterDependencies = Depends(get_telemetry_dependencies),
) -> list[GpuHealthFinding]:
    return await _store_call(
        dependencies,
        dependencies.context.gpu_metrics.findings,
        cluster_id,
        node_id,
        active_only=active_only,
    )
