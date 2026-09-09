from __future__ import annotations

import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from fastapi import APIRouter, Depends, Header, HTTPException, Request

from gpu_fault.app.authorization import authorization_bucket
from gpu_fault.async_store import (
    AsyncStoreExecutor,
    StoreIoCapacityExceeded,
)
from gpu_fault.channel_registry import (
    COLLECTOR_HEALTH_PATH,
    FABRIC_MANAGER_PATH,
    GPU_INVENTORY_PATH,
    GPU_METRICS_PATH,
    HOST_TELEMETRY_PATH,
    NODE_LOG_PATH,
    NVIDIA_KERNEL_PATH,
)
from gpu_fault.env import env_bool
from gpu_fault.gpu_metrics import (
    GpuInventorySnapshot,
    GpuMetricBatch,
    GpuMetricsIngestionResult,
)
from gpu_fault.hma import (
    FabricManagerLogEvent,
    HmaIngestionResult,
    HmaNormalizedBatch,
    NvidiaKernelLogEvent,
)
from gpu_fault.host_health import (
    HostTelemetryBatch,
    NodeHealthCategory,
    NodeHealthFinding,
    NodeHealthIngestionResult,
    NodeLogBatch,
    SyntheticNodeReplacementRequest,
)
from gpu_fault.models import (
    EfaTrafficAdminDecision,
    EfaTrafficAdminRequest,
    RecoveryAction,
    Severity,
    WorkloadState,
)
from gpu_fault.processor_diagnostics import (
    report_processor_replay_phase,
)
from gpu_fault.store import (
    EfaTrafficAdminConflict,
    NotFoundError,
)
from gpu_fault.telemetry import (
    CollectorHealthSummary,
    CollectorKind,
    EvidenceKind,
)


def _ignore_unresolved_signals(
    normalized: HmaNormalizedBatch, *, batch_id: str
) -> NodeHealthIngestionResult | None:
    return None


@dataclass(frozen=True)
class CollectorRouterDependencies:
    context: Any
    store_io: AsyncStoreExecutor
    processor: Any | None
    replay_authorized: Callable[[Request], bool]
    enrich_workload_context: Callable
    enrich_sxid_scope: Callable
    resolve_kernel_xid_gpu_uuid: Callable
    record_collector_status: Callable
    capture_evidence: Callable
    ingest_xid: Callable
    ingest_sxid: Callable
    ingest_gpu_inventory_batch: Callable
    persist_gpu_metrics: Callable
    finish_gpu_metrics: Callable
    ingest_node_health_findings: Callable
    persist_host_telemetry: Callable
    finish_host_telemetry: Callable
    ingest_telemetry_batch: Callable
    # Test fakes may leave this at the no-op; the production factory passes
    # FaultIngestionService.ingest_unresolved_signals so an Xid/SXid line the
    # normalizer could not read becomes a WARNING finding instead of silent
    # evidence (G7).
    ingest_unresolved_signals: Callable[..., NodeHealthIngestionResult | None] = (
        _ignore_unresolved_signals
    )


def get_collector_dependencies() -> CollectorRouterDependencies:
    raise RuntimeError("collector router dependencies are not configured")


router = APIRouter(tags=["collector-events"])


async def _store_call(
    dependencies: CollectorRouterDependencies,
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


@router.post(COLLECTOR_HEALTH_PATH)
@authorization_bucket("cluster-token")
async def ingest_collector_health(
    summary: CollectorHealthSummary,
    dependencies: CollectorRouterDependencies = Depends(get_collector_dependencies),
) -> dict[str, bool]:
    if summary.collector not in {
        CollectorKind.NVIDIA_KERNEL,
        CollectorKind.FABRIC_MANAGER_LOG,
    }:
        raise HTTPException(
            status_code=422,
            detail="collector-health only accepts event-driven collectors",
        )

    def ingest() -> dict[str, bool]:
        received_at = datetime.now(timezone.utc)
        # The kernel collector reports its own losses as ``<name>:<count>``
        # tokens (delivery-failures, kmsg-overflow, boot-time-reestimates)
        # next to the bare "health-summary" marker. Those are collection
        # errors: they set ``last_error_at`` so the erroring-nodes gauge sees
        # a collector that is dropping lines. An all-zero summary carries no
        # such token and only refreshes ``last_success_at`` (G4).
        failure_tokens = [
            reason
            for reason in summary.edge_filter_reasons
            if reason != "health-summary" and ":" in reason
        ]
        dependencies.record_collector_status(
            summary.collector,
            summary.cluster_id,
            summary.node_id,
            received_at,
            summary.summary_id,
            0,
            [f"kernel collector reported {token}" for token in failure_tokens] or None,
        )
        return {"accepted": True}

    return await _store_call(dependencies, ingest)


@router.post(
    NVIDIA_KERNEL_PATH,
    response_model=HmaIngestionResult,
)
@authorization_bucket("cluster-token")
async def ingest_nvidia_kernel(
    event: NvidiaKernelLogEvent,
    dependencies: CollectorRouterDependencies = Depends(get_collector_dependencies),
) -> HmaIngestionResult:
    def ingest() -> HmaIngestionResult:
        ctx = dependencies.context
        report_processor_replay_phase("kernel_workload_context")
        enriched = dependencies.enrich_workload_context(event, event.observed_at)
        report_processor_replay_phase("kernel_normalize")
        normalized = ctx.hma.normalize_kernel(enriched)
        normalized = normalized.model_copy(
            update={
                "sxid_events": [
                    dependencies.enrich_sxid_scope(item)
                    for item in normalized.sxid_events
                ]
            }
        )
        report_processor_replay_phase("kernel_gpu_identity")
        xids = [
            dependencies.resolve_kernel_xid_gpu_uuid(item)
            for item in normalized.xid_events
        ]
        normalized = normalized.model_copy(update={"xid_events": xids})
        report_processor_replay_phase("kernel_collector_status")
        dependencies.record_collector_status(
            CollectorKind.NVIDIA_KERNEL,
            enriched.cluster_id,
            enriched.node_id,
            enriched.observed_at,
            enriched.record_id,
            1,
        )
        report_processor_replay_phase("kernel_evidence")
        dependencies.capture_evidence(
            record_id=f"nvidia-kernel/{enriched.record_id}",
            cluster_id=enriched.cluster_id,
            node_id=enriched.node_id,
            kind=EvidenceKind.NVIDIA_KERNEL,
            observed_at=enriched.observed_at,
            payload=enriched.model_dump(mode="json"),
        )
        dependencies.ingest_unresolved_signals(normalized, batch_id=enriched.record_id)
        report_processor_replay_phase("kernel_fault_policy")
        result = HmaIngestionResult(
            normalized=normalized,
            decisions=[
                *(dependencies.ingest_xid(item) for item in xids),
                *(dependencies.ingest_sxid(item) for item in normalized.sxid_events),
            ],
        )
        report_processor_replay_phase("kernel_dispatch_wake")
        ctx.dispatcher.wake()
        return result

    return await _store_call(dependencies, ingest)


@router.post(
    FABRIC_MANAGER_PATH,
    response_model=HmaIngestionResult,
)
@authorization_bucket("cluster-token")
async def ingest_fabric_manager_log(
    event: FabricManagerLogEvent,
    dependencies: CollectorRouterDependencies = Depends(get_collector_dependencies),
) -> HmaIngestionResult:
    def ingest() -> HmaIngestionResult:
        ctx = dependencies.context
        report_processor_replay_phase("fabric_workload_context")
        enriched = dependencies.enrich_workload_context(event, event.observed_at)
        report_processor_replay_phase("fabric_normalize")
        normalized = ctx.hma.normalize_fabric_manager(enriched)
        normalized = normalized.model_copy(
            update={
                "xid_events": [
                    dependencies.resolve_kernel_xid_gpu_uuid(item)
                    for item in normalized.xid_events
                ],
                "sxid_events": [
                    dependencies.enrich_sxid_scope(item)
                    for item in normalized.sxid_events
                ],
            }
        )
        dependencies.record_collector_status(
            CollectorKind.FABRIC_MANAGER_LOG,
            enriched.cluster_id,
            enriched.node_id,
            enriched.observed_at,
            enriched.record_id,
            1,
        )
        dependencies.capture_evidence(
            record_id=f"fabric-manager/{enriched.record_id}",
            cluster_id=enriched.cluster_id,
            node_id=enriched.node_id,
            kind=EvidenceKind.FABRIC_MANAGER_LOG,
            observed_at=enriched.observed_at,
            payload=enriched.model_dump(mode="json"),
        )
        dependencies.ingest_unresolved_signals(normalized, batch_id=enriched.record_id)
        result = HmaIngestionResult(
            normalized=normalized,
            decisions=[
                *(dependencies.ingest_xid(item) for item in normalized.xid_events),
                *(dependencies.ingest_sxid(item) for item in normalized.sxid_events),
            ],
        )
        ctx.dispatcher.wake()
        return result

    return await _store_call(dependencies, ingest)


@router.post(
    GPU_INVENTORY_PATH,
    response_model=GpuInventorySnapshot,
)
@authorization_bucket("cluster-token")
async def ingest_gpu_inventory(
    snapshot: GpuInventorySnapshot,
    dependencies: CollectorRouterDependencies = Depends(get_collector_dependencies),
) -> GpuInventorySnapshot:
    values = await _store_call(
        dependencies,
        dependencies.ingest_gpu_inventory_batch,
        [snapshot],
    )
    return values[0]


@router.post(
    GPU_METRICS_PATH,
    response_model=GpuMetricsIngestionResult,
)
@authorization_bucket("cluster-token")
async def ingest_gpu_metrics(
    batch: GpuMetricBatch,
    dependencies: CollectorRouterDependencies = Depends(get_collector_dependencies),
) -> GpuMetricsIngestionResult:
    def persist():
        with dependencies.context.store.collector_ingestion_transaction(
            batch.cluster_id, batch.node_id, batch.batch_id
        ):
            return dependencies.persist_gpu_metrics(batch)

    persisted, result = await _store_call(dependencies, persist)
    return await _store_call(
        dependencies,
        dependencies.finish_gpu_metrics,
        persisted,
        result,
    )


@router.post(
    "/v1/admin/test/node-replacement",
    response_model=NodeHealthIngestionResult,
)
@authorization_bucket("execution-token")
async def inject_test_node_replacement(
    request: SyntheticNodeReplacementRequest,
    execution_token: str | None = Header(
        default=None,
        alias="X-GPU-Fault-Execution-Token",
    ),
    dependencies: CollectorRouterDependencies = Depends(get_collector_dependencies),
) -> NodeHealthIngestionResult:
    ctx = dependencies.context
    if not env_bool("GPU_FAULT_ENABLE_SYNTHETIC_REPLACEMENT_TESTS", False):
        raise HTTPException(status_code=404, detail="resource not found")
    if (
        not ctx.execution_token
        or not execution_token
        or not secrets.compare_digest(execution_token, ctx.execution_token)
    ):
        raise HTTPException(
            status_code=403,
            detail="invalid workflow execution token",
        )
    finding = NodeHealthFinding(
        finding_id=f"finding-{request.event_id}",
        event_id=request.event_id,
        cluster_id=request.cluster_id,
        node_id=request.node_id,
        observed_at=request.observed_at,
        category=NodeHealthCategory.GPU,
        severity=Severity.CRITICAL,
        reason=request.reason,
        recommended_action=RecoveryAction.REPLACE_NODE,
        gpu_uuids=request.gpu_uuids,
        runtime_profile_version=request.runtime_profile_version,
        workload_state=WorkloadState.ACTIVE,
        affected_workload_ids=request.affected_workload_ids,
        job_id=request.job_id,
        attempt_id=request.attempt_id,
        diagnostic_parameters={
            "replacement_strategy": request.replacement_strategy,
            "synthetic": True,
        },
        policy_source="SITE_SYNTHETIC_REPLACEMENT_TEST",
        policy_reference=("execution-token protected warm-spare E2E test"),
        official_action="REPLACE_NODE",
    )
    result = await _store_call(
        dependencies,
        dependencies.ingest_node_health_findings,
        request.event_id,
        [finding],
    )
    ctx.dispatcher.wake()
    return result


@router.post(
    HOST_TELEMETRY_PATH,
    response_model=NodeHealthIngestionResult,
)
@authorization_bucket("cluster-token")
async def ingest_host_telemetry(
    batch: HostTelemetryBatch,
    dependencies: CollectorRouterDependencies = Depends(get_collector_dependencies),
) -> NodeHealthIngestionResult:
    def persist():
        with dependencies.context.store.collector_ingestion_transaction(
            batch.cluster_id, batch.node_id, batch.batch_id
        ):
            return dependencies.persist_host_telemetry(batch)

    persisted, findings = await _store_call(dependencies, persist)
    return await _store_call(
        dependencies,
        dependencies.finish_host_telemetry,
        persisted,
        findings,
    )


@router.post("/v1/internal/processor/telemetry-batch")
@authorization_bucket("execution-token")
async def ingest_processor_telemetry_batch(
    http_request: Request,
    batch_request: dict,
    replay_token: str | None = Header(
        default=None,
        alias="X-GPU-Fault-Processor-Replay",
    ),
    dependencies: CollectorRouterDependencies = Depends(get_collector_dependencies),
) -> dict:
    if dependencies.processor is None or not dependencies.replay_authorized(
        http_request
    ):
        raise HTTPException(
            status_code=403,
            detail=(
                "processor replay is accepted only from loopback "
                "with a valid replay secret"
            ),
        )
    return await _store_call(
        dependencies,
        dependencies.ingest_telemetry_batch,
        batch_request,
    )


@router.post(
    "/v1/efa-traffic/admin-actions",
    response_model=EfaTrafficAdminDecision,
)
@authorization_bucket("execution-token")
async def apply_efa_traffic_admin_action(
    request: EfaTrafficAdminRequest,
    execution_token: str | None = Header(
        default=None,
        alias="X-GPU-Fault-Execution-Token",
    ),
    dependencies: CollectorRouterDependencies = Depends(get_collector_dependencies),
) -> EfaTrafficAdminDecision:
    ctx = dependencies.context
    if (
        not ctx.execution_token
        or not execution_token
        or not secrets.compare_digest(execution_token, ctx.execution_token)
    ):
        raise HTTPException(
            status_code=403,
            detail="invalid workflow execution token",
        )

    def apply() -> EfaTrafficAdminDecision:
        incident = ctx.store.get_incident_by_event(request.event_id)
        if incident is None:
            raise NotFoundError(request.event_id)
        if (
            incident.event_type != "NODE_HEALTH"
            or incident.policy_source != "SITE_EFA_TRAFFIC"
            or incident.cluster_id != request.cluster_id
            or incident.node_ids != [request.node_id]
        ):
            raise EfaTrafficAdminConflict(
                "event_id is not the requested node's EFA traffic event"
            )
        state_key = ctx.store.efa_traffic_state_key(
            request.cluster_id,
            request.node_id,
            request.job_id,
            request.attempt_id,
        )
        decision = ctx.store.apply_efa_traffic_admin_action(
            state_key=state_key,
            event_id=request.event_id,
            action=request.action,
            operator=request.operator,
            reason=request.reason,
            decided_at=datetime.now(timezone.utc),
        )
        ctx.evidence.capture(
            record_id=decision.decision_id,
            cluster_id=request.cluster_id,
            node_id=request.node_id,
            kind=EvidenceKind.ADMIN_ACTION,
            observed_at=decision.decided_at,
            attempt_ids=[request.attempt_id],
            payload=decision.model_dump(mode="json"),
        )
        return decision

    return await _store_call(dependencies, apply)


@router.post(
    NODE_LOG_PATH,
    response_model=NodeHealthIngestionResult,
)
@authorization_bucket("cluster-token")
async def ingest_node_logs(
    batch: NodeLogBatch,
    dependencies: CollectorRouterDependencies = Depends(get_collector_dependencies),
) -> NodeHealthIngestionResult:
    def ingest() -> NodeHealthIngestionResult:
        enriched = dependencies.enrich_workload_context(batch, batch.collected_at)
        result = dependencies.ingest_node_health_findings(
            enriched.batch_id,
            (
                dependencies.context.node_health.evaluate_logs(enriched)
                if enriched.entries
                else []
            ),
        )
        # The errors are handed over so this batch does not count as a success:
        # a node whose journal read failed produced no entries, and recording it
        # as a success is what would make a broken collector look healthy.
        dependencies.record_collector_status(
            CollectorKind.NODE_LOGS,
            enriched.cluster_id,
            enriched.node_id,
            enriched.collected_at,
            enriched.batch_id,
            len(enriched.entries),
            enriched.collection_errors,
        )
        # An error-only batch has no entries and is still the only record that
        # the node's log collection is broken, so it is kept as evidence too.
        if enriched.entries or enriched.collection_errors:
            dependencies.capture_evidence(
                record_id=f"node-logs/{enriched.batch_id}",
                cluster_id=enriched.cluster_id,
                node_id=enriched.node_id,
                kind=EvidenceKind.NODE_LOGS,
                observed_at=enriched.collected_at,
                payload=enriched.model_dump(mode="json"),
            )
        return result

    return await _store_call(dependencies, ingest)
