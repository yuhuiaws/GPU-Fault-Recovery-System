from __future__ import annotations

from gpu_fault.app.authorization import authorization_bucket

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from fastapi import APIRouter, Depends, HTTPException

from gpu_fault.async_store import (
    AsyncStoreExecutor,
    StoreIoCapacityExceeded,
)
from gpu_fault.hma import (
    HmaCloudWatchLogEvent,
    HmaDeploymentDiscovery,
    HmaDeploymentProbe,
    HmaIngestionResult,
    HmaKubernetesNodeEvent,
    HmaNodeSnapshot,
)
from gpu_fault.policy import (
    DistributedXidBatch,
    DistributedXidIngestionResult,
    FaultPolicyDecision,
    SxidEvent,
    XidEvent,
)


@dataclass(frozen=True)
class GpuEventRouterDependencies:
    context: Any
    store_io: AsyncStoreExecutor
    ingest_xid: Callable[[XidEvent], FaultPolicyDecision]
    ingest_sxid: Callable[[SxidEvent], FaultPolicyDecision]
    enrich_sxid_scope: Callable[[SxidEvent], SxidEvent]


def get_gpu_event_dependencies() -> GpuEventRouterDependencies:
    raise RuntimeError("GPU event router dependencies are not configured")


router = APIRouter(tags=["gpu-events"])


async def _store_call(
    dependencies: GpuEventRouterDependencies,
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


@router.post(
    "/v1/gpu-events/xid",
    response_model=FaultPolicyDecision,
)
@authorization_bucket("cluster-token")
async def evaluate_xid(
    event: XidEvent,
    dependencies: GpuEventRouterDependencies = Depends(get_gpu_event_dependencies),
) -> FaultPolicyDecision:
    return await _store_call(dependencies, dependencies.ingest_xid, event)


@router.get("/v1/gpu-events/xid/{event_id}/correlation")
@authorization_bucket("cluster-token")
async def xid_correlation_status(
    event_id: str,
    dependencies: GpuEventRouterDependencies = Depends(get_gpu_event_dependencies),
) -> dict:
    def lookup() -> dict:
        correlation = dependencies.context.store.get_xid_correlation(event_id)
        decision = dependencies.context.store.get_xid_policy_decision(event_id)
        return {
            "correlation": correlation.model_dump(mode="json"),
            "decision": (
                decision.model_dump(mode="json") if decision is not None else None
            ),
        }

    return await _store_call(dependencies, lookup)


@router.post(
    "/v1/gpu-events/xid/distributed",
    response_model=DistributedXidIngestionResult,
)
@authorization_bucket("cluster-token")
async def evaluate_distributed_xids(
    batch: DistributedXidBatch,
    dependencies: GpuEventRouterDependencies = Depends(get_gpu_event_dependencies),
) -> DistributedXidIngestionResult:
    def ingest() -> DistributedXidIngestionResult:
        ctx = dependencies.context
        now = datetime.now(timezone.utc)
        normalized = batch.model_copy(
            update={
                "events": [
                    event
                    if event.ingested_at is not None
                    else event.model_copy(update={"ingested_at": now})
                    for event in batch.events
                ]
            }
        )
        prepared = [
            ctx.xid_correlation.prepare_xid74(ctx.xid_correlation.prepare_xid154(event))
            for event in normalized.events
        ]
        normalized = normalized.model_copy(update={"events": prepared})
        decisions = [
            ctx.orchestrator.apply_fault_action_generation_fence(
                event,
                ctx.policy.evaluate_xid(
                    event,
                    companion_events=prepared,
                    xid154_window_closed=True,
                ),
            )
            for event in prepared
        ]
        try:
            incident, workflow = ctx.orchestrator.ingest_distributed_xids(
                normalized, decisions
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        notification = ctx.advisory_notifications.try_preview(incident.incident_id)
        linked = []
        for event, decision in zip(prepared, decisions, strict=True):
            investigatory = ctx.advisory_notifications.preview_xid_investigatory(
                incident.incident_id, event, decision
            )
            ctx.advisory_notifications.send(investigatory.notification_id)
            decision = decision.model_copy(
                update={
                    "marker": decision.marker.model_copy(
                        update={"incident_id": incident.incident_id}
                    ),
                    "incident_id": incident.incident_id,
                    "workflow_request_id": workflow.request_id,
                    "advisory_notification_id": (
                        notification.notification_id if notification else None
                    ),
                    "investigatory_notification_id": (investigatory.notification_id),
                }
            )
            ctx.completion.add_marker(decision.marker)
            linked.append(decision)
        return DistributedXidIngestionResult(
            batch_id=batch.batch_id,
            decisions=linked,
            incident=incident,
            workflow=workflow,
        )

    return await _store_call(dependencies, ingest)


@router.post(
    "/v1/gpu-events/sxid",
    response_model=FaultPolicyDecision,
)
@authorization_bucket("cluster-token")
async def evaluate_sxid(
    event: SxidEvent,
    dependencies: GpuEventRouterDependencies = Depends(get_gpu_event_dependencies),
) -> FaultPolicyDecision:
    return await _store_call(dependencies, dependencies.ingest_sxid, event)


def _ingest_hma(
    normalized,
    dependencies: GpuEventRouterDependencies,
) -> HmaIngestionResult:
    normalized = normalized.model_copy(
        update={
            "sxid_events": [
                dependencies.enrich_sxid_scope(item) for item in normalized.sxid_events
            ]
        }
    )
    result = HmaIngestionResult(
        normalized=normalized,
        decisions=[
            *(dependencies.ingest_xid(item) for item in normalized.xid_events),
            *(dependencies.ingest_sxid(item) for item in normalized.sxid_events),
        ],
    )
    dependencies.context.dispatcher.wake()
    return result


@router.post(
    "/v1/provider-events/hyperpod-hma/cloudwatch",
    response_model=HmaIngestionResult,
)
@authorization_bucket("cluster-token")
async def ingest_hma_cloudwatch(
    event: HmaCloudWatchLogEvent,
    dependencies: GpuEventRouterDependencies = Depends(get_gpu_event_dependencies),
) -> HmaIngestionResult:
    return await _store_call(
        dependencies,
        lambda: _ingest_hma(
            dependencies.context.hma.normalize_cloudwatch(event),
            dependencies,
        ),
    )


@router.post(
    "/v1/provider-events/hyperpod-hma/node",
    response_model=HmaIngestionResult,
)
@authorization_bucket("cluster-token")
async def ingest_hma_node(
    snapshot: HmaNodeSnapshot,
    dependencies: GpuEventRouterDependencies = Depends(get_gpu_event_dependencies),
) -> HmaIngestionResult:
    return await _store_call(
        dependencies,
        lambda: _ingest_hma(
            dependencies.context.hma.normalize_node(snapshot),
            dependencies,
        ),
    )


@router.post(
    "/v1/provider-events/hyperpod-hma/kubernetes-node",
    response_model=HmaIngestionResult,
)
@authorization_bucket("cluster-token")
async def ingest_hma_kubernetes_node(
    event: HmaKubernetesNodeEvent,
    dependencies: GpuEventRouterDependencies = Depends(get_gpu_event_dependencies),
) -> HmaIngestionResult:
    return await _store_call(
        dependencies,
        lambda: _ingest_hma(
            dependencies.context.hma.normalize_kubernetes_node(event),
            dependencies,
        ),
    )


@router.post(
    "/v1/provider-events/hyperpod-hma/discovery",
    response_model=HmaDeploymentDiscovery,
)
@authorization_bucket("cluster-token")
async def discover_hma(
    probe: HmaDeploymentProbe,
    dependencies: GpuEventRouterDependencies = Depends(get_gpu_event_dependencies),
) -> HmaDeploymentDiscovery:
    return await _store_call(
        dependencies,
        dependencies.context.hma.discover_deployment,
        probe.daemonset,
        probe.services,
    )
