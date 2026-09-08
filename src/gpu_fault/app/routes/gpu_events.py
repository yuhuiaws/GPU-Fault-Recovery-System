from __future__ import annotations

from gpu_fault.app.authorization import authorization_bucket

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from fastapi import APIRouter, Depends, Header, HTTPException

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
    authenticated_cluster: str | None = Header(
        default=None,
        alias="X-GPU-Fault-Cluster-ID",
    ),
    dependencies: GpuEventRouterDependencies = Depends(get_gpu_event_dependencies),
) -> dict:
    def lookup() -> dict:
        ctx = dependencies.context
        # The correlation is addressed by a global event id, so its owning
        # cluster must be checked against the caller before any record is
        # returned; the XID event carries the cluster_id (the correlation row
        # does not) and shares the event id one-to-one.
        event = ctx.store.get_xid_event(event_id)
        if (
            ctx.regional_mode
            and authenticated_cluster
            and event.cluster_id != authenticated_cluster
        ):
            raise HTTPException(
                status_code=403,
                detail=(
                    "authenticated cluster cannot read an XID correlation "
                    "from another cluster"
                ),
            )
        correlation = ctx.store.get_xid_correlation(event_id)
        decision = ctx.store.get_xid_policy_decision(event_id)
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
    # An HMA fault the normalizer could not read (a node cordoned for
    # EfaError/InstanceUnreachable, an Xid line whose format drifted) used to
    # be a silent 200: no decision, no counter, no finding. It now takes the
    # same route the kernel and Fabric Manager collectors take
    # (``collector_events.py``), so the cordon becomes a WARNING
    # operator-review finding instead of nothing (F4). The service is reached
    # through the context -- the same handle ``/metrics`` renders
    # ``unresolved_signal_totals`` from -- because this router's dependencies
    # are assembled per request and the fault ingestion service is a
    # process-lifetime object.
    fault_ingestion = getattr(dependencies.context, "fault_ingestion", None)
    if fault_ingestion is not None and normalized.provider_signals:
        fault_ingestion.ingest_unresolved_signals(
            normalized,
            batch_id=normalized.provider_signals[0].signal_id,
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
