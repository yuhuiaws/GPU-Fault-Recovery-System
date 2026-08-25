from __future__ import annotations

from gpu_fault.app.authorization import authorization_bucket

import secrets
from dataclasses import dataclass
from typing import Any, Callable

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import JSONResponse, Response

from gpu_fault.async_store import (
    AsyncStoreExecutor,
    StoreIoCapacityExceeded,
)
from gpu_fault.processor import ProcessorRequestStatus


@dataclass(frozen=True)
class ProcessorRouterDependencies:
    context: Any
    processor: Any | None
    processor_mode: str
    store_io: AsyncStoreExecutor
    telemetry_spool_store_io: AsyncStoreExecutor
    evidence_store_io: AsyncStoreExecutor
    diagnostics: Callable[[], dict]
    diagnostics_publisher: Any
    max_queue_depth: int
    max_cluster_queue_depth: int
    fault_reserved_queue_depth: int
    fault_reserved_cluster_depth: int
    max_request_bytes: int
    global_admission_guard: int


def get_processor_dependencies() -> ProcessorRouterDependencies:
    raise RuntimeError("processor router dependencies are not configured")


router = APIRouter(prefix="/v1/processor", tags=["processor"])


@router.get("/status")
@authorization_bucket("execution-token")
async def processor_status(
    execution_token: str | None = Header(
        default=None,
        alias="X-GPU-Fault-Execution-Token",
    ),
    dependencies: ProcessorRouterDependencies = Depends(get_processor_dependencies),
) -> dict:
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
    try:
        leadership = (
            None
            if dependencies.processor is not None
            and dependencies.processor.active_consumers
            else await dependencies.store_io.run(ctx.store.get_processor_leadership)
        )
        queue = await dependencies.store_io.run(ctx.store.processor_queue_stats)
    except StoreIoCapacityExceeded as exc:
        raise HTTPException(
            status_code=503,
            detail="store I/O capacity exceeded",
            headers={"Retry-After": "2"},
        ) from exc
    processor = dependencies.processor
    local_diagnostics = dependencies.diagnostics()
    return {
        "mode": dependencies.processor_mode,
        "local_owner_id": (processor.owner_id if processor is not None else None),
        "local_role": (
            "active-consumer"
            if processor is not None and processor.active_consumers
            else "leader"
            if processor is not None and processor.is_leader()
            else "standby"
            if processor is not None
            else "active"
        ),
        "leadership": (
            leadership.model_dump(mode="json") if leadership is not None else None
        ),
        "queue": queue,
        "limits": {
            "global_depth": dependencies.max_queue_depth,
            "cluster_depth": dependencies.max_cluster_queue_depth,
            "fault_reserved_global_depth": (dependencies.fault_reserved_queue_depth),
            "fault_reserved_cluster_depth": (dependencies.fault_reserved_cluster_depth),
            "max_request_bytes": dependencies.max_request_bytes,
            "global_admission_guard": (dependencies.global_admission_guard),
        },
        "store_io": _executor_status(dependencies.store_io),
        "telemetry_spool_store_io": _executor_status(
            dependencies.telemetry_spool_store_io
        ),
        "evidence_store_io": _executor_status(dependencies.evidence_store_io),
        **local_diagnostics,
        "pod_processes": dependencies.diagnostics_publisher.read_all(),
    }


def _executor_status(executor: AsyncStoreExecutor) -> dict:
    return {
        "workers": executor.workers,
        "max_in_flight": executor.max_in_flight,
        "in_flight": executor.in_flight,
        "rejected_total": executor.rejected_total,
    }


@router.get("/requests/{request_id}")
@authorization_bucket("cluster-token")
async def processor_request_receipt(
    request_id: str,
    authenticated_cluster: str | None = Header(
        default=None,
        alias="X-GPU-Fault-Cluster-ID",
    ),
    dependencies: ProcessorRouterDependencies = Depends(get_processor_dependencies),
):
    try:
        current = await dependencies.store_io.run(
            dependencies.context.store.get_processor_request,
            request_id,
        )
    except StoreIoCapacityExceeded as exc:
        raise HTTPException(
            status_code=503,
            detail="store I/O capacity exceeded",
            headers={"Retry-After": "2"},
        ) from exc
    if (
        dependencies.context.regional_mode
        and current.cluster_id != authenticated_cluster
    ):
        raise HTTPException(
            status_code=403,
            detail=(
                "authenticated cluster cannot read another cluster's processor request"
            ),
        )
    if current.status is not ProcessorRequestStatus.COMPLETED:
        return JSONResponse(
            status_code=202,
            headers={"Cache-Control": "no-store"},
            content={
                "processor_request_id": current.request_id,
                "status": current.status.value,
            },
        )
    headers = {
        "Cache-Control": "no-store",
        "X-GPU-Fault-Processor-Request-ID": current.request_id,
        "X-GPU-Fault-Processor-Status": "COMPLETED",
    }
    if current.response_content_type:
        headers["Content-Type"] = current.response_content_type
    return Response(
        content=current.response_body(),
        status_code=current.response_status or 500,
        headers=headers,
    )
