from __future__ import annotations

from gpu_fault.app.authorization import authorization_bucket

from dataclasses import dataclass
from typing import Any, Callable

from fastapi import APIRouter, Depends, HTTPException

from gpu_fault.async_store import (
    AsyncStoreExecutor,
    StoreIoCapacityExceeded,
)
from gpu_fault.models import (
    AdvisoryNotification,
    FaultIncident,
    NotificationDispatchReport,
    NotificationDispatchRequest,
    NotificationResult,
)


@dataclass(frozen=True)
class IncidentRouterDependencies:
    context: Any
    store_io: AsyncStoreExecutor
    notification_owner: str
    notification_lease_seconds: int
    notification_max_attempts: int


def get_incident_dependencies() -> IncidentRouterDependencies:
    raise RuntimeError("incident router dependencies are not configured")


router = APIRouter(tags=["incidents"])


async def _store_call(
    dependencies: IncidentRouterDependencies,
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


@router.get(
    "/v1/incidents/{incident_id}",
    response_model=FaultIncident,
)
@authorization_bucket("execution-token")
async def get_incident(
    incident_id: str,
    dependencies: IncidentRouterDependencies = Depends(get_incident_dependencies),
) -> FaultIncident:
    return await _store_call(
        dependencies,
        dependencies.context.store.get_incident,
        incident_id,
    )


@router.post(
    "/v1/incidents/{incident_id}/advisory-notifications",
    response_model=AdvisoryNotification,
)
@authorization_bucket("execution-token")
async def preview_advisory_notification(
    incident_id: str,
    dependencies: IncidentRouterDependencies = Depends(get_incident_dependencies),
) -> AdvisoryNotification:
    return await _store_call(
        dependencies,
        dependencies.context.advisory_notifications.preview,
        incident_id,
    )


@router.get(
    "/v1/advisory-notifications/{notification_id}",
    response_model=AdvisoryNotification,
)
@authorization_bucket("execution-token")
async def get_advisory_notification(
    notification_id: str,
    dependencies: IncidentRouterDependencies = Depends(get_incident_dependencies),
) -> AdvisoryNotification:
    return await _store_call(
        dependencies,
        dependencies.context.store.get_notification,
        notification_id,
    )


@router.post(
    "/v1/advisory-notifications/{notification_id}/send",
    response_model=NotificationResult,
)
@authorization_bucket("execution-token")
async def send_advisory_notification(
    notification_id: str,
    dependencies: IncidentRouterDependencies = Depends(get_incident_dependencies),
) -> NotificationResult:
    return await _store_call(
        dependencies,
        dependencies.context.advisory_notifications.send,
        notification_id,
    )


@router.post(
    "/v1/advisory-notifications/dispatch",
    response_model=NotificationDispatchReport,
)
@authorization_bucket("execution-token")
async def dispatch_advisory_notifications(
    request: NotificationDispatchRequest,
    dependencies: IncidentRouterDependencies = Depends(get_incident_dependencies),
) -> NotificationDispatchReport:
    notifications = dependencies.context.advisory_notifications
    if notifications.async_delivery:
        return await _store_call(
            dependencies,
            notifications.dispatch_outbox,
            dependencies.notification_owner,
            limit=request.limit,
            lease_seconds=dependencies.notification_lease_seconds,
            max_attempts=dependencies.notification_max_attempts,
        )
    return await _store_call(
        dependencies,
        notifications.dispatch_pending,
        request.limit,
    )
