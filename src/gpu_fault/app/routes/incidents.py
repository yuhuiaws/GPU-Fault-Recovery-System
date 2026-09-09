from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import Any, Callable

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import Field

from gpu_fault.app.authorization import EXECUTION_TOKEN_HEADER, authorization_bucket
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
    StrictModel,
)
from gpu_fault.orchestration.incident_closure import IncidentNotClosable
from gpu_fault.store.shared.errors import StaleWriteError


class IncidentCloseRequest(StrictModel):
    """Body of ``POST /v1/incidents/{incident_id}/close``."""

    reason: str = Field(min_length=1, max_length=512)
    operator: str = Field(min_length=1, max_length=256)
    # Approved-change or ticket reference; goes on the audit event.
    reference: str | None = Field(default=None, max_length=128)


class IncidentCloseResponse(StrictModel):
    incident: FaultIncident
    # False when the incident was already RECOVERED and nothing was written.
    closed: bool


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
    "/v1/incidents/{incident_id}/close",
    response_model=IncidentCloseResponse,
)
@authorization_bucket("execution-token")
async def close_incident(
    incident_id: str,
    request: IncidentCloseRequest,
    execution_token: str | None = Header(default=None, alias=EXECUTION_TOKEN_HEADER),
    dependencies: IncidentRouterDependencies = Depends(get_incident_dependencies),
) -> IncidentCloseResponse:
    """Close an ESCALATED incident RECOVERED on an operator's authority.

    The operator exit for an incident whose remediation was handed to a human
    (DESTR-018 product gap): once the node is back, the incident stops
    absorbing the node's faults as record-only. 409 when the incident is not
    ESCALATED or a workflow of it is still open, with the reason in ``detail``;
    idempotent -- an already RECOVERED incident answers 200 with
    ``closed=false``. 404 for an unknown incident. The audit event carries
    ``operator`` and ``reference``.
    """

    # An operator write: the token is checked here, fail-closed, like the
    # workflow router's dispatch/execute -- not left to the bucket inventory.
    expected = dependencies.context.execution_token
    if (
        not expected
        or not execution_token
        or not secrets.compare_digest(execution_token, expected)
    ):
        raise HTTPException(status_code=403, detail="invalid execution token")
    try:
        incident, closed = await _store_call(
            dependencies,
            dependencies.context.incident_closure.close_incident,
            incident_id,
            reason=request.reason,
            operator=request.operator,
            reference=request.reference,
        )
    except IncidentNotClosable as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except StaleWriteError as exc:
        raise HTTPException(
            status_code=409,
            detail=f"incident {incident_id} changed while it was being closed; retry",
        ) from exc
    return IncidentCloseResponse(incident=incident, closed=closed)


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
