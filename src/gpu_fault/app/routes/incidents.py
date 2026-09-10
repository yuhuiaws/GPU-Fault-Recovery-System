from __future__ import annotations

import secrets
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Sequence

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import Field

from gpu_fault.app.authorization import EXECUTION_TOKEN_HEADER, authorization_bucket
from gpu_fault.async_store import (
    AsyncStoreExecutor,
    StoreIoCapacityExceeded,
)
from gpu_fault.models import (
    AdvisoryNotification,
    FaultIncident,
    IncidentState,
    NotificationDispatchReport,
    NotificationDispatchRequest,
    NotificationResult,
    RecoveryAction,
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


INCIDENT_LIST_DEFAULT_LIMIT = 200
INCIDENT_LIST_MAX_LIMIT = 1000
INCIDENT_LIST_FIRST_REASON_CHARS = 200


class IncidentSummary(StrictModel):
    """One row of ``GET /v1/incidents``: what an operator needs to pick an
    incident out of a queue, without the full record."""

    incident_id: str
    state: IncidentState
    cluster_id: str
    node_ids: list[str]
    created_at: datetime
    updated_at: datetime
    event_type: str
    official_action: str | None = None
    effective_action: RecoveryAction | None = None
    workflow_request_id: str | None = None
    # The first recorded reason, cut to INCIDENT_LIST_FIRST_REASON_CHARS.
    first_reason: str | None = None
    reasons_count: int


class IncidentListResponse(StrictModel):
    incidents: list[IncidentSummary]
    # True when more incidents matched than ``limit`` allowed to return.
    truncated: bool
    states: list[IncidentState]
    # The clusters that were searched: the one named, else every registered one.
    cluster_ids: list[str]
    limit: int


def _summarize_incident(incident: FaultIncident) -> IncidentSummary:
    first_reason = incident.reasons[0] if incident.reasons else None
    if (
        first_reason is not None
        and len(first_reason) > INCIDENT_LIST_FIRST_REASON_CHARS
    ):
        first_reason = first_reason[: INCIDENT_LIST_FIRST_REASON_CHARS - 1] + "\u2026"
    return IncidentSummary(
        incident_id=incident.incident_id,
        state=incident.state,
        cluster_id=incident.cluster_id,
        node_ids=list(incident.node_ids),
        created_at=incident.created_at,
        updated_at=incident.updated_at,
        event_type=incident.event_type,
        official_action=incident.official_action,
        effective_action=incident.effective_action,
        workflow_request_id=incident.workflow_request_id,
        first_reason=first_reason,
        reasons_count=len(incident.reasons),
    )


def list_incidents_across_clusters(
    store: Any,
    *,
    cluster_ids: Sequence[str],
    states: Sequence[IncidentState],
    node_ids: set[str] | None,
    limit: int,
) -> tuple[list[FaultIncident], bool]:
    """``list_incidents_by_state`` over several clusters, merged newest
    ``updated_at`` first and cut to ``limit``; the flag says whether the cut
    dropped anything. Runs on the store executor: one indexed read per cluster,
    asking for ``limit + 1`` rows so truncation is detected without a count."""

    merged: list[FaultIncident] = []
    for cluster_id in cluster_ids:
        merged.extend(
            store.list_incidents_by_state(
                cluster_id, states, node_ids=node_ids, limit=limit + 1
            )
        )
    merged.sort(key=lambda item: (item.updated_at, item.incident_id), reverse=True)
    return merged[:limit], len(merged) > limit


def parse_incident_states(raw: Sequence[str]) -> list[IncidentState]:
    """``state`` query values, repeatable and/or comma-separated, in the order
    given without duplicates; 422 for an unknown value or none at all."""

    states: list[IncidentState] = []
    for chunk in raw:
        for token in str(chunk).split(","):
            token = token.strip()
            if not token:
                continue
            try:
                state = IncidentState(token)
            except ValueError:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"unknown incident state {token!r}; one of "
                        + ", ".join(item.value for item in IncidentState)
                    ),
                ) from None
            if state not in states:
                states.append(state)
    if not states:
        raise HTTPException(
            status_code=422,
            detail="query parameter 'state' is required (repeatable or comma-separated)",
        )
    return states


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
    "/v1/incidents",
    response_model=IncidentListResponse,
)
@authorization_bucket("execution-token")
async def list_incidents(
    state: list[str] = Query(
        default=[],
        description=(
            "IncidentState values to list; repeatable or comma-separated. "
            "Required: the route never dumps every incident."
        ),
    ),
    cluster_id: str | None = Query(default=None),
    node_id: str | None = Query(default=None),
    limit: int = Query(
        default=INCIDENT_LIST_DEFAULT_LIMIT, ge=1, le=INCIDENT_LIST_MAX_LIMIT
    ),
    dependencies: IncidentRouterDependencies = Depends(get_incident_dependencies),
) -> IncidentListResponse:
    """List the incidents of one or every registered cluster in the given
    states, newest ``updated_at`` first.

    The operator face of a queue the alerts only count: with
    ``GpuFaultIncidentsAwaitingOperator`` firing, ``?state=ESCALATED`` is the
    list of ids to read and close, replacing the escalation e-mails and the raw
    SQL that stood in for it. Without ``cluster_id`` every cluster in the
    regional registry is searched; ``truncated`` says the ``limit`` cut the
    merged list.
    """

    states = parse_incident_states(state)
    store = dependencies.context.store
    if cluster_id is not None and cluster_id.strip():
        cluster_ids = [cluster_id.strip()]
    else:
        cluster_ids = list(
            await _store_call(dependencies, store.list_regional_cluster_ids)
        )
    node_ids = {node_id.strip()} if node_id and node_id.strip() else None
    incidents, truncated = await _store_call(
        dependencies,
        list_incidents_across_clusters,
        store,
        cluster_ids=cluster_ids,
        states=states,
        node_ids=node_ids,
        limit=limit,
    )
    return IncidentListResponse(
        incidents=[_summarize_incident(incident) for incident in incidents],
        truncated=truncated,
        states=states,
        cluster_ids=cluster_ids,
        limit=limit,
    )


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

    ESCALATED only, on purpose. A QUARANTINED incident is closable solely on
    node isolation evidence (``NodeIsolationEvidence``: no cordon, no
    quarantine taint of this incident, no isolation annotation of it), and
    this API has no kubeconfig to read the nodes, so it never accepts such
    evidence from a caller. That close goes through ``gpu-fault-admin
    workflow-reconcile --close-quarantined`` (or ``--close-incident`` on a
    QUARANTINED id), which reads the evidence through the site's GPU
    kubeconfig and calls the same service function with it.
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
