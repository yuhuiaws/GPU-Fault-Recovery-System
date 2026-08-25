from __future__ import annotations

from gpu_fault.app.authorization import authorization_bucket

import secrets
from dataclasses import dataclass
from typing import Any, Callable

from fastapi import APIRouter, Depends, Header, HTTPException

from gpu_fault.async_store import (
    AsyncStoreExecutor,
    StoreIoCapacityExceeded,
)
from gpu_fault.fleet import (
    AgentRecord,
    AgentTransitionRequest,
    DeploymentNodeUpdate,
    DeploymentWaveLease,
    FleetDeployment,
    FleetDeploymentRequest,
    FleetReadinessReport,
    FleetReadinessRequest,
    MultiNodeBarrier,
    SignedAgentHeartbeat,
)
from gpu_fault.store import NotFoundError


@dataclass(frozen=True)
class FleetRouterDependencies:
    context: Any
    store_io: AsyncStoreExecutor


def get_fleet_dependencies() -> FleetRouterDependencies:
    raise RuntimeError("fleet router dependencies are not configured")


router = APIRouter(prefix="/v1/fleet", tags=["fleet"])


async def _store_call(
    dependencies: FleetRouterDependencies,
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


def _require_execution_token(
    expected: str | None,
    supplied: str | None,
) -> None:
    if not expected or not supplied or not secrets.compare_digest(supplied, expected):
        raise HTTPException(
            status_code=403,
            detail="invalid workflow execution token",
        )


@router.post("/agents/heartbeat", response_model=AgentRecord)
@authorization_bucket("cluster-token")
async def register_agent(
    envelope: SignedAgentHeartbeat,
    dependencies: FleetRouterDependencies = Depends(get_fleet_dependencies),
) -> AgentRecord:
    registry = dependencies.context.fleet_registry
    if registry is None:
        raise HTTPException(
            status_code=503,
            detail="agent registry is disabled",
        )
    try:
        return await _store_call(dependencies, registry.register, envelope)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/agents", response_model=list[AgentRecord])
@authorization_bucket("dual-credential")
async def list_agents(
    cluster_id: str | None = None,
    authenticated_cluster: str | None = Header(
        default=None,
        alias="X-GPU-Fault-Cluster-ID",
    ),
    dependencies: FleetRouterDependencies = Depends(get_fleet_dependencies),
) -> list[AgentRecord]:
    ctx = dependencies.context
    if ctx.fleet_registry is None:
        return []
    if ctx.regional_mode and authenticated_cluster:
        if cluster_id and cluster_id != authenticated_cluster:
            raise HTTPException(
                status_code=403,
                detail=("authenticated cluster cannot read agents for another cluster"),
            )
        cluster_id = authenticated_cluster
    return await _store_call(dependencies, ctx.store.list_agents, cluster_id)


@router.get(
    "/agents/{cluster_id}/{node_id}",
    response_model=AgentRecord,
)
@authorization_bucket("dual-credential")
async def get_agent(
    cluster_id: str,
    node_id: str,
    authenticated_cluster: str | None = Header(
        default=None,
        alias="X-GPU-Fault-Cluster-ID",
    ),
    dependencies: FleetRouterDependencies = Depends(get_fleet_dependencies),
) -> AgentRecord:
    ctx = dependencies.context
    if (
        ctx.regional_mode
        and authenticated_cluster
        and authenticated_cluster != cluster_id
    ):
        raise HTTPException(
            status_code=403,
            detail=("authenticated cluster cannot read an agent from another cluster"),
        )
    return await _store_call(
        dependencies,
        ctx.store.get_agent,
        cluster_id,
        node_id,
    )


async def _transition(
    operation: str,
    cluster_id: str,
    node_id: str,
    request: AgentTransitionRequest,
    execution_token: str | None,
    dependencies: FleetRouterDependencies,
) -> AgentRecord:
    ctx = dependencies.context
    _require_execution_token(ctx.execution_token, execution_token)
    if ctx.fleet_registry is None:
        raise HTTPException(
            status_code=503,
            detail="agent registry is disabled",
        )
    try:
        handler = getattr(ctx.fleet_registry, f"{operation}_agent")
        return await _store_call(
            dependencies,
            handler,
            cluster_id,
            node_id,
            request,
        )
    except (NotFoundError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post(
    "/agents/{cluster_id}/{node_id}/drain",
    response_model=AgentRecord,
)
@authorization_bucket("dual-credential")
async def drain_agent(
    cluster_id: str,
    node_id: str,
    request: AgentTransitionRequest,
    execution_token: str | None = Header(
        default=None,
        alias="X-GPU-Fault-Execution-Token",
    ),
    dependencies: FleetRouterDependencies = Depends(get_fleet_dependencies),
) -> AgentRecord:
    return await _transition(
        "drain",
        cluster_id,
        node_id,
        request,
        execution_token,
        dependencies,
    )


@router.post(
    "/agents/{cluster_id}/{node_id}/revoke",
    response_model=AgentRecord,
)
@authorization_bucket("dual-credential")
async def revoke_agent(
    cluster_id: str,
    node_id: str,
    request: AgentTransitionRequest,
    execution_token: str | None = Header(
        default=None,
        alias="X-GPU-Fault-Execution-Token",
    ),
    dependencies: FleetRouterDependencies = Depends(get_fleet_dependencies),
) -> AgentRecord:
    return await _transition(
        "revoke",
        cluster_id,
        node_id,
        request,
        execution_token,
        dependencies,
    )


@router.post(
    "/agents/{cluster_id}/{node_id}/reactivate",
    response_model=AgentRecord,
)
@authorization_bucket("dual-credential")
async def reactivate_agent(
    cluster_id: str,
    node_id: str,
    request: AgentTransitionRequest,
    execution_token: str | None = Header(
        default=None,
        alias="X-GPU-Fault-Execution-Token",
    ),
    dependencies: FleetRouterDependencies = Depends(get_fleet_dependencies),
) -> AgentRecord:
    return await _transition(
        "reactivate",
        cluster_id,
        node_id,
        request,
        execution_token,
        dependencies,
    )


@router.post("/readiness", response_model=FleetReadinessReport)
@authorization_bucket("dual-credential")
async def fleet_readiness(
    request: FleetReadinessRequest,
    dependencies: FleetRouterDependencies = Depends(get_fleet_dependencies),
) -> FleetReadinessReport:
    registry = dependencies.context.fleet_registry
    if registry is None:
        raise HTTPException(
            status_code=503,
            detail="agent registry is disabled",
        )
    return await _store_call(
        dependencies,
        registry.readiness,
        request.cluster_id,
        request.node_ids,
    )


@router.post("/deployments", response_model=FleetDeployment)
@authorization_bucket("execution-token")
async def create_fleet_deployment(
    request: FleetDeploymentRequest,
    execution_token: str | None = Header(
        default=None,
        alias="X-GPU-Fault-Execution-Token",
    ),
    dependencies: FleetRouterDependencies = Depends(get_fleet_dependencies),
) -> FleetDeployment:
    ctx = dependencies.context
    _require_execution_token(ctx.execution_token, execution_token)
    if ctx.fleet_registry is None:
        raise HTTPException(
            status_code=503,
            detail="agent registry is disabled",
        )
    try:
        return await _store_call(
            dependencies,
            ctx.fleet_registry.create_deployment,
            request,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/deployments", response_model=list[FleetDeployment])
@authorization_bucket("execution-token")
async def list_fleet_deployments(
    dependencies: FleetRouterDependencies = Depends(get_fleet_dependencies),
) -> list[FleetDeployment]:
    return await _store_call(
        dependencies,
        dependencies.context.store.list_fleet_deployments,
    )


@router.get(
    "/deployments/{deployment_id}",
    response_model=FleetDeployment,
)
@authorization_bucket("execution-token")
async def get_fleet_deployment(
    deployment_id: str,
    dependencies: FleetRouterDependencies = Depends(get_fleet_dependencies),
) -> FleetDeployment:
    return await _store_call(
        dependencies,
        dependencies.context.store.get_fleet_deployment,
        deployment_id,
    )


@router.post(
    "/deployments/{deployment_id}/next-wave",
    response_model=DeploymentWaveLease,
)
@authorization_bucket("execution-token")
async def start_fleet_deployment_wave(
    deployment_id: str,
    execution_token: str | None = Header(
        default=None,
        alias="X-GPU-Fault-Execution-Token",
    ),
    dependencies: FleetRouterDependencies = Depends(get_fleet_dependencies),
) -> DeploymentWaveLease:
    ctx = dependencies.context
    _require_execution_token(ctx.execution_token, execution_token)
    if ctx.fleet_registry is None:
        raise HTTPException(
            status_code=503,
            detail="agent registry is disabled",
        )
    try:
        return await _store_call(
            dependencies,
            ctx.fleet_registry.start_next_wave,
            deployment_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post(
    "/deployments/{deployment_id}/nodes/{node_id}",
    response_model=FleetDeployment,
)
@authorization_bucket("execution-token")
async def update_fleet_deployment_node(
    deployment_id: str,
    node_id: str,
    update: DeploymentNodeUpdate,
    execution_token: str | None = Header(
        default=None,
        alias="X-GPU-Fault-Execution-Token",
    ),
    dependencies: FleetRouterDependencies = Depends(get_fleet_dependencies),
) -> FleetDeployment:
    ctx = dependencies.context
    _require_execution_token(ctx.execution_token, execution_token)
    if ctx.fleet_registry is None:
        raise HTTPException(
            status_code=503,
            detail="agent registry is disabled",
        )
    try:
        return await _store_call(
            dependencies,
            ctx.fleet_registry.update_deployment_node,
            deployment_id,
            node_id,
            update,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/barriers", response_model=list[MultiNodeBarrier])
@authorization_bucket("execution-token")
async def list_barriers(
    dependencies: FleetRouterDependencies = Depends(get_fleet_dependencies),
) -> list[MultiNodeBarrier]:
    return await _store_call(
        dependencies,
        dependencies.context.store.list_barriers,
    )


@router.get(
    "/barriers/{barrier_id:path}",
    response_model=MultiNodeBarrier,
)
@authorization_bucket("execution-token")
async def get_barrier(
    barrier_id: str,
    dependencies: FleetRouterDependencies = Depends(get_fleet_dependencies),
) -> MultiNodeBarrier:
    return await _store_call(
        dependencies,
        dependencies.context.store.get_barrier,
        barrier_id,
    )
