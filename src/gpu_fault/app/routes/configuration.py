from __future__ import annotations

from gpu_fault.app.authorization import authorization_bucket

from dataclasses import dataclass
from typing import Any, Callable

from fastapi import APIRouter, Depends, HTTPException

from gpu_fault.async_store import (
    AsyncStoreExecutor,
    StoreIoCapacityExceeded,
)
from gpu_fault.capabilities import compile_runtime_profile
from gpu_fault.models import (
    EffectiveRuntimeProfile,
    NodeMarker,
    RuntimeProfile,
)


@dataclass(frozen=True)
class ConfigurationRouterDependencies:
    context: Any
    store_io: AsyncStoreExecutor


def get_configuration_dependencies() -> ConfigurationRouterDependencies:
    raise RuntimeError("configuration router dependencies are not configured")


router = APIRouter(tags=["configuration"])


async def _store_call(
    dependencies: ConfigurationRouterDependencies,
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
    "/v1/runtime-profiles",
    response_model=EffectiveRuntimeProfile,
)
@authorization_bucket("execution-token")
async def register_profile(
    profile: RuntimeProfile,
    dependencies: ConfigurationRouterDependencies = Depends(
        get_configuration_dependencies
    ),
) -> EffectiveRuntimeProfile:
    ctx = dependencies.context

    def register() -> EffectiveRuntimeProfile:
        if ctx.regional_mode:
            try:
                ctx.store.get_regional_cluster(profile.cluster_id)
            except KeyError as exc:
                raise HTTPException(
                    status_code=403,
                    detail=(
                        "runtime profile cluster_id is not a "
                        "registered regional cluster"
                    ),
                ) from exc
        effective = compile_runtime_profile(profile)
        ctx.store.save_profile(effective)
        return effective

    return await _store_call(dependencies, register)


@router.post("/v1/markers", response_model=NodeMarker)
@authorization_bucket("execution-token")
async def create_marker(
    marker: NodeMarker,
    dependencies: ConfigurationRouterDependencies = Depends(
        get_configuration_dependencies
    ),
) -> NodeMarker:
    return await _store_call(
        dependencies,
        dependencies.context.completion.add_marker,
        marker,
    )
