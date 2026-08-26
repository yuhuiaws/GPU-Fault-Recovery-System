from __future__ import annotations

from gpu_fault.app.authorization import authorization_bucket

from dataclasses import dataclass
from typing import Any, Callable, cast

from fastapi import APIRouter, Depends, HTTPException

from gpu_fault.async_store import (
    AsyncStoreExecutor,
    StoreIoCapacityExceeded,
)
from gpu_fault.capabilities import compile_runtime_profile
from gpu_fault.installation_resources import (
    InstallationResource,
    InstallationResourceSnapshot,
)
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


@router.post(  # type: ignore[untyped-decorator]
    "/v1/installation-resources/sync",
    response_model=list[InstallationResource],
)
@authorization_bucket("execution-token")
async def sync_installation_resources(
    snapshot: InstallationResourceSnapshot,
    dependencies: ConfigurationRouterDependencies = Depends(
        get_configuration_dependencies
    ),
) -> list[InstallationResource]:
    saved: list[InstallationResource] = []
    for resource in snapshot.resources:
        saved.append(
            cast(
                InstallationResource,
                await _store_call(
                    dependencies,
                    dependencies.context.store.save_installation_resource,
                    resource,
                ),
            )
        )
    return saved


@router.get(  # type: ignore[untyped-decorator]
    "/v1/installation-resources",
    response_model=list[InstallationResource],
)
@authorization_bucket("execution-token")
async def list_installation_resources(
    site_id: str | None = None,
    dependencies: ConfigurationRouterDependencies = Depends(
        get_configuration_dependencies
    ),
) -> list[InstallationResource]:
    return cast(
        list[InstallationResource],
        await _store_call(
            dependencies,
            dependencies.context.store.list_installation_resources,
            site_id,
        ),
    )


@router.put(  # type: ignore[untyped-decorator]
    "/v1/installation-resources/{site_id}/{resource_key:path}",
    response_model=InstallationResource,
)
@authorization_bucket("execution-token")
async def update_installation_resource(
    site_id: str,
    resource_key: str,
    resource: InstallationResource,
    dependencies: ConfigurationRouterDependencies = Depends(
        get_configuration_dependencies
    ),
) -> InstallationResource:
    if resource.site_id != site_id or resource.resource_key != resource_key:
        raise HTTPException(
            status_code=409, detail="installation resource identity mismatch"
        )
    return cast(
        InstallationResource,
        await _store_call(
            dependencies,
            dependencies.context.store.save_installation_resource,
            resource,
        ),
    )
