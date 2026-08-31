from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable

from fastapi import APIRouter, Depends, HTTPException

from gpu_fault.app.authorization import authorization_bucket
from gpu_fault.app.routes.regional import (
    RegionalRouterDependencies,
    get_regional_dependencies,
)
from gpu_fault.async_store import StoreIoCapacityExceeded
from gpu_fault.regional import (
    RegionalRegistryMember,
    RegionalRegistryPublishRequest,
    RegionalRegistryRevision,
    RegionalRegistryRollbackRequest,
    RegionalRegistryStatus,
)
from gpu_fault.regional_registry_runtime import (
    active_registry_member_ids,
    registry_revision_converged,
)


router = APIRouter(prefix="/v1/regional/registry", tags=["regional-registry"])


async def _store_call(
    dependencies: RegionalRouterDependencies,
    function: Callable[..., Any],
    /,
    *args: Any,
    **kwargs: Any,
) -> Any:
    try:
        return await dependencies.store_io.run(function, *args, **kwargs)
    except StoreIoCapacityExceeded as exc:
        raise HTTPException(
            status_code=503,
            detail="store I/O capacity exceeded",
            headers={"Retry-After": "2"},
        ) from exc


async def _publish(
    dependencies: RegionalRouterDependencies,
    revision: RegionalRegistryRevision,
    *,
    expected_generation: int,
) -> None:
    try:
        await _store_call(
            dependencies,
            dependencies.context.store.publish_regional_registry_revision,
            revision,
            expected_generation=expected_generation,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


def _status(
    revision: RegionalRegistryRevision,
    members: list[RegionalRegistryMember],
    *,
    observed_at: datetime,
    stale_seconds: float,
) -> RegionalRegistryStatus:
    active_ids = active_registry_member_ids(
        members,
        observed_at=observed_at,
        stale_seconds=stale_seconds,
    )
    by_id = {member.member_id: member for member in members}
    acked = sorted(
        member_id
        for member_id in revision.required_member_ids
        if (
            (member := by_id.get(member_id)) is not None
            and member.ready
            and member.generation == revision.generation
            and member.content_sha256 == revision.content_sha256
            and member.member_id in active_ids
        )
    )
    missing = sorted(set(revision.required_member_ids) - set(acked))
    return RegionalRegistryStatus(
        generation=revision.generation,
        content_sha256=revision.content_sha256,
        cluster_states={
            item.cluster_id: item.lifecycle_state for item in revision.registrations
        },
        required_member_ids=revision.required_member_ids,
        acked_member_ids=acked,
        missing_member_ids=missing,
        active_member_ids=active_ids,
        members=members,
        converged=registry_revision_converged(
            revision,
            members,
            observed_at=observed_at,
            stale_seconds=stale_seconds,
        ),
    )


async def _current_status(
    dependencies: RegionalRouterDependencies,
) -> RegionalRegistryStatus:
    runtime = dependencies.auth_registry
    head = await _store_call(
        dependencies,
        dependencies.context.store.get_regional_registry_head,
    )
    revision = await _store_call(
        dependencies,
        dependencies.context.store.get_regional_registry_revision,
        head.generation,
    )
    members = await _store_call(
        dependencies,
        dependencies.context.store.list_regional_registry_members,
    )
    return _status(
        revision,
        members,
        observed_at=datetime.now(timezone.utc),
        stale_seconds=runtime.stale_seconds,
    )


@router.get(  # type: ignore[untyped-decorator]
    "/status",
    response_model=RegionalRegistryStatus,
)
@authorization_bucket("execution-token")
async def regional_registry_status(
    dependencies: RegionalRouterDependencies = Depends(get_regional_dependencies),
) -> RegionalRegistryStatus:
    return await _current_status(dependencies)


@router.post(  # type: ignore[untyped-decorator]
    "/revisions",
    response_model=RegionalRegistryStatus,
)
@authorization_bucket("execution-token")
async def publish_regional_registry_revision(
    request: RegionalRegistryPublishRequest,
    dependencies: RegionalRouterDependencies = Depends(get_regional_dependencies),
) -> RegionalRegistryStatus:
    runtime = dependencies.auth_registry
    members = await _store_call(
        dependencies,
        dependencies.context.store.list_regional_registry_members,
    )
    observed = datetime.now(timezone.utc)
    required = (
        request.required_member_ids
        if request.required_member_ids is not None
        else active_registry_member_ids(
            members,
            observed_at=observed,
            stale_seconds=runtime.stale_seconds,
        )
    )
    revision = RegionalRegistryRevision.build(
        generation=request.expected_generation + 1,
        registrations=request.registrations,
        previous_generation=(
            request.expected_generation if request.expected_generation else None
        ),
        required_member_ids=required,
        reason=request.reason,
        created_at=observed,
    )
    await _publish(
        dependencies,
        revision,
        expected_generation=request.expected_generation,
    )
    await _store_call(
        dependencies,
        runtime.refresh_once,
        raise_on_failure=True,
    )
    return await _current_status(dependencies)


@router.post(  # type: ignore[untyped-decorator]
    "/rollback",
    response_model=RegionalRegistryStatus,
)
@authorization_bucket("execution-token")
async def rollback_regional_registry_revision(
    request: RegionalRegistryRollbackRequest,
    dependencies: RegionalRouterDependencies = Depends(get_regional_dependencies),
) -> RegionalRegistryStatus:
    target = await _store_call(
        dependencies,
        dependencies.context.store.get_regional_registry_revision,
        request.target_generation,
    )
    members = await _store_call(
        dependencies,
        dependencies.context.store.list_regional_registry_members,
    )
    runtime = dependencies.auth_registry
    observed = datetime.now(timezone.utc)
    revision = RegionalRegistryRevision.build(
        generation=request.expected_generation + 1,
        registrations=target.registrations,
        previous_generation=request.expected_generation,
        required_member_ids=active_registry_member_ids(
            members,
            observed_at=observed,
            stale_seconds=runtime.stale_seconds,
        ),
        reason=request.reason,
        created_at=observed,
    )
    await _publish(
        dependencies,
        revision,
        expected_generation=request.expected_generation,
    )
    await _store_call(
        dependencies,
        runtime.refresh_once,
        raise_on_failure=True,
    )
    return await _current_status(dependencies)
