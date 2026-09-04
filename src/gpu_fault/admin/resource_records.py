from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

from gpu_fault.installation_resources import (
    InstallationResource,
    InstallationResourceDeletePolicy,
    InstallationResourceOwnership,
    InstallationResourceStatus,
)


def ownership(
    value: object,
    *,
    default: InstallationResourceOwnership = InstallationResourceOwnership.EXTERNAL,
) -> InstallationResourceOwnership:
    try:
        result = InstallationResourceOwnership(str(value))
    except ValueError:
        return default
    if result is InstallationResourceOwnership.REUSED:
        return InstallationResourceOwnership.CREATED
    return result


def foundation_ownership(value: object) -> InstallationResourceOwnership:
    try:
        result = InstallationResourceOwnership(str(value))
    except ValueError:
        return InstallationResourceOwnership.EXTERNAL
    if result is InstallationResourceOwnership.REUSED:
        return InstallationResourceOwnership.EXTERNAL
    return result


def policy(
    value: InstallationResourceOwnership,
    *,
    created: InstallationResourceDeletePolicy = (
        InstallationResourceDeletePolicy.DELETE
    ),
) -> InstallationResourceDeletePolicy:
    if value is InstallationResourceOwnership.CREATED:
        return created
    return InstallationResourceDeletePolicy.PRESERVE


def record(
    *,
    site_id: str,
    resource_key: str,
    resource_type: str,
    resource_id: str,
    ownership: InstallationResourceOwnership,
    delete_policy: InstallationResourceDeletePolicy,
    provider: str = "aws",
    resource_arn: str | None = None,
    region: str | None = None,
    account_id: str | None = None,
    dependencies: Iterable[str] = (),
    attributes: Mapping[str, object] | None = None,
) -> InstallationResource:
    now = datetime.now(timezone.utc)
    return InstallationResource(
        site_id=site_id,
        resource_key=resource_key,
        provider=provider,
        resource_type=resource_type,
        resource_id=resource_id,
        resource_arn=resource_arn,
        region=region,
        account_id=account_id,
        ownership=ownership,
        delete_policy=delete_policy,
        status=InstallationResourceStatus.ACTIVE,
        dependencies=list(dependencies),
        attributes={
            str(key): str(item)
            for key, item in (attributes or {}).items()
            if item is not None and str(item)
        },
        created_at=now,
        updated_at=now,
    )


def release_repository_resources(
    *,
    site_id: str,
    region: str,
    account_id: str,
    state: Mapping[str, Any],
) -> list[InstallationResource]:
    repositories = state.get("release_repositories") or {}
    resources = []
    for name in ("runtime", "cache"):
        value = repositories.get(name) or {}
        repository_name = value.get("repository_name")
        if not repository_name:
            continue
        resource_ownership = ownership(value.get("ownership"))
        resources.append(
            record(
                site_id=site_id,
                resource_key=f"aws/ecr/{name}",
                resource_type="ecr_repository",
                resource_id=str(repository_name),
                resource_arn=(
                    str(value["repository_arn"])
                    if value.get("repository_arn")
                    else None
                ),
                region=region,
                account_id=account_id,
                ownership=resource_ownership,
                delete_policy=policy(resource_ownership),
                attributes={"purpose": value.get("purpose") or name},
            )
        )
    return resources
