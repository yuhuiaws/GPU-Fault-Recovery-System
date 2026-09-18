"""Plan and execute removal of exclusively owned installation resources."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from datetime import datetime, timezone
from typing import Any, Callable

from gpu_fault.admin.aws_cleanup import ResourceCleaner
from gpu_fault.admin.bootstrap_common import Arn, BootstrapError
from gpu_fault.admin.cluster_removal_network import NetworkRemovalRequest
from gpu_fault.admin.diagnostics import diagnostic_text
from gpu_fault.admin.resource_registry_dns import (
    find_vpc_association_resource,
    network_vpc_identity,
)
from gpu_fault.admin.site import RenderedSite
from gpu_fault.installation_resources import (
    InstallationResource,
    InstallationResourceDeletePolicy,
    InstallationResourceSnapshot,
    InstallationResourceStatus,
    TERMINAL_INSTALLATION_RESOURCE_STATUSES,
)


def target_resource(resource: InstallationResource, cluster_id: str) -> bool:
    key = resource.resource_key
    return (
        key.startswith(f"aws/iam/executor/{cluster_id}/")
        or key.startswith(f"aws/iam/adot-writer/{cluster_id}/")
        or key.startswith(f"cluster/{cluster_id}/")
    )


def association_to_detach(
    site: RenderedSite,
    snapshot: InstallationResourceSnapshot,
    *,
    target_network: dict[str, Any],
    remaining_networks: list[dict[str, Any]],
    cpu_vpc_id: str,
) -> InstallationResource | None:
    """Authorize only an exclusive association with an exact immutable DETACH row."""

    snapshot.require_source_binding()
    if snapshot.site_id != site.registry_site_id:
        raise BootstrapError("installation registry belongs to another site")
    config = site.release_config
    region = str(config["aws_region"])
    target = network_vpc_identity(target_network, region=region)
    shared = {
        network_vpc_identity(item, region=region) for item in remaining_networks
    } | {(region, cpu_vpc_id)}
    zone = str((config.get("dns") or {}).get("hosted_zone_id") or "")
    if not zone or target in shared:
        return None
    if target[0] != region:
        raise BootstrapError("target Route53 association is outside the site Region")
    account = Arn.parse(str(config["cpu_eks_arn"])).account
    resource = find_vpc_association_resource(
        snapshot.resources,
        site_id=site.registry_site_id,
        hosted_zone_id=zone,
        vpc_region=target[0],
        vpc_id=target[1],
        region=region,
        account_id=account,
    )
    if resource is None:
        raise BootstrapError(
            "target Route53 association has no registry ownership; reconcile it"
        )
    if resource.delete_policy is InstallationResourceDeletePolicy.PRESERVE:
        return None
    if resource.status is not InstallationResourceStatus.ACTIVE:
        raise BootstrapError("target Route53 association registry state is not ACTIVE")
    return resource


def detach_registered_network(
    request: NetworkRemovalRequest,
    snapshot: InstallationResourceSnapshot,
    *,
    target_network: dict[str, Any],
    remaining_networks: list[dict[str, Any]],
    cpu_vpc_id: str,
    detach: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    association = association_to_detach(
        request.site,
        snapshot,
        target_network=target_network,
        remaining_networks=remaining_networks,
        cpu_vpc_id=cpu_vpc_id,
    )
    result = detach(
        request,
        target_network=target_network,
        remaining_networks=remaining_networks,
        cpu_vpc_id=cpu_vpc_id,
        detach_dns=association is not None,
    )
    expected_vpc = association.attributes["vpc_id"] if association else None
    if result.get("detached_vpc_id") != expected_vpc:
        raise BootstrapError("Route53 detach evidence differs from its authorization")
    return result


def deletion_waves(
    resources: list[InstallationResource],
) -> list[list[InstallationResource]]:
    remaining = {resource.resource_key: resource for resource in resources}
    waves = []
    while remaining:
        dependencies = {
            dependency
            for resource in remaining.values()
            for dependency in resource.dependencies
            if dependency in remaining
        }
        wave = [
            resource
            for key, resource in sorted(remaining.items())
            if key not in dependencies
        ]
        if not wave:
            raise BootstrapError(
                "installation registry target resources contain a dependency cycle"
            )
        waves.append(wave)
        for resource in wave:
            remaining.pop(resource.resource_key)
    return waves


def target_resource_plan(
    site: RenderedSite,
    cluster_id: str,
    snapshot: InstallationResourceSnapshot,
    *,
    detached_vpc_id: str | None = None,
) -> tuple[list[InstallationResource], list[list[InstallationResource]]]:
    selected = [
        resource
        for resource in snapshot.resources
        if target_resource(resource, cluster_id)
        and resource.resource_type != "route53_vpc_association"
    ]
    # A VPC association outlives the member whose join first registered it.
    if detached_vpc_id:
        zone = str((site.release_config.get("dns") or {}).get("hosted_zone_id") or "")
        region = str(site.release_config["aws_region"])
        association = find_vpc_association_resource(
            snapshot.resources,
            site_id=site.registry_site_id,
            hosted_zone_id=zone,
            vpc_id=detached_vpc_id,
            vpc_region=region,
            region=region,
            account_id=Arn.parse(str(site.release_config["cpu_eks_arn"])).account,
        )
        if association is None:
            raise BootstrapError(
                "detached Route53 association has no registry identity"
            )
        selected.append(association)
    roles = [
        resource.resource_key
        for resource in selected
        if resource.resource_type == "iam_role"
    ]
    executor_key = f"aws/iam/executor/{cluster_id}/role"
    allowed_keys = {executor_key, f"aws/iam/adot-writer/{cluster_id}/role"}
    if roles.count(executor_key) != 1 or set(roles) - allowed_keys:
        raise BootstrapError(
            "installation registry must contain exactly one target Executor IAM role"
            " (and at most one data-plane ADOT writer role)"
        )
    snapshot.require_source_binding()
    if snapshot.site_id != site.registry_site_id:
        raise BootstrapError("installation registry belongs to another site")
    account = Arn.parse(str(site.release_config["cpu_eks_arn"])).account
    for resource in selected:
        if (
            resource.region != site.release_config["aws_region"]
            or resource.account_id != account
            or resource.resource_type in {"gpu_eks", "gpu_hyperpod"}
            and resource.delete_policy is not InstallationResourceDeletePolicy.PRESERVE
            or resource.resource_type == "route53_vpc_association"
            and resource.delete_policy is not InstallationResourceDeletePolicy.DETACH
        ):
            raise BootstrapError(
                "target AWS resource scope or preservation policy conflicts"
            )
    deleted = [
        resource
        for resource in selected
        if resource.delete_policy is InstallationResourceDeletePolicy.DELETE
    ]
    retiring = [
        resource
        for resource in selected
        if resource.delete_policy is InstallationResourceDeletePolicy.DELETE
        or resource.resource_type == "route53_vpc_association"
    ]
    deleting = {resource.resource_key for resource in retiring}
    selected_keys = {resource.resource_key for resource in selected}
    association_keys = {
        resource.resource_key
        for resource in selected
        if resource.resource_type == "route53_vpc_association"
    }
    for resource in selected:
        if association_keys.intersection(resource.dependencies):
            raise BootstrapError(
                "target resource still depends on the Route53 association"
            )
        if resource.resource_type == "route53_vpc_association" and not any(
            item.resource_key == "aws/route53/zone"
            and item.resource_type == "route53_zone"
            and item.resource_id == resource.attributes["hosted_zone_id"]
            and item.region == resource.region
            and item.account_id == resource.account_id
            for item in snapshot.resources
        ):
            raise BootstrapError("Route53 association zone dependency is missing")
    for resource in snapshot.resources:
        if (
            resource.resource_key in selected_keys
            or resource.status in TERMINAL_INSTALLATION_RESOURCE_STATUSES
        ):
            continue
        if deleting.intersection(resource.dependencies) or any(
            resource.resource_type == target.resource_type
            and (resource.resource_arn or resource.resource_id)
            == (target.resource_arn or target.resource_id)
            for target in retiring
        ):
            raise BootstrapError(
                "target AWS resource is still owned by a preserved resource"
            )
    return selected, deletion_waves(deleted)


def remove_target_resources(
    site: RenderedSite,
    cluster_id: str,
    snapshot: InstallationResourceSnapshot,
    *,
    cleaner_factory: Callable[[RenderedSite], ResourceCleaner],
    detached_vpc_id: str | None = None,
) -> tuple[InstallationResourceSnapshot, dict[str, Any]]:
    selected, waves = target_resource_plan(
        site, cluster_id, snapshot, detached_vpc_id=detached_vpc_id
    )
    cleaner_factory(site).validate_supported(selected)
    deleted = []
    detached = [
        resource.resource_key
        for resource in selected
        if resource.delete_policy is not InstallationResourceDeletePolicy.DELETE
    ]
    for wave in waves:
        with ThreadPoolExecutor(max_workers=min(4, len(wave))) as executor:
            futures = {
                executor.submit(
                    copy_context().run, cleaner_factory(site).delete, resource
                ): resource
                for resource in wave
            }
            try:
                for future in as_completed(futures):
                    resource = futures[future]
                    try:
                        future.result()
                    except Exception as exc:
                        raise BootstrapError(
                            f"failed to delete {resource.resource_key}: {diagnostic_text(str(exc))}"
                        ) from exc
                    deleted.append(resource.resource_key)
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
    now = datetime.now(timezone.utc)
    updated = [
        resource.model_copy(
            update={
                "status": (
                    InstallationResourceStatus.DELETED
                    if resource.resource_key in deleted
                    else InstallationResourceStatus.DETACHED
                ),
                "updated_at": now,
                "error": None,
            }
        )
        if resource.resource_key in deleted or resource.resource_key in detached
        else resource
        for resource in snapshot.resources
    ]
    value = InstallationResourceSnapshot(site_id=snapshot.site_id, resources=updated)
    value = value.model_copy(update={"source_sha256": value.digest()})
    return value, {"deleted": sorted(deleted), "detached": sorted(detached)}


def merge_removal_resources(
    before: InstallationResourceSnapshot,
    removed: InstallationResourceSnapshot,
    current: InstallationResourceSnapshot,
) -> InstallationResourceSnapshot:
    for snapshot in (before, removed, current):
        snapshot.require_source_binding()
    if len({before.site_id, removed.site_id, current.site_id}) != 1:
        raise BootstrapError("removal registry snapshots belong to different sites")
    original = {resource.resource_key: resource for resource in before.resources}
    fresh = {resource.resource_key: resource for resource in current.resources}
    for resource in removed.resources:
        old = original.get(resource.resource_key)
        if old is None or resource.status == old.status:
            continue
        observed = fresh.get(resource.resource_key)
        if (
            observed is None
            or old.immutable_identity() != resource.immutable_identity()
            or old.immutable_identity() != observed.immutable_identity()
            or old.attributes != observed.attributes
            or old.created_at != observed.created_at
            or resource.status not in TERMINAL_INSTALLATION_RESOURCE_STATUSES
        ):
            raise BootstrapError(
                "target installation resource identity changed during removal"
            )
        fresh[resource.resource_key] = observed.model_copy(
            update={
                "status": resource.status,
                "updated_at": resource.updated_at,
                "error": None,
            }
        )
    value = InstallationResourceSnapshot(
        site_id=current.site_id,
        resources=sorted(fresh.values(), key=lambda item: item.resource_key),
    )
    return value.model_copy(update={"source_sha256": value.digest()})
