"""Private-zone association inventory with stable physical resource identities."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from gpu_fault.admin.bootstrap_common import BootstrapError, ClusterIdentity, safe_name
from gpu_fault.admin.resource_records import policy, record
from gpu_fault.installation_resources import (
    RESOURCE_KEY_PATTERN,
    InstallationResource,
    InstallationResourceDeletePolicy,
    InstallationResourceOwnership,
)

VPC_ASSOCIATION_KEY_PREFIX = "aws/route53/vpc-association/"
VpcIdentity = tuple[str, str]


def _identifier(value: object, description: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or not RESOURCE_KEY_PATTERN.fullmatch(value)
        or "/" in value
        or ":" in value
    ):
        raise BootstrapError(f"Route53 association {description} is invalid")
    return value


def association_identity(association: Mapping[str, Any]) -> VpcIdentity:
    if not isinstance(association, Mapping):
        raise BootstrapError("Route53 association identity must be an object")
    return (
        _identifier(association.get("vpc_region"), "Region"),
        _identifier(association.get("vpc_id"), "VPC ID"),
    )


def network_vpc_identity(network: Mapping[str, Any], *, region: str) -> VpcIdentity:
    return association_identity(
        {
            "vpc_region": network.get("vpc_region", region),
            "vpc_id": network.get("vpc_id"),
        }
    )


def _ownership(value: object) -> InstallationResourceOwnership:
    if not isinstance(value, str):
        raise BootstrapError(
            "Route53 association ownership is unknown; reconcile its provenance"
        )
    try:
        return InstallationResourceOwnership(value)
    except ValueError:
        raise BootstrapError(
            "Route53 association ownership is unknown; reconcile its provenance"
        ) from None


def vpc_association_key(*, vpc_region: str, vpc_id: str) -> str:
    region, vpc = association_identity({"vpc_region": vpc_region, "vpc_id": vpc_id})
    return f"{VPC_ASSOCIATION_KEY_PREFIX}{region}/{vpc}"


def vpc_association_entry(
    *,
    vpc_id: str,
    vpc_region: str,
    cluster_ids: Iterable[str],
    ownership: str,
    native: bool = False,
) -> dict[str, Any]:
    """Record explicit ownership; membership never determines the registry key."""

    if isinstance(cluster_ids, (str, bytes)) or type(native) is not bool:
        raise BootstrapError("Route53 association membership or native flag is invalid")
    return {
        "vpc_id": vpc_id,
        "vpc_region": vpc_region,
        "ownership": _ownership(ownership).value,
        "cluster_ids": sorted(
            {_identifier(cluster_id, "cluster ID") for cluster_id in cluster_ids}
        ),
        "resource_key": vpc_association_key(vpc_region=vpc_region, vpc_id=vpc_id),
        "native": native,
    }


def zone_vpc_associations(
    cpu: ClusterIdentity,
    gpu_clusters: Sequence[ClusterIdentity],
    *,
    ownership_by_vpc: Mapping[VpcIdentity, str] | None = None,
) -> list[dict[str, Any]]:
    """Describe the CPU/native and unique GPU associations without adopting them.

    The caller supplies creation provenance. Merely observing an association
    in an existing zone never grants DETACH ownership.
    """

    cpu_vpc = (cpu.region, cpu.vpc_id)
    members: dict[VpcIdentity, set[str]] = {cpu_vpc: set()}
    clusters: dict[str, ClusterIdentity] = {}
    for cluster in gpu_clusters:
        cluster_id = safe_name(cluster.hyperpod_name)
        previous = clusters.setdefault(cluster_id, cluster)
        if previous != cluster:
            raise BootstrapError("Route53 association cluster identity is ambiguous")
        members.setdefault((cluster.region, cluster.vpc_id), set()).add(cluster_id)
    if ownership_by_vpc is not None and set(members) - set(ownership_by_vpc):
        raise BootstrapError("Route53 association ownership inventory is incomplete")
    return [
        vpc_association_entry(
            vpc_id=vpc_id,
            vpc_region=region,
            cluster_ids=members[(region, vpc_id)],
            ownership=(
                ownership_by_vpc[(region, vpc_id)]
                if ownership_by_vpc is not None
                else InstallationResourceOwnership.EXTERNAL.value
            ),
            native=(region, vpc_id) == cpu_vpc,
        )
        for region, vpc_id in [cpu_vpc, *sorted(set(members) - {cpu_vpc})]
    ]


def recorded_zone_vpc_ownership(
    pki: Mapping[str, Any] | None,
    *,
    hosted_zone_id: str,
    cpu: ClusterIdentity,
) -> dict[VpcIdentity, str]:
    """Return only explicit claims from the same zone and native CPU binding."""

    if pki is None or pki == {}:
        return {}
    if not isinstance(pki, Mapping) or pki.get("hosted_zone_id") != hosted_zone_id:
        raise BootstrapError("recorded Route53 hosted zone identity conflicts")
    _identifier(hosted_zone_id, "hosted zone ID")
    associations = pki.get("vpc_associations", [])
    if not isinstance(associations, list):
        raise BootstrapError("recorded Route53 association inventory is invalid")
    recorded: dict[VpcIdentity, str] = {}
    seen: dict[VpcIdentity, Mapping[str, Any]] = {}
    for association in associations:
        if not isinstance(association, Mapping):
            raise BootstrapError("recorded Route53 association is invalid")
        identity = association_identity(association)
        if "native" in association and (
            type(association["native"]) is not bool
            or association["native"] != (identity == (cpu.region, cpu.vpc_id))
        ):
            raise BootstrapError("recorded Route53 native CPU binding conflicts")
        ownership = _ownership(association.get("ownership")).value
        if identity in seen and seen[identity] != association:
            raise BootstrapError("recorded Route53 association ownership conflicts")
        seen[identity] = association
        recorded[identity] = ownership
    return recorded


def vpc_association_resource(
    *,
    site_id: str,
    hosted_zone_id: str,
    vpc_id: str,
    vpc_region: str,
    region: str,
    account_id: str,
    ownership: InstallationResourceOwnership,
    resource_key: str | None = None,
) -> InstallationResource:
    ownership = _ownership(ownership)
    zone = _identifier(hosted_zone_id, "hosted zone ID")
    key = (
        vpc_association_key(vpc_region=vpc_region, vpc_id=vpc_id)
        if resource_key is None
        else resource_key
    )
    if (
        not isinstance(key, str)
        or not key.startswith(VPC_ASSOCIATION_KEY_PREFIX)
        or not RESOURCE_KEY_PATTERN.fullmatch(key)
        or key == VPC_ASSOCIATION_KEY_PREFIX
    ):
        raise BootstrapError("Route53 association registry key is invalid")
    association_identity({"vpc_region": vpc_region, "vpc_id": vpc_id})
    return record(
        site_id=site_id,
        resource_key=key,
        resource_type="route53_vpc_association",
        resource_id=f"{zone}:{vpc_region}:{vpc_id}",
        region=region,
        account_id=account_id,
        ownership=ownership,
        delete_policy=policy(
            ownership, created=InstallationResourceDeletePolicy.DETACH
        ),
        dependencies=["aws/route53/zone"],
        attributes={
            "hosted_zone_id": zone,
            "vpc_id": vpc_id,
            "vpc_region": vpc_region,
        },
    )


def find_vpc_association_resource(
    resources: Sequence[InstallationResource],
    *,
    site_id: str,
    hosted_zone_id: str,
    vpc_id: str,
    vpc_region: str,
    region: str,
    account_id: str,
) -> InstallationResource | None:
    identity = f"{hosted_zone_id}:{vpc_region}:{vpc_id}"
    matches = [
        item
        for item in resources
        if item.resource_type == "route53_vpc_association"
        and (
            item.resource_id == identity
            or (
                item.attributes.get("hosted_zone_id") == hosted_zone_id
                and item.attributes.get("vpc_region") == vpc_region
                and item.attributes.get("vpc_id") == vpc_id
            )
        )
    ]
    if len(matches) > 1:
        raise BootstrapError(
            "Route53 association has ambiguous immutable registry rows"
        )
    if not matches:
        return None
    previous = matches[0]
    expected = vpc_association_resource(
        site_id=site_id,
        hosted_zone_id=hosted_zone_id,
        vpc_id=vpc_id,
        vpc_region=vpc_region,
        region=region,
        account_id=account_id,
        ownership=previous.ownership,
        resource_key=previous.resource_key,
    )
    if previous.immutable_identity() != expected.immutable_identity() or any(
        previous.attributes.get(key) != value
        for key, value in expected.attributes.items()
    ):
        raise BootstrapError(
            "Route53 association immutable identity or policy conflicts"
        )
    return previous


def reconcile_vpc_association_resource(
    resource: InstallationResource,
    existing: Sequence[InstallationResource],
    *,
    revive: bool = False,
) -> InstallationResource:
    """Use an existing physical association's key without rewriting its identity."""

    if any(
        item.resource_key == resource.resource_key
        and item.resource_id != resource.resource_id
        for item in existing
    ):
        raise BootstrapError(
            "Route53 association registry key already names a resource"
        )
    previous = find_vpc_association_resource(
        existing,
        site_id=resource.site_id,
        hosted_zone_id=resource.attributes["hosted_zone_id"],
        vpc_id=resource.attributes["vpc_id"],
        vpc_region=resource.attributes["vpc_region"],
        region=str(resource.region),
        account_id=str(resource.account_id),
    )
    if previous is None:
        if any(item.resource_key == resource.resource_key for item in existing):
            raise BootstrapError(
                "Route53 association registry key already names a resource"
            )
        return resource
    bound = resource.model_copy(update={"resource_key": previous.resource_key})
    if bound.immutable_identity() != previous.immutable_identity():
        raise BootstrapError(
            "Route53 association immutable ownership or policy conflicts"
        )
    return previous.model_copy(
        update={
            "status": resource.status if revive else previous.status,
            "updated_at": resource.updated_at,
            "error": resource.error if revive else previous.error,
        }
    )


def vpc_association_resources(
    *,
    site_id: str,
    region: str,
    account_id: str,
    hosted_zone_id: str | None,
    state: Mapping[str, Any],
    existing_resources: Sequence[InstallationResource] = (),
) -> list[InstallationResource]:
    """Build GPU rows using explicit CPU/native identity, never list position."""

    pki = state.get("pki") or {}
    if not isinstance(pki, Mapping):
        raise BootstrapError("Route53 checkpoint must be an object")
    associations = pki.get("vpc_associations") or []
    if not isinstance(associations, list) or any(
        not isinstance(item, Mapping) for item in associations
    ):
        raise BootstrapError("Route53 association checkpoint is invalid")
    if not associations:
        return []
    zone = _identifier(hosted_zone_id, "hosted zone ID")
    nlb = state.get("nlb_network") or {}
    if not isinstance(nlb, Mapping):
        raise BootstrapError("Route53 CPU network checkpoint is invalid")
    cpu_vpc = (
        (region, _identifier(nlb["vpc_id"], "CPU VPC ID"))
        if nlb.get("vpc_id")
        else None
    )
    native = set()
    for association in associations:
        if "native" in association and type(association["native"]) is not bool:
            raise BootstrapError("Route53 association native flag is invalid")
        if association.get("native"):
            native.add(association_identity(association))
    if len(native) > 1 or native and cpu_vpc is not None and native != {cpu_vpc}:
        raise BootstrapError("Route53 native association conflicts with CPU identity")
    cpu_vpc = cpu_vpc or next(iter(native), None)
    if cpu_vpc is None or cpu_vpc[0] != region:
        raise BootstrapError(
            "Route53 native CPU association is unknown; reconcile the checkpoint"
        )
    resources = []
    seen: dict[VpcIdentity, Mapping[str, Any]] = {}
    for association in associations:
        vpc_region, vpc_id = identity = association_identity(association)
        if "native" in association and association["native"] != (identity == cpu_vpc):
            raise BootstrapError("Route53 association native CPU binding conflicts")
        if identity in seen:
            if seen[identity] != association:
                raise BootstrapError(
                    "Route53 association checkpoint contains conflicts"
                )
            continue
        seen[identity] = association
        previous = find_vpc_association_resource(
            existing_resources,
            site_id=site_id,
            hosted_zone_id=zone,
            vpc_id=vpc_id,
            vpc_region=vpc_region,
            region=region,
            account_id=account_id,
        )
        if identity == cpu_vpc:
            if previous is not None:
                if (
                    previous.delete_policy
                    is not InstallationResourceDeletePolicy.PRESERVE
                ):
                    raise BootstrapError(
                        "Route53 native association has a deletion row; reconcile it"
                    )
                resources.append(previous)
            continue
        ownership = _ownership(
            association.get("ownership")
            if association.get("ownership") is not None or previous is None
            else previous.ownership
        )
        resource = vpc_association_resource(
            site_id=site_id,
            hosted_zone_id=zone,
            vpc_id=vpc_id,
            vpc_region=vpc_region,
            region=region,
            account_id=account_id,
            ownership=ownership,
            resource_key=association.get("resource_key"),
        )
        resources.append(
            reconcile_vpc_association_resource(resource, existing_resources)
        )
    return resources
