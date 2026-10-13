"""NLB network rows of the installation registry, numbered stably across rebuilds.

The CPU public subnets with their route tables and route-table associations, and
the NLB listeners and target groups, have no natural registry key, so they are
numbered: ``aws/network/public-subnet/<n>``, ``aws/nlb/listener/<n>`` and so on.
The number used to be the position in the list the builder was handed, and those
lists are AWS responses: bootstrap rediscovers its owned subnets with
``describe-subnets`` whenever the ``nlb_network`` task re-runs (its input covers
``bootstrap.py``), in an unspecified order. Live 2026-10-03 the re-run listed the
us-west-2b subnet first, ``/1`` and ``/2`` (with their route tables and
associations) swapped resource IDs, and every later deploy and ``config`` sync
was refused with "installation resource identity cannot change" -- a renumbering,
not a replaced resource (batch 79).

``registered_slots`` keeps the number the registry already gave a physical
resource and gives one it does not hold the lowest free number, so a fresh
registry is numbered 1, 2, ... in list order as before and a rediscovery in any
order rebuilds the same rows. A resource that really was replaced gets a new
number beside the old row instead of overwriting it.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.resource_records import (
    foundation_ownership as _foundation_ownership,
)
from gpu_fault.admin.resource_records import ownership as _ownership
from gpu_fault.admin.resource_records import policy as _policy
from gpu_fault.admin.resource_records import record as _record
from gpu_fault.installation_resources import (
    InstallationResource,
    InstallationResourceDeletePolicy,
    InstallationResourceOwnership,
)


def registered_slots(
    prefix: str,
    resource_ids: Sequence[str],
    existing: Sequence[InstallationResource],
) -> list[int]:
    """The ``<prefix><n>`` number of each of ``resource_ids``, in order.

    A resource the registry already holds under ``<prefix><n>`` keeps ``n``;
    any other takes the lowest number that neither a registry row nor an
    earlier resource of this call holds.
    """

    recorded: dict[str, int] = {}
    for item in existing:
        number = item.resource_key.removeprefix(prefix)
        if number == item.resource_key or not number.isdigit():
            continue
        if recorded.setdefault(item.resource_id, int(number)) != int(number):
            raise BootstrapError(
                f"installation registry names {item.resource_id} twice under {prefix}"
            )
    taken = set(recorded.values())
    slots: list[int] = []
    for resource_id in resource_ids:
        slot = recorded.get(resource_id)
        if slot is None or slot in slots:
            slot = min(set(range(1, len(taken) + 2)) - taken)
        taken.add(slot)
        slots.append(slot)
    return slots


def network_resources(
    *,
    site_id: str,
    region: str,
    account_id: str,
    config: Mapping[str, Any],
    state: Mapping[str, Any],
    runtime: Mapping[str, Any],
    existing: Sequence[InstallationResource] = (),
) -> list[InstallationResource]:
    resources: list[InstallationResource] = []
    nlb = state.get("nlb_network") or {}
    subnet_keys = []
    owned = list(nlb.get("subnet_resources") or ())
    slots = registered_slots(
        "aws/network/public-subnet/",
        [str(subnet["subnet_id"]) for subnet in owned],
        existing,
    )
    for index, subnet in zip(slots, owned, strict=True):
        ownership = _foundation_ownership(subnet.get("ownership"))
        subnet_key = f"aws/network/public-subnet/{index}"
        subnet_keys.append(subnet_key)
        resources.append(
            _record(
                site_id=site_id,
                resource_key=subnet_key,
                resource_type="ec2_subnet",
                resource_id=str(subnet["subnet_id"]),
                region=region,
                account_id=account_id,
                ownership=ownership,
                delete_policy=_policy(ownership),
                attributes={
                    "vpc_id": subnet.get("vpc_id"),
                    "availability_zone": subnet.get("availability_zone"),
                },
            )
        )
        route_table = subnet.get("route_table_id")
        association = subnet.get("route_table_association_id")
        if route_table:
            route_key = f"aws/network/public-route-table/{index}"
            resources.append(
                _record(
                    site_id=site_id,
                    resource_key=route_key,
                    resource_type="ec2_route_table",
                    resource_id=str(route_table),
                    region=region,
                    account_id=account_id,
                    ownership=ownership,
                    delete_policy=_policy(ownership),
                    attributes={"vpc_id": subnet.get("vpc_id")},
                )
            )
            if association:
                resources.append(
                    _record(
                        site_id=site_id,
                        resource_key=(f"aws/network/public-route-association/{index}"),
                        resource_type="ec2_route_table_association",
                        resource_id=str(association),
                        region=region,
                        account_id=account_id,
                        ownership=ownership,
                        delete_policy=_policy(
                            ownership,
                            created=InstallationResourceDeletePolicy.DETACH,
                        ),
                        dependencies=[subnet_key, route_key],
                    )
                )
    if not subnet_keys:
        raw_subnets = nlb.get("public_subnets")
        if not raw_subnets:
            raw_subnets = str(config["nlb"].get("public_subnets") or "").split(",")
        for index, subnet_id in enumerate(raw_subnets, 1):
            if not subnet_id:
                continue
            subnet_key = f"aws/network/public-subnet/{index}"
            subnet_keys.append(subnet_key)
            resources.append(
                _record(
                    site_id=site_id,
                    resource_key=subnet_key,
                    resource_type="ec2_subnet",
                    resource_id=str(subnet_id),
                    region=region,
                    account_id=account_id,
                    ownership=InstallationResourceOwnership.EXTERNAL,
                    delete_policy=InstallationResourceDeletePolicy.PRESERVE,
                )
            )
    gateway = nlb.get("internet_gateway") or {}
    if gateway.get("internet_gateway_id"):
        ownership = _foundation_ownership(gateway.get("ownership"))
        resources.append(
            _record(
                site_id=site_id,
                resource_key="aws/network/internet-gateway",
                resource_type="internet_gateway",
                resource_id=str(gateway["internet_gateway_id"]),
                region=region,
                account_id=account_id,
                ownership=ownership,
                delete_policy=_policy(ownership),
                attributes={"vpc_id": gateway.get("vpc_id")},
            )
        )
    security_group_id = nlb.get("security_group") or config["nlb"].get("security_group")
    if security_group_id:
        ownership = _ownership(
            nlb.get("security_group_ownership"),
            default=(
                InstallationResourceOwnership.CREATED
                if nlb
                else InstallationResourceOwnership.EXTERNAL
            ),
        )
        resources.append(
            _record(
                site_id=site_id,
                resource_key="aws/nlb/security-group",
                resource_type="security_group",
                resource_id=str(security_group_id),
                region=region,
                account_id=account_id,
                ownership=ownership,
                delete_policy=_policy(ownership),
                attributes={"vpc_id": nlb.get("vpc_id")},
            )
        )
    nlb_details = runtime.get("nlb") or {}
    nlb_dependencies = [
        *subnet_keys,
        *(["aws/nlb/security-group"] if security_group_id else []),
    ]
    resources.append(
        _record(
            site_id=site_id,
            resource_key="aws/nlb",
            resource_type="nlb",
            resource_id=str(config["nlb"]["name"]),
            resource_arn=nlb_details.get("arn"),
            region=region,
            account_id=account_id,
            ownership=InstallationResourceOwnership.CREATED,
            delete_policy=InstallationResourceDeletePolicy.DELETE,
            dependencies=nlb_dependencies,
            attributes={"dns_name": nlb_details.get("dns_name")},
        )
    )
    listeners = [str(item) for item in nlb_details.get("listeners") or ()]
    for index, listener in zip(
        registered_slots("aws/nlb/listener/", listeners, existing),
        listeners,
        strict=True,
    ):
        resources.append(
            _record(
                site_id=site_id,
                resource_key=f"aws/nlb/listener/{index}",
                resource_type="nlb_listener",
                resource_id=listener,
                resource_arn=listener,
                region=region,
                account_id=account_id,
                ownership=InstallationResourceOwnership.CREATED,
                delete_policy=InstallationResourceDeletePolicy.DELETE,
                dependencies=["aws/nlb"],
            )
        )
    target_groups = [str(item) for item in nlb_details.get("target_groups") or ()]
    for index, target_group in zip(
        registered_slots("aws/nlb/target-group/", target_groups, existing),
        target_groups,
        strict=True,
    ):
        resources.append(
            _record(
                site_id=site_id,
                resource_key=f"aws/nlb/target-group/{index}",
                resource_type="nlb_target_group",
                resource_id=target_group,
                resource_arn=target_group,
                region=region,
                account_id=account_id,
                ownership=InstallationResourceOwnership.CREATED,
                delete_policy=InstallationResourceDeletePolicy.DELETE,
                dependencies=["aws/nlb"],
            )
        )
    return resources
