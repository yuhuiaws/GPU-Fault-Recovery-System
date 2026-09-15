"""Route53 private-zone VPC associations: one registry rule for bootstrap and join.

Finding A (staging-1, 2026-09-15): the association bootstrap created for a GPU
VPC never reached the installation registry -- the checkpoint reader skipped
every association of a zone the site created -- while the same association
created by ``join-cluster`` was registered as
``aws/route53/vpc-association/<cluster_id>``. Both writers now record the
checkpoint entry through ``vpc_association_entry`` and both registry paths build
the row through ``vpc_association_resource``; the keying, CPU-VPC and legacy
rules live in ``vpc_association_resources``.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from gpu_fault.admin.bootstrap_common import ClusterIdentity, safe_name
from gpu_fault.admin.bootstrap_site import unique_gpu_vpcs
from gpu_fault.admin.resource_records import ownership as _ownership
from gpu_fault.admin.resource_records import policy as _policy
from gpu_fault.admin.resource_records import record as _record
from gpu_fault.installation_resources import (
    InstallationResource,
    InstallationResourceDeletePolicy,
    InstallationResourceOwnership,
)

VPC_ASSOCIATION_KEY_PREFIX = "aws/route53/vpc-association/"


def vpc_association_key(owner: str) -> str:
    """Registry key of one private-zone VPC association.

    ``owner`` is the id of the GPU cluster the association is registered under
    (the first of the checkpoint entry's ``cluster_ids``); only a legacy entry
    no cluster can be derived for falls back to its 1-based position in
    ``pki.vpc_associations`` (see ``vpc_association_resources``).
    """
    return VPC_ASSOCIATION_KEY_PREFIX + owner


def vpc_association_entry(
    *,
    vpc_id: str,
    vpc_region: str,
    cluster_ids: Iterable[str],
    ownership: str = "CREATED",
) -> dict[str, Any]:
    """One ``pki.vpc_associations`` checkpoint item, the same shape from both writers.

    Bootstrap (``_ensure_private_zone``) and join (``_update_bootstrap_state``)
    record each association through this helper. ``cluster_ids`` -- sorted, and
    additive so states written before it load unchanged -- names the GPU
    clusters whose VPC this is; the registry keys the association by the first
    of them, so the association bootstrap created for a VPC and the one join
    would have created for the same VPC resolve to the same
    ``aws/route53/vpc-association/<cluster_id>`` row.
    """
    return {
        "vpc_id": vpc_id,
        "vpc_region": vpc_region,
        "ownership": ownership,
        "cluster_ids": sorted(set(cluster_ids)),
    }


def zone_vpc_associations(
    cpu: ClusterIdentity,
    gpu_clusters: Sequence[ClusterIdentity],
) -> list[dict[str, Any]]:
    """The private zone's VPC associations in checkpoint shape, as bootstrap writes them.

    The CPU VPC comes first: the zone is created with it, so it is the zone's
    native association and never a registry row of its own. Then every distinct
    GPU VPC in ``unique_gpu_vpcs`` order. Each item carries the sorted
    ``cluster_ids`` of the GPU clusters in that VPC -- the same id the site
    document and the per-cluster tasks use (``safe_name(hyperpod_name)``) -- so
    the registry keys the association by the first of them exactly as join keys
    the association it creates by the joining cluster. A GPU cluster inside the
    CPU VPC is listed on the CPU entry and gets no association of its own.
    """
    clusters_by_vpc: dict[tuple[str, str], set[str]] = {}
    for cluster in gpu_clusters:
        clusters_by_vpc.setdefault((cluster.region, cluster.vpc_id), set()).add(
            safe_name(cluster.hyperpod_name)
        )
    associations = [
        vpc_association_entry(
            vpc_id=cpu.vpc_id,
            vpc_region=cpu.region,
            cluster_ids=clusters_by_vpc.get((cpu.region, cpu.vpc_id), set()),
        )
    ]
    for region, vpc_id in unique_gpu_vpcs(cpu, gpu_clusters):
        associations.append(
            vpc_association_entry(
                vpc_id=vpc_id,
                vpc_region=region,
                cluster_ids=clusters_by_vpc[(region, vpc_id)],
            )
        )
    return associations


def vpc_association_resource(
    *,
    site_id: str,
    owner: str,
    hosted_zone_id: str,
    vpc_id: str,
    vpc_region: str,
    region: str,
    account_id: str,
    ownership: InstallationResourceOwnership = InstallationResourceOwnership.CREATED,
) -> InstallationResource:
    """The registry row of one GPU-VPC association, whichever path created it.

    ``vpc_association_resources`` (reading the bootstrap checkpoint) and
    ``cluster_join_commit._joined_resources`` (registering what a join created)
    both build the row here so the two paths cannot drift: an association the
    site CREATED is detached with its cluster, an EXTERNAL one is preserved, and
    the row depends on the zone so the detach wave runs before the zone delete.
    """
    return _record(
        site_id=site_id,
        resource_key=vpc_association_key(owner),
        resource_type="route53_vpc_association",
        resource_id=f"{hosted_zone_id}:{vpc_region}:{vpc_id}",
        region=region,
        account_id=account_id,
        ownership=ownership,
        delete_policy=_policy(
            ownership,
            created=InstallationResourceDeletePolicy.DETACH,
        ),
        dependencies=["aws/route53/zone"],
        attributes={
            "hosted_zone_id": hosted_zone_id,
            "vpc_id": vpc_id,
            "vpc_region": vpc_region,
        },
    )


def _association_owners(
    associations: Sequence[Any],
    *,
    cpu_vpc_id: str | None,
    site_cluster_ids: Sequence[str],
    joined_clusters: Mapping[str, Any],
) -> dict[int, str]:
    """Map each GPU-VPC association (1-based position) to the cluster id keying it.

    Recorded ``cluster_ids`` win (their first id; bootstrap and join sort them).
    A legacy entry without them is matched to the cluster that joined with its
    VPC, and when exactly one such entry and exactly one GPU cluster of the site
    are left unaccounted for, the two belong together: bootstrap associated
    exactly the GPU clusters' VPCs. Entries no cluster can be derived for are
    absent from the result; the CPU VPC and repeats of a VPC are never mapped.
    """
    owners: dict[int, str] = {}
    accounted: set[str] = set(joined_clusters)
    joined_by_vpc: dict[str, list[str]] = {}
    for cluster_id, joined in joined_clusters.items():
        if isinstance(joined, Mapping) and joined.get("vpc_id"):
            joined_by_vpc.setdefault(str(joined["vpc_id"]), []).append(str(cluster_id))
    unresolved: list[int] = []
    seen_vpcs: set[str] = set()
    for index, association in enumerate(associations, 1):
        if not isinstance(association, Mapping) or not association.get("vpc_id"):
            continue
        vpc_id = str(association["vpc_id"])
        cluster_ids = [
            str(item) for item in association.get("cluster_ids") or () if item
        ]
        accounted.update(cluster_ids)
        if vpc_id == cpu_vpc_id or vpc_id in seen_vpcs:
            continue
        seen_vpcs.add(vpc_id)
        if cluster_ids:
            owners[index] = cluster_ids[0]
        elif vpc_id in joined_by_vpc:
            owners[index] = sorted(joined_by_vpc[vpc_id])[0]
        else:
            unresolved.append(index)
    unaccounted = [item for item in site_cluster_ids if item not in accounted]
    if len(unresolved) == 1 and len(unaccounted) == 1:
        owners[unresolved[0]] = unaccounted[0]
    return owners


def vpc_association_resources(
    *,
    site_id: str,
    region: str,
    account_id: str,
    hosted_zone_id: str | None,
    config: Mapping[str, Any],
    state: Mapping[str, Any],
    joined_clusters: Mapping[str, Any],
) -> list[InstallationResource]:
    """Registry rows of the private zone's GPU-VPC associations from the checkpoint.

    One rule whether bootstrap or join created them: every association of a
    GPU VPC is one ``vpc_association_resource`` row keyed
    ``aws/route53/vpc-association/<cluster_id>``, its policy set by the
    association's recorded ownership (CREATED detaches with the cluster,
    EXTERNAL is preserved). Zone ownership does not decide whether the row
    exists -- the detach wave runs before the zone delete either way. The CPU
    VPC is never registered: a private zone cannot lose its last VPC, it goes
    with the zone. It is the NLB network's VPC or, for a state that never
    recorded one, the first association bootstrap wrote (the zone is created
    with it). One row per distinct VPC, keyed by the first of the entry's
    ``cluster_ids``; a legacy entry without them is keyed as
    ``_association_owners`` derives, and when no cluster can be derived it
    keeps the pre-2026-09-15 key: its 1-based position in
    ``pki.vpc_associations``, the CPU VPC counting as position 1.
    """
    pki = state.get("pki") or {}
    associations = list(pki.get("vpc_associations") or ())
    if not hosted_zone_id or not associations:
        return []
    first = associations[0] if isinstance(associations[0], Mapping) else {}
    cpu_vpc_id = (state.get("nlb_network") or {}).get("vpc_id") or first.get("vpc_id")
    owners = _association_owners(
        associations,
        cpu_vpc_id=cpu_vpc_id,
        site_cluster_ids=[
            str(item["cluster_id"]) for item in config.get("clusters") or ()
        ],
        joined_clusters=joined_clusters,
    )
    resources: list[InstallationResource] = []
    registered_vpcs: set[str] = set()
    for index, association in enumerate(associations, 1):
        if not isinstance(association, Mapping) or not association.get("vpc_id"):
            continue
        vpc_id = str(association["vpc_id"])
        if vpc_id == cpu_vpc_id or vpc_id in registered_vpcs:
            continue
        registered_vpcs.add(vpc_id)
        ownership = _ownership(
            association.get("ownership"),
            default=InstallationResourceOwnership.CREATED,
        )
        resources.append(
            vpc_association_resource(
                site_id=site_id,
                owner=owners.get(index) or str(index),
                hosted_zone_id=hosted_zone_id,
                vpc_id=vpc_id,
                vpc_region=str(association.get("vpc_region") or region),
                region=region,
                account_id=account_id,
                ownership=ownership,
            )
        )
    return resources
