"""Which provider stands behind a GPU cluster ARN, and what a plain EKS one is.

``gpu_fault.admin.bootstrap.discover_cluster`` keeps the HyperPod path it always
had; this module holds the two reads it delegates: the HyperPod lookup for an
EKS ARN (now allowed to answer "none") and the identity of an EKS cluster that
has no HyperPod orchestrator at all -- a plain EKS cluster whose GPU capacity
is a managed node group (provider ``eks-managed``).
"""

from __future__ import annotations

from pathlib import Path

import yaml  # type: ignore[import-untyped,unused-ignore]
from typing import Any, Callable, Mapping, Sequence

from gpu_fault.admin.bootstrap_common import (
    Arn,
    BootstrapError,
    ClusterIdentity,
    CommandRunner,
    describe_or_absent,
)
from gpu_fault.admin.bootstrap_ownership import (
    NodeScope,
    RecordedNodeScope,
    eks_nodegroup_facts,
    groups_for_selector,
    resolve_node_scope,
)
from gpu_fault.admin.bootstrap_site import hyperpod_inventory
from gpu_fault.eks_node_repair import (
    aggregate_node_repair,
    node_repair_error,
    node_repair_state,
)
from gpu_fault.cluster_scope import ClusterProvider, NodeSelectorSource

__all__ = [
    "cluster_provider_entries",
    "registration_cluster_provider",
    "discover_eks_managed_cluster",
    "find_hyperpod_for_eks",
    "gpu_nodegroups",
    "orchestrator_eks_arn",
    "regional_profile_for",
]

SubnetCidrs = Callable[..., tuple[str, ...]]


HYPERPOD_PROFILE = (
    "config/runtime-profile.regional-hyperpod-safe.example.yaml",
    "hyperpod-v1",
)
EKS_MANAGED_PROFILE = (
    "config/runtime-profile.regional-eks-safe.example.yaml",
    "eks-v1",
)


def registration_cluster_provider(
    state_dir: Path, gpu_clusters: Sequence[ClusterIdentity]
) -> str | None:
    """The provider of a managed site's registration cluster, or None for a fresh site.

    Read from ``site.yaml`` (``spec.runtimeProfile.registrationClusterId`` and
    the matching ``spec.clusters[]`` entry). A registration cluster the deploy
    no longer manages, or an unreadable site, is refused: the site base profile
    must not be re-derived from whatever cluster happens to come first.
    """

    site_file = state_dir / "site.yaml"
    if not site_file.is_file():
        return None
    try:
        document = yaml.safe_load(site_file.read_text(encoding="utf-8")) or {}
        spec = document["spec"]
        registration = str(spec["runtimeProfile"]["registrationClusterId"])
        entries = {str(item["clusterId"]): item for item in spec.get("clusters") or []}
    except (OSError, yaml.YAMLError, KeyError, TypeError) as exc:
        raise BootstrapError("cannot read the site's registration cluster") from exc
    entry = entries.get(registration)
    if entry is None:
        # Zero GPU clusters (remove-cluster) keep the site's own profile: the
        # anchor is stable and no cluster re-derives it.
        return None if not entries else _refuse_registration(registration)
    arn = str(entry.get("eksClusterArn") or "")
    if not any(cluster.eks_arn == arn for cluster in gpu_clusters):
        _refuse_registration(registration)
    return str(entry.get("provider") or ClusterProvider.HYPERPOD.value)


def _refuse_registration(registration: str) -> str:
    raise BootstrapError(
        f"registration cluster {registration} is not among the managed GPU "
        "clusters; the site Runtime Profile base cannot be re-derived"
    )


def regional_profile_for(
    repository_root: Path,
    gpu_clusters: Sequence[ClusterIdentity],
    *,
    state_dir: Path | None = None,
) -> tuple[str, str]:
    """``(profile source path, profile version)`` the site profile starts from.

    A fresh site: the HyperPod profile unless every GPU cluster is plain EKS.
    A managed site (``state_dir`` holds ``site.yaml``): the family of the
    registration cluster's provider, whatever joined later -- an eks-managed
    registration cluster keeps ``eks-v1`` with a HyperPod override on the
    HyperPod member, never an eks -> hyperpod re-base (live 2026-10-08: the
    upgrade of an eks-first site re-derived a HyperPod base because a HyperPod
    cluster had joined, and the next deploy then wanted an eks override).
    """

    source, version = HYPERPOD_PROFILE
    provider = (
        registration_cluster_provider(state_dir, gpu_clusters)
        if state_dir is not None
        else None
    )
    if provider is not None:
        if provider == ClusterProvider.EKS_MANAGED.value:
            source, version = EKS_MANAGED_PROFILE
    elif gpu_clusters and all(cluster.is_eks_managed for cluster in gpu_clusters):
        source, version = EKS_MANAGED_PROFILE
    return str(repository_root / source), version


def cluster_provider_entries(cluster: ClusterIdentity) -> dict[str, str]:
    """The provider keys of a ``spec.clusters[]`` entry for ``cluster``.

    A HyperPod cluster writes ``hyperpodClusterName``; a plain EKS cluster
    writes ``provider`` and ``nodeSelector`` and no HyperPod name. Both record
    the selector discovery resolved (a HyperPod cluster joined before the
    ownership convention recorded none and keeps selecting on its cluster-name
    label) and ``nodeSelectorSource``, the audit trail of where it came from.
    """

    entries: dict[str, str] = (
        {"provider": ClusterProvider.EKS_MANAGED.value}
        if cluster.is_eks_managed
        else {"hyperpodClusterName": cluster.hyperpod_name}
    )
    if cluster.node_selector or cluster.is_eks_managed:
        entries["nodeSelector"] = cluster.node_selector
    if cluster.node_selector_source:
        entries["nodeSelectorSource"] = cluster.node_selector_source
    return entries


def orchestrator_eks_arn(hyperpod: Mapping[str, Any]) -> str:
    return str(
        ((hyperpod.get("Orchestrator") or {}).get("Eks") or {}).get("ClusterArn", "")
    )


def find_hyperpod_for_eks(
    runner: CommandRunner,
    *,
    eks_arn: str,
    region: str,
    hint: str | None = None,
) -> dict[str, Any] | None:
    """The HyperPod cluster orchestrated by ``eks_arn``, or ``None`` when none is.

    With a ``hint`` (the HyperPod ARN a previous run resolved) one describe
    answers; it is trusted only if the described cluster still names this EKS
    cluster as its orchestrator, otherwise the region inventory decides as on a
    first run. No match means the EKS cluster is a plain one (``eks-managed``);
    more than one is a configuration this solution refuses to guess about.
    """

    if hint:
        described = describe_or_absent(
            runner,
            region,
            "sagemaker",
            "describe-cluster",
            "--cluster-name",
            hint,
            not_found=("ResourceNotFound",),
        )
        if described is not None and orchestrator_eks_arn(described) == eks_arn:
            return described
    matches = [
        value
        for value in hyperpod_inventory(runner, region=region)
        if orchestrator_eks_arn(value) == eks_arn
    ]
    if len(matches) > 1:
        raise BootstrapError(
            f"EKS cluster {eks_arn} must belong to at most one HyperPod cluster; "
            f"found {len(matches)}"
        )
    return matches[0] if matches else None


def gpu_nodegroups(
    runner: CommandRunner,
    *,
    region: str,
    eks_name: str,
    site_id: str | None,
    recorded: RecordedNodeScope | None = None,
) -> tuple[list[dict[str, Any]], NodeScope]:
    """The managed node groups that hold this cluster's GPU nodes, and the scope.

    A cluster joining now resolves through the ownership label
    (``bootstrap_ownership.resolve_node_scope``): the groups labelled
    ``gpu-fault.io/control-plane=<site id>``, refused when none is, when one
    names another site or when one runs a non-GPU instance type. An
    already-managed cluster (``recorded``) keeps the selector its site entry
    records and the groups are the ones that selector covers. Every selected
    group must be ACTIVE and must not repair its own nodes; the CPU node groups
    never qualify, so their subnets never widen the Agent endpoint CIDRs.
    """

    names = [
        str(item)
        for item in runner.aws_json(
            region,
            "eks",
            "list-nodegroups",
            "--cluster-name",
            eks_name,
        ).get("nodegroups", [])
        if item
    ]
    described = [
        runner.aws_json(
            region,
            "eks",
            "describe-nodegroup",
            "--cluster-name",
            eks_name,
            "--nodegroup-name",
            name,
        ).get("nodegroup")
        or {}
        for name in sorted(names)
    ]
    facts = [eks_nodegroup_facts(item) for item in described]
    if recorded is not None and not recorded.selector:
        # An identity-only read (remove-cluster, rotate-token): no scope is
        # resolved; the GPU inventory groups stand in for subnets and repair.
        covered = [item for item in facts if item.hosts_gpu_types]
        scope = NodeScope(
            selector="",
            source=None,
            owned_groups=tuple(item.name for item in covered),
            evidence=tuple(facts),
        )
    elif recorded is not None:
        covered = groups_for_selector(facts, recorded.selector)
        if not covered:
            raise BootstrapError(
                f"the recorded node selector {recorded.selector!r} of EKS cluster "
                f"{eks_name} covers none of its managed node groups ("
                + (", ".join(item.describe() for item in facts) or "none")
                + ")"
            )
        scope = NodeScope(
            selector=recorded.selector,
            source=NodeSelectorSource(recorded.source),
            owned_groups=tuple(item.name for item in covered),
            evidence=tuple(facts),
        )
    else:
        if site_id is None:
            raise BootstrapError(
                f"discovering the GPU node groups of EKS cluster {eks_name} needs "
                "the site id the ownership label is compared with"
            )
        scope = resolve_node_scope(
            facts,
            provider=ClusterProvider.EKS_MANAGED,
            cluster=eks_name,
            site_id=site_id,
        )
    selected = [
        item for item in described if item.get("nodegroupName") in scope.owned_groups
    ]
    for item in selected:
        if item.get("status") != "ACTIVE":
            raise BootstrapError(
                f"EKS node group {item.get('nodegroupName')} of {eks_name} is not ACTIVE"
            )
        # The plain EKS counterpart of HyperPod NodeRecovery=None: a node group
        # that repairs its own nodes would race this solution's recovery.
        reason = node_repair_error(
            str(item.get("nodegroupName")), eks_name, node_repair_state(item)
        )
        if reason is not None:
            raise BootstrapError(reason)
    return selected, scope


def discover_eks_managed_cluster(
    runner: CommandRunner,
    *,
    cluster_arn: str,
    role: str,
    context: str,
    subnet_cidrs: SubnetCidrs,
    site_id: str | None = None,
    recorded: RecordedNodeScope | None = None,
) -> ClusterIdentity:
    """Identity of a plain EKS GPU cluster (no HyperPod orchestrator).

    ``eks describe-cluster`` must be ACTIVE; the node subnets and therefore the
    Agent endpoint CIDRs come from the GPU managed node groups (the EKS
    ``subnetIds`` only place control-plane ENIs); ``subnet_ids`` stays the EKS
    set that places Aurora and the NLB, as for HyperPod. There is no
    ``NodeRecovery`` to gate on -- the field belongs to HyperPod -- and the CPU
    control plane of a site stays a HyperPod cluster, so only ``role="gpu"`` is
    accepted here. The GPU node groups are the ones labelled for ``site_id``
    (``gpu_nodegroups``), or the ones the cluster's existing site entry
    (``recorded``) already selects.
    """

    if role != "gpu":
        raise BootstrapError(
            f"EKS cluster {cluster_arn} has no HyperPod orchestrator; the CPU "
            "control plane must be a HyperPod EKS cluster"
        )
    parsed = Arn.parse(cluster_arn)
    if parsed.service != "eks":
        raise BootstrapError("a plain EKS cluster is identified by its EKS ARN")
    eks_name = parsed.resource_name
    eks = (
        runner.aws_json(
            parsed.region,
            "eks",
            "describe-cluster",
            "--name",
            eks_name,
        ).get("cluster")
        or {}
    )
    if eks.get("status") != "ACTIVE":
        raise BootstrapError(f"EKS cluster {eks_name} is not ACTIVE")
    if str(eks.get("arn") or cluster_arn) != cluster_arn:
        raise BootstrapError(f"EKS cluster {eks_name} describes a different ARN")
    vpc = eks.get("resourcesVpcConfig") or {}
    subnet_ids = tuple(str(item) for item in vpc.get("subnetIds") or () if item)
    node_subnets: list[str] = []
    repair_states: list[str] = []
    nodegroups, scope = gpu_nodegroups(
        runner,
        region=parsed.region,
        eks_name=eks_name,
        site_id=site_id,
        recorded=recorded,
    )
    for nodegroup in nodegroups:
        node_subnets.extend(str(item) for item in nodegroup.get("subnets") or ())
        repair_states.append(node_repair_state(nodegroup))
    cidrs = subnet_cidrs(
        runner,
        region=parsed.region,
        subnet_ids=tuple(dict.fromkeys(node_subnets)),
    )
    return ClusterIdentity(
        input_arn=cluster_arn,
        role=role,
        region=parsed.region,
        account_id=parsed.account,
        hyperpod_arn="",
        hyperpod_name="",
        eks_arn=cluster_arn,
        eks_name=eks_name,
        vpc_id=str(vpc.get("vpcId") or ""),
        subnet_ids=subnet_ids,
        node_recovery="",
        context=context,
        subnet_cidrs=cidrs,
        provider=ClusterProvider.EKS_MANAGED.value,
        node_selector=scope.selector,
        node_selector_source=scope.source.value if scope.source is not None else "",
        node_repair=aggregate_node_repair(repair_states),
    )
