"""Which provider stands behind a GPU cluster ARN, and what a plain EKS one is.

``gpu_fault.admin.bootstrap.discover_cluster`` keeps the HyperPod path it always
had; this module holds the two reads it delegates: the HyperPod lookup for an
EKS ARN (now allowed to answer "none") and the identity of an EKS cluster that
has no HyperPod orchestrator at all -- a plain EKS cluster whose GPU capacity
is a managed node group (provider ``eks-managed``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from gpu_fault.admin.bootstrap_common import (
    Arn,
    BootstrapError,
    ClusterIdentity,
    CommandRunner,
    describe_or_absent,
)
from gpu_fault.admin.bootstrap_site import hyperpod_inventory
from gpu_fault.cluster_scope import (
    DEFAULT_EKS_MANAGED_NODE_SELECTOR,
    EKS_NODEGROUP_LABEL,
    ClusterProvider,
    validate_node_selector,
)
from gpu_fault.gpu_instance_inventory import GPU_INSTANCE_INVENTORY

__all__ = [
    "cluster_provider_entries",
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


def regional_profile_for(
    repository_root: Path, gpu_clusters: Sequence[ClusterIdentity]
) -> tuple[str, str]:
    """``(profile source path, profile version)`` a fresh site starts from.

    The HyperPod profile unless every GPU cluster is plain EKS: the EKS profile
    owns nodeReboot through the EKS adapter and keeps nodeReplace OBSERVE, and
    a site with any HyperPod cluster must not register a profile whose owners
    its HyperPod executors never advertise.
    """

    source, version = HYPERPOD_PROFILE
    if gpu_clusters and all(cluster.is_eks_managed for cluster in gpu_clusters):
        source, version = EKS_MANAGED_PROFILE
    return str(repository_root / source), version


def cluster_provider_entries(cluster: ClusterIdentity) -> dict[str, str]:
    """The provider keys of a ``spec.clusters[]`` entry for ``cluster``.

    A HyperPod cluster writes ``hyperpodClusterName`` and nothing else, as
    every site before plain EKS support did; a plain EKS cluster writes
    ``provider`` and ``nodeSelector`` and no HyperPod name.
    """

    if cluster.is_eks_managed:
        return {
            "provider": ClusterProvider.EKS_MANAGED.value,
            "nodeSelector": cluster.node_selector,
        }
    return {"hyperpodClusterName": cluster.hyperpod_name}


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


def _nodegroup_instance_types(nodegroup: Mapping[str, Any]) -> tuple[str, ...]:
    types = [str(item) for item in nodegroup.get("instanceTypes") or () if item]
    launch_template_type = str(nodegroup.get("instanceType") or "")
    if launch_template_type:
        types.append(launch_template_type)
    return tuple(dict.fromkeys(types))


def _is_gpu_nodegroup(nodegroup: Mapping[str, Any]) -> bool:
    return any(
        item.removeprefix("ml.") in GPU_INSTANCE_INVENTORY
        for item in _nodegroup_instance_types(nodegroup)
    )


def _selected_nodegroup_name(selector: str) -> str:
    """The node group a ``eks.amazonaws.com/nodegroup=<name>`` selector names."""

    for term in selector.split(","):
        key, separator, value = term.strip().partition("=")
        if separator and key.strip().rstrip("=") == EKS_NODEGROUP_LABEL:
            return value.strip()
    return ""


def gpu_nodegroups(
    runner: CommandRunner,
    *,
    region: str,
    eks_name: str,
    node_selector: str,
) -> list[dict[str, Any]]:
    """The managed node groups that hold this cluster's GPU nodes.

    Named explicitly by an ``eks.amazonaws.com/nodegroup=<name>`` selector, or
    recognised by an instance type in the GPU inventory the node installer
    sizes from. The CPU node groups never qualify, so their subnets never
    widen the Agent endpoint CIDRs.
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
    wanted = _selected_nodegroup_name(node_selector)
    if wanted and wanted not in names:
        raise BootstrapError(
            f"EKS cluster {eks_name} has no managed node group named {wanted}"
        )
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
        if not wanted or name == wanted
    ]
    selected = [
        item
        for item in described
        if (wanted and item.get("nodegroupName") == wanted) or _is_gpu_nodegroup(item)
    ]
    if not selected:
        raise BootstrapError(
            f"EKS cluster {eks_name} has no managed node group with a supported GPU "
            "instance type; name the GPU node group with --gpu-nodegroup"
        )
    for item in selected:
        if item.get("status") != "ACTIVE":
            raise BootstrapError(
                f"EKS node group {item.get('nodegroupName')} of {eks_name} is not ACTIVE"
            )
    return selected


def discover_eks_managed_cluster(
    runner: CommandRunner,
    *,
    cluster_arn: str,
    role: str,
    context: str,
    node_selector: str | None,
    subnet_cidrs: SubnetCidrs,
) -> ClusterIdentity:
    """Identity of a plain EKS GPU cluster (no HyperPod orchestrator).

    ``eks describe-cluster`` must be ACTIVE; the node subnets and therefore the
    Agent endpoint CIDRs come from the GPU managed node groups (the EKS
    ``subnetIds`` only place control-plane ENIs); ``subnet_ids`` stays the EKS
    set that places Aurora and the NLB, as for HyperPod. There is no
    ``NodeRecovery`` to gate on -- the field belongs to HyperPod -- and the CPU
    control plane of a site stays a HyperPod cluster, so only ``role="gpu"`` is
    accepted here.
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
    try:
        selector = validate_node_selector(
            node_selector or DEFAULT_EKS_MANAGED_NODE_SELECTOR
        )
    except ValueError as exc:
        raise BootstrapError(f"invalid --node-selector: {exc}") from None
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
    for nodegroup in gpu_nodegroups(
        runner, region=parsed.region, eks_name=eks_name, node_selector=selector
    ):
        node_subnets.extend(str(item) for item in nodegroup.get("subnets") or ())
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
        node_selector=selector,
    )
