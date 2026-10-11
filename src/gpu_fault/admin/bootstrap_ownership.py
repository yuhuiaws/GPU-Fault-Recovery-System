"""Which node groups of a GPU cluster this site owns: the ownership label.

A site manages the nodes of the node groups that carry
``cluster_scope.OWNERSHIP_LABEL`` with the site id as the value -- an EKS
managed node group's ``labels``, a HyperPod instance group's
``KubernetesConfig.Labels`` (``CurrentLabels`` once applied), a Karpenter
NodePool's template labels. The label sits on the group so that every
replacement node the group creates is born owned, and the administrator
declares ownership once, up front, with the id ``gpu-fault-admin site-id``
prints; ``deploy`` and ``join-cluster`` then need nothing but ARNs.

Both providers resolve through the same function and fail closed the same
way: a group labelled for another site, a labelled group whose instance type
is not in the GPU inventory, or no labelled group at all each refuse before
anything is created, naming the groups, types and labels actually seen so the
remedy is the next command rather than a search.

A cluster joined before the convention keeps the selector it recorded
(``NodeSelectorSource.PROVIDER_DEFAULT_LEGACY`` / ``FLAG_LEGACY``): discovery
of an already-managed cluster never re-derives membership -- that is the
journaled ``node-scope --converge`` step -- so a deploy over a legacy site
keeps selecting exactly what it selected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.cluster_scope import (
    EKS_NODEGROUP_LABEL,
    HYPERPOD_CLUSTER_LABEL,
    HYPERPOD_INSTANCE_GROUP_LABEL,
    INSTANCE_TYPE_LABEL,
    OWNERSHIP_LABEL,
    ClusterProvider,
    NodeSelectorSource,
    cluster_provider,
    node_selector_source,
    ownership_node_selector,
    selector_matches,
    selector_ownership_value,
    selector_requirements,
)
from gpu_fault.gpu_instance_inventory import GPU_INSTANCE_INVENTORY

__all__ = [
    "NodeGroupFacts",
    "NodeScope",
    "RecordedNodeScope",
    "eks_nodegroup_facts",
    "groups_for_selector",
    "hyperpod_instance_group_facts",
    "hyperpod_node_scope",
    "label_remedy",
    "recorded_node_scope",
    "resolve_node_scope",
]

#: What each provider calls the unit the label is applied to.
GROUP_KINDS: dict[str, str] = {
    ClusterProvider.HYPERPOD.value: "instance group",
    ClusterProvider.EKS_MANAGED.value: "managed node group",
}


@dataclass(frozen=True)
class NodeGroupFacts:
    """One node group as the provider API describes it, for selection and evidence."""

    name: str
    instance_types: tuple[str, ...]
    #: The labels the group stamps on its nodes, plus the provider's own
    #: stamped labels (node group / instance group name, cluster name,
    #: instance type), so a recorded selector can be evaluated against a group.
    labels: Mapping[str, str]
    #: The ownership label value the group applies today, ``None`` when absent.
    ownership: str | None
    #: HyperPod only: the value declared but not yet applied (``DesiredLabels``
    #: differs from ``CurrentLabels``); ``None`` when nothing is pending.
    pending_ownership: str | None = None

    @property
    def unsupported_types(self) -> tuple[str, ...]:
        """Instance types outside the GPU inventory (``ml.`` prefix ignored)."""

        return tuple(
            item
            for item in self.instance_types
            if item.removeprefix("ml.") not in GPU_INSTANCE_INVENTORY
        )

    @property
    def hosts_gpu_types(self) -> bool:
        return bool(self.instance_types) and not self.unsupported_types

    def describe(self) -> str:
        """The evidence line a refusal prints for this group."""

        label = f"{OWNERSHIP_LABEL}=" + (self.ownership or "absent")
        if self.pending_ownership is not None:
            label += f" (desired {self.pending_ownership}, not applied yet)"
        types = ",".join(self.instance_types) or "unknown"
        return f"{self.name} instanceTypes={types} {label}"


@dataclass(frozen=True)
class NodeScope:
    """The selector a cluster records and the groups it resolved from."""

    selector: str
    #: ``None`` only for an identity-only read that resolved no scope.
    source: NodeSelectorSource | None
    owned_groups: tuple[str, ...]
    evidence: tuple[NodeGroupFacts, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "selector": self.selector,
            "source": self.source.value if self.source is not None else "",
            "owned_groups": list(self.owned_groups),
            "node_groups": [item.describe() for item in self.evidence],
        }


@dataclass(frozen=True)
class RecordedNodeScope:
    """The selector an already-managed cluster recorded (the runtime truth)."""

    selector: str
    source: str


def _text_map(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): str(item) for key, item in value.items()}


def eks_nodegroup_facts(nodegroup: Mapping[str, Any]) -> NodeGroupFacts:
    """Facts of one ``eks describe-nodegroup`` ``nodegroup`` document.

    The group's ``labels`` reach its nodes through the kubelet, and EKS stamps
    ``eks.amazonaws.com/nodegroup=<name>`` on top; ``instanceTypes`` (or the
    launch template's ``instanceType``) are the types it launches.
    """

    name = str(nodegroup.get("nodegroupName") or "")
    types = [str(item) for item in nodegroup.get("instanceTypes") or () if item]
    launch_template_type = str(nodegroup.get("instanceType") or "")
    if launch_template_type:
        types.append(launch_template_type)
    labels = _text_map(nodegroup.get("labels"))
    labels[EKS_NODEGROUP_LABEL] = name
    return NodeGroupFacts(
        name=name,
        instance_types=tuple(dict.fromkeys(types)),
        labels=labels,
        ownership=labels.get(OWNERSHIP_LABEL),
    )


def hyperpod_instance_group_facts(
    group: Mapping[str, Any], *, cluster_name: str
) -> NodeGroupFacts:
    """Facts of one ``sagemaker describe-cluster`` ``InstanceGroups[]`` entry.

    ``KubernetesConfig.CurrentLabels`` is what the group's nodes carry;
    ``DesiredLabels`` is what ``update-cluster`` asked for and the service is
    still reconciling (``Labels`` is the create-time spelling some documents
    keep). Only applied labels count as ownership; a pending value is kept as
    evidence so the refusal says "not yet" rather than "absent".
    """

    name = str(group.get("InstanceGroupName") or "")
    instance_type = str(group.get("InstanceType") or "")
    kubernetes = group.get("KubernetesConfig")
    kubernetes = kubernetes if isinstance(kubernetes, Mapping) else {}
    current = _text_map(kubernetes.get("CurrentLabels"))
    if not current and "CurrentLabels" not in kubernetes:
        current = _text_map(kubernetes.get("Labels"))
    desired = _text_map(kubernetes.get("DesiredLabels"))
    labels = dict(current)
    labels[HYPERPOD_CLUSTER_LABEL] = cluster_name
    labels[HYPERPOD_INSTANCE_GROUP_LABEL] = name
    if instance_type:
        labels[INSTANCE_TYPE_LABEL] = instance_type
    ownership = current.get(OWNERSHIP_LABEL)
    pending = desired.get(OWNERSHIP_LABEL)
    return NodeGroupFacts(
        name=name,
        instance_types=(instance_type,) if instance_type else (),
        labels=labels,
        ownership=ownership,
        pending_ownership=pending if pending != ownership else None,
    )


def label_remedy(
    provider: ClusterProvider | str, *, cluster: str, site_id: str, group: str
) -> str:
    """The provider command that applies the ownership label to ``group``."""

    if cluster_provider(provider) is ClusterProvider.HYPERPOD:
        return (
            f"aws sagemaker update-cluster --cluster-name {cluster} "
            f"--instance-groups '<the full definition of {group}>' with "
            f'KubernetesConfig={{Labels={{"{OWNERSHIP_LABEL}": "{site_id}"}}}}'
        )
    return (
        f"aws eks update-nodegroup-config --cluster-name {cluster} "
        f"--nodegroup-name {group} --labels "
        f"addOrUpdateLabels={{{OWNERSHIP_LABEL}={site_id}}}"
    )


def resolve_node_scope(
    facts: Sequence[NodeGroupFacts],
    *,
    provider: ClusterProvider | str,
    cluster: str,
    site_id: str,
    hyperpod_cluster_name: str = "",
) -> NodeScope:
    """The node scope of a cluster joining under the ownership convention.

    ``cluster`` is the provider's bare cluster name (EKS or HyperPod); the
    refusals spell out which kind it is. Refuses, naming the facts, when a
    group carries the label with another site's id (one GPU cluster belongs to
    one site), when no group carries this site's id, or when a group that does
    runs an instance type outside the GPU inventory (the node installer could
    never size it, and its nodes would keep every wave from converging).
    Otherwise the selector is ``ownership_node_selector`` and the owned groups
    are returned as evidence.
    """

    kind = cluster_provider(provider)
    group_kind = GROUP_KINDS[kind.value]
    display = (
        f"HyperPod cluster {cluster}"
        if kind is ClusterProvider.HYPERPOD
        else f"EKS cluster {cluster}"
    )
    foreign = [
        item
        for item in facts
        if item.ownership is not None and item.ownership != site_id
    ]
    if foreign:
        names = "; ".join(
            f"{item.name} carries {OWNERSHIP_LABEL}={item.ownership}"
            for item in foreign
        )
        raise BootstrapError(
            f"{group_kind}s of {display} name another site: {names}; this site is "
            f"{OWNERSHIP_LABEL}={site_id}. One GPU cluster belongs to one site: "
            "remove or change the label before joining"
        )
    owned = [item for item in facts if item.ownership == site_id]
    if not owned:
        seen = "; ".join(item.describe() for item in facts) or "none"
        example = next(
            (item.name for item in facts if item.hosts_gpu_types),
            f"<{group_kind}>",
        )
        raise BootstrapError(
            f"{display} has no {group_kind} labelled {OWNERSHIP_LABEL}={site_id} "
            "(the site id gpu-fault-admin site-id prints for this CPU cluster); "
            f"{group_kind}s seen: {seen}. Label the GPU {group_kind}s this site "
            "manages, for example "
            + label_remedy(kind, cluster=cluster, site_id=site_id, group=example)
            + ", wait until the label is applied, and rerun"
        )
    unsupported = [item for item in owned if not item.hosts_gpu_types]
    if unsupported:
        details = "; ".join(
            f"{item.name} runs {','.join(item.unsupported_types) or 'unknown'}"
            for item in unsupported
        )
        raise BootstrapError(
            f"{group_kind}s of {display} carry {OWNERSHIP_LABEL}={site_id} but run "
            f"instance types outside the GPU instance inventory: {details}; the "
            "inventory is " + ", ".join(sorted(GPU_INSTANCE_INVENTORY)) + ". The "
            "label marks GPU node groups only: remove it from these groups"
        )
    return NodeScope(
        selector=ownership_node_selector(
            site_id, provider=kind, hyperpod_cluster_name=hyperpod_cluster_name
        ),
        source=NodeSelectorSource.LABEL_CONVENTION,
        owned_groups=tuple(item.name for item in owned),
        evidence=tuple(facts),
    )


def groups_for_selector(
    facts: Sequence[NodeGroupFacts], selector: str
) -> list[NodeGroupFacts]:
    """The groups a recorded selector covers, for subnets and node-repair facts.

    Evaluated against each group's labels (its own plus the provider-stamped
    ones). A legacy selector that names only a label nodes gain at runtime
    (``nvidia.com/gpu.present=true`` from GPU feature discovery) matches no
    group's labels; such a selector falls back to the groups running a GPU
    inventory type, which is what that default meant.
    """

    matched = [item for item in facts if selector_matches(item.labels, selector)]
    if matched:
        return matched
    names_a_group = selector_ownership_value(selector) is not None or any(
        key in {EKS_NODEGROUP_LABEL, HYPERPOD_INSTANCE_GROUP_LABEL}
        for key, _operator, _values in selector_requirements(selector)
    )
    if names_a_group:
        return []
    return [item for item in facts if item.hosts_gpu_types]


def _entry_field(entry: Mapping[str, Any], *names: str) -> str:
    for name in names:
        if name in entry and entry[name] is not None:
            return str(entry[name])
    return ""


def recorded_node_scope(
    entries: Sequence[Mapping[str, Any]],
    *,
    eks_arn: str,
    hyperpod_name: str = "",
) -> RecordedNodeScope | None:
    """The scope an existing site entry records for the cluster, or ``None``.

    Accepts both spellings of a cluster entry -- ``site.yaml``
    (``eksClusterArn`` / ``nodeSelector``) and the rendered release config
    (``eks_cluster_arn`` / ``node_selector``) -- and matches by EKS ARN, or
    by HyperPod name for a cluster given as its SageMaker ARN. The source is
    the recorded one, else the legacy derivation of ``node_selector_source``.
    """

    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        entry_eks = _entry_field(entry, "eksClusterArn", "eks_cluster_arn")
        entry_hyperpod = _entry_field(
            entry, "hyperpodClusterName", "hyperpod_cluster_name"
        )
        if entry_eks != eks_arn and not (
            hyperpod_name and entry_hyperpod and entry_hyperpod == hyperpod_name
        ):
            continue
        selector = _entry_field(entry, "nodeSelector", "node_selector")
        provider = _entry_field(entry, "provider") or ClusterProvider.HYPERPOD.value
        source = node_selector_source(
            None,
            provider=provider,
            selector=selector,
            source=_entry_field(entry, "nodeSelectorSource", "node_selector_source"),
        )
        return RecordedNodeScope(selector=selector, source=source.value)
    return None


def hyperpod_node_scope(
    hyperpod: Mapping[str, Any],
    *,
    site_id: str | None,
    recorded: RecordedNodeScope | None,
) -> tuple[str, str]:
    """``(selector, source)`` of a HyperPod GPU cluster.

    An already-managed cluster keeps what its site entry records (an empty
    selector is the legacy cluster-name default). A cluster joining now
    resolves through its instance groups' ``KubernetesConfig`` labels and is
    refused without a group labelled for this site.
    """

    if recorded is not None:
        return recorded.selector, recorded.source
    hyperpod_name = str(hyperpod.get("ClusterName") or "")
    if site_id is None:
        raise BootstrapError(
            f"discovering the GPU instance groups of HyperPod cluster "
            f"{hyperpod_name} needs the site id the ownership label is compared with"
        )
    scope = resolve_node_scope(
        [
            hyperpod_instance_group_facts(group, cluster_name=hyperpod_name)
            for group in hyperpod.get("InstanceGroups") or ()
            if isinstance(group, Mapping)
        ],
        provider=ClusterProvider.HYPERPOD,
        cluster=hyperpod_name,
        site_id=site_id,
        hyperpod_cluster_name=hyperpod_name,
    )
    return scope.selector, scope.source.value if scope.source is not None else ""
