"""Whether a GPU cluster whose selector selects no node is parked or unavailable.

One pure rule over the provider documents a reader hands in: nothing here reads
a source. The deploy host (``gpu_fault_release.regional_parked_cluster``:
``status``, every release step and the administrator verbs) reads them through
the AWS CLI and ``kubectl``; any other reader that hands in the same documents
gets the same verdict and the same reason, so no second copy of the rule exists
to drift.

A cluster is *parked* when its provider declares that capacity zero:

* ``eks-managed``: at least one managed node group the selector covers
  (``eks_node_repair.nodegroup_hosts_gpu_nodes``), every one of them
  ``ACTIVE`` with ``scalingConfig.desiredSize`` 0, and no Karpenter NodePool
  that could hand the selector a node (``nodepools_supplying``);
* ``hyperpod``: at least one instance group the selector covers
  (``hyperpod_groups_for_selector``), every one of them ``InService`` with
  ``TargetCount`` 0 and ``CurrentCount`` 0;

and *unavailable* otherwise (``state_of``), the reason naming why it is not
parked: a covered group that wants nodes (an outage, or a scale-up still
joining), one that is not ``ACTIVE``/``InService`` (being created, updated or
replaced), a selector that covers no group, a NodePool that could supply the
scope, a provider read that failed (``capacity_proof``).

The node-group facts the HyperPod coverage is judged on (``NodeGroupFacts``,
``hyperpod_instance_group_facts``, ``groups_for_selector``) live here with it;
``gpu_fault.admin.bootstrap_ownership`` keeps its names for them.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from gpu_fault.cluster_scope import (
    EKS_NODEGROUP_LABEL,
    HYPERPOD_CLUSTER_LABEL,
    HYPERPOD_INSTANCE_GROUP_LABEL,
    INSTANCE_TYPE_LABEL,
    OWNERSHIP_LABEL,
    selector_matches,
    selector_ownership_value,
    selector_requirements,
)
from gpu_fault.eks_node_repair import nodegroup_hosts_gpu_nodes, selected_nodegroup
from gpu_fault.gpu_instance_inventory import GPU_INSTANCE_INVENTORY

#: The state of a cluster whose provider declares zero capacity.
PARKED = "parked"
#: The state of every other cluster without nodes (``state_of``).
UNAVAILABLE = "unavailable"
EKS_MANAGED = "eks-managed"
HYPERPOD = "hyperpod"
#: The prefix of the reason of a proof whose provider could not be read.
UNREADABLE_REASON = "the provider's node capacity could not be read: "
_REASON_LIMIT = 300

NodePoolRead = Callable[[], Mapping[str, Any] | None]


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _count(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def reason(exc: BaseException) -> str:
    """``exc`` as one bounded line of a proof's reason."""

    text = re.sub(r"\s+", " ", str(exc)).strip() or type(exc).__name__
    return text if len(text) <= _REASON_LIMIT else text[: _REASON_LIMIT - 3] + "..."


def proof(
    provider: str, parked: bool, reason: str, groups: list[dict[str, Any]], **extra: Any
) -> dict[str, Any]:
    """A proof: parked or not, the provider, why, and the groups it read."""

    return {
        "parked": parked,
        "provider": provider,
        "reason": reason,
        "groups": groups,
        **extra,
    }


# --- the node groups a selector covers ------------------------------------


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


def text_map(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): str(item) for key, item in value.items()}


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
    current = text_map(kubernetes.get("CurrentLabels"))
    if not current and "CurrentLabels" not in kubernetes:
        current = text_map(kubernetes.get("Labels"))
    desired = text_map(kubernetes.get("DesiredLabels"))
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


def hyperpod_groups_for_selector(
    groups: Sequence[Any], *, cluster_name: str, selector: str
) -> list[str]:
    """The ``describe-cluster`` instance groups a HyperPod cluster's recorded
    selector covers; an empty selector is the legacy cluster-name label, which
    covers every group of the cluster."""

    facts = [
        hyperpod_instance_group_facts(group, cluster_name=cluster_name)
        for group in groups
        if isinstance(group, Mapping)
    ]
    effective = selector or f"{HYPERPOD_CLUSTER_LABEL}={cluster_name}"
    return [item.name for item in groups_for_selector(facts, effective)]


# --- the provider's declared capacity --------------------------------------


def _eks_waiting(group: Mapping[str, Any]) -> str:
    name, status, desired = group["name"], group["status"], group["desired"]
    if status != "ACTIVE":
        return (
            f"node group {name} is {status or 'of unknown status'} "
            "(being created, updated or replaced)"
        )
    if desired is None:
        return f"node group {name} reports no scalingConfig.desiredSize"
    return (
        f"node group {name} wants {desired} node(s) (desiredSize {desired}): "
        "its nodes are lost or still joining"
    )


def eks_parked(
    described: Mapping[str, Mapping[str, Any]],
    *,
    selector: str,
    nodepools: NodePoolRead,
) -> dict[str, Any]:
    """The proof of an eks-managed cluster from its ``describe-nodegroup``
    documents (name -> ``nodegroup``) and its Karpenter NodePools."""

    groups = [
        {
            "name": str(name),
            "status": str(group.get("status") or ""),
            "desired": _count(_mapping(group.get("scalingConfig")).get("desiredSize")),
            "min": _count(_mapping(group.get("scalingConfig")).get("minSize")),
            "max": _count(_mapping(group.get("scalingConfig")).get("maxSize")),
            # What a node of the group is born with: the installer template of
            # a cluster without nodes is rendered from them (``render_node``).
            "instance_types": [
                str(item)
                for item in (
                    *(group.get("instanceTypes") or ()),
                    *(
                        (group.get("instanceType"),)
                        if group.get("instanceType")
                        else ()
                    ),
                )
                if item
            ],
            "labels": {
                str(key): str(value)
                for key, value in _mapping(group.get("labels")).items()
            },
        }
        for name, group in sorted(described.items())
        if isinstance(group, Mapping) and nodegroup_hosts_gpu_nodes(group, selector)
    ]
    if not groups:
        return proof(
            EKS_MANAGED,
            False,
            "the node selector covers no managed node group (a selector that "
            "selects nothing, or a group not labelled for this site)",
            groups,
        )
    waiting = [
        item for item in groups if item["status"] != "ACTIVE" or item["desired"] != 0
    ]
    if waiting:
        return proof(
            EKS_MANAGED,
            False,
            "; ".join(_eks_waiting(item) for item in waiting),
            groups,
        )
    try:
        document = nodepools()
    except Exception as exc:  # noqa: BLE001 - an unreadable list proves nothing
        return proof(
            EKS_MANAGED,
            False,
            f"the Karpenter NodePools could not be read: {reason(exc)}",
            groups,
        )
    suppliers = nodepools_supplying(document, selector)
    if suppliers:
        return proof(
            EKS_MANAGED,
            False,
            "Karpenter NodePool(s) "
            + ", ".join(suppliers)
            + " can still launch nodes the node selector selects",
            groups,
            nodepools=suppliers,
        )
    return proof(
        EKS_MANAGED,
        True,
        "every managed node group the node selector covers is ACTIVE at "
        "desiredSize 0 and no Karpenter NodePool can launch a node it selects",
        groups,
        proven_by=[
            "eks:DescribeNodegroup status and scalingConfig",
            "Karpenter NodePools (" + ("absent" if document is None else "none") + ")",
        ],
    )


def _requirement_values(
    requirements: Sequence[Mapping[str, Any]], key: str
) -> frozenset[str] | None:
    """The values a NodePool's ``requirements`` may give ``key`` on a node:
    ``None`` when none sets it, ``frozenset({"*"})`` for any value."""

    produced: set[str] = set()
    found = False
    for item in requirements:
        if str(item.get("key") or "") != key:
            continue
        operator = str(item.get("operator") or "")
        if operator == "In":
            found = True
            produced.update(str(value) for value in item.get("values") or ())
        elif operator in {"Exists", "Gt", "Lt"}:
            return frozenset({"*"})
    return frozenset(produced) if found else None


def _could_select(
    labels: Mapping[str, str],
    requirements: Sequence[Mapping[str, Any]],
    selector: str,
) -> bool:
    """Whether a node a NodePool launches could satisfy ``selector``: its
    template labels, or a requirement that puts the key on the node."""

    for key, operator, values in selector_requirements(selector):
        if key in labels:
            value = labels[key]
            if (
                (operator == "in" and value not in values)
                or (operator == "notin" and value in values)
                or operator == "absent"
            ):
                return False
            continue
        produced = _requirement_values(requirements, key)
        if operator in {"in", "exists"} and (
            produced is None
            or (operator == "in" and "*" not in produced and not produced & values)
        ):
            return False
    return True


def nodepools_supplying(document: Mapping[str, Any] | None, selector: str) -> list[str]:
    """The Karpenter NodePools that could launch a node ``selector`` selects.

    A selector that names the ownership label or a node group is judged
    against each NodePool's template labels and requirements (a requirement
    ``<key> In [...]`` or ``Exists`` puts the label on the node too). Any other
    selector names labels a node gains at runtime (``nvidia.com/gpu.present``),
    which no template predicts, so every NodePool could supply it. ``None`` is
    a cluster without the Karpenter CRD.
    """

    if document is None:
        return []
    predictable = selector_ownership_value(selector) is not None or bool(
        selected_nodegroup(selector)
    )
    found = []
    for item in document.get("items") or ():
        if not isinstance(item, Mapping):
            continue
        template = _mapping(_mapping(item.get("spec")).get("template"))
        labels = {
            str(key): str(value)
            for key, value in _mapping(
                _mapping(template.get("metadata")).get("labels")
            ).items()
        }
        requirements = [
            requirement
            for requirement in _mapping(template.get("spec")).get("requirements") or ()
            if isinstance(requirement, Mapping)
        ]
        if not predictable or _could_select(labels, requirements, selector):
            found.append(str(_mapping(item.get("metadata")).get("name") or "<unnamed>"))
    return sorted(found)


def _hyperpod_waiting(group: Mapping[str, Any]) -> str:
    name, status = group["name"], group["status"]
    target, current = group["target"], group["current"]
    if status != "InService":
        return f"instance group {name} is {status or 'of unknown status'}"
    if target is None or current is None:
        return f"instance group {name} reports no TargetCount/CurrentCount"
    if target:
        return (
            f"instance group {name} wants {target} instance(s) (TargetCount {target}, "
            f"CurrentCount {current}): its nodes are lost or still joining"
        )
    return f"instance group {name} still has {current} instance(s) at TargetCount 0"


def hyperpod_parked(
    groups: Sequence[Any], *, cluster_name: str, selector: str
) -> dict[str, Any]:
    """The proof of a HyperPod cluster from its ``describe-cluster`` instance
    groups (``HYPERPOD_CONTEXT_QUERY``)."""

    covered = set(
        hyperpod_groups_for_selector(
            groups, cluster_name=cluster_name, selector=selector
        )
    )
    entries = [
        {
            "name": str(group.get("InstanceGroupName") or ""),
            "status": str(group.get("Status") or ""),
            "target": _count(group.get("TargetCount")),
            "current": _count(group.get("CurrentCount")),
            "instance_types": (
                [str(group["InstanceType"])] if group.get("InstanceType") else []
            ),
            "labels": {
                str(key): str(value)
                for key, value in _mapping(
                    _mapping(group.get("KubernetesConfig")).get("CurrentLabels")
                ).items()
            },
        }
        for group in groups
        if isinstance(group, Mapping)
        and str(group.get("InstanceGroupName") or "") in covered
    ]
    if not entries:
        return proof(
            HYPERPOD, False, "the node selector covers no instance group", entries
        )
    waiting = [
        item
        for item in entries
        if item["status"] != "InService" or item["target"] != 0 or item["current"] != 0
    ]
    if waiting:
        return proof(
            HYPERPOD,
            False,
            "; ".join(_hyperpod_waiting(item) for item in waiting),
            entries,
        )
    return proof(
        HYPERPOD,
        True,
        "every instance group the node selector covers is InService at "
        "TargetCount 0 and CurrentCount 0",
        entries,
        proven_by=["sagemaker:DescribeCluster InstanceGroups"],
    )


def state_of(proof: Mapping[str, Any]) -> str:
    """``parked`` for a proof of declared zero capacity, else ``unavailable``."""

    return PARKED if proof.get("parked") is True else UNAVAILABLE


def unreadable(provider: str, exc: BaseException) -> dict[str, Any]:
    """The proof of a cluster whose provider could not be read: unavailable."""

    return proof(
        EKS_MANAGED if provider == EKS_MANAGED else HYPERPOD,
        False,
        UNREADABLE_REASON + reason(exc),
        [],
    )


def capacity_proof(
    provider: str,
    *,
    selector: str,
    cluster_name: str,
    nodegroups: Callable[[], Mapping[str, Mapping[str, Any]]],
    instance_groups: Callable[[], Sequence[Any]],
    nodepools: NodePoolRead,
) -> dict[str, Any]:
    """The proof of a cluster from its provider documents, read by the
    callables -- ``nodegroups`` (name -> ``describe-nodegroup`` ``nodegroup``)
    and ``nodepools`` on an eks-managed cluster, ``instance_groups``
    (``describe-cluster`` ``InstanceGroups``) on a HyperPod one; never raises:
    a provider that cannot be read proves nothing."""

    try:
        if provider == EKS_MANAGED:
            return eks_parked(nodegroups(), selector=selector, nodepools=nodepools)
        return hyperpod_parked(
            instance_groups(), cluster_name=cluster_name, selector=selector
        )
    except Exception as exc:  # noqa: BLE001 - recorded as the reason, never raised
        return unreadable(provider, exc)
