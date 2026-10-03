"""The provider-neutral scope of one managed GPU cluster.

Two kinds of GPU cluster are managed: a SageMaker HyperPod cluster orchestrated
by EKS (``hyperpod``, the original and fully validated form) and a plain EKS
cluster with managed node groups and no HyperPod orchestrator (``eks-managed``).
Every component that enumerates "the GPU nodes of cluster X" used to spell the
HyperPod label ``sagemaker.amazonaws.com/cluster-name=<name>`` itself; this
module is the one place that turns a cluster's provider and configuration into
that label selector, so the administrator CLI, the release engine, the deploy
scripts and the data-plane runtime all select the same node set.

The module is pure and import-light on purpose: both ``gpu_fault.admin`` and
the data-plane runtime import it, and the two must agree byte for byte on the
environment variable names, the connection Secret keys and the selector text.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Any, Iterable, Mapping

from gpu_fault.failure_domains import FAILURE_DOMAIN_LABELS

__all__ = [
    "CONNECTION_SECRET_NODE_SELECTOR_KEY",
    "CONNECTION_SECRET_PROVIDER_KEY",
    "DEFAULT_EKS_MANAGED_NODE_SELECTOR",
    "EKS_MANAGED_FAILURE_DOMAIN_LABELS",
    "EKS_NODEGROUP_LABEL",
    "HYPERPOD_CLUSTER_LABEL",
    "NODE_SELECTOR_ENV",
    "PROVIDER_ENV",
    "ClusterProvider",
    "cluster_provider",
    "default_failure_domain_labels",
    "ec2_instance_id_from_provider_id",
    "is_eks_managed",
    "node_selector",
    "nodegroup_node_selector",
    "provider_cluster_name",
    "selector_matches",
    "validate_node_selector",
]

#: Environment variable that tells a data-plane component which provider the
#: cluster it runs in uses. Absent means ``hyperpod`` (every site deployed
#: before plain EKS support existed).
PROVIDER_ENV = "GPU_FAULT_CLUSTER_PROVIDER"
#: Environment variable carrying the GPU node label selector. Absent means the
#: HyperPod default derived from ``GPU_FAULT_HYPERPOD_CLUSTER``.
NODE_SELECTOR_ENV = "GPU_FAULT_NODE_SELECTOR"
#: Keys of the per-cluster ``gpu-fault-regional-connection`` Secret that feed
#: the two variables above. Both are optional on the manifests so a HyperPod
#: site renders exactly as it did before the keys existed.
CONNECTION_SECRET_PROVIDER_KEY = "cluster-provider"
CONNECTION_SECRET_NODE_SELECTOR_KEY = "node-selector"

#: The label SageMaker HyperPod stamps on every node it manages.
HYPERPOD_CLUSTER_LABEL = "sagemaker.amazonaws.com/cluster-name"
#: The label EKS stamps on every node of a managed node group.
EKS_NODEGROUP_LABEL = "eks.amazonaws.com/nodegroup"
#: The GPU node selector a plain EKS join uses unless told otherwise: the
#: label NVIDIA GPU feature discovery (and the EKS NVIDIA AMI device plugin
#: bundle) sets on nodes that actually expose GPUs. CPU node groups never
#: carry it.
DEFAULT_EKS_MANAGED_NODE_SELECTOR = "nvidia.com/gpu.present=true"
#: Failure-domain label priority for an ``eks-managed`` cluster. The managed
#: node group plays the role the HyperPod instance group plays, so it comes
#: first; the zone labels follow as they do for HyperPod.
EKS_MANAGED_FAILURE_DOMAIN_LABELS: tuple[str, ...] = (
    EKS_NODEGROUP_LABEL,
    "topology.kubernetes.io/zone",
    "failure-domain.beta.kubernetes.io/zone",
)


def default_failure_domain_labels(
    providers: Iterable[ClusterProvider | str],
) -> tuple[str, ...]:
    """Failure-domain label priority for a site that configured none.

    A HyperPod-only site keeps ``FAILURE_DOMAIN_LABELS`` unchanged. As soon as
    one managed cluster is ``eks-managed`` the EKS node group label joins the
    priority list right after the HyperPod instance group label: a HyperPod
    node still resolves to its instance group first, a plain EKS node to its
    node group, and both fall back to the zone labels.
    """

    kinds = {cluster_provider(item) for item in providers}
    if ClusterProvider.EKS_MANAGED not in kinds:
        return FAILURE_DOMAIN_LABELS
    merged: list[str] = []
    for label in (
        FAILURE_DOMAIN_LABELS[0],
        *EKS_MANAGED_FAILURE_DOMAIN_LABELS,
        *FAILURE_DOMAIN_LABELS,
    ):
        if label not in merged:
            merged.append(label)
    return tuple(merged)


_LABEL_KEY = re.compile(
    r"^(?:[a-z0-9]([-a-z0-9]*[a-z0-9])?(\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*/)?"
    r"[A-Za-z0-9]([-A-Za-z0-9_.]{0,61}[A-Za-z0-9])?$"
)
_LABEL_VALUE = re.compile(r"^(?:[A-Za-z0-9]([-A-Za-z0-9_.]{0,61}[A-Za-z0-9])?)?$")
_SET_REQUIREMENT = re.compile(
    r"^(?P<key>\S+)\s+(?P<operator>in|notin)\s*\((?P<values>[^()]*)\)$"
)
_PROVIDER_ID = re.compile(r"^aws:///[^/]+/(?P<instance>i-[0-9a-f]{8,17})$")


class ClusterProvider(StrEnum):
    """Who owns the machines behind a managed GPU cluster."""

    HYPERPOD = "hyperpod"
    EKS_MANAGED = "eks-managed"


def cluster_provider(value: object) -> ClusterProvider:
    """Parse a provider name; empty or absent means ``hyperpod``.

    Any other spelling is a configuration error and fails closed rather than
    being read as the default.
    """

    if value is None:
        return ClusterProvider.HYPERPOD
    text = str(value).strip()
    if not text:
        return ClusterProvider.HYPERPOD
    try:
        return ClusterProvider(text)
    except ValueError:
        raise ValueError(
            "cluster provider must be one of "
            + ", ".join(item.value for item in ClusterProvider)
            + f", got {text!r}"
        ) from None


def _field(target: object, *names: str) -> Any:
    """First present attribute or mapping key among ``names``."""

    for name in names:
        if isinstance(target, Mapping):
            if name in target:
                return target[name]
        elif hasattr(target, name):
            return getattr(target, name)
    return None


def is_eks_managed(target: object) -> bool:
    """Whether ``target`` (identity, site entry, release target) is plain EKS."""

    return cluster_provider(_field(target, "provider")) is ClusterProvider.EKS_MANAGED


def provider_cluster_name(target: object) -> str:
    """The name the provider scope variables carry for ``target``.

    ``GPU_FAULT_HYPERPOD_CLUSTER`` and the ``hyperpod-cluster-name`` Secret keys
    keep meaning "the provider cluster scope": the HyperPod cluster name for a
    HyperPod cluster, and the stable ``cluster_id`` for a plain EKS cluster,
    which has no other provider-side name.
    """

    hyperpod = str(_field(target, "hyperpod_cluster_name", "hyperpod_name") or "")
    if cluster_provider(_field(target, "provider")) is ClusterProvider.HYPERPOD:
        return hyperpod
    return hyperpod or str(_field(target, "cluster_id") or "")


def nodegroup_node_selector(nodegroup: str) -> str:
    """The selector for one EKS managed node group."""

    name = nodegroup.strip()
    if not name or not _LABEL_VALUE.fullmatch(name):
        raise ValueError(f"invalid EKS node group name: {nodegroup!r}")
    return f"{EKS_NODEGROUP_LABEL}={name}"


def node_selector(
    target: object,
    *,
    provider: object = None,
    hyperpod_cluster_name: str | None = None,
    selector: str | None = None,
) -> str:
    """The label selector that enumerates the GPU nodes of one cluster.

    ``target`` may be anything that carries ``provider``, a HyperPod name
    (``hyperpod_cluster_name`` or ``hyperpod_name``) and ``node_selector`` as
    attributes or mapping keys: a ``ClusterIdentity``, a ``ClusterTarget``, a
    site cluster entry or a rendered release-config entry. Keyword arguments
    override what ``target`` carries.

    * ``hyperpod``: an explicit ``node_selector`` wins; otherwise the HyperPod
      label ``sagemaker.amazonaws.com/cluster-name=<name>``. Every site that
      existed before plain EKS support therefore selects exactly as before.
    * ``eks-managed``: the configured ``node_selector`` is required. There is
      no HyperPod label to fall back on, and a selector that matched every
      node would hand the CPU node group to the GPU data plane.
    """

    kind = cluster_provider(
        provider if provider is not None else _field(target, "provider")
    )
    configured = (
        selector
        if selector is not None
        else str(_field(target, "node_selector", "nodeSelector") or "")
    ).strip()
    if configured:
        return validate_node_selector(configured)
    if kind is ClusterProvider.HYPERPOD:
        name = (
            hyperpod_cluster_name
            if hyperpod_cluster_name is not None
            else str(_field(target, "hyperpod_cluster_name", "hyperpod_name") or "")
        ).strip()
        if not name:
            raise ValueError("a HyperPod cluster needs its HyperPod cluster name")
        return f"{HYPERPOD_CLUSTER_LABEL}={name}"
    raise ValueError("an eks-managed cluster requires an explicit node selector")


def _terms(selector: str) -> list[str]:
    """Split on the commas between requirements, not those inside ``(a, b)``."""

    terms: list[str] = []
    depth = 0
    current: list[str] = []
    for character in selector:
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth < 0:
                raise ValueError(
                    f"unbalanced parentheses in node selector {selector!r}"
                )
        if character == "," and depth == 0:
            terms.append("".join(current))
            current = []
        else:
            current.append(character)
    if depth:
        raise ValueError(f"unbalanced parentheses in node selector {selector!r}")
    terms.append("".join(current))
    return terms


def _requirements(selector: str) -> list[tuple[str, str, frozenset[str]]]:
    requirements: list[tuple[str, str, frozenset[str]]] = []
    for raw in _terms(selector):
        term = raw.strip()
        if not term:
            raise ValueError(f"empty requirement in node selector {selector!r}")
        matched = _SET_REQUIREMENT.fullmatch(term)
        if matched:
            values = frozenset(
                value.strip() for value in matched.group("values").split(",")
            )
            key, operator = matched.group("key"), matched.group("operator")
            if not values or any(
                not value or not _LABEL_VALUE.fullmatch(value) for value in values
            ):
                raise ValueError(f"invalid set requirement in node selector: {term}")
        elif "!=" in term:
            key, _, value = term.partition("!=")
            operator, values = "notin", frozenset({value.strip()})
        elif "==" in term:
            key, _, value = term.partition("==")
            operator, values = "in", frozenset({value.strip()})
        elif "=" in term:
            key, _, value = term.partition("=")
            operator, values = "in", frozenset({value.strip()})
        elif term.startswith("!"):
            key, operator, values = term[1:], "absent", frozenset()
        else:
            key, operator, values = term, "exists", frozenset()
        key = key.strip()
        if not _LABEL_KEY.fullmatch(key):
            raise ValueError(f"invalid label key in node selector: {term}")
        if operator in {"in", "notin"} and any(
            not _LABEL_VALUE.fullmatch(value) for value in values
        ):
            raise ValueError(f"invalid label value in node selector: {term}")
        requirements.append((key, operator, values))
    return requirements


def validate_node_selector(selector: str) -> str:
    """Return ``selector`` normalised, or raise ``ValueError``.

    Accepts the equality (``k=v``, ``k==v``, ``k!=v``), existence (``k``,
    ``!k``) and set (``k in (a,b)``, ``k notin (a)``) forms kubectl's ``-l``
    accepts. A selector must name at least one requirement: the empty
    selector means "every node", which is never the GPU set.
    """

    text = selector.strip()
    if not text:
        raise ValueError("node selector must not be empty")
    _requirements(text)
    return text


def selector_matches(labels: Mapping[str, object] | None, selector: str) -> bool:
    """Evaluate ``selector`` against one node's labels, as the API server does.

    Used to re-check a node inventory locally after the API server filtered it:
    every caller that pushes the selector to ``kubectl get nodes -l`` also
    confirms each returned item carries the labels, so a proxy or cache that
    returned the wrong set is caught rather than trusted.
    """

    present = {str(key): str(value) for key, value in (labels or {}).items()}
    for key, operator, values in _requirements(selector):
        if operator == "exists":
            if key not in present:
                return False
        elif operator == "absent":
            if key in present:
                return False
        elif operator == "in":
            if present.get(key) not in values:
                return False
        elif present.get(key, None) in values:
            return False
    return True


def ec2_instance_id_from_provider_id(provider_id: object) -> str:
    """``i-...`` from a node's ``spec.providerID`` (``aws:///<zone>/i-...``).

    Returns ``""`` when the value is not an AWS provider id, so a caller can
    fall back to the Kubernetes Node UID instead of inventing an identity.
    """

    matched = _PROVIDER_ID.fullmatch(str(provider_id or "").strip())
    return matched.group("instance") if matched else ""
