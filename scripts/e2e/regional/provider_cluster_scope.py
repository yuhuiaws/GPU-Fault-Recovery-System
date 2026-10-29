"""Which CloudTrail provider events belong to the case's own GPU cluster.

Two GPU clusters share one Region and one control plane. A CloudTrail lookup by
event source returns every HyperPod cluster's ``sagemaker.amazonaws.com`` calls,
so a node reboot on one cluster used to fail a negative "no provider mutation"
claim of a case on the other ("an HP reboot fails a concurrent check on EKS"),
although nothing touched that case's nodes.

Every SageMaker cluster-node mutation names its target cluster in its request
(``clusterName`` holds the name or the ARN; some calls carry ``clusterArn``), so
an event is attributed by that binding against the case's own cluster:

- ``OWN``: every identity it names is this cluster's name or ARN;
- ``FOREIGN``: none is -- another HyperPod cluster's call. A plain EKS cluster
  has no SageMaker cluster at all, so on it every bound SageMaker event is
  foreign: a SageMaker call cannot touch an EKS node;
- ``UNBOUND``: no readable binding (the request names no cluster, an identity
  is not a nonempty string, or the record failed normalisation);
- ``CONFLICTING``: it names this cluster and another one.

Only ``FOREIGN`` is set aside; ``UNBOUND`` and ``CONFLICTING`` stay with the
case's own events (fail closed). Names and ARNs compare case-insensitively, so a
differently-cased spelling of this cluster still counts as its own.

Only the SageMaker source is attributed. An EC2 event names instances, not a
cluster; its callers keep their own instance and executor-role filters (and on
this site HyperPod instances never appear in the account's EC2 trail).
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from scripts.e2e.regional.regional_commands import RegionalFixtureError

if TYPE_CHECKING:
    from scripts.e2e.regional.regional_live_fixture import RegionalLiveFixture

SAGEMAKER_EVENT_SOURCE = "sagemaker.amazonaws.com"
#: HyperPod labels every Node it joins with its cluster's name.
HYPERPOD_CLUSTER_LABEL = "sagemaker.amazonaws.com/cluster-name"
#: A plain EKS GPU Node belongs to a managed node group or a Karpenter NodePool.
EKS_NODE_LABELS = ("eks.amazonaws.com/nodegroup", "karpenter.sh/nodepool")
HYPERPOD = "hyperpod"
EKS_MANAGED = "eks-managed"
OWN = "own"
FOREIGN = "foreign"
UNBOUND = "unbound"
CONFLICTING = "conflicting"
_REQUEST_CLUSTER_KEYS = ("clusterName", "ClusterName", "clusterArn", "ClusterArn")
_CLUSTER_ARN = re.compile(
    r"arn:aws[a-z-]*:sagemaker:[a-z0-9-]+:[0-9]{12}:cluster/[A-Za-z0-9-]+\Z"
)
_SET_ASIDE_FIELDS = (
    "event_name",
    "event_time",
    "event_id",
    "request_cluster_name",
    "request_cluster_arn",
)


@dataclass(frozen=True)
class ProviderClusterScope:
    """The case's own GPU cluster as a SageMaker request names it.

    ``identities`` is the HyperPod cluster's name and ARN; a plain EKS cluster
    has no SageMaker cluster, so its set is empty.
    """

    provider: str
    cluster_name: str = ""
    cluster_arn: str = ""

    @property
    def identities(self) -> frozenset[str]:
        return frozenset(
            item.casefold() for item in (self.cluster_name, self.cluster_arn) if item
        )

    def evidence(self) -> dict[str, str]:
        return {
            "provider": self.provider,
            "cluster_name": self.cluster_name,
            "cluster_arn": self.cluster_arn,
        }


def request_identities(parameters: object) -> list[str] | None:
    """The cluster identities a raw SageMaker ``requestParameters`` names, or
    ``None`` when it names none or names one by anything but a nonempty string."""

    if not isinstance(parameters, Mapping):
        return None
    values = [parameters[key] for key in _REQUEST_CLUSTER_KEYS if key in parameters]
    if not values or any(not isinstance(item, str) or not item for item in values):
        return None
    return [str(item) for item in values]


def event_identities(event: Mapping[str, Any]) -> list[str] | None:
    """The same for a ``normalize_provider_event`` record: its request cluster
    name and ARN, or ``None`` when it names neither or failed normalisation."""

    if event.get("normalization_errors"):
        return None
    return request_identities(
        {
            key: event[field]
            for key, field in (
                ("clusterName", "request_cluster_name"),
                ("clusterArn", "request_cluster_arn"),
            )
            if event.get(field) is not None
        }
    )


def attribution(identities: Iterable[str] | None, scope: ProviderClusterScope) -> str:
    """``OWN``, ``FOREIGN``, ``UNBOUND`` or ``CONFLICTING`` (module docstring)."""

    names = list(identities) if identities is not None else []
    if not names:
        return UNBOUND
    own = [name.casefold() in scope.identities for name in names]
    if all(own):
        return OWN
    if not any(own):
        return FOREIGN
    return CONFLICTING


def scope_from_nodes(
    document: object, describe_cluster: Any, *, region: str
) -> ProviderClusterScope:
    """The scope of the GPU cluster whose ``kubectl get node -o json`` this is.

    HyperPod: the one cluster name its Nodes are labelled with, bound to its ARN
    by ``describe_cluster(name)`` (the parsed ``describe-cluster`` answer). Plain
    EKS: no Node carries that label and at least one is a node-group or NodePool
    member. Anything else cannot be attributed and raises (fail closed).
    """

    items = document.get("items") if isinstance(document, Mapping) else None
    if not isinstance(items, list) or not items:
        raise RegionalFixtureError(
            "GPU Node inventory is empty or unreadable; provider events cannot "
            "be attributed to this cluster"
        )
    names: set[str] = set()
    eks_member = False
    for item in items:
        metadata = item.get("metadata") if isinstance(item, Mapping) else None
        labels = metadata.get("labels", {}) if isinstance(metadata, Mapping) else None
        if not isinstance(labels, Mapping):
            raise RegionalFixtureError("GPU Node labels are unreadable")
        name = labels.get(HYPERPOD_CLUSTER_LABEL)
        if name is not None:
            if not isinstance(name, str) or not name:
                raise RegionalFixtureError("HyperPod cluster label is malformed")
            names.add(name)
        eks_member = eks_member or any(labels.get(key) for key in EKS_NODE_LABELS)
    if len(names) > 1:
        raise RegionalFixtureError(
            f"GPU Nodes name more than one HyperPod cluster: {sorted(names)}"
        )
    if not names:
        if not eks_member:
            raise RegionalFixtureError(
                "GPU Nodes carry neither a HyperPod cluster label nor an EKS node "
                "group label; provider events cannot be attributed"
            )
        return ProviderClusterScope(EKS_MANAGED)
    (name,) = names
    described = describe_cluster(name)
    arn = described.get("ClusterArn") if isinstance(described, Mapping) else None
    if (
        not isinstance(described, Mapping)
        or described.get("ClusterName") != name
        or not isinstance(arn, str)
        or _CLUSTER_ARN.fullmatch(arn) is None
        or arn.split(":")[3] != region
    ):
        raise RegionalFixtureError(
            "HyperPod cluster name/ARN of the GPU Nodes is unreadable; provider "
            "events cannot be attributed to this cluster"
        )
    return ProviderClusterScope(HYPERPOD, name, arn)


def cluster_scope(regional: RegionalLiveFixture) -> ProviderClusterScope:
    """The fixture's own cluster scope, read once (Node labels, then SageMaker
    ``describe-cluster`` on HyperPod) and kept on the fixture."""

    cached = getattr(regional, "_provider_cluster_scope", None)
    if isinstance(cached, ProviderClusterScope):
        return cached
    region = regional.settings.region

    def describe(name: str) -> Any:
        return json.loads(
            regional.run(
                [
                    "aws",
                    "sagemaker",
                    "describe-cluster",
                    "--region",
                    region,
                    "--cluster-name",
                    name,
                    "--output",
                    "json",
                ],
                timeout=120,
            ).stdout
        )

    scope = scope_from_nodes(
        json.loads(regional.kubectl("gpu", "get", "node", "-o", "json")),
        describe,
        region=region,
    )
    setattr(regional, "_provider_cluster_scope", scope)
    return scope


def own_cluster_events(
    regional: RegionalLiveFixture,
    events: Sequence[dict[str, Any]],
    event_source: str,
) -> list[dict[str, Any]]:
    """``events`` without the ones another cluster's request names.

    The scope is read only when a SageMaker event names a cluster at all. What
    was set aside (name, time, id, binding) goes to stderr -- the case's log --
    and stays on the fixture as ``provider_events_set_aside`` next to
    ``provider_events_scope``; both are reset by every lookup.
    """

    kept = list(events)
    set_aside: list[dict[str, Any]] = []
    scope: ProviderClusterScope | None = None
    if event_source == SAGEMAKER_EVENT_SOURCE and any(
        event_identities(item) for item in kept
    ):
        scope = cluster_scope(regional)
        own = [attribution(event_identities(item), scope) != FOREIGN for item in kept]
        set_aside = [
            {field: item.get(field) for field in _SET_ASIDE_FIELDS}
            for item, mine in zip(kept, own)
            if not mine
        ]
        kept = [item for item, mine in zip(kept, own) if mine]
    logged = getattr(regional, "_provider_events_logged", set())
    fresh = [item for item in set_aside if str(item.get("event_id")) not in logged]
    if scope is not None and fresh:
        # Once per event: wait_provider_events polls the same window.
        print(
            f"provider events of other clusters set aside for {scope.evidence()}: "
            + json.dumps(fresh, sort_keys=True),
            file=sys.stderr,
        )
        logged = logged | {str(item.get("event_id")) for item in fresh}
        setattr(regional, "_provider_events_logged", logged)
    setattr(regional, "provider_events_set_aside", set_aside)
    setattr(regional, "provider_events_scope", scope.evidence() if scope else None)
    return kept
