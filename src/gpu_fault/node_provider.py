"""Which provider runs the GPU cluster this process operates, and its node scope.

A GPU cluster is either a SageMaker HyperPod cluster (``hyperpod``, the only
value before plain EKS support and therefore the default when the variable is
absent) or a plain EKS cluster with managed node groups (``eks-managed``). The
data plane learns which from ``GPU_FAULT_CLUSTER_PROVIDER``; the node scope it
operates comes from ``GPU_FAULT_NODE_SELECTOR``, a Kubernetes label selector
that defaults, for HyperPod, to the cluster-name label every existing
deployment selected by. ``GPU_FAULT_HYPERPOD_CLUSTER`` keeps meaning "provider
cluster scope": the HyperPod cluster name for ``hyperpod`` and the stable
``cluster_id`` for ``eks-managed``.

The enum values and variable names are a contract shared with the admin and
release layers (``gpu_fault.cluster_scope`` renders them into the connection
Secret); they must not drift. This module carries only what the runtime needs
and imports nothing provider-specific, so every adapter can import it.
"""

from __future__ import annotations

import os
import re
from enum import StrEnum
from typing import Any, Mapping, Protocol, TypeVar

from gpu_fault.models import Environment, RecoveryAction

PROVIDER_ENV = "GPU_FAULT_CLUSTER_PROVIDER"
NODE_SELECTOR_ENV = "GPU_FAULT_NODE_SELECTOR"
PROVIDER_CLUSTER_ENV = "GPU_FAULT_HYPERPOD_CLUSTER"
EFA_DEVICE_PLUGIN_ENV = "GPU_FAULT_EFA_DEVICE_PLUGIN_NAME"

#: The label HyperPod stamps on every node of a cluster; the historical scope.
HYPERPOD_CLUSTER_LABEL = "sagemaker.amazonaws.com/cluster-name"
HYPERPOD_INSTANCE_GROUP_LABEL = "sagemaker.amazonaws.com/instance-group-name"
#: EKS managed node groups play the instance-group role on a plain cluster.
EKS_NODEGROUP_LABEL = "eks.amazonaws.com/nodegroup"

#: The reboot confirmation each provider's lifecycle adapter records. The
#: HyperPod string predates this module and is bound into persisted
#: ``RESTART_NODE`` executions, so it stays exactly as it was.
HYPERPOD_REBOOT_CONFIRMATION_SOURCE = "hyperpod-running-and-new-agent-incarnation"
EKS_REBOOT_CONFIRMATION_SOURCE = "eks-running-and-new-agent-incarnation"
REBOOT_CONFIRMATION_SOURCES = frozenset(
    {HYPERPOD_REBOOT_CONFIRMATION_SOURCE, EKS_REBOOT_CONFIRMATION_SOURCE}
)

#: ``spec.providerID`` of an EC2-backed node: ``aws:///<zone>/<instance id>``.
PROVIDER_ID_PATTERN = re.compile(
    r"^aws:///(?P<zone>[a-z0-9-]+)/(?P<instance_id>i-[0-9a-f]{8,17})$"
)
INSTANCE_ID_PATTERN = re.compile(r"^i-[0-9a-f]{8,17}$")


class ClusterProvider(StrEnum):
    HYPERPOD = "hyperpod"
    EKS_MANAGED = "eks-managed"


#: The EFA device plugin DaemonSet (its ``name`` Pod label) per provider.
EFA_DEVICE_PLUGIN_NAMES: dict[ClusterProvider, str] = {
    ClusterProvider.HYPERPOD: "hyperpod-dependencies-aws-efa-k8s-device-plugin",
    ClusterProvider.EKS_MANAGED: "aws-efa-k8s-device-plugin",
}


def _environ(environ: Mapping[str, str] | None) -> Mapping[str, str]:
    return os.environ if environ is None else environ


def cluster_provider(environ: Mapping[str, str] | None = None) -> ClusterProvider:
    """The provider named by ``GPU_FAULT_CLUSTER_PROVIDER``; absent means HyperPod.

    An unknown value is a misconfiguration, never a default: the adapters a
    process constructs and the IAM it needs follow from this answer.
    """

    raw = _environ(environ).get(PROVIDER_ENV, "").strip().lower()
    if not raw:
        return ClusterProvider.HYPERPOD
    try:
        return ClusterProvider(raw)
    except ValueError as exc:
        raise ValueError(
            f"{PROVIDER_ENV} must be one of "
            + ", ".join(item.value for item in ClusterProvider)
            + f"; got {raw!r}"
        ) from exc


def node_selector(environ: Mapping[str, str] | None = None) -> str | None:
    """The label selector of this cluster's nodes, or None when unscoped.

    ``GPU_FAULT_NODE_SELECTOR`` wins when set. Without it a HyperPod cluster
    selects by its cluster-name label, exactly as before; an ``eks-managed``
    cluster has no implied label, so the selector is required there.
    """

    values = _environ(environ)
    explicit = values.get(NODE_SELECTOR_ENV, "").strip()
    if explicit:
        return explicit
    provider = cluster_provider(values)
    if provider is ClusterProvider.EKS_MANAGED:
        raise ValueError(
            f"{NODE_SELECTOR_ENV} is required when {PROVIDER_ENV}="
            f"{ClusterProvider.EKS_MANAGED.value}"
        )
    hyperpod_cluster = (values.get(PROVIDER_CLUSTER_ENV) or "").strip()
    if not hyperpod_cluster:
        return None
    return f"{HYPERPOD_CLUSTER_LABEL}={hyperpod_cluster}"


def instance_group_label(provider: ClusterProvider) -> str:
    """The node label that names the failure-domain group on ``provider``."""

    if provider is ClusterProvider.EKS_MANAGED:
        return EKS_NODEGROUP_LABEL
    return HYPERPOD_INSTANCE_GROUP_LABEL


def efa_device_plugin_name(environ: Mapping[str, str] | None = None) -> str:
    """The EFA device plugin DaemonSet name to restart on this cluster."""

    values = _environ(environ)
    explicit = values.get(EFA_DEVICE_PLUGIN_ENV, "").strip()
    if explicit:
        return explicit
    return EFA_DEVICE_PLUGIN_NAMES[cluster_provider(values)]


def parse_provider_id(value: object) -> tuple[str | None, str] | None:
    """``(zone, instance id)`` from a node's ``spec.providerID``, or None.

    Accepts the EC2 shape ``aws:///us-west-2a/i-00000000000000001``. The
    historical fallback -- a trailing ``...-i-<id>`` segment, read exactly as
    the HyperPod spare coordinator always has -- is kept for the provider IDs
    HyperPod nodes carry; anything else is unknown.
    """

    if not isinstance(value, str) or not value:
        return None
    match = PROVIDER_ID_PATTERN.match(value.strip())
    if match is not None:
        return match.group("zone"), match.group("instance_id")
    tail = value.strip().rsplit("/", 1)[-1]
    if "-i-" in tail:
        return None, "i-" + tail.rsplit("-i-", 1)[-1]
    return None


def restart_vm_effective_action(
    environment: Environment,
) -> tuple[RecoveryAction | None, str | None]:
    """How NVIDIA's ``RESTART_VM`` lands on a cluster whose nodes are not VMs.

    HyperPod reboots through ``BatchRebootClusterNodes``; a plain EKS cluster
    through EC2 ``RebootInstances`` on the bound instance. Neither is an EC2
    stop/start, so neither may be described as one. A bare Kubernetes profile
    maps nothing and keeps the engine's BLOCKED verdict.
    """

    if environment in {Environment.HYPERPOD_EKS, Environment.HYPERPOD_SLURM}:
        return (
            RecoveryAction.REBOOT_NODE,
            "HyperPod maps NVIDIA RESTART_VM to REBOOT_NODE via "
            "BatchRebootClusterNodes; this is not EC2 stop/start",
        )
    if environment is Environment.EKS:
        return (
            RecoveryAction.REBOOT_NODE,
            "eks-managed maps NVIDIA RESTART_VM to REBOOT_NODE via EC2 "
            "RebootInstances on the bound instance; this is not EC2 stop/start",
        )
    return None, None


class _HasAliases(Protocol):
    @property
    def node_logical_id(self) -> str: ...

    @property
    def aliases(self) -> set[str]: ...


NodeT = TypeVar("NodeT", bound=_HasAliases)


def resolve_provider_nodes(
    identifiers: list[str],
    candidates: list[NodeT],
    *,
    error: type[ValueError],
    membership: str,
) -> list[NodeT]:
    """The provider nodes ``identifiers`` name, each exactly once.

    An identifier no node answers to and one two nodes answer to both raise
    ``error``: a lifecycle mutation must never be aimed by guesswork.
    """

    if not identifiers:
        raise error("at least one target node is required")
    index: dict[str, list[NodeT]] = {}
    for node in candidates:
        for alias in node.aliases:
            index.setdefault(alias, []).append(node)
    resolved = []
    for identifier in identifiers:
        matches = index.get(identifier, [])
        if not matches:
            raise error(f"node is not part of {membership}: {identifier}")
        unique = {item.node_logical_id: item for item in matches}
        if len(unique) != 1:
            raise error(f"node identifier is ambiguous: {identifier}")
        resolved.append(next(iter(unique.values())))
    return list({node.node_logical_id: node for node in resolved}.values())


class NodeLifecycleProvider(Protocol):
    """What the lifecycle step adapter and its confirmation need of a provider.

    ``HyperPodLifecycleAdapter`` and ``EksManagedNodeProvider`` both satisfy
    it; the step adapter is written against this surface and nothing else.
    """

    @property
    def config(self) -> Any: ...

    @property
    def provider_label(self) -> str: ...

    @property
    def confirmation_source(self) -> str: ...

    def list_nodes(self, *, enrich: bool = False) -> list[Any]: ...

    def resolve_nodes(
        self, identifiers: list[str], *, nodes: list[Any] | None = None
    ) -> list[Any]: ...

    def preflight(
        self,
        action: Any,
        node_identifiers: list[str],
        *,
        isolation_verified_nodes: list[str] | None = None,
        require_execution_enabled: bool = True,
    ) -> Any: ...

    def execute_step(
        self,
        step: Any,
        *,
        isolation_verified_nodes: list[str],
        confirm_cluster_name: str,
        workflow_fencing_token: int,
        expected_fencing_token: int,
        idempotency_key: str,
    ) -> Any: ...
