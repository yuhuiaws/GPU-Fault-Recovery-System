"""Shared node preflight classification: business workloads and blocking taints."""

from __future__ import annotations

from typing import Any

from gpu_fault.cluster_scope import EKS_NODEGROUP_LABEL, HYPERPOD_CLUSTER_LABEL

#: Taints a plain-EKS GPU node carries by construction: the managed node
#: group's GPU taint (``nvidia.com/gpu=present:NoSchedule``), which keeps
#: non-GPU Pods off the node and says nothing about this solution's state.
EKS_MANAGED_NODE_TAINTS = frozenset({("nvidia.com/gpu", "NoSchedule")})
#: Taint keys that always block a drill, whatever the provider: this
#: solution's own quarantine/ownership taints and the kubelet's unschedulable.
BLOCKING_TAINT_PREFIXES = ("gpu-fault.io/", "node.kubernetes.io/unschedulable")


def node_provider(labels: dict[str, Any] | None) -> str:
    """``hyperpod`` for a node carrying the HyperPod cluster label, ``eks-managed``
    for one carrying only an EKS node group label, else ``unknown``."""

    labels = labels or {}
    if labels.get(HYPERPOD_CLUSTER_LABEL):
        return "hyperpod"
    if labels.get(EKS_NODEGROUP_LABEL):
        return "eks-managed"
    return "unknown"


def blocking_node_taints(
    taints: list[dict[str, Any]] | None, labels: dict[str, Any] | None
) -> list[dict[str, Any]]:
    """The taints a preflight must treat as pre-existing isolation.

    HyperPod nodes carry no taint in steady state, so every taint blocks, as
    before. A plain-EKS node (live 2026-10-04: DESTR-001/010 refused on
    ``nvidia.com/gpu=present:NoSchedule``) legitimately carries its node group's
    GPU taint; that one is tolerated, and only on an eks-managed node. Anything
    under ``gpu-fault.io/``, the unschedulable taint and every ``NoExecute``
    taint block on every provider.
    """

    blocking = []
    tolerated = (
        EKS_MANAGED_NODE_TAINTS
        if node_provider(labels) == "eks-managed"
        else frozenset()
    )
    for taint in taints or []:
        key = str(taint.get("key") or "")
        effect = str(taint.get("effect") or "")
        if key.startswith(BLOCKING_TAINT_PREFIXES) or effect == "NoExecute":
            blocking.append(taint)
        elif (key, effect) not in tolerated:
            blocking.append(taint)
    return blocking


def tolerated_node_taints(
    taints: list[dict[str, Any]] | None, labels: dict[str, Any] | None
) -> list[dict[str, Any]]:
    blocking = blocking_node_taints(taints, labels)
    return [taint for taint in taints or [] if taint not in blocking]


SYSTEM_NAMESPACES = frozenset(
    {
        "aws-hyperpod",
        "cert-manager",
        "hyperpod-inference-system",
        "kube-system",
        "kubeflow",
    }
)


def pod_gpu_count(item: dict[str, Any]) -> int:
    """The largest GPU request or limit across a Pod's containers."""
    gpu_count = 0
    for container in item.get("spec", {}).get("containers", []):
        resources = container.get("resources", {})
        for values in (resources.get("requests", {}), resources.get("limits", {})):
            try:
                gpu_count = max(gpu_count, int(values.get("nvidia.com/gpu", 0)))
            except (TypeError, ValueError):
                pass
    return gpu_count


#: Owner kinds that mark a Pod as node infrastructure rather than a workload:
#: one replica per node, re-created by the controller, never a user's job.
INFRASTRUCTURE_OWNER_KINDS = frozenset({"DaemonSet"})


def pod_owner_kind(item: dict[str, Any]) -> str:
    owners = item.get("metadata", {}).get("ownerReferences") or []
    return str((owners[0] if owners else {}).get("kind") or "")


def business_workload_reason(
    item: dict[str, Any],
    *,
    namespace: str,
    allowed_namespaces: frozenset[str] | set[str] = frozenset(),
) -> str | None:
    """Why a Pod counts as a business workload, or ``None`` for infrastructure.

    The historical rule first: Pods in the system namespaces are infrastructure,
    and in the solution namespace only a Pod requesting a GPU is a workload.
    Elsewhere a GPU request is a workload wherever it runs; without one, a
    DaemonSet-owned Pod (node agents such as device plugins, CNI/EFA plugins,
    exporters; live 2026-10-04 a plain-EKS node carried the customer's
    node-exporter) is infrastructure, and any other Pod is a workload: in an
    allowed workload namespace by declaration, elsewhere because it is
    unclassified and the drill must not disturb it.
    """

    pod_namespace = str(item["metadata"].get("namespace", ""))
    if pod_namespace in SYSTEM_NAMESPACES:
        return None
    gpu = pod_gpu_count(item) > 0
    if pod_namespace == namespace:
        return "gpu-request" if gpu else None
    if gpu:
        return "gpu-request"
    if pod_owner_kind(item) in INFRASTRUCTURE_OWNER_KINDS:
        return None
    if pod_namespace in allowed_namespaces:
        return "allowed-workload-namespace"
    return "unclassified-workload"


def business_workload_details(
    items: list[dict[str, Any]],
    *,
    namespace: str,
    allowed_namespaces: frozenset[str] | set[str] = frozenset(),
) -> list[dict[str, str]]:
    """Business Pods with their owner kind and the reason they count."""

    result = []
    for item in items:
        reason = business_workload_reason(
            item, namespace=namespace, allowed_namespaces=allowed_namespaces
        )
        if reason is None:
            continue
        result.append(
            {
                "namespace": str(item["metadata"].get("namespace", "")),
                "name": str(item["metadata"].get("name", "")),
                "owner_kind": pod_owner_kind(item),
                "reason": reason,
            }
        )
    return result


def business_workload_items(
    items: list[dict[str, Any]],
    *,
    namespace: str,
    allowed_namespaces: frozenset[str] | set[str] = frozenset(),
) -> list[dict[str, str]]:
    """Keep business Pods, including GPU jobs in the solution namespace."""
    return [
        {"namespace": item["namespace"], "name": item["name"]}
        for item in business_workload_details(
            items, namespace=namespace, allowed_namespaces=allowed_namespaces
        )
    ]


#: The workload kinds the managed-workload audits enumerate, as ``kubectl
#: api-resources -o name`` spells them. A plain-EKS cluster without the Kubeflow
#: training operator serves no ``pytorchjobs.kubeflow.org`` (live 2026-10-04:
#: DESTR-012 and DESTR-015 stopped on "the server doesn't have a resource type").
WORKLOAD_RESOURCE_KINDS = ("pytorchjobs.kubeflow.org", "jobs.batch", "pods")


def served_resource_names(api_resources_output: str) -> frozenset[str]:
    """The resource names a cluster serves, from ``kubectl api-resources -o name``."""

    return frozenset(
        line.strip() for line in api_resources_output.splitlines() if line.strip()
    )


def resource_kind_served(served: frozenset[str], kind: str) -> bool:
    """Whether ``kind`` (``pods``, ``jobs.batch``, ``pytorchjobs.kubeflow.org``)
    is served; a core kind may be listed without its empty group suffix."""

    return kind in served or kind.split(".", 1)[0] in served
