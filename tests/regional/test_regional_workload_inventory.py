"""Which Pods on a GPU node stop a drill: workloads, not node infrastructure.

Live 2026-10-04 (NET-001 on the mixed site's plain-EKS cluster): the target
node carried the customer's ``monitoring/kube-prometheus-stack-prometheus-
node-exporter`` DaemonSet Pod and the preflight refused with "target node has
an active business workload"; a HyperPod node only carries HyperPod system Pods,
so the rule never met a foreign DaemonSet before.
"""

from __future__ import annotations

from typing import Any

from scripts.e2e.regional import run_net001_collector_replay as net001
from scripts.e2e.regional.regional_workload_inventory import (
    WORKLOAD_RESOURCE_KINDS,
    blocking_node_taints,
    business_workload_details,
    business_workload_items,
    business_workload_reason,
    node_provider,
    resource_kind_served,
    served_resource_names,
    tolerated_node_taints,
)


def _pod(
    namespace: str, name: str, *, owner: str | None = "ReplicaSet", gpus: int = 0
) -> dict[str, Any]:
    metadata: dict[str, Any] = {"namespace": namespace, "name": name}
    if owner:
        metadata["ownerReferences"] = [{"kind": owner, "name": f"{name}-owner"}]
    containers = [{"name": "main", "resources": {}}]
    if gpus:
        containers[0]["resources"] = {"limits": {"nvidia.com/gpu": str(gpus)}}
    return {"metadata": metadata, "spec": {"containers": containers}}


def test_daemonset_agents_without_a_gpu_are_infrastructure() -> None:
    node_exporter = _pod(
        "monitoring",
        "kube-prometheus-stack-prometheus-node-exporter-x",
        owner="DaemonSet",
    )
    assert business_workload_reason(node_exporter, namespace="gpu-fault-system") is None
    assert (
        business_workload_reason(
            _pod("kube-system", "aws-node-q", owner="DaemonSet"),
            namespace="gpu-fault-system",
        )
        is None
    )
    assert (
        business_workload_reason(
            _pod("gpu-fault-system", "gpu-fault-dcgm-exporter-k", owner="DaemonSet"),
            namespace="gpu-fault-system",
        )
        is None
    ), "the solution's own non-GPU Pods are infrastructure, as before"


def test_gpu_requests_are_workloads_wherever_they_run() -> None:
    assert (
        business_workload_reason(
            _pod("monitoring", "gpu-burn", owner="DaemonSet", gpus=1),
            namespace="gpu-fault-system",
        )
        == "gpu-request"
    ), "a DaemonSet holding a GPU is still a workload the drill would disturb"
    assert (
        business_workload_reason(
            _pod("kube-system", "gpu-health", owner="DaemonSet", gpus=1),
            namespace="gpu-fault-system",
        )
        is None
    ), "the system namespaces stay infrastructure even with a GPU, as before"
    assert (
        business_workload_reason(
            _pod("gpu-fault-system", "validation-job", owner="Job", gpus=8),
            namespace="gpu-fault-system",
        )
        == "gpu-request"
    )


def test_non_gpu_pods_outside_system_namespaces_stay_fail_closed() -> None:
    """A Deployment Pod in a customer namespace is a workload we cannot classify
    (as before); a Pod in a declared workload namespace is one by declaration."""

    grafana = _pod("monitoring", "grafana-7d", owner="ReplicaSet")
    assert (
        business_workload_reason(grafana, namespace="gpu-fault-system")
        == "unclassified-workload"
    )
    launcher = _pod("training", "launcher-0", owner="Job")
    assert (
        business_workload_reason(
            launcher, namespace="gpu-fault-system", allowed_namespaces={"training"}
        )
        == "allowed-workload-namespace"
    )


def test_items_keep_the_legacy_shape_and_details_carry_owner_and_reason() -> None:
    pods = [
        _pod("monitoring", "node-exporter-a", owner="DaemonSet"),
        _pod("training", "trainer-0", owner="Job", gpus=4),
        _pod("kube-system", "coredns-1"),
    ]
    assert business_workload_items(pods, namespace="gpu-fault-system") == [
        {"namespace": "training", "name": "trainer-0"}
    ], "every preflight keeps reading the fleet-wide namespace/name items"
    details = business_workload_details(pods, namespace="gpu-fault-system")
    assert details == [
        {
            "namespace": "training",
            "name": "trainer-0",
            "owner_kind": "Job",
            "reason": "gpu-request",
        }
    ]
    assert net001.business_workload_refusal(details) == (
        "target node has an active business workload: training/trainer-0 (gpu-request)"
    ), "the NET-001 refusal names the Pods instead of a bare sentence"


# -- taints: a plain-EKS node's node-group GPU taint is not isolation -------------

GPU_TAINT = {"key": "nvidia.com/gpu", "value": "present", "effect": "NoSchedule"}
QUARANTINE = {
    "key": "gpu-fault.io/quarantine",
    "value": "inc-1",
    "effect": "NoSchedule",
}
UNSCHEDULABLE = {"key": "node.kubernetes.io/unschedulable", "effect": "NoSchedule"}
EKS_LABELS = {
    "eks.amazonaws.com/nodegroup": "gpu-p5en",
    "nvidia.com/gpu.present": "true",
}
HP_LABELS = {"sagemaker.amazonaws.com/cluster-name": "hp-a"}


def test_node_provider_follows_the_provider_labels() -> None:
    assert node_provider(HP_LABELS) == "hyperpod"
    assert node_provider(EKS_LABELS) == "eks-managed"
    assert node_provider({}) == "unknown"
    assert node_provider({**EKS_LABELS, **HP_LABELS}) == "hyperpod", (
        "a HyperPod label wins: HyperPod nodes also carry EKS node group labels"
    )


def test_eks_managed_nodes_tolerate_only_their_node_group_gpu_taint() -> None:
    """Live 2026-10-04: DESTR-001/010 refused the plain-EKS node for
    ``nvidia.com/gpu=present:NoSchedule``, the taint its managed node group puts
    on every GPU node by construction."""

    assert blocking_node_taints([GPU_TAINT], EKS_LABELS) == []
    assert tolerated_node_taints([GPU_TAINT], EKS_LABELS) == [GPU_TAINT]
    assert blocking_node_taints([GPU_TAINT, QUARANTINE], EKS_LABELS) == [QUARANTINE]
    assert blocking_node_taints([UNSCHEDULABLE], EKS_LABELS) == [UNSCHEDULABLE]
    no_execute = {**GPU_TAINT, "effect": "NoExecute"}
    assert blocking_node_taints([no_execute], EKS_LABELS) == [no_execute], (
        "a NoExecute GPU taint evicts Pods: never a steady-state node group taint"
    )


def test_hyperpod_and_unknown_nodes_block_on_every_taint_as_before() -> None:
    assert blocking_node_taints([GPU_TAINT], HP_LABELS) == [GPU_TAINT]
    assert blocking_node_taints([GPU_TAINT], {}) == [GPU_TAINT]
    assert blocking_node_taints([], HP_LABELS) == []
    assert tolerated_node_taints([GPU_TAINT], HP_LABELS) == []


# -- served resource kinds: no Kubeflow on a plain EKS cluster -------------------


def test_served_resource_names_follow_kubectl_api_resources_output() -> None:
    """Live 2026-10-04: DESTR-012 and DESTR-015 stopped on the plain-EKS cluster
    because `kubectl get pytorchjobs.kubeflow.org,...` exits 1 when the Kubeflow
    training operator is not installed."""

    served = served_resource_names("pods\njobs.batch\nnodes\n\n")
    assert resource_kind_served(served, "pods"), "core kinds are listed bare"
    assert resource_kind_served(served, "jobs.batch"), "grouped kinds carry their group"
    assert not resource_kind_served(served, "pytorchjobs.kubeflow.org"), (
        "an unserved CRD is reported, never assumed"
    )
    assert [k for k in WORKLOAD_RESOURCE_KINDS if resource_kind_served(served, k)] == [
        "jobs.batch",
        "pods",
    ]
    hyperpod = served_resource_names("pytorchjobs.kubeflow.org\njobs.batch\npods\n")
    assert all(resource_kind_served(hyperpod, k) for k in WORKLOAD_RESOURCE_KINDS), (
        "a HyperPod cluster with Kubeflow enumerates every kind, as before"
    )
