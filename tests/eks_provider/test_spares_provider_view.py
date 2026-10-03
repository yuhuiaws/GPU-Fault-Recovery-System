"""``hyperpod_spares`` reads plain-EKS nodes: providerID, node group, no HMA gate.

Driven through ``HyperPodSpareCoordinator.allocate(local_only=True)``, the
Kubernetes-only path: the fault node is named by the EC2 instance id its
``aws:///<zone>/i-...`` providerID carries, the spare must share the EKS
managed node group and instance type, and on ``eks-managed`` the HyperPod
health-monitoring-agent label is not a health gate.
"""

from __future__ import annotations

from typing import Any

from gpu_fault.hyperpod_spares import HyperPodSpareCoordinator
from gpu_fault.node_provider import ClusterProvider
from gpu_fault.store import InMemoryStore
from tests.eks_provider._support import (
    INSTANCE_A,
    INSTANCE_B,
    NODE_A,
    NODE_B,
    NODEGROUP,
    FakeCore,
    gpu_node,
    provider,
)

SPARE_LABEL = "gpu-fault.io/spare"


class _PodlessCore(FakeCore):
    def list_pod_for_all_namespaces(self, **_kwargs: Any) -> dict[str, Any]:
        return {"items": []}


class _RemoteHealth:
    def spare_health_reasons(self, **_kwargs: Any) -> list[str]:
        return []


def _cluster(
    *, spare_nodegroup: str = NODEGROUP, spare_zone: str = "us-west-2a"
) -> _PodlessCore:
    fault = gpu_node(NODE_A, INSTANCE_A)
    spare = gpu_node(
        NODE_B,
        INSTANCE_B,
        zone=spare_zone,
        nodegroup=spare_nodegroup,
        labels={SPARE_LABEL: "true"},
    )
    spare["spec"]["unschedulable"] = True
    return _PodlessCore([fault, spare])


def _coordinator(
    core: _PodlessCore, cluster_provider: ClusterProvider
) -> HyperPodSpareCoordinator:
    lifecycle, *_ = provider(core=core)
    return HyperPodSpareCoordinator(
        lifecycle,
        InMemoryStore(),
        core,
        remote_health_provider=_RemoteHealth(),
        spare_label=SPARE_LABEL,
        provider=cluster_provider,
    )


def test_local_allocation_resolves_the_fault_node_by_its_provider_id_instance() -> None:
    core = _cluster()
    coordinator = _coordinator(core, ClusterProvider.EKS_MANAGED)

    allocation = coordinator.allocate(
        cluster_id="gpu-a",
        incident_id="incident-a",
        fault_node_ids=[INSTANCE_A],
        local_only=True,
    )

    assert allocation.applicable is True
    assert allocation.sufficient is True, allocation.reason
    assert allocation.selected_node_ids == (NODE_B,)
    assert allocation.rejected_candidates == ()
    reserved = core.nodes[NODE_B]["metadata"]["annotations"]
    assert reserved["gpu-fault.io/spare-reservation"] == "incident-a"


def test_local_allocation_matches_spares_by_eks_node_group() -> None:
    coordinator = _coordinator(
        _cluster(spare_nodegroup="other-node-group"), ClusterProvider.EKS_MANAGED
    )

    allocation = coordinator.allocate(
        cluster_id="gpu-a",
        incident_id="incident-a",
        fault_node_ids=[NODE_A],
        local_only=True,
    )

    assert allocation.applicable is True
    assert allocation.sufficient is False
    assert allocation.reason is not None
    assert NODEGROUP in allocation.reason, allocation.reason


def test_the_hma_health_label_gates_spares_only_on_hyperpod() -> None:
    eks = _coordinator(_cluster(), ClusterProvider.EKS_MANAGED)
    hyperpod = _coordinator(_cluster(), ClusterProvider.HYPERPOD)

    eks_allocation = eks.allocate(
        cluster_id="gpu-a",
        incident_id="incident-a",
        fault_node_ids=[NODE_A],
        local_only=True,
    )
    hyperpod_allocation = hyperpod.allocate(
        cluster_id="gpu-a",
        incident_id="incident-a",
        fault_node_ids=[NODE_A],
        local_only=True,
    )

    assert eks_allocation.sufficient is True
    assert hyperpod_allocation.sufficient is False
    rejections = dict(hyperpod_allocation.rejected_candidates)
    assert "HyperPod node health is unknown" in rejections[NODE_B], rejections
    assert "HyperPod topology is unknown" in rejections[NODE_B], (
        "a HyperPod reading looks for the SageMaker instance-group label"
    )
