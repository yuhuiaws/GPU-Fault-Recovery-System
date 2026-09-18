from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pytest
from kubernetes import client
from kubernetes.client.exceptions import ApiException

from gpu_fault.hyperpod import HyperPodNode
from gpu_fault.hyperpod_spares import (
    SPARE_POOL_STATE_ANNOTATION,
    SPARE_RESERVATION_ANNOTATION,
    HyperPodSpareCoordinator,
)
from gpu_fault.store import InMemoryStore
from tests.hyperpod._cov95_runtime_spares import TypedCore
from tests.hyperpod.test_hyperpod_spares import (
    FakeCore,
    FakeLifecycle,
    coordinator,
    hyperpod_node,
    kubernetes_node,
)


@pytest.mark.parametrize("unresolved", [False, True])
def test_allocator_never_selects_faulted_or_unaddressable_spare(
    unresolved: bool,
) -> None:
    fault = hyperpod_node("fault", "i-fault", spare=not unresolved)
    nodes = [fault]
    if unresolved:
        nodes.append(
            HyperPodNode(
                node_logical_id="unaddressable",
                status="Running",
                instance_group_name="workers",
                instance_type="ml.p5.48xlarge",
                kubernetes_labels={"gpu-fault.io/spare": "true"},
            )
        )
    service = HyperPodSpareCoordinator(
        FakeLifecycle(nodes), InMemoryStore(), FakeCore({})
    )
    result = service.allocate(
        cluster_id="hp-cluster", incident_id="owned", fault_node_ids=["fault"]
    )
    assert result.applicable and not result.sufficient, result
    assert result.selected_node_ids == () and service.core.patches == [], result
    if unresolved:
        assert result.rejected_candidates == (
            ("unaddressable", ("spare has no resolvable Kubernetes node name",)),
        ), result


@pytest.mark.parametrize("provider_tail", ["cluster-i-spare", "opaque-provider"])
def test_local_only_allocation_reads_typed_node_topology_and_never_lists_provider(
    provider_tail: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    nodes = {name: kubernetes_node() for name in ("compute-a", "compute-b")}
    for name, node in nodes.items():
        node["metadata"]["labels"].update(
            {
                "sagemaker.amazonaws.com/instance-group-name": "workers",
                "beta.kubernetes.io/instance-type": "ml.p5.48xlarge",
            }
        )
        node["spec"]["providerID"] = (
            f"aws:///fake/{provider_tail}"
            if name == "compute-b"
            else "aws:///fake/cluster-i-fault"
        )
    nodes["compute-b"]["metadata"]["labels"]["gpu-fault.io/spare"] = "true"
    core = TypedCore(nodes)
    lifecycle = FakeLifecycle([])

    def forbidden(**kwargs: Any) -> Any:
        pytest.fail("local-only allocation must not query the provider")

    monkeypatch.setattr(lifecycle, "list_nodes", forbidden)
    proofs: list[dict[str, Any]] = []
    service = HyperPodSpareCoordinator(
        lifecycle,
        InMemoryStore(),
        core,
        remote_health_provider=SimpleNamespace(
            spare_health_reasons=lambda **kwargs: proofs.append(kwargs) or []
        ),
    )
    result = service.allocate(
        cluster_id="hp-cluster",
        incident_id="owned",
        fault_node_ids=["i-fault"],
        local_only=True,
    )
    assert result.sufficient and result.selected_node_ids == ("compute-b",), result
    assert nodes["compute-b"]["spec"]["unschedulable"] is False, nodes
    assert (
        nodes["compute-b"]["metadata"]["annotations"][SPARE_RESERVATION_ANNOTATION]
        == "owned"
    ), nodes
    assert len(proofs) == 1 and proofs[0]["cluster_id"] == "hp-cluster", proofs
    assert ("i-spare" in proofs[0]["node_aliases"]) is (
        provider_tail == "cluster-i-spare"
    ), proofs


def test_local_node_inventory_without_a_name_never_reaches_reservation() -> None:
    node = kubernetes_node()
    node["metadata"]["name"] = None
    core = TypedCore({"invalid": node})
    service = HyperPodSpareCoordinator(FakeLifecycle([]), InMemoryStore(), core)
    with pytest.raises(ValueError, match="missing metadata.name"):
        service.allocate(
            cluster_id="hp-cluster",
            incident_id="owned",
            fault_node_ids=["invalid"],
            local_only=True,
        )
    assert core.patches == [], core.patches


@pytest.mark.parametrize(
    ("defect", "expected"),
    [
        ("provider-status", "HyperPod status is Pending"),
        ("reservation", "reserved by incident other"),
        ("pool", "spare pool state is REMEDIATING"),
        ("registry", "agent registry is unavailable"),
        ("agent", "expected one matching agent, found 0"),
        ("finding", "active GPU health finding exists"),
    ],
)
def test_spare_health_keeps_each_unproven_candidate_out_of_the_pool(
    defect: str, expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    node = hyperpod_node("spare", "i-spare", spare=True)
    core_node = kubernetes_node()
    service, store = coordinator([node], {"hyperpod-i-spare": core_node})
    if defect == "provider-status":
        node = node.model_copy(update={"status": "Pending"})
    elif defect == "reservation":
        core_node["metadata"]["annotations"][SPARE_RESERVATION_ANNOTATION] = "other"
    elif defect == "pool":
        core_node["metadata"]["annotations"][SPARE_POOL_STATE_ANNOTATION] = (
            "REMEDIATING"
        )
    elif defect == "registry":
        service.registry = None
    elif defect == "agent":
        monkeypatch.setattr(store, "list_agents", lambda _cluster: [])
    else:
        monkeypatch.setattr(
            store,
            "list_gpu_findings",
            lambda *_a, **_k: [SimpleNamespace(observed_at=datetime.now(timezone.utc))],
        )
    before = deepcopy(core_node)
    reasons = service.health_reasons(
        "hp-cluster", node, "hyperpod-i-spare", incident_id="owned"
    )
    assert expected in reasons, reasons
    assert service.core.patches == [] and core_node == before, core_node


@pytest.mark.parametrize("resource", ["1", "unreadable"])
def test_typed_gpu_pod_or_unknown_gpu_request_blocks_spare(resource: str) -> None:
    node = hyperpod_node("spare", "i-spare", spare=True)
    pod = client.V1Pod(
        metadata=client.V1ObjectMeta(),
        spec=client.V1PodSpec(
            containers=[
                client.V1Container(
                    name="trainer",
                    resources=client.V1ResourceRequirements(
                        requests={"nvidia.com/gpu": resource}
                    ),
                )
            ]
        ),
        status=client.V1PodStatus(phase="Pending"),
    )
    core_nodes = {"hyperpod-i-spare": kubernetes_node()}
    core = FakeCore(core_nodes, pod_batches=[[pod]])
    service, _ = coordinator([node], core_nodes, core=core)
    reasons = service.health_reasons("hp-cluster", node, "hyperpod-i-spare")
    assert "active GPU resource pods exist: default/unknown" in reasons, reasons
    assert core.patches == [], core.patches


@pytest.mark.parametrize("owner", ["owned", "foreign"])
def test_reserve_rechecks_racing_reservation_before_conditional_patch(
    owner: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    node = hyperpod_node("spare", "i-spare", spare=True)
    core_node = kubernetes_node()
    service, _ = coordinator([node], {"hyperpod-i-spare": core_node})
    original = service.core.read_node
    reads = 0

    def read(name: str) -> Any:
        nonlocal reads
        reads += 1
        if reads == 2:
            core_node["metadata"]["annotations"][SPARE_RESERVATION_ANNOTATION] = owner
            core_node["metadata"]["resourceVersion"] = "2"
        return original(name)

    monkeypatch.setattr(service.core, "read_node", read)
    if owner == "foreign":
        with pytest.raises(ValueError, match="reserved by foreign"):
            service.reserve(node, "hyperpod-i-spare", "owned")
    else:
        service.reserve(node, "hyperpod-i-spare", "owned")
    assert reads == 2 and service.core.patches == [], (reads, service.core.patches)
    assert (
        core_node["metadata"]["annotations"][SPARE_RESERVATION_ANNOTATION] == owner
    ), core_node


def test_release_ignores_confirmed_absence_but_cleans_every_remaining_owned_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core_nodes = {name: kubernetes_node() for name in ("owned", "foreign")}
    for name, node in core_nodes.items():
        node["metadata"]["annotations"][SPARE_RESERVATION_ANNOTATION] = name
    core = FakeCore(core_nodes)
    original = core.read_node

    def read(name: str) -> Any:
        if name == "absent":
            raise ApiException(status=404)
        return original(name)

    monkeypatch.setattr(core, "read_node", read)
    service = HyperPodSpareCoordinator(FakeLifecycle([]), InMemoryStore(), core)
    service.release(["owned", "foreign", "absent"], "owned")
    assert [name for name, _ in core.patches] == ["owned"], core.patches
    assert (
        core_nodes["owned"]["metadata"]["annotations"].get(SPARE_RESERVATION_ANNOTATION)
        is None
    ), core_nodes
    assert (
        core_nodes["foreign"]["metadata"]["annotations"][SPARE_RESERVATION_ANNOTATION]
        == "foreign"
    ), core_nodes


def test_storeless_shortage_is_reported_without_fabricating_persistent_notification() -> (
    None
):
    nodes = [
        hyperpod_node("fault", "i-fault"),
        hyperpod_node("spare", "i-spare", spare=True),
    ]
    service = HyperPodSpareCoordinator(
        FakeLifecycle(nodes),
        SimpleNamespace(),
        FakeCore({"hyperpod-i-spare": kubernetes_node()}),
    )
    result = service.allocate(
        cluster_id="hp-cluster", incident_id="owned", fault_node_ids=["fault"]
    )
    assert result.applicable and not result.sufficient, result
    assert result.notification_id is None, result
    assert "agent registry is unavailable" in (result.reason or ""), result
    assert service.core.patches == [], service.core.patches
