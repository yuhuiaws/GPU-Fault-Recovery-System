"""``EksManagedNodeProvider`` on the API shapes the main suite never feeds it.

The Kubernetes client's object models (not dicts), a node without a name,
``DescribeInstances`` pages and malformed instances, EKS answers without the
cluster or node group, a batch over the configured limit, a concurrent
submission of the same key by another executor, and the boto3 dependency the
provider builds its EC2 and EKS clients from. Every fake is the neighbouring
``tests/eks_provider/_support.py`` one or a subclass of it.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from gpu_fault.eks_provider import EksProviderError, eks_provider_from_environment
from gpu_fault.hyperpod import HyperPodAction
from gpu_fault.store import InMemoryStore
from tests.eks_provider._support import (
    CLUSTER_ID,
    EKS_CLUSTER,
    INSTANCE_A,
    INSTANCE_B,
    NODE_A,
    NODE_B,
    NODEGROUP,
    REGION,
    SELECTOR,
    FakeCore,
    FakeEc2,
    FakeEks,
    config,
    default_instances,
    default_nodes,
    gpu_node,
    provider,
)

SUBMIT = {
    "isolation_verified_nodes": [NODE_A],
    "confirm_cluster_name": CLUSTER_ID,
    "workflow_fencing_token": 3,
    "expected_fencing_token": 3,
    "idempotency_key": "workflow-1/RESTART_NODE/1",
}
ENVIRON = {
    "GPU_FAULT_CLUSTER_PROVIDER": "eks-managed",
    "GPU_FAULT_HYPERPOD_CLUSTER": CLUSTER_ID,
    "GPU_FAULT_NODE_SELECTOR": SELECTOR,
    "AWS_REGION": REGION,
}


def test_a_blank_cluster_id_is_refused() -> None:
    with pytest.raises(ValueError, match="cluster_id must not be blank"):
        config(cluster_id="   ")


class ObjectCore:
    """``CoreV1Api.list_node`` as the Kubernetes client returns it: models with
    snake_case attributes (``spec.provider_id``), not dicts."""

    def __init__(self) -> None:
        self.selectors: list[str | None] = []

    def list_node(self, label_selector: str | None = None, **_: Any) -> Any:
        self.selectors.append(label_selector)
        node = SimpleNamespace(
            metadata=SimpleNamespace(
                name=NODE_A,
                labels={
                    "nvidia.com/gpu.present": "true",
                    "eks.amazonaws.com/nodegroup": NODEGROUP,
                    "node.kubernetes.io/instance-type": "p5en.48xlarge",
                },
            ),
            spec=SimpleNamespace(
                provider_id=f"aws:///us-west-2a/{INSTANCE_A}",
                taints=[
                    SimpleNamespace(
                        key="nvidia.com/gpu", value="present", effect="NoSchedule"
                    )
                ],
            ),
        )
        return SimpleNamespace(items=[node])


def test_listing_reads_the_kubernetes_client_object_models() -> None:
    core = ObjectCore()
    adapter, *_ = provider(core=core)  # type: ignore[arg-type]

    (node,) = adapter.list_nodes()

    assert core.selectors == [SELECTOR]
    assert (node.node_logical_id, node.instance_id, node.status) == (
        NODE_A,
        INSTANCE_A,
        "Running",
    ), node
    assert node.instance_group_name == NODEGROUP
    assert node.availability_zone == "us-west-2a"
    assert node.kubernetes_taints == [
        {"key": "nvidia.com/gpu", "value": "present", "effect": "NoSchedule"}
    ]
    assert node.kubernetes_labels["kubernetes.io/hostname"] == NODE_A, (
        "the hostname label defaults to the node name"
    )


def test_a_node_without_a_name_is_refused() -> None:
    core = FakeCore(default_nodes())
    nameless = gpu_node(NODE_B, INSTANCE_B)
    nameless["metadata"]["name"] = ""
    core.nodes[NODE_B] = nameless
    adapter, *_ = provider(core=core)

    with pytest.raises(EksProviderError, match="missing metadata.name"):
        adapter.list_nodes()


class PagedEc2(FakeEc2):
    """``DescribeInstances`` one instance per page, with a malformed entry."""

    def describe_instances(self, **kwargs: Any) -> dict[str, Any]:
        self.describe_calls.append(kwargs)
        if "NextToken" not in kwargs:
            return {
                "Reservations": [
                    {"Instances": [self.instances[INSTANCE_A], {"InstanceId": ""}]},
                    {"Instances": [{"State": {"Name": "running"}}]},
                ],
                "NextToken": "page-2",
            }
        assert kwargs["NextToken"] == "page-2", kwargs
        return {"Reservations": [{"Instances": [self.instances[INSTANCE_B]]}]}


def test_instances_are_read_across_every_page_and_malformed_ones_are_skipped() -> None:
    ec2 = PagedEc2(default_instances())
    adapter, *_ = provider(ec2=ec2)

    nodes = {node.node_logical_id: node for node in adapter.list_nodes()}

    assert {name: node.status for name, node in nodes.items()} == {
        NODE_A: "Running",
        NODE_B: "Running",
    }, "the second page's instance is joined too"
    assert [call.get("NextToken") for call in ec2.describe_calls] == [None, "page-2"]


class SilentEks(FakeEks):
    """EKS answers that carry no cluster, or no node group."""

    def __init__(self, *, cluster: bool, nodegroup: bool) -> None:
        super().__init__()
        self.cluster = cluster
        self.nodegroup = nodegroup

    def describe_cluster(self, **kwargs: Any) -> dict[str, Any]:
        answer = super().describe_cluster(**kwargs)
        return answer if self.cluster else {"cluster": None}

    def describe_nodegroup(self, **kwargs: Any) -> dict[str, Any]:
        answer = super().describe_nodegroup(**kwargs)
        return answer if self.nodegroup else {}


@pytest.mark.parametrize(
    ("cluster", "nodegroup", "failure"),
    [
        (False, True, f"DescribeCluster returned no cluster for {EKS_CLUSTER}"),
        (
            True,
            False,
            f"DescribeNodegroup returned no nodegroup for {EKS_CLUSTER}/{NODEGROUP}",
        ),
    ],
)
def test_an_eks_answer_without_the_cluster_or_node_group_fails_preflight(
    cluster: bool, nodegroup: bool, failure: str
) -> None:
    adapter, *_ = provider(eks=SilentEks(cluster=cluster, nodegroup=nodegroup))

    result = adapter.preflight(
        HyperPodAction.REBOOT, [NODE_A], isolation_verified_nodes=[NODE_A]
    )

    assert result.safe_to_submit is False, "an unreadable control plane fails closed"
    assert failure in result.gate_failures, result.gate_failures


def test_a_batch_over_the_configured_limit_fails_preflight() -> None:
    adapter, *_ = provider(max_batch_size=1)

    result = adapter.preflight(
        HyperPodAction.REBOOT,
        [NODE_A, NODE_B],
        isolation_verified_nodes=[NODE_A, NODE_B],
    )

    assert result.gate_failures == ["batch exceeds limit 1"], result.gate_failures


def test_a_concurrent_submission_of_the_same_key_is_replayed_not_rebooted_twice() -> (
    None
):
    """Another executor sharing the durable ledger submits the same step while
    this one is inside its preflight: this one's reservation finds the other's
    outcome and replays it."""

    store = InMemoryStore()
    winner, _, winner_ec2, _ = provider(store=store)
    racing: list[Any] = []

    class RacedEc2(FakeEc2):
        def describe_instances(self, **kwargs: Any) -> dict[str, Any]:
            if not racing:
                racing.append(winner.submit(HyperPodAction.REBOOT, [NODE_A], **SUBMIT))
            return super().describe_instances(**kwargs)

    loser_ec2 = RacedEc2(default_instances())
    loser, *_ = provider(store=store, ec2=loser_ec2)

    replay = loser.submit(HyperPodAction.REBOOT, [NODE_A], **SUBMIT)

    (first,) = racing
    assert winner_ec2.reboots == [[INSTANCE_A]]
    assert loser_ec2.reboots == [], "the instance is rebooted once, by the winner"
    assert replay.duplicate is True
    assert replay.operation_id == first.operation_id


def test_the_provider_builds_its_clients_with_boto3(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[tuple[str, str | None]] = []
    clients = {"ec2": FakeEc2(default_instances()), "eks": FakeEks()}

    def client(service: str, region_name: str | None = None) -> Any:
        created.append((service, region_name))
        return clients[service]

    boto3 = ModuleType("boto3")
    boto3.client = client  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "boto3", boto3)

    adapter = eks_provider_from_environment(FakeCore(default_nodes()), environ=ENVIRON)

    assert created == [("ec2", REGION), ("eks", REGION)]
    assert (adapter.ec2, adapter.eks) == (clients["ec2"], clients["eks"])


def test_the_provider_without_boto3_names_the_extra_to_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "boto3", None)

    with pytest.raises(
        RuntimeError, match=r"install gpu-fault-control-plane\[hyperpod\]"
    ):
        eks_provider_from_environment(FakeCore(default_nodes()), environ=ENVIRON)
