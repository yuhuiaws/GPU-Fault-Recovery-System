"""The warm-spare cases read provider facts per provider, HyperPod or plain EKS.

On ``eks-managed`` there is no ``sagemaker describe-cluster``, no
health-monitoring-agent label and no instance group: the cluster fact is the
EKS cluster status, the inventory is the GPU nodes as Kubernetes reports them,
the health gate is the Ready condition alone and the topology is the managed
node group. The HyperPod reads are byte for byte what they were.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import run_destr003_warm_spare_failover as destr003
from scripts.e2e.regional import warm_spare_provider as provider
from scripts.e2e.regional.warm_spare_fixture import WarmSpareLiveFixture

EKS_NODEGROUP = "eks.amazonaws.com/nodegroup"
SAGEMAKER_CLUSTER = "sagemaker.amazonaws.com/cluster-name"
NODE_A = "ip-10-0-33-205.us-west-2.compute.internal"
NODE_B = "ip-10-0-41-164.us-west-2.compute.internal"
NOW = datetime(2026, 10, 5, 2, tzinfo=timezone.utc)


def _eks_node(name: str, instance: str, *, ready: bool = True) -> dict[str, Any]:
    return {
        "metadata": {
            "name": name,
            "labels": {
                EKS_NODEGROUP: "gpu-p5en-efa-spot-node-group",
                "node.kubernetes.io/instance-type": "p5en.48xlarge",
            },
        },
        "spec": {"providerID": f"aws:///us-west-2c/{instance}"},
        "status": {
            "allocatable": {"nvidia.com/gpu": "8"},
            "conditions": [{"type": "Ready", "status": "True" if ready else "False"}],
        },
    }


def _snapshot(labels: dict[str, Any]) -> dict[str, Any]:
    return {"labels": labels, "annotations": {}, "ready": "True", "unschedulable": True}


def test_the_gpu_nodes_labels_name_the_provider() -> None:
    eks = {"items": [_eks_node(NODE_A, "i-0a"), {"metadata": {"labels": {}}}]}
    assert provider.gpu_node_provider(eks) == "eks-managed"
    hyperpod = {
        "items": [
            {
                "metadata": {"labels": {SAGEMAKER_CLUSTER: "hp"}},
                "status": {"allocatable": {"nvidia.com/gpu": "8"}},
            }
        ]
    }
    assert provider.gpu_node_provider(hyperpod) == "hyperpod"
    assert provider.gpu_node_provider({"items": []}) == "hyperpod", (
        "nothing saying EKS keeps the HyperPod reading"
    )


def test_topology_and_health_follow_the_provider() -> None:
    eks = _snapshot({EKS_NODEGROUP: "ng-a", "node.kubernetes.io/instance-type": "p5en"})
    assert provider.topology(eks, "eks-managed") == {
        "instance_group": "ng-a",
        "instance_type": "p5en",
    }
    assert provider.spare_health_errors(eks, "eks-managed") == []
    assert provider.spare_health_errors(eks, "hyperpod") == [
        "spare HyperPod health label is not Schedulable"
    ]
    assert provider.spare_health_errors(_snapshot({}), "eks-managed") == [
        "spare eks-managed node carries no managed node group label"
    ]
    hyperpod = _snapshot(
        {
            provider.HYPERPOD_HEALTH_LABEL: "Schedulable",
            provider.INSTANCE_GROUP_LABEL: "workers",
            "beta.kubernetes.io/instance-type": "ml.p5.48xlarge",
        }
    )
    assert provider.topology(hyperpod, None) == {
        "instance_group": "workers",
        "instance_type": "ml.p5.48xlarge",
    }, "absent provider means HyperPod, as before"
    assert provider.spare_health_errors(hyperpod, None) == []


def test_cluster_facts_are_judged_per_provider() -> None:
    assert (
        provider.cluster_recovery_errors(
            {"status": "InService", "node_recovery": "None"}, "hyperpod"
        )
        == []
    )
    assert provider.cluster_recovery_errors(
        {"status": "InService", "node_recovery": "Automatic"}, "hyperpod"
    ) == ["HyperPod cluster is not InService with NodeRecovery=None"]
    eks = provider.eks_cluster_recovery(
        {"cluster": {"name": "gpu-eks-cluster", "status": "ACTIVE"}}, "gpu-eks-cluster"
    )
    assert eks == {
        "cluster_name": "gpu-eks-cluster",
        "status": "ACTIVE",
        "node_recovery": "not-applicable",
        "provider": "eks-managed",
    }
    assert provider.cluster_recovery_errors(eks, "eks-managed") == []
    assert provider.cluster_recovery_errors(
        {**eks, "status": "UPDATING"}, "eks-managed"
    ) == ["EKS cluster is not ACTIVE (plain EKS has no managed node recovery)"]
    assert provider.lifecycle_owner("eks-managed") == "gpu-fault-eks-adapter"
    assert provider.lifecycle_owner(None) == "gpu-fault-hyperpod-adapter"


def test_the_eks_inventory_is_the_gpu_nodes_as_kubernetes_reports_them() -> None:
    nodes = {
        "items": [
            _eks_node(NODE_B, "i-0b", ready=False),
            _eks_node(NODE_A, "i-0a"),
            {"metadata": {"name": "cpu-node", "labels": {}}, "status": {}},
        ]
    }
    inventory = provider.eks_provider_inventory(nodes)
    assert inventory["count"] == 2
    assert [row["node_logical_id"] for row in inventory["nodes"]] == [NODE_A, NODE_B]
    assert inventory["nodes"][0] == {
        "node_logical_id": NODE_A,
        "instance_id": "i-0a",
        "instance_group": "gpu-p5en-efa-spot-node-group",
        "instance_type": "p5en.48xlarge",
        "status": "Running",
    }
    assert inventory["nodes"][1]["status"] == "NotReady"
    again = provider.eks_provider_inventory(nodes)
    assert again["sha256"] == inventory["sha256"], (
        "the digest is stable for the same nodes"
    )
    replaced = provider.eks_provider_inventory(
        {"items": [_eks_node(NODE_A, "i-0a"), _eks_node(NODE_B, "i-0c")]}
    )
    assert replaced["sha256"] != inventory["sha256"], "a replaced instance is drift"


class _Regional:
    """The fixture's transport: ``kubectl`` for nodes, ``run`` for aws calls."""

    def __init__(self, nodes: dict[str, Any], describe: dict[str, Any]) -> None:
        self.nodes = nodes
        self.describe = describe
        self.commands: list[list[str]] = []
        self.settings = SimpleNamespace(region="us-west-2")

    def kubectl(self, plane: str, *args: str, **_kwargs: Any) -> str:
        assert (plane, args[:2]) == ("gpu", ("get", "node")), (plane, args)
        return json.dumps(self.nodes)

    def run(self, argv: list[str], **_kwargs: Any) -> SimpleNamespace:
        self.commands.append(list(argv))
        if argv[:3] == ["aws", "eks", "describe-cluster"]:
            return SimpleNamespace(stdout=json.dumps(self.describe))
        if argv[:3] == ["aws", "cloudtrail", "lookup-events"]:
            return SimpleNamespace(
                stdout=json.dumps(
                    {
                        "Events": [
                            {"EventName": "TerminateInstances", "EventTime": "t"},
                            {"EventName": "DescribeInstances", "EventTime": "t"},
                            {"EventName": "BatchReplaceClusterNodes", "EventTime": "t"},
                        ]
                    }
                )
            )
        raise AssertionError(f"unexpected command {argv}")


def test_the_fixture_reads_eks_facts_without_sagemaker() -> None:
    regional = _Regional(
        {"items": [_eks_node(NODE_A, "i-0a"), _eks_node(NODE_B, "i-0b")]},
        {"cluster": {"name": "gpu-eks-cluster", "status": "ACTIVE"}},
    )
    warm = WarmSpareLiveFixture(regional, "gpu-eks-cluster")  # type: ignore[arg-type]

    assert warm.provider == "eks-managed", "derived once from the GPU nodes' labels"
    assert warm.cluster_recovery() == {
        "cluster_name": "gpu-eks-cluster",
        "status": "ACTIVE",
        "node_recovery": "not-applicable",
        "provider": "eks-managed",
    }
    assert warm.provider_inventory()["count"] == 2
    calls = warm.provider_calls(NOW, NOW)
    assert {item["event_name"] for item in calls["mutations"]} == {
        "TerminateInstances",
        "BatchReplaceClusterNodes",
    }, "both event sources are read and only replacement mutations kept"
    assert calls["refused_calls"] == [], "accepted calls only; nothing was refused"
    sources = [
        item[item.index("--lookup-attributes") + 1]
        for item in regional.commands
        if item[:3] == ["aws", "cloudtrail", "lookup-events"]
    ]
    assert sources == [
        "AttributeKey=EventSource,AttributeValue=sagemaker.amazonaws.com",
        "AttributeKey=EventSource,AttributeValue=ec2.amazonaws.com",
    ]
    assert not any(item[:2] == ["aws", "sagemaker"] for item in regional.commands), (
        "no SageMaker call is made for a plain EKS cluster"
    )
    explicit = WarmSpareLiveFixture(regional, "hp", provider="hyperpod")  # type: ignore[arg-type]
    assert explicit.provider == "hyperpod", "a given provider is not re-derived"


def _settings() -> SimpleNamespace:
    return SimpleNamespace(fault_node=NODE_A, spare_node=NODE_B)


def _node(name: str, *, spare: bool) -> dict[str, Any]:
    return {
        "name": name,
        "uid": f"uid-{name}",
        "ready": "True",
        "unschedulable": spare,
        "taints": [],
        "labels": {
            "gpu-fault.io/spare": "true" if spare else None,
            EKS_NODEGROUP: "gpu-p5en-efa-spot-node-group",
            "node.kubernetes.io/instance-type": "p5en.48xlarge",
        },
        "annotations": {},
    }


def _eks_profile() -> dict[str, Any]:
    return {
        "capabilities": [
            {
                "capability": name,
                "mode": "OWN",
                "owner": owner,
                "adapter": "regional-cluster-executor",
            }
            for name, owner in (
                ("nodeReplace", "gpu-fault-eks-adapter"),
                ("workloadStop", "gpu-fault-kubernetes-adapter"),
                ("workloadRestart", "gpu-fault-kubernetes-adapter"),
            )
        ]
    }


def test_destr003_preflight_accepts_an_eks_managed_cluster_with_a_declared_spare() -> (
    None
):
    state = {
        "profile": _eks_profile(),
        "agents": [
            {"node_id": NODE_A, "lifecycle_state": "ACTIVE"},
            {"node_id": NODE_B, "lifecycle_state": "ACTIVE"},
        ],
        "queue": {"fault_backlog_depth": 0},
        "remote_commands": {"open_by_cluster": {}},
    }
    cluster = {
        "cluster_name": "gpu-eks-cluster",
        "status": "ACTIVE",
        "node_recovery": "not-applicable",
        "provider": "eks-managed",
    }
    arguments = dict(
        fault=_node(NODE_A, spare=False),
        spare=_node(NODE_B, spare=True),
        state=state,
        spare_nodes=[NODE_B],
        cluster=cluster,
        executor_env=[
            {"spare_failover": "true", "remote_state": "true", "allow_replace": "false"}
        ],
        synthetic_gates=[{"enabled": "true"}],
        gpu_workloads=[],
        predecessor={"valid": True},
        tests={"passed": True},
    )
    assert destr003.preflight_errors(_settings(), **arguments) == []
    # The same inputs judged as HyperPod refuse on every provider-specific fact.
    hyperpod = destr003.preflight_errors(
        _settings(), **{**arguments, "cluster": {**cluster, "provider": "hyperpod"}}
    )
    assert "spare HyperPod health label is not Schedulable" in hyperpod
    assert "nodeReplace has the wrong owner or adapter" in hyperpod
    assert "HyperPod cluster is not InService with NodeRecovery=None" in hyperpod


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"status": "UPDATING"}, "EKS cluster is not ACTIVE"),
        ({"node_recovery": "Automatic"}, "EKS cluster is not ACTIVE"),
    ],
)
def test_destr003_preflight_refuses_an_unhealthy_eks_cluster(
    overrides: dict[str, Any], expected: str
) -> None:
    errors = destr003.preflight_errors(
        _settings(),
        fault=_node(NODE_A, spare=False),
        spare=_node(NODE_B, spare=True),
        state={
            "profile": _eks_profile(),
            "agents": [
                {"node_id": NODE_A, "lifecycle_state": "ACTIVE"},
                {"node_id": NODE_B, "lifecycle_state": "ACTIVE"},
            ],
            "queue": {"fault_backlog_depth": 0},
            "remote_commands": {"open_by_cluster": {}},
        },
        spare_nodes=[NODE_B],
        cluster={
            "status": "ACTIVE",
            "node_recovery": "not-applicable",
            "provider": "eks-managed",
            **overrides,
        },
        executor_env=[
            {"spare_failover": "true", "remote_state": "true", "allow_replace": "false"}
        ],
        synthetic_gates=[{"enabled": "true"}],
        gpu_workloads=[],
        predecessor={"valid": True},
        tests={"passed": True},
    )
    assert any(item.startswith(expected) for item in errors), errors
