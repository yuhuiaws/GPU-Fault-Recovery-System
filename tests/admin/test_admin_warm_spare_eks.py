"""``config spare`` on a plain-EKS (eks-managed) GPU node.

The HyperPod health label is a gate only where the health-monitoring-agent
exists; on eks-managed the readiness is the Kubernetes Ready condition plus the
Agent, and the topology a spare must share with the fault node is the EKS
managed node group and the instance type. The site's cluster entry names the
provider; a node's own labels say it when the caller has no site.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from gpu_fault.admin.warm_spare import (
    EKS_NODEGROUP_LABEL,
    HYPERPOD_HEALTH_LABEL,
    INSTANCE_GROUP_LABEL,
    SPARE_LABEL,
    AgentState,
    WarmSpareError,
    WarmSpareRequest,
    declare_refusals,
    node_snapshot,
    perform_warm_spare,
    run_warm_spare,
    topology,
)
from tests.admin.test_admin_warm_spare import ACTIVE, FakeCoreApi, _raw_node

NODE = "ip-10-0-41-164.us-west-2.compute.internal"
FAULT = "ip-10-0-33-205.us-west-2.compute.internal"
NODEGROUP = "gpu-p5en-efa-spot-node-group"


def _eks_node(name: str, *, nodegroup: str = NODEGROUP, **overrides: Any) -> dict:
    node = _raw_node(name, **overrides)
    labels = node["metadata"]["labels"]
    labels.pop(HYPERPOD_HEALTH_LABEL, None)
    labels.pop(INSTANCE_GROUP_LABEL, None)
    labels[EKS_NODEGROUP_LABEL] = nodegroup
    labels["node.kubernetes.io/instance-type"] = "p5en.48xlarge"
    return node


def test_the_snapshot_names_the_provider_from_the_node_labels() -> None:
    eks = node_snapshot(_eks_node(NODE))
    assert eks["provider"] == "eks-managed"
    assert eks["labels"][EKS_NODEGROUP_LABEL] == NODEGROUP
    assert eks["labels"][HYPERPOD_HEALTH_LABEL] is None
    assert topology(eks) == {
        "instance_group": NODEGROUP,
        "instance_type": "p5en.48xlarge",
    }, "the node group plays the instance group's role"
    hyperpod = node_snapshot(_raw_node(NODE))
    assert hyperpod["provider"] == "hyperpod"
    assert topology(hyperpod)["instance_group"] == "group-a"


def test_an_eks_node_without_the_hyperpod_health_label_can_be_declared() -> None:
    spare = node_snapshot(_eks_node(NODE))
    assert (
        declare_refusals(
            NODE,
            spare=spare,
            fault=None,
            fault_node="",
            declared=[],
            workloads=[],
            agent=ACTIVE,
        )
        == []
    ), "no health-monitoring-agent on plain EKS, so its label is not a gate"
    # The same node judged as HyperPod would be refused: the rules follow the provider.
    assert "HyperPod health label is not Schedulable" in declare_refusals(
        NODE,
        spare=spare,
        fault=None,
        fault_node="",
        declared=[],
        workloads=[],
        agent=ACTIVE,
        provider="hyperpod",
    )


def test_an_eks_spare_must_share_the_fault_nodes_node_group() -> None:
    spare = node_snapshot(_eks_node(NODE))
    fault = node_snapshot(_eks_node(FAULT, nodegroup="gpu-other-group"))
    refusals = declare_refusals(
        NODE,
        spare=spare,
        fault=fault,
        fault_node=FAULT,
        declared=[],
        workloads=[],
        agent=ACTIVE,
    )
    assert any(item.startswith("topology does not match") for item in refusals), (
        refusals
    )
    assert "gpu-other-group" in " ".join(refusals)
    same_group = node_snapshot(_eks_node(FAULT))
    assert (
        declare_refusals(
            NODE,
            spare=spare,
            fault=same_group,
            fault_node=FAULT,
            declared=[],
            workloads=[],
            agent=ACTIVE,
        )
        == []
    )


def test_an_eks_node_without_a_node_group_label_is_refused() -> None:
    bare = _eks_node(NODE)
    del bare["metadata"]["labels"][EKS_NODEGROUP_LABEL]
    refusals = declare_refusals(
        NODE,
        spare=node_snapshot(bare),
        fault=None,
        fault_node="",
        declared=[],
        workloads=[],
        agent=ACTIVE,
        provider="eks-managed",
    )
    assert refusals == ["eks-managed node carries no managed node group label"]


def test_declaring_an_eks_spare_records_the_provider_and_cordons(
    tmp_path: Path,
) -> None:
    api = FakeCoreApi(_eks_node(NODE), _eks_node(FAULT))
    report = perform_warm_spare(
        api,
        WarmSpareRequest(
            state_dir=tmp_path,
            node=NODE,
            fault_node=FAULT,
            mode="declare",
            reference="CHG-1",
            confirmation="DECLARE_WARM_SPARE_CORDON",
        ),
        cluster_id="gpu-eks-cluster",
        record=tmp_path / "warm-spares" / f"{NODE}.json",
        agent_lookup=lambda *_: ACTIVE,
        actor="alice",
    )
    assert report["provider"] == "eks-managed"
    assert report["refusals"] == [] and report["ready"] is True
    assert report["topology"] == report["fault_topology"]
    assert api.nodes[NODE]["spec"]["unschedulable"] is True
    assert api.nodes[NODE]["metadata"]["labels"][SPARE_LABEL] == "true"
    assert (tmp_path / "warm-spares" / f"{NODE}.json").is_file(), (
        "the declaration record is written for an eks-managed node too"
    )


def test_run_warm_spare_passes_the_sites_provider_for_the_cluster(
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace

    api = FakeCoreApi(_raw_node(NODE))  # a HyperPod-labelled node ...
    site = SimpleNamespace(
        release_config={
            "clusters": [
                {
                    "cluster_id": "gpu-eks-cluster",
                    "context": "ctx",
                    "provider": "eks-managed",
                }
            ],
            "gpu_kubeconfig": "/state/gpu.kubeconfig",
        },
        environment={},
    )
    report = run_warm_spare(
        site,
        WarmSpareRequest(state_dir=tmp_path, node=NODE, mode="check"),
        api=api,
        agent_lookup=lambda *_: AgentState(lifecycle_state="ACTIVE"),
    )
    # ... judged by the site's word for the cluster, not by its labels.
    assert report["provider"] == "eks-managed"
    assert "HyperPod health label is not Schedulable" not in report["refusals"]
    assert report["refusals"] == [
        "eks-managed node carries no managed node group label"
    ]
    with pytest.raises(WarmSpareError, match="not in the managed site"):
        run_warm_spare(
            site,
            WarmSpareRequest(state_dir=tmp_path, node=NODE, cluster_id="other"),
            api=api,
            agent_lookup=lambda *_: ACTIVE,
        )
