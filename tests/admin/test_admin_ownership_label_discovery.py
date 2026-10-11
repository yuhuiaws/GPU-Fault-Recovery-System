"""The ownership label convention: node group facts, resolution and legacy scope.

``bootstrap_ownership`` is the one place both providers resolve "which nodes
does this site manage" from. The discovery integration (``discover_cluster``
on EKS and HyperPod answers) lives in ``test_admin_eks_managed_join``; this
file pins the pure parts: how a provider document becomes facts, how a
recorded legacy selector maps back onto node groups, and the exact wording
of the three refusals.
"""

from __future__ import annotations

import pytest

from gpu_fault.admin import bootstrap_ownership as ownership
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.cluster_scope import OWNERSHIP_LABEL, NodeSelectorSource

SITE_ID = "us-west-2-control-plane-gpu-fault-solut-985fd370"


def _eks(name: str, instance_type: str, labels: dict | None = None) -> dict:
    return {
        "nodegroupName": name,
        "instanceTypes": [instance_type],
        "labels": labels or {},
        "status": "ACTIVE",
    }


def test_eks_nodegroup_facts_carry_the_stamped_nodegroup_label() -> None:
    facts = ownership.eks_nodegroup_facts(
        _eks("gpu-ng", "p5en.48xlarge", {OWNERSHIP_LABEL: SITE_ID, "node-type": "gpu"})
    )
    assert facts.name == "gpu-ng"
    assert facts.instance_types == ("p5en.48xlarge",)
    assert facts.ownership == SITE_ID
    assert facts.labels["eks.amazonaws.com/nodegroup"] == "gpu-ng", (
        "EKS stamps the node group name on its nodes; a recorded selector may name it"
    )
    assert facts.hosts_gpu_types and facts.unsupported_types == ()
    launch_template = ownership.eks_nodegroup_facts(
        {"nodegroupName": "lt-ng", "instanceType": "m7i-flex.xlarge"}
    )
    assert launch_template.instance_types == ("m7i-flex.xlarge",)
    assert launch_template.unsupported_types == ("m7i-flex.xlarge",)
    assert launch_template.ownership is None
    assert launch_template.describe() == (
        f"lt-ng instanceTypes=m7i-flex.xlarge {OWNERSHIP_LABEL}=absent"
    )


def test_hyperpod_instance_group_facts_read_current_labels_only() -> None:
    applied = ownership.hyperpod_instance_group_facts(
        {
            "InstanceGroupName": "spot-p5en",
            "InstanceType": "ml.p5en.48xlarge",
            "KubernetesConfig": {
                "CurrentLabels": {OWNERSHIP_LABEL: SITE_ID},
                "DesiredLabels": {OWNERSHIP_LABEL: SITE_ID},
            },
        },
        cluster_name="hp",
    )
    assert applied.ownership == SITE_ID and applied.pending_ownership is None
    assert applied.labels["sagemaker.amazonaws.com/cluster-name"] == "hp"
    assert applied.labels["sagemaker.amazonaws.com/instance-group-name"] == "spot-p5en"
    assert applied.labels["node.kubernetes.io/instance-type"] == "ml.p5en.48xlarge"
    assert applied.hosts_gpu_types, "the ml. prefix is ignored against the inventory"
    pending = ownership.hyperpod_instance_group_facts(
        {
            "InstanceGroupName": "spot-p5en",
            "InstanceType": "ml.p5en.48xlarge",
            "KubernetesConfig": {
                "CurrentLabels": {},
                "DesiredLabels": {OWNERSHIP_LABEL: SITE_ID},
            },
        },
        cluster_name="hp",
    )
    assert pending.ownership is None, "a desired label is not applied yet"
    assert pending.pending_ownership == SITE_ID
    assert "(desired " in pending.describe() and "not applied yet" in pending.describe()
    create_spelling = ownership.hyperpod_instance_group_facts(
        {
            "InstanceGroupName": "g",
            "InstanceType": "ml.p5.48xlarge",
            "KubernetesConfig": {"Labels": {OWNERSHIP_LABEL: SITE_ID}},
        },
        cluster_name="hp",
    )
    assert create_spelling.ownership == SITE_ID, (
        "the create-time Labels spelling is accepted when CurrentLabels is absent"
    )
    bare = ownership.hyperpod_instance_group_facts(
        {"InstanceGroupName": "g6", "InstanceType": "ml.g6.12xlarge"}, cluster_name="hp"
    )
    assert bare.ownership is None and not bare.hosts_gpu_types


def test_resolution_selects_the_labelled_gpu_groups_for_both_providers() -> None:
    facts = [
        ownership.eks_nodegroup_facts(_eks("cpu-ng", "m7i-flex.xlarge")),
        ownership.eks_nodegroup_facts(
            _eks("gpu-ng", "p5en.48xlarge", {OWNERSHIP_LABEL: SITE_ID})
        ),
        ownership.eks_nodegroup_facts(
            _eks("gpu-ng-2", "p5.48xlarge", {OWNERSHIP_LABEL: SITE_ID})
        ),
    ]
    scope = ownership.resolve_node_scope(
        facts, provider="eks-managed", cluster="gpu-eks", site_id=SITE_ID
    )
    assert scope.selector == f"{OWNERSHIP_LABEL}={SITE_ID}"
    assert scope.source is NodeSelectorSource.LABEL_CONVENTION
    assert scope.owned_groups == ("gpu-ng", "gpu-ng-2"), "every labelled group is owned"
    assert scope.as_dict()["node_groups"][0].startswith("cpu-ng instanceTypes="), (
        "the evidence lists every group seen, owned or not"
    )
    hyperpod = ownership.resolve_node_scope(
        [
            ownership.hyperpod_instance_group_facts(
                {
                    "InstanceGroupName": "spot-p5en",
                    "InstanceType": "ml.p5en.48xlarge",
                    "KubernetesConfig": {"CurrentLabels": {OWNERSHIP_LABEL: SITE_ID}},
                },
                cluster_name="hp",
            )
        ],
        provider="hyperpod",
        cluster="hp",
        site_id=SITE_ID,
        hyperpod_cluster_name="hp",
    )
    assert hyperpod.selector == (
        f"{OWNERSHIP_LABEL}={SITE_ID},sagemaker.amazonaws.com/cluster-name=hp"
    )


def test_the_three_refusals_name_the_facts() -> None:
    foreign = [
        ownership.eks_nodegroup_facts(
            _eks("gpu-ng", "p5en.48xlarge", {OWNERSHIP_LABEL: "another-site"})
        )
    ]
    with pytest.raises(BootstrapError, match="name another site") as refusal:
        ownership.resolve_node_scope(
            foreign, provider="eks-managed", cluster="gpu-eks", site_id=SITE_ID
        )
    assert "gpu-ng carries gpu-fault.io/control-plane=another-site" in str(
        refusal.value
    )
    assert "One GPU cluster belongs to one site" in str(refusal.value)

    unlabelled = [ownership.eks_nodegroup_facts(_eks("gpu-ng", "p5en.48xlarge"))]
    with pytest.raises(BootstrapError, match="has no managed node group labelled"):
        ownership.resolve_node_scope(
            unlabelled, provider="eks-managed", cluster="gpu-eks", site_id=SITE_ID
        )

    non_gpu = [
        ownership.eks_nodegroup_facts(
            _eks("cpu-ng", "m7i-flex.xlarge", {OWNERSHIP_LABEL: SITE_ID})
        )
    ]
    with pytest.raises(
        BootstrapError, match="outside the GPU instance inventory"
    ) as exc:
        ownership.resolve_node_scope(
            non_gpu, provider="eks-managed", cluster="gpu-eks", site_id=SITE_ID
        )
    assert "cpu-ng runs m7i-flex.xlarge" in str(exc.value)
    assert "remove it from these groups" in str(exc.value)


def test_label_remedy_is_the_provider_command() -> None:
    assert ownership.label_remedy(
        "eks-managed", cluster="gpu-eks", site_id=SITE_ID, group="gpu-ng"
    ) == (
        "aws eks update-nodegroup-config --cluster-name gpu-eks --nodegroup-name "
        f"gpu-ng --labels addOrUpdateLabels={{{OWNERSHIP_LABEL}={SITE_ID}}}"
    )
    hyperpod = ownership.label_remedy(
        "hyperpod", cluster="hp", site_id=SITE_ID, group="spot-p5en"
    )
    assert hyperpod.startswith("aws sagemaker update-cluster --cluster-name hp "), (
        hyperpod
    )
    assert "KubernetesConfig" in hyperpod and SITE_ID in hyperpod


def test_groups_for_a_recorded_selector_follow_its_meaning() -> None:
    facts = [
        ownership.eks_nodegroup_facts(_eks("cpu-ng", "m7i-flex.xlarge")),
        ownership.eks_nodegroup_facts(
            _eks("gpu-ng", "p5en.48xlarge", {OWNERSHIP_LABEL: SITE_ID})
        ),
    ]
    names = lambda items: [item.name for item in items]  # noqa: E731
    assert names(
        ownership.groups_for_selector(facts, "eks.amazonaws.com/nodegroup=gpu-ng")
    ) == ["gpu-ng"], "a node group selector names its group"
    assert names(
        ownership.groups_for_selector(facts, f"{OWNERSHIP_LABEL}={SITE_ID}")
    ) == ["gpu-ng"], "the ownership selector covers the labelled groups"
    assert names(
        ownership.groups_for_selector(facts, "nvidia.com/gpu.present=true")
    ) == ["gpu-ng"], (
        "the retired default names a runtime label: fall back to the GPU inventory"
    )
    assert (
        ownership.groups_for_selector(facts, "eks.amazonaws.com/nodegroup=gone") == []
    )
    assert ownership.groups_for_selector(facts, f"{OWNERSHIP_LABEL}=other-site") == []


def test_recorded_scope_reads_both_entry_spellings_and_derives_the_source() -> None:
    site_entries = [
        {
            "clusterId": "gpu-eks",
            "provider": "eks-managed",
            "eksClusterArn": "arn:aws:eks:r:1:cluster/gpu-eks",
            "nodeSelector": "eks.amazonaws.com/nodegroup=gpu-ng",
        },
        {
            "clusterId": "hp",
            "hyperpodClusterName": "hp",
            "eksClusterArn": "arn:aws:eks:r:1:cluster/hp-eks",
        },
    ]
    eks = ownership.recorded_node_scope(
        site_entries, eks_arn="arn:aws:eks:r:1:cluster/gpu-eks"
    )
    assert eks == ownership.RecordedNodeScope(
        "eks.amazonaws.com/nodegroup=gpu-ng", "flag-legacy"
    )
    hyperpod = ownership.recorded_node_scope(
        site_entries, eks_arn="arn:aws:eks:r:1:cluster/other", hyperpod_name="hp"
    )
    assert hyperpod == ownership.RecordedNodeScope("", "provider-default-legacy"), (
        "a HyperPod entry is found by name when the ARN given was the SageMaker one"
    )
    release_entries = [
        {
            "cluster_id": "gpu-eks",
            "provider": "eks-managed",
            "eks_cluster_arn": "arn:aws:eks:r:1:cluster/gpu-eks",
            "node_selector": f"{OWNERSHIP_LABEL}={SITE_ID}",
            "node_selector_source": "label-convention",
        }
    ]
    assert ownership.recorded_node_scope(
        release_entries, eks_arn="arn:aws:eks:r:1:cluster/gpu-eks"
    ) == ownership.RecordedNodeScope(f"{OWNERSHIP_LABEL}={SITE_ID}", "label-convention")
    assert ownership.recorded_node_scope(release_entries, eks_arn="arn:other") is None
