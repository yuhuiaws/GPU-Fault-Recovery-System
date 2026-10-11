"""The provider-neutral GPU cluster scope shared by admin and runtime code."""

from __future__ import annotations

import pytest

from gpu_fault import cluster_scope
from gpu_fault.cluster_scope import (
    ClusterProvider,
    NodeSelectorSource,
    cluster_provider,
    default_failure_domain_labels,
    ec2_instance_id_from_provider_id,
    node_selector,
    node_selector_source,
    nodegroup_node_selector,
    ownership_node_selector,
    provider_cluster_name,
    selector_matches,
    selector_ownership_value,
    selector_requirements,
    validate_node_selector,
)
from gpu_fault.failure_domains import FAILURE_DOMAIN_LABELS

HYPERPOD_SELECTOR = "sagemaker.amazonaws.com/cluster-name=hp-gpu-a"


def test_contract_names_are_the_ones_the_runtime_agent_imports() -> None:
    """The env and Secret key spellings are a merge contract, not a style choice."""

    assert cluster_scope.PROVIDER_ENV == "GPU_FAULT_CLUSTER_PROVIDER", (
        "provider env name differs from the design brief"
    )
    assert cluster_scope.NODE_SELECTOR_ENV == "GPU_FAULT_NODE_SELECTOR", (
        "selector env name differs from the design brief"
    )
    assert cluster_scope.CONNECTION_SECRET_PROVIDER_KEY == "cluster-provider", (
        "provider Secret key differs from the design brief"
    )
    assert cluster_scope.CONNECTION_SECRET_NODE_SELECTOR_KEY == "node-selector", (
        "selector Secret key differs from the design brief"
    )
    assert [item.value for item in ClusterProvider] == ["hyperpod", "eks-managed"], (
        "provider enum values differ from the design brief"
    )
    assert cluster_scope.OWNERSHIP_LABEL == "gpu-fault.io/control-plane", (
        "the ownership label key is the convention operators label node groups with"
    )
    assert cluster_scope.CONNECTION_SECRET_NODE_SELECTOR_SOURCE_KEY == (
        "node-selector-source"
    ), "the selector-source Secret key differs from the design brief"
    assert [item.value for item in NodeSelectorSource] == [
        "label-convention",
        "provider-default-legacy",
        "flag-legacy",
    ], "selector source values differ from the design brief"
    assert not hasattr(cluster_scope, "DEFAULT_EKS_MANAGED_NODE_SELECTOR"), (
        "nvidia.com/gpu.present=true is no discovery input any more"
    )


def test_absent_or_empty_provider_means_hyperpod_and_unknown_fails_closed() -> None:
    assert cluster_provider(None) is ClusterProvider.HYPERPOD, (
        "absent provider must read as hyperpod"
    )
    assert cluster_provider("") is ClusterProvider.HYPERPOD, (
        "empty provider must read as hyperpod"
    )
    assert cluster_provider(" eks-managed ") is ClusterProvider.EKS_MANAGED, (
        "whitespace around a provider name must be tolerated"
    )
    with pytest.raises(ValueError, match="cluster provider must be one of"):
        cluster_provider("eks")


def test_hyperpod_selector_is_byte_identical_to_the_historical_label() -> None:
    """Every HyperPod site keeps selecting on the SageMaker label text."""

    identity = {"provider": "hyperpod", "hyperpod_name": "hp-gpu-a"}
    target = {"hyperpod_cluster_name": "hp-gpu-a"}
    assert node_selector(identity) == HYPERPOD_SELECTOR, (
        "identity-shaped hyperpod target rendered a different selector"
    )
    assert node_selector(target) == HYPERPOD_SELECTOR, (
        "release-config-shaped hyperpod target rendered a different selector"
    )
    assert (
        node_selector(None, provider="hyperpod", hyperpod_cluster_name="hp-gpu-a")
        == HYPERPOD_SELECTOR
    ), "keyword-only hyperpod call rendered a different selector"


def test_hyperpod_without_a_name_fails_closed() -> None:
    with pytest.raises(ValueError, match="needs its HyperPod cluster name"):
        node_selector({"provider": "hyperpod", "hyperpod_cluster_name": ""})


def test_eks_managed_requires_an_explicit_selector() -> None:
    """A selector that matched every node would hand the CPU node group over."""

    with pytest.raises(ValueError, match="has no recorded node selector"):
        node_selector({"provider": "eks-managed", "cluster_id": "gpu-eks"})
    assert (
        node_selector(
            {"provider": "eks-managed", "node_selector": "nvidia.com/gpu.present=true"}
        )
        == "nvidia.com/gpu.present=true"
    ), "configured eks-managed selector was not returned verbatim"
    assert (
        node_selector({"provider": "eks-managed", "nodeSelector": "node-type=gpu"})
        == "node-type=gpu"
    ), "site-document spelling nodeSelector was not read"


def test_an_explicit_selector_overrides_the_hyperpod_label() -> None:
    assert (
        node_selector(
            {"provider": "hyperpod", "hyperpod_name": "hp", "node_selector": "a=b"}
        )
        == "a=b"
    ), "an explicit selector must win over the derived HyperPod label"


def test_nodegroup_selector_uses_the_eks_nodegroup_label() -> None:
    assert (
        nodegroup_node_selector("gpu-p5en-efa-spot-node-group")
        == "eks.amazonaws.com/nodegroup=gpu-p5en-efa-spot-node-group"
    ), "nodegroup selector label or value differs"
    with pytest.raises(ValueError, match="invalid EKS node group name"):
        nodegroup_node_selector("bad name")


@pytest.mark.parametrize(
    "selector",
    [
        "nvidia.com/gpu.present=true",
        "a==b",
        "a!=b",
        "nvidia.com/gpu.present",
        "!node-type",
        "node-type in (gpu, accelerated)",
        "node-type notin (cpu)",
        "nvidia.com/gpu.present=true,eks.amazonaws.com/nodegroup=gpu",
    ],
)
def test_valid_selectors_are_accepted(selector: str) -> None:
    assert validate_node_selector(f"  {selector} ") == selector, (
        "a valid selector must be returned stripped and otherwise unchanged"
    )


@pytest.mark.parametrize(
    "selector",
    ["", "   ", "a=b,", ",a=b", "bad key=1", "a=has space", "a in ()", "a in (b"],
)
def test_invalid_selectors_are_rejected(selector: str) -> None:
    with pytest.raises(ValueError):
        validate_node_selector(selector)


def test_selector_matches_evaluates_every_requirement_form() -> None:
    gpu = {
        "nvidia.com/gpu.present": "true",
        "node-type": "gpu",
        "eks.amazonaws.com/nodegroup": "gpu-p5en-efa-spot-node-group",
        "topology.kubernetes.io/zone": "us-west-2c",
    }
    cpu = {"node-type": "cpu", "eks.amazonaws.com/nodegroup": "cpu-node-group-20260101"}
    assert selector_matches(gpu, "nvidia.com/gpu.present=true"), (
        "GPU node must match the default eks-managed selector"
    )
    assert not selector_matches(cpu, "nvidia.com/gpu.present=true"), (
        "the m7i-flex CPU node must never match the GPU selector"
    )
    assert selector_matches(gpu, "node-type in (gpu,accelerated)"), "set in failed"
    assert not selector_matches(gpu, "node-type notin (gpu)"), "set notin failed"
    assert selector_matches(cpu, "!nvidia.com/gpu.present"), "absence failed"
    assert selector_matches(gpu, "node-type!=cpu"), "inequality failed"
    assert not selector_matches(cpu, "node-type!=cpu"), "inequality must exclude"
    assert selector_matches(
        gpu,
        "nvidia.com/gpu.present=true,eks.amazonaws.com/nodegroup=gpu-p5en-efa-spot-node-group",
    ), "conjunction failed"
    assert not selector_matches(
        gpu, "nvidia.com/gpu.present=true,eks.amazonaws.com/nodegroup=other"
    ), "a conjunction must require every term"
    assert selector_matches(
        {"sagemaker.amazonaws.com/cluster-name": "hp-gpu-a"}, HYPERPOD_SELECTOR
    ), "HyperPod label match failed"
    assert not selector_matches(None, HYPERPOD_SELECTOR), (
        "a node without labels must not match"
    )


def test_provider_cluster_name_keeps_hyperpod_name_and_uses_cluster_id_for_eks() -> (
    None
):
    assert (
        provider_cluster_name({"hyperpod_cluster_name": "hp-gpu-a", "cluster_id": "a"})
        == "hp-gpu-a"
    ), "HyperPod clusters keep their HyperPod name as the provider scope"
    assert (
        provider_cluster_name(
            {
                "provider": "eks-managed",
                "hyperpod_cluster_name": "",
                "cluster_id": "gpu-eks",
            }
        )
        == "gpu-eks"
    ), "eks-managed clusters carry their cluster_id as the provider scope"


def test_ec2_instance_id_is_parsed_from_the_aws_provider_id() -> None:
    assert (
        ec2_instance_id_from_provider_id("aws:///us-west-2c/i-00000000000000001")
        == "i-00000000000000001"
    ), "the EC2 instance id was not extracted from spec.providerID"
    assert ec2_instance_id_from_provider_id("") == "", "empty id must give empty"
    assert ec2_instance_id_from_provider_id("gce://x/y/z") == "", (
        "a non-AWS provider id must give empty, not a guess"
    )
    assert ec2_instance_id_from_provider_id("aws:///zone/not-an-instance") == "", (
        "a malformed AWS provider id must give empty"
    )


def test_failure_domain_defaults_follow_the_providers() -> None:
    assert default_failure_domain_labels([]) == FAILURE_DOMAIN_LABELS, (
        "a site without clusters keeps the HyperPod default labels"
    )
    assert (
        default_failure_domain_labels(["hyperpod", ClusterProvider.HYPERPOD])
        == FAILURE_DOMAIN_LABELS
    ), "a HyperPod-only site keeps the HyperPod default labels unchanged"
    mixed = default_failure_domain_labels(["hyperpod", "eks-managed"])
    assert mixed[:2] == (
        "sagemaker.amazonaws.com/instance-group-name",
        "eks.amazonaws.com/nodegroup",
    ), "the EKS node group label must follow the HyperPod instance group label"
    assert "topology.kubernetes.io/zone" in mixed, "zone must stay a fallback"
    assert len(mixed) == len(set(mixed)), "label priority must not repeat a label"
    assert default_failure_domain_labels(["eks-managed"]) == mixed, (
        "an eks-managed-only site uses the same merged priority"
    )


# --- the ownership label convention ---------------------------------------------

SITE_ID = "us-west-2-control-plane-gpu-fault-solut-985fd370"


def test_ownership_selector_names_the_site_and_cross_checks_hyperpod() -> None:
    eks = ownership_node_selector(SITE_ID, provider="eks-managed")
    assert eks == f"gpu-fault.io/control-plane={SITE_ID}", (
        "a plain EKS cluster selects on the ownership label alone"
    )
    hyperpod = ownership_node_selector(
        SITE_ID, provider="hyperpod", hyperpod_cluster_name="hp-gpu-a"
    )
    assert hyperpod == (
        f"gpu-fault.io/control-plane={SITE_ID},sagemaker.amazonaws.com/cluster-name=hp-gpu-a"
    ), "a HyperPod cluster adds its cluster-name label as the cross-check"
    assert validate_node_selector(hyperpod) == hyperpod, (
        "the ownership selector must be a selector every reader accepts"
    )
    with pytest.raises(ValueError, match="needs its HyperPod cluster name"):
        ownership_node_selector(SITE_ID, provider="hyperpod")
    with pytest.raises(ValueError, match="invalid ownership label value"):
        ownership_node_selector("not a label value", provider="eks-managed")


def test_ownership_selector_matches_only_this_sites_nodes() -> None:
    selector = ownership_node_selector(
        SITE_ID, provider="hyperpod", hyperpod_cluster_name="hp-gpu-a"
    )
    owned = {
        "gpu-fault.io/control-plane": SITE_ID,
        "sagemaker.amazonaws.com/cluster-name": "hp-gpu-a",
        "node.kubernetes.io/instance-type": "ml.p5en.48xlarge",
    }
    assert selector_matches(owned, selector), (
        "a labelled node of the cluster is selected"
    )
    assert not selector_matches(
        {**owned, "gpu-fault.io/control-plane": "other"}, selector
    ), "a node labelled for another site is never selected"
    assert not selector_matches(
        {**owned, "sagemaker.amazonaws.com/cluster-name": "hp-gpu-b"}, selector
    ), "a label copied onto another HyperPod cluster's node selects nothing"
    unlabelled = dict(owned)
    del unlabelled["gpu-fault.io/control-plane"]
    assert not selector_matches(unlabelled, selector), (
        "an unlabelled node of the cluster (a non-GPU instance group) is left out"
    )


def test_selector_ownership_value_reads_the_site_id_from_a_selector() -> None:
    assert selector_ownership_value(f"gpu-fault.io/control-plane={SITE_ID},a=b") == (
        SITE_ID
    ), "the ownership requirement's value is the site id"
    assert selector_ownership_value(HYPERPOD_SELECTOR) is None, (
        "a selector without the ownership label names no site"
    )
    assert selector_ownership_value("gpu-fault.io/control-plane in (a,b)") is None, (
        "a set requirement with two values names no single site"
    )


def test_selector_requirements_is_the_one_public_parser() -> None:
    assert selector_requirements("a=b, c in (d,e),!f,g") == (
        ("a", "in", frozenset({"b"})),
        ("c", "in", frozenset({"d", "e"})),
        ("f", "absent", frozenset()),
        ("g", "exists", frozenset()),
    ), "requirements must be split on the commas between terms, not inside sets"


def test_node_selector_source_is_recorded_or_derived() -> None:
    assert node_selector_source({"node_selector_source": "flag-legacy"}) is (
        NodeSelectorSource.FLAG_LEGACY
    ), "a recorded source wins"
    assert node_selector_source({"nodeSelectorSource": "label-convention"}) is (
        NodeSelectorSource.LABEL_CONVENTION
    ), "the site-document spelling is read too"
    assert node_selector_source({"provider": "hyperpod"}) is (
        NodeSelectorSource.PROVIDER_DEFAULT_LEGACY
    ), "a HyperPod entry without a selector is the legacy cluster-name default"
    assert (
        node_selector_source(
            {"provider": "eks-managed", "node_selector": "nvidia.com/gpu.present=true"}
        )
        is NodeSelectorSource.FLAG_LEGACY
    ), "a plain EKS selector recorded before the source field came from the flags"
    assert (
        node_selector_source(
            {
                "provider": "hyperpod",
                "node_selector": f"gpu-fault.io/control-plane={SITE_ID}",
            }
        )
        is NodeSelectorSource.LABEL_CONVENTION
    ), "a selector naming the ownership label is the convention"
    with pytest.raises(ValueError, match="node selector source must be one of"):
        node_selector_source({"node_selector_source": "guessed"})
