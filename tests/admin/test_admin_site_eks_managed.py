"""``spec.clusters[].provider`` / ``nodeSelector``: plain EKS clusters in site.yaml."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from gpu_fault.admin.site import SiteConfigError, load_site
from gpu_fault.cluster_scope import node_selector
from gpu_fault.cluster_scope import EKS_MANAGED_FAILURE_DOMAIN_LABELS
from gpu_fault.failure_domains import FAILURE_DOMAIN_LABELS
from tests.admin.test_admin_site import site_file

EKS_ARN = "arn:aws:eks:us-east-1:123456789012:cluster/gpu-eks-cluster"


def _document(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _write(path: Path, document: dict) -> None:
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")


def _eks_cluster(document: dict, **overrides: object) -> dict:
    base = dict(document["spec"]["clusters"][0])
    base.pop("hyperpodClusterName", None)
    base.update(
        {
            "clusterId": "gpu-eks-cluster",
            "context": "gpu-eks-cluster",
            "provider": "eks-managed",
            "nodeSelector": "nvidia.com/gpu.present=true",
            "eksClusterArn": EKS_ARN,
            **overrides,
        }
    )
    return base


def test_hyperpod_site_renders_exactly_as_before(tmp_path: Path) -> None:
    """No ``provider``/``node_selector`` keys leak into a HyperPod release config."""

    site = load_site(site_file(tmp_path))
    (entry,) = site.release_config["clusters"]
    assert "provider" not in entry, "hyperpod entries must not gain a provider key"
    assert "node_selector" not in entry, "hyperpod entries must not gain a selector"
    assert entry["hyperpod_cluster_name"] == "hp-gpu-a", "HyperPod name must stay"
    assert site.release_config["failure_domain_labels"] == list(
        FAILURE_DOMAIN_LABELS
    ), "a HyperPod-only site keeps the historical failure-domain labels"
    assert node_selector(entry) == "sagemaker.amazonaws.com/cluster-name=hp-gpu-a", (
        "the HyperPod selector must be the historical label text"
    )


def test_eks_managed_cluster_is_accepted_without_a_hyperpod_name(
    tmp_path: Path,
) -> None:
    path = site_file(tmp_path)
    document = _document(path)
    document["spec"]["clusters"].append(_eks_cluster(document))
    _write(path, document)

    site = load_site(path)

    hyperpod, eks = site.release_config["clusters"]
    assert eks["provider"] == "eks-managed", "provider must be rendered for EKS"
    assert eks["node_selector"] == "nvidia.com/gpu.present=true", (
        "the node selector must be rendered for EKS"
    )
    assert eks["hyperpod_cluster_name"] == "", (
        "an eks-managed entry carries an empty HyperPod name, never a guess"
    )
    assert "provider" not in hyperpod, "the HyperPod sibling must render unchanged"
    assert node_selector(eks) == "nvidia.com/gpu.present=true", (
        "the eks-managed selector is the configured one"
    )
    labels = site.release_config["failure_domain_labels"]
    assert labels[:2] == [
        "sagemaker.amazonaws.com/instance-group-name",
        EKS_MANAGED_FAILURE_DOMAIN_LABELS[0],
    ], "a mixed site adds the EKS node group label after the instance group"


def test_eks_managed_cluster_requires_a_node_selector(tmp_path: Path) -> None:
    path = site_file(tmp_path)
    document = _document(path)
    cluster = _eks_cluster(document)
    cluster.pop("nodeSelector")
    document["spec"]["clusters"].append(cluster)
    _write(path, document)

    with pytest.raises(SiteConfigError, match=r"nodeSelector.*explicit node selector"):
        load_site(path)


def test_invalid_node_selector_is_refused(tmp_path: Path) -> None:
    path = site_file(tmp_path)
    document = _document(path)
    document["spec"]["clusters"].append(
        _eks_cluster(document, nodeSelector="nvidia.com/gpu.present=true,")
    )
    _write(path, document)

    with pytest.raises(SiteConfigError, match="nodeSelector"):
        load_site(path)


def test_unknown_provider_is_refused(tmp_path: Path) -> None:
    path = site_file(tmp_path)
    document = _document(path)
    document["spec"]["clusters"].append(_eks_cluster(document, provider="eks"))
    _write(path, document)

    with pytest.raises(SiteConfigError, match="provider"):
        load_site(path)


def test_hyperpod_cluster_still_requires_its_hyperpod_name(tmp_path: Path) -> None:
    path = site_file(tmp_path)
    document = _document(path)
    document["spec"]["clusters"][0].pop("hyperpodClusterName")
    _write(path, document)

    with pytest.raises(SiteConfigError, match="hyperpodClusterName"):
        load_site(path)


def test_configured_failure_domain_labels_still_win(tmp_path: Path) -> None:
    path = site_file(tmp_path)
    document = _document(path)
    document["spec"]["clusters"].append(_eks_cluster(document))
    document["spec"]["failureDomainLabels"] = ["topology.kubernetes.io/zone"]
    _write(path, document)

    site = load_site(path)

    assert site.release_config["failure_domain_labels"] == [
        "topology.kubernetes.io/zone"
    ], "spec.failureDomainLabels must override the provider default"
