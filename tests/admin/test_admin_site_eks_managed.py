"""``spec.clusters[].provider`` / ``nodeSelector``: plain EKS clusters in site.yaml."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from gpu_fault.admin.site import SiteConfigError, load_site
from gpu_fault.cluster_scope import (
    EKS_MANAGED_FAILURE_DOMAIN_LABELS,
    NodeSelectorSource,
    node_selector,
    node_selector_source,
)
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

    with pytest.raises(
        SiteConfigError, match=r"nodeSelector.*no recorded node selector"
    ):
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


# --- spec.clusters[].runtimeProfile -------------------------------------------------


def _eks_profile_override(root: Path) -> dict:
    return {
        "source": "config/runtime-profile.regional-eks-safe.example.yaml",
        "templateSource": "config/runtime-profile.regional-eks-safe.example.yaml",
        "version": "eks-v1",
        "agentConfigDigest": "d" * 64,
    }


def test_cluster_without_runtime_profile_renders_no_profile_keys(
    tmp_path: Path,
) -> None:
    site = load_site(site_file(tmp_path))
    (entry,) = site.release_config["clusters"]
    assert "runtime_profile" not in entry, (
        "a cluster on the site profile must render exactly as before"
    )


def test_cluster_runtime_profile_override_is_rendered_with_resolved_paths(
    tmp_path: Path,
) -> None:
    path = site_file(tmp_path)
    document = _document(path)
    root = Path(document["spec"]["repositoryRoot"])
    cluster = _eks_cluster(document)
    cluster["runtimeProfile"] = _eks_profile_override(root)
    document["spec"]["clusters"].append(cluster)
    _write(path, document)

    site = load_site(path)

    hyperpod, eks = site.release_config["clusters"]
    assert "runtime_profile" not in hyperpod, "the inheriting sibling is unchanged"
    override = eks["runtime_profile"]
    assert override["version"] == "eks-v1", "the override version is rendered"
    assert override["agent_config_digest"] == "d" * 64, "the digest is rendered"
    assert override["source"] == str(
        (root / "config/runtime-profile.regional-eks-safe.example.yaml").resolve()
    ), "the override source is resolved against the repository root"
    assert override["template_source"] == override["source"], "template resolved"


def test_cluster_runtime_profile_override_without_digest_renders_without_it(
    tmp_path: Path,
) -> None:
    path = site_file(tmp_path)
    document = _document(path)
    root = Path(document["spec"]["repositoryRoot"])
    cluster = _eks_cluster(document)
    override = _eks_profile_override(root)
    override.pop("agentConfigDigest")
    override.pop("templateSource")
    cluster["runtimeProfile"] = override
    document["spec"]["clusters"].append(cluster)
    _write(path, document)

    site = load_site(path)
    rendered = site.release_config["clusters"][1]["runtime_profile"]
    assert "agent_config_digest" not in rendered, "no digest is invented"
    assert rendered["template_source"] == rendered["source"], (
        "templateSource defaults to source as for the site profile"
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("version", "bad version!", "unsupported characters"),
        ("agentConfigDigest", "nope", "SHA-256"),
        ("source", "REPLACE_WITH_PROFILE", "placeholder"),
        ("extra", "x", "unknown fields"),
    ],
)
def test_invalid_cluster_runtime_profile_is_refused(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    path = site_file(tmp_path)
    document = _document(path)
    root = Path(document["spec"]["repositoryRoot"])
    cluster = _eks_cluster(document)
    override = _eks_profile_override(root)
    override[field] = value
    cluster["runtimeProfile"] = override
    document["spec"]["clusters"].append(cluster)
    _write(path, document)

    with pytest.raises(SiteConfigError, match=message):
        load_site(path)


# --- nodeSelectorSource: where a recorded selector came from --------------------

SITE_ID = "us-east-1-control-985fd370"


def test_node_selector_source_is_rendered_and_derived(tmp_path: Path) -> None:
    path = site_file(tmp_path)
    document = _document(path)
    labelled = _eks_cluster(
        document,
        nodeSelector=f"gpu-fault.io/control-plane={SITE_ID}",
        nodeSelectorSource="label-convention",
    )
    document["spec"]["clusters"].append(labelled)
    _write(path, document)

    site = load_site(path)

    hyperpod, eks = site.release_config["clusters"]
    assert eks["node_selector_source"] == "label-convention", (
        "a recorded source is rendered into the release config for audit"
    )
    assert "node_selector_source" not in hyperpod, (
        "a legacy HyperPod entry renders byte for byte as before"
    )
    assert (
        node_selector_source(hyperpod) is NodeSelectorSource.PROVIDER_DEFAULT_LEGACY
    ), "a legacy HyperPod entry derives the legacy default source on read"
    assert node_selector_source(eks) is NodeSelectorSource.LABEL_CONVENTION


def test_legacy_eks_entry_without_a_source_derives_flag_legacy(tmp_path: Path) -> None:
    path = site_file(tmp_path)
    document = _document(path)
    document["spec"]["clusters"].append(
        _eks_cluster(document, nodeSelector="eks.amazonaws.com/nodegroup=gpu-ng")
    )
    _write(path, document)

    _hyperpod, eks = load_site(path).release_config["clusters"]

    assert "node_selector_source" not in eks, "nothing is invented into the entry"
    assert node_selector_source(eks) is NodeSelectorSource.FLAG_LEGACY, (
        "a selector recorded by the retired flags reads as flag-legacy"
    )


def test_unknown_node_selector_source_is_refused(tmp_path: Path) -> None:
    path = site_file(tmp_path)
    document = _document(path)
    document["spec"]["clusters"].append(
        _eks_cluster(document, nodeSelectorSource="guessed")
    )
    _write(path, document)

    with pytest.raises(SiteConfigError, match=r"nodeSelectorSource.*must be one of"):
        load_site(path)
