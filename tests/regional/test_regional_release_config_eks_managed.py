"""``ClusterTarget`` and the registry entry for a plain EKS (``eks-managed``) cluster."""

from __future__ import annotations

from pathlib import Path

import pytest

from gpu_fault.regional import (
    RegionalClusterRegistration,
    regional_registry_content_sha256,
)
from gpu_fault.regional_registry import regional_registry_config_sha256
from gpu_fault_release import regional_release_registry as REGISTRY
from gpu_fault_release.regional_release_config import ClusterTarget, ReleaseError
from tests.regional._release_orchestrator_support import (
    GPU_EKS_ARN,
    REGION,
    config_file,
)

EKS_ARN = "arn:aws:eks:us-east-1:123456789012:cluster/gpu-eks-cluster"


def _hyperpod_target() -> dict:
    return {
        "cluster_id": "gpu-a",
        "context": "gpu-a-context",
        "executor_irsa_role_arn": "arn:aws:iam::1:role/a",
        "region": REGION,
        "hyperpod_cluster_name": "hp-gpu-a",
        "eks_cluster_arn": GPU_EKS_ARN,
        "agent_endpoint_allowed_cidrs": ["10.0.0.0/16"],
    }


def _eks_target(**overrides: object) -> dict:
    value = {
        "cluster_id": "gpu-eks-cluster",
        "context": "gpu-eks-cluster",
        "executor_irsa_role_arn": "arn:aws:iam::1:role/eks",
        "region": REGION,
        "provider": "eks-managed",
        "node_selector": "nvidia.com/gpu.present=true",
        "eks_cluster_arn": EKS_ARN,
        "agent_endpoint_allowed_cidrs": ["10.0.0.0/20", "10.0.16.0/20"],
    }
    value.update(overrides)
    return value


def test_hyperpod_target_keeps_its_defaults_and_selector() -> None:
    target = ClusterTarget.from_mapping(_hyperpod_target(), expected_region=REGION)
    assert target.provider == "hyperpod", "absent provider must read as hyperpod"
    assert target.node_selector == "", "no selector is stored for a HyperPod target"
    assert not target.is_eks_managed, "a HyperPod target is not eks-managed"
    assert (
        target.gpu_node_selector == "sagemaker.amazonaws.com/cluster-name=hp-gpu-a"
    ), "the HyperPod selector must be the historical label text"


def test_eks_managed_target_needs_no_hyperpod_name() -> None:
    target = ClusterTarget.from_mapping(_eks_target(), expected_region=REGION)
    assert target.is_eks_managed, "provider eks-managed must be recognised"
    assert target.hyperpod_cluster_name == "", "no HyperPod name is invented"
    assert target.gpu_node_selector == "nvidia.com/gpu.present=true", (
        "the configured selector is the GPU node selector"
    )


def test_eks_managed_target_requires_a_node_selector() -> None:
    with pytest.raises(ReleaseError, match="node_selector"):
        ClusterTarget.from_mapping(
            _eks_target(node_selector=""), expected_region=REGION
        )


def test_hyperpod_target_still_requires_its_hyperpod_name() -> None:
    value = _hyperpod_target()
    value.pop("hyperpod_cluster_name")
    with pytest.raises(ReleaseError, match="hyperpod_cluster_name"):
        ClusterTarget.from_mapping(value, expected_region=REGION)


def test_unknown_provider_fails_closed() -> None:
    with pytest.raises(ReleaseError, match="provider"):
        ClusterTarget.from_mapping(_eks_target(provider="eks"), expected_region=REGION)


def test_registry_entry_accepts_an_eks_managed_target(tmp_path: Path) -> None:
    token = tmp_path / "token"
    token.write_text("t" * 64, encoding="utf-8")
    target = ClusterTarget.from_mapping(
        _eks_target(token_file=str(token)), expected_region=REGION
    )
    entry = REGISTRY.registry_entry(target)
    assert entry["hyperpod_cluster_name"] == "", (
        "the registry entry carries an empty HyperPod name for eks-managed"
    )
    assert entry["cluster_id"] == "gpu-eks-cluster", "cluster id must be kept"
    assert entry["provider"] == "eks-managed", (
        "an eks-managed entry tells the control plane its provider"
    )
    hyperpod = _hyperpod_target()
    hyperpod["token_file"] = str(token)
    hyperpod_entry = REGISTRY.registry_entry(
        ClusterTarget.from_mapping(hyperpod, expected_region=REGION)
    )
    assert "provider" not in hyperpod_entry, (
        "a HyperPod entry keeps the key set it always had (additive only)"
    )


def test_registry_entry_still_requires_the_hyperpod_name_for_hyperpod(
    tmp_path: Path,
) -> None:
    token = tmp_path / "token"
    token.write_text("t" * 64, encoding="utf-8")
    value = _hyperpod_target()
    value["token_file"] = str(token)
    target = ClusterTarget.from_mapping(value, expected_region=REGION)
    stripped = ClusterTarget(**{**target.__dict__, "hyperpod_cluster_name": ""})
    with pytest.raises(ReleaseError, match="hyperpod_cluster_name"):
        REGISTRY.registry_entry(stripped)


# Digests computed at commit 0d151512, before ``hyperpod_cluster_name`` gained
# its default. A registry revision published by that release must digest to
# exactly these values with the current model, or every durable revision in
# the field would fail its load-time verification.
PRE_EXISTING_REGISTRATION = {
    "cluster_id": "gpu-prod-a",
    "region": "us-west-2",
    "hyperpod_cluster_name": "hp-gpu-prod-a",
    "eks_cluster_arn": "arn:aws:eks:us-west-2:123456789012:cluster/gpu-prod-a",
    "token_sha256": "a" * 64,
    "allowed_namespaces": ["training", "gpu-fault-system"],
    "agent_endpoint_allowed_cidrs": ["10.0.0.0/20"],
    "created_at": "2026-01-01T00:00:00+00:00",
    "updated_at": "2026-01-01T00:00:00+00:00",
}
PRE_EXISTING_CONFIG_SHA256 = (
    "5bdae79e00022dbc3b8e4cfc56afba78bf7890505aa9eefb0d83fe0c8fb54d9b"
)
PRE_EXISTING_CONTENT_SHA256 = (
    "fde99025208d4e428f0a3de39120fb6d62401b27d34ea3d913e03eee8e9574d9"
)


def test_pre_existing_hyperpod_registration_digests_identically() -> None:
    registration = RegionalClusterRegistration.model_validate(PRE_EXISTING_REGISTRATION)
    assert (
        regional_registry_config_sha256([registration]) == PRE_EXISTING_CONFIG_SHA256
    ), "the operator-config digest of a pre-existing registration changed"
    assert (
        regional_registry_content_sha256([registration]) == PRE_EXISTING_CONTENT_SHA256
    ), "the durable content digest of a pre-existing registration changed"


def test_registration_without_a_hyperpod_name_is_valid() -> None:
    value = dict(PRE_EXISTING_REGISTRATION)
    value.pop("hyperpod_cluster_name")
    value["provider"] = "eks-managed"
    registration = RegionalClusterRegistration.model_validate(value)
    assert registration.hyperpod_cluster_name == "", (
        "an eks-managed registration carries an empty HyperPod name"
    )
    assert registration.provider == "eks-managed", "the provider is a real field"
    assert (
        RegionalClusterRegistration.model_validate(PRE_EXISTING_REGISTRATION).provider
        == "hyperpod"
    ), "a registration without the key is HyperPod"
    with pytest.raises(ValueError, match="cluster provider must be one of"):
        RegionalClusterRegistration.model_validate({**value, "provider": "eks"})
    assert regional_registry_content_sha256([registration]) != (
        PRE_EXISTING_CONTENT_SHA256
    ), "an eks-managed registration digests differently from a HyperPod one"
    assert regional_registry_config_sha256([registration]) != (
        PRE_EXISTING_CONFIG_SHA256
    ), "dropping the HyperPod name is a configuration change and must digest so"


# --- per-cluster runtime profile -----------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
EKS_PROFILE = ROOT / "config/runtime-profile.regional-eks-safe.example.yaml"


def _load(tmp_path: Path, clusters: list[dict]):
    from gpu_fault_release import rollout as MODULE

    return MODULE.ReleaseConfig.load(config_file(tmp_path, clusters=clusters))


def test_clusters_inherit_the_site_profile_by_default(tmp_path: Path) -> None:
    config = _load(tmp_path, [_hyperpod_target()])
    (target,) = config.clusters
    assert target.runtime_profile_version == config.runtime_profile_version, (
        "a cluster without an override pins to the site version"
    )
    assert target.agent_config_digest == config.agent_config_digest, (
        "a cluster without an override carries the site Agent config digest"
    )
    assert target.runtime_profile_source == config.runtime_profile_source, "source"
    assert not target.overrides_runtime_profile, "no override is recorded"
    assert config.runtime_profiles() == (config.site_runtime_profile(),), (
        "a single-profile site registers exactly one profile"
    )
    assert config.cluster_runtime_profiles() == {}, "no overrides to record"
    assert config.cluster_runtime_profile_version(target) == "hyperpod-v1", "version"


def test_cluster_override_pins_its_own_profile_version(tmp_path: Path) -> None:
    override = _eks_target(
        runtime_profile={
            "source": str(EKS_PROFILE),
            "version": "eks-v1",
            "agent_config_digest": "e" * 64,
        }
    )
    config = _load(tmp_path, [_hyperpod_target(), override])
    hyperpod, eks = config.clusters
    assert eks.overrides_runtime_profile, "the override is recorded"
    assert config.cluster_runtime_profile_version(eks) == "eks-v1", "cluster version"
    assert config.cluster_agent_config_digest(eks) == "e" * 64, "cluster digest"
    assert config.cluster_runtime_profile_version(hyperpod) == "hyperpod-v1", (
        "the sibling keeps the site version"
    )
    versions = [spec.version for spec in config.runtime_profiles()]
    assert versions == ["hyperpod-v1", "eks-v1"], "both profiles are registered"
    spec = eks.runtime_profile_spec()
    assert spec.registration_cluster_id == "gpu-eks-cluster", (
        "an override is anchored on its own cluster"
    )
    assert spec.template_source == EKS_PROFILE, "template defaults to the source"
    overrides = config.cluster_runtime_profiles()
    assert set(overrides) == {"gpu-eks-cluster"}, "only overriding clusters appear"
    assert overrides["gpu-eks-cluster"]["version"] == "eks-v1", "version recorded"
    assert len(overrides["gpu-eks-cluster"]["policy_sha256"]) == 64, "policy digest"


def test_cluster_override_requires_an_existing_file_and_a_digest(
    tmp_path: Path,
) -> None:
    missing = _eks_target(
        runtime_profile={
            "source": str(tmp_path / "missing.yaml"),
            "version": "eks-v1",
            "agent_config_digest": "e" * 64,
        }
    )
    with pytest.raises(ReleaseError, match="existing file"):
        _load(tmp_path, [missing])
    without_digest = _eks_target(
        runtime_profile={"source": str(EKS_PROFILE), "version": "eks-v1"}
    )
    with pytest.raises(ReleaseError, match="agent_config_digest"):
        _load(tmp_path, [without_digest])
    bad_version = _eks_target(
        runtime_profile={
            "source": str(EKS_PROFILE),
            "version": "bad version",
            "agent_config_digest": "e" * 64,
        }
    )
    with pytest.raises(ReleaseError, match="runtime_profile.version"):
        _load(tmp_path, [bad_version])


def test_registry_entry_carries_the_pins_only_for_overriding_clusters(
    tmp_path: Path,
) -> None:
    token = tmp_path / "token"
    token.write_text("t" * 64, encoding="utf-8")
    override = _eks_target(
        token_file=str(token),
        runtime_profile={
            "source": str(EKS_PROFILE),
            "version": "eks-v1",
            "agent_config_digest": "e" * 64,
        },
    )
    hyperpod = _hyperpod_target()
    hyperpod["token_file"] = str(token)
    config = _load(tmp_path, [hyperpod, override])
    hyperpod_entry, eks_entry = (REGISTRY.registry_entry(t) for t in config.clusters)
    assert "runtime_profile_version" not in hyperpod_entry, (
        "an inheriting cluster keeps the registration shape it always had"
    )
    assert eks_entry["runtime_profile_version"] == "eks-v1", "the override pins"
    assert eks_entry["agent_config_digest"] == "e" * 64, "the override digest"
    single_root = tmp_path / "single"
    single_root.mkdir()
    alone = _load(single_root, [hyperpod]).clusters[0]
    assert REGISTRY.registry_config_digest((config.clusters[0],)) == (
        REGISTRY.registry_config_digest((alone,))
    ), "an inheriting cluster's identity digest does not depend on its siblings"


def test_registration_pins_digest_only_when_set() -> None:
    legacy = RegionalClusterRegistration.model_validate(PRE_EXISTING_REGISTRATION)
    assert (legacy.runtime_profile_version, legacy.agent_config_digest) == ("", ""), (
        "a registration without pins inherits the site defaults"
    )
    assert regional_registry_content_sha256([legacy]) == PRE_EXISTING_CONTENT_SHA256, (
        "the empty pins must not change a pre-existing registration's digest"
    )
    pinned = RegionalClusterRegistration.model_validate(
        {
            **PRE_EXISTING_REGISTRATION,
            "runtime_profile_version": "eks-v1",
            "agent_config_digest": "e" * 64,
        }
    )
    assert regional_registry_config_sha256([pinned]) != (
        regional_registry_config_sha256([legacy])
    ), "a pinned registration is a different configuration"
    with pytest.raises(ValueError, match="SHA-256"):
        RegionalClusterRegistration.model_validate(
            {**PRE_EXISTING_REGISTRATION, "agent_config_digest": "short"}
        )


def test_membership_digest_ignores_the_profile_pins_but_legacy_still_matches(
    tmp_path: Path,
) -> None:
    """Live 2026-10-04: the recorded `cluster_registry_digest` folded the pins
    in, so a plain EKS entry that lost (or refreshed) its pins read as a
    `clusters` change -- non-transactional, deploy refused. Membership is
    membership; the pins ride the rollbackable cluster_runtime_profiles."""

    token = tmp_path / "token"
    token.write_text("t" * 64, encoding="utf-8")
    hyperpod = _hyperpod_target()
    hyperpod["token_file"] = str(token)
    override = _eks_target(
        token_file=str(token),
        runtime_profile={
            "source": str(EKS_PROFILE),
            "version": "eks-v1",
            "agent_config_digest": "e" * 64,
        },
    )
    with_pins = _load(tmp_path, [hyperpod, override]).clusters
    bare_root = tmp_path / "bare"
    bare_root.mkdir()
    without_pins = _load(
        bare_root, [hyperpod, _eks_target(token_file=str(token))]
    ).clusters
    assert REGISTRY.registry_config_digest(with_pins) == (
        REGISTRY.registry_config_digest(without_pins)
    ), "the same two clusters are the same membership, pins or not"
    assert REGISTRY.legacy_registry_config_digest(with_pins) != (
        REGISTRY.registry_config_digest(with_pins)
    ), "the legacy digest is the pin-folded one"
    other_root = tmp_path / "other"
    other_root.mkdir()
    narrowed = _eks_target(
        token_file=str(token), allowed_namespaces=["gpu-fault-system"]
    )
    assert REGISTRY.registry_config_digest(
        _load(other_root, [hyperpod, narrowed]).clusters
    ) != REGISTRY.registry_config_digest(without_pins), (
        "a real membership attribute change still changes the digest"
    )
