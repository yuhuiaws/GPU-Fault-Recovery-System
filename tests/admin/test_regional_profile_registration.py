"""The site Runtime Profile base follows the registration cluster's provider on a
managed site; only a fresh site derives it from the cluster list.

Live 2026-10-08, eks-first site (registration gpu-eks-cluster, HyperPod joined
with the hyperpod-v1 override): the upgrade re-derived a HyperPod-family base
because a HyperPod cluster was present, and the next deploy then demanded an
eks override under that base."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from gpu_fault.admin.bootstrap_common import BootstrapError, ClusterIdentity
from gpu_fault.admin.bootstrap_discovery import (
    EKS_MANAGED_PROFILE,
    HYPERPOD_PROFILE,
    regional_profile_for,
    registration_cluster_provider,
)

ACCOUNT = "123456789012"


def _cluster(name: str, provider: str) -> ClusterIdentity:
    arn = f"arn:aws:eks:us-west-2:{ACCOUNT}:cluster/{name}"
    hyperpod = provider == "hyperpod"
    return ClusterIdentity(
        input_arn=arn,
        role="gpu",
        region="us-west-2",
        account_id=ACCOUNT,
        hyperpod_arn=f"arn:aws:sagemaker:us-west-2:{ACCOUNT}:cluster/{name}"
        if hyperpod
        else "",
        hyperpod_name=f"hp-{name}" if hyperpod else "",
        eks_arn=arn,
        eks_name=name,
        vpc_id="vpc-a",
        subnet_ids=("subnet-a",),
        node_recovery="None" if hyperpod else "",
        context=f"gpu-fault-gpu-{name}",
        provider=provider,
    )


def _site(
    tmp_path: Path, *, registration: str, clusters: list[tuple[str, str]]
) -> Path:
    document = {
        "spec": {
            "runtimeProfile": {"version": "x", "registrationClusterId": registration},
            "clusters": [
                {
                    "clusterId": name,
                    "eksClusterArn": f"arn:aws:eks:us-west-2:{ACCOUNT}:cluster/{name}",
                    **({"provider": provider} if provider == "eks-managed" else {}),
                }
                for name, provider in clusters
            ],
        }
    }
    (tmp_path / "site.yaml").write_text(yaml.safe_dump(document), encoding="utf-8")
    return tmp_path


EKS = _cluster("gpu-eks-cluster", "eks-managed")
HP = _cluster("hp-cluster", "hyperpod")


def test_an_eks_first_site_keeps_the_eks_base_after_a_hyperpod_join(
    tmp_path: Path,
) -> None:
    state = _site(
        tmp_path,
        registration="gpu-eks-cluster",
        clusters=[("gpu-eks-cluster", "eks-managed"), ("hp-cluster", "hyperpod")],
    )

    assert registration_cluster_provider(state, [HP, EKS]) == "eks-managed"
    assert regional_profile_for(tmp_path, [HP, EKS], state_dir=state) == (
        str(tmp_path / EKS_MANAGED_PROFILE[0]),
        "eks-v1",
    ), "the registration cluster's family wins over the cluster list, in any order"


def test_a_hyperpod_first_site_keeps_the_hyperpod_base_after_an_eks_join(
    tmp_path: Path,
) -> None:
    state = _site(
        tmp_path,
        registration="hp-cluster",
        clusters=[("hp-cluster", "hyperpod"), ("gpu-eks-cluster", "eks-managed")],
    )

    assert regional_profile_for(tmp_path, [EKS, HP], state_dir=state) == (
        str(tmp_path / HYPERPOD_PROFILE[0]),
        "hyperpod-v1",
    ), "the eks member is an override, never a re-base"


def test_a_fresh_site_derives_the_base_from_its_single_provider(tmp_path: Path) -> None:
    assert regional_profile_for(tmp_path, [EKS], state_dir=tmp_path)[1] == "eks-v1"
    assert regional_profile_for(tmp_path, [HP], state_dir=tmp_path)[1] == "hyperpod-v1"
    assert regional_profile_for(tmp_path, [HP, EKS])[1] == "hyperpod-v1", (
        "without a site the old rule holds (a mixed first deploy is refused earlier)"
    )


def test_a_registration_cluster_the_deploy_no_longer_manages_is_refused(
    tmp_path: Path,
) -> None:
    state = _site(
        tmp_path,
        registration="gpu-eks-cluster",
        clusters=[("gpu-eks-cluster", "eks-managed"), ("hp-cluster", "hyperpod")],
    )
    with pytest.raises(BootstrapError, match="registration cluster gpu-eks-cluster"):
        regional_profile_for(tmp_path, [HP], state_dir=state)
    (tmp_path / "empty").mkdir()
    empty = _site(tmp_path / "empty", registration="hp-cluster", clusters=[])
    assert registration_cluster_provider(empty, []) is None, (
        "a site without GPU clusters keeps its own profile anchor"
    )
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken/site.yaml").write_text("spec: [", encoding="utf-8")
    with pytest.raises(BootstrapError, match="cannot read"):
        registration_cluster_provider(tmp_path / "broken", [EKS])


def test_override_rows_follow_the_registration_family_not_the_live_base(
    tmp_path: Path,
) -> None:
    """Live 2026-10-08: an eks-first site under a HyperPod-family base refused
    the deploy for a missing eks override before the base plan back to eks-v1
    could be offered. Overrides are judged against the base this deploy renders
    (the registration cluster's family); the registration cluster is never one."""
    from gpu_fault.admin.cluster_join_profile import _cluster_entries

    def document(registration: str) -> dict:
        return {
            "spec": {
                "runtimeProfile": {
                    "source": "config/runtime-profile.regional-hyperpod-safe.example.yaml",
                    "version": "regional-hyperpod-08d0bbcfa39d",
                    "registrationClusterId": registration,
                },
                "clusters": [
                    {"clusterId": "gpu-eks-cluster", "provider": "eks-managed"},
                    {
                        "clusterId": "hp-cluster",
                        "runtimeProfile": {"version": "hyperpod-v1"},
                    },
                ],
            }
        }

    eks_first = _cluster_entries(
        document("gpu-eks-cluster"),
        repository_root=tmp_path,
        state_dir=tmp_path,
        strict=False,
    )
    assert [(e["clusterId"], v) for e, _p, v in eks_first] == [
        ("hp-cluster", "hyperpod-v1")
    ], (
        "the eks registration cluster needs no override; the HyperPod member keeps its pin"
    )
    hyperpod_first = _cluster_entries(
        document("hp-cluster"),
        repository_root=tmp_path,
        state_dir=tmp_path,
        strict=False,
    )
    assert [(e["clusterId"], p) for e, p, _v in hyperpod_first] == [
        ("gpu-eks-cluster", "eks-managed"),
        ("hp-cluster", "hyperpod"),
    ], "on a HyperPod-first site the eks member still needs its eks-family override"
