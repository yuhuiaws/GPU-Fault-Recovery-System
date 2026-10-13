"""A 2 HyperPod + 2 plain-EKS site through its Runtime Profile lifecycle.

The site runs ``hyperpod-v1``; both plain EKS clusters run the one reviewed
``eks-v1``. Joined one after another (a batch for the first, a single join
for the second), released, upgraded by a ``deploy`` that lists the clusters
by EKS ARN, then moved together to the successor version when the EKS
template changes policy: one review per version, never an automatic
approval, and the release engine accepting its own unchanged policy whichever
member the cluster order makes its anchor.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from gpu_fault.admin import cluster_batch_join as batch
from gpu_fault.admin import cluster_join as join
from gpu_fault.admin import cluster_join_profile, profile_approval
from gpu_fault.admin.cluster_join_types import JoinClusterRequest
from gpu_fault.admin.site import load_site, materialized_release_config
from gpu_fault_release import rollout as RELEASE
from gpu_fault_release.regional_release_profile_model import (
    runtime_profile_policy_sha256,
    runtime_profiles,
)
from gpu_fault_release.regional_runtime_profile import (
    ensure_runtime_profile,
    verify_runtime_profile,
)
from tests.admin.test_admin_eks_managed_join import EKS_PROFILE, _approve_pending_plan
from tests.admin.test_multi_cluster_profile_review import MixedJoinScenario, eks
from tests.regional.test_release_profile_anchor import StoreRunner

#: ``--cluster-id`` of the plain EKS cluster ``gpu-f``: listed first by cluster
#: id (the order every join commit writes), last by EKS ARN (the order a
#: deploy writes), so the two orders anchor ``eks-v1`` on different members.
FIRST_EKS = "eks-alpha"


def _release(
    scenario: MixedJoinScenario, store: StoreRunner
) -> RELEASE.RegionalRelease:
    site = load_site(scenario.path)
    with materialized_release_config(site) as path:
        config = RELEASE.ReleaseConfig.load(path)
    return RELEASE.RegionalRelease(config, store)


def _deploy_finalize(scenario: MixedJoinScenario) -> dict[str, Any]:
    """What a whole-site ``deploy`` writes: the clusters in EKS ARN order (the
    order ``deploy`` requests them in), each cluster profile finalized."""

    document = yaml.safe_load(scenario.path.read_text(encoding="utf-8"))
    document["spec"]["clusters"] = sorted(
        document["spec"]["clusters"], key=lambda item: str(item["eksClusterArn"])
    )
    cluster_join_profile.reconcile_cluster_runtime_profiles(
        scenario,
        document,
        repository_root=scenario.site.repository_root,
        state_dir=scenario.state_dir,
    )
    scenario.path.write_text(yaml.safe_dump(document, sort_keys=False))
    return dict(document)


def _pins(document: dict[str, Any]) -> dict[str, str | None]:
    return {
        str(item["clusterId"]): (item.get("runtimeProfile") or {}).get("version")
        for item in document["spec"]["clusters"]
    }


def _anchors(release: RELEASE.RegionalRelease) -> dict[str, str]:
    return {
        spec.version: spec.registration_cluster_id
        for spec in runtime_profiles(release.config)
    }


def _change_the_eks_template(scenario: MixedJoinScenario) -> str:
    """The EKS example profile gives ``nodeReplace`` back to OBSERVE; the
    successor version is named after the new policy."""

    template = scenario.site.repository_root / "config" / EKS_PROFILE.name
    document = yaml.safe_load(template.read_text(encoding="utf-8"))
    for claim in document["claims"]:
        if claim.get("capability") == "nodeReplace":
            claim["mode"] = "OBSERVE"
            claim.pop("adapter", None)
    template.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return f"regional-eks-{runtime_profile_policy_sha256(template)[:12]}"


@pytest.fixture
def scenario(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MixedJoinScenario:
    return MixedJoinScenario(tmp_path, monkeypatch)


def test_a_two_plus_two_site_keeps_one_reviewed_eks_profile_through_its_lifecycle(
    scenario: MixedJoinScenario,
) -> None:
    store = StoreRunner()
    state_dir = scenario.state_dir

    # 1. The second HyperPod cluster follows the site profile: nothing to review.
    assert scenario.join("b")["phase"] == "COMPLETED"
    assert not profile_approval.profile_plan_path(state_dir).exists(), "no review"

    # 2. The first plain EKS cluster joins through a batch (what
    #    `deploy --gpu-cluster-arn` runs): it stops for review like a join.
    request = JoinClusterRequest(
        site=load_site(scenario.path),
        gpu_cluster_arn=eks("f").eks_arn,
        cluster_id=FIRST_EKS,
    )
    with pytest.raises(cluster_join_profile.JoinProfileApprovalRequired):
        batch.join_clusters((request,), runner_factory=lambda: scenario)
    plan = scenario.plan()
    assert (plan["registration_cluster_id"], plan["desired_version"]) == (
        FIRST_EKS,
        "eks-v1",
    ), plan
    _approve_pending_plan(state_dir, plan["plan_sha256"])
    request = JoinClusterRequest(
        site=load_site(scenario.path),
        gpu_cluster_arn=eks("f").eks_arn,
        cluster_id=FIRST_EKS,
    )
    joined = batch.join_clusters((request,), runner_factory=lambda: scenario)
    assert joined["joined"] == [FIRST_EKS], joined
    ensure_runtime_profile(_release(scenario, store), anchor=FIRST_EKS)
    assert [(item["profile_version"], item["cluster_id"]) for item in store.posted] == [
        ("hyperpod-v1", "gpu-a"),
        ("eks-v1", FIRST_EKS),
    ], "the join registers eks-v1 under the cluster it joins"

    # 3. The second plain EKS cluster reuses the approved eks-v1, unreviewed.
    second = join.join_cluster(scenario.eks_request("e"), runner=scenario)
    assert second["phase"] == "COMPLETED", second
    assert not profile_approval.profile_plan_path(state_dir).exists(), (
        "an approved version is reused without a second review"
    )
    release = _release(scenario, store)
    ensure_runtime_profile(release, anchor="gpu-e")
    verify_runtime_profile(release)
    assert _anchors(release)["eks-v1"] == FIRST_EKS, "commit order: by cluster id"
    assert len(store.posted) == 2, "eks-v1 is registered once"

    # 4. A whole-site deploy lists the clusters by EKS ARN; the anchor moves,
    #    the policy does not, and the release accepts what it registered.
    upgraded = _deploy_finalize(scenario)
    assert _pins(upgraded) == {
        "gpu-a": None,
        "hp-gpu-b": None,
        "gpu-e": "eks-v1",
        FIRST_EKS: "eks-v1",
    }, upgraded["spec"]["clusters"]
    release = _release(scenario, store)
    assert _anchors(release)["eks-v1"] == "gpu-e", "deploy order: by EKS ARN"
    ensure_runtime_profile(release)
    verify_runtime_profile(release)
    assert len(store.posted) == 2, "nothing is re-registered after the re-sort"
    assert store.store["eks-v1"]["cluster_id"] == FIRST_EKS, "stored as registered"

    # 5. The EKS template changes policy: one plan moves both clusters.
    successor = _change_the_eks_template(scenario)
    with pytest.raises(cluster_join_profile.JoinProfileApprovalRequired) as stop:
        cluster_join_profile.preflight_cluster_profile_plans(
            state_dir, repository_root=scenario.site.repository_root
        )
    plan = scenario.plan()
    assert stop.value.cluster_ids == (FIRST_EKS, "gpu-e"), stop.value.cluster_ids
    assert (plan["current_version"], plan["desired_version"]) == (
        "eks-v1",
        successor,
    ), plan
    _approve_pending_plan(state_dir, plan["plan_sha256"])
    cluster_join_profile.preflight_cluster_profile_plans(
        state_dir, repository_root=scenario.site.repository_root
    )
    moved = _deploy_finalize(scenario)
    assert _pins(moved) == {
        "gpu-a": None,
        "hp-gpu-b": None,
        "gpu-e": successor,
        FIRST_EKS: successor,
    }, moved["spec"]["clusters"]
    release = _release(scenario, store)
    ensure_runtime_profile(release)
    verify_runtime_profile(release)
    assert [item["profile_version"] for item in store.posted] == [
        "hyperpod-v1",
        "eks-v1",
        successor,
    ], "the successor is registered once for both clusters"
    consumed = sorted(
        json.loads(path.read_text(encoding="utf-8"))["release_id"]
        for path in (state_dir / "release-deploy" / "profile-approvals").glob(
            "*/consumed.json"
        )
    )
    assert consumed == [f"deploy-{FIRST_EKS}", f"join-{FIRST_EKS}"], consumed
    assert (state_dir / "profiles" / "eks-v1.yaml").is_file(), "eks-v1 is immutable"
