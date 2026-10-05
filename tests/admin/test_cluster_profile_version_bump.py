"""A cluster-owned Runtime Profile moves to a new version when its template
policy changes; the pinned snapshot stays immutable.

Live 2026-10-05: ``gpu-eks-cluster`` was pinned to ``eks-v1`` by its join;
the plain-EKS example profile then took ownership of ``nodeReplace``. The
deploy ran the release bundle and its gates and only then refused in the
site finalization ("approved Runtime Profile snapshot ... has a different
policy ... choose a new version or remove the stale snapshot") -- no plan, no
``plan_sha256``, nothing for ``--approve-profile-plan`` to approve, and the
runbook's promised ``regional-eks-<digest12>`` version did not exist.

Now the deploy computes the successor version from the template policy
digest, writes the ``CLUSTER_PROFILE_NEW`` plan anchored on the cluster and
stops -- in ``run_deploy`` before any release work -- and the approved rerun
writes ``profiles/regional-eks-<digest12>.yaml``, re-pins the cluster entry
and continues. ``eks-v1`` is never rewritten; a cluster joining after the
change derives the same successor.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from gpu_fault.admin import cluster_join_profile, profile_approval
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.site import load_site
from gpu_fault_release.regional_release_profile_model import (
    runtime_profile_policy_sha256,
)
from tests.admin.test_admin_eks_managed_join import (
    EKS_PROFILE,
    HYPERPOD_PROFILE,
    JoinScenario,
    _approve_pending_plan,
    _approved_snapshot,
    _deploy_document,
    _install_example_profiles,
    _use_site_profile,
    eks_target,
)

CLUSTER = "gpu-eks-cluster"
EKS_ARN = f"arn:aws:eks:us-west-2:123456789012:cluster/{CLUSTER}"
PINNED = {
    "source": "/secure/gpu-fault/profiles/eks-v1.yaml",
    "templateSource": "config/runtime-profile.regional-eks-safe.example.yaml",
    "version": "eks-v1",
    "agentConfigDigest": "1" * 64,
}


def _site_identity(document: dict) -> dict:
    document["metadata"] = {"name": "staging"}
    document["spec"]["awsRegion"] = "us-west-2"
    document["spec"]["cpu"] = {
        "eksArn": "arn:aws:eks:us-west-2:123456789012:cluster/control-plane"
    }
    document["spec"]["runtimeProfile"]["version"] = "hyperpod-v1"
    return document


def _older_policy(snapshot: Path) -> None:
    """Rewrite the pinned snapshot with the policy the join approved: the
    template before it took ownership of nodeReplace."""

    document = yaml.safe_load(snapshot.read_text(encoding="utf-8"))
    for claim in document["claims"]:
        if claim.get("capability") == "nodeReplace":
            claim["mode"] = "OBSERVE"
            claim.pop("adapter", None)
    snapshot.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")


def _digest_version(repo: Path) -> str:
    digest = runtime_profile_policy_sha256(repo / "config" / EKS_PROFILE.name)
    return f"regional-eks-{digest[:12]}"


def _fake_digest(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cluster_join_profile,
        "compute_agent_config_digest",
        lambda _runner, *, repository_root, runtime_profile_version: (
            runtime_profile_version[:1] * 64
        ),
    )


def _bumped_site(tmp_path: Path) -> tuple[Path, dict, Path, Path]:
    repo, document = _deploy_document(tmp_path, eks_profile=dict(PINNED))
    _site_identity(document)
    state_dir = tmp_path / "state"
    snapshot = _approved_snapshot(state_dir, repo)
    _older_policy(snapshot)
    return repo, document, state_dir, snapshot


def test_the_provider_version_is_the_declared_one_until_its_snapshot_disagrees(
    tmp_path: Path,
) -> None:
    repo, _document, state_dir, snapshot = _bumped_site(tmp_path)
    template = repo / "config" / EKS_PROFILE.name
    policy = runtime_profile_policy_sha256(template)

    assert (
        cluster_join_profile.provider_profile_version(
            "eks-managed", policy, state_dir=tmp_path / "fresh"
        )
        == "eks-v1"
    ), "no snapshot yet: the declared version"
    assert (
        cluster_join_profile.provider_profile_version(
            "eks-managed", runtime_profile_policy_sha256(snapshot), state_dir=state_dir
        )
        == "eks-v1"
    ), "the snapshot carries this policy: still the declared version"
    successor = cluster_join_profile.provider_profile_version(
        "eks-managed", policy, state_dir=state_dir
    )
    assert successor == f"regional-eks-{policy[:12]}", (
        "another policy under the declared name: the digest names the version"
    )
    assert (
        cluster_join_profile.provider_profile_version(
            "hyperpod", policy, state_dir=state_dir
        )
        == "hyperpod-v1"
    )


def test_deploy_plans_the_successor_version_and_stops_without_touching_the_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, document, state_dir, snapshot = _bumped_site(tmp_path)
    before = snapshot.read_text(encoding="utf-8")
    _fake_digest(monkeypatch)

    with pytest.raises(cluster_join_profile.JoinProfileApprovalRequired) as stop:
        cluster_join_profile.reconcile_cluster_runtime_profiles(
            object(), document, repository_root=repo, state_dir=state_dir
        )

    plan = json.loads(profile_approval.profile_plan_path(state_dir).read_text())
    successor = _digest_version(repo)
    assert plan["change_kind"] == "CLUSTER_PROFILE_NEW"
    assert plan["registration_cluster_id"] == CLUSTER
    assert (plan["current_version"], plan["desired_version"]) == ("eks-v1", successor)
    assert plan["current_policy_digest"] == runtime_profile_policy_sha256(snapshot)
    assert plan["policy_digest"] == runtime_profile_policy_sha256(
        repo / "config" / EKS_PROFILE.name
    )
    assert plan["site_identity"] == {
        "site_name": "staging",
        "aws_region": "us-west-2",
        "cpu_eks_arn": "arn:aws:eks:us-west-2:123456789012:cluster/control-plane",
    }
    assert plan["approval_required"] is True
    assert (
        "moves from its own Runtime Profile eks-v1 to " + successor
        in (plan["changes"][0])
    )
    message = str(stop.value)
    assert "the deploy stopped for review" in message
    assert f"version: eks-v1 -> {successor} (this cluster only)" in message
    assert plan["plan_sha256"] in message
    assert (
        f"gpu-fault-admin deploy --state-dir {state_dir} --approve-profile-plan "
        f"{plan['plan_sha256']} --reference CHG-<id>"
    ) in message
    assert snapshot.read_text(encoding="utf-8") == before, "eks-v1 stays immutable"
    assert not (state_dir / "profiles" / f"{successor}.yaml").exists(), (
        "no snapshot before the approval"
    )
    assert document["spec"]["clusters"][1]["runtimeProfile"] == PINNED, (
        "the entry is not rewritten by a stopped deploy"
    )

    with pytest.raises(cluster_join_profile.JoinProfileApprovalRequired):
        cluster_join_profile.reconcile_cluster_runtime_profiles(
            object(), document, repository_root=repo, state_dir=state_dir
        )
    assert (
        json.loads(profile_approval.profile_plan_path(state_dir).read_text())[
            "plan_sha256"
        ]
        == plan["plan_sha256"]
    ), "a rerun without approval keeps the same plan"


def test_the_approved_rerun_writes_the_successor_snapshot_and_repins_the_cluster(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, document, state_dir, snapshot = _bumped_site(tmp_path)
    before = snapshot.read_text(encoding="utf-8")
    _fake_digest(monkeypatch)
    with pytest.raises(cluster_join_profile.JoinProfileApprovalRequired):
        cluster_join_profile.reconcile_cluster_runtime_profiles(
            object(), document, repository_root=repo, state_dir=state_dir
        )
    plan = json.loads(profile_approval.profile_plan_path(state_dir).read_text())
    _approve_pending_plan(state_dir, plan["plan_sha256"])

    cluster_join_profile.reconcile_cluster_runtime_profiles(
        object(), document, repository_root=repo, state_dir=state_dir
    )

    successor = _digest_version(repo)
    new_snapshot = state_dir / "profiles" / f"{successor}.yaml"
    assert new_snapshot.is_file(), (
        "the approved policy is snapshotted under its version"
    )
    written = yaml.safe_load(new_snapshot.read_text(encoding="utf-8"))
    assert written["profile_version"] == successor and written["cluster_id"] == CLUSTER
    assert runtime_profile_policy_sha256(new_snapshot) == plan["policy_digest"]
    assert snapshot.read_text(encoding="utf-8") == before, "eks-v1 is untouched"
    hyperpod, eks = document["spec"]["clusters"]
    assert "runtimeProfile" not in hyperpod
    assert eks["runtimeProfile"] == {
        "source": str(new_snapshot),
        "templateSource": "config/runtime-profile.regional-eks-safe.example.yaml",
        "version": successor,
        "agentConfigDigest": "r" * 64,
    }, "the entry pins the successor and this release's Agent config digest"
    assert not profile_approval.profile_plan_path(state_dir).exists(), "consumed"
    consumed = list(
        (state_dir / "release-deploy" / "profile-approvals").glob("*/consumed.json")
    )
    assert len(consumed) == 1
    assert json.loads(consumed[0].read_text())["release_id"] == f"deploy-{CLUSTER}"

    # The next deploy finds the successor approved and asks nothing.
    again = {
        **document,
        "spec": {**document["spec"], "clusters": [dict(hyperpod), dict(eks)]},
    }
    cluster_join_profile.reconcile_cluster_runtime_profiles(
        object(), again, repository_root=repo, state_dir=state_dir
    )
    assert again["spec"]["clusters"][1]["runtimeProfile"]["version"] == successor
    assert not profile_approval.profile_plan_path(state_dir).exists(), (
        "an approved successor asks for no second review"
    )


def test_the_deploy_preflight_stops_early_and_leaves_an_approval_to_the_finalization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, document, state_dir, _snapshot = _bumped_site(tmp_path)
    (state_dir / "site.yaml").write_text(
        yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
    )
    _fake_digest(monkeypatch)

    with pytest.raises(cluster_join_profile.JoinProfileApprovalRequired) as stop:
        cluster_join_profile.preflight_cluster_profile_plans(
            state_dir, repository_root=repo
        )
    plan = json.loads(profile_approval.profile_plan_path(state_dir).read_text())
    assert plan["desired_version"] == _digest_version(repo)
    assert plan["plan_sha256"] in str(stop.value)
    assert not (state_dir / "profiles" / f"{plan['desired_version']}.yaml").exists(), (
        "the preflight writes no snapshot"
    )

    _approve_pending_plan(state_dir, plan["plan_sha256"])
    cluster_join_profile.preflight_cluster_profile_plans(
        state_dir, repository_root=repo
    )
    assert profile_approval.profile_approval_path(state_dir).is_file(), (
        "the preflight consumes nothing; the finalization writes and consumes"
    )
    assert not (state_dir / "profiles" / f"{plan['desired_version']}.yaml").exists(), (
        "an approved plan still leaves the snapshot to the finalization"
    )

    # No site, a site without clusters, or an unchanged pin: silent.
    cluster_join_profile.preflight_cluster_profile_plans(
        tmp_path / "nowhere", repository_root=repo
    )
    (tmp_path / "bare").mkdir()
    (tmp_path / "bare" / "site.yaml").write_text("name: staging\n", encoding="utf-8")
    cluster_join_profile.preflight_cluster_profile_plans(
        tmp_path / "bare", repository_root=repo
    )


def test_a_successor_snapshot_that_does_not_carry_its_policy_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, document, state_dir, snapshot = _bumped_site(tmp_path)
    _fake_digest(monkeypatch)
    successor = state_dir / "profiles" / f"{_digest_version(repo)}.yaml"
    successor.write_text(snapshot.read_text(encoding="utf-8"), encoding="utf-8")

    with pytest.raises(BootstrapError, match="does not carry the policy its version"):
        cluster_join_profile.reconcile_cluster_runtime_profiles(
            object(), document, repository_root=repo, state_dir=state_dir
        )


def test_a_plan_needs_the_site_identity(tmp_path: Path) -> None:
    repo, document, state_dir, _snapshot = _bumped_site(tmp_path)
    del document["spec"]["cpu"]

    with pytest.raises(BootstrapError, match="lacks the identity"):
        cluster_join_profile.reconcile_cluster_runtime_profiles(
            object(), document, repository_root=repo, state_dir=state_dir
        )


def test_a_cluster_joining_after_the_change_derives_the_successor_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenario = JoinScenario(tmp_path, monkeypatch, suffixes=("b",))
    _install_example_profiles(scenario)
    _use_site_profile(scenario, HYPERPOD_PROFILE.name, "hyperpod-v1")
    site = load_site(scenario.path)
    state_dir = scenario.path.parent
    repo = Path(yaml.safe_load(scenario.path.read_text())["spec"]["repositoryRoot"])
    pinned = _approved_snapshot(state_dir, repo)
    _older_policy(pinned)

    override = cluster_join_profile.cluster_profile_override(
        site, replace(eks_target(), eks_name="gpu-f"), "gpu-f"
    )

    assert override is not None
    successor = _digest_version(repo)
    assert override.version == successor, "eks-v1 names another policy already"
    assert override.snapshot_file == state_dir / "profiles" / f"{successor}.yaml"
    with pytest.raises(cluster_join_profile.JoinProfileApprovalRequired) as stop:
        cluster_join_profile.ensure_cluster_profile_approved(
            site, override, rerun_command="gpu-fault-admin join-cluster ..."
        )
    plan = json.loads(profile_approval.profile_plan_path(state_dir).read_text())
    assert (plan["current_version"], plan["desired_version"]) == (
        "hyperpod-v1",
        successor,
    ), "a first own version is reviewed against the site profile"
    assert "the join stopped for review" in str(stop.value)
    _approve_pending_plan(state_dir, plan["plan_sha256"])
    written = cluster_join_profile.ensure_cluster_profile_approved(
        site, override, rerun_command="x"
    )
    assert written == override.snapshot_file and written.is_file()
    assert (
        yaml.safe_load(pinned.read_text())["claims"]
        != yaml.safe_load(written.read_text())["claims"]
    ), "two versions, two policies, both snapshots kept"
