"""Runtime Profile review on sites with several clusters per provider.

A cluster whose provider the site profile cannot reboot runs its own profile
version, reviewed once (``CLUSTER_PROFILE_NEW``) and kept as the immutable
snapshot ``profiles/<version>.yaml``. Two gaps of the review showed on a
2 HyperPod + 2 plain-EKS site (the release engine's comparison is covered by
``tests/regional/test_release_profile_anchor.py``):

* the batch join (every ``deploy --gpu-cluster-arn``, ``join-cluster`` with
  several ARNs) recorded the discovery without the cluster's profile, so a
  plain EKS cluster joined a HyperPod site with no plan, no review and no
  override, and ran the other provider's profile;
* the deploy preflight planned per cluster and left the approval for the
  finalization, so a second cluster needing the same successor built its own
  plan and found the first one's approval "stale" on every rerun.

A version is now one plan naming every cluster of the operation that needs
it, anchored on the first by cluster id, whichever order lists them; a batch
plans it exactly as a single join does and stops before it mutates anything;
an approval stays explicit and covers exactly the plan it was given for.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml

from gpu_fault.admin import cluster_batch_join as batch
from gpu_fault.admin import cluster_join_profile, profile_approval
from gpu_fault.admin.bootstrap_common import ClusterIdentity
from gpu_fault.admin.cluster_join_types import JoinClusterRequest
from gpu_fault.admin.site import load_site
from gpu_fault_release.regional_release_profile_model import (
    runtime_profile_policy_sha256,
)
from tests.admin._cov95_join_support import ACCOUNT, REGION, JoinScenario
from tests.admin.test_admin_eks_managed_join import (
    EKS_PROFILE,
    GPU_NODE,
    HYPERPOD_PROFILE,
    _approve_pending_plan,
    _eks_node,
    _install_example_profiles,
    _use_site_profile,
    eks_target,
)

EKS_NODES = {"e": GPU_NODE, "f": "ip-10-0-40-7.us-west-2.compute.internal"}


def agent_config_digest(version: str) -> str:
    """The fake Agent config digest of a profile version (a valid SHA-256)."""

    return hashlib.sha256(version.encode()).hexdigest()


def eks(name: str) -> ClusterIdentity:
    """A plain EKS cluster ``gpu-<name>`` as discovery returns it."""

    arn = f"arn:aws:eks:{REGION}:{ACCOUNT}:cluster/gpu-{name}"
    return replace(
        eks_target(),
        input_arn=arn,
        eks_arn=arn,
        eks_name=f"gpu-{name}",
        vpc_id=f"vpc-gpu-{name}",
        context=f"gpu-{name}",
    )


class MixedJoinScenario(JoinScenario):
    """The join world of a HyperPod site (``gpu-a``, the HyperPod ``gpu-b`` to
    join) with plain EKS clusters ``gpu-e``/``gpu-f`` that can join it."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        super().__init__(tmp_path, monkeypatch, suffixes=("b",))
        _install_example_profiles(self)
        _use_site_profile(self, HYPERPOD_PROFILE.name, "hyperpod-v1")
        self.site = load_site(self.path)
        for name, node in EKS_NODES.items():
            self.targets[eks(name).eks_arn] = eks(name)
            self.node_names[name] = node
        monkeypatch.setattr(
            cluster_join_profile,
            "compute_agent_config_digest",
            lambda _runner, *, repository_root, runtime_profile_version: (
                agent_config_digest(runtime_profile_version)
            ),
        )

    @property
    def state_dir(self) -> Path:
        return self.path.parent

    def run(self, arguments: list[str], **options: Any) -> Any:
        context = (
            arguments[arguments.index("--context") + 1]
            if "--context" in arguments
            else "cpu"
        )
        if (
            "get" in arguments
            and "nodes" in arguments
            and context in {"gpu-e", "gpu-f"}
        ):
            self.commands.append((list(arguments), dict(options)))
            name = self.node_names[context.rsplit("-", 1)[-1]]
            if any(str(item).startswith("go-template=") for item in arguments):
                return json.dumps([name])
            return json.dumps({"items": [_eks_node(name)]})
        return super().run(arguments, **options)

    def eks_request(self, name: str) -> JoinClusterRequest:
        return JoinClusterRequest(
            site=load_site(self.path), gpu_cluster_arn=eks(name).eks_arn
        )

    def join_batch(self, *names: str) -> dict[str, Any]:
        requests = tuple(
            self.request("b") if name == "b" else self.eks_request(name)
            for name in names
        )
        return batch.join_clusters(requests, runner_factory=lambda: self)

    def plan(self) -> dict[str, Any]:
        path = profile_approval.profile_plan_path(self.state_dir)
        return dict(json.loads(path.read_text(encoding="utf-8")))

    def entry(self, cluster_id: str) -> dict[str, Any]:
        document = yaml.safe_load(self.path.read_text(encoding="utf-8"))
        return dict(
            next(
                item
                for item in document["spec"]["clusters"]
                if item["clusterId"] == cluster_id
            )
        )

    def journal(self, name: str) -> dict[str, Any]:
        arn = self.request("b").gpu_cluster_arn if name == "b" else eks(name).eks_arn
        identity = hashlib.sha256(arn.encode()).hexdigest()[:12]
        path = self.state_dir / "join-cluster" / identity / "state.json"
        return dict(json.loads(path.read_text(encoding="utf-8")))


@pytest.fixture
def scenario(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MixedJoinScenario:
    return MixedJoinScenario(tmp_path, monkeypatch)


def _consumed(state_dir: Path) -> list[dict[str, Any]]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(
            (state_dir / "release-deploy" / "profile-approvals").glob("*/consumed.json")
        )
    ]


# --- the batch join plans a cluster's own profile like a single join ------------


def test_a_batch_join_stops_for_the_clusters_own_profile_like_a_single_join(
    scenario: MixedJoinScenario,
) -> None:
    """The 1+1 shape every ``deploy --gpu-cluster-arn <plain EKS>`` takes: one
    plain EKS cluster batch-joined into a hyperpod-v1 site, beside a HyperPod
    cluster that needs no profile of its own."""

    with pytest.raises(cluster_join_profile.JoinProfileApprovalRequired) as stop:
        scenario.join_batch("b", "e")

    plan = scenario.plan()
    assert plan["change_kind"] == "CLUSTER_PROFILE_NEW", plan
    assert (plan["current_version"], plan["desired_version"]) == (
        "hyperpod-v1",
        "eks-v1",
    ), "the first own version is reviewed against the site profile"
    assert plan["registration_cluster_id"] == "gpu-e", "anchored on the EKS cluster"
    assert plan["approval_required"] is True, "a new version always needs review"
    message = str(stop.value)
    assert "the join stopped for review" in message, message
    assert "  cluster: gpu-e\n" in message, "the stop names the cluster it plans"
    assert (
        f"gpu-fault-admin join-cluster --state-dir {scenario.state_dir} "
        f"--gpu-cluster-arn {scenario.request('b').gpu_cluster_arn} "
        f"--gpu-cluster-arn {eks('e').eks_arn} --approve-profile-plan "
        f"{plan['plan_sha256']} --reference CHG-<id>"
    ) in message, "the rerun repeats the batch with the approval"
    assert not any(event.startswith("executor-role") for event in scenario.events), (
        "the batch stopped before any AWS mutation"
    )
    assert not any(event.startswith("namespace:") for event in scenario.events), (
        "and before anything was created on a cluster"
    )
    assert not (scenario.state_dir / "profiles" / "eks-v1.yaml").exists(), (
        "no snapshot is written before the approval"
    )
    assert scenario.journal("e")["phase"] in {"FAILED", "ROLLED_BACK"}, (
        "the cluster the plan covers records the stop as a single join does"
    )
    assert "JoinProfileApprovalRequired" in scenario.journal("e")["failure"]["error"]
    assert scenario.journal("b")["completed_steps"] == [], (
        "the HyperPod cluster of the batch is left as it was"
    )

    with pytest.raises(cluster_join_profile.JoinProfileApprovalRequired):
        scenario.join_batch("b", "e")
    assert scenario.plan()["plan_sha256"] == plan["plan_sha256"], (
        "a rerun without approval stops on the same plan"
    )

    _approve_pending_plan(scenario.state_dir, plan["plan_sha256"])
    result = scenario.join_batch("b", "e")

    assert result["phase"] == "COMPLETED", result
    assert result["joined"] == ["gpu-e", "hp-gpu-b"], result
    snapshot = scenario.state_dir / "profiles" / "eks-v1.yaml"
    assert snapshot.is_file(), "the approved profile snapshot is written"
    assert yaml.safe_load(snapshot.read_text(encoding="utf-8"))["cluster_id"] == (
        "gpu-e"
    )
    own = {
        "source": str(snapshot),
        "templateSource": "config/runtime-profile.regional-eks-safe.example.yaml",
        "version": "eks-v1",
        "agentConfigDigest": agent_config_digest("eks-v1"),
    }
    assert scenario.entry("gpu-e")["runtimeProfile"] == own, (
        "the batch-joined cluster pins its own profile, as a single join does"
    )
    assert "runtimeProfile" not in scenario.entry("hp-gpu-b"), (
        "the HyperPod cluster stays on the site profile"
    )
    assert scenario.journal("e")["evidence"]["DISCOVERED"]["runtime_profile"] == own
    assert [item["release_id"] for item in _consumed(scenario.state_dir)] == [
        "join-gpu-e"
    ], "the approval is consumed once, by the join"
    assert not profile_approval.profile_plan_path(scenario.state_dir).exists(), (
        "nothing is left pending"
    )


def test_two_clusters_needing_one_version_share_one_plan_and_one_approval(
    scenario: MixedJoinScenario,
) -> None:
    """Two plain EKS clusters joined in one batch: one review of eks-v1, the
    same plan whichever cluster the batch lists first."""

    pending = [(eks("f"), "gpu-f"), (eks("e"), "gpu-e")]
    rerun = "gpu-fault-admin join-cluster --state-dir X --gpu-cluster-arn ..."

    with pytest.raises(cluster_join_profile.JoinProfileApprovalRequired) as stop:
        cluster_join_profile.cluster_runtime_profiles(
            scenario.site, scenario, pending, rerun_command=rerun
        )

    plan = scenario.plan()
    assert stop.value.cluster_ids == ("gpu-e", "gpu-f"), "the plan covers both"
    assert plan["registration_cluster_id"] == "gpu-e", "anchored on the first id"
    assert (plan["current_version"], plan["desired_version"]) == (
        "hyperpod-v1",
        "eks-v1",
    )
    assert [line.split(" (", 1)[0] for line in plan["changes"][:2]] == [
        "cluster gpu-e",
        "cluster gpu-f",
    ], plan["changes"]
    assert "for these 2 clusters" in plan["changes"][2], plan["changes"]
    message = str(stop.value)
    assert "  clusters: gpu-e, gpu-f\n" in message, message
    assert "-> eks-v1 (these clusters only)" in message, message

    with pytest.raises(cluster_join_profile.JoinProfileApprovalRequired):
        cluster_join_profile.cluster_runtime_profiles(
            scenario.site, scenario, list(reversed(pending)), rerun_command=rerun
        )
    assert scenario.plan()["plan_sha256"] == plan["plan_sha256"], (
        "the order the batch lists the clusters in does not change the plan"
    )

    _approve_pending_plan(scenario.state_dir, plan["plan_sha256"])
    profiles = cluster_join_profile.cluster_runtime_profiles(
        scenario.site, scenario, pending, rerun_command=rerun
    )

    snapshot = scenario.state_dir / "profiles" / "eks-v1.yaml"
    assert sorted(profiles) == ["gpu-e", "gpu-f"], profiles
    assert (
        profiles["gpu-e"]
        == profiles["gpu-f"]
        == {
            "source": str(snapshot),
            "templateSource": "config/runtime-profile.regional-eks-safe.example.yaml",
            "version": "eks-v1",
            "agentConfigDigest": agent_config_digest("eks-v1"),
        }
    ), "both clusters pin the one approved snapshot"
    assert [item["release_id"] for item in _consumed(scenario.state_dir)] == [
        "join-gpu-e"
    ], "one approval, consumed once"
    assert (
        cluster_join_profile.cluster_runtime_profiles(
            scenario.site, scenario, pending, rerun_command=rerun
        )
        == profiles
    ), "an approved version asks nothing on a rerun"


def test_a_cluster_on_the_site_profile_needs_no_plan_in_a_batch(
    scenario: MixedJoinScenario,
) -> None:
    result = scenario.join_batch("b")

    assert result["phase"] == "COMPLETED", result
    assert "runtimeProfile" not in scenario.entry("hp-gpu-b"), "no override needed"
    assert not profile_approval.profile_plan_path(scenario.state_dir).exists(), (
        "nothing to review"
    )


# --- deploy: one plan per version for every member that needs it ---------------

ARN = f"arn:aws:eks:{REGION}:{ACCOUNT}:cluster/{{}}"
PINNED_V1 = "eks-v1"


def _hyperpod(cluster_id: str) -> dict[str, Any]:
    return {
        "clusterId": cluster_id,
        "context": f"ctx-{cluster_id}",
        "hyperpodClusterName": cluster_id,
        "eksClusterArn": ARN.format(f"{cluster_id}-eks"),
    }


def _plain_eks(state_dir: Path, cluster_id: str, *, pinned: bool) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "clusterId": cluster_id,
        "context": f"ctx-{cluster_id}",
        "provider": "eks-managed",
        "eksClusterArn": ARN.format(cluster_id),
    }
    if pinned:
        entry["runtimeProfile"] = {
            "source": str(state_dir / "profiles" / f"{PINNED_V1}.yaml"),
            "templateSource": "config/runtime-profile.regional-eks-safe.example.yaml",
            "version": PINNED_V1,
            "agentConfigDigest": "0" * 64,
        }
    return entry


def _two_plus_two(tmp_path: Path, *, pinned: bool = True) -> tuple[Path, Path, dict]:
    """A 2 HyperPod + 2 plain-EKS site document whose EKS clusters pin
    ``eks-v1``, approved under the template policy before it took over
    ``nodeReplace``; the repository's template carries the new policy."""

    repo = tmp_path / "repo"
    (repo / "config").mkdir(parents=True)
    for source in (HYPERPOD_PROFILE, EKS_PROFILE):
        (repo / "config" / source.name).write_text(source.read_text(encoding="utf-8"))
    state_dir = tmp_path / "state"
    (state_dir / "profiles").mkdir(parents=True)
    older = yaml.safe_load(EKS_PROFILE.read_text(encoding="utf-8"))
    for claim in older["claims"]:
        if claim.get("capability") == "nodeReplace":
            claim["mode"] = "OBSERVE"
            claim.pop("adapter", None)
    older["cluster_id"] = "eks-1"
    older["profile_version"] = PINNED_V1
    (state_dir / "profiles" / f"{PINNED_V1}.yaml").write_text(
        yaml.safe_dump(older, sort_keys=False), encoding="utf-8"
    )
    document = {
        "apiVersion": "gpu-fault.aws/v1alpha1",
        "kind": "RegionalSite",
        "metadata": {"name": "site"},
        "spec": {
            "awsRegion": REGION,
            "cpu": {"eksArn": ARN.format("cpu")},
            "runtimeProfile": {
                "source": str(repo / "config" / HYPERPOD_PROFILE.name),
                "version": "hyperpod-v1",
                "registrationClusterId": "hp-1",
            },
            "clusters": [
                _hyperpod("hp-1"),
                _hyperpod("hp-2"),
                _plain_eks(state_dir, "eks-1", pinned=pinned),
                _plain_eks(state_dir, "eks-2", pinned=pinned),
            ],
        },
    }
    (state_dir / "site.yaml").write_text(
        yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
    )
    return repo, state_dir, document


def _successor(repo: Path) -> str:
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


def _plan(state_dir: Path) -> dict[str, Any]:
    path = profile_approval.profile_plan_path(state_dir)
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _reordered(document: dict[str, Any]) -> dict[str, Any]:
    """The document a deploy regenerates, listing the clusters in another order
    (``deploy`` sorts them by EKS ARN): here ``eks-2`` comes before ``eks-1``."""

    spec = dict(document["spec"])
    spec["clusters"] = sorted(
        (dict(item) for item in spec["clusters"]),
        key=lambda item: str(item["eksClusterArn"]),
        reverse=True,
    )
    return {**document, "spec": spec}


def test_the_deploy_preflight_converges_when_two_clusters_need_one_successor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, state_dir, document = _two_plus_two(tmp_path)
    _fake_digest(monkeypatch)
    successor = _successor(repo)

    with pytest.raises(cluster_join_profile.JoinProfileApprovalRequired) as stop:
        cluster_join_profile.preflight_cluster_profile_plans(
            state_dir, repository_root=repo
        )
    plan = _plan(state_dir)
    assert stop.value.cluster_ids == ("eks-1", "eks-2"), "one plan for both"
    assert (plan["current_version"], plan["desired_version"]) == (PINNED_V1, successor)
    assert plan["registration_cluster_id"] == "eks-1", plan
    assert "the deploy stopped for review" in str(stop.value)

    _approve_pending_plan(state_dir, plan["plan_sha256"])
    for _rerun in range(2):
        cluster_join_profile.preflight_cluster_profile_plans(
            state_dir, repository_root=repo
        )
    assert profile_approval.profile_approval_path(state_dir).is_file(), (
        "the preflight consumes nothing; the finalization writes and consumes"
    )

    regenerated = _reordered(document)
    cluster_join_profile.reconcile_cluster_runtime_profiles(
        object(), regenerated, repository_root=repo, state_dir=state_dir
    )

    snapshot = state_dir / "profiles" / f"{successor}.yaml"
    assert snapshot.is_file(), "the successor is snapshotted once approved"
    assert yaml.safe_load(snapshot.read_text(encoding="utf-8"))["cluster_id"] == (
        "eks-1"
    ), "the snapshot is the reviewed one, whatever order the deploy lists"
    pins = {
        item["clusterId"]: item.get("runtimeProfile")
        for item in regenerated["spec"]["clusters"]
    }
    assert pins["hp-1"] is None and pins["hp-2"] is None, pins
    assert (
        pins["eks-1"]
        == pins["eks-2"]
        == {
            "source": str(snapshot),
            "templateSource": "config/runtime-profile.regional-eks-safe.example.yaml",
            "version": successor,
            "agentConfigDigest": "r" * 64,
        }
    ), "both clusters move to the one approved successor"
    assert [item["release_id"] for item in _consumed(state_dir)] == ["deploy-eks-1"]
    assert (state_dir / "profiles" / f"{PINNED_V1}.yaml").is_file(), (
        "eks-v1 keeps its immutable snapshot"
    )

    (state_dir / "site.yaml").write_text(
        yaml.safe_dump(regenerated, sort_keys=False), encoding="utf-8"
    )
    cluster_join_profile.preflight_cluster_profile_plans(
        state_dir, repository_root=repo
    )
    assert not profile_approval.profile_plan_path(state_dir).exists(), (
        "the next deploy asks nothing"
    )


def test_a_member_that_never_got_its_own_profile_is_planned_by_deploy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The exit for a plain EKS cluster a batch join completed without its
    override: no approved snapshot exists and ``join-cluster`` has nothing left
    to join, so the deploy reviews the version the members need -- one plan,
    explicit approval -- instead of refusing every run."""

    repo, state_dir, document = _two_plus_two(tmp_path, pinned=False)
    (state_dir / "profiles" / f"{PINNED_V1}.yaml").unlink()
    _fake_digest(monkeypatch)

    with pytest.raises(cluster_join_profile.JoinProfileApprovalRequired) as stop:
        cluster_join_profile.preflight_cluster_profile_plans(
            state_dir, repository_root=repo
        )
    plan = _plan(state_dir)
    assert stop.value.cluster_ids == ("eks-1", "eks-2"), "every member needing it"
    assert (plan["current_version"], plan["desired_version"]) == (
        "hyperpod-v1",
        PINNED_V1,
    ), "a first own version is reviewed against the site profile"
    assert plan["current_policy_digest"] is None, plan
    assert all(
        " runs its own Runtime Profile eks-v1" in line for line in plan["changes"][:2]
    ), plan["changes"]

    with pytest.raises(cluster_join_profile.JoinProfileApprovalRequired):
        cluster_join_profile.reconcile_cluster_runtime_profiles(
            object(), _reordered(document), repository_root=repo, state_dir=state_dir
        )
    assert _plan(state_dir)["plan_sha256"] == plan["plan_sha256"], (
        "the finalization plans the same version the preflight did"
    )

    _approve_pending_plan(state_dir, plan["plan_sha256"])
    cluster_join_profile.preflight_cluster_profile_plans(
        state_dir, repository_root=repo
    )
    regenerated = _reordered(document)
    cluster_join_profile.reconcile_cluster_runtime_profiles(
        object(), regenerated, repository_root=repo, state_dir=state_dir
    )
    snapshot = state_dir / "profiles" / f"{PINNED_V1}.yaml"
    pinned = [
        item["runtimeProfile"]["version"]
        for item in regenerated["spec"]["clusters"]
        if item.get("provider") == "eks-managed"
    ]
    assert pinned == [PINNED_V1, PINNED_V1], regenerated["spec"]["clusters"]
    assert snapshot.is_file(), "the reviewed snapshot is written"
    assert [item["release_id"] for item in _consumed(state_dir)] == ["deploy-eks-1"]


def test_an_approval_for_another_plan_is_superseded_and_the_deploy_stops_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An approval covers exactly its plan. When the clusters needing the
    version change before it is consumed (here ``eks-2`` left the site), the
    old approval is archived SUPERSEDED and the new plan is written for
    review, instead of a dead end "approval is unusable" on every rerun."""

    repo, state_dir, document = _two_plus_two(tmp_path)
    _fake_digest(monkeypatch)
    with pytest.raises(cluster_join_profile.JoinProfileApprovalRequired):
        cluster_join_profile.preflight_cluster_profile_plans(
            state_dir, repository_root=repo
        )
    approved = _plan(state_dir)
    _approve_pending_plan(state_dir, approved["plan_sha256"])
    document["spec"]["clusters"] = [
        item for item in document["spec"]["clusters"] if item["clusterId"] != "eks-2"
    ]
    (state_dir / "site.yaml").write_text(
        yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
    )

    with pytest.raises(cluster_join_profile.JoinProfileApprovalRequired) as stop:
        cluster_join_profile.preflight_cluster_profile_plans(
            state_dir, repository_root=repo
        )

    replacement = _plan(state_dir)
    assert replacement["plan_sha256"] != approved["plan_sha256"], replacement
    assert stop.value.cluster_ids == ("eks-1",), "the plan names who needs it now"
    assert "SUPERSEDED" in str(stop.value), str(stop.value)
    assert replacement["plan_sha256"] in str(stop.value), "the new digest is printed"
    assert not profile_approval.profile_approval_path(state_dir).exists(), (
        "the old approval no longer stands"
    )
    archive = profile_approval.profile_approval_archive_path(
        state_dir, approved["plan_sha256"]
    )
    superseded = json.loads((archive / "superseded.json").read_text(encoding="utf-8"))
    assert superseded["status"] == "SUPERSEDED", superseded
    assert superseded["replacement_plan_sha256"] == replacement["plan_sha256"]
    assert not (state_dir / "profiles" / f"{_successor(repo)}.yaml").exists(), (
        "nothing was applied under the superseded approval"
    )
