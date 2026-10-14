"""A new cluster's Runtime Profile approval is consumed by the join it was planned for.

``deploy --gpu-cluster-arn NEW`` joins the new cluster after its application
release. A cluster that needs its own profile version stops that join with a
``CLUSTER_PROFILE_NEW`` plan; ``deploy --approve-profile-plan SHA`` approved it
and went on -- but the release planned the site profile first, judged the
pending approval stale against its own unchanged plan, archived it SUPERSEDED
and cleared the plan, and the join then stopped again with the same digest,
which could not be approved a second time ("archived Profile approval differs
from the active approval"). The site profile resolution now leaves a cluster
plan and its approval to the join, and a site profile change still takes the
one plan file first -- every approval stays explicit.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from gpu_fault.admin import cluster_join_profile
from gpu_fault.admin import profile_approval as approvals
from gpu_fault.admin.site import load_site
from scripts import release_deploy
from tests.admin.test_admin_eks_managed_join import eks_target
from tests.admin.test_release_deploy import (
    _release_diff,
    _release_summary,
    _site,
    _stability_report,
    _verification_report,
)

ROOT = Path(__file__).resolve().parents[2]
EKS_TEMPLATE = "config/runtime-profile.regional-eks-safe.example.yaml"
HYPERPOD_TEMPLATE = "config/runtime-profile.regional-hyperpod-safe.example.yaml"
RERUN = "gpu-fault-admin join-cluster --state-dir STATE --gpu-cluster-arn NEW"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hyperpod_site(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A released hyperpod-v1 site (HyperPod ``gpu-a``) no plain EKS cluster
    has joined yet: the first one needs ``eks-v1`` of its own."""

    site = _site(tmp_path, monkeypatch)
    repo = tmp_path / "repo"
    for name in (EKS_TEMPLATE, HYPERPOD_TEMPLATE):
        (repo / name).write_bytes((ROOT / name).read_bytes())
    document = yaml.safe_load(site.read_text(encoding="utf-8"))
    document["spec"]["runtimeProfile"] = {
        "source": HYPERPOD_TEMPLATE,
        "templateSource": HYPERPOD_TEMPLATE,
        "version": "hyperpod-v1",
        "registrationClusterId": "gpu-a",
    }
    site.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    plan = release_deploy.plan_runtime_profile(
        site, live_profile_sha256=_sha256(repo / HYPERPOD_TEMPLATE)
    )
    assert plan.change_kind == "UNCHANGED", plan
    release_deploy.prepare_site_release(site, profile_plan=plan)
    monkeypatch.setattr(
        cluster_join_profile,
        "compute_agent_config_digest",
        lambda _runner, *, repository_root, runtime_profile_version: "e" * 64,
    )
    monkeypatch.setattr(release_deploy, "_run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(release_deploy, "_run_json", _run_json)
    return site


def _run_json(arguments: list[str], **_kwargs: Any) -> dict[str, Any]:
    if "verify" in arguments:
        return _verification_report()
    if "stability" in arguments:
        return _stability_report()
    if "release-diff" in arguments:
        return _release_diff("CONTROL_PLANE_ONLY", ["control_plane_wheel"])
    return _release_summary()


def _live_state(site: Path) -> dict[str, str]:
    """The live state of the released site: its snapshot is the live profile."""

    source = yaml.safe_load(site.read_text(encoding="utf-8"))["spec"]["runtimeProfile"]
    return {"runtime_profile_sha256": _sha256(Path(source["source"]))}


def _join_profiles(site: Path) -> dict[str, dict[str, str]]:
    """The profile step of joining the plain EKS cluster ``gpu-e``."""

    return cluster_join_profile.cluster_runtime_profiles(
        load_site(site), object(), [(eks_target(), "gpu-e")], rerun_command=RERUN
    )


def _stopped_join_plan(site: Path) -> dict[str, Any]:
    with pytest.raises(cluster_join_profile.JoinProfileApprovalRequired):
        _join_profiles(site)
    plan = json.loads(approvals.profile_plan_path(site.parent).read_text())
    assert plan["change_kind"] == "CLUSTER_PROFILE_NEW", plan
    return dict(plan)


def _archive(site: Path, digest: str) -> Path:
    return approvals.profile_approval_archive_path(site.parent, digest)


def test_a_release_before_the_join_leaves_the_clusters_approval_to_the_join(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = _hyperpod_site(tmp_path, monkeypatch)
    plan = _stopped_join_plan(site)
    # deploy --approve-profile-plan SHA --reference REF approves inline ...
    approvals.approve_profile_plan_inline(
        site.parent, plan_sha256=str(plan["plan_sha256"]), reference="CHG-join-1"
    )

    # ... and its application release runs before the new cluster joins.
    prepared = release_deploy.execute_release(
        site, run_checks=False, live_state=_live_state(site)
    )

    state = json.loads((prepared.state_dir / "state.json").read_text())
    assert state["phase"] == "COMPLETED", state
    assert state["profile_change"]["kind"] == "UNCHANGED", state["profile_change"]
    assert state["profile_change"]["approval"] is None, (
        "the site profile release consumed nothing of the cluster's approval"
    )
    assert "profile_approval_audit" not in state, state
    assert approvals.profile_approval_path(site.parent).is_file(), (
        "the release must leave the new cluster's approval for its join"
    )
    assert json.loads(approvals.profile_plan_path(site.parent).read_text()) == plan, (
        "the plan the approval covers stays pending, unchanged"
    )
    assert not (_archive(site, plan["plan_sha256"]) / "superseded.json").exists(), (
        "the approval was archived SUPERSEDED by a release that had nothing to approve"
    )

    profiles = _join_profiles(site)

    assert profiles["gpu-e"]["version"] == "eks-v1", profiles
    consumed = json.loads(
        (_archive(site, plan["plan_sha256"]) / "consumed.json").read_text()
    )
    assert (consumed["release_id"], consumed["relation"]) == ("join-gpu-e", "EXACT")
    assert consumed["reference"] == "CHG-join-1", consumed
    assert not approvals.profile_approval_path(site.parent).exists(), (
        "the join consumed the approval"
    )
    assert not approvals.profile_plan_path(site.parent).exists(), (
        "nothing is left pending once the join consumed the approval"
    )


def test_a_pending_cluster_plan_without_approval_survives_the_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = _hyperpod_site(tmp_path, monkeypatch)
    plan = _stopped_join_plan(site)

    release_deploy.execute_release(site, run_checks=False, live_state=_live_state(site))

    assert json.loads(approvals.profile_plan_path(site.parent).read_text()) == plan, (
        "the join's plan stays pending for the review it stopped for"
    )
    approvals.approve_profile_plan_inline(
        site.parent, plan_sha256=str(plan["plan_sha256"]), reference="CHG-join-2"
    )
    assert _join_profiles(site)["gpu-e"]["version"] == "eks-v1"


def _change_site_profile(tmp_path: Path) -> None:
    """A release whose own site profile template changed (a capability left)."""

    template = tmp_path / "repo" / HYPERPOD_TEMPLATE
    document = yaml.safe_load(template.read_text(encoding="utf-8"))
    document["claims"] = [
        claim
        for claim in document["claims"]
        if claim.get("capability") != "nvlinkDiagnostics"
    ]
    template.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")


def test_a_site_profile_change_still_takes_the_plan_file_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One plan file per site: a release whose own site profile changed stops
    for that review and archives the cluster's approval SUPERSEDED, as any
    approval of another plan is; the site profile is never applied under it."""

    site = _hyperpod_site(tmp_path, monkeypatch)
    plan = _stopped_join_plan(site)
    approvals.approve_profile_plan_inline(
        site.parent, plan_sha256=str(plan["plan_sha256"]), reference="CHG-join-3"
    )
    _change_site_profile(tmp_path)

    with pytest.raises(release_deploy.ReleaseDeployError, match="SUPERSEDED"):
        release_deploy.execute_release(
            site, run_checks=False, live_state=_live_state(site)
        )

    base = json.loads(approvals.profile_plan_path(site.parent).read_text())
    assert base["change_kind"] == "RESTRICTIVE", base
    assert base["approval_required"] is True, base
    superseded = json.loads(
        (_archive(site, plan["plan_sha256"]) / "superseded.json").read_text()
    )
    assert superseded["replacement_plan_sha256"] == base["plan_sha256"], superseded
    assert not approvals.profile_approval_path(site.parent).exists(), (
        "no approval of the cluster plan covers the site profile's"
    )
