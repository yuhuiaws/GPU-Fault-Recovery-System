"""A plan whose approval was superseded can be approved again -- explicitly.

Batch 90 reproduced the dead end with the real ``profile_approval`` functions:
a new cluster's join stopped for its ``CLUSTER_PROFILE_NEW`` plan, the
operator resumed with ``deploy --approve-profile-plan``, the deploy's release
archived that approval SUPERSEDED, and the join wrote the same plan again. The
digest could never be approved again ("archived Profile approval differs from
the active approval"): the round's archive is immutable and a new record is a
different record. No command resumed the join. The order fix keeps new joins
off that path; sites already in that state (and any plan that comes back under
its digest after another review displaced it) are approved anew: the earlier
round moves, as it is, to ``profile-approvals/<plan_sha256>.<n>/`` and
``earlier-rounds.json`` records it beside the new approval. Nothing is
approved for the operator, and an approval still covers exactly its plan.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.admin import profile_approval as approvals
from scripts import release_deploy
from tests.admin.test_profile_approval_join_order import (
    _archive,
    _change_site_profile,
    _hyperpod_site,
    _join_profiles,
    _live_state,
    _stopped_join_plan,
)

SITE_IDENTITY = {
    "site_name": "test-site",
    "aws_region": "us-east-1",
    "cpu_eks_arn": "arn:aws:eks:us-east-1:123456789012:cluster/control",
}


def _cluster_plan(**updates: Any) -> dict[str, Any]:
    """A new cluster's ``CLUSTER_PROFILE_NEW`` plan, as a join writes it."""

    value: dict[str, Any] = {
        "schema_version": 1,
        "site_identity": dict(SITE_IDENTITY),
        "site_identity_sha256": approvals.profile_site_identity_sha256(SITE_IDENTITY),
        "registration_cluster_id": "gpu-e",
        "template_source": "/repo-1/config/runtime-profile.regional-eks-safe.example.yaml",
        "active_source": "/state/profiles/hyperpod-v1.yaml",
        "snapshot_file": "/state/profiles/eks-v1.yaml",
        "current_version": "hyperpod-v1",
        "desired_version": "eks-v1",
        "current_policy_digest": None,
        "policy_digest": "6" * 64,
        "live_profile_sha256": None,
        "active_source_sha256": None,
        "source_sha256": "7" * 64,
        "snapshot_sha256": "8" * 64,
        "change_kind": "CLUSTER_PROFILE_NEW",
        "changes": ["cluster gpu-e (eks-managed) runs its own Runtime Profile eks-v1"],
        "approval_required": True,
    }
    value.update(updates)
    value["plan_sha256"] = approvals.profile_plan_sha256(value)
    return value


def _read(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _approve(state_dir: Path, plan: dict[str, Any], reference: str) -> dict[str, Any]:
    return approvals.approve_profile(
        state_dir,
        reference=reference,
        expected_plan_sha256=str(plan["plan_sha256"]),
        approver_identity=f"{reference.lower()}@fixture",
    )


def _dead_end(state_dir: Path, plan: dict[str, Any]) -> None:
    """The state the old order left: the join's approval archived SUPERSEDED by
    the deploy's release (plan and approval cleared), then the join stopped
    again and wrote the same plan."""

    approvals.write_profile_plan(state_dir, plan)
    _approve(state_dir, plan, "CHG-1")
    approvals.supersede_profile_approval(
        state_dir,
        replacement_plan=None,
        reason="Profile approval is stale because the pending plan or live baseline changed",
    )
    assert not approvals.profile_approval_path(state_dir).exists(), (
        "the old order archived the approval and removed it"
    )
    approvals.write_profile_plan(state_dir, plan)


def test_a_plan_whose_approval_was_superseded_is_approvable_again(
    tmp_path: Path,
) -> None:
    plan = _cluster_plan()
    _dead_end(tmp_path, plan)
    digest = str(plan["plan_sha256"])
    archive = approvals.profile_approval_archive_path(tmp_path, digest)
    assert (archive / "superseded.json").is_file(), "the dead end as the sites hold it"
    assert approvals.resolve_profile_approval(tmp_path, current_plan=plan) is None, (
        "a superseded approval never comes back by itself"
    )

    record = _approve(tmp_path, plan, "CHG-2")

    assert record["reference"] == "CHG-2", record
    earlier = archive.with_name(f"{digest}.1")
    assert _read(earlier / "approval.json")["reference"] == "CHG-1"
    assert _read(earlier / "superseded.json")["status"] == "SUPERSEDED"
    assert _read(earlier / "plan.json") == plan, "the earlier round moved as it was"
    assert _read(archive / "approval.json") == record, "a new round for the new record"
    index = _read(archive / approvals.EARLIER_ROUNDS)
    assert index["reference"] == "CHG-2", index
    assert index["earlier_rounds"] == [
        {
            "archive": f"{digest}.1",
            "status": "SUPERSEDED",
            "reference": "CHG-1",
            "approver_identity": "chg-1@fixture",
            "approved_at": _read(earlier / "approval.json")["approved_at"],
        }
    ], index

    resolved = approvals.resolve_profile_approval(tmp_path, current_plan=plan)
    assert resolved is not None
    assert (resolved.relation, resolved.reference) == ("EXACT", "CHG-2")
    approvals.consume_profile_approval(
        tmp_path,
        expected_plan_sha256=digest,
        release_id="join-gpu-e",
        relation=resolved.relation,
    )
    consumed = _read(archive / "consumed.json")
    assert (consumed["reference"], consumed["release_id"]) == ("CHG-2", "join-gpu-e")
    assert not (earlier / "consumed.json").exists(), "the earlier round stays closed"


def test_a_plan_rewritten_with_other_paths_is_approvable_again(tmp_path: Path) -> None:
    """A deploy moves ``repositoryRoot``, so the plan the join writes again has
    another ``template_source`` under the same digest; its archived copy used
    to refuse the approval ("archived Profile plan differs")."""

    plan = _cluster_plan()
    _dead_end(tmp_path, plan)
    moved = {
        **plan,
        "template_source": "/repo-2/config/runtime-profile.regional-eks-safe.example.yaml",
    }
    assert moved["plan_sha256"] == approvals.profile_plan_sha256(moved)
    approvals.write_profile_plan(tmp_path, moved)

    _approve(tmp_path, moved, "CHG-3")

    archive = approvals.profile_approval_archive_path(
        tmp_path, str(plan["plan_sha256"])
    )
    assert _read(archive / "plan.json") == moved, "the round archives the plan approved"


def test_every_approval_still_covers_exactly_its_plan(tmp_path: Path) -> None:
    plan = _cluster_plan()
    _dead_end(tmp_path, plan)

    with pytest.raises(
        approvals.ProfileApprovalError, match="does not match the reviewed"
    ):
        approvals.approve_profile(
            tmp_path, reference="CHG-4", expected_plan_sha256="f" * 64
        )
    first = _approve(tmp_path, plan, "CHG-4")
    assert _approve(tmp_path, plan, "CHG-4") == first, "an approval is idempotent"
    with pytest.raises(
        approvals.ProfileApprovalError, match="different approval reference"
    ):
        _approve(tmp_path, plan, "CHG-5")
    changed = _cluster_plan(snapshot_sha256="9" * 64)
    with pytest.raises(approvals.StaleProfileApprovalError):
        approvals.resolve_profile_approval(tmp_path, current_plan=changed)
    digest = str(plan["plan_sha256"])
    rounds = sorted(
        path.name
        for path in approvals.profile_approval_archive_path(
            tmp_path, digest
        ).parent.iterdir()
    )
    assert rounds == [digest, f"{digest}.1"], "only a new approval opens a round"


def test_a_consumed_plan_that_comes_back_is_approved_anew(tmp_path: Path) -> None:
    """A rolled-back change plans the same digest again: approved explicitly,
    the consumed round stays as the audit of the first application."""

    plan = _cluster_plan()
    approvals.write_profile_plan(tmp_path, plan)
    _approve(tmp_path, plan, "CHG-6")
    digest = str(plan["plan_sha256"])
    approvals.consume_profile_approval(
        tmp_path, expected_plan_sha256=digest, release_id="release-a", relation="EXACT"
    )
    approvals.write_profile_plan(tmp_path, plan)

    _approve(tmp_path, plan, "CHG-7")

    archive = approvals.profile_approval_archive_path(tmp_path, digest)
    earlier = archive.with_name(f"{digest}.1")
    assert _read(earlier / "consumed.json")["release_id"] == "release-a"
    assert _read(archive / approvals.EARLIER_ROUNDS)["earlier_rounds"][0]["status"] == (
        "CONSUMED"
    )


def test_a_join_resumed_after_a_site_profile_review_took_its_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The deploy's release stopped for its own site profile change and archived
    the new cluster's approval SUPERSEDED; the operator resumes the join first:
    the same plan is written again and approved again, and the join consumes it."""

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

    assert _stopped_join_plan(site)["plan_sha256"] == plan["plan_sha256"]
    approvals.approve_profile_plan_inline(
        site.parent, plan_sha256=str(plan["plan_sha256"]), reference="CHG-join-4"
    )

    assert _join_profiles(site)["gpu-e"]["version"] == "eks-v1"
    archive = _archive(site, plan["plan_sha256"])
    assert _read(archive / "consumed.json")["reference"] == "CHG-join-4"
    earlier = archive.with_name(f"{plan['plan_sha256']}.1")
    assert _read(earlier / "approval.json")["reference"] == "CHG-join-3"
    assert (earlier / "superseded.json").is_file(), "its audit stays complete"
