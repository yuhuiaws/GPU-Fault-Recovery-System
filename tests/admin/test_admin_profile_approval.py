from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from gpu_fault.admin import profile_approval as admin_profile_approval


def _plan(**updates: object) -> dict[str, object]:
    site_identity = {
        "site_name": "test-site",
        "aws_region": "us-east-1",
        "cpu_eks_arn": "arn:aws:eks:us-east-1:123456789012:cluster/control",
    }
    value: dict[str, object] = {
        "schema_version": 1,
        "site_identity": site_identity,
        "site_identity_sha256": (
            admin_profile_approval.profile_site_identity_sha256(site_identity)
        ),
        "registration_cluster_id": "gpu-a",
        "template_source": "/repo/config/profile.yaml",
        "active_source": "/state/profiles/profile-v1.yaml",
        "snapshot_file": "/state/profiles/profile-v2.yaml",
        "current_version": "profile-v1",
        "desired_version": "profile-v2",
        "current_policy_digest": "1" * 64,
        "policy_digest": "2" * 64,
        "live_profile_sha256": "3" * 64,
        "active_source_sha256": "3" * 64,
        "source_sha256": "4" * 64,
        "snapshot_sha256": "5" * 64,
        "change_kind": "EXPANSIVE",
        "changes": ["gpuReset: mode OBSERVE->OWN"],
        "approval_required": True,
    }
    value.update(updates)
    value["plan_sha256"] = admin_profile_approval.profile_plan_sha256(value)
    return value


def test_profile_approval_is_bound_audited_and_idempotent(tmp_path: Path) -> None:
    plan = _plan()
    admin_profile_approval.write_profile_plan(tmp_path, plan)
    approved_at = datetime(2026, 8, 30, 12, 0, tzinfo=UTC)

    first = admin_profile_approval.approve_profile(
        tmp_path,
        reference="CHG-12345",
        expected_plan_sha256=str(plan["plan_sha256"]),
        approved_at=approved_at,
    )
    repeated = admin_profile_approval.approve_profile(
        tmp_path,
        reference="CHG-12345",
        expected_plan_sha256=str(plan["plan_sha256"]),
        approved_at=datetime(2026, 8, 30, 13, 0, tzinfo=UTC),
    )

    assert repeated == first
    assert first["plan_sha256"] == plan["plan_sha256"]
    approval_path = admin_profile_approval.profile_approval_path(tmp_path)
    assert approval_path.stat().st_mode & 0o777 == 0o600
    assert approval_path.parent.stat().st_mode & 0o777 == 0o700
    archive = admin_profile_approval.profile_approval_archive_path(
        tmp_path, str(plan["plan_sha256"])
    )
    assert json.loads((archive / "plan.json").read_text()) == plan
    assert json.loads((archive / "approval.json").read_text()) == first
    assert first["site_identity"]["cpu_eks_arn"].endswith("/control"), (
        "approval record omitted the reviewed CPU control-plane identity"
    )

    with pytest.raises(
        admin_profile_approval.ProfileApprovalError,
        match="different approval reference",
    ):
        admin_profile_approval.approve_profile(
            tmp_path,
            reference="CHG-99999",
            expected_plan_sha256=str(plan["plan_sha256"]),
        )


def test_profile_approval_rejects_a_different_reviewed_plan_sha(tmp_path: Path) -> None:
    plan = _plan()
    admin_profile_approval.write_profile_plan(tmp_path, plan)

    with pytest.raises(
        admin_profile_approval.ProfileApprovalError, match="does not match the reviewed"
    ):
        admin_profile_approval.approve_profile(
            tmp_path, reference="CHG-12345", expected_plan_sha256="f" * 64
        )

    assert not admin_profile_approval.profile_approval_path(tmp_path).exists(), (
        "mismatched reviewed SHA created an active approval"
    )


def test_profile_approval_resolves_resume_and_already_applied_state(
    tmp_path: Path,
) -> None:
    plan = _plan()
    admin_profile_approval.write_profile_plan(tmp_path, plan)
    admin_profile_approval.approve_profile(
        tmp_path,
        reference="MW-2026-08-30",
        expected_plan_sha256=str(plan["plan_sha256"]),
    )

    exact = admin_profile_approval.resolve_profile_approval(tmp_path, current_plan=plan)
    assert exact is not None
    assert exact.relation == "EXACT"

    resume = _plan(
        current_version="profile-v2",
        current_policy_digest=None,
        active_source="/state/profiles/profile-v2.yaml",
        active_source_sha256="5" * 64,
        change_kind="UNKNOWN_BASELINE",
        changes=["active Profile content is unavailable or differs from live SHA"],
    )
    resumed = admin_profile_approval.resolve_profile_approval(
        tmp_path, current_plan=resume
    )
    assert resumed is not None
    assert resumed.relation == "PREPARED_RESUME"

    applied = _plan(
        current_version="profile-v2",
        current_policy_digest="2" * 64,
        active_source="/state/profiles/profile-v2.yaml",
        live_profile_sha256="5" * 64,
        active_source_sha256="5" * 64,
        change_kind="UNCHANGED",
        changes=[],
        approval_required=False,
    )
    observed = admin_profile_approval.resolve_profile_approval(
        tmp_path, current_plan=applied
    )
    assert observed is not None
    assert observed.relation == "ALREADY_APPLIED"


def test_profile_approval_is_consumed_once_after_success(tmp_path: Path) -> None:
    plan = _plan()
    admin_profile_approval.write_profile_plan(tmp_path, plan)
    admin_profile_approval.approve_profile(
        tmp_path, reference="CHG-12345", expected_plan_sha256=str(plan["plan_sha256"])
    )

    archive = admin_profile_approval.consume_profile_approval(
        tmp_path,
        expected_plan_sha256=str(plan["plan_sha256"]),
        release_id="release-a",
        relation="EXACT",
        consumed_at=datetime(2026, 8, 30, 14, 0, tzinfo=UTC),
    )

    assert not admin_profile_approval.profile_plan_path(tmp_path).exists(), (
        "consumed Profile plan remained active"
    )
    assert not admin_profile_approval.profile_approval_path(tmp_path).exists(), (
        "consumed Profile approval remained active"
    )
    result = json.loads((archive / "consumed.json").read_text())
    assert result["status"] == "CONSUMED"
    assert result["release_id"] == "release-a"
    assert (
        admin_profile_approval.resolve_profile_approval(tmp_path, current_plan=plan)
        is None
    )
    with pytest.raises(
        admin_profile_approval.ProfileApprovalError, match="no pending Profile plan"
    ):
        admin_profile_approval.approve_profile(
            tmp_path,
            reference="CHG-12345",
            expected_plan_sha256=str(plan["plan_sha256"]),
        )


def test_stale_profile_approval_is_archived_before_reapproval(tmp_path: Path) -> None:
    original = _plan()
    admin_profile_approval.write_profile_plan(tmp_path, original)
    admin_profile_approval.approve_profile(
        tmp_path,
        reference="CHG-12345",
        expected_plan_sha256=str(original["plan_sha256"]),
    )
    replacement = _plan(
        desired_version="profile-v3",
        policy_digest="6" * 64,
        source_sha256="7" * 64,
        snapshot_sha256="8" * 64,
        changes=["nodeReboot: mode OBSERVE->OWN"],
    )

    with pytest.raises(
        admin_profile_approval.StaleProfileApprovalError, match="live baseline changed"
    ):
        admin_profile_approval.resolve_profile_approval(
            tmp_path, current_plan=replacement
        )

    archive = admin_profile_approval.supersede_profile_approval(
        tmp_path,
        replacement_plan=replacement,
        reason="pending plan changed",
        superseded_at=datetime(2026, 8, 30, 15, 0, tzinfo=UTC),
    )

    assert archive is not None
    assert not admin_profile_approval.profile_approval_path(tmp_path).exists(), (
        "superseded Profile approval remained active"
    )
    assert (
        json.loads(admin_profile_approval.profile_plan_path(tmp_path).read_text())
        == replacement
    )
    superseded = json.loads((archive / "superseded.json").read_text())
    assert superseded["status"] == "SUPERSEDED"
    assert superseded["replacement_plan_sha256"] == replacement["plan_sha256"]


def test_tampered_profile_plan_cannot_be_approved(tmp_path: Path) -> None:
    plan = _plan()
    admin_profile_approval.write_profile_plan(tmp_path, plan)
    plan["desired_version"] = "tampered"
    admin_profile_approval.profile_plan_path(tmp_path).write_text(
        json.dumps(plan), encoding="utf-8"
    )

    with pytest.raises(
        admin_profile_approval.ProfileApprovalError, match="digest does not match"
    ):
        admin_profile_approval.approve_profile(
            tmp_path,
            reference="CHG-12345",
            expected_plan_sha256=str(plan["plan_sha256"]),
        )


def test_tampered_readable_site_identity_cannot_be_approved(tmp_path: Path) -> None:
    plan = _plan()
    admin_profile_approval.write_profile_plan(tmp_path, plan)
    identity = dict(plan["site_identity"])
    identity["cpu_eks_arn"] = (
        "arn:aws:eks:us-east-1:123456789012:cluster/different-control"
    )
    plan["site_identity"] = identity
    plan["plan_sha256"] = admin_profile_approval.profile_plan_sha256(plan)
    admin_profile_approval.profile_plan_path(tmp_path).write_text(
        json.dumps(plan), encoding="utf-8"
    )

    with pytest.raises(
        admin_profile_approval.ProfileApprovalError,
        match="site identity digest does not match",
    ):
        admin_profile_approval.approve_profile(
            tmp_path,
            reference="CHG-12345",
            expected_plan_sha256=str(plan["plan_sha256"]),
        )


APPROVER = "arn:aws:sts::123456789012:assumed-role/Admin/alice"


def test_the_approval_records_who_approved_outside_the_plan_digest(
    tmp_path: Path,
) -> None:
    """An approval carried only a free-text reference and a time (I2).

    The approver's identity now sits on the record, in the archive and in the
    consumption audit. It stays *outside* ``plan_sha256``: that digest binds the
    reviewed plan and is computed before anyone approves, so folding the
    approver in would change the digest the operator was asked to type back.
    """

    plan = _plan()
    admin_profile_approval.write_profile_plan(tmp_path, plan)

    record = admin_profile_approval.approve_profile(
        tmp_path,
        reference="CHG-12345",
        expected_plan_sha256=str(plan["plan_sha256"]),
        approver_identity=APPROVER,
    )

    assert record["approver_identity"] == APPROVER
    assert record["plan_sha256"] == plan["plan_sha256"], (
        "the approver must not change the plan digest the operator reviewed"
    )
    archive = admin_profile_approval.profile_approval_archive_path(
        tmp_path, str(plan["plan_sha256"])
    )
    archived = json.loads((archive / "approval.json").read_text())
    assert archived["approver_identity"] == APPROVER, (
        "the archived approval lost the approver"
    )
    resolved = admin_profile_approval.resolve_profile_approval(
        tmp_path, current_plan=plan
    )
    assert resolved is not None
    assert resolved.relation == "EXACT"
    assert resolved.approver_identity == APPROVER

    admin_profile_approval.consume_profile_approval(
        tmp_path,
        expected_plan_sha256=str(plan["plan_sha256"]),
        release_id="release-1",
        relation="EXACT",
    )
    consumed = json.loads((archive / "consumed.json").read_text())
    assert consumed["approver_identity"] == APPROVER, (
        "the consumption audit must name who approved what was consumed"
    )


def test_the_approver_defaults_to_the_resolved_operator(tmp_path: Path) -> None:
    from tests.admin.conftest import TEST_OPERATOR_ARN

    plan = _plan()
    admin_profile_approval.write_profile_plan(tmp_path, plan)

    record = admin_profile_approval.approve_profile(
        tmp_path, reference="CHG-12345", expected_plan_sha256=str(plan["plan_sha256"])
    )

    assert record["approver_identity"] == TEST_OPERATOR_ARN, (
        "without an explicit approver the STS caller identity must be recorded"
    )


def test_an_approval_written_before_the_field_existed_still_resolves(
    tmp_path: Path,
) -> None:
    plan = _plan()
    admin_profile_approval.write_profile_plan(tmp_path, plan)
    record = admin_profile_approval.approve_profile(
        tmp_path, reference="CHG-12345", expected_plan_sha256=str(plan["plan_sha256"])
    )
    legacy = {key: value for key, value in record.items() if key != "approver_identity"}
    approval_path = admin_profile_approval.profile_approval_path(tmp_path)
    approval_path.write_text(json.dumps(legacy), encoding="utf-8")
    archive = admin_profile_approval.profile_approval_archive_path(
        tmp_path, str(plan["plan_sha256"])
    )
    (archive / "approval.json").write_text(json.dumps(legacy), encoding="utf-8")

    resolved = admin_profile_approval.resolve_profile_approval(
        tmp_path, current_plan=plan
    )

    assert resolved is not None
    assert resolved.reference == "CHG-12345"
    assert resolved.approver_identity is None, (
        "an older record has no approver; it must not be invented"
    )
