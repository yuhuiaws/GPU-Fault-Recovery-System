from __future__ import annotations

import copy
import json
from datetime import UTC, datetime

import pytest

from gpu_fault.admin import profile_approval as approval
from gpu_fault.admin.operation_lock import site_operation_lock


def plan(**changes):
    identity = {
        "site_name": "example-site",
        "aws_region": "us-east-1",
        "cpu_eks_arn": "arn:aws:eks:us-east-1:123456789012:cluster/control",
    }
    value = {
        "schema_version": 1,
        "site_identity": identity,
        "site_identity_sha256": approval.profile_site_identity_sha256(identity),
        "registration_cluster_id": "gpu-a",
        "current_version": "profile-v1",
        "desired_version": "profile-v2",
        "current_policy_digest": "1" * 64,
        "policy_digest": "2" * 64,
        "live_profile_sha256": "3" * 64,
        "active_source_sha256": "3" * 64,
        "source_sha256": "4" * 64,
        "snapshot_sha256": "5" * 64,
        "change_kind": "EXPANSIVE",
        "changes": ["example-reviewed-change"],
        "approval_required": True,
        **changes,
    }
    value["plan_sha256"] = approval.profile_plan_sha256(value)
    return value


@pytest.fixture
def approved(tmp_path):
    value = plan()
    approval.write_profile_plan(tmp_path, value)
    record = approval.approve_profile(
        tmp_path,
        reference="CHG-EXAMPLE",
        expected_plan_sha256=value["plan_sha256"],
        approved_at=datetime(2026, 9, 12, tzinfo=UTC),
        approver_identity="example-operator",
    )
    return tmp_path, value, record


@pytest.mark.parametrize("identity", [None, [], {}, {"site_name": "example"}])
def test_site_identity_requires_exact_object_fields(identity):
    with pytest.raises(approval.ProfileApprovalError, match="object|exactly"):
        approval.profile_site_identity_sha256(identity)


@pytest.mark.parametrize("field", ["site_name", "aws_region", "cpu_eks_arn"])
@pytest.mark.parametrize("value", [None, "", " padded "])
def test_site_identity_does_not_normalize_invalid_binding(field, value):
    identity = plan()["site_identity"]
    identity[field] = value
    with pytest.raises(approval.ProfileApprovalError, match=field):
        approval.profile_site_identity_sha256(identity)


@pytest.mark.parametrize(
    "field,value",
    [("schema_version", 2), ("changes", {}), ("approval_required", "true")],
)
def test_plan_digest_requires_schema_and_field_types(field, value):
    document = plan()
    document[field] = value
    with pytest.raises(approval.ProfileApprovalError):
        approval.profile_plan_sha256(document)


def test_plan_digest_requires_every_identity_field():
    document = plan()
    del document["policy_digest"]
    with pytest.raises(approval.ProfileApprovalError, match="missing identity field"):
        approval.profile_plan_sha256(document)


@pytest.mark.parametrize("reference", ["", "ab", "invalid reference", "x" * 129])
def test_invalid_approval_reference_writes_nothing(tmp_path, reference):
    with pytest.raises(approval.ProfileApprovalError, match="reference"):
        approval.approve_profile(
            tmp_path, reference=reference, expected_plan_sha256="a" * 64
        )
    assert not (tmp_path / "release-deploy").exists(), (
        "invalid reference created approval state"
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", 2),
        ("reference", "invalid reference"),
        ("plan_sha256", "invalid"),
        ("site_identity_sha256", "0" * 64),
        ("approved_at", "invalid"),
        ("approved_at", "2026-09-12T00:00:00"),
        ("approver_identity", None),
        ("approver_identity", " "),
    ],
)
def test_resolve_refuses_corrupt_approval_before_trusting_archive(
    approved, field, value
):
    root, document, record = approved
    record[field] = value
    approval.profile_approval_path(root).write_text(json.dumps(record))
    with pytest.raises(approval.ProfileApprovalError):
        approval.resolve_profile_approval(root, current_plan=document)
    assert approval.profile_approval_path(root).exists(), (
        "corrupt approval refusal removed its evidence"
    )
    assert approval.profile_plan_path(root).exists(), (
        "corrupt approval refusal removed the pending plan"
    )


@pytest.mark.parametrize("payload", ["invalid-json", "[]"])
def test_invalid_plan_files_cannot_be_approved(tmp_path, payload):
    path = approval.profile_plan_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(payload)
    with pytest.raises(approval.ProfileApprovalError, match="invalid|JSON object"):
        approval.approve_profile(
            tmp_path,
            reference="CHG-EXAMPLE",
            expected_plan_sha256="a" * 64,
            approver_identity="example",
        )


def test_plan_not_requiring_approval_cannot_gain_one(tmp_path):
    document = plan(approval_required=False)
    approval.write_profile_plan(tmp_path, document)
    with pytest.raises(approval.ProfileApprovalError, match="does not require"):
        approval.approve_profile(
            tmp_path,
            reference="CHG-EXAMPLE",
            expected_plan_sha256=document["plan_sha256"],
            approver_identity="example",
        )
    assert not approval.profile_approval_path(tmp_path).exists(), (
        "non-approvable plan gained an approval"
    )


def test_approval_lock_reuses_site_mutation_exclusion(tmp_path):
    with site_operation_lock(tmp_path, wait=False):
        with pytest.raises(
            approval.ProfileApprovalError, match="mutation is in progress"
        ):
            with approval.profile_approval_lock(tmp_path):
                pytest.fail("Profile approval bypassed the site mutation lock")


def test_approving_revised_plan_archives_prior_approval_as_superseded(approved):
    root, previous, _record = approved
    changed = plan(policy_digest="a" * 64)
    approval.write_profile_plan(root, changed)
    result = approval.approve_profile(
        root,
        reference="CHG-REVISED",
        expected_plan_sha256=changed["plan_sha256"],
        approver_identity="example-operator",
    )
    prior = approval.profile_approval_archive_path(root, previous["plan_sha256"])
    superseded = json.loads((prior / "superseded.json").read_text())
    assert superseded["status"] == "SUPERSEDED"
    assert superseded["replacement_plan_sha256"] == changed["plan_sha256"]
    assert result["plan_sha256"] == changed["plan_sha256"]


def test_cannot_clear_an_active_approval_plan(approved):
    root = approved[0]
    with pytest.raises(approval.ProfileApprovalError, match="approval is active"):
        approval.clear_profile_plan(root)
    approval.profile_approval_path(root).unlink()
    approval.clear_profile_plan(root)
    approval.clear_profile_plan(root)
    assert not approval.profile_plan_path(root).exists(), "cleared plan remained active"


@pytest.mark.parametrize("kind", ["missing-plan", "mismatched-plan"])
def test_approval_record_requires_its_original_pending_plan(approved, kind):
    root, document, _record = approved
    if kind == "missing-plan":
        approval.profile_plan_path(root).unlink()
    else:
        approval.write_profile_plan(root, plan(policy_digest="a" * 64))
    with pytest.raises(
        approval.ProfileApprovalError, match="without its pending plan|does not match"
    ):
        approval.resolve_profile_approval(root, current_plan=document)


@pytest.mark.parametrize("kind", ["plan", "approval"])
def test_archive_identity_drift_cannot_be_overwritten_by_retry(approved, kind):
    root, document, record = approved
    archive = approval.profile_approval_archive_path(root, document["plan_sha256"])
    original = copy.deepcopy(document if kind == "plan" else record)
    changed = {**original, "extra": "unexpected"}
    path = archive / f"{kind}.json"
    path.write_text(json.dumps(changed))
    with pytest.raises(
        approval.ProfileApprovalError, match=f"archived Profile {kind} differs"
    ):
        approval.approve_profile(
            root,
            reference=record["reference"],
            expected_plan_sha256=document["plan_sha256"],
            approver_identity="example",
        )
    assert json.loads(path.read_text()) == changed


@pytest.mark.parametrize(
    "release,relation", [("../outside", "EXACT"), ("release-a", "UNKNOWN")]
)
def test_consumption_rejects_unbound_release_or_relation(approved, release, relation):
    root, document, _record = approved
    with pytest.raises(approval.ProfileApprovalError, match="invalid"):
        approval.consume_profile_approval(
            root,
            expected_plan_sha256=document["plan_sha256"],
            release_id=release,
            relation=relation,
        )
    assert approval.profile_approval_path(root).exists(), (
        "invalid consumption removed the active approval"
    )


@pytest.mark.parametrize("kind", ["expected-digest", "pending-digest", "audit"])
def test_consumption_rejects_inconsistent_plan_or_audit(approved, kind):
    root, document, _record = approved
    digest = document["plan_sha256"]
    if kind == "expected-digest":
        digest = "0" * 64
    elif kind == "pending-digest":
        approval.write_profile_plan(root, plan(policy_digest="a" * 64))
    else:
        archive = approval.profile_approval_archive_path(root, digest)
        (archive / "consumed.json").write_text(
            json.dumps({"plan_sha256": digest, "release_id": "other-release"})
        )
    with pytest.raises(
        approval.ProfileApprovalError,
        match="different plan|changed before|audit is inconsistent",
    ):
        approval.consume_profile_approval(
            root, expected_plan_sha256=digest, release_id="release-a", relation="EXACT"
        )
    assert approval.profile_approval_path(root).exists(), (
        "inconsistent audit removed the active approval"
    )


def test_consumption_resumes_after_audit_write_before_active_files_were_removed(
    approved,
):
    root, document, record = approved
    archive = approval.consume_profile_approval(
        root,
        expected_plan_sha256=document["plan_sha256"],
        release_id="release-a",
        relation="EXACT",
    )
    result = (archive / "consumed.json").read_bytes()
    approval.write_profile_plan(root, document)
    approval.profile_approval_path(root).write_text(json.dumps(record))
    assert (
        approval.consume_profile_approval(
            root,
            expected_plan_sha256=document["plan_sha256"],
            release_id="release-a",
            relation="EXACT",
        )
        == archive
    )
    assert (archive / "consumed.json").read_bytes() == result
    assert not approval.profile_plan_path(root).exists(), (
        "consumed plan remained active after replay"
    )
    assert not approval.profile_approval_path(root).exists(), (
        "consumed approval remained replayable"
    )


@pytest.mark.parametrize("changed", [False, True])
def test_supersession_resumes_only_for_the_same_replacement(approved, changed):
    root, document, record = approved
    replacement = plan(policy_digest="a" * 64)
    archive = approval.supersede_profile_approval(
        root, replacement_plan=replacement, reason="example"
    )
    prior = (archive / "superseded.json").read_bytes()
    approval.write_profile_plan(root, document)
    approval.profile_approval_path(root).write_text(json.dumps(record))
    if changed:
        with pytest.raises(
            approval.ProfileApprovalError, match="audit is inconsistent"
        ):
            approval.supersede_profile_approval(
                root, replacement_plan=plan(policy_digest="b" * 64), reason="different"
            )
    else:
        assert (
            approval.supersede_profile_approval(
                root, replacement_plan=replacement, reason="retry"
            )
            == archive
        )
    assert (archive / "superseded.json").read_bytes() == prior


def test_supersession_can_clear_an_unapproved_or_active_plan(tmp_path):
    document = plan()
    approval.write_profile_plan(tmp_path, document)
    assert (
        approval.supersede_profile_approval(
            tmp_path, replacement_plan=None, reason="cancelled"
        )
        is None
    )
    assert not approval.profile_plan_path(tmp_path).exists(), (
        "cancelled unapproved plan remained active"
    )
    approval.write_profile_plan(tmp_path, document)
    approval.approve_profile(
        tmp_path,
        reference="CHG-EXAMPLE",
        expected_plan_sha256=document["plan_sha256"],
        approver_identity="example",
    )
    archive = approval.supersede_profile_approval(
        tmp_path, replacement_plan=None, reason="cancelled"
    )
    assert (
        json.loads((archive / "superseded.json").read_text())["replacement_plan_sha256"]
        is None
    )
    assert not approval.profile_plan_path(tmp_path).exists(), (
        "superseded approved plan remained active"
    )


def test_supersession_refuses_a_different_pending_plan(approved):
    root, _document, _record = approved
    approval.write_profile_plan(root, plan(policy_digest="a" * 64))
    with pytest.raises(approval.ProfileApprovalError, match="does not match"):
        approval.supersede_profile_approval(
            root, replacement_plan=None, reason="cancelled"
        )
