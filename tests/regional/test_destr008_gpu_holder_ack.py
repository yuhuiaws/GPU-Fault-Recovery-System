from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import destr008_gpu_holder as holder
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from tests.regional._destr008_gpu_holder import (
    HolderHarness,
    admitted_pod,
    build_holder,
)
from tests.regional.test_destr008_gpu_holder import set_field


@pytest.fixture
def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> HolderHarness:
    return build_holder(tmp_path, monkeypatch)


def unapproved(harness: HolderHarness, path: tuple[str | int, ...], value: Any) -> None:
    def change(pod: dict[str, Any]) -> None:
        set_field(pod, path, value)
        harness.pod = copy.deepcopy(pod)

    harness.ack_change = change
    with pytest.raises(RegionalFixtureError):
        harness.controller.create()
    record = harness.journal()
    assert record["pod_uid"] == "holder-uid", (
        "a direct ACK must retain deletion-only custody"
    )
    assert record["phase"] == "CREATED" and record["pod_approved"] is False, (
        "admission mutation must never be approved for execution"
    )
    assert harness.pod is not None, "the local API committed the unapproved Pod"
    assert record["pod_ack_sha256"] == holder.fingerprint(harness.pod), (
        "the fingerprint must come from the direct response before approval"
    )


def test_direct_ack_is_durable_before_semantic_approval_starts(
    harness: HolderHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    validate = harness.controller.validate
    checks = []

    def observe(pod: dict[str, Any], *, acknowledged: bool) -> tuple[str, str]:
        assert not acknowledged, "the first semantic check must inspect the direct ACK"
        record = harness.journal()
        assert record["pod_uid"] == pod["metadata"]["uid"], (
            "UID custody must already be durable"
        )
        assert record["pod_ack_sha256"] == holder.fingerprint(pod), (
            "the direct-ACK fingerprint must be durable before approval"
        )
        assert record["pod_spec_sha256"] == holder.digest(pod["spec"]), (
            "the full admitted spec must already be pinned"
        )
        assert record["phase"] == "CREATED" and record["pod_approved"] is False, (
            "only deletion authority exists at the semantic approval boundary"
        )
        checks.append(record)
        return validate(pod, acknowledged=acknowledged)

    monkeypatch.setattr(harness.controller, "validate", observe)
    unapproved(harness, ("spec", "containers", 0, "image"), "unapproved")
    assert len(checks) == 1, "approval must inspect the original ACK exactly once"
    assert len([args for args in harness.calls if args[:2] == ("get", "pod")]) == 1, (
        "the only Pod read before approval is the pre-CREATE absence check"
    )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("spec", "containers", 0, "image"), "unapproved"),
        (("spec", "containers", 0, "securityContext"), {"privileged": True}),
        (("spec", "containers"), []),
        (("spec",), []),
        (("spec", "nodeName"), "other-node"),
        (("metadata", "labels"), {}),
        (("metadata", "labels"), ["invalid"]),
        (("metadata", "ownerReferences"), [{"uid": "unexpected-owner"}]),
        (("metadata", "deletionTimestamp"), "already-deleting"),
    ],
)
def test_direct_ack_pins_unapproved_object_for_deletion_only(
    harness: HolderHarness, path: tuple[str | int, ...], value: Any
) -> None:
    unapproved(harness, path, value)
    deadline = harness.journal()["deadline_at"]
    fresh = harness.new_holder()
    with pytest.raises(RegionalFixtureError, match="cleanup-only"):
        fresh.create()
    assert fresh.resume_cleanup() is False, (
        "the directly acknowledged Pod can be removed"
    )
    record = harness.journal()
    assert record["phase"] == "CLOSED" and not record["pod_approved"], (
        "cleanup must not retroactively approve a rejected creation"
    )
    assert record["deadline_at"] == deadline, "deletion-only recovery never rearms"
    assert [args[0] for args in harness.mutations()] == ["create", "delete"], (
        "only the original CREATE and a UID/RV DELETE are allowed"
    )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("metadata", "uid"), "replacement"),
        (("metadata", "labels"), {"same": "public"}),
        (("metadata", "annotations"), {"changed": "true"}),
        (("metadata", "ownerReferences"), [{"uid": "later-owner"}]),
        (("spec", "nodeName"), "another-node"),
        (("spec", "containers", 0, "image"), "third-image"),
    ],
)
def test_deletion_only_ack_fingerprint_cannot_be_rebased_from_readback(
    harness: HolderHarness, path: tuple[str | int, ...], value: Any
) -> None:
    unapproved(harness, ("spec", "containers", 0, "image"), "unapproved")
    assert harness.pod is not None, "this control needs a directly acknowledged Pod"
    set_field(harness.pod, path, value)
    with pytest.raises(RegionalFixtureError, match="replaced|acknowledged spec"):
        harness.new_holder().resume_cleanup()
    assert len(harness.mutations()) == 1, "changed unapproved objects cannot be deleted"


def test_deletion_only_ack_allows_current_rv_and_status_not_new_identity(
    harness: HolderHarness,
) -> None:
    unapproved(harness, ("metadata", "labels"), {})
    assert harness.pod is not None, "this control needs the unapproved Pod"
    harness.pod["metadata"]["resourceVersion"] = "9"
    harness.pod["status"] = {"phase": "Failed"}
    harness.new_holder().cleanup()
    assert harness.pod is None, "the raw DELETE must carry the new verified RV"
    assert not harness.journal()["pod_approved"], "terminal state cannot grant approval"


@pytest.mark.parametrize("committed", [False, True])
def test_deletion_only_unknown_delete_ack_keeps_create_forbidden(
    harness: HolderHarness, committed: bool
) -> None:
    unapproved(harness, ("metadata", "labels"), {})
    if committed:
        harness.lost_delete = True
        harness.new_holder().resume_cleanup()
        assert harness.journal()["phase"] == "CLOSED", (
            "known custody plus absence resolves DELETE"
        )
    else:
        harness.delete_error = TimeoutError("local fake")
        with pytest.raises(
            RegionalFixtureError, match="deletion-only request is unconfirmed"
        ):
            harness.new_holder().resume_cleanup()
        assert harness.journal()["phase"] == "CLEANING", (
            "unconfirmed DELETE remains unresolved"
        )
        harness.delete_error = None
        harness.new_holder().resume_cleanup()
    with pytest.raises(RegionalFixtureError, match="cleanup-only"):
        harness.new_holder().create()
    assert harness.pod is None, "a resumed DELETE must prove absence"


@pytest.mark.parametrize("eventually_removed", [True, False])
def test_deletion_only_termination_wait_is_bounded_and_not_a_timer_proof(
    harness: HolderHarness, eventually_removed: bool
) -> None:
    unapproved(harness, ("metadata", "labels"), {})

    def deleting(args: tuple[str, ...]) -> None:
        if args[0] == "delete" and harness.pod is not None:
            harness.pod["metadata"]["deletionTimestamp"] = "now"
        if (
            args[:2] == ("get", "pod")
            and eventually_removed
            and harness.clock.elapsed >= 2
        ):
            harness.pod = None

    harness.before = deleting
    harness.delete_error = TimeoutError("local fake: deleting after lost ACK")
    if eventually_removed:
        harness.new_holder().resume_cleanup()
        assert harness.clock.elapsed == 2, "cleanup waits for actual absence"
        assert harness.journal()["phase"] == "CLOSED", (
            "absence finishes deletion-only cleanup"
        )
    else:
        with pytest.raises(RegionalFixtureError, match="removal timed out"):
            harness.new_holder().resume_cleanup()
        assert harness.clock.elapsed == 120, "termination observation must be bounded"
        assert harness.journal()["phase"] == "CLEANING", "timeout is not removal"
        assert harness.pod is not None, "a still-present Pod remains unresolved"


def test_deletion_only_replacement_during_delete_is_not_adopted(
    harness: HolderHarness,
) -> None:
    unapproved(harness, ("metadata", "labels"), {})

    def replacement(args: tuple[str, ...]) -> None:
        if args[:2] == ("get", "pod") and harness.pod is None:
            harness.pod = admitted_pod(harness.controller.manifest())
            harness.pod["metadata"]["uid"] = "replacement"

    harness.before = replacement
    with pytest.raises(RegionalFixtureError, match="replaced"):
        harness.new_holder().resume_cleanup()
    assert harness.journal()["phase"] == "CLEANING", "replacement is not absence"
    assert len(harness.mutations()) == 2, (
        "no follow-up delete may target the replacement"
    )


def test_live_evidence_may_add_fields_but_required_scope_still_must_match(
    harness: HolderHarness,
) -> None:
    harness.identity["region"] = harness.regional.settings.region
    harness.controller.create()
    harness.identity["release_id"] = "changed"
    with pytest.raises(RegionalFixtureError, match="release or cluster"):
        harness.new_holder().cleanup()
    assert len(harness.mutations()) == 1, "extra evidence must not hide release drift"


@pytest.mark.parametrize("identity", [{}, {"release_id": "release-a"}, [], None])
def test_incomplete_or_unknown_live_evidence_is_not_a_valid_binding(
    harness: HolderHarness, monkeypatch: pytest.MonkeyPatch, identity: Any
) -> None:
    monkeypatch.setattr(harness.regional, "evidence_identity", lambda: identity)
    with pytest.raises(RegionalFixtureError, match="release or cluster"):
        harness.controller.create()
    assert not harness.mutations(), "missing binding fields cannot authorize CREATE"
