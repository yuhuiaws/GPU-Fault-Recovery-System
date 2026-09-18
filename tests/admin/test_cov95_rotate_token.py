from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest

from gpu_fault.admin import rotate_token as rotation
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault_release.regional_release_config import ReleaseError
from tests.admin.test_admin_rotate_token import OLD_DIGEST, OLD_TOKEN, Harness


@pytest.fixture
def harness(tmp_path, monkeypatch):
    return Harness(tmp_path, monkeypatch)


def pause(harness):
    harness.acceptance_error = BootstrapError("example acceptance interruption")
    with pytest.raises(BootstrapError, match="acceptance interruption"):
        harness.rotate()
    harness.acceptance_error = None
    return harness.state


@pytest.mark.parametrize("kind", ["missing", "short", "unreadable"])
def test_rotation_cannot_read_unknown_current_credential(harness, monkeypatch, kind):
    if kind == "missing":
        harness.token_file.unlink()
    elif kind == "short":
        harness.token_file.write_text("short")
    else:
        read_text = Path.read_text

        def read(path, *args, **kwargs):
            if path == harness.token_file:
                raise PermissionError("example read permission failure")
            return read_text(path, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", read)
    with pytest.raises(BootstrapError, match="missing|shorter|cannot read"):
        harness.rotate()
    assert harness.calls == []
    assert harness.publishes == []


def test_pending_credential_must_differ_from_current(harness):
    pending = (
        rotation.rotation_state_path(harness.site, "gpu-a").parent
        / rotation.PENDING_TOKEN_FILE
    )
    pending.parent.mkdir(parents=True)
    pending.write_text(OLD_TOKEN)
    pending.chmod(0o600)
    with pytest.raises(BootstrapError, match="equals the current"):
        harness.rotate()
    assert not harness.publishes, "identical pending token reached registry publication"
    assert not harness.calls, "identical pending token reached data-plane mutation"


@pytest.mark.parametrize(
    "kind", ["missing-pending", "changed-pending", "changed-current"]
)
def test_rotation_resume_requires_original_current_and_pending_digests(harness, kind):
    state = pause(harness)
    pending = Path(state["pending_token_file"])
    before = list(harness.calls)
    if kind == "missing-pending":
        pending.unlink()
    elif kind == "changed-pending":
        pending.write_text("d" * 64)
    else:
        harness.token_file.write_text("e" * 64)
    with pytest.raises(BootstrapError, match="missing|digest|changed under"):
        harness.rotate()
    assert harness.calls == before
    assert harness.state["status"] == rotation.STATUS_IN_PROGRESS


@pytest.mark.parametrize(
    "field,value",
    [
        ("site_id", "other"),
        ("cluster_id", "other"),
        ("token_file", "/example/other"),
        ("schema_version", 99),
        ("status", "UNKNOWN"),
    ],
)
def test_rotation_resume_rejects_journal_identity_drift_before_commands(
    harness, field, value
):
    state = pause(harness)
    state[field] = value
    rotation.rotation_state_path(harness.site, "gpu-a").write_text(json.dumps(state))
    before = list(harness.calls)
    with pytest.raises(BootstrapError, match="conflicts|unknown schema or status"):
        harness.rotate()
    assert harness.calls == before


def test_rotation_resume_extends_expired_overlap_without_rerolling_completed_nodes(
    harness,
):
    state = pause(harness)
    rolled = sum(call[0] == "roll_node_tokens" for call in harness.calls)
    harness.clock += timedelta(hours=1)
    result = harness.rotate()
    assert result["status"] == rotation.STATUS_COMPLETED
    assert result["expires_at"] != state["expires_at"]
    assert any("window extended" in warning for warning in result["warnings"]), (
        "renewed overlap was omitted from the rotation audit"
    )
    assert sum(call[0] == "roll_node_tokens" for call in harness.calls) == rolled
    assert any("window extended" in value["reason"] for value in harness.publishes), (
        "expired overlap was not republished before resume"
    )


def test_failed_window_extension_does_not_drop_overlap_or_advance_rotation(harness):
    state = pause(harness)
    harness.clock += timedelta(hours=1)
    harness.publish_errors = [ReleaseError("example extension failure")]
    before = list(harness.calls)
    with pytest.raises(BootstrapError, match="window extension failed"):
        harness.rotate()
    assert harness.state["expires_at"] == state["expires_at"]
    assert harness.calls == before


def test_failed_overlap_and_compensation_are_both_reported(harness):
    harness.publish_errors = [
        ReleaseError("example publish failure"),
        ReleaseError("example restoration failure"),
    ]
    with pytest.raises(BootstrapError, match="old-only republish also failed"):
        harness.rotate()
    assert harness.calls == []
    assert rotation.STEP_OVERLAP_PUBLISHED not in harness.state["steps"]


def test_rotation_rollback_refuses_changed_original_credential(harness):
    pause(harness)
    harness.token_file.write_text("e" * 64)
    before = list(harness.calls)
    with pytest.raises(BootstrapError, match="rollback is refused"):
        harness.rotate(rollback=True)
    assert harness.calls == before


@pytest.mark.parametrize("finished", [False, True])
def test_rotation_rollback_requires_an_inflight_transaction(harness, finished):
    if finished:
        harness.rotate()
    before = list(harness.calls)
    with pytest.raises(BootstrapError, match="no rotation is in progress"):
        harness.rotate(rollback=True)
    assert harness.calls == before


def test_cleanup_only_resume_cannot_reverse_committed_direction(harness):
    harness.rotate()
    state = harness.state
    state["pending_token_cleanup_completed"] = False
    rotation.rotation_state_path(harness.site, "gpu-a").write_text(json.dumps(state))
    before = list(harness.calls)
    with pytest.raises(BootstrapError, match="committed direction"):
        harness.rotate(rollback=True)
    assert harness.calls == before


def test_terminal_token_digest_is_revalidated_before_fail_forward(harness, monkeypatch):
    original = rotation.publish_registry_revision

    def publish(*args, **kwargs):
        if kwargs["payload"]["reason"].endswith(" final"):
            raise ReleaseError("example final publish failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(rotation, "publish_registry_revision", publish)
    with pytest.raises(ReleaseError, match="final publish failure"):
        harness.rotate()
    assert rotation.STEP_TOKEN_FILE_WRITTEN in harness.state["steps"]
    harness.token_file.write_text("d" * 64)
    before = list(harness.calls)
    with pytest.raises(BootstrapError, match="committed token file"):
        harness.rotate()
    assert harness.calls == before


@pytest.mark.parametrize("kind", ["missing", "retiring"])
def test_overlap_revision_requires_one_current_nonretiring_target(harness, kind):
    if kind == "missing":
        harness.secret = [
            entry for entry in harness.secret if entry["cluster_id"] != "gpu-a"
        ]
    else:
        harness.secret[0]["retiring_token_sha256"] = "b" * 64
    with pytest.raises(BootstrapError, match="not in|already carries"):
        rotation.publish_overlap_revision(
            harness.release,
            "gpu-a",
            new_digest="c" * 64,
            old_digest=OLD_DIGEST,
            expires_at=harness.clock + timedelta(minutes=30),
            reason="example",
        )
    assert harness.publishes == []


def test_overlap_publish_requires_ack_from_every_control_plane_member(
    harness, monkeypatch
):
    calls = []

    def publish(_release, **options):
        calls.append(options)
        return {"missing_member_ids": ["example-member"]}

    monkeypatch.setattr(rotation, "publish_registry_revision", publish)
    with pytest.raises(ReleaseError, match="every control-plane member"):
        rotation.publish_overlap_revision(
            harness.release,
            "gpu-a",
            new_digest="c" * 64,
            old_digest=OLD_DIGEST,
            expires_at=harness.clock + timedelta(minutes=30),
            reason="example",
        )
    assert len(calls) == 1
    assert calls[0]["use_current_generation"] is True
    assert (
        calls[0]["payload"]["registrations"][0]["retiring_token_sha256"] == OLD_DIGEST
    )


def test_registry_secret_rewrite_refuses_missing_target(harness):
    with pytest.raises(BootstrapError, match="not in"):
        rotation.rewrite_registry_secret_token(
            harness.release, "unknown", "example-placeholder"
        )
    assert harness.calls == []


@pytest.mark.parametrize("value", [None, [], "invalid"])
def test_rotation_state_must_be_a_json_object(tmp_path, value):
    path = tmp_path / "state.json"
    assert rotation.load_rotation_state(path) is None
    path.write_text(json.dumps(value))
    with pytest.raises(BootstrapError, match="not an object"):
        rotation.load_rotation_state(path)


def test_unknown_cluster_cannot_create_rotation_state(harness):
    with pytest.raises(BootstrapError, match="unknown cluster_id"):
        harness.rotate(cluster_id="unknown")
    assert harness.calls == []
