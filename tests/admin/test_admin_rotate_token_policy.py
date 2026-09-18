"""Rotation retries keep their recorded security policy, not their direction."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest

from gpu_fault.admin import rotate_token as module
from gpu_fault.admin.bootstrap_common import BootstrapError
from tests.admin.test_admin_rotate_token import Harness


def _pause(
    harness: Harness, monkeypatch: pytest.MonkeyPatch, *, after_commit: bool
) -> None:
    if after_commit:

        def fail_rewrite(*args, **kwargs):
            raise BootstrapError("paused after token file commit")

        monkeypatch.setattr(module, "rewrite_registry_secret_token", fail_rewrite)
    else:
        harness.acceptance_error = BootstrapError("paused before token file commit")
    with pytest.raises(BootstrapError, match="paused"):
        harness.rotate()
    assert (
        module.step_done(harness.state, module.STEP_TOKEN_FILE_WRITTEN) is after_commit
    )


@pytest.mark.parametrize("after_commit", [False, True])
@pytest.mark.parametrize(
    ("changed", "field"),
    [
        ({"keep_window": True}, "keep_window"),
        ({"window": timedelta(minutes=45)}, "window_minutes"),
        ({"quiet_seconds": 120}, "quiet_seconds"),
        ({"acceptance_timeout_seconds": 900}, "acceptance_timeout_seconds"),
    ],
)
def test_resume_refuses_changed_policy_before_any_new_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    after_commit: bool,
    changed: dict,
    field: str,
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    _pause(harness, monkeypatch, after_commit=after_commit)
    before = harness.state
    calls = list(harness.calls)

    with pytest.raises(BootstrapError, match=f"resume policy conflicts on {field}"):
        harness.rotate(**changed)
    assert harness.state == before
    assert harness.calls == calls


@pytest.mark.parametrize(
    "field",
    ["keep_window", "window_minutes", "quiet_seconds", "acceptance_timeout_seconds"],
)
@pytest.mark.parametrize("unbound", ["missing", "wrong-type"])
def test_resume_refuses_unbound_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, unbound: str
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    _pause(harness, monkeypatch, after_commit=False)
    state = harness.state
    if unbound == "missing":
        del state[field]
    else:
        state[field] = 0 if field == "keep_window" else float(state[field])
    path = module.rotation_state_path(harness.site, "gpu-a")
    path.write_text(json.dumps(state), encoding="utf-8")
    calls = list(harness.calls)

    with pytest.raises(BootstrapError, match=f"resume policy.*{field}"):
        harness.rotate()
    assert harness.calls == calls
    assert harness.state == state


def test_rollback_direction_is_not_pinned_before_the_write_intent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    harness.acceptance_error = BootstrapError("paused")
    with pytest.raises(BootstrapError, match="paused"):
        harness.rotate(keep_window=True)
    assert harness.state["keep_window"] is True
    assert "rollback" not in harness.state

    result = harness.rotate(keep_window=True, rollback=True)

    assert result["status"] == module.STATUS_ROLLED_BACK


@pytest.mark.parametrize("rollback", [False, True])
def test_keep_window_cannot_be_dropped_on_resume_or_rollback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rollback: bool
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    harness.acceptance_error = BootstrapError("paused")
    with pytest.raises(BootstrapError, match="paused"):
        harness.rotate(keep_window=True)
    before = harness.state
    calls = list(harness.calls)

    with pytest.raises(BootstrapError, match="resume policy conflicts on keep_window"):
        harness.rotate(rollback=rollback)
    assert harness.state == before
    assert harness.calls == calls


@pytest.mark.parametrize(
    "invalid",
    [
        {"keep_window": 1},
        {"keep_window": "false"},
        {"quiet_seconds": True},
        {"quiet_seconds": 1.0},
        {"acceptance_timeout_seconds": True},
        {"acceptance_timeout_seconds": 1200.0},
    ],
)
def test_policy_types_cannot_use_boolean_or_numeric_coercion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid: dict
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    with pytest.raises(BootstrapError, match="policy|quiet"):
        harness.rotate(**invalid)
    assert harness.calls == []
    assert not module.rotation_state_path(harness.site, "gpu-a").exists(), (
        "invalid policy types must not create rotation state"
    )


def test_a_new_rotation_can_choose_a_new_policy_after_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    first = harness.rotate()
    result = harness.rotate(keep_window=True, window=timedelta(minutes=45))
    assert result["rotation_id"] != first["rotation_id"]
    assert harness.state["keep_window"] is True
    assert harness.state["window_minutes"] == 45


def test_fractional_window_minutes_cannot_evade_the_policy_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    with pytest.raises(BootstrapError, match="whole minutes"):
        harness.request(window=timedelta(minutes=30, seconds=1))


def test_cleanup_only_resume_also_requires_the_recorded_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = Harness(tmp_path, monkeypatch)
    unlink = Path.unlink

    def fail_pending(path, *args, **kwargs):
        if path.name == module.PENDING_TOKEN_FILE:
            raise OSError("pending cleanup paused")
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_pending)
    with pytest.raises(OSError, match="cleanup paused"):
        harness.rotate()
    before = harness.state
    calls = list(harness.calls)

    with pytest.raises(BootstrapError, match="resume policy conflicts on keep_window"):
        harness.rotate(keep_window=True)
    assert harness.state == before
    assert harness.calls == calls

    monkeypatch.setattr(Path, "unlink", unlink)
    assert harness.rotate()["rotation_id"] == before["rotation_id"]
    assert harness.calls == calls
