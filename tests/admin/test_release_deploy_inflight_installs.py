"""The deploy driver's side of the in-flight install gate (F3).

Moved out of ``test_release_deploy.py`` when that module crossed the 1500-line
architecture cap in the merge; the fixtures and helpers stay there.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from scripts import release_deploy, release_failure_recovery
from tests.admin.test_release_deploy import _release_diff, _release_summary, _site


def test_the_driver_and_the_engine_agree_on_the_refusal_exit_code() -> None:
    from gpu_fault_release import regional_release_store_preflight as STORE

    assert (
        release_failure_recovery.INFLIGHT_INSTALLS_REFUSED_EXIT_CODE
        == STORE.INFLIGHT_INSTALLS_REFUSED_EXIT_CODE
    )
    assert release_failure_recovery.refused_inflight_installs(
        release_deploy.ReleaseDeployError("command failed (3): rollout.sh")
    ), "the engine's refusal exit code is what the driver classifies on"
    assert not release_failure_recovery.refused_inflight_installs(
        release_deploy.ReleaseDeployError("command failed (2): rollout.sh")
    ), "an ordinary engine error stays a rollback failure"


def test_an_engine_refusal_is_recorded_as_refused_not_as_a_failed_rollback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The engine refused before touching anything (an install step is in
    flight), so the driver must not say the rollback failed: nothing was rolled
    back, and the operator's next step is to wait, not to repair a half-restore.
    The automatic rollback is also told it is automatic, so a control plane that
    cannot answer the store read does not wedge it."""

    site = _site(tmp_path, monkeypatch)
    commands: list[list[str]] = []

    def run(arguments, **_kwargs):
        commands.append(list(arguments))
        if "rollback" in arguments:
            raise release_deploy.ReleaseDeployError(
                f"command failed (3): {arguments[0]}"
            )

    monkeypatch.setattr(release_deploy, "_run", run)

    def run_json(arguments, **_kwargs):
        if "verify" in arguments:
            raise release_deploy.ReleaseDeployError("verify failed")
        if "release-diff" in arguments:
            return _release_diff("CONTROL_PLANE_ONLY", ["control_plane_wheel"])
        return _release_summary()

    monkeypatch.setattr(release_deploy, "_run_json", run_json)
    profile = tmp_path / "repo/config/profile.yaml"

    with pytest.raises(release_deploy.ReleaseDeployError) as failure:
        release_deploy.execute_release(
            site,
            run_checks=False,
            live_state={
                "runtime_profile_sha256": hashlib.sha256(
                    profile.read_bytes()
                ).hexdigest()
            },
        )

    assert "verify failed" in str(failure.value)
    assert "rollback also failed" not in str(failure.value)
    rollback = next(command for command in commands if command[1] == "rollback")
    assert "--automatic" in rollback
    state = json.loads(
        (site.parent / "release-deploy/release-a/state.json").read_text()
    )
    assert state["phase"] == "FAILED"
    assert state["rollback"]["status"] == "REFUSED_INFLIGHT_INSTALLS"
    assert "nothing was rolled back" in state["rollback"]["reason"]
    assert "GPU_FAULT_RELEASE_ALLOW_INFLIGHT_INSTALLS" in state["rollback"]["reason"]
    verdict = state["rollback"]["inflight_installs"]
    assert verdict["verdict"] == "refused"
    assert verdict["checked"] is True, "a refusal is the check having run"
    assert "rollback log" in verdict["reason"], "the steps live in the engine log"


def _verify_failure_after_deploy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, site: Path
) -> dict[str, Any]:
    """Run a release whose verify fails and whose engine rollback passes; return
    the driver's persisted state."""

    monkeypatch.setattr(release_deploy, "_run", lambda *_args, **_kwargs: None)

    def run_json(arguments, **_kwargs):
        if "verify" in arguments:
            raise release_deploy.ReleaseDeployError("verify failed")
        if "release-diff" in arguments:
            return _release_diff("CONTROL_PLANE_ONLY", ["control_plane_wheel"])
        return _release_summary()

    monkeypatch.setattr(release_deploy, "_run_json", run_json)
    profile = tmp_path / "repo/config/profile.yaml"
    with pytest.raises(release_deploy.ReleaseDeployError, match="verify failed"):
        release_deploy.execute_release(
            site,
            run_checks=False,
            live_state={
                "runtime_profile_sha256": hashlib.sha256(
                    profile.read_bytes()
                ).hexdigest()
            },
        )
    return json.loads((site.parent / "release-deploy/release-a/state.json").read_text())


def test_the_rollback_record_carries_the_engines_inflight_install_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An automatic rollback that proceeded ``inflight-installs-unchecked`` (no
    Running control-plane Pod could run the probe) used to leave one stderr
    line behind. The engine now writes the verdict into its transaction state
    before ``rollback-started``; the driver copies it into its own rollback
    record, so the release history says whether the rollback was checked and
    on what (fix round 3, MEDIUM-3)."""

    site = _site(tmp_path, monkeypatch)
    engine_verdict = {
        "checked": False,
        "verdict": "unchecked",
        "reason": "automatic rollback: no Running control-plane Pod could run the probe",
        "steps": [],
    }
    monkeypatch.setattr(
        release_deploy,
        "read_live_release_state",
        lambda _site: {
            "phase": "rolled-back",
            "rollback_result": {"status": "PASSED"},
            "previous": {},
            "inflight_installs": dict(engine_verdict),
        },
    )

    state = _verify_failure_after_deploy(monkeypatch, tmp_path, site)

    assert state["rollback"]["status"] == "PASSED"
    assert state["rollback"]["inflight_installs"] == engine_verdict


def test_a_rollback_record_says_when_the_engine_left_no_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An engine that predates the verdict leaves no ``inflight_installs`` in
    its state; the record must say so rather than imply the check passed. It
    must not blame a rollback re-entered after ``rollback-cpu-restored``: that
    re-entry skips the gate but the FIRST attempt's verdict stays in the state
    (with its ``checked_at``), so the state is not empty there (fix round 4,
    LOW-4)."""

    site = _site(tmp_path, monkeypatch)

    state = _verify_failure_after_deploy(monkeypatch, tmp_path, site)

    verdict = state["rollback"]["inflight_installs"]
    assert verdict["checked"] is False
    assert verdict["verdict"] == "unrecorded"
    assert "inflight_installs" in verdict["reason"], (
        "the reason names the missing engine-state key"
    )
    assert "predates" in verdict["reason"], "the one cause that leaves no verdict"
    assert "re-enter" not in verdict["reason"], (
        "a re-entry keeps the first attempt's verdict; it is not unrecorded"
    )
