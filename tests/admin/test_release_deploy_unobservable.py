"""The release driver's record for a verify that could not observe health.

Live on 2026-10-15 the record of a rolled-back healthy config change said only
``command failed (2)``: the engine had never observed the release, it had failed
to fetch a credential. Exit status 4 now says that health was not observed, and
the transaction still fails closed.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from scripts import release_deploy
from tests.admin.test_release_deploy import _release_diff, _release_summary, _site


def test_unobservable_health_is_named_in_the_failure_and_still_rolls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exit 4: the engine could not observe health for its whole budget.

    The record has to say that health was not observed -- the live 2026-10-15
    record said only `command failed (2)` -- and the transaction still fails
    closed: a release whose health was never shown is not committed, so
    `spec.autoRollback: true` rolls it back as for any verify failure.
    """

    site = _site(tmp_path, monkeypatch)
    monkeypatch.setattr(release_deploy, "_run", lambda *_args, **_kwargs: None)

    def run_driver(arguments, **_kwargs):
        if "verify" in arguments:
            return subprocess.CompletedProcess(
                arguments, release_deploy.HEALTH_UNOBSERVABLE_EXIT_CODE, "{}", None
            )
        if "release-diff" in arguments:
            document = _release_diff("CONTROL_PLANE_ONLY", ["control_plane_wheel"])
        elif "stability" in arguments:
            document = {"mode": "stability", "healthy": True, "window_seconds": 120}
        else:
            document = _release_summary()
        return subprocess.CompletedProcess(arguments, 0, json.dumps(document), None)

    monkeypatch.setattr(release_deploy, "run_driver", run_driver)
    profile = tmp_path / "repo/config/profile.yaml"

    with pytest.raises(
        release_deploy.ReleaseDeployError, match="health could not be observed"
    ):
        release_deploy.execute_release(
            site,
            run_checks=False,
            live_state={
                "runtime_profile_sha256": hashlib.sha256(
                    profile.read_bytes()
                ).hexdigest()
            },
        )

    state = json.loads(
        (site.parent / "release-deploy/release-a/state.json").read_text()
    )
    assert state["phase"] == "FAILED"
    assert "command failed (4)" in state["error"], state["error"]
    assert "release health could not be observed" in state["error"], state["error"]
    assert state["rollback"]["status"] == "PASSED", (
        "an unobserved release must still be rolled back under autoRollback"
    )


def test_a_verify_that_needed_a_retry_commits_without_a_rollback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The live case after the fix: one unobserved attempt, then a healthy report."""

    site = _site(tmp_path, monkeypatch)
    commands: list[list[str]] = []
    monkeypatch.setattr(
        release_deploy,
        "_run",
        lambda arguments, **_kwargs: commands.append(list(arguments)),
    )
    retried = {
        "mode": "verify",
        "site_name": "test-site",
        "healthy": True,
        "summary": {"PASS": 12, "WARN": 0, "FAIL": 0, "SKIP": 0},
        "checks": [{"name": "runtime_profile", "status": "PASS"}],
        "observation": {
            "observed": True,
            "budget_seconds": 120.0,
            "unobserved_seconds": 7.0,
            "retries": 1,
            "reasons": {"no-credentials": 1},
        },
    }

    def run_json(arguments, **_kwargs):
        if "verify" in arguments:
            return retried
        if "stability" in arguments:
            return {"mode": "stability", "healthy": True, "window_seconds": 120}
        if "release-diff" in arguments:
            return _release_diff("CONTROL_PLANE_ONLY", ["control_plane_wheel"])
        return _release_summary()

    monkeypatch.setattr(release_deploy, "_run_json", run_json)
    profile = tmp_path / "repo/config/profile.yaml"

    prepared = release_deploy.execute_release(
        site,
        run_checks=False,
        live_state={
            "release_id": "release-a",
            "runtime_profile_sha256": hashlib.sha256(profile.read_bytes()).hexdigest(),
        },
    )

    state = json.loads((prepared.state_dir / "state.json").read_text())
    assert state["phase"] == "COMPLETED"
    assert state.get("rollback") is None, "a healthy release was rolled back"
    assert ["commit"] in [command[1:2] for command in commands], "never committed"
    assert not any("rollback" in command for command in commands), (
        "the driver ran a rollback for a verify that passed"
    )
    report = json.loads((prepared.state_dir / "verification-report.json").read_text())
    assert report["observation"]["retries"] == 1, "the retry left no evidence"
