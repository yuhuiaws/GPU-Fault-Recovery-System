"""The installed-resource registry is written after the commit, as a record.

The admin command used to synchronize it inside its deploy branch, right after
`rollout deploy` returned 0, so any failure there (three AWS calls, a kubectl
exec, a Postgres fallback) exited 2 and the release driver marked a fully applied
release FAILED. These cases pin the new contract: the sync runs after the
commit, its failure is a completion warning, and the NOOP fast path skips it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from gpu_fault.admin.bootstrap_common import BootstrapError
from scripts import release_deploy, release_deploy_evidence
from tests.admin.test_release_deploy import (
    _release_diff,
    _release_summary,
    _site,
    _stability_report,
    _verification_report,
)


def _non_noop_run_json(arguments, **_kwargs):
    if "verify" in arguments:
        return _verification_report()
    if "stability" in arguments:
        return _stability_report()
    if "release-diff" in arguments:
        return _release_diff("CONTROL_PLANE_ONLY", ["control_plane_wheel"])
    return _release_summary()


def test_registry_sync_failure_after_a_green_rollout_is_a_completion_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The installed-resource registry is a record of the release, not a gate.

    The admin command used to synchronize the registry right after `rollout
    deploy` returned 0, inside the same process: three AWS calls, a kubectl exec
    and a Postgres fallback. Any one of them failing exited 2, and the driver
    read that as the deploy failing -- the release was marked FAILED with every
    component applied, recovery had nothing to roll back, and the administrator
    re-ran the eight-minute preamble to reach the pending commit. The sync now
    runs after the commit, and a failure there is a completion warning.
    """

    site = _site(tmp_path, monkeypatch)
    monkeypatch.setattr(release_deploy, "_run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(release_deploy, "_run_json", _non_noop_run_json)
    synced_sites: list[str] = []

    def failing_sync(rendered):
        synced_sites.append(rendered.metadata_name)
        raise BootstrapError("aws sts get-caller-identity timed out")

    monkeypatch.setattr(
        release_deploy_evidence, "sync_installation_resource_registry", failing_sync
    )
    profile = tmp_path / "repo/config/profile.yaml"

    prepared = release_deploy.execute_release(
        site,
        run_checks=False,
        live_state={
            "runtime_profile_sha256": hashlib.sha256(profile.read_bytes()).hexdigest()
        },
    )

    state = json.loads((prepared.state_dir / "state.json").read_text())
    assert synced_sites == ["test-site"]
    assert state["phase"] == "COMPLETED", "a registry failure must not fail the release"
    assert "rollback" not in state
    assert state["installation_registry"]["status"] == "UNAVAILABLE"
    assert (
        "aws sts get-caller-identity timed out"
        in (state["installation_registry"]["error"])
    )
    assert state["release_summary"]["status"] == "AVAILABLE_WITH_WARNINGS"
    assert any(
        "installation resource registry" in warning
        for warning in state["completion_warnings"]
    ), state["completion_warnings"]


def test_registry_sync_runs_after_commit_and_is_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A successful sync is part of the completion record, after the commit.

    The order matters: a sync that ran before the commit would advertise the
    new release's resources while the transaction could still be rolled back.
    """

    site = _site(tmp_path, monkeypatch)
    order: list[str] = []

    def run(arguments, **_kwargs):
        if "commit" in list(arguments):
            order.append("commit")

    def sync(rendered):
        order.append("sync")
        return tmp_path / "installation-resources.json"

    monkeypatch.setattr(release_deploy, "_run", run)
    monkeypatch.setattr(release_deploy, "_run_json", _non_noop_run_json)
    monkeypatch.setattr(
        release_deploy_evidence, "sync_installation_resource_registry", sync
    )
    profile = tmp_path / "repo/config/profile.yaml"

    prepared = release_deploy.execute_release(
        site,
        run_checks=False,
        live_state={
            "runtime_profile_sha256": hashlib.sha256(profile.read_bytes()).hexdigest()
        },
    )

    state = json.loads((prepared.state_dir / "state.json").read_text())
    assert order == ["commit", "sync"]
    assert state["phase"] == "COMPLETED"
    assert state["installation_registry"] == {
        "status": "SYNCED",
        "path": str(tmp_path / "installation-resources.json"),
        "synced_at": state["installation_registry"]["synced_at"],
    }
    assert state["release_summary"]["status"] == "AVAILABLE"
    assert state["completion_warnings"] == []


def test_noop_fast_path_does_not_synchronize_the_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing was installed, so there is nothing new to record.

    The fast path has already proved the live state equals the desired state;
    discovering resources again would only add AWS calls to a verification.
    """

    site = _site(tmp_path, monkeypatch)
    monkeypatch.setattr(release_deploy, "_run", lambda *_args, **_kwargs: None)

    def run_json(arguments, **_kwargs):
        if "verify" in arguments:
            return _verification_report()
        if "release-diff" in arguments:
            return _release_diff()
        return _release_summary()

    def sync(_rendered):
        raise AssertionError("the fast path must not discover resources")

    monkeypatch.setattr(release_deploy, "_run_json", run_json)
    monkeypatch.setattr(
        release_deploy_evidence, "sync_installation_resource_registry", sync
    )
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
    assert state["deployment"]["fast_path"] is True
    assert state["installation_registry"]["status"] == "SKIPPED"
