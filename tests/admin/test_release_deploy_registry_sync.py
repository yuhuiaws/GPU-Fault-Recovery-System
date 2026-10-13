"""The installed-resource registry is written after the commit, as a record.

The admin command used to synchronize it inside its deploy branch, right after
`rollout deploy` returned 0, so any failure there (three AWS calls, a kubectl
exec, a Postgres fallback) exited 2 and the release driver marked a fully applied
release FAILED. These cases pin the new contract: the sync runs after the
commit, its failure is a completion warning, and the NOOP fast path skips it
while the deploy host's record is a success.

A warning alone went unread for nine days, so every outcome is also kept in
``<state-dir>/release-deploy/installation-registry-sync.json`` (success with its
digest, failure with its reason, both timed): ``status`` reports it and the
deletion verbs refuse while it is not a success. A NOOP deploy must not erase
it, and must retry the sync while it is not a success -- the remedy those
refusals name.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.admin import resource_registry_outcome as outcome
from gpu_fault.admin.bootstrap_common import BootstrapError
from scripts import release_deploy, release_deploy_evidence
from tests.admin._registry_sync_support import record_registry_sync
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


def _noop_run_json(arguments, **_kwargs):
    if "verify" in arguments:
        return _verification_report()
    if "release-diff" in arguments:
        return _release_diff()
    return _release_summary()


def _execute(site: Path, tmp_path: Path, *, live_release_id: str | None = None):
    profile = tmp_path / "repo/config/profile.yaml"
    live: dict[str, Any] = {
        "runtime_profile_sha256": hashlib.sha256(profile.read_bytes()).hexdigest()
    }
    if live_release_id is not None:
        live["release_id"] = live_release_id
    return release_deploy.execute_release(site, run_checks=False, live_state=live)


def _durable(tmp_path: Path) -> dict[str, Any]:
    record = outcome.read_record(tmp_path)
    assert record is not None, "the completion step must leave a durable record"
    return record


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
    runs after the commit, and a failure there is a completion warning -- and,
    durably, the deploy host's record of the failure with its time and reason.
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

    prepared = _execute(site, tmp_path)

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
    durable = _durable(tmp_path)
    assert durable == {
        **state["installation_registry"],
        "schema_version": 1,
        "release_id": prepared.release_id,
        "noop_release_ids": [],
    }, "the durable record is the per-attempt outcome, bound to its release"
    assert (outcome.record_path(tmp_path).stat().st_mode & 0o777) == 0o600


def test_registry_sync_runs_after_commit_and_is_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A successful sync is part of the completion record, after the commit.

    The order matters: a sync that ran before the commit would advertise the
    new release's resources while the transaction could still be rolled back.
    The record carries the digest of the snapshot the sync wrote.
    """

    site = _site(tmp_path, monkeypatch)
    order: list[str] = []
    snapshot = tmp_path / "installation-resources.json"

    def run(arguments, **_kwargs):
        if "commit" in list(arguments):
            order.append("commit")

    def sync(rendered):
        order.append("sync")
        snapshot.write_text('{"resources": ["synced"]}\n', encoding="utf-8")
        return snapshot

    monkeypatch.setattr(release_deploy, "_run", run)
    monkeypatch.setattr(release_deploy, "_run_json", _non_noop_run_json)
    monkeypatch.setattr(
        release_deploy_evidence, "sync_installation_resource_registry", sync
    )

    prepared = _execute(site, tmp_path)

    state = json.loads((prepared.state_dir / "state.json").read_text())
    assert order == ["commit", "sync"]
    assert state["phase"] == "COMPLETED"
    digest = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    assert state["installation_registry"] == {
        "status": "SYNCED",
        "path": str(snapshot),
        "sha256": digest,
        "synced_at": state["installation_registry"]["synced_at"],
    }
    assert state["release_summary"]["status"] == "AVAILABLE"
    assert state["completion_warnings"] == []
    durable = _durable(tmp_path)
    assert (durable["status"], durable["sha256"], durable["release_id"]) == (
        "SYNCED",
        digest,
        prepared.release_id,
    )
    assert durable["synced_at"] == state["installation_registry"]["synced_at"]


def test_noop_fast_path_does_not_synchronize_the_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing was installed, so there is nothing new to record.

    The fast path has already proved the live state equals the desired state;
    discovering resources again would only add AWS calls to a verification.
    The last outcome stays exactly as it was: a NOOP must not erase it.
    """

    site = _site(tmp_path, monkeypatch)
    monkeypatch.setattr(release_deploy, "_run", lambda *_args, **_kwargs: None)

    def sync(_rendered):
        raise AssertionError("the fast path must not discover resources")

    monkeypatch.setattr(release_deploy, "_run_json", _noop_run_json)
    monkeypatch.setattr(
        release_deploy_evidence, "sync_installation_resource_registry", sync
    )
    path = record_registry_sync(tmp_path, release_id="release-a")
    before = path.read_bytes()

    prepared = _execute(site, tmp_path, live_release_id="release-a")

    state = json.loads((prepared.state_dir / "state.json").read_text())
    assert state["phase"] == "COMPLETED"
    assert state["deployment"]["fast_path"] is True
    assert state["installation_registry"]["status"] == "SKIPPED"
    assert state["installation_registry"]["last_sync"] == {
        "status": "SYNCED",
        "release_id": "release-a",
        "synced_at": "2026-01-01T00:00:00Z",
    }
    assert path.read_bytes() == before, "a NOOP must leave the last outcome untouched"
    assert state["completion_warnings"] == []


def test_a_noop_that_stages_a_new_release_id_keeps_the_record_for_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A NOOP may stage a new release id over the live one without installing.

    The record of the live release's sync still describes the site, so it is
    kept and only learns the staged id -- otherwise ``status`` would call the
    ordinary NOOP deploy "no sync outcome since the live release".
    """

    site = _site(tmp_path, monkeypatch)
    monkeypatch.setattr(release_deploy, "_run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(release_deploy, "_run_json", _noop_run_json)
    monkeypatch.setattr(
        release_deploy_evidence,
        "sync_installation_resource_registry",
        lambda _site: pytest.fail("the staged NOOP must not discover resources"),
    )
    record_registry_sync(tmp_path, release_id="release-previous")

    prepared = _execute(site, tmp_path, live_release_id="release-previous")

    durable = _durable(tmp_path)
    assert prepared.release_id == "release-a"
    assert (durable["release_id"], durable["noop_release_ids"]) == (
        "release-previous",
        ["release-a"],
    )
    assert durable["synced_at"] == "2026-01-01T00:00:00Z", "the outcome is unchanged"
    view = outcome.status_view(tmp_path, release_id="release-a")
    assert view["finding"] is None, view


@pytest.mark.parametrize("previous", ["missing", "failed", "unreadable", "foreign"])
def test_a_noop_deploy_retries_a_sync_whose_record_is_no_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, previous: str
) -> None:
    """The remedy every refusal names is the next deploy, and that is a NOOP.

    So a NOOP deploy skips the sync only over a success of the release it
    starts from; no record, an unreadable one, a failure or a success of some
    other release is retried, and the outcome replaces the record.
    """

    site = _site(tmp_path, monkeypatch)
    monkeypatch.setattr(release_deploy, "_run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(release_deploy, "_run_json", _noop_run_json)
    if previous == "failed":
        record_registry_sync(tmp_path, error="direct Aurora sync failed: ValueError")
    elif previous == "unreadable":
        outcome.record_path(tmp_path).parent.mkdir(parents=True)
        outcome.record_path(tmp_path).write_text("{not json", encoding="utf-8")
    elif previous == "foreign":
        record_registry_sync(tmp_path, release_id="release-older")

    prepared = _execute(site, tmp_path, live_release_id="release-a")

    state = json.loads((prepared.state_dir / "state.json").read_text())
    assert state["deployment"]["fast_path"] is True
    assert state["installation_registry"]["status"] == "SYNCED"
    assert state["installation_registry"]["noop_retry"] is True
    durable = _durable(tmp_path)
    assert (durable["status"], durable["release_id"]) == ("SYNCED", "release-a")
    assert outcome.status_view(tmp_path, release_id="release-a")["finding"] is None


def test_an_outcome_that_cannot_be_recorded_does_not_leave_an_older_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed sync whose record cannot be written must not read as a success.

    The older record is removed (the deletion verbs then refuse on a missing
    record) and the completion warning says the outcome was not recorded.
    """

    site = _site(tmp_path, monkeypatch)
    monkeypatch.setattr(release_deploy, "_run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(release_deploy, "_run_json", _non_noop_run_json)
    monkeypatch.setattr(
        release_deploy_evidence,
        "sync_installation_resource_registry",
        lambda _site: (_ for _ in ()).throw(BootstrapError("fetch failed")),
    )

    def unwritable(*_args, **_kwargs):
        raise OSError("No space left on device")

    monkeypatch.setattr(release_deploy_evidence, "write_outcome", unwritable)
    record_registry_sync(tmp_path, release_id="release-a")

    prepared = _execute(site, tmp_path)

    state = json.loads((prepared.state_dir / "state.json").read_text())
    assert state["phase"] == "COMPLETED", "recording is never a release gate"
    assert outcome.read_record(tmp_path) is None, "the older success must not stand"
    (warning,) = [
        item
        for item in state["completion_warnings"]
        if "installation resource registry" in item
    ]
    assert "fetch failed" in warning
    assert "outcome not recorded: OSError: No space left on device" in warning
