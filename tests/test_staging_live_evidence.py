"""The staging fast path must not call a site UNCHANGED while its Runtime
Profile template differs from the live Profile.

Observed live on 2026-09-07: code and site.yaml were untouched, only the
Profile template had gained a capability claim, and ``gpu-fault-admin deploy``
exited 0 in two minutes as UNCHANGED -- the release engine, and with it the
``profile-plan.json`` stop, never ran.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts import release_deploy, staging_deploy, staging_live_evidence


def _completed(stdout: str, returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["release_deploy"], returncode=returncode, stdout=stdout, stderr="err"
    )


def _plan(change_kind: str) -> dict[str, object]:
    return {
        "change_kind": change_kind,
        "policy_digest": "c" * 64,
        "live_profile_sha256": "b" * 64,
    }


def test_policy_evidence_runs_the_release_engine_planner(tmp_path: Path) -> None:
    site_file = tmp_path / "site.yaml"
    site_file.write_text("kind: RegionalSite\n", encoding="utf-8")
    repository = tmp_path / "snapshot"
    calls: list[dict[str, object]] = []

    def runner(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append({"command": command, **kwargs})
        return _completed(json.dumps(_plan("UNCHANGED")))

    evidence = staging_live_evidence.runtime_profile_policy_evidence(
        site_file, repository_root=repository, runner=runner
    )
    assert evidence == {
        "runtime_profile_sha256": "b" * 64,
        "runtime_profile_policy_digest": "c" * 64,
    }
    (call,) = calls
    command = call["command"]
    assert isinstance(command, list), "planner must be invoked as an argv list"
    # Same planner and same interpreter arrangement as ``gpu-fault-admin deploy``:
    # the snapshot's release_deploy.py with the snapshot's src on PYTHONPATH.
    assert command[1] == str(repository / "scripts/release_deploy.py")
    assert command[2:] == ["--site", str(site_file), "--profile-plan-json"]
    environment = call["env"]
    assert isinstance(environment, dict), "planner needs an explicit environment"
    assert environment["PYTHONPATH"] == str(repository / "src")
    assert call["cwd"] == repository


def test_pending_template_change_is_not_unchanged_evidence(tmp_path: Path) -> None:
    site_file = tmp_path / "site.yaml"
    site_file.write_text("kind: RegionalSite\n", encoding="utf-8")
    with pytest.raises(
        staging_live_evidence.RuntimeProfileChangePending, match="EXPANSIVE"
    ):
        staging_live_evidence.runtime_profile_policy_evidence(
            site_file,
            repository_root=tmp_path,
            runner=lambda *_a, **_k: _completed(json.dumps(_plan("EXPANSIVE"))),
        )
    # The pending signal is a LiveEvidenceError too, so callers that only know
    # the base class still fall back to the full release rather than UNCHANGED.
    assert issubclass(
        staging_live_evidence.RuntimeProfileChangePending,
        staging_live_evidence.LiveEvidenceError,
    ), "pending profile change must still be a live-evidence failure"

    # A planner that fails (unreadable template, bad site) is no evidence either.
    with pytest.raises(staging_live_evidence.LiveEvidenceError, match="err"):
        staging_live_evidence.runtime_profile_policy_evidence(
            site_file,
            repository_root=tmp_path,
            runner=lambda *_a, **_k: _completed("", returncode=2),
        )
    with pytest.raises(staging_live_evidence.LiveEvidenceError, match="invalid"):
        staging_live_evidence.runtime_profile_policy_evidence(
            site_file,
            repository_root=tmp_path,
            runner=lambda *_a, **_k: _completed("not json"),
        )
    with pytest.raises(staging_live_evidence.LiveEvidenceError, match="object"):
        staging_live_evidence.runtime_profile_policy_evidence(
            site_file,
            repository_root=tmp_path,
            runner=lambda *_a, **_k: _completed(
                json.dumps({"change_kind": "UNCHANGED"})
            ),
        )


def _identities() -> dict[str, object]:
    return {
        "schema_version": 1,
        "sha256": "f" * 64,
        "application": {"sha256": "a" * 64},
        "deploy_host": {"sha256": "b" * 64, "bundle": {"sha256": "c" * 64}},
    }


def test_pending_profile_change_forces_application_release() -> None:
    source = staging_deploy.SourceCheckout(
        repository_root=Path("/repo"),
        git_commit="a" * 40,
        fingerprint="a" * 64,
        snapshot=False,
        isolated=True,
    )
    previous = {
        "identities": _identities(),
        "source": {"fingerprint": source.fingerprint},
    }
    host_changed = _identities()
    host_changed["deploy_host"] = {"sha256": "9" * 64, "bundle": {"sha256": "8" * 64}}
    quality_only = {"identities": _identities(), "source": {"fingerprint": "0" * 64}}
    # Every mode that applies no release would record the pending change as a
    # success and the next deploy would read UNCHANGED again.
    for stored, identities, live_matches, ordinary in (
        (previous, _identities(), True, "UNCHANGED"),
        (previous, host_changed, False, "DEPLOY_HOST_ONLY"),
        (quality_only, _identities(), False, "QUALITY_ONLY"),
    ):
        assert (
            staging_deploy.classify_source_deploy(
                stored,
                identities,
                source=source,
                site_exists=True,
                live_matches=live_matches,
            )
            == ordinary
        )
        assert (
            staging_deploy.classify_source_deploy(
                stored,
                identities,
                source=source,
                site_exists=True,
                live_matches=live_matches,
                profile_change_pending=True,
            )
            == "APPLICATION_RELEASE"
        )


def test_release_deploy_profile_plan_json_is_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    site_file = tmp_path / "site.yaml"
    site_file.write_text("kind: RegionalSite\n", encoding="utf-8")
    site_file.chmod(0o600)
    plan = release_deploy.ProfilePlan(
        site_identity={"site_name": "site"},
        site_identity_sha256="1" * 64,
        registration_cluster_id="cluster",
        template_source=tmp_path / "template.yaml",
        template_reference="template.yaml",
        active_source=tmp_path / "profiles/hyperpod-v1.yaml",
        current_version="hyperpod-v1",
        desired_version="regional-hyperpod-08d0bbcfa39d",
        current_policy_digest="2" * 64,
        policy_digest="08d0bbcfa39d" + "0" * 52,
        live_profile_sha256="3" * 64,
        active_source_sha256="3" * 64,
        source_sha256="4" * 64,
        snapshot_sha256="5" * 64,
        change_kind="EXPANSIVE",
        changes=("efaDriverRemediation: mode DISABLED->OWN",),
        approval=None,
        approval_plan_sha256=None,
        approval_relation=None,
        approval_required=True,
        snapshot_file=tmp_path / "snapshot.yaml",
        snapshot_document={},
    )
    seen: dict[str, object] = {}

    def planner(
        path: Path, *, live_profile_sha256: str | None
    ) -> release_deploy.ProfilePlan:
        seen["path"] = path
        seen["live"] = live_profile_sha256
        return plan

    monkeypatch.setattr(
        release_deploy,
        "read_live_release_state",
        lambda _site: {"runtime_profile_sha256": "3" * 64},
    )
    monkeypatch.setattr(release_deploy, "plan_runtime_profile", planner)
    for name in ("verify_attestation", "execute_release", "write_profile_plan"):
        monkeypatch.setattr(
            release_deploy,
            name,
            lambda *_a, **_k: pytest.fail(f"{name} ran during a read-only plan"),
        )

    assert release_deploy.main(["--site", str(site_file), "--profile-plan-json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert seen == {"path": site_file.resolve(), "live": "3" * 64}
    assert payload["change_kind"] == "EXPANSIVE"
    assert payload["approval_required"] is True
    assert payload["plan_sha256"] == release_deploy.profile_plan_sha256(
        {key: value for key, value in payload.items() if key != "plan_sha256"}
    )
    assert not (tmp_path / "release-deploy").exists(), "read-only plan wrote state"

    # No live state yet: the plan is computed against no baseline, not refused.
    def missing(_site: Path) -> dict[str, object]:
        raise release_deploy.ReleaseStateNotFound("absent")

    monkeypatch.setattr(release_deploy, "read_live_release_state", missing)
    assert release_deploy.main(["--site", str(site_file), "--profile-plan-json"]) == 0
    assert seen["live"] is None
