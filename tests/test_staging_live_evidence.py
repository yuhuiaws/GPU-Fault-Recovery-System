"""The staging fast path must not call a site UNCHANGED while its Runtime
Profile template differs from the live Profile.

Observed live on 2026-09-07: code and site.yaml were untouched, only the
Profile template had gained a capability claim, and ``gpu-fault-admin deploy``
exited 0 in two minutes as UNCHANGED -- the release engine, and with it the
``profile-plan.json`` stop, never ran.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from scripts import (
    release_deploy,
    release_deploy_profile,
    staging_deploy,
    staging_live_evidence,
)


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
    for module, name in (
        (release_deploy, "verify_attestation"),
        (release_deploy, "execute_release"),
        (release_deploy_profile, "write_profile_plan"),
    ):
        monkeypatch.setattr(
            module,
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


# --- The success record built from the release driver's own files ---------
#
# ``apply_source_deploy`` used to end an APPLICATION_RELEASE with a second
# ``gpu-fault-admin status`` (about 45 seconds) only to read back the release
# the driver had just committed and summarized. The record is now assembled
# from ``release-deploy/<id>/state.json`` and ``release-summary.json``; the
# next run compares it key for key with its own live reading, so the two
# constructions must agree exactly.

RELEASE_ID = "release-20260908-abc123"
STATE_SHA = "e" * 64
PROFILE_SHA = "f" * 64
POLICY_DIGEST = "c" * 64


def _release_summary() -> dict[str, object]:
    # The shape ``rollout release-summary`` prints, trimmed to what is read.
    return {
        "mode": "release-summary",
        "site_name": "site",
        "configured_release": {"release_id": RELEASE_ID},
        "configured_runtime_profile": {
            "version": "hyperpod-v2",
            "source_sha256": PROFILE_SHA,
            "policy_sha256": POLICY_DIGEST,
        },
        "live_release": {
            "release_id": RELEASE_ID,
            "phase": "complete",
            "transaction_committed": True,
            "release_lifecycle": None,
            "state_sha256": STATE_SHA,
            "schema_change_acceptance": None,
        },
        "next_deploy": {"kind": "NOOP", "changed": []},
    }


def _write_release_record(
    state_dir: Path,
    *,
    release_id: str = RELEASE_ID,
    phase: str = "COMPLETED",
    summary: dict[str, object] | None = None,
) -> Path:
    record_dir = state_dir / "release-deploy" / release_id
    record_dir.mkdir(parents=True)
    summary_path = record_dir / "release-summary.json"
    summary_path.write_text(
        json.dumps(summary if summary is not None else _release_summary(), indent=2)
        + "\n",
        encoding="utf-8",
    )
    # The fields ``release_deploy.py`` writes on the way to COMPLETED that the
    # record reader consults; the real file carries the plan and images too.
    state = {
        "schema_version": 1,
        "release_id": release_id,
        "phase": phase,
        "profile_change": {
            "kind": "UNCHANGED",
            "changes": [],
            "policy_digest": POLICY_DIGEST,
            "source_sha256": PROFILE_SHA,
        },
        "verification": {
            "status": "PASSED",
            "path": str(record_dir / "verification-report.json"),
            "summary": {"PASS": 12, "WARN": 0, "FAIL": 0, "SKIP": 0},
        },
        "release_summary": {
            "status": "AVAILABLE",
            "path": str(summary_path),
            "sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
            "next_deploy": {"kind": "NOOP", "changed": []},
        },
    }
    record = record_dir / "state.json"
    record.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    return record


def _status_report() -> dict[str, object]:
    return {**_release_summary(), "mode": "status", "healthy": True, "health": {}}


def test_release_record_evidence_equals_the_live_status_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "site.yaml").write_text("kind: RegionalSite\n", encoding="utf-8")
    _write_release_record(state_dir)

    def run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        assert command[1] == "status", command
        return _completed(json.dumps(_status_report()))

    monkeypatch.setattr(staging_live_evidence.subprocess, "run", run)
    # The planner's reading of the live Profile: the live state's
    # ``runtime_profile_sha256`` is what the engine wrote at commit, i.e. the
    # configured Profile's source digest, and the policy digest is the plan's.
    monkeypatch.setattr(
        staging_live_evidence,
        "runtime_profile_policy_evidence",
        lambda *_a, **_k: {
            "runtime_profile_sha256": PROFILE_SHA,
            "runtime_profile_policy_digest": POLICY_DIGEST,
        },
    )

    from_status = staging_live_evidence.collect_live_deploy_evidence(
        repository_root=tmp_path / "snapshot",
        state_dir=state_dir,
        venv=tmp_path / "venv",
    )
    from_record = staging_live_evidence.live_evidence_from_release_record(
        state_dir=state_dir
    )

    assert from_record == from_status
    assert from_record["release_id"] == RELEASE_ID
    assert from_record["state_sha256"] == STATE_SHA
    assert staging_live_evidence.successful_source_live_matches(
        {"live": from_record}, from_status
    ), "the next run must recognize the record as its own live reading"


def test_release_record_evidence_reads_the_newest_completed_record(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "site.yaml").write_text("kind: RegionalSite\n", encoding="utf-8")
    older = _write_release_record(state_dir, release_id="release-old")
    newer = _write_release_record(state_dir, release_id="release-new")
    os.utime(older, (1_700_000_000, 1_700_000_000))
    os.utime(newer, (1_700_000_100, 1_700_000_100))

    assert staging_live_evidence.latest_release_record(state_dir) == newer


@pytest.mark.parametrize(
    ("mutate", "reason"),
    (
        (lambda state, _summary: state.update(phase="FAILED"), "not COMPLETED"),
        (
            lambda state, _summary: state.update(verification={"status": "SKIPPED"}),
            "no passed verification",
        ),
        (
            lambda state, _summary: state.update(
                release_summary={"status": "UNAVAILABLE"}
            ),
            "no release summary",
        ),
        (
            lambda _state, summary: summary["next_deploy"].update(
                kind="CONTROL_PLANE_ONLY"
            ),
            "committed NOOP release",
        ),
        (
            lambda _state, summary: summary["live_release"].update(
                transaction_committed=False
            ),
            "committed NOOP release",
        ),
        (
            lambda state, _summary: state.update(release_id="somebody-else"),
            "disagree on the release",
        ),
    ),
)
def test_release_record_evidence_refuses_an_unfinished_release(
    tmp_path: Path, mutate, reason: str
) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "site.yaml").write_text("kind: RegionalSite\n", encoding="utf-8")
    record = _write_release_record(state_dir)
    state = json.loads(record.read_text(encoding="utf-8"))
    summary_path = record.parent / "release-summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    mutate(state, summary)
    summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")
    state["release_summary"]["sha256"] = hashlib.sha256(
        summary_path.read_bytes()
    ).hexdigest()
    record.write_text(json.dumps(state) + "\n", encoding="utf-8")

    with pytest.raises(staging_live_evidence.LiveEvidenceError, match=reason):
        staging_live_evidence.live_evidence_from_release_record(state_dir=state_dir)


def test_release_record_evidence_requires_the_summary_the_record_names(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "site.yaml").write_text("kind: RegionalSite\n", encoding="utf-8")
    record = _write_release_record(state_dir)
    summary_path = record.parent / "release-summary.json"
    summary_path.write_text(
        json.dumps(_release_summary()) + "\n# edited\n", encoding="utf-8"
    )

    with pytest.raises(
        staging_live_evidence.LiveEvidenceError, match="does not match its record"
    ):
        staging_live_evidence.live_evidence_from_release_record(state_dir=state_dir)

    (state_dir / "release-deploy").rename(state_dir / "gone")
    with pytest.raises(staging_live_evidence.LiveEvidenceError, match="no release"):
        staging_live_evidence.live_evidence_from_release_record(state_dir=state_dir)


def test_pre_deploy_gate_accepts_the_quick_status_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pre-deploy reading is the default ``status``: summary + cheap checks.

    It is ``mode: status`` with ``health_scope: quick`` and no GPU-cluster or
    role-split checks; the classification only needs the live release identity
    and a healthy control plane.
    """

    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "site.yaml").write_text("kind: RegionalSite\n", encoding="utf-8")
    commands: list[list[str]] = []

    monkeypatch.setattr(
        staging_live_evidence,
        "runtime_profile_policy_evidence",
        lambda *_a, **_k: {
            "runtime_profile_sha256": PROFILE_SHA,
            "runtime_profile_policy_digest": POLICY_DIGEST,
        },
    )

    def run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        report = {
            **_status_report(),
            "health_scope": "quick",
            "health": {
                "mode": "status",
                "scope": "quick",
                "healthy": True,
                "checks": [
                    {"name": "cpu_workloads", "status": "PASS"},
                    {"name": "control_api", "status": "PASS"},
                ],
            },
        }
        return _completed(json.dumps(report))

    monkeypatch.setattr(staging_live_evidence.subprocess, "run", run)

    evidence = staging_live_evidence.collect_live_deploy_evidence(
        repository_root=tmp_path / "snapshot",
        state_dir=state_dir,
        venv=tmp_path / "venv",
    )

    assert evidence["release_id"] == RELEASE_ID
    status_command = commands[0]
    assert status_command[1:] == ["status", "--state-dir", str(state_dir)], (
        "the gate runs the default (quick) status, not --full"
    )


def test_read_live_status_keeps_the_report_of_an_unhealthy_site(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``status`` exits 1 for an unhealthy site and still prints the report.

    A deploy over a failed transaction needs exactly that report (live release
    phase and id feed the consent refusals and the classification); live
    2026-09-11 the exit code alone was treated as "no report" and
    --supersede-failed-transaction could not start. Only an exit code other
    than 0/1, or an unparsable body, is a failed reading.
    """

    state_dir = tmp_path / "state"
    unhealthy = {
        **_status_report(),
        "healthy": False,
        "live_release": {
            "phase": "failed",
            "release_id": "cand",
            "transaction_committed": False,
        },
    }
    answers = {"rc": 1, "body": json.dumps(unhealthy)}

    def run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        assert command[1] == "status", command
        return _completed(answers["body"], returncode=answers["rc"])

    monkeypatch.setattr(staging_live_evidence.subprocess, "run", run)

    report = staging_live_evidence.read_live_status(
        repository_root=tmp_path, state_dir=state_dir, venv=tmp_path / "venv"
    )
    assert report["healthy"] is False
    assert report["live_release"]["phase"] == "failed"

    answers.update(rc=2, body="")
    with pytest.raises(
        staging_live_evidence.LiveEvidenceError, match="status failed \\(2\\)"
    ):
        staging_live_evidence.read_live_status(
            repository_root=tmp_path, state_dir=state_dir, venv=tmp_path / "venv"
        )
    answers.update(rc=1, body="not json")
    with pytest.raises(
        staging_live_evidence.LiveEvidenceError, match="status failed \\(1\\)"
    ):
        staging_live_evidence.read_live_status(
            repository_root=tmp_path, state_dir=state_dir, venv=tmp_path / "venv"
        )
