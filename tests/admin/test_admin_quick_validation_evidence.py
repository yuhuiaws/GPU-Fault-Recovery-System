"""Which command may name the quick-validation evidence file, and when.

The file itself is written by the release engine and validated against the live
release state. The first cases cover the CLI's half, which is deciding the path
and nothing else; the last two cover the release driver's half, which is deciding
whether the state a deploy left behind still describes the release the evidence
was written for. Reuse always has to get past `quick_validation_evidence`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest

from gpu_fault.admin import cli as admin_cli
from gpu_fault_release import regional_validation_evidence as evidence_module
from scripts import release_deploy
from tests.admin.test_release_deploy import (
    _pending_commit_diff,
    _release_diff,
    _release_summary,
    _site,
    _stability_report,
    _verification_report,
)


def test_deploy_names_the_quick_validation_evidence_under_the_state_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deploy driven by the admin command writes reusable evidence.

    Only the wrapper scripts used to name this file, so the read-only verifiers a
    deploy had just run were re-run by the next `status`. A stale file is removed
    first: a deploy that fails before quick validation must leave no evidence
    rather than evidence describing the release it replaced.
    """

    monkeypatch.delenv(admin_cli.QUICK_VALIDATION_EVIDENCE_ENV, raising=False)
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    stale = state_dir / admin_cli.QUICK_VALIDATION_EVIDENCE_FILE
    stale.write_text("{}", encoding="utf-8")
    arguments = argparse.Namespace(command="deploy", state_dir=state_dir)

    environment = admin_cli.quick_validation_evidence_environment(arguments)

    assert environment == {
        admin_cli.QUICK_VALIDATION_EVIDENCE_ENV: str(stale.resolve())
    }
    assert not stale.exists(), (
        "a deploy must remove the previous release's evidence before it writes"
    )


def test_status_reuses_a_present_evidence_file_and_invents_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`status` points at the evidence only when the deploy left one.

    Naming a file that does not exist would make every report carry a fallback
    reason for evidence nobody claimed to have written.
    """

    monkeypatch.delenv(admin_cli.QUICK_VALIDATION_EVIDENCE_ENV, raising=False)
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    arguments = argparse.Namespace(command="status", state_dir=state_dir)

    assert admin_cli.quick_validation_evidence_environment(arguments) == {}

    evidence = state_dir / admin_cli.QUICK_VALIDATION_EVIDENCE_FILE
    evidence.write_text("{}", encoding="utf-8")

    assert admin_cli.quick_validation_evidence_environment(arguments) == {
        admin_cli.QUICK_VALIDATION_EVIDENCE_ENV: str(evidence.resolve())
    }


def test_the_driver_and_status_name_the_same_evidence_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One canonical evidence path, shared by the writer and the reader.

    The release driver used to finalize the evidence under its per-release
    directory (`release-deploy/<release_id>/quick-validation.json`) while `status`
    looked for `<state-dir>/quick-validation.json`; on the live state directory
    the root file never existed, so every `status` re-ran all fifteen probes the
    deploy had just proved. Both sides now derive the path from one helper, and
    this pins them equal without a cluster or a subprocess.
    """

    monkeypatch.delenv(admin_cli.QUICK_VALIDATION_EVIDENCE_ENV, raising=False)
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    site_file = state_dir / "site.yaml"

    canonical = evidence_module.quick_validation_evidence_path(state_dir)
    assert (
        canonical
        == state_dir.resolve() / evidence_module.QUICK_VALIDATION_EVIDENCE_FILE
    )
    assert release_deploy.quick_validation_evidence_path(site_file.parent) == (
        canonical
    ), "the release driver must finalize evidence where status reads it"
    assert admin_cli.QUICK_VALIDATION_EVIDENCE_FILE == (
        evidence_module.QUICK_VALIDATION_EVIDENCE_FILE
    )

    canonical.write_text("{}", encoding="utf-8")
    status = argparse.Namespace(command="status", state_dir=state_dir)
    assert admin_cli.quick_validation_evidence_environment(status) == {
        admin_cli.QUICK_VALIDATION_EVIDENCE_ENV: str(canonical)
    }
    by_file = argparse.Namespace(command="status", state_dir=None, file=site_file)
    assert admin_cli.quick_validation_evidence_environment(by_file) == {
        admin_cli.QUICK_VALIDATION_EVIDENCE_ENV: str(canonical)
    }, "status -f <site> lives in the same state directory as the deploy"


def test_verify_and_explicit_settings_keep_running_the_verifiers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`verify` is the gate, so it is never handed discovered evidence.

    An administrator who asks for the gate gets the probes. An explicit
    environment setting also wins everywhere, so the wrapper scripts keep naming
    their own path and a deploy driven by them does not have its file deleted.
    """

    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / admin_cli.QUICK_VALIDATION_EVIDENCE_FILE).write_text(
        "{}", encoding="utf-8"
    )

    monkeypatch.delenv(admin_cli.QUICK_VALIDATION_EVIDENCE_ENV, raising=False)
    verify = argparse.Namespace(command="verify", state_dir=state_dir)
    assert admin_cli.quick_validation_evidence_environment(verify) == {}

    monkeypatch.setenv(admin_cli.QUICK_VALIDATION_EVIDENCE_ENV, "/tmp/explicit.json")
    for command in ("deploy", "status"):
        arguments = argparse.Namespace(command=command, state_dir=state_dir)
        assert admin_cli.quick_validation_evidence_environment(arguments) == {}
    assert (state_dir / admin_cli.QUICK_VALIDATION_EVIDENCE_FILE).is_file(), (
        "an explicit setting must not delete the conventional evidence file"
    )


def test_evidence_finalizes_when_deploy_completed_but_not_committed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The state a successful deploy leaves behind is a reusable state.

    `rollout deploy` ends at `phase=complete, transaction_committed=False`: every
    component is applied and quick validation passed, and only the commit the
    driver runs after the stability window is outstanding. The finalizer used to
    demand a clean NOOP diff there, which that state never is, so *every*
    production deploy printed "quick validation evidence was not reusable" and
    verify re-ran the two read-only verifier scripts the deploy had run ninety
    seconds earlier. `pending_commit` plus the deployed release id is what tells
    that state apart from a half-applied transaction.
    """

    monkeypatch.delenv("GPU_FAULT_ADMIN_LOG", raising=False)
    monkeypatch.delenv(admin_cli.QUICK_VALIDATION_EVIDENCE_ENV, raising=False)
    site = _site(tmp_path, monkeypatch)
    diffs = iter(
        (
            _release_diff("CONTROL_PLANE_ONLY", ["control_plane_wheel"]),
            _pending_commit_diff(state_sha256="b" * 64),
        )
    )
    verify_environments: list[dict[str, str]] = []

    def run(arguments, **kwargs):
        command = list(arguments)
        if "deploy" in command:
            path = Path(
                kwargs["environment"][release_deploy.QUICK_VALIDATION_EVIDENCE_ENV]
            )
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "release_id": "release-a",
                        "checks": ["control_plane_role_split"],
                        "verified_at_epoch": 1,
                    }
                ),
                encoding="utf-8",
            )

    def run_json(arguments, **kwargs):
        command = list(arguments)
        if "verify" in command:
            verify_environments.append(dict(kwargs["environment"]))
            return _verification_report()
        if "stability" in command:
            return _stability_report()
        if "release-diff" in command:
            return next(diffs)
        return _release_summary()

    monkeypatch.setattr(release_deploy, "_run", run)
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

    evidence = evidence_module.quick_validation_evidence_path(site.parent)
    assert not (
        prepared.state_dir / admin_cli.QUICK_VALIDATION_EVIDENCE_FILE
    ).exists(), "the per-release copy is the location status never read"
    finalized = json.loads(evidence.read_text(encoding="utf-8"))
    assert finalized["release_state_sha256"] == "b" * 64
    assert finalized["finalized_at"]
    assert evidence.stat().st_mode & 0o777 == 0o600, (
        "the evidence reader refuses a file other users can read"
    )
    assert len(verify_environments) == 1
    assert verify_environments[0][release_deploy.QUICK_VALIDATION_EVIDENCE_ENV] == str(
        evidence
    ), "verify was not handed the evidence the deploy had just proved"
    status = argparse.Namespace(command="status", state_dir=site.parent)
    assert admin_cli.quick_validation_evidence_environment(status) == {
        admin_cli.QUICK_VALIDATION_EVIDENCE_ENV: str(evidence)
    }, "the next status must find the evidence the driver finalized"


def test_evidence_is_dropped_when_the_post_deploy_state_is_not_ours(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pending commit for another release is not evidence for this one.

    `pending_commit` alone would let a concurrent transaction's state vouch for
    probes that ran against a different release, so the release id has to match
    the release this driver prepared. When it does not, verify gets no evidence
    and runs the full gate.
    """

    monkeypatch.delenv("GPU_FAULT_ADMIN_LOG", raising=False)
    monkeypatch.delenv(admin_cli.QUICK_VALIDATION_EVIDENCE_ENV, raising=False)
    site = _site(tmp_path, monkeypatch)
    diffs = iter(
        (
            _release_diff("CONTROL_PLANE_ONLY", ["control_plane_wheel"]),
            _pending_commit_diff(state_sha256="b" * 64, release_id="release-other"),
        )
    )
    verify_environments: list[dict[str, str]] = []

    def run(arguments, **kwargs):
        if "deploy" in list(arguments):
            Path(
                kwargs["environment"][release_deploy.QUICK_VALIDATION_EVIDENCE_ENV]
            ).write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "release_id": "release-a",
                        "checks": ["control_plane_role_split"],
                        "verified_at_epoch": 1,
                    }
                ),
                encoding="utf-8",
            )

    def run_json(arguments, **kwargs):
        command = list(arguments)
        if "verify" in command:
            verify_environments.append(dict(kwargs["environment"]))
            return _verification_report()
        if "stability" in command:
            return _stability_report()
        if "release-diff" in command:
            return next(diffs)
        return _release_summary()

    monkeypatch.setattr(release_deploy, "_run", run)
    monkeypatch.setattr(release_deploy, "_run_json", run_json)
    profile = tmp_path / "repo/config/profile.yaml"

    release_deploy.execute_release(
        site,
        run_checks=False,
        live_state={
            "release_id": "release-a",
            "runtime_profile_sha256": hashlib.sha256(profile.read_bytes()).hexdigest(),
        },
    )

    assert len(verify_environments) == 1
    assert release_deploy.QUICK_VALIDATION_EVIDENCE_ENV not in verify_environments[0]
    assert not evidence_module.quick_validation_evidence_path(site.parent).exists(), (
        "evidence the driver declined must not be left for the next status to find"
    )
