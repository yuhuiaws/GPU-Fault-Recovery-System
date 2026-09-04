"""Which command may name the quick-validation evidence file, and when.

The file itself is written by the release engine and validated against the live
release state; these cases only cover the CLI's half, which is deciding the path
and nothing else. Reuse still has to get past `quick_validation_evidence`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from gpu_fault.admin import cli as admin_cli


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
