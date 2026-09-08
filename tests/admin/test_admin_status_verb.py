"""``status`` is the administrator's one read-only verb.

``status`` and ``verify`` had become the same 44-second report, with
``healthy`` on line 1150 of the JSON, and ``preflight`` was never in the task
table. The public CLI now offers ``status`` (cheap checks by default, ``--full``
for the whole report); ``preflight`` and ``verify`` survive only as unlisted
passthroughs for the release and staging drivers.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pytest

from gpu_fault.admin import cli as admin_cli
from tests.admin.test_admin_site import site_file


def test_public_help_offers_status_and_not_verify_or_preflight(capsys) -> None:
    with pytest.raises(SystemExit, match="0"):
        admin_cli.parser().parse_args(["--help"])

    help_text = capsys.readouterr().out
    assert "status" in help_text
    for verb in admin_cli.INTERNAL_READONLY_COMMANDS:
        assert verb not in help_text.replace("deploy", ""), (
            f"{verb} is a driver passthrough, not a public verb"
        )
    assert admin_cli.PUBLIC_READONLY_COMMANDS == ("status",)
    assert set(admin_cli.READONLY_COMMANDS) == {"status", "preflight", "verify"}


def test_status_accepts_full_and_defaults_to_the_quick_report() -> None:
    quick = admin_cli.parser().parse_args(["status", "--state-dir", "/tmp/s"])
    full = admin_cli.parser().parse_args(["status", "--state-dir", "/tmp/s", "--full"])

    assert (quick.command, quick.full) == ("status", False)
    assert (full.command, full.full) == ("status", True)


@pytest.mark.parametrize("verb", admin_cli.INTERNAL_READONLY_COMMANDS)
def test_driver_passthrough_verbs_still_parse_for_the_drivers(verb: str) -> None:
    """``release_deploy.py verify -f`` and ``staging_deploy.py preflight`` keep working."""

    arguments = admin_cli.parser().parse_args([verb, "-f", "/tmp/site.yaml"])

    assert arguments.command == verb
    assert arguments.file == Path("/tmp/site.yaml")
    assert arguments.full is False
    with pytest.raises(SystemExit):
        admin_cli.parser().parse_args([verb, "--full"])


def _status_report() -> dict[str, object]:
    return {
        "mode": "status",
        "healthy": True,
        "health_scope": "quick",
        "live_release": {
            "release_id": "release-a",
            "phase": "complete",
            "transaction_committed": True,
        },
        "next_deploy": {"kind": "NOOP", "changed": []},
        "health": {
            "summary": {"PASS": 2, "WARN": 0, "FAIL": 0, "SKIP": 0},
            "checks": [
                {"name": "cpu_workloads", "status": "PASS"},
                {"name": "control_api", "status": "PASS"},
            ],
        },
    }


def test_status_prints_a_header_on_stderr_and_the_json_unchanged_on_stdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    site_file(tmp_path)
    document = json.dumps(_status_report(), indent=2, sort_keys=True) + "\n"
    calls: list[list[str]] = []

    def fake_run(arguments, **kwargs):
        calls.append([str(item) for item in arguments])
        assert kwargs.get("stdout") is subprocess.PIPE, "status output is captured"
        return subprocess.CompletedProcess(arguments, 0, stdout=document)

    monkeypatch.setattr(admin_cli.subprocess, "run", fake_run)
    monkeypatch.setattr(admin_cli, "_live_release_state", lambda _site: {})
    monkeypatch.setenv(admin_cli.ADMIN_LOG_ENVIRONMENT, "/var/log/status.log")
    arguments = argparse.Namespace(
        command="status",
        file=None,
        state_dir=tmp_path,
        repo_root=None,
        show_effective_config=False,
        full=False,
    )

    assert admin_cli.run(arguments) == 0

    captured = capsys.readouterr()
    assert captured.out == document, "consumers parse stdout byte for byte"
    header = [
        line for line in captured.err.splitlines() if "gpu-fault-admin status:" in line
    ]
    assert header == [
        "gpu-fault-admin status: healthy: yes (quick health, 2 checks)",
        "gpu-fault-admin status: live release: release-a phase=complete committed=yes",
        "gpu-fault-admin status: next deploy: NOOP",
        "gpu-fault-admin status: failing checks: none",
        "gpu-fault-admin status: full JSON report: stdout (also in /var/log/status.log)",
    ]
    (command,) = calls
    assert command[1:2] == ["status"] and "--full" not in command


def test_status_full_reaches_the_engine(tmp_path: Path, monkeypatch) -> None:
    site_file(tmp_path)
    calls: list[list[str]] = []

    def fake_run(arguments, **_kwargs):
        calls.append([str(item) for item in arguments])
        return subprocess.CompletedProcess(arguments, 1, stdout="not json\n")

    monkeypatch.setattr(admin_cli.subprocess, "run", fake_run)
    monkeypatch.setattr(admin_cli, "_live_release_state", lambda _site: {})
    arguments = argparse.Namespace(
        command="status",
        file=None,
        state_dir=tmp_path,
        repo_root=None,
        show_effective_config=False,
        full=True,
    )

    assert admin_cli.run(arguments) == 1, "the engine's exit status passes through"
    (command,) = calls
    assert command[1] == "status" and command[-1] == "--full"


def test_status_without_a_document_still_reports_and_passes_output(capsys) -> None:
    admin_cli.print_status_report("release engine died\n")

    captured = capsys.readouterr()
    assert captured.out == "release engine died\n"
    assert "printed no JSON report" in captured.err
    # The rollout wrapper may print plain lines ahead of the document.
    report = admin_cli.status_document('narration\n{\n  "healthy": true\n}\n')
    assert report == {"healthy": True}
    assert admin_cli.status_document("") is None
