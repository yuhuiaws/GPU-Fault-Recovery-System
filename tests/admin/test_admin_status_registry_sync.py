"""``status`` reports the last installation registry sync as a finding.

Every deploy and ``config`` printed "installation resource registry not
synchronized" for nine days and nobody saw it. ``status`` now carries the deploy
host's record as the ``installation_registry_sync`` field of its JSON document
and, unless the record is a success for the release the site runs, as one
``finding:`` header line with the time, the reason and the remedy. It is a
finding, not a failing check: ``healthy`` and the exit code stay the engine's,
so an ordinary NOOP deploy (whose own first step reads ``status``) never looks
unhealthy because of it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.admin import cli as admin_cli
from gpu_fault.admin import resource_registry_outcome as outcome
from tests.admin._registry_sync_support import SYNCED_AT, record_registry_sync
from tests.admin.test_admin_site import site_file

FAILURE = (
    "installation resource registry not synchronized: BootstrapError: direct "
    "Aurora registry synchronization failed: ValueError"
)


def _report(*, healthy: bool = True, configured: str | None = None) -> dict[str, Any]:
    report: dict[str, Any] = {
        "mode": "status",
        "healthy": healthy,
        "health_scope": "quick",
        "live_release": {
            "release_id": "release-a",
            "phase": "complete",
            "transaction_committed": True,
        },
        "next_deploy": {"kind": "NOOP", "changed": []},
        "health": {
            "summary": {"PASS": 1, "WARN": 0, "FAIL": 0 if healthy else 1, "SKIP": 0},
            "checks": [
                {"name": "control_api", "status": "PASS" if healthy else "FAIL"}
            ],
        },
    }
    if configured is not None:
        report["configured_release"] = {"release_id": configured}
    return report


def _status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    stdout: str,
    *,
    returncode: int = 0,
) -> tuple[int, str, list[str]]:
    monkeypatch.setattr(
        admin_cli,
        "run_driver",
        lambda arguments, **_kwargs: subprocess.CompletedProcess(
            arguments, returncode, stdout=stdout
        ),
    )
    monkeypatch.setattr(admin_cli, "_live_release_state", lambda _site: {})
    arguments = argparse.Namespace(
        command="status",
        file=None,
        state_dir=tmp_path,
        repo_root=None,
        show_effective_config=False,
        full=False,
    )
    code = admin_cli.run(arguments)
    captured = capsys.readouterr()
    findings = [
        line.removeprefix("gpu-fault-admin status: finding: ")
        for line in captured.err.splitlines()
        if line.startswith("gpu-fault-admin status: finding: ")
    ]
    return code, captured.out, findings


def test_a_failed_sync_is_a_finding_and_a_field_and_healthy_stays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    site_file(tmp_path)
    record_registry_sync(tmp_path, release_id="release-a", error=FAILURE)
    engine = json.dumps(_report(), indent=2, sort_keys=True) + "\n"

    code, stdout, findings = _status(tmp_path, monkeypatch, capsys, engine)

    document = json.loads(stdout)
    field = document.pop(outcome.FIELD)
    assert code == 0, "a finding never changes the engine's exit code"
    assert document == _report(), "every engine field is passed through"
    assert document["healthy"] is True
    assert (field["status"], field["synced_at"], field["error"]) == (
        "UNAVAILABLE",
        SYNCED_AT,
        FAILURE,
    )
    assert field["deletion_verbs_refused"] is True
    assert findings == [field["finding"]]
    (finding,) = findings
    assert finding.startswith(
        f"the last installation registry sync failed at {SYNCED_AT} "
        f"(release release-a): {FAILURE}; uninstall and remove-cluster refuse"
    ), finding
    assert f"gpu-fault-admin deploy --state-dir {tmp_path}" in finding
    assert "`gpu-fault-admin config` re-roll" in finding


def test_an_unhealthy_engine_report_keeps_its_exit_code_and_gains_the_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    site_file(tmp_path)
    record_registry_sync(tmp_path, release_id="release-a")
    engine = json.dumps(_report(healthy=False), indent=2, sort_keys=True) + "\n"

    code, stdout, findings = _status(
        tmp_path, monkeypatch, capsys, engine, returncode=1
    )

    assert code == 1
    assert json.loads(stdout)[outcome.FIELD]["status"] == "SYNCED"
    assert findings == [], "a success of the running release is no finding"


def test_no_recorded_outcome_is_a_finding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    site_file(tmp_path)
    engine = json.dumps(_report(), indent=2, sort_keys=True) + "\n"

    _code, stdout, findings = _status(tmp_path, monkeypatch, capsys, engine)

    assert json.loads(stdout)[outcome.FIELD]["status"] == "MISSING"
    (finding,) = findings
    assert finding.startswith("no installation registry sync outcome is recorded"), (
        finding
    )


def test_an_unreadable_record_is_a_finding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    site_file(tmp_path)
    outcome.record_path(tmp_path).parent.mkdir(parents=True)
    outcome.record_path(tmp_path).write_text("[]", encoding="utf-8")
    engine = json.dumps(_report(), indent=2, sort_keys=True) + "\n"

    _code, stdout, findings = _status(tmp_path, monkeypatch, capsys, engine)

    field = json.loads(stdout)[outcome.FIELD]
    assert (field["status"], field["deletion_verbs_refused"]) == ("UNREADABLE", True)
    (finding,) = findings
    assert "sync record is unreadable" in finding and "not a JSON object" in finding


def test_the_release_rolled_back_to_is_the_one_compared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """After a rollback the live state names the candidate; the site runs the
    release it rolled back to, which the report gives as ``configured_release``."""

    site_file(tmp_path)
    record_registry_sync(tmp_path, release_id="release-previous")
    engine = (
        json.dumps(_report(configured="release-previous"), indent=2, sort_keys=True)
        + "\n"
    )

    _code, stdout, findings = _status(tmp_path, monkeypatch, capsys, engine)

    assert findings == []
    assert json.loads(stdout)[outcome.FIELD]["live_release_id"] == "release-previous"


def test_a_success_of_an_older_release_is_reported_without_a_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    site_file(tmp_path)
    record_registry_sync(tmp_path, release_id="release-old")
    engine = json.dumps(_report(), indent=2, sort_keys=True) + "\n"

    _code, stdout, findings = _status(tmp_path, monkeypatch, capsys, engine)

    (finding,) = findings
    assert finding.startswith(
        "no installation registry sync outcome since the live release release-a"
    ), finding
    assert json.loads(stdout)[outcome.FIELD]["deletion_verbs_refused"] is False


def test_output_that_is_not_one_json_object_passes_through_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    site_file(tmp_path)
    record_registry_sync(tmp_path, release_id="release-a", error=FAILURE)
    engine = "engine warning\n" + json.dumps(_report(), indent=2) + "\n"

    _code, stdout, findings = _status(tmp_path, monkeypatch, capsys, engine)

    assert stdout == engine, "a document the CLI cannot rewrite is left alone"
    assert len(findings) == 1, "the finding still reaches the header"
