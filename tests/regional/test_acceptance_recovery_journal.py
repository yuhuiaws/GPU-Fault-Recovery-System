from __future__ import annotations

import json
from pathlib import Path

import pytest

from gpu_fault.admin.process_supervisor import ProcessSupervisionLost
from scripts.e2e.regional import regional_commands as commands
from scripts.e2e.regional.acceptance_runner_common import EvidenceRecorder
from scripts.e2e.regional.acceptance_supervision import (
    bind_command_supervision,
    require_supervision_clear,
)


def test_lost_supervision_survives_a_new_run_initialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bind_command_supervision(tmp_path)

    def lost(*_args, **_kwargs):
        raise ProcessSupervisionLost("private-command-details")

    monkeypatch.setattr(commands, "run_command", lost)
    with pytest.raises(ProcessSupervisionLost):
        commands.run_fixture_command(["owned-test-command"])
    marker = tmp_path / "command-supervision-lost.json"
    assert marker.stat().st_mode & 0o077 == 0
    assert json.loads(marker.read_text())["status"] == "RECOVERY_REQUIRED"
    assert "private-command-details" not in marker.read_text()
    with pytest.raises(RuntimeError, match="independent recovery"):
        bind_command_supervision(tmp_path)
    with pytest.raises(RuntimeError, match="independent recovery"):
        require_supervision_clear(tmp_path)
    bind_command_supervision(tmp_path / "separate-run")


def test_failed_loss_marker_write_cannot_downgrade_a_process_abort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def lost(*_args, **_kwargs):
        raise ProcessSupervisionLost("uncontained command")

    def unavailable():
        raise OSError("private-file-path")

    monkeypatch.setattr(commands, "run_command", lost)
    monkeypatch.setattr(commands, "record_supervision_loss", unavailable)
    with pytest.raises(ProcessSupervisionLost, match="could not be persisted"):
        commands.run_fixture_command(["owned-test-command"])


def test_retry_invalidates_old_success_before_running_any_stage(tmp_path: Path) -> None:
    path = tmp_path / "case.json"
    first = EvidenceRecorder(path, case_id="case", inputs={"run": "one"})
    first.stage("snapshot", lambda: {"observed": True})
    first.note("verdict", "PASS")
    first.complete()
    resumed = EvidenceRecorder(path, case_id="case", inputs={"run": "one"})
    value = json.loads(path.read_text())
    assert value["status"] == "RUNNING"
    assert value["verdict"] == "NOT_RUN"
    assert "completed_at" not in value
    assert resumed.stage(
        "snapshot", lambda: pytest.fail("completed stage replayed")
    ) == {"observed": True}
    resumed.fail(RuntimeError("new verification failed"))
    assert json.loads(path.read_text())["verdict"] == "FAIL"


@pytest.mark.parametrize(
    "change",
    [
        lambda value: value.update(stages=[]),
        lambda value: value.update(stages={"snapshot": "PASS"}),
        lambda value: value.update(status="unknown"),
        lambda value: value.update(schema_version=0),
    ],
)
def test_malformed_evidence_is_not_replayable(tmp_path: Path, change) -> None:
    path = tmp_path / "case.json"
    EvidenceRecorder(path, case_id="case", inputs={})
    value = json.loads(path.read_text())
    change(value)
    path.write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match="malformed"):
        EvidenceRecorder(path, case_id="case", inputs={})


def test_non_object_stage_is_not_checkpointed(tmp_path: Path) -> None:
    path = tmp_path / "case.json"
    recorder = EvidenceRecorder(path, case_id="case", inputs={})
    with pytest.raises(RuntimeError, match="object"):
        recorder.stage("snapshot", lambda: None)
    assert json.loads(path.read_text())["stages"] == {}
