"""Recovery cleanup refusals through the real runner and existing fake I/O."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import destr014_recovery as recovery
from scripts.e2e.regional import live_driver_guard
from scripts.e2e.regional.destr014_cleanup import _cleanup
from scripts.e2e.regional.destr014_verdicts import recovery_cleanup_hold
from tests.regional.test_destr014_recovery_runner import RunnerHarness


def settled_cleanup(runner: RunnerHarness) -> dict[str, Any]:
    """Exercise known-outcome cleanup without entering the unknown-reboot drill."""
    assert recovery_cleanup_hold(runner.snapshot()) is False, (
        "the cleanup premise needs complete known workflow/command evidence"
    )
    runner.terminal = True
    probe = SimpleNamespace(cleanup=lambda: runner.cleanup_result("probe.cleanup", {}))
    recovered_agent = SimpleNamespace(
        cleanup=lambda: runner.event("recovery.cleanup") or {"phase": "CLOSED"}
    )
    return _cleanup(
        regional=runner.regional,
        warm=runner.warm,
        workload=runner.workload,
        prewarm=runner.prewarm,
        fault_probe=probe,
        sibling_probe=probe,
        inject_fault=probe,
        inject_sibling=probe,
        settings=runner.settings,
        run_id="settled-cleanup",
        incident_id=runner.incident["incident_id"],
        env_baseline=runner.case_dir / "executor.json",
        env_opened=True,
        control_env_baseline=runner.case_dir / "cpu.json",
        control_env_opened=True,
        holder_armed=False,
        agent_disabled=False,
        profile_version=runner.preflight["store"]["profile"]["profile_version"],
        recovery_window=recovered_agent,
        operator_hold=False,
    )


def test_source_digest_binds_the_new_cleanup_helper_bytes_without_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    relative = Path("scripts/e2e/regional/destr014_cleanup.py")
    helper = tmp_path / relative
    helper.parent.mkdir(parents=True)
    helper.write_text("revision = 1\n")
    calls: list[list[str]] = []

    def git(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        assert command[0] == "git"
        assert kwargs["cwd"] == tmp_path
        if command[1:] == ["rev-parse", "HEAD"]:
            output = "a" * 40
        else:
            assert command[1] == "ls-files"
            selected = command[command.index("--") + 1 :]
            assert any(relative.is_relative_to(Path(path)) for path in selected), (
                "source inventory must include the extracted cleanup helper"
            )
            if "--deleted" in command:
                output = ""
            else:
                assert {"--cached", "--others", "--exclude-standard", "-z"} <= set(
                    command
                )
                output = relative.as_posix() + "\0"
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")

    monkeypatch.setattr(live_driver_guard, "ROOT", tmp_path)
    monkeypatch.setattr(subprocess, "run", git)
    original = live_driver_guard.source_digest()
    helper.write_text("revision = 2\n")
    assert live_driver_guard.source_digest() != original
    helper.write_text("revision = 1\n")
    assert live_driver_guard.source_digest() == original
    helper.unlink()
    with pytest.raises(RuntimeError, match="focused-test input is unavailable"):
        live_driver_guard.source_digest()
    assert len(calls) == 12


@pytest.mark.parametrize("disabled", [False, True])
def test_missing_recovery_binding_cannot_use_a_legacy_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, disabled: bool
) -> None:
    runner = RunnerHarness(tmp_path, monkeypatch)
    if disabled:
        runner.recovery_transport.lost_ack = "disable"
        runner.failures["recovery.cleanup"] = RuntimeError("temporary cleanup outage")
        code, first = runner.execute()
        assert code == 1 and first["cleanup"]["errors"]
        saved = json.loads(runner.journal_path.read_text())
        assert saved["run"]["agent_disabled"] is True
        runner.failures.clear()
    before = len(runner.calls)
    monkeypatch.setattr(recovery, "AgentRecoveryWindow", lambda *_args: None)
    code, report = runner.execute()
    assert code == 1 and report["verdict"] == "FAIL"
    calls = runner.calls[before:]
    assert "recovery.disable" not in calls
    assert "recovery.restore" not in calls
    assert "recovery.cleanup" not in calls
    assert "probe.restore-agent" not in calls
    assert "probe.disable-agent-restart" not in calls
    if disabled:
        assert report["cleanup"]["errors"] == [
            "Agent recovery binding is missing; legacy restore is not authorized"
        ]
        assert (
            json.loads(runner.journal_path.read_text())["phase"] == "RECOVERY_REQUIRED"
        )
    else:
        assert "independent Node Agent recovery safeguard is missing" in report["error"]
        assert report["cleanup"]["errors"] == []
        assert runner.host.enable.exists(), (
            "missing recovery binding must preserve Agent boot activation"
        )


def test_case_owned_isolation_is_restored_once_without_reclosing_the_incident(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = RunnerHarness(tmp_path, monkeypatch)
    cleanup_started = False
    original_delete = runner.workload.delete
    original_snapshot = runner.regional.node_snapshot

    def delete() -> None:
        nonlocal cleanup_started
        original_delete()
        cleanup_started = True

    def snapshot(node: str) -> dict[str, Any]:
        result: dict[str, Any] = original_snapshot(node)
        if cleanup_started and result["ownership_annotations"]:
            result["ownership_annotations"]["gpu-fault.io/incident-id"] = (
                runner.incident["incident_id"]
            )
        return result

    monkeypatch.setattr(runner.workload, "delete", delete)
    monkeypatch.setattr(runner.regional, "node_snapshot", snapshot)
    cleanup = settled_cleanup(runner)
    assert cleanup["errors"] == []
    restored = cleanup["isolation_restore"]
    sibling = restored["nodes"][runner.settings.sibling_node]
    assert sibling["quarantine_owner"] == runner.incident["incident_id"]
    assert "successor_incident" not in sibling
    assert sibling["restore"]["status"] == "SUCCEEDED"
    assert restored["incident_state_before_close"] == "RECOVERED"
    assert restored["incident_state_after"] == "RECOVERED"
    assert "close" not in restored
    assert runner.calls.count("isolation.restore") == 1
    assert "probe.write-xid46" not in runner.calls, (
        "known-outcome helper coverage must not enter an unknown physical drill"
    )


@pytest.mark.parametrize("failed_wait", [1, 2])
def test_failed_node_restore_or_incident_close_keeps_outer_cleanup_unfinished(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_wait: int
) -> None:
    runner = RunnerHarness(tmp_path, monkeypatch)
    waits: list[str] = []

    def wait(request_id: str) -> dict[str, str]:
        waits.append(request_id)
        return {
            "status": "FAILED" if len(waits) == failed_wait else "SUCCEEDED",
            "error": "fake restore rejection" if len(waits) == failed_wait else "",
        }

    monkeypatch.setattr(runner.warm, "wait_workflow_id", wait)
    cleanup = settled_cleanup(runner)
    assert len(waits) == failed_wait
    error = (
        "restore workflow did not succeed: FAILED fake restore rejection"
        if failed_wait == 1
        else "case incident close workflow did not succeed"
    )
    assert len(cleanup["errors"]) == 1
    assert error in cleanup["errors"][0]
    assert cleanup["agent_recovery"]["phase"] == "CLOSED"
    assert runner.calls.count("prewarm.cleanup") == 1
    assert runner.calls.count("probe.cleanup") == 4


def test_unknown_drill_cannot_borrow_known_looking_rows_as_cleanup_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = RunnerHarness(tmp_path, monkeypatch)
    assert recovery_cleanup_hold(runner.snapshot()) is False, (
        "the stored fixture alone looks settled before the unknown injection"
    )
    code, report = runner.execute()
    assert code == 1 and report["verdict"] == "FAIL", (
        "confirmed-failure rows cannot pass the unknown-reboot scenario"
    )
    assert report["cleanup"]["operator_hold_preserved"] is True
    assert report["cleanup"]["agent_recovery"]["phase"] == "CLOSED"
    assert "workload.delete" not in runner.calls
    assert "isolation.restore" not in runner.calls
    saved = json.loads(runner.journal_path.read_text())
    assert saved["phase"] == "RECOVERY_REQUIRED"
    assert saved["host_cleanup"]["phase"] == "CLOSED"
    assert saved["host_cleanup"]["record_kind"] == "FORENSIC_TOMBSTONE"


def test_failed_quiescence_without_a_holder_still_defers_workload_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = RunnerHarness(tmp_path, monkeypatch)

    def unsettled(_incident: str) -> None:
        raise RuntimeError("remote action is still in flight")

    monkeypatch.setattr(runner.warm, "wait_incident_idle", unsettled)
    cleanup = settled_cleanup(runner)
    assert "workload.delete" not in runner.calls, (
        "absence of a holder is not proof that the incident is idle"
    )
    assert "isolation.restore" not in runner.calls
    assert cleanup["workload_cleanup_deferred"] is True
    assert any("quiescence" in error for error in cleanup["errors"]), cleanup
    assert runner.calls.count("prewarm.cleanup") == 1
    assert runner.calls.count("probe.cleanup") == 4
