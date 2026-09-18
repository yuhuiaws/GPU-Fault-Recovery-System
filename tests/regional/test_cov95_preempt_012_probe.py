from __future__ import annotations

import json
import signal
import subprocess
from argparse import Namespace
from types import SimpleNamespace

import pytest

from scripts.e2e.regional.probes import preempt012_node_probe as probe


@pytest.fixture
def machine(tmp_path, monkeypatch):
    state = SimpleNamespace(
        calls=[],
        services="active\n",
        timer="ActiveState=active\n",
        returncode=0,
        quiesce_error=None,
        restore_error=None,
        handlers={},
        sleeps=[],
    )
    monkeypatch.setattr(probe, "STATE_ROOT", tmp_path / "state")
    monkeypatch.setattr(probe, "QUIESCE_ROOT", tmp_path / "quiesce")
    probe.QUIESCE_ROOT.mkdir()

    def run(command, **_kwargs):
        state.calls.append(list(command))
        output = state.services
        if command[:2] == ["systemctl", "show"]:
            output = state.timer
        elif command[0] == "nvidia-smi":
            output = "GPU-a\n\nGPU-b\n"
        return subprocess.CompletedProcess(
            command, state.returncode, output, "fixture failure"
        )

    def quiesce(**kwargs):
        state.calls.append(["quiesce", kwargs["incident_id"]])
        if state.quiesce_error:
            raise state.quiesce_error
        return {"quiesced": True}

    def restore(**kwargs):
        state.calls.append(["restore", kwargs["incident_id"]])
        if state.restore_error:
            raise state.restore_error
        return {"restored": True}

    manager = SimpleNamespace(quiesce=quiesce, restore=restore)
    monkeypatch.setattr(
        probe,
        "subprocess",
        SimpleNamespace(
            run=run, PIPE=subprocess.PIPE, SubprocessError=subprocess.SubprocessError
        ),
    )
    monkeypatch.setattr(probe, "GpuServiceQuiesceManager", lambda **_kwargs: manager)
    monkeypatch.setattr(
        probe,
        "signal",
        SimpleNamespace(
            SIGTERM=signal.SIGTERM,
            SIGINT=signal.SIGINT,
            signal=lambda number, handler: state.handlers.update({number: handler}),
        ),
    )
    monkeypatch.setattr(probe, "time", SimpleNamespace(sleep=state.sleeps.append))
    return state


@pytest.mark.parametrize("run_id", ["", "../x", "x/y", "a" * 64, "white space"])
def test_probe_rejects_unowned_paths_before_commands(run_id, machine) -> None:
    with pytest.raises(probe.ProbeError, match="unsafe run ID"):
        probe.arm(Namespace(run_id=run_id))
    assert machine.calls == [], "unsafe identity must never reach systemd"


@pytest.mark.parametrize(
    "timer", ["ActiveState=inactive\n", "ActiveState=\n", "Other=active\n"]
)
def test_arm_requires_positive_timer_acknowledgement(timer, machine) -> None:
    machine.timer = timer
    with pytest.raises(probe.ProbeError, match="timer is not active"):
        probe.arm(
            Namespace(
                run_id="run-a",
                delay_seconds=5,
                hold_seconds=45,
                failsafe_seconds=180,
                probe_script="/fixture/probe.py",
            )
        )
    assert [call[0] for call in machine.calls] == [
        "systemctl",
        "systemctl",
        "systemd-run",
        "systemctl",
    ], "the timer must be read back after arming"
    expected = "inactive" if "inactive" in timer else "unknown"
    assert probe.timer_active_state("unit") == expected, "unknown is not inactive"


@pytest.mark.parametrize("failure", ["", "quiesce", "restore", "interrupt"])
def test_cycle_always_attempts_restore_and_persists_final_state(
    failure, machine
) -> None:
    if failure == "quiesce":
        machine.quiesce_error = RuntimeError("fixture quiesce failure")
    elif failure == "restore":
        machine.restore_error = KeyboardInterrupt()
    elif failure == "interrupt":
        machine.quiesce_error = probe.ProbeInterrupted("fixture signal")
    probe.cycle(Namespace(run_id="run-a", hold_seconds=45, failsafe_seconds=180))
    result = probe.read_cycle(Namespace(run_id="run-a"))
    expected = {
        "": "COMPLETED",
        "quiesce": "FAILED",
        "restore": "FAILED",
        "interrupt": "INTERRUPTED",
    }
    assert result["status"] == expected[failure], (
        "the final state must reflect restoration"
    )
    assert [call[0] for call in machine.calls if call[0] in {"quiesce", "restore"}] == [
        "quiesce",
        "restore",
    ], "every quiesce attempt must enter the restore path"
    assert probe.evidence_path("run-a").stat().st_mode & 0o777 == 0o600, (
        "host cycle evidence must be private"
    )
    assert "restored_at" in result, "failed restoration also needs an observation time"
    with pytest.raises(probe.ProbeInterrupted, match="signal"):
        machine.handlers[signal.SIGTERM](signal.SIGTERM, None)


@pytest.mark.parametrize("payload", [None, [], "invalid"])
def test_read_refuses_nonobject_cycle_evidence(payload, machine) -> None:
    path = probe.evidence_path("run-a")
    probe.write_json(path, {"status": "RUNNING"})
    path.write_text(json.dumps(payload))
    with pytest.raises(probe.ProbeError, match="evidence is invalid"):
        probe.read_cycle(Namespace(run_id="run-a"))


def test_pending_read_and_snapshot_use_only_owned_temp_state(machine) -> None:
    assert probe.read_cycle(Namespace(run_id="absent")) == {
        "run_id": "absent",
        "status": "PENDING",
    }, "a missing local cycle file is pending, not completed"
    (probe.QUIESCE_ROOT / "quiesce-fixture.json").write_text("{}")
    machine.services = ""
    result = probe.snapshot()
    assert result["gpu_count"] == 2, "blank GPU output lines are not devices"
    assert result["quiesce_state_files"] == ["quiesce-fixture.json"], (
        "snapshot must retain existing quiesce evidence"
    )
    assert set(result["services"].values()) == {"unknown"}, (
        "empty service reads stay unknown"
    )


@pytest.mark.parametrize(
    "args",
    [
        ["snapshot"],
        ["arm", "--run-id", "run-a", "--probe-script", "/fixture/probe.py"],
        [
            "cycle",
            "--run-id",
            "run-a",
            "--hold-seconds",
            "1",
            "--failsafe-seconds",
            "180",
        ],
        ["read", "--run-id", "run-a"],
        ["cleanup", "--run-id", "run-a"],
    ],
)
def test_public_probe_entry_dispatches_through_fake_system_boundary(
    args, machine, monkeypatch, capsys
) -> None:
    parser = probe.parser()
    monkeypatch.setattr(
        probe,
        "parser",
        lambda: SimpleNamespace(parse_args=lambda: parser.parse_args(args)),
    )
    assert probe.main() == 0, "the fake probe command should complete"
    output = capsys.readouterr().out
    if args[0] == "cycle":
        assert probe.read_cycle(Namespace(run_id="run-a"))["status"] == "COMPLETED", (
            "cycle dispatch must persist terminal evidence"
        )
        assert output == "", "cycle writes its bounded evidence file"
    else:
        assert isinstance(json.loads(output), dict), "the command emits a JSON object"


def test_command_failure_is_reported_by_entry_without_success_payload(
    machine, monkeypatch, capsys
) -> None:
    machine.returncode = 1
    parser = probe.parser()
    monkeypatch.setattr(
        probe,
        "parser",
        lambda: SimpleNamespace(parse_args=lambda: parser.parse_args(["snapshot"])),
    )
    assert probe.main() == 1, "failed GPU query must fail the probe"
    assert json.loads(capsys.readouterr().out) == {
        "error": "command failed (1): fixture failure"
    }, "a failed snapshot must not fabricate a GPU count"
