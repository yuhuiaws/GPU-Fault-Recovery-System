from __future__ import annotations

import json
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.admin.process_supervisor import ProcessSupervisionLost
from scripts.e2e.regional import notify008_fixture as fixture
from scripts.e2e.regional import notify008_runner as runner
from scripts.e2e.regional.probes import notify008_probe as probe
from scripts.e2e.regional.probes.notify008_protocol import CASE_ID, ProbeError
from scripts.e2e.regional.regional_commands import RegionalCommandFailed
from tests.regional._cov95_notify008_lifecycle import setup_run, target
from tests.regional._cov95_notify008_probe import argv, probe_environment

UNTRUSTED = "untrusted-private-diagnostic-value"
REMOTE_FAILURE = {
    "case_id": CASE_ID,
    "verdict": "FAIL",
    "command": "prepare",
    "stage": "prepare",
    "error_type": "ProbeError",
    "reason": "process-budget-expired",
}


@pytest.mark.parametrize(
    "output",
    [
        json.dumps({**REMOTE_FAILURE, "message": UNTRUSTED, "env": {"key": UNTRUSTED}}),
        UNTRUSTED,
        json.dumps(
            {
                **REMOTE_FAILURE,
                "stage": [UNTRUSTED],
                "reason": UNTRUSTED,
                "error_type": UNTRUSTED,
                "command": UNTRUSTED,
            }
        ),
        "[" * 2000,
        "x" * 65537,
    ],
    ids=["structured", "not-json", "unknown-fields", "deep-json", "oversized"],
)
def test_nonzero_cpu_exec_retains_only_allowlisted_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, output: str
) -> None:
    path = tmp_path / "cpu"
    path.write_text("local fake connection")
    calls: list[dict[str, Any]] = []

    def command(
        arguments: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        calls.append(kwargs)
        return subprocess.CompletedProcess(arguments, 127, output, UNTRUSTED)

    monkeypatch.setattr(fixture, "run_fixture_command", command)
    with pytest.raises(RegionalCommandFailed) as raised:
        fixture.CpuAPI(path, "local-cpu").call("exec", "owned-pod")
    error = raised.value
    assert isinstance(error, fixture.CpuCommandFailed), (
        "test_nonzero_cpu_exec_retains_only_allowlisted_fields: expected isinstance(error, fixture.CpuCommandFailed)"
    )
    assert calls[0]["check"] is False, "nonzero stdout must reach the diagnostic filter"
    assert error.diagnostic["exit_code"] == 127
    assert error.diagnostic["command_stage"] == "exec"
    assert error.diagnostic["reason"] == "command-nonzero"
    assert UNTRUSTED not in json.dumps(error.diagnostic)
    assert UNTRUSTED not in str(error)
    if output.startswith('{"case_id"'):
        remote = error.diagnostic["probe"]
        assert set(remote) == {"command", "stage", "reason", "error_type"}
        assert remote["reason"] in {"process-budget-expired", "unclassified-error"}


@pytest.mark.parametrize("returncode", [0, 1])
def test_failed_probe_report_is_durable_before_failed_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, returncode: int
) -> None:
    settings, api, case_dir, deadline = setup_run(tmp_path, monkeypatch)
    cpu = fixture.CpuAPI(settings.cpu_kubeconfig, settings.cpu_context)
    original_call = api.call
    retained: list[dict[str, Any]] = []

    def completed(
        arguments: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            arguments,
            returncode,
            json.dumps({**REMOTE_FAILURE, "stderr": UNTRUSTED, "image": UNTRUSTED}),
            UNTRUSTED,
        )

    def call(*arguments: str, **kwargs: Any) -> str:
        if arguments[0] == "exec" and "prepare" in arguments:
            return cpu.call(*arguments, **kwargs)
        if arguments[0] == "exec" and "stop" in arguments:
            journal = json.loads((case_dir / "notify008-ownership.json").read_text())
            retained.append(journal["first_failure"])
            raise RuntimeError(UNTRUSTED)
        if arguments[0] == "delete":
            raise OSError(UNTRUSTED)
        return original_call(*arguments, **kwargs)

    monkeypatch.setattr(fixture, "run_fixture_command", completed)
    monkeypatch.setattr(api, "call", call)
    assert runner.execute_case(settings, tmp_path, 1, deadline) == 1
    journal_text = (case_dir / "notify008-ownership.json").read_text()
    journal = json.loads(journal_text)
    result_text = (case_dir / f"{CASE_ID}.json").read_text()
    result = json.loads(result_text)
    assert retained == [journal["first_failure"]], (
        "cleanup must not replace the first error"
    )
    first = journal["first_failure"]
    assert first["stage"] == "prepare"
    assert first["exit_code"] == (1 if returncode else None)
    assert first["probe"] == {
        key: REMOTE_FAILURE[key] for key in ("command", "stage", "error_type", "reason")
    }
    assert first["pod_status"]["containers"]["database"]["ready"] is True
    assert result["verdict"] == "FAIL" and result["cleanup"]["errors"]
    assert result["cleanup"]["namespace_absent"] is False
    assert result["cleanup"]["process_termination_proven"] is False
    assert UNTRUSTED not in journal_text + result_text
    assert not any("run" in command for command in api.calls), (
        "failed preparation cannot start runtime work"
    )


def test_container_exit_is_retained_before_cleanup_without_raw_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, api, case_dir, deadline = setup_run(tmp_path, monkeypatch)
    sandbox = fixture.Sandbox(api, target(), case_dir, deadline, lambda: None)
    sandbox.create()
    sandbox.admit()
    database = api.pod["status"]["containerStatuses"][1]
    database.update(
        ready=False,
        image=UNTRUSTED,
        imageID=UNTRUSTED,
        state={
            "terminated": {"exitCode": 127, "reason": "Error", "message": UNTRUSTED}
        },
    )
    with pytest.raises(ProbeError):
        sandbox.execute_probe("inspect")
    saved = deepcopy(sandbox.record["first_failure"])
    assert saved["stage"] == "pod-admission"
    assert saved["pod_status"]["containers"]["database"] == {
        "ready": False,
        "restart_count": 0,
        "state": "terminated",
        "exit_code": 127,
        "reason": "Error",
    }
    sandbox.cleanup()
    assert sandbox.record["first_failure"] == saved
    assert sandbox.record["pre_cleanup_pod_status"] == saved["pod_status"]
    assert UNTRUSTED not in sandbox.path.read_text()
    assert not api.armed, (
        "test_container_exit_is_retained_before_cleanup_without_raw_status: expected no api.armed"
    )


@pytest.mark.parametrize(
    "status",
    [
        {"phase": UNTRUSTED, "containerStatuses": [{"name": UNTRUSTED}]},
        {
            "containerStatuses": [
                {
                    "name": "database",
                    "ready": UNTRUSTED,
                    "restartCount": UNTRUSTED,
                    "state": UNTRUSTED,
                }
            ]
        },
        {
            "containerStatuses": [
                {
                    "name": "database",
                    "state": {
                        "terminated": {
                            "reason": UNTRUSTED,
                            "exitCode": True,
                            "message": UNTRUSTED,
                        }
                    },
                }
            ]
        },
        {"containerStatuses": UNTRUSTED},
        {"containerStatuses": [{"name": "database", "state": {"unknown": UNTRUSTED}}]},
    ],
    ids=[
        "unknown-name",
        "invalid-state",
        "unknown-reason",
        "invalid-statuses",
        "unknown-state",
    ],
)
def test_container_diagnostics_reject_arbitrary_fields(status: dict[str, Any]) -> None:
    result = fixture.pod_diagnostics({"status": status})
    assert set(result["containers"]) == {"runtime", "database"}
    assert result["containers"]["database"]["exit_code"] is None
    assert UNTRUSTED not in json.dumps(result)


def test_supervision_loss_is_recorded_and_reraised_without_extra_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, api, case_dir, deadline = setup_run(tmp_path, monkeypatch)
    sandbox = fixture.Sandbox(api, target(), case_dir, deadline, lambda: None)
    sandbox.create()
    sandbox.admit()
    calls: list[str] = []

    def lost(*arguments: str, **kwargs: Any) -> str:
        calls.append(arguments[0])
        raise ProcessSupervisionLost(UNTRUSTED)

    monkeypatch.setattr(api, "call", lost)
    with pytest.raises(ProcessSupervisionLost):
        sandbox.execute_probe("inspect")
    assert calls == ["exec"], (
        "diagnostics cannot issue follow-up commands after supervision loss"
    )
    first = sandbox.record["first_failure"]
    assert first["error_type"] == "ProcessSupervisionLost"
    assert UNTRUSTED not in sandbox.path.read_text()


def test_probe_reports_environment_guard_without_secret_or_dynamic_type(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _, arguments, _ = probe_environment(tmp_path, monkeypatch)
    monkeypatch.setenv("GPU_FAULT_EXECUTION_TOKEN", UNTRUSTED)
    assert probe.main(argv("arm", arguments)) == 1
    output = capsys.readouterr()
    report = json.loads(output.out)
    assert report == {
        "case_id": CASE_ID,
        "verdict": "FAIL",
        "command": "arm",
        "stage": "environment",
        "reason": "ambient-authority",
        "error_type": "ProbeError",
    }
    assert UNTRUSTED not in output.out + output.err
    assert not probe.marker("arm").exists(), (
        'test_probe_reports_environment_guard_without_secret_or_dynamic_type: expected no probe.marker("arm").exists()'
    )


def test_probe_unknown_exception_does_not_expose_class_name_or_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _, arguments, _ = probe_environment(tmp_path, monkeypatch)

    def fail(*args: Any) -> dict[str, Any]:
        raise type(UNTRUSTED, (RuntimeError,), {})(UNTRUSTED)

    monkeypatch.setattr(probe, "run", fail)
    assert probe.main(argv("run", arguments)) == 1
    output = capsys.readouterr()
    report = json.loads(output.out)
    assert report["command"] == report["stage"] == "run"
    assert report["error_type"] == "Exception"
    assert report["reason"] == "unclassified-error"
    assert UNTRUSTED not in output.out + output.err


def test_prepare_timeout_reports_known_reason_without_database_error_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import psycopg

    value, arguments, _ = probe_environment(tmp_path, monkeypatch)
    probe.arm(value)

    class ExpiredBudget:
        def __init__(self, seconds: int) -> None:
            assert seconds == 60

        def remaining(self) -> float:
            raise ProbeError("isolated process budget expired")

    def unavailable() -> Any:
        raise psycopg.OperationalError(UNTRUSTED)

    monkeypatch.setattr(probe, "Budget", ExpiredBudget)
    monkeypatch.setattr(probe, "connect", unavailable)
    assert probe.main(argv("prepare", arguments)) == 1
    output = capsys.readouterr()
    report = json.loads(output.out)
    assert report["command"] == report["stage"] == "prepare"
    assert report["reason"] == "process-budget-expired"
    assert report["error_type"] == "ProbeError"
    assert not (probe.WORK / "prepared.json").exists(), (
        'test_prepare_timeout_reports_known_reason_without_database_error_text: expected no (probe.WORK / "prepared.json").exists()'
    )
    assert UNTRUSTED not in output.out + output.err
