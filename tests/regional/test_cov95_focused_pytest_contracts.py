from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import pytest

from gpu_fault.admin.deadlines import DeploymentDeadlineExceeded
from gpu_fault.admin.process_supervisor import ProcessSupervisionLost
from scripts.e2e.regional import focused_pytest, regional_commands
from tools import pytest_result_identity

IDENTITY = "a" * 64
NODEID = "test_unit.py::test_control"
COMMAND = ["python3", "-m", "pytest", "test_unit.py"]


def receipt() -> dict:
    return {
        "schema_version": 1,
        "source_identity": IDENTITY,
        "session": {
            "source_identity": IDENTITY,
            "exitstatus": 0,
            "collected_nodeids": [NODEID],
            "discovered_nodeids": [NODEID],
            "collection_errors": [],
            "collection_skips": [],
        },
        "records": {
            NODEID: {
                "status": "PASS",
                "phases": {"setup": "passed", "call": "passed", "teardown": "passed"},
            }
        },
    }


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ([], False),
        (["pytest"], True),
        (["/tools/py.test", "-q"], True),
        (["python3", "-m", "pytest"], True),
        (["/venv/python3.12", "-B", "-X", "dev", "-W", "error", "-m", "pytest"], True),
        (["pypy3", "-mpytest"], True),
        (["python", "-m"], False),
        (["python", "-m", "unittest"], False),
        (["python", "-munittest"], False),
        (["python", "-c", "print('pytest')", "-m", "pytest"], False),
        (["python", "script.py", "-m", "pytest"], False),
        (["python", "--", "-m", "pytest"], False),
        (["python", "-B"], False),
        (["kubectl", "exec", "pod", "--", "python", "-m", "pytest"], False),
        (["sh", "-c", "python -m pytest"], False),
    ],
)
def test_only_direct_local_pytest_is_given_a_local_receipt(command, expected):
    assert focused_pytest.is_local_pytest(command) is expected, (
        "a remote exec or ordinary Python script must not be rewritten as local pytest",
        command,
    )


@pytest.fixture
def transport(monkeypatch: pytest.MonkeyPatch):
    state = {"value": receipt(), "returncode": 0, "reports": [], "calls": []}
    monkeypatch.setattr(
        pytest_result_identity, "source_identity", lambda root: IDENTITY
    )

    def execute(command, **kwargs):
        state["calls"].append((command, kwargs))
        path = Path(kwargs["environment"]["PYTEST_GPU_FAULT_CASE_REPORT"])
        state["reports"].append(path)
        if state["value"] is not None:
            path.write_text(json.dumps(state["value"]), encoding="utf-8")
        if state.get("source_changed"):
            monkeypatch.setattr(
                pytest_result_identity, "source_identity", lambda root: "b" * 64
            )
        return subprocess.CompletedProcess(
            command, state["returncode"], "child output", "child diagnostic"
        )

    monkeypatch.setattr(regional_commands, "run_command", execute)
    return state


@pytest.mark.parametrize(
    "defect",
    [
        "missing",
        "malformed",
        "foreign-source",
        "source-changed",
        "collector-skip",
        "unexecuted-discovery",
        "nonrecord",
        "failed",
        "missing-phase",
    ],
)
def test_missing_or_incomplete_receipts_reject_a_zero_exit(
    defect: str, transport, tmp_path: Path
) -> None:
    value = copy.deepcopy(transport["value"])
    if defect == "missing":
        value = None
    elif defect == "malformed":
        value = []
    elif defect == "foreign-source":
        value["source_identity"] = "b" * 64
    elif defect == "source-changed":
        transport["source_changed"] = True
    elif defect == "collector-skip":
        value["session"]["collection_skips"] = ["test_optional.py"]
    elif defect == "unexecuted-discovery":
        value["session"]["discovered_nodeids"].append("test_unit.py::test_second")
    elif defect == "nonrecord":
        value["records"][NODEID] = None
    elif defect == "failed":
        value["records"][NODEID]["status"] = "FAIL"
    else:
        del value["records"][NODEID]["phases"]["teardown"]
    transport["value"] = value
    result = regional_commands.run_fixture_command(
        COMMAND, cwd=tmp_path, env={"PATH": "/unit/bin"}, check=False, timeout=11
    )
    assert result.returncode == 1, "an incomplete receipt cannot authorize preflight"
    assert "focused pytest evidence rejected" in result.stderr, result.stderr
    assert result.stdout == "child output", "failed test diagnostics must be retained"
    assert transport["calls"][0][1]["timeout_seconds"] == 11, (
        "receipt collection must preserve the supervised command deadline"
    )
    assert all(not path.parent.exists() for path in transport["reports"]), (
        "temporary receipt directories must be cleaned on failure"
    )


def test_failed_pytest_retains_its_original_exit_and_private_receipt_cleanup(
    transport, tmp_path: Path
) -> None:
    transport["returncode"] = 7
    transport["value"] = None
    result = regional_commands.run_fixture_command(
        COMMAND, cwd=tmp_path, check=False, timeout=4
    )
    assert result.returncode == 7 and result.stderr == "child diagnostic", result
    assert all(not path.parent.exists() for path in transport["reports"]), (
        "a process failure still owns and removes its receipt directory"
    )


def test_checked_call_rejects_failed_receipt_even_when_pytest_exited_zero(
    transport, tmp_path: Path
) -> None:
    transport["value"] = None
    with pytest.raises(regional_commands.RegionalCommandFailed):
        regional_commands.run_fixture_command(COMMAND, cwd=tmp_path, timeout=4)
    assert len(transport["calls"]) == 1, "an invalid receipt is not retried"


def test_nonpytest_command_preserves_its_environment_and_avoids_pytest_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = []

    def unexpected_identity(root):
        raise AssertionError(
            "ordinary commands must not request pytest source identity"
        )

    def execute(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, "ordinary output", "")

    monkeypatch.setattr(pytest_result_identity, "source_identity", unexpected_identity)
    monkeypatch.setattr(regional_commands, "run_command", execute)
    command = ["kubectl", "exec", "owned-probe", "--", "python3", "-m", "pytest"]
    environment = {"KUBECONFIG": str(tmp_path / "private-reference")}
    result = regional_commands.run_fixture_command(
        command, cwd=tmp_path, env=environment, timeout=4
    )
    assert result.stdout == "ordinary output", result
    assert calls[0][0] == command and calls[0][1]["environment"] == environment, (
        "local evidence guards must not strip a remote command's connection context",
        calls,
    )


@pytest.mark.parametrize("command", [[], ["python3", "\0"], ["python3", 1]])
def test_invalid_command_never_reaches_pytest_or_the_supervisor(command, transport):
    with pytest.raises(regional_commands.RegionalFixtureError, match="malformed"):
        regional_commands.run_fixture_command(command)
    assert transport["calls"] == [], "malformed arguments must fail before execution"


@pytest.mark.parametrize("timeout", [True, 0, -1, float("nan"), float("inf")])
def test_invalid_budget_never_starts_focused_pytest(timeout, transport):
    with pytest.raises(
        regional_commands.RegionalFixtureError, match="finite and positive"
    ):
        regional_commands.run_fixture_command(COMMAND, timeout=timeout)
    assert transport["calls"] == [], "invalid budgets must fail before execution"


@pytest.mark.parametrize(
    "failure", ["spawn", "timeout", "deadline", "supervision", "marker"]
)
def test_focused_receipts_preserve_transport_failure_and_supervision_contract(
    failure: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    reports = []
    markers = []
    monkeypatch.setattr(
        pytest_result_identity, "source_identity", lambda root: IDENTITY
    )
    raised = {
        "spawn": OSError(5, "unit-private-startup-diagnostic"),
        "timeout": subprocess.TimeoutExpired(COMMAND, 3),
        "deadline": DeploymentDeadlineExceeded("unit deadline"),
        "supervision": ProcessSupervisionLost("unit supervision lost"),
        "marker": ProcessSupervisionLost("unit supervision lost"),
    }[failure]

    def execute(command, **kwargs):
        reports.append(Path(kwargs["environment"]["PYTEST_GPU_FAULT_CASE_REPORT"]))
        raise raised

    def mark():
        markers.append("lost")
        if failure == "marker":
            raise OSError(5, "unit-private-marker-diagnostic")

    monkeypatch.setattr(regional_commands, "run_command", execute)
    monkeypatch.setattr(regional_commands, "record_supervision_loss", mark)
    expected = (
        ProcessSupervisionLost
        if failure in {"supervision", "marker"}
        else regional_commands.RegionalCommandTimeout
        if failure in {"timeout", "deadline"}
        else regional_commands.RegionalFixtureError
    )
    with pytest.raises(expected) as stopped:
        regional_commands.run_fixture_command(COMMAND, cwd=tmp_path, timeout=3)
    assert len(reports) == 1 and not reports[0].parent.exists(), (
        "failed transport must clean only its temporary pytest receipt directory"
    )
    assert markers == (["lost"] if failure in {"supervision", "marker"} else []), (
        "the supervisor's lost-ownership marker must not be skipped"
    )
    assert "unit-private" not in str(stopped.value), (
        "process and marker failures must not disclose raw private diagnostics"
    )
