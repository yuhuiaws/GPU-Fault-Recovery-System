from __future__ import annotations

import re
import subprocess
import threading
from pathlib import Path

import pytest

from scripts import run_release_gates, run_static_gates


def test_release_gate_parallelism_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GPU_FAULT_RELEASE_GATE_PARALLELISM", "2")
    assert run_release_gates.gate_parallelism() == 2

    monkeypatch.setenv("GPU_FAULT_RELEASE_GATE_PARALLELISM", "4")
    with pytest.raises(run_release_gates.ReleaseGateError, match="within 1..3"):
        run_release_gates.gate_parallelism()


def test_static_gate_parallelism_and_groups_are_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GPU_FAULT_STATIC_GATE_PARALLELISM", "7")
    assert run_static_gates.gate_parallelism() == 7
    assert set(run_static_gates.gate_groups("python3")) == {
        "architecture",
        "compile",
        "contracts",
        "deployment",
        "docs",
        "mypy",
        "ruff",
        "safety",
        "shell",
        "yaml",
    }


def test_static_gate_failure_is_aggregated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GPU_FAULT_STATIC_GATE_PARALLELISM", "10")
    monkeypatch.setattr(
        run_static_gates,
        "run_gate_group",
        lambda name, *_args, **_kwargs: 1 if name == "yaml" else 0,
    )

    with pytest.raises(run_static_gates.StaticGateError, match="yaml=1"):
        run_static_gates.run_static_gates("python3")


def test_release_gates_build_artifacts_only_after_parallel_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gates: list[str] = []
    commands: list[list[str]] = []
    monkeypatch.setenv(
        "GPU_FAULT_TEST_POSTGRES_URL", "postgresql://postgres@127.0.0.1:5432/test"
    )
    monkeypatch.setenv("GPU_FAULT_RELEASE_GATE_PARALLELISM", "3")
    monkeypatch.setattr(
        run_release_gates,
        "_stream_gate",
        lambda name, *_args, **_kwargs: gates.append(name) or 0,
    )
    monkeypatch.setattr(
        run_release_gates.subprocess,
        "run",
        lambda command, **_kwargs: (
            commands.append(list(command)) or subprocess.CompletedProcess(command, 0)
        ),
    )

    run_release_gates.run_release_gates("python3")

    assert set(gates) == {"postgres", "pytest", "static"}
    assert commands == [
        ["make", "artifact-check", "PYTHON=python3"],
        ["make", "python-cache-clean", "PYTHON=python3"],
    ]


def test_release_gate_failure_blocks_artifact_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "GPU_FAULT_TEST_POSTGRES_URL", "postgresql://postgres@127.0.0.1:5432/test"
    )
    monkeypatch.setenv("GPU_FAULT_RELEASE_GATE_PARALLELISM", "3")
    monkeypatch.setattr(
        run_release_gates,
        "_stream_gate",
        lambda name, *_args, **_kwargs: 1 if name == "postgres" else 0,
    )
    monkeypatch.setattr(
        run_release_gates.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail(
            "artifact build ran after a parallel gate failure"
        ),
    )

    with pytest.raises(run_release_gates.ReleaseGateError, match="postgres=1"):
        run_release_gates.run_release_gates("python3")


def test_default_check_runs_static_then_parallel_artifact_and_pytest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []
    parallel: list[tuple[dict[str, list[str]], int]] = []
    monkeypatch.setattr(
        run_release_gates.subprocess,
        "run",
        lambda command, **_kwargs: (
            commands.append(list(command)) or subprocess.CompletedProcess(command, 0)
        ),
    )
    monkeypatch.setattr(
        run_release_gates,
        "_run_parallel",
        lambda gates, **kwargs: (
            parallel.append((gates, int(kwargs["max_workers"]))) or {}
        ),
    )

    run_release_gates.run_check_gates("python3")

    assert commands == [
        ["make", "check-static", "PYTHON=python3"],
        ["make", "python-cache-clean", "PYTHON=python3"],
    ]
    assert parallel == [
        (
            {
                "artifact": ["make", "artifact-check", "PYTHON=python3"],
                "pytest": ["make", "test-parallel-release", "PYTHON=python3"],
            },
            2,
        )
    ]


class _FinishedProcess:
    """A subprocess that produced no output and exited with ``status``."""

    def __init__(self, status: int) -> None:
        self.stdout = iter(())
        self._status = status

    def wait(self) -> int:
        return self._status


def test_a_static_gate_reports_its_own_wall_time(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """`make check` runs ten gates in parallel and never said which was the pole.

    Wall time is the slowest gate, so a per-gate duration is the only number that
    points at what to go fix; and a gate that bundles four commands needs its slow
    command named too, or "contracts took 90s" is as far as the log goes. Fast
    steps stay unannotated so the ten pass lines do not turn into forty.
    """

    monkeypatch.setattr(run_static_gates, "SLOW_COMMAND_SECONDS", 0.0)
    monkeypatch.setattr(
        run_static_gates.subprocess,
        "Popen",
        lambda *_args, **_keywords: _FinishedProcess(0),
    )
    group = run_static_gates.GateGroup(
        commands=(
            ("/usr/bin/python3", "-m", "ruff", "check", "src"),
            ("/usr/bin/python3", "scripts/check-assert-messages.py"),
        )
    )

    status = run_static_gates.run_gate_group(
        "contracts", group, cache_root=tmp_path, output_lock=threading.Lock()
    )

    assert status == 0
    err = capsys.readouterr().err
    assert "static-gates: contracts step took" in err
    assert "python3 ruff check src" in err, (
        "the interpreter path is the same for every gate, so the tool has to show"
    )
    assert re.search(r"static-gates: contracts passed in \d+\.\ds", err), err


def test_a_failing_static_gate_names_the_command_that_failed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """A gate is several commands, and the exit status alone identifies none.

    The failing command's own output is already interleaved with nine other
    gates', so the line that reports the status is where the reader finds out
    which command it belonged to.
    """

    monkeypatch.setattr(
        run_static_gates.subprocess,
        "Popen",
        lambda *_args, **_keywords: _FinishedProcess(2),
    )
    group = run_static_gates.GateGroup(
        commands=(("/usr/bin/python3", "scripts/check-deploy-layout.py"),)
    )

    status = run_static_gates.run_gate_group(
        "deployment", group, cache_root=tmp_path, output_lock=threading.Lock()
    )

    assert status == 2
    assert "check-deploy-layout.py" in capsys.readouterr().err
