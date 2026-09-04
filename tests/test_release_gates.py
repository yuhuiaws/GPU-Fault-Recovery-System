from __future__ import annotations

import subprocess

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
        "_run_group",
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
