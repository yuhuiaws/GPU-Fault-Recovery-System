"""The release engine pins its own interpreter directory onto every child's PATH.

Every deploy tool the engine drives (``apply-control-plane-role-split.sh`` and
its siblings) shells out to bare ``python3`` and imports ``gpu_fault``. The
admin CLI routes its children through ``effective_environment``; a driver that
builds ``rollout.Runner`` directly used to inherit the shell's ``python3``,
which on a plain deploy host is the system interpreter (2026-09-30: BOOT-020
failed its upgrade and the automatic rollback with ``ModuleNotFoundError``).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from gpu_fault_release import rollout

INTERPRETER_DIRECTORY = str(Path(sys.executable).absolute().parent)


@pytest.fixture
def captured_environment(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    seen: dict[str, Any] = {}

    def fake_run_command(
        arguments: list[str],
        *,
        input_text: str | None = None,
        capture: bool = True,
        environment: dict[str, str] | None = None,
        timeout_seconds: float | None = None,
    ) -> subprocess.CompletedProcess[str]:
        seen["arguments"] = list(arguments)
        seen["environment"] = environment
        return subprocess.CompletedProcess(arguments, 0, "", "")

    monkeypatch.setattr(rollout, "run_command", fake_run_command)
    return seen


def test_inherited_environment_gets_the_interpreter_directory_first(
    monkeypatch: pytest.MonkeyPatch, captured_environment: dict[str, Any]
) -> None:
    monkeypatch.setenv("PATH", os.pathsep.join(["/usr/bin", "/bin"]))
    monkeypatch.setenv("PYTHONHOME", "/nowhere")

    rollout.Runner().run(["true"])

    environment = captured_environment["environment"]
    assert environment is not None, "the engine must hand its children an explicit env"
    path = environment["PATH"].split(os.pathsep)
    assert path[0] == INTERPRETER_DIRECTORY, path
    assert path[1:] == ["/usr/bin", "/bin"], path
    assert "PYTHONHOME" not in environment, "PYTHONHOME would redirect python3"


def test_explicit_environment_keeps_its_values_and_gains_the_interpreter(
    captured_environment: dict[str, Any],
) -> None:
    rollout.Runner().run(
        ["true"], env={"PATH": "/usr/bin", "GPU_FAULT_ROLE_SPLIT_OUT_DIR": "/tmp/out"}
    )

    environment = captured_environment["environment"]
    assert environment["PATH"].split(os.pathsep) == [INTERPRETER_DIRECTORY, "/usr/bin"]
    assert environment["GPU_FAULT_ROLE_SPLIT_OUT_DIR"] == "/tmp/out"


def test_interpreter_directory_is_not_duplicated(
    captured_environment: dict[str, Any],
) -> None:
    rollout.Runner().run(
        ["true"], env={"PATH": os.pathsep.join(["/usr/bin", INTERPRETER_DIRECTORY])}
    )

    path = captured_environment["environment"]["PATH"].split(os.pathsep)
    assert path == [INTERPRETER_DIRECTORY, "/usr/bin"], path
