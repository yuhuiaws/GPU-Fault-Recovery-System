from __future__ import annotations

import contextlib
import os
import sqlite3
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.admin import api_budget, execution


@pytest.mark.parametrize("hard", [False, True])
def test_exclusive_protocol_lock_cannot_start_a_command_after_its_deadline(
    tmp_path: Path, hard: bool
) -> None:
    marker = tmp_path / "late-command"
    with api_budget.deployment_api_budget():
        root = api_budget.budget_root()
        assert root is not None, "test did not establish its private API budget"
        with sqlite3.connect(
            root / "budget.sqlite3", check_same_thread=False
        ) as blocker:
            blocker.execute("BEGIN EXCLUSIVE")

            def release_lock() -> None:
                time.sleep(0.4)
                blocker.commit()

            releaser = threading.Thread(target=release_lock)
            releaser.start()
            started = time.monotonic()
            try:
                scope = (
                    execution.deployment_deadline(
                        "exclusive protocol lock", 0.1, recovery_seconds=0
                    )
                    if hard
                    else execution.deadline_scope("exclusive protocol lock", 0.1)
                )
                with (
                    scope,
                    pytest.raises(
                        execution.DeploymentDeadlineExceeded, match="deadline"
                    ),
                ):
                    execution.run_command(
                        [
                            sys.executable,
                            "-c",
                            f"from pathlib import Path; Path({str(marker)!r}).touch()",
                        ]
                    )
                elapsed = time.monotonic() - started
            finally:
                releaser.join(timeout=2)
            assert not releaser.is_alive(), "test ledger lock was not released"
            assert elapsed < 0.35, "protocol validation ignored the short deadline"
            assert not marker.exists(), "a command started after its absolute deadline"


def test_command_rechecks_deadline_after_environment_preparation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [100.0]
    calls: list[object] = []
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])

    def prepare(_environment: object) -> None:
        clock[0] = 102.0

    def execute(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((args, kwargs))
        return subprocess.CompletedProcess([], 0, "", "")

    monkeypatch.setattr(execution, "command_environment", prepare)
    monkeypatch.setattr(execution, "run_owned_command", execute)
    with execution.deadline_scope("environment preparation", 1):
        with pytest.raises(execution.DeploymentDeadlineExceeded, match="deadline"):
            execution.run_command(["not-started"])
    assert calls == [], "expired preparation reached the subprocess supervisor"


def test_relative_command_limit_is_exported_as_an_absolute_cutoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [100.0]
    captured: dict[str, Any] = {}
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])

    def prepare(_environment: object) -> None:
        clock[0] = 100.25

    def execute(*_args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured.update(kwargs)
        return subprocess.CompletedProcess([], 0, "", "")

    monkeypatch.setattr(execution, "command_environment", prepare)
    monkeypatch.setattr(execution, "run_owned_command", execute)
    execution.run_command(["test-command"], timeout_seconds=1)
    assert captured["expires_at"] == 101.0, "environment preparation renewed the budget"
    assert captured["timeout"] == 0.75, (
        "the supervisor received a stale relative timeout"
    )


def test_root_deadline_keeps_shared_interruption_scope_until_work_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []

    @contextlib.contextmanager
    def interruption_scope(*, wait_all: bool = False) -> Iterator[None]:
        events.append(("enter", wait_all))
        try:
            yield
        finally:
            deadline = execution.current_deadline()
            events.append(("exit", deadline.label if deadline else None))

    monkeypatch.setattr(execution, "interruption_scope", interruption_scope)
    with execution.deployment_deadline("shared task pool", 30):
        events.append("body")
    assert events == [("enter", True), "body", ("exit", "shared task pool")], (
        "the root deadline did not retain shared cancellation through task completion"
    )


def test_command_and_driver_only_allow_interruption_during_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admitted: list[bool] = []
    checked: list[bool] = []

    def safe(*, allow_interrupted: bool = False) -> None:
        checked.append(allow_interrupted)

    def execute(*_args: object, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        admitted.append(kwargs["allow_interrupted"])
        assert kwargs["expires_at"] is not None, "recovery lost its absolute deadline"
        return subprocess.CompletedProcess([], 0, "", "")

    monkeypatch.setattr(execution, "ensure_supervision_safe", safe)
    monkeypatch.setattr(execution, "run_owned_command", execute)
    execution.run_command(["normal-command"])
    execution.run_driver(["normal-driver"])
    with execution.cleanup_deadline("bounded cleanup"):
        execution.run_command(["cleanup-command"])
        execution.run_driver(["cleanup-driver"])
    assert checked == [False, False, True, True], "preparation bypassed cancellation"
    assert admitted == checked, "the supervisor received different recovery permissions"


def test_expired_budget_setup_restores_the_callers_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = {
        name: os.environ.get(name)
        for name in (api_budget.ROOT_ENV, api_budget.PHASE_ENV, "PATH")
    }
    monkeypatch.setenv(execution.DEADLINE_ENV, str(time.monotonic() - 1))
    with pytest.raises(execution.DeploymentDeadlineExceeded, match="deadline"):
        with api_budget.deployment_api_budget():
            pytest.fail("expired API budget setup entered its body")
    assert {name: os.environ.get(name) for name in original} == original, (
        "failed budget setup leaked its deleted temporary directory or shim PATH"
    )
