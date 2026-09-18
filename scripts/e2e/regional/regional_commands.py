"""Supervised local commands and private diagnostics for acceptance fixtures."""

from __future__ import annotations

import math
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

from gpu_fault.admin.deadlines import DeploymentDeadlineExceeded
from gpu_fault.admin.diagnostics import diagnostic_text
from gpu_fault.admin.execution import run_command
from gpu_fault.admin.process_supervisor import ProcessSupervisionLost
from scripts.e2e.regional.acceptance_runner_common import replica_vanished
from scripts.e2e.regional.acceptance_supervision import record_supervision_loss
from scripts.e2e.regional.focused_pytest import prepare_focused_pytest


class RegionalFixtureError(RuntimeError):
    pass


class RegionalCommandTimeout(RegionalFixtureError):
    """An ambiguous command timeout must never be retried as a failed read."""

    def __init__(self, command: Sequence[str], timeout: float | None) -> None:
        super().__init__(f"command timed out after {timeout}s")
        self.command = list(command)
        self.timeout = timeout


class RegionalCommandFailed(RegionalFixtureError):
    def __init__(self, returncode: int, stderr: str) -> None:
        self.replica_disappeared = replica_vanished(RuntimeError(stderr))
        super().__init__(
            f"command failed ({returncode}): " + diagnostic_text(stderr, sensitive=True)
        )


def run_fixture_command(
    command: Sequence[str],
    *,
    input_text: str | None = None,
    check: bool = True,
    timeout: float = 300,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    isolated_postgres_url: str | None = None,
) -> subprocess.CompletedProcess[str]:
    if not command or any(
        not isinstance(item, str) or "\0" in item for item in command
    ):
        raise RegionalFixtureError("command arguments are empty or malformed")
    if isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
        raise RegionalFixtureError("command timeout must be finite and positive")
    try:
        with prepare_focused_pytest(
            command,
            cwd=cwd,
            environment=env,
            isolated_postgres_url=isolated_postgres_url,
        ) as focused:
            completed: subprocess.CompletedProcess[str] = run_command(
                focused.command if focused else command,
                input_text=input_text,
                environment=focused.environment if focused else env,
                cwd=cwd,
                timeout_seconds=timeout,
            )
            if focused:
                completed = focused.verify(completed)
    except ProcessSupervisionLost:
        try:
            record_supervision_loss()
        except BaseException:
            raise ProcessSupervisionLost(
                "command supervision was lost and the recovery marker could not be persisted"
            ) from None
        raise
    except (subprocess.TimeoutExpired, DeploymentDeadlineExceeded):
        raise RegionalCommandTimeout(command, timeout) from None
    except OSError as exc:
        raise RegionalFixtureError(
            f"command could not start ({type(exc).__name__}, errno={exc.errno})"
        ) from None
    if check and completed.returncode:
        raise RegionalCommandFailed(completed.returncode, completed.stderr)
    return completed
