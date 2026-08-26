from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Literal

from gpu_fault.admin_bootstrap_common import BootstrapError


FinalSnapshotPolicy = Literal["retain", "skip"]


@dataclass(frozen=True)
class CommandResult:
    stdout: str
    stderr: str
    returncode: int


def run_command(arguments: list[str]) -> CommandResult:
    completed = subprocess.run(
        arguments,
        text=True,
        capture_output=True,
    )
    return CommandResult(
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
        returncode=completed.returncode,
    )


def matches_not_found(
    result: CommandResult,
    patterns: Iterable[str],
) -> bool:
    message = result.stdout + "\n" + result.stderr
    return any(pattern in message for pattern in patterns)


def checked_command(
    arguments: list[str],
    *,
    not_found: Iterable[str] = (),
) -> str | None:
    result = run_command(arguments)
    if result.returncode == 0:
        return result.stdout.strip()
    if matches_not_found(result, not_found):
        return None
    raise BootstrapError(
        f"command failed ({result.returncode}): {' '.join(arguments[:3])}: "
        f"{result.stderr.strip()}"
    )


def json_command(
    arguments: list[str],
    *,
    not_found: Iterable[str] = (),
) -> dict[str, Any] | None:
    output = checked_command(
        [*arguments, "--output", "json"],
        not_found=not_found,
    )
    return json.loads(output) if output is not None else None


def wait_until(
    predicate: Callable[[], bool],
    *,
    description: str,
    timeout_seconds: float = 900,
    interval_seconds: float = 5,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(interval_seconds)
    raise BootstrapError(f"timed out waiting for {description}")
