from __future__ import annotations

import json
import math
import re
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Literal

from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.deadlines import deadline_scope
from gpu_fault.admin.diagnostics import diagnostic_command, diagnostic_text
from gpu_fault.admin.execution import run_command as bounded_command
from gpu_fault.admin.process_supervisor import split_progress_lines

FinalSnapshotPolicy = Literal["retain", "skip"]
# AWS CLI 2.35 prefixes every error line with ``aws: [ERROR]: `` (live
# 2026-09-20: an uninstall treated DBClusterSnapshotNotFoundFault as a hard
# failure). Only that ERROR prefix is tolerated; other levels stay opaque.
_AWS_ERROR = re.compile(
    r"(?:aws: \[ERROR\]: )?An error occurred \(([A-Za-z0-9_.-]+)\)(?:\s.*|:.*)?",
    re.S,
)


@dataclass(frozen=True)
class CommandResult:
    stdout: str
    stderr: str
    returncode: int


def run_command(arguments: list[str]) -> CommandResult:
    try:
        completed = bounded_command(arguments)
    except (subprocess.TimeoutExpired, TimeoutError):
        raise BootstrapError(
            "AWS command exceeded its deployment time budget: "
            + diagnostic_command(arguments[:3])
        ) from None
    except OSError as exc:
        raise BootstrapError(
            f"cannot execute cleanup command ({type(exc).__name__}): "
            + diagnostic_command(arguments[:3])
        ) from None
    return CommandResult(
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
        returncode=completed.returncode,
    )


def command_error_text(stderr: str) -> str:
    """The command's own error text: wait heartbeats and blank lines removed.

    The supervisor already keeps its heartbeat out of a captured result (a live
    uninstall once read ``deployment-wait command=aws ...`` ahead of an exact
    NoSuchEntity error and aborted a wait for a role that was gone). Absence
    detection stays strict either way: after this normalisation the whole text
    must still be exactly one AWS error.
    """
    own, _heartbeats = split_progress_lines(stderr)
    return "\n".join(line for line in own.splitlines() if line.strip()).strip()


def matches_not_found(
    result: CommandResult,
    patterns: Iterable[str],
) -> bool:
    if result.returncode not in {254, 255}:
        return False
    message = command_error_text(result.stderr)
    match = _AWS_ERROR.fullmatch(message)
    # A bare code remains supported for command adapters; arbitrary prose and
    # stdout are never evidence of absence.
    code = match.group(1) if match else message
    return code in patterns


def helm_release_absent(result: CommandResult) -> bool:
    return (
        result.returncode == 1
        and command_error_text(result.stderr).lower() == "error: release: not found"
    )


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
        f"{diagnostic_text(result.stderr.strip())}"
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
    if output is None:
        return None
    try:
        document = json.loads(output)
    except ValueError:
        raise BootstrapError(
            "invalid JSON response from cleanup command: "
            + diagnostic_command(arguments[:3])
        ) from None
    if not isinstance(document, dict):
        raise BootstrapError(
            "cleanup command did not return a JSON object: "
            + diagnostic_command(arguments[:3])
        )
    return document


def wait_until(
    predicate: Callable[[], bool],
    *,
    description: str,
    timeout_seconds: float = 900,
    interval_seconds: float = 5,
) -> None:
    if not math.isfinite(interval_seconds) or interval_seconds <= 0:
        raise BootstrapError("cleanup polling interval must be finite and positive")
    try:
        with deadline_scope(description, timeout_seconds) as deadline:
            while True:
                deadline.remaining()
                complete = predicate()
                remaining = deadline.remaining()
                if complete:
                    return
                time.sleep(min(interval_seconds, remaining))
    except TimeoutError:
        raise BootstrapError(f"timed out waiting for {description}") from None
