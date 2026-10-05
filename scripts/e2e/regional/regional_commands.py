"""Supervised local commands and private diagnostics for acceptance fixtures."""

from __future__ import annotations

import math
import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath

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


#: Positional subcommand words a CLI takes before its values; absent means none.
SUBCOMMAND_WORDS = {"aws": 2, "gpu-fault-admin": 2, "helm": 1}
SUBCOMMAND_PATTERN = re.compile(r"[a-z][a-z0-9-]{0,40}")


def command_step_label(command: Sequence[str]) -> str:
    """The step a failed command belongs to, without any of its values.

    The CLI and the subcommand words (``aws sagemaker describe-cluster``,
    ``kubectl get``) and nothing else: no names, flags, paths or payloads, so the
    label is safe beside redacted output. Live 2026-10-04: several cases
    stopped with only ``command failed (1): <sensitive output redacted>``, which
    names neither the command class nor the step.
    """

    words = [item for item in command if not item.startswith("-")]
    if not words:
        return ""
    if words[0] == "kubectl":
        # ``kubectl --kubeconfig P --context C -n NS exec -i POD -- ...``: the
        # kubeconfig, context and namespace values precede the verb, so the verb
        # is the first word that is a kubectl verb; nothing after it is kept.
        verbs = {
            "exec",
            "get",
            "auth",
            "describe",
            "logs",
            "config",
            "version",
            "create",
            "apply",
            "delete",
            "patch",
            "annotate",
            "label",
            "api-resources",
            "wait",
            "rollout",
            "cordon",
            "uncordon",
            "drain",
            "taint",
        }
        verb = next((item for item in words[1:] if item in verbs), None)
        return "kubectl" if verb is None else f"kubectl {verb}"
    # Only CLIs whose leading positionals are subcommand words keep any; for
    # every other program (``python -c ... VALUE``) the arguments are values and
    # the label is the program name alone. 2026-10-05: the first three words of
    # an interpreter command were its script and an argument.
    cli = PurePosixPath(words[0]).name
    label = [cli]
    for item in words[1 : 1 + SUBCOMMAND_WORDS.get(cli, 0)]:
        if not SUBCOMMAND_PATTERN.fullmatch(item):
            break
        label.append(item)
    return " ".join(label)


class RegionalCommandFailed(RegionalFixtureError):
    def __init__(
        self, returncode: int, stderr: str, command: Sequence[str] | None = None
    ) -> None:
        self.replica_disappeared = replica_vanished(RuntimeError(stderr))
        self.step = command_step_label(command) if command else ""
        message = f"command failed ({returncode}): " + diagnostic_text(
            stderr, sensitive=True
        )
        if self.step:
            message += f" (step: {self.step})"
        super().__init__(message)


def run_fixture_command(
    command: Sequence[str],
    *,
    input_text: str | None = None,
    check: bool = True,
    timeout: float = 300,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    isolated_postgres_url: str | None = None,
    identity_root: Path | None = None,
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
            identity_root=identity_root,
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
        raise RegionalCommandFailed(completed.returncode, completed.stderr, command)
    return completed
