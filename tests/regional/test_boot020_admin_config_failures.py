"""BOOT-020's public AdminConfig drill when the CLI refuses a step.

Live 2026-10-29 (BOOT-020 attempt 24) ``gpu-fault-admin config`` refused the
apply -- the site was still on an interrupted drill candidate -- and the drill
raised a bare "command failed; preserve its pending state and audit": no
reason, no log, and there was no pending state. It also left its own edit in
``admin-config.yaml``, which the next run refuses as an administrator edit.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from typing import Any

import pytest

from gpu_fault.admin.config import (
    load_desired_admin_config,
    persist_desired_admin_config,
)
from gpu_fault.admin.config_file import load_admin_config_file, write_admin_config_file
from scripts.e2e.regional import boot020_admin_config as admin_config
from tests.regional import test_full01_boot_lifecycle_edges as edges

config_io = edges.config_io
ConfigIO = edges.ConfigIO

LOG = "/state/logs/mutating/config-unit.log"
REFUSAL = (
    "gpu-fault-admin: current signed release differs from the live regional release"
)
CLI_STDERR = "\n".join(
    (
        f"gpu-fault-admin: logging to {LOG}",
        "admin-phase-begin unit command=config phase=live-release",
        "admin-phase-end unit command=config phase=live-release status=failed",
        REFUSAL,
        f"gpu-fault-admin: full output in {LOG}",
    )
)


def _cli_says(
    config_io: ConfigIO,
    monkeypatch: pytest.MonkeyPatch,
    stderr: str,
    *,
    during: Callable[[], object] | None = None,
) -> None:
    """A failing CLI step prints ``stderr``; ``during`` runs inside that step."""

    def command(arguments: list[str], **options: Any) -> Any:
        completed = config_io.command(arguments, **options)
        if not completed.returncode:
            return completed
        if during is not None:
            during()
        return subprocess.CompletedProcess(
            completed.args, completed.returncode, completed.stdout, stderr
        )

    monkeypatch.setattr(admin_config, "run_driver", command)


@pytest.mark.parametrize("pending", [False, True])
def test_a_refused_apply_names_what_the_cli_said_and_where_it_logged(
    config_io: ConfigIO, monkeypatch: pytest.MonkeyPatch, pending: bool
) -> None:
    config_io.exit_codes["apply"] = 2
    if pending:
        config_io.pending_at.add("apply")
    _cli_says(config_io, monkeypatch, CLI_STDERR)

    with pytest.raises(admin_config.PublicConfigCommandError) as failure:
        config_io.run()

    error = failure.value
    message = str(error)
    assert isinstance(error, ValueError), "callers that catch ValueError still do"
    assert (error.step, error.returncode) == ("apply", 2), message
    assert error.admin_message == REFUSAL and error.log_path == LOG, message
    assert REFUSAL in message and f"full output in {LOG}" in message, message
    assert ("preserve its pending state" in message) is pending, message
    assert ("no pending apply was recorded" in message) is not pending, message
    assert config_io.calls == ["dry-run", "apply"], "nothing applied, nothing to undo"
    assert load_desired_admin_config(config_io.root) == config_io.before
    # Without a pending apply the drill's own edit goes back to the committed
    # config; a pending apply keeps its target where the CLI resumes it from.
    assert load_admin_config_file(config_io.root / "admin-config.yaml") == (
        config_io.desired if pending else config_io.before
    )


def test_the_next_run_is_not_refused_by_the_drills_own_leftover_edit(
    config_io: ConfigIO, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_io.exit_codes["apply"] = 2
    _cli_says(config_io, monkeypatch, CLI_STDERR)
    with pytest.raises(admin_config.PublicConfigCommandError, match="exit 2"):
        config_io.run()

    config_io.exit_codes.clear()
    config_io.calls.clear()
    result = config_io.run()

    assert result["restored_config_sha256"] == config_io.before.sha256(), result
    assert config_io.calls == ["dry-run", "apply", "noop", "restore"]


@pytest.mark.parametrize("change", ["file-edited", "config-committed"])
def test_an_edit_or_commit_made_during_the_failed_step_is_left_alone(
    config_io: ConfigIO, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    editable = config_io.root / "admin-config.yaml"
    other = config_io.before.patched(
        {
            "capacity": {
                "controlWorkerReplicas": config_io.before.capacity.control_worker_replicas
                - 2
            }
        }
    )
    if change == "file-edited":
        expected = other

        def during() -> object:
            return write_admin_config_file(editable, other, overwrite=True)

    else:
        expected = config_io.desired

        def during() -> object:
            return persist_desired_admin_config(
                config_io.root, config=config_io.desired, source="unit-commit"
            )

    config_io.exit_codes["apply"] = 2
    _cli_says(config_io, monkeypatch, CLI_STDERR, during=during)

    with pytest.raises(admin_config.PublicConfigCommandError):
        config_io.run()

    assert load_admin_config_file(editable) == expected, (
        "the drill overwrote a file it no longer owned"
    )


def test_failure_report_reads_only_the_last_cli_line_and_its_log() -> None:
    long_line = "gpu-fault-admin: " + "x" * 1000
    log, message = admin_config.admin_failure_report(
        "\n".join(
            (
                "gpu-fault-admin: logging to /a.log",
                "gpu-fault-admin: a note the CLI attached first",
                "+ kubectl get configmap",
                long_line,
                "gpu-fault-admin: full output in /a.log",
            )
        )
    )
    assert log == "/a.log"
    assert message is not None and message.startswith("gpu-fault-admin: xxx")
    assert len(message) == admin_config.ADMIN_MESSAGE_LIMIT and message.endswith(
        "..."
    ), message
    assert admin_config.admin_failure_report(REFUSAL) == (None, REFUSAL)
    assert admin_config.admin_failure_report("Traceback\nValueError: x") == (None, None)
    silent = str(
        admin_config.PublicConfigCommandError(
            "dry-run", 7, stderr="owned failure", pending=False
        )
    )
    assert "printed no gpu-fault-admin line" in silent, silent
    assert "named no log file" in silent and "exit 7" in silent, silent
