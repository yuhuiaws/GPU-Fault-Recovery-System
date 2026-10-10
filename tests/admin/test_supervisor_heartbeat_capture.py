"""The supervisor's wait heartbeat never becomes part of a captured command result.

In capture mode the supervisors and the command share one stderr pipe. A command
slower than the heartbeat interval used to return ``deployment-wait ...`` as the
first line of its own stderr, which corrupted every result a caller parses (live:
an uninstall's ``aws iam get-role`` poll on a role that was already deleted).
"""

from __future__ import annotations

import sys

import pytest

from gpu_fault.admin import process_supervisor
from gpu_fault.admin.diagnostics import DriverDiagnostics
from gpu_fault.admin.process_supervisor import (
    PROGRESS_LINE,
    run_owned_command,
    split_progress_lines,
)

HEARTBEAT = "deployment-wait command=aws state=running elapsed=30s remaining=89s\n"
ROLE_GONE = (
    "aws: [ERROR]: An error occurred (NoSuchEntity) when calling the GetRole "
    "operation: The role with name gpu-fault-example cannot be found.\n"
)


def test_split_takes_out_exact_heartbeats_and_keeps_every_other_byte() -> None:
    own, progress = split_progress_lines(f"{HEARTBEAT}\n{ROLE_GONE}")
    assert own == f"\n{ROLE_GONE}"
    assert progress == [HEARTBEAT]


@pytest.mark.parametrize(
    "text",
    [
        "",
        ROLE_GONE,
        "partial line without newline",
        f"glued {HEARTBEAT}",
        HEARTBEAT.rstrip("\n") + " trailing\n",
        "deployment-wait command=aws state=unknown elapsed=30s remaining=89s\n",
        "line one\r\nline two\n\n\tindented\n",
    ],
)
def test_split_leaves_text_without_an_exact_heartbeat_untouched(text: str) -> None:
    assert split_progress_lines(text) == (text, [])


def test_split_handles_cleanup_heartbeats_without_a_final_newline() -> None:
    cleanup = "deployment-wait command=command state=cleanup elapsed=61s remaining=0s"
    own, progress = split_progress_lines(f"{ROLE_GONE}{cleanup}")
    assert own == ROLE_GONE
    assert progress == [cleanup]


def test_captured_stderr_holds_only_the_commands_output(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(process_supervisor, "REPORT_INTERVAL_SECONDS", 0.2)
    result = run_owned_command(
        [
            sys.executable,
            "-c",
            "import sys, time; time.sleep(0.7); "
            f"sys.stderr.write({ROLE_GONE!r}); sys.stderr.flush(); sys.exit(254)",
        ],
        timeout=10,
        capture=True,
    )
    assert result.returncode == 254
    assert result.stderr == ROLE_GONE
    assert result.stdout == ""
    forwarded = capsys.readouterr().err.splitlines()
    heartbeats = [line for line in forwarded if PROGRESS_LINE.fullmatch(line)]
    assert heartbeats, f"no heartbeat reached the operator channel: {forwarded}"
    assert heartbeats == forwarded, "something other than heartbeats was forwarded"
    assert any(" state=running " in line for line in heartbeats), heartbeats


def test_streaming_mode_forwards_the_same_heartbeat_shape(
    capsys: pytest.CaptureFixture[str],
) -> None:
    diagnostics = DriverDiagnostics()
    diagnostics.feed(HEARTBEAT)
    assert capsys.readouterr().err == HEARTBEAT, "heartbeat was not streamed live"
    diagnostics.feed(ROLE_GONE)
    assert capsys.readouterr().err == "", "ordinary stderr was streamed unredacted"
    diagnostics.finish()
    assert capsys.readouterr().err == ROLE_GONE


@pytest.mark.parametrize("interval", [0, -1, float("nan"), float("inf")])
def test_supervisor_rejects_an_invalid_heartbeat_interval(interval: float) -> None:
    with pytest.raises(ValueError, match="report interval"):
        process_supervisor.supervise(
            ["example"], owner=1, expires=1.0, pass_fds=(), report_interval=interval
        )
