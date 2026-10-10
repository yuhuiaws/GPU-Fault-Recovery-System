"""A command slower than the supervisor's wait heartbeat must still prove absence.

Live uninstall: an ``aws iam get-role`` poll on a role that was already deleted
took longer than the heartbeat interval, the captured stderr started with
``deployment-wait command=aws ...`` and ``wait_absent`` aborted the phase with
"resource verification failed" although the service had answered NoSuchEntity.
"""

from __future__ import annotations

import pytest

from gpu_fault.admin.aws_commands import (
    CommandResult,
    command_error_text,
    helm_release_absent,
    matches_not_found,
)

HEARTBEAT = "deployment-wait command=aws state=running elapsed=30s remaining=89s"
ROLE_GONE = (
    "aws: [ERROR]: An error occurred (NoSuchEntity) when calling the GetRole "
    "operation: The role with name "
    "gpu-fault-us-west-2-control-plane-gpu-fault-solut-985fd-72db0ca1 "
    "cannot be found."
)
LIVE_STDERR = f"{HEARTBEAT}\n\n{ROLE_GONE}\n"


def test_live_get_role_absence_behind_a_wait_heartbeat_is_recognised() -> None:
    result = CommandResult(stdout="", stderr=LIVE_STDERR, returncode=254)
    assert matches_not_found(result, ("NoSuchEntity",)) is True, LIVE_STDERR


@pytest.mark.parametrize(
    "stderr",
    [
        f"{HEARTBEAT}\n",
        f"{HEARTBEAT}\n\n{HEARTBEAT}\n",
        f"{HEARTBEAT}\n\nsome prose mentioning NoSuchEntity\n",
        f"{HEARTBEAT}\n\naws: [ERROR]: An error occurred (AccessDenied) when "
        "calling the GetRole operation: no\n",
        f"{HEARTBEAT}\n\naws: [WARN]: An error occurred (NoSuchEntity) when "
        "calling the GetRole operation: x\n",
        # A heartbeat glued to other text is not a heartbeat; the line stays and
        # the strict full match then fails closed.
        f"glued {HEARTBEAT}\n\n{ROLE_GONE}\n",
    ],
)
def test_heartbeats_prose_and_other_codes_are_not_absence(stderr: str) -> None:
    result = CommandResult(stdout="", stderr=stderr, returncode=254)
    assert matches_not_found(result, ("NoSuchEntity",)) is False, stderr


def test_stdout_behind_a_heartbeat_is_not_absence() -> None:
    result = CommandResult(
        stdout="NoSuchEntity", stderr=f"{HEARTBEAT}\n", returncode=254
    )
    assert matches_not_found(result, ("NoSuchEntity",)) is False, result


def test_error_text_drops_only_heartbeats_and_blank_lines() -> None:
    assert command_error_text(LIVE_STDERR) == ROLE_GONE
    assert command_error_text(f"{HEARTBEAT}\n") == ""
    assert command_error_text("  NoSuchEntity \n\n") == "NoSuchEntity"
    multiline = "first line\n\n  second line\n"
    assert (
        command_error_text(f"{HEARTBEAT}\n{multiline}{HEARTBEAT}\n")
        == "first line\n  second line"
    )
    glued = f"glued {HEARTBEAT}"
    assert command_error_text(glued) == glued


@pytest.mark.parametrize(
    "stderr",
    ["Error: release: not found\n", f"{HEARTBEAT}\n\nError: release: not found\n"],
)
def test_helm_release_absence_is_recognised_behind_a_heartbeat(stderr: str) -> None:
    assert helm_release_absent(CommandResult("", stderr, 1)) is True, stderr


@pytest.mark.parametrize(
    ("stderr", "returncode"),
    [
        (f"{HEARTBEAT}\nError: release: not found\nmore\n", 1),
        ("Error: release: not found\n", 2),
        (f"{HEARTBEAT}\n", 1),
    ],
)
def test_helm_release_absence_still_requires_the_exact_message(
    stderr: str, returncode: int
) -> None:
    assert helm_release_absent(CommandResult("", stderr, returncode)) is False, stderr
