from __future__ import annotations

import subprocess
from typing import Any

import pytest

from gpu_fault.admin import aws_commands
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.deadlines import current_deadline, deadline_scope


@pytest.mark.parametrize(
    ("returncode", "stdout", "stderr"),
    [
        (
            254,
            "",
            "An error occurred (AccessDenied) when calling GetRole: NoSuchEntity",
        ),
        (254, "NoSuchEntity", "Unable to locate credentials"),
        (254, "", "An error occurred (NoSuchEntityException) when calling GetRole"),
        (254, "", "tool configuration NoSuchEntity could not be opened"),
        (127, "", "An error occurred (NoSuchEntity) when calling GetRole"),
        (0, "", "An error occurred (NoSuchEntity) when calling GetRole"),
    ],
)
def test_absence_requires_the_exact_aws_service_error(
    returncode: int, stdout: str, stderr: str
) -> None:
    result = aws_commands.CommandResult(stdout, stderr, returncode)
    assert not aws_commands.matches_not_found(result, ("NoSuchEntity",)), (
        "an inexact or unsuccessful AWS error match was accepted as confirmed absence"
    )


@pytest.mark.parametrize("returncode", [254, 255])
@pytest.mark.parametrize(
    "stderr",
    [
        "An error occurred (NoSuchEntity) when calling the GetRole operation: gone",
        "An error occurred (NoSuchEntity)",
        "NoSuchEntity",
    ],
)
def test_confirmed_absence_accepts_cli_versions_and_code_only_adapters(
    returncode: int, stderr: str
) -> None:
    result = aws_commands.CommandResult("", stderr, returncode)
    assert aws_commands.matches_not_found(result, ("NoSuchEntity",)), (
        "an exact NoSuchEntity error was not recognized as confirmed absence"
    )


@pytest.mark.parametrize(
    "stdout", ["", "not-json", "[]", "null", '"resource missing"', "true"]
)
def test_json_tool_failures_are_not_absence(
    monkeypatch: pytest.MonkeyPatch, stdout: str
) -> None:
    def command(arguments: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(arguments, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(aws_commands, "bounded_command", command)
    with pytest.raises(BootstrapError, match="JSON"):
        aws_commands.json_command(["aws", "iam", "get-role"])


@pytest.mark.parametrize(
    "error", [FileNotFoundError("missing tool"), PermissionError("unusable tool")]
)
def test_missing_or_unusable_tools_fail_closed(
    monkeypatch: pytest.MonkeyPatch, error: OSError
) -> None:
    def command(_arguments: list[str]) -> Any:
        raise error

    monkeypatch.setattr(aws_commands, "bounded_command", command)
    with pytest.raises(BootstrapError, match="cannot execute cleanup command"):
        aws_commands.checked_command(
            ["aws", "iam", "get-role"], not_found=("NoSuchEntity",)
        )


def test_wait_caps_sleep_and_each_probe_to_its_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [10.0]
    sleeps: list[float] = []
    probes: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock[0] += seconds

    def predicate() -> bool:
        deadline = current_deadline()
        assert deadline is not None
        probes.append(deadline.remaining())
        return False

    monkeypatch.setattr(aws_commands.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(aws_commands.time, "sleep", sleep)
    with pytest.raises(BootstrapError, match="timed out"):
        aws_commands.wait_until(
            predicate,
            description="test deletion",
            timeout_seconds=2,
            interval_seconds=5,
        )
    assert probes == [2.0]
    assert sleeps == [2.0]
    assert current_deadline() is None


def test_wait_never_accepts_a_late_success(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = [10.0]
    monkeypatch.setattr(aws_commands.time, "monotonic", lambda: clock[0])

    def predicate() -> bool:
        clock[0] += 3
        return True

    with deadline_scope("parent", 1):
        with pytest.raises(BootstrapError, match="timed out"):
            aws_commands.wait_until(
                predicate, description="test deletion", timeout_seconds=2
            )
