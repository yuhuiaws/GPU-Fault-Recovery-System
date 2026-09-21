"""AWS CLI 2.35 prefixes its error lines with ``aws: [ERROR]:``; absence must
still be recognised (live 2026-09-20: the uninstall's Aurora final-snapshot probe
treated ``DBClusterSnapshotNotFoundFault`` as a hard failure)."""

from __future__ import annotations

import pytest

from gpu_fault.admin.aws_commands import CommandResult, matches_not_found

NOT_FOUND = (
    "An error occurred (DBClusterSnapshotNotFoundFault) when calling the "
    "DescribeDBClusterSnapshots operation: DBClusterSnapshot not found: gpu-fault-x"
)


@pytest.mark.parametrize(
    "stderr",
    [NOT_FOUND, "aws: [ERROR]: " + NOT_FOUND, "\naws: [ERROR]: " + NOT_FOUND + "\n"],
)
def test_not_found_is_recognised_with_and_without_the_cli_prefix(stderr: str) -> None:
    result = CommandResult(stdout="", stderr=stderr, returncode=254)
    assert matches_not_found(result, ("DBClusterSnapshotNotFoundFault",)) is True, (
        stderr
    )


@pytest.mark.parametrize(
    "stderr",
    [
        "aws: [ERROR]: An error occurred (AccessDenied) when calling the X operation: no",
        "aws: [WARN]: An error occurred (DBClusterSnapshotNotFoundFault) when calling X",
        "some prose mentioning DBClusterSnapshotNotFoundFault",
    ],
)
def test_other_errors_and_prose_are_not_absence(stderr: str) -> None:
    result = CommandResult(stdout="", stderr=stderr, returncode=254)
    assert matches_not_found(result, ("DBClusterSnapshotNotFoundFault",)) is False, (
        stderr
    )
