"""A redacted command failure names its step in every regional runner.

Live 2026-10-04: DESTR-015 and DESTR-002 stopped with only
``command failed (1): <sensitive output redacted>``; the evidence could not say
which command class failed.
"""

from __future__ import annotations

import subprocess
from typing import Any

import pytest

from scripts.e2e.regional import regional_commands


def test_command_step_label_keeps_the_cli_and_subcommand_words_only() -> None:
    assert (
        regional_commands.command_step_label(
            [
                "aws",
                "sagemaker",
                "describe-cluster",
                "--cluster-name",
                "",
                "--region",
                "x",
            ]
        )
        == "aws sagemaker describe-cluster"
    )
    assert (
        regional_commands.command_step_label(
            [
                "kubectl",
                "--kubeconfig",
                "/secure/g",
                "--context",
                "ctx",
                "-n",
                "ns",
                "exec",
                "-i",
                "pod-7",
                "--",
                "/opt/python",
                "-",
                "secret",
            ]
        )
        == "kubectl exec"
    ), "a kubectl label carries the verb only"
    assert (
        regional_commands.command_step_label(
            ["kubectl", "--context", "ctx", "api-resources", "-o", "name"]
        )
        == "kubectl api-resources"
    )
    assert (
        regional_commands.command_step_label(
            [
                "kubectl",
                "--context",
                "ctx",
                "get",
                "pytorchjobs.kubeflow.org,jobs.batch",
                "-o",
                "json",
            ]
        )
        == "kubectl get"
    )
    assert regional_commands.command_step_label([]) == ""


def test_a_failed_fixture_command_names_its_step_and_stays_redacted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def run_command(
        command: list[str], **_kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            command, 1, "", "token=SECRET-VALUE no such resource"
        )

    monkeypatch.setattr(regional_commands, "run_command", run_command)
    with pytest.raises(regional_commands.RegionalCommandFailed) as failure:
        regional_commands.run_fixture_command(
            [
                "kubectl",
                "--context",
                "ctx",
                "-n",
                "training",
                "get",
                "pytorchjobs.kubeflow.org",
                "-o",
                "json",
            ]
        )
    assert str(failure.value) == (
        "command failed (1): <sensitive output redacted> (step: kubectl get)"
    )
    assert failure.value.step == "kubectl get"
    assert "SECRET-VALUE" not in str(failure.value)


def test_a_command_failure_without_its_command_reads_exactly_as_before() -> None:
    message = str(regional_commands.RegionalCommandFailed(1, "Forbidden"))
    assert message.startswith("command failed (1): ") and "(step:" not in message, (
        "without the command the message keeps its historical shape"
    )
