"""The quick DCGM diagnostic on a plain EKS node whose image ships dcgmi but no
host nv-hostengine.

Live 2026-10-06 (gpu-eks-cluster, COLLECT-020): ``dcgmi diag -r 1 -j`` exited
226 (``DCGM_ST_CONNECTION_NOT_VALID``) with no JSON, the step graded it FAIL
with ``DCGM_EXECUTION_REVIEW`` and the identity incident's workflow drained
and quarantined a healthy node. On ``eks-managed`` that reading is the same
host configuration fact as a missing dcgmi; on HyperPod, which runs a host
hostengine, a connection failure stays a failure.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from gpu_fault.models import WorkflowOperation
from gpu_fault.node_agent import NodeActionStatus
from gpu_fault.node_agent.config import (
    agent_config_digest,
    dcgm_hostengine_from_environment,
    executor_from_environment,
)
from gpu_fault.node_agent.operations import diagnostics
from tests.node_agent._support import command, envelope, node_action_executor

HOSTENGINE_DOWN = (
    "Error: unable to establish a connection to the specified host: localhost\n"
    "Error: Unable to connect to host engine. Host engine connection "
    "invalid/disconnected.\n"
)


def _runner(returncode: int, stderr: str, stdout: str = ""):
    def runner(argv, **_kwargs):
        return subprocess.CompletedProcess(argv, returncode, stdout, stderr)

    return runner


def _execute(tmp_path: Path, name: str, runner, **overrides):
    agent = node_action_executor(
        tmp_path,
        f"{name}.db",
        allowed_operations={WorkflowOperation.RUN_DCGM_DIAGNOSTIC},
        diagnostic_output_dir=str(tmp_path / name),
        runner=runner,
        **overrides,
    )
    return agent.execute(envelope(command(WorkflowOperation.RUN_DCGM_DIAGNOSTIC)))


def test_no_hostengine_on_eks_is_a_configuration_warning(tmp_path: Path) -> None:
    result = _execute(
        tmp_path,
        "eks-no-hostengine",
        _runner(226, HOSTENGINE_DOWN),
        cluster_provider="eks-managed",
    )

    assert result.status is NodeActionStatus.SUCCEEDED
    assert result.details["diagnostic_outcome"] == "WARN"
    assert result.details["configuration_only_failures"] is True
    assert result.details["returncode"] == 226
    assert [item["action_code"] for item in result.details["recommended_actions"]] == [
        "DCGM_HOSTENGINE_UNAVAILABLE"
    ]
    evidence = Path(result.details["evidence_ref"].removeprefix("file://"))
    assert "Unable to connect to host engine" in evidence.read_text(), (
        "the evidence file keeps dcgmi's own words"
    )


def test_the_connection_error_text_counts_without_the_exit_status(
    tmp_path: Path,
) -> None:
    result = _execute(
        tmp_path,
        "eks-text-only",
        _runner(255, HOSTENGINE_DOWN),
        cluster_provider="eks-managed",
    )
    assert result.details["diagnostic_outcome"] == "WARN"
    assert result.details["configuration_only_failures"] is True


def test_hyperpod_keeps_failing_when_its_host_hostengine_is_down(
    tmp_path: Path,
) -> None:
    result = _execute(
        tmp_path, "hyperpod-hostengine-down", _runner(226, HOSTENGINE_DOWN)
    )

    assert result.details["diagnostic_outcome"] == "FAIL"
    assert result.details["configuration_only_failures"] is False
    assert [item["action_code"] for item in result.details["recommended_actions"]] == [
        "DCGM_EXECUTION_REVIEW"
    ]


def test_another_dcgmi_failure_on_eks_is_still_a_failure(tmp_path: Path) -> None:
    result = _execute(
        tmp_path,
        "eks-other-failure",
        _runner(1, "Error: the diagnostic could not start the plugin\n"),
        cluster_provider="eks-managed",
    )
    assert result.details["diagnostic_outcome"] == "FAIL"
    assert result.details["configuration_only_failures"] is False
    assert [item["action_code"] for item in result.details["recommended_actions"]] == [
        "DCGM_EXECUTION_REVIEW"
    ]


def test_a_graded_json_result_is_never_read_as_unreachable() -> None:
    graded = subprocess.CompletedProcess(
        ["dcgmi"], 226, '{"DCGM GPU Diagnostic": {"test_categories": []}}', ""
    )
    assert diagnostics.dcgm_hostengine_unreachable(graded) is False
    assert diagnostics.dcgm_hostengine_unreachable(
        subprocess.CompletedProcess(["dcgmi"], 226, "", "")
    ), "the DCGM_ST_CONNECTION_NOT_VALID exit status alone is enough"


# --- host mode: the node group declares a host DCGM ---------------------------
#
# An eks-managed node whose node group carries gpu-fault.io/dcgm-hostengine=host
# runs nvidia-dcgm (the node preflight required it), and its Agent is installed
# with GPU_FAULT_DCGM_HOSTENGINE=host. There an absent dcgmi or an unreachable
# engine is graded exactly as an unreachable engine on HyperPod: a failure, not
# a configuration-only warning.


def _missing_dcgmi(argv, **_kwargs):
    raise FileNotFoundError(2, "No such file or directory", argv[0])


def _grading(result) -> tuple[str, bool, list[str]]:
    return (
        result.details["diagnostic_outcome"],
        result.details["configuration_only_failures"],
        [item["action_code"] for item in result.details["recommended_actions"]],
    )


def test_host_mode_grades_an_unreachable_engine_like_hyperpod(tmp_path: Path) -> None:
    host = _execute(
        tmp_path,
        "eks-host-hostengine-down",
        _runner(226, HOSTENGINE_DOWN),
        cluster_provider="eks-managed",
        dcgm_hostengine="host",
    )
    hyperpod = _execute(tmp_path, "hyperpod-reference", _runner(226, HOSTENGINE_DOWN))

    assert host.status is NodeActionStatus.SUCCEEDED
    assert (
        _grading(host)
        == _grading(hyperpod)
        == ("FAIL", False, ["DCGM_EXECUTION_REVIEW"])
    ), "a declared host engine that does not answer is a failure"
    assert host.details["dcgm_hostengine"] == "host", "the grading mode is recorded"
    evidence = Path(host.details["evidence_ref"].removeprefix("file://"))
    assert "Unable to connect to host engine" in evidence.read_text(), (
        "the evidence file keeps dcgmi's own words"
    )


def test_host_mode_fails_when_dcgmi_is_missing(tmp_path: Path) -> None:
    result = _execute(
        tmp_path,
        "eks-host-no-dcgmi",
        _missing_dcgmi,
        cluster_provider="eks-managed",
        dcgm_hostengine="host",
    )

    assert _grading(result) == ("FAIL", False, ["DCGM_EXECUTION_REVIEW"]), (
        "the node group promised DCGM; its absence is not a configuration note"
    )
    assert result.details["returncode"] == 127, "the missing tool's exit status"
    evidence = Path(result.details["evidence_ref"].removeprefix("file://"))
    assert "dcgmi unavailable on this node" in evidence.read_text(), (
        "the evidence file carries the reason"
    )


@pytest.mark.parametrize(
    ("runner", "code"),
    [
        (_runner(226, HOSTENGINE_DOWN), "DCGM_HOSTENGINE_UNAVAILABLE"),
        (_missing_dcgmi, "DCGM_TOOL_UNAVAILABLE"),
    ],
)
def test_embedded_mode_keeps_the_configuration_warning(
    tmp_path: Path, runner, code: str
) -> None:
    result = _execute(
        tmp_path,
        f"eks-embedded-{code.lower()}",
        runner,
        cluster_provider="eks-managed",
        dcgm_hostengine="embedded",
    )

    assert _grading(result) == ("WARN", True, [code]), (
        "the embedded engine leaves the node nothing to dial: a configuration fact"
    )
    assert result.details["dcgm_hostengine"] == "embedded", "the mode is recorded"


def test_hyperpod_results_do_not_change_with_the_mode(tmp_path: Path) -> None:
    unreachable = _execute(tmp_path, "hyperpod-down", _runner(226, HOSTENGINE_DOWN))
    missing = _execute(tmp_path, "hyperpod-no-dcgmi", _missing_dcgmi)

    assert _grading(unreachable) == ("FAIL", False, ["DCGM_EXECUTION_REVIEW"])
    assert _grading(missing) == ("WARN", True, ["DCGM_TOOL_UNAVAILABLE"]), (
        "HyperPod keeps grading a missing dcgmi as before"
    )
    for result in (unreachable, missing):
        assert "dcgm_hostengine" not in result.details, (
            "a HyperPod result carries exactly the fields it always did"
        )
    executor = node_action_executor(
        tmp_path,
        "hyperpod-mode.db",
        allowed_operations={WorkflowOperation.RUN_DCGM_DIAGNOSTIC},
    )
    assert executor.dcgm_hostengine == "host", "HyperPod always runs a host engine"


def test_the_mode_is_not_part_of_the_agent_config_digest(tmp_path: Path) -> None:
    digests = {
        agent_config_digest(
            node_action_executor(
                tmp_path,
                f"digest-{mode or 'unset'}.db",
                allowed_operations={WorkflowOperation.RUN_DCGM_DIAGNOSTIC},
                cluster_provider="eks-managed",
                dcgm_hostengine=mode,
            ),
            "eks-v1",
        )
        for mode in ("", "embedded", "host")
    }
    assert len(digests) == 1, (
        "like the provider, the mode is a node-image fact outside the pinned "
        "Agent configuration: declaring it changes no config digest"
    )


@pytest.mark.parametrize(
    ("provider", "value", "expected"),
    [
        (None, None, "host"),
        ("hyperpod", "host", "host"),
        ("eks-managed", None, "embedded"),
        ("eks-managed", "embedded", "embedded"),
        ("eks-managed", "host", "host"),
    ],
)
def test_the_agent_reads_its_mode_from_its_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider: str | None,
    value: str | None,
    expected: str,
) -> None:
    monkeypatch.setenv("GPU_FAULT_NODE_ACTION_SECRET", "s" * 32)
    monkeypatch.setenv("NODE_NAME", "node-a")
    monkeypatch.setenv("GPU_FAULT_NODE_ALLOWED_OPERATIONS", "RUN_DCGM_DIAGNOSTIC")
    monkeypatch.setenv("GPU_FAULT_NODE_ACTION_DB", str(tmp_path / "actions.db"))
    for name, setting in (
        ("GPU_FAULT_CLUSTER_PROVIDER", provider),
        ("GPU_FAULT_DCGM_HOSTENGINE", value),
    ):
        if setting is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, setting)

    executor = executor_from_environment(validate_runtime_paths=False)

    assert executor.dcgm_hostengine == expected, (
        "GPU_FAULT_DCGM_HOSTENGINE as the node installer wrote it decides the grading"
    )


@pytest.mark.parametrize(
    ("provider", "value"),
    [("eks-managed", "Host"), ("eks-managed", "on"), ("hyperpod", "embedded")],
)
def test_a_mode_the_agent_cannot_honour_fails_its_start(
    provider: str, value: str
) -> None:
    with pytest.raises(ValueError, match="DCGM host engine|HyperPod cluster runs"):
        dcgm_hostengine_from_environment(provider, {"GPU_FAULT_DCGM_HOSTENGINE": value})
