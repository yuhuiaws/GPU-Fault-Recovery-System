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

from gpu_fault.models import WorkflowOperation
from gpu_fault.node_agent import NodeActionStatus
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
