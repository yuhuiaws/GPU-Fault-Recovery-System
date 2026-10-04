"""The Node Agent P1 probe runs end to end without a GPU host.

``audit_node_agent_p1`` drives a real ``NodeActionExecutor``, ledger and quiesce
manager against an in-process flaky ``nvidia-smi`` runner. It has no command
line; the only contract is that it reaches its final summary line, which it can
only do when every embedded assertion -- retry after a transport error, ledger
retention, archive pruning, the Fabric Manager active-client gate -- held.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from gpu_fault.models import WorkflowOperation
from gpu_fault.node_agent import NodeActionStatus, sign_node_action
from scripts.e2e.regional import audit_node_agent_p1 as probe

SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts/e2e/regional/audit_node_agent_p1.py"
)


def test_flaky_runner_fails_the_first_reset_only_and_answers_queries() -> None:
    runner = probe.FlakyRunner()

    clients = runner(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name"])
    assert isinstance(clients, CompletedProcess) and clients.stdout == ""
    inventory = runner(["nvidia-smi", "--query-gpu=uuid,index"])
    assert inventory.stdout == "GPU-a, 0\n"

    with pytest.raises(OSError, match="temporary transport error"):
        runner(["nvidia-smi", "--gpu-reset", "-i", "0"])
    retry = runner(["nvidia-smi", "--gpu-reset", "-i", "0"])
    assert retry.returncode == 0
    assert runner.reset_attempts == 2

    other = runner(["systemctl", "is-active", "kubelet"])
    assert other.returncode == 0
    assert len(runner.commands) == 5


def test_signed_command_verifies_against_the_probe_secret() -> None:
    signed = probe.signed_command(WorkflowOperation.RESET_GPU)

    assert signed.command.command_id == "audit/RESET_GPU"
    assert signed.command.node_id == "audit-node"
    assert signed.signature == sign_node_action(signed.command, probe.SECRET)
    assert signed.signature != sign_node_action(signed.command, "other-" + "y" * 32)


def test_probe_main_reaches_its_summary_line(
    capsys: pytest.CaptureFixture[str],
) -> None:
    probe.main()

    out = capsys.readouterr().out
    assert "node_agent_p1_probe=PASS" in out
    assert "retry_attempts=2" in out
    assert "archives_removed=2" in out
    assert "fm_active_client_gate=PASS" in out
    assert NodeActionStatus.FAILED.value not in out


def test_probe_module_entry_point_runs_main(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", [str(SCRIPT)])

    runpy.run_path(str(SCRIPT), run_name="__main__")

    assert "node_agent_p1_probe=PASS" in capsys.readouterr().out
