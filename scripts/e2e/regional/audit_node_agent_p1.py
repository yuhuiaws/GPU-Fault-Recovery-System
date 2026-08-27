from __future__ import annotations

import hashlib
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from subprocess import CompletedProcess

from gpu_fault.models import WorkflowOperation
from gpu_fault.node_agent import (
    GpuServiceQuiesceManager,
    NodeActionCommand,
    NodeActionExecutor,
    NodeActionLedger,
    NodeActionResult,
    NodeActionStatus,
    SignedNodeAction,
    agent_config_payload,
    sign_node_action,
)


SECRET = "audit-node-action-" + "x" * 32
NOW = datetime.now(timezone.utc)


class FlakyRunner:
    def __init__(self) -> None:
        self.reset_attempts = 0
        self.commands: list[list[str]] = []

    def __call__(self, command, **_kwargs):
        self.commands.append(command)
        if "--query-compute-apps=gpu_uuid,pid,process_name" in command:
            return CompletedProcess(command, 0, stdout="", stderr="")
        if "--query-gpu=uuid,index" in command:
            return CompletedProcess(command, 0, stdout="GPU-a, 0\n", stderr="")
        if "--gpu-reset" in command:
            self.reset_attempts += 1
            if self.reset_attempts == 1:
                raise OSError("temporary transport error")
        return CompletedProcess(command, 0, stdout="", stderr="")


def signed_command(operation: WorkflowOperation) -> SignedNodeAction:
    command = NodeActionCommand(
        command_id=f"audit/{operation.value}",
        workflow_request_id="audit-workflow",
        incident_id="audit-incident",
        fencing_token=1,
        operation=operation,
        node_id="audit-node",
        gpu_uuids=["GPU-a"],
        issued_at=NOW,
        expires_at=NOW + timedelta(minutes=5),
    )
    return SignedNodeAction(
        command=command,
        signature=sign_node_action(command, SECRET),
    )


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="gpu-fault-node-p1-") as scratch:
        root = Path(scratch)
        runner = FlakyRunner()
        ledger = NodeActionLedger(
            str(root / "actions.db"),
            retention_seconds=600,
            max_results=100,
        )
        executor = NodeActionExecutor(
            secret=SECRET,
            node_ids={"audit-node"},
            allowed_operations={WorkflowOperation.RESET_GPU},
            reset_enabled=True,
            ledger=ledger,
            runner=runner,
            device_client_finder=lambda _targets: [],
            device_client_samples=1,
            now=lambda: NOW,
        )
        command = signed_command(WorkflowOperation.RESET_GPU)
        first = executor.execute(command)
        second = executor.execute(command)
        assert first.status is NodeActionStatus.FAILED
        assert first.retryable is True
        assert second.status is NodeActionStatus.SUCCEEDED
        assert runner.reset_attempts == 2

        old = NOW - timedelta(minutes=20)
        ledger.save(
            NodeActionResult(
                command_id="old-result",
                operation=WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
                status=NodeActionStatus.SUCCEEDED,
                completed_at=old,
            )
        )
        removed = ledger.cleanup(now=NOW)
        assert ledger.get("old-result") is None
        assert removed["results"] >= 1

        manager = GpuServiceQuiesceManager(
            state_dir=str(root / "quiesce"),
            services=("kubelet",),
            containers=("kube-system/nvidia-device-plugin-ctr",),
            failsafe_seconds=30,
            retry_seconds=10,
            container_restore_timeout_seconds=180,
            restore_command="/bin/true",
        )
        manager._running_container_ids = lambda _selector: [
            "a" * 64,
            "b" * 64,
        ]
        assert len(manager._resolve_container_targets()) == 2

        archive_dir = root / "diagnostics"
        archive_dir.mkdir()
        for index in range(3):
            path = archive_dir / (f"gpu-diagnostic-{index}.tar.gz")
            path.write_bytes(b"archive")
            modified = (NOW - timedelta(hours=2 - index)).timestamp()
            os.utime(path, (modified, modified))
        diagnostic_executor = NodeActionExecutor(
            secret=SECRET,
            node_ids={"audit-node"},
            allowed_operations={WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE},
            reset_enabled=False,
            diagnostic_output_dir=str(archive_dir),
            diagnostic_retention_seconds=3600,
            diagnostic_max_archives=1,
            ledger=NodeActionLedger(str(root / "diagnostic.db")),
            now=lambda: NOW,
        )
        removed_archives = diagnostic_executor._cleanup_diagnostic_archives(now=NOW)
        assert len(removed_archives) == 2

        assert NodeActionExecutor._field_diagnostic_line_failed("Overall Result: FAIL")
        assert not NodeActionExecutor._field_diagnostic_line_failed("0 errors")
        diagnostic_executor._compute_clients = lambda: [
            {
                "gpu_uuid": "GPU-a",
                "pid": "123",
                "process_name": "python",
            }
        ]
        diagnostic_executor.fabric_manager_restart_enabled = True
        try:
            diagnostic_executor._restart_fabric_manager(
                signed_command(WorkflowOperation.RESTART_FABRIC_MANAGER).command
            )
        except RuntimeError as exc:
            assert "compute clients are still active" in str(exc)
        else:
            raise AssertionError("Fabric Manager restart did not fail closed")

        payload = agent_config_payload(diagnostic_executor, "audit-profile-v1")
        for field in (
            "diagnostic_retention_seconds",
            "diagnostic_max_archives",
            "device_client_samples",
            "inflight_wait_timeout_seconds",
            "ledger_retention_seconds",
            "ledger_max_results",
        ):
            assert field in payload

        executable = root / "fielddiag"
        executable.write_text("#!/bin/sh\n")
        executable.chmod(0o755)
        assert len(hashlib.sha256(executable.read_bytes()).hexdigest()) == 64

        print(
            "node_agent_p1_probe=PASS",
            f"retry_attempts={runner.reset_attempts}",
            f"ledger_removed={removed['results']}",
            f"archives_removed={len(removed_archives)}",
            "multi_container_targets=2",
            "fm_active_client_gate=PASS",
        )


if __name__ == "__main__":
    main()
