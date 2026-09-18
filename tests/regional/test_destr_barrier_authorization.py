from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.models import WorkflowOperation, WorkflowStatus
from gpu_fault.node_agent.ledger import NodeActionLedger
from gpu_fault.node_agent.protocol import (
    NodeActionCommand,
    NodeActionResult,
    NodeActionStatus,
)
from scripts.e2e.regional import destr_barrier_authorization as authorization
from scripts.e2e.regional.probes import destr016_node_probe as delayed
from scripts.e2e.regional.probes import destr017_node_probe as reboot
from scripts.e2e.regional.probes import destructive_node_probe as probe

NOW = datetime(2026, 9, 12, 12, tzinfo=timezone.utc)
QUIESCE = "QUIESCE_GPU_SERVICES"
VERIFY = "VERIFY_NO_GPU_CLIENTS"


class Clock(datetime):
    @classmethod
    def now(cls, tz: Any = None) -> datetime:
        return NOW


def barrier_state() -> dict[str, Any]:
    return {
        "incident": {"incident_id": "incident-a", "drill_id": "review-r"},
        "workflow": {
            "request_id": "workflow-a",
            "incident_id": "incident-a",
            "status": WorkflowStatus.RUNNING,
            "fencing_token": 3,
            "step_executions": [
                {
                    "operation": QUIESCE,
                    "status": "SUCCEEDED",
                    "details": {
                        "agent_generations": {"node-a": 4},
                        "maintenance_window_expires_at": (
                            NOW + timedelta(minutes=5)
                        ).isoformat(),
                    },
                },
                {"operation": VERIFY, "status": "WAITING"},
            ],
        },
        "commands": [
            {
                "workflow_request_id": "workflow-a",
                "idempotency_key": operation.lower(),
                "step": {"operation": operation, "node_ids": ["node-a"]},
            }
            for operation in (QUIESCE, VERIFY)
        ],
    }


def proof(state: dict[str, Any] | None = None) -> dict[str, Any]:
    return authorization.barrier_authorization(
        barrier_state() if state is None else state,
        run_id="review",
        node="node-a",
        boot_id="boot-a",
        device="/dev/nvidia0",
        drill_id="review-r",
        maintenance_window_end=NOW + timedelta(minutes=10),
    )


@pytest.fixture
def frozen(monkeypatch: pytest.MonkeyPatch) -> None:
    for module in (authorization, probe, delayed, reboot):
        monkeypatch.setattr(module, "datetime", Clock)


def test_controller_proof_uses_exact_batched_step_identity(frozen: None) -> None:
    state = barrier_state()
    first, second = state["commands"]
    first["batched_steps"] = [second]
    state["commands"] = [first]

    result = proof(state)

    assert result["command_ids"] == {
        QUIESCE: "quiesce_gpu_services/node-a/agent-4",
        VERIFY: "verify_no_gpu_clients/node-a/agent-4",
    }
    assert result["workflow_request_id"] == "workflow-a"


@pytest.mark.parametrize(
    "change",
    ["drill", "workflow", "fence", "verify", "reset", "command", "node", "window"],
)
def test_controller_refuses_ambiguous_or_advanced_barrier(
    change: str, frozen: None
) -> None:
    state = barrier_state()
    workflow = state["workflow"]
    if change == "drill":
        state["incident"]["drill_id"] = "another-drill"
    elif change == "workflow":
        workflow["incident_id"] = "another-incident"
    elif change == "fence":
        workflow["fencing_token"] = True
    elif change == "verify":
        workflow["step_executions"][-1]["status"] = "SUCCEEDED"
    elif change == "reset":
        workflow["step_executions"].append(
            {"operation": "RESET_GPU", "status": "FAILED"}
        )
    elif change == "command":
        state["commands"].append(deepcopy(state["commands"][-1]))
    elif change == "node":
        state["commands"][-1]["step"]["node_ids"].append("node-b")
    else:
        workflow["step_executions"][0]["details"]["maintenance_window_expires_at"] = (
            NOW.isoformat()
        )

    with pytest.raises(authorization.RegionalFixtureError):
        proof(state)


def install_ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "node-actions.db"
    ledger = NodeActionLedger(str(path))
    try:
        for operation in (
            WorkflowOperation.QUIESCE_GPU_SERVICES,
            WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
        ):
            command = NodeActionCommand(
                command_id=f"{operation.value.lower()}/node-a/agent-4",
                workflow_request_id="workflow-a",
                incident_id="incident-a",
                node_id="node-a",
                agent_generation=4,
                fencing_token=3,
                operation=operation,
                gpu_uuids=["GPU-a"],
                issued_at=NOW - timedelta(seconds=2),
                expires_at=NOW + timedelta(minutes=1),
            )
            ledger.mark_in_progress(command, 1, agent_generation=4)
            verify = operation is WorkflowOperation.VERIFY_NO_GPU_CLIENTS
            ledger.save(
                NodeActionResult(
                    command_id=command.command_id,
                    operation=operation,
                    attempt=1,
                    status=NodeActionStatus.FAILED
                    if verify
                    else NodeActionStatus.SUCCEEDED,
                    error="GPU device clients are still active" if verify else None,
                    completed_at=NOW,
                )
            )
    finally:
        ledger.close()
    state = tmp_path / "quiesce-owned.json"
    state.write_text(
        json.dumps(
            {
                "incident_id": "incident-a",
                "workflow_request_id": "workflow-a",
                "boot_id": "boot-a",
                "phase": "QUIESCED",
                "reset_issued": None,
                "target_device_paths": ["/dev/nvidia0"],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(probe, "LEDGER", path)
    monkeypatch.setattr(probe, "QUIESCE_STATE_DIR", tmp_path)
    monkeypatch.setattr(probe, "boot_id", lambda: "boot-a")
    return state


def test_on_node_guard_reads_real_audit_rows_and_quiesce_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, frozen: None
) -> None:
    install_ledger(tmp_path, monkeypatch)

    probe.check_barrier(proof())


@pytest.mark.parametrize(
    "change",
    [
        {"waiting": False},
        {"workflow_request_id": "other"},
        {"incident_id": "other"},
        {"fencing_token": 4},
        {"agent_generation": 5},
        {"boot_id": "other"},
        {"device": "/dev/nvidia1"},
        {"observed_at": (NOW + timedelta(seconds=1)).isoformat()},
        {"observed_at": (NOW - timedelta(seconds=91)).isoformat()},
        {"maintenance_window_end": NOW.isoformat()},
        {"window_expires_at": "2026-09-12T13:00:00"},
        {"command_ids": {}},
    ],
)
def test_on_node_guard_refuses_wrong_or_expired_identity(
    change: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen: None,
) -> None:
    install_ledger(tmp_path, monkeypatch)

    with pytest.raises(probe.ProbeError):
        probe.check_barrier({**proof(), **change})


@pytest.mark.parametrize(
    "change",
    [
        {"phase": "RESTORED"},
        {"reset_issued": {"command_id": "reset"}},
        {"workflow_request_id": "other"},
        {"boot_id": "other"},
    ],
)
def test_on_node_guard_refuses_released_or_replaced_quiesce(
    change: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen: None,
) -> None:
    path = install_ledger(tmp_path, monkeypatch)
    path.write_text(
        json.dumps({**json.loads(path.read_text()), **change}), encoding="utf-8"
    )

    with pytest.raises(probe.ProbeError, match="quiesce"):
        probe.check_barrier(proof())


def test_scheduled_xid_refuses_without_explicit_controller_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "state.json"
    delayed.write_state(path, {"run_id": "review", "boot_id": "boot-a"})
    calls: list[Any] = []
    monkeypatch.setattr(delayed, "state_path", lambda _run: path)
    monkeypatch.setattr(delayed, "_boot_id", lambda: "boot-a")
    monkeypatch.setattr(delayed, "run", lambda *args, **kwargs: calls.append(args))

    with pytest.raises(delayed.ProbeError, match="not authorized"):
        delayed.fire_injection(argparse.Namespace(run_id="review", phase="escalate"))
    assert calls == []


def test_delayed_xid_consumes_intent_before_lost_ack_and_never_repeats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, frozen: None
) -> None:
    path = tmp_path / "state.json"
    script = tmp_path / "guard.py"
    script.write_text("# isolated fake probe\n", encoding="utf-8")
    delayed.write_state(
        path,
        {
            "run_id": "review",
            "boot_id": "boot-a",
            "injection_script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
            "injections": [
                {
                    "phase": "escalate",
                    "script": str(script),
                    "subcommand": "write-xid79",
                    "marker": "review-e",
                    "drill_id": "review-e",
                    "pci_bdf": "0000:01:00",
                    "maintenance_window_end": (NOW + timedelta(minutes=10)).isoformat(),
                }
            ],
            "authorizations": {
                "escalate": {"proof": proof(), "fire_requested_at": None}
            },
        },
    )
    monkeypatch.setattr(delayed, "state_path", lambda _run: path)
    monkeypatch.setattr(delayed, "_boot_id", lambda: "boot-a")
    calls: list[list[str]] = []

    def run(command: list[str], **kwargs: Any) -> None:
        calls.append(command)
        if "write-xid79" in command:
            state = delayed.read_state(path)
            assert state["authorizations"]["escalate"]["fire_requested_at"]
            raise TimeoutError("fake ACK loss")

    monkeypatch.setattr(delayed, "run", run)
    arguments = argparse.Namespace(run_id="review", phase="escalate")
    with pytest.raises(TimeoutError):
        delayed.fire_injection(arguments)
    with pytest.raises(delayed.ProbeError, match="consumed"):
        delayed.fire_injection(arguments)
    assert sum("write-xid79" in command for command in calls) == 1


def test_probe_state_cannot_be_cleared_while_a_timer_is_active(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "state.json"
    reboot.write_state(path, {"run_id": "review"})
    monkeypatch.setattr(reboot, "state_path", lambda _run: path)
    monkeypatch.setattr(
        reboot,
        "_unit_state",
        lambda *_args: {"LoadState": "loaded", "ActiveState": "active"},
    )

    with pytest.raises(reboot.ProbeError, match="may still run"):
        reboot.clear_state(argparse.Namespace(run_id="review"))
    assert path.exists(), "active timer state must remain available for cancellation"
