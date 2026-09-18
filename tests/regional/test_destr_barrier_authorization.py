from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
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


def conditional(**overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "run_id": "review",
        "node": "node-a",
        "boot_id": "boot-a",
        "device": "/dev/nvidia0",
        "drill_id": "review-r",
        "marker": "review-m",
        "maintenance_window_end": NOW + timedelta(minutes=10),
        "maintenance_window_seconds": 420,
        "valid_for_seconds": 300,
        "not_before_seconds": {"absorb": 90, "escalate": 240},
        "now": NOW,
    }
    kwargs.update(overrides)
    return authorization.conditional_pre_authorization(**kwargs)


@pytest.fixture
def frozen(monkeypatch: pytest.MonkeyPatch) -> None:
    for module in (authorization, probe, delayed, reboot):
        monkeypatch.setattr(module, "datetime", Clock)


# --------------------------------------------------------------------------- #
# The store-side barrier observation
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# The conditional pre-authorization the runner delivers before the injection
# --------------------------------------------------------------------------- #
def test_conditional_pre_authorization_binds_the_run_and_its_deadlines() -> None:
    result = conditional()
    assert result["kind"] == authorization.CONDITIONAL_KIND, result
    assert result["conditional"] is True, result
    assert result["run_id"] == "review" and result["node_id"] == "node-a", result
    assert result["boot_id"] == "boot-a" and result["device"] == "/dev/nvidia0", result
    assert result["drill_id"] == "review-r" and result["marker"] == "review-m", result
    assert result["authorized_at"] == NOW.isoformat(), result
    assert result["expires_at"] == (NOW + timedelta(seconds=300)).isoformat(), result
    assert result["maintenance_window_seconds"] == 420, result
    assert result["not_before_seconds"] == {"absorb": 90, "escalate": 240}, result
    assert result["ledger"] == authorization.LEDGER_SHAPE, result
    assert "command_ids" not in result and "waiting" not in result, (
        "nothing in the pre-authorization may claim the barrier already exists"
    )


def test_conditional_pre_authorization_never_outlives_the_maintenance_window() -> None:
    result = conditional(maintenance_window_end=NOW + timedelta(seconds=100))
    assert result["expires_at"] == (NOW + timedelta(seconds=100)).isoformat(), result


@pytest.mark.parametrize(
    "change",
    [
        {"run_id": "bad run id"},
        {"node": ""},
        {"device": "/dev/sda"},
        {"boot_id": ""},
        {"marker": "m;rm -rf /"},
        {"drill_id": "no spaces allowed"},
        {"maintenance_window_end": NOW},
        {"maintenance_window_end": datetime(2026, 9, 12, 13)},
        {"maintenance_window_seconds": 10},
        {"maintenance_window_seconds": 7200},
        {"valid_for_seconds": 30},
        {"valid_for_seconds": 7200},
        {"not_before_seconds": {}},
        {"not_before_seconds": {"absorb": 300}},
        {"not_before_seconds": {"absorb": -1}},
        {"not_before_seconds": {"absorb": "90"}},
        {"not_before_seconds": {"bad phase": 10}},
    ],
)
def test_conditional_pre_authorization_refuses_unbound_or_unbounded_inputs(
    change: dict[str, Any],
) -> None:
    with pytest.raises(authorization.RegionalFixtureError):
        conditional(**change)


def barrier_rows(node: str = "node-a") -> list[dict[str, Any]]:
    """Ledger rows of ``workflow-a`` parked at the client verification at NOW."""

    stamp = NOW.isoformat()
    return [
        {
            "command_id": f"workflow-a/2/{QUIESCE}/{node}/agent-4",
            "operation": QUIESCE,
            "state": "SUCCEEDED",
            "attempt": 1,
            "started_at": stamp,
            "completed_at": stamp,
            "workflow_request_id": "workflow-a",
            "incident_id": "incident-a",
            "agent_generation": 4,
            "error": None,
        },
        {
            "command_id": f"workflow-a/3/{VERIFY}/{node}/agent-4",
            "operation": VERIFY,
            "state": "FAILED",
            "attempt": 1,
            "started_at": stamp,
            "completed_at": stamp,
            "workflow_request_id": "workflow-a",
            "incident_id": "incident-a",
            "agent_generation": 4,
            "error": "GPU device clients are still active: GPU-a:4242",
        },
    ]


def test_scheduled_xid_refuses_without_a_conditional_pre_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "state.json"
    delayed.write_state(
        path,
        {
            "run_id": "review",
            "boot_id": "boot-a",
            "injections": [{"phase": "escalate"}],
        },
    )
    calls: list[Any] = []
    monkeypatch.setattr(delayed, "state_path", lambda _run: path)
    monkeypatch.setattr(delayed, "_boot_id", lambda: "boot-a")
    monkeypatch.setattr(delayed, "holder_active", lambda _run: True)
    monkeypatch.setattr(delayed, "run", lambda *args, **kwargs: calls.append(args))

    with pytest.raises(delayed.ProbeError, match="no conditional pre-authorization"):
        delayed.fire_injection(argparse.Namespace(run_id="review", phase="escalate"))
    assert not any("write-xid" in str(command) for command in calls), calls
    recorded = delayed.read_state(path)
    assert recorded["injection_refusals"]["escalate"]["reason"] == (
        "no conditional pre-authorization is recorded"
    ), recorded
    assert recorded["disarmed_at"], "a timer that can never fire disarms the holder"


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
            "drill_id": "review-r",
            "device": "/dev/nvidia0",
            "boot_id": "boot-a",
            "max_hold_seconds": 900,
            "injection_script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
            "injections": [
                {
                    "phase": "escalate",
                    "script": str(script),
                    "subcommand": "write-xid79",
                    "marker": "review-e",
                    "drill_id": "review-e",
                    "pci_bdf": "0000:01:00",
                    "after_seconds": 10,
                    "maintenance_window_end": (NOW + timedelta(minutes=10)).isoformat(),
                }
            ],
            "pre_authorization": conditional(not_before_seconds={"escalate": 0}),
            "pre_authorized_at": NOW.isoformat(),
            "ledger_baseline_workflow_ids": [],
            "hold_started_at": NOW.isoformat(),
            "injections_fired": {},
        },
    )
    monkeypatch.setattr(delayed, "state_path", lambda _run: path)
    monkeypatch.setattr(delayed, "_boot_id", lambda: "boot-a")
    monkeypatch.setattr(delayed, "holder_active", lambda _run: True)
    monkeypatch.setattr(delayed, "ledger_rows", barrier_rows)
    calls: list[list[str]] = []

    def run(command: list[str], **kwargs: Any) -> None:
        calls.append(command)
        if "write-xid79" in command:
            state = delayed.read_state(path)
            assert state["injections_fired"]["escalate"]["fire_requested_at"], state
            assert "--barrier-authorization" not in command, command
            raise TimeoutError("fake ACK loss")

    monkeypatch.setattr(delayed, "run", run)
    arguments = argparse.Namespace(run_id="review", phase="escalate")
    with pytest.raises(TimeoutError):
        delayed.fire_injection(arguments)
    with pytest.raises(delayed.ProbeError, match="consumed"):
        delayed.fire_injection(arguments)
    assert sum("write-xid79" in command for command in calls) == 1
    condition = delayed.read_state(path)["injections_fired"]["escalate"]["condition"]
    assert condition["workflow_request_id"] == "workflow-a", condition
    assert condition["incident_id"] == "incident-a" and condition["boot_id"] == (
        "boot-a"
    ), condition


# --------------------------------------------------------------------------- #
# The post-hoc ordering verdict
# --------------------------------------------------------------------------- #
def fire_record(offset_seconds: int, **condition: Any) -> dict[str, Any]:
    fired = (NOW + timedelta(seconds=offset_seconds)).isoformat()
    return {
        "fire_requested_at": fired,
        "fired_at": fired,
        "condition": {
            "workflow_request_id": "workflow-a",
            "incident_id": "incident-a",
            "boot_id": "boot-a",
            "observed_at": fired,
            **condition,
        },
    }


def test_fired_after_barrier_accepts_a_later_fire_on_the_same_barrier(
    frozen: None,
) -> None:
    assert (
        authorization.fired_after_barrier_errors(
            fire_record(30), store_proof=proof(), label="absorb injection"
        )
        == []
    )


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        ("early", "not after the runner observed the barrier"),
        ("same-instant", "not after the runner observed the barrier"),
        ("workflow", "matched workflow_request_id 'workflow-b'"),
        ("incident", "matched incident_id 'incident-b'"),
        ("boot", "matched boot_id 'boot-b'"),
        ("missing-record", "the host recorded no fire"),
        ("missing-time", "timestamp is missing"),
        ("no-observation", "barrier observation has no workflow_request_id"),
    ],
)
def test_fired_after_barrier_refuses_early_or_foreign_fires(
    change: str, expected: str, frozen: None
) -> None:
    store = proof()
    record: dict[str, Any] | None = fire_record(30)
    if change == "early":
        record = fire_record(-30)
    elif change == "same-instant":
        record = fire_record(0)
    elif change == "workflow":
        record = fire_record(30, workflow_request_id="workflow-b")
    elif change == "incident":
        record = fire_record(30, incident_id="incident-b")
    elif change == "boot":
        record = fire_record(30, boot_id="boot-b")
    elif change == "missing-record":
        record = None
    elif change == "missing-time":
        record = {"condition": fire_record(30)["condition"]}
    else:
        store = {**store, "workflow_request_id": ""}
    errors = authorization.fired_after_barrier_errors(
        record, store_proof=store, label="out-of-band reboot"
    )
    assert any(expected in error for error in errors), errors
    assert all(error.startswith("out-of-band reboot: ") for error in errors), errors


# --------------------------------------------------------------------------- #
# Cleanup while kubelet may still be down
# --------------------------------------------------------------------------- #
def test_holder_disarm_decision_assumes_a_holder_gone_only_on_a_new_boot_or_lifetime() -> (
    None
):
    ready = authorization.holder_disarm_decision(
        node_ready=True,
        baseline_boot_id="boot-a",
        current_boot_id="boot-a",
        failsafe_deadline=NOW + timedelta(hours=1),
        now=NOW,
    )
    assert ready["exec_allowed"] is True and ready["assume_disarmed"] is False, ready
    rebooted = authorization.holder_disarm_decision(
        node_ready=False,
        baseline_boot_id="boot-a",
        current_boot_id="boot-b",
        failsafe_deadline=NOW + timedelta(hours=1),
        now=NOW,
    )
    assert rebooted["exec_allowed"] is False and rebooted["assume_disarmed"] is True, (
        rebooted
    )
    expired = authorization.holder_disarm_decision(
        node_ready=False,
        baseline_boot_id="boot-a",
        current_boot_id="boot-a",
        failsafe_deadline=NOW - timedelta(seconds=1),
        now=NOW,
    )
    assert expired["assume_disarmed"] is True and "lifetime" in expired["reason"], (
        expired
    )
    alive = authorization.holder_disarm_decision(
        node_ready=False,
        baseline_boot_id="boot-a",
        current_boot_id=None,
        failsafe_deadline=NOW + timedelta(hours=1),
        now=NOW,
    )
    assert alive["exec_allowed"] is False and alive["assume_disarmed"] is False, alive


class _Regional:
    def __init__(self, *, ready: bool, boot: str) -> None:
        self.ready = ready
        self.boot = boot
        self.waits: list[dict[str, Any]] = []

    def wait_node_ready(self, node: str, **kwargs: Any) -> dict[str, Any]:
        self.waits.append({"node": node, **kwargs})
        if not self.ready:
            raise authorization.RegionalFixtureError("node did not return Ready")
        return {"boot_id": self.boot}

    def node_snapshot(self, node: str) -> dict[str, Any]:
        return {"boot_id": self.boot}


def test_host_reachability_waits_for_the_node_before_allowing_any_exec() -> None:
    regional = _Regional(ready=True, boot="boot-a")
    decision = authorization.host_reachability(
        regional,
        node="node-a",
        baseline_boot_id="boot-a",
        holder_armed_at=NOW,
        max_hold_seconds=900,
        budget_seconds=600,
    )
    assert decision["exec_allowed"] is True and decision["node_ready"] is True, decision
    assert regional.waits == [{"node": "node-a", "timeout_seconds": 600}], (
        regional.waits
    )


def test_host_reachability_decides_from_the_boot_id_and_the_failsafe(
    frozen: None,
) -> None:
    rebooted = authorization.host_reachability(
        _Regional(ready=False, boot="boot-b"),
        node="node-a",
        baseline_boot_id="boot-a",
        holder_armed_at=NOW,
        max_hold_seconds=900,
        budget_seconds=600,
    )
    assert rebooted["exec_allowed"] is False and rebooted["assume_disarmed"] is True, (
        rebooted
    )
    assert "did not return Ready" in rebooted["wait_error"], rebooted
    expired = authorization.host_reachability(
        _Regional(ready=False, boot="boot-a"),
        node="node-a",
        baseline_boot_id="boot-a",
        holder_armed_at=NOW - timedelta(seconds=2 * 900 + 121),
        max_hold_seconds=900,
        budget_seconds=600,
    )
    assert expired["assume_disarmed"] is True and "lifetime" in expired["reason"], (
        expired
    )
    alive = authorization.host_reachability(
        _Regional(ready=False, boot="boot-a"),
        node="node-a",
        baseline_boot_id="boot-a",
        holder_armed_at=NOW,
        max_hold_seconds=900,
        budget_seconds=600,
    )
    assert alive["exec_allowed"] is False and alive["assume_disarmed"] is False, alive


def test_executor_bounds_are_ints_or_a_refusal() -> None:
    bounds = {
        "node_workflow_lifetime_seconds": "3600",
        "step_waiting_timeout_seconds": 600,
        "verify_waiting_limit_seconds": 600,
        "restore_waiting_limit_seconds": 600,
        "agent_maintenance_window_seconds": "420",
        "gpu_client_verify_max_attempts": 60,
    }
    regional = SimpleNamespace(executor_python=lambda script: dict(bounds))
    result = authorization.executor_bounds(regional)
    assert result["agent_maintenance_window_seconds"] == 420, result
    assert all(type(value) is int for value in result.values()), result
    bounds.pop("agent_maintenance_window_seconds")
    with pytest.raises(
        authorization.RegionalFixtureError, match="executor bounds lack"
    ):
        authorization.executor_bounds(regional)


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
