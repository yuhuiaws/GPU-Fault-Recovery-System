from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import destr_barrier_authorization as authorization
from scripts.e2e.regional.probes import destr017_node_probe as reboot
from scripts.e2e.regional.probes import destructive_node_probe as injection

WINDOW_END = datetime(2099, 1, 1, tzinfo=timezone.utc)
NODE = "node-a"
WORKFLOW = "workflow-a"


@pytest.mark.parametrize("deadline", ["2000-01-01T00:00:00Z", "invalid", "2099-01-01"])
def test_delayed_xid_refuses_before_opening_kmsg(
    deadline: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[str] = []
    monkeypatch.setattr(
        injection, "os", SimpleNamespace(open=lambda path, *_args: opened.append(path))
    )
    arguments = argparse.Namespace(maintenance_window_end=deadline)

    with pytest.raises(injection.ProbeError, match="maintenance window"):
        injection.write_xid79(arguments)
    assert opened == []


def reboot_state(deadline: str) -> dict[str, Any]:
    """A holder pre-authorized for the reboot, opened on this run's quiesce."""

    now = datetime.now(timezone.utc)
    proof = authorization.conditional_pre_authorization(
        run_id="review",
        node=NODE,
        boot_id="boot-a",
        device="/dev/nvidia0",
        drill_id="review",
        marker="review-m",
        maintenance_window_end=WINDOW_END,
        maintenance_window_seconds=600,
        valid_for_seconds=600,
        not_before_seconds={"reboot": 30},
        now=now,
    )
    return {
        "run_id": "review",
        "device": "/dev/nvidia0",
        "drill_id": "review",
        "boot_id": "boot-a",
        "max_hold_seconds": 900,
        "maintenance_window_end": deadline,
        "boot_id_before_reboot": "boot-a",
        "reboot_cancelled_at": None,
        "pre_authorization": proof,
        "pre_authorized_at": now.isoformat(),
        "ledger_baseline_workflow_ids": [],
        "hold_started_at": (now - timedelta(seconds=60)).isoformat(),
    }


def barrier_rows() -> list[dict[str, Any]]:
    """The ledger of a workflow parked at the client verification, right now."""

    stamp = datetime.now(timezone.utc).isoformat()
    return [
        {
            "command_id": f"{WORKFLOW}/2/QUIESCE_GPU_SERVICES/{NODE}/agent-4",
            "operation": "QUIESCE_GPU_SERVICES",
            "state": "SUCCEEDED",
            "attempt": 1,
            "started_at": stamp,
            "completed_at": stamp,
            "workflow_request_id": WORKFLOW,
            "incident_id": "incident-a",
            "agent_generation": 4,
            "error": None,
        },
        {
            "command_id": f"{WORKFLOW}/3/VERIFY_NO_GPU_CLIENTS/{NODE}/agent-4",
            "operation": "VERIFY_NO_GPU_CLIENTS",
            "state": "FAILED",
            "attempt": 1,
            "started_at": stamp,
            "completed_at": stamp,
            "workflow_request_id": WORKFLOW,
            "incident_id": "incident-a",
            "agent_generation": 4,
            "error": "GPU device clients are still active: GPU-a:4242",
        },
    ]


def install(
    monkeypatch: pytest.MonkeyPatch, path: Path, commands: list[list[str]]
) -> None:
    monkeypatch.setattr(reboot, "state_path", lambda _run: path)
    monkeypatch.setattr(reboot, "boot_id", lambda: "boot-a")
    monkeypatch.setattr(reboot, "ledger_rows", barrier_rows)
    monkeypatch.setattr(reboot, "holder_active", lambda _run: True)
    monkeypatch.setattr(
        reboot, "run", lambda argv, **_kwargs: commands.append(list(argv))
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"maintenance_window_end": "2000-01-01T00:00:00Z"},
        {"maintenance_window_end": ""},
        {"maintenance_window_end": "2099-01-01"},
        {"reboot_cancelled_at": "2026-09-12T00:00:00Z"},
        {"fire_requested_at": "2026-09-12T00:00:00Z"},
        {"boot_id_before_reboot": "another-boot"},
        {"run_id": "another-run"},
    ],
)
def test_timer_fire_rechecks_deadline_boot_and_single_use(
    overrides: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "state.json"
    reboot.write_state(path, {**reboot_state("2099-01-01T00:00:00Z"), **overrides})
    commands: list[list[str]] = []
    install(monkeypatch, path, commands)

    with pytest.raises(reboot.ProbeError):
        reboot.fire_reboot(argparse.Namespace(run_id="review"))
    assert reboot.reboot_command() not in commands, commands


def test_fire_intent_is_persisted_before_the_only_reboot_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "state.json"
    reboot.write_state(path, reboot_state("2099-01-01T00:00:00Z"))
    commands: list[list[str]] = []
    install(monkeypatch, path, commands)

    def run(argv: list[str], **_kwargs: Any) -> None:
        if list(argv) == reboot.reboot_command():
            assert reboot.read_state(path).get("fire_requested_at"), (
                "durable intent must precede the reboot request"
            )
        commands.append(list(argv))

    monkeypatch.setattr(reboot, "run", run)
    arguments = argparse.Namespace(run_id="review")

    reboot.fire_reboot(arguments)
    with pytest.raises(reboot.ProbeError, match="consumed"):
        reboot.fire_reboot(arguments)
    assert [argv for argv in commands if argv == reboot.reboot_command()] == [
        reboot.reboot_command()
    ], commands
    condition = reboot.read_state(path)["fire_condition"]
    assert condition["workflow_request_id"] == WORKFLOW, condition
    assert condition["boot_id"] == "boot-a" and condition["incident_id"] == (
        "incident-a"
    ), condition


def test_reboot_waits_for_the_verification_row_and_gives_up_at_the_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "state.json"
    state = reboot_state("2099-01-01T00:00:00Z")
    state["pre_authorization"]["expires_at"] = (
        datetime.now(timezone.utc) + timedelta(seconds=12)
    ).isoformat()
    reboot.write_state(path, state)
    commands: list[list[str]] = []
    install(monkeypatch, path, commands)
    monkeypatch.setattr(reboot, "ledger_rows", lambda: barrier_rows()[:1])
    slept: list[float] = []
    monkeypatch.setattr(
        reboot, "time", SimpleNamespace(sleep=slept.append, monotonic=lambda: 0.0)
    )
    monkeypatch.setattr(reboot, "POLL_SECONDS", 0)

    def ticking_now(tz: Any = None) -> datetime:
        return datetime.now(timezone.utc) + timedelta(seconds=5 * len(slept))

    monkeypatch.setattr(
        reboot,
        "datetime",
        SimpleNamespace(now=ticking_now, fromisoformat=datetime.fromisoformat),
    )

    with pytest.raises(reboot.ProbeError, match="expired"):
        reboot.fire_reboot(argparse.Namespace(run_id="review"))
    assert reboot.reboot_command() not in commands, commands
    assert len(slept) >= 2, "the timer re-checked the ledger before giving up"
    recorded = reboot.read_state(path)
    assert "expired" in recorded["reboot_refusal"]["reason"], recorded
    assert recorded["reboot_cancelled_at"] and recorded["disarmed_at"], recorded


def test_timer_cannot_be_placed_beyond_the_approved_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "state.json"
    end = datetime.now(timezone.utc) + timedelta(seconds=10)
    reboot.write_state(path, reboot_state(end.isoformat()))
    commands: list[Any] = []
    monkeypatch.setattr(reboot, "run", lambda *args, **kwargs: commands.append(args))

    with pytest.raises(reboot.ProbeError, match="outlive"):
        reboot.arm_reboot("review", 30, path)
    assert commands == []
