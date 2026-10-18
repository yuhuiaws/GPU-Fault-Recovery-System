"""GF-REGIONAL-DESTR-017 node probe: the refusals and acceptance of the
conditional pre-authorization check, the ledger-shaped barrier verdicts that a
real ledger cannot produce (a quiesce that failed or never finished, rows with
no timestamps, malformed proofs), the holder arm with a reboot delay, and
``clear-state`` after a completed cancellation. Synthetic ledger rows stand in
for the SQLite ledger where the shape under test is what the ledger would never
write on its own."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional.probes import destr017_node_probe as probe

RUN_ID = "destr017-refusal-a1"
NODE = "node-a"
WORKFLOW = "wf-new"
T0 = datetime(2026, 9, 6, 10, 0, tzinfo=timezone.utc)
NOW = T0 + timedelta(seconds=60)
BOOT_A = "11111111-1111-4111-8111-111111111111"
BOOT_B = "22222222-2222-4222-8222-222222222222"


def proof(**overrides: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "kind": probe.CONDITIONAL_KIND,
        "conditional": True,
        "run_id": RUN_ID,
        "node_id": NODE,
        "boot_id": BOOT_A,
        "device": "/dev/nvidia0",
        "drill_id": RUN_ID,
        "maintenance_window_end": (T0 + timedelta(hours=1)).isoformat(),
        "maintenance_window_seconds": 420,
        "authorized_at": T0.isoformat(),
        "expires_at": (T0 + timedelta(seconds=600)).isoformat(),
        "not_before_seconds": {probe.REBOOT_PHASE: 30},
        "ledger": dict(probe.LEDGER_SHAPE),
    }
    value.update(overrides)
    return value


def armed_state(**overrides: Any) -> dict[str, Any]:
    state: dict[str, Any] = {
        "run_id": RUN_ID,
        "boot_id": BOOT_A,
        "device": "/dev/nvidia0",
        "drill_id": RUN_ID,
        "max_hold_seconds": 900,
    }
    state.update(overrides)
    return state


def parked_state(**overrides: Any) -> dict[str, Any]:
    state = armed_state(
        boot_id_before_reboot=BOOT_A,
        hold_started_at=(T0 + timedelta(seconds=20)).isoformat(),
        pre_authorization=proof(),
        pre_authorized_at=T0.isoformat(),
        ledger_baseline_workflow_ids=["wf-old"],
    )
    state.update(overrides)
    return state


def row(
    operation: str,
    state: str,
    *,
    index: int,
    started: int | None = 20,
    completed: int | None = 25,
    error: str | None = None,
    attempt: int = 1,
) -> dict[str, Any]:
    return {
        "command_id": f"{WORKFLOW}/{index}/{operation}/{NODE}/agent-4",
        "attempt": attempt,
        "state": state,
        "operation": operation,
        "started_at": None
        if started is None
        else (T0 + timedelta(seconds=started)).isoformat(),
        "completed_at": None
        if completed is None
        else (T0 + timedelta(seconds=completed)).isoformat(),
        "workflow_request_id": WORKFLOW,
        "incident_id": "inc-1",
        "agent_generation": 4,
        "error": error,
    }


QUIESCE = probe.LEDGER_SHAPE["quiesce"]
VERIFY = probe.LEDGER_SHAPE["verify"]
REFUSED_VERIFY = row(
    VERIFY,
    "FAILED",
    index=3,
    started=40,
    completed=45,
    error="GPU device clients are still active: GPU-a:4242",
)


# --------------------------------------------------------------------------- #
# run()
# --------------------------------------------------------------------------- #
def test_run_names_the_failed_command_and_its_stderr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[dict[str, Any]] = []

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        seen.append({"command": list(command), **kwargs})
        return subprocess.CompletedProcess(command, 4, "", " unit not found \n")

    monkeypatch.setattr(probe.subprocess, "run", fake_run)
    with pytest.raises(
        probe.ProbeError,
        match=r"command failed \(4\): systemctl show x: unit not found",
    ):
        probe.run(["systemctl", "show", "x"])
    assert seen[0]["timeout"] == 180
    assert seen[0]["check"] is False
    unchecked = probe.run(["systemctl", "show", "x"], check=False, timeout=7)
    assert unchecked.returncode == 4
    assert seen[1]["timeout"] == 7


# --------------------------------------------------------------------------- #
# check_pre_authorization
# --------------------------------------------------------------------------- #
def test_a_binding_pre_authorization_is_accepted_unchanged() -> None:
    value = proof()
    accepted = probe.check_pre_authorization(
        armed_state(), value, run_id=RUN_ID, boot_id=BOOT_A, now=NOW
    )
    assert accepted is value, "the accepted proof is the proof that was offered"


@pytest.mark.parametrize(
    ("state_changes", "proof_changes", "message"),
    [
        ({"run_id": "other-run"}, {}, "no armed holder is recorded"),
        ({"disarmed_at": T0.isoformat()}, {}, "already disarmed"),
        ({"pre_authorization": {"kind": "x"}}, {}, "already pre-authorized"),
        ({"reboot_armed_at": T0.isoformat()}, {}, "already pre-authorized"),
        ({"reboot_cancelled_at": T0.isoformat()}, {}, "was cancelled"),
        ({}, {"kind": "unconditional"}, "not a conditional barrier"),
        ({}, {"conditional": "yes"}, "not a conditional barrier"),
        ({}, {"device": "/dev/nvidia1"}, "does not bind this holder"),
        ({}, {"drill_id": "other-drill"}, "does not bind this holder"),
        ({}, {"boot_id": BOOT_B}, "belongs to another boot"),
        ({"boot_id": BOOT_B}, {}, "belongs to another boot"),
        ({}, {"node_id": "../node"}, "names no node"),
        ({}, {"node_id": None}, "names no node"),
        ({}, {"ledger": {"quiesce": "RESET_GPU"}}, "ledger shape is not this probe's"),
        ({}, {"expires_at": "not a time"}, "no valid deadline"),
        ({}, {"expires_at": "2026-09-06T10:10:00"}, "no valid deadline"),
        ({}, {"maintenance_window_end": None}, "no valid deadline"),
        ({}, {"authorized_at": ""}, "no valid deadline"),
        ({}, {"expires_at": T0.isoformat()}, "expired or future-dated"),
        (
            {},
            {"maintenance_window_end": (T0 + timedelta(seconds=30)).isoformat()},
            "expired or future-dated",
        ),
        (
            {},
            {"authorized_at": (NOW + timedelta(seconds=91)).isoformat()},
            "expired or future-dated",
        ),
        ({"max_hold_seconds": 120}, {}, "outlives the holder's bounded lifetime"),
        ({}, {"maintenance_window_seconds": 29}, "maintenance window is out of bounds"),
        (
            {},
            {"maintenance_window_seconds": "420"},
            "maintenance window is out of bounds",
        ),
        ({}, {"not_before_seconds": [30]}, "exactly the reboot delay"),
        (
            {},
            {"not_before_seconds": {"reboot": 30, "x": 1}},
            "exactly the reboot delay",
        ),
        ({}, {"not_before_seconds": {"reboot": "30"}}, "exactly the reboot delay"),
    ],
)
def test_pre_authorization_defects_are_refused(
    state_changes: dict[str, Any], proof_changes: dict[str, Any], message: str
) -> None:
    with pytest.raises(probe.ProbeError, match=message):
        probe.check_pre_authorization(
            armed_state(**state_changes),
            proof(**proof_changes),
            run_id=RUN_ID,
            boot_id=BOOT_A,
            now=NOW,
        )


def test_pre_authorization_without_a_boot_id_is_refused() -> None:
    with pytest.raises(probe.ProbeError, match="belongs to another boot"):
        probe.check_pre_authorization(
            armed_state(), proof(), run_id=RUN_ID, boot_id="", now=NOW
        )


# --------------------------------------------------------------------------- #
# barrier_condition on shapes a real ledger never writes
# --------------------------------------------------------------------------- #
def _barrier(
    rows: list[dict[str, Any]], state: dict[str, Any] | None = None
) -> dict[str, Any]:
    return probe.barrier_condition(
        state or parked_state(), rows, now=NOW, boot_id=BOOT_A, holder_active=True
    )


def test_the_barrier_holds_on_synthetic_rows_shaped_like_the_ledger() -> None:
    verdict = _barrier([row(QUIESCE, "SUCCEEDED", index=2), REFUSED_VERIFY])
    assert verdict["holds"] is True, verdict
    assert verdict["condition"]["workflow_request_id"] == WORKFLOW
    assert (
        verdict["condition"]["pinned_window_expires_at"]
        == (T0 + timedelta(seconds=20 + 420)).isoformat()
    )


@pytest.mark.parametrize(
    ("proof_changes", "reason"),
    [
        ({"ledger": {"quiesce": "RESET_GPU"}}, "ledger shape is not this probe's"),
        ({"expires_at": "garbage"}, "no valid deadline"),
        ({"maintenance_window_end": "2026-09-06T11:00:00"}, "no valid deadline"),
        ({"not_before_seconds": {"reboot": "30"}}, "names no reboot delay"),
        ({"not_before_seconds": None}, "names no reboot delay"),
    ],
)
def test_a_malformed_pre_authorization_is_a_final_refusal(
    proof_changes: dict[str, Any], reason: str
) -> None:
    state = parked_state(pre_authorization=proof(**proof_changes))
    verdict = _barrier([row(QUIESCE, "SUCCEEDED", index=2), REFUSED_VERIFY], state)
    assert verdict["holds"] is False
    assert verdict["final"] is True, verdict
    assert reason in verdict["reason"], verdict


def test_a_failed_quiesce_is_final_and_an_unfinished_one_waits() -> None:
    failed = _barrier([row(QUIESCE, "FAILED", index=2), REFUSED_VERIFY])
    assert (failed["final"], failed["reason"]) == (True, "the quiesce did not succeed")
    pending = _barrier([row(QUIESCE, probe.IN_PROGRESS_STATE, index=2, completed=None)])
    assert (pending["holds"], pending["final"]) == (False, False)
    assert pending["reason"] == "the quiesce has not succeeded"
    absent = _barrier([REFUSED_VERIFY])
    assert absent["reason"] == "the quiesce has not succeeded"
    assert absent["final"] is False


def test_a_quiesce_row_without_timestamps_cannot_pin_the_window() -> None:
    verdict = _barrier(
        [
            row(QUIESCE, "SUCCEEDED", index=2, started=None, completed=None),
            REFUSED_VERIFY,
        ]
    )
    assert verdict["final"] is True
    assert verdict["reason"] == "the pinned maintenance window cannot be derived"


def test_the_window_is_pinned_from_the_earliest_timestamp() -> None:
    late_start = row(QUIESCE, "SUCCEEDED", index=2, started=None, completed=400)
    verdict = _barrier([late_start, REFUSED_VERIFY])
    assert verdict["holds"] is True, verdict
    assert (
        verdict["condition"]["pinned_window_expires_at"]
        == (T0 + timedelta(seconds=400 + 420)).isoformat()
    )


# --------------------------------------------------------------------------- #
# arm_holder
# --------------------------------------------------------------------------- #
def _arm_arguments(**overrides: Any) -> argparse.Namespace:
    values: dict[str, Any] = {
        "run_id": RUN_ID,
        "drill_id": RUN_ID,
        "device": "/dev/nvidia0",
        "max_hold_seconds": 600,
        "after_ledger_op": "QUIESCE_GPU_SERVICES",
        "probe_script": "/opt/acceptance/destr017_node_probe.py",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_arm_holder_refuses_a_foreign_probe_script(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        probe,
        "run",
        lambda command, **kwargs: pytest.fail("no unit may start on a refusal"),
    )
    with pytest.raises(probe.ProbeError, match="probe script identity mismatch"):
        probe.arm_holder(_arm_arguments(probe_script="/opt/other/holder.py"))
    with pytest.raises(probe.ProbeError, match="not permitted for arming"):
        probe.arm_holder(_arm_arguments(after_ledger_op="RESET_GPU"))


def test_arm_holder_with_a_reboot_delay_needs_a_maintenance_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(probe, "state_path", lambda run_id: tmp_path / "state.json")
    monkeypatch.setattr(probe, "ledger_rows", lambda: [])
    monkeypatch.setattr(
        probe,
        "run",
        lambda command, **kwargs: pytest.fail("no unit may start on a refusal"),
    )
    with pytest.raises(probe.ProbeError, match="explicit maintenance deadline"):
        probe.arm_holder(_arm_arguments(reboot_delay_seconds=45))
    with pytest.raises(probe.ProbeError, match="maintenance window ended"):
        probe.arm_holder(
            _arm_arguments(
                reboot_delay_seconds=45,
                maintenance_window_end="2000-01-01T00:00:00+00:00",
            )
        )
    with pytest.raises(probe.ProbeError, match="reboot delay"):
        probe.arm_holder(
            _arm_arguments(
                reboot_delay_seconds=5,
                maintenance_window_end="2099-01-01T00:00:00+00:00",
            )
        )
    assert not (tmp_path / "state.json").exists(), "no state before the checks pass"


def test_arm_holder_records_the_reboot_delay_and_starts_the_watch_unit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(probe, "state_path", lambda run_id: tmp_path / "state.json")
    monkeypatch.setattr(
        probe,
        "ledger_rows",
        lambda: [
            {
                "command_id": "wf-old/2/QUIESCE_GPU_SERVICES/node-a/agent-3",
                "operation": "QUIESCE_GPU_SERVICES",
            },
            {
                "command_id": "wf-old/5/RESET_GPU/node-a/agent-3",
                "operation": "RESET_GPU",
            },
        ],
    )
    monkeypatch.setattr(probe, "boot_id", lambda: BOOT_A)
    monkeypatch.setattr(
        probe,
        "run",
        lambda command, **kwargs: calls.append(list(command))
        or subprocess.CompletedProcess(command, 0, "", ""),
    )
    probe.arm_holder(
        _arm_arguments(
            reboot_delay_seconds=45, maintenance_window_end="2099-01-01T00:00:00+00:00"
        )
    )
    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert state["reboot_delay_seconds"] == 45
    assert state["maintenance_window_end"] == "2099-01-01T00:00:00+00:00"
    assert state["baseline_command_ids"] == [
        "wf-old/2/QUIESCE_GPU_SERVICES/node-a/agent-3"
    ]
    assert state["boot_id"] == BOOT_A
    started = [call for call in calls if call[0] == "systemd-run"]
    assert len(started) == 1, calls
    assert started[0][-3:] == ["watch-ledger", "--run-id", RUN_ID]
    assert "--property=RuntimeMaxSec=660" in started[0]
    emitted = json.loads(capsys.readouterr().out.strip())
    assert emitted["reboot_delay_seconds"] == 45
    assert emitted["arm_unit"] == probe.arm_unit(RUN_ID) + ".service"


# --------------------------------------------------------------------------- #
# clear_state
# --------------------------------------------------------------------------- #
def test_clear_state_removes_a_completed_cancellation_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "state.json"
    monkeypatch.setattr(probe, "state_path", lambda run_id: path)
    monkeypatch.setattr(
        probe,
        "_unit_state",
        lambda unit, *properties: {"LoadState": "not-found", "ActiveState": "inactive"},
    )
    probe.write_state(
        path,
        {
            "run_id": RUN_ID,
            "reboot_cancelled_at": T0.isoformat(),
            "disarmed_at": T0.isoformat(),
        },
    )
    probe.clear_state(argparse.Namespace(run_id=RUN_ID))
    assert not path.exists(), "the completed record is removed"
    assert json.loads(capsys.readouterr().out.strip()) == {
        "run_id": RUN_ID,
        "state_cleared": True,
    }

    probe.write_state(path, {"run_id": RUN_ID, "reboot_cancelled_at": T0.isoformat()})
    with pytest.raises(probe.ProbeError, match="no completed cancellation proof"):
        probe.clear_state(argparse.Namespace(run_id=RUN_ID))
    assert path.exists(), "an incomplete record is kept for the operator"

    monkeypatch.setattr(
        probe,
        "_unit_state",
        lambda unit, *properties: {"LoadState": "loaded", "ActiveState": "active"},
    )
    with pytest.raises(probe.ProbeError, match="probe unit may still run"):
        probe.clear_state(argparse.Namespace(run_id=RUN_ID))
