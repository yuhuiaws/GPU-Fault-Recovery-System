from __future__ import annotations

import argparse
import hashlib
import json
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional.probes import destr014_node_probe as p014
from scripts.e2e.regional.probes import destr016_node_probe as p016
from scripts.e2e.regional.probes import destr017_node_probe as p017
from scripts.e2e.regional.probes import destr018_node_probe as p018
from tests.regional._cov95_destr_probes import (
    GUARD,
    INCIDENT,
    NOW,
    RUN_ID,
    WORKFLOW,
    ProbeHarness,
)

MODULES = (p014, p016, p017, p018)


@pytest.mark.parametrize("module", MODULES)
@pytest.mark.parametrize("matched", [False, True])
def test_holder_arms_only_after_new_ledger_row_and_disarms_its_units(
    module: Any, matched: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ProbeHarness(module, tmp_path, monkeypatch)
    module.arm_holder(h.arm_arguments())
    armed = json.loads(h.state.read_text())
    assert armed["device"] == "/dev/nvidia0" and armed["baseline_command_ids"] == [], (
        armed
    )
    h.calls.clear()
    if matched:
        h.append_arm_row()
        h.row_delays = 1
    module.watch_ledger(argparse.Namespace(run_id=RUN_ID))
    state = json.loads(h.state.read_text())
    holders = [command for command, _ in h.calls if command[0] == "systemd-run"]
    if matched:
        assert (
            len(holders) == 1
            and state["matched_row"]["command_id"] == "arm-command-owned"
        ), (state, holders)
        assert "/dev/nvidia0" in holders[0], holders
        assert state["hold_started_at"], state
    else:
        assert (
            holders == [] and state["holder_error"] == "arm ledger row never appeared"
        ), (state, holders)
    module.holder_status(argparse.Namespace(run_id=RUN_ID))
    assert h.records[-1]["run_id"] == RUN_ID, h.records
    module.disarm_holder(argparse.Namespace(run_id=RUN_ID))
    assert json.loads(h.state.read_text())["disarmed_at"], h.state
    assert all(values["ActiveState"] == "inactive" for values in h.units.values()), (
        h.units
    )


@pytest.mark.parametrize("module", MODULES)
def test_holder_refuses_invalid_arm_operation_without_running_a_command(
    module: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ProbeHarness(module, tmp_path, monkeypatch)
    args = h.arm_arguments()
    args.after_ledger_op = "RESET_GPU"
    with pytest.raises(module.ProbeError):
        module.arm_holder(args)
    assert h.calls == [], h.calls
    h.state.write_text(json.dumps({"after_ledger_op": "RESET_GPU"}), encoding="utf-8")
    with pytest.raises(module.ProbeError):
        module.watch_ledger(argparse.Namespace(run_id=RUN_ID))
    assert h.calls == [], h.calls


@pytest.mark.parametrize("module", MODULES)
@pytest.mark.parametrize("with_state", [False, True])
def test_snapshot_and_idempotent_disarm_work_without_a_previous_holder(
    module: Any, with_state: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ProbeHarness(module, tmp_path, monkeypatch)
    if with_state:
        module.write_state(h.state, {"run_id": RUN_ID})
    module.disarm_holder(argparse.Namespace(run_id=RUN_ID))
    code, report = h.main(
        monkeypatch, "snapshot", *(["--run-id", RUN_ID] if with_state else [])
    )
    assert code == 0 and report["boot_id"] == "boot-before", report
    if with_state:
        assert report["state"]["run_id"] == RUN_ID, report
    else:
        assert "state" not in report, report


@pytest.mark.parametrize("module", MODULES)
def test_probe_main_reports_invalid_device_as_failure(
    module: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ProbeHarness(module, tmp_path, monkeypatch)
    code, report = h.main(
        monkeypatch,
        "arm-holder",
        "--run-id",
        RUN_ID,
        "--drill-id",
        "drill-owned",
        "--device",
        "/dev/nvidia999",
        "--probe-script",
        str(module.__file__),
        "--max-hold-seconds",
        "60",
        "--after-ledger-op",
        h.arm_operation,
    )
    assert code == 1 and "error" in report, report
    assert h.calls == [], h.calls


def test_immediate_lifetime_holder_reports_clients_and_rejects_failed_disarm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ProbeHarness(p018, tmp_path, monkeypatch)
    h.clients = [{"pid": "222"}]
    p018.arm_holder(h.arm_arguments(ledger=False))
    assert h.records[-1]["arm_mode"] == "immediate", h.records
    assert h.records[-1]["device_clients"] == [{"pid": "222"}], h.records
    p018.update_state(h.state, {"holder_pid": "222"})
    with pytest.raises(p018.ProbeError, match="still holds"):
        p018.disarm_holder(argparse.Namespace(run_id=RUN_ID))
    assert json.loads(h.state.read_text())["disarmed_at"], h.state
    h.clients = []
    p018.disarm_holder(argparse.Namespace(run_id=RUN_ID))
    assert h.records[-1]["device_clients"] == [], h.records


@pytest.mark.parametrize("module", (p016, p017))
def test_watch_ledger_refuses_expired_delayed_action_window(
    module: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ProbeHarness(module, tmp_path, monkeypatch)
    module.arm_holder(h.arm_arguments())
    h.append_arm_row()
    if module is p016:
        module.update_state(
            h.state, {"injections": [{"maintenance_window_end": NOW.isoformat()}]}
        )
    else:
        module.update_state(
            h.state,
            {"reboot_delay_seconds": 30, "maintenance_window_end": NOW.isoformat()},
        )
    h.calls.clear()
    with pytest.raises(module.ProbeError, match="maintenance window ended"):
        module.watch_ledger(argparse.Namespace(run_id=RUN_ID))
    assert h.calls == [], h.calls


def test_branch_exhaustion_keeps_agent_disable_refused_and_restores_legacy_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ProbeHarness(p014, tmp_path, monkeypatch)
    args = argparse.Namespace(run_id=RUN_ID)
    with pytest.raises(p014.ProbeError, match="recovery safeguard"):
        p014.disable_agent_restart(args)
    assert h.calls == [], h.calls
    with pytest.raises(p014.ProbeError, match="unbound legacy"):
        p014.restore_agent(args)
    p014.write_state(
        h.state,
        {
            "agent_baseline": p014.agent_baseline_record(
                enabled_state="enabled", active_state="active"
            )
        },
    )
    with pytest.raises(p014.ProbeError, match="unbound legacy"):
        p014.restore_agent(args)
    assert h.calls == [], (
        "a baseline alone cannot authorize unbound service restoration"
    )
    h.agent_active = False
    h.agent_start_stuck = True
    with pytest.raises(p014.ProbeError, match="unbound legacy"):
        p014.restore_agent(args)
    assert h.calls == [], (
        "an inactive service does not repair missing recovery authority"
    )


# --------------------------------------------------------------------------- #
# DESTR-016: conditional pre-authorization, timers that decide for themselves
# --------------------------------------------------------------------------- #
def seed_injection(h: ProbeHarness) -> None:
    """An armed holder with both writes planned, before pre-authorization."""

    p016.write_state(
        h.state,
        {
            "run_id": RUN_ID,
            "drill_id": "drill-owned",
            "device": "/dev/nvidia0",
            "boot_id": "boot-before",
            "max_hold_seconds": 60,
            "armed_at": NOW.isoformat(),
            "injection_script_sha256": hashlib.sha256(
                h.guard_path.read_bytes()
            ).hexdigest(),
            "injections": [
                {
                    "phase": "absorb",
                    "script": GUARD,
                    "subcommand": "write-xid46",
                    "marker": "marker-absorb",
                    "drill_id": "drill-owned-s",
                    "pci_bdf": "0000:01:00",
                    "after_seconds": 10,
                    "maintenance_window_end": (NOW + timedelta(minutes=10)).isoformat(),
                },
                {
                    "phase": "escalate",
                    "script": GUARD,
                    "subcommand": "write-xid79",
                    "marker": "marker-owned",
                    "drill_id": "drill-owned",
                    "pci_bdf": "0000:01:00",
                    "after_seconds": 20,
                    "maintenance_window_end": (NOW + timedelta(minutes=10)).isoformat(),
                },
            ],
        },
    )


def pre_authorize(
    h: ProbeHarness, proof: dict[str, Any] | None = None
) -> dict[str, Any]:
    proof = proof or h.pre_authorization({"absorb": 10, "escalate": 20})
    p016.pre_authorize(
        argparse.Namespace(run_id=RUN_ID, authorization=json.dumps(proof))
    )
    return proof


def armed_and_parked(h: ProbeHarness) -> None:
    """Pre-authorized, holder open on the quiesce, ledger parked at the barrier."""

    seed_injection(h)
    pre_authorize(h)
    p016.update_state(h.state, {"hold_started_at": h.clock.now().isoformat()})
    h.holder_alive()
    h.rows = h.barrier_rows()


def test_pre_authorization_places_both_timers_and_records_the_ledger_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ProbeHarness(p016, tmp_path, monkeypatch)
    seed_injection(h)
    h.rows = h.barrier_rows(workflow="workflow-old", offset_seconds=-600)
    proof = pre_authorize(h)
    state = json.loads(h.state.read_text())
    assert state["pre_authorization"] == proof, state
    assert state["ledger_baseline_workflow_ids"] == ["workflow-old"], state
    timers = [command for command, _ in h.calls if command[0] == "systemd-run"]
    assert [command[command.index("--phase") + 1] for command in timers] == [
        "absorb",
        "escalate",
    ], timers
    assert all(
        "fire-injection" in command and "--on-active=1s" in command
        for command in timers
    ), timers
    assert not any("write-xid" in " ".join(command) for command in timers), (
        "the timers decide; they never carry the write themselves"
    )
    assert h.records[-1]["not_before_seconds"] == {"absorb": 10, "escalate": 20}, (
        h.records
    )
    assert h.records[-1]["expires_at"] == proof["expires_at"], h.records


@pytest.mark.parametrize(
    "defect",
    [
        "device",
        "drill",
        "boot",
        "kind",
        "ledger",
        "outlives-hold",
        "phases",
        "repeat",
        "disarmed",
        "race",
        "window-ended",
    ],
)
def test_pre_authorization_refuses_a_proof_that_does_not_bind_this_holder(
    defect: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ProbeHarness(p016, tmp_path, monkeypatch)
    seed_injection(h)
    overrides: dict[str, Any] = {}
    phases = {"absorb": 10, "escalate": 20}
    if defect == "device":
        overrides["device"] = "/dev/nvidia1"
    elif defect == "drill":
        overrides["drill_id"] = "another-drill"
    elif defect == "boot":
        overrides["boot_id"] = "boot-other"
    elif defect == "kind":
        overrides["kind"] = "exact-barrier-authorization"
    elif defect == "ledger":
        overrides["ledger"] = {"quiesce": "QUIESCE_GPU_SERVICES"}
    elif defect == "outlives-hold":
        overrides["expires_at"] = (NOW + timedelta(seconds=61)).isoformat()
    elif defect == "phases":
        phases = {"escalate": 20}
    elif defect == "repeat":
        pre_authorize(h)
    elif defect == "disarmed":
        p016.update_state(h.state, {"disarmed_at": NOW.isoformat()})
    elif defect == "race":
        p016.update_state(h.state, {"arm_race_lost": True})
    else:
        overrides["maintenance_window_end"] = NOW.isoformat()
    h.calls.clear()
    with pytest.raises(p016.ProbeError):
        pre_authorize(h, h.pre_authorization(phases, **overrides))
    assert not any(command[0] == "systemd-run" for command, _ in h.calls), h.calls


def test_injection_fires_once_the_ledger_shows_the_barrier_and_never_repeats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ProbeHarness(p016, tmp_path, monkeypatch)
    armed_and_parked(h)
    h.calls.clear()
    p016.fire_injection(argparse.Namespace(run_id=RUN_ID, phase="absorb"))
    writes = [command for command, _ in h.calls if "write-xid46" in command]
    assert len(writes) == 1 and "marker-absorb" in writes[0], h.calls
    assert "--barrier-authorization" not in writes[0], (
        "no exact store proof exists on the node; the ledger is the authority"
    )
    # The relative delay is the earliest time, honoured by waiting for it.
    assert h.clock.elapsed >= 10, h.clock.elapsed
    record = json.loads(h.state.read_text())["injections_fired"]["absorb"]
    assert record["condition"]["workflow_request_id"] == WORKFLOW, record
    assert record["condition"]["incident_id"] == INCIDENT, record
    assert record["condition"]["boot_id"] == "boot-before", record
    assert record["fire_requested_at"] <= record["fired_at"], record
    with pytest.raises(p016.ProbeError, match="consumed"):
        p016.fire_injection(argparse.Namespace(run_id=RUN_ID, phase="absorb"))
    assert sum("write-xid46" in command for command, _ in h.calls) == 1, h.calls


def test_escalation_waits_for_the_absorbed_fault_and_gives_up_at_the_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ProbeHarness(p016, tmp_path, monkeypatch)
    armed_and_parked(h)
    h.clock.sleep(20)
    h.calls.clear()
    with pytest.raises(p016.ProbeError, match="expired"):
        p016.fire_injection(argparse.Namespace(run_id=RUN_ID, phase="escalate"))
    state = json.loads(h.state.read_text())
    assert "expired" in state["injection_refusals"]["escalate"]["reason"], state
    assert state["disarmed_at"] and "escalate" in state["disarm_reason"], state
    assert not any("write-xid79" in command for command, _ in h.calls), h.calls
    assert h.units[p016.holder_unit(RUN_ID) + ".service"]["ActiveState"] == (
        "inactive"
    ), h.units


def test_escalation_fires_after_the_absorbed_fault_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ProbeHarness(p016, tmp_path, monkeypatch)
    armed_and_parked(h)
    p016.fire_injection(argparse.Namespace(run_id=RUN_ID, phase="absorb"))
    h.calls.clear()
    p016.fire_injection(argparse.Namespace(run_id=RUN_ID, phase="escalate"))
    assert sum("write-xid79" in command for command, _ in h.calls) == 1, h.calls
    assert h.clock.elapsed >= 20, h.clock.elapsed
    fired = json.loads(h.state.read_text())["injections_fired"]
    assert fired["absorb"]["fired_at"] <= fired["escalate"]["fire_requested_at"], fired


@pytest.mark.parametrize(
    "defect",
    [
        "no-verify",
        "reset-row",
        "restore-row",
        "verify-succeeded",
        "boot",
        "holder-dead",
        "ambiguous",
        "race",
        "no-preauth",
    ],
)
def test_injection_refuses_and_disarms_when_the_barrier_cannot_hold(
    defect: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ProbeHarness(p016, tmp_path, monkeypatch)
    if defect == "no-preauth":
        seed_injection(h)
        p016.update_state(h.state, {"hold_started_at": NOW.isoformat()})
        h.holder_alive()
        h.rows = h.barrier_rows()
    else:
        armed_and_parked(h)
    h.clock.sleep(10)
    if defect == "no-verify":
        h.rows = h.barrier_rows()[:1]
    elif defect == "reset-row":
        h.rows = h.barrier_rows(extra_operations=("RESET_GPU",))
    elif defect == "restore-row":
        h.rows = h.barrier_rows(extra_operations=("RESTORE_GPU_SERVICES",))
    elif defect == "verify-succeeded":
        h.rows = h.barrier_rows(verify_state="SUCCEEDED", verify_error=None)
    elif defect == "boot":
        h.boot_path.write_text("boot-after\n", encoding="utf-8")
    elif defect == "holder-dead":
        h.holder_alive(False)
    elif defect == "ambiguous":
        h.rows = h.barrier_rows() + h.barrier_rows(workflow="workflow-second")
    elif defect == "race":
        p016.update_state(h.state, {"arm_race_lost": True})
    h.calls.clear()
    with pytest.raises(p016.ProbeError, match="refused"):
        p016.fire_injection(argparse.Namespace(run_id=RUN_ID, phase="absorb"))
    state = json.loads(h.state.read_text())
    assert "absorb" in state["injection_refusals"] and state["disarmed_at"], state
    assert not any("write-xid46" in command for command, _ in h.calls), h.calls
    assert h.units[p016.holder_unit(RUN_ID) + ".service"]["ActiveState"] == (
        "inactive"
    ), h.units
    if defect == "no-verify":
        assert "expired" in state["injection_refusals"]["absorb"]["reason"], state
    else:
        assert "expired" not in state["injection_refusals"]["absorb"]["reason"], state


# --------------------------------------------------------------------------- #
# DESTR-017: the reboot is pre-authorized, armed, and fires on the condition
# --------------------------------------------------------------------------- #
def seed_reboot(h: ProbeHarness) -> None:
    p017.write_state(
        h.state,
        {
            "run_id": RUN_ID,
            "drill_id": "drill-owned",
            "device": "/dev/nvidia0",
            "boot_id": "boot-before",
            "max_hold_seconds": 60,
            "armed_at": NOW.isoformat(),
            "maintenance_window_end": (NOW + timedelta(minutes=10)).isoformat(),
        },
    )


def pre_authorize_reboot(
    h: ProbeHarness, *, delay: int = 30, proof: dict[str, Any] | None = None
) -> dict[str, Any]:
    proof = proof or h.pre_authorization({"reboot": delay})
    p017.pre_authorize_reboot(
        argparse.Namespace(
            run_id=RUN_ID, authorization=json.dumps(proof), delay_seconds=delay
        )
    )
    return proof


def parked_for_reboot(h: ProbeHarness) -> None:
    seed_reboot(h)
    pre_authorize_reboot(h)
    p017.update_state(h.state, {"hold_started_at": h.clock.now().isoformat()})
    h.holder_alive()
    h.rows = h.barrier_rows()


def test_reboot_pre_authorization_writes_the_marker_then_arms_a_conditional_timer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ProbeHarness(p017, tmp_path, monkeypatch)
    seed_reboot(h)
    proof = pre_authorize_reboot(h)
    state = json.loads(h.state.read_text())
    assert state["boot_id_before_reboot"] == "boot-before", state
    assert state["reboot_armed_at"] and state["pre_authorization"] == proof, state
    assert state["reboot_not_before_seconds"] == 30, state
    timers = [command for command, _ in h.calls if command[0] == "systemd-run"]
    assert len(timers) == 1 and "--on-active=30s" in timers[0], timers
    assert timers[0][-3:] == ["fire-reboot", "--run-id", RUN_ID], timers
    assert "reboot" not in " ".join(timers[0][:-3]).replace(
        p017.reboot_unit(RUN_ID), ""
    ), "the timer runs the conditional fire, never systemctl reboot itself"
    assert h.records[-1]["reboot_delay_seconds"] == 30, h.records
    assert h.records[-1]["expires_at"] == proof["expires_at"], h.records


def test_reboot_fires_once_after_the_barrier_and_keeps_the_boot_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ProbeHarness(p017, tmp_path, monkeypatch)
    parked_for_reboot(h)
    h.calls.clear()
    args = argparse.Namespace(run_id=RUN_ID)
    p017.fire_reboot(args)
    assert sum(command == p017.reboot_command() for command, _ in h.calls) == 1, h.calls
    assert h.clock.elapsed >= 30, h.clock.elapsed
    state = json.loads(h.state.read_text())
    assert state["fire_requested_at"], state
    assert state["fire_condition"]["workflow_request_id"] == WORKFLOW, state
    with pytest.raises(p017.ProbeError, match="consumed"):
        p017.fire_reboot(args)
    h.boot_path.write_text("boot-after\n", encoding="utf-8")
    p017.reboot_status(args)
    status = h.records[-1]
    assert status["fired"] is True and status["boot_changes"] == 1, status
    assert status["condition"]["workflow_request_id"] == WORKFLOW, status
    assert status["fire_requested_at"] == state["fire_requested_at"], status
    p017.cancel_reboot(args)
    assert h.records[-1]["already_fired"] is True, h.records
    p017.disarm_holder(args)
    p017.clear_state(args)
    assert not h.state.exists() and h.records[-1]["state_cleared"] is True, h.records
    assert sum(command == p017.reboot_command() for command, _ in h.calls) == 1, h.calls


@pytest.mark.parametrize(
    "defect",
    [
        "cancelled",
        "wrong-holder",
        "kind",
        "window",
        "outlives-hold",
        "delay-mismatch",
        "repeat",
        "boot",
    ],
)
def test_reboot_cannot_be_pre_authorized_outside_its_holder_and_deadlines(
    defect: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ProbeHarness(p017, tmp_path, monkeypatch)
    seed_reboot(h)
    proof = h.pre_authorization({"reboot": 30})
    if defect == "cancelled":
        p017.update_state(h.state, {"reboot_cancelled_at": NOW.isoformat()})
    elif defect == "wrong-holder":
        p017.update_state(h.state, {"device": "/dev/nvidia1"})
    elif defect == "kind":
        proof["kind"] = "exact-barrier-authorization"
    elif defect == "window":
        p017.update_state(
            h.state,
            {"maintenance_window_end": (NOW + timedelta(seconds=20)).isoformat()},
        )
    elif defect == "outlives-hold":
        proof["expires_at"] = (NOW + timedelta(seconds=61)).isoformat()
    elif defect == "delay-mismatch":
        proof = h.pre_authorization({"reboot": 45})
    elif defect == "repeat":
        pre_authorize_reboot(h)
    else:
        proof["boot_id"] = "boot-other"
    h.calls.clear()
    with pytest.raises(p017.ProbeError):
        pre_authorize_reboot(h, proof=proof)
    assert not any("fire-reboot" in command for command, _ in h.calls), h.calls


@pytest.mark.parametrize(
    "defect",
    ["no-verify", "reset-row", "verify-succeeded", "boot", "holder-dead", "ambiguous"],
)
def test_reboot_refuses_cancels_and_disarms_when_the_barrier_cannot_hold(
    defect: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ProbeHarness(p017, tmp_path, monkeypatch)
    parked_for_reboot(h)
    h.clock.sleep(30)
    if defect == "no-verify":
        h.rows = h.barrier_rows()[:1]
    elif defect == "reset-row":
        h.rows = h.barrier_rows(extra_operations=("RESET_GPU",))
    elif defect == "verify-succeeded":
        h.rows = h.barrier_rows(verify_state="SUCCEEDED", verify_error=None)
    elif defect == "boot":
        h.boot_path.write_text("boot-after\n", encoding="utf-8")
    elif defect == "holder-dead":
        h.holder_alive(False)
    else:
        h.rows = h.barrier_rows() + h.barrier_rows(workflow="workflow-second")
    h.calls.clear()
    with pytest.raises(p017.ProbeError, match="refused"):
        p017.fire_reboot(argparse.Namespace(run_id=RUN_ID))
    state = json.loads(h.state.read_text())
    assert state["reboot_refusal"]["reason"] and state["reboot_cancelled_at"], state
    assert state["disarmed_at"], state
    assert not any(command == p017.reboot_command() for command, _ in h.calls), h.calls
    assert h.units[p017.holder_unit(RUN_ID) + ".service"]["ActiveState"] == (
        "inactive"
    ), h.units
    p017.reboot_status(argparse.Namespace(run_id=RUN_ID))
    assert h.records[-1]["fired"] is False, h.records
    assert h.records[-1]["refusal"] == state["reboot_refusal"], h.records


def test_reboot_cannot_fire_after_its_own_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ProbeHarness(p017, tmp_path, monkeypatch)
    parked_for_reboot(h)
    h.clock.sleep(30)
    p017.cancel_reboot(argparse.Namespace(run_id=RUN_ID))
    h.calls.clear()
    with pytest.raises(p017.ProbeError, match="cancelled"):
        p017.fire_reboot(argparse.Namespace(run_id=RUN_ID))
    assert h.calls == [], h.calls


def test_reboot_cancel_and_clear_are_idempotent_without_existing_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ProbeHarness(p017, tmp_path, monkeypatch)
    args = argparse.Namespace(run_id=RUN_ID)
    p017.reboot_status(args)
    assert h.records[-1]["armed"] is False and h.records[-1]["fired"] is False, (
        h.records
    )
    p017.cancel_reboot(args)
    p017.clear_state(args)
    assert h.records[-1]["state_cleared"] is True, h.records
