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
from tests.regional._cov95_destr_probes import GUARD, NOW, RUN_ID, ProbeHarness

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


def seed_injection(h: ProbeHarness) -> argparse.Namespace:
    p016.write_state(
        h.state,
        {
            "run_id": RUN_ID,
            "drill_id": "drill-owned",
            "device": "/dev/nvidia0",
            "boot_id": "boot-before",
            "injection_script_sha256": hashlib.sha256(
                h.guard_path.read_bytes()
            ).hexdigest(),
            "injections": [
                {
                    "phase": "escalate",
                    "script": GUARD,
                    "subcommand": "write-xid79",
                    "marker": "marker-owned",
                    "drill_id": "drill-owned",
                    "pci_bdf": "0000:01:00",
                    "after_seconds": 10,
                    "maintenance_window_end": (NOW + timedelta(minutes=10)).isoformat(),
                }
            ],
        },
    )
    return argparse.Namespace(
        run_id=RUN_ID, phase="escalate", authorization=json.dumps(h.proof())
    )


def test_delayed_injection_records_authorization_and_consumes_once_before_firing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ProbeHarness(p016, tmp_path, monkeypatch)
    args = seed_injection(h)
    p016.authorize_injection(args)
    state = json.loads(h.state.read_text())
    assert state["authorizations"]["escalate"]["fire_requested_at"] is None, state
    assert any("check-barrier" in command for command, _ in h.calls), h.calls
    p016.fire_injection(args)
    state = json.loads(h.state.read_text())
    assert state["authorizations"]["escalate"]["fire_requested_at"], state
    with pytest.raises(p016.ProbeError, match="consumed"):
        p016.fire_injection(args)
    assert sum("write-xid79" in command for command, _ in h.calls) == 1, h.calls


@pytest.mark.parametrize("defect", ["holder", "phase", "repeat", "digest"])
def test_delayed_injection_refuses_changed_or_replayed_authorization(
    defect: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ProbeHarness(p016, tmp_path, monkeypatch)
    args = seed_injection(h)
    if defect == "holder":
        p016.update_state(h.state, {"arm_race_lost": True})
    elif defect == "phase":
        args.phase = "absorb"
    else:
        p016.authorize_injection(args)
    if defect == "digest":
        h.guard_path.write_text("# changed fake guard\n", encoding="utf-8")
        with pytest.raises(p016.ProbeError, match="script changed"):
            p016.fire_injection(args)
    else:
        with pytest.raises(p016.ProbeError):
            p016.authorize_injection(args)
    assert not any("write-xid79" in command for command, _ in h.calls), h.calls


def seed_reboot(h: ProbeHarness) -> argparse.Namespace:
    p017.write_state(
        h.state,
        {
            "run_id": RUN_ID,
            "drill_id": "drill-owned",
            "device": "/dev/nvidia0",
            "maintenance_window_end": (NOW + timedelta(minutes=10)).isoformat(),
        },
    )
    return argparse.Namespace(
        run_id=RUN_ID,
        authorization=json.dumps(h.proof()),
        barrier_script=GUARD,
        delay_seconds=30,
    )


def test_reboot_timer_requires_proof_and_preserves_one_shot_fire_and_boot_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ProbeHarness(p017, tmp_path, monkeypatch)
    args = seed_reboot(h)
    p017.arm_reboot(args)
    state = json.loads(h.state.read_text())
    assert (
        state["boot_id_before_reboot"] == "boot-before" and state["reboot_armed_at"]
    ), state
    assert any("fire-reboot" in command for command, _ in h.calls), h.calls
    p017.fire_reboot(args)
    assert json.loads(h.state.read_text())["fire_requested_at"], h.state
    with pytest.raises(p017.ProbeError, match="consumed"):
        p017.fire_reboot(args)
    h.boot_path.write_text("boot-after\n", encoding="utf-8")
    p017.reboot_status(args)
    assert h.records[-1]["fired"] is True and h.records[-1]["boot_changes"] == 1, (
        h.records
    )
    p017.cancel_reboot(args)
    assert h.records[-1]["already_fired"] is True, h.records
    p017.disarm_holder(args)
    p017.clear_state(args)
    assert not h.state.exists() and h.records[-1]["state_cleared"] is True, h.records
    assert (
        sum(command == ["/bin/systemctl", "reboot"] for command, _ in h.calls) == 1
    ), h.calls


@pytest.mark.parametrize(
    "defect",
    ["cancelled", "wrong-holder", "bad-path", "window", "pinned-window", "too-long"],
)
def test_reboot_cannot_be_armed_outside_its_holder_and_deadlines(
    defect: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = ProbeHarness(p017, tmp_path, monkeypatch)
    args = seed_reboot(h)
    if defect == "cancelled":
        p017.update_state(h.state, {"reboot_cancelled_at": NOW.isoformat()})
    elif defect == "wrong-holder":
        p017.update_state(h.state, {"device": "/dev/nvidia1"})
    elif defect == "bad-path":
        args.barrier_script = "/tmp/untrusted.py"
    elif defect == "window":
        p017.update_state(
            h.state,
            {"maintenance_window_end": (NOW + timedelta(seconds=30)).isoformat()},
        )
    elif defect == "pinned-window":
        args.authorization = json.dumps(
            {
                **h.proof(),
                "window_expires_at": (NOW + timedelta(seconds=30)).isoformat(),
            }
        )
    else:
        args.delay_seconds = 61
    with pytest.raises(p017.ProbeError):
        p017.arm_reboot(args)
    assert not any("fire-reboot" in command for command, _ in h.calls), h.calls


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
