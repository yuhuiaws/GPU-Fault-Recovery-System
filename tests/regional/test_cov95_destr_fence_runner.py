from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from scripts.e2e.regional import run_destr017_out_of_band_reboot_fence as case
from scripts.e2e.regional.regional_live_fixture import (
    RegionalFixtureAbort,
    RegionalFixtureError,
)
from tests.regional._cov95_destr_fence import FenceHarness, data


@pytest.mark.parametrize("recovered", [False, True])
def test_reboot_fence_keeps_original_request_and_authorizes_only_waiting_generation(
    recovered: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = FenceHarness(tmp_path, monkeypatch)
    preflight = h.plan(tmp_path)
    assert preflight["errors"] == [], preflight
    h.recovered = recovered
    h.calls.clear()
    code, report = h.execute(tmp_path)
    assert code == 0 and report["verdict"] == "PASS", report
    assert report["errors"] == report["cleanup"]["errors"] == [], report
    assert report["workflow_request_id"] == data.REQUEST, report
    assert report["reconcile_application"] == "NOT_APPLIED", report
    proof = report["reboot"]["authorization"]
    assert proof["run_id"] == h.run_id and proof["drill_id"] == h.run_id, proof
    assert (
        proof["boot_id"] == data.BOOT_BEFORE
        and proof["agent_generation"] == data.GENERATION
    ), proof
    assert proof["command_ids"] == {
        op: f"key-{op}/{data.NODE}/agent-{data.GENERATION}"
        for op in ("QUIESCE_GPU_SERVICES", "VERIFY_NO_GPU_CLIENTS")
    }, proof
    names = [name for name, _ in h.calls]
    order = [
        "fence.arm-holder",
        "host.write-xid",
        "waiting.read",
        "fence.arm-reboot",
        "node.ready",
        "aftermath",
        "reconcile.plan",
        "fence.cancel-reboot",
        "fence.disarm-holder",
        "restore.create",
        "fence.clear-state",
    ]
    assert [names.index(name) for name in order] == sorted(
        names.index(name) for name in order
    ), names
    assert [args for name, args in h.calls if name == "aftermath"] == [
        (data.REQUEST,)
    ], h.calls


@pytest.mark.parametrize(
    ("defect", "message"),
    [
        ("pin", "never pinned"),
        ("verify", "never reached WAITING"),
        ("workflow", "workflow changed"),
        ("drill", "exact drill/workflow"),
        ("index", "device index"),
        ("window", "window"),
    ],
)
def test_reboot_is_not_armed_without_the_exact_live_barrier(
    defect: str, message: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = FenceHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    if defect == "pin":
        h.pin_missing = True
    elif defect == "verify":
        h.verify_missing = True
    elif defect == "workflow":
        h.authorization_drift = True
    elif defect == "drill":
        h.bad_drill = True
    elif defect == "index":
        h.baseline["gpu_inventory"][0].pop("index")
    else:
        h.bounds["agent_maintenance_window_seconds"] = 30
    code, report = h.execute(tmp_path)
    assert code == 1 and message in report["error"], report
    assert h.reboot_armed is False, h.calls
    assert any(name == "host.cleanup" for name, _ in h.calls), h.calls


def test_fence_waits_through_transient_missing_pin_and_old_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = FenceHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.pin_delays = 1
    h.verify_delays = 2
    h.agent_delays = 1
    h.terminal_delays = 4
    code, report = h.execute(tmp_path)
    assert code == 0, report
    assert h.clock.elapsed >= 20, h.clock.elapsed
    assert h.waiting_reads >= 5, h.waiting_reads


@pytest.mark.parametrize(
    "phase",
    [
        "fence.arm-holder",
        "host.write-xid",
        "fence.arm-reboot",
        "node.ready",
        "aftermath",
        "escalations",
        "provider.events",
        "reconcile.plan",
        "fence.cancel-reboot",
        "fence.disarm-holder",
        "agent.active",
        "incident.idle",
        "restore.create",
        "restore.wait",
        "host.restore-quiesce",
        "fence.clear-state",
        "runtime.verify",
    ],
)
def test_fence_failure_is_retained_and_remaining_cleanup_is_attempted(
    phase: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = FenceHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.failures[phase] = RuntimeError(f"fake failure at {phase}")
    code, report = h.execute(tmp_path)
    assert code == 1 and phase in str(report), report
    names = [name for name, _ in h.calls]
    assert "host.cleanup" in names and "fence.cleanup" in names, names
    if phase in {"fence.cancel-reboot", "fence.disarm-holder"}:
        assert "fence.clear-state" not in names, names


def test_fence_generation_timeout_cancels_timers_without_fabricating_new_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = FenceHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.agent_delays = 10000
    code, report = h.execute(tmp_path)
    assert code == 1 and "did not re-register a new generation" in report["error"], (
        report
    )
    names = [name for name, _ in h.calls]
    assert "aftermath" not in names and "fence.cancel-reboot" in names, names


@pytest.mark.parametrize("phase", ["before", "host.snapshot", "fence.arm-holder"])
def test_fence_rechecks_maintenance_window_before_arming_or_injection(
    phase: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = FenceHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    if phase == "before":
        h.clock.sleep(61)
    else:
        h.advance_at[phase] = 61
    code, report = h.execute(tmp_path, seconds=60)
    assert code == 1 and "maintenance window ended" in report["error"], report
    assert h.injected is False, h.calls


def test_fence_abort_after_timer_ack_still_cancels_owned_timers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = FenceHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.failures["fence.arm-reboot"] = RegionalFixtureAbort(2)
    with pytest.raises(RegionalFixtureAbort):
        h.execute(tmp_path)
    names = [name for name, _ in h.calls]
    assert names.index("fence.cancel-reboot") < names.index("fence.disarm-holder"), (
        names
    )
    assert "fence.cleanup" in names and "host.cleanup" in names, names


@pytest.mark.parametrize(
    "defect", ["predecessor", "tests", "residual", "armed", "no-gpu"]
)
def test_fence_preflight_fails_closed_and_always_cleans_readonly_probes(
    defect: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = FenceHarness(tmp_path, monkeypatch)
    if defect == "predecessor":
        h.predecessor_valid = False
    elif defect == "tests":
        h.tests_pass = False
    elif defect == "residual":
        h.residuals["pod"] = True
    elif defect == "armed":
        h.preflight_reboot = {"armed": True}
    else:
        h.baseline["gpu_inventory"] = []
        h.settings = replace(h.settings, pci_bdf="0000:00:00.0")
    preflight = h.plan(tmp_path)
    assert preflight["errors"], preflight
    with pytest.raises(RegionalFixtureError, match="preflight failed"):
        h.execute(tmp_path)
    names = [name for name, _ in h.calls]
    assert "host.cleanup" in names and "fence.cleanup" in names, names
    assert "fence.arm-holder" not in names, names


def test_fence_plan_drift_is_refused_before_arming(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = FenceHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.node["uid"] = "new-uid"
    with pytest.raises(RegionalFixtureError, match="plan identity drifted"):
        h.execute(tmp_path)
    assert not h.injected, h.calls


@pytest.mark.parametrize(
    "variation",
    [
        "provider",
        "restore-failed",
        "new-quiesce",
        "hardware-escalation",
        "reconcile-revoke",
    ],
)
def test_fence_bad_aftermath_cannot_become_pass(
    variation: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = FenceHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    if variation == "provider":
        h.events = [{"event_name": "BatchRebootClusterNodes"}]
    elif variation == "restore-failed":
        h.restore_status = "FAILED"
    elif variation == "new-quiesce":
        h.after["quiesce_states"] = [{"name": "residue"}]
    elif variation == "hardware-escalation":
        h.extra_escalations["reboot"] = {"workflow": "unexpected"}
    else:
        h.reconcile["items"] = [{"request_id": data.REQUEST}]
    code, report = h.execute(tmp_path)
    assert code == 1 and (report["errors"] or report["cleanup"]["errors"]), report


def test_fence_authorization_is_persisted_with_no_raw_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = FenceHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    code, report = h.execute(tmp_path)
    assert code == 0, report
    path = tmp_path / "cases" / case.CASE_ID / "reboot-armed.json"
    proof = json.loads(path.read_text())["authorization"]
    assert proof["waiting"] is True and proof["fencing_token"] == 5, proof
    assert proof["workflow_request_id"] == data.REQUEST, proof
    assert set(proof["command_ids"]) == {
        "QUIESCE_GPU_SERVICES",
        "VERIFY_NO_GPU_CLIENTS",
    }, proof
