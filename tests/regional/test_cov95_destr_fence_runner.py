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

CONDITIONAL_KIND = "conditional-barrier-pre-authorization"


@pytest.mark.parametrize("recovered", [False, True])
def test_reboot_fence_pre_authorizes_before_injection_and_never_execs_after_barrier(
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
    assert proof["kind"] == CONDITIONAL_KIND and proof["conditional"] is True, proof
    assert proof["run_id"] == h.run_id and proof["drill_id"] == h.run_id, proof
    assert proof["boot_id"] == data.BOOT_BEFORE and proof["node_id"] == data.NODE, proof
    assert proof["not_before_seconds"] == {"reboot": 60}, proof
    assert "command_ids" not in proof, (
        "a pre-authorization cannot name barrier rows that do not exist yet"
    )
    barrier = report["reboot"]["barrier"]
    assert barrier["waiting"] is True and barrier["fencing_token"] == 5, barrier
    assert barrier["workflow_request_id"] == data.REQUEST, barrier
    assert barrier["agent_generation"] == data.GENERATION, barrier
    assert barrier["command_ids"] == {
        op: f"key-{op}/{data.NODE}/agent-{data.GENERATION}"
        for op in ("QUIESCE_GPU_SERVICES", "VERIFY_NO_GPU_CLIENTS")
    }, barrier
    names = h.names()
    order = [
        "fence.arm-holder",
        "fence.pre-authorize-reboot",
        "host.write-xid",
        "waiting.read",
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
    # kubelet is down from the quiesce until the node is back: nothing may be
    # exec'd into the node between the barrier observation and its return.
    assert h.host_execs_between("waiting.read", "node.ready") == [], names
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
    ],
)
def test_barrier_defects_fail_closed_without_an_exec_after_the_barrier(
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
    else:
        h.bad_drill = True
    code, report = h.execute(tmp_path)
    assert code == 1 and message in report["error"], report
    assert h.reboot_armed is True, "the reboot is pre-authorized before the injection"
    assert h.host_execs_between("waiting.read", "node.ready") == [], h.names()
    names = h.names()
    assert (
        names.index("node.ready")
        < names.index("fence.cancel-reboot")
        < names.index("fence.disarm-holder")
    ), names
    assert any(name == "host.cleanup" for name, _ in h.calls), h.calls


@pytest.mark.parametrize(
    ("defect", "message"), [("index", "device index"), ("window", "window")]
)
def test_premise_defects_refuse_before_the_reboot_is_pre_authorized(
    defect: str, message: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = FenceHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    if defect == "index":
        h.baseline["gpu_inventory"][0].pop("index")
    else:
        h.bounds["agent_maintenance_window_seconds"] = 30
    code, report = h.execute(tmp_path)
    assert code == 1 and message in report["error"], report
    assert h.reboot_armed is False, h.calls
    assert "fence.pre-authorize-reboot" not in h.names(), h.names()
    assert h.injected is False, h.calls
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
        "fence.pre-authorize-reboot",
        "host.write-xid",
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
    if phase == "node.ready":
        # A node that never answers again gets no exec at all; the holder
        # cannot be assumed gone while its boot id is unchanged.
        assert "fence.cancel-reboot" not in names, names
        assert "fence.disarm-holder" not in names, names
        assert any(
            "never answered again" in error for error in report["cleanup"]["errors"]
        ), report


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


def test_fence_abort_after_pre_authorization_ack_still_cancels_owned_timers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = FenceHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.failures["fence.pre-authorize-reboot"] = RegionalFixtureAbort(2)
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


def test_fence_evidence_persists_the_pre_authorization_and_the_store_barrier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = FenceHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    code, report = h.execute(tmp_path)
    assert code == 0, report
    case_dir = tmp_path / "cases" / case.CASE_ID
    armed = json.loads((case_dir / "reboot-armed.json").read_text())
    assert armed["authorization"]["kind"] == CONDITIONAL_KIND, armed
    assert armed["authorization"] == h.pre_authorization, armed
    assert "lease_token" not in json.dumps(armed), armed
    waiting = json.loads((case_dir / "waiting-verify.json").read_text())
    barrier = waiting["barrier"]
    assert barrier["waiting"] is True and barrier["fencing_token"] == 5, barrier
    assert barrier["workflow_request_id"] == data.REQUEST, barrier
    assert set(barrier["command_ids"]) == {
        "QUIESCE_GPU_SERVICES",
        "VERIFY_NO_GPU_CLIENTS",
    }, barrier
    window = json.loads((case_dir / "reboot-window.json").read_text())
    assert window["barrier"] == barrier and window["window_remaining_seconds"] > 60, (
        window
    )


def test_fence_ordering_check_fails_when_the_host_fired_before_the_barrier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = FenceHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.host_fired_early = True
    code, report = h.execute(tmp_path)
    assert code == 1 and report["verdict"] == "FAIL", report
    assert any(
        "not after the runner observed the barrier" in error
        for error in report["errors"]
    ), report["errors"]


def test_fence_ordering_check_fails_when_the_host_matched_another_workflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = FenceHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.host_condition_workflow = "workflow-someone-else"
    code, report = h.execute(tmp_path)
    assert code == 1, report
    assert any(
        "matched workflow_request_id 'workflow-someone-else'" in error
        for error in report["errors"]
    ), report["errors"]


def test_fence_cleanup_never_execs_into_a_node_that_does_not_return(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = FenceHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.node_unreachable = True
    code, report = h.execute(tmp_path)
    assert code == 1 and "did not return Ready" in report["error"], report
    names = h.names()
    for forbidden in (
        "fence.cancel-reboot",
        "fence.disarm-holder",
        "fence.clear-state",
        "host.restore-quiesce",
    ):
        assert forbidden not in names, names
    assert h.host_execs_between("waiting.read", "fence.cleanup") == [], names
    errors = report["cleanup"]["errors"]
    assert any("cancel_reboot: the node never answered" in e for e in errors), errors
    assert any("host_final: the node never answered" in e for e in errors), errors


def test_fence_cleanup_assumes_the_timer_gone_when_the_node_rebooted_but_hid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = FenceHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.node_unreachable = True
    h.unreachable_boot_id = data.BOOT_AFTER
    code, report = h.execute(tmp_path)
    assert code == 1, report
    cleanup = report["cleanup"]
    assert cleanup["node_ready"]["assume_disarmed"] is True, cleanup
    assert "boot id changed" in cleanup["cancel_reboot"]["skipped"], cleanup
    assert "boot id changed" in cleanup["disarm_holder"]["skipped"], cleanup
    names = h.names()
    assert "fence.cancel-reboot" not in names and "fence.disarm-holder" not in names, (
        names
    )
    # Quiescence still has to be proven on the host; it cannot be assumed.
    assert any("host_final" in error for error in cleanup["errors"]), cleanup
