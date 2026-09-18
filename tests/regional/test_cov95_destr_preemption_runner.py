from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional.regional_live_fixture import (
    RegionalFixtureAbort,
    RegionalFixtureError,
)
from tests.regional._cov95_destr_preemption import PreemptionHarness, data


def test_preemption_authorizes_absorb_and_escalation_from_same_running_barrier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = PreemptionHarness(tmp_path, monkeypatch)
    preflight = h.plan(tmp_path)
    assert preflight["errors"] == [], preflight
    h.barrier_delays = h.absorb_delays = h.successor_delays = 1
    h.supersede_delays = h.handoff_delays = h.terminal_delays = 1
    code, report = h.execute(tmp_path)
    assert code == 0 and report["errors"] == report["cleanup"]["errors"] == [], report
    assert report["predecessor_workflow_id"] == data.RESET_ID, report
    assert report["successor_workflow_id"] == data.REBOOT_ID, report
    assert h.authorized == ["absorb", "escalate"], h.calls
    proofs = [detail["proof"] for name, detail in h.calls if name == "authorization"]
    assert all(
        proof["workflow_request_id"] == data.RESET_ID and proof["agent_generation"] == 4
        for proof in proofs
    ), proofs
    names = [name for name, _ in h.calls]
    assert names.index("holder.arm-holder") < names.index("injector.write-xid46"), names
    assert names.index("holder.disarm-holder") < names.index("incident.idle"), names
    assert h.rebooted, h.calls


@pytest.mark.parametrize(
    "defect", ["clients", "kmsg", "quiesce", "inventory", "barrier", "absorb"]
)
def test_preemption_refuses_bad_premise_before_escalation(
    defect: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = PreemptionHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    if defect == "clients":
        h.baseline["compute_clients"] = ["foreign"]
    elif defect == "kmsg":
        h.baseline["kmsg_writable"] = False
    elif defect == "quiesce":
        h.baseline["quiesce_states"] = ["existing"]
    elif defect == "inventory":
        h.baseline["gpu_inventory"] = []
    elif defect == "barrier":
        h.bad_barrier = True
    else:
        h.bad_absorb = True
    code, report = h.execute(tmp_path)
    assert code == 1 and report["error"], report
    assert "escalate" not in h.authorized, h.calls
    assert not h.rebooted, h.calls


@pytest.mark.parametrize(
    "phase",
    [
        "holder.arm-holder",
        "injector.write-xid46",
        "holder.authorize-injection",
        "predecessor.read",
        "handoff.read",
        "commands.read",
        "node.ready",
        "provider.events",
        "holder.disarm-holder",
        "incident.idle",
        "holder.cleanup",
        "injector.cleanup",
        "runtime.verify",
    ],
)
def test_preemption_phase_failure_retains_failure_and_attempts_cleanup(
    phase: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = PreemptionHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.failures[phase] = RuntimeError(f"fake failure at {phase}")
    code, report = h.execute(tmp_path)
    assert code == 1 and phase in str(report), report
    names = [name for name, _ in h.calls]
    assert "holder.cleanup" in names and "injector.cleanup" in names, names
    if "holder.arm-holder" in names:
        assert "holder.disarm-holder" in names, names


@pytest.mark.parametrize("stage", ["barrier", "successor", "supersede", "handoff"])
def test_preemption_waiters_fail_when_the_requested_state_never_arrives(
    stage: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = PreemptionHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    setattr(h, f"{stage}_delays", 10000)
    code, report = h.execute(tmp_path)
    assert code == 1 and report["error"], report
    assert any(name == "holder.disarm-holder" for name, _ in h.calls), h.calls


@pytest.mark.parametrize("succeeded", [True, False])
def test_preemption_restores_isolation_through_bound_incident(
    succeeded: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = PreemptionHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.restored = "SUCCEEDED" if succeeded else "FAILED"
    original = h.regional.provider_events

    def events(*args: Any) -> list[dict[str, Any]]:
        result = original(*args)
        h.node["unschedulable"] = True
        return result

    monkeypatch.setattr(h.regional, "provider_events", events)
    code, report = h.execute(tmp_path)
    assert code == (0 if succeeded else 1), report
    assert any(
        name == "restore.create" and detail["incident_id"] == data.INCIDENT
        for name, detail in h.calls
    ), h.calls


def test_preemption_cannot_start_without_a_full_reboot_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = PreemptionHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    code, report = h.execute(tmp_path, seconds=60)
    assert code == 1 and "needs at least" in report["error"], report
    assert not h.injected, h.calls


def test_preemption_abort_propagates_after_disarming_and_probe_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = PreemptionHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.failures["holder.authorize-injection"] = RegionalFixtureAbort(2)
    with pytest.raises(RegionalFixtureAbort):
        h.execute(tmp_path)
    names = [name for name, _ in h.calls]
    assert "holder.disarm-holder" in names and "injector.cleanup" in names, names


def test_preemption_plan_identity_drift_never_arms_a_holder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = PreemptionHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.node["uid"] = "changed"
    with pytest.raises(RegionalFixtureError, match="plan identity drifted"):
        h.execute(tmp_path)
    assert not h.injected, h.calls
