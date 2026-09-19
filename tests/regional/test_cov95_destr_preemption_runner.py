from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import run_destr016_preempting_reboot as case
from scripts.e2e.regional.regional_live_fixture import (
    RegionalFixtureAbort,
    RegionalFixtureError,
)
from tests.regional._cov95_destr_preemption import PreemptionHarness, data

CONDITIONAL_KIND = "conditional-barrier-pre-authorization"


def test_preemption_pre_authorizes_both_writes_and_never_execs_after_the_barrier(
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
    proof = h.pre_authorization
    assert proof is not None and proof["kind"] == CONDITIONAL_KIND, h.calls
    assert proof["not_before_seconds"] == {
        "absorb": case.ABSORB_DELAY_SECONDS,
        "escalate": case.ESCALATE_DELAY_SECONDS,
    }, proof
    # The proof binds the HOLDER (armed with ``--drill-id run_id``), not the
    # reset injection's ``-r`` drill id: the probe refuses any other binding.
    assert proof["drill_id"] == h.run_id == h.armed_drill_id, (proof, h.armed_drill_id)
    assert proof["boot_id"] == "boot-before", proof
    assert proof["maintenance_window_seconds"] == 420, proof
    assert report["pre_authorization"] == proof, report
    names = h.names()
    assert (
        names.index("holder.arm-holder")
        < names.index("holder.pre-authorize")
        < names.index("injector.write-xid46")
    ), names
    # kubelet is down from the quiesce until the reboot brings the node back:
    # nothing may be exec'd into the node between the first barrier read and
    # the node's return.
    assert h.host_execs_between("barrier.read", "node.ready") == [], names
    assert names.index("holder.disarm-holder") < names.index("incident.idle"), names
    assert h.rebooted is True, h.calls
    assert sorted(h.fired) == ["absorb", "escalate"], h.fired
    assert report["barrier_observed_at"] is not None, report


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
    assert h.rebooted is False, h.calls
    if defect in {"clients", "kmsg", "quiesce", "inventory"}:
        assert h.pre_authorization is None, (
            "the holder is not pre-authorized before the premise holds"
        )
        assert h.injected is False, h.calls
    else:
        assert h.host_execs_between("barrier.read", "node.ready") == [], h.names()


@pytest.mark.parametrize(
    "phase",
    [
        "holder.arm-holder",
        "holder.pre-authorize",
        "injector.write-xid46",
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
    if phase == "node.ready":
        # The node never answered again and its boot id did not change: the
        # holder cannot be disarmed through the probe nor assumed gone.
        assert "holder.disarm-holder" not in names, names
        assert any(
            "never answered again" in error for error in report["cleanup"]["errors"]
        ), report
    elif "holder.arm-holder" in names:
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
    assert h.pre_authorization is None, h.calls


def test_preemption_abort_propagates_after_disarming_and_probe_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = PreemptionHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.failures["holder.pre-authorize"] = RegionalFixtureAbort(2)
    with pytest.raises(RegionalFixtureAbort):
        h.execute(tmp_path)
    names = [name for name, _ in h.calls]
    assert "holder.disarm-holder" in names and "injector.cleanup" in names, names
    assert "injector.write-xid46" not in names, names


def test_preemption_plan_identity_drift_never_arms_a_holder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = PreemptionHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.node["uid"] = "changed"
    with pytest.raises(RegionalFixtureError, match="plan identity drifted"):
        h.execute(tmp_path)
    assert not h.injected, h.calls


def test_preemption_ordering_check_fails_when_the_host_fired_before_the_barrier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = PreemptionHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.host_fired_early = True
    code, report = h.execute(tmp_path)
    assert code == 1 and report["verdict"] == "FAIL", report
    early = [
        error
        for error in report["errors"]
        if "not after the runner observed the barrier" in error
    ]
    assert len(early) == 2, report["errors"]
    assert any(error.startswith("absorb injection") for error in early), early
    assert any(error.startswith("escalation injection") for error in early), early


def test_preemption_cleanup_never_execs_into_a_node_that_does_not_return(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = PreemptionHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.node_unreachable = True
    code, report = h.execute(tmp_path)
    assert code == 1 and "did not return Ready" in report["error"], report
    names = h.names()
    assert "holder.disarm-holder" not in names, names
    assert h.host_execs_between("barrier.read", "holder.cleanup") == [], names
    assert any(
        "never answered again" in error for error in report["cleanup"]["errors"]
    ), report["cleanup"]


def test_preemption_cleanup_assumes_the_holder_gone_after_a_boot_it_could_not_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = PreemptionHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.node_unreachable = True
    h.unreachable_boot_id = "boot-after"
    code, report = h.execute(tmp_path)
    assert code == 1, report
    disarm = report["cleanup"]["holder_disarm"]
    assert disarm["disarmed"] == "assumed" and "boot id changed" in disarm["reason"], (
        disarm
    )
    assert "holder.disarm-holder" not in h.names(), h.names()
