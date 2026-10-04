"""GF-REGIONAL-DESTR-016 runner: a preflight that fails inside ``execute_case``
itself, the barrier wait that ends on a terminal workflow, the absorb and
observation waits that run out of budget, a reset workflow that changes before
the escalation, a control-plane blast radius that moved, and residual probe
resources found only at cleanup."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import run_destr016_preempting_reboot as case
from scripts.e2e.regional.regional_live_fixture import RegionalFixtureError
from tests.regional._cov95_destr_preemption import PreemptionHarness


def test_execute_refuses_a_preflight_that_fails_on_its_own_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = PreemptionHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.preflight["tests"] = {"passed": False}
    with pytest.raises(RegionalFixtureError, match="preflight failed"):
        h.execute(tmp_path)
    assert "holder.arm-holder" not in h.names(), h.names()


def test_a_barrier_wait_ends_when_the_reset_workflow_goes_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = PreemptionHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    barrier = h.barrier

    def failed_barrier(absorbed: bool = False) -> dict[str, Any]:
        state = barrier(absorbed)
        state["workflow"]["status"] = "FAILED"
        return state

    monkeypatch.setattr(h, "barrier", failed_barrier)
    code, report = h.execute(tmp_path)
    assert code == 1, report
    assert "barrier.read" in h.names(), h.names()
    assert h.names().count("barrier.read") == 1, "a terminal workflow ends the wait"
    recorded = json.loads(
        (tmp_path / "cases" / case.CASE_ID / "barrier-state.json").read_text()
    )
    assert recorded["workflow"]["status"] == "FAILED"
    assert "holder.disarm-holder" in h.names(), h.names()


def test_the_absorb_wait_stops_at_its_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = PreemptionHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.absorb_delays = 10**6
    h.advance_at["absorb.read"] = case.ABSORB_BUDGET_SECONDS / 2 + 1
    code, report = h.execute(tmp_path)
    assert code == 1, report
    assert h.names().count("absorb.read") == 2, "the wait stops at its budget"
    recorded = json.loads(
        (tmp_path / "cases" / case.CASE_ID / "absorb-state.json").read_text()
    )
    assert recorded["workflow"]["request_id"], "the last read is what is recorded"
    assert "holder.disarm-holder" in h.names(), h.names()


def test_the_observation_stops_at_its_budget_when_nothing_terminalizes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = PreemptionHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.terminal_delays = 10**6
    store_snapshot = h.regional.store_snapshot

    def slow_terminal_reads(**kwargs: Any) -> dict[str, Any]:
        if h.rebooted and str(kwargs.get("marker") or "").endswith("-e"):
            h.clock.sleep(case.OBSERVATION_BUDGET_SECONDS / 2 + 1)
        return store_snapshot(**kwargs)

    monkeypatch.setattr(h.regional, "store_snapshot", slow_terminal_reads)
    code, report = h.execute(tmp_path, seconds=40000)
    assert code == 1, report
    assert h.clock.elapsed >= case.OBSERVATION_BUDGET_SECONDS, h.clock.elapsed
    recorded = json.loads(
        (tmp_path / "cases" / case.CASE_ID / "workflow-state.json").read_text()
    )
    assert recorded["workflow"]["status"] == "RUNNING", (
        "the unsettled state is what the budget ran out on"
    )
    assert "holder.disarm-holder" in h.names(), h.names()


def test_a_reset_workflow_that_changes_before_escalation_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = PreemptionHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    barrier = h.barrier

    def replaced_workflow(absorbed: bool = False) -> dict[str, Any]:
        state = barrier(absorbed)
        # The absorb itself lands on the parked reset; the next plain read of
        # the reset, taken right before the escalation, names another workflow.
        if not absorbed and "absorb" in h.fired:
            state["workflow"]["request_id"] = "workflow-replacement"
            for command in state.get("commands") or []:
                if "workflow_request_id" in command:
                    command["workflow_request_id"] = "workflow-replacement"
        return state

    monkeypatch.setattr(h, "barrier", replaced_workflow)
    code, report = h.execute(tmp_path)
    assert code == 1, report
    assert "reset workflow changed before the escalation was due" in report["error"]
    assert "injector.write-xid79" not in h.names(), h.names()
    assert "holder.disarm-holder" in h.names(), h.names()


def test_a_moved_control_plane_blast_radius_fails_the_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = PreemptionHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.cpu_after = {"cpu": ["api-a", "api-b"]}
    code, report = h.execute(tmp_path)
    assert code == 1, report
    assert "control-plane EKS state differs from baseline" in report["errors"], report
    after = json.loads(
        (tmp_path / "cases" / case.CASE_ID / "cpu-blast-after.json").read_text()
    )
    assert after == {"cpu": ["api-a", "api-b"]}


def test_cleanup_refuses_residual_probe_resources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = PreemptionHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    original = case.HostProbeFixture

    def probe_with_residue(settings: Any) -> Any:
        probe = original(settings)

        def cleanup() -> dict[str, bool]:
            h.call(f"{probe.label}.cleanup")
            return {"pod": h.injected}

        probe.cleanup = cleanup
        return probe

    monkeypatch.setattr(case, "HostProbeFixture", probe_with_residue)
    code, report = h.execute(tmp_path)
    assert code == 1, report
    residual = [
        item for item in report["cleanup"]["errors"] if "residual resources" in item
    ]
    assert len(residual) == 2, report["cleanup"]
    assert all("'pod': True" in item for item in residual), residual


def test_a_parked_reset_with_a_wrong_decision_is_refused_before_absorbing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = PreemptionHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    barrier = h.barrier

    def foreign_decision(absorbed: bool = False) -> dict[str, Any]:
        state = barrier(absorbed)
        state["decision"] = {"official_action": "RESTART_BM"}
        return state

    monkeypatch.setattr(h, "barrier", foreign_decision)
    code, report = h.execute(tmp_path)
    assert code == 1, report
    assert "policy did not resolve the first XID 46 to RESET_GPU" in report["error"]
    assert "RESTART_BM" in report["error"], report
    names = h.names()
    assert names.count("injector.write-xid46") == 1, "nothing is absorbed into it"
    assert "injector.write-xid79" not in names, "nothing escalates a refused reset"
    assert "holder.disarm-holder" in names, names
