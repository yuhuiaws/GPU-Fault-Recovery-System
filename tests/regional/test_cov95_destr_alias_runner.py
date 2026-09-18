from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional.regional_live_fixture import (
    RegionalFixtureAbort,
    RegionalFixtureError,
)
from tests.regional._cov95_destr_alias import AliasHarness


def test_alias_case_proves_absence_and_bounded_escalation_without_node_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = AliasHarness(tmp_path, monkeypatch)
    preflight = h.plan(tmp_path)
    assert preflight["errors"] == [], preflight
    h.chain_pending = 1
    code, report = h.execute(tmp_path)
    assert code == 0 and report["errors"] == report["cleanup"]["errors"] == [], report
    assert h.posted["node_id"] == report["alias"] == h.alias, report
    assert h.posted["cluster_id"] == "cluster-a" and h.posted["product"] == "A100", (
        h.posted
    )
    assert h.posted["affected_workload_ids"] == [], h.posted
    assert h.clock.elapsed >= 35, h.clock.elapsed
    assert report["leftover_records"], report
    assert all(
        detail["args"][0] == "get" for name, detail in h.calls if name == "node.present"
    ), h.calls


@pytest.mark.parametrize("defect", ["node", "agent", "event", "predecessor", "tests"])
def test_alias_preflight_rejects_prior_identity_or_failed_proofs(
    defect: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = AliasHarness(tmp_path, monkeypatch)
    if defect == "node":
        h.alias_present = True
    elif defect == "agent":
        h.alias_agent = {"lifecycle_state": "ACTIVE"}
    elif defect == "event":
        h.alias_event = {"event_id": "earlier"}
    elif defect == "predecessor":
        h.predecessor_valid = False
    else:
        h.tests_pass = False
    assert h.plan(tmp_path)["errors"], h.calls
    with pytest.raises(RegionalFixtureError, match="preflight failed"):
        h.execute(tmp_path)
    assert not h.posted, h.calls


@pytest.mark.parametrize("busy", ["queue", "command"])
def test_alias_injection_rechecks_quiet_control_plane(
    busy: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = AliasHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    if busy == "queue":
        h.quiet_queue = {"depth": 1}
    else:
        h.quiet_commands = {"open_by_cluster": {"cluster-a": 1}}
    code, report = h.execute(tmp_path)
    assert code == 1 and report["error"], report
    assert not h.posted, h.calls
    assert "runtime_identity" in report["cleanup"], report


@pytest.mark.parametrize(
    "phase",
    [
        "metadata",
        "event.post",
        "workflow.wait",
        "commands.read",
        "chain.read",
        "provider.events",
        "runtime.verify",
    ],
)
def test_alias_phase_failure_is_recorded_and_cleanup_still_checks_fleet(
    phase: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = AliasHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.failures[phase] = RuntimeError(f"fake failure at {phase}")
    code, report = h.execute(tmp_path)
    assert code == 1 and phase in str(report), report
    assert "fleet_after" in report["cleanup"], report


@pytest.mark.parametrize(
    "defect", ["receipt", "decision", "command", "chain", "provider"]
)
def test_alias_bad_verdict_evidence_is_never_promoted_to_pass(
    defect: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = AliasHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    if defect == "receipt":
        h.receipt["receipt"]["status"] = 500
    elif defect == "decision":
        h.state["decision"]["action"] = "RESET_GPU"
    elif defect == "command":
        h.remote_commands[0]["status"] = "SUCCEEDED"
    elif defect == "chain":
        h.chain["second_order_workflow"] = {"request_id": "unexpected"}
    else:
        h.events = [{"event_name": "BatchRebootClusterNodes"}]
    code, report = h.execute(tmp_path)
    assert code == 1 and (report.get("errors") or report.get("error")), report
    if defect == "chain":
        assert h.clock.elapsed == 0, h.clock.elapsed
        assert any("second-order" in error for error in report["errors"]), report


def test_alias_cannot_pass_when_real_node_scheduling_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = AliasHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    original = h.regional.post_xid_event

    def post(payload: dict[str, Any]) -> dict[str, Any]:
        result = original(payload)
        h.node["unschedulable"] = True
        return result

    monkeypatch.setattr(h.regional, "post_xid_event", post)
    code, report = h.execute(tmp_path)
    assert code == 1 and report["cleanup"]["errors"], report


def test_alias_expired_window_and_abort_both_preserve_no_mutation_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = AliasHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    code, report = h.execute(tmp_path, seconds=-1)
    assert code == 1 and "window ended" in report["error"], report
    assert not h.posted, h.calls
    h.failures["event.post"] = RegionalFixtureAbort(2)
    with pytest.raises(RegionalFixtureAbort):
        h.execute(tmp_path)
    assert any(name == "runtime.verify" for name, _ in h.calls), h.calls


def test_alias_plan_reference_uid_drift_refuses_before_post(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = AliasHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.node["uid"] = "replacement"
    with pytest.raises(RegionalFixtureError, match="plan drifted"):
        h.execute(tmp_path)
    assert not h.posted, h.calls
