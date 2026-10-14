from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import destr020_verdicts as verdicts
from scripts.e2e.regional import run_destr020_identity_mismatch_isolation as case
from scripts.e2e.regional.regional_live_fixture import (
    RegionalFixtureAbort,
    RegionalFixtureError,
)
from tests.regional._cov95_destr_alias import AliasHarness

NO_ALIAS_AGENT = {
    "found": False,
    "before": None,
    "lifecycle_state": None,
    "transition_id": None,
    "revoked_by_case": False,
}


def _alias_agent_evidence(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "cases" / case.CASE_ID / "alias-agent-cleanup.json"
    return dict(json.loads(path.read_text(encoding="utf-8")))


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
    # The cleanup asked the CPU Pod to revoke the alias's Agent under the
    # case's own transition -- before it proved the fleet and the runtime --
    # and kept the read-back as evidence: nothing had registered one.
    run_id = case.run_identity(tmp_path, 1)
    [revoke] = [detail for name, detail in h.calls if name == "alias.agent.cleanup"]
    assert revoke == (
        "cluster-a",
        h.alias,
        verdicts.alias_cleanup_transition(run_id),
        verdicts.alias_cleanup_reason(run_id, h.alias),
    ), revoke
    names = [name for name, _ in h.calls]
    revoked_at = names.index("alias.agent.cleanup")
    assert "gpu.nodes" in names[revoked_at:], "the fleet is compared after it"
    assert "runtime.verify" in names[revoked_at:], "the runtime is verified after it"
    assert report["cleanup"]["alias_agent"] == NO_ALIAS_AGENT, report["cleanup"]
    assert _alias_agent_evidence(tmp_path) == NO_ALIAS_AGENT


@pytest.mark.parametrize("lifecycle", ["ACTIVE", "DRAINING", "REVOKED"])
def test_an_alias_agent_left_behind_is_revoked_by_the_cleanup(
    lifecycle: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = AliasHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    drained = None if lifecycle == "ACTIVE" else "earlier/drain"
    h.cleanup_agent = {"lifecycle_state": lifecycle, "transition_id": drained}

    code, report = h.execute(tmp_path)

    run_id = case.run_identity(tmp_path, 1)
    cleanup = report["cleanup"]["alias_agent"]
    assert code == 0 and report["cleanup"]["errors"] == [], report
    assert cleanup == {
        "found": True,
        "before": lifecycle,
        "lifecycle_state": "REVOKED",
        "transition_id": drained or verdicts.alias_cleanup_transition(run_id),
        "revoked_by_case": lifecycle != "REVOKED",
    }, cleanup
    assert _alias_agent_evidence(tmp_path) == cleanup
    if lifecycle != "REVOKED":
        assert h.cleanup_agent is not None, h.calls
        assert h.cleanup_agent["transition_reason"] == (
            verdicts.alias_cleanup_reason(run_id, h.alias)
        ), "audited as the case's cleanup"


def test_an_alias_agent_the_cleanup_cannot_revoke_fails_the_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = AliasHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.cleanup_agent = {"lifecycle_state": "ACTIVE", "transition_id": None}
    h.cleanup_agent_returns = True

    code, report = h.execute(tmp_path)

    assert code == 1 and report["verdict"] == "FAIL", report
    assert report["cleanup"]["errors"] == [
        "alias_agent: RegionalFixtureError: the alias Agent record is 'ACTIVE' "
        "after the cleanup, not REVOKED"
    ], report["cleanup"]
    assert _alias_agent_evidence(tmp_path)["lifecycle_state"] == "ACTIVE"
    assert {"fleet_after", "runtime_identity"} <= set(report["cleanup"]), (
        "the rest of the cleanup still runs"
    )


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
        "alias.agent.cleanup",
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
