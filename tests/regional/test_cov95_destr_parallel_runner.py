from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import run_destr015_parallel_branch_join as case
from scripts.e2e.regional.regional_live_fixture import (
    RegionalFixtureAbort,
    RegionalFixtureError,
)
from tests.regional._cov95_destr_branches import NODES, BranchHarness


def test_two_node_join_executes_through_real_verdicts_and_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = BranchHarness(tmp_path, monkeypatch)
    preflight = h.plan(tmp_path)
    assert preflight["errors"] == [], preflight
    h.pending_reads = 1
    code, report = h.execute(tmp_path)
    assert code == 0 and report["verdict"] == "PASS", report
    assert report["errors"] == report["cleanup"]["errors"] == [], report
    assert set(report["injected_at"]) == set(NODES), report
    assert report["restart_budget"]["restart_count"] == 1, report
    assert report["provider_events"] == {"count": 0, "provisional": True}, report
    assert set(report["components"]) == {
        "preflight_identity",
        "workflow",
        "incident",
        "hosts",
    }, report
    names = [name for name, _ in h.calls]
    assert names.index("workload.submit") < names.index("probe.write-xid46"), names
    assert names.index("workload.authorize_restart") < names.index(
        "workload.restarted"
    ), names
    assert names.index("workload.restarted") < names.index("workload.delete"), names
    assert sum(name == "probe.write-xid46" for name in names) == 2, names
    assert sum(name == "probe.cleanup" for name in names) == 2, names
    path = tmp_path / "cases" / case.CASE_ID / "step-timeline.json"
    assert json.loads(path.read_text())["transitions"], path


@pytest.mark.parametrize("fault", ["split", "late"])
def test_invalid_merge_stops_before_terminal_observation_or_restart(
    fault: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = BranchHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.split = fault == "split"
    h.skew = 20 if fault == "late" else 0
    code, report = h.execute(tmp_path)
    assert code == 1 and "stopping before observation" in report["error"], report
    assert report["errors"], report
    names = [name for name, _ in h.calls]
    assert "workload.restarted" not in names, names
    assert "workload.delete" in names, names


@pytest.mark.parametrize(
    "phase",
    [
        "prewarm.create",
        "workload.submit",
        "workload.running",
        "probe.create",
        "probe.write-xid46",
        "workflow.wait",
        "workload.authorize_restart",
        "workload.restarted",
        "provider.events",
        "workload.delete",
        "prewarm.cleanup",
        "probe.cleanup",
        "runtime.verify",
        "witness.start",
        "witness.poll",
        "witness.finish",
        "witness.close",
    ],
)
def test_parallel_runner_failure_reports_error_without_skipping_other_cleanup(
    phase: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = BranchHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.failures[phase] = RuntimeError(f"fake failure at {phase}")
    code, report = h.execute(tmp_path)
    assert code == 1 and phase in str(report), report
    names = [name for name, _ in h.calls]
    assert "prewarm.cleanup" in names, names
    assert names.count("probe.cleanup") == 2, names
    if phase == "workload.authorize_restart":
        assert "workload.restarted" not in names


@pytest.mark.parametrize(
    ("defect", "message"),
    [
        ("cache", "not cached"),
        ("gpu_inventory", "host GPU inventory"),
        ("quiesce_states", "pre-existing quiesce"),
        ("kmsg_writable", "not writable"),
    ],
)
def test_parallel_baseline_refusal_never_injects(
    defect: str, message: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = BranchHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    if defect == "cache":
        h.cached = [NODES[0]]
    else:
        h.baselines[NODES[0]][defect] = {
            "gpu_inventory": [],
            "quiesce_states": ["foreign"],
            "kmsg_writable": False,
        }[defect]
    code, report = h.execute(tmp_path)
    assert code == 1 and message in report["error"], report
    assert not h.injected, h.injected


@pytest.mark.parametrize("phase", ["before", "prewarm.create", "workload.running"])
def test_parallel_deadline_never_authorizes_the_next_mutation(
    phase: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = BranchHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    if phase == "before":
        h.clock.sleep(61)
    else:
        h.advance_at[phase] = 61
    code, report = h.execute(tmp_path, seconds=60)
    assert code == 1 and "window ended" in report["error"], report
    assert not h.injected, h.calls


@pytest.mark.parametrize(
    "cleanup_fault",
    ["leased-command", "workload-residual", "prewarm-residual", "probe-residual"],
)
def test_parallel_cleanup_requires_terminal_commands_and_zero_residuals(
    cleanup_fault: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = BranchHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    if cleanup_fault == "leased-command":
        h.cleanup_busy = True
    elif cleanup_fault == "workload-residual":
        h.workload_residual = "pytorchjob/job-owned"
    elif cleanup_fault == "prewarm-residual":
        h.prewarm_residuals["pod"] = True
    else:
        h.probe_residuals["pod"] = True
    code, report = h.execute(tmp_path)
    assert code == 1 and report["cleanup"]["errors"], report
    if cleanup_fault == "leased-command":
        assert report["cleanup"]["workload_cleanup_deferred"] is True, report
        assert report["cleanup"]["isolation_restore_deferred"] is True, report
        assert "workload.delete" not in [name for name, _ in h.calls], h.calls
    else:
        assert "residual" in str(report["cleanup"]["errors"]) or "still exists" in str(
            report["cleanup"]["errors"]
        ), report


@pytest.mark.parametrize("restored", [True, False])
def test_failed_parallel_branch_cleans_up_through_validation_workflow(
    restored: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = BranchHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.workflow["status"] = "FAILED"
    h.restore_status = "SUCCEEDED" if restored else "FAILED"
    original_wait = h.regional.wait_for_workflow

    def wait(**kwargs: Any) -> dict[str, Any]:
        state = original_wait(**kwargs)
        h.nodes[NODES[0]]["unschedulable"] = True
        return state

    monkeypatch.setattr(h.regional, "wait_for_workflow", wait)
    code, report = h.execute(tmp_path)
    assert code == 1, report
    assert "workload.restarted" not in [name for name, _ in h.calls], h.calls
    assert any(name == "restore.create" for name, _ in h.calls), h.calls
    assert bool(report["cleanup"]["errors"]) is not restored, report


def test_isolated_node_without_incident_does_not_get_guessed_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = BranchHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.incident["incident_id"] = ""
    h.failures["probe.write-xid46"] = RuntimeError("write ACK lost")
    original_execute = h.workload.delete

    def delete() -> None:
        original_execute()
        h.nodes[NODES[0]]["taints"] = [{"key": "gpu-fault.io/quarantined"}]

    monkeypatch.setattr(h.workload, "delete", delete)
    code, report = h.execute(tmp_path)
    assert code == 1 and "no incident is known" in str(report["cleanup"]["errors"]), (
        report
    )
    assert not any(name == "restore.create" for name, _ in h.calls), h.calls


def test_parallel_abort_propagates_only_after_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = BranchHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.failures["probe.write-xid46"] = RegionalFixtureAbort(2)
    with pytest.raises(RegionalFixtureAbort):
        h.execute(tmp_path)
    names = [name for name, _ in h.calls]
    assert "workload.delete" in names and names.count("probe.cleanup") == 2, names


def test_parallel_plan_drift_is_refused_before_resource_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = BranchHarness(tmp_path, monkeypatch)
    h.plan(tmp_path)
    h.nodes[NODES[0]]["uid"] = "replacement-uid"
    with pytest.raises(RegionalFixtureError, match="plan identity drifted"):
        h.execute(tmp_path)
    assert "prewarm.create" not in [name for name, _ in h.calls], h.calls


@pytest.mark.parametrize("source", ["site_file", "manifest"])
def test_parallel_missing_input_is_refused_without_cluster_reads(
    source: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = BranchHarness(tmp_path, monkeypatch)
    h.settings = replace(h.settings, **{source: tmp_path / "missing"})
    with pytest.raises(RegionalFixtureError, match="does not exist"):
        case.read_only_preflight(h.settings, tmp_path)
    assert h.calls == [], h.calls


def test_arbiter_inventory_ignores_terminating_and_unscheduled_pods(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = BranchHarness(tmp_path, monkeypatch)
    h.arbiters = [
        {"metadata": {"deletionTimestamp": "now"}, "spec": {"nodeName": NODES[0]}},
        {"metadata": {"namespace": "system", "name": "pending"}, "spec": {}},
        {
            "metadata": {"namespace": "system", "name": "executor"},
            "spec": {"nodeName": "outside"},
        },
    ]
    result = h.plan(tmp_path)
    assert result["arbiter_pods"] == {"outside": ["system/executor"]}, result
    assert result["errors"] == [], result
