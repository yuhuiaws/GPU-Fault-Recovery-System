from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from scripts.e2e.regional import blast_acceptance_base as base
from scripts.e2e.regional import run_workload_acceptance as workload
from scripts.e2e.regional.blast_acceptance_cases_1 import BlastCasesOne
from tests.regional._blast_containment_support import produce_containment
from tests.regional.test_blast_acceptance_runner import _runner
from tests.regional.test_identity_causal_review import lifecycle_harness


def test_blast001_consumes_both_real_producer_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = _runner(tmp_path, case_id="GF-REGIONAL-BLAST-001", cls=BlastCasesOne)
    runner.e2e_dir.mkdir(parents=True)
    site, targets, _events = lifecycle_harness(monkeypatch, runner.e2e_dir)
    outcome = workload.run_e2e001(
        site=site,
        target=targets[0],
        case_dir=runner.e2e_dir,
        job_id="job-test",
        attempt_id="attempt-test",
        host_probe_image="unit@sha256:" + "a" * 64,
        attempt=1,
        maintenance_window_end=datetime.now(timezone.utc) + timedelta(minutes=30),
    )
    assert outcome["verdict"] == "PASS"
    nodes = json.loads(runner.trusted_cpu_baseline.read_text())
    runner.targets = [base.ClusterTarget("a", "ctx-a", "hp-a", "eks-a", "role-a")]
    runner.evidence_identity = lambda: {"release_id": "release-test", "cluster_id": "a"}
    runner.cpu_json = lambda *args: nodes if "nodes" in args else {"items": []}
    runner.predecessor = produce_containment(
        monkeypatch, runner.root_run_dir, nodes, release="release-test", cluster="a"
    )
    runner.blast_001()
    analysis = json.loads((runner.run_dir / "BLAST-001-analysis.json").read_text())
    assert analysis["workload_operations"] == ["RESTART_WORKLOAD", "STOP_WORKLOADS"]
    assert analysis["isolation_producers"] == [base.CONTAINMENT_CASE_ID], (
        "the real DESTR-001 producer is discovered through its SUCCEEDED records"
    )
    assert base.E2E001_CASE_ID in analysis["workload_producers"], (
        "the real E2E-001 producer is discovered through its SUCCEEDED records"
    )
    assert analysis["required_operations_present"] is True


@pytest.mark.parametrize("card", ["bound", "foreign-release"])
def test_blast001_counts_e2e001_from_its_card_window_not_its_verdict_start(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, card: str
) -> None:
    """The verdict file run_workload_acceptance.main writes next to the card
    stamps started_at before E2E-001 takes the baseline; read alone, it put the
    one producer of STOP_WORKLOADS/RESTART_WORKLOAD before its own baseline
    (live round 13: BLAST-001 failed with E2E-001 under
    producers_before_baseline once DESTR-010 ran on another node). E2E-001
    contributes from the window its card opened -- while the card agrees."""

    runner = _runner(tmp_path, case_id="GF-REGIONAL-BLAST-001", cls=BlastCasesOne)
    runner.e2e_dir.mkdir(parents=True)
    site, targets, _events = lifecycle_harness(monkeypatch, runner.e2e_dir)
    started_at = datetime.now(timezone.utc).isoformat()
    outcome = workload.run_e2e001(
        site=site,
        target=targets[0],
        case_dir=runner.e2e_dir,
        job_id="job-test",
        attempt_id="attempt-test",
        host_probe_image="unit@sha256:" + "a" * 64,
        attempt=1,
        maintenance_window_end=datetime.now(timezone.utc) + timedelta(minutes=30),
    )
    assert outcome["verdict"] == "PASS"
    # The verdict file as main() assembles it around the outcome.
    base.write_json(
        workload.case_evidence_path(runner.root_run_dir, base.E2E001_CASE_ID),
        {
            "schema_version": 2,
            "report_type": "fault-acceptance",
            "case_id": base.E2E001_CASE_ID,
            "verdict": outcome["verdict"],
            "started_at": started_at,
            "executed_at": datetime.now(timezone.utc).isoformat(),
            "predecessor": {"valid": True, "case_id": None, "verdict": "NOT_REQUIRED"},
            **{key: value for key, value in outcome.items() if key != "verdict"},
        },
    )
    card_path = runner.e2e_dir / base.E2E001_EXECUTION_CARD
    handoff = json.loads(card_path.read_text())
    window_start = base.parse_time(handoff["maintenance_window"]["start"])
    assert base.parse_time(started_at) < window_start
    nodes = json.loads(runner.trusted_cpu_baseline.read_text())
    runner.targets = [base.ClusterTarget("a", "ctx-a", "hp-a", "eks-a", "role-a")]
    runner.evidence_identity = lambda: {"release_id": "release-test", "cluster_id": "a"}
    runner.cpu_json = lambda *args: nodes if "nodes" in args else {"items": []}
    # DESTR-001 proves MARK_UNSCHEDULABLE only, as DESTR-010/001 did live.
    runner.predecessor = produce_containment(
        monkeypatch, runner.root_run_dir, nodes, release="release-test", cluster="a"
    )
    if card == "foreign-release":
        handoff["release_id"] = "release-other"
        base.write_json(card_path, handoff)
        with pytest.raises(
            base.CheckError,
            match=(
                "proves RESTART_WORKLOAD, STOP_WORKLOADS on release release-test.*"
                r"before the baseline: GF-REGIONAL-E2E-001\)"
            ),
        ):
            runner.blast_001()
        return
    runner.blast_001()
    analysis = json.loads((runner.run_dir / "BLAST-001-analysis.json").read_text())
    assert analysis["producers_before_baseline"] == []
    assert analysis["workload_producers"] == [base.E2E001_CASE_ID]
    assert (
        base.parse_time(analysis["producers"][base.E2E001_CASE_ID]["started_at"])
        == window_start
    )
    assert analysis["required_operations_present"] is True


@pytest.mark.parametrize(
    "defect",
    [
        "failed-step",
        "declared-only",
        "foreign-command",
        "wrong-node",
        "missing-command",
        "failed-verdict",
        "cleanup",
        "release",
        "cpu",
        "window",
        "predecessor",
        "missing-fence",
        "missing-step-index",
    ],
)
def test_containment_source_rejects_unproven_or_unbound_mark(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, defect: str
) -> None:
    nodes = {"items": [{"metadata": {"name": "cpu", "uid": "cpu-uid"}, "spec": {}}]}
    predecessor = produce_containment(monkeypatch, tmp_path, nodes)
    source_dir = tmp_path / "cases" / base.CONTAINMENT_CASE_ID
    state_path = source_dir / "workflow-state.json"
    source_path = source_dir / f"{base.CONTAINMENT_CASE_ID}.json"
    state = json.loads(state_path.read_text())
    source = json.loads(source_path.read_text())
    if defect == "failed-step":
        state["workflow"]["step_executions"][0]["status"] = "FAILED"
    elif defect == "declared-only":
        state["workflow"]["step_executions"].pop(0)
    elif defect == "foreign-command":
        state["commands"][0]["cluster_id"] = "other"
    elif defect == "wrong-node":
        state["commands"][0]["step"]["node_ids"] = ["other"]
    elif defect == "missing-command":
        state["commands"].pop(0)
    elif defect == "failed-verdict":
        source["verdict"] = "FAIL"
    elif defect == "cleanup":
        source["probe_residuals"]["pod"] = True
    elif defect == "release":
        source["release_id"] = "other"
    elif defect == "cpu":
        base.write_json(source_dir / "cpu-blast-after.json", {"nodes": {}})
    elif defect == "window":
        base.write_json(
            source_dir / "host-after.json", {"captured_at": "2000-01-01T00:00:00Z"}
        )
    elif defect == "missing-fence":
        state["workflow"].pop("fencing_token")
        for command in state["commands"]:
            command.pop("fencing_token")
    elif defect == "missing-step-index":
        state["workflow"]["step_executions"][0].pop("step_index")
        state["commands"][0].pop("step_index")
    else:
        predecessor["path"] = str(tmp_path / "other-run" / source_path.name)
    base.write_json(state_path, state)
    base.write_json(source_path, source)
    with pytest.raises(base.CheckError):
        base.containment_source(
            tmp_path,
            predecessor=predecessor,
            release_id="release-1",
            cluster_id="cluster-a",
            cpu_snapshot=BlastCasesOne.node_security_snapshot(nodes),
        )
