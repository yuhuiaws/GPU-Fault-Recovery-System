"""Unit contracts for the BLAST-001..004 runner (scripts/e2e/regional/*blast*)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.e2e.regional import blast_acceptance_base as base
from scripts.e2e.regional import blast_acceptance_cases_2 as cases_2
from scripts.e2e.regional import run_blast_acceptance as entry
from scripts.e2e.regional.blast_acceptance_cases_1 import BlastCasesOne
from scripts.e2e.regional.blast_acceptance_cases_2 import BlastCasesTwo

ROOT = Path(__file__).resolve().parents[2]


def _runner(
    tmp_path: Path,
    *,
    case_id: str = "GF-REGIONAL-BLAST-002",
    cls: type[BlastCasesTwo] | type[BlastCasesOne] = BlastCasesTwo,
) -> Any:
    """A runner with the live-site constructor bypassed."""

    runner = cls.__new__(cls)
    runner.site_path = tmp_path / "site.yaml"
    runner.root_run_dir = tmp_path / "run"
    runner.case_id = case_id
    runner.run_dir = runner.root_run_dir / "cases" / case_id
    runner.run_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    runner.e2e_dir = runner.root_run_dir / "cases" / base.E2E001_CASE_ID
    runner.trusted_cpu_baseline = runner.e2e_dir / base.E2E001_CPU_NODES_BEFORE
    runner.predecessor = {"valid": True}
    runner.preflight_reuse_seconds = base.PREFLIGHT_REUSE_SECONDS
    # The release-state read is a live call; every case result is bound to it.
    runner.evidence_identity = lambda: {
        "release_id": "release-1",
        "cluster_id": "cluster-a",
    }
    runner.namespace = "gpu-fault-system"
    runner.targets = [
        base.ClusterTarget(
            cluster_id="cluster-a",
            context="ctx-a",
            hyperpod_cluster_name="hp-a",
            eks_cluster_arn="arn:aws:eks:us-west-2:000000000000:cluster/gpu-a",
            executor_role_arn="arn:aws:iam::000000000000:role/executor-a",
        )
    ]
    runner.case_statuses = []
    return runner


def test_blast001_inputs_default_to_this_runs_e2e001_case_dir(tmp_path: Path) -> None:
    arguments = entry.parse_args(
        [
            "--case",
            "GF-REGIONAL-BLAST-001",
            "--site",
            str(tmp_path / "site.yaml"),
            "--run-dir",
            str(tmp_path / "run"),
        ]
    )

    e2e_dir, baseline = entry.resolve_blast001_inputs(arguments)

    assert e2e_dir == tmp_path / "run" / "cases" / "GF-REGIONAL-E2E-001"
    assert baseline == e2e_dir / "cpu-nodes-before.json"
    # The old default `.` passed exists() on any working directory and then
    # crashed on read_text(); every input is now checked as a file.
    errors = base.blast001_input_errors(e2e_dir, baseline)
    assert any("execution-card.json" in item for item in errors) or any(
        "not a directory" in item for item in errors
    )
    assert any("trusted CPU baseline" in item for item in errors), (
        "the default baseline path must be reported as missing, not silently accepted"
    )


def test_blast001_inputs_accept_a_complete_e2e001_case_dir(tmp_path: Path) -> None:
    e2e_dir = tmp_path / "cases" / base.E2E001_CASE_ID
    e2e_dir.mkdir(parents=True)
    for name in (
        base.E2E001_EXECUTION_CARD,
        base.E2E001_CONTROL_PLANE_STATE,
        base.E2E001_CPU_NODES_BEFORE,
    ):
        (e2e_dir / name).write_text("{}", encoding="utf-8")

    assert base.blast001_input_errors(e2e_dir, e2e_dir / "cpu-nodes-before.json") == []
    # A directory where a file is expected is not "exists"-good-enough.
    assert base.blast001_input_errors(e2e_dir, e2e_dir) == [
        f"trusted CPU baseline is not a file: {e2e_dir}"
    ]


def test_write_json_is_the_atomic_scoped_writer(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "evidence.json"

    base.write_json(path, {"case_id": "GF-REGIONAL-BLAST-002", "verdict": "PASS"})

    assert json.loads(path.read_text(encoding="utf-8"))["verdict"] == "PASS"
    assert path.stat().st_mode & 0o777 == 0o600
    assert not list(path.parent.glob(".*.tmp")), (
        "the atomic writer leaves no temp file behind"
    )


def test_run_writes_a_fail_evidence_file_for_a_non_check_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _runner(tmp_path)

    def broken_preflight() -> None:
        raise KeyError("Orchestrator")

    monkeypatch.setattr(runner, "preflight", broken_preflight)

    assert runner.run() == 1

    evidence = json.loads(
        (runner.run_dir / f"{runner.case_id}.json").read_text(encoding="utf-8")
    )
    assert evidence["verdict"] == "FAIL"
    assert evidence["error"].startswith("KeyError"), (
        "the raised exception type opens the recorded error"
    )
    # The result is bound to the release/cluster it was read against.
    assert evidence["release_id"] == "release-1"
    assert evidence["cluster_id"] == "cluster-a"
    summary = json.loads(
        (runner.run_dir / "phase-summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "FAIL"


def test_preflight_is_reused_within_the_window_for_the_same_site(
    tmp_path: Path,
) -> None:
    runner = _runner(tmp_path)
    cache = runner.root_run_dir / base.PREFLIGHT_CACHE_NAME
    scope = {
        "captured_at": base.utc_now(),
        "site": str(runner.site_path),
        "gpu_clusters": [{"cluster_id": "cluster-a"}],
    }
    base.write_json(cache, scope)

    assert runner.reusable_preflight() is not None
    runner.preflight()
    written = json.loads(
        (runner.run_dir / "execution-scope.json").read_text(encoding="utf-8")
    )
    assert written["reused_from"] == str(cache)

    stale = dict(scope)
    stale["captured_at"] = (
        datetime.now(timezone.utc)
        - timedelta(seconds=runner.preflight_reuse_seconds + 1)
    ).isoformat()
    base.write_json(cache, stale)
    assert runner.reusable_preflight() is None

    other_site = dict(scope)
    other_site["site"] = str(tmp_path / "other.yaml")
    base.write_json(cache, other_site)
    assert runner.reusable_preflight() is None

    other_clusters = dict(scope)
    other_clusters["gpu_clusters"] = [{"cluster_id": "cluster-b"}]
    base.write_json(cache, other_clusters)
    assert runner.reusable_preflight() is None


def test_sagemaker_read_only_accepts_a_role_with_no_sagemaker_permission() -> None:
    assert BlastCasesOne.sagemaker_read_only([]) is True
    assert BlastCasesOne.sagemaker_read_only(
        ["sagemaker:DescribeCluster", "sagemaker:ListClusterNodes"]
    ), "describe/list are read-only SageMaker actions"
    assert not BlastCasesOne.sagemaker_read_only(
        ["sagemaker:DescribeCluster", "sagemaker:BatchDeleteClusterNodes"]
    ), "BatchDeleteClusterNodes is a mutation"
    assert BlastCasesOne.sagemaker_patterns(
        ["ses:SendEmail", "sagemaker:ListClusters", "SageMaker:DescribeCluster"]
    ) == ["sagemaker:ListClusters", "SageMaker:DescribeCluster"]


def test_expected_executor_role_is_parsed_from_the_shipped_manifest() -> None:
    expected = cases_2.expected_executor_role()

    documents = list(
        yaml.safe_load_all(cases_2.EXECUTOR_MANIFEST.read_text(encoding="utf-8"))
    )
    role = next(
        item
        for item in documents
        if isinstance(item, dict)
        and item.get("kind") == "ClusterRole"
        and item["metadata"]["name"] == "gpu-fault-cluster-executor"
    )
    assert expected == BlastCasesTwo.normalized_role_rules(role)
    assert "delete" not in expected["core:nodes"], expected["core:nodes"]
    assert "create" not in expected["core:pods"], expected["core:pods"]
    # Since the security review (459bcc1) the ClusterRole grants pods read-only
    # verbs; pods delete/patch live in per-namespace Roles, so the cluster-wide
    # expectation is exactly the read set and nothing wider.
    assert set(expected["core:pods"]) <= {"get", "list", "watch"}, expected["core:pods"]


def test_expected_executor_role_refuses_a_manifest_without_the_role(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "executor.yaml"
    manifest.write_text("apiVersion: v1\nkind: ServiceAccount\n", encoding="utf-8")

    with pytest.raises(base.CheckError, match="declares no ClusterRole"):
        cases_2.expected_executor_role(manifest)


def _node(name: str, managed_fields: list[dict[str, Any]]) -> dict[str, Any]:
    return {"metadata": {"name": name, "managedFields": managed_fields}}


def test_managed_field_writes_ignore_entries_already_in_the_baseline() -> None:
    since = datetime(2026, 9, 7, 10, 0, tzinfo=timezone.utc)
    pre_window = {
        "manager": "kubectl",
        "operation": "Update",
        "time": "2026-09-07T10:30:00Z",
        "fieldsV1": {"f:spec": {"f:taints": {}}},
    }
    new_write = {
        "manager": "gpu-fault-executor",
        "operation": "Update",
        "time": "2026-09-07T10:31:00Z",
        "fieldsV1": {"f:spec": {"f:unschedulable": {}}},
    }
    baseline = {"items": [_node("cpu-1", [pre_window])]}
    current = {"items": [_node("cpu-1", [pre_window, new_write])]}

    hits = BlastCasesOne.managed_field_writes(
        current, baseline_nodes=baseline, since=since
    )

    # The time filter alone would have flagged `pre_window` too; it is inside
    # the window by timestamp but was already present before the fault.
    assert hits == [
        {
            "node": "cpu-1",
            "manager": "gpu-fault-executor",
            "operation": "Update",
            "time": "2026-09-07T10:31:00Z",
        }
    ]
    assert (
        BlastCasesOne.managed_field_writes(
            baseline, baseline_nodes=baseline, since=since
        )
        == []
    )


def _e2e001_handoff(e2e_dir: Path, nodes: dict[str, Any]) -> None:
    """The files E2E-001 leaves for BLAST-001, in the shape ``run_e2e001`` writes.

    ``run_e2e001`` writes the execution card and ``control-plane-current.json``
    inline at the end of a live run; the shape here mirrors that write.
    """

    e2e_dir.mkdir(parents=True)
    base.write_json(
        e2e_dir / base.E2E001_EXECUTION_CARD,
        {
            "case_id": base.E2E001_CASE_ID,
            "maintenance_window": {
                "start": "2026-09-07T10:00:00+00:00",
                "end": "2026-09-07T11:00:00+00:00",
            },
            "cluster_id": "cluster-a",
            "node": "gpu-1",
            "job_id": "job-a",
            "attempt_id": "job-a-a001",
        },
    )
    base.write_json(
        e2e_dir / base.E2E001_CONTROL_PLANE_STATE,
        {
            "workflows": [
                {
                    "official_steps": [
                        {"operation": "MARK_UNSCHEDULABLE"},
                        {"operation": "STOP_WORKLOADS"},
                    ],
                    "safety_steps": [{"operation": "FREEZE_EVIDENCE"}],
                    "step_executions": [{"operation": "RESTART_WORKLOAD"}],
                }
            ]
        },
    )
    base.write_json(e2e_dir / base.E2E001_CPU_NODES_BEFORE, nodes)


def _blast001_run(tmp_path: Path) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Run BLAST-001 against a quiet fake CPU cluster; return runner, evidence, analysis."""

    runner = _runner(tmp_path, case_id="GF-REGIONAL-BLAST-001", cls=BlastCasesOne)
    nodes = {"items": [_node("cpu-1", [])]}
    _e2e001_handoff(runner.e2e_dir, nodes)
    listings = {
        ("get", "nodes", "-o", "json"): nodes,
        ("get", "jobs", "-A", "-o", "json"): {"items": []},
        ("get", "events", "-A", "-o", "json"): {"items": []},
    }
    runner.cpu_json = lambda *args: listings[args]

    runner.blast_001()

    evidence = json.loads(
        (runner.run_dir / "GF-REGIONAL-BLAST-001.json").read_text(encoding="utf-8")
    )
    analysis = json.loads(
        (runner.run_dir / "BLAST-001-analysis.json").read_text(encoding="utf-8")
    )
    return runner, evidence, analysis


def test_blast001_reads_the_window_and_the_workflow_steps_e2e001_hands_over(
    tmp_path: Path,
) -> None:
    """BLAST-001 consumes ``maintenance_window.start/end`` and ``workflows[]`` steps.

    The producer side (``run_e2e001`` writing those keys) is a live-run tail
    with no seam of its own; this proves the consumer reads exactly the keys
    the handoff fixture mirrors from that write.
    """

    _runner_, evidence, analysis = _blast001_run(tmp_path)

    assert evidence["verdict"] == "PASS", evidence
    assert analysis["e2e_window_start"] == "2026-09-07T10:00:00+00:00"
    assert analysis["e2e_window_end"] == "2026-09-07T11:00:00+00:00"
    # Steps are gathered from all three step groups of the handed-over workflow.
    assert analysis["workflow_operations"] == [
        "FREEZE_EVIDENCE",
        "MARK_UNSCHEDULABLE",
        "RESTART_WORKLOAD",
        "STOP_WORKLOADS",
    ]
    assert evidence["checks"]["required_e2e_operations_present"] is True


def test_blast001_limitation_names_the_before_snapshot_it_compares_against(
    tmp_path: Path,
) -> None:
    runner, evidence, analysis = _blast001_run(tmp_path)

    assert analysis["trusted_baseline"] == str(runner.trusted_cpu_baseline)
    assert evidence["checks"]["cpu_nodes_identical_to_trusted_baseline"] is True
    limitations = evidence["limitations"]
    assert any(base.E2E001_CPU_NODES_BEFORE in item for item in limitations), (
        "the limitation names the E2E-001 before-snapshot as the baseline"
    )
    assert not any(
        "did not preserve an immediate CPU node" in item for item in limitations
    ), "the old claim that no before-snapshot exists is gone"
