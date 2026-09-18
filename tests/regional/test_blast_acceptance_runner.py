"""Unit contracts for the BLAST-001..004 runner (scripts/e2e/regional/*blast*)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import yaml

from scripts.e2e.regional import blast_acceptance_base as base
from scripts.e2e.regional import blast_acceptance_cases_2 as cases_2
from scripts.e2e.regional import run_blast_acceptance as entry
from scripts.e2e.regional.blast_acceptance_cases_1 import BlastCasesOne
from scripts.e2e.regional.blast_acceptance_cases_2 import BlastCasesTwo
from tests.regional._blast_containment_support import produce_containment

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def local_source_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(base, "source_digest", lambda: "a" * 64)


def _runner(
    tmp_path: Path,
    *,
    case_id: str = "GF-REGIONAL-BLAST-002",
    cls: type[BlastCasesTwo] | type[BlastCasesOne] = BlastCasesTwo,
) -> Any:
    """Initialize the real runner against an isolated, already parsed site."""

    site_path = tmp_path / "site.yaml"
    site_path.write_text("test site", encoding="ascii")
    run_dir = tmp_path / "run"
    e2e_dir = run_dir / "cases" / base.E2E001_CASE_ID
    for plane in ("cpu", "gpu"):
        (tmp_path / f"{plane}.kubeconfig").write_text(
            f"{plane} connection fixture", encoding="ascii"
        )
    site = SimpleNamespace(
        source_sha256=base.sha256_bytes(site_path.read_bytes()),
        environment={},
        release_config={
            "namespace": "gpu-fault-system",
            "aws_region": "us-west-2",
            "cpu_kubeconfig": str(tmp_path / "cpu.kubeconfig"),
            "gpu_kubeconfig": str(tmp_path / "gpu.kubeconfig"),
            "cpu_eks_arn": "arn:aws:eks:us-west-2:000000000000:cluster/cpu",
            "clusters": [
                {
                    "cluster_id": "cluster-a",
                    "context": "ctx-a",
                    "hyperpod_cluster_name": "hp-a",
                    "eks_cluster_arn": "arn:aws:eks:us-west-2:000000000000:cluster/gpu-a",
                    "executor_irsa_role_arn": "arn:aws:iam::000000000000:role/executor-a",
                }
            ],
        },
    )
    with patch.object(base, "load_site", return_value=site):
        runner = cls(
            site_path=site_path,
            run_dir=run_dir,
            case_id=case_id,
            e2e_dir=e2e_dir,
            trusted_cpu_baseline=e2e_dir / base.E2E001_CPU_NODES_BEFORE,
            predecessor={"valid": True},
        )
    # The release-state read is a live call; every case result is bound to it.
    runner.evidence_identity = lambda: {
        "release_id": "release-1",
        "cluster_id": "cluster-a",
    }
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
        "binding": runner.preflight_binding(),
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
    return {
        "metadata": {
            "name": name,
            "uid": name + "-uid",
            "managedFields": managed_fields,
        }
    }


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
    e2e_dir.mkdir(parents=True)
    state_path = e2e_dir / base.E2E001_CONTROL_PLANE_STATE
    baseline_path = e2e_dir / base.E2E001_CPU_NODES_BEFORE
    base.write_json(
        state_path,
        {
            "workflows": [
                {
                    "status": "SUCCEEDED",
                    "step_executions": [
                        {"operation": operation, "status": "SUCCEEDED"}
                        for operation in (
                            "FREEZE_EVIDENCE",
                            "STOP_WORKLOADS",
                            "RESTART_WORKLOAD",
                        )
                    ],
                }
            ]
        },
    )
    base.write_json(baseline_path, nodes)
    base.write_json(
        e2e_dir / base.E2E001_EXECUTION_CARD,
        {
            "case_id": base.E2E001_CASE_ID,
            "verdict": "PASS",
            "cleanup_complete": True,
            "release_id": "release-1",
            "baseline_sha256": base.sha256_bytes(baseline_path.read_bytes()),
            "state_sha256": base.sha256_bytes(state_path.read_bytes()),
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


def _blast001_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Run BLAST-001 against a quiet fake CPU cluster; return runner, evidence, analysis."""

    runner = _runner(tmp_path, case_id="GF-REGIONAL-BLAST-001", cls=BlastCasesOne)
    nodes = {"items": [_node("cpu-1", [])]}
    _e2e001_handoff(runner.e2e_dir, nodes)
    runner.predecessor = produce_containment(monkeypatch, runner.root_run_dir, nodes)
    listings = {
        ("get", "nodes", "-o", "json"): nodes,
        ("get", "jobs", "-A", "-o", "json"): {"items": []},
        ("get", "events", "-A", "-o", "json"): {"items": []},
    }
    runner.cpu_json = lambda *args: listings[args]
    runner.evidence_identity = lambda: {
        "release_id": "release-1",
        "cluster_id": "cluster-a",
    }

    runner.blast_001()

    evidence = json.loads(
        (runner.run_dir / "GF-REGIONAL-BLAST-001.json").read_text(encoding="utf-8")
    )
    analysis = json.loads(
        (runner.run_dir / "BLAST-001-analysis.json").read_text(encoding="utf-8")
    )
    return runner, evidence, analysis


def test_blast001_reads_the_window_and_the_workflow_steps_e2e001_hands_over(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """BLAST-001 consumes ``maintenance_window.start/end`` and ``workflows[]`` steps.

    The producer side (``run_e2e001`` writing those keys) is a live-run tail
    with no seam of its own; this proves the consumer reads exactly the keys
    the handoff fixture mirrors from that write.
    """

    _runner_, evidence, analysis = _blast001_run(tmp_path, monkeypatch)

    assert evidence["verdict"] == "PASS", evidence
    assert analysis["e2e_window_start"] == "2026-09-07T10:00:00+00:00"
    assert analysis["e2e_window_end"] == "2026-09-07T11:00:00+00:00"
    assert analysis["workload_operations"] == ["RESTART_WORKLOAD", "STOP_WORKLOADS"], (
        "the workload side is read from SUCCEEDED step executions only"
    )
    assert analysis["isolation_producers"] == [base.CONTAINMENT_CASE_ID], (
        "DESTR-001's SUCCEEDED MARK_UNSCHEDULABLE makes it the isolation producer"
    )
    assert analysis["workload_producers"] == [base.E2E001_CASE_ID], (
        "E2E-001's SUCCEEDED STOP/RESTART make it the workload producer"
    )
    assert {"MARK_UNSCHEDULABLE", "RESTART_WORKLOAD", "STOP_WORKLOADS"} <= set(
        analysis["workflow_operations"]
    ), analysis["workflow_operations"]
    assert evidence["checks"]["required_e2e_operations_present"] is True


def test_blast001_limitation_names_the_before_snapshot_it_compares_against(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, evidence, analysis = _blast001_run(tmp_path, monkeypatch)

    assert analysis["trusted_baseline"] == str(runner.trusted_cpu_baseline)
    assert evidence["checks"]["cpu_nodes_identical_to_trusted_baseline"] is True
    limitations = evidence["limitations"]
    assert any(base.E2E001_CPU_NODES_BEFORE in item for item in limitations), (
        "the limitation names the E2E-001 before-snapshot as the baseline"
    )
    assert not any(
        "did not preserve an immediate CPU node" in item for item in limitations
    ), "the old claim that no before-snapshot exists is gone"


def _pass_case(
    run_dir: Path,
    case_id: str,
    *,
    release: str,
    operations: list[str],
    started_at: str,
    sidecar: str | None = None,
) -> None:
    """A PASS verdict whose SUCCEEDED workflow proves ``operations``.

    ``sidecar`` puts the workflow into a second JSON file, as COLLECT-021 does
    with passive-completion.json, so discovery must read the whole case dir.
    """

    case_dir = run_dir / "cases" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    workflow = {
        "request_id": f"{case_id.lower()}-workflow",
        "status": "SUCCEEDED",
        "official_steps": [{"operation": item} for item in operations],
        "step_executions": [
            {"operation": item, "status": "SUCCEEDED"} for item in operations
        ],
    }
    verdict: dict[str, Any] = {
        "case_id": case_id,
        "verdict": "PASS",
        "release_id": release,
        "cluster_id": "cluster-a",
        "errors": [],
        "started_at": started_at,
    }
    if sidecar:
        base.write_json(case_dir / sidecar, {"recovery_workflow": workflow})
    else:
        verdict["workflow"] = workflow
    base.write_json(case_dir / f"{case_id}.json", verdict)


def test_blast001_binds_producers_by_release_not_by_case_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The baseline snapshot comes from an E2E-001 card of an OLDER release; the
    # producers are whichever PASS cases of the run prove the operations on the
    # live release -- here a passive restart (sidecar file) and a reboot case.
    runner = _runner(tmp_path, case_id="GF-REGIONAL-BLAST-001", cls=BlastCasesOne)
    nodes = {"items": [_node("cpu-1", [])]}
    _e2e001_handoff(runner.e2e_dir, nodes)
    card_path = runner.e2e_dir / base.E2E001_EXECUTION_CARD
    card = json.loads(card_path.read_text())
    card["release_id"] = "release-0"
    base.write_json(card_path, card)
    _pass_case(
        runner.root_run_dir,
        "GF-REGIONAL-COLLECT-021",
        release="release-1",
        operations=["FREEZE_EVIDENCE", "STOP_WORKLOADS", "RESTART_WORKLOAD"],
        started_at="2026-09-08T10:00:00+00:00",
        sidecar="passive-completion.json",
    )
    _pass_case(
        runner.root_run_dir,
        "GF-REGIONAL-COLLECT-015",
        release="release-1",
        operations=["MARK_UNSCHEDULABLE", "RESTART_NODE"],
        started_at="2026-09-08T11:00:00+00:00",
    )
    _pass_case(
        runner.root_run_dir,
        "GF-REGIONAL-DESTR-009",
        release="release-0",
        operations=["STOP_WORKLOADS", "RESTART_WORKLOAD"],
        started_at="2026-09-08T09:00:00+00:00",
    )
    listings = {
        ("get", "nodes", "-o", "json"): nodes,
        ("get", "jobs", "-A", "-o", "json"): {"items": []},
        ("get", "events", "-A", "-o", "json"): {"items": []},
    }
    runner.cpu_json = lambda *args: listings[args]

    runner.blast_001()

    analysis = json.loads(
        (runner.run_dir / "BLAST-001-analysis.json").read_text(encoding="utf-8")
    )
    assert sorted(analysis["producers"]) == [
        "GF-REGIONAL-COLLECT-015",
        "GF-REGIONAL-COLLECT-021",
    ], "only PASS cases bound to the live release count; release-0 evidence does not"
    assert analysis["workload_producers"] == ["GF-REGIONAL-COLLECT-021"], (
        "the sidecar file's SUCCEEDED workflow proves the workload side"
    )
    assert analysis["isolation_producers"] == ["GF-REGIONAL-COLLECT-015"], (
        "a reboot case's SUCCEEDED MARK_UNSCHEDULABLE proves the isolation side"
    )
    assert analysis["required_operations_present"] is True, analysis


def test_blast001_refuses_when_no_live_release_case_proves_an_operation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _runner(tmp_path, case_id="GF-REGIONAL-BLAST-001", cls=BlastCasesOne)
    nodes = {"items": [_node("cpu-1", [])]}
    _e2e001_handoff(runner.e2e_dir, nodes)
    card_path = runner.e2e_dir / base.E2E001_EXECUTION_CARD
    card = json.loads(card_path.read_text())
    card["release_id"] = "release-0"
    base.write_json(card_path, card)
    _pass_case(
        runner.root_run_dir,
        "GF-REGIONAL-COLLECT-015",
        release="release-1",
        operations=["MARK_UNSCHEDULABLE"],
        started_at="2026-09-08T11:00:00+00:00",
    )
    runner.cpu_json = lambda *args: nodes if "nodes" in args else {"items": []}
    with pytest.raises(base.CheckError, match="RESTART_WORKLOAD, STOP_WORKLOADS"):
        runner.blast_001()


def test_blast001_does_not_count_a_producer_that_started_before_the_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _runner(tmp_path, case_id="GF-REGIONAL-BLAST-001", cls=BlastCasesOne)
    nodes = {"items": [_node("cpu-1", [])]}
    _e2e001_handoff(runner.e2e_dir, nodes)
    # The card's window starts 2026-09-07T10:00; a producer that started earlier
    # cannot be judged against a baseline taken after it ran, so it is listed
    # and excluded -- and with it the only MARK_UNSCHEDULABLE proof is gone.
    _pass_case(
        runner.root_run_dir,
        "GF-REGIONAL-COLLECT-015",
        release="release-1",
        operations=["MARK_UNSCHEDULABLE"],
        started_at="2026-09-07T09:00:00+00:00",
    )
    runner.cpu_json = lambda *args: nodes if "nodes" in args else {"items": []}
    with pytest.raises(
        base.CheckError,
        match=(
            "inside the audited window proves MARK_UNSCHEDULABLE.*"
            "producers before the baseline: GF-REGIONAL-COLLECT-015"
        ),
    ):
        runner.blast_001()


def test_succeeded_operations_read_only_succeeded_records() -> None:
    document = {
        "workflows": [
            {
                "request_id": "wf-ok",
                "status": "SUCCEEDED",
                "official_steps": [{"operation": "RESET_GPU"}],
                "step_executions": [
                    {"operation": "MARK_UNSCHEDULABLE", "status": "SUCCEEDED"},
                    {"operation": "RESET_GPU", "status": "FAILED"},
                ],
            },
            {
                "request_id": "wf-open",
                "status": "RUNNING",
                "step_executions": [
                    {"operation": "STOP_WORKLOADS", "status": "SUCCEEDED"}
                ],
            },
        ],
        "commands": [
            {
                "command_id": "cmd-1",
                "workflow_request_id": "wf-ok",
                "status": "SUCCEEDED",
                "step": {"operation": "WorkflowOperation.RESTART_WORKLOAD"},
            },
            {
                "command_id": "cmd-2",
                "workflow_request_id": "wf-ok",
                "status": "FAILED",
                "step": {"operation": "REPLACE_NODE"},
            },
        ],
    }
    found = base.succeeded_operations(document)
    assert found == {"MARK_UNSCHEDULABLE": {"wf-ok"}, "RESTART_WORKLOAD": {"wf-ok"}}, (
        "declared steps, FAILED executions and non-terminal workflows prove nothing"
    )
