"""BLAST runner input and producer-scan edges.

A BLAST-001 case directory that exists but lacks one evidence file, a site that
does not resolve a GPU kubeconfig or lists no GPU clusters, the containment
command binding when the only candidate command covers a different node, and
the release-bound producer scan skipping stray files, unreadable or oversized
documents and cases that prove no SUCCEEDED operation, and placing E2E-001 at
its card's window while leaving out store snapshots a later case's plan read.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import blast_acceptance_base as base

RELEASE = "release-unit"
CLUSTER = "cluster-a"


def test_blast001_inputs_name_each_missing_evidence_file(tmp_path: Path) -> None:
    e2e_dir = tmp_path / "cases" / base.E2E001_CASE_ID
    e2e_dir.mkdir(parents=True)
    (e2e_dir / base.E2E001_EXECUTION_CARD).write_text("{}")
    baseline = e2e_dir / base.E2E001_CPU_NODES_BEFORE
    baseline.write_text("{}")
    errors = base.blast001_input_errors(e2e_dir, baseline)
    assert errors == [
        f"E2E-001 evidence file is missing: {e2e_dir / base.E2E001_CONTROL_PLANE_STATE}"
    ]


def site_config(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "namespace": "gpu-fault-system",
        "aws_region": "us-west-2",
        "cpu_kubeconfig": str(tmp_path / "cpu.kubeconfig"),
        "gpu_kubeconfig": str(tmp_path / "gpu.kubeconfig"),
        "cpu_eks_arn": "arn:aws:eks:us-west-2:000000000000:cluster/cpu",
        "clusters": [
            {
                "cluster_id": CLUSTER,
                "context": "ctx-a",
                "hyperpod_cluster_name": "hp-a",
                "eks_cluster_arn": "arn:aws:eks:us-west-2:000000000000:cluster/gpu-a",
                "executor_irsa_role_arn": "arn:aws:iam::000000000000:role/executor-a",
            }
        ],
    }
    config.update(overrides)
    return config


def construct_runner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **overrides: Any
) -> base.BlastRunnerBase:
    site_path = tmp_path / "site.yaml"
    site_path.write_text("test site", encoding="ascii")
    site = SimpleNamespace(
        source_sha256=base.sha256_bytes(site_path.read_bytes()),
        environment={},
        release_config=site_config(tmp_path, **overrides),
    )
    monkeypatch.setattr(base, "load_site", lambda path: site)
    e2e_dir = tmp_path / "run" / "cases" / base.E2E001_CASE_ID
    return base.BlastRunnerBase(
        site_path=site_path,
        run_dir=tmp_path / "run",
        case_id="GF-REGIONAL-BLAST-002",
        e2e_dir=e2e_dir,
        trusted_cpu_baseline=e2e_dir / base.E2E001_CPU_NODES_BEFORE,
        predecessor={"valid": True},
    )


def test_runner_requires_a_gpu_kubeconfig_from_the_site_or_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with pytest.raises(base.CheckError, match="does not resolve a GPU kubeconfig"):
        construct_runner(monkeypatch, tmp_path, gpu_kubeconfig="")
    assert not (tmp_path / "run").exists(), "a refused site creates no run directory"


def test_runner_requires_at_least_one_gpu_cluster(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with pytest.raises(base.CheckError, match="contains no GPU clusters"):
        construct_runner(monkeypatch, tmp_path, clusters=[])
    assert not (tmp_path / "run").exists(), "a refused site creates no run directory"


def containment_state(node_ids: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    source = {
        "workflow_request_id": "wf-1",
        "incident_id": "inc-1",
        "cluster_id": CLUSTER,
        "node": "node-a",
    }
    state = {
        "workflow": {
            "request_id": "wf-1",
            "incident_id": "inc-1",
            "status": "SUCCEEDED",
            "fencing_token": 7,
            "step_executions": [
                {
                    "operation": "MARK_UNSCHEDULABLE",
                    "status": "SUCCEEDED",
                    "step_index": 1,
                    "adapter_operation_id": "remote/cmd-1",
                }
            ],
        },
        "incident": {
            "incident_id": "inc-1",
            "workflow_request_id": "wf-1",
            "cluster_id": CLUSTER,
            "event_id": "ev-1",
            "node_ids": ["node-a"],
        },
        "event": {"event_id": "ev-1", "cluster_id": CLUSTER, "node_id": "node-a"},
        "commands": [
            {
                "command_id": "cmd-1",
                "status": "SUCCEEDED",
                "cluster_id": CLUSTER,
                "workflow_request_id": "wf-1",
                "incident_id": "inc-1",
                "fencing_token": 7,
                "last_lease_owner": "executor-a",
                "step_index": 1,
                "step": {"operation": "MARK_UNSCHEDULABLE", "node_ids": node_ids},
            }
        ],
    }
    return state, source


def test_containment_command_binds_only_the_command_covering_the_source_node() -> None:
    state, source = containment_state(["node-a"])
    assert base.successful_containment_command(state, source) == "cmd-1"
    state, source = containment_state(["node-b"])
    with pytest.raises(base.CheckError, match="missing or ambiguous"):
        base.successful_containment_command(state, source)


def verdict(case_id: str, **extra: Any) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "verdict": "PASS",
        "release_id": RELEASE,
        "cluster_id": CLUSTER,
        "errors": [],
        **extra,
    }


def test_producer_scan_skips_stray_files_unreadable_or_oversized_documents(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cases = tmp_path / "cases"
    cases.mkdir()
    (cases / "GF-REGIONAL-STRAY-001").write_text("not a directory")
    quiet = cases / "GF-REGIONAL-HA-001"
    quiet.mkdir()
    (quiet / "GF-REGIONAL-HA-001.json").write_text(
        json.dumps(verdict("GF-REGIONAL-HA-001"))
    )
    (quiet / "broken.json").write_text("{not json")
    (quiet / "oversized.json").write_text(
        json.dumps(
            {
                "status": "SUCCEEDED",
                "request_id": "wf-big",
                "step_executions": [{"operation": "RESET_GPU", "status": "SUCCEEDED"}],
                "padding": "x" * 400,
            }
        )
    )
    proving = cases / "GF-REGIONAL-HA-002"
    proving.mkdir()
    (proving / "GF-REGIONAL-HA-002.json").write_text(
        json.dumps(
            verdict(
                "GF-REGIONAL-HA-002",
                started_at="2026-01-01T00:00:00+00:00",
                ended_at="2026-01-01T00:10:00+00:00",
            )
        )
    )
    (proving / "workflow.json").write_text(
        json.dumps(
            {
                "status": "SUCCEEDED",
                "request_id": "wf-2",
                "step_executions": [
                    {"operation": "MARK_UNSCHEDULABLE", "status": "SUCCEEDED"}
                ],
            }
        )
    )
    monkeypatch.setattr(base, "PRODUCER_EVIDENCE_LIMIT_BYTES", 350)

    producers = base.release_bound_producers(
        tmp_path, release_id=RELEASE, cluster_ids={CLUSTER}
    )
    assert list(producers) == ["GF-REGIONAL-HA-002"]
    assert producers["GF-REGIONAL-HA-002"]["operations"] == {
        "MARK_UNSCHEDULABLE": ["wf-2"]
    }
    assert producers["GF-REGIONAL-HA-002"]["files"] == [
        "GF-REGIONAL-HA-002.json",
        "workflow.json",
    ]
    assert producers["GF-REGIONAL-HA-002"]["started_at"] == "2026-01-01T00:00:00+00:00"


def test_producer_scan_places_e2e001_at_its_card_and_leaves_plan_snapshots_out(
    tmp_path: Path,
) -> None:
    """Live round 12: DESTR-010 ran on E2E-001's node, so the store read in its
    plan carried E2E-001's SUCCEEDED STOP/RESTART and BLAST-001 credited them
    to DESTR-010, while E2E-001 -- its verdict's started_at precedes the
    baseline it takes -- sat before its own baseline. The workload proof is
    E2E-001's, from its card's window on; DESTR-010 keeps what it ran."""

    cases = tmp_path / "cases"
    e2e = cases / base.E2E001_CASE_ID
    e2e.mkdir(parents=True)
    restart = {
        "request_id": "wf-restart",
        "status": "SUCCEEDED",
        "created_at": "2026-01-01T10:01:05Z",
        "step_executions": [
            {
                "operation": operation,
                "status": "SUCCEEDED",
                "started_at": f"2026-01-01T10:01:{second}Z",
            }
            for operation, second in (("STOP_WORKLOADS", 10), ("RESTART_WORKLOAD", 20))
        ],
    }
    (e2e / f"{base.E2E001_CASE_ID}.json").write_text(
        json.dumps(
            verdict(
                base.E2E001_CASE_ID,
                started_at="2026-01-01T10:00:00+00:00",
                executed_at="2026-01-01T10:03:00+00:00",
                state={"workflow": restart},
            )
        )
    )
    (e2e / base.E2E001_EXECUTION_CARD).write_text(
        json.dumps(
            {
                **verdict(base.E2E001_CASE_ID),
                "maintenance_window": {
                    "start": "2026-01-01T10:01:00+00:00",
                    "end": "2026-01-01T10:03:00+00:00",
                },
            }
        )
    )
    reset = cases / "GF-REGIONAL-DESTR-010"
    reset.mkdir()
    (reset / "plan.json").write_text(
        json.dumps({"details": {"preflight": {"store": {"workflow": restart}}}})
    )
    (reset / "workflow-state.json").write_text(
        json.dumps(
            {
                "workflow": {
                    "request_id": "wf-reset",
                    "status": "SUCCEEDED",
                    "created_at": "2026-01-01T11:05:05Z",
                    "step_executions": [
                        {
                            "operation": "MARK_UNSCHEDULABLE",
                            "status": "SUCCEEDED",
                            "started_at": "2026-01-01T11:05:10Z",
                        }
                    ],
                }
            }
        )
    )
    (reset / "GF-REGIONAL-DESTR-010.json").write_text(
        json.dumps(verdict("GF-REGIONAL-DESTR-010"))
    )
    # DESTR-010's verdict has no start stamp: its earliest file, the plan, is it.
    for name, minute in (
        ("plan.json", 0),
        ("workflow-state.json", 20),
        ("GF-REGIONAL-DESTR-010.json", 25),
    ):
        stamp = datetime(2026, 1, 1, 11, minute, tzinfo=timezone.utc).timestamp()
        os.utime(reset / name, (stamp, stamp))

    producers = base.release_bound_producers(
        tmp_path, release_id=RELEASE, cluster_ids={CLUSTER}
    )
    assert producers[base.E2E001_CASE_ID]["started_at"] == "2026-01-01T10:01:00+00:00"
    assert producers[base.E2E001_CASE_ID]["operations"] == {
        "RESTART_WORKLOAD": ["wf-restart"],
        "STOP_WORKLOADS": ["wf-restart"],
    }
    assert producers["GF-REGIONAL-DESTR-010"]["started_at"] == (
        "2026-01-01T11:00:00+00:00"
    )
    assert producers["GF-REGIONAL-DESTR-010"]["operations"] == {
        "MARK_UNSCHEDULABLE": ["wf-reset"]
    }, "the plan's snapshot of E2E-001's workflow is not DESTR-010's proof"
