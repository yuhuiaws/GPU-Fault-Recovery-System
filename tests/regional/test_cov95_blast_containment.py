from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import blast_acceptance_base as base
from scripts.e2e.regional.blast_acceptance_cases_1 import BlastCasesOne
from tests.regional._cov95_blast_support import containment_runner
from tests.regional._cov95_identity_support import offline_guard as offline_guard
from tests.regional.test_blast_acceptance_review import evidence


@pytest.mark.parametrize("defect", ["empty", "uid", "duplicate"])
def test_blast_cpu_node_comparison_requires_unambiguous_identity(defect: str) -> None:
    node = {"metadata": {"name": "cpu", "uid": "uid"}}
    document = {"items": [node]}
    if defect == "empty":
        document["items"] = []
    elif defect == "uid":
        node["metadata"]["uid"] = ""
    else:
        document["items"].append(copy.deepcopy(node))
    with pytest.raises(
        base.CheckError, match="empty or malformed|missing or duplicated"
    ):
        BlastCasesOne.node_security_snapshot(document)


@pytest.mark.parametrize(
    ("stamp", "fields", "expected"),
    [
        (None, {"f:spec": {"f:taints": {}}}, False),
        ("2000-01-01T00:00:00Z", {"f:spec": {"f:taints": {}}}, False),
        ("invalid", {}, True),
        ("2026-01-01T00:00:00Z", {"f:other": {}}, False),
        ("2026-01-01T00:00:00Z", {"f:spec": {"f:unschedulable": {}}}, True),
    ],
)
def test_managed_fields_use_time_and_security_fields_not_just_manager_name(
    stamp: str | None, fields: dict[str, Any], expected: bool
) -> None:
    entry = {"time": stamp, "fieldsV1": fields, "manager": "unit"}
    since = datetime(2025, 1, 1, tzinfo=timezone.utc)
    assert BlastCasesOne.managed_field_relevant(entry, since=since) is expected
    nodes = {"items": [{"metadata": {"name": "cpu", "managedFields": [entry]}}]}
    hits = BlastCasesOne.managed_field_writes(
        nodes, baseline_nodes={"items": []}, since=since
    )
    assert bool(hits) is expected


@pytest.mark.parametrize("defect", ["none", "inside", "missing-time"])
def test_blast001_counts_only_window_changes_and_performs_no_extra_action(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, defect: str
) -> None:
    runner, docs, events, reads = containment_runner(monkeypatch, tmp_path)
    card = json.loads((runner.e2e_dir / base.E2E001_EXECUTION_CARD).read_text())
    stamp = (
        card["maintenance_window"]["start"]
        if defect == "inside"
        else "2000-01-01T00:00:00Z"
    )
    when = None if defect == "missing-time" else stamp
    docs["nodes"]["items"][0]["metadata"]["managedFields"] = [
        {"manager": "unit", "time": stamp, "fieldsV1": {"f:spec": {"f:taints": {}}}},
        {"manager": "unit", "fieldsV1": {}},
    ]
    docs["jobs"]["items"] = [
        {"metadata": {"name": "other", "labels": {}, "creationTimestamp": when}},
        {
            "metadata": {
                "name": "unit-job",
                "namespace": "gpu-system",
                "labels": {"gpu-fault.io/test": "unit"},
                "creationTimestamp": when,
            }
        },
    ]
    docs["events"]["items"] = [
        {"reason": "Scheduled", "message": "ordinary event"},
        {"reason": "Evicted", "eventTime": when, "metadata": {"name": "unit-event"}},
        {
            "reason": "Other",
            "message": "TaintManagerEviction",
            "lastTimestamp": when,
            "metadata": {"name": "unit-event-2"},
        },
    ]
    before_events = list(events)
    assert runner.run() == (0 if defect == "none" else 1)
    assert events == before_events, (
        "BLAST001 must not execute another injection or cleanup action"
    )
    assert all(args[0] == "get" or "configmap" in args for args in reads), (
        "containment audit issued a command outside its read-only scope"
    )
    result = evidence(runner)
    assert result["verdict"] == ("PASS" if defect == "none" else "FAIL")
    analysis = json.loads((runner.run_dir / "BLAST-001-analysis.json").read_text())
    assert analysis["isolation_producers"] == [base.CONTAINMENT_CASE_ID], (
        "the real DESTR-001 producer proves MARK_UNSCHEDULABLE"
    )
    assert analysis["workload_operations"] == ["RESTART_WORKLOAD", "STOP_WORKLOADS"]
    assert result["checks"]["gpu_fault_jobs_created_in_window"] == (
        0 if defect == "none" else 1
    )
    assert result["checks"]["eviction_events_in_window"] == (
        0 if defect == "none" else 2
    )
    assert result["checks"]["relevant_node_writes_in_window"] == int(defect == "inside")


@pytest.mark.parametrize(
    "defect", ["directory", "baseline", "verdict", "window", "digest"]
)
def test_blast001_refuses_unbound_handoffs_before_any_further_action(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, defect: str
) -> None:
    runner, _docs, events, _reads = containment_runner(monkeypatch, tmp_path)
    card_path = runner.e2e_dir / base.E2E001_EXECUTION_CARD
    card = json.loads(card_path.read_text())
    if defect == "directory":
        runner.e2e_dir = tmp_path
    elif defect == "baseline":
        runner.trusted_cpu_baseline = tmp_path / "foreign"
    elif defect == "verdict":
        card["verdict"] = "FAIL"
    elif defect == "window":
        card["maintenance_window"]["start"] = card["maintenance_window"]["end"]
    else:
        card["state_sha256"] = "0" * 64
    base.write_json(card_path, card)
    before_events = list(events)
    assert runner.run() == 1
    assert evidence(runner)["verdict"] == "FAIL"
    assert events == before_events


def test_blast001_lists_a_producer_that_started_before_the_baseline_without_counting_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A PASS producer that started before the baseline snapshot cannot be
    judged against it: it is reported as producers_before_baseline and left
    out of the coverage -- the in-window E2E-001 and DESTR-001 still prove
    the required operations, so the audit passes on them alone."""
    runner, _docs, _events, _reads = containment_runner(monkeypatch, tmp_path)
    early = runner.root_run_dir / "cases" / "GF-REGIONAL-COLLECT-015"
    early.mkdir(parents=True)
    base.write_json(
        early / "GF-REGIONAL-COLLECT-015.json",
        {
            "case_id": "GF-REGIONAL-COLLECT-015",
            "verdict": "PASS",
            "release_id": "release-test",
            "cluster_id": "a",
            "errors": [],
            "started_at": "2000-01-01T00:00:00+00:00",
            "workflow": {
                "request_id": "early-workflow",
                "status": "SUCCEEDED",
                "step_executions": [
                    {"operation": "MARK_UNSCHEDULABLE", "status": "SUCCEEDED"}
                ],
            },
        },
    )
    assert runner.run() == 0
    assert evidence(runner)["verdict"] == "PASS"
    analysis = json.loads((runner.run_dir / "BLAST-001-analysis.json").read_text())
    assert analysis["producers_before_baseline"] == ["GF-REGIONAL-COLLECT-015"]
    assert "GF-REGIONAL-COLLECT-015" not in analysis["producers"]
    assert analysis["isolation_producers"] == [base.CONTAINMENT_CASE_ID]


@pytest.mark.parametrize("defect", ["not-object", "escaped-file", "node-uid"])
def test_containment_source_refuses_malformed_or_escaped_producer_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, defect: str
) -> None:
    runner, documents, events, _reads = containment_runner(monkeypatch, tmp_path)
    source_dir = runner.root_run_dir / "cases" / base.CONTAINMENT_CASE_ID
    if defect == "not-object":
        (source_dir / "host-after.json").write_text("[]", encoding="ascii")
    elif defect == "escaped-file":
        path = source_dir / "host-after.json"
        outside = tmp_path / "outside.json"
        path.rename(outside)
        path.symlink_to(outside)
    else:
        path = source_dir / f"{base.CONTAINMENT_CASE_ID}.json"
        value = json.loads(path.read_text())
        value["final_node"]["uid"] = "replacement"
        base.write_json(path, value)
    before_events = list(events)
    with pytest.raises(base.CheckError, match="not an object|escaped|node identity"):
        base.containment_source(
            runner.root_run_dir,
            predecessor=runner.predecessor,
            release_id="release-test",
            cluster_id="a",
            cpu_snapshot=BlastCasesOne.node_security_snapshot(documents["nodes"]),
        )
    assert events == before_events
