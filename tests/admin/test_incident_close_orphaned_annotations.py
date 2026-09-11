"""``--close-quarantined`` and orphaned isolation annotations.

A node whose ``gpu-fault.io/quarantined`` taint an operator released by hand
keeps the incident's isolation annotations. Live 2026-09-11 that orphaned the
incident between the levers: ``submit-remediation --disposition restore`` found
no isolation to restore, ``--close-quarantined`` refused on the annotations.
The evidence pass now strips the incident's own leftover annotations (fenced
on the node's resourceVersion) before the close, and a dry run says what it
would strip without touching the node. Split from ``test_incident_close.py``
to keep that file under the size ratchet.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.adapters.common import quarantine_taint_value
from gpu_fault.admin import incident_close
from gpu_fault.admin import workflow_reconcile as reconcile


def _gpu_site(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        release_config={
            "site_name": "staging",
            "aws_region": "us-west-2",
            "cpu_eks_arn": "arn:aws:eks:us-west-2:123456789012:cluster/cpu-control",
            "namespace": "gpu-fault",
            "cpu_kubeconfig": str(tmp_path / "cpu.kubeconfig"),
            "gpu_kubeconfig": str(tmp_path / "gpu.kubeconfig"),
            "clusters": [{"cluster_id": "gpu-a", "context": "gpu-a-context"}],
        },
        environment={},
        source_sha256="a" * 64,
    )


def _node(
    name: str,
    *,
    unschedulable: bool = False,
    taint_value: str | None = None,
    annotations: dict[str, str] | None = None,
) -> dict[str, Any]:
    taints = (
        [
            {
                "key": reconcile.QUARANTINE_TAINT,
                "value": taint_value,
                "effect": "NoSchedule",
            }
        ]
        if taint_value is not None
        else []
    )
    return {
        "metadata": {"name": name, "annotations": dict(annotations or {})},
        "spec": {"unschedulable": unschedulable, "taints": taints},
    }


def _quarantined_pending(incident_id: str, *node_ids: str) -> dict[str, Any]:
    """What the first Pod pass reports for a QUARANTINED incident."""

    return {
        "incident_id": incident_id,
        "outcome": "refused",
        "state": "QUARANTINED",
        "reason": f"incident {incident_id} is QUARANTINED; only an ESCALATED ...",
        "open_workflow_id": None,
        "cluster_id": "gpu-a",
        "node_ids": list(node_ids),
        "evidence_required": True,
        "isolation_reasons": [],
    }


def _two_pass_runner(first: dict[str, Any], calls: list[dict[str, Any]]):
    """A Pod double: the first call answers ``first``; the evidence pass judges
    each incident from the evidence it is handed, as the service would (cordon
    and the incident's own taint refuse; annotations are the admin layer's
    concern here and pass through untouched)."""

    def run(_site: Any, payload: dict[str, Any], *, script: str) -> dict[str, Any]:
        calls.append({"payload": payload, "script": script})
        if "evidence" not in payload:
            return {"mode": "incident-close", "dry_run": payload["dry_run"], **first}
        results = []
        for incident_id in payload["incident_ids"]:
            evidence = payload["evidence"][incident_id]
            own = quarantine_taint_value(incident_id)
            blocked = [
                item
                for item in evidence
                if item["unschedulable"] or item["quarantine_taint_value"] == own
            ]
            if blocked:
                results.append(
                    {
                        "incident_id": incident_id,
                        "outcome": "refused",
                        "state": "QUARANTINED",
                        "reason": f"node {blocked[0]['node_id']} still isolated",
                    }
                )
                continue
            results.append(
                {
                    "incident_id": incident_id,
                    "outcome": "would-close" if payload["dry_run"] else "closed",
                    "state": "QUARANTINED" if payload["dry_run"] else "RECOVERED",
                    "isolation_nodes": sorted(item["node_id"] for item in evidence),
                }
            )
        return {
            "mode": "incident-close",
            "dry_run": payload["dry_run"],
            "results": results,
        }

    return run


def _orphaned_node(incident_id: str, name: str = "node-a") -> dict[str, Any]:
    node = _node(
        name,
        annotations={
            "gpu-fault.io/incident-id": incident_id,
            "gpu-fault.io/fencing-token": "1",
            "gpu-fault.io/previous-unschedulable": "false",
        },
    )
    node["metadata"]["resourceVersion"] = "4242"
    return node


def _kubectl_strip_double(node: dict[str, Any], calls: list[list[str]]):
    """``kubectl`` that answers the annotated node until it sees the patch,
    and the clean node afterwards -- what the cluster does."""

    state = {"stripped": False}

    def run(args, **_kwargs):
        calls.append(list(args))
        if "patch" in args:
            state["stripped"] = True
            return SimpleNamespace(
                returncode=0, stdout="node/node-a patched", stderr=""
            )
        current = json.loads(json.dumps(node))
        if state["stripped"]:
            current["metadata"]["annotations"] = {}
        return SimpleNamespace(
            returncode=0, stdout=json.dumps({"items": [current]}), stderr=""
        )

    return run


def test_orphaned_isolation_nodes_need_the_incident_annotation_without_its_taint():
    own = quarantine_taint_value("inc-q1")
    clean = {
        "node_id": "n",
        "exists": True,
        "unschedulable": False,
        "quarantine_taint_value": None,
        "isolation_annotations": {"gpu-fault.io/incident-id": "inc-q1"},
    }
    assert incident_close.orphaned_isolation_nodes("inc-q1", [clean]) == ["n"]
    for blocked in (
        {**clean, "unschedulable": True},
        {**clean, "quarantine_taint_value": own},
        {**clean, "quarantine_taint_value": "inc-q1"},
        {**clean, "isolation_annotations": {"gpu-fault.io/incident-id": "inc-q2"}},
        {**clean, "isolation_annotations": {}},
        {**clean, "exists": False},
    ):
        assert incident_close.orphaned_isolation_nodes("inc-q1", [blocked]) == [], (
            blocked
        )


def test_close_quarantined_dry_run_judges_orphaned_annotations_as_stripped_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []
    kubectl: list[list[str]] = []
    monkeypatch.setattr(
        incident_close,
        "_run_reconcile",
        _two_pass_runner(
            {
                "discovered_total": 1,
                "discovered_cluster_ids": ["gpu-a"],
                "discovery_limit_reached": [],
                "discovered_incident_ids": ["inc-q1"],
                "results": [_quarantined_pending("inc-q1", "node-a")],
            },
            calls,
        ),
    )
    monkeypatch.setattr(
        reconcile.subprocess,
        "run",
        _kubectl_strip_double(_orphaned_node("inc-q1"), kubectl),
    )

    result = incident_close.run_incident_close(
        _gpu_site(tmp_path),
        tmp_path,
        selector=incident_close.quarantined_selector(),
        reason="",
        reference=None,
        dry_run=True,
    )

    assert all("patch" not in call for call in kubectl), "a dry run touches no node"
    assert calls[1]["payload"]["evidence"]["inc-q1"] == [
        {
            "node_id": "node-a",
            "exists": True,
            "unschedulable": False,
            "quarantine_taint_value": None,
            "isolation_annotations": {},
        }
    ], "the Pod judges the node as it would stand after the strip"
    assert result["would_strip_isolation_nodes"] == {"inc-q1": ["node-a"]}
    assert incident_close.result_lines(result) == [
        "discovered 1 QUARANTINED incident(s) across 1 cluster(s): gpu-a",
        "inc-q1: would-close (isolation absent on node-a; would strip orphaned "
        "isolation annotations on node-a)",
    ]


def test_close_quarantined_strips_orphaned_annotations_with_a_resource_version_fence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []
    kubectl: list[list[str]] = []
    monkeypatch.setattr(
        incident_close,
        "_run_reconcile",
        _two_pass_runner(
            {
                "discovered_total": 1,
                "discovered_cluster_ids": ["gpu-a"],
                "discovery_limit_reached": [],
                "discovered_incident_ids": ["inc-q1"],
                "results": [_quarantined_pending("inc-q1", "node-a")],
            },
            calls,
        ),
    )
    monkeypatch.setattr(
        reconcile.subprocess,
        "run",
        _kubectl_strip_double(_orphaned_node("inc-q1"), kubectl),
    )

    result = incident_close.run_incident_close(
        _gpu_site(tmp_path),
        tmp_path,
        selector=incident_close.quarantined_selector(),
        reason="taint released by hand after the node was repaired",
        reference="CHG-3",
        dry_run=False,
    )

    [patch] = [call for call in kubectl if "patch" in call]
    assert patch[:5] == [
        "kubectl",
        "--kubeconfig",
        str(tmp_path / "gpu.kubeconfig"),
        "--context",
        "gpu-a-context",
    ]
    assert patch[5:11] == ["patch", "node", "node-a", "--type", "merge", "-p"]
    assert json.loads(patch[11]) == {
        "metadata": {
            "resourceVersion": "4242",
            "annotations": {
                "gpu-fault.io/incident-id": None,
                "gpu-fault.io/fencing-token": None,
                "gpu-fault.io/previous-unschedulable": None,
            },
        }
    }, "the patch removes exactly the isolation record, fenced on resourceVersion"
    assert (
        calls[1]["payload"]["evidence"]["inc-q1"][0]["isolation_annotations"] == {}
    ), "the Pod is handed the node as re-read after the strip"
    assert incident_close.result_lines(result) == [
        "discovered 1 QUARANTINED incident(s) across 1 cluster(s): gpu-a",
        "inc-q1: closed (isolation absent on node-a; orphaned isolation annotations "
        "stripped on node-a)",
    ]
    assert result["closed_incident_ids"] == ["inc-q1"]
    history = tmp_path / incident_close.INCIDENT_CLOSE_HISTORY_PATH
    [archived] = sorted(history.glob("*.json"))
    document = json.loads(archived.read_text(encoding="utf-8"))
    assert document["stripped_isolation_nodes"] == {"inc-q1": ["node-a"]}


def test_close_quarantined_leaves_another_incidents_annotations_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []
    kubectl: list[list[str]] = []
    monkeypatch.setattr(
        incident_close,
        "_run_reconcile",
        _two_pass_runner(
            {
                "discovered_total": 1,
                "discovered_cluster_ids": ["gpu-a"],
                "discovery_limit_reached": [],
                "discovered_incident_ids": ["inc-q1"],
                "results": [_quarantined_pending("inc-q1", "node-a")],
            },
            calls,
        ),
    )
    monkeypatch.setattr(
        reconcile.subprocess,
        "run",
        _kubectl_strip_double(_orphaned_node("inc-other"), kubectl),
    )

    result = incident_close.run_incident_close(
        _gpu_site(tmp_path),
        tmp_path,
        selector=incident_close.quarantined_selector(),
        reason="cleanup",
        reference="CHG-4",
        dry_run=False,
    )

    assert all("patch" not in call for call in kubectl), (
        "annotations naming another incident are that incident's business"
    )
    assert calls[1]["payload"]["evidence"]["inc-q1"][0]["isolation_annotations"] == {
        "gpu-fault.io/incident-id": "inc-other",
        "gpu-fault.io/fencing-token": "1",
        "gpu-fault.io/previous-unschedulable": "false",
    }
    assert result["stripped_isolation_nodes"] == {}
