"""Refusals that only appear once a case is already under way.

HA-006 stops when the lease owner Pod vanished or the window closed right
before the kill; HA-005's entry point refuses a rollout scope that differs from
the approved plan; E2E-002's recovery verdict notices GPU node drift in the
workload-only isolation case; and the destructive reset helper fails closed
when the post-restart workload proof is not a populated mapping.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import run_collector_destructive as destructive
from scripts.e2e.regional import run_e2e002_multicluster_fault as e2e002
from scripts.e2e.regional import run_ha005_rollout_continuity as ha005
from scripts.e2e.regional import run_ha006_executor_takeover as ha006
from scripts.e2e.regional.regional_commands import RegionalFixtureError

DEADLINE = datetime(2099, 1, 1, tzinfo=timezone.utc)
T0 = datetime(2030, 6, 1, 12, 0, tzinfo=timezone.utc)


class FakeResources:
    def __init__(self, *_arguments: Any, owned: dict[str, Any] | None) -> None:
        self.owned_pod = owned
        self.records: dict[str, Any] = {}
        self.created: list[str] = []
        self.deleted: list[tuple[str, str]] = []

    def create(self, manifest: dict[str, Any]) -> None:
        self.created.append(manifest["kind"])

    def owned(self, kind: str, name: str) -> dict[str, Any] | None:
        return self.owned_pod

    def delete(self, kind: str, name: str) -> None:
        self.deleted.append((kind, name))


def arm_ha006_until_the_kill(
    monkeypatch: pytest.MonkeyPatch, resources: FakeResources, *, snapshot_delay: float
) -> dict[str, list[str]]:
    calls: dict[str, list[str]] = {"kills": [], "teardowns": [], "seeds": []}
    for name in ("database_residuals", "registry_residuals", "kubernetes_residuals"):
        monkeypatch.setattr(ha006, name, lambda: {"total": 0, "count": 0})
    monkeypatch.setattr(ha006, "OwnedProbeResources", lambda *a, **k: resources)
    monkeypatch.setattr(ha006, "register", lambda *a, **k: None)
    monkeypatch.setattr(ha006, "executor_identity", lambda **k: {"sa": "executor"})
    deployment = {
        "spec": {"template": {"spec": {"containers": [{"image": "img@sha256:ab"}]}}}
    }
    monkeypatch.setattr(
        ha006,
        "dataplane",
        lambda *a, **k: json.dumps(deployment) if a[0] == "get" else "",
    )
    monkeypatch.setattr(ha006, "run_manifest", lambda manifest, run_id: manifest)
    monkeypatch.setattr(
        ha006,
        "pod_manifest",
        lambda pod, *a: {"kind": "Pod", "metadata": {"name": pod}},
    )
    monkeypatch.setattr(ha006, "wait_file", lambda *a: None)
    monkeypatch.setattr(ha006, "read_state", lambda *a: {"ready": True})
    monkeypatch.setattr(ha006, "seed_command", lambda run_id: {"command_id": "cmd-1"})
    monkeypatch.setattr(
        ha006,
        "wait_first_owner",
        lambda seed: (
            {"lease_owner": ha006.PODS[0], "status": "LEASED"},
            {"notification_id": "n-1"},
            [{"status": "WAITING"}],
        ),
    )
    monkeypatch.setattr(
        ha006, "waiting_branch", lambda timeline: {"reclaimed_by_other_replica": True}
    )

    def command_snapshot(command_id: str) -> dict[str, Any]:
        time.sleep(snapshot_delay)
        return {"status": "LEASED", "lease_owner": ha006.PODS[0]}

    monkeypatch.setattr(ha006, "command_snapshot", command_snapshot)
    monkeypatch.setattr(
        ha006, "delete_pod", lambda *a, **k: calls["kills"].append("kill")
    )
    monkeypatch.setattr(
        ha006, "teardown", lambda **k: calls["teardowns"].append("teardown")
    )
    monkeypatch.setattr(
        ha006, "cleanup_seed", lambda *a: calls["seeds"].append("seed") or {}
    )
    return calls


def read_report(run_dir: Path, case_id: str) -> dict[str, Any]:
    return json.loads(
        (run_dir / "cases" / case_id / f"{case_id}.json").read_text(encoding="utf-8")
    )


def test_ha006_refuses_the_kill_when_the_lease_owner_pod_disappeared(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    resources = FakeResources(owned=None)
    calls = arm_ha006_until_the_kill(monkeypatch, resources, snapshot_delay=0)

    assert ha006.run_case(tmp_path, 1, DEADLINE) == 1
    report = read_report(tmp_path, ha006.CASE_ID)
    assert report["error"] == "CaseError: lease owner Pod disappeared before the kill"
    assert calls["kills"] == [], "no owner Pod means nothing may be force deleted"
    assert resources.created == ["ConfigMap", "Pod", "Pod"]
    assert resources.deleted == [("ConfigMap", ha006.CONFIGMAP)]
    assert calls["seeds"] == ["seed"] and calls["teardowns"] == ["teardown"]
    capsys.readouterr()


def test_ha006_refuses_the_kill_once_the_window_closed_during_setup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    resources = FakeResources(owned={"metadata": {"uid": "owner-uid"}})
    calls = arm_ha006_until_the_kill(monkeypatch, resources, snapshot_delay=0.3)
    window_end = datetime.now(timezone.utc) + timedelta(seconds=0.15)

    assert ha006.run_case(tmp_path, 1, window_end) == 1
    report = read_report(tmp_path, ha006.CASE_ID)
    assert report["error"] == (
        "CaseError: maintenance window ended before owner deletion"
    )
    assert calls["kills"] == [], "a closed window must never force delete the owner"
    assert (tmp_path / "cases" / ha006.CASE_ID / "owner-before-kill.json").is_file(), (
        "the owner state read before the refused kill is kept as evidence"
    )
    capsys.readouterr()


def test_ha005_entry_refuses_a_rollout_scope_that_differs_from_the_plan(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    case_dir = tmp_path / "cases" / ha005.CASE_ID
    case_dir.mkdir(parents=True)
    (case_dir / "plan.json").write_text(
        json.dumps({"details": {"chain": {}, "all_deployments": False}})
    )
    monkeypatch.setattr(ha005, "install_site_profile", lambda: None)
    monkeypatch.setattr(ha005.os, "umask", lambda mode: None)
    monkeypatch.setattr(ha005, "authorize_execution", lambda *a, **k: DEADLINE)
    monkeypatch.setattr(
        ha005, "chain_preflight", lambda *a: {"identity": {}, "predecessor": None}
    )
    monkeypatch.setattr(ha005, "require_chain", lambda *a: None)
    started: list[str] = []
    monkeypatch.setattr(ha005, "run_case", lambda *a, **k: started.append("run"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ha005",
            "--run-dir",
            str(tmp_path),
            "--all-deployments",
            "--execute",
            "--confirm",
            ha005.CONFIRMATION,
        ],
    )
    with pytest.raises(ha005.CaseError, match="rollout scope changed"):
        ha005.main()
    assert started == [], "a changed scope must never start the rollout"


def test_e2e002_isolation_verdict_notices_gpu_node_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("budget_denial_errors", "command_scope_errors", "notification_errors"):
        monkeypatch.setattr(e2e002, name, lambda *a, **k: [])
    monkeypatch.setattr(e2e002.workload_case, "workflow_errors", lambda *a, **k: [])
    monkeypatch.setattr(
        e2e002.workload_acceptance, "recovery_identity_errors", lambda *a, **k: []
    )
    baseline_a = [{"name": "gpu-a-1", "ready": "True"}]
    baseline_b = [{"name": "gpu-b-1", "ready": "True"}]
    fixtures = [
        SimpleNamespace(
            settings=SimpleNamespace(cluster_id="a"), gpu_nodes=lambda: baseline_a
        ),
        SimpleNamespace(
            settings=SimpleNamespace(cluster_id="b"),
            gpu_nodes=lambda: [{"name": "gpu-b-1", "ready": "False"}],
        ),
    ]
    states = [
        {
            "workflow": {"request_id": "wf-a"},
            "incident": {"incident_id": "inc-a"},
            "restart_budget": {"restart_count": 0, "cluster_id": "a", "job_id": "job"},
        },
        {
            "workflow": {"request_id": "wf-b"},
            "incident": {"incident_id": "inc-b"},
            "restart_budget": {"restart_count": 1, "cluster_id": "b", "job_id": "job"},
        },
    ]
    result: dict[str, Any] = {"errors": []}
    e2e002.verify_cluster_recoveries(
        settings=SimpleNamespace(job_id="job", attempt_id="job-a001"),  # type: ignore[arg-type]
        fixtures=fixtures,  # type: ignore[arg-type]
        cluster_ids=["a", "b"],
        sources=[{"pods": [{"uid": "u-a", "node": "n-a"}]}, {"pods": []}],
        states=states,
        targets_after=[{"pods": []}, {"pods": []}],
        payloads=[
            {"node_id": "n-a", "record_id": "r-a"},
            {"node_id": "n-b", "record_id": "r-b"},
        ],
        injection_starts={0: T0, 1: T0},
        preflight={
            "executor_identities": {},
            "registrations": [],
            "nodes_a": baseline_a,
            "nodes_b": baseline_b,
        },
        result=result,
        expect_a_budget_denial=True,
    )
    assert result["errors"] == [
        "GPU node state changed in the workload-only isolation case"
    ]


def test_single_reset_requires_a_populated_post_restart_workload_proof(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    baseline = {
        "gpu_inventory": [{"pci_bdf": "0000:3a:00.0"}],
        "boot_id": "boot-a",
        "ledger": [],
    }
    host_calls: list[str] = []
    host = SimpleNamespace(
        host_script="/opt/probe.sh",
        execute=lambda action, *a, **k: host_calls.append(action) or dict(baseline),
    )
    collector = SimpleNamespace(node="node-a", execute=lambda *a, **k: {"records": []})
    monkeypatch.setattr(
        destructive, "wait_xid_workflow", lambda *a, **k: {"workflow": {"status": "X"}}
    )
    settings = destructive.Settings(
        regional=SimpleNamespace(namespace="gpu-fault-system"),  # type: ignore[arg-type]
        case_id="GF-REGIONAL-COLLECT-013",
        node="node-a",
        second_node=None,
        host_probe_image="img@sha256:" + "a" * 64,
        hyperpod_cluster="hp-unit",
        executor_role_arn="arn:aws:iam::123456789012:role/unit",
        site_file=None,
        predecessor_path=tmp_path / "predecessor.json",
    )
    cleanup = destructive.CaseCleanup()
    with pytest.raises(RegionalFixtureError, match="post-restart workload proof"):
        destructive.run_single_reset(
            settings,
            SimpleNamespace(),  # type: ignore[arg-type]
            host,  # type: ignore[arg-type]
            tmp_path,
            collector=collector,  # type: ignore[arg-type]
            cleanup=cleanup,
            xid=109,
            marker="c013-unit",
            run_id="collect013-1",
            observe_post_restart_workload=lambda state: {},
        )
    assert "write-xid" in host_calls, "the proof is judged after the injection"
    assert not (tmp_path / "post-restart-workload.json").exists(), (
        "a refused proof is never written as evidence"
    )
