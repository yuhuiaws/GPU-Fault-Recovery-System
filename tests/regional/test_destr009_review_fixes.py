"""DESTR-009 ``execute_case`` evidence contracts, driven through fakes.

The runner is executed end to end against a spy fixture set; the assertions
read the evidence file it writes, not its source text.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import run_destr009_workload_restart as destr009
from scripts.e2e.regional.regional_live_fixture import RegionalLiveSettings

CANDIDATES = [
    {"name": "gpu-a", "uid": "uid-a"},
    {"name": "gpu-b", "uid": "uid-b"},
    {"name": "gpu-c", "uid": "uid-c"},
]
RUNTIME_IDENTITY = {"release_state": {"phase": "deployed"}}
WORKFLOW_STATE: dict[str, Any] = {
    "event": {"xid": 11},
    "decision": {"official_action": "RESTART_APP"},
    "workflow": {
        "request_id": "wf-9",
        "status": "SUCCEEDED",
        "official_steps": [
            {
                "operation": "FREEZE_EVIDENCE",
                "execution_owner": "gpu-fault-control-plane",
                "node_ids": ["gpu-a"],
            },
            {
                "operation": "STOP_WORKLOADS",
                "execution_owner": "gpu-fault-kubernetes-adapter",
            },
            {
                "operation": "RESTART_WORKLOAD",
                "execution_owner": "gpu-fault-kubernetes-adapter",
            },
        ],
        "step_executions": [
            {
                "operation": "STOP_WORKLOADS",
                "status": "SUCCEEDED",
                "adapter_operation_id": "remote/stop",
            },
            {
                "operation": "RESTART_WORKLOAD",
                "status": "SUCCEEDED",
                "adapter_operation_id": "remote/restart",
                "details": {"source_gpu_count": 24, "target_gpu_count": 24},
            },
        ],
    },
    "commands": [
        {"step": {"operation": "STOP_WORKLOADS"}, "status": "SUCCEEDED"},
        {"step": {"operation": "RESTART_WORKLOAD"}, "status": "SUCCEEDED"},
    ],
    "restart_budget": {"restart_count": 1, "budget": 1},
}


class FakeRegional:
    """The live fixture's read surface, recording what the runner asked for."""

    def __init__(self, _settings: Any) -> None:
        self.calls: list[str] = []

    def node_metadata(self, _node: str) -> dict[str, Any]:
        return {"product": "NVIDIA H100 80GB HBM3"}

    def verify_runtime_identity(self, *_args: Any, **_kwargs: Any) -> None:
        self.calls.append("verify_runtime_identity")

    def post_xid_event(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append("post_xid_event")
        return {"receipt": {"status": 200}, "payload": payload}

    def wait_for_workflow(self, **_kwargs: Any) -> dict[str, Any]:
        self.calls.append("wait_for_workflow")
        return json.loads(json.dumps(WORKFLOW_STATE))

    def provider_events(self, *_args: Any) -> list[dict[str, str]]:
        self.calls.append("provider_events")
        return []

    @staticmethod
    def provider_events_provisional(_ended_at: datetime) -> bool:
        return True

    def node_snapshot(self, name: str) -> dict[str, Any]:
        return {"name": name, "ownership_annotations": {}, "taints": []}

    def ready_pods(self, _plane: str, app: str) -> list[dict[str, Any]]:
        return [{"name": f"{app}-0"}]

    def kubectl(self, _plane: str, *arguments: str, **_kwargs: Any) -> str:
        if arguments[0] == "logs":
            return "2026-09-07 INFO gpu_fault.api served /healthz"
        return ""

    def cpu_blast_snapshot(self) -> dict[str, Any]:
        return {}


class FakeWorkload:
    name = "training-a"
    resource = "pytorchjob"

    def __init__(self, _regional: Any, _settings: Any) -> None:
        self.deleted = 0

    def submit(self) -> dict[str, Any]:
        return {"submitted": True}

    def wait_running(self, *, timeout_seconds: int) -> dict[str, Any]:
        return {
            "pods": [
                {"name": f"src-{i}", "uid": f"src-{i}", "node": item["name"]}
                for i, item in enumerate(CANDIDATES)
            ]
        }

    def wait_restarted(
        self, _uids: set[str], *, timeout_seconds: int
    ) -> dict[str, Any]:
        return {
            "pods": [
                {"name": f"dst-{i}", "uid": f"dst-{i}", "node": item["name"]}
                for i, item in enumerate(CANDIDATES)
            ]
        }

    def delete(self) -> None:
        self.deleted += 1


class FakePrewarm:
    def __init__(self, _regional: Any, *, case_id: str, run_id: str) -> None:
        self.case_id = case_id
        self.run_id = run_id

    def create(self, nodes: list[str]) -> dict[str, Any]:
        return {"skipped": [], "created": list(nodes)}

    def cached_nodes(self) -> list[str]:
        return [item["name"] for item in CANDIDATES]

    def cleanup(self) -> dict[str, bool]:
        return {}


def _settings(tmp_path: Path) -> destr009.Settings:
    manifest = tmp_path / "workload.yaml"
    manifest.write_text("kind: PyTorchJob\n", encoding="utf-8")
    site = tmp_path / "site.yaml"
    site.write_text("clusters: []\n", encoding="utf-8")
    cpu = tmp_path / "cpu.kubeconfig"
    gpu = tmp_path / "gpu.kubeconfig"
    cpu.write_text("apiVersion: v1\n", encoding="utf-8")
    gpu.write_text("apiVersion: v1\n", encoding="utf-8")
    return destr009.Settings(
        regional=RegionalLiveSettings(
            cpu_kubeconfig=cpu,
            gpu_kubeconfig=gpu,
            gpu_context="gpu-context",
            namespace="gpu-fault-system",
            cluster_id="cluster-a",
            region="us-west-2",
        ),
        site_file=site,
        manifest=manifest,
        job_id="job-a",
        attempt_id="job-a-a001",
        predecessor_path=tmp_path / "predecessor.json",
    )


def _drive_execute_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[int, dict[str, Any], list[dict[str, Any]]]:
    """Run ``execute_case`` against the fakes; return exit code, evidence, and
    the keyword arguments each ``read_only_preflight`` call received."""

    preflight_calls: list[dict[str, Any]] = []
    preflight = {
        "errors": [],
        "release_id": "release-a",
        "evidence_identity": {"release_id": "release-a", "cluster_id": "cluster-a"},
        "candidate_nodes": CANDIDATES,
        "store": {"profile": {"profile_version": "profile-a"}},
        "runtime_identity": RUNTIME_IDENTITY,
        "focused_tests": {"passed": True, "focused_tests_reused": True},
        "cpu_blast": {},
        "predecessor": {"valid": True},
    }

    def fake_preflight(
        _settings: Any, _case_dir: Path, **kwargs: Any
    ) -> dict[str, Any]:
        preflight_calls.append(kwargs)
        return preflight

    run_dir = tmp_path / "run-a"
    case_dir = run_dir / "cases" / destr009.CASE_ID
    case_dir.mkdir(parents=True)
    (case_dir / "plan.json").write_text(
        json.dumps(
            {
                "details": {
                    "preflight_identity": {
                        "release_id": "release-a",
                        "runtime_profile_version": "profile-a",
                        "candidate_node_uids": ["uid-a", "uid-b", "uid-c"],
                        "runtime_identity": RUNTIME_IDENTITY,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    observation = {
        "workload_phase": "RUNNING",
        "runtime_profile_version": "profile-a",
        "workload_ids": ["job-a"],
    }
    monkeypatch.setattr(destr009, "read_only_preflight", fake_preflight)
    monkeypatch.setattr(destr009, "RegionalLiveFixture", FakeRegional)
    monkeypatch.setattr(destr009, "ManagedWorkloadFixture", FakeWorkload)
    monkeypatch.setattr(destr009, "ImagePrewarmFixture", FakePrewarm)
    monkeypatch.setattr(destr009, "wait_observation", lambda *_a, **_k: observation)
    monkeypatch.setattr(
        destr009,
        "wait_for_cleanup_quiescence",
        lambda **_k: {"safe_to_delete": True, "entries": []},
    )

    exit_code = destr009.execute_case(
        _settings(tmp_path), run_dir, 1, datetime.now(timezone.utc) + timedelta(hours=1)
    )
    evidence = json.loads(
        (case_dir / f"{destr009.CASE_ID}.json").read_text(encoding="utf-8")
    )
    return exit_code, evidence, preflight_calls


def test_destr009_evidence_records_the_step_owners_of_the_matched_workflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exit_code, evidence, _calls = _drive_execute_case(tmp_path, monkeypatch)

    assert exit_code == 0, evidence
    assert evidence["verdict"] == "PASS", evidence
    # DESTR-012 group A reads exactly this projection from the DESTR-009
    # evidence, so the runner must persist what ``workflow_official_steps``
    # derives from the workflow it waited for.
    assert evidence["workflow_official_steps"] == destr009.workflow_official_steps(
        WORKFLOW_STATE
    )
    assert evidence["workflow_official_steps"] == [
        {"operation": "FREEZE_EVIDENCE", "execution_owner": "gpu-fault-control-plane"},
        {
            "operation": "STOP_WORKLOADS",
            "execution_owner": "gpu-fault-kubernetes-adapter",
        },
        {
            "operation": "RESTART_WORKLOAD",
            "execution_owner": "gpu-fault-kubernetes-adapter",
        },
    ]
    assert evidence["workflow_request_id"] == "wf-9"


def test_destr009_evidence_is_bound_to_the_preflight_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _exit_code, evidence, preflight_calls = _drive_execute_case(tmp_path, monkeypatch)

    # ``execute_case`` reuses the plan's focused pytest instead of paying for
    # it twice, and says so in the evidence.
    assert preflight_calls == [{"reuse_focused_tests": True}]
    assert evidence["focused_tests_reused"] is True
    # The result is bound to the release/cluster the preflight read against.
    assert evidence["release_id"] == "release-a"
    assert evidence["cluster_id"] == "cluster-a"
    # An empty provider read inside CloudTrail's delivery window is recorded
    # as provisional, never as proof of "no mutation".
    assert evidence["provider_events"] == []
    assert evidence["provider_events_provisional"] is True
