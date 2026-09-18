from __future__ import annotations

import copy
import json
from argparse import Namespace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from scripts.e2e.regional import run_preempt012_acceptance as runner


def control_report() -> dict:
    return {
        "executed_at": "2026-09-01T12:00:10+00:00",
        "completed_at": "2026-09-01T12:00:20+00:00",
        "clean": {
            "status": "SUPERSEDED",
            "new_adapter_calls": [],
            "predecessor_id": "clean-pred",
            "preempted_by": "clean-succ",
            "successor_id": "clean-succ",
            "successor_adapter_calls": ["RESTART_NODE"],
            "successor_inherited_step_indexes": [0, 1],
            "successor_completed_operations": ["MARK_UNSCHEDULABLE", "STOP_WORKLOADS"],
            "successor_inherited_from": ["clean-pred"],
        },
        "dirty": {
            "status": "SUPERSEDED",
            "handoff_after_claim": "dirty-pred",
            "predecessor_id": "dirty-pred",
        },
        "physical_operations_called": [],
        "remote_commands_for_audit_workflows": 0,
        "residual_objects": [],
    }


class PreemptionModel:
    def __init__(self, tmp_path: Path, monkeypatch) -> None:
        self.calls: list[str] = []
        self.predecessors: list[tuple[str, Path]] = []
        self.failure = ""
        self.preflight_errors: list[str] = []
        self.minutes = 60
        self.control = control_report()
        self.node = {
            "ready": "True",
            "unschedulable": False,
            "taints": [],
            "ownership_annotations": {},
        }
        self.host_state = {
            "services": {"kubelet": "active", "dcgm": "active"},
            "quiesce_state_files": [],
            "gpu_count": 8,
        }
        self.cycle = {
            "status": "COMPLETED",
            "quiesced_at": "2026-09-01T12:00:00+00:00",
            "restored_at": "2026-09-01T12:00:30+00:00",
        }
        self.settings = SimpleNamespace(
            gpu_kubeconfig=tmp_path / "empty-kubeconfig",
            gpu_context="fixture-context",
            namespace="fixture-namespace",
            cluster_id="fixture-cluster",
            environment=lambda: {"cluster": "fixture-cluster"},
        )
        self.settings.gpu_kubeconfig.touch()
        self.arguments = Namespace(run_dir=tmp_path, attempt=2)
        self.host_script = "/fixture/probe.py"
        self.reads = 0
        monkeypatch.setattr(runner, "HostProbeFixture", lambda _settings: self)
        monkeypatch.setattr(
            runner,
            "authorize_execution",
            lambda *a, **kw: datetime.now(timezone.utc)
            + timedelta(minutes=self.minutes),
        )
        monkeypatch.setattr(runner, "read_only_preflight", self.preflight)

    def preflight(
        self, *_args, predecessor_id: str, predecessor_path_value: Path, **_kwargs
    ) -> dict:
        self.predecessors.append((predecessor_id, predecessor_path_value))
        return {
            "errors": list(self.preflight_errors),
            "predecessor": {"valid": True},
            "cpu_blast": {"pods": ["cpu-one"]},
            "focused_tests": {"passed": True},
        }

    def evidence_identity(self) -> dict:
        return {"release_id": "fixture-release", "cluster_id": self.settings.cluster_id}

    def create(self) -> None:
        self.calls.append("create")

    def execute(self, command, *_args, **_kwargs) -> dict:
        self.calls.append(command)
        if command == "snapshot":
            value = copy.deepcopy(self.host_state)
            if self.failure == "baseline":
                value["quiesce_state_files"] = ["pre-existing.json"]
            return value
        if command == "arm":
            return {"timer_active": self.failure != "timer"}
        if command == "read":
            self.reads += 1
            if self.reads == 1:
                return {"status": "FAILED" if self.failure == "quiesce" else "QUIESCED"}
            value = dict(self.cycle)
            if self.failure == "cycle":
                value["status"] = "FAILED"
            return value
        if command == "cleanup":
            if self.failure == "host-cleanup":
                raise RuntimeError("host cleanup read failed")
            return {"timer_active_state": "inactive"}
        raise AssertionError(f"unexpected host probe command: {command}")

    def cleanup(self) -> dict:
        self.calls.append("remove")
        if self.failure == "probe-cleanup":
            raise RuntimeError("probe resource deletion failed")
        return {"pod": self.failure == "residual"}

    def node_snapshot(self, _node) -> dict:
        self.calls.append("node")
        if self.failure == "node":
            raise RuntimeError("node inventory is unavailable")
        return copy.deepcopy(self.node)

    def cpu_python(self, _script, *args, **kwargs) -> dict:
        assert kwargs["attempts"] == 1, "synthetic workflow mutations must not replay"
        if "--cleanup-only" in args:
            self.calls.append("audit-cleanup")
            if self.failure == "audit-cleanup":
                raise RuntimeError("cleanup acknowledgement lost")
            return {
                "residual_objects": ["remaining"]
                if self.failure == "audit-residual"
                else [],
                "residual_links": 0,
            }
        self.calls.append("audit")
        return (
            {"error": "ids already exist"} if self.failure == "audit" else self.control
        )

    def gpu_nodes(self) -> list:
        return [copy.deepcopy(self.node)]

    def provider_events(self, *_args) -> list:
        return []

    def cpu_blast_snapshot(self) -> dict:
        return {"pods": ["cpu-one"]}

    def run(self) -> tuple[int, dict]:
        predecessor_id, predecessor_file = runner.predecessor_path(
            self.arguments.run_dir, runner.CASE_ID, None
        )
        assert predecessor_id is not None and predecessor_file is not None, (
            "the lifecycle fixture must resolve the formal predecessor"
        )
        code = runner.execute_case(
            self.arguments,
            self,
            node="node-a",
            image="fixture/probe@sha256:" + "a" * 64,
            predecessor_id=predecessor_id,
            predecessor_path_value=predecessor_file,
            environment={},
        )
        path = (
            self.arguments.run_dir / "cases" / runner.CASE_ID / f"{runner.CASE_ID}.json"
        )
        return code, json.loads(path.read_text())
