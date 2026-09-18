from __future__ import annotations

import json
import os
import subprocess
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from gpu_fault.orchestration.validated_restore import VALIDATED_RESTORE_OPERATIONS
from scripts.e2e.regional import run_destr001_gpu_reset as reset_case
from scripts.e2e.regional import run_destr023_idle_cluster_reset as idle_case
from scripts.e2e.regional import run_destr024_watcher_down_fail_closed as watcher_case
from tests.regional._cov95_destr_warm import NOW, Clock, regional_settings
from tests.regional._destructive_acceptance_support import reset_host_pair
from tests.regional.test_destr023_idle_cluster_reset import deployment, reset_state
from tests.regional.test_destr024_watcher_down_fail_closed import blocked_state


class IdleHarness:
    def __init__(
        self, module: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self.module = module
        self.settings = module.Settings(
            regional_settings(tmp_path),
            "node-a",
            "example.test/probe",
            tmp_path / "predecessor.json",
        )
        self.clock = Clock()
        self.calls: list[tuple[str, Any]] = []
        self.failures: dict[str, BaseException] = {}
        self.advance_at: dict[str, float] = {}
        self.node = {
            "name": "node-a",
            "uid": "node-uid",
            "ready": "True",
            "unschedulable": False,
            "taints": [],
            "ownership_annotations": {},
            "gpu_allocatable": 8,
        }
        self.final_node_changes: dict[str, Any] = {}
        self.state = {
            "release_id": "release-test",
            "agent": {
                "lifecycle_state": "ACTIVE",
                "generation": 3,
                "allowed_operations": sorted(reset_case.REQUIRED_AGENT_OPERATIONS),
            },
            "profile": {
                "profile_version": "profile-v1",
                "warnings": [],
                "capabilities": [
                    {
                        "capability": "gpuReset",
                        "mode": "OWN",
                        "owner": "gpu-fault-node-agent",
                        "adapter": "node-action",
                    }
                ],
            },
            "queue": {"depth": 0, "fault_backlog_depth": 0},
            "remote_commands": {"open_by_cluster": {}},
        }
        self.cpu = {"nodes": ["cpu-a"]}
        self.cpu_after = self.cpu
        self.workloads: list[dict[str, Any]] = []
        self.pods: list[dict[str, Any]] = []
        self.jobs: list[dict[str, Any]] = []
        self.coverage_changes: dict[str, Any] = {}
        self.predecessor_valid = True
        self.tests_pass = True
        self.injected = False
        self.watcher_replicas = 1
        self.watcher_ready = True
        self.watcher_stuck = False
        self.watcher_recovered = False
        self.events: list[dict[str, Any]] = []
        self.baseline, self.after = reset_host_pair(minimum=7, last=8)
        self.after = deepcopy(self.after)
        self.baseline.update(compute_clients=[], quiesce_states=[], kmsg_writable=True)
        self.after["kmsg_writable"] = True
        if module is watcher_case:
            self.workflow = blocked_state()
            self.after = deepcopy(self.baseline)
        else:
            self.workflow = reset_state()
            self.after["ledger"][-1].update(
                incident_id=self.workflow["incident"]["incident_id"],
                workflow_request_id=self.workflow["workflow"]["request_id"],
            )
        self.restored_workflow = {
            "status": "SUCCEEDED",
            "completed_operations": [op.value for op in VALIDATED_RESTORE_OPERATIONS],
            "step_executions": [
                {"operation": op.value, "status": "SUCCEEDED"}
                for op in VALIDATED_RESTORE_OPERATIONS
            ],
        }
        self.residuals = {"pod": False, "configmap": False}
        self.regional = IdleRegional(self)
        self.host = IdleHost(self)
        self.warm = IdleWarm(self)
        self.process = WatchdogProcess(self)
        for target in {module, idle_case}:
            monkeypatch.setattr(target, "time", self.clock)
            monkeypatch.setattr(target, "datetime", self.clock)
        monkeypatch.setattr(module, "RegionalLiveFixture", lambda _s: self.regional)
        monkeypatch.setattr(module, "HostProbeFixture", lambda _s: self.host)
        monkeypatch.setattr(module, "focused_tests", self.focused)
        monkeypatch.setattr(module, "predecessor_evidence", self.predecessor)
        monkeypatch.setattr(module, "record_focused_tests", self.record_focused)
        if module is watcher_case:
            monkeypatch.setattr(module, "WarmSpareLiveFixture", lambda *_a: self.warm)
            monkeypatch.setattr(
                module,
                "subprocess",
                SimpleNamespace(
                    Popen=self.popen,
                    STDOUT=subprocess.STDOUT,
                    TimeoutExpired=subprocess.TimeoutExpired,
                ),
            )
            monkeypatch.setattr(
                module, "os", SimpleNamespace(killpg=self.killpg, getenv=os.getenv)
            )

    def call(self, name: str, detail: Any = None) -> None:
        self.calls.append((name, deepcopy(detail)))
        self.clock.sleep(self.advance_at.get(name, 0))
        if name in self.failures:
            raise self.failures[name]

    def focused(self, _path: Path, *, reuse: bool = False) -> dict[str, Any]:
        self.call("focused", reuse)
        return {"passed": self.tests_pass, "focused_tests_reused": reuse}

    def predecessor(self, _path: Path, case_id: str, **identity: Any) -> dict[str, Any]:
        self.call("predecessor", {"case_id": case_id, **identity})
        return {"valid": self.predecessor_valid}

    def record_focused(self, details: dict[str, Any], result: dict[str, Any]) -> None:
        details["focused_tests"] = result

    def popen(self, argv: list[str], **kwargs: Any) -> Any:
        self.call(
            "watchdog.arm",
            {"argv": argv, "start_new_session": kwargs["start_new_session"]},
        )
        return self.process

    def killpg(self, pid: int, signum: int) -> None:
        self.call("watchdog.kill", {"pid": pid, "signal": signum})

    def plan(self, run_dir: Path) -> dict[str, Any]:
        case_dir = run_dir / "cases" / self.module.CASE_ID
        case_dir.mkdir(parents=True, exist_ok=True)
        result = self.module.read_only_preflight(self.settings, case_dir)
        details = self.module.plan_details(self.settings, result)
        (case_dir / "plan.json").write_text(
            json.dumps({"details": details}), encoding="utf-8"
        )
        return result

    def execute(self, run_dir: Path, seconds: int = 3600) -> tuple[int, dict[str, Any]]:
        code = self.module.execute_case(
            self.settings, run_dir, 1, NOW + timedelta(seconds=seconds)
        )
        path = run_dir / "cases" / self.module.CASE_ID / f"{self.module.CASE_ID}.json"
        return code, json.loads(path.read_text(encoding="utf-8"))


class WatchdogProcess:
    def __init__(self, harness: IdleHarness) -> None:
        self.h = harness
        self.pid = 456789
        self.returncode: int | None = None
        self.timeout_once = False

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: int) -> int:
        self.h.call("watchdog.wait", timeout)
        if self.timeout_once:
            self.timeout_once = False
            raise subprocess.TimeoutExpired("fake-watchdog", timeout)
        self.returncode = -15
        return self.returncode


class IdleRegional:
    def __init__(self, harness: IdleHarness) -> None:
        self.h = harness
        self.settings = harness.settings.regional

    def evidence_identity(self) -> dict[str, Any]:
        return {
            "release_id": "release-test",
            "cluster_id": "cluster-a",
            "region": "us-west-2",
        }

    def store_snapshot(self, **kwargs: Any) -> dict[str, Any]:
        self.h.call("store.snapshot", kwargs)
        return deepcopy(self.h.state)

    def node_snapshot(self, node: str) -> dict[str, Any]:
        self.h.call("node.snapshot", node)
        return deepcopy(self.h.node)

    def business_workloads(self, node: str) -> list[dict[str, Any]]:
        return deepcopy(self.h.workloads)

    def cpu_blast_snapshot(self) -> dict[str, Any]:
        self.h.call("cpu.snapshot")
        return deepcopy(self.h.cpu_after if self.h.injected else self.h.cpu)

    def cpu_python(self, script: str, cluster: str, node: str) -> dict[str, Any]:
        self.h.call("coverage", {"cluster": cluster, "node": node})
        now = self.h.clock.now()
        age = 0 if self.h.watcher_replicas else 700
        result = {
            "heartbeat_supported": True,
            "probed_at": now.isoformat(),
            "freshness_seconds": 600,
            "heartbeat": {
                "cluster_id": cluster,
                "observed_at": (now - timedelta(seconds=age)).isoformat(),
                "watched_pods": 0,
                "watched_attempts": 0,
            },
            "heartbeat_age_seconds": age,
            "observation_count": 0,
            "latest_observation_at": None,
            "latest_observation_age_seconds": None,
            "workload_state": "IDLE" if self.h.watcher_replicas else "UNKNOWN",
        }
        result.update(deepcopy(self.h.coverage_changes))
        return result

    def kubectl(self, plane: str, *args: str, **kwargs: Any) -> str:
        self.h.call("kubectl", {"plane": plane, "args": args, **kwargs})
        if args[0] == "scale":
            replicas = int(args[-1].split("=")[1])
            self.h.call("watcher.down" if replicas == 0 else "watcher.restore")
            self.h.watcher_replicas = replicas
            return ""
        if args[0] == "rollout":
            self.h.call("watcher.rollout")
            return ""
        if args[:2] == ("get", "deployment"):
            replicas = 1 if self.h.watcher_stuck else self.h.watcher_replicas
            value = deployment(
                replicas=replicas, ready=replicas if self.h.watcher_ready else 0
            )
            return json.dumps(value)
        if args[:2] == ("get", "jobs"):
            return json.dumps({"items": self.h.jobs})
        if args[:2] == ("get", "pods"):
            if f"{idle_case.MANAGED_LABEL}=true" in args:
                return json.dumps({"items": self.h.pods})
            replicas = 1 if self.h.watcher_stuck else self.h.watcher_replicas
            items = (
                [
                    {
                        "metadata": {"name": "watcher-pod", "uid": "watcher-pod-uid"},
                        "spec": {
                            "containers": [{"name": "watcher"}],
                            "nodeName": "node-a",
                        },
                        "status": {
                            "phase": "Running",
                            "conditions": [{"type": "Ready", "status": "True"}],
                            "containerStatuses": [{"name": "watcher", "ready": True}],
                        },
                    }
                ]
                if replicas
                else []
            )
            return json.dumps({"items": items})
        raise AssertionError(f"unhandled fake Kubernetes request: {args}")

    def wait_for_workflow(self, **kwargs: Any) -> dict[str, Any]:
        self.h.call("workflow.wait", kwargs)
        if self.h.module is watcher_case:
            self.h.node.update(
                unschedulable=True,
                taints=[{"key": "gpu-fault.io/quarantined", "effect": "NoSchedule"}],
                ownership_annotations={
                    "gpu-fault.io/incident-id": self.h.workflow["incident"][
                        "incident_id"
                    ]
                },
            )
        return deepcopy(self.h.workflow)

    def provider_events(self, *args: Any) -> list[dict[str, Any]]:
        self.h.call("provider.events")
        return deepcopy(self.h.events)

    def provider_events_provisional(self, *args: Any) -> bool:
        return True


class IdleHost:
    host_script = "/fake/probe.py"

    def __init__(self, harness: IdleHarness) -> None:
        self.h = harness

    def create(self) -> None:
        self.h.call("host.create")

    def execute(self, action: str, *args: str, **kwargs: Any) -> dict[str, Any]:
        self.h.call(f"host.{action}", {"args": args, **kwargs})
        if action == "snapshot":
            return deepcopy(self.h.after if args else self.h.baseline)
        if action == "write-xid46":
            self.h.injected = True
        return {"action": action, "ok": True}

    def cleanup(self) -> dict[str, bool]:
        self.h.call("host.cleanup")
        self.h.node.update(deepcopy(self.h.final_node_changes))
        return dict(self.h.residuals)


class IdleWarm:
    def __init__(self, harness: IdleHarness) -> None:
        self.h = harness

    def wait_agent_active(self, node: str) -> None:
        self.h.call("agent.active", node)

    def create_restore_workflow(self, **kwargs: Any) -> dict[str, Any]:
        self.h.call("restore.create", kwargs)
        return {"workflow_request_id": "restore-owned"}

    def wait_workflow_id(self, request_id: str, **kwargs: Any) -> dict[str, Any]:
        self.h.call("restore.wait", {"request_id": request_id, **kwargs})
        self.h.node.update(unschedulable=False, taints=[], ownership_annotations={})
        return deepcopy(self.h.restored_workflow)
