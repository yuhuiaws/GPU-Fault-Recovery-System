from __future__ import annotations

import json
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import pytest

from scripts.e2e.regional import run_destr002_hyperpod_reboot as reboot
from scripts.e2e.regional import run_destr009_workload_restart as workload_case
from scripts.e2e.regional import run_destr010_fabric_manager_restart as fabric
from tests.regional import test_destr002_review_fixes as reboot_data
from tests.regional import test_destr009_review_fixes as workload_data
from tests.regional._cov95_destr_branches import ready_runtime
from tests.regional._cov95_destr_warm import NOW, Clock, regional_settings


class Boundary:
    def __init__(self, harness: ActionHarness, label: str, target: Any) -> None:
        self.h = harness
        self.label = label
        self.target = target

    def __getattr__(self, name: str) -> Any:
        method = getattr(self.target, name)
        if not callable(method):
            return method

        def invoke(*args: Any, **kwargs: Any) -> Any:
            key = f"{self.label}.{name}"
            self.h.calls.append((key, args, kwargs))
            self.h.clock.sleep(self.h.advance_at.get(key, 0))
            if key in self.h.failures:
                raise self.h.failures[key]
            result = method(*args, **kwargs)
            transform = self.h.transforms.get(key)
            return transform(result, args, kwargs) if transform else result

        return invoke


class ActionHarness:
    def __init__(
        self, module: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self.module = module
        regional = regional_settings(tmp_path)
        self.clock = Clock()
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.failures: dict[str, BaseException] = {}
        self.advance_at: dict[str, float] = {}
        self.transforms: dict[str, Callable[..., Any]] = {}
        self.node = {
            "name": "node-a",
            "uid": "uid-node",
            "boot_id": "boot-1",
            "ready": "True",
            "unschedulable": False,
            "taints": [],
            "ownership_annotations": {},
            "gpu_allocatable": "8",
        }
        self.preflight = {
            "errors": [],
            "release_id": "release-a",
            "evidence_identity": {"release_id": "release-a", "cluster_id": "cluster-a"},
            "node": deepcopy(self.node),
            "store": {
                "agent": {
                    "generation": 3,
                    "artifact_sha256": "artifact-a",
                    "boot_id": "agent-boot-1",
                },
                "profile": {"profile_version": "profile-a"},
            },
            "focused_tests": {"passed": True, "focused_tests_reused": True},
            "predecessor": {"valid": True},
            "cpu_blast": {},
        }
        if module is reboot:
            self.settings = reboot.Settings(
                regional,
                "node-a",
                "example.test/probe",
                "hp-a",
                "arn:aws:iam::123456789012:role/executor-a",
                tmp_path / "previous.json",
            )
            self.regional = Boundary(
                self, "regional", reboot_data.SpyRegional(regional)
            )
            self.host = Boundary(self, "host", reboot_data.FakeHost(regional))
            self.preflight["provider_preflight"] = {
                "positive": {"targets": [{"instance_id": "i-0123"}]}
            }
        elif module is workload_case:
            site = tmp_path / "site.yaml"
            site.write_text("fake site\n", encoding="utf-8")
            self.settings = workload_case.Settings(
                regional,
                site,
                workload_case.DEFAULT_MANIFEST,
                "job-owned",
                "job-owned-a001",
                tmp_path / "previous.json",
            )
            self.preflight.update(
                candidate_nodes=deepcopy(workload_data.CANDIDATES),
                runtime_identity=ready_runtime(),
            )
            self.state = deepcopy(workload_data.WORKFLOW_STATE)
            self.regional = Boundary(self, "regional", WorkloadRegional(self))
            self.workload = Boundary(
                self,
                "workload",
                workload_data.FakeWorkload(self.regional, self.settings),
            )
            self.prewarm = Boundary(
                self,
                "prewarm",
                workload_data.FakePrewarm(
                    self.regional, case_id=module.CASE_ID, run_id="unit"
                ),
            )
            monkeypatch.setattr(
                module, "ManagedWorkloadFixture", lambda *_a: self.workload
            )
            monkeypatch.setattr(
                module, "ImagePrewarmFixture", lambda *_a, **_k: self.prewarm
            )
        else:
            self.settings = fabric.Settings(regional, "node-a", "example.test/probe")
            self.preflight["fabric"] = {"companion_window_seconds": 60}
            self.state = {
                "agent": {"generation": 3},
                "event": {"xid": 45, "evidence_ref": "kmsg://node-a/1"},
                "decision": {
                    "official_action": "RESTART_FM",
                    "disposition": "EXECUTABLE",
                },
                "workflow": {
                    "request_id": "workflow-fabric",
                    "status": "SUCCEEDED",
                    "official_action": "RESTART_FM",
                    "official_steps": [
                        {"operation": op, "execution_owner": owner}
                        for op, owner in zip(
                            fabric.EXPECTED_STEPS, fabric.EXPECTED_OWNERS, strict=True
                        )
                    ],
                    "completed_operations": list(fabric.EXPECTED_STEPS),
                },
                "commands": [
                    {
                        "command_id": "command-fabric",
                        "status": "SUCCEEDED",
                        "idempotency_key": "key-fabric",
                    }
                ],
                "notifications": [
                    {
                        "notification_id": "notification-fabric",
                        "category": "ACTION_COMPLETED",
                        "subject": "Fabric Manager restarted",
                        "status": "SENT",
                    }
                ],
            }
            self.regional = Boundary(self, "regional", FabricRegional(self))
            self.host = Boundary(self, "host", FabricHost(self))
        monkeypatch.setattr(
            module, "read_only_preflight", lambda *_a, **_k: deepcopy(self.preflight)
        )
        monkeypatch.setattr(
            module, "record_focused_tests", lambda d, r: d.update(focused_tests=r)
        )
        monkeypatch.setattr(module, "RegionalLiveFixture", lambda _s: self.regional)
        if module is not workload_case:
            monkeypatch.setattr(module, "HostProbeFixture", lambda _s: self.host)
        monkeypatch.setattr(module, "time", self.clock)
        monkeypatch.setattr(module, "datetime", self.clock)

    def plan(self, run_dir: Path) -> None:
        path = run_dir / "cases" / self.module.CASE_ID
        path.mkdir(parents=True, exist_ok=True)
        details = self.module.plan_details(self.settings, deepcopy(self.preflight))
        (path / "plan.json").write_text(
            json.dumps({"details": details}), encoding="utf-8"
        )

    def execute(self, run_dir: Path, seconds: int = 7200) -> tuple[int, dict[str, Any]]:
        code = self.module.execute_case(
            self.settings, run_dir, 1, NOW + timedelta(seconds=seconds)
        )
        path = run_dir / "cases" / self.module.CASE_ID / f"{self.module.CASE_ID}.json"
        return code, json.loads(path.read_text(encoding="utf-8"))


class WorkloadRegional(workload_data.FakeRegional):
    def __init__(self, harness: ActionHarness) -> None:
        super().__init__(harness.settings.regional)
        self.h = harness

    def store_snapshot(self, **kwargs: Any) -> dict[str, Any]:
        if kwargs.get("marker"):
            return {
                **deepcopy(self.h.state),
                "observations": [{"workload_phase": "STOPPED"}],
            }
        return {
            "observations": [
                {
                    "workload_phase": "RUNNING",
                    "runtime_profile_version": "profile-a",
                    "workload_ids": ["training/pytorchjob/job-owned"],
                    "containers": [
                        {"gpu_count": 24, "gpu_uuids": [f"GPU-{i}" for i in range(24)]}
                    ],
                }
            ]
        }

    def wait_for_workflow(self, **kwargs: Any) -> dict[str, Any]:
        return deepcopy(self.h.state)


class FabricRegional:
    def __init__(self, harness: ActionHarness) -> None:
        self.h = harness
        self.settings = harness.settings.regional

    def cpu_python(self, script: str, *args: str) -> dict[str, Any]:
        return {
            "companion_window_seconds": 60,
            "recent_xid_events": [],
            "active_workflow_incidents": [],
            "evidence": [{"record_id": "owned-evidence"}],
        }

    def executor_python(self, *args: Any, **kwargs: Any) -> dict[str, str]:
        return {"status": "SUCCEEDED"}

    def wait_for_workflow(self, **kwargs: Any) -> dict[str, Any]:
        return deepcopy(self.h.state)

    def store_snapshot(self, **kwargs: Any) -> dict[str, Any]:
        return deepcopy(self.h.state)

    def node_snapshot(self, node: str) -> dict[str, Any]:
        return deepcopy(self.h.node)

    def business_workloads(self, node: str) -> list[dict[str, Any]]:
        return []

    def provider_events(self, *args: Any) -> list[dict[str, Any]]:
        return []

    def provider_events_provisional(self, *args: Any) -> bool:
        return True


class FabricHost:
    def __init__(self, harness: ActionHarness) -> None:
        self.h = harness

    def create(self) -> None:
        pass

    def execute(self, action: str, *args: str, **kwargs: Any) -> dict[str, Any]:
        if action != "snapshot":
            return {"ok": True}
        after = bool(args)
        return {
            "kmsg_writable": True,
            "compute_clients": [],
            "gpu_pci_bdf": "0000:01:00.0",
            "fabric_manager": {
                "ActiveState": "active",
                "MainPID": "200" if after else "100",
                "InvocationID": "new" if after else "old",
            },
            "ledger": [{"command_id": "key-fabric/node-a/agent-3"}] if after else [],
            "gpu_fault_timers": [],
            "journal": {"started_count": 1 if after else 0},
        }

    def cleanup(self) -> dict[str, bool]:
        return {}
