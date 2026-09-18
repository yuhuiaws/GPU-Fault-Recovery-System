from __future__ import annotations

import json
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import destr021_verdicts as verdicts
from scripts.e2e.regional import run_destr021_adversarial_node_metadata as case
from tests.regional._cov95_destr_branches import ready_runtime
from tests.regional._cov95_destr_warm import NOW, Clock, regional_settings

NODE = "node-a"
EFA_COUNT = 4


def observed_node(cordoned: bool, version: str) -> dict[str, Any]:
    return {
        "unschedulable": cordoned,
        "taint_keys": [verdicts.QUARANTINE_TAINT] if cordoned else [],
        "resource_version": version,
    }


def recovery_bundle(foreign: str) -> dict[str, Any]:
    return {
        "workflow": {
            "request_id": "workflow-owned",
            "status": "SUCCEEDED",
            "official_steps": [
                {
                    "operation": op,
                    "parameters": {"expected_count": EFA_COUNT}
                    if op == "RESTART_EFA_DEVICE_PLUGIN"
                    else {},
                }
                for op in verdicts.EXPECTED_STEPS
            ],
            "step_executions": [
                {
                    "step_index": i,
                    "operation": op,
                    "status": "SUCCEEDED",
                    "phase": "official",
                }
                for i, op in enumerate(verdicts.EXPECTED_STEPS)
            ],
        },
        "incident": {"incident_id": "incident-owned", "state": "RECOVERED"},
        "remote_commands": [
            {
                "operation": "MARK_UNSCHEDULABLE",
                "status": "SUCCEEDED",
                "result_details": {
                    "isolated_nodes": [NODE],
                    "node_baselines": {
                        NODE: {
                            "before": observed_node(False, "1"),
                            "after": observed_node(True, "2"),
                        }
                    },
                },
            },
            {
                "operation": "RESTART_EFA_DEVICE_PLUGIN",
                "status": "SUCCEEDED",
                "step": {"parameters": {"expected_count": EFA_COUNT}},
                "result_details": {
                    "node_results": {
                        NODE: {
                            "allocatable": EFA_COUNT,
                            "expected": EFA_COUNT,
                            "already_healthy": False,
                            "replacement_pod_uids": ["replacement-pod"],
                            "took_over_incident": foreign,
                        }
                    }
                },
            },
            {
                "operation": "RESTORE_SCHEDULING",
                "status": "SUCCEEDED",
                "result_details": {
                    "restored_nodes": [NODE],
                    "node_baselines": {
                        NODE: {
                            "before": observed_node(True, "3"),
                            "after": observed_node(False, "4"),
                        }
                    },
                },
            },
        ],
    }


class MetadataHarness:
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        site = tmp_path / "site.yaml"
        site.write_text("fake site\n", encoding="utf-8")
        self.settings = case.Settings(
            regional_settings(tmp_path),
            NODE,
            site,
            "example.test/probe",
            tmp_path / "predecessor.json",
            0.5,
            600,
        )
        self.clock = Clock()
        self.calls: list[tuple[str, Any]] = []
        self.failures: dict[str, BaseException] = {}
        self.advance_at: dict[str, float] = {}
        self.node = {
            "name": NODE,
            "uid": "node-uid",
            "resource_version": "1",
            "boot_id": "boot-1",
            "ready": "True",
            "unschedulable": False,
            "taints": [],
            "labels": {},
            "ownership_annotations": {},
            "annotations": {},
        }
        self.efa_count: Any = EFA_COUNT
        self.bound = True
        self.no_bound_device = False
        self.injection_attempted = False
        self.recovery_missing = False
        self.workflow_delays = 0
        self.incident_delays = 0
        self.workflow_missing = False
        self.restore_waiting = True
        self.queue = {"depth": 0, "fault_backlog_depth": 0}
        self.commands: Any = {"open_by_cluster": {}}
        self.quiet_queue = self.queue
        self.quiet_commands = self.commands
        self.predecessor_valid = True
        self.tests_pass = True
        self.residuals = {"pod": False}
        self.drop_patch = False
        self.events: list[dict[str, Any]] = []
        self.foreign = f"acceptance-foreign-{int(NOW.timestamp())}-a1"
        self.bundle = recovery_bundle(self.foreign)
        self.regional = MetadataRegional(self)
        self.warm = MetadataWarm(self)
        self.collector = MetadataCollector(self)
        self.writer = MetadataWriter(self)
        monkeypatch.setattr(case, "RegionalLiveFixture", lambda _s: self.regional)
        monkeypatch.setattr(case, "WarmSpareLiveFixture", lambda *_a: self.warm)
        monkeypatch.setattr(
            case, "CollectorAcceptanceFixture", lambda *_a, **_k: self.collector
        )
        monkeypatch.setattr(case, "AnnotationWriter", lambda *_a, **_k: self.writer)
        monkeypatch.setattr(
            case, "focused_tests", lambda _p: {"passed": self.tests_pass}
        )
        monkeypatch.setattr(
            case, "predecessor_evidence", lambda *_a: {"valid": self.predecessor_valid}
        )
        monkeypatch.setattr(case, "time", self.clock)
        monkeypatch.setattr(case, "datetime", self.clock)

    def call(self, name: str, detail: Any = None) -> None:
        self.calls.append((name, deepcopy(detail)))
        self.clock.sleep(self.advance_at.get(name, 0))
        if name in self.failures:
            raise self.failures[name]

    def plan(self, run_dir: Path) -> dict[str, Any]:
        path = run_dir / "cases" / case.CASE_ID
        path.mkdir(parents=True, exist_ok=True)
        preflight = case.read_only_preflight(self.settings, path)
        (path / "plan.json").write_text(
            json.dumps({"details": case.plan_details(self.settings, preflight)}),
            encoding="utf-8",
        )
        return preflight

    def execute(self, run_dir: Path, seconds: int = 3600) -> tuple[int, dict[str, Any]]:
        code = case.execute_case(
            self.settings, run_dir, 1, NOW + timedelta(seconds=seconds)
        )
        path = run_dir / "cases" / case.CASE_ID / f"{case.CASE_ID}.json"
        return code, json.loads(path.read_text(encoding="utf-8"))


class MetadataWarm:
    def __init__(self, harness: MetadataHarness) -> None:
        self.h = harness
        self.regional = harness.regional

    def node_snapshot(self, node: str, **kwargs: Any) -> dict[str, Any]:
        return deepcopy(self.h.node)


class MetadataRegional:
    def __init__(self, harness: MetadataHarness) -> None:
        self.h = harness
        self.settings = harness.settings.regional

    def node_snapshot(self, node: str) -> dict[str, Any]:
        self.h.call("node.snapshot", node)
        return deepcopy(self.h.node)

    def kubectl(self, plane: str, *args: str, **kwargs: Any) -> str:
        if args[0] == "get":
            self.h.call("node.document")
            return json.dumps(
                {
                    "metadata": {
                        "annotations": self.h.node["annotations"],
                        "resourceVersion": self.h.node["resource_version"],
                    },
                    "status": {
                        "allocatable": {verdicts.EFA_RESOURCE: self.h.efa_count}
                    },
                }
            )
        if args[0] == "patch":
            self.h.call("node.patch")
            patch = json.loads(args[args.index("-p") + 1])["metadata"]
            assert patch["uid"] == self.h.node["uid"], patch
            assert patch["resourceVersion"] == self.h.node["resource_version"], patch
            if not self.h.drop_patch:
                self.h.node["annotations"].update(patch.get("annotations", {}))
                self.h.node["resource_version"] = str(
                    int(self.h.node["resource_version"]) + 1
                )
            return ""
        raise AssertionError(f"unexpected fake metadata command: {args}")

    def business_workloads(self, node: str) -> list[dict[str, Any]]:
        return []

    def store_snapshot(self, **kwargs: Any) -> dict[str, Any]:
        quiet = "queue_attempts" in kwargs
        self.h.call("store.quiet" if quiet else "store.preflight")
        return {
            "release_id": "release-test",
            "agent": {"generation": 4},
            "queue": deepcopy(self.h.quiet_queue if quiet else self.h.queue),
            "remote_commands": deepcopy(
                self.h.quiet_commands if quiet else self.h.commands
            ),
        }

    def cpu_blast_snapshot(self) -> dict[str, Any]:
        return {"cpu": ["api-a"]}

    def runtime_identity(self) -> dict[str, Any]:
        return ready_runtime()

    def cpu_python(self, script: str, *args: str) -> dict[str, Any]:
        if script == case.base.LATEST_NODE_WORKFLOW:
            self.h.call("workflow.latest")
            if self.h.workflow_missing:
                return {"matches": []}
            bundle = deepcopy(self.h.bundle)
            if self.h.workflow_delays:
                self.h.workflow_delays -= 1
                bundle["workflow"]["status"] = "RUNNING"
                if self.h.restore_waiting:
                    execution = bundle["workflow"]["step_executions"][-1]
                    execution.update(
                        status="WAITING", details={"patch_conflict_retry": True}
                    )
            if self.h.incident_delays:
                self.h.incident_delays -= 1
                bundle["incident"]["state"] = "RECOVERING"
            return {"matches": [bundle]}
        if script == case.c017.REMOTE_COMMANDS:
            self.h.call("commands.read", args)
            if not self.h.recovery_missing:
                self.h.bound = True
            for key in verdicts.PLUGIN_RESTART_ANNOTATIONS:
                self.h.node["annotations"][key] = None
            return {"remote_commands": deepcopy(self.h.bundle["remote_commands"])}
        raise AssertionError("unexpected fake CPU metadata request")

    def provider_events(self, *args: Any) -> list[dict[str, Any]]:
        self.h.call("provider.events")
        return deepcopy(self.h.events)

    def verify_runtime_identity(self, identity: Any, **kwargs: Any) -> None:
        self.h.call("runtime.verify")
        assert identity == ready_runtime(), (
            "metadata case must preserve runtime identity"
        )


class MetadataCollector:
    def __init__(self, harness: MetadataHarness) -> None:
        self.h = harness

    def create(self) -> None:
        self.h.call("collector.create")

    def snapshot(self) -> dict[str, Any]:
        self.h.call("collector.snapshot")
        devices = [
            {"pci_bdf": f"0000:{i + 10:02x}:00.0", "driver": "efa"}
            for i in range(EFA_COUNT)
        ]
        if self.h.no_bound_device:
            devices = []
        return {
            "efa_inventory": {
                "devices": devices,
                "discovered_count": EFA_COUNT if self.h.bound else EFA_COUNT - 1,
            }
        }

    def execute(self, action: str, *args: str, **kwargs: Any) -> dict[str, Any]:
        self.h.call(f"collector.{action}", {"args": args, **kwargs})
        if action == "unbind-efa":
            self.h.injection_attempted = True
            self.h.bound = False
        if action == "restore-efa":
            self.h.bound = True
        return {"action": action, "ok": True}

    def cleanup(self) -> dict[str, bool]:
        self.h.call("collector.cleanup")
        return dict(self.h.residuals)


class MetadataWriter:
    def __init__(self, harness: MetadataHarness) -> None:
        self.h = harness
        self.started_at: float | None = None
        self.running = False
        self.stuck = False
        self.report_changes: dict[str, Any] = {}

    def start(self) -> None:
        self.h.call("writer.start")
        self.started_at = self.h.clock.monotonic()
        self.running = True
        self.h.node["annotations"][verdicts.TICK_ANNOTATION] = "tick"

    def stop(self) -> None:
        self.h.call("writer.stop")
        if not self.stuck:
            self.running = False

    def clear(self) -> None:
        self.h.call("writer.clear")
        if self.running:
            raise RuntimeError("fake writer still running")
        self.h.node["annotations"][verdicts.TICK_ANNOTATION] = None

    def report(self) -> dict[str, Any]:
        return {
            "patches": 400,
            "elapsed_seconds": 200.0,
            "conflicts": 3,
            "stopped": not self.running,
            "running": self.running,
            "cleared": True,
            "exceeded_max_seconds": False,
            **self.report_changes,
        }
