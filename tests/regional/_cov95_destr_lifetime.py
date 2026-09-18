from __future__ import annotations

import json
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional import control_plane_env_window as window
from scripts.e2e.regional import destr018_verdicts as verdicts
from scripts.e2e.regional import run_destr018_lifetime_deadline as case
from tests.regional import test_destr018_lifetime_deadline as data
from tests.regional._cov95_destr_branches import ready_runtime
from tests.regional._cov95_destr_warm import NOW, Clock, regional_settings


class LifetimeHarness:
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.settings = case.Settings(
            regional_settings(tmp_path),
            data.NODE,
            "example.test/probe",
            tmp_path / "predecessor.json",
            verdicts.LIFETIME_SECONDS,
            verdicts.EXECUTION_TIMEOUT_SECONDS,
            verdicts.STEP_TIMEOUT_SECONDS,
            verdicts.MANAGED_RECOVERY_SECONDS,
            verdicts.STEP_WARNING_SECONDS,
            verdicts.LEASE_DURATION_SECONDS,
            1200,
        )
        self.clock = Clock()
        self.calls: list[tuple[str, Any]] = []
        self.failures: dict[str, BaseException] = {}
        self.advance_at: dict[str, float] = {}
        self.node = {
            "name": data.NODE,
            "uid": "node-uid",
            "ready": "True",
            "unschedulable": False,
            "taints": [],
            "ownership_annotations": {},
            "gpu_allocatable": 8,
        }
        self.store = {
            "release_id": "release-test",
            "agent": {
                "lifecycle_state": "ACTIVE",
                "generation": 3,
                "allowed_operations": list(case.REQUIRED_AGENT_OPERATIONS),
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
        self.open_incidents: list[dict[str, Any]] = []
        self.runtime = ready_runtime()
        self.runtime["deployments"]["cpu"][window.DEPLOYMENT]["template_sha256"] = (
            "original"
        )
        self.survey_data = {
            "deployment": {"variables": {}},
            "replicas": [
                {
                    "pod": "worker",
                    "values": {
                        case.POLL_INTERVAL_VARIABLE: "5",
                        case.STEP_TIMEOUT_VARIABLE: "600",
                    },
                }
            ],
        }
        self.predecessor_valid = True
        self.tests_pass = True
        self.injected = False
        self.baseline = {
            "gpu_inventory": [
                {"index": i, "pci_bdf": f"0000:{i:02x}:00.0"} for i in range(8)
            ],
            "compute_clients": [],
            "quiesce_states": [],
            "kmsg_writable": True,
            "ledger": [data.happy_ledger()[0]],
        }
        self.after = {
            **deepcopy(self.baseline),
            "ledger": data.happy_ledger(),
            "kernel_reset_journal": deepcopy(data.EMPTY_JOURNAL),
        }
        self.holder_clients: list[Any] = ["fake-device-client"]
        self.state = {
            "event": {"xid": 46},
            "workflow": data.happy_workflow(),
            "incident": data.happy_incident(),
            "commands": data.happy_commands(),
        }
        self.escalation = data.happy_escalation()
        self.absorb = data.happy_absorb()
        self.absorb_delays = 0
        self.restore_status = "SUCCEEDED"
        self.incident_states = {
            "inc-1": "ESCALATED",
            data.SUPPORT_INCIDENT_ID: "ESCALATED",
        }
        self.incident_close_stuck = False
        self.events: list[dict[str, Any]] = []
        self.metric_after = 2
        self.residuals = {"pod": False}
        self.regional = LifetimeRegional(self)
        self.warm = LifetimeWarm(self)
        self.window = SimpleNamespace(
            DEPLOYMENT=window.DEPLOYMENT,
            PLANE=window.PLANE,
            Settings=window.Settings,
            assignment_errors=window.assignment_errors,
            without_survey=window.without_survey,
            survey=lambda _r: deepcopy(self.survey_data),
            open_window=self.open_window,
            close_window=self.close_window,
            **{
                name: getattr(window, name)
                for name in (
                    "LIFETIME_VARIABLE",
                    "EXECUTION_TIMEOUT_VARIABLE",
                    "STEP_TIMEOUT_VARIABLE",
                    "MANAGED_RECOVERY_VARIABLE",
                    "STEP_WARNING_VARIABLE",
                    "LEASE_DURATION_VARIABLE",
                )
            },
        )
        monkeypatch.setattr(case, "env_window", self.window)
        monkeypatch.setattr(case, "RegionalLiveFixture", lambda _s: self.regional)
        monkeypatch.setattr(case, "WarmSpareLiveFixture", lambda *_a: self.warm)
        monkeypatch.setattr(case, "HostProbeFixture", self.probe)
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

    def probe(self, settings: Any) -> Any:
        return LifetimeProbe(
            self,
            settings,
            "holder" if settings.run_id.endswith("-hold") else "injector",
        )

    def open_window(
        self, settings: Any, regional: Any, survey: Any, assignments: Any
    ) -> dict[str, Any]:
        self.call("window.open", assignments)
        worker = self.runtime["deployments"]["cpu"][window.DEPLOYMENT]
        worker.update(generation=2, observed_generation=2, template_sha256="compressed")
        return {"opened": True, "assignments": assignments}

    def close_window(self, *args: Any) -> dict[str, Any]:
        self.call("window.close")
        worker = self.runtime["deployments"]["cpu"][window.DEPLOYMENT]
        worker.update(generation=3, observed_generation=3, template_sha256="original")
        return {"closed": True}

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


class LifetimeRegional:
    def __init__(self, harness: LifetimeHarness) -> None:
        self.h = harness
        self.settings = harness.settings.regional

    def store_snapshot(self, **kwargs: Any) -> dict[str, Any]:
        self.h.call("store", kwargs)
        if str(kwargs.get("marker") or "").endswith("-absorb"):
            if self.h.absorb_delays:
                self.h.absorb_delays -= 1
                return {}
            return deepcopy(self.h.absorb)
        return deepcopy(self.h.store)

    def cpu_python(self, script: str, *args: str) -> dict[str, Any]:
        if script == case.OPEN_INCIDENTS:
            return {"open_incidents": deepcopy(self.h.open_incidents)}
        if script == case.ESCALATION_CHAIN:
            self.h.call("escalation.read")
            return deepcopy(self.h.escalation)
        raise AssertionError("unexpected CPU request")

    def node_snapshot(self, node: str) -> dict[str, Any]:
        self.h.call("node.snapshot", node)
        return deepcopy(self.h.node)

    def business_workloads(self, node: str) -> list[dict[str, Any]]:
        return []

    def cpu_blast_snapshot(self) -> dict[str, Any]:
        return {"cpu": ["api-a"]}

    def runtime_identity(self) -> dict[str, Any]:
        self.h.call("runtime.identity")
        return deepcopy(self.h.runtime)

    def wait_for_workflow(self, **kwargs: Any) -> dict[str, Any]:
        self.h.call("workflow.wait", kwargs)
        self.h.node.update(
            unschedulable=True,
            taints=[{"key": "gpu-fault.io/quarantined", "effect": "NoSchedule"}],
            ownership_annotations={
                "gpu-fault.io/incident-id": data.SUPPORT_INCIDENT_ID
            },
        )
        return deepcopy(self.h.state)

    def ready_pods(self, *args: str) -> list[dict[str, str]]:
        return [{"name": "worker-pod"}]

    def kubectl(self, plane: str, *args: str, **kwargs: Any) -> str:
        self.h.call("metrics", {"plane": plane, "args": args})
        count = self.h.metric_after if self.h.injected else 1
        return json.dumps(
            {
                "metrics": "\n".join(
                    f"{name} {count}"
                    for name in (verdicts.LIFETIME_METRIC, verdicts.RECORD_ONLY_METRIC)
                )
            }
        )

    def provider_events(self, *args: Any) -> list[dict[str, Any]]:
        self.h.call("provider.events")
        return deepcopy(self.h.events)


class LifetimeProbe:
    host_script = "/fake/probe.py"

    def __init__(self, harness: LifetimeHarness, settings: Any, label: str) -> None:
        self.h = harness
        self.settings = settings
        self.label = label

    def create(self) -> None:
        self.h.call(f"{self.label}.create")

    def execute(self, action: str, *args: str, **kwargs: Any) -> dict[str, Any]:
        self.h.call(f"{self.label}.{action}", {"args": args, **kwargs})
        if action == "snapshot":
            return deepcopy(self.h.after if self.h.injected else self.h.baseline)
        if action == "holder-status":
            return {"device_clients": list(self.h.holder_clients)}
        if action == "write-xid46":
            self.h.injected = True
        return {"action": action, "ok": True}

    def cleanup(self) -> dict[str, bool]:
        self.h.call(f"{self.label}.cleanup")
        return dict(self.h.residuals)


class LifetimeWarm:
    def __init__(self, harness: LifetimeHarness) -> None:
        self.h = harness
        self.restoring = ""

    def wait_incident_idle(self, incident_id: str) -> None:
        self.h.call("incident.idle", incident_id)
        self.h.call(f"incident.idle/{incident_id}")

    def create_restore_workflow(self, **kwargs: Any) -> dict[str, Any]:
        self.h.call("restore.create", kwargs)
        self.restoring = kwargs["incident_id"]
        return {"workflow_request_id": f"restore-{self.restoring}"}

    def wait_workflow_id(self, request_id: str) -> dict[str, Any]:
        self.h.call("restore.wait", request_id)
        if self.h.restore_status == "SUCCEEDED":
            self.h.node.update(unschedulable=False, taints=[], ownership_annotations={})
            if not self.h.incident_close_stuck:
                self.h.incident_states[self.restoring] = "RECOVERED"
        return {"status": self.h.restore_status}

    def incident_by_id(self, incident_id: str) -> dict[str, Any]:
        self.h.call("incident.read", incident_id)
        return {"state": self.h.incident_states[incident_id]}
