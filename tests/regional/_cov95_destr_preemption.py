from __future__ import annotations

import json
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import destr_barrier_authorization as authorization
from scripts.e2e.regional import run_destr016_preempting_reboot as case
from tests.regional import test_destr016_preempting_reboot as data
from tests.regional._cov95_destr_branches import ready_runtime
from tests.regional._cov95_destr_warm import NOW, Clock, regional_settings


class PreemptionHarness:
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.settings = case.Settings(
            regional_settings(tmp_path),
            data.NODE,
            "example.test/probe",
            "fake-hyperpod",
            "arn:aws:iam::123456789012:role/gpu-fault-executor-role",
            "",
            "",
            1800,
            tmp_path / "predecessor.json",
        )
        self.run_id = f"destr016-{tmp_path.name.rsplit('-', 1)[-1].lower()}-a1"
        self.clock = Clock()
        self.calls: list[tuple[str, Any]] = []
        self.failures: dict[str, BaseException] = {}
        self.advance_at: dict[str, float] = {}
        self.preflight = data.happy_preflight_arguments()
        self.preflight["agent"].update(generation=4, artifact_sha256="artifact-sha")
        self.node = {**data.node_baseline(), "boot_id": "boot-before"}
        self.baseline = {**data.host_baseline(), "boot_id": "boot-before"}
        self.after = {**data.host_after(), "boot_id": "boot-after"}
        self.injected = False
        self.rebooted = False
        self.authorized: list[str] = []
        self.barrier_delays = 0
        self.absorb_delays = 0
        self.successor_delays = 0
        self.supersede_delays = 0
        self.handoff_delays = 0
        self.terminal_delays = 0
        self.bad_barrier = False
        self.bad_absorb = False
        self.changed_predecessor = False
        self.restored = "SUCCEEDED"
        self.residuals = {"pod": False}
        self.holder_status = data.holder_status()
        self.events = data.provider_events()
        self.terminal = data.terminal_state()
        self.runtime = ready_runtime()
        self.cpu = {"cpu": ["api-a"]}
        self.cpu_after = self.cpu
        self.regional = PreemptionRegional(self)
        self.warm = PreemptionWarm(self)
        monkeypatch.setattr(case, "RegionalLiveFixture", lambda _s: self.regional)
        monkeypatch.setattr(case, "WarmSpareLiveFixture", lambda *_a: self.warm)
        monkeypatch.setattr(case, "HostProbeFixture", self.probe)
        monkeypatch.setattr(
            case, "focused_tests", lambda _p: deepcopy(self.preflight["tests"])
        )
        monkeypatch.setattr(
            case,
            "predecessor_evidence",
            lambda *_a: deepcopy(self.preflight["predecessor"]),
        )
        monkeypatch.setattr(
            case,
            "replica_env",
            lambda *_a, **_k: [
                {
                    "values": {
                        case.STEP_TIMEOUT_VARIABLE: "600",
                        case.NODE_LIFETIME_VARIABLE: "3600",
                    }
                }
            ],
        )
        monkeypatch.setattr(case, "time", self.clock)
        monkeypatch.setattr(case, "datetime", self.clock)
        monkeypatch.setattr(authorization, "datetime", self.clock)

    def call(self, name: str, detail: Any = None) -> None:
        self.calls.append((name, deepcopy(detail)))
        self.clock.sleep(self.advance_at.get(name, 0))
        if name in self.failures:
            raise self.failures[name]

    def probe(self, settings: Any) -> Any:
        return PreemptionProbe(
            self, "injector" if settings.case_id.endswith("-inject") else "holder"
        )

    def barrier(self, absorbed: bool = False) -> dict[str, Any]:
        state = data.absorbed_snapshot() if absorbed else data.barrier_snapshot()
        workflow = state["workflow"]
        workflow["fencing_token"] = 7
        state["incident"]["drill_id"] = f"{self.run_id}-r"
        workflow["step_executions"][2]["details"] = {
            "agent_generations": {data.NODE: 4},
            "maintenance_window_expires_at": (
                NOW + timedelta(seconds=3600)
            ).isoformat(),
        }
        state["commands"][0]["idempotency_key"] = "verify-owned"
        state["commands"].append(
            {
                "workflow_request_id": data.RESET_ID,
                "idempotency_key": "quiesce-owned",
                "step": {"operation": "QUIESCE_GPU_SERVICES", "node_ids": [data.NODE]},
                "status": "SUCCEEDED",
            }
        )
        if self.bad_barrier:
            workflow["status"] = "FAILED"
            workflow["step_executions"][-1]["status"] = "SUCCEEDED"
        if self.bad_absorb and absorbed:
            workflow["request_id"] = "wrong-workflow"
        if self.changed_predecessor and self.authorized:
            workflow["request_id"] = "changed-predecessor"
        return state

    def plan(self, run_dir: Path) -> dict[str, Any]:
        path = run_dir / "cases" / case.CASE_ID
        path.mkdir(parents=True, exist_ok=True)
        preflight = case.read_only_preflight(self.settings, path)
        (path / "plan.json").write_text(
            json.dumps({"details": case.plan_details(self.settings, preflight)}),
            encoding="utf-8",
        )
        return preflight

    def execute(self, run_dir: Path, seconds: int = 7200) -> tuple[int, dict[str, Any]]:
        code = case.execute_case(
            self.settings, run_dir, 1, NOW + timedelta(seconds=seconds)
        )
        path = run_dir / "cases" / case.CASE_ID / f"{case.CASE_ID}.json"
        return code, json.loads(path.read_text(encoding="utf-8"))


class PreemptionRegional:
    def __init__(self, harness: PreemptionHarness) -> None:
        self.h = harness
        self.settings = harness.settings.regional

    def store_snapshot(self, **kwargs: Any) -> dict[str, Any]:
        self.h.call("store", kwargs)
        marker = str(kwargs.get("marker") or "")
        if not marker:
            return {
                "release_id": "release-test",
                "agent": deepcopy(self.h.preflight["agent"]),
                "profile": deepcopy(self.h.preflight["profile"]),
                "queue": deepcopy(self.h.preflight["queue"]),
                "remote_commands": deepcopy(self.h.preflight["remote_commands"]),
            }
        if marker.endswith("-r"):
            self.h.call("barrier.read")
            if self.h.barrier_delays:
                self.h.barrier_delays -= 1
                return {}
            return self.h.barrier()
        if marker.endswith("-s"):
            self.h.call("absorb.read")
            if self.h.absorb_delays:
                self.h.absorb_delays -= 1
                return self.h.barrier()
            return self.h.barrier(absorbed=True)
        if marker.endswith("-e"):
            self.h.call("successor.read")
            if self.h.successor_delays:
                self.h.successor_delays -= 1
                return {}
            if self.h.rebooted:
                state = deepcopy(self.h.terminal)
                if self.h.terminal_delays:
                    self.h.terminal_delays -= 1
                    state["workflow"]["status"] = "RUNNING"
                return state
            return {
                "workflow": data.successor_workflow(),
                "incident": data.escalated_incident(),
                "decision": {"official_action": "RESTART_BM"},
            }
        raise AssertionError(f"unexpected marker {marker}")

    def node_snapshot(self, node: str) -> dict[str, Any]:
        return deepcopy(self.h.node)

    def runtime_identity(self) -> dict[str, Any]:
        return deepcopy(self.h.runtime)

    def cpu_blast_snapshot(self) -> dict[str, Any]:
        return deepcopy(self.h.cpu_after if self.h.injected else self.h.cpu)

    def business_workloads(self, node: str) -> list[dict[str, Any]]:
        return deepcopy(self.h.preflight["business_workloads"])

    def gpu_workloads(self) -> list[dict[str, Any]]:
        return deepcopy(self.h.preflight["gpu_workloads"])

    def cpu_python(self, script: str, request_id: str) -> dict[str, Any]:
        if script == case.WORKFLOW_BY_ID:
            if request_id == data.RESET_ID:
                self.h.call("predecessor.read", request_id)
                if self.h.supersede_delays:
                    self.h.supersede_delays -= 1
                    return data.parked_reset_workflow()
                return data.superseded_reset_workflow()
            self.h.call("handoff.read", request_id)
            workflow = data.successor_workflow()
            if self.h.handoff_delays:
                self.h.handoff_delays -= 1
                workflow["quiesce_handoff_from_workflow_id"] = None
            return workflow
        if script == case.COMMANDS_BY_WORKFLOW:
            self.h.call("commands.read", request_id)
            return {"commands": data.cancelled_commands()}
        raise AssertionError("unexpected fake preemption CPU request")

    def wait_node_ready(self, node: str, **kwargs: Any) -> dict[str, Any]:
        self.h.call("node.ready", kwargs)
        self.h.rebooted = True
        self.h.node["boot_id"] = "boot-after"
        return deepcopy(self.h.node)

    def provider_events(self, *args: Any) -> list[dict[str, Any]]:
        self.h.call("provider.events")
        return deepcopy(self.h.events)

    def verify_runtime_identity(self, identity: Any, **kwargs: Any) -> None:
        self.h.call("runtime.verify", kwargs["stage"])
        assert identity == self.h.runtime, "preemption must preserve release identity"


class PreemptionProbe:
    host_script = "/fake/probe.py"

    def __init__(self, harness: PreemptionHarness, label: str) -> None:
        self.h = harness
        self.label = label

    def create(self) -> None:
        self.h.call(f"{self.label}.create")

    def execute(self, action: str, *args: str, **kwargs: Any) -> dict[str, Any]:
        self.h.call(f"{self.label}.{action}", {"args": args, **kwargs})
        if action == "snapshot":
            return deepcopy(self.h.after if self.h.rebooted else self.h.baseline)
        if action == "write-xid46":
            self.h.injected = True
        if action == "authorize-injection":
            phase = args[args.index("--phase") + 1]
            proof = json.loads(args[args.index("--authorization") + 1])
            self.h.call("authorization", {"phase": phase, "proof": proof})
            self.h.authorized.append(phase)
        if action == "holder-status":
            return deepcopy(self.h.holder_status)
        return {"action": action, "ok": True}

    def cleanup(self) -> dict[str, bool]:
        self.h.call(f"{self.label}.cleanup")
        return dict(self.h.residuals)


class PreemptionWarm:
    def __init__(self, harness: PreemptionHarness) -> None:
        self.h = harness

    def wait_incident_idle(self, incident: str) -> None:
        self.h.call("incident.idle", incident)

    def wait_agent_active(self, node: str) -> None:
        self.h.call("agent.active", node)

    def create_restore_workflow(self, **kwargs: Any) -> dict[str, Any]:
        self.h.call("restore.create", kwargs)
        return {"workflow_request_id": "restore-owned"}

    def wait_workflow_id(self, request_id: str) -> dict[str, Any]:
        self.h.call("restore.wait", request_id)
        return {"status": self.h.restored}
