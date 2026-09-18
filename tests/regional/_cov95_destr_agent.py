from __future__ import annotations

import json
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import destr019_verdicts as verdicts
from scripts.e2e.regional import run_destr019_agent_restart_ledger as case
from scripts.e2e.regional.probes.destr019_node_probe import migration_drill_report
from tests.regional._cov95_destr_branches import ready_runtime
from tests.regional._cov95_destr_warm import NOW, Clock, regional_settings

NODE = "node-a"
INCIDENT = "incident-owned"
WORKFLOW = "workflow-owned"
GENERATION = 4
COMMAND = {"idempotency_key": f"{WORKFLOW}/1/RESTART_FABRIC_MANAGER"}
COMMAND_ID = verdicts.expected_command_id(COMMAND, node=NODE, generation=GENERATION)


def health(counters: dict[str, int]) -> dict[str, Any]:
    return {
        "http_status": 200,
        "payload": {
            "status": "ok",
            "ledger": {"writable": True},
            "heartbeat": {
                "configured": True,
                "consecutive_failures": 0,
                "last_success_at": NOW.isoformat(),
                "last_success_age_seconds": 1,
            },
            "counters": counters,
        },
    }


def journal() -> dict[str, Any]:
    fields = {
        "command_id": COMMAND_ID,
        "incident_id": INCIDENT,
        "workflow_request_id": WORKFLOW,
        "operation": "RESTART_FABRIC_MANAGER",
        "node_id": NODE,
        "fencing_token": "2",
        "agent_generation": str(GENERATION),
        "gpu_count": "0",
        "attempt": "1",
    }
    return {
        "lines": [
            {"phase": "accepted", "fields": dict(fields)},
            {"phase": "started", "fields": dict(fields)},
            {
                "phase": "completed",
                "fields": {**fields, "status": "SUCCEEDED", "duration_ms": "20"},
            },
        ]
    }


def audit() -> dict[str, Any]:
    return {
        "present": True,
        "user_version": verdicts.LEDGER_SCHEMA_VERSION,
        "columns": ["command_id", "payload", *verdicts.AUDIT_COLUMNS],
        "primary_key": list(verdicts.LEDGER_PRIMARY_KEY),
        "row_count": 1,
        "interrupted_count": 0,
        "rows": [
            {
                "command_id": COMMAND_ID,
                "attempt": 1,
                "state": "SUCCEEDED",
                "operation": "RESTART_FABRIC_MANAGER",
                "started_at": NOW.isoformat(),
                "completed_at": (NOW + timedelta(seconds=1)).isoformat(),
                "incident_id": INCIDENT,
                "workflow_request_id": WORKFLOW,
                "fencing_token": 2,
                "gpu_uuid_count": 0,
                "gpu_uuids_present": True,
                "parameters_digest": "a" * 64,
                "signature_digest": "b" * 64,
                "exit_code": None,
            }
        ],
    }


class AgentHarness:
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.settings = case.Settings(
            regional_settings(tmp_path),
            NODE,
            "example.test/probe",
            tmp_path / "predecessor.json",
            180,
        )
        self.tmp_path = tmp_path
        self.clock = Clock()
        self.calls: list[tuple[str, Any]] = []
        self.failures: dict[str, BaseException] = {}
        self.advance_at: dict[str, float] = {}
        self.restarted = False
        self.injected = False
        self.predecessor_valid = True
        self.tests_pass = True
        self.heartbeat_delays = 0
        self.agent = {
            "lifecycle_state": "ACTIVE",
            "generation": GENERATION,
            "agent_incarnation_id": "incarnation-owned",
            "boot_id": "boot-1",
            "allowed_operations": ["RESTART_FABRIC_MANAGER"],
            "last_seen_at": NOW.isoformat(),
        }
        self.changed_agent: dict[str, Any] = {}
        self.node = {
            "name": NODE,
            "uid": "node-uid",
            "boot_id": "boot-1",
            "ready": "True",
            "unschedulable": False,
            "taints": [],
            "ownership_annotations": {},
        }
        self.workloads: list[dict[str, Any]] = []
        self.queue: dict[str, Any] = {"depth": 0, "fault_backlog_depth": 0}
        self.commands: Any = {"open_by_cluster": {}}
        self.quiet_queue: dict[str, Any] = self.queue
        self.quiet_commands: Any = self.commands
        self.host = {
            "boot_id": "boot-1",
            "agent": {"ActiveState": "active", "MainPID": "100", "InvocationID": "i1"},
            "ledger": {
                "present": True,
                "user_version": verdicts.LEDGER_SCHEMA_VERSION,
                "interrupted_count": 0,
            },
            "gpu_fault_timers": [],
            "restore_timer": {"ActiveState": "inactive"},
        }
        self.fabric = {
            "kmsg_writable": True,
            "compute_clients": [],
            "fabric_manager": {"ActiveState": "active"},
            "gpu_pci_bdf": "0000:01:00.0",
        }
        self.health_before = health(dict.fromkeys(verdicts.COUNTER_NAMES, 0))
        self.health_after = health(
            {"accepted": 1, "completed": 1, "failed": 0, "rejected": 0}
        )
        self.migration_override: dict[str, Any] | None = None
        self.restart_report = {
            "before": {"ActiveState": "active", "MainPID": "100", "InvocationID": "i1"},
            "after": {"ActiveState": "active", "MainPID": "200", "InvocationID": "i2"},
            "boot_id": "boot-1",
            "restore_unit": "owned-restore.timer",
            "armed_at": (NOW - timedelta(seconds=1)).isoformat(),
            "restarted_at": NOW.isoformat(),
        }
        self.state = {
            "event": {"xid": 45},
            "decision": {"official_action": "RESTART_FM", "disposition": "EXECUTABLE"},
            "workflow": {
                "request_id": WORKFLOW,
                "status": "SUCCEEDED",
                "fencing_token": 2,
                "official_steps": [
                    {"operation": op, "execution_owner": owner}
                    for op, owner in zip(
                        verdicts.EXPECTED_STEPS, verdicts.EXPECTED_OWNERS, strict=True
                    )
                ],
                "step_executions": [
                    {"operation": op, "status": "SUCCEEDED"}
                    for op in verdicts.EXPECTED_STEPS
                ],
            },
            "incident": {"incident_id": INCIDENT, "node_ids": [NODE]},
            "commands": [{**COMMAND, "status": "SUCCEEDED"}],
        }
        self.journal = journal()
        self.audit = audit()
        self.disarm = {
            "restore_timer": {"ActiveState": "inactive"},
            "agent": {"ActiveState": "active"},
            "gpu_fault_timers": [],
        }
        self.events: list[dict[str, Any]] = []
        self.residuals = {"pod": False}
        self.regional = AgentRegional(self)
        monkeypatch.setattr(case, "RegionalLiveFixture", lambda _s: self.regional)
        monkeypatch.setattr(case, "HostProbeFixture", self.probe)
        monkeypatch.setattr(
            case, "focused_tests", lambda _p: {"passed": self.tests_pass}
        )
        monkeypatch.setattr(
            case, "predecessor_evidence", lambda *_a: {"valid": self.predecessor_valid}
        )
        monkeypatch.setattr(case, "time", self.clock)
        monkeypatch.setattr(case, "datetime", self.clock)

    def probe(self, settings: Any) -> Any:
        label = "injector" if settings.run_id.endswith("-inject") else "agent"
        return AgentProbe(self, label)

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


class AgentRegional:
    def __init__(self, harness: AgentHarness) -> None:
        self.h = harness
        self.settings = harness.settings.regional

    def store_snapshot(self, **kwargs: Any) -> dict[str, Any]:
        self.h.call("store", kwargs)
        agent = deepcopy(self.h.agent)
        if self.h.restarted:
            agent.update(self.h.changed_agent)
            if self.h.heartbeat_delays:
                self.h.heartbeat_delays -= 1
                agent["last_seen_at"] = (NOW - timedelta(seconds=10)).isoformat()
            else:
                agent["last_seen_at"] = self.h.clock.now().isoformat()
        quiet = "observed_after" not in kwargs
        return {
            "release_id": "release-test",
            "agent": agent,
            "profile": {"profile_version": "profile-v1"},
            "queue": deepcopy(self.h.quiet_queue if quiet else self.h.queue),
            "remote_commands": deepcopy(
                self.h.quiet_commands if quiet else self.h.commands
            ),
        }

    def node_snapshot(self, node: str) -> dict[str, Any]:
        self.h.call("node.snapshot", node)
        return deepcopy(self.h.node)

    def business_workloads(self, node: str) -> list[dict[str, Any]]:
        return deepcopy(self.h.workloads)

    def runtime_identity(self) -> dict[str, Any]:
        return ready_runtime()

    def cpu_blast_snapshot(self) -> dict[str, Any]:
        return {"cpu": ["api-a"]}

    def wait_for_workflow(self, **kwargs: Any) -> dict[str, Any]:
        self.h.call("workflow.wait", kwargs)
        return deepcopy(self.h.state)

    def provider_events(self, *args: Any) -> list[dict[str, Any]]:
        self.h.call("provider.events")
        return deepcopy(self.h.events)

    def verify_runtime_identity(self, identity: dict[str, Any], **kwargs: Any) -> None:
        self.h.call("runtime.verify", kwargs["stage"])
        assert identity == ready_runtime(), "case cleanup must retain its release"


class AgentProbe:
    def __init__(self, harness: AgentHarness, label: str) -> None:
        self.h = harness
        self.label = label

    def create(self) -> None:
        self.h.call(f"{self.label}.create")

    def execute(self, action: str, *args: str, **kwargs: Any) -> dict[str, Any]:
        self.h.call(f"{self.label}.{action}", {"args": args, **kwargs})
        if action == "snapshot":
            return deepcopy(self.h.host if self.label == "agent" else self.h.fabric)
        if action == "agent-health":
            return deepcopy(
                self.h.health_after if self.h.injected else self.h.health_before
            )
        if action == "migration-drill":
            if self.h.migration_override is not None:
                return deepcopy(self.h.migration_override)
            return migration_drill_report(self.h.tmp_path / "scratch.db", now=NOW)
        if action == "restart-agent":
            self.h.restarted = True
            return deepcopy(self.h.restart_report)
        if action == "write-xid45":
            self.h.injected = True
        if action == "journal":
            return deepcopy(self.h.journal)
        if action == "ledger-audit":
            return deepcopy(self.h.audit)
        if action == "disarm-restore":
            return deepcopy(self.h.disarm)
        return {"action": action, "ok": True}

    def cleanup(self) -> dict[str, bool]:
        self.h.call(f"{self.label}.cleanup")
        return dict(self.h.residuals)
