from __future__ import annotations

import json
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import destr020_verdicts as verdicts
from scripts.e2e.regional import run_destr020_identity_mismatch_isolation as case
from tests.regional._cov95_destr_branches import ready_runtime
from tests.regional._cov95_destr_warm import NOW, Clock, regional_settings


class AliasHarness:
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.settings = case.Settings(
            regional_settings(tmp_path), "node-a", tmp_path / "predecessor.json", 30
        )
        self.alias = verdicts.alias_for_run(case.run_identity(tmp_path, 1))
        self.clock = Clock()
        self.calls: list[tuple[str, Any]] = []
        self.failures: dict[str, BaseException] = {}
        self.posted: dict[str, Any] = {}
        self.alias_present = False
        self.alias_agent: dict[str, Any] | None = None
        self.alias_event: dict[str, Any] | None = None
        self.predecessor_valid = True
        self.tests_pass = True
        self.queue = {"depth": 0, "fault_backlog_depth": 0}
        self.commands = {"open_by_cluster": {}}
        self.quiet_queue = self.queue
        self.quiet_commands = self.commands
        self.chain_pending = 0
        self.node = {
            "name": "node-a",
            "uid": "node-uid",
            "ready": "True",
            "unschedulable": False,
            "taints": [],
            "ownership_annotations": {},
        }
        self.receipt = {
            "accepted": {"processor_request_id": "processor-owned"},
            "receipt": {"status": 200},
        }
        absent = f"node {self.alias} is absent; isolation cannot be observed"
        self.state = {
            "event": {"xid": verdicts.INJECTED_XID},
            "decision": {
                "disposition": verdicts.EXPECTED_DISPOSITION,
                "safety_action": verdicts.EXPECTED_SAFETY_ACTION,
                "action": None,
            },
            "workflow": {
                "request_id": "workflow-owned",
                "status": "FAILED",
                "safety_only": True,
                "safety_steps": [
                    {"operation": op, "step_index": i}
                    for i, op in enumerate(verdicts.SAFETY_STEPS)
                ],
                "official_steps": [{"operation": "FREEZE_EVIDENCE", "step_index": 0}],
                "completed_operations": ["FREEZE_EVIDENCE"],
                "step_executions": [
                    {
                        "step_index": 0,
                        "operation": "FREEZE_EVIDENCE",
                        "status": "SUCCEEDED",
                    },
                    {
                        "step_index": 1,
                        "operation": "MARK_UNSCHEDULABLE",
                        "status": "FAILED",
                        "error": absent,
                    },
                ],
            },
            "incident": {
                "incident_id": "incident-owned",
                "node_ids": [self.alias],
                "state": "ESCALATED",
            },
        }
        self.remote_commands = [
            {
                "command_id": "command-owned",
                "operation": "MARK_UNSCHEDULABLE",
                "status": "FAILED",
                "error": absent,
                "result_details": {
                    "safety_rejection": True,
                    "absent": True,
                    "node_id": self.alias,
                },
            }
        ]
        self.chain = {
            "incident": {
                "incident_id": "support-owned",
                "effective_action": verdicts.SUPPORT_ESCALATION_ACTION,
                "reasons": [
                    f"{verdicts.SUPPORT_ESCALATION_STAGE} remediation failed; failed operation or validation: MARK_UNSCHEDULABLE"
                ],
                "node_ids": [self.alias],
            },
            "workflow": {
                "request_id": "workflow-support-owned",
                "status": "SUCCEEDED",
                "official_steps": [
                    {"operation": op} for op in verdicts.SUPPORT_ESCALATION_OPERATIONS
                ],
                "step_executions": [
                    {"operation": op, "status": "SUCCEEDED"}
                    for op in verdicts.SUPPORT_ESCALATION_OPERATIONS
                ],
            },
            "second_order_incident": None,
            "second_order_workflow": None,
        }
        self.events: list[dict[str, Any]] = []
        self.regional = AliasRegional(self)
        monkeypatch.setattr(case, "RegionalLiveFixture", lambda _s: self.regional)
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
        if name in self.failures:
            raise self.failures[name]

    def plan(self, run_dir: Path) -> dict[str, Any]:
        path = run_dir / "cases" / case.CASE_ID
        path.mkdir(parents=True, exist_ok=True)
        preflight = case.read_only_preflight(self.settings, path, alias=self.alias)
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


class AliasRegional:
    def __init__(self, harness: AliasHarness) -> None:
        self.h = harness
        self.settings = harness.settings.regional

    def store_snapshot(self, **kwargs: Any) -> dict[str, Any]:
        self.h.call("store", kwargs)
        if kwargs["node"] == self.h.alias:
            return {"agent": self.h.alias_agent, "event": self.h.alias_event}
        quiet = "observed_after" not in kwargs
        return {
            "release_id": "release-test",
            "agent": {
                "lifecycle_state": "ACTIVE",
                "runtime_profile_version": "profile-v1",
            },
            "profile": {"warnings": []},
            "queue": deepcopy(self.h.quiet_queue if quiet else self.h.queue),
            "remote_commands": deepcopy(
                self.h.quiet_commands if quiet else self.h.commands
            ),
        }

    def node_snapshot(self, node: str) -> dict[str, Any]:
        self.h.call("node.snapshot", node)
        return deepcopy(self.h.node)

    def gpu_nodes(self) -> list[dict[str, Any]]:
        self.h.call("gpu.nodes")
        return [{"name": "node-a"}]

    def business_workloads(self, node: str) -> list[dict[str, Any]]:
        return []

    def kubectl(self, plane: str, *args: str, **kwargs: Any) -> str:
        self.h.call("node.present", {"plane": plane, "args": args, **kwargs})
        assert args[:3] == ("get", "node", self.h.alias), args
        return f"node/{self.h.alias}" if self.h.alias_present else ""

    def cpu_python(self, script: str, *args: str) -> dict[str, Any]:
        if script == case.FLEET_AGENTS:
            return {"agents": [{"node_id": "node-a"}]}
        if script == case.c017.REMOTE_COMMANDS:
            self.h.call("commands.read", args)
            return {"remote_commands": deepcopy(self.h.remote_commands)}
        if script == case.ESCALATION_CHAIN:
            self.h.call("chain.read", args)
            chain = deepcopy(self.h.chain)
            if self.h.chain_pending:
                self.h.chain_pending -= 1
                chain["workflow"]["status"] = "RUNNING"
            return chain
        raise AssertionError("unexpected fake CPU request")

    def node_metadata(self, node: str) -> dict[str, Any]:
        self.h.call("metadata", node)
        return {"product": "NVIDIA A100"}

    def post_xid_event(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.h.call("event.post", payload)
        self.h.posted = deepcopy(payload)
        return deepcopy(self.h.receipt)

    def wait_for_workflow(self, **kwargs: Any) -> dict[str, Any]:
        self.h.call("workflow.wait", kwargs)
        return deepcopy(self.h.state)

    def runtime_identity(self) -> dict[str, Any]:
        return ready_runtime()

    def cpu_blast_snapshot(self) -> dict[str, Any]:
        return {"cpu": ["api-a"]}

    def provider_events(self, *args: Any) -> list[dict[str, Any]]:
        self.h.call("provider.events")
        return deepcopy(self.h.events)

    def verify_runtime_identity(self, identity: Any, **kwargs: Any) -> None:
        self.h.call("runtime.verify")
        assert identity == ready_runtime(), "alias case must preserve the runtime"
