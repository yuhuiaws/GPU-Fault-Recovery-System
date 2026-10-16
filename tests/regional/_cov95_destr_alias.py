from __future__ import annotations

import json
import subprocess
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import destr020_verdicts as verdicts
from scripts.e2e.regional import run_destr020_identity_mismatch_isolation as case
from tests.regional._cov95_destr_branches import ready_runtime
from tests.regional._cov95_destr_warm import NOW, Clock, regional_settings

LOCK_BUSY = (
    "gpu-fault-admin: another administrator mutation is in progress (held by "
    "pid 4242: gpu-fault-admin config --state-dir /state)\n"
    "gpu-fault-admin: full output in /state/logs/mutating/admin.log\n"
)
# A product answer: "lock", ("refuse", message), a JSON document (exit 0),
# (returncode, document) or a callable of the harness returning one of them.
Answer = Any


def eligible_item(alias: str, **overrides: Any) -> dict[str, Any]:
    """The dry run's item for an alias the exit accepts (no Agent to retire)."""

    item: dict[str, Any] = {
        "agent": f"cluster-a/{alias}",
        "cluster_id": "cluster-a",
        "node_id": alias,
        "terminalization": "accept-unknown-provider-identity",
        "retire": False,
        "kubernetes_reports": {"open": True, "never_reported": True},
        "kubernetes_reports_read": {
            "absent_since": (NOW - timedelta(hours=3)).isoformat(),
            "chain_started_at": (NOW - timedelta(hours=3)).isoformat(),
            "received_at": NOW.isoformat(),
            "opens_at": None,
        },
        "eligible": True,
        "reasons": [],
    }
    item.update(overrides)
    return item


def nothing_names(alias: str) -> dict[str, Any]:
    return {
        "agent": alias,
        "node_id": alias,
        "terminalization": "accept-unknown-provider-identity",
        "eligible": False,
        "reasons": [f"{verdicts.NOTHING_NAMES_THE_NODE} in any managed cluster"],
    }


def accepted(alias: str) -> dict[str, Any]:
    return {
        "mode": "retire-departed-agents-apply",
        "retired_agents": [],
        "accepted_identities": [f"cluster-a/{alias}"],
        "failed_agents": [],
        "failures": {},
        "reference": "applied",
        "dry_run": False,
    }


class DepartedExit:
    """The product's departed-node exit for the alias, as its CLI answers it.

    ``line`` is the alias's ledger line before the acceptance (None: none);
    ``plans`` and ``applies`` script the dry runs and the applies in order
    (the last answer repeats). An apply that accepts starts the cleanup's
    clock: ``cycle`` seconds later its next cycle ends the line as ``ends``
    says -- ``CLEANED``, ``gone`` or ``stuck`` (still REFUSED).
    """

    def __init__(self, harness: AliasHarness) -> None:
        self.h = harness
        self.line: dict[str, Any] | None = {
            "status": "REFUSED",
            "blocker_classes": ["provider_identity_unknown"],
            "blockers": [
                "provider membership unknown: the fleet recorded no EC2 instance "
                "id for the node (Agent node_instance_id); the administrator may "
                "accept the departure"
            ],
            "admin_commands": ["gpu-fault-admin workflow-reconcile ..."],
            "actions": [],
        }
        self.plans: list[Answer] = [eligible_item(harness.alias)]
        self.applies: list[Answer] = [accepted(harness.alias)]
        self.cycle = 0.0
        self.ends = "CLEANED"
        self.accepted_at: float | None = None

    def answer(self, argv: list[str]) -> subprocess.CompletedProcess[str]:
        dry_run = "--dry-run" in argv
        queue = self.plans if dry_run else self.applies
        reply = queue.pop(0) if len(queue) > 1 else queue[0]
        if callable(reply):
            reply = reply(self.h)
        if reply == "lock":
            return subprocess.CompletedProcess(argv, 2, "", LOCK_BUSY)
        if isinstance(reply, tuple) and reply[0] == "refuse":
            return subprocess.CompletedProcess(
                argv, 2, "", f"gpu-fault-admin: {reply[1]}\n"
            )
        code, document = reply if isinstance(reply, tuple) else (0, reply)
        if dry_run and "items" not in document:
            document = {
                "mode": "accept-unknown-provider-identity-plan",
                "evaluated_at": self.h.clock.now().isoformat(),
                "discovery": {
                    "named": 1,
                    "eligible": int(bool(document.get("eligible"))),
                },
                "items": [document],
                "plan_sha256": "a" * 64,
                "dry_run": True,
            }
        if not dry_run and code == 0 and document.get("accepted_identities"):
            self.accepted_at = self.h.clock.elapsed
        return subprocess.CompletedProcess(
            argv, code, json.dumps(document, indent=2), ""
        )

    def ledger(self) -> dict[str, Any]:
        line = deepcopy(self.line)
        done = (
            self.accepted_at is not None
            and self.h.clock.elapsed - self.accepted_at >= self.cycle
        )
        if done and self.ends == "gone":
            line = None
        elif done and self.ends == "CLEANED" and line is not None:
            line.update(
                status="CLEANED",
                blockers=[],
                blocker_classes=[],
                admin_commands=[],
                actions=[
                    "incident-closed:incident-owned",
                    "incident-closed:support-owned",
                ],
            )
        acceptance = (
            {"change_reference": "applied", "actor": "case"}
            if self.accepted_at is not None
            else None
        )
        return {"ledger": line, "acceptance": acceptance}


class AliasHarness:
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # The site's state directory with its own deploy-host CLI.
        state_dir = tmp_path / "state"
        (state_dir / "deployer-venv" / "bin").mkdir(parents=True, exist_ok=True)
        (state_dir / "deployer-venv" / "bin" / "gpu-fault-admin").write_text(
            "fake deploy-host CLI\n", encoding="utf-8"
        )
        (state_dir / "site.yaml").write_text("fake site\n", encoding="utf-8")
        self.settings = case.Settings(
            regional_settings(tmp_path),
            "node-a",
            tmp_path / "predecessor.json",
            30,
            state_dir,
        )
        self.alias = verdicts.alias_for_run(case.run_identity(tmp_path, 1))
        self.clock = Clock()
        self.calls: list[tuple[str, Any]] = []
        self.failures: dict[str, BaseException] = {}
        self.posted: dict[str, Any] = {}
        self.alias_present = False
        self.alias_agent: dict[str, Any] | None = None
        self.alias_event: dict[str, Any] | None = None
        # The alias's Agent record as the cleanup finds it (None: nothing ever
        # registered one), and whether something re-registers it after the
        # revoke (the cleanup's read-back then finds it ACTIVE again).
        self.cleanup_agent: dict[str, Any] | None = None
        self.cleanup_agent_returns = False
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
        self.departed = DepartedExit(self)
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

    def alias_agent_cleanup(
        self, cluster_id: str, alias: str, transition_id: str, reason: str
    ) -> dict[str, Any]:
        """What ``case.ALIAS_AGENT_CLEANUP`` prints: revoke through the
        registry (drain first when ACTIVE; a DRAINING record finishes under the
        transition that drained it), then the record read back."""

        before = deepcopy(self.cleanup_agent)
        revoked = False
        if before is not None and before["lifecycle_state"] != "REVOKED":
            transition = (
                transition_id
                if before["lifecycle_state"] == "ACTIVE"
                else before.get("transition_id") or transition_id
            )
            self.cleanup_agent = {
                **before,
                "lifecycle_state": "REVOKED",
                "transition_id": transition,
                "transition_reason": reason,
            }
            revoked = True
        if before is not None and self.cleanup_agent_returns:
            self.cleanup_agent = {**before, "lifecycle_state": "ACTIVE"}
        after = self.cleanup_agent
        return {
            "found": before is not None,
            "before": None if before is None else before["lifecycle_state"],
            "lifecycle_state": None if after is None else after["lifecycle_state"],
            "transition_id": None if after is None else after.get("transition_id"),
            "revoked_by_case": revoked,
        }

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
        if script == case.ALIAS_AGENT_CLEANUP:
            self.h.call("alias.agent.cleanup", args)
            cluster_id, alias, transition_id, reason = args
            return self.h.alias_agent_cleanup(cluster_id, alias, transition_id, reason)
        if script == case.ALIAS_DEPARTED_LEDGER:
            self.h.call("departed.ledger", args)
            assert args == ("cluster-a", self.h.alias), args
            return self.h.departed.ledger()
        raise AssertionError("unexpected fake CPU request")

    def run(
        self, command: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        """The deploy-host CLI: only the departed-node exit for the alias."""

        cli = case.admin_command(self.h.settings.admin_state_dir)
        assert command[: len(cli)] == cli, command
        argv = list(command[len(cli) :])
        self.h.call("admin.dry-run" if "--dry-run" in argv else "admin.apply", argv)
        return self.h.departed.answer(argv)

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
