from __future__ import annotations

import json
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.models import WorkflowStatus
from scripts.e2e.regional import destr017_verdicts as verdicts
from scripts.e2e.regional import destr_barrier_authorization as authorization
from scripts.e2e.regional import run_destr017_out_of_band_reboot_fence as case
from scripts.e2e.regional.acceptance_runner_common import OPEN_INCIDENTS_PROBE
from scripts.e2e.regional.regional_live_fixture import RegionalFixtureError
from tests.regional import test_destr017_out_of_band_reboot_fence as data
from tests.regional._cov95_destr_branches import ready_runtime
from tests.regional._cov95_destr_warm import NOW, Clock, regional_settings


class FenceHarness:
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.settings = case.Settings(
            regional_settings(tmp_path),
            data.NODE,
            "example.test/probe",
            "",
            60,
            900,
            tmp_path / "predecessor.json",
        )
        self.run_id = case.derived_identity(tmp_path, 1)
        self.clock = Clock()
        self.calls: list[tuple[str, Any]] = []
        self.failures: dict[str, BaseException] = {}
        self.advance_at: dict[str, float] = {}
        self.runtime = ready_runtime()
        self.node = {
            "name": data.NODE,
            "uid": "node-uid",
            "boot_id": data.BOOT_BEFORE,
            "ready": "True",
            "unschedulable": False,
            "taints": [],
            "ownership_annotations": {},
            "gpu_allocatable": 8,
        }
        self.baseline = data.host_before()
        self.after = data.host_after()
        self.baseline_agent = data.agent_before()
        self.new_agent = data.agent_after()
        self.bounds = {
            "node_workflow_lifetime_seconds": 7200,
            "step_waiting_timeout_seconds": 300,
            "verify_waiting_limit_seconds": 300,
            "restore_waiting_limit_seconds": 300,
            "agent_maintenance_window_seconds": 600,
            "gpu_client_verify_max_attempts": 6,
        }
        self.predecessor_valid = True
        self.tests_pass = True
        self.preflight_reboot: dict[str, Any] = {"armed": False}
        self.residuals = {"pod": False}
        self.injected = False
        self.reboot_armed = False
        self.pre_authorization: dict[str, Any] | None = None
        self.rebooted = False
        # What the on-node fire record says once the node is back: when the
        # host fired and which ledger rows it matched. ``host_fired_early``
        # makes the host claim a fire before the runner's barrier observation.
        self.fired_at: str | None = None
        self.host_fired_early = False
        self.host_condition_workflow = data.REQUEST
        # ``node_unreachable`` keeps the node from ever returning Ready, i.e.
        # kubelet never answers again during cleanup; ``unreachable_boot_id``
        # is what a later node read then shows (a reboot cleanup could not see).
        self.node_unreachable = False
        self.unreachable_boot_id: str | None = None
        self.recovered = False
        self.waiting_reads = 0
        self.pin_missing = False
        self.verify_missing = False
        self.pin_delays = 0
        self.verify_delays = 0
        self.agent_delays = 0
        self.terminal_delays = 0
        self.authorization_drift = False
        self.bad_drill = False
        self.restore_status = "SUCCEEDED"
        self.fenced = data.fenced_workflow()
        self.quiesce = data.quiesce_execution()
        self.quiesce["details"].update(
            maintenance_window_started_at=NOW.isoformat(),
            maintenance_window_expires_at=(NOW + timedelta(seconds=600)).isoformat(),
        )
        self.support = {
            "workflow": data.support_workflow(),
            "incident": data.support_incident(),
        }
        self.extra_escalations: dict[str, Any] = {}
        # Open incidents the preflight reads by state; a rerun that trips over
        # one is refused before the holder is armed.
        self.open_incidents: list[dict[str, Any]] = []
        # The support successor owns the node's isolation until its validated
        # restore; the fenced record is then closed on isolation evidence.
        self.close_refusal: str | None = None
        self.events: list[dict[str, Any]] = []
        self.reconcile = {
            "mode": verdicts.RETIRED_GENERATION_PLAN_MODE,
            "items": [],
            "plan_sha256": "a" * 64,
        }
        self.regional = FenceRegional(self)
        self.warm = FenceWarm(self)
        self.host = FenceProbe(self, "host")
        self.fence = FenceProbe(self, "fence")
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
        monkeypatch.setattr(authorization, "datetime", self.clock)

    def probe(self, settings: Any) -> Any:
        self.call(
            "probe.build",
            {"script": settings.probe_script.name, "run_id": settings.run_id},
        )
        return self.fence if settings.run_id.endswith("-fence") else self.host

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

    def names(self) -> list[str]:
        return [name for name, _ in self.calls]

    def host_execs_between(self, start: str, end: str) -> list[str]:
        """Probe execs recorded after the first ``start`` and before the first
        ``end`` call: kubelet is down there, so this must stay empty."""

        names = self.names()
        first = names.index(start)
        last = names.index(end) if end in names else len(names)
        return [
            name
            for name in names[first:last]
            if name.startswith(("fence.", "host.")) and not name.endswith(".create")
        ]

    def fire_record(self) -> dict[str, Any]:
        """The host's durable record of the reboot it fired, read back later."""

        if self.fired_at is None:
            return {}
        return {
            "fire_requested_at": self.fired_at,
            "condition": {
                "workflow_request_id": self.host_condition_workflow,
                "incident_id": data.INCIDENT,
                "boot_id": data.BOOT_BEFORE,
                "agent_generation": data.GENERATION,
                "quiesce_command_id": f"key-QUIESCE_GPU_SERVICES/{data.NODE}/agent-7",
                "verify_command_id": f"key-VERIFY_NO_GPU_CLIENTS/{data.NODE}/agent-7",
                "observed_at": self.fired_at,
            },
        }

    def waiting_state(self) -> dict[str, Any]:
        self.waiting_reads += 1
        self.call("waiting.read", self.waiting_reads)
        quiesce = (
            []
            if self.pin_missing or self.waiting_reads <= self.pin_delays
            else [deepcopy(self.quiesce)]
        )
        verifying = (
            not self.verify_missing
            and self.waiting_reads > self.pin_delays + self.verify_delays
        )
        verify = (
            [
                {
                    "operation": "VERIFY_NO_GPU_CLIENTS",
                    "step_index": 3,
                    "status": "WAITING",
                    "phase": "official",
                }
            ]
            if verifying
            else []
        )
        workflow = {
            "request_id": data.REQUEST,
            "incident_id": data.INCIDENT,
            "fencing_token": 5,
            "status": WorkflowStatus.RUNNING,
            "step_executions": [*quiesce, *verify],
        }
        if self.pin_missing or self.verify_missing:
            workflow["status"] = "FAILED"
        if self.authorization_drift and self.waiting_reads >= 3:
            workflow["request_id"] = "different-request"
        return {
            "workflow": workflow,
            "incident": {
                "incident_id": data.INCIDENT,
                "drill_id": "foreign-drill" if self.bad_drill else self.run_id,
            },
            "commands": [
                {
                    "workflow_request_id": data.REQUEST,
                    "idempotency_key": f"key-{operation}",
                    "step": {"operation": operation, "node_ids": [data.NODE]},
                }
                for operation in ("QUIESCE_GPU_SERVICES", "VERIFY_NO_GPU_CLIENTS")
            ],
        }


class FenceRegional:
    def __init__(self, harness: FenceHarness) -> None:
        self.h = harness
        self.settings = harness.settings.regional

    def node_snapshot(self, node: str) -> dict[str, Any]:
        self.h.call("node.snapshot", node)
        return deepcopy(self.h.node)

    def runtime_identity(self) -> dict[str, Any]:
        return deepcopy(self.h.runtime)

    def cpu_blast_snapshot(self) -> dict[str, Any]:
        return {"cpu": ["api-a"]}

    def executor_python(self, script: str) -> dict[str, Any]:
        self.h.call("bounds")
        return deepcopy(self.h.bounds)

    def business_workloads(self, node: str) -> list[dict[str, Any]]:
        return []

    def store_snapshot(self, **kwargs: Any) -> dict[str, Any]:
        self.h.call("store", kwargs)
        if not self.h.injected:
            return {
                "release_id": "release-test",
                "agent": deepcopy(self.h.baseline_agent),
                "profile": {
                    "profile_version": "profile-v1",
                    "capabilities": [
                        {
                            "capability": "gpuReset",
                            "mode": "OWN",
                            "owner": "gpu-fault-node-agent",
                        }
                    ],
                },
                "queue": {"depth": 0, "fault_backlog_depth": 0},
                "remote_commands": {},
            }
        if not self.h.rebooted:
            return self.h.waiting_state()
        agent = self.h.new_agent
        if self.h.agent_delays:
            self.h.agent_delays -= 1
            agent = self.h.baseline_agent
        incident = data.fenced_incident(
            state="RECOVERED" if self.h.recovered else "QUARANTINED"
        )
        if self.h.terminal_delays:
            self.h.terminal_delays -= 1
            incident["state"] = "ACTION_PENDING"
        return {
            "workflow": deepcopy(self.h.fenced),
            "incident": incident,
            "agent": deepcopy(agent),
        }

    def wait_node_ready(self, node: str, **kwargs: Any) -> dict[str, Any]:
        self.h.call("node.ready", {"node": node, **kwargs})
        if self.h.node_unreachable:
            if self.h.unreachable_boot_id:
                self.h.node["boot_id"] = self.h.unreachable_boot_id
            raise RegionalFixtureError("node did not return Ready: fake outage")
        if kwargs.get("expected_boot_id") and not self.h.rebooted:
            # The host reboots on its own once its ledger shows the barrier,
            # no earlier than the reboot delay after the holder opened.
            self.h.clock.sleep(self.h.settings.reboot_delay_seconds)
            fired = self.h.clock.now()
            if self.h.host_fired_early:
                fired -= timedelta(seconds=self.h.settings.reboot_delay_seconds + 1)
            self.h.fired_at = fired.isoformat()
            self.h.rebooted = True
            self.h.node["boot_id"] = data.BOOT_AFTER
        return deepcopy(self.h.node)

    def cpu_python(self, script: str, *args: str) -> dict[str, Any]:
        if script == OPEN_INCIDENTS_PROBE:
            self.h.call("open_incidents", args)
            return {"open_incidents": deepcopy(self.h.open_incidents)}
        if script == case.AFTERMATH_PROBE:
            self.h.call("aftermath", args)
            # The support successor has quarantined the node by now: cleanup
            # restores through it and closes the fenced record on evidence.
            self.h.node.update(
                unschedulable=True,
                taints=[{"key": verdicts.QUARANTINE_TAINT, "value": "abc"}],
                ownership_annotations={
                    "gpu-fault.io/incident-id": (
                        self.h.support["incident"]["incident_id"]
                    )
                },
            )
            return {
                "fenced": deepcopy(self.h.fenced),
                "incident": data.fenced_incident(
                    state="RECOVERED" if self.h.recovered else "QUARANTINED"
                ),
                "recovery_successors": [data.recovery_successor()]
                if self.h.recovered
                else [],
                "commands": [
                    {"status": "FAILED", "step": {"operation": "VERIFY_NO_GPU_CLIENTS"}}
                ],
            }
        if script == case.ESCALATION_PROBE:
            self.h.call("escalations", args)
            return {
                "support": deepcopy(self.h.support),
                **deepcopy(self.h.extra_escalations),
            }
        self.h.call("reconcile.plan")
        return deepcopy(self.h.reconcile)

    def provider_events(self, *args: Any) -> list[dict[str, Any]]:
        self.h.call("provider.events")
        return deepcopy(self.h.events)

    def verify_runtime_identity(self, identity: dict[str, Any], **kwargs: Any) -> None:
        self.h.call("runtime.verify", kwargs["stage"])
        assert identity == self.h.runtime, "cleanup must remain release-bound"


class FenceProbe:
    host_script = "/fake/barrier.py"

    def __init__(self, harness: FenceHarness, label: str) -> None:
        self.h = harness
        self.label = label

    def create(self) -> None:
        self.h.call(f"{self.label}.create")

    def execute(self, action: str, *args: str, **kwargs: Any) -> dict[str, Any]:
        self.h.call(f"{self.label}.{action}", {"args": args, **kwargs})
        if action == "snapshot":
            return deepcopy(self.h.after if self.h.rebooted else self.h.baseline)
        if action == "reboot-status":
            if not self.h.rebooted:
                return deepcopy(self.h.preflight_reboot)
            return {**data.reboot_status(), **self.h.fire_record()}
        if action == "write-xid":
            self.h.injected = True
        if action == "pre-authorize-reboot":
            self.h.reboot_armed = True
            self.h.pre_authorization = json.loads(
                args[args.index("--authorization") + 1]
            )
        if action == "holder-status":
            return {
                "matched_row": {"command_id": "owned-quiesce"},
                "hold_started_at": NOW.isoformat(),
            }
        return {"action": action, "ok": True}

    def cleanup(self) -> dict[str, bool]:
        self.h.call(f"{self.label}.cleanup")
        return dict(self.h.residuals)


class FenceWarm:
    def __init__(self, harness: FenceHarness) -> None:
        self.h = harness

    def wait_agent_active(self, node: str, **kwargs: Any) -> dict[str, Any]:
        self.h.call("agent.active", node)
        return deepcopy(self.h.new_agent)

    def wait_incident_idle(self, incident: str) -> None:
        self.h.call("incident.idle", incident)

    def incident_by_id(self, incident_id: str) -> dict[str, Any]:
        self.h.call("incident.read", incident_id)
        if incident_id == data.INCIDENT:
            state = "RECOVERED" if self.h.recovered else "QUARANTINED"
        else:
            state = str(self.h.support["incident"]["state"])
        return {"incident_id": incident_id, "state": state}

    def close_incident_with_evidence(
        self, incident_id: str, **kwargs: Any
    ) -> dict[str, Any]:
        self.h.call("incident.close", {"incident_id": incident_id, **kwargs})
        if self.h.close_refusal:
            return {"closed": False, "refusal": self.h.close_refusal, "state": None}
        return {"closed": True, "refusal": None, "state": "RECOVERED"}

    def create_restore_workflow(self, **kwargs: Any) -> dict[str, Any]:
        self.h.call("restore.create", kwargs)
        return {"workflow_request_id": f"restore-{kwargs['incident_id']}"}

    def wait_workflow_id(self, request_id: str) -> dict[str, Any]:
        self.h.call("restore.wait", request_id)
        if self.h.restore_status == "SUCCEEDED":
            self.h.node.update(unschedulable=False, taints=[], ownership_annotations={})
        return {"status": self.h.restore_status}
