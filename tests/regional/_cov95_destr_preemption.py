from __future__ import annotations

import json
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import destr_barrier_authorization as authorization
from scripts.e2e.regional import run_destr016_preempting_reboot as case
from scripts.e2e.regional.host_probe_fixture import HostProbeError
from scripts.e2e.regional.regional_live_fixture import RegionalFixtureError
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
        self.bounds = {
            "node_workflow_lifetime_seconds": 3600,
            "step_waiting_timeout_seconds": 600,
            "verify_waiting_limit_seconds": 600,
            "restore_waiting_limit_seconds": 600,
            "agent_maintenance_window_seconds": 420,
            "gpu_client_verify_max_attempts": 60,
        }
        self.node = {**data.node_baseline(), "boot_id": "boot-before"}
        self.baseline = {**data.host_baseline(), "boot_id": "boot-before"}
        self.after = {**data.host_after(), "boot_id": "boot-after"}
        self.injected = False
        self.rebooted = False
        # The conditional pre-authorization the holder probe received, and the
        # host's fire records (one per phase) that holder-status reads back
        # after the reboot. ``host_fired_early`` makes the host claim a fire
        # before the runner's barrier observation; ``node_unreachable`` keeps
        # kubelet from ever answering again during cleanup.
        self.pre_authorization: dict[str, Any] | None = None
        self.fired: dict[str, str] = {}
        self.host_fired_early = False
        self.node_unreachable = False
        self.unreachable_boot_id: str | None = None
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
            if name.startswith(("holder.", "injector."))
            and not name.endswith(".create")
        ]

    def probe(self, settings: Any) -> Any:
        return PreemptionProbe(
            self, "injector" if settings.case_id.endswith("-inject") else "holder"
        )

    def fire(self, phase: str) -> None:
        """The host fires ``phase`` now: one clock tick after the store read
        that let the runner see the barrier, or before it when asked to."""

        if phase in self.fired:
            return
        self.clock.sleep(1)
        fired = self.clock.now()
        if self.host_fired_early:
            fired -= timedelta(minutes=5)
        self.fired[phase] = fired.isoformat()

    def fire_records(self) -> dict[str, Any]:
        return {
            phase: {
                "fire_requested_at": fired_at,
                "fired_at": fired_at,
                "condition": {
                    "workflow_request_id": data.RESET_ID,
                    "incident_id": data.INCIDENT,
                    "boot_id": "boot-before",
                    "agent_generation": 4,
                    "quiesce_command_id": f"quiesce-owned/{data.NODE}/agent-4",
                    "verify_command_id": f"verify-owned/{data.NODE}/agent-4",
                    "observed_at": fired_at,
                },
            }
            for phase, fired_at in self.fired.items()
        }

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
        if self.changed_predecessor and self.fired:
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

    def executor_python(self, script: str) -> dict[str, Any]:
        self.h.call("bounds")
        return deepcopy(self.h.bounds)

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
            self.h.fire("absorb")
            return self.h.barrier(absorbed=True)
        if marker.endswith("-e"):
            self.h.call("successor.read")
            if self.h.successor_delays:
                self.h.successor_delays -= 1
                return {}
            self.h.fire("escalate")
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
        if self.h.node_unreachable:
            if self.h.unreachable_boot_id:
                self.h.node["boot_id"] = self.h.unreachable_boot_id
            raise RegionalFixtureError("node did not return Ready: fake outage")
        if kwargs.get("expected_boot_id"):
            # Only a wait for a *new* boot models the provider reboot; the
            # plain Ready wait cleanup takes reboots nothing.
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
        if action == "arm-holder":
            self.h.armed_drill_id = args[args.index("--drill-id") + 1]
            self.h.armed_device = args[args.index("--device") + 1]
        if action == "pre-authorize":
            proof = json.loads(args[args.index("--authorization") + 1])
            # The real probe (check_pre_authorization) refuses a proof whose
            # drill id or device is not the armed holder's; the live run of
            # 2026-09-18 died there with the reset injection's "-r" drill id.
            if proof.get("drill_id") != getattr(self.h, "armed_drill_id", None):
                raise HostProbeError(
                    "host probe failed (exit 1); output withheld"
                    " [fake: pre-authorization does not bind this holder]"
                )
            if proof.get("device") != getattr(self.h, "armed_device", None):
                raise HostProbeError(
                    "host probe failed (exit 1); output withheld"
                    " [fake: pre-authorization device is not the holder's]"
                )
            self.h.pre_authorization = proof
            self.h.call("pre-authorization", proof)
        if action == "holder-status":
            return {
                **deepcopy(self.h.holder_status),
                "injections_fired": (self.h.fire_records()),
            }
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
