from __future__ import annotations

import json
from copy import deepcopy
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import control_plane_env_window, executor_env_window
from scripts.e2e.regional import destr014_recovery as recovery
from scripts.e2e.regional import run_destr014_branch_exhaustion as case
from scripts.e2e.regional.destr014_verdicts import quarantine_taint_value
from scripts.e2e.regional.host_probe_fixture import HostProbeSettings
from scripts.e2e.regional.probes import destr014_recovery_probe as host_probe
from scripts.e2e.regional.warm_spare_fixture import (
    INSTANCE_GROUP_LABEL,
    QUARANTINE_TAINT,
)
from tests.regional import test_destr014_branch_exhaustion as data
from tests.regional._cov95_destr_branches import (
    BranchHarness,
    BranchRegional,
    BranchWarm,
    BranchWorkload,
)
from tests.regional._cov95_destr_exhaustion_recovery import (
    RecoveryClock,
    RecoveryTransport,
)
from tests.regional._cov95_destr_warm import NOW
from tests.regional.test_destr014_recovery_probe import HostHarness

RECOVERY_IDENTITY_REFUSAL = (
    "sibling Node Agent recovery identity is incomplete or mismatched"
)


class ExhaustionHarness(BranchHarness):
    settings: Any
    nodes: dict[str, dict[str, Any]]

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        super().__init__(tmp_path, monkeypatch)
        self.settings = case.Settings(
            self.settings.regional,
            self.settings.site_file,
            case.DEFAULT_MANIFEST,
            "fake-hyperpod",
            "example.test/probe",
            data.FAULT,
            "0000:01:00.0",
            "/dev/nvidia0",
            data.SIBLING,
            "0000:02:00.0",
            "destr014-job",
            "destr014-job-a001",
            6,
            600,
            "unknown-reboot",
            tmp_path / "previous.json",
        )
        for node in self.nodes.values():
            node.update(
                labels={
                    INSTANCE_GROUP_LABEL: "group-a",
                    "node.kubernetes.io/instance-type": "p4d.24xlarge",
                },
                annotations={},
            )
        from tests.regional.test_acceptance_reboot_hold_alignment import unknown_reboot

        self.workflow = unknown_reboot()
        self.workflow["terminal_failure_reason"] = (
            f"node branch escalation exhausted: branch:{data.SIBLING}"
        )
        self.incident = data.happy_incident()
        self.support_incident = "support-owned"
        self.incident_state = "QUARANTINED"
        self.pending_reads = 0
        self.rebooted = False
        self.agent_settle_reads = 0
        self.directory = tmp_path
        host_directory = tmp_path / "recovery-host"
        host_directory.mkdir(mode=0o700)
        self.host = HostHarness(host_directory, monkeypatch)
        self.host.scope.update(
            cluster_id=self.settings.regional.cluster_id,
            node=data.SIBLING,
            node_uid=self.nodes[data.SIBLING]["uid"],
            boot_id=self.nodes[data.SIBLING]["boot_id"],
            profile_version=self.profile["profile_version"],
        )
        self.host.boot = self.host.scope["boot_id"]
        host_probe.AGENT_ENV.write_text(
            "\n".join(
                f"{name}={self.host.scope[key]}"
                for key, name in host_probe.PIN_KEYS.items()
            ),
            encoding="utf-8",
        )
        self.clock = RecoveryClock(self.host)
        self.agents = {
            node: {
                "node_id": node,
                "node_instance_id": snapshot["uid"],
                "lifecycle_state": "ACTIVE",
                "artifact_sha256": self.host.scope["artifact_sha256"],
                "installer_bundle_sha256": self.host.scope["bundle_sha256"],
                "runtime_profile_version": self.host.scope["profile_version"],
            }
            for node, snapshot in self.nodes.items()
        }
        self.recovery_transport = RecoveryTransport(self, self.host)
        self.provider = {"count": 2, "sha256": "f" * 64}
        self.events = [
            {
                "event_name": "BatchRebootClusterNodes",
                "session_issuer_role_name": "executor",
            }
        ] * 2
        self.regional = ExhaustionRegional(self)
        self.workload = ExhaustionWorkload(self)
        self.warm = ExhaustionWarm(self)
        monkeypatch.setattr(case, "RegionalLiveFixture", lambda _s: self.regional)
        monkeypatch.setattr(case, "WarmSpareLiveFixture", lambda *_a: self.warm)
        monkeypatch.setattr(case, "ManagedWorkloadFixture", lambda *_a: self.workload)
        monkeypatch.setattr(case, "ImagePrewarmFixture", lambda *_a, **_k: self.prewarm)
        monkeypatch.setattr(case, "HostProbeFixture", self.make_probe)
        monkeypatch.setattr(case, "predecessor_evidence", self.predecessor)
        monkeypatch.setattr(case, "time", self.clock)
        monkeypatch.setattr(case, "datetime", self.clock)
        self.patch_windows(monkeypatch)

    @property
    def journal_path(self) -> Path:
        return (
            self.directory / "cases" / case.CASE_ID / "destr014-recovery-journal.json"
        )

    @property
    def disabled(self) -> bool:
        return not self.host.enable.exists()

    def make_probe(
        self, settings: HostProbeSettings
    ) -> ExhaustionProbe | RecoveryTransport:
        if settings.probe_script == recovery.PROBE:
            return self.recovery_transport
        return ExhaustionProbe(self, settings.node)

    def patch_windows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for module, label in (
            (control_plane_env_window, "control"),
            (executor_env_window, "executor"),
        ):
            monkeypatch.setattr(module, "survey", lambda _regional: {})
            monkeypatch.setattr(
                module, "open_window", partial(self.window_action, label, "open")
            )
            monkeypatch.setattr(
                module, "close_window", partial(self.window_action, label, "close")
            )
        monkeypatch.setattr(
            control_plane_env_window,
            "replica_env",
            lambda *_args, **_kwargs: [
                {"values": {"GPU_FAULT_BRANCH_ESCALATION_MAX_RUNGS": "2"}}
            ],
        )
        monkeypatch.setattr(
            control_plane_env_window, "without_survey", lambda value: value
        )

    def window_action(self, label: str, action: str, *args: Any) -> dict[str, bool]:
        if action == "open":
            saved = json.loads(self.journal_path.read_text())
            field = "control_env_opened" if label == "control" else "env_opened"
            assert saved["run"][field] is True, (
                "window intent must be durable before opening"
            )
        self.call(f"{label}.{action}", args[-1] if action == "open" else None)
        return {"opened" if action == "open" else "closed": True}

    def plan(self, run_dir: Path) -> dict[str, Any]:
        self.directory = run_dir
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


class ExhaustionRegional(BranchRegional):
    h: ExhaustionHarness

    def kubectl(self, plane: str, *args: str, **kwargs: Any) -> str:
        self.h.call("kubectl", {"plane": plane, "args": args})
        if args[0] == "exec":
            return json.dumps(
                {
                    "poll": "5",
                    "spare_failover": "true",
                    "remote_state": "true",
                    "allow_replace": "false",
                    "allow_reboot": "true",
                }
            )
        if args[:2] == ("get", "configmap"):
            return json.dumps(
                {
                    "data": {
                        "state.json": json.dumps(
                            {"job_workflow_lifetime_seconds": 3600}
                        )
                    }
                }
            )
        raise AssertionError(f"unexpected fake exhaustion request: {args}")

    def pod_python(self, *args: Any) -> dict[str, Any]:
        self.h.call("budget.read")
        return {"readable": True, "scopes": {"region": {"limit": 2, "active": 0}}}

    def store_snapshot(self, **kwargs: Any) -> dict[str, Any]:
        if not kwargs.get("marker"):
            return {"release_id": "release-test", "profile": deepcopy(self.h.profile)}
        self.h.call("workflow.read")
        workflow = deepcopy(self.h.workflow)
        if self.h.pending_reads:
            self.h.pending_reads -= 1
            workflow["status"] = "RUNNING"
        return {
            "workflow": workflow,
            "incident": deepcopy(self.h.incident),
            "restart_budget": {"restart_count": 0},
        }

    def wait_for_workflow(self, **kwargs: Any) -> dict[str, Any]:
        self.h.call("workflow.wait", kwargs)
        request = (
            "different-workflow"
            if self.h.split and kwargs["node"] == data.SIBLING
            else data.REQUEST
        )
        return {
            "workflow": {"request_id": request, "dag_enabled": True},
            "incident": {"incident_id": data.INCIDENT},
        }

    def release_id(self) -> str:
        return "release-test"

    def cpu_python(self, script: str, request_id: str) -> dict[str, Any]:
        self.h.call("support.read", request_id)
        return {
            "incident": {
                "incident_id": self.h.support_incident,
                "node_ids": [data.SIBLING],
            },
            "workflow": {"official_steps": [{"operation": "ESCALATE_SUPPORT"}]},
        }

    def wait_node_ready(self, node: str, **kwargs: Any) -> dict[str, Any]:
        self.h.call("node.ready", node)
        self.h.rebooted = True
        boot_id = f"new-boot-{node}"
        self.h.nodes[node]["boot_id"] = boot_id
        if node == data.SIBLING:
            self.h.host.reboot()
            self.h.host.boot = boot_id
        sibling = self.h.nodes[data.SIBLING]
        sibling.update(
            unschedulable=True,
            taints=[
                {
                    "key": QUARANTINE_TAINT,
                    "effect": "NoSchedule",
                    "value": quarantine_taint_value(self.h.support_incident),
                }
            ],
            ownership_annotations={"gpu-fault.io/incident-id": self.h.support_incident},
        )
        return deepcopy(self.h.nodes[node])


class ExhaustionWarm(BranchWarm):
    h: ExhaustionHarness

    def __init__(self, harness: ExhaustionHarness) -> None:
        super().__init__(harness)
        self.regional = harness.regional
        self.restoring: dict[str, Any] = {}

    def node_snapshot(self, node: str) -> dict[str, Any]:
        return deepcopy(self.h.nodes[node])

    def store_snapshot(self) -> dict[str, Any]:
        return {"agents": deepcopy(list(self.h.agents.values()))}

    def provider_inventory(self) -> dict[str, Any]:
        self.h.call("provider.inventory")
        return deepcopy(self.h.provider)

    def spare_nodes(self) -> list[str]:
        return []

    def cluster_recovery(self) -> dict[str, str]:
        return {"status": "InService", "node_recovery": "None"}

    def reactivate_agent(self, node: str) -> dict[str, Any]:
        self.h.call("agent.reactivate", node)
        return {"active": True}

    def create_restore_workflow(self, **kwargs: Any) -> dict[str, Any]:
        self.restoring = dict(kwargs)
        return super().create_restore_workflow(**kwargs)

    def wait_workflow_id(self, request: str) -> dict[str, Any]:
        result = super().wait_workflow_id(request)
        if result["status"] == "SUCCEEDED":
            self.h.nodes[self.restoring["node"]].update(
                unschedulable=False, taints=[], ownership_annotations={}
            )
            if self.restoring["incident_id"] == data.INCIDENT:
                self.h.incident_state = "RECOVERED"
        return result

    def incident_by_id(self, incident: str) -> dict[str, str]:
        return {"state": self.h.incident_state}


class ExhaustionWorkload(BranchWorkload):
    h: ExhaustionHarness

    def pods(self) -> list[dict[str, Any]]:
        self.h.call("workload.pods")
        return []


class ExhaustionProbe:
    host_script = "/fake/probe.py"

    def __init__(self, harness: ExhaustionHarness, node: str) -> None:
        self.h = harness
        self.node = node

    def create(self) -> None:
        self.h.call("probe.create", self.node)

    def execute(self, action: str, *args: str, **kwargs: Any) -> dict[str, Any]:
        self.h.call(f"probe.{action}", {"node": self.node, "args": args})
        assert action not in {"disable-agent-restart", "restore-agent"}, (
            "legacy unbound Agent mutation must never be used"
        )
        if action in {"write-xid46", "write-xid79"}:
            saved = json.loads(self.h.journal_path.read_text())
            assert saved["run"]["injection_started"] is True
            assert saved["host_ack"]["phase"] == "DISABLED"
            self.h.injected.add(self.node)
        elif action == "holder-status":
            return {
                "matched_row": {"command_id": "verify-owned"},
                "hold_started_at": NOW.isoformat(),
            }
        elif action == "snapshot":
            sibling = self.node == data.SIBLING
            active = self.h.host.agent_active if sibling else True
            if (
                sibling
                and not self.h.disabled
                and self.h.rebooted
                and self.h.agent_settle_reads
            ):
                self.h.agent_settle_reads -= 1
                active = False
            return {
                "boot_id": self.h.nodes[self.node]["boot_id"],
                "agent_unit": {
                    "ActiveState": "active" if active else "inactive",
                    "UnitFileState": "disabled"
                    if sibling and self.h.disabled
                    else "enabled",
                },
                "ledger": [
                    {
                        "command_id": "reset-owned",
                        "operation": "RESET_GPU",
                        "state": "FAILED",
                    }
                ]
                if self.h.rebooted and self.node == data.FAULT
                else [],
            }
        return {"action": action, "ok": True}

    def cleanup(self) -> dict[str, bool]:
        self.h.call("probe.cleanup", self.node)
        return dict(self.h.probe_residuals)
