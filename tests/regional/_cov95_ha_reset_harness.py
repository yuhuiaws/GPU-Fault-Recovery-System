from __future__ import annotations

import copy
import functools
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gpu_fault.models import (
    FaultIncident,
    IncidentState,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowStepExecution,
    WorkflowStepSpec,
    WorkflowStepStatus,
)
from gpu_fault.regional import RemoteActionCommand, RemoteCommandStatus
from scripts.e2e.regional import run_destr001_gpu_reset as reset
from scripts.e2e.regional import run_ha003_aurora_failover_reset as ha003
from scripts.e2e.regional.regional_commands import RegionalFixtureError
from scripts.e2e.regional.regional_live_fixture import RegionalLiveSettings
from tests.regional._cov95_ha001_harness import DEADLINE, Clock
from tests.regional._cov95_ha002_harness import Watchdog
from tests.regional.test_ha_resource_guards import WindowApi


def workflow_state() -> dict[str, Any]:
    steps = [
        WorkflowStepSpec(
            operation=WorkflowOperation(name),
            execution_owner="gpu-fault-node-agent",
            node_ids=["node-a"],
            gpu_uuids=["GPU-a"],
        )
        for name in reset.EXPECTED_STEPS
    ]
    incident = FaultIncident(
        incident_id="incident-unit",
        event_id="event-unit",
        event_type="XID",
        cluster_id="unit-cluster",
        node_ids=["node-a"],
        policy_version="unit-policy",
        policy_source="UNIT",
        state=IncidentState.ACTION_PENDING,
        fencing_token=1,
    )
    workflow = WorkflowRequest(
        request_id="workflow-unit",
        incident_id=incident.incident_id,
        status=WorkflowStatus.SUCCEEDED,
        official_action="RESET_GPU",
        fencing_token=1,
        official_steps=steps,
        completed_operations=[step.operation for step in steps],
        step_executions=[
            WorkflowStepExecution(
                step_index=index,
                operation=step.operation,
                status=WorkflowStepStatus.SUCCEEDED,
                adapter_operation_id=f"remote/command-{index}",
            )
            for index, step in enumerate(steps)
        ],
    )
    commands = [
        RemoteActionCommand(
            command_id=f"command-{index}",
            cluster_id=incident.cluster_id,
            incident_id=incident.incident_id,
            workflow_request_id=workflow.request_id,
            step_index=index,
            fencing_token=1,
            idempotency_key=f"workflow-unit/{index}",
            step=step,
            incident=incident,
            workflow=workflow,
            status=RemoteCommandStatus.SUCCEEDED,
            last_lease_owner="unit/pod-1",
        ).model_dump(mode="json")
        for index, step in enumerate(steps)
    ]
    return {
        "event": {"xid": 46, "evidence_ref": "kmsg://unit-synthetic"},
        "decision": {"official_action": "RESET_GPU"},
        "incident": incident.model_dump(mode="json"),
        "workflow": workflow.model_dump(mode="json"),
        "commands": commands,
        "agent": {"generation": 1},
    }


def host_documents() -> tuple[dict[str, Any], dict[str, Any]]:
    inventory = [{"uuid": "GPU-a", "pci_bdf": "0000:59:00.0"}]
    before = {
        "gpu_inventory": inventory,
        "compute_clients": [],
        "ledger": [],
        "gpu_fault_timers": [],
        "quiesce_states": [],
        "services": {"unit-service": {"ActiveState": "active"}},
    }
    after = {
        **copy.deepcopy(before),
        "ledger": [
            {
                "command_id": "workflow-unit/4/node-a/agent-1",
                "attempt": 1,
                "state": "SUCCEEDED",
                "operation": "RESET_GPU",
                "gpu_uuids": ["GPU-a"],
            }
        ],
        "sampler": {
            "sample_count": 3,
            "min_gpu_count": 0,
            "observed_gpu_uuid_sets": [["GPU-a"], [], ["GPU-a"]],
            "last": {"gpu_count": 1, "gpu_uuids": ["GPU-a"]},
        },
    }
    return before, after


class Host:
    host_script = "/unit/probe.py"

    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.before, self.after = host_documents()
        self.snapshots = 0
        self.failure_at: dict[str, BaseException] = {}
        self.residuals = {"pod": False, "configmap": False}
        self.arguments: list[tuple[str, tuple[str, ...]]] = []
        self.settings: Any = None

    def action(self, name: str) -> None:
        self.events.append(name)
        if name in self.failure_at:
            raise self.failure_at[name]

    def create(self) -> None:
        self.action("host-create")

    def execute(self, operation: str, *args: str, **kwargs: Any) -> dict[str, Any]:
        self.arguments.append((operation, args))
        self.action(operation)
        if operation == "snapshot":
            self.snapshots += 1
            return copy.deepcopy(self.before if self.snapshots == 1 else self.after)
        return {"observed": True}

    def cleanup(self) -> dict[str, bool]:
        self.action("host-cleanup")
        return self.residuals.copy()


class RegionalAPI(WindowApi):
    def __init__(self, path: Path, module: Any, events: list[str]) -> None:
        super().__init__(path)
        (path / "cpu").write_text("apiVersion: v1\n")
        (path / "gpu").write_text("apiVersion: v1\n")
        self.settings = RegionalLiveSettings(
            cpu_kubeconfig=path / "cpu",
            gpu_kubeconfig=path / "gpu",
            gpu_context="unit-gpu",
            namespace="unit",
            cluster_id="unit-cluster",
            region="us-east-1",
        )
        self.document["spec"]["replicas"] = 2
        self.document["status"].update(
            replicas=2, readyReplicas=2, updatedReplicas=2, availableReplicas=2
        )
        self.module, self.events = module, events
        self.node = {
            "uid": "node-uid",
            "ready": "True",
            "unschedulable": True,
            "ownership_annotations": {},
            "gpu_allocatable": 1,
        }
        self.baseline = {
            "release_id": "unit-release",
            "agent": {"lifecycle_state": "ACTIVE", "generation": 1},
            "profile": {
                "profile_version": "unit-profile",
                "capabilities": [
                    {
                        "capability": "gpuReset",
                        "mode": "OWN",
                        "owner": "gpu-fault-node-agent",
                        "adapter": "node-action",
                    }
                ],
            },
            "queue": {"depth": 0},
            "remote_commands": {"open_by_cluster": {}},
        }
        self.workloads: list[dict[str, Any]] = []
        self.final = workflow_state()
        self.states: list[dict[str, Any] | BaseException] = []
        statuses = (
            [
                ("PENDING", "pod-0"),
                ("WAITING", "pod-0"),
                ("WAITING", "pod-0"),
                ("WAITING", "pod-0"),
                ("SUCCEEDED", "pod-1"),
            ]
            if module is ha003
            else [
                ("PENDING", "pod-0"),
                ("WAITING", "pod-0"),
                ("WAITING", "pod-1"),
                ("SUCCEEDED", "pod-1"),
            ]
        )
        for status, owner in statuses:
            state = copy.deepcopy(self.final)
            command = state["commands"][4]
            command.update(status=status, last_lease_owner=f"unit/{owner}")
            self.states.append(state)
        self.state_reads = 0
        self.provider_events: list[dict[str, Any]] = []
        self.cpu_blast: dict[str, Any] = {"pods": ["unit-cpu"]}
        self.log_text = "INFO connection recovered\n"
        self.failover = False
        self.rds_reads = 0
        self.rds_members = 2
        self.rds_status = "available"
        # "writer" or "status": the describe answered at the barrier before the
        # failover request (reset injected, failover not yet requested) reports
        # a moved writer or a cluster that is not available.
        self.rds_drift_before_failover: str | None = None
        self.focused_returncode = 0
        self.removed = False
        self.replace_on_delete = False

    def ready_pods(self, plane: str, app: str) -> list[dict[str, str]]:
        if plane == "gpu":
            return [
                {
                    "name": "pod-0-replacement"
                    if self.removed and i == 0
                    else f"pod-{i}",
                    "uid": f"uid-{i}",
                }
                for i in range(self.document["spec"]["replicas"])
            ]
        return [{"name": f"{app}-unit", "uid": "cpu-uid"}]

    def node_snapshot(self, node: str) -> dict[str, Any]:
        assert node == "node-a"
        return copy.deepcopy(self.node)

    def store_snapshot(self, **kwargs: Any) -> dict[str, Any]:
        if "marker" not in kwargs:
            return copy.deepcopy(self.baseline)
        self.events.append("store-observe")
        index = min(self.state_reads, len(self.states) - 1)
        self.state_reads += 1
        value = self.states[index]
        if isinstance(value, BaseException):
            raise value
        return copy.deepcopy(value)

    def business_workloads(self, node: str) -> list[dict[str, Any]]:
        return copy.deepcopy(self.workloads)

    def cpu_blast_snapshot(self) -> dict[str, Any]:
        return copy.deepcopy(self.cpu_blast)

    def wait_for_workflow(self, **kwargs: Any) -> dict[str, Any]:
        self.events.append("workflow-terminal")
        return copy.deepcopy(self.final)

    def wait_provider_events(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return copy.deepcopy(self.provider_events)

    def provider_events_provisional(self, *args: Any) -> bool:
        return True

    def kubectl(self, plane: str, *args: str, **kwargs: Any) -> str:
        if args[0] == "logs":
            self.events.append(f"logs:{plane}")
            return self.log_text
        if args[:2] == ("delete", "--raw"):
            options = json.loads(kwargs["input_text"])
            assert plane == "gpu"
            assert args[2].endswith("/pods/pod-0"), (
                "only the captured executor owner may be deleted"
            )
            assert options["preconditions"] == {"uid": "uid-0"}
            assert options["gracePeriodSeconds"] == 0
            if self.replace_on_delete:
                raise RegionalFixtureError("unit API rejected replacement Pod UID")
            self.removed = True
            self.events.append("delete-owner")
            return ""
        return super().kubectl(plane, *args, **kwargs)

    def run(self, argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if argv[1:3] == ["-m", "pytest"]:
            self.events.append("focused-tests")
            return subprocess.CompletedProcess(
                argv, self.focused_returncode, "unit tests", ""
            )
        assert argv[:2] == ["aws", "rds"]
        if argv[2] == "failover-db-cluster":
            self.events.append("failover-request")
            self.failover = True
            return subprocess.CompletedProcess(
                argv, 0, json.dumps({"accepted": True}), ""
            )
        assert argv[2] == "describe-db-clusters"
        self.events.append("rds-describe")
        if self.failover:
            self.rds_reads += 1
        drift = (
            self.rds_drift_before_failover
            if "write-xid46" in self.events and not self.failover
            else None
        )
        cluster = {
            "Status": "failing-over"
            if self.rds_reads == 1 or drift == "status"
            else self.rds_status,
            "DBClusterMembers": [
                {
                    "DBInstanceIdentifier": name,
                    "IsClusterWriter": (index == 0)
                    != (self.rds_reads >= 2 or drift == "writer"),
                }
                for index, name in enumerate(
                    ["old-writer", "new-writer"][: self.rds_members]
                )
            ],
        }
        return subprocess.CompletedProcess(
            argv, 0, json.dumps({"DBClusters": [cluster]}), ""
        )


class RegionalFactory:
    def __init__(self, api: RegionalAPI) -> None:
        self.api = api

    def __call__(self, settings: Any) -> RegionalAPI:
        return self.api

    def run(self, *args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return self.api.run(*args, **kwargs)


class ResetHarness:
    def __init__(
        self, monkeypatch: pytest.MonkeyPatch, path: Path, module: Any
    ) -> None:
        self.module, self.path = module, path
        self.events: list[str] = []
        self.api = RegionalAPI(path, module, self.events)
        self.host = Host(self.events)
        self.clock = Clock()
        self.binding_reads = 0
        self.binding_failure_at: int | None = None
        self.reuse_tests = False
        self.watchdog = Watchdog()
        self.restore_status = "SUCCEEDED"
        self.restore_failure: BaseException | None = None
        self.deadline = DEADLINE
        self.directory = path / "cases" / module.CASE_ID
        self.directory.mkdir(parents=True)
        previous = path / "previous.json"
        previous.write_text(
            json.dumps(
                {
                    "case_id": module.PREDECESSOR_CASE_ID,
                    "verdict": "PASS",
                    **self.api.evidence_identity(),
                }
            )
        )
        values = {
            "regional": self.api.settings,
            "node": "node-a",
            "host_probe_image": "registry.invalid/probe@sha256:" + "a" * 64,
            "predecessor_path": previous,
        }
        if module is ha003:
            values["rds_cluster_id"] = "unit-rds"
        self.settings = module.Settings(**values)
        monkeypatch.setattr(module, "RegionalLiveFixture", RegionalFactory(self.api))
        monkeypatch.setattr(module, "HostProbeFixture", self.host_factory)
        monkeypatch.setattr(module, "WarmSpareLiveFixture", lambda *a: self)
        monkeypatch.setattr(
            module,
            "time",
            SimpleNamespace(
                monotonic=self.clock.monotonic,
                sleep=self.clock.sleep,
                time=lambda: 1_000_000,
            ),
        )
        monkeypatch.setattr(
            module,
            "reusable_focused_tests",
            lambda _: {"passed": True} if self.reuse_tests else None,
        )
        monkeypatch.setattr(
            module,
            "record_focused_tests",
            lambda details, tests: details.update(focused_tests=tests),
        )
        if module is ha003:
            monkeypatch.setattr(
                module,
                "regional_binding",
                lambda *a: SimpleNamespace(read=self.binding),
            )
            monkeypatch.setattr(
                module,
                "wait_rds_failover",
                functools.partial(module.wait_rds_failover, sleep=self.clock.sleep),
            )
        else:
            monkeypatch.setattr(
                module,
                "subprocess",
                SimpleNamespace(**{**vars(subprocess), "Popen": self.spawn}),
            )
            monkeypatch.setattr(
                module,
                "os",
                SimpleNamespace(**{**vars(module.os), "killpg": self.killpg}),
            )
        preflight = module.read_only_preflight(self.settings, self.directory)
        assert preflight["errors"] == [], preflight["errors"]
        details = module.plan_details(self.settings, preflight)
        (self.directory / "plan.json").write_text(json.dumps({"details": details}))
        self.events.clear()
        self.binding_reads = 0

    def binding(self, expected: dict[str, Any] | None = None) -> dict[str, Any]:
        self.binding_reads += 1
        self.events.append("binding")
        if self.binding_reads == self.binding_failure_at:
            raise RegionalFixtureError("unit Aurora binding drift")
        value = {"identity": {"database": {"resource_id": "unit-rds"}}}
        if expected is not None:
            assert expected == value
        return value

    def host_factory(self, settings: Any) -> Host:
        self.host.settings = settings
        return self.host

    def spawn(self, *args: Any, **kwargs: Any) -> Watchdog:
        self.events.append("watchdog-start")
        return self.watchdog

    def killpg(self, pid: int, sig: int) -> None:
        assert pid == self.watchdog.pid
        self.events.append("watchdog-stop")

    def create_restore_workflow(self, **kwargs: Any) -> dict[str, str]:
        self.events.append("node-restore")
        assert kwargs["incident_id"] == "incident-unit"
        if self.restore_failure is not None:
            raise self.restore_failure
        return {"workflow_request_id": "restore-unit"}

    def wait_workflow_id(self, workflow_id: str) -> dict[str, str]:
        assert workflow_id == "restore-unit"
        return {"status": self.restore_status}

    def execute(self) -> tuple[int, dict[str, Any]]:
        code = self.module.execute_case(self.settings, self.path, 1, self.deadline)
        report = json.loads(
            (self.directory / f"{self.module.CASE_ID}.json").read_text()
        )
        return code, report
