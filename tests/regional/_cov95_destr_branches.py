from __future__ import annotations

import json
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import run_destr009_workload_restart as workload_case
from scripts.e2e.regional import run_destr015_parallel_branch_join as case
from scripts.e2e.regional.destr015_physical_evidence import (
    ResetIntervalScope,
    evidence_digest,
)
from scripts.e2e.regional.regional_live_fixture import RUNTIME_IDENTITY_DEPLOYMENTS
from tests.regional._cov95_destr_warm import NOW, Clock, profile, regional_settings
from tests.regional.test_acceptance_physical_interval_alignment import (
    BASE,
    SECOND,
    interval_fixture,
)
from tests.regional.test_destr015_parallel_branch_join import (
    INCIDENT,
    NODES,
    REQUEST,
    happy_budget,
    happy_incident,
    happy_workflow,
)


def ready_runtime() -> dict[str, Any]:
    return {
        "release_state": {"phase": "COMMITTED"},
        "deployments": {
            plane: {
                name: {
                    "generation": 1,
                    "desired_replicas": 1,
                    "observed_generation": 1,
                    "updated_replicas": 1,
                    "ready_replicas": 1,
                    "available_replicas": 1,
                }
                for name in names
            }
            for plane, names in RUNTIME_IDENTITY_DEPLOYMENTS.items()
        },
    }


def branch_host(node: str, *, after: bool = False) -> dict[str, Any]:
    return {
        "boot_id": f"boot-{node}",
        "gpu_inventory": [
            {"pci_bdf": f"0000:0{i}:00.0", "uuid": f"GPU-{node}-{i}"} for i in range(8)
        ],
        "quiesce_states": [],
        "gpu_fault_timers": [],
        "kmsg_writable": True,
        "services": {"kubelet.service": {"ActiveState": "active"}},
        "ledger": [
            {
                "command_id": f"{node}-{op}",
                "operation": op,
                "state": "SUCCEEDED",
                "attempt": 1,
                "workflow_request_id": REQUEST,
                "incident_id": INCIDENT,
                "fencing_token": 3,
                "agent_generation": 4,
                "gpu_uuids": [f"GPU-{node}-0"],
                "started_at": NOW.isoformat(),
                "completed_at": (NOW + timedelta(seconds=40)).isoformat(),
            }
            for op in (
                "QUIESCE_GPU_SERVICES",
                "VERIFY_NO_GPU_CLIENTS",
                "RESET_GPU",
                "RESTORE_GPU_SERVICES",
            )
        ]
        if after
        else [],
    }


def branch_pods(*, restarted: bool = False) -> list[dict[str, Any]]:
    return [
        {
            "uid": f"{'new' if restarted else 'old'}-{node}",
            "name": f"pod-{node}",
            "node": node,
            "phase": "Running",
            "ready": True,
        }
        for node in NODES
    ]


class BranchHarness:
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        site = tmp_path / "site.yaml"
        site.write_text("fake site\n", encoding="utf-8")
        self.settings = case.Settings(
            regional_settings(tmp_path),
            site,
            case.DEFAULT_MANIFEST,
            "example.test/probe",
            *NODES,
            "",
            "",
            "destr015-job",
            "destr015-job-a001",
            tmp_path / "predecessor.json",
        )
        self.clock = Clock()
        self.calls: list[tuple[str, Any]] = []
        self.failures: dict[str, BaseException] = {}
        self.advance_at: dict[str, float] = {}
        self.nodes = {
            node: {
                "name": node,
                "uid": f"uid-{node}",
                "boot_id": f"boot-{node}",
                "ready": "True",
                "unschedulable": False,
                "taints": [],
                "ownership_annotations": {},
                "gpu_allocatable": 8,
            }
            for node in NODES
        }
        self.agent = {
            "lifecycle_state": "ACTIVE",
            "allowed_operations": list(case.AGENT_OPERATIONS),
            "generation": 4,
        }
        self.profile = profile()
        self.profile["capabilities"].append(
            {"capability": "gpuReset", "mode": "OWN", "owner": "gpu-fault-node-agent"}
        )
        self.workflow = happy_workflow()
        self.workflow["fencing_token"] = 3
        self.incident = happy_incident()
        self.budget = happy_budget()
        self.runtime = ready_runtime()
        self.predecessor_valid = True
        self.tests_pass = True
        self.window = 5
        self.arbiters: list[dict[str, Any]] = []
        self.dns: list[dict[str, Any]] = [
            {"metadata": {}, "spec": {"nodeName": "outside"}}
        ]
        self.busy: list[dict[str, Any]] = []
        self.queue = {"depth": 0, "fault_backlog_depth": 0}
        self.commands: list[dict[str, Any]] = []
        self.physical_captures: dict[str, dict[str, Any]] = {}
        self.cached = list(NODES)
        self.baselines = {node: branch_host(node) for node in NODES}
        self.hosts_after = {node: branch_host(node, after=True) for node in NODES}
        self.probe_residuals: dict[str, bool] = {"pod": False}
        self.prewarm_residuals: dict[str, bool] = {"pod": False}
        self.workload_residual = ""
        self.injected: set[str] = set()
        self.split = False
        self.skew = 0
        self.pending_reads = 0
        self.cleanup_busy = False
        self.cpu = {"nodes": ["cpu-a"]}
        self.cpu_after = self.cpu
        self.events: list[dict[str, Any]] = []
        self.log_text = "healthy control plane"
        self.restore_status = "SUCCEEDED"
        self.regional = BranchRegional(self)
        self.workload = BranchWorkload(self)
        self.prewarm = BranchPrewarm(self)
        self.warm = BranchWarm(self)
        monkeypatch.setattr(case, "RegionalLiveFixture", lambda _s: self.regional)
        monkeypatch.setattr(case, "ManagedWorkloadFixture", lambda *_a: self.workload)
        monkeypatch.setattr(case, "ImagePrewarmFixture", lambda *_a, **_k: self.prewarm)
        monkeypatch.setattr(case, "WarmSpareLiveFixture", lambda *_a: self.warm)
        monkeypatch.setattr(
            case, "HostProbeFixture", lambda settings: BranchProbe(self, settings.node)
        )
        monkeypatch.setattr(case, "ResetIntervalWitness", BranchWitness)
        monkeypatch.setattr(case, "focused_tests", self.focused)
        monkeypatch.setattr(case, "predecessor_evidence", self.predecessor)
        monkeypatch.setattr(case, "replica_env", self.replica_env)
        monkeypatch.setattr(
            case, "record_focused_tests", lambda d, r: d.update(focused_tests=r)
        )
        for module in (case, workload_case):
            monkeypatch.setattr(module, "time", self.clock)
            monkeypatch.setattr(module, "datetime", self.clock)

    def call(self, name: str, detail: Any = None) -> None:
        self.calls.append((name, deepcopy(detail)))
        self.clock.sleep(self.advance_at.get(name, 0))
        if name in self.failures:
            raise self.failures[name]

    def focused(self, _path: Path, **kwargs: Any) -> dict[str, Any]:
        return {"passed": self.tests_pass, "focused_tests_reused": bool(kwargs)}

    def predecessor(self, _path: Path, case_id: str, **identity: Any) -> dict[str, Any]:
        self.call("predecessor", {"case_id": case_id, **identity})
        return {"valid": self.predecessor_valid}

    def replica_env(self, *_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        return [
            {
                "pod": "worker",
                "values": {
                    case.AGGREGATION_WINDOW_VARIABLE: str(self.window),
                    case.JOB_LIFETIME_VARIABLE: "3600",
                },
            }
        ]

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


class BranchWitness:
    """Complete physical receipts at the transport boundary, real verdict downstream."""

    def __init__(self, probe: Any, scope: ResetIntervalScope) -> None:
        self.h = probe.h
        self.scope = scope
        self.capture = interval_fixture()[0]["node-a"]
        shift = int(NOW.timestamp() * SECOND) - BASE
        start = self.capture["start"]
        start.update(scope_sha256=scope.digest(), witness_id=f"witness-{scope.node}")
        start["tracee"]["boot_id"] = scope.boot_id
        start["producer"]["boot_id"] = scope.boot_id
        end = self.capture["end"]
        end.update(**start, start_sha256=evidence_digest(start))
        end["wall_minus_monotonic_min_ns"] += shift
        end["wall_minus_monotonic_max_ns"] += shift
        action = end["actions"][0]
        action["gpu_uuid"] = scope.gpu_uuid
        delay = 0 if scope.node == NODES[0] else 5 * SECOND
        action["started_ns"] += shift + delay
        action["ended_ns"] += shift + delay
        self.h.physical_captures[scope.node] = self.capture

    def start(self) -> None:
        self.h.call("witness.start", self.scope.node)
        for row in self.h.hosts_after[self.scope.node]["ledger"]:
            if row["operation"] == "RESET_GPU":
                row["gpu_uuids"] = [self.scope.gpu_uuid]

    def poll(self) -> None:
        self.h.call("witness.poll", self.scope.node)

    def finish(self) -> dict[str, Any]:
        self.h.call("witness.finish", self.scope.node)
        return deepcopy(self.capture)

    def close(self) -> dict[str, bool]:
        self.h.call("witness.close", self.scope.node)
        return {"closed": True}


class BranchRegional:
    def __init__(self, harness: BranchHarness) -> None:
        self.h = harness
        self.settings = harness.settings.regional

    def evidence_identity(self) -> dict[str, str]:
        return {
            "release_id": "release-test",
            "cluster_id": "cluster-a",
            "region": "us-west-2",
        }

    def node_snapshot(self, node: str) -> dict[str, Any]:
        self.h.call("node.snapshot", node)
        return deepcopy(self.h.nodes[node])

    def store_snapshot(self, **kwargs: Any) -> dict[str, Any]:
        self.h.call("store", kwargs)
        if not kwargs.get("job_id"):
            return {
                "profile": deepcopy(self.h.profile),
                "release_id": "release-test",
                "agent": deepcopy(self.h.agent),
                "queue": deepcopy(self.h.queue),
                "remote_commands": {"open_by_cluster": {}},
            }
        if not kwargs.get("marker"):
            return {
                "observations": [
                    {
                        "workload_phase": "RUNNING",
                        "containers": [{"gpu_uuids": [f"GPU-{i}" for i in range(16)]}],
                    }
                ]
            }
        self.h.call("store.workflow")
        state = {
            "workflow": deepcopy(self.h.workflow),
            "incident": deepcopy(self.h.incident),
            "restart_budget": deepcopy(self.h.budget),
            "event": {"event_id": "event-owned"},
            "observations": [{"workload_phase": "STOPPED"}],
            "commands": deepcopy(self.h.commands),
        }
        if self.h.pending_reads:
            self.h.pending_reads -= 1
            state["workflow"]["status"] = "RUNNING"
        if self.h.cleanup_busy:
            state["commands"] = [{"status": "LEASED"}]
        return state

    def runtime_identity(self) -> dict[str, Any]:
        return deepcopy(self.h.runtime)

    def verify_runtime_identity(self, baseline: dict[str, Any], **kwargs: Any) -> None:
        self.h.call("runtime.verify", kwargs["stage"])
        assert baseline == self.h.runtime, "fake runtime identity must remain bound"

    def business_workloads(self, node: str) -> list[dict[str, Any]]:
        return []

    def gpu_workloads(self) -> list[dict[str, Any]]:
        return deepcopy(self.h.busy)

    def cpu_blast_snapshot(self) -> dict[str, Any]:
        return deepcopy(self.h.cpu_after if self.h.injected else self.h.cpu)

    def kubectl(self, plane: str, *args: str, **kwargs: Any) -> str:
        self.h.call("kubectl", {"plane": plane, "args": args, **kwargs})
        if args[0] == "logs":
            return self.h.log_text
        if args[:2] == ("get", "pod"):
            return json.dumps(
                {"items": self.h.dns if "kube-system" in args else self.h.arbiters}
            )
        if args[0] == "get" and "--ignore-not-found" in args:
            return self.h.workload_residual
        raise AssertionError(f"unhandled fake kubectl request: {args}")

    def ready_pods(self, plane: str, app: str) -> list[dict[str, Any]]:
        return [{"name": f"{app}-pod", "uid": "pod-uid"}]

    def wait_for_workflow(self, **kwargs: Any) -> dict[str, Any]:
        self.h.call("workflow.wait", kwargs)
        node = kwargs["node"]
        request = (
            "split-workflow"
            if self.h.split and node == NODES[1]
            else self.h.workflow["request_id"]
        )
        at = NOW + timedelta(seconds=self.h.skew if node == NODES[1] else 0)
        return {
            "event": {
                "xid": 46,
                "evidence_ref": f"kmsg://{node}/event",
                "observed_at": at.isoformat(),
            },
            "decision": {"official_action": "RESET_GPU"},
            "incident": {"workflow_request_id": request},
            "workflow": {"request_id": request},
        }

    def provider_events(self, *args: Any) -> list[dict[str, Any]]:
        self.h.call("provider.events")
        return deepcopy(self.h.events)

    def provider_events_provisional(self, *args: Any) -> bool:
        return True


class BranchWorkload:
    name = "job-owned"
    resource = "pytorchjob"

    def __init__(self, harness: BranchHarness) -> None:
        self.h = harness
        self.restart_state: dict[str, Any] | None = None

    def submit(self) -> dict[str, Any]:
        self.h.call("workload.submit")
        return {"submitted": True}

    def wait_running(self, **kwargs: Any) -> dict[str, Any]:
        self.h.call("workload.running", kwargs)
        return {"pods": branch_pods()}

    def authorize_restart(self, state: dict[str, Any]) -> None:
        self.h.call("workload.authorize_restart", state)
        self.restart_state = state

    def wait_restarted(self, uids: set[str], **kwargs: Any) -> dict[str, Any]:
        assert self.restart_state is not None
        self.h.call("workload.restarted", {"source_uids": uids, **kwargs})
        return {"pods": branch_pods(restarted=True)}

    def pods(self) -> list[dict[str, Any]]:
        self.h.call("workload.pods")
        return branch_pods(restarted=bool(self.h.injected))

    def delete(self) -> None:
        self.h.call("workload.delete")


class BranchPrewarm:
    def __init__(self, harness: BranchHarness) -> None:
        self.h = harness

    def create(self, nodes: list[str]) -> None:
        self.h.call("prewarm.create", nodes)

    def cached_nodes(self) -> list[str]:
        return self.h.cached

    def cleanup(self) -> dict[str, bool]:
        self.h.call("prewarm.cleanup")
        return dict(self.h.prewarm_residuals)


class BranchProbe:
    def __init__(self, harness: BranchHarness, node: str) -> None:
        self.h = harness
        self.node = node
        self.host_script = "/fake/probe.py"

    def create(self) -> None:
        self.h.call("probe.create", self.node)

    def execute(self, action: str, *args: str, **kwargs: Any) -> dict[str, Any]:
        self.h.call(f"probe.{action}", {"node": self.node, "args": args, **kwargs})
        if action == "snapshot":
            return deepcopy(
                self.h.hosts_after[self.node]
                if "--since-epoch" in args
                else self.h.baselines[self.node]
            )
        if action == "write-xid46":
            self.h.injected.add(self.node)
            return {"xid": 46, "node": self.node}
        raise AssertionError(f"unhandled fake host action: {action}")

    def cleanup(self) -> dict[str, bool]:
        self.h.call("probe.cleanup", self.node)
        return dict(self.h.probe_residuals)


class BranchWarm:
    def __init__(self, harness: BranchHarness) -> None:
        self.h = harness

    def wait_incident_idle(self, incident: str) -> dict[str, Any]:
        self.h.call("incident.idle", incident)
        return {"idle": True}

    def create_restore_workflow(self, **kwargs: Any) -> dict[str, Any]:
        self.h.call("restore.create", kwargs)
        return {"workflow_request_id": f"restore-{kwargs['node']}"}

    def wait_workflow_id(self, request: str) -> dict[str, Any]:
        self.h.call("restore.wait", request)
        return {"status": self.h.restore_status}
