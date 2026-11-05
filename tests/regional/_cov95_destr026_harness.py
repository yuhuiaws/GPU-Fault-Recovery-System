"""A DESTR-026 live run against fakes at the transport boundary.

The fakes stand in for the Kubernetes/Store reads (``RegionalLiveFixture``),
the host probe Pods, the managed PyTorchJob, the image prewarm and the
validated-restore workflow; everything downstream of them -- the DESTR-026,
DESTR-015 and DESTR-001 verdicts, the cleanup-quiescence wait, the control
plane log scan and the reachability decision -- is the real code. The pass
shapes are the synthetic records of ``test_destr026_same_node_two_gpu_reset``.

The fault node's kubelet can be held down after both XID writes
(``outage_seconds``): a probe exec or a Pod create into a NotReady node then
fails the way the real transport does, so a runner that execs too early fails
the test instead of passing on a fake that answers.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import destr_barrier_authorization as authorization
from scripts.e2e.regional import run_destr009_workload_restart as workload_case
from scripts.e2e.regional import run_destr015_parallel_branch_join as join_case
from scripts.e2e.regional import run_destr026_same_node_two_gpu_reset as case
from scripts.e2e.regional.destr015_verdicts import AGENT_OPERATIONS
from scripts.e2e.regional.regional_live_fixture import RegionalFixtureError
from tests.orchestration.test_same_node_job_reset_merge import LINEAR_PLAN
from tests.regional._cov95_destr_branches import ready_runtime
from tests.regional._cov95_destr_warm import NOW, Clock, profile, regional_settings
from tests.regional.test_destr026_same_node_two_gpu_reset import (
    GPU_A,
    GPU_B,
    NODE,
    PARTNER,
    SOURCE_ATTEMPT,
    event_state,
    gpu_nodes,
    hosts,
    partner_snapshot,
    pass_incident,
    pass_workflow,
    restarted_pods,
)

JOB_ID = "destr026-job"
RESTARTED_ATTEMPT = "destr026-job-a002"
MASTER_GPUS = [f"GPU-a{index}" for index in range(8)]
WORKER_GPUS = [f"GPU-b{index}" for index in range(8)]
BDF_A, BDF_B = "0000:59:00", "0000:5a:00"
# The remote steps the control plane delegated to the GPU-side executor.
REMOTE_OPERATIONS = [
    operation
    for operation in LINEAR_PLAN
    if operation not in {"FREEZE_EVIDENCE", "VALIDATE_GPU"}
]
QUARANTINE = {"key": "gpu-fault.io/quarantined", "effect": "NoSchedule"}


def node_snapshot(node: str) -> dict[str, Any]:
    return {
        "name": node,
        "uid": f"uid-{node}",
        "boot_id": "boot-a" if node == NODE else "boot-b",
        "ready": "True",
        "unschedulable": False,
        "taints": [],
        "ownership_annotations": {},
        "gpu_allocatable": "8",
    }


def source_pods() -> list[dict[str, Any]]:
    return [
        {"uid": f"old-{node}", "node": node, "phase": "Running", "ready": True}
        for node in (NODE, PARTNER)
    ]


def isolate(snapshot: dict[str, Any], incident: str) -> None:
    """What a failed job workflow leaves on its node: cordoned and owned."""

    snapshot.update(
        unschedulable=True,
        taints=[dict(QUARANTINE)],
        ownership_annotations={"gpu-fault.io/incident-id": incident},
    )


class Destr026Harness:
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        site = tmp_path / "site.yaml"
        site.write_text("fake site\n", encoding="utf-8")
        self.run_dir = tmp_path / "run-r9-hp"
        self.case_dir = self.run_dir / "cases" / case.CASE_ID
        self.settings = case.Settings(
            regional=regional_settings(tmp_path),
            site_file=site,
            manifest=join_case.DEFAULT_MANIFEST,
            host_probe_image="example.test/probe",
            fault_node=NODE,
            node_b=PARTNER,
            pci_bdf_a="",
            pci_bdf_b="",
            job_id=JOB_ID,
            attempt_id=SOURCE_ATTEMPT,
            predecessor_path=tmp_path / "predecessor.json",
        )
        self.clock = Clock()
        self.calls: list[tuple[str, Any]] = []
        self.failures: dict[str, BaseException] = {}
        self.advance_at: dict[str, float] = {}
        self.hooks: dict[str, Callable[[], None]] = {}
        self.nodes = {node: node_snapshot(node) for node in (NODE, PARTNER)}
        self.agent = {
            "lifecycle_state": "ACTIVE",
            "allowed_operations": list(AGENT_OPERATIONS),
            "generation": 4,
        }
        self.profile = profile()
        self.profile["capabilities"].append(
            {"capability": "gpuReset", "mode": "OWN", "owner": "gpu-fault-node-agent"}
        )
        self.runtime = ready_runtime()
        self.runtime_after_cleanup: dict[str, Any] | None = None
        self.predecessor_valid = True
        self.tests_pass = True
        self.window = 5
        self.recent_event: dict[str, Any] | None = None
        self.arbiters: list[dict[str, Any]] = []
        self.dns: list[dict[str, Any]] = [
            {"metadata": {}, "spec": {"nodeName": "cpu-node"}}
        ]
        self.queue = {"depth": 0, "fault_backlog_depth": 0}
        self.cpu = {"nodes": ["cpu-a"]}
        self.cpu_after = self.cpu
        self.cached = [NODE, PARTNER]
        self.allocation = list(MASTER_GPUS)
        before, after = hosts()
        self.baselines = {
            "a": {**before, "kmsg_writable": True},
            "partner": partner_snapshot(),
        }
        self.hosts_after = {
            "a": after,
            "b": {"kernel_reset_journal": {"target_reset_count": 1}},
            "partner": partner_snapshot(),
        }
        self.event_gpus = {"a": GPU_A, "b": GPU_B}
        self.split = False
        self.workflow = pass_workflow()
        self.incident = pass_incident()
        self.budget = {"restart_count": 1, "budget": 1}
        self.commands = [
            {"status": "SUCCEEDED", "step": {"operation": operation}}
            for operation in REMOTE_OPERATIONS
        ]
        self.pending_reads = 0
        self.cleanup_busy = False
        self.attempts_missing = 0
        self.attempt_reads = 0
        self.restarted = restarted_pods()
        self.gpu_nodes_before = gpu_nodes()
        self.gpu_nodes_after = gpu_nodes()
        self.events: list[dict[str, Any]] = []
        self.log_text = "healthy control plane"
        self.injected: set[str] = set()
        # Seconds the fault node stays NotReady after both XID writes (the
        # quiesce stops kubelet); ``None`` keeps it Ready, ``inf`` never back.
        self.outage_seconds: float | None = None
        self.not_ready_until: dict[str, float] = {}
        self.probe_residuals: dict[str, dict[str, bool]] = {}
        self.prewarm_residuals = {"pod": False}
        self.workload_residual = ""
        self.restore_status = "SUCCEEDED"
        self.regional = FakeRegional(self)
        self.workload = FakeWorkload(self)
        self.prewarm = FakePrewarm(self)
        self.warm = FakeWarm(self)
        self.probes: dict[str, FakeProbe] = {}
        monkeypatch.setattr(case, "RegionalLiveFixture", lambda _s: self.regional)
        monkeypatch.setattr(case, "ManagedWorkloadFixture", self.managed_workload)
        monkeypatch.setattr(case, "ImagePrewarmFixture", lambda *_a, **_k: self.prewarm)
        monkeypatch.setattr(case, "WarmSpareLiveFixture", lambda *_a: self.warm)
        monkeypatch.setattr(case, "HostProbeFixture", self.host_probe)
        monkeypatch.setattr(case, "focused_tests", self.focused)
        monkeypatch.setattr(case, "predecessor_evidence", self.predecessor)
        monkeypatch.setattr(
            case, "record_focused_tests", lambda d, r: d.update(focused_tests=r)
        )
        for module in (case, workload_case):
            monkeypatch.setattr(module, "time", self.clock)
            monkeypatch.setattr(module, "datetime", self.clock)
        # The reachability decision measures the same time the harness advances.
        monkeypatch.setattr(authorization, "datetime", self.clock)

    # -- bookkeeping ---------------------------------------------------------
    def call(self, name: str, detail: Any = None) -> None:
        self.calls.append((name, deepcopy(detail)))
        self.clock.sleep(self.advance_at.get(name, 0))
        hook = self.hooks.pop(name, None)
        if hook is not None:
            hook()
        if name in self.failures:
            raise self.failures[name]

    def names(self) -> list[str]:
        return [name for name, _ in self.calls]

    def details(self, name: str) -> list[Any]:
        return [detail for called, detail in self.calls if called == name]

    def node_ready(self, node: str) -> bool:
        return self.clock.elapsed >= self.not_ready_until.get(node, 0.0)

    def written(self, label: str) -> None:
        self.injected.add(label)
        if self.injected == {"a", "b"} and self.outage_seconds is not None:
            self.not_ready_until[NODE] = self.clock.elapsed + self.outage_seconds

    # -- fixture factories ---------------------------------------------------
    def managed_workload(self, regional: Any, settings: Any) -> FakeWorkload:
        self.call(
            "workload.fixture",
            {
                "restart_budget": settings.restart_budget,
                "expected_pods": settings.expected_pods,
                "expected_gpu_count": settings.expected_gpu_count,
                "manifest": settings.manifest.name,
            },
        )
        return self.workload

    def host_probe(self, settings: Any) -> FakeProbe:
        if settings.node == PARTNER:
            label = "partner"
        else:
            label = "b" if settings.run_id.endswith("-b") else "a"
        probe = FakeProbe(self, label, settings.node, settings.run_id)
        self.probes[label] = probe
        return probe

    def focused(self, _path: Path, *, reuse: bool = False) -> dict[str, Any]:
        return {"passed": self.tests_pass, "focused_tests_reused": reuse}

    def predecessor(self, _path: Path, case_id: str, **identity: Any) -> dict[str, Any]:
        self.call("predecessor", {"case_id": case_id, **identity})
        return {"valid": self.predecessor_valid}

    # -- driving -------------------------------------------------------------
    def plan(self) -> dict[str, Any]:
        self.case_dir.mkdir(parents=True, exist_ok=True)
        preflight = case.read_only_preflight(self.settings, self.case_dir)
        (self.case_dir / "plan.json").write_text(
            json.dumps({"details": case.plan_details(self.settings, preflight)}),
            encoding="utf-8",
        )
        return preflight

    def execute(self, seconds: float = 3600) -> tuple[int, dict[str, Any]]:
        code = case.execute_case(
            self.settings, self.run_dir, 1, NOW + timedelta(seconds=seconds)
        )
        return code, self.evidence(f"{case.CASE_ID}.json")

    def evidence(self, name: str) -> dict[str, Any]:
        return json.loads((self.case_dir / name).read_text(encoding="utf-8"))


class FakeRegional:
    def __init__(self, harness: Destr026Harness) -> None:
        self.h = harness
        self.settings = harness.settings.regional

    def evidence_identity(self) -> dict[str, str]:
        return {
            "release_id": "release-test",
            "cluster_id": "cluster-a",
            "region": "us-west-2",
        }

    def runtime_identity(self) -> dict[str, Any]:
        return deepcopy(self.h.runtime)

    def verify_runtime_identity(self, baseline: dict[str, Any], **kwargs: Any) -> None:
        self.h.call("runtime.verify", kwargs["stage"])
        current = self.h.runtime
        if "cleanup" in kwargs["stage"] and self.h.runtime_after_cleanup is not None:
            current = self.h.runtime_after_cleanup
        if baseline != current:
            raise RegionalFixtureError(f"runtime identity drifted {kwargs['stage']}")

    def node_snapshot(self, node: str) -> dict[str, Any]:
        self.h.call("node.snapshot", node)
        snapshot = deepcopy(self.h.nodes[node])
        if not self.h.node_ready(node):
            snapshot["ready"] = "False"
        return snapshot

    def wait_node_ready(self, node: str, *, timeout_seconds: int) -> dict[str, Any]:
        self.h.call("node.wait_ready", {"node": node, "timeout": timeout_seconds})
        deadline = self.h.clock.monotonic() + timeout_seconds
        while not self.h.node_ready(node):
            if self.h.clock.monotonic() >= deadline:
                raise RegionalFixtureError(
                    f"{node} did not return Ready in {timeout_seconds}s"
                )
            self.h.clock.sleep(10)
        return self.node_snapshot(node)

    def gpu_nodes(self) -> list[dict[str, Any]]:
        self.h.call("gpu.nodes")
        return deepcopy(
            self.h.gpu_nodes_after if self.h.injected else self.h.gpu_nodes_before
        )

    def business_workloads(self, node: str) -> list[dict[str, Any]]:
        return []

    def gpu_workloads(self) -> list[dict[str, Any]]:
        return []

    def cpu_blast_snapshot(self) -> dict[str, Any]:
        return deepcopy(self.h.cpu_after if self.h.injected else self.h.cpu)

    def store_snapshot(self, **kwargs: Any) -> dict[str, Any]:
        if "marker" in kwargs:
            return self.workflow_state(kwargs)
        if "job_id" in kwargs and "attempt_id" in kwargs:
            self.h.call("store.observation", kwargs)
            containers = [
                {"node_id": NODE, "gpu_uuids": list(self.h.allocation)},
                {"node_id": PARTNER, "gpu_uuids": list(WORKER_GPUS)},
            ]
            return {
                "observations": [
                    {"workload_phase": "RUNNING", "containers": containers}
                ]
            }
        if "job_id" in kwargs:
            self.h.call("store.attempts", kwargs)
            self.h.attempt_reads += 1
            attempts = [{"attempt_id": SOURCE_ATTEMPT}]
            if self.h.attempt_reads > self.h.attempts_missing:
                attempts.append({"attempt_id": RESTARTED_ATTEMPT})
            return {"observations": attempts}
        self.h.call("store.preflight", kwargs)
        return {
            "release_id": "release-test",
            "agent": deepcopy(self.h.agent),
            "profile": deepcopy(self.h.profile),
            "queue": deepcopy(self.h.queue),
            "remote_commands": {"open_by_cluster": {}},
            "event": deepcopy(self.h.recent_event),
            "gpu_identity": {
                "product": "H200",
                "products": ["H200"],
                "device_count": 8,
                "source_boot_id": self.h.nodes[NODE]["boot_id"],
                "observed_at": datetime.now(timezone.utc).isoformat(),
            },
        }

    def workflow_state(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        self.h.call("store.workflow", kwargs)
        state = {
            **event_state("a", GPU_A),
            "incident": deepcopy(self.h.incident),
            "workflow": deepcopy(self.h.workflow),
            "restart_budget": deepcopy(self.h.budget),
            "commands": deepcopy(self.h.commands),
            "observations": [{"workload_phase": "STOPPED"}],
        }
        if self.h.pending_reads:
            self.h.pending_reads -= 1
            state["workflow"]["status"] = "RUNNING"
        if self.h.cleanup_busy:
            state["commands"] = [
                {"status": "LEASED", "step": {"operation": "RESET_GPU"}}
            ]
        return state

    def wait_for_workflow(self, **kwargs: Any) -> dict[str, Any]:
        self.h.call("workflow.wait", kwargs)
        label = str(kwargs["marker"]).rsplit("-g", 1)[-1]
        state = event_state(label, self.h.event_gpus[label])
        if self.h.split and label == "b":
            state["workflow"] = {"request_id": "workflow-second"}
            state["incident"] = {
                "incident_id": "inc-second",
                "workflow_request_id": "workflow-second",
            }
        return state

    def kubectl(self, plane: str, *args: str, **kwargs: Any) -> str:
        self.h.call("kubectl", {"plane": plane, "args": list(args)})
        if args[0] == "logs":
            return self.h.log_text
        if args[0] == "exec":
            return json.dumps(
                {
                    join_case.AGGREGATION_WINDOW_VARIABLE: str(self.h.window),
                    join_case.JOB_LIFETIME_VARIABLE: "3600",
                }
            )
        if args[:2] == ("get", "pod"):
            items = self.h.dns if "kube-system" in args else self.h.arbiters
            return json.dumps({"items": items})
        if args[0] == "get" and "--ignore-not-found" in args:
            return self.h.workload_residual
        raise AssertionError(f"unhandled fake kubectl request: {args}")

    def ready_pods(self, plane: str, app: str) -> list[dict[str, Any]]:
        return [{"name": f"{app}-pod", "uid": f"{app}-uid"}]

    def provider_events(self, started: Any, ended: Any) -> list[dict[str, Any]]:
        self.h.call("provider.events")
        return deepcopy(self.h.events)

    def provider_events_provisional(self, ended: Any) -> bool:
        return True


class FakeWorkload:
    name = "gpu-fault-two-node-destr026"
    resource = "pytorchjob"

    def __init__(self, harness: Destr026Harness) -> None:
        self.h = harness

    def submit(self) -> dict[str, Any]:
        self.h.call("workload.submit")
        return {"submitted": True}

    def wait_running(self, **kwargs: Any) -> dict[str, Any]:
        self.h.call("workload.running", kwargs)
        return {"pods": source_pods()}

    def authorize_restart(self, state: dict[str, Any]) -> None:
        self.h.call("workload.authorize_restart", state["workflow"]["status"])

    def wait_restarted(self, uids: set[str], **kwargs: Any) -> dict[str, Any]:
        self.h.call("workload.restarted", {"source_uids": sorted(uids), **kwargs})
        return {"pods": deepcopy(self.h.restarted)}

    def wait_pod_uids_unchanged(self, uids: set[str], **kwargs: Any) -> dict[str, Any]:
        self.h.call("workload.stable", {"uids": sorted(uids), **kwargs})
        return {"pods": deepcopy(self.h.restarted)}

    def pods(self) -> list[dict[str, Any]]:
        self.h.call("workload.pods")
        return source_pods()

    def delete(self) -> dict[str, str]:
        self.h.call("workload.delete")
        return {"deleted": self.name}


class FakePrewarm:
    def __init__(self, harness: Destr026Harness) -> None:
        self.h = harness

    def create(self, nodes: list[str]) -> None:
        self.h.call("prewarm.create", nodes)

    def cached_nodes(self) -> list[str]:
        return list(self.h.cached)

    def cleanup(self) -> dict[str, bool]:
        self.h.call("prewarm.cleanup")
        return dict(self.h.prewarm_residuals)


class FakeProbe:
    """One host probe Pod. Every action is a ``kubectl exec`` (or a Pod create
    pinned to the node), which a NotReady node cannot answer."""

    host_script = "/fake/destructive_node_probe.py"

    def __init__(self, harness: Destr026Harness, label: str, node: str, run_id: str):
        self.h = harness
        self.label = label
        self.node = node
        self.run_id = run_id

    def _require_kubelet(self, action: str) -> None:
        if not self.h.node_ready(self.node):
            raise RegionalFixtureError(
                f"{action}: exec into {self.node} failed; kubelet is not answering"
            )

    def create(self) -> None:
        self.h.call("probe.create", self.label)
        self._require_kubelet("probe.create")

    def execute(self, action: str, *args: str, **kwargs: Any) -> dict[str, Any]:
        self.h.call(f"probe.{action}", {"probe": self.label, "args": list(args)})
        self._require_kubelet(f"probe.{action}")
        if action == "snapshot":
            source = self.h.hosts_after if "--since-epoch" in args else self.h.baselines
            return deepcopy(source[self.label])
        if action == "write-xid46":
            self.h.written(self.label)
            return {"xid": 46, "probe": self.label}
        if action in {"start-reset-sampler", "stop-reset-sampler"}:
            return {"action": action, "run_id": self.run_id}
        raise AssertionError(f"unhandled fake host action: {action}")

    def cleanup(self) -> dict[str, bool]:
        self.h.call("probe.cleanup", self.label)
        self._require_kubelet("probe.cleanup")
        return dict(self.h.probe_residuals.get(self.label, {"pod": False}))


class FakeWarm:
    def __init__(self, harness: Destr026Harness) -> None:
        self.h = harness

    def wait_incident_idle(self, incident: str) -> dict[str, Any]:
        self.h.call("restore.idle", incident)
        return {"idle": True}

    def create_restore_workflow(self, **kwargs: Any) -> dict[str, Any]:
        self.h.call("restore.create", kwargs)
        return {"workflow_request_id": f"restore-{kwargs['node']}"}

    def wait_workflow_id(self, request: str) -> dict[str, Any]:
        self.h.call("restore.wait", request)
        if self.h.restore_status == "SUCCEEDED":
            node = request.removeprefix("restore-")
            self.h.nodes[node].update(node_snapshot(node))
        return {"status": self.h.restore_status}
