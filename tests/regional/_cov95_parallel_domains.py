"""A fake cluster for the parallel workload-domain runners DESTR-027/028/029.

The fakes stand at the runners' transport boundary only: the regional fixture
(store, Kubernetes and CPU-plane reads), the host probes, the managed training
jobs and the image prewarm. Everything downstream is real: the shared
workload-domain oracle judges records built with its own test's builders,
DESTR-001's reset contract and host checks read the probe snapshots, and the
job reboot contract reads the node and Agent records. Every fake action is
logged in ``calls`` and can be made to fail through ``failures``.
"""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from gpu_fault.cluster_scope import EKS_NODEGROUP_LABEL
from scripts.e2e.regional import run_destr001_gpu_reset as reset_case
from scripts.e2e.regional import workload_domain_oracle as oracle
from scripts.e2e.regional.managed_workload_fixture import managed_workload_name
from scripts.e2e.regional.regional_live_fixture import RegionalLiveSettings
from tests.regional._cov95_destr_branches import ready_runtime
from tests.regional._destructive_acceptance_support import reset_host_pair
from tests.regional.test_destr023_idle_cluster_reset import reset_state
from tests.regional.test_workload_domain_oracle import (
    CLUSTER,
    Part,
    premise_snapshot,
    snapshot,
)

GROUP = "gpu-group"
NODES = ("node-a", "node-b", "node-c", "node-d")
AGENT_ARTIFACT = "sha256:" + "a" * 64
WORKLOAD_NAMESPACE = "gpu-fault-system/pytorchjob/"
#: The steps a job workflow ran live on HyperPod (DESTR-029 attempt 39).
JOB_STEPS = (
    "FREEZE_EVIDENCE",
    "MARK_UNSCHEDULABLE",
    "STOP_WORKLOADS",
    "RESTART_NODE",
    "VALIDATE_GPU",
    "VALIDATE_HOST",
    "VALIDATE_FABRIC",
    "RESTORE_SCHEDULING",
    "RESTART_WORKLOAD",
)


class Clock:
    """The runner's ``time`` and ``datetime``: it starts at ``base`` and moves
    only when a fake call is charged time (``World.advance_at``)."""

    fromisoformat = staticmethod(datetime.fromisoformat)

    def __init__(self, base: datetime) -> None:
        self.base = base
        self.elapsed = 0.0

    def monotonic(self) -> float:
        return self.elapsed

    def time(self) -> float:
        return self.base.timestamp() + self.elapsed

    def sleep(self, seconds: float) -> None:
        self.elapsed += seconds

    def now(self, tz: Any = None) -> datetime:
        return self.base + timedelta(seconds=self.elapsed)


def regional_settings(tmp_path: Path) -> RegionalLiveSettings:
    cpu = tmp_path / "cpu.config"
    gpu = tmp_path / "gpu.config"
    cpu.write_text("fake CPU transport\n", encoding="utf-8")
    gpu.write_text("fake GPU transport\n", encoding="utf-8")
    return RegionalLiveSettings(
        cpu, gpu, "fake-gpu", "test-system", CLUSTER, "us-west-2"
    )


def capability(name: str, owner: str, adapter: str) -> dict[str, Any]:
    return {"capability": name, "mode": "OWN", "owner": owner, "adapter": adapter}


RESET_OWNER = capability("gpuReset", "gpu-fault-node-agent", "node-action")
REBOOT_OWNER = capability("nodeReboot", "gpu-fault-node-agent", "hyperpod")
WORKLOAD_OWNERS = [
    capability(name, "gpu-fault-kubernetes-adapter", "regional-cluster-executor")
    for name in ("workloadStop", "workloadRestart")
]


def profile(*capabilities: dict[str, Any]) -> dict[str, Any]:
    return {
        "profile_version": "profile-v1",
        "warnings": [],
        "capabilities": [deepcopy(item) for item in capabilities],
    }


def node_snapshot(node: str) -> dict[str, Any]:
    return {
        "name": node,
        "uid": f"uid-{node}",
        "boot_id": f"boot-{node}",
        "ready": "True",
        "unschedulable": False,
        "taints": [],
        "gpu_allocatable": "8",
        "ownership_annotations": {},
        "spare_pool_annotations": {},
    }


def store_snapshot(node: str, runtime_profile: dict[str, Any]) -> dict[str, Any]:
    """A clean preflight store read of one node: Agent ACTIVE, no recent XID,
    a fresh GPU identity from the node's current boot, empty queues."""

    return {
        "agent": {
            "lifecycle_state": "ACTIVE",
            "allowed_operations": sorted(reset_case.REQUIRED_AGENT_OPERATIONS),
            "generation": 7,
            "boot_id": f"agent-boot-{node}",
            "artifact_sha256": AGENT_ARTIFACT,
        },
        "event": None,
        "gpu_identity": {
            "product": "NVIDIA H200",
            "products": ["NVIDIA H200"],
            "device_count": 8,
            "source_boot_id": f"boot-{node}",
            "observed_at": datetime.now(timezone.utc).isoformat(),
        },
        "profile": deepcopy(runtime_profile),
        "queue": {"depth": 0, "fault_backlog_depth": 0},
        "remote_commands": {"open_by_cluster": {}},
    }


def reset_node_state(label: str) -> dict[str, Any]:
    """DESTR-001's terminal reset workflow, named after the oracle part
    ``label`` so the store records and the runner's state agree."""

    state = reset_state()
    state["workflow"]["request_id"] = f"workflow-{label}"
    state["incident"]["incident_id"] = f"incident-{label}"
    return state


def job_node_state(label: str) -> dict[str, Any]:
    """The job workflow's terminal record on its rebooted master."""

    return {
        "event": {"xid": 79, "evidence_ref": "kmsg://master/1"},
        "decision": {"action": "REBOOT_NODE"},
        "incident": {"incident_id": f"incident-{label}", "state": "RECOVERED"},
        "workflow": {
            "request_id": f"workflow-{label}",
            "status": "SUCCEEDED",
            "official_steps": [{"operation": step} for step in JOB_STEPS],
            "step_executions": [
                {"operation": step, "status": "SUCCEEDED"} for step in JOB_STEPS
            ],
        },
        "agent": {
            "lifecycle_state": "ACTIVE",
            "boot_id": "agent-boot-after-reboot",
            "artifact_sha256": AGENT_ARTIFACT,
        },
    }


def reset_hosts(label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """An idle node's host before and after its one-GPU reset: the sampler saw
    GPU-0 leave and return and the ledger gained this workflow's RESET_GPU."""

    baseline, after = reset_host_pair(minimum=7, last=8)
    after = deepcopy(after)
    baseline.update(compute_clients=[], quiesce_states=[], kmsg_writable=True)
    after["ledger"][-1].update(
        incident_id=f"incident-{label}", workflow_request_id=f"workflow-{label}"
    )
    return baseline, after


def job_host(*, clients: bool = True) -> dict[str, Any]:
    """A job master's host baseline: its ranks hold the GPUs by design."""

    baseline, _after = reset_host_pair(minimum=7, last=8)
    baseline.update(
        compute_clients=[{"pid": 4242, "gpu_uuid": "GPU-0"}] if clients else [],
        quiesce_states=[],
        kmsg_writable=True,
    )
    return baseline


def bind_job(item: Part, workload_id: str) -> Part:
    """The oracle part's job tokens renamed after the runner's workload id
    (the builder names ``default/pytorchjob/<job>``)."""

    old = f"default/pytorchjob/{item.expectation.job_id}"
    item.workflow = json.loads(json.dumps(item.workflow).replace(old, workload_id))
    return item


def workload_id(base: str, job_id: str) -> str:
    return WORKLOAD_NAMESPACE + managed_workload_name(base, job_id)


class World:
    """The fake cluster one runner sees. Configure it, then ``plan`` and
    ``execute`` exactly as ``--plan`` and ``--execute`` would."""

    def __init__(
        self,
        module: ModuleType,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        *,
        settings: Any,
        nodes: tuple[str, ...],
        runtime_profile: dict[str, Any],
    ) -> None:
        self.module = module
        self.settings = settings
        self.run_dir = tmp_path
        self.case_dir = tmp_path / "cases" / module.CASE_ID
        self.clock = Clock(datetime.now(timezone.utc).replace(microsecond=0))
        self.calls: list[tuple[str, Any]] = []
        self.failures: dict[str, BaseException] = {}
        self.advance_at: dict[str, float] = {}
        self.gpu_nodes = [
            {
                "name": node,
                "uid": f"uid-{node}",
                "ready": "True",
                "gpu_allocatable": 8,
                "labels": {EKS_NODEGROUP_LABEL: GROUP},
            }
            for node in nodes
        ]
        self.nodes = {node: node_snapshot(node) for node in nodes}
        # Applied to the Node records when the XID lands (a reboot changes the
        # boot id) and when the probe Pods are deleted (the final read).
        self.on_injection: dict[str, dict[str, Any]] = {}
        self.on_cleanup: dict[str, dict[str, Any]] = {}
        self.stores = {node: store_snapshot(node, runtime_profile) for node in nodes}
        self.views = {node: "IDLE" for node in nodes}
        self.business: dict[str, list[dict[str, Any]]] = {}
        self.gpu_pods: list[dict[str, Any]] = []
        self.arbiters: list[dict[str, Any]] = []
        self.dns = [{"metadata": {}, "spec": {"nodeName": "cpu-node"}}]
        self.runtime = ready_runtime()
        self.premise = premise_snapshot()
        self.records: dict[str, Any] = {}
        self.states: dict[str, dict[str, Any]] = {}
        self.baselines: dict[str, dict[str, Any]] = {}
        self.afters: dict[str, dict[str, Any]] = {}
        self.probe_residuals: dict[str, dict[str, bool]] = {}
        self.prewarm_residuals = {"pods": False}
        self.events: list[dict[str, Any]] = []
        self.cpu = {"nodes": ["cpu-a"]}
        self.cpu_after = self.cpu
        self.tests_pass = True
        self.predecessor_valid = True
        self.injected: set[str] = set()
        self.workloads: dict[str, FakeWorkload] = {}
        self.regional = FakeRegional(self)
        self.prewarm = FakePrewarm(self)
        monkeypatch.setattr(module, "RegionalLiveFixture", lambda _s: self.regional)
        monkeypatch.setattr(
            module, "HostProbeFixture", lambda value: FakeProbe(self, value)
        )
        monkeypatch.setattr(module, "focused_tests", self.focused)
        monkeypatch.setattr(module, "predecessor_evidence", self.predecessor)
        monkeypatch.setattr(
            module, "record_focused_tests", lambda d, r: d.update(focused_tests=r)
        )
        monkeypatch.setattr(module, "time", self.clock)
        monkeypatch.setattr(module, "datetime", self.clock)
        if hasattr(module, "ManagedWorkloadFixture"):
            monkeypatch.setattr(module, "ManagedWorkloadFixture", self.workload)
            monkeypatch.setattr(
                module, "ImagePrewarmFixture", lambda *_a, **_k: self.prewarm
            )

    # -- the fake transport --------------------------------------------------
    def call(self, name: str, detail: Any = None) -> None:
        self.calls.append((name, deepcopy(detail)))
        self.clock.sleep(self.advance_at.get(name, 0))
        if name in self.failures:
            raise self.failures[name]

    def names(self) -> list[str]:
        return [name for name, _detail in self.calls]

    def inject(self, node: str) -> None:
        self.injected.add(node)
        self.nodes[node].update(deepcopy(self.on_injection.get(node, {})))

    def focused(self, _path: Path, *, reuse: bool = False) -> dict[str, Any]:
        self.call("focused", reuse)
        return {"passed": self.tests_pass, "focused_tests_reused": reuse}

    def predecessor(self, _path: Path, case_id: str, **identity: Any) -> dict[str, Any]:
        self.call("predecessor", {"case_id": case_id, **identity})
        return {"valid": self.predecessor_valid}

    def workload(self, _regional: Any, settings: Any) -> FakeWorkload:
        value = FakeWorkload(self, settings)
        self.workloads[settings.job_id] = value
        return value

    # -- the runner's two modes -----------------------------------------------
    def plan(self) -> dict[str, Any]:
        self.case_dir.mkdir(parents=True, exist_ok=True)
        preflight = self.module.read_only_preflight(self.settings, self.case_dir)
        details = self.module.plan_details(self.settings, preflight)
        (self.case_dir / "plan.json").write_text(
            json.dumps({"details": details}), encoding="utf-8"
        )
        return preflight

    def execute(self, seconds: int = 3600) -> tuple[int, dict[str, Any]]:
        end = self.clock.now() + timedelta(seconds=seconds)
        code = self.module.execute_case(self.settings, self.run_dir, 1, end)
        return code, self.evidence()

    def evidence(self, name: str = "") -> dict[str, Any]:
        path = self.case_dir / (name or f"{self.module.CASE_ID}.json")
        return json.loads(path.read_text(encoding="utf-8"))

    def records_of(self, *parts: Part, **kwargs: Any) -> None:
        self.records = snapshot(*parts, **kwargs)

    def run_id(self) -> str:
        return str(self.module.run_identity(self.run_dir, 1))


class FakeRegional:
    def __init__(self, world: World) -> None:
        self.w = world
        self.settings = world.settings.regional

    def evidence_identity(self) -> dict[str, str]:
        self.w.call("identity")
        return {"release_id": "release-test", "cluster_id": CLUSTER, "region": "r"}

    def gpu_nodes(self) -> list[dict[str, Any]]:
        self.w.call("gpu_nodes")
        return deepcopy(self.w.gpu_nodes)

    def node_snapshot(self, node: str) -> dict[str, Any]:
        self.w.call("node.snapshot", node)
        return deepcopy(self.w.nodes[node])

    def store_snapshot(self, **kwargs: Any) -> dict[str, Any]:
        self.w.call("store", kwargs["node"])
        return deepcopy(self.w.stores[kwargs["node"]])

    def kubectl(self, plane: str, *args: str, **kwargs: Any) -> str:
        self.w.call("kubectl", {"plane": plane, "args": args})
        if args[:2] == ("get", "pod"):
            pods = self.w.dns if "kube-system" in args else self.w.arbiters
            return json.dumps({"items": pods})
        raise AssertionError(f"unhandled fake kubectl request: {args}")

    def gpu_workloads(self) -> list[dict[str, Any]]:
        return deepcopy(self.w.gpu_pods)

    def node_workload_view(self, node: str) -> dict[str, Any]:
        return {"node": node, "workload_state": self.w.views[node]}

    def business_workloads(self, node: str) -> list[dict[str, Any]]:
        return deepcopy(self.w.business.get(node, []))

    def runtime_identity(self) -> dict[str, Any]:
        return deepcopy(self.w.runtime)

    def verify_runtime_identity(self, baseline: dict[str, Any], **kwargs: Any) -> None:
        self.w.call(f"runtime.verify:{kwargs['stage']}", str(kwargs["evidence_path"]))
        assert baseline == self.w.runtime, "the runner binds the planned identity"

    def cpu_python(self, script: str, *args: str) -> dict[str, Any]:
        assert script == oracle.ORACLE_PROBE, "only the oracle reads the CPU plane"
        cluster, workflow_ids, node_ids, states = args
        assert cluster == CLUSTER and states, "the read names the cluster and states"
        if not workflow_ids:
            self.w.call("oracle.premise", node_ids)
            return deepcopy(self.w.premise)
        self.w.call("oracle.read", {"workflows": workflow_ids, "nodes": node_ids})
        return deepcopy(self.w.records)

    def cpu_blast_snapshot(self) -> dict[str, Any]:
        self.w.call("cpu.snapshot")
        return deepcopy(self.w.cpu_after if self.w.injected else self.w.cpu)

    def wait_for_workflow(self, **kwargs: Any) -> dict[str, Any]:
        node = kwargs["node"]
        self.w.call(f"workflow.wait:{node}", kwargs["timeout_seconds"])
        return deepcopy(self.w.states[node])

    def provider_events(self, *args: Any) -> list[dict[str, Any]]:
        self.w.call("provider.events")
        return deepcopy(self.w.events)

    def provider_events_provisional(self, *args: Any) -> bool:
        return True


class FakeProbe:
    """The owned host probe Pod on one node."""

    host_script = "/fake/probe.py"

    def __init__(self, world: World, settings: Any) -> None:
        self.w = world
        self.node = settings.node
        self.run_id = settings.run_id
        world.call("probe.new", {"node": self.node, "run_id": self.run_id})

    def create(self) -> None:
        self.w.call(f"probe.create:{self.node}")

    def execute(self, action: str, *args: str, **kwargs: Any) -> dict[str, Any]:
        self.w.call(f"probe.{action}:{self.node}", {"args": args, **kwargs})
        if action == "snapshot":
            source = self.w.afters if "--since-epoch" in args else self.w.baselines
            return deepcopy(source[self.node])
        if action.startswith("write-xid"):
            self.w.inject(self.node)
        return {"action": action, "node": self.node, "run_id": self.run_id}

    def cleanup(self) -> dict[str, bool]:
        self.w.call(f"probe.cleanup:{self.node}")
        self.w.nodes[self.node].update(deepcopy(self.w.on_cleanup.get(self.node, {})))
        return dict(self.w.probe_residuals.get(self.node, {"pod": False}))


class FakeWorkload:
    """One managed training job (``ManagedWorkloadFixture``)."""

    def __init__(self, world: World, settings: Any) -> None:
        self.w = world
        self.settings = settings
        self.job_id = settings.job_id

    def submit(self) -> dict[str, Any]:
        self.w.call(f"job.submit:{self.job_id}")
        return {"submitted": self.job_id}

    def wait_running(self, **kwargs: Any) -> dict[str, Any]:
        self.w.call(f"job.running:{self.job_id}", kwargs)
        return {"pods": [{"uid": f"pod-{self.job_id}", "phase": "Running"}]}

    def delete(self) -> dict[str, Any]:
        self.w.call(f"job.delete:{self.job_id}")
        return {"deleted": self.job_id}


class FakePrewarm:
    def __init__(self, world: World) -> None:
        self.w = world

    def create(self, nodes: list[str]) -> dict[str, Any]:
        self.w.call("prewarm.create", nodes)
        return {"cached_nodes": list(nodes)}

    def cleanup(self) -> dict[str, bool]:
        self.w.call("prewarm.cleanup")
        return dict(self.w.prewarm_residuals)
