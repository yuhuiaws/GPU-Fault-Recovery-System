from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts.e2e.regional import destr_barrier_authorization as authorization
from scripts.e2e.regional import run_destr009_workload_restart as workload_case
from scripts.e2e.regional import run_destr015_parallel_branch_join as case
from scripts.e2e.regional.destr015_physical_evidence import (
    ResetIntervalScope,
    evidence_digest,
)
from scripts.e2e.regional.late_ownership_barrier import BoundaryDenied
from scripts.e2e.regional.regional_live_fixture import (
    RUNTIME_IDENTITY_DEPLOYMENTS,
    RegionalFixtureError,
)
from tests.regional._cov95_destr_warm import NOW, Clock, profile, regional_settings
from tests.regional.test_acceptance_physical_interval_alignment import (
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

# The fake hosts' monotonic clocks start here and advance in lockstep with the
# harness clock; the runner's own stamps start at RUNNER_BASE. Both are
# arbitrary, the verdict only ever reads differences and exchange pairs.
HOST_BASE_NS = 50 * SECOND
RUNNER_BASE_NS = 1000 * SECOND
# The minutes the two parallel reset branches take on the real cluster between
# the arm and the collection; the fake advances the clock by this much when the
# records are collected so the host and runner spans agree.
COLLECT_ELAPSED_SECONDS = 100
WITNESS_STATE_ROOT = "/var/lib/gpu-fault-acceptance/destr015"


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
        # The kubelet outage the parallel quiesce causes: after the XID writes
        # every node is NotReady for this many harness seconds (``None`` keeps
        # the nodes Ready, ``inf`` never brings one back). ``arm_refusal`` is
        # the host's own reason for refusing to place a witness.
        self.outage_seconds: float | None = None
        self.permanent_outage: set[str] = set()
        self.reboot_during_outage: set[str] = set()
        self.not_ready_until: dict[str, float] = {}
        self.arm_refusal: dict[str, str] = {}
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
        monkeypatch.setattr(case, "DetachedResetWitness", BranchWitness)
        monkeypatch.setattr(case, "focused_tests", self.focused)
        monkeypatch.setattr(case, "predecessor_evidence", self.predecessor)
        monkeypatch.setattr(case, "replica_env", self.replica_env)
        monkeypatch.setattr(
            case, "record_focused_tests", lambda d, r: d.update(focused_tests=r)
        )
        for module in (case, workload_case):
            monkeypatch.setattr(module, "time", self.clock)
            monkeypatch.setattr(module, "datetime", self.clock)
        # The reachability decision reads the clock too: cleanup's failsafe must
        # measure the same time the harness advances.
        monkeypatch.setattr(authorization, "datetime", self.clock)

    def call(self, name: str, detail: Any = None) -> None:
        self.calls.append((name, deepcopy(detail)))
        self.clock.sleep(self.advance_at.get(name, 0))
        if name in self.failures:
            raise self.failures[name]

    def node_ready(self, node: str) -> bool:
        """What kubelet answers right now: Ready unless the quiesce outage holds."""

        if self.nodes[node].get("ready") != "True":
            return False
        return self.clock.elapsed >= self.not_ready_until.get(node, 0.0)

    def begin_outage(self) -> None:
        for node in NODES:
            if node in self.permanent_outage:
                self.not_ready_until[node] = float("inf")
            elif self.outage_seconds is not None:
                self.not_ready_until[node] = self.clock.elapsed + self.outage_seconds
            if node in self.reboot_during_outage:
                self.nodes[node]["boot_id"] = f"boot-{node}-rebooted"

    def runner_ns(self) -> int:
        return RUNNER_BASE_NS + int(self.clock.elapsed * SECOND)

    def host_ns(self) -> int:
        return HOST_BASE_NS + int(self.clock.elapsed * SECOND)

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
    """The detached witness at the transport boundary; real verdicts downstream.

    Mirrors the probe refusals the runner depends on: ``arm`` refuses a
    lifetime outside the probe's bounds or a scope whose boot id is not this
    host's, ``collect`` refuses a changed boot id, and every request is a
    ``kubectl exec`` that a NotReady node cannot answer. The records it returns
    have the shape the real probe writes, with a host clock that advances in
    lockstep with the harness clock, so the interval arithmetic downstream is
    the real one.
    """

    def __init__(
        self,
        regional: Any,
        probe: Any,
        scope: ResetIntervalScope,
        *,
        lifetime_seconds: int,
    ) -> None:
        self.h: BranchHarness = probe.h
        self.regional = regional
        self.scope = scope
        self.node = scope.node
        self.lifetime_seconds = lifetime_seconds
        self.bundle_sha256 = "f" * 64
        self.receipt: dict[str, Any] | None = None
        self.collection: dict[str, Any] | None = None
        self.last_status: dict[str, Any] | None = None
        self.exchanges: list[dict[str, Any]] = []
        self.armed_at: Any = None
        self.arm_mono_ns = 0
        self.wall_ns = 0
        self._last: tuple[dict[str, int], dict[str, int]] = ({}, {})
        capture = interval_fixture()[0]["node-a"]
        start = capture["start"]
        start.update(scope_sha256=scope.digest(), witness_id=f"witness-{scope.node}")
        start["tracee"]["boot_id"] = scope.boot_id
        start["producer"]["boot_id"] = scope.boot_id
        self.start = start
        self.end_template = capture["end"]
        digest = hashlib.sha256(scope.run_id.encode()).hexdigest()[:16]
        self.unit = f"gpu-fault-destr015-witness-{digest}.service"
        self.state_dir = f"{WITNESS_STATE_ROOT}/{scope.run_id}"

    def _exec(self, kind: str) -> tuple[int, dict[str, int]]:
        self.h.call(f"witness.{kind}", self.node)
        if not self.h.node_ready(self.node):
            raise BoundaryDenied(
                f"physical witness {kind}: exec into {self.node} failed; "
                "kubelet is not answering (node NotReady)"
            )
        return self.h.runner_ns(), self._host_clock()

    def _host_clock(self) -> dict[str, int]:
        mono = self.h.host_ns()
        return {"monotonic_ns": mono, "realtime_ns": self.wall_ns + mono}

    def _finish(self, kind: str, sent: int, start_clock: dict[str, int]) -> None:
        self.h.clock.sleep(1)
        clock = self._host_clock()
        self.exchanges.append(
            {
                "kind": kind,
                "sent_ns": sent,
                "received_ns": self.h.runner_ns(),
                "monotonic_ns": clock["monotonic_ns"],
                "realtime_ns": clock["realtime_ns"],
                "runner_sent_realtime_ns": int(self.h.clock.time() * SECOND) - SECOND,
                "runner_received_realtime_ns": int(self.h.clock.time() * SECOND),
            }
        )
        self._last = (start_clock, clock)

    def arm(self) -> dict[str, Any]:
        sent, start_clock = self._exec("arm")
        if not 600 <= self.lifetime_seconds <= 7200:
            raise BoundaryDenied(
                "physical witness arm refused: witness lifetime is outside its bounds"
            )
        if self.scope.boot_id != self.h.baselines[self.node]["boot_id"]:
            raise BoundaryDenied(
                "physical witness arm refused: host boot id differs from the scope"
            )
        if self.h.arm_refusal.get(self.node):
            raise BoundaryDenied(
                f"physical witness arm refused: {self.h.arm_refusal[self.node]}"
            )
        # The host's realtime clock is pinned so the reset lands inside the
        # ledger command window the fake host snapshot reports (NOW..NOW+40s).
        self.wall_ns = int(NOW.timestamp() * SECOND) - self.h.host_ns() - SECOND
        start_clock = self._host_clock()
        armed_mono = start_clock["monotonic_ns"] + SECOND // 2
        armed = {
            "case_id": case.CASE_ID,
            "record": "armed",
            "run_id": self.scope.run_id,
            "node": self.node,
            "scope_sha256": self.scope.digest(),
            "unit": self.unit,
            "invocation_id": f"invocation-{self.node}",
            "witness_id": self.start["witness_id"],
            "boot_id": self.scope.boot_id,
            "monotonic_ns": armed_mono,
            "realtime_ns": self.wall_ns + armed_mono,
            "start": deepcopy(self.start),
            "ledger_baseline_command_ids": [],
        }
        self._finish("arm", sent, start_clock)
        self.arm_mono_ns = self._last[1]["monotonic_ns"]
        self.receipt = {
            "run_id": self.scope.run_id,
            "node": self.node,
            "unit": self.unit,
            "state_dir": self.state_dir,
            "lifetime_seconds": self.lifetime_seconds,
            "boot_id": self.scope.boot_id,
            "program_sha256": self.bundle_sha256,
            "armed": armed,
            "host_clock_start": self._last[0],
            "host_clock": self._last[1],
            "runner_clock": {"monotonic_ns": sent, "realtime_ns": 0},
            "unit_state": {"ActiveState": "active", "SubState": "running"},
        }
        self.armed_at = self.h.clock.now(timezone.utc)
        for row in self.h.hosts_after[self.node]["ledger"]:
            if row["operation"] == "RESET_GPU":
                row["gpu_uuids"] = [self.scope.gpu_uuid]
        self.h.physical_captures[self.node] = deepcopy(self.receipt)
        return self.receipt

    def _final(self, mono: int) -> dict[str, Any]:
        # The reset the tracer saw: two seconds after the arm response (four
        # for the second node), nine seconds long, so the two intervals overlap
        # by five seconds after the clock uncertainty is charged.
        delay = 2 * SECOND if self.node == NODES[0] else 4 * SECOND
        began = self.arm_mono_ns + delay + self.wall_ns
        end = deepcopy(self.end_template)
        end.update(**self.start, start_sha256=evidence_digest(self.start))
        action = end["actions"][0]
        action.update(
            gpu_uuid=self.scope.gpu_uuid, started_ns=began, ended_ns=began + 9 * SECOND
        )
        return {
            "case_id": case.CASE_ID,
            "record": "final",
            "run_id": self.scope.run_id,
            "node": self.node,
            "scope_sha256": self.scope.digest(),
            "unit": self.unit,
            "invocation_id": f"invocation-{self.node}",
            "witness_id": self.start["witness_id"],
            "boot_id": self.scope.boot_id,
            "monotonic_ns": mono,
            "realtime_ns": self.wall_ns + mono,
            "reason": "finish-request",
            "start_sha256": end["start_sha256"],
            "trace_complete": True,
            "closed": True,
            "lost_events": 0,
            "trace_sha256": end["trace_sha256"],
            "trace_bytes": end["trace_bytes"],
            "calibration_execs": end["calibration_execs"],
            "actions": end["actions"],
            "wall_minus_monotonic_min_ns": self.wall_ns,
            "wall_minus_monotonic_max_ns": self.wall_ns,
            "refusal": None,
            "ledger_reset_rows": [],
        }

    def collect(self) -> dict[str, Any]:
        if self.receipt is None:
            raise BoundaryDenied("physical witness was never armed")
        # The parallel reset took minutes on the cluster before the runner could
        # reach the node again.
        self.h.clock.sleep(COLLECT_ELAPSED_SECONDS)
        sent, start_clock = self._exec("collect")
        if self.h.nodes[self.node]["boot_id"] != self.scope.boot_id:
            raise BoundaryDenied(
                "physical witness collect refused: host boot id changed since the "
                "witness was armed; its records are not this boot's"
            )
        final = self._final(start_clock["monotonic_ns"] + SECOND // 2)
        self._finish("collect", sent, start_clock)
        self.collection = {
            "run_id": self.scope.run_id,
            "node": self.node,
            "unit": self.unit,
            "state_dir": self.state_dir,
            "boot_id": self.scope.boot_id,
            "state": {
                "run_id": self.scope.run_id,
                "node": self.node,
                "phase": "ARMED",
                "scope_sha256": self.scope.digest(),
                "boot_id": self.scope.boot_id,
                "unit": self.unit,
                "lifetime_seconds": self.lifetime_seconds,
            },
            "armed": deepcopy(self.receipt["armed"]),
            "final": final,
            "finish_requested": start_clock,
            "unit_state": {"ActiveState": "inactive", "SubState": "dead"},
            "host_clock_start": self._last[0],
            "host_clock": self._last[1],
        }
        self.h.physical_captures[self.node] = deepcopy(self.collection)
        return self.collection

    def disarm(self) -> dict[str, Any]:
        self._exec("disarm")
        return {
            "run_id": self.scope.run_id,
            "unit": self.unit,
            "unit_state": {"ActiveState": "inactive"},
            "state_present": self.receipt is not None,
            "boot_id": self.h.nodes[self.node]["boot_id"],
        }

    def status(self) -> dict[str, Any]:
        self._exec("status")
        self.last_status = {
            "run_id": self.scope.run_id,
            "unit": self.unit,
            "unit_state": {"ActiveState": "inactive"},
            "state": {"phase": "REFUSED", "refusal": self.h.arm_refusal.get(self.node)},
            "armed": None,
            "final": None,
            "boot_id": self.h.nodes[self.node]["boot_id"],
        }
        return self.last_status

    def evidence(self) -> dict[str, Any]:
        return {
            "scope": self.scope.model_dump(mode="json"),
            "lifetime_seconds": self.lifetime_seconds,
            "bundle_sha256": self.bundle_sha256,
            "receipt": self.receipt,
            "collection": self.collection,
            "exchanges": list(self.exchanges),
        }


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
        snapshot = deepcopy(self.h.nodes[node])
        if not self.h.node_ready(node):
            snapshot["ready"] = "False"
        return snapshot

    def wait_node_ready(
        self, node: str, *, timeout_seconds: int, expected_boot_id: str | None = None
    ) -> dict[str, Any]:
        self.h.call(
            "node.wait_ready", {"node": node, "timeout_seconds": timeout_seconds}
        )
        deadline = self.h.clock.monotonic() + timeout_seconds
        while True:
            snapshot = self.node_snapshot(node)
            rebooted = expected_boot_id is None or (
                bool(snapshot.get("boot_id"))
                and snapshot["boot_id"] != expected_boot_id
            )
            if snapshot.get("ready") == "True" and rebooted:
                return snapshot
            if self.h.clock.monotonic() >= deadline:
                raise RegionalFixtureError(f"node did not return Ready: {snapshot}")
            self.h.clock.sleep(10)

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
        assert self.restart_state is not None, "a restart needs its authorizing state"
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
    """The host probe Pod on one node. Every action is a ``kubectl exec`` (or a
    Ready wait), which a NotReady node cannot answer: the fake refuses exactly
    where the real transport would fail, so a runner that execs into a
    quiesced node fails the test instead of passing on a fake that answers."""

    def __init__(self, harness: BranchHarness, node: str) -> None:
        self.h = harness
        self.node = node
        self.host_script = "/fake/probe.py"

    def _require_kubelet(self, action: str) -> None:
        if not self.h.node_ready(self.node):
            raise RegionalFixtureError(
                f"{action}: exec into {self.node} failed; kubelet is not answering "
                "(node NotReady)"
            )

    def create(self) -> None:
        self.h.call("probe.create", self.node)
        self._require_kubelet("probe.create")

    def execute(self, action: str, *args: str, **kwargs: Any) -> dict[str, Any]:
        self.h.call(f"probe.{action}", {"node": self.node, "args": args, **kwargs})
        self._require_kubelet(f"probe.{action}")
        if action == "snapshot":
            return deepcopy(
                self.h.hosts_after[self.node]
                if "--since-epoch" in args
                else self.h.baselines[self.node]
            )
        if action == "write-xid46":
            self.h.injected.add(self.node)
            if len(self.h.injected) == len(NODES):
                self.h.begin_outage()
            return {"xid": 46, "node": self.node}
        raise AssertionError(f"unhandled fake host action: {action}")

    def cleanup(self) -> dict[str, bool]:
        self.h.call("probe.cleanup", self.node)
        self._require_kubelet("probe.cleanup")
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
