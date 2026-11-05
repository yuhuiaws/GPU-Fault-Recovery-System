"""A fake site for the GF-REGIONAL-BOOT-038 runner.

The fakes stand where the runner leaves the process: the regional fixture's
store and Kubernetes reads, the deploy-host CLI (``subprocess``), the warm-spare
fixtures that read the other node and drive the kubelet window, and the
ownership, node-scope and registry helpers that talk to AWS. Everything the
runner decides with is real: the narration grammar and verdicts of
``boot038_membership``, its record readers over the state directory, the
service-window restore through the node, the journals' residue check and the
plan identity gate. The kubelet window is modelled: ``stop`` takes the other
node NotReady and the window's ``cleanup`` brings it back, so the runner's
NotReady poller reads what a real window would show.
"""

from __future__ import annotations

import io
import json
import subprocess
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from scripts.e2e.regional import boot038_membership as membership
from scripts.e2e.regional import run_boot038_cross_cluster_membership as runner
from scripts.e2e.regional import warm_spare_fixture
from scripts.e2e.regional.regional_live_fixture import RegionalLiveSettings

TARGET_ID = "hp-cluster"
OTHER_ID = "gpu-eks-cluster"
OTHER_NODE = "ip-10-0-44-18"
PEER_NODE = "ip-10-0-44-19"
ACCOUNT = "arn:aws:eks:us-west-2:000000000000:cluster"
TARGET_ARN = f"{ACCOUNT}/hp-backing"
OTHER_ARN = f"{ACCOUNT}/gpu-eks"
CPU_ARN = f"{ACCOUNT}/cpu"
FINDING = {"check": "control_api", "cluster_id": OTHER_ID, "finding": "coverage drift"}
STATUS_LINE = (
    f"gpu-fault-admin status: finding: join-cluster {TARGET_ID}: not gating: "
    f"{OTHER_ID} control_api: coverage drift"
)


@contextmanager
def threads_end_inside() -> Iterator[None]:
    """Every thread started inside (verb followers, NotReady pollers) has
    ended when the block is left."""

    before = set(threading.enumerate())
    yield
    for thread in set(threading.enumerate()) - before:
        thread.join(timeout=5)
        assert not thread.is_alive(), f"{thread.name} outlived its test"


class Clock:
    """``CLOCK_NOW`` and ``time.sleep`` of the runner: virtual seconds."""

    def __init__(self) -> None:
        self.base = datetime(2026, 11, 4, 18, 30, tzinfo=timezone.utc)
        self.elapsed = 0.0
        self.lock = threading.Lock()

    def now(self) -> datetime:
        with self.lock:
            return self.base + timedelta(seconds=self.elapsed)

    def sleep(self, seconds: float) -> None:
        with self.lock:
            self.elapsed += seconds


def write_site(state_dir: Path, **changes: Any) -> Path:
    """The site's ``site.yaml`` (YAML, as the deploy host writes it)."""

    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "cpu.kubeconfig").write_text("fake cpu\n", encoding="utf-8")
    (state_dir / "gpu.kubeconfig").write_text("fake gpu\n", encoding="utf-8")
    spec: dict[str, Any] = {
        "awsRegion": "us-west-2",
        "namespace": "gpu-fault-system",
        "gpuKubeconfig": str(state_dir / "gpu.kubeconfig"),
        "cpu": {"kubeconfig": str(state_dir / "cpu.kubeconfig"), "eksArn": CPU_ARN},
        "clusters": [
            {"clusterId": OTHER_ID, "context": "ctx-other", "eksClusterArn": OTHER_ARN},
            {"clusterId": TARGET_ID, "context": "ctx-hp", "eksClusterArn": TARGET_ARN},
        ],
    }
    spec.update(changes)
    document = {"apiVersion": "v1", "kind": "RegionalSite", "spec": spec}
    path = state_dir / "site.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


def settings(tmp_path: Path, *, window: str = membership.WINDOW_KUBELET) -> Any:
    state_dir = tmp_path / "state"
    write_site(state_dir)
    return runner.Settings(
        state_dir=state_dir,
        target_cluster_arn=TARGET_ARN,
        target_cluster_id=TARGET_ID,
        other_cluster_id=OTHER_ID,
        other_node=OTHER_NODE,
        window=window,
        dry_run=False,
        regional=RegionalLiveSettings(
            cpu_kubeconfig=state_dir / "cpu.kubeconfig",
            gpu_kubeconfig=state_dir / "gpu.kubeconfig",
            gpu_context="ctx-other",
            namespace="gpu-fault-system",
            cluster_id=OTHER_ID,
            region="us-west-2",
        ),
        host_probe_image="example.test/probe@sha256:" + "e" * 64,
        cpu_cluster_arn=CPU_ARN,
        predecessor_id=None,
        predecessor_path=None,
        run_token="deadbeef",
    )


def verb_result(**changes: Any) -> list[str]:
    """The verb's own result object as ``json.dumps(indent=2)`` prints it."""

    value = {
        "cluster_id": TARGET_ID,
        "cpu_control_plane": "PRESERVED",
        "gpu_cluster": "PRESERVED",
        "phase": "COMPLETED",
        "not_gating_findings": [dict(FINDING)],
        **changes,
    }
    return [line + "\n" for line in json.dumps(value, indent=2).splitlines()]


VERIFY_REPORT = ["{\n", '  "checks": [], "healthy": true, "mode": "verify"\n']


def remove_stream(*, result: list[str] | None = None) -> list[str]:
    return [
        "release-begin 2026-11-04T18:37:33Z mode=drain-cluster dry_run=false\n",
        "release-begin 2026-11-04T18:47:47Z mode=remove-cluster dry_run=false\n",
        "release-end 2026-11-04T18:49:40Z mode=remove-cluster exit_code=0 total=113.2s\n",
        "release-begin 2026-11-04T18:51:27Z mode=verify dry_run=false\n",
        *VERIFY_REPORT,
        "}release-end 2026-11-04T18:52:00Z mode=verify exit_code=0 total=32.9s\n",
        *(verb_result() if result is None else result),
    ]


def join_stream(*, result: list[str] | None = None) -> list[str]:
    return [
        "fleet-wave 2026-11-04T19:01:43Z cluster=hp phase=join wave=1/2 ready=0/4 "
        "nodes=n total=246.1s\n",
        "fleet-wave 2026-11-04T19:02:54Z cluster=hp phase=join wave=2/2 ready=1/4 "
        "nodes=n total=316.5s\n",
        "release-end 2026-11-04T19:06:38Z mode=join-cluster exit_code=0 total=540.9s\n",
        "release-begin 2026-11-04T19:06:48Z mode=verify dry_run=false\n",
        *VERIFY_REPORT,
        "}release-end 2026-11-04T19:07:38Z mode=verify exit_code=0 total=49.1s\n",
        *(verb_result() if result is None else result),
    ]


class FakeProcess:
    """A spawned verb: its merged stdout and, once waited for, its exit code."""

    def __init__(self, lines: list[str], returncode: int) -> None:
        self.stdout = io.StringIO("".join(lines))
        self.returncode: int | None = None
        self.exit_code = returncode

    def wait(self, timeout: float | None = None) -> int:
        self.returncode = self.exit_code
        return self.exit_code


class World:
    """The fake site one runner sees; configure it, then drive the runner."""

    def __init__(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        *,
        window: str = membership.WINDOW_KUBELET,
    ) -> None:
        self.settings = settings(tmp_path, window=window)
        self.state_dir = self.settings.state_dir
        self.run_dir = tmp_path / "run"
        self.case_dir = self.run_dir / "cases" / runner.CASE_ID
        self.case_dir.mkdir(parents=True)
        self.clock = Clock()
        self.calls: list[tuple[str, Any]] = []
        self.failures: dict[str, BaseException] = {}
        self.lock = threading.Lock()
        self.identity = {"release_id": "release-test", "cluster_id": OTHER_ID}
        self.store: dict[str, Any] = {
            "open_incidents": [],
            "node_workflows": [],
            "target_agents": [{"node_id": "hp-node-1", "lifecycle_state": "ACTIVE"}],
            "other_agents": [{"node_id": OTHER_NODE, "lifecycle_state": "ACTIVE"}],
        }
        self.after: dict[str, Any] = {
            "open_incidents": [],
            "target_agents": [{"node_id": "hp-node-1", "lifecycle_state": "ACTIVE"}],
            "other_agents": [{"node_id": OTHER_NODE, "lifecycle_state": "ACTIVE"}],
            "departed_ledger": [],
            "notifications": [],
        }
        self.baseline_notifications: list[dict[str, Any]] = []
        self.gpu_nodes = [
            {"name": OTHER_NODE, "ready": "True"},
            {"name": PEER_NODE, "ready": "True"},
        ]
        self.generation = '{"b250": true, "record": "membership-verify/findings.json"}'
        self.streams = {
            membership.REMOVE_VERB: remove_stream(),
            membership.JOIN_VERB: join_stream(),
        }
        self.exit_codes = {membership.REMOVE_VERB: 0, membership.JOIN_VERB: 0}
        self.status_rc = 0
        self.status_stderr = STATUS_LINE + "\n"
        self.rerun_rc = 0
        self.rerun_stream = verb_result()
        self.pending_plan: dict[str, Any] | None = None
        self.node_scope_status = "ok"
        self.registry_verdict = "PASS"
        self.kubelet_stopped = False
        self.stop_takes_node_down = True
        self.node_stays_down = False
        self.windows: list[FakeWindow] = []
        self.write_surfaces()
        self.fixture = FakeFixture(self)
        self.warm = FakeWarm(self)
        monkeypatch.setattr(runner, "RegionalLiveFixture", lambda _s: self.fixture)
        monkeypatch.setattr(
            warm_spare_fixture, "WarmSpareLiveFixture", lambda *_a, **_k: self.warm
        )
        monkeypatch.setattr(warm_spare_fixture, "WarmSpareServiceFixture", self.window)
        monkeypatch.setattr(
            runner,
            "subprocess",
            SimpleNamespace(
                Popen=self.popen,
                run=self.run,
                PIPE=subprocess.PIPE,
                STDOUT=subprocess.STDOUT,
                SubprocessError=subprocess.SubprocessError,
            ),
        )
        monkeypatch.setattr(runner, "CLOCK_NOW", self.clock.now)
        monkeypatch.setattr(runner, "time", SimpleNamespace(sleep=self.clock.sleep))
        monkeypatch.setattr(runner, "pending_profile_plan", self.pending)
        monkeypatch.setattr(runner.ownership, "reapply_report", self.reapply)
        monkeypatch.setattr(runner.node_scope, "precondition_report", self.scope)
        monkeypatch.setattr(runner.registry, "site_cpu_target", self.cpu_target)
        monkeypatch.setattr(runner.registry, "kubectl_control", self.control)
        monkeypatch.setattr(runner.registry, "cpu_inventory", self.inventory)
        monkeypatch.setattr(runner.registry, "check_operation", self.check)
        monkeypatch.setattr(membership, "NOT_READY_POLL_SECONDS", 0.01)

    # -- bookkeeping ----------------------------------------------------------
    def call(self, name: str, detail: Any = None, *, key: str = "") -> None:
        """Log one fake action; raise what ``failures`` holds for its name or
        its narrower ``key`` (one window of several)."""

        with self.lock:
            self.calls.append((name, deepcopy(detail)))
        for failure in (key, name):
            if failure and failure in self.failures:
                raise self.failures[failure]

    def names(self) -> list[str]:
        with self.lock:
            return [name for name, _detail in self.calls]

    def details(self, name: str) -> list[Any]:
        with self.lock:
            return [detail for item, detail in self.calls if item == name]

    def evidence(self, name: str = "") -> dict[str, Any]:
        path = self.case_dir / (name or f"{runner.CASE_ID}.json")
        return json.loads(path.read_text(encoding="utf-8"))

    def write_surfaces(self) -> None:
        """b250's state-directory surfaces: VERIFIED evidence of both verbs
        and the findings record, each naming the other cluster's finding."""

        verified = {"VERIFIED": {"not_gating_findings": [dict(FINDING)]}}
        for verb, key in (
            (membership.REMOVE_VERB, "cluster_id"),
            (membership.JOIN_VERB, "requested_cluster_id"),
        ):
            path = self.state_dir / verb / "run-1" / "state.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            state = {key: TARGET_ID, "phase": "COMPLETED", "evidence": verified}
            path.write_text(json.dumps(state), encoding="utf-8")
        record = membership.findings_record_path(self.state_dir)
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text(
            json.dumps({"not_gating_findings": [dict(FINDING)]}), encoding="utf-8"
        )

    def plan(self) -> dict[str, Any]:
        preflight = runner.read_only_preflight(self.settings, self.case_dir)
        details = runner.plan_details(self.settings, preflight)
        (self.case_dir / "plan.json").write_text(
            json.dumps({"details": details}), encoding="utf-8"
        )
        return preflight

    def execute(self) -> tuple[int, dict[str, Any]]:
        deadline = self.clock.now() + timedelta(hours=2)
        code = runner.execute_case(self.settings, self.run_dir, 1, deadline)
        return code, self.evidence()

    def driver(self) -> Any:
        return runner.LiveMembershipDriver(
            self.settings, self.case_dir, self.clock.now() + timedelta(hours=2)
        )

    # -- the deploy-host CLI ------------------------------------------------
    def popen(self, command: list[str], **kwargs: Any) -> FakeProcess:
        verb = next(item for item in membership.VERBS if item in command)
        self.call(f"verb.spawn:{verb}", {"command": command, "cwd": kwargs["cwd"]})
        return FakeProcess(self.streams[verb], self.exit_codes[verb])

    def run(self, command: list[str], **kwargs: Any) -> Any:
        if command[1:3] == ["-I", "-c"]:
            self.call("generation", {"interpreter": command[0], **kwargs})
            return subprocess.CompletedProcess(command, 0, self.generation, "")
        if "status" in command:
            self.call("status", command)
            return subprocess.CompletedProcess(
                command, self.status_rc, "", self.status_stderr
            )
        self.call("verb.rerun", command)
        return subprocess.CompletedProcess(
            command, self.rerun_rc, "".join(self.rerun_stream), ""
        )

    def pending(self, state: Path) -> dict[str, Any] | None:
        self.call("profile-plan", str(state))
        return deepcopy(self.pending_plan)

    # -- AWS-side helpers ----------------------------------------------------
    def reapply(self, cpu_arn: str, arns: list[str], **kwargs: Any) -> Any:
        self.call("ownership.reapply", {"cpu": cpu_arn, "arns": arns, **kwargs})
        return {"reapplied": list(arns)}, ["reapplied"]

    def scope(self, cpu_arn: str, arns: list[str]) -> Any:
        self.call("node_scope.precondition", {"cpu": cpu_arn, "arns": arns})
        clusters = [{"arn": arn, "status": self.node_scope_status} for arn in arns]
        return {"clusters": clusters}, []

    def cpu_target(self, state: Path) -> tuple[str, str]:
        self.call("registry.target", str(state))
        return CPU_ARN, "ctx-cpu"

    def control(self, *target: str) -> Any:
        self.call("registry.control", list(target))
        return lambda *args, **kwargs: ""

    def inventory(self, control: Any) -> dict[str, Any]:
        self.call("registry.inventory")
        return {"api-ha": {"registered": [TARGET_ID, OTHER_ID]}}

    def check(self, control: Any, *, baseline: Any, operation: str) -> Any:
        self.call("registry.check", {"baseline": baseline, "operation": operation})
        return {"verdict": self.registry_verdict, "operation": operation}

    def window(self, warm: Any, **kwargs: Any) -> FakeWindow:
        value = FakeWindow(self, kwargs)
        self.windows.append(value)
        return value


class FakeFixture:
    """``RegionalLiveFixture`` on the other cluster: the store and Node reads."""

    def __init__(self, world: World) -> None:
        self.w = world
        self.settings = world.settings.regional

    def evidence_identity(self) -> dict[str, str]:
        self.w.call("identity")
        return dict(self.w.identity)

    def release_id(self) -> str:
        return self.w.identity["release_id"]

    def cpu_python(self, script: str, *args: str) -> dict[str, Any]:
        if "node_workflows" in script:
            self.w.call("store.precondition", list(args))
            return deepcopy(self.w.store)
        if "departed_ledger" in script:
            self.w.call("store.after", list(args))
            return deepcopy(self.w.after)
        self.w.call("store.notifications", list(args))
        return {"notifications": deepcopy(self.w.baseline_notifications)}

    def gpu_nodes(self) -> list[dict[str, Any]]:
        self.w.call("gpu_nodes")
        return deepcopy(self.w.gpu_nodes)


class FakeWarm:
    """The warm-spare fixture's Node read of the other node."""

    def __init__(self, world: World) -> None:
        self.w = world

    def node_snapshot(self, node: str) -> dict[str, Any]:
        self.w.call("warm.snapshot", node)
        ready = "Unknown" if self.w.kubelet_stopped else "True"
        return {"name": node, "uid": f"uid-{node}", "ready": ready}


class FakeWindow:
    """The DESTR-008 kubelet window on the other node."""

    def __init__(self, world: World, options: dict[str, Any]) -> None:
        self.w = world
        self.options = options
        self.run_id = str(options["run_id"])
        #: The verb the window was opened for (the run id's last token).
        self.verb = next(
            verb for verb in membership.VERBS if self.run_id.endswith(verb)
        )
        self.window = SimpleNamespace(data={"binding": None})

    def act(self, action: str) -> None:
        self.w.call(f"window.{action}", self.run_id, key=f"window.{action}:{self.verb}")

    def create(self) -> None:
        self.act("create")

    def stop(self, service: str, **kwargs: Any) -> dict[str, Any]:
        self.w.call("window.stop", {"service": service, **kwargs})
        if self.w.stop_takes_node_down:
            self.w.kubelet_stopped = True
        return {"stopped": service}

    def cleanup(self) -> dict[str, Any]:
        self.act("cleanup")
        if not self.w.node_stays_down:
            self.w.kubelet_stopped = False
        return {"phase": "CLOSED", "run_id": self.run_id}

    def resume_cleanup(self) -> dict[str, Any]:
        self.act("resume")
        return {"resumed": self.run_id}

    def close(self) -> None:
        self.act("close")
