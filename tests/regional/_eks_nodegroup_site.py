"""A plain EKS acceptance site for the BOOT-036/037 runners, without a cluster.

The tool side is the recorded world of the verb's own tests
(``tests/admin/_eks_nodegroup_world``): ``ToolDouble`` runs the real
``gpu-fault-admin prepare-eks-nodegroup`` in-process against it, with the
CLI's argument parsing and its failure reporting (``report_failure``: exit 2,
one ``gpu-fault-admin: <reason>`` line), and its fake AWS echoes every call the
way ``CommandRunner`` does. The runner side is ``FixtureDouble``: the same
world's provider reads, Kubernetes Nodes made from the group's instances, the
control plane's store (incidents, workflows, Agents) and the release identity.
"""

from __future__ import annotations

import contextlib
import copy
import io
import json
import sys
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from gpu_fault.admin import cli as admin_cli
from gpu_fault.admin import eks_nodegroup_prepare as verb
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.command_log import report_failure
from gpu_fault.admin.diagnostics import diagnostic_command
from scripts.e2e.regional import eks_nodegroup_case as case
from tests.admin import _eks_nodegroup_world as tool_world

REGION = tool_world.REGION
ACCOUNT = tool_world.ACCOUNT
SITE_ID = tool_world.SITE_ID
GPU_GROUP = tool_world.GPU_GROUP
CPU_GROUP = tool_world.CPU_GROUP
GPU_ARN = tool_world.GPU_ARN
CPU_EKS_ARN = tool_world.CPU_EKS_ARN
CPU_HYPERPOD = "control-plane-fixture"
HP_NAME = "hp-member-fixture"
HP_EKS_ARN = f"arn:aws:eks:{REGION}:{ACCOUNT}:cluster/hp-member-eks"
HP_ARN = f"arn:aws:sagemaker:{REGION}:{ACCOUNT}:cluster/hpmember01"
CLUSTER_ID = "gpu-eks-cluster"
RELEASE_ID = "rel-147"
NOW = datetime(2030, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
PIN = "4.6.1-1"
PREPARE = (
    "--nodegroup",
    GPU_GROUP,
    "--dcgm-hostengine",
    "host",
    "--dcgm-version",
    PIN,
    "--adopt-user-data-part",
    "2",
)


def site_document(
    *, provider: str = "eks-managed", members: bool = True, dcgm: str = "host"
) -> dict[str, Any]:
    clusters: list[dict[str, Any]] = []
    if members:
        clusters.append(
            {
                "clusterId": "hp-member",
                "hyperpodClusterName": HP_NAME,
                "eksClusterArn": HP_EKS_ARN,
                "allowedNamespaces": ["training"],
            }
        )
    entry: dict[str, Any] = {
        "clusterId": CLUSTER_ID,
        "provider": provider,
        "eksClusterArn": GPU_ARN,
        "allowedNamespaces": ["training"],
    }
    if dcgm:
        entry["dcgmHostEngine"] = dcgm
    clusters.append(entry)
    return {
        "apiVersion": "gpu-fault.aws/v1alpha1",
        "kind": "RegionalSite",
        "metadata": {"name": SITE_ID},
        "spec": {
            "awsRegion": REGION,
            "namespace": "gpu-fault-system",
            "cpu": {"eksArn": CPU_EKS_ARN, "hyperpodClusterName": CPU_HYPERPOD},
            "clusters": clusters,
        },
    }


def write_site(state_dir: Path, **options: Any) -> Path:
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / "site.yaml"
    path.write_text(yaml.safe_dump(site_document(**options)), encoding="utf-8")
    return path


class EchoAws(tool_world.FakeAws):
    """The verb's fake AWS, echoing each call to stderr as the CLI does."""

    def run(
        self, arguments: Sequence[str], *, mutate: bool = False, **kwargs: Any
    ) -> str:
        print(f"+ {diagnostic_command(list(arguments))}", file=sys.stderr, flush=True)
        return super().run(arguments, mutate=mutate, **kwargs)


class ToolDouble:
    """``gpu-fault-admin prepare-eks-nodegroup`` in-process, as the CLI runs it."""

    def __init__(self, world: tool_world.World) -> None:
        self.world = world
        self.clock = tool_world.FakeClock()
        self.runs: list[list[str]] = []

    def __call__(self, arguments: Sequence[str], *, timeout: float) -> case.ToolRun:
        self.runs.append(list(arguments))
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                parsed = admin_cli.parser().parse_args([verb.VERB, *arguments])
                code = verb.run_command(
                    parsed,
                    runner=EchoAws(self.world),
                    discover=lambda _runner, **_arguments: tool_world.cpu_identity(),
                    identity=lambda: f"arn:aws:sts::{ACCOUNT}:assumed-role/Admin/op",
                    sleep=self.clock.sleep,
                    clock=self.clock.monotonic,
                    now=lambda: NOW,
                )
            except SystemExit as exc:  # --help
                code = int(exc.code or 0)
            except BootstrapError as exc:
                code = report_failure("gpu-fault-admin", exc)
        return case.ToolRun(tuple(arguments), code, out.getvalue(), err.getvalue())


class RunnerClock:
    """The runners' clock: sleeping moves both clocks."""

    def __init__(self, now: datetime = NOW) -> None:
        self.current = now
        self.ticks = 0.0
        self.slept: list[float] = []

    def now(self) -> datetime:
        return self.current

    def monotonic(self) -> float:
        return self.ticks

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.ticks += seconds
        self.current += timedelta(seconds=seconds)


def node_name(instance: str) -> str:
    return f"ip-{instance}.{REGION}.compute.internal"


class FixtureDouble:
    """``RegionalLiveFixture`` as the runners use it, over one ``World``."""

    def __init__(self, site: Site) -> None:
        self.site = site
        self.aws = EchoAws(site.world)
        self.runs: list[list[str]] = []

    def run(
        self, command: list[str], *, check: bool = True, timeout: float = 300, **_: Any
    ) -> SimpleNamespace:
        self.runs.append(list(command))
        if command[0] != "aws":
            raise AssertionError(f"unexpected command {command}")
        with contextlib.redirect_stderr(io.StringIO()):
            stdout = self.aws.run(command)
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    def kubectl(self, plane: str, *arguments: str, **_: Any) -> str:
        assert plane == "gpu" and arguments[:2] == ("get", "nodes"), arguments
        selector = arguments[arguments.index("-l") + 1]
        group = selector.split("=", 1)[1]
        return json.dumps({"items": self.site.nodes(group)})

    def cpu_python(self, script: str, *arguments: str, **_: Any) -> dict[str, Any]:
        assert script == case.STORE_PROBE, "the runners read the store probe only"
        return self.site.store(arguments[0], arguments[1])

    def evidence_identity(self) -> dict[str, str]:
        return {"release_id": self.site.release_id, "cluster_id": CLUSTER_ID}

    def business_workloads(self, node: str) -> list[dict[str, str]]:
        return list(self.site.workloads.get(node, []))

    def node_workload_view(self, node: str) -> dict[str, Any]:
        return {
            "workload_state": "IDLE",
            "attempt_ids": list(self.site.attempts.get(node, [])),
            "workload_ids": [],
        }


class Site:
    """The world, the state directory, the Nodes, the store and the clocks."""

    def __init__(
        self, root: Path, *, user_data: str | None = None, **site: Any
    ) -> None:
        self.state_dir = root / "state"
        write_site(self.state_dir, **site)
        self.world = tool_world.World(
            user_data=tool_world.site_user_data() if user_data is None else user_data
        )
        self.world.hyperpods = [
            {
                "ClusterName": HP_NAME,
                "ClusterArn": HP_ARN,
                "Orchestrator": {"Eks": {"ClusterArn": HP_EKS_ARN}},
            },
            {
                "ClusterName": CPU_HYPERPOD,
                "ClusterArn": tool_world.CPU_HYPERPOD_ARN,
                "Orchestrator": {"Eks": {"ClusterArn": CPU_EKS_ARN}},
            },
        ]
        self.tool = ToolDouble(self.world)
        self.fixture = FixtureDouble(self)
        self.clock = RunnerClock()
        self.release_id = RELEASE_ID
        self.ready_age = timedelta(hours=1)
        self.cordoned: set[str] = set()
        self.taints: dict[str, list[dict[str, Any]]] = {}
        self.unready: set[str] = set()
        self.workloads: dict[str, list[dict[str, str]]] = {}
        self.attempts: dict[str, list[str]] = {}
        self.incidents: list[dict[str, Any]] = []
        self.workflows: list[dict[str, Any]] = []
        #: Agents not yet ACTIVE (the installer has not run there).
        self.pending_agents: set[str] = set()
        #: Departed nodes whose Agent the cleanup never retired.
        self.lingering_agents: set[str] = set()
        #: Departed Node objects still listed (NotReady) for this many reads.
        self.linger_reads = 0
        self.seen: dict[str, str] = {}
        self.store_reads: list[tuple[str, str]] = []

    # --- the tool ------------------------------------------------------------

    def prepare(self, *flags: str) -> case.ToolRun:
        """Run the verb's apply once: the group as a site would have it."""

        run = self.tool(
            [
                "--cpu-cluster-arn",
                CPU_EKS_ARN,
                "--gpu-cluster-arn",
                GPU_ARN,
                "--state-dir",
                str(self.state_dir),
                *(flags or PREPARE),
                "--apply",
                "--confirm",
                verb.CONFIRMATION,
                "--reference",
                "CHG-147",
            ],
            timeout=60,
        )
        assert run.returncode == 0, run.stderr[-2000:]
        self.tool.runs.clear()
        self.world.calls.clear()
        return run

    # --- Kubernetes ----------------------------------------------------------

    def nodes(self, group: str) -> list[dict[str, Any]]:
        labels = {
            "eks.amazonaws.com/nodegroup": group,
            **(self.world.nodegroups.get(group, {}).get("labels") or {}),
        }
        instances = list(self.world.instances.get(group, []))
        departed = [
            instance
            for name, instance in self.seen.items()
            if instance not in instances and self.linger_reads > 0
        ]
        if departed:
            self.linger_reads -= 1
        items = []
        for instance in [*instances, *departed]:
            name = node_name(instance)
            if instance in instances:
                self.seen.setdefault(name, instance)
            ready = "False" if name in self.unready or instance in departed else "True"
            items.append(
                {
                    "metadata": {
                        "name": name,
                        "uid": f"uid-{instance}",
                        "labels": copy.deepcopy(labels),
                    },
                    "spec": {
                        "providerID": f"aws:///{REGION}c/{instance}",
                        "unschedulable": name in self.cordoned,
                        "taints": [
                            {
                                "key": "nvidia.com/gpu",
                                "value": "present",
                                "effect": "NoSchedule",
                            },
                            *self.taints.get(name, []),
                        ],
                    },
                    "status": {
                        "conditions": [
                            {
                                "type": "Ready",
                                "status": ready,
                                "lastTransitionTime": (
                                    self.clock.now() - self.ready_age
                                ).isoformat(),
                            }
                        ]
                    },
                }
            )
        return items

    # --- the store -----------------------------------------------------------

    def store(self, cluster_id: str, since: str) -> dict[str, Any]:
        self.store_reads.append((cluster_id, since))
        current = {
            node_name(instance)
            for instances in self.world.instances.values()
            for instance in instances
        }
        for name in current:
            self.seen.setdefault(name, name.removeprefix("ip-").split(".", 1)[0])
        agents = []
        for name in sorted(self.seen):
            if name in current:
                if name in self.pending_agents:
                    continue
                state = "ACTIVE"
            else:
                state = "ACTIVE" if name in self.lingering_agents else "REVOKED"
            agents.append(
                {
                    "node_id": name,
                    "lifecycle_state": state,
                    "node_instance_id": self.seen[name],
                    "last_seen_at": NOW.isoformat(),
                }
            )
        return {
            "cluster_id": cluster_id,
            "observed_at": self.clock.now().isoformat(),
            "incidents": copy.deepcopy(self.incidents),
            "workflows": copy.deepcopy(self.workflows),
            "agents": agents,
        }


def install(monkeypatch: Any, module: Any, site: Site) -> None:
    """Point a runner module at ``site``."""

    monkeypatch.setattr(module, "RegionalLiveFixture", lambda _settings: site.fixture)
    monkeypatch.setattr(module, "admin_tool", lambda _fixture, _state_dir: site.tool)
    monkeypatch.setattr(module, "CLOCK", site.clock)


def kubeconfigs(root: Path) -> tuple[Path, Path]:
    cpu, gpu = root / "cpu.kubeconfig", root / "gpu.kubeconfig"
    for path in (cpu, gpu):
        path.write_text("apiVersion: v1\n", encoding="utf-8")
    return cpu, gpu


def runner_arguments(root: Path, site: Site, *extra: str) -> list[str]:
    cpu, gpu = kubeconfigs(root)
    return [
        "--run-dir",
        str(root / "run"),
        "--cpu-kubeconfig",
        str(cpu),
        "--gpu-kubeconfig",
        str(gpu),
        "--gpu-context",
        "gpu-context",
        "--cluster-id",
        CLUSTER_ID,
        "--region",
        REGION,
        "--state-dir",
        str(site.state_dir),
        *extra,
    ]


def approve(case_dir: Path, details: dict[str, Any]) -> None:
    """Write the ``plan.json`` an execute compares its preflight with."""

    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "plan.json").write_text(
        json.dumps({"details": details}, default=str), encoding="utf-8"
    )
