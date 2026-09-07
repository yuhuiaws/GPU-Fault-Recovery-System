#!/usr/bin/env python3
"""GF-REGIONAL-DESTR-015 live acceptance runner.

One two-node PyTorchJob (node-a + node-b, 16 GPUs). Both nodes take a real
``/dev/kmsg`` XID 46 inside the multi-node aggregation window, so the two faults
land in ONE job DAG: a shared STOP_WORKLOADS, one hardware branch per node
(cordon, quiesce, verify, reset, restore services, validate, restore
scheduling) running in parallel, and a single RESTART_WORKLOAD join that runs
only after both branches released their nodes. The job is restarted exactly
once, on the same two nodes, with its 16 GPUs; nothing reboots or replaces.

DESTR-014 proves the failure side of this DAG (a branch exhausts, the join
never runs). This case proves the happy side, which is the system's core
multi-node promise and had no live evidence.

The runner defaults to ``--plan``; ``--execute`` needs ``--confirm
DESTR015_EXECUTE``. The verdict functions are pure and unit-tested; the runner
records digests only.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e2e.regional import (  # noqa: E402
    run_destr009_workload_restart as workload_case,
)
from scripts.e2e.regional.acceptance_runner_common import (  # noqa: E402
    write_json_atomic,
)
from scripts.e2e.regional.control_plane_env_window import (  # noqa: E402
    observed_value,
    replica_env,
)
from scripts.e2e.regional.destr015_verdicts import (  # noqa: E402
    AGENT_OPERATIONS,
    cloudtrail_errors,
    estimated_duration_seconds,
    event_observed_at,
    host_errors,
    injection_errors,
    lifetime_errors,
    schedulability_errors,
    spread_errors,
    step_transitions,
    workflow_errors,
    workload_errors,
)
from scripts.e2e.regional.host_probe_fixture import (  # noqa: E402
    HostProbeFixture,
    HostProbeSettings,
)
from scripts.e2e.regional.live_driver_guard import (  # noqa: E402
    CaseRunner,
    add_live_arguments,
    record_focused_tests,
    reusable_focused_tests,
    run_standard_case,
)
from scripts.e2e.regional.managed_workload_fixture import (  # noqa: E402
    ImagePrewarmFixture,
    ManagedWorkloadFixture,
    ManagedWorkloadSettings,
)
from scripts.e2e.regional.regional_live_fixture import (  # noqa: E402
    RegionalFixtureError,
    RegionalLiveFixture,
    RegionalLiveSettings,
    predecessor_evidence,
    required,
    run_case_main,
    runtime_identity_errors,
    settings_from_arguments,
)
from scripts.e2e.regional.warm_spare_fixture import (  # noqa: E402
    WarmSpareLiveFixture,
)

yaml = importlib.import_module("yaml")

PROBE_SCRIPT = Path(__file__).with_name("probes") / "destructive_node_probe.py"
DEFAULT_MANIFEST = (
    Path(__file__).with_name("manifests")
    / "training"
    / "two-node-replace-pytorchjob.yaml"
)
CASE_ID = "GF-REGIONAL-DESTR-015"
PREDECESSOR_CASE_ID = "GF-REGIONAL-DESTR-012"
CONFIRMATION = "DESTR015_EXECUTE"
EXPECTED_GPU_COUNT = 16
GPUS_PER_NODE = 8
# The two kmsg writes must land inside one aggregation window; a window this
# narrow cannot absorb two kubectl execs plus collector latency.
MIN_AGGREGATION_WINDOW_SECONDS = 5
OBSERVATION_BUDGET_SECONDS = 2400
# The bounds this case fits inside, read from the control-worker replicas that
# apply them. ``orchestration/coordinator.py`` reads the aggregation window,
# ``execution/config.py`` the job workflow lifetime; the release-state ConfigMap
# carries neither.
CONTROL_PLANE = "cpu"
CONTROL_DEPLOYMENT = "gpu-fault-control-worker"
AGGREGATION_WINDOW_VARIABLE = "GPU_FAULT_MULTI_NODE_AGGREGATION_WINDOW_SECONDS"
JOB_LIFETIME_VARIABLE = "GPU_FAULT_JOB_WORKFLOW_MAX_LIFETIME_SECONDS"
CONTROL_VARIABLES = (AGGREGATION_WINDOW_VARIABLE, JOB_LIFETIME_VARIABLE)
# Shipped defaults, used only when the deployment does not override them.
DEFAULT_AGGREGATION_WINDOW_SECONDS = 5
DEFAULT_JOB_LIFETIME_SECONDS = 3600


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested)
# --------------------------------------------------------------------------- #
def preflight_errors(
    *,
    nodes: tuple[str, str],
    node_snapshots: dict[str, dict[str, Any]],
    agents: dict[str, dict[str, Any]],
    profile: dict[str, Any] | None,
    queue: dict[str, Any],
    remote_commands: dict[str, Any],
    gpu_workloads: list[dict[str, Any]],
    business_workloads: dict[str, list[dict[str, Any]]],
    predecessor: dict[str, Any],
    tests: dict[str, Any],
    control_env: dict[str, Any],
    runtime_identity_errors: list[str],
    arbiter_pods: dict[str, list[str]] | None = None,
    dns_nodes: list[str] | None = None,
) -> list[str]:
    errors: list[str] = list(runtime_identity_errors)
    if len(set(nodes)) != 2:
        errors.append("the two target nodes are identical")
    # Precondition 2 of the spec: neither node may host a controller or
    # executor arbiter replica. And a pair that holds every kube-dns endpoint
    # takes cluster DNS down when both are quiesced (kubelet stops, the nodes
    # go NotReady, the endpoints drop) -- the executor then cannot resolve the
    # control plane and the RESTORE step fails with gaierror, as it did live.
    for node in nodes:
        hosted = (arbiter_pods or {}).get(node) or []
        if hosted:
            errors.append(f"{node} hosts controller/executor arbiter Pods: {hosted}")
    if dns_nodes is not None and dns_nodes and set(dns_nodes) <= set(nodes):
        errors.append(
            "the two target nodes hold every kube-dns endpoint; quiescing both "
            "would take cluster DNS down"
        )
    if not predecessor.get("valid"):
        errors.append("DESTR-012 predecessor evidence is not PASS")
    if not tests.get("passed"):
        errors.append("focused regression tests failed")
    for node in nodes:
        snapshot = node_snapshots.get(node) or {}
        if (
            snapshot.get("ready") != "True"
            or snapshot.get("unschedulable")
            or snapshot.get("taints")
            or snapshot.get("ownership_annotations")
        ):
            errors.append(f"{node} is not Ready, schedulable and untainted")
        if int(snapshot.get("gpu_allocatable") or 0) != GPUS_PER_NODE:
            errors.append(f"{node} does not allocate {GPUS_PER_NODE} GPUs")
        agent = agents.get(node) or {}
        if agent.get("lifecycle_state") != "ACTIVE":
            errors.append(f"{node} Node Agent is not ACTIVE")
        missing = sorted(
            set(AGENT_OPERATIONS) - set(agent.get("allowed_operations") or [])
        )
        if missing:
            errors.append(f"{node} Node Agent does not allow {missing}")
        if business_workloads.get(node):
            errors.append(
                f"{node} carries a critical system workload: {business_workloads[node]}"
            )
    errors.extend(workload_case.managed_owner_errors(profile))
    reset = workload_case.capability(profile, "gpuReset")
    if reset is None:
        errors.append("runtime profile has no gpuReset capability")
    elif reset.get("mode") != "OWN" or reset.get("owner") != "gpu-fault-node-agent":
        errors.append("gpuReset is not OWN by the Node Agent")
    if (profile or {}).get("warnings"):
        errors.append("runtime profile has warnings")
    if int(queue.get("depth") or 0):
        errors.append("processor queue is not empty")
    if remote_commands.get("open_by_cluster"):
        errors.append("remote command queue is not empty")
    if gpu_workloads:
        errors.append(f"GPU cluster already has a GPU workload: {gpu_workloads}")
    window = int(control_env.get("aggregation_window_seconds") or 0)
    if window < MIN_AGGREGATION_WINDOW_SECONDS:
        errors.append(
            f"multi-node aggregation window {window}s is below "
            f"{MIN_AGGREGATION_WINDOW_SECONDS}s; two kmsg writes cannot land in it"
        )
    errors.extend(
        lifetime_errors(
            estimated_seconds=estimated_duration_seconds(),
            lifetime_seconds=control_env.get("job_lifetime_seconds"),
        )
    )
    return errors


def control_env_record(observations: dict[str, str | None]) -> dict[str, Any]:
    """Fold what every ready control-worker replica agrees on into the two
    bounds this case does arithmetic against.

    A value no replica agrees on (mid-rollout) or one that is not a number is
    reported as the shipped default rather than averaged; the rollout itself is
    caught by the runtime-identity check.
    """

    def number(name: str, default: int) -> int:
        value = observations.get(name)
        try:
            return int(str(value)) if value is not None else default
        except ValueError:
            return default

    return {
        "observed": dict(observations),
        "aggregation_window_seconds": number(
            AGGREGATION_WINDOW_VARIABLE, DEFAULT_AGGREGATION_WINDOW_SECONDS
        ),
        "job_lifetime_seconds": number(
            JOB_LIFETIME_VARIABLE, DEFAULT_JOB_LIFETIME_SECONDS
        ),
    }


def stop_before_observation(errors: list[str]) -> None:
    """Refuse to keep going once the injection itself has failed.

    Two faults that did not land in one DAG, or landed outside the window,
    cannot prove the join; observing the workflow for forty minutes after that
    would only bury the real error. The faults are already on the nodes, so the
    cleanup still waits for quiescence and restores anything left isolated.
    """

    if errors:
        raise RegionalFixtureError(
            "stopping before observation: the injection did not set up the case: "
            + "; ".join(errors)
        )


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str).encode()
    ).hexdigest()


def plan_identity(
    preflight: dict[str, Any], *, nodes: tuple[str, str]
) -> dict[str, Any]:
    snapshots = preflight.get("nodes") or {}
    store = preflight.get("store") or {}
    return {
        "release_id": preflight.get("release_id"),
        "node_uids": {node: (snapshots.get(node) or {}).get("uid") for node in nodes},
        "node_boot_ids": {
            node: (snapshots.get(node) or {}).get("boot_id") for node in nodes
        },
        "runtime_profile_version": (store.get("profile") or {}).get("profile_version"),
        "runtime_identity": preflight.get("runtime_identity"),
    }


def identity_digest(identity: dict[str, Any]) -> str:
    return _digest(identity)


def evidence_components(details: dict[str, Any]) -> dict[str, str]:
    return {name: _digest(value) for name, value in details.items()}


def case_digest(components: dict[str, str]) -> str:
    return _digest(components)


def render_two_node_manifest(
    source: Path,
    destination: Path,
    *,
    master_node: str,
    worker_node: str,
) -> Path:
    if master_node == worker_node:
        raise ValueError("master and worker nodes must differ")
    document = yaml.safe_load(source.read_text(encoding="utf-8"))
    replicas = document["spec"]["pytorchReplicaSpecs"]
    replicas["Master"]["template"]["spec"]["nodeName"] = master_node
    replicas["Worker"]["template"]["spec"]["nodeName"] = worker_node
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    destination.chmod(0o600)
    return destination


def derived_identity(run_dir: Path, attempt: int) -> tuple[str, str]:
    suffix = hashlib.sha256(
        f"{run_dir.resolve()}\0{attempt}\0{CASE_ID}".encode()
    ).hexdigest()[:12]
    job_id = f"destr015-{suffix}"
    return job_id, f"{job_id}-a001"


# --------------------------------------------------------------------------- #
# Settings / plan
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Settings:
    regional: RegionalLiveSettings
    site_file: Path
    manifest: Path
    host_probe_image: str
    node_a: str
    node_b: str
    pci_bdf_a: str
    pci_bdf_b: str
    job_id: str
    attempt_id: str
    predecessor_path: Path

    @property
    def nodes(self) -> tuple[str, str]:
        return (self.node_a, self.node_b)

    def environment(self) -> dict[str, str]:
        return {
            **self.regional.environment(),
            "GPU_FAULT_SITE_FILE": str(self.site_file),
            "GPU_FAULT_TRAINING_MANIFEST": str(self.manifest),
            "GPU_FAULT_HOST_PROBE_IMAGE": self.host_probe_image,
            "GPU_FAULT_NODE_A": self.node_a,
            "GPU_FAULT_NODE_B": self.node_b,
            "GPU_FAULT_TEST_JOB_ID": self.job_id,
            "GPU_FAULT_TEST_ATTEMPT_ID": self.attempt_id,
            "GPU_FAULT_PREDECESSOR_EVIDENCE": str(self.predecessor_path),
        }


def configure(arguments: argparse.Namespace) -> Settings:
    default_job, default_attempt = derived_identity(
        arguments.run_dir, arguments.attempt
    )
    job_id = arguments.job_id.strip() or default_job
    predecessor = (
        Path(arguments.predecessor_evidence).expanduser().resolve()
        if arguments.predecessor_evidence
        else (
            arguments.run_dir
            / "cases"
            / PREDECESSOR_CASE_ID
            / f"{PREDECESSOR_CASE_ID}.json"
        ).resolve()
    )
    return Settings(
        regional=settings_from_arguments(arguments),
        site_file=Path(
            required(
                arguments.site_file or os.getenv("GPU_FAULT_SITE_FILE", ""),
                "regional site file",
            )
        )
        .expanduser()
        .resolve(),
        manifest=Path(arguments.manifest).expanduser().resolve(),
        host_probe_image=required(
            arguments.host_probe_image or os.getenv("GPU_FAULT_HOST_PROBE_IMAGE", ""),
            "host probe image",
        ),
        node_a=required(
            arguments.node_a or os.getenv("GPU_FAULT_NODE_A", ""), "node A"
        ),
        node_b=required(
            arguments.node_b or os.getenv("GPU_FAULT_NODE_B", ""), "node B"
        ),
        pci_bdf_a=arguments.pci_bdf_a.strip(),
        pci_bdf_b=arguments.pci_bdf_b.strip(),
        job_id=job_id,
        attempt_id=arguments.attempt_id.strip() or default_attempt,
        predecessor_path=predecessor,
    )


def plan_details(settings: Settings, preflight: dict[str, Any]) -> dict[str, Any]:
    identity = plan_identity(preflight, nodes=settings.nodes)
    details: dict[str, Any] = {
        "risk": "destructive",
        "predecessor": preflight.get("predecessor"),
        "node_a": settings.node_a,
        "node_b": settings.node_b,
        "job_id": settings.job_id,
        "attempt_id": settings.attempt_id,
        "manifest": str(settings.manifest),
        "control_env": preflight.get("control_env"),
        "mutation": (
            "submit one two-node 16-GPU PyTorchJob; write one XID 46 to each "
            "node's /dev/kmsg inside the aggregation window so both land in one "
            "job DAG; allow the two branches to cordon, quiesce, reset and "
            "restore their node in parallel; allow the single join to restart "
            "the job once. No reboot, no replacement, no provider call."
        ),
        "preflight_identity": identity,
        "preflight_identity_digest": identity_digest(identity),
        "stop_conditions": [
            "DESTR-012 predecessor evidence is not PASS",
            "either node is busy, tainted, owned, or its Node Agent is not ACTIVE",
            "gpuReset is not OWN by the Node Agent, or workload stop/restart not OWN",
            "the aggregation window is too narrow for two kmsg writes",
            "the estimated duration does not fit the job workflow lifetime",
            "the two XID injections do not land in one DAG workflow",
            "a branch escalates, a step fails, or any reboot/replace appears",
            "the join runs before both branches released their nodes, or twice",
            "the job is not restarted exactly once on the same two nodes",
            "CloudTrail shows any provider mutation",
            "cleanup cannot prove quiescence or leaves a node isolated",
        ],
        "rollback": {
            "runner_waits_for_async_quiescence_before_workload_delete": True,
            "runner_deletes_test_PyTorchJob_only_after_quiescence": True,
            "runner_restores_a_node_left_isolated_via_validated_restore": True,
            "runner_finally_deletes_probe_and_prewarm_Pods": True,
            "no_node_reboot_or_replacement_is_authorized": True,
        },
    }
    # The --plan's focused-test result is recorded with a source digest so
    # --execute can reuse it instead of paying for the same run twice.
    record_focused_tests(
        details,
        dict(preflight.get("focused_tests") or {"passed": False}),
    )
    return details


# --------------------------------------------------------------------------- #
# Live preflight (not unit-tested; delegates every assertion to the pure funcs)
# --------------------------------------------------------------------------- #
def focused_tests(case_dir: Path, *, reuse: bool = False) -> dict[str, Any]:
    """Run the focused pytest, or reuse the --plan's result in --execute.

    ``reuse`` is set by the execution path only: the result recorded in
    ``plan.json`` is taken when it passed against exactly this source tree
    (``reusable_focused_tests`` compares the digest), so the same tests are
    not paid for twice minutes apart; any edit in between forces a rerun.
    """

    if reuse:
        recorded = reusable_focused_tests(case_dir / "plan.json")
        if recorded is not None:
            return {**recorded, "focused_tests_reused": True}
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/orchestration/test_merge.py::"
        "test_parallel_job_dag_accepts_three_and_five_node_branches",
        "tests/execution/_executor_cases_1.py::"
        "test_dag_fans_out_node_branches_and_joins_once",
        "tests/hyperpod/test_e2e_distributed_xid_reset.py::"
        "test_three_node_job_stops_before_two_fault_gpu_resets",
        "tests/execution/test_restart_safety.py::"
        "test_restart_budget_blocks_second_restart_for_same_job",
        "tests/regional/test_destr015_parallel_branch_join.py",
    ]
    completed = RegionalLiveFixture.run(command, cwd=ROOT, check=False, timeout=600)
    path = case_dir / "focused-tests.log"
    path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    path.chmod(0o600)
    return {
        "passed": completed.returncode == 0,
        "returncode": completed.returncode,
        "command": command,
    }


def control_env(regional: RegionalLiveFixture) -> dict[str, Any]:
    """What the deployed control-worker replicas actually read for the two
    bounds this case has to fit inside. Read-only."""

    replicas = replica_env(
        regional,
        plane=CONTROL_PLANE,
        deployment=CONTROL_DEPLOYMENT,
        names=CONTROL_VARIABLES,
    )
    record = control_env_record(
        {name: observed_value(replicas, name) for name in CONTROL_VARIABLES}
    )
    record["replicas"] = replicas
    return record


ARBITER_APPS = (
    "gpu-fault-cluster-executor",
    "gpu-fault-completion-watcher",
    "gpu-fault-node-installer-reconciler",
)


def cluster_arbiter_placement(
    regional: RegionalLiveFixture,
) -> tuple[dict[str, list[str]], list[str]]:
    """Where the GPU cluster's arbiter replicas and kube-dns endpoints run.

    Returns ``({node: [namespace/pod, ...]}, [kube-dns node, ...])`` for the
    running Pods only; terminating Pods have already left the node's fate.
    """

    arbiters: dict[str, list[str]] = {}
    pods = json.loads(
        regional.kubectl(
            "gpu",
            "get",
            "pod",
            "-l",
            f"app in ({','.join(ARBITER_APPS)})",
            "-o",
            "json",
        )
    )
    for item in pods.get("items", []):
        if item.get("metadata", {}).get("deletionTimestamp"):
            continue
        node = item.get("spec", {}).get("nodeName") or ""
        if node:
            name = f"{item['metadata'].get('namespace')}/{item['metadata'].get('name')}"
            arbiters.setdefault(node, []).append(name)
    dns = json.loads(
        regional.kubectl(
            "gpu",
            "get",
            "pod",
            "-n",
            "kube-system",
            "-l",
            "k8s-app=kube-dns",
            "-o",
            "json",
        )
    )
    dns_nodes = sorted(
        {
            str(item.get("spec", {}).get("nodeName") or "")
            for item in dns.get("items", [])
            if not item.get("metadata", {}).get("deletionTimestamp")
            and item.get("spec", {}).get("nodeName")
        }
    )
    return {node: sorted(names) for node, names in arbiters.items()}, dns_nodes


def read_only_preflight(
    settings: Settings,
    case_dir: Path,
    *,
    reuse_focused_tests: bool = False,
) -> dict[str, Any]:
    if not settings.site_file.is_file() or not settings.manifest.is_file():
        raise RegionalFixtureError("site file or training manifest does not exist")
    regional = RegionalLiveFixture(settings.regional)
    snapshots = {node: regional.node_snapshot(node) for node in settings.nodes}
    stores = {node: regional.store_snapshot(node=node) for node in settings.nodes}
    state = stores[settings.node_a]
    runtime_identity = regional.runtime_identity()
    tests = focused_tests(case_dir, reuse=reuse_focused_tests)
    predecessor = predecessor_evidence(
        settings.predecessor_path,
        PREDECESSOR_CASE_ID,
        **regional.evidence_identity(),
    )
    env = control_env(regional)
    arbiter_pods, dns_nodes = cluster_arbiter_placement(regional)
    result: dict[str, Any] = {
        "release_id": state.get("release_id"),
        "nodes": snapshots,
        "arbiter_pods": arbiter_pods,
        "dns_nodes": dns_nodes,
        "agents": {node: stores[node].get("agent") for node in settings.nodes},
        "store": state,
        "runtime_identity": runtime_identity,
        "focused_tests": tests,
        "cpu_blast": regional.cpu_blast_snapshot(),
        "predecessor": predecessor,
        "control_env": env,
    }
    result["errors"] = preflight_errors(
        nodes=settings.nodes,
        node_snapshots=snapshots,
        agents={node: stores[node].get("agent") or {} for node in settings.nodes},
        profile=state.get("profile"),
        queue=state.get("queue") or {},
        remote_commands=state.get("remote_commands") or {},
        gpu_workloads=regional.gpu_workloads(),
        business_workloads={
            node: regional.business_workloads(node) for node in settings.nodes
        },
        predecessor=predecessor,
        tests=tests,
        control_env=env,
        runtime_identity_errors=runtime_identity_errors(runtime_identity),
        arbiter_pods=arbiter_pods,
        dns_nodes=dns_nodes,
    )
    write_json_atomic(case_dir / "preflight.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Run the guarded DESTR-015 acceptance: two nodes of one job fault "
            "inside the aggregation window, reset on parallel branches, and the "
            "job is restarted once by the single join."
        )
    )
    add_live_arguments(value, confirmation=CONFIRMATION)
    value.add_argument("--cpu-kubeconfig", default="")
    value.add_argument("--gpu-kubeconfig", default="")
    value.add_argument("--gpu-context", default="")
    value.add_argument("--namespace", default="gpu-fault-system")
    value.add_argument("--cluster-id", default="")
    value.add_argument("--region", default="")
    value.add_argument("--site-file", default="")
    value.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    value.add_argument("--host-probe-image", default="")
    value.add_argument(
        "--node-a", default="", help="node the Master replica is pinned to"
    )
    value.add_argument(
        "--node-b", default="", help="node the Worker replica is pinned to"
    )
    value.add_argument(
        "--pci-bdf-a",
        default="",
        help="GPU PCI BDF to fault on node A; defaults to the first inventory entry",
    )
    value.add_argument(
        "--pci-bdf-b",
        default="",
        help="GPU PCI BDF to fault on node B; defaults to the first inventory entry",
    )
    value.add_argument("--job-id", default="")
    value.add_argument("--attempt-id", default="")
    value.add_argument("--predecessor-evidence", default="")
    return value


# --------------------------------------------------------------------------- #
# Live execution
# --------------------------------------------------------------------------- #
def _host_probe(settings: Settings, node: str, run_id: str) -> HostProbeFixture:
    return HostProbeFixture(
        HostProbeSettings(
            kubeconfig=settings.regional.gpu_kubeconfig,
            context=settings.regional.gpu_context,
            namespace=settings.regional.namespace,
            node=node,
            image=settings.host_probe_image,
            case_id=CASE_ID,
            run_id=run_id,
            probe_script=PROBE_SCRIPT,
            active_deadline_seconds=3600,
        )
    )


@dataclass
class _LiveRun:
    """Everything one execution owns: fixtures, probes and the flags cleanup
    reads. Kept on one object so the phases below stay short."""

    settings: Settings
    case_dir: Path
    attempt: int
    maintenance_window_end: datetime
    preflight: dict[str, Any]
    regional: RegionalLiveFixture
    run_id: str
    workload: ManagedWorkloadFixture
    prewarm: ImagePrewarmFixture
    probes: dict[str, HostProbeFixture]
    marker: str = ""
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    injected_at: dict[str, str] = field(default_factory=dict)
    bdf: dict[str, str] = field(default_factory=dict)
    baselines: dict[str, dict[str, Any]] = field(default_factory=dict)
    source_uids: set[str] = field(default_factory=set)
    incident_id: str = ""
    workflow_request_id: str = ""
    workload_submitted: bool = False


def _prepare_live_run(
    settings: Settings,
    run_dir: Path,
    attempt: int,
    maintenance_window_end: datetime,
) -> _LiveRun:
    case_dir = run_dir / "cases" / CASE_ID
    case_dir.mkdir(parents=True, exist_ok=True)
    preflight = read_only_preflight(settings, case_dir, reuse_focused_tests=True)
    if preflight["errors"]:
        raise RegionalFixtureError(
            "preflight failed: " + "; ".join(preflight["errors"])
        )
    planned = json.loads((case_dir / "plan.json").read_text(encoding="utf-8"))
    current = identity_digest(plan_identity(preflight, nodes=settings.nodes))
    if planned["details"]["preflight_identity_digest"] != current:
        raise RegionalFixtureError("DESTR-015 plan identity drifted before execution")

    regional = RegionalLiveFixture(settings.regional)
    run_id = f"destr015-{run_dir.name.rsplit('-', 1)[-1].lower()}-a{attempt}"
    pinned = render_two_node_manifest(
        settings.manifest,
        case_dir / "pinned-workload.yaml",
        master_node=settings.node_a,
        worker_node=settings.node_b,
    )
    workload = ManagedWorkloadFixture(
        regional,
        ManagedWorkloadSettings(
            manifest=pinned,
            site_file=settings.site_file,
            job_id=settings.job_id,
            attempt_id=settings.attempt_id,
            restart_budget=1,
            expected_pods=2,
            expected_gpu_count=EXPECTED_GPU_COUNT,
        ),
    )
    return _LiveRun(
        settings=settings,
        case_dir=case_dir,
        attempt=attempt,
        maintenance_window_end=maintenance_window_end,
        preflight=preflight,
        regional=regional,
        run_id=run_id,
        workload=workload,
        prewarm=ImagePrewarmFixture(regional, case_id=CASE_ID, run_id=run_id),
        probes={node: _host_probe(settings, node, run_id) for node in settings.nodes},
    )


def _target_bdf(settings: Settings, node: str, baseline: dict[str, Any]) -> str:
    explicit = settings.pci_bdf_a if node == settings.node_a else settings.pci_bdf_b
    if explicit:
        return explicit
    inventory = baseline.get("gpu_inventory") or []
    if not inventory:
        raise RegionalFixtureError(f"{node} host probe found no GPU inventory")
    return str(inventory[0]["pci_bdf"])


def _start_job_and_probes(run: _LiveRun) -> None:
    settings, case_dir = run.settings, run.case_dir
    run.prewarm.create(list(settings.nodes))
    cached = run.prewarm.cached_nodes()
    write_json_atomic(case_dir / "image-cache.json", {"cached_nodes": cached})
    if not set(settings.nodes) <= set(cached):
        raise RegionalFixtureError("training image is not cached on both nodes")
    run.workload_submitted = True
    write_json_atomic(case_dir / "submission.json", run.workload.submit())
    source = run.workload.wait_running(timeout_seconds=900)
    run.source_uids = {str(item["uid"]) for item in source["pods"]}
    write_json_atomic(case_dir / "workload-source.json", source)
    observation = workload_case.wait_observation(
        run.regional,
        settings,
        node=settings.node_a,
        expected_gpu_count=EXPECTED_GPU_COUNT,
    )
    write_json_atomic(case_dir / "source-observation.json", observation)
    for node, probe in run.probes.items():
        probe.create()
        baseline = probe.execute("snapshot", "--run-id", run.run_id)
        if len(baseline.get("gpu_inventory") or []) != GPUS_PER_NODE:
            raise RegionalFixtureError(
                f"{node} host GPU inventory is not {GPUS_PER_NODE}"
            )
        if baseline.get("quiesce_states"):
            raise RegionalFixtureError(f"{node} has a pre-existing quiesce state")
        if not baseline.get("kmsg_writable"):
            raise RegionalFixtureError(
                f"{node} /dev/kmsg is not writable from the probe"
            )
        run.baselines[node] = baseline
        run.bdf[node] = _target_bdf(settings, node, baseline)
        write_json_atomic(case_dir / f"host-baseline-{node}.json", baseline)
    run.regional.verify_runtime_identity(
        run.preflight["runtime_identity"],
        evidence_path=case_dir / "runtime-identity-before-injection.json",
        stage="before DESTR-015 injection",
    )


def _inject_both(run: _LiveRun) -> tuple[dict[str, Any], dict[str, Any]]:
    """Write one XID 46 to each node concurrently so both land inside the
    aggregation window, then wait until both events show one workflow."""

    settings, case_dir = run.settings, run.case_dir
    if datetime.now(timezone.utc) >= run.maintenance_window_end:
        raise RegionalFixtureError("maintenance window ended before injection")
    run.marker = f"destr015-{int(time.time())}-a{run.attempt}"
    run.started_at = datetime.now(timezone.utc)

    def write(node: str, label: str) -> dict[str, Any]:
        payload = run.probes[node].execute(
            "write-xid46",
            "--marker",
            run.marker,
            "--drill-id",
            f"{run.run_id}-{label}",
            "--pci-bdf",
            run.bdf[node],
        )
        # Stamped on return, i.e. one kubectl round trip after the write. Kept
        # as evidence of when the runner acted; the spread verdict reads the
        # events' own ``observed_at`` from the store instead.
        run.injected_at[node] = datetime.now(timezone.utc).isoformat()
        return payload

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="destr015-xid") as pool:
        futures = {
            node: pool.submit(write, node, label)
            for node, label in zip(settings.nodes, ("a", "b"), strict=True)
        }
        injections = {node: future.result() for node, future in futures.items()}
    write_json_atomic(
        case_dir / "injection.json",
        {"marker": run.marker, "injected_at": run.injected_at, "writes": injections},
    )
    states = []
    for node in settings.nodes:
        (case_dir / node).mkdir(mode=0o700, parents=True, exist_ok=True)
        states.append(
            run.regional.wait_for_workflow(
                node=node,
                marker=run.marker,
                observed_after=run.started_at,
                case_dir=case_dir / node,
                timeout_seconds=300,
                job_id=settings.job_id,
                attempt_id=settings.attempt_id,
                terminal=False,
            )
        )
    run.workflow_request_id = str(
        (states[0].get("workflow") or {}).get("request_id") or ""
    )
    return states[0], states[1]


def _observe_until_terminal(run: _LiveRun) -> dict[str, Any]:
    settings = run.settings
    transitions: dict[str, str] = {}
    timeline: list[dict[str, Any]] = []
    deadline = time.monotonic() + OBSERVATION_BUDGET_SECONDS
    state: dict[str, Any] = {}
    while time.monotonic() < deadline:
        state = run.regional.store_snapshot(
            node=settings.node_a,
            marker=run.marker,
            observed_after=run.started_at,
            job_id=settings.job_id,
            attempt_id=settings.attempt_id,
            queue_attempts=1,
            workflow_request_ids=(
                [run.workflow_request_id] if run.workflow_request_id else None
            ),
        )
        workflow = state.get("workflow") or {}
        transitions, changes = step_transitions(
            transitions, workflow.get("step_executions") or []
        )
        timeline.extend(changes)
        write_json_atomic(
            run.case_dir / "step-timeline.json", {"transitions": timeline}
        )
        if workflow.get("status") in {"SUCCEEDED", "FAILED", "BLOCKED", "SUPERSEDED"}:
            break
        time.sleep(5)
    write_json_atomic(run.case_dir / "workflow-state.json", state)
    return state


def _data_plane_errors(
    run: _LiveRun, state: dict[str, Any]
) -> tuple[list[str], dict[str, Any]]:
    settings, case_dir = run.settings, run.case_dir
    errors: list[str] = []
    if (state.get("workflow") or {}).get("status") == "SUCCEEDED":
        target = run.workload.wait_restarted(run.source_uids, timeout_seconds=900)
    else:
        # A workflow that did not succeed owes no restart; waiting for one
        # would only bury the workflow verdicts under a timeout.
        target = {"pods": run.workload.pods()}
    write_json_atomic(case_dir / "workload-target.json", target)
    errors.extend(
        workload_errors(
            pods=target["pods"], source_uids=run.source_uids, nodes=settings.nodes
        )
    )
    hosts: dict[str, dict[str, dict[str, Any]]] = {}
    for node, probe in run.probes.items():
        after = probe.execute(
            "snapshot",
            "--since-epoch",
            str(run.started_at.timestamp()),
            "--pci-bdf",
            run.bdf[node],
            "--run-id",
            run.run_id,
            timeout=180,
        )
        write_json_atomic(case_dir / f"host-after-{node}.json", after)
        hosts[node] = {"before": run.baselines[node], "after": after}
    errors.extend(
        host_errors(hosts, nodes=settings.nodes, expected_gpu_count=GPUS_PER_NODE)
    )
    snapshots = {node: run.regional.node_snapshot(node) for node in settings.nodes}
    write_json_atomic(case_dir / "nodes-after.json", snapshots)
    errors.extend(schedulability_errors(snapshots, nodes=settings.nodes))
    ended_at = datetime.now(timezone.utc)
    provider = run.regional.provider_events(run.started_at, ended_at)
    # A negative CloudTrail claim cannot be settled inside the delivery window;
    # it is recorded as provisional and re-checked by DESTR-013.
    provisional = run.regional.provider_events_provisional(ended_at)
    write_json_atomic(
        case_dir / "provider-events.json",
        {"events": provider, "provisional": provisional},
    )
    errors.extend(cloudtrail_errors(provider))
    logs = workload_case.control_plane_log_snapshot(
        run.regional, run.started_at, run.workload.name
    )
    write_json_atomic(case_dir / "control-plane-logs.json", logs)
    if logs["suspicious"]:
        errors.append("control-plane logs show a Kubernetes workload write")
    cpu_after = run.regional.cpu_blast_snapshot()
    write_json_atomic(case_dir / "cpu-blast-after.json", cpu_after)
    if cpu_after != run.preflight["cpu_blast"]:
        errors.append("control-plane EKS state differs from baseline")
    run.regional.verify_runtime_identity(
        run.preflight["runtime_identity"],
        evidence_path=case_dir / "runtime-identity-after-restart.json",
        stage="after DESTR-015 workload restart",
    )
    summary: dict[str, Any] = {
        node: {
            "boot_id_before": run.baselines[node].get("boot_id"),
            "boot_id_after": hosts[node]["after"].get("boot_id"),
            "ledger_rows_added": len(hosts[node]["after"].get("ledger") or [])
            - len(run.baselines[node].get("ledger") or []),
        }
        for node in settings.nodes
    }
    summary["provider_events"] = {"count": len(provider), "provisional": provisional}
    return errors, summary


def execute_case(
    settings: Settings,
    run_dir: Path,
    attempt: int,
    maintenance_window_end: datetime,
) -> int:
    """Drive the live case. Not exercised by the unit suite; the verdicts it
    calls are. Every cleanup failure downgrades the verdict to FAIL."""

    run = _prepare_live_run(settings, run_dir, attempt, maintenance_window_end)
    result: dict[str, Any] = {
        "case_id": CASE_ID,
        "attempt": attempt,
        "verdict": "FAIL",
        "job_id": settings.job_id,
        "attempt_id": settings.attempt_id,
        "nodes": list(settings.nodes),
        "maintenance_window_end": maintenance_window_end.isoformat(),
    }
    try:
        _start_job_and_probes(run)
        first, second = _inject_both(run)
        errors = injection_errors(first, second, nodes=settings.nodes)
        observed = event_observed_at(dict(zip(settings.nodes, (first, second))))
        write_json_atomic(
            run.case_dir / "injection-observed.json",
            {"event_observed_at": observed, "exec_returned_at": run.injected_at},
        )
        errors.extend(
            spread_errors(
                observed,
                aggregation_window_seconds=int(
                    run.preflight["control_env"]["aggregation_window_seconds"]
                ),
            )
        )
        # Two faults that did not form one in-window DAG cannot prove the join;
        # the plan lists that as a stop condition, so stop here.
        result["errors"] = list(errors)
        stop_before_observation(errors)
        state = _observe_until_terminal(run)
        workflow = state.get("workflow") or {}
        incident = state.get("incident") or {}
        run.incident_id = str(incident.get("incident_id") or "")
        errors.extend(
            workflow_errors(
                workflow,
                incident,
                nodes=settings.nodes,
                expected_gpu_count=EXPECTED_GPU_COUNT,
                restart_budget=state.get("restart_budget") or {},
            )
        )
        data_errors, hosts = _data_plane_errors(run, state)
        errors.extend(data_errors)
        details = {
            "preflight_identity": plan_identity(run.preflight, nodes=settings.nodes),
            "workflow": workflow,
            "incident": incident,
            "hosts": hosts,
        }
        components = evidence_components(details)
        result.update(
            {
                **run.regional.evidence_identity(),
                "verdict": "PASS" if not errors else "FAIL",
                "errors": errors,
                "marker": run.marker,
                "incident_id": run.incident_id,
                "workflow_request_id": workflow.get("request_id"),
                "dag_revision": workflow.get("dag_revision"),
                "injected_at": run.injected_at,
                "event_observed_at": observed,
                "provider_events": hosts.get("provider_events"),
                "restart_budget": state.get("restart_budget"),
                "case_digest": case_digest(components),
                "components": components,
            }
        )
    except Exception as exc:  # noqa: BLE001 - recorded as the case error
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        cleanup = _cleanup(run)
        result["cleanup"] = cleanup
        if cleanup["errors"]:
            result["verdict"] = "FAIL"
    write_json_atomic(run.case_dir / f"{CASE_ID}.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["verdict"] == "PASS" else 1


def _cleanup(run: _LiveRun) -> dict[str, Any]:
    settings = run.settings
    result: dict[str, Any] = {"errors": []}

    def guard(label: str, action: Any) -> None:
        try:
            result[label] = action()
        except Exception as exc:  # noqa: BLE001 - a cleanup failure is a FAIL
            result["errors"].append(f"{label}: {type(exc).__name__}: {exc}")

    if run.marker:
        guard(
            "quiescence",
            lambda: workload_case.wait_for_cleanup_quiescence(
                regional=run.regional,
                node=settings.node_a,
                marker=run.marker,
                observed_after=run.started_at,
                job_id=settings.job_id,
                attempt_id=settings.attempt_id,
                case_dir=run.case_dir,
            ),
        )
    if run.workload_submitted:
        guard("workload_delete", run.workload.delete)
        guard(
            "workload_residual",
            lambda: _refuse_residual(
                run.regional.kubectl(
                    "gpu",
                    "get",
                    run.workload.resource,
                    run.workload.name,
                    "--ignore-not-found",
                    "-o",
                    "name",
                    check=False,
                ).strip()
            ),
        )
    guard("restore_isolated_nodes", lambda: _restore_isolated_nodes(run))
    guard("prewarm_cleanup", lambda: _refuse_residual_map(run.prewarm.cleanup()))
    for node, probe in run.probes.items():
        guard(
            f"probe_cleanup_{node}",
            lambda probe=probe: _refuse_residual_map(probe.cleanup()),
        )
    guard(
        "runtime_identity",
        lambda: run.regional.verify_runtime_identity(
            run.preflight["runtime_identity"],
            evidence_path=run.case_dir / "runtime-identity-after-cleanup.json",
            stage="after DESTR-015 cleanup",
        ),
    )
    return result


def _refuse_residual(name: str) -> dict[str, Any]:
    if name:
        raise RegionalFixtureError(f"test workload still exists: {name}")
    return {"residual": False}


def _refuse_residual_map(residuals: dict[str, bool]) -> dict[str, bool]:
    if any(residuals.values()):
        raise RegionalFixtureError(f"residual resources remain: {residuals}")
    return residuals


def _restore_isolated_nodes(run: _LiveRun) -> dict[str, Any]:
    """A branch that failed leaves its node cordoned and owned. Never delete
    the taint by hand: restore through the validation-first workflow, exactly
    as DESTR-003's cleanup does."""

    settings = run.settings
    report: dict[str, Any] = {}
    profile_version = str(
        (run.preflight["store"].get("profile") or {}).get("profile_version") or ""
    )
    warm = WarmSpareLiveFixture(run.regional, "")
    for node in settings.nodes:
        snapshot = run.regional.node_snapshot(node)
        isolated = bool(
            snapshot.get("unschedulable")
            or snapshot.get("ownership_annotations")
            or any(
                str(taint.get("key") or "").startswith("gpu-fault.io/")
                for taint in snapshot.get("taints") or []
            )
        )
        if not isolated:
            report[node] = {"isolated": False}
            continue
        if not run.incident_id:
            raise RegionalFixtureError(f"{node} is isolated but no incident is known")
        warm.wait_incident_idle(run.incident_id)
        created = warm.create_restore_workflow(
            incident_id=run.incident_id,
            node=node,
            profile_version=profile_version,
            reason="DESTR-015 validated cleanup",
        )
        restored = warm.wait_workflow_id(str(created["workflow_request_id"]))
        report[node] = {"isolated": True, "restore": restored.get("status")}
        if restored.get("status") != "SUCCEEDED":
            raise RegionalFixtureError(f"{node} restore workflow did not succeed")
    return report


CASE = CaseRunner(
    case_id=CASE_ID,
    confirmation=CONFIRMATION,
    parser=parser,
    configure=configure,
    read_only_preflight=read_only_preflight,
    plan_details=plan_details,
    execute_case=execute_case,
)


def main() -> int:
    return run_standard_case(CASE)


if __name__ == "__main__":
    raise SystemExit(run_case_main(main))
