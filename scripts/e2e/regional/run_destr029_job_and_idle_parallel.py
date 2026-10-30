#!/usr/bin/env python3
"""GF-REGIONAL-DESTR-029 live acceptance runner.

A two-node distributed training job and one idle GPU node of ONE node group
recover in parallel from faults of two different domain TYPES (b220 + b223): a
reboot-class fault on one node of the job, and a reset-class fault on the idle
node, written at the same moment.

The job is its own merged workflow (STOP_WORKLOADS / RESTART_WORKLOAD take its
job tokens, the faulted node reboots, the healthy peer is only stopped and
restarted); the idle node is its own merged workflow (its single RESET_GPU).
They are kept apart only by their ``node:<cluster>:<node>`` scopes -- never by
a node-group ``domain:`` slot (retired in b220) -- so a job domain and an
idle-node domain run at once. The storm breaker counts the job's one rebooted
node and the idle node -- two nodes across two workload domains, below its
floor of 3 -- so it stays CLOSED. Before b223 the job's healthy peer counted
too (three nodes), which tripped the floor and serialized the two domains.

The job is the GPU-active two-node PyTorchJob the managed fixtures use
(``manifests/training/two-node-replace-pytorchjob.yaml`` + b141's GPU-load
snippet). The reboot is the node-reboot case's XID 79; the idle reset is
DESTR-001's XID 46, with its host probe and reset sampler. The job's faulted
node is judged with DESTR-002's reboot proof, the idle node with DESTR-001's
reset contract and host checks, and the parallel admission by the shared
read-only oracle (``workload_domain_oracle``): a job-domain workflow and an
idle-node workflow, different domain types, neither held on a node-group
``domain:`` or a retired scope, their remediation steps overlapping in
wall-clock time, the storm breaker CLOSED with exactly two counted nodes, both
incidents RECOVERED and nothing left open.

Precondition, read live: at least ``MIN_GROUP_NODES`` GPU nodes in one node
group; short of it the case is ``NOT_RUN(CAPABILITY_ABSENT)``. The group and
its three target nodes are chosen from the live cluster (the largest GPU node
group, or ``--node-group``; the first three eligible idle nodes in name order),
never from a site-profile node list.

Defaults to ``--plan``; ``--execute`` needs ``--confirm DESTR029_EXECUTE``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e2e.regional import run_destr001_gpu_reset as reset_case  # noqa: E402
from scripts.e2e.regional import (  # noqa: E402
    run_destr002_hyperpod_reboot as reboot_case,
)
from scripts.e2e.regional import (  # noqa: E402
    run_destr009_workload_restart as workload_case,
)
from scripts.e2e.regional import (  # noqa: E402
    run_destr015_parallel_branch_join as join_case,
)
from scripts.e2e.regional import (  # noqa: E402
    run_destr027_two_idle_node_parallel_reset as idle_case,
)
from scripts.e2e.regional import workload_domain_oracle as oracle  # noqa: E402
from scripts.e2e.regional.acceptance_runner_common import (  # noqa: E402
    processor_queue_backlog,
    remote_command_queue_errors,
    write_json_atomic,
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
from scripts.e2e.regional.reset_escalation_evidence import (  # noqa: E402
    product_escalation,
    product_reboot,
    reboot_proof_error,
)

PROBE_SCRIPT = Path(__file__).with_name("probes") / "destructive_node_probe.py"
TRAINING_MANIFEST = (
    Path(__file__).with_name("manifests")
    / "training"
    / "two-node-replace-pytorchjob.yaml"
)
CASE_ID = "GF-REGIONAL-DESTR-029"
PREDECESSOR_CASE_ID = "GF-REGIONAL-DESTR-002"
CONFIRMATION = "DESTR029_EXECUTE"
#: One two-node job plus one idle node need a node group of at least three.
MIN_GROUP_NODES = 3
GPUS_PER_NODE = 8
EXPECTED_GPU_COUNT = GPUS_PER_NODE * 2
#: The breaker counts the job's one rebooted node and the idle node.
CASE_COUNTED_NODES = 2
WORKFLOW_BUDGET_SECONDS = 1800
RECENT_EVENT_SECONDS = 600
FOCUSED_TESTS = (
    "tests/execution/test_remediation_budget.py::"
    "test_two_idle_nodes_are_repaired_together",
    "tests/execution/test_remediation_storm_admission.py::"
    "test_a_two_node_job_counts_only_its_faulted_node",
    "tests/regional/test_workload_domain_oracle.py",
    "tests/regional/test_destr029_job_and_idle_parallel.py",
)


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Settings:
    regional: RegionalLiveSettings
    host_probe_image: str
    site_file: Path
    manifest: Path
    predecessor_path: Path
    node_group: str

    def environment(self) -> dict[str, str]:
        return {
            **self.regional.environment(),
            "GPU_FAULT_HOST_PROBE_IMAGE": self.host_probe_image,
            "GPU_FAULT_SITE_FILE": str(self.site_file),
            "GPU_FAULT_TRAINING_MANIFEST": str(self.manifest),
            "GPU_FAULT_PREDECESSOR_EVIDENCE": str(self.predecessor_path),
            "GPU_FAULT_NODE_GROUP": self.node_group,
        }


def configure(arguments: argparse.Namespace) -> Settings:
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
    manifest = (
        Path(arguments.manifest).expanduser().resolve()
        if arguments.manifest
        else TRAINING_MANIFEST
    )
    return Settings(
        regional=settings_from_arguments(arguments),
        host_probe_image=required(
            arguments.host_probe_image or os.getenv("GPU_FAULT_HOST_PROBE_IMAGE", ""),
            "host probe image",
        ),
        site_file=Path(
            required(
                arguments.site_file or os.getenv("GPU_FAULT_SITE_FILE", ""),
                "regional site file",
            )
        )
        .expanduser()
        .resolve(),
        manifest=manifest,
        predecessor_path=predecessor,
        node_group=str(arguments.node_group or "").strip(),
    )


def run_identity(run_dir: Path, attempt: int) -> str:
    token = run_dir.name.rsplit("-", 1)[-1].lower()
    digest = hashlib.sha256(str(run_dir.resolve()).encode()).hexdigest()[:8]
    return f"destr029-{token}-{digest}-a{attempt}"


# --------------------------------------------------------------------------- #
# Target selection and layout (pure, unit-tested)
# --------------------------------------------------------------------------- #
def choose_targets(
    candidates: dict[str, list[str]], *, count: int, dns_nodes: list[str]
) -> tuple[str, ...] | None:
    """The first ``count`` eligible nodes in name order that do not hold every
    kube-dns endpoint between them; ``None`` when no such set exists."""

    eligible = sorted(node for node, reasons in candidates.items() if not reasons)
    if len(eligible) < count:
        return None
    chosen = tuple(eligible[:count])
    if dns_nodes and set(dns_nodes) <= set(chosen):
        for extra in eligible[count:]:
            trimmed = (*chosen[:-1], extra)
            if not set(dns_nodes) <= set(trimmed):
                return trimmed
        return None
    return chosen


@dataclass(frozen=True)
class Layout:
    """The case's three nodes: a job (master + worker) and one idle node."""

    job_id: str
    master: str
    worker: str
    idle: str

    @property
    def job_nodes(self) -> tuple[str, str]:
        return (self.master, self.worker)

    @property
    def workload_id(self) -> str:
        return f"gpu-fault-system/pytorchjob/{self.job_id}"


def layout(targets: tuple[str, ...], run_id: str) -> Layout:
    """Assign the chosen nodes: job master=t0, job worker=t1, idle node=t2.

    The master takes the reboot-class XID; the idle node takes the reset."""

    if len(targets) != MIN_GROUP_NODES:
        raise ValueError(f"need {MIN_GROUP_NODES} targets, got {len(targets)}")
    return Layout(
        job_id=f"{run_id}-job",
        master=targets[0],
        worker=targets[1],
        idle=targets[2],
    )


def profile_errors(profile: dict[str, Any] | None) -> list[str]:
    """The job recovery needs workload stop/restart and node reboot owned; the
    idle reset needs gpuReset owned by the Node Agent (DESTR-001's rule)."""

    errors = list(workload_case.managed_owner_errors(profile))
    if reboot_case.capability(profile, "nodeReboot") is None:
        errors.append("runtime profile has no nodeReboot capability")
    errors.extend(idle_case.profile_errors(profile))
    return errors


# --------------------------------------------------------------------------- #
# Preflight / plan
# --------------------------------------------------------------------------- #
def focused_tests(case_dir: Path, *, reuse: bool = False) -> dict[str, Any]:
    if reuse:
        recorded = reusable_focused_tests(case_dir / "plan.json")
        if recorded is not None:
            return {**recorded, "focused_tests_reused": True}
    command = [sys.executable, "-m", "pytest", "-q", *FOCUSED_TESTS]
    completed = RegionalLiveFixture.run(command, cwd=ROOT, check=False, timeout=600)
    path = case_dir / "focused-tests.log"
    path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    path.chmod(0o600)
    return {
        "passed": completed.returncode == 0,
        "returncode": completed.returncode,
        "command": command,
    }


def read_only_preflight(
    settings: Settings,
    case_dir: Path,
    *,
    reuse_focused_tests: bool = False,
) -> dict[str, Any]:
    regional = RegionalLiveFixture(settings.regional)
    identity = regional.evidence_identity()
    cluster_id = regional.settings.cluster_id
    gpu_nodes = regional.gpu_nodes()
    gate = oracle.node_group_gate(
        gpu_nodes,
        minimum=MIN_GROUP_NODES,
        cluster_id=cluster_id,
        requested=settings.node_group,
    )
    result: dict[str, Any] = {
        "release_id": identity["release_id"],
        "evidence_identity": identity,
        "gpu_nodes": gpu_nodes,
        "node_group_gate": gate.as_dict(),
    }
    if gate.not_run is not None or gate.group is None:
        result.update(not_run=gate.not_run, errors=[])
        write_json_atomic(case_dir / "preflight.json", result)
        return result
    group = gate.group
    since = datetime.now(timezone.utc) - timedelta(seconds=RECENT_EVENT_SECONDS)
    snapshots = {node: regional.node_snapshot(node) for node in group.nodes}
    stores = {
        node: regional.store_snapshot(node=node, observed_after=since)
        for node in group.nodes
    }
    arbiter_pods, dns_nodes = join_case.cluster_arbiter_placement(regional)
    gpu_workloads = regional.gpu_workloads()
    views = {node: regional.node_workload_view(node) for node in group.nodes}
    candidates = {
        node: idle_case.node_ineligibility(
            node,
            snapshot=snapshots[node],
            store=stores[node],
            business_workloads=regional.business_workloads(node),
            arbiter_pods=arbiter_pods.get(node) or [],
            gpu_workloads=gpu_workloads,
            workload_view=views[node],
        )
        for node in group.nodes
    }
    targets = choose_targets(candidates, count=MIN_GROUP_NODES, dns_nodes=dns_nodes)
    errors: list[str] = []
    if targets is None:
        errors.append(
            f"node group {group.name} has no {MIN_GROUP_NODES} eligible idle "
            "nodes now: "
            + "; ".join(reason for reasons in candidates.values() for reason in reasons)
        )
    plan = layout(targets, "destr029-plan") if targets else None
    probe_node = targets[0] if targets else group.nodes[0]
    first = stores[probe_node]
    runtime_identity = regional.runtime_identity()
    errors.extend(runtime_identity_errors(runtime_identity))
    for node in targets or ():
        errors.extend(
            f"{node}: {error}" for error in profile_errors(stores[node].get("profile"))
        )
    if processor_queue_backlog(first.get("queue") or {}):
        errors.append("processor queue is not empty")
    errors.extend(remote_command_queue_errors(first.get("remote_commands"), cluster_id))
    storm = oracle.read_oracle_snapshot(regional, workflow_ids=(), node_ids=group.nodes)
    if plan is not None:
        counted = [plan.master, plan.idle]
        errors.extend(
            oracle.breaker_premise_errors(
                storm,
                cluster_id=cluster_id,
                targets=counted,
                now=datetime.now(timezone.utc),
            )
        )
        errors.extend(idle_case.residue_errors(storm, tuple(targets)))
    tests = focused_tests(case_dir, reuse=reuse_focused_tests)
    if not tests.get("passed"):
        errors.append("focused regression tests failed")
    predecessor = predecessor_evidence(
        settings.predecessor_path, PREDECESSOR_CASE_ID, **identity
    )
    if not predecessor.get("valid"):
        errors.append("DESTR-002 predecessor evidence is not PASS")
    result.update(
        {
            "node_group": group.as_dict(),
            "targets": list(targets) if targets else [],
            "layout": (
                {
                    "job_id": plan.job_id,
                    "master": plan.master,
                    "worker": plan.worker,
                    "idle": plan.idle,
                }
                if plan is not None
                else None
            ),
            "candidates": candidates,
            "workload_views": views,
            "nodes": snapshots,
            "stores": stores,
            "arbiter_pods": arbiter_pods,
            "dns_nodes": dns_nodes,
            "storm": storm,
            "runtime_identity": runtime_identity,
            "focused_tests": tests,
            "cpu_blast": regional.cpu_blast_snapshot(),
            "predecessor": predecessor,
            "errors": errors,
        }
    )
    write_json_atomic(case_dir / "preflight.json", result)
    return result


def plan_identity(preflight: dict[str, Any]) -> dict[str, Any]:
    targets = list(preflight.get("targets") or [])
    snapshots = preflight.get("nodes") or {}
    stores = preflight.get("stores") or {}
    return {
        "release_id": preflight.get("release_id"),
        "node_group": {
            key: (preflight.get("node_group") or {}).get(key)
            for key in ("label", "name")
        },
        "targets": {
            node: {
                "uid": (snapshots.get(node) or {}).get("uid"),
                "boot_id": (snapshots.get(node) or {}).get("boot_id"),
                "agent_generation": ((stores.get(node) or {}).get("agent") or {}).get(
                    "generation"
                ),
            }
            for node in targets
        },
        "runtime_identity": preflight.get("runtime_identity"),
    }


def plan_details(settings: Settings, preflight: dict[str, Any]) -> dict[str, Any]:
    if preflight.get("not_run"):
        return {
            "risk": "destructive",
            "not_run": preflight["not_run"],
            "node_group_gate": preflight["node_group_gate"],
            "mutation": "none: the cluster cannot carry the case "
            "(NOT_RUN(CAPABILITY_ABSENT)); --execute records the verdict only",
        }
    identity = plan_identity(preflight)
    storm = preflight.get("storm") or {}
    details: dict[str, Any] = {
        "risk": "destructive",
        "predecessor": preflight.get("predecessor"),
        "node_group": preflight.get("node_group"),
        "node_group_gate": preflight.get("node_group_gate"),
        "targets": preflight.get("targets"),
        "layout": preflight.get("layout"),
        "manifest": str(settings.manifest),
        "mutation": (
            "submit one two-node 16-GPU PyTorchJob on two idle GPU nodes and "
            "leave a third idle; write one synthetic XID 79 to one job node and "
            "one XID 46 to the idle node at the same moment (the reboot case's "
            "and DESTR-001's host probes), so the job stops and reboots its "
            "faulted node while the idle node resets one GPU, both at once. No "
            "replacement, no provider call beyond the one reboot."
        ),
        "storm_premise": {
            "threshold": oracle.storm_threshold(storm.get("cluster_gpu_agents")),
            "cluster_gpu_agents": storm.get("cluster_gpu_agents"),
            "breaker_state": (storm.get("breaker") or {}).get("state", "CLOSED"),
            "case_counted_nodes": CASE_COUNTED_NODES,
        },
        "preflight_identity": identity,
        "preflight_identity_digest": join_case.identity_digest(identity),
        "stop_conditions": [
            f"fewer than {MIN_GROUP_NODES} GPU nodes in one node group "
            "(NOT_RUN(CAPABILITY_ABSENT))",
            "DESTR-002 has not passed on this release",
            f"no {MIN_GROUP_NODES} eligible idle nodes: not Ready, cordoned, "
            "tainted, owned, a declared spare, running Pods, hosting arbiters, "
            "a recent XID, an inactive Agent or a product-less GPU inventory",
            "the targets hold every kube-dns endpoint",
            "workload stop/restart, node reboot or gpuReset is not owned",
            "the storm breaker is open, holds workflows, or still counts nodes "
            "that would make the case's second admission the trip",
            "an open workflow or incident on a target",
            "plan identity drift (release, runtime identity, node UID/boot id, "
            "Agent generation)",
            "the two faults do not land in two separate workflows of different "
            "domain types",
            "a workflow claims a node-group domain or a retired scope, is held, "
            "or the remediation steps do not overlap",
            "the breaker counts other than the rebooted node and the idle node, "
            "or trips",
            "provider mutation beyond the one reboot; cleanup leaves a residual",
        ],
        "rollback": {
            "node_reboot_and_gpu_reset_are_owned_by_the_product_workflows": True,
            "failed_validation_keeps_a_node_quarantined": True,
            "runner_deletes_the_test_job_after_quiescence": True,
            "detached_samplers_are_stopped_after_the_post_fault_snapshots": True,
            "runner_finally_deletes_probe_and_prewarm_resources": True,
            "no_storm_breaker_or_control_plane_setting_is_changed": True,
        },
    }
    record_focused_tests(details, dict(preflight.get("focused_tests") or {}))
    return details


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Run the guarded DESTR-029 acceptance: a two-node training job and "
            "an idle GPU node of one node group recover in parallel from a "
            "reboot-class and a reset-class fault (b220 + b223)."
        ),
        allow_abbrev=False,
    )
    add_live_arguments(value, confirmation=CONFIRMATION)
    value.add_argument("--cpu-kubeconfig", default="")
    value.add_argument("--gpu-kubeconfig", default="")
    value.add_argument("--gpu-context", default="")
    value.add_argument("--namespace", default="gpu-fault-system")
    value.add_argument("--cluster-id", default="")
    value.add_argument("--region", default="")
    value.add_argument("--host-probe-image", default="")
    value.add_argument("--site-file", default="")
    value.add_argument("--manifest", default="")
    value.add_argument("--predecessor-evidence", default="")
    value.add_argument(
        "--node-group",
        default="",
        help=(
            "GPU node group (node-group or instance-group name) to use; default: "
            "the cluster's largest GPU node group, read live"
        ),
    )
    return value


# --------------------------------------------------------------------------- #
# Oracle domains
# --------------------------------------------------------------------------- #
def oracle_domains(
    plan: Layout, *, job_workflow_id: str, idle_workflow_id: str
) -> list[oracle.DomainExpectation]:
    """The job domain (its job tokens and two-node allocation, faulted on the
    master) and the idle-node domain (its node scope only) -- two different
    domain types."""

    return [
        oracle.DomainExpectation(
            label="job",
            faulted_node=plan.master,
            workflow_id=job_workflow_id,
            remediation="RESTART_NODE",
            job_id=plan.job_id,
            workload_ids=(plan.workload_id,),
            allocation=plan.job_nodes,
        ),
        oracle.DomainExpectation(
            label="idle",
            faulted_node=plan.idle,
            workflow_id=idle_workflow_id,
            remediation="RESET_GPU",
        ),
    ]


# --------------------------------------------------------------------------- #
# Live execution (not exercised by the unit suite; the oracle, the reboot proof
# and the reset contract it calls are)
# --------------------------------------------------------------------------- #
@dataclass
class _Target:
    """A faulted node: the job master (reboot) or the idle node (reset)."""

    label: str
    node: str
    kind: str  # "reboot" or "reset"
    probe: HostProbeFixture
    run_id: str
    directory: Path
    marker: str = ""
    baseline: dict[str, Any] = field(default_factory=dict)
    expected_gpu_count: int = 0
    target_bdf: str = ""
    injected_at: str = ""
    state: dict[str, Any] = field(default_factory=dict)
    after: dict[str, Any] = field(default_factory=dict)
    sampler_started: bool = False
    sampler_stopped: bool = False
    injection_attempted: bool = False

    @property
    def workflow(self) -> dict[str, Any]:
        return dict(self.state.get("workflow") or {})

    @property
    def workflow_id(self) -> str:
        return str(self.workflow.get("request_id") or "")

    @property
    def incident_id(self) -> str:
        return str((self.state.get("incident") or {}).get("incident_id") or "")


@dataclass
class _LiveRun:
    settings: Settings
    case_dir: Path
    attempt: int
    maintenance_window_end: datetime
    preflight: dict[str, Any]
    regional: RegionalLiveFixture
    plan: Layout
    prewarm: ImagePrewarmFixture
    workload: ManagedWorkloadFixture
    targets: dict[str, _Target]
    job_submitted: bool = False
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def window_open(self, stage: str) -> None:
        if datetime.now(timezone.utc) >= self.maintenance_window_end:
            raise RegionalFixtureError(f"maintenance window ended before {stage}")


def _target(
    settings: Settings, case_dir: Path, run_id: str, *, label: str, node: str, kind: str
) -> _Target:
    directory = case_dir / label
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    node_run_id = f"{run_id}-{label}"
    return _Target(
        label=label,
        node=node,
        kind=kind,
        run_id=node_run_id,
        directory=directory,
        probe=HostProbeFixture(
            HostProbeSettings(
                state_directory=case_dir / "host-probes",
                kubeconfig=settings.regional.gpu_kubeconfig,
                context=settings.regional.gpu_context,
                namespace=settings.regional.namespace,
                node=node,
                image=settings.host_probe_image,
                case_id=CASE_ID,
                run_id=node_run_id,
                probe_script=PROBE_SCRIPT,
            )
        ),
    )


def _prepare(
    settings: Settings,
    run_dir: Path,
    attempt: int,
    maintenance_window_end: datetime,
    preflight: dict[str, Any],
) -> _LiveRun:
    case_dir = run_dir / "cases" / CASE_ID
    run_id = run_identity(run_dir, attempt)
    regional = RegionalLiveFixture(settings.regional)
    plan = layout(tuple(preflight["targets"]), run_id)
    job_manifest = join_case.render_two_node_manifest(
        settings.manifest,
        case_dir / "job" / "job.yaml",
        master_node=plan.master,
        worker_node=plan.worker,
    )
    workload = ManagedWorkloadFixture(
        regional,
        ManagedWorkloadSettings(
            manifest=job_manifest,
            site_file=settings.site_file,
            job_id=plan.job_id,
            attempt_id=f"{plan.job_id}-a001",
            restart_budget=1,
            expected_pods=2,
            expected_gpu_count=EXPECTED_GPU_COUNT,
        ),
    )
    targets = {
        "job": _target(
            settings, case_dir, run_id, label="job", node=plan.master, kind="reboot"
        ),
        "idle": _target(
            settings, case_dir, run_id, label="idle", node=plan.idle, kind="reset"
        ),
    }
    return _LiveRun(
        settings=settings,
        case_dir=case_dir,
        attempt=attempt,
        maintenance_window_end=maintenance_window_end,
        preflight=preflight,
        regional=regional,
        plan=plan,
        prewarm=ImagePrewarmFixture(regional, case_id=CASE_ID, run_id=run_id),
        workload=workload,
        targets=targets,
    )


def _launch_job(run: _LiveRun) -> None:
    run.window_open("image prewarm")
    write_json_atomic(
        run.case_dir / "prewarm.json", run.prewarm.create(list(run.plan.job_nodes))
    )
    run.window_open(f"submit {run.plan.job_id}")
    run.job_submitted = True
    write_json_atomic(run.case_dir / "job" / "submission.json", run.workload.submit())
    source = run.workload.wait_running(timeout_seconds=WORKFLOW_BUDGET_SECONDS)
    write_json_atomic(run.case_dir / "job" / "workload-source.json", source)


def _baselines_and_samplers(run: _LiveRun) -> None:
    for item in run.targets.values():
        run.window_open(f"the host baseline of {item.node}")
        baseline, count, bdf = reset_case.prepare_host_baseline(
            item.probe, {"node": run.preflight["nodes"][item.node]}, item.directory
        )
        item.baseline, item.expected_gpu_count, item.target_bdf = baseline, count, bdf
    run.regional.verify_runtime_identity(
        run.preflight["runtime_identity"],
        evidence_path=run.case_dir / "runtime-identity-before-injection.json",
        stage="before DESTR-029 injection",
    )
    for item in run.targets.values():
        run.window_open(f"the sampler start on {item.node}")
        item.sampler_started = True
        sampler = item.probe.execute(
            "start-reset-sampler",
            "--run-id",
            item.run_id,
            "--probe-script",
            item.probe.host_script,
            timeout=60,
        )
        write_json_atomic(item.directory / "sampler-start.json", sampler)


def _inject_both(run: _LiveRun) -> None:
    """XID 79 on the job's faulted node and XID 46 on the idle node, together."""

    run.window_open("injection")
    base = f"destr029-{int(time.time())}-a{run.attempt}"
    for item in run.targets.values():
        item.marker = f"{base}-{item.label}"
    run.started_at = datetime.now(timezone.utc)

    def write(item: _Target) -> dict[str, Any]:
        item.injection_attempted = True
        command = "write-xid79" if item.kind == "reboot" else "write-xid46"
        payload = item.probe.execute(
            command,
            "--marker",
            item.marker,
            "--drill-id",
            item.run_id,
            "--pci-bdf",
            item.target_bdf,
            "--maintenance-window-end",
            run.maintenance_window_end.isoformat(),
        )
        item.injected_at = datetime.now(timezone.utc).isoformat()
        return payload

    with ThreadPoolExecutor(
        max_workers=len(run.targets), thread_name_prefix="d029"
    ) as pool:
        futures = {
            label: pool.submit(write, item) for label, item in run.targets.items()
        }
        writes = {label: future.result() for label, future in futures.items()}
    write_json_atomic(
        run.case_dir / "injection.json",
        {
            "started_at": run.started_at.isoformat(),
            "markers": {label: item.marker for label, item in run.targets.items()},
            "injected_at": {
                label: item.injected_at for label, item in run.targets.items()
            },
            "writes": writes,
        },
    )


def _wait_terminal(run: _LiveRun) -> None:
    deadline = time.monotonic() + WORKFLOW_BUDGET_SECONDS
    for item in run.targets.values():
        item.state = run.regional.wait_for_workflow(
            node=item.node,
            marker=item.marker,
            observed_after=run.started_at,
            case_dir=item.directory,
            timeout_seconds=max(60, int(deadline - time.monotonic())),
        )
        write_json_atomic(item.directory / "workflow-state.json", item.state)


def _judge_records(run: _LiveRun) -> oracle.OracleVerdict:
    domains = oracle_domains(
        run.plan,
        job_workflow_id=run.targets["job"].workflow_id,
        idle_workflow_id=run.targets["idle"].workflow_id,
    )
    if not all(domain.workflow_id for domain in domains):
        return oracle.unreadable_verdict(
            "a fault never produced a workflow the store could name"
        )
    try:
        snapshot = oracle.read_oracle_snapshot(
            run.regional,
            workflow_ids=[domain.workflow_id for domain in domains],
            node_ids=[item.node for item in run.targets.values()],
        )
    except RegionalFixtureError as exc:
        return oracle.unreadable_verdict(f"the store could not be read: {exc}")
    write_json_atomic(run.case_dir / "oracle-records.json", snapshot)
    group = run.preflight.get("node_group") or {}
    return oracle.judge_parallel_domains(
        snapshot,
        cluster_id=run.regional.settings.cluster_id,
        domains=domains,
        started_at=run.started_at,
        node_group=oracle.NodeGroup(
            label=str(group.get("label") or ""),
            name=str(group.get("name") or ""),
            nodes=tuple(group.get("nodes") or ()),
        ),
        overlap=oracle.OVERLAP_REMEDIATION,
        expected_counted_nodes=CASE_COUNTED_NODES,
    )


def _reboot_contract(run: _LiveRun, item: _Target) -> list[str]:
    errors: list[str] = [
        f"{item.node}: {error}" for error in workload_case.workflow_errors(item.state)
    ]
    escalation = product_escalation(item.state)
    before = run.preflight["nodes"][item.node]
    after = run.regional.node_snapshot(item.node)
    rebooted = product_reboot(escalation, before, after)
    if not rebooted:
        errors.append(f"{item.node}: the job's faulted node did not reboot")
    else:
        item.sampler_stopped = True
        errors.append(f"{item.node}: {reboot_proof_error(escalation, rebooted)}")
    return errors


def _reset_contract(run: _LiveRun, item: _Target) -> list[str]:
    errors = [
        f"{item.node}: {error}" for error in reset_case.workflow_errors(item.state)
    ]
    restored, _node_after = reset_case.node_restored_errors(
        run.regional, item.node, item.directory
    )
    errors.extend(f"{item.node}: {error}" for error in restored)
    errors.extend(
        f"{item.node}: {error}"
        for error in reset_case.host_errors(
            item.baseline,
            item.after,
            expected_gpu_count=item.expected_gpu_count,
            target_bdf=item.target_bdf,
            incident_id=item.incident_id,
            workflow_request_id=item.workflow_id,
        )
    )
    return errors


def not_run_result(
    attempt: int, identity: dict[str, Any], record: dict[str, Any]
) -> dict[str, Any]:
    return {
        "case_id": CASE_ID,
        "attempt": attempt,
        **identity,
        "verdict": oracle.NOT_RUN,
        "reason": record.get("reason", oracle.CAPABILITY_ABSENT),
        "detail": record.get("detail"),
        "node_group_gate": record.get("node_group_gate"),
        "mutation_performed": False,
        "ended_at": datetime.now(timezone.utc).isoformat(),
    }


def _record_not_run(
    case_dir: Path, attempt: int, identity: dict[str, Any], record: dict[str, Any]
) -> int:
    result = not_run_result(attempt, identity, record)
    write_json_atomic(case_dir / f"{CASE_ID}.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0


def verify_plan_identity(case_dir: Path, preflight: dict[str, Any]) -> None:
    plan = json.loads((case_dir / "plan.json").read_text(encoding="utf-8"))
    planned = plan["details"]["preflight_identity_digest"]
    current = join_case.identity_digest(plan_identity(preflight))
    if planned != current or plan["details"].get("targets") != preflight["targets"]:
        raise RegionalFixtureError("DESTR-029 plan identity drifted before execution")


def execute_case(
    settings: Settings,
    run_dir: Path,
    attempt: int,
    maintenance_window_end: datetime,
) -> int:
    case_dir = run_dir / "cases" / CASE_ID
    case_dir.mkdir(parents=True, exist_ok=True)
    planned = json.loads((case_dir / "plan.json").read_text(encoding="utf-8"))
    planned_not_run = (planned.get("details") or {}).get("not_run")
    if planned_not_run:
        identity = RegionalLiveFixture(settings.regional).evidence_identity()
        gate = (planned.get("details") or {}).get("node_group_gate")
        return _record_not_run(
            case_dir, attempt, identity, {**planned_not_run, "node_group_gate": gate}
        )
    preflight = read_only_preflight(settings, case_dir, reuse_focused_tests=True)
    if preflight.get("not_run"):
        return _record_not_run(
            case_dir,
            attempt,
            preflight["evidence_identity"],
            {**preflight["not_run"], "node_group_gate": preflight["node_group_gate"]},
        )
    if preflight["errors"]:
        raise RegionalFixtureError(
            "preflight failed: " + "; ".join(preflight["errors"])
        )
    verify_plan_identity(case_dir, preflight)
    run = _prepare(settings, run_dir, attempt, maintenance_window_end, preflight)
    result: dict[str, Any] = {
        "case_id": CASE_ID,
        "attempt": attempt,
        "verdict": oracle.FAIL,
        **preflight["evidence_identity"],
        "node_group": preflight["node_group"],
        "targets": preflight["targets"],
        "layout": preflight["layout"],
        "maintenance_window_end": maintenance_window_end.isoformat(),
    }
    try:
        _launch_job(run)
        _baselines_and_samplers(run)
        _inject_both(run)
        _wait_terminal(run)
        verdict = _judge_records(run)
        errors = list(verdict.errors)
        errors.extend(_reboot_contract(run, run.targets["job"]))
        errors.extend(_reset_contract(run, run.targets["idle"]))
        cpu_after = run.regional.cpu_blast_snapshot()
        write_json_atomic(case_dir / "cpu-blast-after.json", cpu_after)
        if cpu_after != preflight["cpu_blast"]:
            errors.append("control-plane EKS state differs from baseline")
        result.update(
            {
                "verdict": oracle.FAIL
                if errors
                else (oracle.INCONCLUSIVE if verdict.inconclusive else oracle.PASS),
                "errors": errors,
                "oracle": verdict.as_dict(),
                "started_at": run.started_at.isoformat(),
                "window_clear_at": (verdict.facts.get("breaker") or {}).get(
                    "window_clear_at"
                ),
                "markers": {item.node: item.marker for item in run.targets.values()},
                "workflow_request_ids": {
                    item.label: item.workflow_id for item in run.targets.values()
                },
                "incident_ids": {
                    item.label: item.incident_id for item in run.targets.values()
                },
            }
        )
    except Exception as exc:  # noqa: BLE001 - recorded as the case error
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        _cleanup(run, result)
    write_json_atomic(case_dir / f"{CASE_ID}.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["verdict"] == oracle.PASS else 1


# --------------------------------------------------------------------------- #
# Cleanup
# --------------------------------------------------------------------------- #
def _cleanup(run: _LiveRun, result: dict[str, Any]) -> None:
    """Stop the samplers, delete the test job and the probe/prewarm resources.
    The runner never restores a node itself: a workflow that failed keeps its
    node quarantined for the operator, and the case FAILs."""

    targets: dict[str, Any] = {}
    for item in run.targets.values():
        report: dict[str, Any] = {}
        if item.sampler_started and not item.sampler_stopped:
            try:
                stopped = item.probe.execute(
                    "stop-reset-sampler", "--run-id", item.run_id, timeout=60
                )
                item.sampler_stopped = True
                write_json_atomic(item.directory / "sampler-final.json", stopped)
            except Exception as exc:  # noqa: BLE001 - a cleanup failure is a FAIL
                report["sampler_cleanup_error"] = f"{type(exc).__name__}: {exc}"
                result["verdict"] = oracle.FAIL
        try:
            residuals = item.probe.cleanup()
        except Exception as exc:  # noqa: BLE001 - a cleanup failure is a FAIL
            residuals = {"cleanup_error": True}
            report["probe_cleanup_error"] = f"{type(exc).__name__}: {exc}"
        report["probe_residuals"] = residuals
        if any(residuals.values()):
            result["verdict"] = oracle.FAIL
        targets[item.label] = report
    result["target_reports"] = targets
    if run.job_submitted:
        try:
            result["job_delete"] = run.workload.delete()
        except Exception as exc:  # noqa: BLE001 - a cleanup failure is a FAIL
            result["job_delete_error"] = f"{type(exc).__name__}: {exc}"
            result["verdict"] = oracle.FAIL
    try:
        prewarm_residual = run.prewarm.cleanup()
    except Exception as exc:  # noqa: BLE001 - a cleanup failure is a FAIL
        prewarm_residual = {"cleanup_error": True}
        result["prewarm_cleanup_error"] = f"{type(exc).__name__}: {exc}"
    result["prewarm_residuals"] = prewarm_residual
    if any(prewarm_residual.values()):
        result["verdict"] = oracle.FAIL
    try:
        run.regional.verify_runtime_identity(
            run.preflight["runtime_identity"],
            evidence_path=run.case_dir / "runtime-identity-after-cleanup.json",
            stage="after DESTR-029 cleanup",
        )
    except Exception as exc:  # noqa: BLE001 - a cleanup failure is a FAIL
        result["runtime_identity_error"] = f"{type(exc).__name__}: {exc}"
        result["verdict"] = oracle.FAIL


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
