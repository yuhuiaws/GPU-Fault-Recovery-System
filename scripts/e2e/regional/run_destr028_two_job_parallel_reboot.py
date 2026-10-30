#!/usr/bin/env python3
"""GF-REGIONAL-DESTR-028 live acceptance runner.

Two distributed training jobs on GPU nodes of ONE node group recover in
parallel when a reboot-class fault hits ONE node of each (b220 + b223). Job a
is always a two-node job; job b is a second two-node job when the group offers
four eligible idle nodes ("2+2"), else a single-node job on the third ("2+1":
a four-node group whose fourth node hosts the arbiters, HyperPod today).

Each job is its own merged workflow: a fault on one of its nodes merges into
it, STOP_WORKLOADS / RESTART_WORKLOAD take the job's tokens, and the faulted
node's RESTART_NODE reboots it while a healthy peer is only stopped and
restarted with the job. Two such jobs are kept apart only by their
``node:<cluster>:<node>`` scopes -- never by a node-group ``domain:`` slot
(retired in b220) -- so they run at once. The storm breaker counts only the two
rebooted nodes (b223: a job's workload-stop peer is not entering destructive
NODE remediation), two nodes across two job domains, below its floor of 3, so
it stays CLOSED. Before b223 the healthy peers counted too -- four nodes in
2+2, three in 2+1, either way at the floor -- which tripped it and held the
second job; on a cluster whose threshold is higher the oracle still fails any
count other than the two rebooted nodes.

The jobs are the GPU-active PyTorchJobs the managed fixtures use
(``manifests/training/two-node-replace-pytorchjob.yaml`` and, for 2+1's job b,
``single-node-warm-spare-pytorchjob.yaml``, both with b141's GPU-load snippet),
pinned to their nodes, renamed after their job ids (the product's job token
names the submitted object) and submitted by ``ManagedWorkloadFixture``. The
fault is the node-reboot case's, unchanged: one synthetic XID 79 written to the
faulted node's ``/dev/kmsg`` by the destructive host probe (``write-xid79``),
which the policy resolves to a node reboot. Each faulted node then runs its
job's STOP_WORKLOADS -> RESTART_NODE -> validations -> RESTORE_SCHEDULING ->
RESTART_WORKLOAD recovery, judged with DESTR-002's reboot proof minus its
provider half (``job_reboot_contract``), and the parallel admission is judged
by the shared read-only oracle (``workload_domain_oracle``) on the durable
store records: two job workflows, each claiming its node scopes and its job
tokens, neither ever held on a node-group ``domain:`` or a retired scope, their
reboot steps overlapping in wall-clock time, the storm breaker CLOSED with
exactly the two rebooted nodes counted, both incidents RECOVERED and nothing
left open.

Precondition, read live: at least ``MIN_GROUP_NODES`` (3) GPU nodes in one
node group; short of it the case is ``NOT_RUN(CAPABILITY_ABSENT)`` -- the plan
says so and ``--execute`` records it without touching anything. The group and
its target nodes are chosen from the live cluster (the largest GPU node group,
or ``--node-group``; the first four, else three, eligible idle nodes in name
order), never from a site-profile node list. The plan and the result record
the layout.

Two counted nodes are below the storm breaker's floor (3), so the case cannot
trip it alone; the preflight refuses while the breaker still counts other nodes
from the last 600 s that would make the case's second admission the trip, and
the result records ``window_clear_at``.

Defaults to ``--plan``; ``--execute`` needs ``--confirm DESTR028_EXECUTE``.
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
from scripts.e2e.regional import job_reboot_contract  # noqa: E402
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
    managed_workload_name,
    render_node_pinned_manifest,
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
)

PROBE_SCRIPT = Path(__file__).with_name("probes") / "destructive_node_probe.py"
TRAINING_MANIFEST = (
    Path(__file__).with_name("manifests")
    / "training"
    / "two-node-replace-pytorchjob.yaml"
)
#: The 2+1 layout's job b: the GPU-active single-node 8-GPU PyTorchJob.
SINGLE_NODE_TRAINING_MANIFEST = (
    Path(__file__).with_name("manifests")
    / "training"
    / "single-node-warm-spare-pytorchjob.yaml"
)
#: The fixed names the two manifests ship with, by a job's node count; each job
#: is submitted renamed after its job id (``name_managed_workload``).
WORKLOAD_BASES = {
    2: "gpu-fault-two-node-replace",
    1: "gpu-fault-single-node-warm-spare",
}
CASE_ID = "GF-REGIONAL-DESTR-028"
PREDECESSOR_CASE_ID = "GF-REGIONAL-DESTR-002"
CONFIRMATION = "DESTR028_EXECUTE"
#: Two jobs, labelled a and b, each faulted on its master (its first node).
JOB_LABELS = ("a", "b")
#: Nodes per job by the number of eligible targets, most first: two two-node
#: jobs on four nodes ("2+2"); a two-node and a single-node job on three
#: ("2+1"). The targets order as a-master, a-worker, b-master[, b-worker].
LAYOUTS: dict[int, tuple[int, ...]] = {4: (2, 2), 3: (2, 1)}
#: The smallest layout's node count: a node group with fewer GPU nodes cannot
#: carry the case.
MIN_GROUP_NODES = min(LAYOUTS)
GPUS_PER_NODE = 8
#: The breaker counts one node per job, in either layout: the two rebooted
#: masters (before b223: 4 in 2+2, 3 in 2+1).
CASE_COUNTED_NODES = len(JOB_LABELS)
#: Two parallel jobs each run a node reboot; DESTR-002 allows one node ~1500 s.
WORKFLOW_BUDGET_SECONDS = 1800
#: A recent XID on a target refuses the plan.
RECENT_EVENT_SECONDS = 600
FOCUSED_TESTS = (
    "tests/execution/test_remediation_budget.py::"
    "test_two_idle_nodes_are_repaired_together",
    "tests/execution/test_remediation_storm_admission.py::"
    "test_two_two_node_jobs_stay_below_the_floor",
    "tests/execution/test_remediation_storm_admission.py::"
    "test_a_two_node_job_counts_only_its_faulted_node",
    "tests/regional/test_workload_domain_oracle.py",
    "tests/regional/test_job_reboot_contract.py",
    "tests/regional/test_destr028_two_job_parallel_reboot.py",
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
    #: The node group to use; empty takes the cluster's largest GPU group.
    node_group: str
    #: The 2+1 layout's single-node job b.
    single_node_manifest: Path = SINGLE_NODE_TRAINING_MANIFEST

    def environment(self) -> dict[str, str]:
        return {
            **self.regional.environment(),
            "GPU_FAULT_HOST_PROBE_IMAGE": self.host_probe_image,
            "GPU_FAULT_SITE_FILE": str(self.site_file),
            "GPU_FAULT_TRAINING_MANIFEST": str(self.manifest),
            "GPU_FAULT_SINGLE_NODE_TRAINING_MANIFEST": str(self.single_node_manifest),
            "GPU_FAULT_PREDECESSOR_EVIDENCE": str(self.predecessor_path),
            "GPU_FAULT_NODE_GROUP": self.node_group,
        }

    def workload_bases(self) -> dict[int, str]:
        """The fixed manifest name each job size is renamed from."""

        return {
            2: job_reboot_contract.manifest_workload_name(self.manifest),
            1: job_reboot_contract.manifest_workload_name(self.single_node_manifest),
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
    single_node_manifest = (
        Path(arguments.single_node_manifest).expanduser().resolve()
        if arguments.single_node_manifest
        else SINGLE_NODE_TRAINING_MANIFEST
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
        single_node_manifest=single_node_manifest,
    )


def run_identity(run_dir: Path, attempt: int) -> str:
    """The probes', samplers' and jobs' run id. Run directories end in the lane
    token every round, so the resolved directory's digest keeps rounds apart."""

    token = run_dir.name.rsplit("-", 1)[-1].lower()
    digest = hashlib.sha256(str(run_dir.resolve()).encode()).hexdigest()[:8]
    return f"destr028-{token}-{digest}-a{attempt}"


# --------------------------------------------------------------------------- #
# Target selection (pure, unit-tested)
# --------------------------------------------------------------------------- #
def choose_targets(
    candidates: dict[str, list[str]], *, count: int, dns_nodes: list[str]
) -> tuple[str, ...] | None:
    """The first ``count`` eligible nodes in name order that do not hold every
    kube-dns endpoint between them (the jobs' nodes are rebooted or stopped);
    ``None`` when no such set exists."""

    eligible = sorted(node for node, reasons in candidates.items() if not reasons)
    if len(eligible) < count:
        return None
    chosen = tuple(eligible[:count])
    if dns_nodes and set(dns_nodes) <= set(chosen):
        # Swap in the next eligible node so the targets never hold all of DNS.
        for extra in eligible[count:]:
            trimmed = (*chosen[:-1], extra)
            if not set(dns_nodes) <= set(trimmed):
                return trimmed
        return None
    return chosen


def choose_layout_targets(
    candidates: dict[str, list[str]], *, dns_nodes: list[str]
) -> tuple[str, ...] | None:
    """The largest layout's targets the eligible nodes allow: four (2+2), else
    three (2+1); ``None`` when not even three are eligible."""

    for count in sorted(LAYOUTS, reverse=True):
        chosen = choose_targets(candidates, count=count, dns_nodes=dns_nodes)
        if chosen is not None:
            return chosen
    return None


def layout_sizes(targets: tuple[str, ...] | list[str]) -> tuple[int, ...]:
    """The nodes per job of the layout ``targets`` carry."""

    sizes = LAYOUTS.get(len(targets))
    if sizes is None:
        raise ValueError(
            f"need {' or '.join(str(count) for count in sorted(LAYOUTS))} targets, "
            f"got {len(targets)}"
        )
    return sizes


def layout_name(targets: tuple[str, ...] | list[str]) -> str:
    """``2+2`` or ``2+1``: the layout ``targets`` carry."""

    return "+".join(str(size) for size in layout_sizes(targets))


@dataclass(frozen=True)
class JobSpec:
    """One job domain the case creates: its label, nodes and faulted master."""

    label: str
    job_id: str
    #: Master first; a single-node job is its master alone.
    nodes: tuple[str, ...]
    #: The PyTorchJob's Kubernetes name: its manifest's fixed name renamed after
    #: the job id, which is the object the product's job token names.
    workload_name: str

    @property
    def master(self) -> str:
        return self.nodes[0]

    @property
    def workers(self) -> tuple[str, ...]:
        return self.nodes[1:]

    @property
    def faulted_node(self) -> str:
        return self.master

    @property
    def gpu_count(self) -> int:
        return GPUS_PER_NODE * len(self.nodes)

    @property
    def workload_id(self) -> str:
        return f"gpu-fault-system/pytorchjob/{self.workload_name}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "job_id": self.job_id,
            "nodes": list(self.nodes),
            "master": self.master,
            "workers": list(self.workers),
            "faulted_node": self.faulted_node,
            "workload_name": self.workload_name,
            "gpu_count": self.gpu_count,
        }


def job_layout(
    targets: tuple[str, ...],
    run_id: str,
    *,
    workload_bases: dict[int, str] | None = None,
) -> list[JobSpec]:
    """Assign the chosen nodes to the two jobs in order: a=(t0,t1) and
    b=(t2,t3) on four targets, b=(t2,) on three.

    The master (first node of each job) is the one the reboot-class XID hits; a
    worker is only stopped and restarted with its job. ``workload_bases`` maps
    a job's node count to the fixed manifest name it is renamed from."""

    sizes = layout_sizes(targets)
    bases = workload_bases or WORKLOAD_BASES
    specs: list[JobSpec] = []
    start = 0
    for label, size in zip(JOB_LABELS, sizes, strict=True):
        job_id = f"{run_id}-{label}"
        specs.append(
            JobSpec(
                label=label,
                job_id=job_id,
                nodes=tuple(targets[start : start + size]),
                workload_name=managed_workload_name(bases[size], job_id),
            )
        )
        start += size
    return specs


def render_job_manifest(settings: Settings, spec: JobSpec, directory: Path) -> Path:
    """The job's manifest pinned to its nodes and renamed after its job id.

    A two-node job is the two-node manifest with its master and worker pinned;
    a single-node job is the single-node manifest pinned to its master. The
    manifests' fixed names would make two jobs of one manifest collide and the
    product's job token name an object the oracle cannot expect."""

    destination = directory / "job.yaml"
    if len(spec.nodes) == 2:
        join_case.render_two_node_manifest(
            settings.manifest,
            destination,
            master_node=spec.master,
            worker_node=spec.workers[0],
        )
    elif len(spec.nodes) == 1:
        render_node_pinned_manifest(
            settings.single_node_manifest, destination, node=spec.master
        )
    else:
        raise ValueError(f"{spec.job_id}: no manifest for {len(spec.nodes)} nodes")
    named = job_reboot_contract.name_rendered_job(destination, spec.job_id)
    if named != spec.workload_name:
        raise RegionalFixtureError(
            f"rendered job is named {named}, the plan expects {spec.workload_name}"
        )
    return destination


def profile_errors(profile: dict[str, Any] | None) -> list[str]:
    """The ownership the job recovery needs: the Kubernetes adapter must own
    STOP_WORKLOADS / RESTART_WORKLOAD, and the node agent must reboot."""

    errors = list(workload_case.managed_owner_errors(profile))
    if reboot_case.capability(profile, "nodeReboot") is None:
        errors.append("runtime profile has no nodeReboot capability")
    if (profile or {}).get("warnings"):
        errors.append("runtime profile has warnings")
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
    targets = choose_layout_targets(candidates, dns_nodes=dns_nodes)
    errors: list[str] = []
    if targets is None:
        errors.append(
            f"node group {group.name} has no {MIN_GROUP_NODES} eligible idle "
            "nodes now: "
            + "; ".join(reason for reasons in candidates.values() for reason in reasons)
        )
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
    jobs = (
        job_layout(targets, "destr028-plan", workload_bases=settings.workload_bases())
        if targets
        else []
    )
    if targets is not None:
        counted = [job.faulted_node for job in jobs]
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
            "layout": layout_name(targets) if targets else None,
            "jobs": [job.as_dict() for job in jobs],
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
    """What must not change between the approved plan and the execution."""

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
        "jobs": preflight.get("jobs"),
        "manifest": str(settings.manifest),
        "single_node_manifest": str(settings.single_node_manifest),
        "mutation": (
            "submit two PyTorchJobs on idle GPU nodes of one node group -- two "
            "two-node 16-GPU jobs on four nodes (2+2), or a two-node 16-GPU and "
            "a single-node 8-GPU job on three (2+1); write one synthetic XID 79 "
            "to /dev/kmsg on the master node of each job at the same moment "
            "(the node-reboot case's host probe and reboot sampler), so each "
            "job stops and reboots its faulted node and restarts once, both "
            "jobs at the same time. No replacement, no provider call beyond "
            "the two reboots."
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
            f"not even {MIN_GROUP_NODES} eligible idle nodes: not Ready, "
            "cordoned, tainted, owned, a declared spare, running Pods, hosting "
            "arbiters, a recent XID, an inactive Agent or a product-less GPU "
            "inventory",
            "the targets hold every kube-dns endpoint",
            "workload stop/restart or node reboot is not owned by the adapters",
            "the storm breaker is open, holds workflows, or still counts nodes "
            "that would make the case's second admission the trip",
            "an open workflow or incident on a target",
            "plan identity drift (release, runtime identity, node UID/boot id, "
            "Agent generation)",
            "the two faults do not land in two separate job workflows",
            "a workflow claims a node-group domain or a retired scope, is held, "
            "or the reboot steps do not overlap",
            "the breaker counts other than the two rebooted nodes, or trips",
            "provider mutation beyond the two reboots; cleanup leaves a residual",
        ],
        "rollback": {
            "node_reboots_are_owned_by_the_product_workflow": True,
            "failed_validation_keeps_the_node_quarantined": True,
            "runner_deletes_the_test_jobs_after_quiescence": True,
            "detached_samplers_are_stopped_after_the_post_reboot_snapshots": True,
            "runner_finally_deletes_probe_and_prewarm_resources": True,
            "no_storm_breaker_or_control_plane_setting_is_changed": True,
        },
    }
    record_focused_tests(details, dict(preflight.get("focused_tests") or {}))
    return details


def parser() -> argparse.ArgumentParser:
    # No abbreviations: a typed ``--node`` must not become ``--node-group``.
    value = argparse.ArgumentParser(
        description=(
            "Run the guarded DESTR-028 acceptance: two training jobs of one "
            "node group (two-node + two-node, or two-node + single-node) "
            "recover in parallel from a reboot-class fault on one node of each "
            "(b220 + b223)."
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
    value.add_argument(
        "--single-node-manifest",
        default="",
        help="the single-node job of the 2+1 layout; default: the shipped one",
    )
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
    jobs: list[JobSpec], workflow_ids: dict[str, str]
) -> list[oracle.DomainExpectation]:
    """One job-domain expectation per job, carrying its job tokens (named by
    its renamed workload) and its allocation -- two nodes, or the single-node
    job's one (``workflow_ids`` maps a job label to the live workflow its fault
    produced)."""

    return [
        oracle.DomainExpectation(
            label=f"job-{job.label}",
            faulted_node=job.faulted_node,
            workflow_id=workflow_ids.get(job.label, ""),
            remediation="RESTART_NODE",
            job_id=job.job_id,
            workload_ids=(job.workload_id,),
            allocation=job.nodes,
        )
        for job in jobs
    ]


# --------------------------------------------------------------------------- #
# Live execution (not exercised by the unit suite; the oracle and the reboot
# proof it calls are)
# --------------------------------------------------------------------------- #
@dataclass
class JobRun:
    """One job's fixture, faulted-node probe and records."""

    spec: JobSpec
    fixture: ManagedWorkloadFixture
    probe: HostProbeFixture
    run_id: str
    directory: Path
    marker: str = ""
    baseline: dict[str, Any] = field(default_factory=dict)
    target_bdf: str = ""
    injected_at: str = ""
    state: dict[str, Any] = field(default_factory=dict)
    sampler_started: bool = False
    sampler_stopped: bool = False
    injection_attempted: bool = False
    job_submitted: bool = False

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
    prewarm: ImagePrewarmFixture
    jobs: dict[str, JobRun]
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def window_open(self, stage: str) -> None:
        if datetime.now(timezone.utc) >= self.maintenance_window_end:
            raise RegionalFixtureError(f"maintenance window ended before {stage}")


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
    specs = job_layout(
        tuple(preflight["targets"]), run_id, workload_bases=settings.workload_bases()
    )
    jobs: dict[str, JobRun] = {}
    for spec in specs:
        directory = case_dir / f"job-{spec.label}"
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        fixture = ManagedWorkloadFixture(
            regional,
            ManagedWorkloadSettings(
                manifest=render_job_manifest(settings, spec, directory),
                site_file=settings.site_file,
                job_id=spec.job_id,
                attempt_id=f"{spec.job_id}-a001",
                restart_budget=1,
                expected_pods=len(spec.nodes),
                expected_gpu_count=spec.gpu_count,
            ),
        )
        jobs[spec.label] = JobRun(
            spec=spec,
            fixture=fixture,
            run_id=f"{run_id}-{spec.label}",
            directory=directory,
            probe=HostProbeFixture(
                HostProbeSettings(
                    state_directory=case_dir / "host-probes",
                    kubeconfig=settings.regional.gpu_kubeconfig,
                    context=settings.regional.gpu_context,
                    namespace=settings.regional.namespace,
                    node=spec.master,
                    image=settings.host_probe_image,
                    case_id=CASE_ID,
                    run_id=f"{run_id}-{spec.label}",
                    probe_script=PROBE_SCRIPT,
                )
            ),
        )
    return _LiveRun(
        settings=settings,
        case_dir=case_dir,
        attempt=attempt,
        maintenance_window_end=maintenance_window_end,
        preflight=preflight,
        regional=regional,
        prewarm=ImagePrewarmFixture(regional, case_id=CASE_ID, run_id=run_id),
        jobs=jobs,
    )


def _launch_jobs(run: _LiveRun) -> None:
    """Prewarm the image on every target, submit both GPU-active jobs, wait for
    every rank healthy."""

    run.window_open("image prewarm")
    nodes = sorted({node for item in run.jobs.values() for node in item.spec.nodes})
    write_json_atomic(run.case_dir / "prewarm.json", run.prewarm.create(nodes))
    for item in run.jobs.values():
        run.window_open(f"submit {item.spec.job_id}")
        item.job_submitted = True
        write_json_atomic(item.directory / "submission.json", item.fixture.submit())
    for item in run.jobs.values():
        run.window_open(f"running {item.spec.job_id}")
        source = item.fixture.wait_running(timeout_seconds=WORKFLOW_BUDGET_SECONDS)
        write_json_atomic(item.directory / "workload-source.json", source)


def _baselines_and_samplers(run: _LiveRun) -> None:
    for item in run.jobs.values():
        run.window_open(f"the host baseline of {item.spec.master}")
        # The master runs its job's ranks by design: their compute clients are
        # not a pre-existing fault (the idle-reset rule is not this case's).
        baseline, _count, bdf = reset_case.prepare_host_baseline(
            item.probe,
            {"node": run.preflight["nodes"][item.spec.master]},
            item.directory,
            require_idle=False,
        )
        item.baseline, item.target_bdf = baseline, bdf
    run.regional.verify_runtime_identity(
        run.preflight["runtime_identity"],
        evidence_path=run.case_dir / "runtime-identity-before-injection.json",
        stage="before DESTR-028 injection",
    )
    for item in run.jobs.values():
        run.window_open(f"the sampler start on {item.spec.master}")
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
    """One XID 79 per job's faulted node, written at the same moment."""

    run.window_open("injection")
    base = f"destr028-{int(time.time())}-a{run.attempt}"
    for item in run.jobs.values():
        item.marker = f"{base}-{item.spec.label}"
    run.started_at = datetime.now(timezone.utc)

    def write(item: JobRun) -> dict[str, Any]:
        item.injection_attempted = True
        payload = item.probe.execute(
            "write-xid79",
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
        max_workers=len(run.jobs), thread_name_prefix="d028"
    ) as pool:
        futures = {label: pool.submit(write, item) for label, item in run.jobs.items()}
        writes = {label: future.result() for label, future in futures.items()}
    write_json_atomic(
        run.case_dir / "injection.json",
        {
            "started_at": run.started_at.isoformat(),
            "markers": {label: item.marker for label, item in run.jobs.items()},
            "injected_at": {
                label: item.injected_at for label, item in run.jobs.items()
            },
            "writes": writes,
        },
    )


def _wait_terminal(run: _LiveRun) -> None:
    deadline = time.monotonic() + WORKFLOW_BUDGET_SECONDS
    for item in run.jobs.values():
        item.state = run.regional.wait_for_workflow(
            node=item.spec.master,
            marker=item.marker,
            observed_after=run.started_at,
            case_dir=item.directory,
            timeout_seconds=max(60, int(deadline - time.monotonic())),
        )
        write_json_atomic(item.directory / "workflow-state.json", item.state)


def _judge_records(run: _LiveRun) -> oracle.OracleVerdict:
    workflow_ids = {label: item.workflow_id for label, item in run.jobs.items()}
    domains = oracle_domains([item.spec for item in run.jobs.values()], workflow_ids)
    if not all(domain.workflow_id for domain in domains):
        return oracle.unreadable_verdict(
            "a job's XID never produced a workflow the store could name"
        )
    try:
        snapshot = oracle.read_oracle_snapshot(
            run.regional,
            workflow_ids=[domain.workflow_id for domain in domains],
            # Every node of both jobs: a healthy worker is stopped and
            # restarted too, so residue the case left could sit on it.
            node_ids=[node for item in run.jobs.values() for node in item.spec.nodes],
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


def job_node_contract(run: _LiveRun, item: JobRun) -> list[str]:
    """The job's faulted master: rebooted by its job workflow, which stopped
    and restarted the job; a healthy worker only stopped and restarted with it
    (``job_reboot_contract``: DESTR-002's reboot proof without its provider
    half). A single-node job has no worker to judge."""

    master = item.spec.master
    before = run.preflight["nodes"][master]
    after = run.regional.node_snapshot(master)
    write_json_atomic(item.directory / "node-after.json", after)
    if product_reboot(product_escalation(item.state), before, after):
        item.sampler_stopped = True  # the boot ended the sampler's transient unit
    errors = [
        f"{item.spec.job_id}: {master}: {error}"
        for error in job_reboot_contract.faulted_node_errors(
            item.state,
            node_before=before,
            node_after=after,
            store_before=(run.preflight.get("stores") or {}).get(master),
        )
    ]
    errors.extend(
        f"{item.spec.job_id}: {error}"
        for error in job_reboot_contract.healthy_peer_errors(
            {node: run.preflight["nodes"][node] for node in item.spec.workers},
            {node: run.regional.node_snapshot(node) for node in item.spec.workers},
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
        raise RegionalFixtureError("DESTR-028 plan identity drifted before execution")


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
        "jobs": [item.spec.as_dict() for item in run.jobs.values()],
        "maintenance_window_end": maintenance_window_end.isoformat(),
    }
    try:
        _launch_jobs(run)
        _baselines_and_samplers(run)
        _inject_both(run)
        _wait_terminal(run)
        verdict = _judge_records(run)
        errors = list(verdict.errors)
        for item in run.jobs.values():
            errors.extend(job_node_contract(run, item))
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
                "markers": {
                    item.spec.master: item.marker for item in run.jobs.values()
                },
                "workflow_request_ids": {
                    item.spec.label: item.workflow_id for item in run.jobs.values()
                },
                "incident_ids": {
                    item.spec.label: item.incident_id for item in run.jobs.values()
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
    """Per job: stop the sampler, delete the test job, delete the probe
    resources; then the image prewarm. The runner never restores a node
    itself: a workflow that failed keeps its node quarantined for the operator,
    and the case FAILs."""

    jobs: dict[str, Any] = {}
    for item in run.jobs.values():
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
        if item.job_submitted:
            try:
                report["job_delete"] = item.fixture.delete()
            except Exception as exc:  # noqa: BLE001 - a cleanup failure is a FAIL
                report["job_delete_error"] = f"{type(exc).__name__}: {exc}"
                result["verdict"] = oracle.FAIL
        try:
            residuals = item.probe.cleanup()
        except Exception as exc:  # noqa: BLE001 - a cleanup failure is a FAIL
            residuals = {"cleanup_error": True}
            report["probe_cleanup_error"] = f"{type(exc).__name__}: {exc}"
        report["probe_residuals"] = residuals
        if any(residuals.values()):
            result["verdict"] = oracle.FAIL
        jobs[item.spec.label] = report
    result["job_reports"] = jobs
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
            stage="after DESTR-028 cleanup",
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
