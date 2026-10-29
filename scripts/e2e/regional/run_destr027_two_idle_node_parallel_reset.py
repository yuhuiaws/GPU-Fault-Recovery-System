#!/usr/bin/env python3
"""GF-REGIONAL-DESTR-027 live acceptance runner.

Two idle GPU nodes of ONE node group recover in parallel (b220). Before b220
every budgeted claim took one ``domain:<cluster>:<group>`` slot per node group
or instance group, so two independent faults on two nodes of a group were
repaired one at a time: the second workflow waited PENDING behind a
``REMEDIATION_BUDGET_WAIT`` hold on that scope. Since b220 an idle node is
its own merged workflow, kept apart from the others only by its
``node:<cluster>:<node>`` scope.

The fault is DESTR-001's, unchanged: one synthetic XID 46 written to the
host's ``/dev/kmsg`` by the destructive host probe (``write-xid46``), here on
each of two idle nodes at the same moment, each node with DESTR-001's detached
reset sampler armed. Each node then runs DESTR-001's eight-step RESET_GPU
workflow, judged per node with DESTR-001's own contract and host checks. What
this case adds is judged by the shared read-only oracle
(``workload_domain_oracle``) on the durable store records: two workflows, each
claiming only its node scope (plus a node-local fabric partition when one of
its steps names it), neither ever held on a node-group ``domain:`` or another
retired scope, their budgeted steps overlapping in wall-clock time (step
execution ``started_at``/``updated_at``), the storm breaker CLOSED with the
two nodes it counted for them, both incidents RECOVERED and nothing left open
on either node.

Precondition, read live: at least ``MIN_GROUP_NODES`` GPU nodes in one node
group (``eks.amazonaws.com/nodegroup`` / ``sagemaker.amazonaws.com/
instance-group-name``); short of it the case is ``NOT_RUN(CAPABILITY_ABSENT)``
-- the plan says so and ``--execute`` records it without touching anything.
The group and its two target nodes are chosen from the live cluster (the
largest GPU node group, or ``--node-group``; the first two eligible nodes in
name order), never from a site-profile node list.

Two nodes are below the storm breaker's floor (3) on any cluster, so the case
cannot trip it alone; the preflight refuses while the breaker still counts
other nodes from the last 600 s that would make the case's second admission
the trip, and the result records ``window_clear_at`` -- the next destructive
case on the cluster starts after it.

QUIESCE_GPU_SERVICES stops kubelet on HyperPod nodes, so between the injection
and the terminal workflows the runner reads the store only.

Defaults to ``--plan``; ``--execute`` needs ``--confirm DESTR027_EXECUTE``.
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
    run_destr015_parallel_branch_join as join_case,
)
from scripts.e2e.regional import workload_domain_oracle as oracle  # noqa: E402
from scripts.e2e.regional.acceptance_runner_common import (  # noqa: E402
    processor_queue_backlog,
    remote_command_queue_errors,
    write_json_atomic,
)
from scripts.e2e.regional.gpu_identity_preflight import (  # noqa: E402
    gpu_identity_errors,
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
    escalation_errors,
    product_escalation,
    product_reboot,
    reboot_proof_error,
)

PROBE_SCRIPT = Path(__file__).with_name("probes") / "destructive_node_probe.py"
CASE_ID = "GF-REGIONAL-DESTR-027"
PREDECESSOR_CASE_ID = "GF-REGIONAL-DESTR-001"
CONFIRMATION = "DESTR027_EXECUTE"
#: The precondition: two idle nodes need a node group of at least two.
MIN_GROUP_NODES = 2
LABELS = ("a", "b")
#: Both workflows run at once; DESTR-001 allows one 1200 s, plus margin.
WORKFLOW_BUDGET_SECONDS = 1500
#: DESTR-001's quiet window: a recent XID on a target refuses the plan.
RECENT_EVENT_SECONDS = 600
#: The workflows' own node actions, as DESTR-001 requires of its Agent.
AGENT_OPERATIONS = frozenset(reset_case.REQUIRED_AGENT_OPERATIONS)
FOCUSED_TESTS = (
    "tests/execution/test_remediation_budget.py::"
    "test_two_idle_nodes_are_repaired_together",
    "tests/execution/test_remediation_budget.py::"
    "test_the_policy_reads_no_node_label_failure_domain_map",
    "tests/execution/test_remediation_storm_admission.py::test_two_nodes_never_trip_it",
    "tests/node_agent/test_remediation.py::"
    "test_gpu_reset_is_idempotent_and_rechecks_clients",
    "tests/regional/test_workload_domain_oracle.py",
    "tests/regional/test_destr027_two_idle_node_parallel_reset.py",
)


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Settings:
    regional: RegionalLiveSettings
    host_probe_image: str
    predecessor_path: Path
    #: The node group to use; empty takes the cluster's largest GPU group.
    node_group: str

    def environment(self) -> dict[str, str]:
        return {
            **self.regional.environment(),
            "GPU_FAULT_HOST_PROBE_IMAGE": self.host_probe_image,
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
    return Settings(
        regional=settings_from_arguments(arguments),
        host_probe_image=required(
            arguments.host_probe_image or os.getenv("GPU_FAULT_HOST_PROBE_IMAGE", ""),
            "host probe image",
        ),
        predecessor_path=predecessor,
        node_group=str(arguments.node_group or "").strip(),
    )


def run_identity(run_dir: Path, attempt: int) -> str:
    """The probes' and samplers' run id. Run directories end in the lane token
    every round, so the resolved directory's digest keeps two rounds apart."""

    token = run_dir.name.rsplit("-", 1)[-1].lower()
    digest = hashlib.sha256(str(run_dir.resolve()).encode()).hexdigest()[:8]
    return f"destr027-{token}-{digest}-a{attempt}"


# --------------------------------------------------------------------------- #
# Target selection (pure, unit-tested)
# --------------------------------------------------------------------------- #
def node_ineligibility(
    node: str,
    *,
    snapshot: dict[str, Any],
    store: dict[str, Any],
    business_workloads: list[dict[str, Any]],
    arbiter_pods: list[str],
    gpu_workloads: list[dict[str, Any]],
    workload_view: dict[str, Any],
) -> list[str]:
    """Why ``node`` cannot be one of the two idle targets now (empty: it can).

    ``workload_view`` is the control plane's own resolution of the node
    (``RegionalLiveFixture.node_workload_view``): only an IDLE node compiles
    the reset plan; UNKNOWN refuses it (DESTR-023/024), ACTIVE adds a job.
    """

    reasons: list[str] = []
    state = str(workload_view.get("workload_state") or "")
    if state.rsplit(".", 1)[-1] != "IDLE":
        reasons.append(f"{node} workload state is {state or 'unknown'}, not IDLE")
    if snapshot.get("ready") != "True":
        reasons.append(f"{node} is not Ready")
    if snapshot.get("unschedulable"):
        reasons.append(f"{node} is cordoned")
    if snapshot.get("taints"):
        reasons.append(f"{node} carries isolating taints {snapshot.get('taints')}")
    if snapshot.get("ownership_annotations"):
        reasons.append(f"{node} is owned by an incident")
    if snapshot.get("spare_pool_annotations"):
        reasons.append(f"{node} is a declared warm spare")
    if int(snapshot.get("gpu_allocatable") or 0) <= 0:
        reasons.append(f"{node} allocates no GPU")
    if business_workloads:
        reasons.append(f"{node} runs workloads {business_workloads}")
    on_node = [item for item in gpu_workloads if item.get("node") == node]
    if on_node:
        reasons.append(f"{node} runs GPU Pods {on_node}")
    if arbiter_pods:
        reasons.append(f"{node} hosts controller/executor arbiter Pods {arbiter_pods}")
    agent = store.get("agent") or {}
    if agent.get("lifecycle_state") != "ACTIVE":
        reasons.append(f"{node} Node Agent is not ACTIVE")
    missing = sorted(AGENT_OPERATIONS - set(agent.get("allowed_operations") or []))
    if missing:
        reasons.append(f"{node} Node Agent does not allow {missing}")
    if store.get("event") is not None:
        reasons.append(f"{node} had an XID in the last {RECENT_EVENT_SECONDS}s")
    reasons.extend(
        gpu_identity_errors(
            node, store.get("gpu_identity"), boot_id=snapshot.get("boot_id")
        )
    )
    return reasons


def choose_targets(
    candidates: dict[str, list[str]], *, dns_nodes: list[str]
) -> tuple[str, str] | None:
    """The first two eligible nodes in name order that do not hold every
    kube-dns endpoint between them (quiescing both would take cluster DNS
    down); ``None`` when no such pair exists."""

    eligible = sorted(node for node, reasons in candidates.items() if not reasons)
    for index, first in enumerate(eligible):
        for second in eligible[index + 1 :]:
            if not dns_nodes or not set(dns_nodes) <= {first, second}:
                return first, second
    return None


def profile_errors(profile: dict[str, Any] | None) -> list[str]:
    """DESTR-001's ownership rule for the reset the two workflows run."""

    reset = reset_case.capability(profile, "gpuReset")
    errors: list[str] = []
    if reset is None:
        errors.append("runtime profile has no gpuReset capability")
    elif (
        reset.get("mode") != "OWN"
        or reset.get("owner") != "gpu-fault-node-agent"
        or reset.get("adapter") != "node-action"
    ):
        errors.append("gpuReset is not OWN by the Node Agent")
    if (profile or {}).get("warnings"):
        errors.append("runtime profile has warnings")
    return errors


def residue_errors(store: dict[str, Any], targets: tuple[str, str]) -> list[str]:
    """Open workflows or incidents already on the targets (before the case)."""

    errors = [
        f"{item.get('node_ids')} already have an open workflow "
        f"{item.get('workflow_request_id')} ({item.get('status')})"
        for item in store.get("active_workflow_incidents") or []
        if set(item.get("node_ids") or []) & set(targets)
    ]
    errors.extend(
        f"{item.get('node_ids')} already have incident {item.get('incident_id')} "
        f"in {item.get('state')}"
        for item in store.get("open_incidents") or []
        if set(item.get("node_ids") or []) & set(targets)
    )
    return errors


# --------------------------------------------------------------------------- #
# Preflight / plan
# --------------------------------------------------------------------------- #
def focused_tests(case_dir: Path, *, reuse: bool = False) -> dict[str, Any]:
    """Run the focused pytest, or reuse the plan's passing result for this tree."""

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
        node: node_ineligibility(
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
    targets = choose_targets(candidates, dns_nodes=dns_nodes)
    errors: list[str] = []
    if targets is None:
        errors.append(
            f"node group {group.name} has no two eligible idle nodes now: "
            + "; ".join(reason for reasons in candidates.values() for reason in reasons)
        )
    first = stores[targets[0] if targets else group.nodes[0]]
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
    if targets is not None:
        errors.extend(
            oracle.breaker_premise_errors(
                storm,
                cluster_id=cluster_id,
                targets=targets,
                now=datetime.now(timezone.utc),
            )
        )
        errors.extend(residue_errors(storm, targets))
    tests = focused_tests(case_dir, reuse=reuse_focused_tests)
    if not tests.get("passed"):
        errors.append("focused regression tests failed")
    predecessor = predecessor_evidence(
        settings.predecessor_path, PREDECESSOR_CASE_ID, **identity
    )
    if not predecessor.get("valid"):
        errors.append("DESTR-001 predecessor evidence is not PASS")
    result.update(
        {
            "node_group": group.as_dict(),
            "targets": list(targets) if targets else [],
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
                "runtime_profile_version": (
                    (stores.get(node) or {}).get("profile") or {}
                ).get("profile_version"),
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
        "mutation": (
            "write one synthetic XID 46 to /dev/kmsg on each of two idle GPU "
            "nodes of one node group at the same moment (DESTR-001's host probe "
            "and detached reset sampler on each); allow each node's own "
            "RESET_GPU workflow to cordon, quiesce, reset one GPU, restore, "
            "validate and release it, both at once. No reboot, no replacement, "
            "no provider call, no workload."
        ),
        "storm_premise": {
            "threshold": oracle.storm_threshold(storm.get("cluster_gpu_agents")),
            "cluster_gpu_agents": storm.get("cluster_gpu_agents"),
            "breaker_state": (storm.get("breaker") or {}).get("state", "CLOSED"),
            "case_counted_nodes": MIN_GROUP_NODES,
        },
        "preflight_identity": identity,
        "preflight_identity_digest": join_case.identity_digest(identity),
        "stop_conditions": [
            "fewer than two GPU nodes in one node group (NOT_RUN(CAPABILITY_ABSENT))",
            "DESTR-001 has not passed on this release",
            "no two eligible idle nodes: not Ready, cordoned, tainted, owned, a "
            "declared spare, running Pods, hosting arbiters, a recent XID, an "
            "inactive Agent or a product-less GPU inventory",
            "the two targets hold every kube-dns endpoint",
            "the storm breaker is open, holds workflows, or still counts nodes "
            "that would make the case's second admission the trip",
            "an open workflow or incident on a target",
            "plan identity drift (release, runtime identity, node UID/boot id, "
            "Agent generation, profile)",
            "either workflow differs from DESTR-001's reset contract",
            "a workflow claims a node-group domain or a retired scope, is held, "
            "or the budgeted steps do not overlap",
            "provider mutation appears; cleanup leaves a residual",
        ],
        "rollback": {
            "gpu_service_restoration_is_owned_by_the_product_workflow": True,
            "failed_validation_keeps_the_node_quarantined": True,
            "detached_samplers_are_stopped_after_the_post_reset_snapshots": True,
            "runner_finally_deletes_probe_resources": True,
            "no_storm_breaker_or_control_plane_setting_is_changed": True,
        },
    }
    record_focused_tests(details, dict(preflight.get("focused_tests") or {}))
    return details


def parser() -> argparse.ArgumentParser:
    # No abbreviations: a typed ``--node`` must not become ``--node-group``.
    value = argparse.ArgumentParser(
        description=(
            "Run the guarded DESTR-027 acceptance: two idle GPU nodes of one node "
            "group fault together and recover as two parallel b220 workflows."
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
# Live execution
# --------------------------------------------------------------------------- #
@dataclass
class _NodeRun:
    """One target node's probe, sampler and records."""

    label: str
    node: str
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
    result: dict[str, Any] = field(default_factory=dict)

    @property
    def workflow(self) -> dict[str, Any]:
        return dict(self.state.get("workflow") or {})

    @property
    def incident_id(self) -> str:
        return str((self.state.get("incident") or {}).get("incident_id") or "")

    def stop_sampler(self) -> dict[str, Any]:
        stopped = self.probe.execute(
            "stop-reset-sampler", "--run-id", self.run_id, timeout=60
        )
        self.sampler_stopped = True
        write_json_atomic(self.directory / "sampler-final.json", stopped)
        return stopped


@dataclass
class _LiveRun:
    settings: Settings
    case_dir: Path
    attempt: int
    maintenance_window_end: datetime
    preflight: dict[str, Any]
    regional: RegionalLiveFixture
    nodes: dict[str, _NodeRun]
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
    nodes: dict[str, _NodeRun] = {}
    for label, node in zip(LABELS, preflight["targets"], strict=True):
        directory = case_dir / f"node-{label}"
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        node_run_id = f"{run_id}-n{label}"
        nodes[label] = _NodeRun(
            label=label,
            node=node,
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
    return _LiveRun(
        settings=settings,
        case_dir=case_dir,
        attempt=attempt,
        maintenance_window_end=maintenance_window_end,
        preflight=preflight,
        regional=RegionalLiveFixture(settings.regional),
        nodes=nodes,
    )


def _baselines_and_samplers(run: _LiveRun) -> None:
    """DESTR-001's host baseline on both targets, then both samplers armed."""

    for item in run.nodes.values():
        run.window_open(f"the host baseline of {item.node}")
        baseline, count, bdf = reset_case.prepare_host_baseline(
            item.probe, {"node": run.preflight["nodes"][item.node]}, item.directory
        )
        item.baseline, item.expected_gpu_count, item.target_bdf = baseline, count, bdf
    run.regional.verify_runtime_identity(
        run.preflight["runtime_identity"],
        evidence_path=run.case_dir / "runtime-identity-before-injection.json",
        stage="before DESTR-027 injection",
    )
    for item in run.nodes.values():
        run.window_open(f"the sampler start on {item.node}")
        # Armed before the call: a start that times out after creating the unit
        # must still be stopped by cleanup (the stop is idempotent).
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
    """One XID 46 per target, written at the same moment."""

    run.window_open("injection")
    base = f"destr027-{int(time.time())}-a{run.attempt}"
    for item in run.nodes.values():
        item.marker = f"{base}-n{item.label}"
    run.started_at = datetime.now(timezone.utc)

    def write(item: _NodeRun) -> dict[str, Any]:
        item.injection_attempted = True
        payload = item.probe.execute(
            "write-xid46",
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

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="destr027-xid") as pool:
        futures = {label: pool.submit(write, item) for label, item in run.nodes.items()}
        writes = {label: future.result() for label, future in futures.items()}
    write_json_atomic(
        run.case_dir / "injection.json",
        {
            "started_at": run.started_at.isoformat(),
            "markers": {label: item.marker for label, item in run.nodes.items()},
            "injected_at": {
                label: item.injected_at for label, item in run.nodes.items()
            },
            "writes": writes,
        },
    )


def _wait_terminal(run: _LiveRun) -> None:
    """Store reads only: a quiesced HyperPod node's kubelet is down."""

    deadline = time.monotonic() + WORKFLOW_BUDGET_SECONDS
    for item in run.nodes.values():
        item.state = run.regional.wait_for_workflow(
            node=item.node,
            marker=item.marker,
            observed_after=run.started_at,
            case_dir=item.directory,
            timeout_seconds=max(60, int(deadline - time.monotonic())),
        )
        write_json_atomic(item.directory / "workflow-state.json", item.state)


def _judge_records(run: _LiveRun) -> oracle.OracleVerdict:
    """The shared oracle on the two workflows' durable records."""

    domains = [
        oracle.DomainExpectation(
            label=f"node-{item.label}",
            faulted_node=item.node,
            workflow_id=str(item.workflow.get("request_id") or ""),
        )
        for item in run.nodes.values()
    ]
    if not all(domain.workflow_id for domain in domains):
        return oracle.unreadable_verdict(
            "a target's XID never produced a workflow the store could name"
        )
    try:
        snapshot = oracle.read_oracle_snapshot(
            run.regional,
            workflow_ids=[domain.workflow_id for domain in domains],
            node_ids=[item.node for item in run.nodes.values()],
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
        overlap=oracle.OVERLAP_BUDGETED,
        expected_counted_nodes=len(run.nodes),
    )


def _node_contract(run: _LiveRun, item: _NodeRun) -> list[str]:
    """DESTR-001's reset contract and host proof for one target."""

    escalation = product_escalation(item.state)
    errors = [
        f"{item.node}: {error}"
        for error in (
            *escalation_errors(escalation),
            *reset_case.workflow_errors(item.state),
        )
    ]
    restored, node_after = reset_case.node_restored_errors(
        run.regional, item.node, item.directory
    )
    before = run.preflight["nodes"][item.node]
    rebooted = reset_case.post_reset_host(
        item.probe,
        item.directory,
        item.result,
        item.after,
        arguments=(
            "--since-epoch",
            str(run.started_at.timestamp()),
            "--pci-bdf",
            item.target_bdf,
            "--run-id",
            item.run_id,
        ),
        stop_sampler=item.stop_sampler,
        reboot=product_reboot(escalation, before, node_after),
        recheck=lambda: product_reboot(
            escalation, before, run.regional.node_snapshot(item.node)
        ),
    )
    if rebooted:
        item.sampler_stopped = True  # the boot ended the sampler's transient unit
        errors.append(f"{item.node}: {reboot_proof_error(escalation, rebooted)}")
    else:
        errors.extend(
            f"{item.node}: {error}"
            for error in reset_case.host_errors(
                item.baseline,
                item.after,
                expected_gpu_count=item.expected_gpu_count,
                target_bdf=item.target_bdf,
                incident_id=item.incident_id,
                workflow_request_id=str(item.workflow.get("request_id") or ""),
            )
        )
    errors.extend(f"{item.node}: {error}" for error in restored)
    return errors


def not_run_result(
    attempt: int, identity: dict[str, Any], record: dict[str, Any]
) -> dict[str, Any]:
    """The evidence of a cluster that cannot carry the case: nothing ran."""

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
        raise RegionalFixtureError("DESTR-027 plan identity drifted before execution")


def execute_case(
    settings: Settings,
    run_dir: Path,
    attempt: int,
    maintenance_window_end: datetime,
) -> int:
    """Drive the live case. Not exercised by the unit suite; the oracle and the
    DESTR-001 contract it calls are. Every cleanup failure is a FAIL."""

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
        "maintenance_window_end": maintenance_window_end.isoformat(),
    }
    try:
        _baselines_and_samplers(run)
        _inject_both(run)
        _wait_terminal(run)
        verdict = _judge_records(run)
        errors = list(verdict.errors)
        for item in run.nodes.values():
            errors.extend(_node_contract(run, item))
        provider, provisional = reset_case.provider_evidence(
            run.regional, case_dir, run.started_at
        )
        if provider:
            errors.append("provider mutation appeared during the two GPU resets")
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
                "markers": {item.node: item.marker for item in run.nodes.values()},
                "injected_at": {
                    item.node: item.injected_at for item in run.nodes.values()
                },
                "workflow_request_ids": {
                    item.node: item.workflow.get("request_id")
                    for item in run.nodes.values()
                },
                "incident_ids": {
                    item.node: item.incident_id for item in run.nodes.values()
                },
                "provider_events": provider,
                "provider_events_provisional": provisional,
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
    """Per target: the product's own compensation report, the sampler, the
    probe resources and DESTR-001's postflight; then the identity recheck.
    The runner never restores a node itself: a workflow that failed keeps its
    node quarantined for the operator, and the case FAILs."""

    nodes: dict[str, Any] = {}
    for item in run.nodes.values():
        report = dict(item.result)
        recovery = reset_case.quiesce_recovery_report(
            item.state, item.after, injection_attempted=item.injection_attempted
        )
        report["quiesce_recovery"] = recovery
        if recovery["operator_review_required"]:
            result["verdict"] = oracle.FAIL
        if item.sampler_started and not item.sampler_stopped:
            try:
                item.stop_sampler()
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
        reset_case.postflight(
            run.regional,
            item.node,
            run.preflight["nodes"][item.node],
            report,
            item.incident_id,
        )
        if report.get("verdict") == oracle.FAIL:
            result["verdict"] = oracle.FAIL
            result.setdefault("errors", []).extend(
                f"{item.node}: {error}" for error in report.get("errors") or []
            )
        nodes[item.node] = report
    result["nodes"] = nodes
    try:
        run.regional.verify_runtime_identity(
            run.preflight["runtime_identity"],
            evidence_path=run.case_dir / "runtime-identity-after-cleanup.json",
            stage="after DESTR-027 cleanup",
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
