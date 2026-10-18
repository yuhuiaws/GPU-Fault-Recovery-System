#!/usr/bin/env python3
"""GF-REGIONAL-DESTR-026 live acceptance runner.

Two GPUs of ONE node fault inside the aggregation window while that node runs
the Master of the case's own two-node PyTorchJob (16 GPUs, the Worker pinned to
a second node). Two host probes on the fault node write one XID 46 each,
concurrently, with the two GPUs' PCI addresses, so both faults reach the
control plane before the job workflow's first step is claimed. The product
merges them into one incident and one linear job workflow: the shared
STOP_WORKLOADS, one quiesce window on the fault node whose single RESET_GPU
names both GPUs (``gpu_uuids_by_node``), the Node Agent resetting them one
after another inside it, VALIDATE_GPU for both, RESTORE_SCHEDULING and one
RESTART_WORKLOAD. The job restarts exactly once on the same two nodes; the
second node is never cordoned, quiesced or reset, and no other GPU node moves.

QUIESCE_GPU_SERVICES stops kubelet on HyperPod nodes, so from the injection
until the workflow is terminal the runner reads the store only; DESTR-001's
detached host sampler records the two GPUs leaving and returning meanwhile,
and every host read waits for the fault node to answer again.

A second fault that arrives after the reset already ran is a different shape
(a successor branch before the join, or a successor workflow once the join
started); it is documented with the case and not driven here: the exec
channel and the node's own kernel collector are down inside the window, so
no existing hook can time a kmsg write after the reset claim.

Defaults to ``--plan``; ``--execute`` needs ``--confirm DESTR026_EXECUTE``.
The verdicts are pure (``destr026_verdicts.py``) and unit-tested; the runner
records digests only.
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

from scripts.e2e.regional import destr026_verdicts as verdicts  # noqa: E402
from scripts.e2e.regional import run_destr001_gpu_reset as reset_case  # noqa: E402
from scripts.e2e.regional import (  # noqa: E402
    run_destr009_workload_restart as workload_case,
)
from scripts.e2e.regional import (  # noqa: E402
    run_destr015_parallel_branch_join as join_case,
)
from scripts.e2e.regional.acceptance_runner_common import (  # noqa: E402
    write_json_atomic,
)
from scripts.e2e.regional.destr015_verdicts import (  # noqa: E402
    cloudtrail_errors,
    event_observed_at,
    host_errors,
    schedulability_errors,
    spread_errors,
    step_transitions,
    workload_errors,
)
from scripts.e2e.regional.destr_barrier_authorization import (  # noqa: E402
    host_reachability,
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

PROBE_SCRIPT = Path(__file__).with_name("probes") / "destructive_node_probe.py"
CASE_ID = "GF-REGIONAL-DESTR-026"
PREDECESSOR_CASE_ID = "GF-REGIONAL-DESTR-015"
CONFIRMATION = "DESTR026_EXECUTE"
EXPECTED_GPU_COUNT = 16
LABELS = ("a", "b")
OBSERVATION_BUDGET_SECONDS = 2400
EVENT_WAIT_SECONDS = 300
# The fault node's kubelet is down inside a HyperPod quiesce window; only the
# restore brings it back. No host read is attempted before it answers.
HOST_RETURN_BUDGET_SECONDS = 900
# The detached sampler's own bound (``RuntimeMaxSec`` in the probe).
SAMPLER_LIFETIME_SECONDS = 1800
# After the restart the Pods must stay the same ones this long: a second
# restart would replace them.
RESTART_STABLE_SECONDS = 60
# The watcher reports the restarted attempt on its next pass.
ATTEMPT_OBSERVATION_SECONDS = 180
# A recent XID on the fault node refuses the plan (DESTR-001's quiet window).
RECENT_EVENT_SECONDS = 600
FOCUSED_TESTS = (
    "tests/orchestration/test_merge.py::test_merged_reset_targets_every_faulted_gpu",
    "tests/orchestration/test_same_node_job_reset_merge.py",
    "tests/node_agent/test_reset_two_gpus_one_window.py",
    "tests/execution/test_restart_safety.py::"
    "test_restart_budget_blocks_second_restart_for_same_job",
    "tests/regional/test_destr026_same_node_two_gpu_reset.py",
)


# --------------------------------------------------------------------------- #
# Settings / plan
# --------------------------------------------------------------------------- #
def derived_identity(run_dir: Path, attempt: int) -> tuple[str, str]:
    suffix = hashlib.sha256(
        f"{run_dir.resolve()}\0{attempt}\0{CASE_ID}".encode()
    ).hexdigest()[:12]
    job_id = f"destr026-{suffix}"
    return job_id, f"{job_id}-a001"


def run_identity(run_dir: Path, attempt: int) -> str:
    """The probes' and the sampler's run id. Run directories end in the lane
    token every round, so the resolved directory's digest keeps two rounds
    apart (the DESTR-020 lesson)."""

    token = run_dir.name.rsplit("-", 1)[-1].lower()
    digest = hashlib.sha256(str(run_dir.resolve()).encode()).hexdigest()[:8]
    return f"destr026-{token}-{digest}-a{attempt}"


@dataclass(frozen=True)
class Settings:
    regional: RegionalLiveSettings
    site_file: Path
    manifest: Path
    host_probe_image: str
    fault_node: str
    node_b: str
    pci_bdf_a: str
    pci_bdf_b: str
    job_id: str
    attempt_id: str
    predecessor_path: Path

    @property
    def nodes(self) -> tuple[str, str]:
        return (self.fault_node, self.node_b)

    def environment(self) -> dict[str, str]:
        return {
            **self.regional.environment(),
            "GPU_FAULT_SITE_FILE": str(self.site_file),
            "GPU_FAULT_TRAINING_MANIFEST": str(self.manifest),
            "GPU_FAULT_HOST_PROBE_IMAGE": self.host_probe_image,
            "GPU_FAULT_FAULT_NODE": self.fault_node,
            "GPU_FAULT_NODE_B": self.node_b,
            "GPU_FAULT_TEST_JOB_ID": self.job_id,
            "GPU_FAULT_TEST_ATTEMPT_ID": self.attempt_id,
            "GPU_FAULT_PREDECESSOR_EVIDENCE": str(self.predecessor_path),
        }


def configure(arguments: argparse.Namespace) -> Settings:
    default_job, default_attempt = derived_identity(
        arguments.run_dir, arguments.attempt
    )
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
        fault_node=required(
            arguments.fault_node or os.getenv("GPU_FAULT_FAULT_NODE", ""),
            "fault node",
        ),
        node_b=required(
            arguments.node_b or os.getenv("GPU_FAULT_NODE_B", ""), "node B"
        ),
        pci_bdf_a=arguments.pci_bdf_a.strip(),
        pci_bdf_b=arguments.pci_bdf_b.strip(),
        job_id=arguments.job_id.strip() or default_job,
        attempt_id=arguments.attempt_id.strip() or default_attempt,
        predecessor_path=predecessor,
    )


def plan_details(settings: Settings, preflight: dict[str, Any]) -> dict[str, Any]:
    identity = join_case.plan_identity(preflight, nodes=settings.nodes)
    details: dict[str, Any] = {
        "risk": "destructive",
        "predecessor": preflight.get("predecessor"),
        "fault_node": settings.fault_node,
        "node_b": settings.node_b,
        "pci_bdfs": [
            settings.pci_bdf_a or "inventory[0]",
            settings.pci_bdf_b or "inventory[1]",
        ],
        "job_id": settings.job_id,
        "attempt_id": settings.attempt_id,
        "manifest": str(settings.manifest),
        "control_env": preflight.get("control_env"),
        "mutation": (
            "submit one two-node 16-GPU PyTorchJob (Master on the fault node, "
            "Worker on node B); write one XID 46 for each of two GPUs of the "
            "fault node to its /dev/kmsg at the same moment, so both join one "
            "job workflow before its first step; allow it to stop the job, "
            "quiesce the fault node once, reset both GPUs in that window, "
            "restore and validate them and restart the job once. No reboot, no "
            "replacement, no provider call, nothing on node B but the job."
        ),
        "preflight_identity": identity,
        "preflight_identity_digest": join_case.identity_digest(identity),
        "stop_conditions": [
            "DESTR-015 predecessor evidence is not PASS",
            "either node is busy, tainted, owned, or the fault node's Node Agent "
            "is not ACTIVE",
            "the fault node hosts an arbiter Pod or every kube-dns endpoint",
            "the fault node had an XID in the last ten minutes",
            "gpuReset is not OWN by the Node Agent, or workload stop/restart not OWN",
            "the two faults open two incidents or land in two workflows",
            "a step does not name both GPUs, or a node action lacks gpu_uuids_by_node",
            "a second QUIESCE, RESET_GPU or RESTART_WORKLOAD, or any reboot/replace",
            "the Node Agent does not reset both GPUs one after another in one window",
            "Fabric Manager is stopped or restarted around the per-GPU reset",
            "the job is not restarted exactly once on the same two nodes",
            "node B or any other GPU node is acted on",
            "CloudTrail shows any provider mutation",
            "cleanup cannot prove quiescence or leaves a node isolated",
        ],
        "rollback": {
            "runner_waits_for_async_quiescence_before_workload_delete": True,
            "runner_restores_a_node_left_isolated_via_validated_restore": True,
            "runner_never_execs_the_fault_node_before_it_answers_again": True,
            "detached_sampler_is_stopped_after_the_post_reset_snapshot": True,
            "runner_finally_deletes_probe_and_prewarm_Pods": True,
            "no_node_reboot_or_replacement_is_authorized": True,
        },
        "late_fault_variant": (
            "not driven: a second fault after the reset already ran queues a "
            "successor branch before the join (or a successor workflow once the "
            "join started); see the case procedure"
        ),
    }
    record_focused_tests(
        details, dict(preflight.get("focused_tests") or {"passed": False})
    )
    return details


# --------------------------------------------------------------------------- #
# Live preflight (delegates every assertion to destr026_verdicts)
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
    if not settings.site_file.is_file() or not settings.manifest.is_file():
        raise RegionalFixtureError("site file or training manifest does not exist")
    regional = RegionalLiveFixture(settings.regional)
    snapshots = {node: regional.node_snapshot(node) for node in settings.nodes}
    state = regional.store_snapshot(
        node=settings.fault_node,
        observed_after=datetime.now(timezone.utc)
        - timedelta(seconds=RECENT_EVENT_SECONDS),
    )
    runtime_identity = regional.runtime_identity()
    tests = focused_tests(case_dir, reuse=reuse_focused_tests)
    predecessor = predecessor_evidence(
        settings.predecessor_path,
        PREDECESSOR_CASE_ID,
        **regional.evidence_identity(),
    )
    env = join_case.control_env(regional)
    arbiter_pods, dns_nodes = join_case.cluster_arbiter_placement(regional)
    workloads = {node: regional.business_workloads(node) for node in settings.nodes}
    result: dict[str, Any] = {
        "release_id": state.get("release_id"),
        "nodes": snapshots,
        "arbiter_pods": arbiter_pods,
        "dns_nodes": dns_nodes,
        "agent": state.get("agent"),
        "gpu_identity": state.get("gpu_identity"),
        "recent_event": state.get("event"),
        "store": state,
        "runtime_identity": runtime_identity,
        "focused_tests": tests,
        "cpu_blast": regional.cpu_blast_snapshot(),
        "predecessor": predecessor,
        "control_env": env,
    }
    result["errors"] = verdicts.preflight_errors(
        nodes=settings.nodes,
        node_snapshots=snapshots,
        agent=state.get("agent") or {},
        profile=state.get("profile"),
        queue=state.get("queue") or {},
        remote_commands=state.get("remote_commands") or {},
        gpu_workloads=regional.gpu_workloads(),
        business_workloads=workloads,
        recent_event=state.get("event"),
        predecessor=predecessor,
        tests=tests,
        control_env=env,
        runtime_identity_errors=runtime_identity_errors(runtime_identity),
        gpu_identity=state.get("gpu_identity"),
        arbiter_pods=arbiter_pods,
        dns_nodes=dns_nodes,
    )
    write_json_atomic(case_dir / "preflight.json", result)
    return result


def parser(*, confirmation: str = CONFIRMATION) -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Run the guarded DESTR-026 acceptance: two GPUs of one node fault "
            "inside the aggregation window, one window resets both, and the "
            "training job is restarted once."
        )
    )
    add_live_arguments(value, confirmation=confirmation)
    value.add_argument("--cpu-kubeconfig", default="")
    value.add_argument("--gpu-kubeconfig", default="")
    value.add_argument("--gpu-context", default="")
    value.add_argument("--namespace", default="gpu-fault-system")
    value.add_argument("--cluster-id", default="")
    value.add_argument("--region", default="")
    value.add_argument("--site-file", default="")
    value.add_argument("--manifest", type=Path, default=join_case.DEFAULT_MANIFEST)
    value.add_argument("--host-probe-image", default="")
    value.add_argument(
        "--fault-node",
        default="",
        help="node whose two GPUs fault; the job's Master replica is pinned to it",
    )
    value.add_argument(
        "--node-b",
        default="",
        help="node the job's Worker replica is pinned to; never faulted",
    )
    value.add_argument(
        "--pci-bdf-a",
        default="",
        help="first GPU PCI BDF to fault; defaults to the first inventory entry",
    )
    value.add_argument(
        "--pci-bdf-b",
        default="",
        help="second GPU PCI BDF to fault; defaults to the next inventory entry",
    )
    value.add_argument("--job-id", default="")
    value.add_argument("--attempt-id", default="")
    value.add_argument("--predecessor-evidence", default="")
    return value


# --------------------------------------------------------------------------- #
# Live execution
# --------------------------------------------------------------------------- #
def _host_probe(
    settings: Settings, node: str, run_id: str, case_dir: Path
) -> HostProbeFixture:
    return HostProbeFixture(
        HostProbeSettings(
            state_directory=case_dir / "host-probes",
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
    """Everything one execution owns, and the flags cleanup reads.

    ``probes``: ``a`` and ``b`` on the fault node (one kmsg write each, so the
    two writes run concurrently without sharing a probe; ``a`` also runs the
    sampler and the host reads), ``partner`` on node B (reads only).
    """

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
    markers: dict[str, str] = field(default_factory=dict)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    injected_at: dict[str, str] = field(default_factory=dict)
    bdfs: dict[str, str] = field(default_factory=dict)
    targets: dict[str, str] = field(default_factory=dict)
    baselines: dict[str, dict[str, Any]] = field(default_factory=dict)
    nodes_before: list[dict[str, Any]] = field(default_factory=list)
    source_uids: set[str] = field(default_factory=set)
    incident_id: str = ""
    workflow_request_id: str = ""
    workload_submitted: bool = False
    sampler_started_at: datetime | None = None
    sampler_stopped: bool = False

    @property
    def target_list(self) -> list[str]:
        return [self.targets[label] for label in LABELS]

    def window_open(self, stage: str) -> None:
        if datetime.now(timezone.utc) >= self.maintenance_window_end:
            raise RegionalFixtureError(f"maintenance window ended before {stage}")


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
    current = join_case.identity_digest(
        join_case.plan_identity(preflight, nodes=settings.nodes)
    )
    if planned["details"]["preflight_identity_digest"] != current:
        raise RegionalFixtureError("DESTR-026 plan identity drifted before execution")
    regional = RegionalLiveFixture(settings.regional)
    run_id = run_identity(run_dir, attempt)
    pinned = join_case.render_two_node_manifest(
        settings.manifest,
        case_dir / "pinned-workload.yaml",
        master_node=settings.fault_node,
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
        probes={
            "a": _host_probe(settings, settings.fault_node, run_id, case_dir),
            "b": _host_probe(settings, settings.fault_node, f"{run_id}-b", case_dir),
            "partner": _host_probe(settings, settings.node_b, run_id, case_dir),
        },
    )


def _host_baselines(run: _LiveRun) -> None:
    for probe in run.probes.values():
        probe.create()
    fault = run.probes["a"].execute("snapshot", "--run-id", run.run_id)
    partner = run.probes["partner"].execute("snapshot")
    for node, baseline in (
        (run.settings.fault_node, fault),
        (run.settings.node_b, partner),
    ):
        if len(baseline.get("gpu_inventory") or []) != verdicts.GPUS_PER_NODE:
            raise RegionalFixtureError(
                f"{node} host GPU inventory is not {verdicts.GPUS_PER_NODE}"
            )
        if baseline.get("quiesce_states"):
            raise RegionalFixtureError(f"{node} has a pre-existing quiesce state")
    if not fault.get("kmsg_writable"):
        raise RegionalFixtureError("/dev/kmsg is not writable from the host probe")
    run.baselines = {"fault": fault, "partner": partner}
    write_json_atomic(run.case_dir / "host-baseline-fault.json", fault)
    write_json_atomic(run.case_dir / "host-baseline-node-b.json", partner)


def _start_job_and_probes(run: _LiveRun) -> None:
    settings, case_dir = run.settings, run.case_dir
    run.window_open("image prewarm")
    run.prewarm.create(list(settings.nodes))
    cached = run.prewarm.cached_nodes()
    write_json_atomic(case_dir / "image-cache.json", {"cached_nodes": cached})
    if not set(settings.nodes) <= set(cached):
        raise RegionalFixtureError("training image is not cached on both nodes")
    run.window_open("workload submission")
    run.workload_submitted = True
    write_json_atomic(case_dir / "submission.json", run.workload.submit())
    source = run.workload.wait_running(timeout_seconds=900)
    run.source_uids = {str(item["uid"]) for item in source["pods"]}
    write_json_atomic(case_dir / "workload-source.json", source)
    observation = workload_case.wait_observation(
        run.regional,
        settings,
        node=settings.fault_node,
        expected_gpu_count=EXPECTED_GPU_COUNT,
    )
    write_json_atomic(case_dir / "source-observation.json", observation)
    allocation = {
        str(gpu)
        for container in observation.get("containers") or []
        if container.get("node_id") == settings.fault_node
        for gpu in container.get("gpu_uuids") or []
    }
    _host_baselines(run)
    try:
        chosen = verdicts.target_gpus(
            run.baselines["fault"]["gpu_inventory"],
            pci_bdfs=(settings.pci_bdf_a, settings.pci_bdf_b),
            allocation=allocation,
        )
    except ValueError as exc:
        raise RegionalFixtureError(f"target GPUs: {exc}") from exc
    run.bdfs = dict(zip(LABELS, chosen, strict=True))
    run.targets = {label: chosen[bdf] for label, bdf in run.bdfs.items()}
    run.nodes_before = run.regional.gpu_nodes()
    write_json_atomic(
        case_dir / "targets.json",
        {"bdfs": run.bdfs, "targets": run.targets, "gpu_nodes": run.nodes_before},
    )
    run.regional.verify_runtime_identity(
        run.preflight["runtime_identity"],
        evidence_path=case_dir / "runtime-identity-before-injection.json",
        stage="before DESTR-026 injection",
    )


def _inject_pair(run: _LiveRun) -> dict[str, dict[str, Any]]:
    """Start the sampler, write both XIDs at once, wait for both events."""

    settings, case_dir = run.settings, run.case_dir
    run.window_open("sampler start")
    # Armed before the call: a start that times out after creating the unit
    # must still be stopped by cleanup (the stop is idempotent).
    run.sampler_started_at = datetime.now(timezone.utc)
    sampler = run.probes["a"].execute(
        "start-reset-sampler",
        "--run-id",
        run.run_id,
        "--probe-script",
        run.probes["a"].host_script,
        timeout=60,
    )
    write_json_atomic(case_dir / "sampler-start.json", sampler)
    run.window_open("injection")
    base = f"destr026-{int(time.time())}-a{run.attempt}"
    run.markers = {label: f"{base}-g{label}" for label in LABELS}
    run.started_at = datetime.now(timezone.utc)

    def write(label: str) -> dict[str, Any]:
        payload = run.probes[label].execute(
            "write-xid46",
            "--marker",
            run.markers[label],
            "--drill-id",
            f"{run.run_id}-g{label}",
            "--pci-bdf",
            run.bdfs[label],
            "--maintenance-window-end",
            run.maintenance_window_end.isoformat(),
        )
        run.injected_at[label] = datetime.now(timezone.utc).isoformat()
        return payload

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="destr026-xid") as pool:
        futures = {label: pool.submit(write, label) for label in LABELS}
        writes = {label: future.result() for label, future in futures.items()}
    write_json_atomic(
        case_dir / "injection.json",
        {"markers": run.markers, "injected_at": run.injected_at, "writes": writes},
    )
    states = {}
    for label in LABELS:
        (case_dir / f"gpu-{label}").mkdir(mode=0o700, parents=True, exist_ok=True)
        states[label] = run.regional.wait_for_workflow(
            node=settings.fault_node,
            marker=run.markers[label],
            observed_after=run.started_at,
            case_dir=case_dir / f"gpu-{label}",
            timeout_seconds=EVENT_WAIT_SECONDS,
            job_id=settings.job_id,
            attempt_id=settings.attempt_id,
            terminal=False,
        )
    first = states["a"]
    run.workflow_request_id = str((first.get("workflow") or {}).get("request_id") or "")
    run.incident_id = str((first.get("incident") or {}).get("incident_id") or "")
    return states


def _observe_until_terminal(run: _LiveRun) -> dict[str, Any]:
    """Store reads only: the fault node may be quiesced and unreachable."""

    settings = run.settings
    transitions: dict[str, str] = {}
    timeline: list[dict[str, Any]] = []
    deadline = time.monotonic() + OBSERVATION_BUDGET_SECONDS
    state: dict[str, Any] = {}
    while time.monotonic() < deadline:
        state = run.regional.store_snapshot(
            node=settings.fault_node,
            marker=run.markers["a"],
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


def _fault_node_answers(run: _LiveRun) -> dict[str, Any]:
    """Wait for the fault node to answer before any exec into it; the
    sampler unit is the only detached thing this case leaves there."""

    baseline = (
        run.baselines.get("fault") or run.preflight["nodes"][run.settings.fault_node]
    )
    reach = host_reachability(
        run.regional,
        node=run.settings.fault_node,
        baseline_boot_id=baseline.get("boot_id"),
        holder_armed_at=run.sampler_started_at,
        max_hold_seconds=SAMPLER_LIFETIME_SECONDS,
        budget_seconds=HOST_RETURN_BUDGET_SECONDS,
    )
    write_json_atomic(run.case_dir / "fault-node-return.json", reach)
    return reach


def _stop_sampler(run: _LiveRun) -> dict[str, Any]:
    stopped = run.probes["a"].execute(
        "stop-reset-sampler", "--run-id", run.run_id, timeout=60
    )
    run.sampler_stopped = True
    write_json_atomic(run.case_dir / "sampler-final.json", stopped)
    return stopped


def _restarted_workload(run: _LiveRun, state: dict[str, Any]) -> dict[str, Any]:
    """The job after the workflow: replaced once and then left alone."""

    if (state.get("workflow") or {}).get("status") != "SUCCEEDED":
        # A workflow that did not succeed owes no restart.
        return {"pods": run.workload.pods()}
    run.workload.authorize_restart(state)
    target = run.workload.wait_restarted(run.source_uids, timeout_seconds=900)
    uids = {str(item["uid"]) for item in target["pods"]}
    stable = run.workload.wait_pod_uids_unchanged(
        uids, timeout_seconds=RESTART_STABLE_SECONDS
    )
    return {**target, "pods_after_quiet_period": stable["pods"]}


def _job_attempts(run: _LiveRun, pods: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The store's attempts of the job once the restarted one is among them
    (the watcher reports it on its next pass), or as they are at the bound."""

    wanted = {str(pod.get("attempt_id") or "") for pod in pods}
    wanted -= {"", run.settings.attempt_id}
    deadline = time.monotonic() + ATTEMPT_OBSERVATION_SECONDS
    while True:
        observations = list(
            run.regional.store_snapshot(
                node=run.settings.fault_node,
                job_id=run.settings.job_id,
                queue_attempts=1,
            ).get("observations")
            or []
        )
        known = {str(item.get("attempt_id")) for item in observations}
        if wanted <= known or time.monotonic() >= deadline:
            return observations
        time.sleep(10)


def _host_after(run: _LiveRun) -> dict[str, dict[str, Any]]:
    since = str(run.started_at.timestamp())
    fault = run.probes["a"].execute(
        "snapshot",
        "--since-epoch",
        since,
        "--pci-bdf",
        run.bdfs["a"],
        "--run-id",
        run.run_id,
        timeout=180,
    )
    # The sampler's job ends with the snapshot that read its series.
    _stop_sampler(run)
    second = run.probes["b"].execute(
        "snapshot", "--since-epoch", since, "--pci-bdf", run.bdfs["b"], timeout=180
    )
    partner = run.probes["partner"].execute("snapshot", timeout=180)
    hosts = {"fault": fault, "fault_second_bdf": second, "partner": partner}
    for name, value in hosts.items():
        write_json_atomic(run.case_dir / f"host-after-{name}.json", value)
    return hosts


def _node_errors(
    run: _LiveRun, workflow: dict[str, Any], hosts: dict[str, dict[str, Any]]
) -> list[str]:
    """Everything the two hosts say, graded by the pure verdicts."""

    settings = run.settings
    fault, before = hosts["fault"], run.baselines["fault"]
    errors = host_errors(
        {settings.fault_node: {"before": before, "after": fault}},
        nodes=(settings.fault_node,),
        expected_gpu_count=verdicts.GPUS_PER_NODE,
    )
    for snapshot, bdf in (
        (fault, run.bdfs["a"]),
        (hosts["fault_second_bdf"], run.bdfs["b"]),
    ):
        journal = snapshot.get("kernel_reset_journal") or {}
        if int(journal.get("target_reset_count") or 0) > 1:
            errors.append(f"kernel journal shows more than one reset for {bdf}")
    ledger, rows = verdicts.ledger_errors(
        before,
        fault,
        targets=run.target_list,
        incident_id=run.incident_id,
        workflow_request_id=run.workflow_request_id,
    )
    errors.extend(ledger)
    control = verdicts.node_receipt(workflow, "RESET_GPU", settings.fault_node)
    recorded = (rows.get("RESET_GPU") or {}).get("evidence") or {}
    errors.extend(
        verdicts.reset_receipt_errors(
            control, targets=run.target_list, source="the control plane's RESET_GPU"
        )
    )
    errors.extend(
        verdicts.reset_receipt_errors(
            recorded,
            targets=run.target_list,
            source="the Node Agent ledger's RESET_GPU",
        )
    )
    errors.extend(verdicts.relay_errors(control, recorded))
    errors.extend(
        verdicts.fabric_manager_errors(
            before, fault, workflow=workflow, ledger_rows=rows, node=settings.fault_node
        )
    )
    errors.extend(
        verdicts.sampler_errors(
            before.get("gpu_inventory") or [],
            fault.get("sampler"),
            targets=run.target_list,
        )
    )
    errors.extend(
        verdicts.partner_host_errors(
            run.baselines["partner"], hosts["partner"], node=settings.node_b
        )
    )
    return errors


def _data_plane_errors(
    run: _LiveRun, state: dict[str, Any]
) -> tuple[list[str], dict[str, Any]]:
    settings, case_dir = run.settings, run.case_dir
    target = _restarted_workload(run, state)
    write_json_atomic(case_dir / "workload-target.json", target)
    errors = workload_errors(
        pods=target["pods"], source_uids=run.source_uids, nodes=settings.nodes
    )
    observations = _job_attempts(run, target["pods"])
    write_json_atomic(case_dir / "job-attempts.json", {"observations": observations})
    errors.extend(
        verdicts.attempt_errors(
            source_attempt_id=settings.attempt_id,
            pods=target["pods"],
            observations=observations,
        )
    )
    reach = _fault_node_answers(run)
    if not reach.get("exec_allowed"):
        raise RegionalFixtureError(
            f"{settings.fault_node} did not answer again ({reach.get('reason')}); "
            "no host read is attempted on a NotReady node"
        )
    for probe in run.probes.values():
        probe.create()
    hosts = _host_after(run)
    errors.extend(_node_errors(run, state.get("workflow") or {}, hosts))
    snapshots = {node: run.regional.node_snapshot(node) for node in settings.nodes}
    write_json_atomic(case_dir / "nodes-after.json", snapshots)
    errors.extend(schedulability_errors(snapshots, nodes=settings.nodes))
    nodes_after = run.regional.gpu_nodes()
    write_json_atomic(case_dir / "gpu-nodes-after.json", {"nodes": nodes_after})
    errors.extend(
        verdicts.other_node_errors(
            run.nodes_before, nodes_after, fault_node=settings.fault_node
        )
    )
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
    errors.extend(workload_case.log_write_errors(logs, "control-plane"))
    cpu_after = run.regional.cpu_blast_snapshot()
    write_json_atomic(case_dir / "cpu-blast-after.json", cpu_after)
    if cpu_after != run.preflight["cpu_blast"]:
        errors.append("control-plane EKS state differs from baseline")
    run.regional.verify_runtime_identity(
        run.preflight["runtime_identity"],
        evidence_path=case_dir / "runtime-identity-after-restart.json",
        stage="after DESTR-026 workload restart",
    )
    summary = {
        "boot_ids": {
            "fault": [
                run.baselines["fault"].get("boot_id"),
                hosts["fault"].get("boot_id"),
            ],
            "node_b": [
                run.baselines["partner"].get("boot_id"),
                hosts["partner"].get("boot_id"),
            ],
        },
        "target_pods": sorted(str(item.get("uid")) for item in target["pods"]),
        "provider_events": {"count": len(provider), "provisional": provisional},
        "fault_node_return": reach,
    }
    return errors, summary


def _workflow_verdicts(run: _LiveRun, state: dict[str, Any]) -> list[str]:
    workflow = state.get("workflow") or {}
    incident = state.get("incident") or {}
    run.incident_id = str(incident.get("incident_id") or run.incident_id)
    return [
        # DESTR-001's reset contract: XID, kmsg, decision, the ten-step chain,
        # the remote WAITING delegation and every remote step SUCCEEDED.
        *reset_case.workflow_errors(
            state, expected_steps=verdicts.WORKLOAD_RESET_STEPS
        ),
        *verdicts.workflow_errors(
            workflow,
            incident,
            node=run.settings.fault_node,
            partner=run.settings.node_b,
            targets=run.target_list,
            expected_gpu_count=EXPECTED_GPU_COUNT,
            restart_budget=state.get("restart_budget") or {},
        ),
    ]


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
        run.window_open("case setup")
        _start_job_and_probes(run)
        states = _inject_pair(run)
        errors = verdicts.injection_errors(
            states, node=settings.fault_node, targets=run.targets
        )
        observed = event_observed_at(states)
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
        result["errors"] = list(errors)
        # Two faults that did not form one in-window workflow cannot prove the
        # single window; the plan lists that as a stop condition.
        join_case.stop_before_observation(errors)
        state = _observe_until_terminal(run)
        errors.extend(_workflow_verdicts(run, state))
        data_errors, summary = _data_plane_errors(run, state)
        errors.extend(data_errors)
        details = {
            "preflight_identity": join_case.plan_identity(
                run.preflight, nodes=settings.nodes
            ),
            "workflow": state.get("workflow") or {},
            "incident": state.get("incident") or {},
            "hosts": summary,
        }
        components = join_case.evidence_components(details)
        result.update(
            {
                **run.regional.evidence_identity(),
                "verdict": "PASS" if not errors else "FAIL",
                "errors": errors,
                "markers": run.markers,
                "targets": run.targets,
                "pci_bdfs": run.bdfs,
                "incident_id": run.incident_id,
                "workflow_request_id": run.workflow_request_id,
                "injected_at": run.injected_at,
                "event_observed_at": observed,
                "restart_budget": state.get("restart_budget"),
                "summary": summary,
                "case_digest": join_case.case_digest(components),
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


# --------------------------------------------------------------------------- #
# Cleanup
# --------------------------------------------------------------------------- #
def _cleanup(run: _LiveRun) -> dict[str, Any]:
    """Quiescence, workload, validated restore; then the host side, but only
    once the fault node answers again; then the prewarm Pods and the identity
    recheck. A fault node that stays NotReady is never exec'd: its probes and
    the sampler unit are left for manual reconciliation and the case FAILs."""

    settings = run.settings
    result: dict[str, Any] = {"errors": []}

    def guard(label: str, action: Any) -> None:
        try:
            result[label] = action()
        except Exception as exc:  # noqa: BLE001 - a cleanup failure is a FAIL
            result["errors"].append(f"{label}: {type(exc).__name__}: {exc}")

    if run.markers:
        guard(
            "quiescence",
            lambda: workload_case.wait_for_cleanup_quiescence(
                regional=run.regional,
                node=settings.fault_node,
                marker=run.markers["a"],
                observed_after=run.started_at,
                job_id=settings.job_id,
                attempt_id=settings.attempt_id,
                case_dir=run.case_dir,
            ),
        )
    quiescent = "quiescence" in result or not run.markers
    if run.workload_submitted and quiescent:
        guard("workload_delete", run.workload.delete)
        guard("workload_residual", lambda: _refuse_residual(run))
    elif run.workload_submitted:
        result["workload_cleanup_deferred"] = True
    if quiescent:
        guard("restore_isolated_nodes", lambda: _restore_isolated_nodes(run))
    else:
        result["isolation_restore_deferred"] = True
    fault_answers = True
    if run.sampler_started_at is not None and not run.sampler_stopped:
        reach = _fault_node_answers(run)
        result["fault_node_return"] = reach
        fault_answers = bool(reach.get("exec_allowed"))
        if fault_answers:
            guard("sampler_stop", lambda: _stop_sampler(run))
        else:
            result["errors"].append(
                f"{settings.fault_node} never answered again ({reach.get('reason')}); "
                "the sampler unit and the probes there are left for manual "
                "reconciliation, no exec is attempted on a NotReady node"
            )
    for label, probe in run.probes.items():
        if label == "partner" or fault_answers:
            guard(
                f"probe_cleanup_{label}",
                lambda probe=probe: _refuse_residual_map(probe.cleanup()),
            )
    guard("prewarm_cleanup", lambda: _refuse_residual_map(run.prewarm.cleanup()))
    guard(
        "runtime_identity",
        lambda: run.regional.verify_runtime_identity(
            run.preflight["runtime_identity"],
            evidence_path=run.case_dir / "runtime-identity-after-cleanup.json",
            stage="after DESTR-026 cleanup",
        ),
    )
    return result


def _refuse_residual(run: _LiveRun) -> dict[str, Any]:
    name = run.regional.kubectl(
        "gpu",
        "get",
        run.workload.resource,
        run.workload.name,
        "--ignore-not-found",
        "-o",
        "name",
        check=True,
    ).strip()
    if name:
        raise RegionalFixtureError(f"test workload still exists: {name}")
    return {"residual": False}


def _refuse_residual_map(residuals: dict[str, bool]) -> dict[str, bool]:
    if any(residuals.values()):
        raise RegionalFixtureError(f"residual resources remain: {residuals}")
    return residuals


def _restore_isolated_nodes(run: _LiveRun) -> dict[str, Any]:
    """A workflow that failed leaves its node cordoned and owned. Never delete
    the taint by hand: restore through the validation-first workflow."""

    report: dict[str, Any] = {}
    profile_version = str(
        (run.preflight["store"].get("profile") or {}).get("profile_version") or ""
    )
    warm = WarmSpareLiveFixture(run.regional, "")
    for node in run.settings.nodes:
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
            reason="DESTR-026 validated cleanup",
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
