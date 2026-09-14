#!/usr/bin/env python3
"""GF-REGIONAL-DESTR-016 live acceptance runner.

One idle GPU node, three real ``/dev/kmsg`` writes, one incident.

1. XID 46 opens a RESET_GPU workflow. A GPU device holder, armed to start the
   moment this drill's ``QUIESCE_GPU_SERVICES`` lands in the Node Agent ledger,
   keeps a compute client alive, so ``VERIFY_NO_GPU_CLIENTS`` parks the
   workflow at a WAITING remote command with the quiesce applied and not yet
   restored. That is the dirty boundary in the middle of a physical step.
2. A second XID 46 -- same recovery rank -- is merged onto the same incident and
   the same workflow: no second workflow, no extra step, one more reason.
3. An XID 79 -- RESTART_NODE, recovery rank 50 against RESET_GPU's 30 --
   preempts it. The reset workflow goes SUPERSEDED and its WAITING remote
   command is cancelled; the successor reboot workflow inherits the cordon,
   adopts the quiesce handoff (an appended ``RESTORE_GPU_SERVICES`` carrying
   ``preemption_quiesce_handoff_after_reboot``), reboots the node through
   HyperPod for real, restores GPU services *after* the reboot, validates GPU,
   host and fabric, and restores scheduling. The incident ends RECOVERED and no
   RESET_GPU ever ran on the node.

DESTR-002 proves the real reboot from a clean start; PREEMPT-012 proves the
handoff boundaries against an audited in-process executor. This case is the
first live evidence that a real physical step in flight can be preempted and
its quiesce handed to the stronger recovery.

The runner defaults to ``--plan``; ``--execute`` needs ``--confirm
DESTR016_EXECUTE``. The verdict functions are pure and unit-tested; the
runner records digests only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e2e.regional import (  # noqa: E402
    run_destr002_hyperpod_reboot as reboot_case,
)
from scripts.e2e.regional.acceptance_runner_common import (  # noqa: E402
    processor_queue_backlog,
    write_json_atomic,
)
from scripts.e2e.regional.control_plane_env_window import (  # noqa: E402
    observed_value,
    replica_env,
)
from scripts.e2e.regional.destr016_verdicts import (  # noqa: E402
    AGENT_OPERATIONS,
    FIRST_XID,
    absorb_errors,
    barrier_reason_errors,
    cancelled_command_errors,
    escalation_errors,
    estimated_duration_seconds,
    holder_errors,
    host_errors,
    incident_reason_errors,
    lifetime_errors,
    node_recovery_errors,
    provider_errors,
    reboot_events,
    reset_workflow_errors,
    restore_after_reboot_errors,
    schedulability_errors,
    step_timeout_errors,
    step_transitions,
    successor_step_graph_errors,
    superseded_predecessor_errors,
    terminal_errors,
    waiting_boundary_errors,
)
from scripts.e2e.regional.host_probe_fixture import (  # noqa: E402
    HostProbeFixture,
    HostProbeSettings,
)
from scripts.e2e.regional.live_driver_guard import (  # noqa: E402
    CaseRunner,
    add_live_arguments,
    run_standard_case,
)
from scripts.e2e.regional.regional_live_fixture import (  # noqa: E402
    RegionalFixtureError,
    RegionalLiveFixture,
    RegionalLiveSettings,
    predecessor_evidence,
    provider_event_actor_matches_role,
    required,
    run_case_main,
    runtime_identity_errors,
    settings_from_arguments,
)
from scripts.e2e.regional.warm_spare_fixture import (  # noqa: E402
    WORKFLOW_BY_ID,
    WarmSpareLiveFixture,
)

HOLDER_PROBE = Path(__file__).with_name("probes") / "destr016_node_probe.py"
INJECT_PROBE = Path(__file__).with_name("probes") / "destructive_node_probe.py"
CASE_ID = "GF-REGIONAL-DESTR-016"
PREDECESSOR_CASE_ID = "GF-REGIONAL-DESTR-002"
CONFIRMATION = "DESTR016_EXECUTE"
CONTROL_PLANE = "cpu"
CONTROL_DEPLOYMENT = "gpu-fault-control-worker"
STEP_TIMEOUT_VARIABLE = "GPU_FAULT_WORKFLOW_STEP_TIMEOUT_SECONDS"
NODE_LIFETIME_VARIABLE = "GPU_FAULT_NODE_WORKFLOW_MAX_LIFETIME_SECONDS"
CONTROL_VARIABLES = (STEP_TIMEOUT_VARIABLE, NODE_LIFETIME_VARIABLE)
# Shipped defaults of the two bounds, used only when the deployment does not
# override them (``execution/config.py``: ``from_mapping``).
DEFAULT_STEP_TIMEOUT_SECONDS = 600
DEFAULT_NODE_LIFETIME_SECONDS = 3600
# Phase budgets. The barrier park is the scarce one: it lives inside the step's
# own waiting ceiling, and both later injections have to land inside it.
BARRIER_WAIT_BUDGET_SECONDS = 600
ABSORB_BUDGET_SECONDS = 300
PREEMPTION_BUDGET_SECONDS = 420
# QUIESCE_GPU_SERVICES stops kubelet, and with it the exec channel the host
# probes answer over, so the two writes that must land inside the WAITING
# window are scheduled on the node when the holder is armed: this many seconds
# after the holder starts (i.e. after the quiesce row landed in the ledger).
ABSORB_DELAY_SECONDS = 90
ESCALATE_DELAY_SECONDS = 240
# Supersession happens when the predecessor's executor next reaches a step
# boundary, which can take one lease/reclaim interval after the successor was
# written (``execution/dispatcher.py``: ``_eligible`` holds the successor while
# the predecessor is open).
SUPERSEDE_BUDGET_SECONDS = 900
HANDOFF_BUDGET_SECONDS = 900
REBOOT_BUDGET_SECONDS = 1800
OBSERVATION_BUDGET_SECONDS = 2400
# The maintenance window must have at least this much left at injection time.
MIN_WINDOW_REMAINING_SECONDS = 3600

COMMANDS_BY_WORKFLOW = r"""
import json
import sys

from gpu_fault.app import ApplicationContext

request_id = sys.argv[1]
store = ApplicationContext.from_environment().store
commands = [
    item.model_dump(mode="json")
    for item in store.list_remote_commands()
    if item.workflow_request_id == request_id
]
print(json.dumps(
    {"request_id": request_id, "commands": commands},
    sort_keys=True,
    default=str,
))
"""


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested)
# --------------------------------------------------------------------------- #
def preflight_errors(
    *,
    node: str,
    node_snapshot: dict[str, Any],
    agent: dict[str, Any],
    profile: dict[str, Any] | None,
    queue: dict[str, Any],
    remote_commands: dict[str, Any],
    gpu_workloads: list[dict[str, Any]],
    business_workloads: list[dict[str, Any]],
    event: dict[str, Any] | None,
    predecessor: dict[str, Any],
    tests: dict[str, Any],
    control_env: dict[str, Any],
    identity_errors: list[str],
) -> list[str]:
    """Everything that must hold before a real reboot is authorized."""

    errors: list[str] = list(identity_errors)
    if not predecessor.get("valid"):
        errors.append(f"{PREDECESSOR_CASE_ID} predecessor evidence is not PASS")
    if not tests.get("passed"):
        errors.append("focused regression tests failed")
    if (
        node_snapshot.get("ready") != "True"
        or node_snapshot.get("unschedulable")
        or node_snapshot.get("taints")
        or node_snapshot.get("ownership_annotations")
    ):
        errors.append(f"{node} is not Ready, schedulable, untainted and unowned")
    if int(node_snapshot.get("gpu_allocatable") or 0) < 1:
        errors.append(f"{node} allocates no GPU")
    if agent.get("lifecycle_state") != "ACTIVE":
        errors.append(f"{node} Node Agent is not ACTIVE")
    missing = sorted(set(AGENT_OPERATIONS) - set(agent.get("allowed_operations") or []))
    if missing:
        errors.append(f"{node} Node Agent does not allow {missing}")
    if business_workloads:
        errors.append(f"{node} carries a non-system workload: {business_workloads}")
    if gpu_workloads:
        errors.append(f"the GPU cluster already has a GPU workload: {gpu_workloads}")
    if event is not None:
        errors.append(f"{node} has a recent XID event")
    reset = reboot_case.capability(profile, "gpuReset")
    if reset is None:
        errors.append("the runtime profile has no gpuReset capability")
    elif reset.get("mode") != "OWN" or reset.get("owner") != "gpu-fault-node-agent":
        errors.append("gpuReset is not OWN by the Node Agent")
    reboot = reboot_case.capability(profile, "nodeReboot")
    if reboot is None:
        errors.append("the runtime profile has no nodeReboot capability")
    elif (
        reboot.get("mode") != "OWN"
        or reboot.get("owner") != "gpu-fault-hyperpod-adapter"
        or reboot.get("adapter") != "regional-cluster-executor"
    ):
        errors.append("nodeReboot is not OWN by the HyperPod adapter")
    if (profile or {}).get("warnings"):
        errors.append("the runtime profile has warnings")
    if processor_queue_backlog(queue):
        errors.append("the processor queue is not empty")
    if remote_commands.get("open_by_cluster"):
        errors.append("the remote command queue is not empty")
    if not control_env.get("preemption_enabled", True):
        errors.append("workflow preemption is disabled on the control plane")
    errors.extend(
        step_timeout_errors(
            step_timeout_seconds=control_env.get("step_timeout_seconds")
        )
    )
    errors.extend(
        lifetime_errors(
            estimated_seconds=estimated_duration_seconds(),
            lifetime_seconds=control_env.get("node_lifetime_seconds"),
        )
    )
    return errors


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str).encode()
    ).hexdigest()


def plan_identity(preflight: dict[str, Any], *, node: str) -> dict[str, Any]:
    snapshot = preflight.get("node") or {}
    store = preflight.get("store") or {}
    return {
        "release_id": preflight.get("release_id"),
        "node": node,
        "node_uid": snapshot.get("uid"),
        "node_boot_id": snapshot.get("boot_id"),
        "runtime_profile_version": (store.get("profile") or {}).get("profile_version"),
        "runtime_identity": preflight.get("runtime_identity"),
    }


def identity_digest(identity: dict[str, Any]) -> str:
    return _digest(identity)


def evidence_components(details: dict[str, Any]) -> dict[str, str]:
    return {name: _digest(value) for name, value in details.items()}


def case_digest(components: dict[str, str]) -> str:
    return _digest(components)


def control_env_record(observations: dict[str, str | None]) -> dict[str, Any]:
    """Fold the values every ready control-worker replica agrees on into the
    two bounds the case does arithmetic against.

    A value no replica agrees on (mid-rollout) is reported as the shipped
    default rather than averaged: the preflight then measures against what the
    code would use if the override were absent, and a rollout in progress is
    caught by the runtime-identity check instead.
    """

    def number(name: str, default: int) -> int:
        value = observations.get(name)
        try:
            return int(str(value)) if value is not None else default
        except ValueError:
            return default

    return {
        "observed": dict(observations),
        "step_timeout_seconds": number(
            STEP_TIMEOUT_VARIABLE, DEFAULT_STEP_TIMEOUT_SECONDS
        ),
        "node_lifetime_seconds": number(
            NODE_LIFETIME_VARIABLE, DEFAULT_NODE_LIFETIME_SECONDS
        ),
        "preemption_enabled": True,
    }


def window_errors(
    *,
    now: datetime,
    maintenance_window_end: datetime,
    required_seconds: int = MIN_WINDOW_REMAINING_SECONDS,
) -> list[str]:
    remaining = (maintenance_window_end - now).total_seconds()
    if remaining < required_seconds:
        return [
            f"the maintenance window has {int(remaining)}s left; a real reboot "
            f"needs at least {required_seconds}s"
        ]
    return []


def markers(base: str) -> dict[str, str]:
    """One marker per injection.

    The store probe resolves the *last* matching event, so three writes sharing
    one marker would make every snapshot read the third event. Distinct markers
    keep each phase's observation pinned to its own fault, while the incident
    they all merge into is reached through ``incident_by_event``.
    """

    return {
        "reset": f"{base}-r",
        "absorb": f"{base}-s",
        "escalate": f"{base}-e",
    }


def target_bdf(explicit: str, inventory: list[dict[str, Any]]) -> str:
    if explicit:
        return explicit
    if not inventory:
        raise RegionalFixtureError("the host probe found no GPU inventory")
    return str(inventory[0]["pci_bdf"])


def holder_device(explicit: str, inventory: list[dict[str, Any]]) -> str:
    """The ``/dev/nvidiaN`` node the holder opens.

    Derived from the inventory index of the faulted GPU when not given, so the
    client the barrier step sees is a client of the GPU the case faulted rather
    than of an arbitrary device on the node.
    """

    if explicit:
        return explicit
    if not inventory:
        raise RegionalFixtureError("the host probe found no GPU inventory")
    index = inventory[0].get("index")
    return f"/dev/nvidia{int(index) if index is not None else 0}"


# --------------------------------------------------------------------------- #
# Settings / plan
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Settings:
    regional: RegionalLiveSettings
    node: str
    host_probe_image: str
    hyperpod_cluster: str
    executor_role_arn: str
    pci_bdf: str
    device: str
    max_hold_seconds: int
    predecessor_path: Path

    def environment(self) -> dict[str, str]:
        return {
            **self.regional.environment(),
            "GPU_FAULT_TARGET_NODE": self.node,
            "GPU_FAULT_HOST_PROBE_IMAGE": self.host_probe_image,
            "GPU_FAULT_HYPERPOD_CLUSTER_NAME": self.hyperpod_cluster,
            "GPU_FAULT_EXECUTOR_ROLE_ARN": self.executor_role_arn,
            "GPU_FAULT_PREDECESSOR_EVIDENCE": str(self.predecessor_path),
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
        node=required(
            arguments.node or os.getenv("GPU_FAULT_TARGET_NODE", ""), "target node"
        ),
        host_probe_image=required(
            arguments.host_probe_image or os.getenv("GPU_FAULT_HOST_PROBE_IMAGE", ""),
            "host probe image",
        ),
        hyperpod_cluster=required(
            arguments.hyperpod_cluster
            or os.getenv("GPU_FAULT_HYPERPOD_CLUSTER_NAME", ""),
            "HyperPod cluster name",
        ),
        executor_role_arn=required(
            arguments.executor_role_arn or os.getenv("GPU_FAULT_EXECUTOR_ROLE_ARN", ""),
            "executor role ARN",
        ),
        pci_bdf=arguments.pci_bdf.strip(),
        device=arguments.holder_device.strip(),
        max_hold_seconds=int(arguments.max_hold_seconds),
        predecessor_path=predecessor,
    )


def plan_details(settings: Settings, preflight: dict[str, Any]) -> dict[str, Any]:
    identity = plan_identity(preflight, node=settings.node)
    return {
        "risk": "destructive-provider-reboot",
        "predecessor": preflight.get("predecessor"),
        "node": settings.node,
        "hyperpod_cluster": settings.hyperpod_cluster,
        "control_env": preflight.get("control_env"),
        "mutation": (
            "write one XID 46 to /dev/kmsg on one idle GPU node and hold a GPU "
            "device open so the reset workflow parks at VERIFY_NO_GPU_CLIENTS "
            "with GPU services quiesced; write a second XID 46 that must merge "
            "into the same workflow; write one XID 79 that must preempt it. The "
            "successor workflow reboots the node through HyperPod for real "
            "(BatchRebootClusterNodes, once), restores GPU services after the "
            "reboot, validates GPU/host/fabric and restores scheduling. No GPU "
            "reset is allowed to commit; no replacement or deletion is authorized."
        ),
        "preflight_identity": identity,
        "preflight_identity_digest": identity_digest(identity),
        "stop_conditions": [
            f"{PREDECESSOR_CASE_ID} predecessor evidence is not PASS",
            "the node is busy, tainted, owned, or its Node Agent is not ACTIVE",
            "gpuReset is not OWN by the Node Agent, or nodeReboot not OWN by the "
            "HyperPod adapter",
            "the deployed step timeout leaves no room for both escalation "
            "injections inside the barrier's WAITING window",
            "the estimated duration does not fit the node workflow lifetime the "
            "successor inherits from the reset workflow",
            "the maintenance window has under an hour left at injection time",
            "the GPU holder loses the arming race, so the client verification "
            "succeeds and the reset commits before the escalation arrives",
            "the second XID 46 opens a second workflow or adds a step",
            "the XID 79 does not supersede the reset workflow, or the successor "
            "carries no quiesce handoff",
            "any RESET_GPU or full-fabric reset reaches the Node Agent ledger",
            "CloudTrail shows no reboot, more than one, a replace/delete, or an "
            "actor other than the executor role",
            "cleanup cannot disarm the holder, prove quiescence, or leaves the "
            "node isolated",
        ],
        "rollback": {
            "runner_disarms_the_gpu_device_holder_idempotently": True,
            "runner_waits_for_the_incident_to_go_idle_before_restoring": True,
            "runner_restores_an_isolated_node_via_validated_restore_only": True,
            "runner_never_deletes_a_gpu_fault_taint_by_hand": True,
            "runner_finally_deletes_both_probe_pods": True,
            "no_node_replacement_or_deletion_is_authorized": True,
        },
    }


# --------------------------------------------------------------------------- #
# Live preflight (not unit-tested; delegates every assertion to the pure funcs)
# --------------------------------------------------------------------------- #
def focused_tests(case_dir: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "tests/orchestration/test_quiesce_handoff_on_preemption.py",
        "tests/orchestration/test_preemption_boundary_unified.py",
        "tests/execution/test_preemption_boundary.py",
        "tests/regional/test_destr016_preempting_reboot.py",
        "tests/regional/test_destr016_node_probe.py",
    ]
    completed = RegionalLiveFixture.run(command, cwd=ROOT, check=False, timeout=900)
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


def read_only_preflight(settings: Settings, case_dir: Path) -> dict[str, Any]:
    regional = RegionalLiveFixture(settings.regional)
    state = regional.store_snapshot(
        node=settings.node,
        observed_after=datetime.now(timezone.utc) - timedelta(minutes=10),
        hyperpod_cluster=settings.hyperpod_cluster,
    )
    node = regional.node_snapshot(settings.node)
    runtime_identity = regional.runtime_identity()
    tests = focused_tests(case_dir)
    predecessor = predecessor_evidence(settings.predecessor_path, PREDECESSOR_CASE_ID)
    env = control_env(regional)
    workloads = regional.business_workloads(settings.node)
    result: dict[str, Any] = {
        "release_id": state.get("release_id"),
        "node": node,
        "store": state,
        "business_workloads": workloads,
        "runtime_identity": runtime_identity,
        "focused_tests": tests,
        "cpu_blast": regional.cpu_blast_snapshot(),
        "predecessor": predecessor,
        "control_env": env,
    }
    result["errors"] = preflight_errors(
        node=settings.node,
        node_snapshot=node,
        agent=state.get("agent") or {},
        profile=state.get("profile"),
        queue=state.get("queue") or {},
        remote_commands=state.get("remote_commands") or {},
        gpu_workloads=regional.gpu_workloads(),
        business_workloads=workloads,
        event=state.get("event"),
        predecessor=predecessor,
        tests=tests,
        control_env=env,
        identity_errors=runtime_identity_errors(runtime_identity),
    )
    write_json_atomic(case_dir / "preflight.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Run the guarded DESTR-016 acceptance: a second same-rank XID is "
            "absorbed while a reset workflow is parked mid-quiesce, and an XID "
            "79 preempts it into a real HyperPod reboot that inherits the "
            "quiesce handoff."
        )
    )
    add_live_arguments(value, confirmation=CONFIRMATION)
    value.add_argument("--cpu-kubeconfig", default="")
    value.add_argument("--gpu-kubeconfig", default="")
    value.add_argument("--gpu-context", default="")
    value.add_argument("--namespace", default="gpu-fault-system")
    value.add_argument("--cluster-id", default="")
    value.add_argument("--region", default="")
    value.add_argument("--node", default="", help="the single idle GPU node to fault")
    value.add_argument("--host-probe-image", default="")
    value.add_argument("--hyperpod-cluster", default="")
    value.add_argument("--executor-role-arn", default="")
    value.add_argument(
        "--pci-bdf",
        default="",
        help="GPU PCI BDF to fault; defaults to the first inventory entry",
    )
    value.add_argument(
        "--holder-device",
        default="",
        help="/dev/nvidiaN the holder opens; defaults to the faulted GPU's index",
    )
    value.add_argument(
        "--max-hold-seconds",
        type=int,
        default=1800,
        help="bounded lifetime of the GPU device holder unit",
    )
    value.add_argument("--predecessor-evidence", default="")
    return value


# --------------------------------------------------------------------------- #
# Live execution
# --------------------------------------------------------------------------- #
def _probe(
    settings: Settings,
    *,
    script: Path,
    case_id: str,
    run_id: str,
) -> HostProbeFixture:
    return HostProbeFixture(
        HostProbeSettings(
            kubeconfig=settings.regional.gpu_kubeconfig,
            context=settings.regional.gpu_context,
            namespace=settings.regional.namespace,
            node=settings.node,
            image=settings.host_probe_image,
            case_id=case_id,
            run_id=run_id,
            probe_script=script,
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
    warm: WarmSpareLiveFixture
    run_id: str
    holder_probe: HostProbeFixture
    inject_probe: HostProbeFixture
    marker: dict[str, str] = field(default_factory=dict)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    injected_at: dict[str, str] = field(default_factory=dict)
    bdf: str = ""
    device: str = ""
    baseline: dict[str, Any] = field(default_factory=dict)
    incident_id: str = ""
    predecessor_id: str = ""
    successor_id: str = ""
    holder_armed: bool = False
    rebooted: bool = False


def _prepare_live_run(
    settings: Settings,
    run_dir: Path,
    attempt: int,
    maintenance_window_end: datetime,
) -> _LiveRun:
    case_dir = run_dir / "cases" / CASE_ID
    case_dir.mkdir(parents=True, exist_ok=True)
    preflight = read_only_preflight(settings, case_dir)
    if preflight["errors"]:
        raise RegionalFixtureError(
            "preflight failed: " + "; ".join(preflight["errors"])
        )
    planned = json.loads((case_dir / "plan.json").read_text(encoding="utf-8"))
    current = identity_digest(plan_identity(preflight, node=settings.node))
    if planned["details"]["preflight_identity_digest"] != current:
        raise RegionalFixtureError("DESTR-016 plan identity drifted before execution")
    regional = RegionalLiveFixture(settings.regional)
    run_id = f"destr016-{run_dir.name.rsplit('-', 1)[-1].lower()}-a{attempt}"
    return _LiveRun(
        settings=settings,
        case_dir=case_dir,
        attempt=attempt,
        maintenance_window_end=maintenance_window_end,
        preflight=preflight,
        regional=regional,
        warm=WarmSpareLiveFixture(regional, settings.hyperpod_cluster),
        run_id=run_id,
        holder_probe=_probe(
            settings, script=HOLDER_PROBE, case_id=CASE_ID, run_id=run_id
        ),
        inject_probe=_probe(
            settings,
            script=INJECT_PROBE,
            case_id=f"{CASE_ID}-inject",
            run_id=f"{run_id}-x",
        ),
    )


def _snapshot(run: _LiveRun, marker: str) -> dict[str, Any]:
    return run.regional.store_snapshot(
        node=run.settings.node,
        marker=marker,
        observed_after=run.started_at,
        hyperpod_cluster=run.settings.hyperpod_cluster,
        queue_attempts=1,
    )


def _workflow_by_id(run: _LiveRun, request_id: str) -> dict[str, Any]:
    return run.regional.cpu_python(WORKFLOW_BY_ID, request_id)


def _commands_of(run: _LiveRun, request_id: str) -> list[dict[str, Any]]:
    payload = run.regional.cpu_python(COMMANDS_BY_WORKFLOW, request_id)
    return [
        reboot_case.redact_lease_tokens(item) for item in payload.get("commands") or []
    ]


def _arm_and_park(run: _LiveRun) -> dict[str, Any]:
    """Phase 1: fault the node, hold a GPU client, and wait until the reset
    workflow is parked at the barrier with the quiesce applied."""

    settings, case_dir, run_id = run.settings, run.case_dir, run.run_id
    run.holder_probe.create()
    run.inject_probe.create()
    baseline = run.inject_probe.execute("snapshot")
    write_json_atomic(case_dir / "host-baseline.json", baseline)
    run.baseline = baseline
    if baseline.get("compute_clients"):
        raise RegionalFixtureError("the node already has NVIDIA compute clients")
    if not baseline.get("kmsg_writable"):
        raise RegionalFixtureError("/dev/kmsg is not writable from the host probe")
    if baseline.get("quiesce_states"):
        raise RegionalFixtureError("the node has a pre-existing GPU quiesce state")
    run.bdf = target_bdf(settings.pci_bdf, baseline.get("gpu_inventory") or [])
    run.device = holder_device(settings.device, baseline.get("gpu_inventory") or [])
    window = window_errors(
        now=datetime.now(timezone.utc),
        maintenance_window_end=run.maintenance_window_end,
    )
    if window:
        raise RegionalFixtureError("; ".join(window))
    run.marker = markers(f"destr016-{int(time.time())}-a{run.attempt}")
    armed = run.holder_probe.execute(
        "arm-holder",
        "--device",
        run.device,
        "--drill-id",
        run_id,
        "--after-ledger-op",
        "QUIESCE_GPU_SERVICES",
        "--max-hold-seconds",
        str(settings.max_hold_seconds),
        "--run-id",
        run_id,
        "--probe-script",
        run.holder_probe.host_script,
        "--inject-script",
        run.inject_probe.host_script,
        "--pci-bdf",
        run.bdf,
        "--absorb-marker",
        run.marker["absorb"],
        "--absorb-drill-id",
        f"{run_id}-s",
        "--absorb-after-seconds",
        str(ABSORB_DELAY_SECONDS),
        "--escalate-marker",
        run.marker["escalate"],
        "--escalate-drill-id",
        f"{run_id}-e",
        "--escalate-after-seconds",
        str(ESCALATE_DELAY_SECONDS),
    )
    run.holder_armed = True
    write_json_atomic(case_dir / "holder-armed.json", armed)
    run.started_at = datetime.now(timezone.utc)
    injection = run.inject_probe.execute(
        "write-xid46",
        "--marker",
        run.marker["reset"],
        "--drill-id",
        f"{run_id}-r",
        "--pci-bdf",
        run.bdf,
    )
    run.injected_at["reset"] = datetime.now(timezone.utc).isoformat()
    write_json_atomic(case_dir / "injection-reset.json", injection)
    return _wait_for_barrier(run)


def _wait_for_barrier(run: _LiveRun) -> dict[str, Any]:
    """Poll until the reset workflow is parked at the dirty boundary.

    The predicate is the verdict function itself, so the state the case waits
    for and the state it asserts can never drift apart.
    """

    deadline = time.monotonic() + BARRIER_WAIT_BUDGET_SECONDS
    last: dict[str, Any] = {}
    errors = ["the reset workflow never appeared"]
    while time.monotonic() < deadline:
        last = _snapshot(run, run.marker["reset"])
        workflow = last.get("workflow") or {}
        if workflow:
            # The barrier is only *proven* once the executor has folded the
            # Agent's "clients are still active" verdict into the WAITING
            # command; the first WAITING record is just a pointer to a PENDING
            # node action, so keep polling until the reason is on record.
            errors = waiting_boundary_errors(workflow) + barrier_reason_errors(
                last.get("commands") or [], workflow
            )
            if not errors:
                write_json_atomic(run.case_dir / "barrier-state.json", last)
                return last
            if workflow.get("status") in {
                "SUCCEEDED",
                "FAILED",
                "BLOCKED",
                "SUPERSEDED",
            }:
                break
        time.sleep(5)
    write_json_atomic(run.case_dir / "barrier-state.json", last)
    try:
        holder = run.holder_probe.execute("holder-status", "--run-id", run.run_id)
    except Exception as exc:  # noqa: BLE001 - a quiesced node cannot answer
        holder = {"error": f"{type(exc).__name__}: {exc}"}
    write_json_atomic(run.case_dir / "holder-status-barrier.json", holder)
    raise RegionalFixtureError(
        "the reset workflow never parked at the client-verification barrier: "
        + "; ".join(errors)
        + f"; holder={json.dumps(holder, sort_keys=True)}"
    )


def _absorb(run: _LiveRun, barrier: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    """Phase A: a second same-rank XID 46 must merge into the parked workflow."""

    # The write itself was scheduled on the node by arm-holder; kubelet is
    # stopped behind the quiesce, so the only way to see it land is the store.
    deadline = time.monotonic() + ABSORB_BUDGET_SECONDS
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        last = _snapshot(run, run.marker["absorb"])
        if last.get("workflow") and not incident_reason_errors(
            barrier.get("incident") or {},
            last.get("incident") or {},
            node=run.settings.node,
            xid=FIRST_XID,
        ):
            break
        time.sleep(5)
    write_json_atomic(run.case_dir / "absorb-state.json", last)
    run.injected_at["absorb"] = str(
        (last.get("event") or {}).get("observed_at")
        or f"scheduled {ABSORB_DELAY_SECONDS}s after the holder started"
    )
    errors = absorb_errors(barrier, last, node=run.settings.node)
    # The absorbed fault must not have moved the boundary either: the case can
    # only preempt what is still parked.
    errors.extend(waiting_boundary_errors(last.get("workflow") or {}))
    return errors, last


def _escalate(
    run: _LiveRun, barrier: dict[str, Any]
) -> tuple[list[str], dict[str, Any]]:
    """Phase B: XID 79 must preempt the parked reset and adopt its quiesce."""

    case_dir = run.case_dir
    run.predecessor_id = str((barrier.get("workflow") or {}).get("request_id") or "")
    run.incident_id = str((barrier.get("incident") or {}).get("incident_id") or "")
    # Scheduled on the node by arm-holder (see ESCALATE_DELAY_SECONDS).
    successor: dict[str, Any] = {}
    deadline = time.monotonic() + PREEMPTION_BUDGET_SECONDS
    while time.monotonic() < deadline:
        successor = _snapshot(run, run.marker["escalate"])
        request_id = str((successor.get("workflow") or {}).get("request_id") or "")
        if request_id and request_id != run.predecessor_id:
            run.successor_id = request_id
            break
        time.sleep(5)
    write_json_atomic(case_dir / "successor-created.json", successor)
    run.injected_at["escalate"] = str(
        (successor.get("event") or {}).get("observed_at")
        or f"scheduled {ESCALATE_DELAY_SECONDS}s after the holder started"
    )
    if not run.successor_id:
        raise RegionalFixtureError(
            f"XID 79 did not create a successor workflow: {successor}"
        )

    predecessor = _wait_for(
        lambda: _workflow_by_id(run, run.predecessor_id),
        accept=lambda value: value.get("status") == "SUPERSEDED",
        budget_seconds=SUPERSEDE_BUDGET_SECONDS,
        message="the reset workflow was never superseded",
        path=case_dir / "predecessor-superseded.json",
    )
    adopted = _wait_for(
        lambda: _workflow_by_id(run, run.successor_id),
        accept=lambda value: bool(value.get("quiesce_handoff_from_workflow_id")),
        budget_seconds=HANDOFF_BUDGET_SECONDS,
        message="the successor never adopted the quiesce handoff",
        path=case_dir / "successor-adopted.json",
    )
    commands = _commands_of(run, run.predecessor_id)
    write_json_atomic(case_dir / "predecessor-commands.json", {"commands": commands})
    errors = escalation_errors(
        predecessor,
        adopted,
        successor.get("incident") or {},
        decision=successor.get("decision") or {},
    )
    errors.extend(successor_step_graph_errors(adopted))
    errors.extend(superseded_predecessor_errors(predecessor))
    errors.extend(
        cancelled_command_errors(
            commands, successor_request_id=run.successor_id, workflow=predecessor
        )
    )
    return errors, adopted


def _wait_for(
    read: Any,
    *,
    accept: Any,
    budget_seconds: int,
    message: str,
    path: Path,
) -> dict[str, Any]:
    deadline = time.monotonic() + budget_seconds
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        last = read()
        if accept(last):
            write_json_atomic(path, last)
            return last
        time.sleep(5)
    write_json_atomic(path, last)
    raise RegionalFixtureError(f"{message}: {json.dumps(last, sort_keys=True)}")


def _observe_until_terminal(run: _LiveRun) -> dict[str, Any]:
    transitions: dict[str, str] = {}
    timeline: list[dict[str, Any]] = []
    deadline = time.monotonic() + OBSERVATION_BUDGET_SECONDS
    state: dict[str, Any] = {}
    while time.monotonic() < deadline:
        state = _snapshot(run, run.marker["escalate"])
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
        time.sleep(10)
    write_json_atomic(run.case_dir / "workflow-state.json", state)
    return state


def _data_plane_errors(
    run: _LiveRun, state: dict[str, Any]
) -> tuple[list[str], dict[str, Any]]:
    settings, case_dir, run_id = run.settings, run.case_dir, run.run_id
    errors: list[str] = []
    # The reboot took both probe Pods with it (restartPolicy Never), so the
    # node-side evidence needs them back. The holder's state file lives under
    # /var/lib and survived, which is what holder-status reads.
    run.holder_probe.create()
    run.inject_probe.create()
    holder = run.holder_probe.execute("holder-status", "--run-id", run_id)
    write_json_atomic(case_dir / "holder-status.json", holder)
    errors.extend(holder_errors(holder))
    after = run.inject_probe.execute(
        "snapshot",
        "--since-epoch",
        str(run.started_at.timestamp()),
        "--pci-bdf",
        run.bdf,
        timeout=300,
    )
    write_json_atomic(case_dir / "host-after.json", after)
    errors.extend(
        host_errors(
            run.baseline,
            after,
            expected_gpu_count=len(run.baseline.get("gpu_inventory") or []),
        )
    )
    final_node = run.regional.node_snapshot(settings.node)
    write_json_atomic(case_dir / "node-final.json", final_node)
    errors.extend(schedulability_errors(final_node, node=settings.node))
    errors.extend(
        node_recovery_errors(
            run.preflight["node"],
            run.preflight.get("node_after_boot") or {},
            final_node,
        )
    )
    provider = run.regional.provider_events(run.started_at, datetime.now(timezone.utc))
    write_json_atomic(case_dir / "provider-events.json", {"events": provider})
    unique = reboot_events(provider)
    errors.extend(
        provider_errors(
            provider,
            actor_matches_role=(
                provider_event_actor_matches_role(unique[0], settings.executor_role_arn)
                if len(unique) == 1
                else None
            ),
        )
    )
    errors.extend(restore_after_reboot_errors(state.get("workflow") or {}))
    cpu_after = run.regional.cpu_blast_snapshot()
    write_json_atomic(case_dir / "cpu-blast-after.json", cpu_after)
    if cpu_after != run.preflight["cpu_blast"]:
        errors.append("control-plane EKS state differs from baseline")
    run.regional.verify_runtime_identity(
        run.preflight["runtime_identity"],
        evidence_path=case_dir / "runtime-identity-after-reboot.json",
        stage="after DESTR-016 reboot",
    )
    return errors, {
        "boot_id_before": run.baseline.get("boot_id"),
        "boot_id_after": after.get("boot_id"),
        "ledger_rows_added": len(after.get("ledger") or [])
        - len(run.baseline.get("ledger") or []),
        "holder_hold_started_at": holder.get("hold_started_at"),
    }


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
        "node": settings.node,
        "maintenance_window_end": maintenance_window_end.isoformat(),
    }
    try:
        barrier = _arm_and_park(run)
        errors = reset_workflow_errors(
            barrier.get("workflow") or {},
            barrier.get("incident") or {},
            barrier.get("decision") or {},
        )
        errors.extend(barrier_reason_errors(barrier.get("commands") or []))
        absorb, absorbed = _absorb(run, barrier)
        errors.extend(absorb)
        escalation, adopted = _escalate(run, absorbed if absorbed else barrier)
        errors.extend(escalation)
        node_after_boot = run.regional.wait_node_ready(
            settings.node,
            timeout_seconds=REBOOT_BUDGET_SECONDS,
            expected_boot_id=run.preflight["node"].get("boot_id"),
        )
        run.rebooted = True
        run.preflight["node_after_boot"] = node_after_boot
        write_json_atomic(run.case_dir / "node-after-boot.json", node_after_boot)
        state = _observe_until_terminal(run)
        errors.extend(
            terminal_errors(
                state,
                expected_boot_id=(run.preflight["store"].get("agent") or {}).get(
                    "boot_id"
                ),
                expected_artifact=(run.preflight["store"].get("agent") or {}).get(
                    "artifact_sha256"
                ),
            )
        )
        data_errors, hosts = _data_plane_errors(run, state)
        errors.extend(data_errors)
        details = {
            "preflight_identity": plan_identity(run.preflight, node=settings.node),
            "predecessor_workflow": adopted.get("predecessor_workflow_id"),
            "workflow": state.get("workflow"),
            "incident": state.get("incident"),
            "hosts": hosts,
        }
        components = evidence_components(details)
        result.update(
            {
                "verdict": "PASS" if not errors else "FAIL",
                "errors": errors,
                "markers": run.marker,
                "injected_at": run.injected_at,
                "incident_id": run.incident_id,
                "predecessor_workflow_id": run.predecessor_id,
                "successor_workflow_id": run.successor_id,
                "submission": state.get("submission"),
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
    result["limitations"] = [
        "The boundary this case preempts is VERIFY_NO_GPU_CLIENTS WAITING with "
        "the quiesce applied and unrestored. A single-node RESET_GPU cannot "
        "itself park WAITING, so 'mid-reset' means the physical step sequence "
        "is in flight, not that the reset call was interrupted.",
        "The second XID 46 is recorded as ABSORB when it carries the same GPU "
        "UUID and as WIDEN_IN_PLACE when it carries none or another GPU. The "
        "case asserts the observables both share -- one incident, one workflow, "
        "one request id, an unchanged step list -- and never the label.",
        "Supersession happens when the predecessor's executor next reaches a "
        "step boundary, so the SUPERSEDED transition can lag the successor's "
        "creation by up to one lease interval.",
    ]
    write_json_atomic(run.case_dir / f"{CASE_ID}.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["verdict"] == "PASS" else 1


def _cleanup(run: _LiveRun) -> dict[str, Any]:
    """Disarm, quiesce, restore, then remove. Order matters: the holder must go
    before the incident is allowed to settle, and a node left isolated is only
    ever restored through the validation-first workflow."""

    result: dict[str, Any] = {"errors": []}

    def guard(label: str, action: Any) -> None:
        try:
            result[label] = action()
        except Exception as exc:  # noqa: BLE001 - a cleanup failure is a FAIL
            result["errors"].append(f"{label}: {type(exc).__name__}: {exc}")

    if run.holder_armed:
        guard("holder_disarm", lambda: _disarm_holder(run))
    if run.incident_id:
        guard("incident_idle", lambda: run.warm.wait_incident_idle(run.incident_id))
        guard("restore_isolated_node", lambda: _restore_isolated_node(run))
    for label, probe in (
        ("holder_probe", run.holder_probe),
        ("inject_probe", run.inject_probe),
    ):
        guard(f"{label}_cleanup", lambda probe=probe: _refuse_residual(probe.cleanup()))
    guard(
        "runtime_identity",
        lambda: run.regional.verify_runtime_identity(
            run.preflight["runtime_identity"],
            evidence_path=run.case_dir / "runtime-identity-after-cleanup.json",
            stage="after DESTR-016 cleanup",
        ),
    )
    return result


def _disarm_holder(run: _LiveRun) -> dict[str, Any]:
    """Idempotent: a holder the reboot already took with it is not an error.

    The probe Pod does not survive the reboot, so it is recreated first; a node
    that cannot answer here is a cleanup failure, which is the point.
    """

    run.holder_probe.create()
    return run.holder_probe.execute("disarm-holder", "--run-id", run.run_id)


def _refuse_residual(residuals: dict[str, bool]) -> dict[str, bool]:
    if any(residuals.values()):
        raise RegionalFixtureError(f"residual resources remain: {residuals}")
    return residuals


def _restore_isolated_node(run: _LiveRun) -> dict[str, Any]:
    """A failed workflow leaves the node cordoned and owned. Never delete the
    taint by hand: restore through the validation-first workflow."""

    settings = run.settings
    snapshot = run.regional.node_snapshot(settings.node)
    isolated = bool(
        snapshot.get("unschedulable")
        or snapshot.get("ownership_annotations")
        or any(
            str(taint.get("key") or "").startswith("gpu-fault.io/")
            for taint in snapshot.get("taints") or []
        )
    )
    if not isolated:
        return {"isolated": False}
    run.warm.wait_agent_active(settings.node)
    profile_version = str(
        (run.preflight["store"].get("profile") or {}).get("profile_version") or ""
    )
    created = run.warm.create_restore_workflow(
        incident_id=run.incident_id,
        node=settings.node,
        profile_version=profile_version,
        reason="DESTR-016 validated cleanup",
    )
    restored = run.warm.wait_workflow_id(str(created["workflow_request_id"]))
    if restored.get("status") != "SUCCEEDED":
        raise RegionalFixtureError(
            f"{settings.node} restore workflow did not succeed: {restored.get('status')}"
        )
    return {"isolated": True, "restore": restored.get("status")}


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
