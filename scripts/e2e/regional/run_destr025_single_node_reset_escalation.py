#!/usr/bin/env python3
"""GF-REGIONAL-DESTR-025 live acceptance runner.

One idle GPU node. A GPU device holder is armed to start right after the reset
workflow's own VERIFY_NO_GPU_CLIENTS succeeds, so the RESET_GPU that follows
is refused with a *known* outcome ("clients are still active"). The product
must then climb the hardware ladder inside the same workflow record: the flat
single-node plan grows into a one-branch DAG, a RESTART_NODE rung and its
validation tail are appended, the node really reboots through HyperPod (the
reboot takes the holder with it), comes back, validates and is restored to
scheduling. The record ends SUCCEEDED, the incident RECOVERED, and no
``workflow-reboot-after-<id>`` successor exists.

The runner defaults to ``--plan``; ``--execute`` needs ``--confirm
DESTR025_EXECUTE``. The verdict functions live in ``destr025_verdicts`` and are
unit-tested; the runner records digests only. Cleanup never leaves a cordon:
a failed run restores the node through the validation-first workflow, and
every cleanup error is a FAIL (this case has no unknown-outcome hold).
"""

from __future__ import annotations

import argparse
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

from scripts.e2e.regional import executor_env_window as env_window  # noqa: E402
from scripts.e2e.regional import (  # noqa: E402
    run_destr016_preempting_reboot as preempting,
)
from scripts.e2e.regional.acceptance_runner_common import (  # noqa: E402
    node_open_incidents,
    write_json_atomic,
)
from scripts.e2e.regional.destr014_preflight import (  # noqa: E402
    _control_env,
    budget_headroom,
    executor_env_snapshot,
)
from scripts.e2e.regional.destr016_verdicts import (  # noqa: E402
    node_recovery_errors,
    provider_errors,
    reboot_events,
    schedulability_errors,
    step_transitions,
)
from scripts.e2e.regional.destr025_verdicts import (  # noqa: E402
    CASE_ID,
    CONFIRMATION,
    GPU_HOLD_SECONDS,
    PREDECESSOR_CASE_ID,
    case_digest,
    command_errors,
    evidence_components,
    host_errors,
    identity_digest,
    plan_identity,
    preflight_errors,
    successor_errors,
    window_identity_errors,
    workflow_errors,
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
from scripts.e2e.regional.warm_spare_fixture import WarmSpareLiveFixture  # noqa: E402

HOLDER_PROBE = Path(__file__).with_name("probes") / "destr014_node_probe.py"
INJECT_PROBE = Path(__file__).with_name("probes") / "destructive_node_probe.py"
ARM_AFTER_OPERATION = "VERIFY_NO_GPU_CLIENTS"
WORKFLOW_WAIT_SECONDS = 600
OBSERVATION_BUDGET_SECONDS = 2400
REBOOT_BUDGET_SECONDS = 1800
HOST_RETURN_BUDGET_SECONDS = 900

# Every record the store holds that could have taken the reboot over from the
# case's own workflow: the deterministic whole-workflow successors and any
# record that names ours as its predecessor.
SUCCESSORS = r"""
import json
import sys

from gpu_fault.app import ApplicationContext

store = ApplicationContext.from_environment().store
request_id = sys.argv[1]
suffix = f"-after-{request_id}"
rows = []
for item in store.list_workflows(limit=500):
    if item.request_id.endswith(suffix) or item.predecessor_workflow_id == request_id:
        rows.append(
            {
                "request_id": item.request_id,
                "status": item.status.value,
                "incident_id": item.incident_id,
                "predecessor_workflow_id": item.predecessor_workflow_id,
            }
        )
print(json.dumps({"successors": rows}, sort_keys=True, default=str))
"""


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
    verify_max_attempts: int
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

    # ``budget_headroom`` reads a fault/sibling pair; one node fills both.
    @property
    def fault_node(self) -> str:
        return self.node

    @property
    def sibling_node(self) -> str:
        return self.node


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
        verify_max_attempts=int(arguments.verify_max_attempts),
        predecessor_path=predecessor,
    )


def plan_details(settings: Settings, preflight: dict[str, Any]) -> dict[str, Any]:
    identity = plan_identity(preflight, node=settings.node)
    return {
        "risk": "destructive-provider-reboot",
        "predecessor": preflight.get("predecessor"),
        "node": settings.node,
        "hyperpod_cluster": settings.hyperpod_cluster,
        "verify_max_attempts": settings.verify_max_attempts,
        "gpu_hold_seconds": GPU_HOLD_SECONDS,
        "control_env": preflight.get("control_env"),
        "mutation": (
            "write one XID 46 to /dev/kmsg on one idle GPU node; a GPU device "
            "holder armed to start right after the workflow's own "
            "VERIFY_NO_GPU_CLIENTS succeeds makes the RESET_GPU fail with a "
            "known outcome (clients are still active); the record must grow "
            "into a one-branch DAG and append a RESTART_NODE rung that reboots "
            "the node through HyperPod for real (BatchRebootClusterNodes, once), "
            "then validate GPU/host/fabric and restore scheduling inside the "
            "same record"
        ),
        "preflight_identity": identity,
        "preflight_identity_digest": identity_digest(identity),
        "stop_conditions": [
            f"{PREDECESSOR_CASE_ID} predecessor evidence is not PASS",
            "the node is busy, tainted, owned, or carries a GPU workload",
            "the executor forbids reboots or allows replacement",
            "the estimated duration does not fit the node workflow lifetime",
            "the RESET_GPU failure is not the known clients-are-still-active refusal",
            "a workflow-reboot-after-<id> (or replace/support-after) successor appears",
            "a REPLACE_NODE rung is taken or the branch is exhausted",
            "the node does not come back with a new boot id",
            "CloudTrail shows other than one reboot by the executor role",
            "cleanup cannot disarm the holder, restore the node or close the env window",
        ],
        "rollback": {
            "disarm_the_gpu_device_holder": True,
            "restore_an_isolated_node_through_the_validated_workflow": True,
            "close_the_executor_env_window_to_baseline": True,
            "never_call_provider_replace_or_delete": True,
            "never_leave_a_cordon_or_taint": True,
        },
    }


def focused_tests(case_dir: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "tests/execution/test_branch_escalation.py",
        "tests/regional/test_destr025_verdicts.py",
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
    """The control-worker bounds the case has to fit: node lifetime and step
    timeout (as DESTR-016 reads them), the poll interval and the branch rung
    ceiling (as DESTR-014 reads them). Read-only."""

    record = dict(preempting.control_env(regional))
    branch = _control_env(regional)
    record["poll_interval_seconds"] = branch.get("poll_interval_seconds")
    record["max_rungs"] = branch.get("max_rungs")
    return record


def read_only_preflight(settings: Settings, case_dir: Path) -> dict[str, Any]:
    regional = RegionalLiveFixture(settings.regional)
    state = regional.store_snapshot(
        node=settings.node,
        observed_after=datetime.now(timezone.utc) - timedelta(minutes=10),
        hyperpod_cluster=settings.hyperpod_cluster,
    )
    state["open_incidents"] = node_open_incidents(
        regional.cpu_python, settings.regional.cluster_id, settings.node
    )
    node = regional.node_snapshot(settings.node)
    runtime_identity = regional.runtime_identity()
    tests = focused_tests(case_dir)
    predecessor = predecessor_evidence(settings.predecessor_path, PREDECESSOR_CASE_ID)
    env = control_env(regional)
    workloads = regional.business_workloads(settings.node)
    executor_env = executor_env_snapshot(regional)
    budget = budget_headroom(regional, settings)
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
        "executor_env": executor_env,
        "budget": budget,
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
        open_incidents=state["open_incidents"],
        predecessor=predecessor,
        tests=tests,
        control_env=env,
        executor_env=executor_env,
        budget=budget,
        identity_errors=runtime_identity_errors(runtime_identity),
        verify_max_attempts=settings.verify_max_attempts,
    )
    write_json_atomic(case_dir / "preflight.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Run the guarded DESTR-025 acceptance: a single-node RESET_GPU fails "
            "with a known outcome and escalates to a real HyperPod reboot inside "
            "its own workflow record."
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
        "--verify-max-attempts",
        type=int,
        default=6,
        help="GPU_FAULT_GPU_CLIENT_VERIFY_MAX_ATTEMPTS set on the executor for the run",
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
    case_dir: Path,
) -> HostProbeFixture:
    return HostProbeFixture(
        HostProbeSettings(
            state_directory=case_dir / "host-probes",
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
    env_baseline: Path
    marker: str = ""
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    bdf: str = ""
    device: str = ""
    baseline: dict[str, Any] = field(default_factory=dict)
    incident_id: str = ""
    request_id: str = ""
    env_opened: bool = False
    holder_armed: bool = False
    holder_armed_at: datetime | None = None
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
        raise RegionalFixtureError("DESTR-025 plan identity drifted before execution")
    regional = RegionalLiveFixture(settings.regional)
    run_id = f"destr025-{run_dir.name.rsplit('-', 1)[-1].lower()}-a{attempt}"
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
            settings,
            script=HOLDER_PROBE,
            case_id=CASE_ID,
            run_id=run_id,
            case_dir=case_dir,
        ),
        inject_probe=_probe(
            settings,
            script=INJECT_PROBE,
            case_id=f"{CASE_ID}-inject",
            run_id=f"{run_id}-x",
            case_dir=case_dir,
        ),
        env_baseline=case_dir / "executor-env-window.json",
    )


def _snapshot(run: _LiveRun) -> dict[str, Any]:
    return run.regional.store_snapshot(
        node=run.settings.node,
        marker=run.marker,
        observed_after=run.started_at,
        hyperpod_cluster=run.settings.hyperpod_cluster,
        queue_attempts=1,
    )


def _require_window(run: _LiveRun, stage: str) -> None:
    if datetime.now(timezone.utc) >= run.maintenance_window_end:
        raise RegionalFixtureError(f"maintenance window ended before {stage}")


def _open_env_window(run: _LiveRun) -> None:
    """Lower the executor's client-verify attempts so the refused reset lands
    within minutes; opening rolls the executor replicas, so it goes first."""

    _require_window(run, "executor configuration")
    run.env_opened = True
    report = env_window.open_window(
        env_window.Settings(baseline=run.env_baseline, rollout_timeout_seconds=300),
        run.regional,
        env_window.survey(run.regional),
        {
            "GPU_FAULT_GPU_CLIENT_VERIFY_MAX_ATTEMPTS": str(
                run.settings.verify_max_attempts
            )
        },
    )
    write_json_atomic(run.case_dir / "env-window-open.json", report)


def _arm_and_inject(run: _LiveRun) -> dict[str, Any]:
    """Arm the holder behind the workflow's own client verification, fault the
    node, and wait until the single-node reset workflow exists."""

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
    inventory = baseline.get("gpu_inventory") or []
    run.bdf = preempting.target_bdf(settings.pci_bdf, inventory)
    run.device = preempting.holder_device(settings.device, inventory, pci_bdf=run.bdf)
    _require_window(run, "arming the holder")
    run.holder_armed = True
    armed = run.holder_probe.execute(
        "arm-holder",
        "--device",
        run.device,
        "--drill-id",
        run_id,
        "--after-ledger-op",
        ARM_AFTER_OPERATION,
        "--max-hold-seconds",
        str(GPU_HOLD_SECONDS),
        "--run-id",
        run_id,
        "--probe-script",
        run.holder_probe.host_script,
    )
    write_json_atomic(case_dir / "holder-armed.json", armed)
    run.holder_armed_at = datetime.now(timezone.utc)
    run.started_at = datetime.now(timezone.utc)
    run.marker = f"destr025-{int(time.time())}-a{run.attempt}"
    injection = run.inject_probe.execute(
        "write-xid46",
        "--marker",
        run.marker,
        "--drill-id",
        f"{run_id}-r",
        "--pci-bdf",
        run.bdf,
    )
    write_json_atomic(case_dir / "injection.json", injection)
    state = run.regional.wait_for_workflow(
        node=settings.node,
        marker=run.marker,
        observed_after=run.started_at,
        case_dir=case_dir,
        timeout_seconds=WORKFLOW_WAIT_SECONDS,
        hyperpod_cluster=settings.hyperpod_cluster,
        terminal=False,
    )
    workflow = state.get("workflow") or {}
    run.request_id = str(workflow.get("request_id") or "")
    run.incident_id = str((state.get("incident") or {}).get("incident_id") or "")
    if not run.request_id or not run.incident_id:
        raise RegionalFixtureError("the injected XID opened no workflow record")
    return state


def _observe_until_terminal(run: _LiveRun) -> dict[str, Any]:
    transitions: dict[str, str] = {}
    timeline: list[dict[str, Any]] = []
    deadline = time.monotonic() + OBSERVATION_BUDGET_SECONDS
    state: dict[str, Any] = {}
    while time.monotonic() < deadline:
        state = _snapshot(run)
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


def _control_plane_errors(run: _LiveRun, state: dict[str, Any]) -> list[str]:
    workflow = state.get("workflow") or {}
    incident = state.get("incident") or {}
    errors = workflow_errors(workflow, incident, node=run.settings.node)
    successors = run.regional.cpu_python(SUCCESSORS, run.request_id)
    write_json_atomic(run.case_dir / "successors.json", successors)
    errors.extend(
        successor_errors(
            successors.get("successors") or [],
            request_id=run.request_id,
            workflow=workflow,
        )
    )
    commands = run.regional.cpu_python(preempting.COMMANDS_BY_WORKFLOW, run.request_id)
    redacted = [
        preempting.reboot_case.redact_lease_tokens(item)
        for item in commands.get("commands") or []
    ]
    write_json_atomic(run.case_dir / "commands.json", {"commands": redacted})
    errors.extend(command_errors(redacted, workflow))
    return errors


def _data_plane_errors(
    run: _LiveRun, state: dict[str, Any]
) -> tuple[list[str], dict[str, Any]]:
    settings, case_dir, run_id = run.settings, run.case_dir, run.run_id
    errors: list[str] = []
    node_after_boot = run.regional.wait_node_ready(
        settings.node,
        timeout_seconds=REBOOT_BUDGET_SECONDS,
        expected_boot_id=run.preflight["node"].get("boot_id"),
    )
    run.rebooted = True
    write_json_atomic(case_dir / "node-after-boot.json", node_after_boot)
    # The reboot took both probe Pods with it (restartPolicy Never); the
    # holder's state file under /var/lib survived and is what status reads.
    run.holder_probe.create()
    run.inject_probe.create()
    holder = run.holder_probe.execute("holder-status", "--run-id", run_id)
    write_json_atomic(case_dir / "holder-status.json", holder)
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
            holder_status=holder,
            expected_gpu_count=len(run.baseline.get("gpu_inventory") or []),
        )
    )
    final_node = run.regional.node_snapshot(settings.node)
    write_json_atomic(case_dir / "node-final.json", final_node)
    errors.extend(schedulability_errors(final_node, node=settings.node))
    errors.extend(
        node_recovery_errors(run.preflight["node"], node_after_boot, final_node)
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
    cpu_after = run.regional.cpu_blast_snapshot()
    write_json_atomic(case_dir / "cpu-blast-after.json", cpu_after)
    if cpu_after != run.preflight["cpu_blast"]:
        errors.append("control-plane EKS state differs from baseline")
    # The executor env window is still open here: its Deployment carries the
    # managed variable and one extra generation; everything else must match.
    identity = run.regional.runtime_identity()
    write_json_atomic(case_dir / "runtime-identity-after-reboot.json", identity)
    errors.extend(
        window_identity_errors(
            run.preflight["runtime_identity"],
            identity,
            deployment=env_window.DEPLOYMENT,
            generation_delta=1,
            allow_template_change=True,
        )
    )
    errors.extend(runtime_identity_errors(identity))
    return errors, {
        "boot_id_before": run.baseline.get("boot_id"),
        "boot_id_after": after.get("boot_id"),
        "ledger_rows_added": len(after.get("ledger") or [])
        - len(run.baseline.get("ledger") or []),
        "holder_hold_started_at": holder.get("hold_started_at"),
        "provider_reboot_events": len(unique),
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
        _require_window(run, "case setup")
        _open_env_window(run)
        _arm_and_inject(run)
        state = _observe_until_terminal(run)
        errors = _control_plane_errors(run, state)
        data_errors, hosts = _data_plane_errors(run, state)
        errors.extend(data_errors)
        details = {
            "preflight_identity": plan_identity(run.preflight, node=settings.node),
            "workflow": state.get("workflow"),
            "incident": state.get("incident"),
            "hosts": hosts,
        }
        components = evidence_components(details)
        result.update(
            {
                "verdict": "PASS" if not errors else "FAIL",
                "errors": errors,
                "marker": run.marker,
                "incident_id": run.incident_id,
                "workflow_request_id": run.request_id,
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
    write_json_atomic(run.case_dir / f"{CASE_ID}.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["verdict"] == "PASS" else 1


def env_window_close_action(env_opened: bool, baseline: Path) -> str:
    """What cleanup owes the executor env window.

    ``close`` when this attempt opened it and the record exists; ``never-recorded``
    when the open was attempted but refused before writing its record (there is
    no baseline to restore and the live env was not changed); ``none`` otherwise.
    """

    if not env_opened:
        return "none"
    return "close" if baseline.is_file() else "never-recorded"


def _cleanup(run: _LiveRun) -> dict[str, Any]:
    """Disarm, settle, restore, remove, then close the env window. A node left
    isolated is only ever restored through the validation-first workflow."""

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
    action = env_window_close_action(run.env_opened, run.env_baseline)
    if action == "close":
        guard(
            "env_window_close",
            lambda: env_window.close_window(
                env_window.Settings(
                    baseline=run.env_baseline, rollout_timeout_seconds=300
                ),
                run.regional,
                env_window.survey(run.regional),
            ),
        )
    elif action == "never-recorded":
        # The open was refused before it wrote its record (a foreign window
        # still in effect, a disagreeing live env): nothing of ours to close.
        result["env_window_close"] = {"skipped": "no window record was written"}
    guard(
        "runtime_identity",
        lambda: _restored_identity(run, closed="env_window_close" in result),
    )
    return result


def _restored_identity(run: _LiveRun, *, closed: bool) -> dict[str, Any]:
    """After cleanup the executor template must be back to its pre-window digest;
    only its generation moved (one write per open, one per close)."""

    identity = run.regional.runtime_identity()
    write_json_atomic(run.case_dir / "runtime-identity-after-cleanup.json", identity)
    delta = (1 if run.env_opened else 0) + (1 if closed else 0)
    errors = window_identity_errors(
        run.preflight["runtime_identity"],
        identity,
        deployment=env_window.DEPLOYMENT,
        generation_delta=delta,
        allow_template_change=run.env_opened and not closed,
    )
    errors.extend(runtime_identity_errors(identity))
    if errors:
        raise RegionalFixtureError(
            "after DESTR-025 cleanup runtime identity drifted: " + "; ".join(errors)
        )
    return identity


def _disarm_holder(run: _LiveRun) -> dict[str, Any]:
    """Idempotent: a holder the reboot already took with it is not an error.
    Nothing is exec'd while kubelet is down."""

    reach = host_reachability(
        run.regional,
        node=run.settings.node,
        baseline_boot_id=(run.preflight.get("node") or {}).get("boot_id"),
        holder_armed_at=run.holder_armed_at,
        max_hold_seconds=GPU_HOLD_SECONDS,
        budget_seconds=HOST_RETURN_BUDGET_SECONDS,
    )
    if not reach["exec_allowed"]:
        if reach["assume_disarmed"]:
            return {"disarmed": "assumed", **reach}
        raise RegionalFixtureError(
            "the node never answered again, so the holder could not be "
            f"disarmed: {reach['reason']} ({reach['wait_error']})"
        )
    run.holder_probe.create()
    status = run.holder_probe.execute("holder-status", "--run-id", run.run_id)
    write_json_atomic(run.case_dir / "holder-status-cleanup.json", status)
    report = run.holder_probe.execute("disarm-holder", "--run-id", run.run_id)
    return {
        "disarmed": "by-probe",
        "reachability": reach,
        "status": status,
        "report": report,
    }


def _refuse_residual(residuals: dict[str, bool]) -> dict[str, bool]:
    if any(residuals.values()):
        raise RegionalFixtureError(f"residual resources remain: {residuals}")
    return residuals


def _restore_isolated_node(run: _LiveRun) -> dict[str, Any]:
    """A failed or exhausted record leaves the node cordoned and owned. Never
    delete the taint by hand: restore through the validation-first workflow."""

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
    owner = str(
        (snapshot.get("ownership_annotations") or {}).get("gpu-fault.io/incident-id")
        or run.incident_id
    )
    run.warm.wait_agent_active(settings.node)
    if owner != run.incident_id:
        run.warm.wait_incident_idle(owner)
    profile_version = str(
        (run.preflight["store"].get("profile") or {}).get("profile_version") or ""
    )
    created = run.warm.create_restore_workflow(
        incident_id=owner,
        node=settings.node,
        profile_version=profile_version,
        reason="DESTR-025 validated cleanup",
    )
    restored = run.warm.wait_workflow_id(str(created["workflow_request_id"]))
    if restored.get("status") != "SUCCEEDED":
        raise RegionalFixtureError(
            f"{settings.node} restore workflow did not succeed: {restored.get('status')}"
        )
    return {"isolated": True, "owner": owner, "restore": restored.get("status")}


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
