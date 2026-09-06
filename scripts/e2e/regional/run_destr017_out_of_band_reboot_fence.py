#!/usr/bin/env python3
"""GF-REGIONAL-DESTR-017 live acceptance runner.

One idle GPU node takes a real ``/dev/kmsg`` XID 46, so the control plane opens
the ordinary RESET_GPU workflow: cordon, quiesce, verify no GPU clients, reset,
restore services, validate, restore scheduling. A GPU device holder opened by
the host probe *after* this drill's quiesce lands in the Node Agent ledger keeps
``VERIFY_NO_GPU_CLIENTS`` WAITING, so the workflow is holding a live maintenance
window on a node that has not been reset yet.

While it waits, the node is rebooted from outside the control plane: the probe
arms a bounded ``systemd-run --on-active`` timer that runs ``systemctl reboot``
and returns, because a reboot issued synchronously would kill the ``kubectl
exec`` channel the probe answers over. Nothing about that reboot is a workflow
step, a Node Agent command or a provider call.

The node comes back with a new boot id and its Node Agent re-registers with a
new generation. The generation the workflow pinned at quiesce time no longer
exists, so every in-flight maintenance command is refused: the waiting step
fails closed, the reset is never executed on the new boot, and no command of the
retired generation is ever recorded as a success. The generation and boot-id
fences had unit coverage only; no live case had ever changed a node's boot id
underneath an in-flight workflow.

The runner defaults to ``--plan``; ``--execute`` needs ``--confirm
DESTR017_EXECUTE``. The verdict functions are pure and unit-tested; the runner
records digests only.
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

from scripts.e2e.regional.acceptance_runner_common import (  # noqa: E402
    write_json_atomic,
)
from scripts.e2e.regional.destr017_verdicts import (  # noqa: E402
    EXPECTED_XID,
    FENCE_STEP_OPERATION,
    agent_errors,
    boot_errors,
    cloudtrail_errors,
    command_errors,
    estimated_duration_seconds,
    fence_evidence,
    holder_errors,
    host_final_errors,
    ledger_errors,
    lifetime_errors,
    preflight_errors,
    reboot_window_errors,
    reconcile_plan_errors,
    reset_journal_errors,
    schedulability_errors,
    step_transitions,
    successor_errors,
    workflow_errors,
)
from scripts.e2e.regional.host_probe_fixture import (  # noqa: E402
    HostProbeFixture,
    HostProbeSettings,
)
from scripts.e2e.regional.live_driver_guard import (  # noqa: E402
    add_live_arguments,
    authorize_execution,
    build_plan,
    install_site_profile,
)
from scripts.e2e.regional.regional_live_fixture import (  # noqa: E402
    RegionalFixtureError,
    RegionalLiveFixture,
    RegionalLiveSettings,
    install_abort_signals,
    predecessor_evidence,
    required,
    run_case_main,
    runtime_identity_errors,
    settings_from_arguments,
    waiting_step_executions,
)
from scripts.e2e.regional.warm_spare_fixture import (  # noqa: E402
    WarmSpareLiveFixture,
)

PROBE_SCRIPT = Path(__file__).with_name("probes") / "destructive_node_probe.py"
FENCE_PROBE_SCRIPT = Path(__file__).with_name("probes") / "destr017_node_probe.py"
CASE_ID = "GF-REGIONAL-DESTR-017"
# DESTR-002 is the only live case that has already watched this node go down and
# come back, and it is the case that establishes a real reboot is survivable
# here at all -- with the provider doing it, under a workflow that asked for it.
PREDECESSOR_CASE_ID = "GF-REGIONAL-DESTR-002"
CONFIRMATION = "DESTR017_EXECUTE"
EXPECTED_GPU_COUNT = 8
# The ledger row the device holder waits for. Quiesce, not verify: on a single
# node the holder has to be open before the *first* verify attempt, and quiesce
# sweeps device holders itself, so anything opened earlier is killed.
ARM_LEDGER_OPERATION = "QUIESCE_GPU_SERVICES"
DEFAULT_REBOOT_DELAY_SECONDS = 30
DEFAULT_MAX_HOLD_SECONDS = 1200
# Phase budgets (seconds).
QUIESCE_PIN_BUDGET_SECONDS = 900
WAITING_BUDGET_SECONDS = 600
NODE_RETURN_BUDGET_SECONDS = 900
AGENT_RETURN_BUDGET_SECONDS = 600
OBSERVATION_BUDGET_SECONDS = 2400


# --------------------------------------------------------------------------- #
# Control-plane probes (read-only, except the reconcile plan, which is a read)
# --------------------------------------------------------------------------- #
ESCALATION_PROBE = r"""
import json
import sys

from gpu_fault.app import ApplicationContext
from gpu_fault.store import NotFoundError

request_id = sys.argv[1]
store = ApplicationContext.from_environment().store


def pair(event_id):
    incident = store.get_incident_by_event(event_id)
    if incident is None:
        return None
    workflow = None
    if incident.workflow_request_id:
        try:
            workflow = store.get_workflow(incident.workflow_request_id)
        except NotFoundError:
            workflow = None
    return {
        "incident": incident.model_dump(mode="json"),
        "workflow": (
            workflow.model_dump(mode="json") if workflow is not None else None
        ),
    }


# The escalation ladder names its follow-on records deterministically:
# ``<rung>-after-<predecessor request id>``. Reading all four says both which
# rung was taken and that none of the hardware rungs was.
print(json.dumps({
    name: pair("%s-after-%s" % (name, request_id))
    for name in ("support", "reboot", "replace", "drain")
}, sort_keys=True, default=str))
"""

EXECUTOR_BOUNDS_PROBE = r"""
import json
import os

from gpu_fault.execution.config import ProductionExecutorConfig
from gpu_fault.models import WorkflowOperation

config = ProductionExecutorConfig.from_environment()
print(json.dumps({
    "node_workflow_lifetime_seconds": config.node_workflow_lifetime_seconds,
    "step_waiting_timeout_seconds": config.step_waiting_timeout_seconds,
    "verify_waiting_limit_seconds": config.step_waiting_limit(
        WorkflowOperation.VERIFY_NO_GPU_CLIENTS
    ),
    "restore_waiting_limit_seconds": config.step_waiting_limit(
        WorkflowOperation.RESTORE_GPU_SERVICES
    ),
    "agent_maintenance_window_seconds": int(
        os.getenv("GPU_FAULT_AGENT_MAINTENANCE_WINDOW_SECONDS", "420")
    ),
    "gpu_client_verify_max_attempts": int(
        os.getenv("GPU_FAULT_GPU_CLIENT_VERIFY_MAX_ATTEMPTS", "60")
    ),
}, sort_keys=True))
"""


def reconcile_plan_script() -> str:
    """The shipped retired-generation planner, plan mode only.

    ``gpu_fault.admin.workflow_reconcile`` ships the module's own source plus a
    stdin driver so an operator can plan against a runtime that predates it; this
    runner reuses the same source and calls the same entry point, but only ever
    the ``plan`` half. Applying a revocation is an operator decision and this
    case records NOT_APPLIED.
    """

    from gpu_fault import retired_generation

    source = Path(retired_generation.__file__).read_text(encoding="utf-8")
    return source + (
        "\n\nimport json as _json\n"
        "from gpu_fault.app import ApplicationContext as _Context\n"
        "_store = _Context.from_environment().store\n"
        "print(_json.dumps(build_retired_generation_plan(_store), "
        "sort_keys=True, default=str))\n"
    )


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested)
# --------------------------------------------------------------------------- #
def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str).encode()
    ).hexdigest()


def plan_identity(preflight: dict[str, Any], *, node: str) -> dict[str, Any]:
    snapshot = preflight.get("node") or {}
    store = preflight.get("store") or {}
    return {
        "release_id": preflight.get("release_id"),
        "node_uid": snapshot.get("uid"),
        "node_boot_id": snapshot.get("boot_id"),
        "agent_generation": (store.get("agent") or {}).get("generation"),
        "agent_incarnation_id": (store.get("agent") or {}).get("incarnation_id"),
        "runtime_profile_version": (store.get("profile") or {}).get("profile_version"),
        "runtime_identity": preflight.get("runtime_identity"),
        "node": node,
    }


def identity_digest(identity: dict[str, Any]) -> str:
    return _digest(identity)


def evidence_components(details: dict[str, Any]) -> dict[str, str]:
    return {name: _digest(value) for name, value in details.items()}


def case_digest(components: dict[str, str]) -> str:
    return _digest(components)


def derived_identity(run_dir: Path, attempt: int) -> str:
    """The run id, derived so a rerun of the same attempt reuses its probe Pods
    and its on-node marker file instead of orphaning them."""

    suffix = hashlib.sha256(
        f"{run_dir.resolve()}\0{attempt}\0{CASE_ID}".encode()
    ).hexdigest()[:12]
    return f"destr017-{suffix}-a{attempt}"


def maintenance_pin(workflow: dict[str, Any], *, node: str) -> dict[str, Any]:
    """The generation and window the succeeded quiesce pinned for ``node``.

    ``_decorate_success`` records both on the QUIESCE_GPU_SERVICES execution:
    ``agent_generations`` is exactly what the maintenance barrier compares every
    later step against, and ``maintenance_window_expires_at`` is the deadline it
    fails the step at. Reading them from the record, rather than recomputing
    them, is what makes the expected fence text derivable.
    """

    for item in workflow.get("step_executions") or []:
        if item.get("operation") != "QUIESCE_GPU_SERVICES":
            continue
        if item.get("status") != "SUCCEEDED":
            continue
        details = item.get("details") or {}
        generations = details.get("agent_generations") or {}
        if node not in generations:
            continue
        return {
            "pinned_generation": generations[node],
            "window_started_at": details.get("maintenance_window_started_at"),
            "window_expires_at": details.get("maintenance_window_expires_at"),
        }
    return {}


def window_remaining_seconds(
    pin: dict[str, Any],
    *,
    now: datetime,
) -> float | None:
    expires_at = pin.get("window_expires_at")
    if not expires_at:
        return None
    parsed = datetime.fromisoformat(str(expires_at).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return (parsed - now).total_seconds()


def expected_fence_text(pin: dict[str, Any], *, node: str) -> str:
    """The error the barrier will produce, spelled out for the evidence file.

    Recorded rather than asserted: which of the fail-closed refusals ends the
    waiting step depends on how the reboot raced the window, and the verdict
    accepts any of them as long as the generation fence appears in the terminal
    record.
    """

    generation = pin.get("pinned_generation")
    if not isinstance(generation, int):
        return ""
    return (
        f"maintenance agent fence failed for {node}: agent generation changed "
        f"from {generation} to {generation + 1}"
    )


# --------------------------------------------------------------------------- #
# Settings / plan
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Settings:
    regional: RegionalLiveSettings
    node: str
    host_probe_image: str
    pci_bdf: str
    reboot_delay_seconds: int
    max_hold_seconds: int
    predecessor_path: Path

    def environment(self) -> dict[str, str]:
        return {
            **self.regional.environment(),
            "GPU_FAULT_TARGET_NODE": self.node,
            "GPU_FAULT_HOST_PROBE_IMAGE": self.host_probe_image,
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
    if not 30 <= arguments.reboot_delay_seconds <= 600:
        raise RegionalFixtureError("reboot delay is outside 30..600 seconds")
    if not 60 <= arguments.max_hold_seconds <= 3600:
        raise RegionalFixtureError("device hold is outside 60..3600 seconds")
    return Settings(
        regional=settings_from_arguments(arguments),
        node=required(
            arguments.node or os.getenv("GPU_FAULT_TARGET_NODE", ""),
            "target node",
        ),
        host_probe_image=required(
            arguments.host_probe_image or os.getenv("GPU_FAULT_HOST_PROBE_IMAGE", ""),
            "host probe image",
        ),
        pci_bdf=arguments.pci_bdf.strip(),
        reboot_delay_seconds=int(arguments.reboot_delay_seconds),
        max_hold_seconds=int(arguments.max_hold_seconds),
        predecessor_path=predecessor,
    )


def plan_details(settings: Settings, preflight: dict[str, Any]) -> dict[str, Any]:
    identity = plan_identity(preflight, node=settings.node)
    return {
        "risk": "destructive",
        "predecessor": preflight.get("predecessor"),
        "node": settings.node,
        "reboot_delay_seconds": settings.reboot_delay_seconds,
        "max_hold_seconds": settings.max_hold_seconds,
        "control_env": preflight.get("control_env"),
        "mutation": (
            "write one XID 46 to the idle node's /dev/kmsg; let the RESET_GPU "
            "workflow cordon and quiesce the node; hold a GPU device open so "
            "VERIFY_NO_GPU_CLIENTS stays WAITING; while it waits, reboot the "
            "node from outside the control plane with a bounded systemd timer "
            "running systemctl reboot. One real OS reboot of one node. No "
            "provider call, no reset, no replacement; the reset is what the "
            "generation fence must refuse on the new boot."
        ),
        "preflight_identity": identity,
        "preflight_identity_digest": identity_digest(identity),
        "stop_conditions": [
            "DESTR-002 predecessor evidence is not PASS",
            "the node is busy, tainted, owned, or its Node Agent is not ACTIVE",
            "the node already holds a quiesce state file or a GPU client",
            "an earlier run left an armed reboot timer on the node",
            "the estimated duration does not fit the node workflow lifetime",
            "the quiesce never pins a maintenance generation and window",
            "VERIFY_NO_GPU_CLIENTS never reaches WAITING, so there is nothing "
            "in flight for the reboot to interrupt",
            "the reboot would fire after the pinned window or the per-step cap",
            "the node or its Node Agent does not come back",
            "the Agent generation does not advance by exactly one, or the boot "
            "id changes more than once",
            "any RESET_GPU is executed, dispatched or recorded on the new boot",
            "any command of the retired generation is recorded SUCCEEDED",
            "the workflow does not fail closed, or a reboot/replace successor "
            "is opened for a reboot the control plane did not order",
            "CloudTrail shows any provider mutation",
            "cleanup cannot restore the node through the validated-restore "
            "workflow, or leaves any residue",
        ],
        "rollback": {
            "runner_cancels_an_unfired_reboot_timer": True,
            "runner_disarms_the_device_holder_idempotently": True,
            "runner_waits_for_node_Ready_and_agent_ACTIVE": True,
            "runner_restores_isolation_only_via_validated_restore": True,
            "runner_clears_the_residual_quiesce_state_file_via_product_code": True,
            "runner_records_the_retired_generation_reconcile_as_NOT_APPLIED": True,
            "no_provider_reboot_or_replacement_is_authorized": True,
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
        "tests/execution/test_retired_generation.py",
        "tests/execution/test_abandoned_generation.py",
        "tests/regional/test_destr017_out_of_band_reboot_fence.py",
        "tests/regional/test_destr017_node_probe.py",
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
    """The deployed bounds this case has to fit, read from the executor itself.

    The release-state ConfigMap carries the job-DAG window a multi-node case
    needs; a single-node case needs the node workflow lifetime, the per-step
    waiting cap and the agent maintenance window, and those live in the
    executor's own configuration. Asking the running executor for them is the
    only reading that cannot drift from what will judge the case.
    """

    bounds = regional.executor_python(EXECUTOR_BOUNDS_PROBE)
    return {
        "node_lifetime_seconds": int(bounds["node_workflow_lifetime_seconds"]),
        "step_waiting_timeout_seconds": int(bounds["step_waiting_timeout_seconds"]),
        "verify_waiting_limit_seconds": int(bounds["verify_waiting_limit_seconds"]),
        "restore_waiting_limit_seconds": int(bounds["restore_waiting_limit_seconds"]),
        "agent_maintenance_window_seconds": int(
            bounds["agent_maintenance_window_seconds"]
        ),
        "gpu_client_verify_max_attempts": int(bounds["gpu_client_verify_max_attempts"]),
    }


def read_only_preflight(settings: Settings, case_dir: Path) -> dict[str, Any]:
    regional = RegionalLiveFixture(settings.regional)
    node_snapshot = regional.node_snapshot(settings.node)
    state = regional.store_snapshot(
        node=settings.node,
        observed_after=datetime.now(timezone.utc) - timedelta(hours=1),
    )
    runtime_identity = regional.runtime_identity()
    tests = focused_tests(case_dir)
    predecessor = predecessor_evidence(settings.predecessor_path, PREDECESSOR_CASE_ID)
    env = control_env(regional)
    run_id = derived_identity(case_dir.parents[1], 1)
    host = _host_probe(settings, run_id)
    fence = _fence_probe(settings, run_id)
    try:
        host.create()
        fence.create()
        host_snapshot = host.execute(
            "snapshot",
            "--since-epoch",
            str((datetime.now(timezone.utc) - timedelta(minutes=5)).timestamp()),
            "--pci-bdf",
            settings.pci_bdf or _first_bdf(host),
            timeout=180,
        )
        reboot_state = fence.execute("reboot-status", "--run-id", run_id, timeout=120)
    finally:
        residual = {
            "host": host.cleanup(),
            "fence": fence.cleanup(),
        }
    result: dict[str, Any] = {
        "release_id": state.get("release_id"),
        "node": node_snapshot,
        "store": state,
        "runtime_identity": runtime_identity,
        "focused_tests": tests,
        "cpu_blast": regional.cpu_blast_snapshot(),
        "predecessor": predecessor,
        "control_env": env,
        "host": host_snapshot,
        "reboot_state": reboot_state,
        "probe_residual": residual,
    }
    errors = preflight_errors(
        node=settings.node,
        node_snapshot=node_snapshot,
        agent=state.get("agent") or {},
        profile=state.get("profile") or {},
        business_workloads=regional.business_workloads(settings.node),
        queue=state.get("queue") or {},
        remote_commands=state.get("remote_commands") or {},
        recent_events=[state["event"]] if state.get("event") else [],
        host_snapshot=host_snapshot,
        reboot_status=reboot_state,
        expected_gpu_count=EXPECTED_GPU_COUNT,
    )
    errors.extend(runtime_identity_errors(runtime_identity))
    errors.extend(
        lifetime_errors(
            estimated_seconds=estimated_duration_seconds(),
            lifetime_seconds=env["node_lifetime_seconds"],
        )
    )
    if not predecessor.get("pass"):
        errors.append(f"{PREDECESSOR_CASE_ID} evidence is not PASS")
    if not tests["passed"]:
        errors.append("focused tests did not pass")
    if any(residual["host"].values()) or any(residual["fence"].values()):
        errors.append(f"probe residue remains after the preflight: {residual}")
    result["errors"] = errors
    write_json_atomic(case_dir / "preflight.json", result)
    return result


def _first_bdf(host: HostProbeFixture) -> str:
    """The node's first GPU, when the operator named none.

    Read from the node's own inventory rather than defaulted, because site
    topology must never be hardcoded in a runner.
    """

    snapshot = host.execute("snapshot", timeout=180)
    inventory = snapshot.get("gpu_inventory") or []
    if not inventory:
        raise RegionalFixtureError("host probe found no GPU inventory")
    return str(inventory[0]["pci_bdf"])


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Run the guarded DESTR-017 acceptance: an out-of-band reboot during "
            "a WAITING RESET_GPU workflow, and the generation fence that must "
            "refuse the old generation's commands on the new boot."
        )
    )
    add_live_arguments(value, confirmation=CONFIRMATION)
    value.add_argument("--cpu-kubeconfig", default="")
    value.add_argument("--gpu-kubeconfig", default="")
    value.add_argument("--gpu-context", default="")
    value.add_argument("--namespace", default="gpu-fault-system")
    value.add_argument("--cluster-id", default="")
    value.add_argument("--region", default="")
    value.add_argument("--node", default="", help="the idle GPU node to fault")
    value.add_argument("--host-probe-image", default="")
    value.add_argument(
        "--pci-bdf",
        default="",
        help="GPU PCI BDF to fault; defaults to the node's first inventory entry",
    )
    value.add_argument(
        "--reboot-delay-seconds",
        type=int,
        default=DEFAULT_REBOOT_DELAY_SECONDS,
        help=(
            "seconds between arming and the out-of-band reboot; must leave the "
            "arming exec time to return and still land inside the pinned "
            "maintenance window (30..600)"
        ),
    )
    value.add_argument(
        "--max-hold-seconds",
        type=int,
        default=DEFAULT_MAX_HOLD_SECONDS,
        help="bounded lifetime of the GPU device holder (60..3600)",
    )
    value.add_argument("--predecessor-evidence", default="")
    return value


def main() -> int:
    install_site_profile()
    arguments = parser().parse_args()
    os.umask(0o077)
    install_abort_signals()
    settings = configure(arguments)
    case_dir = arguments.run_dir / "cases" / CASE_ID
    case_dir.mkdir(parents=True, exist_ok=True)
    if not arguments.execute:
        preflight = read_only_preflight(settings, case_dir)
        plan = build_plan(
            run_dir=arguments.run_dir,
            case_id=CASE_ID,
            attempt=arguments.attempt,
            confirmation=CONFIRMATION,
            environment=settings.environment(),
            details=plan_details(settings, preflight),
        )
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0 if not preflight["errors"] else 1
    deadline = authorize_execution(
        arguments,
        case_id=CASE_ID,
        confirmation=CONFIRMATION,
        environment=settings.environment(),
    )
    return execute_case(settings, arguments.run_dir, arguments.attempt, deadline)


# --------------------------------------------------------------------------- #
# Live execution
# --------------------------------------------------------------------------- #
def _host_probe(settings: Settings, run_id: str) -> HostProbeFixture:
    return HostProbeFixture(
        HostProbeSettings(
            kubeconfig=settings.regional.gpu_kubeconfig,
            context=settings.regional.gpu_context,
            namespace=settings.regional.namespace,
            node=settings.node,
            image=settings.host_probe_image,
            case_id=CASE_ID,
            run_id=run_id,
            probe_script=PROBE_SCRIPT,
            active_deadline_seconds=3600,
        )
    )


def _fence_probe(settings: Settings, run_id: str) -> HostProbeFixture:
    """A second Pod on the same node for this case's own probe.

    The fixture names its Pod after (case, run, node), so the fence probe takes
    a distinct run id suffix; sharing one Pod would mean one probe script, and
    the injection probe is shared code this case must not grow a reboot into.
    """

    return HostProbeFixture(
        HostProbeSettings(
            kubeconfig=settings.regional.gpu_kubeconfig,
            context=settings.regional.gpu_context,
            namespace=settings.regional.namespace,
            node=settings.node,
            image=settings.host_probe_image,
            case_id=CASE_ID,
            run_id=f"{run_id}-fence",
            probe_script=FENCE_PROBE_SCRIPT,
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
    host: HostProbeFixture
    fence: HostProbeFixture
    marker: str = ""
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    bdf: str = ""
    baseline: dict[str, Any] = field(default_factory=dict)
    baseline_boot_id: str = ""
    baseline_agent: dict[str, Any] = field(default_factory=dict)
    pin: dict[str, Any] = field(default_factory=dict)
    holder_armed: bool = False
    reboot_armed: bool = False
    reboot_fired: bool = False
    incident_id: str = ""
    successor_incident_id: str = ""
    workflow_request_id: str = ""


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
        raise RegionalFixtureError("DESTR-017 plan identity drifted before execution")
    regional = RegionalLiveFixture(settings.regional)
    run_id = derived_identity(run_dir, attempt)
    return _LiveRun(
        settings=settings,
        case_dir=case_dir,
        attempt=attempt,
        maintenance_window_end=maintenance_window_end,
        preflight=preflight,
        regional=regional,
        warm=WarmSpareLiveFixture(regional, ""),
        run_id=run_id,
        host=_host_probe(settings, run_id),
        fence=_fence_probe(settings, run_id),
    )


def _start_probes(run: _LiveRun) -> None:
    """Baseline the node, then arm the holder *before* the injection.

    The holder opens itself on this drill's quiesce ledger row, and quiesce is
    two steps after the injection, so the watcher has to be running before the
    XID is written. It cannot be armed any earlier than the baseline either: the
    baseline is what proves the node had no GPU client of its own.
    """

    settings = run.settings
    run.host.create()
    run.fence.create()
    run.bdf = settings.pci_bdf or _first_bdf(run.host)
    run.baseline = run.host.execute(
        "snapshot",
        "--since-epoch",
        str(run.started_at.timestamp()),
        "--pci-bdf",
        run.bdf,
        timeout=180,
    )
    write_json_atomic(run.case_dir / "host-before.json", run.baseline)
    run.baseline_boot_id = str(run.baseline.get("boot_id") or "")
    run.baseline_agent = dict(run.preflight["store"].get("agent") or {})
    armed = run.fence.execute(
        "arm-holder",
        "--device",
        _holder_device(run),
        "--drill-id",
        run.run_id,
        "--after-ledger-op",
        ARM_LEDGER_OPERATION,
        "--max-hold-seconds",
        str(settings.max_hold_seconds),
        "--run-id",
        run.run_id,
        "--probe-script",
        f"/host/run/gpu-fault-host-probe-{run.fence.pod.rsplit('-', 1)[-1]}.py",
        timeout=180,
    )
    run.holder_armed = True
    write_json_atomic(run.case_dir / "holder-armed.json", armed)


def _holder_device(run: _LiveRun) -> str:
    """The ``/dev/nvidiaN`` node of the GPU this case faults.

    Taken from the node's inventory, not from an index the runner assumes: the
    holder has to be on a device the node really has, and the probe refuses
    anything that is not an allow-listed GPU device node.
    """

    for entry in run.baseline.get("gpu_inventory") or []:
        if str(entry.get("pci_bdf")) == run.bdf:
            index = entry.get("index")
            if index is None:
                break
            return f"/dev/nvidia{int(index)}"
    raise RegionalFixtureError(
        f"the node's GPU inventory does not name a device index for {run.bdf}"
    )


def _inject(run: _LiveRun) -> dict[str, Any]:
    run.marker = f"{run.run_id}-{int(time.time())}"
    written = run.host.execute(
        "write-xid",
        "--xid",
        str(EXPECTED_XID),
        "--marker",
        run.marker,
        "--drill-id",
        run.run_id,
        "--pci-bdf",
        run.bdf,
        "--case-id",
        CASE_ID,
        timeout=120,
    )
    write_json_atomic(run.case_dir / "injection.json", written)
    return written


def _store_state(run: _LiveRun, *, queue_attempts: int = 1) -> dict[str, Any]:
    return run.regional.store_snapshot(
        node=run.settings.node,
        marker=run.marker,
        observed_after=run.started_at,
        queue_attempts=queue_attempts,
    )


def _wait_for_maintenance_pin(run: _LiveRun) -> dict[str, Any]:
    """Wait until the quiesce pins a generation and a window for this node."""

    deadline = time.monotonic() + QUIESCE_PIN_BUDGET_SECONDS
    state: dict[str, Any] = {}
    while time.monotonic() < deadline:
        state = _store_state(run)
        workflow = state.get("workflow") or {}
        pin = maintenance_pin(workflow, node=run.settings.node)
        if pin:
            run.pin = pin
            write_json_atomic(run.case_dir / "maintenance-pin.json", pin)
            return state
        if workflow.get("status") in {"SUCCEEDED", "FAILED", "BLOCKED", "SUPERSEDED"}:
            break
        time.sleep(5)
    raise RegionalFixtureError(
        "the workflow never pinned a maintenance generation for this node; "
        f"last state: {(state.get('workflow') or {}).get('status')}"
    )


def _wait_for_waiting_verify(run: _LiveRun) -> dict[str, Any]:
    """Wait until the device holder has the verify step WAITING.

    This is the only moment the reboot is worth injecting: a command of the
    pinned generation is in flight, and the node has not been reset.
    """

    deadline = time.monotonic() + WAITING_BUDGET_SECONDS
    state: dict[str, Any] = {}
    while time.monotonic() < deadline:
        state = _store_state(run)
        workflow = state.get("workflow") or {}
        waiting = [
            item
            for item in waiting_step_executions(workflow)
            if item.get("operation") == FENCE_STEP_OPERATION
        ]
        if waiting:
            write_json_atomic(
                run.case_dir / "waiting-verify.json", {"executions": waiting}
            )
            return state
        if workflow.get("status") in {"SUCCEEDED", "FAILED", "BLOCKED", "SUPERSEDED"}:
            break
        time.sleep(5)
    raise RegionalFixtureError(
        f"{FENCE_STEP_OPERATION} never reached WAITING; the device holder did "
        "not hold and there was nothing in flight to fence"
    )


def _arm_out_of_band_reboot(run: _LiveRun) -> tuple[list[str], dict[str, Any]]:
    """Arm the reboot, refusing a placement that would prove another fence."""

    now = datetime.now(timezone.utc)
    remaining = window_remaining_seconds(run.pin, now=now)
    errors = reboot_window_errors(
        delay_seconds=run.settings.reboot_delay_seconds,
        window_remaining_seconds=remaining,
        step_waiting_limit_seconds=int(
            run.preflight["control_env"]["verify_waiting_limit_seconds"]
        ),
    )
    if errors:
        raise RegionalFixtureError("; ".join(errors))
    armed = run.fence.execute(
        "arm-reboot",
        "--run-id",
        run.run_id,
        "--delay-seconds",
        str(run.settings.reboot_delay_seconds),
        timeout=120,
    )
    run.reboot_armed = True
    record = {
        "armed": armed,
        "window_remaining_seconds": remaining,
        "expected_fence": expected_fence_text(run.pin, node=run.settings.node),
    }
    write_json_atomic(run.case_dir / "reboot-armed.json", record)
    return [], record


def _wait_for_new_generation(run: _LiveRun) -> dict[str, Any]:
    """The node returns on a new boot and its Agent re-registers.

    Waiting for the *generation* rather than only for Ready is the point: the
    fence compares generations, so the case has not injected its fault until the
    Agent has published a higher one.
    """

    node_after = run.regional.wait_node_ready(
        run.settings.node,
        timeout_seconds=NODE_RETURN_BUDGET_SECONDS,
        expected_boot_id=run.baseline_boot_id,
    )
    run.reboot_fired = True
    write_json_atomic(run.case_dir / "node-after-reboot.json", node_after)
    baseline_generation = run.baseline_agent.get("generation")
    deadline = time.monotonic() + AGENT_RETURN_BUDGET_SECONDS
    agent: dict[str, Any] = {}
    while time.monotonic() < deadline:
        agent = _store_state(run).get("agent") or {}
        if (
            agent.get("lifecycle_state") == "ACTIVE"
            and isinstance(agent.get("generation"), int)
            and isinstance(baseline_generation, int)
            and agent["generation"] > baseline_generation
        ):
            write_json_atomic(run.case_dir / "agent-after-reboot.json", agent)
            return agent
        time.sleep(5)
    raise RegionalFixtureError(
        f"the Node Agent did not re-register a new generation: {agent}"
    )


def _observe_until_terminal(run: _LiveRun) -> dict[str, Any]:
    transitions: dict[str, str] = {}
    timeline: list[dict[str, Any]] = []
    deadline = time.monotonic() + OBSERVATION_BUDGET_SECONDS
    state: dict[str, Any] = {}
    while time.monotonic() < deadline:
        state = _store_state(run)
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


def _escalations(run: _LiveRun, request_id: str) -> dict[str, Any]:
    result = run.regional.cpu_python(ESCALATION_PROBE, request_id)
    write_json_atomic(run.case_dir / "escalations.json", result)
    return result


def _reconcile_plan(run: _LiveRun) -> dict[str, Any]:
    """Run the shipped retired-generation reconcile in plan mode only."""

    plan = run.regional.cpu_python(reconcile_plan_script())
    record = {"applied": False, "application": "NOT_APPLIED", **plan}
    write_json_atomic(run.case_dir / "reconcile-plan.json", record)
    return record


def _data_plane_errors(run: _LiveRun) -> tuple[list[str], dict[str, Any]]:
    """Read the node back after the reboot. The probe Pods died with the node,
    so they are re-created first; the on-host marker file and the Node Agent
    ledger are what carried the evidence across the boot."""

    errors: list[str] = []
    run.host.create()
    run.fence.create()
    after = run.host.execute(
        "snapshot",
        "--since-epoch",
        str(run.started_at.timestamp()),
        "--pci-bdf",
        run.bdf,
        timeout=180,
    )
    write_json_atomic(run.case_dir / "host-after.json", after)
    reboot_state = run.fence.execute(
        "reboot-status", "--run-id", run.run_id, timeout=120
    )
    write_json_atomic(run.case_dir / "reboot-status.json", reboot_state)
    holder = run.fence.execute("holder-status", "--run-id", run.run_id, timeout=120)
    write_json_atomic(run.case_dir / "holder-status.json", holder)
    errors.extend(ledger_errors(run.baseline, after, node=run.settings.node))
    errors.extend(reset_journal_errors(after, node=run.settings.node))
    errors.extend(holder_errors(holder, node=run.settings.node))
    node_after = run.regional.node_snapshot(run.settings.node)
    errors.extend(
        boot_errors(
            node=run.settings.node,
            baseline_boot_id=run.baseline_boot_id,
            final_boot_id=str(node_after.get("boot_id") or ""),
            host_boot_ids=list(reboot_state.get("observed_boot_ids") or []),
            reboot_status=reboot_state,
        )
    )
    provider = run.regional.provider_events(run.started_at, datetime.now(timezone.utc))
    write_json_atomic(run.case_dir / "provider-events.json", {"events": provider})
    errors.extend(cloudtrail_errors(provider))
    return errors, {
        "host_after": after,
        "reboot_status": reboot_state,
        "holder_status": holder,
        "node_after": node_after,
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
        "reconcile_application": "NOT_APPLIED",
    }
    try:
        _start_probes(run)
        _inject(run)
        _wait_for_maintenance_pin(run)
        _wait_for_waiting_verify(run)
        _, reboot_record = _arm_out_of_band_reboot(run)
        agent_after = _wait_for_new_generation(run)
        state = _observe_until_terminal(run)
        workflow = state.get("workflow") or {}
        incident = state.get("incident") or {}
        run.incident_id = str(incident.get("incident_id") or "")
        run.workflow_request_id = str(workflow.get("request_id") or "")
        evidence = fence_evidence(workflow)
        write_json_atomic(run.case_dir / "fence-evidence.json", evidence)
        errors = workflow_errors(workflow, incident, node=settings.node)
        errors.extend(command_errors(state.get("commands") or [], node=settings.node))
        errors.extend(agent_errors(run.baseline_agent, agent_after, node=settings.node))
        escalations = _escalations(run, run.workflow_request_id)
        support = escalations.get("support") or {}
        run.successor_incident_id = str(
            (support.get("incident") or {}).get("incident_id") or ""
        )
        errors.extend(
            successor_errors(
                support.get("workflow") or {},
                support.get("incident") or {},
                node=settings.node,
                predecessor_request_id=run.workflow_request_id,
                forbidden_escalations={
                    name: bool(escalations.get(name))
                    for name in ("reboot", "replace", "drain")
                },
                compensation_failed="FAILED"
                in (evidence.get("compensation_statuses") or []),
            )
        )
        data_errors, hosts = _data_plane_errors(run)
        errors.extend(data_errors)
        reconcile = _reconcile_plan(run)
        errors.extend(
            reconcile_plan_errors(
                reconcile,
                request_ids={
                    run.workflow_request_id,
                    str((support.get("workflow") or {}).get("request_id") or ""),
                },
            )
        )
        details = {
            "preflight_identity": plan_identity(run.preflight, node=settings.node),
            "workflow": workflow,
            "incident": incident,
            "support": support,
            "fence_evidence": evidence,
            "hosts": hosts,
        }
        components = evidence_components(details)
        result.update(
            {
                "verdict": "PASS" if not errors else "FAIL",
                "errors": errors,
                "marker": run.marker,
                "incident_id": run.incident_id,
                "successor_incident_id": run.successor_incident_id,
                "workflow_request_id": run.workflow_request_id,
                "maintenance_pin": run.pin,
                "reboot": reboot_record,
                "fence_evidence": evidence,
                "reconcile_plan_sha256": reconcile.get("plan_sha256"),
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
    """Undo the case in the order the node can survive.

    Anything armed comes down first (an unfired reboot timer is the one residue
    that could fire into cleanup), then the node is waited back, then isolation
    is released through the validated-restore workflow -- never by deleting a
    taint -- and finally the quiesce state file the reboot orphaned is cleared
    by the product's own restore path.
    """

    result: dict[str, Any] = {"errors": []}

    def guard(label: str, action: Any) -> None:
        try:
            result[label] = action()
        except Exception as exc:  # noqa: BLE001 - a cleanup failure is a FAIL
            result["errors"].append(f"{label}: {type(exc).__name__}: {exc}")

    if run.reboot_armed and not run.reboot_fired:
        guard(
            "cancel_reboot",
            lambda: run.fence.execute(
                "cancel-reboot", "--run-id", run.run_id, timeout=120
            ),
        )
    if run.holder_armed:
        guard(
            "disarm_holder",
            lambda: run.fence.execute(
                "disarm-holder", "--run-id", run.run_id, timeout=120
            ),
        )
    guard(
        "node_ready",
        lambda: run.regional.wait_node_ready(
            run.settings.node, timeout_seconds=NODE_RETURN_BUDGET_SECONDS
        ),
    )
    guard(
        "agent_active",
        lambda: run.warm.wait_agent_active(
            run.settings.node, timeout_seconds=AGENT_RETURN_BUDGET_SECONDS
        ),
    )
    guard("restore_isolation", lambda: _restore_isolation(run))
    guard(
        "quiesce_residue",
        lambda: run.host.execute(
            "restore-quiesce", "--incident-id", run.incident_id, timeout=180
        )
        if run.incident_id
        else {"skipped": "no incident"},
    )
    guard("host_final", lambda: _host_final(run))
    guard("probe_cleanup_host", lambda: _refuse_residual_map(run.host.cleanup()))
    guard("probe_cleanup_fence", lambda: _refuse_residual_map(run.fence.cleanup()))
    guard(
        "runtime_identity",
        lambda: run.regional.verify_runtime_identity(
            run.preflight["runtime_identity"],
            evidence_path=run.case_dir / "runtime-identity-after-cleanup.json",
            stage="after DESTR-017 cleanup",
        ),
    )
    return result


def _host_final(run: _LiveRun) -> dict[str, Any]:
    """The last read of the node, after everything has been undone."""

    final = run.host.execute(
        "snapshot",
        "--since-epoch",
        str(run.started_at.timestamp()),
        "--pci-bdf",
        run.bdf or "0000:00:00.0",
        timeout=180,
    )
    write_json_atomic(run.case_dir / "host-final.json", final)
    errors = host_final_errors(
        run.baseline,
        final,
        node=run.settings.node,
        expected_gpu_count=EXPECTED_GPU_COUNT,
    )
    snapshot = run.regional.node_snapshot(run.settings.node)
    write_json_atomic(run.case_dir / "node-final.json", snapshot)
    errors.extend(schedulability_errors(snapshot, node=run.settings.node))
    if errors:
        raise RegionalFixtureError("; ".join(errors))
    return {"node": snapshot.get("name"), "clean": True}


def _refuse_residual_map(residuals: dict[str, bool]) -> dict[str, bool]:
    if any(residuals.values()):
        raise RegionalFixtureError(f"residual resources remain: {residuals}")
    return residuals


def _restore_isolation(run: _LiveRun) -> dict[str, Any]:
    """Release the isolation through the validated-restore workflow.

    The support escalation takes the node's isolation annotations over from the
    fenced workflow and adds the quarantine taint, so the *successor* incident
    is the owner a restore has to name; the fenced incident is closed after it,
    on the no-op path, so its record does not stay QUARANTINED over a node
    nobody is isolating any more.
    """

    report: dict[str, Any] = {}
    profile_version = str(
        (run.preflight["store"].get("profile") or {}).get("profile_version") or ""
    )
    snapshot = run.regional.node_snapshot(run.settings.node)
    isolated = bool(
        snapshot.get("unschedulable")
        or snapshot.get("ownership_annotations")
        or any(
            str(taint.get("key") or "").startswith("gpu-fault.io/")
            for taint in snapshot.get("taints") or []
        )
    )
    report["isolated"] = isolated
    ordered = [item for item in (run.successor_incident_id, run.incident_id) if item]
    if isolated and not ordered:
        raise RegionalFixtureError("the node is isolated but no incident is known")
    for incident_id in ordered:
        run.warm.wait_incident_idle(incident_id)
        created = run.warm.create_restore_workflow(
            incident_id=incident_id,
            node=run.settings.node,
            profile_version=profile_version,
            reason="DESTR-017 validated cleanup",
        )
        restored = run.warm.wait_workflow_id(str(created["workflow_request_id"]))
        report[incident_id] = restored.get("status")
        if restored.get("status") != "SUCCEEDED":
            raise RegionalFixtureError(
                f"validated restore for {incident_id} did not succeed: "
                f"{restored.get('status')}"
            )
    return report


if __name__ == "__main__":
    raise SystemExit(run_case_main(main))
