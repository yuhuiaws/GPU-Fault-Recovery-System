#!/usr/bin/env python3
"""GF-REGIONAL-DESTR-018: the workflow lifetime is a hard deadline.

One idle GPU node, one compressed lifetime window, one device holder. A
RESET_GPU workflow that cannot get past ``VERIFY_NO_GPU_CLIENTS`` must lose the
whole workflow to its hard lifetime, cancel the command it had in flight, still
run the one compensation it owes, and hand the node to a human -- never climb
to a reboot or a replacement, and never let the node execute a mutation the
control plane already gave up on.

Order matters more here than in any other destructive case:

* The env window opens *before* injection. It rolls the control-worker
  Deployment, which moves the single-instance dispatch lease; doing that inside
  an open workflow re-elects the lease holder mid-lifetime and the timestamps
  this case measures stop meaning one thing.
* The holder is armed *before* injection. ``RESET_GPU`` has no WAITING branch,
  so a holder that appears late does not make the workflow wait -- it makes the
  reset fail and escalates to a reboot, which is a different case. The Agent's
  device-client check also needs the same pid holding the same device across
  every sample, which a holder armed mid-step cannot guarantee.
* The window closes only after quiescence, and the runtime-identity comparison
  is made against the *pre-window* baseline: the drill must give the control
  plane back exactly the deployment it borrowed.
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
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e2e.regional import control_plane_env_window as env_window  # noqa: E402
from scripts.e2e.regional import destr018_verdicts as verdicts  # noqa: E402
from scripts.e2e.regional.acceptance_runner_common import (  # noqa: E402
    write_json_atomic,
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
    required,
    run_case_main,
    runtime_identity_errors,
    settings_from_arguments,
)
from scripts.e2e.regional.warm_spare_fixture import (  # noqa: E402
    WarmSpareLiveFixture,
)

CASE_ID = verdicts.CASE_ID
PREDECESSOR_CASE_ID = verdicts.PREDECESSOR_CASE_ID
CONFIRMATION = "DESTR018_LIFETIME_DEADLINE"
HOLDER_PROBE_SCRIPT = Path(__file__).with_name("probes") / "destr018_node_probe.py"
INJECT_PROBE_SCRIPT = Path(__file__).with_name("probes") / "destructive_node_probe.py"
# The reset contract DESTR-001 proves end to end. This case must compile the
# same steps and then fail inside it, so the sequence is asserted, not the
# outcome of each step.
EXPECTED_STEPS = [
    "FREEZE_EVIDENCE",
    "MARK_UNSCHEDULABLE",
    "QUIESCE_GPU_SERVICES",
    "VERIFY_NO_GPU_CLIENTS",
    "RESET_GPU",
    "RESTORE_GPU_SERVICES",
    "VALIDATE_GPU",
    "RESTORE_SCHEDULING",
]
REQUIRED_AGENT_OPERATIONS = {
    "QUIESCE_GPU_SERVICES",
    "VERIFY_NO_GPU_CLIENTS",
    "RESET_GPU",
    "RESTORE_GPU_SERVICES",
}
POLL_INTERVAL_VARIABLE = "GPU_FAULT_WORKFLOW_POLL_INTERVAL_SECONDS"
STEP_TIMEOUT_VARIABLE = "GPU_FAULT_WORKFLOW_STEP_TIMEOUT_SECONDS"
# execution/config.py: ``poll_interval_seconds: float = 5.0`` and
# ``step_waiting_timeout_seconds: int = 600``. An unset variable is that
# default, and VERIFY_NO_GPU_CLIENTS has no per-operation waiting override.
DEFAULT_POLL_INTERVAL_SECONDS = 5.0
DEFAULT_STEP_CAP_SECONDS = 600
# The control worker's scrape port (deploy/control-plane/tools/
# render_control_plane_role_split.py sets prometheus.io/port=8081 and points the
# worker's local processor URL at the same port).
WORKER_METRICS_PORT = 8081
WORKER_APP = env_window.DEPLOYMENT
WORKFLOW_TIMEOUT_SECONDS = 1200
ABSORB_TIMEOUT_SECONDS = 300

METRICS_PROBE = r"""
import json
import sys
import urllib.request

port = int(sys.argv[1])
with urllib.request.urlopen(
    f"http://127.0.0.1:{port}/metrics", timeout=15
) as response:
    print(json.dumps({"metrics": response.read().decode()}))
"""

ESCALATION_CHAIN = r"""
import json
import sys

from gpu_fault.app import ApplicationContext
from gpu_fault.store import NotFoundError

store = ApplicationContext.from_environment().store
workflow_id = sys.argv[1]


def incident(incident_id):
    try:
        return store.get_incident(incident_id).model_dump(mode="json")
    except (NotFoundError, KeyError):
        return None


def workflow(request_id):
    try:
        return store.get_workflow(request_id).model_dump(mode="json")
    except (NotFoundError, KeyError):
        return None


first = f"workflow-support-after-{workflow_id}"
executable = [
    item.model_dump(mode="json")
    for item in store.list_workflows(limit=200)
    if item.status.value in {"PENDING", "RUNNING"}
]
print(
    json.dumps(
        {
            "incident": incident(f"inc-support-after-{workflow_id}"),
            "workflow": workflow(first),
            "second_order_incident": incident(f"inc-support-after-{first}"),
            "second_order_workflow": workflow(f"workflow-support-after-{first}"),
            "executable_workflows": executable,
        },
        sort_keys=True,
        default=str,
    )
)
"""


@dataclass(frozen=True)
class Settings:
    regional: RegionalLiveSettings
    node: str
    host_probe_image: str
    predecessor_path: Path
    lifetime_seconds: int
    execution_timeout_seconds: int
    step_timeout_seconds: int
    managed_recovery_seconds: int
    step_warning_seconds: int
    lease_duration_seconds: int
    hold_seconds: int

    def environment(self) -> dict[str, str]:
        return {
            **self.regional.environment(),
            "GPU_FAULT_TARGET_NODE": self.node,
            "GPU_FAULT_HOST_PROBE_IMAGE": self.host_probe_image,
            "GPU_FAULT_PREDECESSOR_EVIDENCE": str(self.predecessor_path),
            "GPU_FAULT_DESTR018_LIFETIME_SECONDS": str(self.lifetime_seconds),
            "GPU_FAULT_DESTR018_EXECUTION_TIMEOUT_SECONDS": str(
                self.execution_timeout_seconds
            ),
        }

    def assignments(self) -> dict[str, str]:
        """The full compressed knob set the window opens with.

        The node lifetime alone is not a bootable config: the control plane
        refuses a lifetime below the step ceilings, the managed-recovery window
        or the lease (execution/config.py), so all six are lowered together.
        ``env_window.assignment_errors`` re-checks the ordering.
        """

        return {
            env_window.LIFETIME_VARIABLE: str(self.lifetime_seconds),
            env_window.EXECUTION_TIMEOUT_VARIABLE: str(self.execution_timeout_seconds),
            env_window.STEP_TIMEOUT_VARIABLE: str(self.step_timeout_seconds),
            env_window.MANAGED_RECOVERY_VARIABLE: str(self.managed_recovery_seconds),
            env_window.STEP_WARNING_VARIABLE: str(self.step_warning_seconds),
            env_window.LEASE_DURATION_VARIABLE: str(self.lease_duration_seconds),
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
            arguments.node or os.getenv("GPU_FAULT_TARGET_NODE", ""),
            "target node",
        ),
        host_probe_image=required(
            arguments.host_probe_image or os.getenv("GPU_FAULT_HOST_PROBE_IMAGE", ""),
            "host probe image",
        ),
        predecessor_path=predecessor,
        lifetime_seconds=int(arguments.lifetime_seconds),
        execution_timeout_seconds=int(arguments.execution_timeout_seconds),
        step_timeout_seconds=int(arguments.step_timeout_seconds),
        managed_recovery_seconds=int(arguments.managed_recovery_seconds),
        step_warning_seconds=int(arguments.step_warning_seconds),
        lease_duration_seconds=int(arguments.lease_duration_seconds),
        hold_seconds=int(arguments.hold_seconds),
    )


def focused_tests(case_dir: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/orchestration/test_job_workflow_lifetime.py::"
        "test_a_single_node_workflow_past_its_lifetime_escalates_only_to_support",
        "tests/orchestration/test_job_workflow_lifetime.py::"
        "test_the_quiesce_undo_still_runs_after_the_lifetime_and_counts_once",
        "tests/orchestration/test_job_workflow_lifetime.py::"
        "test_events_after_the_lifetime_are_recorded_on_the_incident_not_replanned",
        "tests/execution/test_workflow_deadline_bounds.py::"
        "test_the_deadline_cancels_the_in_flight_remote_command_it_reports",
        "tests/execution/test_workflow_deadline_bounds.py::"
        "test_a_workflow_past_its_deadline_starts_only_the_restore_it_owes",
        "tests/node_agent/test_remediation.py::"
        "test_gpu_reset_refuses_active_compute_client",
    ]
    completed = RegionalLiveFixture.run(
        command,
        cwd=ROOT,
        check=False,
        timeout=300,
    )
    path = case_dir / "focused-tests.log"
    path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    path.chmod(0o600)
    return {
        "passed": completed.returncode == 0,
        "returncode": completed.returncode,
        "command": command,
    }


def capability(profile: dict[str, Any] | None, name: str) -> dict[str, Any] | None:
    for item in (profile or {}).get("capabilities", []):
        if isinstance(item, dict) and item.get("capability") == name:
            return cast(dict[str, Any], item)
    return None


def agreed_number(
    survey: dict[str, Any],
    name: str,
    default: float,
) -> float | None:
    """The value every ready worker replica agrees on, or ``None``.

    An unset variable is the *shipped default*, not an unknown. Replicas that
    disagree -- a half-finished rollout -- are unknown, and the case refuses
    rather than averaging two different bounds into a margin.
    """

    replicas = survey.get("replicas") or []
    if not replicas:
        return None
    values = {(replica.get("values") or {}).get(name) for replica in replicas}
    if len(values) != 1:
        return None
    single = values.pop()
    if single is None:
        return default
    try:
        return float(single)
    except (TypeError, ValueError):
        return None


def observed_cadence(survey: dict[str, Any]) -> float | None:
    return agreed_number(survey, POLL_INTERVAL_VARIABLE, DEFAULT_POLL_INTERVAL_SECONDS)


def observed_step_cap(survey: dict[str, Any]) -> int | None:
    """The pre-window waiting ceiling the deployed control plane reads.

    Recorded as evidence of the state the drill borrowed. It is *not* the cap
    the margin arithmetic is judged against: the env window compresses the step
    timeout to the lifetime for the run, so the margin is computed against
    ``settings.step_timeout_seconds``. This survey runs before the window opens,
    so on a clean deployment the variable is unset and this returns the shipped
    600s default; a ``None`` means ready replicas disagree, i.e. a half-rollout.
    """

    value = agreed_number(survey, STEP_TIMEOUT_VARIABLE, DEFAULT_STEP_CAP_SECONDS)
    return None if value is None else int(value)


def identity_errors(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    worker_generation_delta: int,
) -> list[str]:
    """Judge a runtime identity across a deliberate control-worker rollout.

    ``verify_runtime_identity`` compares the whole document, and the worker's
    ``generation`` is part of it -- so the plain check cannot be used across an
    env window that rolls that Deployment on purpose. Everything else still has
    to be identical, including the worker's ``template_sha256``: after the
    window closes, that digest returning to its pre-window value is the proof
    that the env was restored exactly and not merely to an equivalent-looking
    list. Only the two generation counters are allowed to move, and only by the
    number of writes the window made.
    """

    errors: list[str] = []
    if after.get("release_state") != before.get("release_state"):
        errors.append("regional release state identity drifted")
    counters = ("generation", "observed_generation")
    for plane, deployments in (before.get("deployments") or {}).items():
        current_plane = (after.get("deployments") or {}).get(plane) or {}
        for name, expected in deployments.items():
            observed = current_plane.get(name)
            if observed is None:
                errors.append(f"{plane} deployment {name} disappeared")
                continue
            if name != env_window.DEPLOYMENT:
                if observed != expected:
                    errors.append(f"{plane} deployment {name} identity drifted")
                continue
            stable = {
                key: value for key, value in observed.items() if key not in counters
            }
            if stable != {
                key: value for key, value in expected.items() if key not in counters
            }:
                errors.append(
                    f"{env_window.DEPLOYMENT} differs from the pre-window "
                    "baseline in something other than its generation"
                )
            expected_generation = int(expected["generation"]) + worker_generation_delta
            if int(observed["generation"]) != expected_generation:
                errors.append(
                    f"{env_window.DEPLOYMENT} generation is "
                    f"{observed['generation']}, not the {expected_generation} "
                    f"the {worker_generation_delta} env-window write(s) imply"
                )
            if int(observed["observed_generation"]) != int(observed["generation"]):
                errors.append(
                    f"{env_window.DEPLOYMENT} rollout is not settled: "
                    f"observedGeneration {observed['observed_generation']} lags "
                    f"generation {observed['generation']}"
                )
    errors.extend(runtime_identity_errors(after))
    return errors


def worker_metrics(regional: RegionalLiveFixture) -> list[str]:
    """The raw ``/metrics`` text of every ready control-worker replica."""

    samples: list[str] = []
    for pod in regional.ready_pods(env_window.PLANE, WORKER_APP):
        output = regional.kubectl(
            env_window.PLANE,
            "exec",
            "-i",
            str(pod["name"]),
            "--",
            "python3",
            "-",
            str(WORKER_METRICS_PORT),
            input_text=METRICS_PROBE,
            timeout=60,
        )
        samples.append(str(json.loads(output.splitlines()[-1])["metrics"]))
    return samples


def preflight_errors(
    settings: Settings,
    state: dict[str, Any],
    node: dict[str, Any],
    workloads: list[dict[str, str]],
    tests: dict[str, Any],
    survey: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    if node["ready"] != "True":
        errors.append("target node is not Ready")
    if node["unschedulable"]:
        errors.append("target node is already unschedulable")
    if node["taints"]:
        errors.append("target node has pre-existing taints")
    if workloads:
        errors.append("target node has non-system running Pods")
    agent = state.get("agent") or {}
    if agent.get("lifecycle_state") != "ACTIVE":
        errors.append("target Node Agent is not ACTIVE")
    if not REQUIRED_AGENT_OPERATIONS <= set(agent.get("allowed_operations") or []):
        errors.append("target Node Agent reset/quiesce allowlist is incomplete")
    reset = capability(state.get("profile"), "gpuReset")
    if reset is None:
        errors.append("runtime profile has no gpuReset capability")
    elif (
        reset.get("mode") != "OWN"
        or reset.get("owner") != "gpu-fault-node-agent"
        or reset.get("adapter") != "node-action"
    ):
        errors.append("gpuReset is not OWN by the Node Agent")
    if (state.get("profile") or {}).get("warnings"):
        errors.append("runtime profile has warnings")
    if int((state.get("queue") or {}).get("depth") or 0):
        errors.append("processor queue is not empty")
    if (state.get("remote_commands") or {}).get("open_by_cluster"):
        errors.append("remote command queue is not empty")
    if state.get("event") is not None:
        errors.append("target node has a recent XID event")
    if not tests["passed"]:
        errors.append("focused regression tests failed")
    errors.extend(env_window.assignment_errors(settings.assignments()))
    for name, item in ((survey.get("deployment") or {}).get("variables") or {}).items():
        if item.get("present"):
            errors.append(
                f"{name} is already set on {env_window.DEPLOYMENT}; an env window "
                "is open that this run did not record"
            )
    # The arithmetic that decides whether the drill can prove anything, judged
    # against the cadence the deployed control plane actually reads and the
    # in-window step cap the drill will run under (the window compresses the
    # step timeout to the lifetime), not the pre-window default.
    cap = observed_step_cap(survey)
    if cap is None:
        errors.append(
            f"{STEP_TIMEOUT_VARIABLE} is not one value every ready control "
            "worker replica agrees on; the step waiting ceiling is unknown"
        )
    errors.extend(
        verdicts.lifetime_margin_errors(
            lifetime_seconds=settings.lifetime_seconds,
            execution_timeout_seconds=settings.execution_timeout_seconds,
            cadence_seconds=observed_cadence(survey),
            step_waiting_cap_seconds=settings.step_timeout_seconds,
        )
    )
    return errors


def read_only_preflight(settings: Settings, case_dir: Path) -> dict[str, Any]:
    fixture = RegionalLiveFixture(settings.regional)
    state = fixture.store_snapshot(
        node=settings.node,
        observed_after=datetime.now(timezone.utc) - timedelta(minutes=10),
    )
    node = fixture.node_snapshot(settings.node)
    workloads = fixture.business_workloads(settings.node)
    survey = env_window.survey(fixture)
    tests = focused_tests(case_dir)
    predecessor = predecessor_evidence(settings.predecessor_path, PREDECESSOR_CASE_ID)
    errors = preflight_errors(settings, state, node, workloads, tests, survey)
    if not predecessor["valid"]:
        errors.append(f"{PREDECESSOR_CASE_ID} predecessor evidence is not PASS")
    cadence = observed_cadence(survey)
    result = {
        "release_id": state.get("release_id"),
        "node": node,
        "business_workloads": workloads,
        "store": state,
        "focused_tests": tests,
        "cpu_blast": fixture.cpu_blast_snapshot(),
        "predecessor": predecessor,
        "env_window_survey": survey,
        "observed_cadence_seconds": cadence,
        "observed_step_cap_seconds": observed_step_cap(survey),
        "timing": verdicts.timing_evidence(
            lifetime_seconds=settings.lifetime_seconds,
            execution_timeout_seconds=settings.execution_timeout_seconds,
            cadence_seconds=cadence,
        ),
        "runtime_identity": fixture.runtime_identity(),
        "errors": errors,
    }
    write_json_atomic(case_dir / "preflight.json", result)
    return result


def plan_identity(preflight: dict[str, Any]) -> dict[str, Any]:
    state = preflight["store"]
    return {
        "release_id": preflight["release_id"],
        "node_uid": preflight["node"]["uid"],
        "agent_generation": (state.get("agent") or {}).get("generation"),
        "runtime_profile_version": (state.get("profile") or {}).get("profile_version"),
    }


def plan_details(settings: Settings, preflight: dict[str, Any]) -> dict[str, Any]:
    return {
        "risk": "live-service-action",
        "predecessor": preflight["predecessor"],
        "target_node": settings.node,
        "mutation": (
            "open a temporary env window on the CPU-plane "
            f"{env_window.DEPLOYMENT} Deployment, compressing the workflow "
            f"lifetime to {settings.lifetime_seconds}s and lowering the "
            "execution timeout, step timeout, managed-recovery window, step "
            "warning and lease in lockstep so the control plane still boots "
            f"({json.dumps(settings.assignments(), sort_keys=True)}); "
            "hold one GPU device open "
            "with a transient systemd unit; write one synthetic XID 46 and one "
            "XID 79 to the real host /dev/kmsg; let the workflow quiesce GPU "
            "services and lose its lifetime while verifying clients"
        ),
        "preflight_identity": plan_identity(preflight),
        "timing": preflight["timing"],
        "stop_conditions": [
            f"{PREDECESSOR_CASE_ID} has not passed in formal sequence",
            "preflight or focused regression failure",
            "target node is not Ready/schedulable/idle",
            "Node Agent/Profile/pin or node UID drift",
            "an env window is already open on the control worker",
            "the compressed lifetime leaves no attempt margin under the "
            "observed redispatch cadence",
            "the device holder is not visible to the Node Agent before injection",
            "the workflow fails for a reason other than its lifetime",
            "the node starts any mutating action after the cancellation",
            "provider mutation appears",
            "cleanup, restore, window close or probe teardown fails",
        ],
        "rollback": {
            "holder_is_disarmed_before_any_restore": True,
            "holder_unit_has_its_own_bounded_lifetime": True,
            "isolation_is_released_only_by_a_validated_restore_workflow": True,
            "quarantine_taint_is_never_deleted_by_hand": True,
            "env_window_closes_to_the_recorded_baseline": True,
            "runtime_identity_is_compared_against_the_pre_window_baseline": True,
            "runner_finally_deletes_probe_resources": True,
        },
        "preflight": preflight,
    }


def verify_plan_identity(case_dir: Path, preflight: dict[str, Any]) -> None:
    plan = json.loads((case_dir / "plan.json").read_text(encoding="utf-8"))
    planned = plan["details"]["preflight_identity"]
    current = plan_identity(preflight)
    if current != planned:
        raise RegionalFixtureError(f"{CASE_ID} plan drifted: {planned} != {current}")


@dataclass
class _LiveRun:
    settings: Settings
    regional: RegionalLiveFixture
    case_dir: Path
    preflight: dict[str, Any]
    run_id: str
    holder: HostProbeFixture
    injector: HostProbeFixture
    window: env_window.Settings
    incident_id: str = ""
    support_incident_id: str = ""
    workflow_request_id: str = ""
    marker: str = ""
    injected_at: datetime | None = None
    holder_armed: bool = False
    window_opened: bool = False
    window_closed: bool = False
    profile_version: str = ""
    baseline_command_ids: set[str] = field(default_factory=set)
    in_window_identity: dict[str, Any] = field(default_factory=dict)
    target_bdf: str = ""
    device: str = ""
    metrics_before: dict[str, Any] = field(default_factory=dict)
    window_record: dict[str, Any] = field(default_factory=dict)
    t_cancel: datetime | None = None
    cadence_sample: dict[str, Any] = field(default_factory=dict)


def _probe(settings: Settings, run_id: str, script: Path) -> HostProbeFixture:
    return HostProbeFixture(
        HostProbeSettings(
            kubeconfig=settings.regional.gpu_kubeconfig,
            context=settings.regional.gpu_context,
            namespace=settings.regional.namespace,
            node=settings.node,
            image=settings.host_probe_image,
            case_id=CASE_ID,
            run_id=run_id,
            probe_script=script,
        )
    )


def _open_window(run: _LiveRun) -> dict[str, Any]:
    report = env_window.survey(run.regional)
    record = env_window.open_window(
        run.window,
        run.regional,
        report,
        run.settings.assignments(),
    )
    run.window_opened = True
    write_json_atomic(
        run.case_dir / "env-window-open.json",
        env_window.without_survey(record),
    )
    return record


def _arm_holder(run: _LiveRun, device: str) -> dict[str, Any]:
    armed = run.holder.execute(
        "arm-holder",
        "--device",
        device,
        "--drill-id",
        run.run_id,
        "--max-hold-seconds",
        str(run.settings.hold_seconds),
        "--run-id",
        run.holder.settings.run_id,
        "--probe-script",
        run.holder.host_script,
        timeout=120,
    )
    run.holder_armed = True
    write_json_atomic(run.case_dir / "holder-armed.json", armed)
    status = run.holder.execute(
        "holder-status",
        "--run-id",
        run.holder.settings.run_id,
        "--device",
        device,
        timeout=60,
    )
    write_json_atomic(run.case_dir / "holder-status.json", status)
    clients = status.get("device_clients") or []
    if not clients:
        raise RegionalFixtureError(
            f"the holder is not visible as a client of {device}; the Node Agent "
            "would not stay WAITING and the drill would reset a live GPU"
        )
    return status


def _absorb(run: _LiveRun, target_bdf: str) -> dict[str, Any]:
    """XID 79 on the same node, after the lifetime failure."""

    marker = f"{run.marker}-absorb"
    started = datetime.now(timezone.utc)
    injection = run.injector.execute(
        "write-xid79",
        "--marker",
        marker,
        "--drill-id",
        run.run_id,
        "--pci-bdf",
        target_bdf,
    )
    write_json_atomic(run.case_dir / "absorb-injection.json", injection)
    deadline = time.monotonic() + ABSORB_TIMEOUT_SECONDS
    snapshot: dict[str, Any] = {}
    while time.monotonic() < deadline:
        snapshot = run.regional.store_snapshot(
            node=run.settings.node,
            marker=marker,
            observed_after=started,
            queue_attempts=1,
        )
        if snapshot.get("event"):
            break
        time.sleep(5)
    write_json_atomic(run.case_dir / "absorb-state.json", snapshot)
    return snapshot


def _prepare_live_run(
    settings: Settings,
    run_dir: Path,
    attempt: int,
) -> _LiveRun:
    case_dir = run_dir / "cases" / CASE_ID
    case_dir.mkdir(parents=True, exist_ok=True)
    preflight = read_only_preflight(settings, case_dir)
    if preflight["errors"]:
        raise RegionalFixtureError(
            "preflight failed: " + "; ".join(preflight["errors"])
        )
    verify_plan_identity(case_dir, preflight)
    run_id = f"destr018-{run_dir.name.rsplit('-', 1)[-1].lower()}-a{attempt}"
    run = _LiveRun(
        settings=settings,
        regional=RegionalLiveFixture(settings.regional),
        case_dir=case_dir,
        preflight=preflight,
        run_id=run_id,
        # One script per fixture, and the fixture identity is a digest of
        # (case, run, node): two fixtures on one node need two run ids.
        holder=_probe(settings, f"{run_id}-hold", HOLDER_PROBE_SCRIPT),
        injector=_probe(settings, f"{run_id}-inject", INJECT_PROBE_SCRIPT),
        window=env_window.Settings(
            baseline=case_dir / "env-window-baseline.json",
            rollout_timeout_seconds=600,
        ),
    )
    run.profile_version = str(
        (preflight["store"].get("profile") or {}).get("profile_version") or ""
    )
    run.marker = f"destr018-{int(time.time())}-a{attempt}"
    return run


def _baseline_host(run: _LiveRun, maintenance_window_end: datetime) -> None:
    """Create both probes, snapshot the idle node and pick the target GPU."""

    run.injector.create()
    run.holder.create()
    baseline_host = run.injector.execute("snapshot")
    write_json_atomic(run.case_dir / "host-baseline.json", baseline_host)
    expected_gpu_count = int(run.preflight["node"]["gpu_allocatable"])
    inventory = baseline_host["gpu_inventory"]
    if len(inventory) != expected_gpu_count:
        raise RegionalFixtureError("host GPU inventory differs from Node allocatable")
    if baseline_host["compute_clients"]:
        raise RegionalFixtureError("target node has active NVIDIA compute clients")
    if baseline_host["quiesce_states"]:
        raise RegionalFixtureError("target node has a pre-existing quiesce state")
    if not baseline_host["kmsg_writable"]:
        raise RegionalFixtureError("/dev/kmsg is not writable from the host probe")
    run.baseline_command_ids = {
        str(item["command_id"]) for item in baseline_host.get("ledger") or []
    }
    target = inventory[0]
    run.target_bdf = str(target["pci_bdf"])
    run.device = f"/dev/nvidia{int(target['index'])}"
    if datetime.now(timezone.utc) >= maintenance_window_end:
        raise RegionalFixtureError("approved maintenance window ended before injection")


def _open_and_arm(run: _LiveRun) -> None:
    """The env window first (it rolls the worker Deployment), then the
    holder, both before the fault is written."""

    run.window_record = _open_window(run)
    in_window_identity = run.regional.runtime_identity()
    write_json_atomic(
        run.case_dir / "runtime-identity-in-window.json", in_window_identity
    )
    drift = identity_errors(
        run.preflight["runtime_identity"],
        in_window_identity,
        worker_generation_delta=1,
    )
    if drift:
        raise RegionalFixtureError(
            "the open env window changed more than the two managed "
            f"variables: {'; '.join(drift)}"
        )
    run.in_window_identity = in_window_identity
    run.metrics_before = worker_metrics(run.regional)
    _arm_holder(run, run.device)


def _inject_and_observe(run: _LiveRun) -> dict[str, Any]:
    run.injected_at = datetime.now(timezone.utc)
    injection = run.injector.execute(
        "write-xid46",
        "--marker",
        run.marker,
        "--drill-id",
        run.run_id,
        "--pci-bdf",
        run.target_bdf,
    )
    write_json_atomic(run.case_dir / "injection.json", injection)
    state = run.regional.wait_for_workflow(
        node=run.settings.node,
        marker=run.marker,
        observed_after=run.injected_at,
        case_dir=run.case_dir,
        timeout_seconds=WORKFLOW_TIMEOUT_SECONDS,
    )
    write_json_atomic(run.case_dir / "workflow-state.json", state)
    workflow = state.get("workflow") or {}
    incident = state.get("incident") or {}
    run.incident_id = str(incident.get("incident_id") or "")
    run.workflow_request_id = str(workflow.get("request_id") or "")
    return state


def _control_plane_errors(run: _LiveRun, state: dict[str, Any]) -> list[str]:
    workflow = state.get("workflow") or {}
    incident = state.get("incident") or {}
    commands = state.get("commands") or []
    errors: list[str] = []
    if (state.get("event") or {}).get("xid") != verdicts.EXPECTED_XID:
        errors.append(f"matched event is not XID {verdicts.EXPECTED_XID}")
    if [item.get("operation") for item in workflow.get("official_steps") or []] != (
        EXPECTED_STEPS
    ):
        errors.append("workflow step sequence differs from the reset contract")
    errors.extend(verdicts.workflow_errors(workflow, incident, node=run.settings.node))
    run.t_cancel = verdicts.cancellation_moment(commands)
    if run.t_cancel is None:
        errors.append(
            "no remote command records a cancellation moment; the deadline "
            "never reached the command queue"
        )
    else:
        errors.extend(verdicts.remote_command_errors(commands, t_cancel=run.t_cancel))
    return errors


def _host_snapshot(run: _LiveRun, name: str) -> dict[str, Any]:
    if run.injected_at is None:
        raise RegionalFixtureError("host snapshots follow the injection")
    snapshot = run.injector.execute(
        "snapshot",
        "--since-epoch",
        str(run.injected_at.timestamp()),
        "--pci-bdf",
        run.target_bdf,
        timeout=180,
    )
    write_json_atomic(run.case_dir / name, snapshot)
    return snapshot


def _data_plane_errors(run: _LiveRun, state: dict[str, Any]) -> list[str]:
    """Ledger, kernel journal and the cadence the run actually saw."""

    after_host = _host_snapshot(run, "host-after.json")
    ledger = after_host.get("ledger") or []
    errors: list[str] = []
    if run.t_cancel is not None:
        errors.extend(
            verdicts.data_plane_errors(
                ledger,
                t_cancel=run.t_cancel,
                baseline_command_ids=run.baseline_command_ids,
                kernel_journal=after_host.get("kernel_reset_journal") or {},
                commands=state.get("commands") or [],
            )
        )
    # Re-assert the arithmetic against the cadence the run actually saw.
    run.cadence_sample = verdicts.cadence_sample(
        ledger, baseline_command_ids=run.baseline_command_ids
    )
    cadence_errors = verdicts.cadence_errors(run.cadence_sample)
    errors.extend(cadence_errors)
    if not cadence_errors:
        errors.extend(
            verdicts.lifetime_margin_errors(
                lifetime_seconds=run.settings.lifetime_seconds,
                execution_timeout_seconds=run.settings.execution_timeout_seconds,
                cadence_seconds=float(run.cadence_sample["min_gap_seconds"]),
                step_waiting_cap_seconds=run.settings.step_timeout_seconds,
            )
        )
    return errors


def _escalation_and_absorb_errors(run: _LiveRun) -> list[str]:
    """The hand-off to the operator, the node's quarantine, and the later
    XID 79 that must be absorbed record-only."""

    regional, settings, case_dir = run.regional, run.settings, run.case_dir
    errors: list[str] = []
    escalation = regional.cpu_python(ESCALATION_CHAIN, run.workflow_request_id)
    write_json_atomic(case_dir / "escalation.json", escalation)
    run.support_incident_id = str(
        ((escalation.get("incident") or {}).get("incident_id")) or ""
    )
    errors.extend(verdicts.escalation_errors(escalation, node=settings.node))
    node_after = regional.node_snapshot(settings.node)
    write_json_atomic(case_dir / "node-after.json", node_after)
    errors.extend(verdicts.quarantine_errors(node_after))

    absorb = _absorb(run, run.target_bdf)
    errors.extend(
        verdicts.absorb_errors(
            absorb,
            incident_id=run.incident_id,
            workflow_request_id=run.workflow_request_id,
            official_step_count=len(EXPECTED_STEPS),
        )
    )
    absorb_chain = regional.cpu_python(ESCALATION_CHAIN, run.workflow_request_id)
    write_json_atomic(case_dir / "escalation-after-absorb.json", absorb_chain)
    errors.extend(
        verdicts.new_executable_workflow_errors(
            absorb_chain.get("executable_workflows") or [],
            known_request_ids={
                run.workflow_request_id,
                f"workflow-support-after-{run.workflow_request_id}",
            },
        )
    )
    absorb_ledger = _host_snapshot(run, "host-after-absorb.json")
    if run.t_cancel is not None:
        errors.extend(
            verdicts.late_row_errors(
                absorb_ledger.get("ledger") or [],
                t_cancel=run.t_cancel,
                baseline_command_ids=run.baseline_command_ids,
            )
        )
    return errors


def _provider_and_metric_errors(run: _LiveRun) -> tuple[list[str], dict[str, Any]]:
    if run.injected_at is None:
        raise RegionalFixtureError("provider events follow the injection")
    provider = run.regional.provider_events(run.injected_at, datetime.now(timezone.utc))
    write_json_atomic(run.case_dir / "provider-events.json", {"events": provider})
    errors: list[str] = []
    if provider:
        errors.append("provider mutation appeared during the lifetime drill")
    metrics = verdicts.metric_evidence(run.metrics_before, worker_metrics(run.regional))
    errors.extend(verdicts.metric_errors(metrics))
    return errors, metrics


def execute_case(
    settings: Settings,
    run_dir: Path,
    attempt: int,
    maintenance_window_end: datetime,
) -> int:
    """Drive the live case. Not exercised by the unit suite; the verdicts it
    calls are. Every cleanup failure downgrades the verdict to FAIL."""

    run = _prepare_live_run(settings, run_dir, attempt)
    result: dict[str, Any] = {
        "case_id": CASE_ID,
        "attempt": attempt,
        "verdict": "FAIL",
        "node": settings.node,
        "maintenance_window_end": maintenance_window_end.isoformat(),
    }
    try:
        _baseline_host(run, maintenance_window_end)
        _open_and_arm(run)
        state = _inject_and_observe(run)
        errors = _control_plane_errors(run, state)
        errors.extend(_data_plane_errors(run, state))
        errors.extend(_escalation_and_absorb_errors(run))
        provider_errors, metrics = _provider_and_metric_errors(run)
        errors.extend(provider_errors)
        result.update(
            {
                "verdict": "PASS" if not errors else "FAIL",
                "errors": errors,
                "marker": run.marker,
                "incident_id": run.incident_id,
                "support_incident_id": run.support_incident_id,
                "workflow_request_id": run.workflow_request_id,
                "injected_at": run.injected_at.isoformat() if run.injected_at else None,
                "cancelled_at": run.t_cancel.isoformat() if run.t_cancel else None,
                "cadence_sample": run.cadence_sample,
                "timing": verdicts.timing_evidence(
                    lifetime_seconds=settings.lifetime_seconds,
                    execution_timeout_seconds=settings.execution_timeout_seconds,
                    cadence_seconds=run.cadence_sample.get("min_gap_seconds"),
                ),
                "metrics": metrics,
                "env_window": env_window.without_survey(run.window_record),
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


def _refuse_residual_map(residuals: dict[str, bool]) -> dict[str, bool]:
    if any(residuals.values()):
        raise RegionalFixtureError(f"residual resources remain: {residuals}")
    return residuals


def _restore_isolated_node(run: _LiveRun) -> dict[str, Any]:
    """Release the isolation through a validated restore, never by hand.

    The taint is owned by whichever incident quarantined the node -- the
    support escalation if it got that far, otherwise the reset incident -- so
    both are offered to the validation-first workflow in that order.
    """

    snapshot = run.regional.node_snapshot(run.settings.node)
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
    candidates = [item for item in (run.support_incident_id, run.incident_id) if item]
    if not candidates:
        raise RegionalFixtureError("the node is isolated but no incident is known")
    warm = WarmSpareLiveFixture(run.regional, "")
    failures: list[str] = []
    for incident_id in candidates:
        try:
            warm.wait_incident_idle(incident_id)
            created = warm.create_restore_workflow(
                incident_id=incident_id,
                node=run.settings.node,
                profile_version=run.profile_version,
                reason=f"{CASE_ID} validated cleanup",
            )
        except Exception as exc:  # noqa: BLE001 - try the next owner
            failures.append(f"{incident_id}: {type(exc).__name__}: {exc}")
            continue
        restored = warm.wait_workflow_id(str(created["workflow_request_id"]))
        if restored.get("status") != "SUCCEEDED":
            raise RegionalFixtureError(
                f"{run.settings.node} restore workflow did not succeed: "
                f"{restored.get('status')}"
            )
        return {
            "isolated": True,
            "incident_id": incident_id,
            "restore": restored.get("status"),
            "refused": failures,
        }
    raise RegionalFixtureError(
        f"no incident could carry the validated restore: {failures}"
    )


def _close_window(run: _LiveRun) -> dict[str, Any]:
    record = env_window.close_window(
        run.window,
        run.regional,
        env_window.survey(run.regional),
    )
    run.window_closed = True
    return env_window.without_survey(record)


def _cleanup(run: _LiveRun) -> dict[str, Any]:
    result: dict[str, Any] = {"errors": []}

    def guard(label: str, action: Any) -> None:
        try:
            result[label] = action()
        except Exception as exc:  # noqa: BLE001 - a cleanup failure is a FAIL
            result["errors"].append(f"{label}: {type(exc).__name__}: {exc}")

    # The holder first: nothing else can succeed while a GPU device is held.
    if run.holder_armed:
        guard(
            "disarm_holder",
            lambda: run.holder.execute(
                "disarm-holder",
                "--run-id",
                run.holder.settings.run_id,
                timeout=120,
            ),
        )
    guard("restore_isolated_node", lambda: _restore_isolated_node(run))
    # Only now may the Deployment roll again.
    if run.window_opened:
        guard("close_env_window", lambda: _close_window(run))
    for label, probe in (("holder", run.holder), ("injector", run.injector)):
        guard(
            f"probe_cleanup_{label}",
            lambda probe=probe: _refuse_residual_map(probe.cleanup()),
        )
    guard("runtime_identity", lambda: _verify_restored_identity(run))
    guard("final_node", lambda: _final_node(run))
    return result


def _verify_restored_identity(run: _LiveRun) -> dict[str, Any]:
    """The deployment the drill borrowed, given back exactly.

    Two writes if the window was opened and closed, one if the run died with it
    open (which the close guard above has already recorded as a failure), none
    if it never opened.
    """

    current = run.regional.runtime_identity()
    write_json_atomic(
        run.case_dir / "runtime-identity-after-cleanup.json",
        current,
    )
    delta = 0
    if run.window_opened:
        delta = 2 if run.window_closed else 1
    errors = identity_errors(
        run.preflight["runtime_identity"],
        current,
        worker_generation_delta=delta,
    )
    if errors:
        raise RegionalFixtureError(
            f"after {CASE_ID} cleanup the runtime identity is not the one the "
            f"drill borrowed: {'; '.join(errors)}"
        )
    return current


def _final_node(run: _LiveRun) -> dict[str, Any]:
    snapshot = run.regional.node_snapshot(run.settings.node)
    if snapshot["ready"] != "True":
        raise RegionalFixtureError("target node is not Ready after cleanup")
    if snapshot["unschedulable"] or snapshot["ownership_annotations"]:
        raise RegionalFixtureError("target node scheduling ownership was not restored")
    if snapshot["taints"]:
        raise RegionalFixtureError(
            f"target node still carries taints: {snapshot['taints']}"
        )
    return snapshot


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Run the guarded DESTR-018 workflow lifetime hard-deadline "
            "acceptance on one idle GPU node."
        )
    )
    add_live_arguments(value, confirmation=CONFIRMATION)
    value.add_argument("--cpu-kubeconfig", default="")
    value.add_argument("--gpu-kubeconfig", default="")
    value.add_argument("--gpu-context", default="")
    value.add_argument("--namespace", default="gpu-fault-system")
    value.add_argument("--cluster-id", default="")
    value.add_argument("--node", default="")
    value.add_argument("--region", default="")
    value.add_argument("--host-probe-image", default="")
    value.add_argument("--predecessor-evidence", default="")
    value.add_argument(
        "--lifetime-seconds",
        type=int,
        default=verdicts.LIFETIME_SECONDS,
        help=(
            "compressed GPU_FAULT_NODE_WORKFLOW_MAX_LIFETIME_SECONDS; the "
            "runner refuses a value that leaves no verify-attempt margin"
        ),
    )
    value.add_argument(
        "--execution-timeout-seconds",
        type=int,
        default=verdicts.EXECUTION_TIMEOUT_SECONDS,
        help=(
            "GPU_FAULT_WORKFLOW_EXECUTION_TIMEOUT_SECONDS; must equal the "
            "lifetime -- below it the execution deadline fires first, above it "
            "claim_deadlines truncates it to the lifetime"
        ),
    )
    value.add_argument(
        "--step-timeout-seconds",
        type=int,
        default=verdicts.STEP_TIMEOUT_SECONDS,
        help=(
            "GPU_FAULT_WORKFLOW_STEP_TIMEOUT_SECONDS; the control plane refuses "
            "a step waiting ceiling above the lifetime, so this is compressed to "
            "the lifetime so the step's own cap cannot fire before the lifetime"
        ),
    )
    value.add_argument(
        "--managed-recovery-seconds",
        type=int,
        default=verdicts.MANAGED_RECOVERY_SECONDS,
        help=(
            "GPU_FAULT_HYPERPOD_MANAGED_RECOVERY_TIMEOUT_SECONDS; pinned to the "
            "step timeout (from_mapping forbids it below the step timeout, "
            "validate_timing forbids it above the lifetime)"
        ),
    )
    value.add_argument(
        "--step-warning-seconds",
        type=int,
        default=verdicts.STEP_WARNING_SECONDS,
        help=(
            "GPU_FAULT_WORKFLOW_STEP_WARNING_SECONDS; must stay below the step "
            "timeout (from_mapping)"
        ),
    )
    value.add_argument(
        "--lease-duration-seconds",
        type=int,
        default=verdicts.LEASE_DURATION_SECONDS,
        help=(
            "GPU_FAULT_WORKFLOW_LEASE_DURATION_SECONDS; must stay below the "
            "execution timeout (validate_timing_relationships)"
        ),
    )
    value.add_argument(
        "--hold-seconds",
        type=int,
        default=1200,
        help="bounded lifetime of the on-node device holder unit",
    )
    return value


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
