#!/usr/bin/env python3
"""GF-REGIONAL-DESTR-023 live acceptance runner.

A truly idle cluster -- no managed attempt, no coverage canary, every attempt
observation aged past ``GPU_FAULT_WORKLOAD_CONTEXT_FRESHNESS_SECONDS`` -- must
still resolve the target node IDLE through the completion watcher's coverage
heartbeat, so one synthetic XID 46 on an idle node runs the eight-step
RESET_GPU contract to SUCCEEDED instead of being BLOCKED with
``node workload state is UNKNOWN``.

The reset itself is DESTR-001's: the injection, the host assertions and the
workflow contract are that runner's functions, called unchanged. What this
case adds is the premise around it:

1. preflight refuses a cluster with any managed Pod or the coverage canary
   Job, and a control plane whose store has no
   ``get_workload_coverage_heartbeat``;
2. before the injection the runner waits until no attempt observation is
   younger than the freshness window while the heartbeat stays fresh (a
   completed full pass that saw nothing running, at most one per 120 s) and
   the node reads IDLE -- the coverage probe is recorded every 15 s;
3. after the run, ``blocked_reasons`` may not contain the UNKNOWN reason.

The runner defaults to ``--plan``; ``--execute`` needs ``--confirm
DESTR023_EXECUTE``. The verdict functions live in ``destr023_verdicts.py``
and are unit-tested; the runner records evidence only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e2e.regional import (  # noqa: E402
    run_destr001_gpu_reset as reset_case,
)
from scripts.e2e.regional.acceptance_runner_common import (  # noqa: E402
    write_json_atomic,
)
from scripts.e2e.regional.destr023_verdicts import (  # noqa: E402
    CASE_ID,
    CONFIRMATION,
    COVERAGE_CANARY_JOB,
    COVERAGE_POLL_SECONDS,
    MANAGED_LABEL,
    PREDECESSOR_CASE_ID,
    SETTLE_BUDGET_SECONDS,
    WATCHER_APP_LABEL,
    WATCHER_DEPLOYMENT,
    coverage_supported_errors,
    deployment_summary,
    expiry_wait_seconds,
    fresh_coverage_errors,
    idle_cluster_errors,
    managed_pod_summary,
    reset_errors,
    watcher_errors,
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
    settings_from_arguments,
)

PROBE_SCRIPT = Path(__file__).with_name("probes") / "destructive_node_probe.py"
WORKFLOW_BUDGET_SECONDS = 1200

# Read in an API Pod: the heartbeat record, the observation feed and the
# topology service's own answer for the target node, at one instant. A
# release before the heartbeat has no ``get_workload_coverage_heartbeat``;
# that is reported, not raised, so the preflight can name the missing deploy.
COVERAGE_PROBE = r"""
import json
import sys
from datetime import datetime, timezone

from gpu_fault.app import ApplicationContext

cluster_id, node_id = sys.argv[1:3]
context = ApplicationContext.from_environment()
store = context.store
now = datetime.now(timezone.utc)
supported = hasattr(store, "get_workload_coverage_heartbeat")
heartbeat = store.get_workload_coverage_heartbeat(cluster_id) if supported else None
observations = store.list_attempt_observations(cluster_id)
latest = max((item.observed_at for item in observations), default=None)
resolved = context.topology.resolve(cluster_id, node_id, now)
print(json.dumps({
    "heartbeat_supported": supported,
    "probed_at": now.isoformat(),
    "freshness_seconds": context.topology.freshness_seconds,
    "max_age_seconds": context.topology.max_age_seconds,
    "heartbeat": (
        heartbeat.model_dump(mode="json") if heartbeat is not None else None
    ),
    "heartbeat_age_seconds": (
        (now - heartbeat.observed_at).total_seconds()
        if heartbeat is not None else None
    ),
    "observation_count": len(observations),
    "latest_observation_at": latest.isoformat() if latest else None,
    "latest_observation_age_seconds": (
        (now - latest).total_seconds() if latest else None
    ),
    "workload_state": str(resolved.workload_state),
}, sort_keys=True, default=str))
"""


@dataclass(frozen=True)
class Settings:
    regional: RegionalLiveSettings
    node: str
    host_probe_image: str
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
    )


# --------------------------------------------------------------------------- #
# Cluster reads shared with DESTR-024
# --------------------------------------------------------------------------- #
def coverage_probe(fixture: RegionalLiveFixture, node: str) -> dict[str, Any]:
    return fixture.cpu_python(COVERAGE_PROBE, fixture.settings.cluster_id, node)


def managed_pods(fixture: RegionalLiveFixture) -> list[dict[str, Any]]:
    value = json.loads(
        fixture.kubectl(
            "gpu",
            "get",
            "pods",
            "-l",
            f"{MANAGED_LABEL}=true",
            "-o",
            "json",
            all_namespaces=True,
        )
    )
    return managed_pod_summary(value.get("items") or [])


def canary_jobs(fixture: RegionalLiveFixture) -> list[str]:
    value = json.loads(fixture.kubectl("gpu", "get", "jobs", "-o", "json"))
    return sorted(
        str(item["metadata"]["name"])
        for item in value.get("items") or []
        if str(item.get("metadata", {}).get("name", "")).startswith(COVERAGE_CANARY_JOB)
    )


def watcher_deployment(fixture: RegionalLiveFixture) -> dict[str, Any]:
    return deployment_summary(
        json.loads(
            fixture.kubectl(
                "gpu", "get", "deployment", WATCHER_DEPLOYMENT, "-o", "json"
            )
        )
    )


def watcher_pods(fixture: RegionalLiveFixture) -> list[dict[str, Any]]:
    value = json.loads(
        fixture.kubectl(
            "gpu", "get", "pods", "-l", f"app={WATCHER_APP_LABEL}", "-o", "json"
        )
    )
    return [
        {
            "name": item["metadata"]["name"],
            "phase": (item.get("status") or {}).get("phase"),
        }
        for item in value.get("items") or []
    ]


def wait_for_coverage(
    fixture: RegionalLiveFixture,
    node: str,
    *,
    case_dir: Path,
    judge: Any,
    budget_seconds: int,
    label: str,
) -> dict[str, Any]:
    """Poll the coverage probe until ``judge(sample)`` returns no errors.

    Every sample is appended to ``coverage-<label>-timeline.json``; the last
    one is returned. A budget that runs out raises with the last errors, so
    the case never injects against a premise it did not observe.
    """

    deadline = time.monotonic() + budget_seconds
    timeline: list[dict[str, Any]] = []
    while True:
        sample = coverage_probe(fixture, node)
        errors = judge(sample)
        timeline.append({**sample, "errors": errors})
        write_json_atomic(
            case_dir / f"coverage-{label}-timeline.json", {"entries": timeline}
        )
        if not errors:
            return sample
        if time.monotonic() >= deadline:
            raise RegionalFixtureError(
                f"coverage did not settle to the {label} premise within "
                f"{budget_seconds}s: {'; '.join(errors)}"
            )
        time.sleep(COVERAGE_POLL_SECONDS)


# --------------------------------------------------------------------------- #
# Preflight / plan
# --------------------------------------------------------------------------- #
def focused_tests(case_dir: Path, *, reuse: bool = False) -> dict[str, Any]:
    if reuse:
        recorded = reusable_focused_tests(case_dir / "plan.json")
        if recorded is not None:
            return {**recorded, "focused_tests_reused": True}
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/completion/test_workload_context_freshness.py::"
        "test_a_fresh_coverage_heartbeat_makes_an_unobserved_node_idle",
        "tests/completion/test_workload_context_freshness.py::"
        "test_no_observations_for_the_cluster_is_unknown_not_idle",
        "tests/app_services/test_coverage_heartbeat_route.py::"
        "test_a_stored_heartbeat_makes_the_cluster_idle_instead_of_unknown",
        "tests/node_agent/test_remediation.py::"
        "test_gpu_reset_is_idempotent_and_rechecks_clients",
        "tests/regional/test_destr023_idle_cluster_reset.py",
    ]
    completed = RegionalLiveFixture.run(command, cwd=ROOT, check=False, timeout=300)
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
    fixture = RegionalLiveFixture(settings.regional)
    identity = fixture.evidence_identity()
    state = fixture.store_snapshot(
        node=settings.node,
        observed_after=datetime.now(timezone.utc) - timedelta(minutes=10),
    )
    node = fixture.node_snapshot(settings.node)
    workloads = fixture.business_workloads(settings.node)
    coverage = coverage_probe(fixture, settings.node)
    pods = managed_pods(fixture)
    canaries = canary_jobs(fixture)
    watcher = watcher_deployment(fixture)
    tests = focused_tests(case_dir, reuse=reuse_focused_tests)
    predecessor = predecessor_evidence(
        settings.predecessor_path, PREDECESSOR_CASE_ID, **identity
    )
    errors = [
        *reset_case.preflight_errors(state, node, workloads, tests),
        *coverage_supported_errors(coverage),
        *idle_cluster_errors(pods, canaries),
        *watcher_errors(watcher),
    ]
    if not predecessor["valid"]:
        errors.append("DESTR-001 predecessor evidence is not PASS")
    result = {
        "release_id": state.get("release_id"),
        "evidence_identity": identity,
        "node": node,
        "business_workloads": workloads,
        "store": state,
        "coverage": coverage,
        "managed_pods": pods,
        "canary_jobs": canaries,
        "watcher": watcher,
        "focused_tests": tests,
        "cpu_blast": fixture.cpu_blast_snapshot(),
        "predecessor": predecessor,
        "errors": errors,
    }
    write_json_atomic(case_dir / "preflight.json", result)
    return result


def plan_identity(preflight: dict[str, Any]) -> dict[str, Any]:
    store = preflight.get("store") or {}
    return {
        "release_id": preflight.get("release_id"),
        "node_uid": (preflight.get("node") or {}).get("uid"),
        "agent_generation": (store.get("agent") or {}).get("generation"),
        "runtime_profile_version": (store.get("profile") or {}).get("profile_version"),
        "watcher_deployment_uid": (preflight.get("watcher") or {}).get("uid"),
    }


def plan_details(settings: Settings, preflight: dict[str, Any]) -> dict[str, Any]:
    details = {
        "risk": "destructive",
        "predecessor": preflight["predecessor"],
        "target_node": settings.node,
        "mutation": (
            "wait until no attempt observation is younger than the freshness "
            "window while the watcher heartbeat keeps the node IDLE; then write "
            "one synthetic XID 46 to the real host /dev/kmsg and allow the "
            "workflow to quiesce services and execute one real GPU reset"
        ),
        "preflight_identity": plan_identity(preflight),
        "coverage_premise": {
            "freshness_seconds": preflight["coverage"].get("freshness_seconds"),
            "expected_state_before_injection": "IDLE",
            "attributable_only_to_heartbeat": True,
        },
        "stop_conditions": [
            "DESTR-001 has not passed in formal sequence on this release",
            "control plane has no coverage heartbeat (deploy the heartbeat release)",
            "any managed Pod or the coverage canary Job exists",
            "the watcher Deployment is not one Ready replica",
            "coverage does not settle to heartbeat-only IDLE in budget",
            "preflight or focused regression failure",
            "target node is not Ready/schedulable/idle",
            "Node Agent/Profile/pin or node UID drift",
            "workflow is BLOCKED (UNKNOWN or otherwise) or differs from the reset contract",
            "provider mutation appears",
            "probe cleanup leaves a residual",
        ],
        "rollback": {
            "runner_finally_invokes_quiesce_restore_for_the_case_incident": True,
            "fail_safe_timer_remains_independent_of_the_probe": True,
            "failed_validation_keeps_the_node_quarantined": True,
            "runner_finally_deletes_probe_resources": True,
            "detached_sampler_is_stopped_after_the_post_reset_snapshot": True,
            "no_watcher_or_control_plane_setting_is_changed": True,
        },
        "preflight": preflight,
    }
    record_focused_tests(details, preflight["focused_tests"])
    return details


def verify_plan_identity(case_dir: Path, preflight: dict[str, Any]) -> None:
    plan = json.loads((case_dir / "plan.json").read_text(encoding="utf-8"))
    planned = plan["details"]["preflight_identity"]
    current = plan_identity(preflight)
    if current != planned:
        raise RegionalFixtureError(f"{CASE_ID} plan drifted: {planned} != {current}")


# --------------------------------------------------------------------------- #
# Execute
# --------------------------------------------------------------------------- #
def execute_case(
    settings: Settings,
    run_dir: Path,
    attempt: int,
    maintenance_window_end: datetime,
) -> int:
    case_dir = run_dir / "cases" / CASE_ID
    case_dir.mkdir(parents=True, exist_ok=True)
    preflight = read_only_preflight(settings, case_dir, reuse_focused_tests=True)
    if preflight["errors"]:
        raise RegionalFixtureError(
            "preflight failed: " + "; ".join(preflight["errors"])
        )
    verify_plan_identity(case_dir, preflight)

    regional = RegionalLiveFixture(settings.regional)
    run_id = f"destr023-{run_dir.name.rsplit('-', 1)[-1].lower()}-a{attempt}"
    marker = f"destr023-{int(time.time())}-a{attempt}"
    host = HostProbeFixture(
        HostProbeSettings(
            kubeconfig=settings.regional.gpu_kubeconfig,
            context=settings.regional.gpu_context,
            namespace=settings.regional.namespace,
            node=settings.node,
            image=settings.host_probe_image,
            case_id=CASE_ID,
            run_id=run_id,
            probe_script=PROBE_SCRIPT,
        )
    )
    result: dict[str, Any] = {
        "case_id": CASE_ID,
        "attempt": attempt,
        "verdict": "FAIL",
        **preflight["evidence_identity"],
        "node": settings.node,
        "maintenance_window_end": maintenance_window_end.isoformat(),
        "focused_tests_reused": bool(
            preflight["focused_tests"].get("focused_tests_reused")
        ),
    }
    baseline_host: dict[str, Any] | None = None
    incident_id = ""
    injection_started: datetime | None = None
    sampler_started = False
    sampler_stopped = False

    def stop_sampler() -> dict[str, Any]:
        nonlocal sampler_stopped
        stopped = host.execute("stop-reset-sampler", "--run-id", run_id, timeout=60)
        sampler_stopped = True
        write_json_atomic(case_dir / "sampler-final.json", stopped)
        return stopped

    try:
        host.create()
        baseline_host = host.execute("snapshot")
        write_json_atomic(case_dir / "host-baseline.json", baseline_host)
        expected_gpu_count = int(preflight["node"]["gpu_allocatable"])
        if len(baseline_host["gpu_inventory"]) != expected_gpu_count:
            raise RegionalFixtureError(
                "host GPU inventory differs from Node allocatable"
            )
        if baseline_host["compute_clients"]:
            raise RegionalFixtureError("target node has active NVIDIA compute clients")
        if baseline_host["quiesce_states"]:
            raise RegionalFixtureError("target node has a pre-existing quiesce state")
        if not baseline_host["kmsg_writable"]:
            raise RegionalFixtureError("/dev/kmsg is not writable from the host probe")
        target_bdf = str(baseline_host["gpu_inventory"][0]["pci_bdf"])

        # The premise. Old observations age out on their own; the heartbeat
        # has to be the only fresh coverage left when the XID lands.
        initial = coverage_probe(regional, settings.node)
        write_json_atomic(case_dir / "coverage-initial.json", initial)
        settle_wait = expiry_wait_seconds(initial)
        result["coverage_settle_wait_seconds"] = settle_wait
        coverage_before = wait_for_coverage(
            regional,
            settings.node,
            case_dir=case_dir,
            judge=fresh_coverage_errors,
            budget_seconds=settle_wait + SETTLE_BUDGET_SECONDS,
            label="idle",
        )
        write_json_atomic(case_dir / "coverage-before.json", coverage_before)
        if managed_pods(regional) or canary_jobs(regional):
            raise RegionalFixtureError(
                "a managed Pod or canary Job appeared during the wait"
            )
        if datetime.now(timezone.utc) >= maintenance_window_end:
            raise RegionalFixtureError(
                "approved maintenance window ended before injection"
            )

        sampler_started = True
        sampler = host.execute(
            "start-reset-sampler",
            "--run-id",
            run_id,
            "--probe-script",
            host.host_script,
            timeout=60,
        )
        write_json_atomic(case_dir / "sampler-start.json", sampler)
        injection_started = datetime.now(timezone.utc)
        injection = host.execute(
            "write-xid46",
            "--marker",
            marker,
            "--drill-id",
            run_id,
            "--pci-bdf",
            target_bdf,
        )
        write_json_atomic(case_dir / "injection.json", injection)
        state = regional.wait_for_workflow(
            node=settings.node,
            marker=marker,
            observed_after=injection_started,
            case_dir=case_dir,
            timeout_seconds=WORKFLOW_BUDGET_SECONDS,
        )
        write_json_atomic(case_dir / "workflow-state.json", state)
        incident_id = str((state.get("incident") or {}).get("incident_id") or "")
        errors = reset_errors(state)
        after = host.execute(
            "snapshot",
            "--since-epoch",
            str(injection_started.timestamp()),
            "--pci-bdf",
            target_bdf,
            "--run-id",
            run_id,
            timeout=180,
        )
        write_json_atomic(case_dir / "host-after.json", after)
        stop_sampler()
        errors.extend(
            reset_case.host_errors(
                baseline_host,
                after,
                expected_gpu_count=expected_gpu_count,
                target_bdf=target_bdf,
            )
        )
        node_after = regional.node_snapshot(settings.node)
        write_json_atomic(case_dir / "node-after.json", node_after)
        if node_after["ready"] != "True":
            errors.append("target node is not Ready after reset")
        if node_after["unschedulable"] or node_after["ownership_annotations"]:
            errors.append("target node scheduling ownership was not restored")
        coverage_after = coverage_probe(regional, settings.node)
        write_json_atomic(case_dir / "coverage-after.json", coverage_after)
        provider_window_end = datetime.now(timezone.utc)
        provider = regional.provider_events(injection_started, provider_window_end)
        provider_provisional = not provider and regional.provider_events_provisional(
            provider_window_end
        )
        write_json_atomic(
            case_dir / "provider-events.json",
            {"events": provider, "provider_events_provisional": provider_provisional},
        )
        if provider:
            errors.append("provider mutation appeared during GPU reset")
        cpu_after = regional.cpu_blast_snapshot()
        write_json_atomic(case_dir / "cpu-blast-after.json", cpu_after)
        if cpu_after != preflight["cpu_blast"]:
            errors.append("control-plane EKS state differs from baseline")
        result.update(
            {
                "verdict": "PASS" if not errors else "FAIL",
                "errors": errors,
                "marker": marker,
                "workflow_request_id": (state.get("workflow") or {}).get("request_id"),
                "incident_id": incident_id,
                "coverage_before": coverage_before,
                "coverage_after_state": coverage_after.get("workload_state"),
                "provider_events": provider,
                "provider_events_provisional": provider_provisional,
                "sampler": after.get("sampler"),
            }
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        recovery: dict[str, Any] = {}
        if incident_id:
            try:
                recovery = host.execute(
                    "restore-quiesce", "--incident-id", incident_id, timeout=300
                )
            except Exception as exc:
                recovery = {"error": f"{type(exc).__name__}: {exc}"}
                result["verdict"] = "FAIL"
        result["quiesce_recovery"] = recovery
        if sampler_started and not sampler_stopped:
            try:
                stop_sampler()
            except Exception as exc:
                result["sampler_cleanup_error"] = f"{type(exc).__name__}: {exc}"
                result["verdict"] = "FAIL"
        try:
            residuals = host.cleanup()
        except Exception as exc:
            residuals = {"cleanup_error": True}
            result["probe_cleanup_error"] = f"{type(exc).__name__}: {exc}"
            result["verdict"] = "FAIL"
        result["probe_residuals"] = residuals
        if any(residuals.values()):
            result["verdict"] = "FAIL"
        try:
            final_node = regional.node_snapshot(settings.node)
            result["final_node"] = final_node
            if final_node["ready"] != "True":
                result["verdict"] = "FAIL"
        except Exception as exc:
            result["postflight_error"] = f"{type(exc).__name__}: {exc}"
            result["verdict"] = "FAIL"
    write_json_atomic(case_dir / f"{CASE_ID}.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["verdict"] == "PASS" else 1


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Run the guarded DESTR-023 acceptance: a real GPU reset on a truly "
            "idle cluster kept IDLE by the watcher coverage heartbeat."
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
