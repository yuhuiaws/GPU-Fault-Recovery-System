#!/usr/bin/env python3
"""GF-REGIONAL-DESTR-024 live acceptance runner.

The fail-closed half of the coverage heartbeat. The completion watcher
Deployment is scaled to zero; once every heartbeat and attempt observation
of the cluster is older than ``GPU_FAULT_WORKLOAD_CONTEXT_FRESHNESS_SECONDS``
the topology service resolves the target node UNKNOWN, and one synthetic XID
46 on that idle node must compile BLOCKED with ``node workload state is
UNKNOWN``: the safety steps still contain the node (cordon + quarantine
taint), but no quiesce, no reset and no Node Agent ledger row ever happen.
The watcher is then scaled back, the heartbeat has to advance and turn the
node IDLE again, and the isolation is lifted only through the
validation-first restore workflow -- never by deleting the taint.

Safety of the runner itself:

* a detached watchdog (its own session, ``sleep N; kubectl scale``) is armed
  *before* the scale-down and disarmed only after the live restore of the
  watcher succeeded, so a runner that dies mid-case still leaves a cluster
  whose coverage feed returns;
* the case never writes to the control plane; the only mutation besides the
  kmsg line is the watcher replica count, and the validated restore is the
  product's own workflow.

The runner defaults to ``--plan``; ``--execute`` needs ``--confirm
DESTR024_EXECUTE``. The verdict functions live in ``destr024_verdicts.py``
(and the coverage half in ``destr023_verdicts.py``) and are unit-tested; the
runner records evidence only.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
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
from scripts.e2e.regional import (  # noqa: E402
    run_destr023_idle_cluster_reset as idle_case,
)
from scripts.e2e.regional.acceptance_runner_common import (  # noqa: E402
    write_json_atomic,
)
from scripts.e2e.regional.destr023_verdicts import (  # noqa: E402
    WATCHER_DEPLOYMENT,
    coverage_supported_errors,
    expiry_wait_seconds,
    fresh_coverage_errors,
    idle_cluster_errors,
    stale_coverage_errors,
    watcher_errors,
)
from scripts.e2e.regional.destr024_verdicts import (  # noqa: E402
    CASE_ID,
    CONFIRMATION,
    HEARTBEAT_RETURN_BUDGET_SECONDS,
    PREDECESSOR_CASE_ID,
    RESTORE_BUDGET_SECONDS,
    STALE_SETTLE_BUDGET_SECONDS,
    WATCHER_GONE_BUDGET_SECONDS,
    WATCHER_ROLLOUT_BUDGET_SECONDS,
    WORKFLOW_BUDGET_SECONDS,
    blocked_workflow_errors,
    containment_errors,
    heartbeat_recovered_errors,
    host_untouched_errors,
    is_isolated,
    node_isolated_errors,
    restore_errors,
    watchdog_delay_seconds,
    watchdog_script,
    watcher_absent_errors,
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
from scripts.e2e.regional.warm_spare_fixture import (  # noqa: E402
    WarmSpareLiveFixture,
)

PROBE_SCRIPT = Path(__file__).with_name("probes") / "destructive_node_probe.py"


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
# Watcher scale with a detached watchdog
# --------------------------------------------------------------------------- #
def kubectl_base(settings: RegionalLiveSettings) -> list[str]:
    return [
        "kubectl",
        "--kubeconfig",
        str(settings.gpu_kubeconfig),
        "--context",
        settings.gpu_context,
        "-n",
        settings.namespace,
    ]


def stop_process_group(process: subprocess.Popen[str] | None) -> dict[str, Any]:
    """Kill a ``start_new_session`` watchdog's whole group; never raise."""

    if process is None:
        return {"armed": False}
    if process.poll() is not None:
        return {"armed": True, "fired": True, "returncode": process.returncode}
    try:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
    except Exception as exc:  # pragma: no cover - platform dependent
        return {
            "armed": True,
            "fired": False,
            "stop_error": f"{type(exc).__name__}: {exc}",
        }
    return {"armed": True, "fired": False, "disarmed": True}


class WatcherScaleFixture:
    """Scale the completion watcher to zero and put it back.

    ``arm`` starts the detached restore first; ``scale_down`` then waits for
    the Pods to be gone; ``restore`` scales back, waits for the rollout and
    disarms the watchdog only once the live restore succeeded.
    """

    def __init__(
        self,
        regional: RegionalLiveFixture,
        *,
        case_dir: Path,
        baseline_replicas: int,
    ) -> None:
        self.regional = regional
        self.case_dir = case_dir
        self.baseline_replicas = baseline_replicas
        self.watchdog: subprocess.Popen[str] | None = None
        self.scaled_down = False
        self.restored = False

    def arm(self, delay_seconds: int) -> dict[str, Any]:
        script = watchdog_script(
            kubectl_base(self.regional.settings),
            replicas=self.baseline_replicas,
            delay_seconds=delay_seconds,
        )
        log_path = self.case_dir / "watcher-rollback-watchdog.log"
        with log_path.open("w", encoding="utf-8") as stream:
            self.watchdog = subprocess.Popen(
                ["bash", "-ceu", script],
                stdout=stream,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
        record = {"pid": self.watchdog.pid, "delay_seconds": delay_seconds}
        write_json_atomic(self.case_dir / "watcher-rollback-watchdog.json", record)
        return record

    def scale_down(self) -> dict[str, Any]:
        if self.watchdog is None:
            raise RegionalFixtureError(
                "watcher scale-down requested before the watchdog was armed"
            )
        self.scaled_down = True
        self.regional.kubectl(
            "gpu", "scale", f"deployment/{WATCHER_DEPLOYMENT}", "--replicas=0"
        )
        deadline = time.monotonic() + WATCHER_GONE_BUDGET_SECONDS
        while True:
            summary = idle_case.watcher_deployment(self.regional)
            pods = idle_case.watcher_pods(self.regional)
            errors = watcher_absent_errors(summary, pods)
            if not errors:
                snapshot = {"deployment": summary, "pods": pods, "scaled_at": _now()}
                write_json_atomic(self.case_dir / "watcher-scaled-down.json", snapshot)
                return snapshot
            if time.monotonic() >= deadline:
                raise RegionalFixtureError(
                    f"watcher did not leave within {WATCHER_GONE_BUDGET_SECONDS}s: "
                    + "; ".join(errors)
                )
            time.sleep(5)

    def restore(self) -> dict[str, Any]:
        self.regional.kubectl(
            "gpu",
            "scale",
            f"deployment/{WATCHER_DEPLOYMENT}",
            f"--replicas={self.baseline_replicas}",
        )
        self.regional.kubectl(
            "gpu",
            "rollout",
            "status",
            f"deployment/{WATCHER_DEPLOYMENT}",
            f"--timeout={WATCHER_ROLLOUT_BUDGET_SECONDS}s",
            timeout=WATCHER_ROLLOUT_BUDGET_SECONDS + 30,
        )
        summary = idle_case.watcher_deployment(self.regional)
        errors = watcher_errors(summary, expected_replicas=self.baseline_replicas)
        if errors:
            raise RegionalFixtureError(
                "watcher restore incomplete: " + "; ".join(errors)
            )
        self.restored = True
        # Disarmed last and only once the live restore ran; if the restore
        # raised above, the watchdog is the remaining rollback and stays.
        watchdog = stop_process_group(self.watchdog)
        record = {"deployment": summary, "watchdog": watchdog, "restored_at": _now()}
        write_json_atomic(self.case_dir / "watcher-restored.json", record)
        return record


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
        "test_a_stale_coverage_heartbeat_leaves_the_node_unknown",
        "tests/completion/test_workload_context_freshness.py::"
        "test_no_observations_for_the_cluster_is_unknown_not_idle",
        "tests/app_services/test_workload_coverage_route.py::"
        "test_a_heartbeat_is_stored_and_makes_the_cluster_idle",
        "tests/regional/test_destr024_watcher_down_fail_closed.py",
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
    coverage = idle_case.coverage_probe(fixture, settings.node)
    pods = idle_case.managed_pods(fixture)
    canaries = idle_case.canary_jobs(fixture)
    watcher = idle_case.watcher_deployment(fixture)
    tests = focused_tests(case_dir, reuse=reuse_focused_tests)
    predecessor = predecessor_evidence(
        settings.predecessor_path, PREDECESSOR_CASE_ID, **identity
    )
    errors = [
        *reset_case.preflight_errors(state, node, workloads, tests),
        *coverage_supported_errors(coverage),
        # The watcher must be alive and covering the cluster when the case
        # starts, otherwise "UNKNOWN after the scale-down" proves nothing.
        *fresh_coverage_errors(coverage, require_stale_observations=False),
        *idle_cluster_errors(pods, canaries),
        *watcher_errors(watcher),
    ]
    if not predecessor["valid"]:
        errors.append("DESTR-023 predecessor evidence is not PASS")
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


def plan_details(settings: Settings, preflight: dict[str, Any]) -> dict[str, Any]:
    details = {
        "risk": "live-isolation",
        "predecessor": preflight["predecessor"],
        "target_node": settings.node,
        "mutation": (
            f"scale deployment/{WATCHER_DEPLOYMENT} to 0 replicas; wait until the "
            "coverage heartbeat and every attempt observation are older than the "
            "freshness window; write one synthetic XID 46 to the real host "
            "/dev/kmsg and expect a BLOCKED, contained, never-reset node; scale "
            "the watcher back; lift the isolation through the validated restore "
            "workflow"
        ),
        "preflight_identity": idle_case.plan_identity(preflight),
        "coverage_premise": {
            "freshness_seconds": preflight["coverage"].get("freshness_seconds"),
            "expected_state_before_injection": "UNKNOWN",
            "expected_blocked_reason": "node workload state is UNKNOWN",
            "expected_blocked_kind": "SAFETY_SETTLED",
        },
        "stop_conditions": [
            "DESTR-023 has not passed in formal sequence on this release",
            "control plane has no coverage heartbeat (deploy CP-8 first)",
            "the heartbeat is not fresh before the scale-down",
            "any managed Pod or the coverage canary Job exists",
            "the watcher Deployment is not one Ready replica",
            "the watcher does not leave, or coverage does not expire, in budget",
            "preflight or focused regression failure",
            "target node is not Ready/schedulable/idle",
            "the workflow runs, or is BLOCKED for a different reason, or a physical step appears",
            "the heartbeat does not return after the watcher is restored",
            "the validated restore does not SUCCEED or leaves the node isolated",
            "provider mutation appears",
        ],
        "rollback": {
            "detached_watchdog_restores_watcher_replicas": True,
            "watchdog_disarmed_only_after_live_restore": True,
            "isolation_lifted_only_by_validated_restore_workflow": True,
            "quiesce_restore_runs_only_if_a_quiesce_state_appears": True,
            "runner_finally_deletes_probe_resources": True,
        },
        "preflight": preflight,
    }
    record_focused_tests(details, preflight["focused_tests"])
    return details


def verify_plan_identity(case_dir: Path, preflight: dict[str, Any]) -> None:
    plan = json.loads((case_dir / "plan.json").read_text(encoding="utf-8"))
    planned = plan["details"]["preflight_identity"]
    current = idle_case.plan_identity(preflight)
    if current != planned:
        raise RegionalFixtureError(f"{CASE_ID} plan drifted: {planned} != {current}")


# --------------------------------------------------------------------------- #
# Execute
# --------------------------------------------------------------------------- #
def validated_restore(
    regional: RegionalLiveFixture,
    warm: WarmSpareLiveFixture,
    *,
    node: str,
    incident_id: str,
    profile_version: str,
) -> dict[str, Any]:
    """Lift the fail-closed isolation the product's own way."""

    if not incident_id:
        raise RegionalFixtureError("the node is isolated but no incident is known")
    warm.wait_agent_active(node)
    created = warm.create_restore_workflow(
        incident_id=incident_id,
        node=node,
        profile_version=profile_version,
        reason="DESTR-024 validated cleanup after fail-closed block",
    )
    restored = warm.wait_workflow_id(
        str(created["workflow_request_id"]),
        timeout_seconds=RESTORE_BUDGET_SECONDS,
    )
    return {
        "created": created,
        "workflow": restored,
        "node": regional.node_snapshot(node),
    }


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
    warm = WarmSpareLiveFixture(regional, "")
    run_id = f"destr024-{run_dir.name.rsplit('-', 1)[-1].lower()}-a{attempt}"
    marker = f"destr024-{int(time.time())}-a{attempt}"
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
    watcher = WatcherScaleFixture(
        regional,
        case_dir=case_dir,
        baseline_replicas=int(preflight["watcher"]["replicas"]),
    )
    profile_version = str(
        (preflight["store"].get("profile") or {}).get("profile_version") or ""
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
    incident_id = ""
    injection_started: datetime | None = None
    after_host: dict[str, Any] | None = None
    restore_done = False

    try:
        host.create()
        baseline_host = host.execute("snapshot")
        write_json_atomic(case_dir / "host-baseline.json", baseline_host)
        if len(baseline_host["gpu_inventory"]) != int(
            preflight["node"]["gpu_allocatable"]
        ):
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

        coverage_alive = idle_case.coverage_probe(regional, settings.node)
        write_json_atomic(case_dir / "coverage-alive.json", coverage_alive)
        settle_wait = expiry_wait_seconds(coverage_alive, include_heartbeat=True)
        result["coverage_expiry_wait_seconds"] = settle_wait
        result["watchdog"] = watcher.arm(
            watchdog_delay_seconds(expiry_wait_seconds=settle_wait)
        )
        result["watcher_scaled_down"] = watcher.scale_down()

        coverage_stale = idle_case.wait_for_coverage(
            regional,
            settings.node,
            case_dir=case_dir,
            judge=stale_coverage_errors,
            budget_seconds=settle_wait + STALE_SETTLE_BUDGET_SECONDS,
            label="stale",
        )
        write_json_atomic(case_dir / "coverage-stale.json", coverage_stale)
        if idle_case.managed_pods(regional) or idle_case.canary_jobs(regional):
            raise RegionalFixtureError(
                "a managed Pod or canary Job appeared during the wait"
            )
        if datetime.now(timezone.utc) >= maintenance_window_end:
            raise RegionalFixtureError(
                "approved maintenance window ended before injection"
            )

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
        errors = [*blocked_workflow_errors(state), *containment_errors(state)]
        after_host = host.execute(
            "snapshot",
            "--since-epoch",
            str(injection_started.timestamp()),
            "--pci-bdf",
            target_bdf,
            "--run-id",
            run_id,
            timeout=180,
        )
        write_json_atomic(case_dir / "host-after.json", after_host)
        errors.extend(host_untouched_errors(baseline_host, after_host))
        node_blocked = regional.node_snapshot(settings.node)
        write_json_atomic(case_dir / "node-blocked.json", node_blocked)
        errors.extend(node_isolated_errors(node_blocked))

        # The watcher returns; the heartbeat has to advance past the stale one.
        result["watcher_restored"] = watcher.restore()
        coverage_back = idle_case.wait_for_coverage(
            regional,
            settings.node,
            case_dir=case_dir,
            judge=lambda sample: heartbeat_recovered_errors(coverage_stale, sample),
            budget_seconds=HEARTBEAT_RETURN_BUDGET_SECONDS,
            label="recovered",
        )
        write_json_atomic(case_dir / "coverage-recovered.json", coverage_back)

        # Close the record the product's way; the taint is never touched by hand.
        if is_isolated(node_blocked):
            restore = validated_restore(
                regional,
                warm,
                node=settings.node,
                incident_id=incident_id,
                profile_version=profile_version,
            )
            restore_done = True
            write_json_atomic(case_dir / "validated-restore.json", restore)
            errors.extend(restore_errors(restore["workflow"], restore["node"]))
            result["restore_workflow_request_id"] = restore["created"].get(
                "workflow_request_id"
            )

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
            errors.append("provider mutation appeared during the fail-closed run")
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
                "workflow_status": (state.get("workflow") or {}).get("status"),
                "blocked_reasons": (state.get("workflow") or {}).get("blocked_reasons"),
                "incident_id": incident_id,
                "coverage_stale": coverage_stale,
                "coverage_recovered": coverage_back,
                "provider_events": provider,
                "provider_events_provisional": provider_provisional,
            }
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        # The watcher comes back whatever happened above; the watchdog stays
        # armed until this live restore has succeeded.
        if watcher.scaled_down and not watcher.restored:
            try:
                result["watcher_restored"] = watcher.restore()
            except Exception as exc:
                result["watcher_restore_error"] = f"{type(exc).__name__}: {exc}"
                result["watchdog_left_armed"] = True
                result["verdict"] = "FAIL"
        elif not watcher.scaled_down:
            result["watchdog_disarmed"] = stop_process_group(watcher.watchdog)
        # No quiesce is expected; if one appeared the contract already failed,
        # and the host still has to be put back.
        if incident_id and after_host is not None and after_host.get("quiesce_states"):
            try:
                result["quiesce_recovery"] = host.execute(
                    "restore-quiesce", "--incident-id", incident_id, timeout=300
                )
            except Exception as exc:
                result["quiesce_recovery"] = {"error": f"{type(exc).__name__}: {exc}"}
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
            if is_isolated(final_node) and not restore_done:
                # Left for the operator: only the validated restore may lift it.
                result["node_left_isolated"] = True
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
            "Run the guarded DESTR-024 acceptance: with the completion watcher "
            "scaled to zero an idle node's XID 46 must be BLOCKED by UNKNOWN "
            "workload state and never reset."
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
