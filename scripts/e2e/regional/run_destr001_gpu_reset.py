#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
CASE_ID = "GF-REGIONAL-DESTR-001"
PREDECESSOR_CASE_ID = "GF-REGIONAL-DESTR-010"
CONFIRMATION = "DESTR001_RESET_IDLE_GPU"
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


def focused_tests(case_dir: Path, *, reuse: bool = False) -> dict[str, Any]:
    """Run the focused pytest, or reuse the plan's result in ``--execute``.

    ``reuse`` consults ``reusable_focused_tests`` on the plan this case wrote:
    a passing result recorded against the same source digest is not re-run.
    """

    if reuse:
        recorded = reusable_focused_tests(case_dir / "plan.json")
        if recorded is not None:
            return {**recorded, "focused_tests_reused": True}
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/node_agent/test_quiesce.py::"
        "test_quiesce_arms_timer_stops_and_restores_active_services",
        "tests/node_agent/test_remediation.py::"
        "test_gpu_reset_is_idempotent_and_rechecks_clients",
        "tests/node_agent/test_remediation.py::"
        "test_gpu_reset_refuses_active_compute_client",
        "tests/regional/test_regional_control_plane.py::"
        "test_remote_command_lease_and_result_advance_adapter",
        "tests/hyperpod/test_e2e_distributed_xid_reset.py::"
        "test_three_node_job_stops_before_two_fault_gpu_resets",
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


def preflight_errors(
    state: dict[str, Any],
    node: dict[str, Any],
    workloads: list[dict[str, str]],
    tests: dict[str, Any],
) -> list[str]:
    errors = []
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
    allowed = set(agent.get("allowed_operations") or [])
    if not REQUIRED_AGENT_OPERATIONS <= allowed:
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
    return errors


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
    tests = focused_tests(case_dir, reuse=reuse_focused_tests)
    predecessor = predecessor_evidence(
        settings.predecessor_path,
        PREDECESSOR_CASE_ID,
        **identity,
    )
    errors = preflight_errors(state, node, workloads, tests)
    if not predecessor["valid"]:
        errors.append("DESTR-010 predecessor evidence is not PASS")
    result = {
        "release_id": state.get("release_id"),
        "evidence_identity": identity,
        "node": node,
        "business_workloads": workloads,
        "store": state,
        "focused_tests": tests,
        "cpu_blast": fixture.cpu_blast_snapshot(),
        "predecessor": predecessor,
        "errors": errors,
    }
    write_json_atomic(case_dir / "preflight.json", result)
    return result


def waiting_details_present(state: dict[str, Any]) -> bool:
    required_operations = {
        "QUIESCE_GPU_SERVICES",
        "VERIFY_NO_GPU_CLIENTS",
        "RESET_GPU",
        "RESTORE_GPU_SERVICES",
    }
    workflow = state.get("workflow") or {}
    executions = [
        *(workflow.get("step_executions") or []),
        *(state.get("observed_waiting_step_executions") or []),
    ]
    observed = {
        str(item.get("operation"))
        for item in executions
        if item.get("status") == "WAITING"
        and (item.get("details") or {}).get("mutation_submitted_by_control_plane")
        is False
    }
    # A step that finishes inside one store poll (~11s) is never seen WAITING,
    # and the terminal record drops the flag. Its adapter_operation_id keeps
    # the same fact: `remote/...` means the GPU-side executor carried out the
    # mutation, not the control plane. COLLECT-016 D observed QUIESCE that way
    # on 2026-09-06 09:29Z with every other remote step WAITING as required.
    observed |= {
        str(item.get("operation"))
        for item in workflow.get("step_executions") or []
        if item.get("status") == "SUCCEEDED"
        and str(item.get("adapter_operation_id") or "").startswith("remote/")
    }
    return required_operations <= observed


def workflow_errors(
    state: dict[str, Any],
    *,
    xid: int = 46,
    official_action: str = "RESET_GPU",
    expected_steps: list[str] | None = None,
) -> list[str]:
    """The reset contract every kmsg-driven single-GPU reset case shares.

    DESTR-001 drives it with XID 46 / RESET_GPU; the XID 63+48 companion drill
    (COLLECT-008/013) reaches the same eight steps through DRAIN_AND_RESET,
    so the event and the decision it must name are parameters. A node that is
    running a managed workload compiles STOP_WORKLOADS/RESTART_WORKLOAD around
    the same reset (COLLECT-016 D), so the step sequence is one as well.
    """

    expected = list(expected_steps) if expected_steps is not None else EXPECTED_STEPS
    errors = []
    event = state.get("event") or {}
    decision = state.get("decision") or {}
    workflow = state.get("workflow") or {}
    if event.get("xid") != xid:
        errors.append(f"matched event is not XID {xid}")
    if not str(event.get("evidence_ref") or "").startswith("kmsg://"):
        errors.append("XID evidence is not backed by kmsg://")
    if decision.get("official_action") != official_action:
        errors.append(f"policy did not finalize XID {xid} as {official_action}")
    steps = [item.get("operation") for item in workflow.get("official_steps", [])]
    if steps != expected:
        errors.append("workflow step sequence differs from the reset contract")
    if workflow.get("status") != "SUCCEEDED":
        errors.append("reset workflow is not SUCCEEDED")
    completed = workflow.get("completed_operations") or []
    if completed != expected:
        errors.append("completed operation sequence differs from the reset contract")
    if not waiting_details_present(state):
        errors.append("remote WAITING evidence lacks control-plane mutation=false")
    commands = state.get("commands") or []
    remote_operations = {
        "MARK_UNSCHEDULABLE",
        "QUIESCE_GPU_SERVICES",
        "VERIFY_NO_GPU_CLIENTS",
        "RESET_GPU",
        "RESTORE_GPU_SERVICES",
        "RESTORE_SCHEDULING",
    }
    observed = {
        item.get("step", {}).get("operation")
        for item in commands
        if item.get("status") == "SUCCEEDED"
    }
    if not remote_operations <= observed:
        errors.append("not every remote reset operation reached SUCCEEDED")
    return errors


def host_errors(
    baseline: dict[str, Any],
    after: dict[str, Any],
    *,
    expected_gpu_count: int,
    target_bdf: str,
) -> list[str]:
    errors = []
    if len(after.get("gpu_inventory") or []) != expected_gpu_count:
        errors.append("GPU inventory count changed after reset")
    if after.get("compute_clients"):
        errors.append("compute clients remain after reset")
    baseline_rows = {
        (item["command_id"], item["operation"]) for item in baseline.get("ledger") or []
    }
    added = [
        item
        for item in after.get("ledger") or []
        if (item["command_id"], item["operation"]) not in baseline_rows
    ]
    reset_rows = [item for item in added if item["operation"] == "RESET_GPU"]
    if len(reset_rows) != 1:
        errors.append("Node Agent ledger did not add exactly one RESET_GPU result")
    elif reset_rows[0].get("attempt") != 1:
        errors.append("RESET_GPU ledger attempt is not one")
    if len({item["command_id"] for item in added}) != len(added):
        errors.append("Node Agent ledger contains duplicate command IDs")
    if after.get("gpu_fault_timers") != baseline.get("gpu_fault_timers"):
        errors.append("gpu-fault timer inventory did not return to baseline")
    if after.get("quiesce_states"):
        errors.append("GPU quiesce state remains after restore")
    for unit, before in (baseline.get("services") or {}).items():
        if before.get("ActiveState") != "active":
            continue
        current = (after.get("services") or {}).get(unit, {})
        if current.get("ActiveState") != "active":
            errors.append(f"service did not return active: {unit}")
    # The physical proof is the detached host sampler: the target GPU stops
    # enumerating while the driver resets it and is back at the end, so the
    # sample series has to dip below the expected count and finish on it. The
    # kernel journal used to be asserted to show exactly one reset line, but
    # the only lines it ever matched were this drill's own injected XID text
    # (see `counts_as_target_reset`); kernel-origin lines are still recorded
    # and more than one of them is still a failure.
    journal = after.get("kernel_reset_journal") or {}
    if int(journal.get("target_reset_count") or 0) > 1:
        errors.append(f"kernel journal shows more than one reset for {target_bdf}")
    sampler = after.get("sampler") or {}
    if int(sampler.get("sample_count") or 0) < 2:
        errors.append("detached host GPU sampler has insufficient samples")
    else:
        minimum = sampler.get("min_gpu_count")
        last = (sampler.get("last") or {}).get("gpu_count")
        if (
            minimum is None
            or int(minimum) >= expected_gpu_count
            or last != expected_gpu_count
        ):
            errors.append(
                "detached host GPU sampler did not see the target GPU leave and "
                f"return (min={minimum}, last={last}, expected={expected_gpu_count})"
            )
    return errors


def plan_details(settings: Settings, preflight: dict[str, Any]) -> dict[str, Any]:
    state = preflight["store"]
    details = {
        "risk": "destructive",
        "predecessor": preflight["predecessor"],
        "target_node": settings.node,
        "mutation": (
            "write one synthetic XID 46 to the real host /dev/kmsg; allow the "
            "workflow to quiesce services and execute one real GPU reset"
        ),
        "preflight_identity": {
            "release_id": preflight["release_id"],
            "node_uid": preflight["node"]["uid"],
            "agent_generation": (state.get("agent") or {}).get("generation"),
            "runtime_profile_version": (state.get("profile") or {}).get(
                "profile_version"
            ),
        },
        "stop_conditions": [
            "DESTR-010 has not passed in formal sequence",
            "preflight or focused regression failure",
            "target node is not Ready/schedulable/idle",
            "Node Agent/Profile/pin or node UID drift",
            "active GPU client or incomplete GPU inventory",
            "workflow differs from the eight-step reset contract",
            "provider mutation appears",
            "GPU validation or service restore fails",
            "probe cleanup leaves a residual",
        ],
        "rollback": {
            "runner_finally_invokes_quiesce_restore_for_the_case_incident": True,
            "fail_safe_timer_remains_independent_of_the_probe": True,
            "failed_validation_keeps_the_node_quarantined": True,
            "reboot_requires_a_separate_approved_case": True,
            "runner_finally_deletes_probe_resources": True,
            "detached_sampler_is_stopped_after_the_post_reset_snapshot": True,
        },
        "preflight": preflight,
    }
    record_focused_tests(details, preflight["focused_tests"])
    return details


def verify_plan_identity(
    case_dir: Path,
    preflight: dict[str, Any],
) -> None:
    plan = json.loads((case_dir / "plan.json").read_text(encoding="utf-8"))
    planned = plan["details"]["preflight_identity"]
    current = {
        "release_id": preflight["release_id"],
        "node_uid": preflight["node"]["uid"],
        "agent_generation": (preflight["store"].get("agent") or {}).get("generation"),
        "runtime_profile_version": (preflight["store"].get("profile") or {}).get(
            "profile_version"
        ),
    }
    if current != planned:
        raise RegionalFixtureError(f"DESTR-001 plan drifted: {planned} != {current}")


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
    run_id = f"destr001-{run_dir.name.rsplit('-', 1)[-1].lower()}-a{attempt}"
    marker = f"destr001-{int(time.time())}-a{attempt}"
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
        stopped = host.execute(
            "stop-reset-sampler",
            "--run-id",
            run_id,
            timeout=60,
        )
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
        if datetime.now(timezone.utc) >= maintenance_window_end:
            raise RegionalFixtureError(
                "approved maintenance window ended before injection"
            )

        # Armed before the call, not after: a start that times out or fails
        # after the transient unit was created would otherwise leave the
        # sampler running on the host with nothing scheduled to stop it. The
        # stop is idempotent, so arming early costs nothing when the start
        # never reached the host.
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
            timeout_seconds=1200,
        )
        write_json_atomic(case_dir / "workflow-state.json", state)
        incident_id = str((state.get("incident") or {}).get("incident_id") or "")
        errors = workflow_errors(state)
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
        # The sampler's job ends with the post-reset snapshot that reads its
        # series; stopping it here rather than in `finally` keeps it from
        # running through the node/provider/CPU checks that follow.
        stop_sampler()
        errors.extend(
            host_errors(
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
        provider_window_end = datetime.now(timezone.utc)
        provider = regional.provider_events(injection_started, provider_window_end)
        # An empty CloudTrail read inside the delivery window is not proof
        # that no provider mutation happened; it is recorded as provisional.
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
                "workflow_request_id": (
                    (state.get("workflow") or {}).get("request_id")
                ),
                "incident_id": incident_id,
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
                    "restore-quiesce",
                    "--incident-id",
                    incident_id,
                    timeout=300,
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
        description="Run the guarded DESTR-001 real GPU reset acceptance."
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
