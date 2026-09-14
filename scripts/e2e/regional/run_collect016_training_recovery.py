#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e2e.regional import run_collector_destructive as base  # noqa: E402
from scripts.e2e.regional import (  # noqa: E402
    run_destr009_workload_restart as workload_case,
)
from scripts.e2e.regional.acceptance_runner_common import (  # noqa: E402
    write_json_atomic,
)
from scripts.e2e.regional.collector_acceptance_fixture import (  # noqa: E402
    CollectorAcceptanceFixture,
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
    settings_from_arguments,
)

CASE_ID = "GF-REGIONAL-COLLECT-016"
PREDECESSOR_CASE_ID = "GF-REGIONAL-COLLECT-014"
CONFIRMATION = "COLLECT016_EXECUTE"
# D section: RESET_GPU on a node that is running the managed workload. The
# compiler wraps the idle-node reset (run_destr001_gpu_reset.EXPECTED_STEPS)
# in STOP_WORKLOADS before the quiesce and RESTART_WORKLOAD after scheduling
# is restored (observed live 2026-09-06 09:29Z, workflow-bec004a7).
WORKLOAD_RESET_STEPS = [
    "FREEZE_EVIDENCE",
    "MARK_UNSCHEDULABLE",
    "STOP_WORKLOADS",
    "QUIESCE_GPU_SERVICES",
    "VERIFY_NO_GPU_CLIENTS",
    "RESET_GPU",
    "RESTORE_GPU_SERVICES",
    "VALIDATE_GPU",
    "RESTORE_SCHEDULING",
    "RESTART_WORKLOAD",
]


@dataclass(frozen=True)
class Settings:
    regional: RegionalLiveSettings
    site_file: Path
    host_probe_image: str
    predecessor_path: Path

    def environment(self) -> dict[str, str]:
        return {
            **self.regional.environment(),
            "GPU_FAULT_SITE_FILE": str(self.site_file),
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
        site_file=Path(
            required(
                arguments.site_file or os.getenv("GPU_FAULT_SITE_FILE", ""),
                "regional site file",
            )
        )
        .expanduser()
        .resolve(),
        host_probe_image=required(
            arguments.host_probe_image or os.getenv("GPU_FAULT_HOST_PROBE_IMAGE", ""),
            "host probe image",
        ),
        predecessor_path=predecessor,
    )


def read_only_preflight(
    settings: Settings,
    case_dir: Path,
) -> dict[str, Any]:
    regional = RegionalLiveFixture(settings.regional)
    candidates = [
        item
        for item in regional.gpu_nodes()
        if item["ready"] == "True" and not item["unschedulable"] and not item["taints"]
    ]
    identity = regional.evidence_identity()
    predecessor = predecessor_evidence(
        settings.predecessor_path,
        PREDECESSOR_CASE_ID,
        **identity,
    )
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/hyperpod/test_e2e_distributed_xid_reset.py::"
        "test_three_node_job_stops_before_two_fault_gpu_resets",
        "tests/execution/test_restart_safety.py::"
        "test_restart_budget_blocks_second_restart_for_same_job",
    ]
    completed = RegionalLiveFixture.run(
        command,
        cwd=ROOT,
        check=False,
        timeout=300,
    )
    (case_dir / "focused-tests.log").write_text(
        completed.stdout + completed.stderr,
        encoding="utf-8",
    )
    errors = []
    if not predecessor["valid"]:
        errors.append("COLLECT-014 predecessor evidence is not PASS")
    if not settings.site_file.is_file():
        errors.append("regional site file does not exist")
    if len(candidates) < 3:
        errors.append("fewer than three idle Ready GPU nodes")
    if regional.gpu_workloads():
        errors.append("GPU cluster already has active GPU workloads")
    if completed.returncode:
        errors.append("focused regression tests failed")
    result = {
        **identity,
        "candidate_nodes": candidates,
        "predecessor": predecessor,
        "focused_tests_passed": completed.returncode == 0,
        "cpu_blast": regional.cpu_blast_snapshot(),
        "errors": errors,
    }
    write_json_atomic(case_dir / "preflight.json", result)
    return result


def workload_settings(
    settings: Settings,
    *,
    manifest: Path,
    job_id: str,
    attempt_id: str,
) -> workload_case.Settings:
    return workload_case.Settings(
        regional=settings.regional,
        site_file=settings.site_file,
        manifest=manifest,
        job_id=job_id,
        attempt_id=attempt_id,
        predecessor_path=settings.predecessor_path,
    )


def managed_fixture(
    settings: Settings,
    *,
    manifest: Path,
    job_id: str,
    attempt_id: str,
) -> ManagedWorkloadFixture:
    return ManagedWorkloadFixture(
        RegionalLiveFixture(settings.regional),
        ManagedWorkloadSettings(
            manifest=manifest,
            site_file=settings.site_file,
            job_id=job_id,
            attempt_id=attempt_id,
            restart_budget=1,
            expected_pods=3,
            expected_gpu_count=24,
        ),
    )


def restart_workload_job_id(workflow: dict[str, Any]) -> str | None:
    """The job the RESTART_WORKLOAD step was compiled for, from its parameters."""

    for step in workflow.get("official_steps") or []:
        if step.get("operation") != "RESTART_WORKLOAD":
            continue
        parameters = step.get("parameters") or {}
        value = parameters.get("job_id")
        return str(value) if value else None
    return None


def restart_budget_section_errors(
    state_a: dict[str, Any],
    state_b: dict[str, Any],
    *,
    job_id: str,
) -> list[str]:
    """What is unique to COLLECT-016 A/B beyond the DESTR-009 restart contract.

    A: the incident identified the workload (``workload_identity_source``), the
    workflow was never BLOCKED, and RESTART_WORKLOAD was compiled for *this*
    job. B: the second RESTART_APP failed for ``RESTART_BUDGET_EXHAUSTED`` and
    the budget still shows the one restart A spent -- not two, not zero.
    """

    errors: list[str] = []
    incident_a = state_a.get("incident") or {}
    workflow_a = state_a.get("workflow") or {}
    if not incident_a.get("workload_identity_source"):
        errors.append("A: incident carries no workload_identity_source")
    if list(workflow_a.get("blocked_reasons") or []):
        errors.append(
            f"A: RESTART_APP workflow was blocked: {workflow_a.get('blocked_reasons')}"
        )
    compiled_job = restart_workload_job_id(workflow_a)
    if compiled_job != job_id:
        errors.append(
            f"A: RESTART_WORKLOAD was compiled for job {compiled_job!r}, not {job_id!r}"
        )
    workflow_b = state_b.get("workflow") or {}
    if workflow_b.get("status") != "FAILED":
        errors.append("B: second RESTART_APP was not budget rejected")
    if state_b.get("commands"):
        errors.append("B: budget rejection created a remote command")
    reasons = {
        str((item.get("details") or {}).get("reason") or "")
        for item in workflow_b.get("step_executions") or []
    }
    if "RESTART_BUDGET_EXHAUSTED" not in reasons:
        errors.append(
            f"B: no step execution failed for RESTART_BUDGET_EXHAUSTED: {sorted(reasons)}"
        )
    budget = state_b.get("restart_budget") or {}
    if budget.get("restart_count") != 1:
        errors.append(
            f"B: restart budget shows restart_count={budget.get('restart_count')!r}, "
            "expected the single restart A spent"
        )
    return errors


def run_restart_budget_sections(
    settings: Settings,
    regional: RegionalLiveFixture,
    case_dir: Path,
    suffix: str,
    *,
    workloads: list[ManagedWorkloadFixture],
    fixtures: list[CollectorAcceptanceFixture | HostProbeFixture],
) -> dict[str, Any]:
    """A and B. Resources land in ``workloads``/``fixtures`` the moment they exist.

    They used to be returned and appended by the caller, so a timeout inside
    this helper (a 900s wait_running, a workflow wait) leaked a 24-GPU
    PyTorchJob and a privileged probe Pod that no finally block knew about.
    """

    job_id = f"c016-a-{suffix}"
    attempt_id = f"{job_id}-a001"
    manifest = base.render_named_training_manifest(
        case_dir / "a-workload.yaml",
        name=f"gpu-fault-c016-a-{suffix}",
    )
    workload = managed_fixture(
        settings,
        manifest=manifest,
        job_id=job_id,
        attempt_id=attempt_id,
    )
    workloads.append(workload)
    workload.submit()
    source = workload.wait_running(timeout_seconds=900)
    source_uids = {str(item["uid"]) for item in source["pods"]}
    target_node = str(source["pods"][0]["node"])
    case_settings = workload_settings(
        settings,
        manifest=manifest,
        job_id=job_id,
        attempt_id=attempt_id,
    )
    workload_case.wait_observation(
        regional,
        case_settings,
        node=target_node,
        expected_gpu_count=24,
    )
    collector = CollectorAcceptanceFixture(
        regional,
        node=target_node,
        image=settings.host_probe_image,
        case_id=CASE_ID,
        run_id=f"c016-a-{suffix}",
    )
    fixtures.append(collector)
    collector.create()
    bdf = str(collector.snapshot()["gpu_inventory"][0]["pci_bdf"])
    marker_a = f"c016-a-{int(time.time())}"
    injected_a = datetime.now(timezone.utc)
    collector.execute(
        "write-xid",
        "--xid",
        "31",
        "--marker",
        marker_a,
        "--pci-bdf",
        bdf,
        "--message",
        "MMU Fault RESTART_APP",
    )
    state_a = regional.wait_for_workflow(
        node=target_node,
        marker=marker_a,
        observed_after=injected_a,
        case_dir=case_dir / "a-restart",
        timeout_seconds=1200,
        job_id=job_id,
        attempt_id=attempt_id,
    )
    errors = workload_case.workflow_errors(state_a, expected_gpu_count=24, xid=31)
    target = workload.wait_restarted(source_uids, timeout_seconds=900)
    marker_b = f"c016-b-{int(time.time())}"
    injected_b = datetime.now(timezone.utc)
    collector.execute(
        "write-xid",
        "--xid",
        "31",
        "--marker",
        marker_b,
        "--pci-bdf",
        bdf,
        "--message",
        "MMU Fault RESTART_APP budget exhausted",
    )
    state_b = regional.wait_for_workflow(
        node=target_node,
        marker=marker_b,
        observed_after=injected_b,
        case_dir=case_dir / "b-budget",
        timeout_seconds=600,
        job_id=job_id,
        attempt_id=attempt_id,
    )
    workflow_b = state_b.get("workflow") or {}
    errors.extend(restart_budget_section_errors(state_a, state_b, job_id=job_id))
    return {
        "errors": errors,
        "a": {
            "marker": marker_a,
            "job_id": job_id,
            "source_pod_uids": sorted(source_uids),
            "target_pod_uids": sorted(str(item["uid"]) for item in target["pods"]),
            "workflow": state_a.get("workflow"),
            "incident": state_a.get("incident"),
        },
        "b": {
            "marker": marker_b,
            "workflow": workflow_b,
            "restart_budget": state_b.get("restart_budget"),
        },
    }


def run_reset_section(
    settings: Settings,
    regional: RegionalLiveFixture,
    case_dir: Path,
    suffix: str,
    *,
    workloads: list[ManagedWorkloadFixture],
    fixtures: list[CollectorAcceptanceFixture | HostProbeFixture],
) -> dict[str, Any]:
    """D. Like A/B, the workload and probe are registered as soon as they exist."""

    job_id = f"c016-d-{suffix}"
    attempt_id = f"{job_id}-a001"
    manifest = base.render_named_training_manifest(
        case_dir / "d-workload.yaml",
        name=f"gpu-fault-c016-d-{suffix}",
    )
    workload = managed_fixture(
        settings,
        manifest=manifest,
        job_id=job_id,
        attempt_id=attempt_id,
    )
    workloads.append(workload)
    workload.submit()
    source = workload.wait_running(timeout_seconds=900)
    source_uids = {str(item["uid"]) for item in source["pods"]}
    target_node = str(source["pods"][0]["node"])
    case_settings = workload_settings(
        settings,
        manifest=manifest,
        job_id=job_id,
        attempt_id=attempt_id,
    )
    workload_case.wait_observation(
        regional,
        case_settings,
        node=target_node,
        expected_gpu_count=24,
    )
    reset_host = HostProbeFixture(
        HostProbeSettings(
            kubeconfig=settings.regional.gpu_kubeconfig,
            context=settings.regional.gpu_context,
            namespace=settings.regional.namespace,
            node=target_node,
            image=settings.host_probe_image,
            case_id=CASE_ID,
            run_id=f"c016-d-{suffix}",
            probe_script=base.DESTRUCTIVE_PROBE,
            active_deadline_seconds=3600,
        )
    )
    fixtures.append(reset_host)
    reset_host.create()
    marker = f"c016-d-{int(time.time())}"
    shim = base.Settings(
        regional=settings.regional,
        case_id=CASE_ID,
        node=target_node,
        second_node=None,
        host_probe_image=settings.host_probe_image,
        hyperpod_cluster="",
        executor_role_arn="",
        site_file=settings.site_file,
        predecessor_path=settings.predecessor_path,
    )
    state, errors = base.run_single_reset(
        shim,
        regional,
        reset_host,
        case_dir / "d-reset",
        xid=109,
        marker=marker,
        run_id=f"c016-d-{suffix}",
        node=target_node,
        expected_steps=WORKLOAD_RESET_STEPS,
    )
    target = workload.wait_restarted(source_uids, timeout_seconds=900)
    return {
        "errors": errors,
        "d": {
            "marker": marker,
            "workflow": state.get("workflow"),
            "source_pod_uids": sorted(source_uids),
            "target_pod_uids": sorted(str(item["uid"]) for item in target["pods"]),
        },
    }


def plan_details(settings: Settings, preflight: dict[str, Any]) -> dict[str, Any]:
    return {
        "risk": "destructive",
        "predecessor": preflight["predecessor"],
        "mutation": (
            "submit managed 24-GPU workloads, inject real kmsg XID31 twice "
            "and XID109 once, then verify restart budget and reset ordering"
        ),
        "preflight_identity": {
            "release_id": preflight["release_id"],
            "candidate_node_uids": sorted(
                str(item["uid"]) for item in preflight["candidate_nodes"]
            ),
        },
        "stop_conditions": [
            "COLLECT-014 predecessor evidence is not PASS",
            "fewer than three idle GPU nodes or image prewarm fails",
            "workload observation lacks 24 GPU UUIDs",
            "restart budget or real reset workflow differs from contract",
            "workload/probe cleanup leaves residuals",
        ],
        "rollback": {
            "delete_both_unique_workloads": True,
            "delete_prewarm_pods": True,
            "restore_quiesce_and_scheduling": True,
        },
        "preflight": preflight,
    }


A_WORKLOAD_DRAIN_TIMEOUT_SECONDS = 300


def wait_a_workload_drained(
    regional: RegionalLiveFixture,
    *,
    job_id: str,
    case_dir: Path,
    timeout_seconds: int = A_WORKLOAD_DRAIN_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Wait until the A/B-phase workload can no longer be node-correlated.

    A and B run on their own 24-GPU PyTorchJob that ``execute_case`` deletes
    before the reset section. ``delete`` force-removes the Pods and the
    PyTorchJob, but the attempt observation the collector last wrote lags: for
    up to the node-correlation window (``host_health._active_attempt`` --
    ``observed_at`` within [-30s, +120s], phase PENDING/RUNNING, a
    non-terminated container on the node) the store still reports the A
    workload as live on the reset node. A D-phase XID 109 fired inside that
    window correlates BOTH the D workload and the just-deleted A workload into
    one incident: STOP_WORKLOADS tolerates the absent A source
    (``already_absent_workloads``) but RESTART_WORKLOAD fails it closed
    (``RESTART_SOURCE_WORKLOAD_NOT_FOUND``, intentional hardening -- 41900a1,
    141a021), so the whole reset workflow FAILs and the node QUARANTINEs.
    Draining the A workload first scopes the reset incident to the D workload
    alone, which is what the case intends.

    "Drained" here is the negation of the product's correlation predicate: no
    observation for ``job_id`` is both non-terminal (PENDING/RUNNING) and still
    holding a non-terminated container. Once the latest observation of every
    attempt is terminal (or all its containers are terminated, or no
    observation is left), the [-30s, +120s] window can never match again, so
    the drain is durable rather than merely aged out.
    """

    deadline = time.monotonic() + timeout_seconds
    entries: list[dict[str, Any]] = []
    last: list[dict[str, Any]] = []
    while time.monotonic() < deadline:
        state = regional.store_snapshot(job_id=job_id, queue_attempts=1)
        last = state.get("observations") or []
        live = [
            {
                "attempt_id": observation.get("attempt_id"),
                "workload_phase": observation.get("workload_phase"),
                "live_nodes": sorted(
                    {
                        str(container.get("node_id"))
                        for container in observation.get("containers") or []
                        if not container.get("terminated")
                    }
                ),
            }
            for observation in last
            if observation.get("workload_phase") in {"PENDING", "RUNNING"}
            and any(
                not container.get("terminated")
                for container in observation.get("containers") or []
            )
        ]
        entries.append(
            {
                "observed_at": datetime.now(timezone.utc).isoformat(),
                "observation_count": len(last),
                "live": live,
            }
        )
        write_json_atomic(
            case_dir / "a-drain.json",
            {"job_id": job_id, "drained": not live, "entries": entries},
        )
        if not live:
            return {"job_id": job_id, "drained": True, "entries": entries}
        time.sleep(5)
    raise RegionalFixtureError(
        f"A-phase workload {job_id} did not drain from node correlation: {last}"
    )


def execute_case(
    settings: Settings,
    run_dir: Path,
    attempt: int,
    maintenance_window_end: datetime,
) -> int:
    case_dir = run_dir / "cases" / CASE_ID
    case_dir.mkdir(parents=True, exist_ok=True)
    preflight = read_only_preflight(settings, case_dir)
    if preflight["errors"]:
        raise RegionalFixtureError(
            "preflight failed: " + "; ".join(preflight["errors"])
        )
    if datetime.now(timezone.utc) >= maintenance_window_end:
        raise RegionalFixtureError("approved maintenance window has ended")
    regional = RegionalLiveFixture(settings.regional)
    candidates = [str(item["name"]) for item in preflight["candidate_nodes"]]
    prewarm = ImagePrewarmFixture(
        regional,
        case_id=CASE_ID,
        run_id=f"c016-{attempt}",
    )
    workloads: list[ManagedWorkloadFixture] = []
    fixtures: list[CollectorAcceptanceFixture | HostProbeFixture] = []
    result: dict[str, Any] = {
        "case_id": CASE_ID,
        "attempt": attempt,
        "verdict": "FAIL",
        "errors": [],
        **regional.evidence_identity(),
    }
    try:
        prewarm.create(candidates)
        suffix = f"{int(time.time())}-{attempt}"
        ab = run_restart_budget_sections(
            settings,
            regional,
            case_dir,
            suffix,
            workloads=workloads,
            fixtures=fixtures,
        )
        result["errors"].extend(ab["errors"])
        result.update({"a": ab["a"], "b": ab["b"]})
        workloads[0].delete()
        # Drain the just-deleted A workload from the store before the
        # reset section injects XID 109: otherwise its lagging attempt
        # observation is still node-correlatable and the reset incident
        # binds it as a restart source it can never satisfy (see
        # wait_a_workload_drained).
        result["a_drain"] = wait_a_workload_drained(
            regional,
            job_id=f"c016-a-{suffix}",
            case_dir=case_dir,
        )
        d = run_reset_section(
            settings,
            regional,
            case_dir,
            suffix,
            workloads=workloads,
            fixtures=fixtures,
        )
        result["errors"].extend(d["errors"])
        result["d"] = d["d"]
        result["verdict"] = "PASS" if not result["errors"] else "FAIL"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        for workload in workloads:
            try:
                workload.delete()
            except Exception as exc:
                result["errors"].append(
                    f"workload cleanup failed: {type(exc).__name__}: {exc}"
                )
        for fixture in fixtures:
            try:
                residuals = fixture.cleanup()
                if isinstance(residuals, dict) and any(residuals.values()):
                    result["errors"].append("probe resources remain")
            except Exception as exc:
                result["errors"].append(
                    f"probe cleanup failed: {type(exc).__name__}: {exc}"
                )
        try:
            residuals = prewarm.cleanup()
            if any(residuals.values()):
                result["errors"].append("prewarm Pods remain")
        except Exception as exc:
            result["errors"].append(
                f"prewarm cleanup failed: {type(exc).__name__}: {exc}"
            )
        if result["errors"]:
            result["verdict"] = "FAIL"
    write_json_atomic(case_dir / f"{CASE_ID}.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["verdict"] == "PASS" else 1


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Run COLLECT-016 managed training recovery acceptance."
    )
    add_live_arguments(value, confirmation=CONFIRMATION)
    value.add_argument("--cpu-kubeconfig", default="")
    value.add_argument("--gpu-kubeconfig", default="")
    value.add_argument("--gpu-context", default="")
    value.add_argument("--namespace", default="gpu-fault-system")
    value.add_argument("--cluster-id", default="")
    value.add_argument("--region", default="")
    value.add_argument("--site-file", default="")
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
