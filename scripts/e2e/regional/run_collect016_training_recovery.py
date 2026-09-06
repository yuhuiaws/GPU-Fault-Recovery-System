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
    add_live_arguments,
    authorize_execution,
    build_plan,
    install_site_profile,
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
    install_abort_signals,
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
    predecessor = predecessor_evidence(
        settings.predecessor_path,
        PREDECESSOR_CASE_ID,
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
        "release_id": regional.release_id(),
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


def run_restart_budget_sections(
    settings: Settings,
    regional: RegionalLiveFixture,
    case_dir: Path,
    suffix: str,
) -> tuple[dict[str, Any], ManagedWorkloadFixture, CollectorAcceptanceFixture]:
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
    if workflow_b.get("status") != "FAILED":
        errors.append("second RESTART_APP was not budget rejected")
    if state_b.get("commands"):
        errors.append("budget rejection created a remote command")
    return (
        {
            "errors": errors,
            "a": {
                "marker": marker_a,
                "source_pod_uids": sorted(source_uids),
                "target_pod_uids": sorted(str(item["uid"]) for item in target["pods"]),
                "workflow": state_a.get("workflow"),
            },
            "b": {"marker": marker_b, "workflow": workflow_b},
        },
        workload,
        collector,
    )


def run_reset_section(
    settings: Settings,
    regional: RegionalLiveFixture,
    case_dir: Path,
    suffix: str,
) -> tuple[dict[str, Any], ManagedWorkloadFixture, HostProbeFixture]:
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
    return (
        {
            "errors": errors,
            "d": {
                "marker": marker,
                "workflow": state.get("workflow"),
                "source_pod_uids": sorted(source_uids),
                "target_pod_uids": sorted(str(item["uid"]) for item in target["pods"]),
            },
        },
        workload,
        reset_host,
    )


def plan_details(preflight: dict[str, Any]) -> dict[str, Any]:
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
    }
    try:
        prewarm.create(candidates)
        suffix = f"{int(time.time())}-{attempt}"
        ab, workload_a, collector = run_restart_budget_sections(
            settings,
            regional,
            case_dir,
            suffix,
        )
        workloads.append(workload_a)
        fixtures.append(collector)
        result["errors"].extend(ab["errors"])
        result.update({"a": ab["a"], "b": ab["b"]})
        workload_a.delete()
        d, workload_d, reset_host = run_reset_section(
            settings,
            regional,
            case_dir,
            suffix,
        )
        workloads.append(workload_d)
        fixtures.append(reset_host)
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
            details=plan_details(preflight),
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


if __name__ == "__main__":
    raise SystemExit(run_case_main(main))
