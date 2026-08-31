#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e2e.regional import run_collector_destructive as base  # noqa: E402
from scripts.e2e.regional.acceptance_runner_common import (  # noqa: E402
    write_json_atomic,
)
from scripts.e2e.regional.collector_acceptance_fixture import (  # noqa: E402
    CollectorAcceptanceFixture,
)
from scripts.e2e.regional.live_driver_guard import (  # noqa: E402
    add_live_arguments,
    authorize_execution,
    build_plan,
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
    settings_from_arguments,
)


CASE_ID = "GF-REGIONAL-COLLECT-017"
PREDECESSOR_CASE_ID = "GF-REGIONAL-COLLECT-016"
CONFIRMATION = "COLLECT017_EXECUTE"


@dataclass(frozen=True)
class Settings:
    regional: RegionalLiveSettings
    node: str
    site_file: Path
    host_probe_image: str
    predecessor_path: Path

    def environment(self) -> dict[str, str]:
        return {
            **self.regional.environment(),
            "GPU_FAULT_TARGET_NODE": self.node,
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
        node=required(
            arguments.node or os.getenv("GPU_FAULT_TARGET_NODE", ""),
            "target node",
        ),
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
    node = regional.node_snapshot(settings.node)
    predecessor = predecessor_evidence(
        settings.predecessor_path,
        PREDECESSOR_CASE_ID,
    )
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/collectors/test_host.py::"
        "test_efa_inventory_distinguishes_unbound_driver_from_missing_pci",
        "tests/orchestration/test_health.py::"
        "test_idle_node_resource_group_upgrades_plugin_to_driver_remediation",
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
        errors.append("COLLECT-016 predecessor evidence is not PASS")
    if node["ready"] != "True" or node["unschedulable"]:
        errors.append("target node is not Ready and schedulable")
    if node["ownership_annotations"]:
        errors.append("target node has pre-existing workflow ownership")
    if regional.business_workloads(settings.node):
        errors.append("target node has a non-system workload")
    if not settings.site_file.is_file():
        errors.append("regional site file does not exist")
    if completed.returncode:
        errors.append("focused regression tests failed")
    result = {
        "release_id": regional.release_id(),
        "node": node,
        "predecessor": predecessor,
        "focused_tests_passed": completed.returncode == 0,
        "cpu_blast": regional.cpu_blast_snapshot(),
        "errors": errors,
    }
    write_json_atomic(case_dir / "preflight.json", result)
    return result


def run_efa_unbind(
    settings: Settings,
    regional: RegionalLiveFixture,
    collector: CollectorAcceptanceFixture,
    attempt: int,
) -> dict[str, Any]:
    snapshot = collector.snapshot()
    efa_devices = [
        item for item in snapshot["efa_inventory"]["devices"] if item.get("pci_bdf")
    ]
    if not efa_devices:
        raise RegionalFixtureError("no bound EFA BDF was discovered")
    bdf = str(efa_devices[0]["pci_bdf"])
    started_at = datetime.now(timezone.utc)
    collector.execute(
        "unbind-efa",
        "--run-id",
        f"c017-a-{attempt}",
        "--pci-bdf",
        bdf,
        "--restore-seconds",
        "300",
    )
    try:
        workflow = base.latest_node_workflow(
            regional,
            base.Settings(
                regional=settings.regional,
                case_id=CASE_ID,
                node=settings.node,
                second_node=None,
                host_probe_image=settings.host_probe_image,
                hyperpod_cluster="",
                executor_role_arn="",
                site_file=settings.site_file,
                predecessor_path=settings.predecessor_path,
            ),
            observed_after=started_at,
            timeout_seconds=600,
        )
    finally:
        collector.execute(
            "restore-efa",
            "--run-id",
            f"c017-a-{attempt}",
            "--pci-bdf",
            bdf,
        )
    errors = []
    if workflow["workflow"].get("status") != "SUCCEEDED":
        errors.append("EFA driver remediation workflow failed")
    return {"errors": errors, "efa_bdf": bdf, "workflow": workflow}


def run_gpu_plugin(
    settings: Settings,
    regional: RegionalLiveFixture,
) -> dict[str, Any]:
    baseline = int(regional.node_snapshot(settings.node)["gpu_allocatable"])
    plugin = base.DevicePluginFixture(
        regional,
        token="nvidia-device-plugin",
        node=settings.node,
        resource="nvidia.com/gpu",
    )
    info = plugin.discover()
    started_at = datetime.now(timezone.utc)
    try:
        plugin.exclude_node()
        plugin.wait_allocatable(0)
        plugin.restore()
        plugin.wait_allocatable(baseline)
        workflow = base.latest_node_workflow(
            regional,
            base.Settings(
                regional=settings.regional,
                case_id=CASE_ID,
                node=settings.node,
                second_node=None,
                host_probe_image=settings.host_probe_image,
                hyperpod_cluster="",
                executor_role_arn="",
                site_file=settings.site_file,
                predecessor_path=settings.predecessor_path,
            ),
            observed_after=started_at,
            timeout_seconds=600,
        )
    finally:
        plugin.restore()
    errors = []
    if workflow["workflow"].get("status") != "SUCCEEDED":
        errors.append("GPU device-plugin workflow failed")
    return {"errors": errors, "plugin": info, "workflow": workflow}


def run_training_plugin(
    settings: Settings,
    regional: RegionalLiveFixture,
    case_dir: Path,
    attempt: int,
) -> dict[str, Any]:
    candidates = [
        str(item["name"])
        for item in regional.gpu_nodes()
        if item["ready"] == "True" and not item["unschedulable"]
    ]
    if len(candidates) < 3:
        raise RegionalFixtureError("three GPU nodes are required")
    prewarm = ImagePrewarmFixture(
        regional,
        case_id=CASE_ID,
        run_id=f"c017-c-{attempt}",
    )
    workload: ManagedWorkloadFixture | None = None
    plugin: base.DevicePluginFixture | None = None
    result: dict[str, Any] = {"errors": []}
    try:
        prewarm.create(candidates)
        suffix = f"{int(time.time())}-{attempt}"
        manifest = base.render_named_training_manifest(
            case_dir / "c-workload.yaml",
            name=f"gpu-fault-efa-plugin-managed-{suffix}",
        )
        job_id = f"efa-plugin-managed-{suffix}"
        attempt_id = f"{job_id}-a001"
        workload = ManagedWorkloadFixture(
            regional,
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
        workload.submit()
        source = workload.wait_running(timeout_seconds=900)
        uids = {str(item["uid"]) for item in source["pods"]}
        target = str(source["pods"][0]["node"])
        node = json.loads(regional.kubectl("gpu", "get", "node", target, "-o", "json"))
        baseline = int(node["status"]["allocatable"].get("vpc.amazonaws.com/efa", 0))
        plugin = base.DevicePluginFixture(
            regional,
            token="efa-k8s-device-plugin",
            node=target,
            resource="vpc.amazonaws.com/efa",
        )
        info = plugin.discover()
        started_at = datetime.now(timezone.utc)
        plugin.exclude_node()
        plugin.wait_allocatable(0)
        time.sleep(30)
        during = workload.snapshot()
        if {str(item["uid"]) for item in during["pods"]} != uids:
            result["errors"].append("EFA plugin loss recreated training Pods")
        plugin.restore()
        plugin.wait_allocatable(baseline)
        workflow = base.latest_node_workflow(
            regional,
            base.Settings(
                regional=settings.regional,
                case_id=CASE_ID,
                node=target,
                second_node=None,
                host_probe_image=settings.host_probe_image,
                hyperpod_cluster="",
                executor_role_arn="",
                site_file=settings.site_file,
                predecessor_path=settings.predecessor_path,
            ),
            observed_after=started_at,
            timeout_seconds=600,
        )
        operations = {
            item.get("operation")
            for item in workflow["workflow"].get("official_steps", [])
        }
        if operations.intersection({"STOP_WORKLOADS", "RESTART_WORKLOAD"}):
            result["errors"].append("EFA plugin recovery restarted workload")
        result.update(
            {
                "target_node": target,
                "pod_uids": sorted(uids),
                "plugin": info,
                "workflow": workflow,
            }
        )
    finally:
        if plugin is not None:
            try:
                plugin.restore()
            except Exception as exc:
                result["errors"].append(
                    f"plugin restore failed: {type(exc).__name__}: {exc}"
                )
        if workload is not None:
            try:
                workload.delete()
            except Exception as exc:
                result["errors"].append(
                    f"workload cleanup failed: {type(exc).__name__}: {exc}"
                )
        try:
            residuals = prewarm.cleanup()
            if any(residuals.values()):
                result["errors"].append("prewarm Pods remain")
        except Exception as exc:
            result["errors"].append(
                f"prewarm cleanup failed: {type(exc).__name__}: {exc}"
            )
    return result


def plan_details(preflight: dict[str, Any]) -> dict[str, Any]:
    return {
        "risk": "live-node-mutation",
        "predecessor": preflight["predecessor"],
        "mutation": (
            "unbind one EFA function with an automatic bind timer, exclude one "
            "node from NVIDIA/EFA plugin DaemonSets, and verify allocatable recovery"
        ),
        "preflight_identity": {
            "release_id": preflight["release_id"],
            "node_uid": preflight["node"]["uid"],
        },
        "stop_conditions": [
            "COLLECT-016 predecessor evidence is not PASS",
            "target node is not idle or plugin DaemonSet discovery is ambiguous",
            "EFA bind fail-safe cannot be armed",
            "allocatable or training Pod UID assertions fail",
            "DaemonSet affinity cannot be restored exactly",
        ],
        "rollback": {
            "EFA_has_host_side_auto_bind": True,
            "restore_original_DaemonSet_affinity": True,
            "delete_managed_training_workload": True,
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
    collector = CollectorAcceptanceFixture(
        regional,
        node=settings.node,
        image=settings.host_probe_image,
        case_id=CASE_ID,
        run_id=f"c017-{attempt}",
    )
    result: dict[str, Any] = {
        "case_id": CASE_ID,
        "attempt": attempt,
        "verdict": "FAIL",
        "errors": [],
    }
    try:
        collector.create()
        a = run_efa_unbind(settings, regional, collector, attempt)
        result["errors"].extend(a["errors"])
        result["a"] = a
        b = run_gpu_plugin(settings, regional)
        result["errors"].extend(b["errors"])
        result["b"] = b
        c = run_training_plugin(settings, regional, case_dir, attempt)
        result["errors"].extend(c["errors"])
        result["c"] = c
        result["verdict"] = "PASS" if not result["errors"] else "FAIL"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            residuals = collector.cleanup()
            result["probe_residuals"] = residuals
            if any(residuals.values()):
                result["errors"].append("collector probe resources remain")
                result["verdict"] = "FAIL"
        except Exception as exc:
            result["cleanup_error"] = f"{type(exc).__name__}: {exc}"
            result["verdict"] = "FAIL"
    write_json_atomic(case_dir / f"{CASE_ID}.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["verdict"] == "PASS" else 1


def abort_on_signal(signum: int, _frame: object) -> None:
    raise RegionalFixtureError(f"received signal {signum}")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Run COLLECT-017 EFA and device-plugin acceptance."
    )
    add_live_arguments(value, confirmation=CONFIRMATION)
    value.add_argument("--cpu-kubeconfig", default="")
    value.add_argument("--gpu-kubeconfig", default="")
    value.add_argument("--gpu-context", default="")
    value.add_argument("--namespace", default="gpu-fault-system")
    value.add_argument("--cluster-id", default="")
    value.add_argument("--region", default="")
    value.add_argument("--node", default="")
    value.add_argument("--site-file", default="")
    value.add_argument("--host-probe-image", default="")
    value.add_argument("--predecessor-evidence", default="")
    return value


def main() -> int:
    arguments = parser().parse_args()
    os.umask(0o077)
    signal.signal(signal.SIGTERM, abort_on_signal)
    signal.signal(signal.SIGINT, abort_on_signal)
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
    raise SystemExit(main())
