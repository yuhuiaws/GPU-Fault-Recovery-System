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
from typing import Any, cast

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


EFA_REMEDIATION_STEPS = (
    "FREEZE_EVIDENCE",
    "MARK_UNSCHEDULABLE",
    "REMEDIATE_EFA_DRIVER",
    "RESTART_EFA_DEVICE_PLUGIN",
    "TRIGGER_HEALTH_SNAPSHOT",
    "VALIDATE_FABRIC",
    "RESTORE_SCHEDULING",
)
# The host-side bind fail-safe only covers a runner that dies mid-case. It has
# to outlive the workflow wait (600s) plus the incident wait, or it rebinds the
# function itself and the Node Agent finds nothing to do -- a false PASS the
# verdict below would catch, but as a flaky FAIL rather than a clean run.
EFA_RESTORE_SECONDS = 900
EFA_WORKFLOW_TIMEOUT_SECONDS = 600
EFA_INCIDENT_TIMEOUT_SECONDS = 180


REMOTE_COMMANDS = r"""
import json
import sys

from gpu_fault.app import ApplicationContext

request_id = sys.argv[1]
store = ApplicationContext.from_environment().store
commands = []
for item in store.list_remote_commands(workflow_request_ids=[request_id]):
    payload = item.model_dump(mode="json")
    commands.append({
        "command_id": payload.get("command_id"),
        "step_index": payload.get("step_index"),
        "operation": item.step.operation.value,
        "status": payload.get("status"),
        "error": payload.get("error"),
        "result_details": payload.get("result_details") or {},
    })
print(json.dumps({"remote_commands": commands}, sort_keys=True, default=str))
"""


def remote_node_result(
    bundle: dict[str, Any], *, operation: str, node: str
) -> dict[str, Any] | None:
    """What the Node Agent reported for ``operation`` on ``node``.

    A remote step's ``details`` stay empty; the agent's result travels on the
    remote command as ``result_details.node_results[<node>]`` (observed live:
    ``rebound_pci_bdfs`` was there and nowhere else).
    """

    commands = [
        item
        for item in bundle.get("remote_commands") or []
        if item.get("operation") == operation
    ]
    if not commands:
        return None
    details = commands[-1].get("result_details") or {}
    node_results = details.get("node_results") or {}
    result = node_results.get(node)
    return cast(dict[str, Any], result) if isinstance(result, dict) else None


def _bound_efa_devices(inventory: dict[str, Any]) -> list[dict[str, Any]]:
    """Functions the collector probe sees bound to ``efa`` (an ``efa_inventory``)."""

    return [
        item
        for item in inventory.get("devices") or []
        if item.get("pci_bdf") and item.get("driver") == "efa"
    ]


def efa_unbind_errors(
    bundle: dict[str, Any],
    *,
    node: str,
    bdf: str,
    baseline: dict[str, Any],
    unbound: dict[str, Any],
    recovered: dict[str, Any],
    restore: dict[str, Any],
) -> list[str]:
    """COLLECT-017 A verdict: the control plane, not the fail-safe, rebound it.

    ``baseline``/``unbound``/``recovered`` are collector-probe ``efa_inventory``
    readings before the unbind, right after it, and after the workflow ended;
    ``restore`` is what ``restore-efa`` observed before disarming the timer;
    ``bundle`` carries the workflow, its incident and its remote commands.
    """

    errors: list[str] = []
    workflow = bundle.get("workflow") or {}
    incident = bundle.get("incident") or {}
    expected = int(baseline["discovered_count"])

    # Injection took effect and the collector saw exactly one function go.
    if int(unbound["discovered_count"]) != expected - 1:
        errors.append(
            "collector did not observe the unbound function: discovered "
            f"{unbound['discovered_count']} != {expected - 1}"
        )
    if any(item.get("pci_bdf") == bdf for item in _bound_efa_devices(unbound)):
        errors.append(f"{bdf} still bound to efa after the unbind")

    # The control plane compiled the documented remediation, in order.
    operations = [
        str(item.get("operation")) for item in workflow.get("official_steps") or []
    ]
    if operations != list(EFA_REMEDIATION_STEPS):
        errors.append(f"official steps {operations} != {list(EFA_REMEDIATION_STEPS)}")
    # Live, the incident's official_action is None for this path; the decided
    # action is effective_action, and the workflow carries it as official_action.
    incident_action = incident.get("effective_action") or incident.get(
        "official_action"
    )
    if incident_action != "REMEDIATE_EFA_DRIVER":
        errors.append(f"incident action {incident_action!r} != REMEDIATE_EFA_DRIVER")
    if workflow.get("official_action") != "REMEDIATE_EFA_DRIVER":
        errors.append(
            f"workflow official_action {workflow.get('official_action')!r} "
            "!= REMEDIATE_EFA_DRIVER"
        )
    reasons = " ".join(str(item) for item in incident.get("reasons") or [])
    if "driver is not bound" not in reasons:
        errors.append("incident reasons do not carry the DRIVER_UNBOUND finding")

    # The Node Agent rebound exactly this function through modprobe + bind.
    if workflow.get("status") != "SUCCEEDED":
        errors.append(f"workflow status {workflow.get('status')!r} != SUCCEEDED")
    remediation = [
        item
        for item in workflow.get("step_executions") or []
        if item.get("operation") == "REMEDIATE_EFA_DRIVER"
    ]
    if not remediation:
        errors.append("REMEDIATE_EFA_DRIVER was never executed")
    else:
        step = remediation[-1]
        if step.get("status") != "SUCCEEDED":
            errors.append(
                f"REMEDIATE_EFA_DRIVER status {step.get('status')!r} != SUCCEEDED"
            )
        if not str(step.get("adapter_operation_id") or "").startswith("remote/"):
            errors.append("REMEDIATE_EFA_DRIVER did not run as a remote node action")
        details = (
            remote_node_result(bundle, operation="REMEDIATE_EFA_DRIVER", node=node)
            or {}
        )
        if not details:
            errors.append(
                f"no Node Agent result for REMEDIATE_EFA_DRIVER on {node} "
                "(remote command result_details.node_results)"
            )
        if details.get("rebound_pci_bdfs") != [bdf]:
            errors.append(
                f"rebound_pci_bdfs {details.get('rebound_pci_bdfs')!r} != [{bdf!r}]"
            )
        if details.get("already_bound") is not False:
            errors.append("Node Agent found the function already bound: nothing to do")
        if details.get("driver_bound_count") != expected:
            errors.append(
                f"driver_bound_count {details.get('driver_bound_count')!r} != {expected}"
            )

    # Recovery is real on the host and the fail-safe never had to act.
    if int(recovered["discovered_count"]) != expected:
        errors.append(
            f"discovered {recovered['discovered_count']} != baseline {expected}"
        )
    if int(recovered["active_count"]) != int(baseline["active_count"]):
        errors.append(
            f"ACTIVE {recovered['active_count']} != baseline {baseline['active_count']}"
        )
    if not any(item.get("pci_bdf") == bdf for item in _bound_efa_devices(recovered)):
        errors.append(f"{bdf} is not bound to efa after the workflow")
    if restore.get("already_bound") is not True:
        errors.append("restore-efa had to bind the function itself")
    if restore.get("timer_fired"):
        errors.append(
            "the host fail-safe timer rebound the function, not the control plane"
        )
    if restore.get("bound") is not True:
        errors.append("function is not bound after restore-efa")

    if incident.get("state") != "RECOVERED":
        errors.append(f"incident state {incident.get('state')!r} != RECOVERED")
    return errors


def _wait_efa_inventory(
    collector: CollectorAcceptanceFixture,
    *,
    discovered_count: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    snapshot = collector.snapshot()
    while (
        int(snapshot["efa_inventory"]["discovered_count"]) != discovered_count
        and time.monotonic() < deadline
    ):
        time.sleep(3)
        snapshot = collector.snapshot()
    return cast(dict[str, Any], snapshot["efa_inventory"])


def run_efa_unbind(
    settings: Settings,
    regional: RegionalLiveFixture,
    collector: CollectorAcceptanceFixture,
    attempt: int,
) -> dict[str, Any]:
    baseline_snapshot = collector.snapshot()
    baseline = baseline_snapshot["efa_inventory"]
    bound = _bound_efa_devices(baseline)
    if not bound:
        raise RegionalFixtureError("no bound EFA BDF was discovered")
    bdf = str(bound[0]["pci_bdf"])
    base_settings = _base_settings(settings, settings.node)
    started_at = datetime.now(timezone.utc)
    injection = collector.execute(
        "unbind-efa",
        "--run-id",
        f"c017-a-{attempt}",
        "--pci-bdf",
        bdf,
        "--restore-seconds",
        str(EFA_RESTORE_SECONDS),
    )
    unbound: dict[str, Any] = {}
    recovered: dict[str, Any] = {}
    bundle: dict[str, Any] = {}
    try:
        unbound = _wait_efa_inventory(
            collector,
            discovered_count=int(baseline["discovered_count"]) - 1,
            timeout_seconds=60,
        )
        bundle = base.latest_node_workflow(
            regional,
            base_settings,
            observed_after=started_at,
            timeout_seconds=EFA_WORKFLOW_TIMEOUT_SECONDS,
        )
        # The workflow closes before the incident does; give it its own wait.
        deadline = time.monotonic() + EFA_INCIDENT_TIMEOUT_SECONDS
        while (bundle.get("incident") or {}).get(
            "state"
        ) != "RECOVERED" and time.monotonic() < deadline:
            time.sleep(5)
            bundle = base.latest_node_workflow(
                regional,
                base_settings,
                observed_after=started_at,
                timeout_seconds=30,
            )
        recovered = _wait_efa_inventory(
            collector,
            discovered_count=int(baseline["discovered_count"]),
            timeout_seconds=30,
        )
        request_id = str((bundle.get("workflow") or {}).get("request_id") or "")
        if request_id:
            bundle = {
                **bundle,
                **regional.cpu_python(REMOTE_COMMANDS, request_id),
            }
    finally:
        restore = collector.execute(
            "restore-efa",
            "--run-id",
            f"c017-a-{attempt}",
            "--pci-bdf",
            bdf,
        )
    errors = efa_unbind_errors(
        bundle,
        node=settings.node,
        bdf=bdf,
        baseline=baseline,
        unbound=unbound,
        recovered=recovered,
        restore=restore,
    )
    return {
        "errors": errors,
        "efa_bdf": bdf,
        "injection": injection,
        "baseline_inventory": baseline,
        "unbound_inventory": unbound,
        "recovered_inventory": recovered,
        "restore": restore,
        "workflow": bundle,
    }


GPU_PLUGIN_STEPS = (
    "FREEZE_EVIDENCE",
    "RESTART_GPU_DEVICE_PLUGIN",
    "TRIGGER_HEALTH_SNAPSHOT",
    "VALIDATE_GPU",
)
EFA_PLUGIN_STEPS = (
    "FREEZE_EVIDENCE",
    "RESTART_EFA_DEVICE_PLUGIN",
    "TRIGGER_HEALTH_SNAPSHOT",
    "VALIDATE_FABRIC",
)
# The Kubernetes node-resource collector samples every 15s and needs two
# consecutive mismatching samples before it reports; add ingestion, triage and
# planning. Restoring the DaemonSet before the workflow exists (attempt 2 did,
# right after allocatable hit 0) leaves the collector nothing to see.
PLUGIN_WORKFLOW_PLAN_TIMEOUT_SECONDS = 240


def _base_settings(settings: Settings, node: str) -> Any:
    return base.Settings(
        regional=settings.regional,
        case_id=CASE_ID,
        node=node,
        second_node=None,
        host_probe_image=settings.host_probe_image,
        hyperpod_cluster="",
        executor_role_arn="",
        site_file=settings.site_file,
        predecessor_path=settings.predecessor_path,
    )


def _wait_planned_workflow(
    regional: RegionalLiveFixture,
    settings: Settings,
    *,
    node: str,
    operation: str,
    observed_after: datetime,
    timeout_seconds: int = PLUGIN_WORKFLOW_PLAN_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """The workflow that planned ``operation`` for ``node``, in any status.

    ``base.latest_node_workflow`` waits for a terminal workflow; here the
    injection must stay in place until the control plane has *planned* the
    restart, and only then may the DaemonSet be restored so the step can run.
    """

    deadline = time.monotonic() + timeout_seconds
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        last = regional.cpu_python(
            base.LATEST_NODE_WORKFLOW,
            settings.regional.cluster_id,
            node,
            observed_after.isoformat(),
        )
        planned = base.workflow_planning(last, operation)
        if planned is not None:
            return planned
        time.sleep(5)
    raise RegionalFixtureError(
        f"no workflow planned {operation} for {node} within {timeout_seconds}s: {last}"
    )


def plugin_workflow_errors(
    bundle: dict[str, Any], *, steps: tuple[str, ...], label: str
) -> list[str]:
    """B/C verdict: the documented steps ran, all SUCCEEDED, incident RECOVERED."""

    errors: list[str] = []
    workflow = bundle.get("workflow") or {}
    incident = bundle.get("incident") or {}
    operations = [
        str(item.get("operation")) for item in workflow.get("official_steps") or []
    ]
    if operations != list(steps):
        errors.append(f"{label} official steps {operations} != {list(steps)}")
    if workflow.get("status") != "SUCCEEDED":
        errors.append(
            f"{label} workflow status {workflow.get('status')!r} != SUCCEEDED"
        )
    statuses = {
        str(item.get("operation")): item.get("status")
        for item in workflow.get("step_executions") or []
    }
    for operation in steps:
        if statuses.get(operation) != "SUCCEEDED":
            errors.append(
                f"{label} {operation} {statuses.get(operation)!r} != SUCCEEDED"
            )
    if incident.get("state") != "RECOVERED":
        errors.append(f"{label} incident state {incident.get('state')!r} != RECOVERED")
    return errors


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
    planned: dict[str, Any] = {}
    try:
        plugin.exclude_node()
        unavailable = plugin.wait_allocatable(0)
        planned = _wait_planned_workflow(
            regional,
            settings,
            node=settings.node,
            operation="RESTART_GPU_DEVICE_PLUGIN",
            observed_after=started_at,
        )
        plugin.restore()
        plugin.wait_allocatable(baseline)
        workflow = base.latest_node_workflow(
            regional,
            _base_settings(settings, settings.node),
            observed_after=started_at,
            timeout_seconds=600,
        )
    finally:
        plugin.restore()
    errors = plugin_workflow_errors(
        workflow, steps=GPU_PLUGIN_STEPS, label="GPU plugin"
    )
    return {
        "errors": errors,
        "plugin": info,
        "baseline_allocatable": baseline,
        "unavailable": unavailable,
        "planned_workflow_id": planned.get("request_id"),
        "workflow": workflow,
    }


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
        # Hold the loss until the control plane has planned the restart (the
        # collector needs two 15s samples), then check the training Pods rode
        # through it untouched before handing the plugin back.
        planned = _wait_planned_workflow(
            regional,
            settings,
            node=target,
            operation="RESTART_EFA_DEVICE_PLUGIN",
            observed_after=started_at,
        )
        during = workload.snapshot()
        if {str(item["uid"]) for item in during["pods"]} != uids:
            result["errors"].append("EFA plugin loss recreated training Pods")
        plugin.restore()
        plugin.wait_allocatable(baseline)
        workflow = base.latest_node_workflow(
            regional,
            _base_settings(settings, target),
            observed_after=started_at,
            timeout_seconds=600,
        )
        operations = {
            item.get("operation")
            for item in workflow["workflow"].get("official_steps", [])
        }
        if operations.intersection({"STOP_WORKLOADS", "RESTART_WORKLOAD"}):
            result["errors"].append("EFA plugin recovery restarted workload")
        result["errors"].extend(
            plugin_workflow_errors(workflow, steps=EFA_PLUGIN_STEPS, label="EFA plugin")
        )
        after = workload.snapshot()
        if {str(item["uid"]) for item in after["pods"]} != uids:
            result["errors"].append("training Pods changed after EFA plugin recovery")
        result.update(
            {
                "target_node": target,
                "pod_uids": sorted(uids),
                "plugin": info,
                "planned_workflow_id": planned.get("request_id"),
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


def plan_details(settings: Settings, preflight: dict[str, Any]) -> dict[str, Any]:
    return {
        "risk": "live-node-mutation",
        "predecessor": preflight["predecessor"],
        "mutation": (
            "unbind one EFA function with a 900s bind fail-safe, exclude one "
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
