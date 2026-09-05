#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
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

from scripts.e2e.regional import run_destr001_gpu_reset as reset_case  # noqa: E402
from scripts.e2e.regional.acceptance_runner_common import (  # noqa: E402
    write_json_atomic,
)
from scripts.e2e.regional.collector_acceptance_fixture import (  # noqa: E402
    CollectorAcceptanceFixture,
    collector_setting,
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
    provider_event_actor_matches_role,
    required,
    run_case_main,
    settings_from_arguments,
)
from scripts.e2e.regional.warm_spare_fixture import (  # noqa: E402
    WarmSpareLiveFixture,
)

CASE_IDS = (
    "GF-REGIONAL-COLLECT-004",
    "GF-REGIONAL-COLLECT-008",
    "GF-REGIONAL-COLLECT-013",
    "GF-REGIONAL-COLLECT-014",
    "GF-REGIONAL-COLLECT-015",
)
PREDECESSORS = {
    "GF-REGIONAL-COLLECT-004": "GF-REGIONAL-COLLECT-003",
    "GF-REGIONAL-COLLECT-008": "GF-REGIONAL-COLLECT-012",
    "GF-REGIONAL-COLLECT-013": "GF-REGIONAL-COLLECT-008",
    "GF-REGIONAL-COLLECT-014": "GF-REGIONAL-COLLECT-013",
    "GF-REGIONAL-COLLECT-015": "GF-REGIONAL-COLLECT-017",
}
CONFIRMATIONS = {
    case_id: case_id.replace("GF-REGIONAL-", "").replace("-", "") + "_EXECUTE"
    for case_id in CASE_IDS
}
DESTRUCTIVE_PROBE = Path(__file__).with_name("probes") / "destructive_node_probe.py"
TRAINING_MANIFEST = (
    Path(__file__).with_name("manifests")
    / "training"
    / "xid11-three-node-pytorchjob.yaml"
)


FABRIC_POST = r"""
import json
import os
import sys

from gpu_fault.collectors.sinks import HttpEventSink

payload = json.loads(sys.argv[1])
os.environ["SSL_CERT_FILE"] = os.environ["GPU_FAULT_CONTROL_PLANE_CA_FILE"]
sink = HttpEventSink(
    os.environ["GPU_FAULT_CONTROL_PLANE_URL"],
    bearer_token=os.environ["GPU_FAULT_CONTROL_PLANE_TOKEN"],
    timeout_seconds=15,
    max_attempts=1,
)
result = sink.post("/v1/collector-events/fabric-manager", payload)
print(json.dumps(result, sort_keys=True))
"""


LATEST_NODE_WORKFLOW = r"""
import json
import sys
from datetime import datetime

from gpu_fault.app import ApplicationContext

cluster_id, node_id, observed_after_text = sys.argv[1:]
observed_after = datetime.fromisoformat(observed_after_text.replace("Z", "+00:00"))
store = ApplicationContext.from_environment().store
matches = []
for workflow in store.list_workflows(limit=500, newest_first=True):
    if workflow.created_at < observed_after:
        continue
    try:
        incident = store.get_incident(workflow.incident_id)
    except Exception:
        continue
    if incident.cluster_id != cluster_id or node_id not in incident.node_ids:
        continue
    matches.append({
        "incident": incident.model_dump(mode="json"),
        "workflow": workflow.model_dump(mode="json"),
    })
print(json.dumps({"matches": matches}, sort_keys=True, default=str))
"""


@dataclass(frozen=True)
class Settings:
    regional: RegionalLiveSettings
    case_id: str
    node: str
    second_node: str | None
    host_probe_image: str
    hyperpod_cluster: str
    executor_role_arn: str
    site_file: Path | None
    predecessor_path: Path

    @property
    def confirmation(self) -> str:
        return CONFIRMATIONS[self.case_id]

    def environment(self) -> dict[str, str]:
        result = {
            **self.regional.environment(),
            "GPU_FAULT_COLLECT_CASE": self.case_id,
            "GPU_FAULT_TARGET_NODE": self.node,
            "GPU_FAULT_SECOND_NODE": self.second_node or "",
            "GPU_FAULT_HOST_PROBE_IMAGE": self.host_probe_image,
            "GPU_FAULT_HYPERPOD_CLUSTER_NAME": self.hyperpod_cluster,
            "GPU_FAULT_EXECUTOR_ROLE_ARN": self.executor_role_arn,
            "GPU_FAULT_PREDECESSOR_EVIDENCE": str(self.predecessor_path),
        }
        if self.site_file is not None:
            result["GPU_FAULT_SITE_FILE"] = str(self.site_file)
        return result


def configure(arguments: argparse.Namespace) -> Settings:
    case_id = required(arguments.case, "case ID")
    predecessor_id = PREDECESSORS[case_id]
    predecessor = (
        Path(arguments.predecessor_evidence).expanduser().resolve()
        if arguments.predecessor_evidence
        else (
            arguments.run_dir / "cases" / predecessor_id / f"{predecessor_id}.json"
        ).resolve()
    )
    site_file = (
        Path(arguments.site_file).expanduser().resolve()
        if arguments.site_file
        else None
    )
    return Settings(
        regional=settings_from_arguments(arguments),
        case_id=case_id,
        node=required(
            arguments.node or os.getenv("GPU_FAULT_TARGET_NODE", ""),
            "target node",
        ),
        second_node=arguments.second_node.strip() or None,
        host_probe_image=required(
            arguments.host_probe_image or os.getenv("GPU_FAULT_HOST_PROBE_IMAGE", ""),
            "host probe image",
        ),
        hyperpod_cluster=(
            arguments.hyperpod_cluster
            or os.getenv("GPU_FAULT_HYPERPOD_CLUSTER_NAME", "")
        ).strip(),
        executor_role_arn=(
            arguments.executor_role_arn or os.getenv("GPU_FAULT_EXECUTOR_ROLE_ARN", "")
        ).strip(),
        site_file=site_file,
        predecessor_path=predecessor,
    )


def focused_tests(case_id: str, case_dir: Path) -> dict[str, Any]:
    definitions = {
        "GF-REGIONAL-COLLECT-004": [
            "tests/collectors/test_gpu.py::"
            "test_host_collector_reports_persistent_gpu_and_efa_card_loss",
        ],
        "GF-REGIONAL-COLLECT-008": [
            "tests/orchestration/test_xid.py::"
            "test_xid_48_solo_and_companion_compile_the_same_containment",
        ],
        "GF-REGIONAL-COLLECT-013": [
            "tests/node_agent/test_remediation.py::"
            "test_gpu_reset_is_idempotent_and_rechecks_clients",
        ],
        "GF-REGIONAL-COLLECT-014": [
            "tests/node_agent/test_remediation.py::"
            "test_full_fabric_reset_rejects_inventory_mismatch",
            "tests/node_agent/test_remediation.py::"
            "test_full_fabric_reset_verifies_inventory_and_is_idempotent",
        ],
        "GF-REGIONAL-COLLECT-015": [
            "tests/policy/test_policy.py::"
            "test_always_fatal_sxid_uses_host_restart_branch",
        ],
    }
    command = [sys.executable, "-m", "pytest", "-q", *definitions[case_id]]
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


def read_only_preflight(
    settings: Settings,
    case_dir: Path,
) -> dict[str, Any]:
    regional = RegionalLiveFixture(settings.regional)
    node = regional.node_snapshot(settings.node)
    state = regional.store_snapshot(node=settings.node)
    predecessor_id = PREDECESSORS[settings.case_id]
    predecessor = predecessor_evidence(settings.predecessor_path, predecessor_id)
    tests = focused_tests(settings.case_id, case_dir)
    errors = []
    if not predecessor["valid"]:
        errors.append(f"{predecessor_id} predecessor evidence is not PASS")
    if node["ready"] != "True":
        errors.append("target node is not Ready")
    if node["ownership_annotations"]:
        errors.append("target node has pre-existing workflow ownership")
    if regional.business_workloads(settings.node):
        errors.append("target node has a non-system workload")
    if (state.get("agent") or {}).get("lifecycle_state") != "ACTIVE":
        errors.append("target Node Agent is not ACTIVE")
    if settings.case_id in {
        "GF-REGIONAL-COLLECT-004",
        "GF-REGIONAL-COLLECT-015",
    } and (not settings.hyperpod_cluster or not settings.executor_role_arn):
        errors.append(f"{settings.case_id} requires HyperPod cluster and executor role")
    if not tests["passed"]:
        errors.append("focused regression tests failed")
    result = {
        "release_id": state.get("release_id"),
        "node": node,
        "store": state,
        "predecessor": predecessor,
        "focused_tests": tests,
        "cpu_blast": regional.cpu_blast_snapshot(),
        "errors": errors,
    }
    write_json_atomic(case_dir / "preflight.json", result)
    return result


def reset_fixture(
    settings: Settings,
    *,
    run_id: str,
) -> HostProbeFixture:
    return HostProbeFixture(
        HostProbeSettings(
            kubeconfig=settings.regional.gpu_kubeconfig,
            context=settings.regional.gpu_context,
            namespace=settings.regional.namespace,
            node=settings.node,
            image=settings.host_probe_image,
            case_id=settings.case_id,
            run_id=run_id,
            probe_script=DESTRUCTIVE_PROBE,
            active_deadline_seconds=3600,
        )
    )


def wait_xid_workflow(
    regional: RegionalLiveFixture,
    settings: Settings,
    *,
    marker: str,
    injected_at: datetime,
    case_dir: Path,
    node: str | None = None,
) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        regional.wait_for_workflow(
            node=node or settings.node,
            marker=marker,
            observed_after=injected_at,
            case_dir=case_dir,
            timeout_seconds=1800,
        ),
    )


def run_single_reset(
    settings: Settings,
    regional: RegionalLiveFixture,
    host: HostProbeFixture,
    case_dir: Path,
    *,
    xid: int,
    marker: str,
    run_id: str,
    node: str | None = None,
) -> tuple[dict[str, Any], list[str]]:
    baseline = host.execute("snapshot")
    target_bdf = str(baseline["gpu_inventory"][0]["pci_bdf"])
    host.execute(
        "start-reset-sampler",
        "--run-id",
        run_id,
        "--probe-script",
        host.host_script,
        timeout=60,
    )
    injected_at = datetime.now(timezone.utc)
    host.execute(
        "write-xid",
        "--xid",
        str(xid),
        "--marker",
        marker,
        "--drill-id",
        run_id,
        "--pci-bdf",
        target_bdf.rsplit(".", 1)[0],
        "--case-id",
        settings.case_id,
    )
    state = wait_xid_workflow(
        regional,
        settings,
        marker=marker,
        injected_at=injected_at,
        case_dir=case_dir,
        node=node,
    )
    after = host.execute(
        "snapshot",
        "--since-epoch",
        str(injected_at.timestamp()),
        "--pci-bdf",
        target_bdf.rsplit(".", 1)[0],
        "--run-id",
        run_id,
        timeout=180,
    )
    errors = reset_case.workflow_errors(state)
    errors.extend(
        reset_case.host_errors(
            baseline,
            after,
            expected_gpu_count=len(baseline["gpu_inventory"]),
            target_bdf=target_bdf.rsplit(".", 1)[0],
        )
    )
    host.execute("stop-reset-sampler", "--run-id", run_id, timeout=60)
    return state, errors


def run_collect008(
    settings: Settings,
    regional: RegionalLiveFixture,
    host: HostProbeFixture,
    case_dir: Path,
    attempt: int,
) -> dict[str, Any]:
    baseline = host.execute("snapshot")
    target_bdf = str(baseline["gpu_inventory"][0]["pci_bdf"]).rsplit(".", 1)[0]
    marker63 = f"c008-63-{int(time.time())}-a{attempt}"
    marker48 = f"c008-48-{int(time.time())}-a{attempt}"
    run_id = f"c008-{attempt}"
    host.execute(
        "start-reset-sampler",
        "--run-id",
        run_id,
        "--probe-script",
        host.host_script,
        timeout=60,
    )
    injected_at = datetime.now(timezone.utc)
    host.execute(
        "write-xid",
        "--xid",
        "63",
        "--marker",
        marker63,
        "--drill-id",
        run_id,
        "--pci-bdf",
        target_bdf,
        "--case-id",
        settings.case_id,
    )
    time.sleep(6)
    host.execute(
        "write-xid",
        "--xid",
        "48",
        "--marker",
        marker48,
        "--drill-id",
        run_id,
        "--pci-bdf",
        target_bdf,
        "--case-id",
        settings.case_id,
    )
    state = wait_xid_workflow(
        regional,
        settings,
        marker=marker48,
        injected_at=injected_at,
        case_dir=case_dir,
    )
    errors = reset_case.workflow_errors(state)
    decision = state.get("decision") or {}
    if decision.get("official_action") != "DRAIN_AND_RESET":
        errors.append("XID48 companion decision is not DRAIN_AND_RESET")
    after = host.execute(
        "snapshot",
        "--since-epoch",
        str(injected_at.timestamp()),
        "--pci-bdf",
        target_bdf,
        "--run-id",
        run_id,
        timeout=180,
    )
    errors.extend(
        reset_case.host_errors(
            baseline,
            after,
            expected_gpu_count=len(baseline["gpu_inventory"]),
            target_bdf=target_bdf,
        )
    )
    host.execute("stop-reset-sampler", "--run-id", run_id, timeout=60)
    return {
        "verdict": "PASS" if not errors else "FAIL",
        "errors": errors,
        "markers": [marker63, marker48],
        "workflow_request_id": (state.get("workflow") or {}).get("request_id"),
    }


def run_collect013(
    settings: Settings,
    regional: RegionalLiveFixture,
    host: HostProbeFixture,
    case_dir: Path,
    attempt: int,
) -> dict[str, Any]:
    results = []
    errors = []
    for xid in (109, 62):
        marker = f"c013-{xid}-{int(time.time())}-a{attempt}"
        state, current_errors = run_single_reset(
            settings,
            regional,
            host,
            case_dir / f"xid-{xid}",
            xid=xid,
            marker=marker,
            run_id=f"c013-{xid}-{attempt}",
        )
        errors.extend(current_errors)
        results.append(
            {
                "xid": xid,
                "marker": marker,
                "workflow_request_id": (
                    (state.get("workflow") or {}).get("request_id")
                ),
            }
        )
        time.sleep(10)
    return {
        "verdict": "PASS" if not errors else "FAIL",
        "errors": errors,
        "runs": results,
    }


def post_fabric_event(
    regional: RegionalLiveFixture,
    payload: dict[str, Any],
) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        regional.executor_python(
            FABRIC_POST,
            json.dumps(payload, sort_keys=True),
        ),
    )


def latest_node_workflow(
    regional: RegionalLiveFixture,
    settings: Settings,
    *,
    observed_after: datetime,
    timeout_seconds: int = 1200,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        last = regional.cpu_python(
            LATEST_NODE_WORKFLOW,
            settings.regional.cluster_id,
            settings.node,
            observed_after.isoformat(),
        )
        matches = last.get("matches") or []
        if matches:
            workflow = matches[0]["workflow"]
            if workflow.get("status") in {"SUCCEEDED", "FAILED", "BLOCKED"}:
                return cast(dict[str, Any], matches[0])
        time.sleep(5)
    raise RegionalFixtureError(f"node workflow did not converge: {last}")


def restore_incident(
    regional: RegionalLiveFixture,
    settings: Settings,
    *,
    incident_id: str,
    profile_version: str,
    reason: str,
) -> dict[str, Any] | None:
    if not incident_id:
        return None
    node = regional.node_snapshot(settings.node)
    if not node["ownership_annotations"]:
        return None
    restore = WarmSpareLiveFixture(regional, "")
    created = restore.create_restore_workflow(
        incident_id=incident_id,
        node=settings.node,
        profile_version=profile_version,
        reason=reason,
    )
    return cast(
        dict[str, Any],
        restore.wait_workflow_id(str(created["workflow_request_id"])),
    )


def run_collect004(
    settings: Settings,
    regional: RegionalLiveFixture,
    collector: CollectorAcceptanceFixture,
    case_dir: Path,
    attempt: int,
) -> dict[str, Any]:
    baseline = collector.snapshot()
    env = baseline["collector_env"]
    expected = collector_setting(env, "GPU_FAULT_EXPECTED_GPU_COUNT")
    interval = collector_setting(env, "GPU_FAULT_HOST_INTERVAL_SECONDS")
    run_id = f"c004-{attempt}"
    started_at = datetime.now(timezone.utc)
    collector.execute(
        "override-expected-gpu-count",
        "--run-id",
        run_id,
        "--value",
        str(expected + 1),
        "--restore-seconds",
        "600",
    )
    result: dict[str, Any] = {"errors": []}
    try:
        time.sleep(interval * 3)
        workflow_state = latest_node_workflow(
            regional,
            settings,
            observed_after=started_at,
        )
        result["workflow_state"] = workflow_state
        workflow = workflow_state["workflow"]
        operations = [
            item.get("operation") for item in workflow.get("official_steps", [])
        ]
        if "RESTART_NODE" not in operations:
            result["errors"].append("inventory mismatch did not plan RESTART_NODE")
        node_after = regional.wait_node_ready(
            settings.node,
            timeout_seconds=1800,
            expected_boot_id=str(baseline["boot_id"]),
        )
        result["node_after"] = node_after
        if node_after["boot_id"] == baseline["boot_id"]:
            result["errors"].append("inventory mismatch did not reboot the node")
        provider = regional.provider_events(started_at, datetime.now(timezone.utc))
        result["provider_events"] = provider
        reboot = [
            item
            for item in provider
            if item["event_name"] in {"BatchRebootClusterNodes", "RebootClusterNodes"}
        ]
        if len(reboot) != 1:
            result["errors"].append("CloudTrail does not contain exactly one reboot")
        elif not provider_event_actor_matches_role(
            reboot[0],
            settings.executor_role_arn,
        ):
            result["errors"].append("reboot actor is not the executor role")
    finally:
        try:
            result["collector_restore"] = collector.execute(
                "restore-collector-env",
                "--run-id",
                run_id,
                timeout=300,
            )
        except Exception as exc:
            result["errors"].append(
                f"collector env restore failed: {type(exc).__name__}: {exc}"
            )
    result["verdict"] = "PASS" if not result["errors"] else "FAIL"
    return result


def run_collect014(
    settings: Settings,
    regional: RegionalLiveFixture,
    reset_host: HostProbeFixture,
    collector: CollectorAcceptanceFixture,
    case_dir: Path,
    attempt: int,
    profile_version: str,
) -> dict[str, Any]:
    baseline = reset_host.execute("snapshot")
    gpu = baseline["gpu_inventory"][0]
    bdf = str(gpu["pci_bdf"])
    fail_marker = f"c014-fail-{int(time.time())}-a{attempt}"
    old_time = datetime.now(timezone.utc) - timedelta(minutes=2)
    fail_payload = {
        "cluster_id": settings.regional.cluster_id,
        "node_id": settings.node,
        "record_id": fail_marker,
        "observed_at": old_time.isoformat(),
        "message": (
            f"nvidia-nvswitch0: SXid (PCI:{bdf}): 10003, Fatal, "
            f"Link 3 NVSWITCH_NON_CORRECTABLE marker={fail_marker}"
        ),
        "source": "/var/log/fabricmanager.log",
        "runtime_profile_version": profile_version,
    }
    post_fabric_event(regional, fail_payload)
    fail_state = collector.wait_marker(
        fail_marker,
        case_dir=case_dir / "fail-closed",
        timeout_seconds=300,
        terminal_workflow=True,
    )
    errors = []
    if (
        not fail_state.get("workflows")
        or fail_state["workflows"][0].get("status") != "BLOCKED"
    ):
        errors.append("incomplete inventory SXID did not fail closed")
    marker = f"c014-full-{int(time.time())}-a{attempt}"
    run_id = f"c014-{attempt}"
    reset_host.execute(
        "start-reset-sampler",
        "--run-id",
        run_id,
        "--probe-script",
        reset_host.host_script,
        timeout=60,
    )
    injected_at = datetime.now(timezone.utc)
    collector.execute(
        "append-sxid",
        "--sxid",
        "10003",
        "--marker",
        marker,
        "--pci-bdf",
        bdf,
        "--classification",
        "Fatal",
        "--message",
        "NVSWITCH_NON_CORRECTABLE",
        "--include-switch",
    )
    state = collector.wait_marker(
        marker,
        case_dir=case_dir / "positive",
        timeout_seconds=1800,
        terminal_workflow=True,
    )
    workflow = (state.get("workflows") or [{}])[0]
    if workflow.get("status") != "SUCCEEDED":
        errors.append("full GPU/NVSwitch reset workflow is not SUCCEEDED")
    execution = next(
        (
            item
            for item in workflow.get("step_executions", [])
            if item.get("operation") == "RESET_ALL_GPUS_NVSWITCHES"
            and item.get("status") == "SUCCEEDED"
        ),
        None,
    )
    if execution is None:
        errors.append("RESET_ALL_GPUS_NVSWITCHES did not succeed")
    after = reset_host.execute(
        "snapshot",
        "--since-epoch",
        str(injected_at.timestamp()),
        "--pci-bdf",
        bdf.rsplit(".", 1)[0],
        "--run-id",
        run_id,
        timeout=180,
    )
    rows = [
        item
        for item in after["ledger"]
        if item.get("operation") == "RESET_ALL_GPUS_NVSWITCHES"
    ]
    if len(rows) != 1:
        errors.append("full fabric reset ledger count is not one")
    if len(after["gpu_inventory"]) != len(baseline["gpu_inventory"]):
        errors.append("GPU inventory changed after full fabric reset")
    reset_host.execute("stop-reset-sampler", "--run-id", run_id, timeout=60)
    return {
        "verdict": "PASS" if not errors else "FAIL",
        "errors": errors,
        "fail_closed_marker": fail_marker,
        "positive_marker": marker,
        "fail_closed": fail_state,
        "positive": state,
    }


def run_collect015(
    settings: Settings,
    regional: RegionalLiveFixture,
    collector: CollectorAcceptanceFixture,
    case_dir: Path,
    attempt: int,
) -> dict[str, Any]:
    baseline_node = regional.node_snapshot(settings.node)
    baseline_provider = WarmSpareLiveFixture(
        regional,
        settings.hyperpod_cluster,
    ).provider_inventory()
    snapshot = collector.snapshot()
    bdf = str(snapshot["gpu_inventory"][0]["pci_bdf"])
    marker = f"c015-{int(time.time())}-a{attempt}"
    started_at = datetime.now(timezone.utc)
    collector.execute(
        "append-sxid",
        "--sxid",
        "23001",
        "--marker",
        marker,
        "--pci-bdf",
        bdf,
        "--classification",
        "Fatal",
        "--message",
        "Always Fatal NVSWITCH_FATAL",
        "--include-switch",
    )
    state = collector.wait_marker(
        marker,
        case_dir=case_dir,
        timeout_seconds=2400,
        terminal_workflow=True,
    )
    node_after = regional.wait_node_ready(
        settings.node,
        timeout_seconds=1800,
        expected_boot_id=str(baseline_node["boot_id"]),
    )
    errors = []
    workflow = (state.get("workflows") or [{}])[0]
    if workflow.get("status") != "SUCCEEDED":
        errors.append("ALWAYS_FATAL reboot workflow is not SUCCEEDED")
    if node_after["boot_id"] == baseline_node["boot_id"]:
        errors.append("node boot ID did not change")
    provider_after = WarmSpareLiveFixture(
        regional,
        settings.hyperpod_cluster,
    ).provider_inventory()
    if provider_after != baseline_provider:
        errors.append("provider node identity changed during reboot")
    events = regional.provider_events(started_at, datetime.now(timezone.utc))
    reboot = [
        item
        for item in events
        if item["event_name"] in {"BatchRebootClusterNodes", "RebootClusterNodes"}
    ]
    replace = [
        item
        for item in events
        if item["event_name"] in {"BatchReplaceClusterNodes", "ReplaceClusterNodes"}
    ]
    if len(reboot) != 1:
        errors.append("CloudTrail reboot count is not one")
    elif not provider_event_actor_matches_role(
        reboot[0],
        settings.executor_role_arn,
    ):
        errors.append("reboot actor is not the executor role")
    if replace:
        errors.append("CloudTrail contains provider replacement")
    return {
        "verdict": "PASS" if not errors else "FAIL",
        "errors": errors,
        "marker": marker,
        "workflow": workflow,
        "provider_events": events,
    }


def render_named_training_manifest(
    destination: Path,
    *,
    name: str,
) -> Path:
    yaml = importlib.import_module("yaml")
    document = yaml.safe_load(TRAINING_MANIFEST.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise RegionalFixtureError("training manifest is not a mapping")
    document.setdefault("metadata", {})["name"] = name
    replicas = document["spec"]["pytorchReplicaSpecs"]
    for item in replicas.values():
        labels = item["template"].setdefault("metadata", {}).setdefault("labels", {})
        labels["app"] = name
        affinity = (
            item["template"]
            .setdefault("spec", {})
            .get("affinity", {})
            .get("podAntiAffinity", {})
            .get("requiredDuringSchedulingIgnoredDuringExecution", [])
        )
        for rule in affinity:
            rule["labelSelector"]["matchLabels"]["app"] = name
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(document, sort_keys=False),
        encoding="utf-8",
    )
    destination.chmod(0o600)
    return destination


class DevicePluginFixture:
    def __init__(
        self,
        regional: RegionalLiveFixture,
        *,
        token: str,
        node: str,
        resource: str,
    ) -> None:
        self.regional = regional
        self.token = token
        self.node = node
        self.resource = resource
        self.namespace = ""
        self.name = ""
        self.affinity: dict[str, Any] | None = None

    def discover(self) -> dict[str, Any]:
        value = json.loads(
            self.regional.kubectl(
                "gpu",
                "get",
                "daemonset",
                "-o",
                "json",
                all_namespaces=True,
            )
        )
        matches = []
        for item in value.get("items", []):
            encoded = json.dumps(item, sort_keys=True).lower()
            if self.token.lower() in encoded:
                matches.append(item)
        if len(matches) != 1:
            raise RegionalFixtureError(
                f"expected one {self.token} DaemonSet, found {len(matches)}"
            )
        item = matches[0]
        self.namespace = str(item["metadata"]["namespace"])
        self.name = str(item["metadata"]["name"])
        self.affinity = item["spec"]["template"]["spec"].get("affinity")
        return {
            "namespace": self.namespace,
            "name": self.name,
            "affinity": self.affinity,
        }

    def _patch_affinity(self, affinity: dict[str, Any] | None) -> None:
        self.regional.kubectl(
            "gpu",
            "patch",
            "daemonset",
            self.name,
            "--type=merge",
            "-p",
            json.dumps(
                {"spec": {"template": {"spec": {"affinity": affinity}}}},
                sort_keys=True,
            ),
            namespace=self.namespace,
        )

    def exclude_node(self) -> None:
        if not self.name:
            self.discover()
        affinity = {
            "nodeAffinity": {
                "requiredDuringSchedulingIgnoredDuringExecution": {
                    "nodeSelectorTerms": [
                        {
                            "matchExpressions": [
                                {
                                    "key": "kubernetes.io/hostname",
                                    "operator": "NotIn",
                                    "values": [self.node],
                                }
                            ]
                        }
                    ]
                }
            }
        }
        self._patch_affinity(affinity)
        value = json.loads(
            self.regional.kubectl(
                "gpu",
                "get",
                "pod",
                "-o",
                "json",
                namespace=self.namespace,
            )
        )
        for pod in value.get("items", []):
            if pod.get("spec", {}).get("nodeName") != self.node:
                continue
            owners = pod["metadata"].get("ownerReferences", [])
            if any(owner.get("name") == self.name for owner in owners):
                self.regional.kubectl(
                    "gpu",
                    "delete",
                    "pod",
                    str(pod["metadata"]["name"]),
                    "--grace-period=0",
                    "--force",
                    namespace=self.namespace,
                )

    def wait_allocatable(
        self,
        expected: int,
        *,
        timeout_seconds: int = 300,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        last: dict[str, Any] = {}
        while time.monotonic() < deadline:
            last = self.regional.node_metadata(self.node)
            node = json.loads(
                self.regional.kubectl(
                    "gpu",
                    "get",
                    "node",
                    self.node,
                    "-o",
                    "json",
                )
            )
            value = int(
                node.get("status", {}).get("allocatable", {}).get(self.resource, 0)
            )
            last["allocatable"] = value
            if value == expected:
                return last
            time.sleep(5)
        raise RegionalFixtureError(
            f"{self.resource} allocatable did not become {expected}: {last}"
        )

    def restore(self) -> None:
        self._patch_affinity(self.affinity)
        self.regional.kubectl(
            "gpu",
            "rollout",
            "status",
            f"daemonset/{self.name}",
            "--timeout=600s",
            namespace=self.namespace,
            timeout=630,
        )


def plan_details(settings: Settings, preflight: dict[str, Any]) -> dict[str, Any]:
    return {
        "risk": "destructive",
        "case_id": settings.case_id,
        "predecessor": preflight["predecessor"],
        "target_node": settings.node,
        "mutation": {
            "GF-REGIONAL-COLLECT-004": "temporary expected-count mismatch and real reboot",
            "GF-REGIONAL-COLLECT-008": "real kmsg XID63+48 and one GPU reset",
            "GF-REGIONAL-COLLECT-013": "real kmsg XID109 and XID62 single-GPU resets",
            "GF-REGIONAL-COLLECT-014": "fail-closed API SXID then real FM full fabric reset",
            "GF-REGIONAL-COLLECT-015": "real FM ALWAYS_FATAL SXID and HyperPod reboot",
        }[settings.case_id],
        "preflight_identity": {
            "release_id": preflight["release_id"],
            "node_uid": preflight["node"]["uid"],
            "agent_generation": (preflight["store"].get("agent") or {}).get(
                "generation"
            ),
        },
        "stop_conditions": [
            "predecessor evidence is not PASS",
            "target node, Agent or release baseline drifts",
            "host-side fail-safe or detached sampling cannot be armed",
            "case-specific workflow or physical evidence differs from contract",
            "cleanup cannot restore services, scheduling or provider identity",
        ],
        "rollback": {
            "host_probe_active_deadline": True,
            "restore_services_and_quiesce_state": True,
            "validation_first_node_restore": True,
            "never_fallback_to_provider_replace": True,
        },
        "preflight": preflight,
    }


def execute_case(
    settings: Settings,
    run_dir: Path,
    attempt: int,
    maintenance_window_end: datetime,
) -> int:
    case_dir = run_dir / "cases" / settings.case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    preflight = read_only_preflight(settings, case_dir)
    if preflight["errors"]:
        raise RegionalFixtureError(
            "preflight failed: " + "; ".join(preflight["errors"])
        )
    if datetime.now(timezone.utc) >= maintenance_window_end:
        raise RegionalFixtureError("approved maintenance window has ended")
    regional = RegionalLiveFixture(settings.regional)
    host = reset_fixture(
        settings,
        run_id=f"{settings.case_id.lower()}-{attempt}",
    )
    result: dict[str, Any] = {
        "case_id": settings.case_id,
        "attempt": attempt,
        "verdict": "FAIL",
    }
    collector: CollectorAcceptanceFixture | None = None
    try:
        if settings.case_id in {
            "GF-REGIONAL-COLLECT-004",
            "GF-REGIONAL-COLLECT-014",
            "GF-REGIONAL-COLLECT-015",
        }:
            collector = CollectorAcceptanceFixture(
                regional,
                node=settings.node,
                image=settings.host_probe_image,
                case_id=settings.case_id,
                run_id=f"{settings.case_id.lower()}-collector-{attempt}",
            )
            collector.create()
        if settings.case_id in {
            "GF-REGIONAL-COLLECT-008",
            "GF-REGIONAL-COLLECT-013",
            "GF-REGIONAL-COLLECT-014",
        }:
            host.create()
        if settings.case_id == "GF-REGIONAL-COLLECT-008":
            result.update(run_collect008(settings, regional, host, case_dir, attempt))
        elif settings.case_id == "GF-REGIONAL-COLLECT-013":
            result.update(run_collect013(settings, regional, host, case_dir, attempt))
        elif settings.case_id == "GF-REGIONAL-COLLECT-004" and collector is not None:
            result.update(
                run_collect004(settings, regional, collector, case_dir, attempt)
            )
        elif settings.case_id == "GF-REGIONAL-COLLECT-014" and collector is not None:
            result.update(
                run_collect014(
                    settings,
                    regional,
                    host,
                    collector,
                    case_dir,
                    attempt,
                    str(
                        (preflight["store"].get("profile") or {}).get("profile_version")
                        or ""
                    ),
                )
            )
        elif settings.case_id == "GF-REGIONAL-COLLECT-015" and collector is not None:
            result.update(
                run_collect015(settings, regional, collector, case_dir, attempt)
            )
        else:
            result["error"] = (
                "case handler requires its dedicated staged mutation fixture; "
                "preflight and safety plan are implemented but execution is blocked"
            )
        if regional.cpu_blast_snapshot() != preflight["cpu_blast"]:
            result.setdefault("errors", []).append(
                "control-plane EKS state differs from baseline"
            )
            result["verdict"] = "FAIL"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        residuals: dict[str, Any] = {}
        if settings.case_id in {
            "GF-REGIONAL-COLLECT-008",
            "GF-REGIONAL-COLLECT-013",
            "GF-REGIONAL-COLLECT-014",
        }:
            try:
                residuals["reset"] = host.cleanup()
            except Exception as exc:
                residuals["reset"] = {"cleanup_error": True}
                result["cleanup_error"] = f"{type(exc).__name__}: {exc}"
                result["verdict"] = "FAIL"
        if collector is not None:
            try:
                residuals["collector"] = collector.cleanup()
            except Exception as exc:
                residuals["collector"] = {"cleanup_error": True}
                result["collector_cleanup_error"] = f"{type(exc).__name__}: {exc}"
                result["verdict"] = "FAIL"
        result["probe_residuals"] = residuals
        if any(
            any(bool(item) for item in value.values()) for value in residuals.values()
        ):
            result["verdict"] = "FAIL"
    write_json_atomic(case_dir / f"{settings.case_id}.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["verdict"] == "PASS" else 1


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Run one guarded destructive Collector acceptance case."
    )
    add_live_arguments(value, confirmation="CASE_SPECIFIC_CONFIRMATION")
    value.add_argument("--case", choices=CASE_IDS, required=True)
    value.add_argument("--cpu-kubeconfig", default="")
    value.add_argument("--gpu-kubeconfig", default="")
    value.add_argument("--gpu-context", default="")
    value.add_argument("--namespace", default="gpu-fault-system")
    value.add_argument("--cluster-id", default="")
    value.add_argument("--region", default="")
    value.add_argument("--node", default="")
    value.add_argument("--second-node", default="")
    value.add_argument("--host-probe-image", default="")
    value.add_argument("--hyperpod-cluster", default="")
    value.add_argument("--executor-role-arn", default="")
    value.add_argument("--site-file", default="")
    value.add_argument("--predecessor-evidence", default="")
    return value


def main() -> int:
    install_site_profile()
    arguments = parser().parse_args()
    os.umask(0o077)
    install_abort_signals()
    settings = configure(arguments)
    case_dir = arguments.run_dir / "cases" / settings.case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    if not arguments.execute:
        preflight = read_only_preflight(settings, case_dir)
        plan = build_plan(
            run_dir=arguments.run_dir,
            case_id=settings.case_id,
            attempt=arguments.attempt,
            confirmation=settings.confirmation,
            environment=settings.environment(),
            details=plan_details(settings, preflight),
        )
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0 if not preflight["errors"] else 1
    if arguments.confirm != settings.confirmation:
        raise RegionalFixtureError(
            f"confirmation must be exactly {settings.confirmation}"
        )
    deadline = authorize_execution(
        arguments,
        case_id=settings.case_id,
        confirmation=settings.confirmation,
        environment=settings.environment(),
    )
    return execute_case(settings, arguments.run_dir, arguments.attempt, deadline)


if __name__ == "__main__":
    raise SystemExit(run_case_main(main))
