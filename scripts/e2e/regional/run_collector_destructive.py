#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
import time
from collections.abc import Callable
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
    select_workflow,
)
from scripts.e2e.regional.collector_action_guard import (  # noqa: E402
    bounded_collector_case,
    require_action_time,
)
from scripts.e2e.regional.collector_case_cleanup import CaseCleanup as CaseCleanup  # noqa: E402
from scripts.e2e.regional.collector_device_plugin_fixture import (  # noqa: E402
    DevicePluginFixture as DevicePluginFixture,
)
from scripts.e2e.regional.collector_reset_evidence import (  # noqa: E402
    physical_reset_errors,
)
from scripts.e2e.regional.collector_reset_runner import (  # noqa: E402
    FULL_RESET_COOLDOWN_SECONDS,
    FULL_RESET_STABLE_SAMPLES,
    run_full_reset_variant as run_full_reset_variant,
    stop_sampler as stop_sampler,
    wait_full_reset_stability as wait_full_reset_stability,
)
from scripts.e2e.regional.collector_env_restore import (  # noqa: E402
    restore_collector_env as restore_collector_env,
)
from scripts.e2e.regional.collector_inventory_reboot import (  # noqa: E402
    HOST_INVENTORY_EVIDENCE as HOST_INVENTORY_EVIDENCE,
    debounce_errors as debounce_errors,
    mismatch_finding as mismatch_finding,
    run_collect004 as run_collect004,
    wait_mismatch_finding as wait_mismatch_finding,
)
from scripts.e2e.regional.collector_reboot_evidence import (  # noqa: E402
    capture_reboot_scope,
)
from scripts.e2e.regional.collector_sxid_evidence import (  # noqa: E402
    FULL_RESET_VARIANTS,
)
from scripts.e2e.regional.host_probe_fixture import (  # noqa: E402
    HostProbeFixture,
    HostProbeSettings,
)
from scripts.e2e.regional.live_driver_guard import (  # noqa: E402
    CaseSurface,
    add_live_arguments,
    record_focused_tests,
    reusable_focused_tests,
    run_selected_case,
)
from scripts.e2e.regional.regional_commands import RegionalFixtureError  # noqa: E402
from scripts.e2e.regional.regional_live_fixture import (  # noqa: E402
    RegionalLiveFixture,
    RegionalLiveSettings,
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
# COLLECT-013 proves one single-GPU reset from a real kmsg XID. XID 109 and 62
# reach the identical RESET_GPU contract, so the case runs one of them (109 by
# default, `--xid 62` for the other) instead of two full quiesce/reset cycles
# with the same verdict.
COLLECT013_XIDS = (109, 62)
DEFAULT_COLLECT013_XID = 109
# COLLECT-014 fail-closed direction: how far back the API SXID is dated so no
# stored inventory sample can count as evidence for it (see run_collect014).
# Must stay below GPU_FAULT_FAULT_ACTION_MAX_AGE_SECONDS (900s).
FAIL_CLOSED_EVENT_AGE = timedelta(minutes=8)
# Ingest accepts an inventory sample as evidence while
# event.observed_at - sample.observed_at >= -30s (src/gpu_fault/app/ingest/
# faults.py, _fresh_gpu_inventory). The back-dated event is only safe when the
# newest stored sample is newer than event + 30s by a margin ingestion lag
# cannot eat; 60s is that margin.
INGEST_FUTURE_TOLERANCE = timedelta(seconds=30)
FAIL_CLOSED_SAFETY_MARGIN = timedelta(seconds=60)
FAIL_CLOSED_REASON = "requires fabric_partition and complete node GPU inventory"
FULL_FABRIC_RESET_ACTIONS = frozenset({"RESET_ALL_GPUS_AND_NVSWITCHES"})
REBOOT_EVENTS = frozenset({"BatchRebootClusterNodes", "RebootClusterNodes"})
REPLACE_EVENTS = frozenset({"BatchReplaceClusterNodes", "ReplaceClusterNodes"})
# `GPU_FAULT_ALLOW_HYPERPOD_REPLACE` unset or any of these spellings is "false".
FALSE_ENV_VALUES = frozenset({"", "0", "false", "no", "off"})
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


GPU_INVENTORY_SNAPSHOT = r"""
import json
import sys

from gpu_fault.app import ApplicationContext

cluster_id, node_id = sys.argv[1:]
store = ApplicationContext.from_environment().store
snapshot = store.get_gpu_inventory_snapshot(cluster_id, node_id)
print(json.dumps({
    "present": snapshot is not None,
    "observed_at": snapshot.observed_at.isoformat() if snapshot else None,
    "source_boot_id": snapshot.source_boot_id if snapshot else None,
    "device_count": len(snapshot.devices) if snapshot else 0,
    "legacy_observed_at": [
        item.observed_at.isoformat()
        for item in store.list_gpu_metrics_latest(cluster_id, node_id)
        if item.sample.gpu_uuid
    ],
}, sort_keys=True, default=str))
"""


HYPERPOD_SUBMISSION = r"""
import json
import sys

from gpu_fault.app import ApplicationContext
from gpu_fault.hyperpod import hyperpod_submission_idempotency_key

request_id, hyperpod_cluster = sys.argv[1:]
store = ApplicationContext.from_environment().store
restart_commands = [
    item
    for item in store.list_remote_commands(workflow_request_ids=[request_id])
    if item.step.operation.value == "RESTART_NODE"
]
submission = None
if restart_commands:
    command = restart_commands[-1]
    key = command.result_details.get(
        "submission_idempotency_key"
    ) or hyperpod_submission_idempotency_key(
        request_id, command.step_index, command.step.operation
    )
    try:
        submission = store.get_hyperpod_submission(hyperpod_cluster, key).model_dump(
            mode="json"
        )
    except Exception:
        submission = None
print(json.dumps({
    "submission": submission,
    "restart_commands": [
        {
            "command_id": item.command_id,
            "status": str(getattr(item.status, "value", item.status)),
            "step_index": item.step_index,
        }
        for item in restart_commands
    ],
}, sort_keys=True, default=str))
"""


EXECUTOR_REPLACE_FLAG = r"""
import json
import os

print(json.dumps({
    "GPU_FAULT_ALLOW_HYPERPOD_REPLACE": os.environ.get(
        "GPU_FAULT_ALLOW_HYPERPOD_REPLACE", ""
    ),
}, sort_keys=True))
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
    xid: int = DEFAULT_COLLECT013_XID
    # Extra collection time allowed between the first mismatch and threshold.
    debounce_tolerance: float = 0.5

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
        if self.case_id == "GF-REGIONAL-COLLECT-013":
            result["GPU_FAULT_COLLECT013_XID"] = str(self.xid)
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
    tolerance = float(getattr(arguments, "debounce_tolerance", 0.5))
    if not math.isfinite(tolerance) or not 0 <= tolerance <= 1:
        raise RegionalFixtureError("debounce tolerance must be finite and within 0..1")
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
        xid=int(getattr(arguments, "xid", DEFAULT_COLLECT013_XID)),
        debounce_tolerance=tolerance,
    )


FOCUSED_TESTS = {
    "GF-REGIONAL-COLLECT-004": [
        "tests/collectors/test_gpu.py::"
        "test_host_collector_reports_persistent_gpu_and_efa_card_loss",
        "tests/regional/test_collector_inventory_sampling.py",
        "tests/regional/test_collector_inventory_reboot_runner.py",
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
        "tests/policy/test_policy.py::"
        "test_code_specific_sxid_requires_full_fabric_reset",
    ],
    "GF-REGIONAL-COLLECT-015": [
        "tests/policy/test_policy.py::test_always_fatal_sxid_uses_host_restart_branch",
    ],
}


def focused_tests(
    case_id: str,
    case_dir: Path,
    *,
    reuse_plan: Path | None = None,
) -> dict[str, Any]:
    """Run the case's focused pytest, or reuse the plan's result on the same tree."""

    if reuse_plan is not None:
        recorded = reusable_focused_tests(reuse_plan)
        if recorded is not None:
            return {**recorded, "reused_from_plan": str(reuse_plan)}
    command = [sys.executable, "-m", "pytest", "-q", *FOCUSED_TESTS[case_id]]
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
    *,
    reuse_plan: Path | None = None,
) -> dict[str, Any]:
    regional = RegionalLiveFixture(settings.regional)
    node = regional.node_snapshot(settings.node)
    state = regional.store_snapshot(node=settings.node)
    predecessor_id = PREDECESSORS[settings.case_id]
    identity = regional.evidence_identity()
    predecessor = predecessor_evidence(
        settings.predecessor_path,
        predecessor_id,
        **identity,
    )
    tests = focused_tests(settings.case_id, case_dir, reuse_plan=reuse_plan)
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
        **identity,
        "node": node,
        "store": state,
        "predecessor": predecessor,
        "focused_tests": tests,
        "cpu_blast": regional.cpu_blast_snapshot(),
        "errors": errors,
    }
    if settings.case_id == "GF-REGIONAL-COLLECT-004":
        try:
            result["reboot_scope"] = capture_reboot_scope(
                regional,
                node=settings.node,
                hyperpod_cluster=settings.hyperpod_cluster,
                executor_role_arn=settings.executor_role_arn,
            )
        except RegionalFixtureError as exc:
            errors.append(f"reboot scope preflight failed: {exc}")
    write_json_atomic(case_dir / "preflight.json", result)
    return result


def reset_fixture(
    settings: Settings,
    *,
    run_id: str,
    case_dir: Path,
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
            state_directory=case_dir / "host-probes",
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
    return regional.wait_for_workflow(
        node=node or settings.node,
        marker=marker,
        observed_after=injected_at,
        case_dir=case_dir,
        timeout_seconds=1800,
    )


def boot_id_errors(baseline: dict[str, Any], after: dict[str, Any]) -> list[str]:
    """A single-GPU or full-fabric reset never reboots the node."""

    if after.get("boot_id") != baseline.get("boot_id"):
        return ["node boot ID changed: the reset became a reboot"]
    return []


def run_single_reset(
    settings: Settings,
    regional: RegionalLiveFixture,
    host: HostProbeFixture,
    case_dir: Path,
    *,
    collector: CollectorAcceptanceFixture,
    cleanup: CaseCleanup,
    xid: int,
    marker: str,
    run_id: str,
    node: str | None = None,
    expected_steps: list[str] | None = None,
    observe_post_restart_workload: (
        Callable[[dict[str, Any]], dict[str, Any]] | None
    ) = None,
) -> tuple[dict[str, Any], list[str]]:
    baseline = host.execute("snapshot")
    audit_before = collector.execute("reset-audit")
    write_json_atomic(case_dir / "reset-audit-before.json", audit_before)
    write_json_atomic(case_dir / "host-baseline.json", baseline)
    target_bdf = str(baseline["gpu_inventory"][0]["pci_bdf"])
    require_action_time(180)
    host.execute(
        "start-reset-sampler",
        "--run-id",
        run_id,
        "--probe-script",
        host.host_script,
        timeout=60,
    )
    errors: list[str] = []
    post_restart_workload: dict[str, Any] | None = None
    try:
        injected_at = datetime.now(timezone.utc)
        require_action_time(180)
        cleanup.register_seed(collector, marker, quiesce_host=host)
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
        if observe_post_restart_workload is not None:
            post_restart_workload = observe_post_restart_workload(state)
            if not isinstance(post_restart_workload, dict) or not post_restart_workload:
                raise RegionalFixtureError("post-restart workload proof is missing")
            state["post_restart_workload"] = post_restart_workload
            write_json_atomic(
                case_dir / "post-restart-workload.json", post_restart_workload
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
        write_json_atomic(case_dir / "host-after.json", after)
        audit_after = collector.execute("reset-audit")
        write_json_atomic(case_dir / "reset-audit-after.json", audit_after)
    finally:
        stop_sampler(host, run_id, errors)
    errors.extend(
        reset_case.workflow_errors(state, xid=xid, expected_steps=expected_steps)
    )
    errors.extend(
        reset_case.host_errors(
            baseline,
            after,
            expected_gpu_count=len(baseline["gpu_inventory"]),
            target_bdf=target_bdf.rsplit(".", 1)[0],
            incident_id=str((state.get("incident") or {}).get("incident_id") or ""),
            workflow_request_id=str(
                (state.get("workflow") or {}).get("request_id") or ""
            ),
            post_restart_workload=post_restart_workload,
        )
    )
    errors.extend(boot_id_errors(baseline, after))
    errors.extend(
        physical_reset_errors(
            audit_before,
            audit_after,
            state,
            node=node or settings.node,
        )
    )
    return state, errors


def companion_xid_errors(solo: dict[str, Any]) -> list[str]:
    """COLLECT-008's first line: XID 63 alone is watched, never acted on.

    ``solo`` is the store's reading for the XID 63 marker. Its event must be
    kmsg-backed like every other kernel XID, its decision MONITOR_ONLY, and
    no workflow may hang off it -- the reset belongs to the XID 48 companion.
    """

    errors = []
    event = solo.get("event") or {}
    decision = solo.get("decision") or {}
    if event.get("xid") != 63:
        errors.append("XID 63 marker did not resolve to an XID 63 event")
    if not str(event.get("evidence_ref") or "").startswith("kmsg://"):
        errors.append("XID 63 evidence is not backed by kmsg://")
    if decision.get("disposition") != "MONITOR_ONLY":
        errors.append(
            f"XID 63 disposition is {decision.get('disposition')!r}, not MONITOR_ONLY"
        )
    if solo.get("workflow"):
        errors.append("solo XID 63 opened a workflow of its own")
    return errors


def run_collect008(
    settings: Settings,
    regional: RegionalLiveFixture,
    host: HostProbeFixture,
    case_dir: Path,
    attempt: int,
    *,
    collector: CollectorAcceptanceFixture,
    cleanup: CaseCleanup,
) -> dict[str, Any]:
    baseline = host.execute("snapshot")
    audit_before = collector.execute("reset-audit")
    write_json_atomic(case_dir / "reset-audit-before.json", audit_before)
    write_json_atomic(case_dir / "host-baseline.json", baseline)
    target_bdf = str(baseline["gpu_inventory"][0]["pci_bdf"]).rsplit(".", 1)[0]
    marker63 = f"c008-63-{int(time.time())}-a{attempt}"
    marker48 = f"c008-48-{int(time.time())}-a{attempt}"
    run_id = f"c008-{attempt}"
    require_action_time(180)
    host.execute(
        "start-reset-sampler",
        "--run-id",
        run_id,
        "--probe-script",
        host.host_script,
        timeout=60,
    )
    errors: list[str] = []
    try:
        injected_at = datetime.now(timezone.utc)
        require_action_time(180)
        cleanup.register_seed(collector, marker63, quiesce_host=host)
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
        require_action_time(180)
        cleanup.register_seed(collector, marker48, quiesce_host=host)
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
        write_json_atomic(case_dir / "host-after.json", after)
        audit_after = collector.execute("reset-audit")
        write_json_atomic(case_dir / "reset-audit-after.json", audit_after)
    finally:
        stop_sampler(host, run_id, errors)
    errors.extend(
        reset_case.workflow_errors(state, xid=48, official_action="DRAIN_AND_RESET")
    )
    errors.extend(
        reset_case.host_errors(
            baseline,
            after,
            expected_gpu_count=len(baseline["gpu_inventory"]),
            target_bdf=target_bdf,
            incident_id=str((state.get("incident") or {}).get("incident_id") or ""),
            workflow_request_id=str(
                (state.get("workflow") or {}).get("request_id") or ""
            ),
        )
    )
    errors.extend(boot_id_errors(baseline, after))
    errors.extend(
        physical_reset_errors(audit_before, audit_after, state, node=settings.node)
    )
    solo = regional.store_snapshot(
        node=settings.node,
        marker=marker63,
        observed_after=injected_at,
        queue_attempts=1,
    )
    errors.extend(companion_xid_errors(solo))
    return {
        "verdict": "PASS" if not errors else "FAIL",
        "errors": errors,
        "markers": [marker63, marker48],
        "workflow_request_id": (state.get("workflow") or {}).get("request_id"),
        "xid63": {
            "event_id": (solo.get("event") or {}).get("event_id"),
            "disposition": (solo.get("decision") or {}).get("disposition"),
        },
    }


def run_collect013(
    settings: Settings,
    regional: RegionalLiveFixture,
    host: HostProbeFixture,
    case_dir: Path,
    attempt: int,
    *,
    collector: CollectorAcceptanceFixture,
    cleanup: CaseCleanup,
) -> dict[str, Any]:
    xid = settings.xid
    if xid not in COLLECT013_XIDS:
        raise RegionalFixtureError(f"COLLECT-013 XID must be one of {COLLECT013_XIDS}")
    marker = f"c013-{xid}-{int(time.time())}-a{attempt}"
    state, errors = run_single_reset(
        settings,
        regional,
        host,
        case_dir / f"xid-{xid}",
        collector=collector,
        cleanup=cleanup,
        xid=xid,
        marker=marker,
        run_id=f"c013-{xid}-{attempt}",
    )
    return {
        "verdict": "PASS" if not errors else "FAIL",
        "errors": errors,
        "xid": xid,
        "runs": [
            {
                "xid": xid,
                "marker": marker,
                "workflow_request_id": (state.get("workflow") or {}).get("request_id"),
            }
        ],
    }


def post_fabric_event(
    regional: RegionalLiveFixture,
    payload: dict[str, Any],
) -> dict[str, Any]:
    # One attempt: the post is a mutation. A retry after a failure that
    # happened *after* the control plane accepted the event posts it twice.
    require_action_time(180)
    return regional.executor_python(
        FABRIC_POST,
        json.dumps(payload, sort_keys=True),
        attempts=1,
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
                # The newest workflow is the one whose terminal state the
                # caller waits for, but a case's injection can leave a chain
                # behind it (reboot -> failed validation -> replace-after), so
                # every workflow the node grew since the injection rides along.
                return cast(dict[str, Any], {**matches[0], "matches": matches})
        time.sleep(5)
    raise RegionalFixtureError(f"node workflow did not converge: {last}")


def wait_planned_workflow(
    regional: RegionalLiveFixture,
    settings: Settings,
    *,
    operation: str,
    observed_after: datetime,
    timeout_seconds: int,
) -> dict[str, Any]:
    """The workflow that planned ``operation`` for the node, in any status.

    ``latest_node_workflow`` waits for a terminal workflow; COLLECT-004 needs
    the moment the reboot is *planned*, because that is when the injected
    override has done its work and must come off the node.
    """

    deadline = time.monotonic() + timeout_seconds
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        last = regional.cpu_python(
            LATEST_NODE_WORKFLOW,
            settings.regional.cluster_id,
            settings.node,
            observed_after.isoformat(),
        )
        planned = workflow_planning(last, operation)
        if planned is not None:
            return planned
        time.sleep(5)
    raise RegionalFixtureError(
        f"no workflow planned {operation} for {settings.node} within "
        f"{timeout_seconds}s: {last}"
    )


def workflow_planning(
    workflow_state: dict[str, Any],
    operation: str,
) -> dict[str, Any] | None:
    """The workflow in a node's chain whose plan contains ``operation``.

    COLLECT-004 asked the *latest* workflow for RESTART_NODE and failed twice
    while the node was demonstrably rebooting: the latest one was the
    replace-after escalation that followed the post-reboot validation failure,
    not the inventory-mismatch workflow that had planned and executed the
    reboot. Look through the whole chain, newest first.
    """

    candidates = workflow_state.get("matches") or [workflow_state]
    for match in candidates:
        workflow = match.get("workflow") or {}
        operations = [
            item.get("operation") for item in workflow.get("official_steps", [])
        ]
        if operation in operations:
            return cast(dict[str, Any], workflow)
    return None


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
    return restore.wait_workflow_id(str(created["workflow_request_id"]))


def fail_closed_timing_errors(
    inventory: dict[str, Any],
    *,
    event_time: datetime,
) -> list[str]:
    """Both dedicated and legacy inventory must be outside the event window."""

    if type(inventory.get("present")) is not bool:
        return ["dedicated GPU inventory presence is unknown; SXID is not posted"]
    legacy = inventory.get("legacy_observed_at")
    if not isinstance(legacy, list):
        return ["legacy GPU inventory was not checked; SXID is not posted"]
    samples = [
        *([inventory.get("observed_at")] if inventory["present"] else []),
        *legacy,
    ]
    needed = INGEST_FUTURE_TOLERANCE + FAIL_CLOSED_SAFETY_MARGIN
    for sample in samples:
        try:
            observed_at = datetime.fromisoformat(str(sample).replace("Z", "+00:00"))
            if observed_at.tzinfo is None:
                raise ValueError("inventory timestamp has no timezone")
            margin = observed_at - event_time
        except (TypeError, ValueError):
            return ["stored GPU inventory time is unknown; SXID is not posted"]
        if margin < needed:
            return [
                f"stored GPU inventory ({observed_at.isoformat()}) is only "
                f"{margin.total_seconds():.0f}s newer than the back-dated event; "
                f"{needed.total_seconds():.0f}s are required for the fail-closed "
                "premise to hold, so the SXID is not posted"
            ]
    return []


def run_collect014(
    settings: Settings,
    regional: RegionalLiveFixture,
    reset_host: HostProbeFixture,
    collector: CollectorAcceptanceFixture,
    case_dir: Path,
    attempt: int,
    profile_version: str,
    *,
    cleanup: CaseCleanup | None = None,
) -> dict[str, Any]:
    cleanup = cleanup or CaseCleanup()
    baseline = reset_host.execute("snapshot")
    audit_before = collector.execute("reset-audit")
    write_json_atomic(case_dir / "reset-audit-before.json", audit_before)
    write_json_atomic(case_dir / "host-baseline.json", baseline)
    gpu = baseline["gpu_inventory"][0]
    bdf = str(gpu["pci_bdf"])
    fail_marker = f"c014-fail-{int(time.time())}-a{attempt}"
    # The fail-closed direction is "SXID 10003 with no inventory evidence".
    # Ingest accepts an inventory sample as evidence only while
    # -30s <= event.observed_at - sample.observed_at <= 180s (snapshot) or
    # 600s (legacy metrics), so the event is back-dated far enough that the
    # newest stored sample is still *newer* than event + 30s. Eight minutes
    # tolerates a 7.5 min delivery lag and stays under the 900s
    # STALE_FAULT_GENERATION fence, which would block for the wrong reason.
    # And because the premise is a bet on inventory freshness, the store is
    # read first and the post refused when the bet would not hold.
    posted_at = datetime.now(timezone.utc)
    old_time = posted_at - FAIL_CLOSED_EVENT_AGE
    inventory = regional.cpu_python(
        GPU_INVENTORY_SNAPSHOT,
        settings.regional.cluster_id,
        settings.node,
    )
    errors = fail_closed_timing_errors(inventory, event_time=old_time)
    if errors:
        return {
            "verdict": "FAIL",
            "errors": errors,
            "fail_closed_marker": None,
            "positive_marker": None,
            "inventory_snapshot": inventory,
            "fail_closed": None,
            "positive": None,
            "restore_workflows": [],
        }
    fail_payload = {
        "cluster_id": settings.regional.cluster_id,
        "node_id": settings.node,
        "record_id": fail_marker,
        "observed_at": old_time.isoformat(),
        "message": (
            f"nvidia-nvswitch0: SXid (PCI:{bdf}): 10003, Fatal, "
            f"Link 3 NVSWITCH_NON_CORRECTABLE marker={fail_marker}"
        ),
        "source": f"api-replay://{settings.case_id}/{fail_marker}",
        "runtime_profile_version": profile_version,
    }
    cleanup.register_seed(collector, fail_marker, quiesce_host=reset_host)
    post_fabric_event(regional, fail_payload)
    # The workflow is created now, not at the back-dated observed_at, so the
    # store scan is scoped to the moment of the post.
    fail_state = collector.wait_marker(
        fail_marker,
        case_dir=case_dir / "fail-closed",
        timeout_seconds=300,
        terminal_workflow=True,
        observed_after=posted_at,
    )
    cleanup.register_state(collector, fail_state)
    after_fail = reset_host.execute("snapshot")
    write_json_atomic(case_dir / "host-after-negative.json", after_fail)
    fail_workflow = (
        select_workflow(
            fail_state.get("workflows") or [],
            official_actions=FULL_FABRIC_RESET_ACTIONS,
        )
        or (fail_state.get("workflows") or [{}])[0]
    )
    if fail_workflow.get("status") != "BLOCKED":
        errors.append("incomplete inventory SXID did not fail closed")
    else:
        reasons = list(fail_workflow.get("blocked_reasons") or [])
        if len(reasons) != 1 or FAIL_CLOSED_REASON not in reasons[0]:
            errors.append("fail-closed reasons are not the dual-evidence gate")
    # The doc requires the fail-closed direction to leave the node untouched:
    # a BLOCKED workflow quarantines through the control plane only.
    if len(after_fail["ledger"]) != len(baseline["ledger"]):
        errors.append("fail-closed SXID produced node-side actions")
    if after_fail["boot_id"] != baseline["boot_id"]:
        errors.append("node rebooted during the fail-closed SXID")
    # The fail-closed SXID quarantines the node (its BLOCKED workflow still
    # owns the isolation). The positive injection that follows must run
    # MARK_UNSCHEDULABLE itself; while the fail-closed incident holds the
    # node that step fails with "node is already isolated by another
    # incident/token" and the positive workflow is FAILED, not SUCCEEDED
    # (same ownership fence as COLLECT-012, observed 2026-09-06 04:00Z). So
    # the node is returned through the validated path between the two SXIDs.
    restores = [
        cleanup.restore(
            collector,
            fail_state,
            profile_version=profile_version,
            reason="COLLECT-014 restore before the positive SXID",
        )
    ]
    if errors:
        # A negative direction that reached an action cannot authorize either
        # positive variant, even if the node subsequently looks healthy.
        return {
            "verdict": "FAIL",
            "errors": errors,
            "fail_closed_marker": fail_marker,
            "positive_marker": None,
            "inventory_snapshot": inventory,
            "fail_closed": fail_state,
            "positive": None,
            "restore_workflows": restores,
        }
    positives = []
    stability = []
    for index, (sxid, classification) in enumerate(FULL_RESET_VARIANTS):
        if index:
            stability = wait_full_reset_stability(
                regional, settings, reset_host, baseline
            )
            baseline = stability[-1]["host"]
            audit_before = collector.execute("reset-audit")
        variant = run_full_reset_variant(
            settings,
            reset_host,
            collector,
            case_dir / f"positive-{sxid}",
            attempt,
            profile_version,
            baseline=baseline,
            audit_before=audit_before,
            sxid=sxid,
            classification=classification,
            cleanup=cleanup,
        )
        positives.append(variant)
        errors.extend(variant["errors"])
        restores.append(variant["restore_workflows"])
        if variant["verdict"] != "PASS":
            errors.append(f"SXID{sxid} failed; later reset variants not run")
            break
    return {
        "verdict": "PASS" if not errors else "FAIL",
        "errors": errors,
        "fail_closed_marker": fail_marker,
        "positive_marker": positives[0]["marker"],
        "inventory_snapshot": inventory,
        "fail_closed": fail_state,
        "positive": positives[0]["state"],
        "positive_variants": positives,
        "required_positive_sxids": [item[0] for item in FULL_RESET_VARIANTS],
        "between_variant_stability": stability,
        "restore_workflows": restores,
        "validation_scope": "live-software-log-and-physical-reset",
        "physical_fault_injected": False,
    }


def collect015_workflow_errors(
    workflow: dict[str, Any] | None,
    *,
    submission: dict[str, Any] | None,
    node_after: dict[str, Any],
    node_recovery: Any,
    allow_replace: str | None,
) -> list[str]:
    """COLLECT-015's control-plane readings around the ALWAYS_FATAL reboot."""

    errors = []
    if workflow is None:
        errors.append("no workflow planned RESTART_NODE for the ALWAYS_FATAL SXID")
        return errors
    if workflow.get("status") != "SUCCEEDED":
        errors.append("ALWAYS_FATAL reboot workflow is not SUCCEEDED")
    restart = [
        item
        for item in workflow.get("step_executions") or []
        if item.get("operation") == "RESTART_NODE"
    ]
    if not restart or restart[-1].get("status") != "SUCCEEDED":
        errors.append("RESTART_NODE step did not reach SUCCEEDED")
    if not submission:
        errors.append("no HyperPod submission record for the RESTART_NODE step")
    elif submission.get("state") != "SUBMITTED":
        errors.append(
            f"HyperPod submission record is {submission.get('state')!r}, not SUBMITTED"
        )
    if node_recovery not in (None, "None"):
        errors.append(
            f"HyperPod cluster NodeRecovery is {node_recovery!r}; the reboot must "
            "not compete with automatic recovery"
        )
    if str(allow_replace or "").strip().lower() not in FALSE_ENV_VALUES:
        errors.append(
            f"executor GPU_FAULT_ALLOW_HYPERPOD_REPLACE={allow_replace!r} is not false"
        )
    if node_after.get("unschedulable"):
        errors.append("node is still unschedulable after the reboot workflow")
    return errors


def run_collect015(
    settings: Settings,
    regional: RegionalLiveFixture,
    collector: CollectorAcceptanceFixture,
    case_dir: Path,
    attempt: int,
    *,
    cleanup: CaseCleanup,
) -> dict[str, Any]:
    baseline_node = regional.node_snapshot(settings.node)
    provider = WarmSpareLiveFixture(regional, settings.hyperpod_cluster)
    baseline_provider = provider.provider_inventory()
    snapshot = collector.snapshot()
    bdf = str(snapshot["gpu_inventory"][0]["pci_bdf"])
    marker = f"c015-{int(time.time())}-a{attempt}"
    started_at = datetime.now(timezone.utc)
    cleanup.register_seed(collector, marker)
    require_action_time(180)
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
    # FABRIC_MANAGER_LOG events do not carry the marker into the workflow, so
    # match by injection time as COLLECT-011/014 do (a bare marker match
    # reported "collector marker did not converge" on 2026-09-06 04:00Z).
    state = collector.wait_marker(
        marker,
        case_dir=case_dir,
        timeout_seconds=2400,
        terminal_workflow=True,
        observed_after=started_at,
    )
    node_after = regional.wait_node_ready(
        settings.node,
        timeout_seconds=1800,
        expected_boot_id=str(baseline_node["boot_id"]),
    )
    errors = []
    workflow = select_workflow(state.get("workflows") or [], operation="RESTART_NODE")
    submission_state: dict[str, Any] = {}
    if workflow is not None:
        submission_state = regional.cpu_python(
            HYPERPOD_SUBMISSION,
            str(workflow.get("request_id") or ""),
            settings.hyperpod_cluster,
        )
    recovery = provider.cluster_recovery()
    executor_flags = regional.executor_python(EXECUTOR_REPLACE_FLAG)
    errors.extend(
        collect015_workflow_errors(
            workflow,
            submission=submission_state.get("submission"),
            node_after=regional.node_snapshot(settings.node),
            node_recovery=recovery.get("node_recovery"),
            allow_replace=executor_flags.get("GPU_FAULT_ALLOW_HYPERPOD_REPLACE"),
        )
    )
    if node_after["boot_id"] == baseline_node["boot_id"]:
        errors.append("node boot ID did not change")
    provider_after = provider.provider_inventory()
    if provider_after != baseline_provider:
        errors.append("provider node identity changed during reboot")
    reboot = regional.wait_provider_events(
        started_at,
        event_names=set(REBOOT_EVENTS),
        expected_count=1,
    )
    ended_at = datetime.now(timezone.utc)
    events = regional.provider_events(started_at, ended_at)
    replace = [item for item in events if item["event_name"] in REPLACE_EVENTS]
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
        "submission": submission_state.get("submission"),
        "cluster_recovery": recovery,
        "executor_flags": executor_flags,
        "provider_events": events,
        "reboot_events": reboot,
        # "no replacement" is a negative claim inside CloudTrail's delivery
        # window; DESTR-013 re-checks the whole run once it has caught up.
        "provider_events_provisional": regional.provider_events_provisional(ended_at),
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


def plan_details(settings: Settings, preflight: dict[str, Any]) -> dict[str, Any]:
    details = {
        "risk": "destructive",
        "case_id": settings.case_id,
        "predecessor": preflight["predecessor"],
        "target_node": settings.node,
        "mutation": {
            "GF-REGIONAL-COLLECT-004": (
                "publish two completed isolated Collector batches and verify one real reboot; "
                "production collector configuration and service are not modified"
            ),
            "GF-REGIONAL-COLLECT-008": "real kmsg XID63+48 and one GPU reset",
            "GF-REGIONAL-COLLECT-013": (
                f"real kmsg XID{settings.xid} single-GPU reset (one cycle)"
            ),
            "GF-REGIONAL-COLLECT-014": (
                "fail-closed API SXID10003, then FM SXID10003/Fatal and "
                "SXID19084/Non-fatal full-fabric reset cycles; restore and "
                "prove stable inventory before the second cycle"
            ),
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
    if settings.case_id == "GF-REGIONAL-COLLECT-004":
        details["verification_layers"] = [
            "isolated real Collector and private sink",
            "exact captured batches accepted by the live collector endpoint",
            "one product reboot workflow and complete cleanup",
        ]
        details["production_collector_mutation"] = False
        details["stop_conditions"][2] = (
            "isolated sampling, source identity or exact delivery cannot be proven"
        )
    if settings.case_id == "GF-REGIONAL-COLLECT-014":
        details["sxid_proof_obligations"] = {
            "live_log_positive_variants": [
                {"sxid": sxid, "classification": classification}
                for sxid, classification in FULL_RESET_VARIANTS
            ],
            "physical_reset_cycles": len(FULL_RESET_VARIANTS),
            "minimum_cooldown_seconds": FULL_RESET_COOLDOWN_SECONDS,
            "between_variant_stable_samples": FULL_RESET_STABLE_SAMPLES,
            "physical_fault_injection": False,
            "component_tests_are_physical_evidence": False,
        }
    record_focused_tests(details, preflight["focused_tests"])
    return details


@bounded_collector_case
def execute_case(
    settings: Settings,
    run_dir: Path,
    attempt: int,
    maintenance_window_end: datetime,
) -> int:
    case_dir = run_dir / "cases" / settings.case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    preflight = read_only_preflight(
        settings,
        case_dir,
        reuse_plan=case_dir / "plan.json",
    )
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
        case_dir=case_dir,
    )
    result: dict[str, Any] = {
        "case_id": settings.case_id,
        "attempt": attempt,
        "verdict": "FAIL",
        **regional.evidence_identity(),
    }
    collector: CollectorAcceptanceFixture | None = None
    cleanup = CaseCleanup()
    profile_version = str(
        (preflight["store"].get("profile") or {}).get("profile_version") or ""
    )
    try:
        if settings.case_id in CASE_IDS:
            collector = CollectorAcceptanceFixture(
                regional,
                node=settings.node,
                image=settings.host_probe_image,
                case_id=settings.case_id,
                run_id=f"{settings.case_id.lower()}-collector-{attempt}",
                case_dir=case_dir,
            )
            collector.create()
        if settings.case_id in {
            "GF-REGIONAL-COLLECT-008",
            "GF-REGIONAL-COLLECT-013",
            "GF-REGIONAL-COLLECT-014",
        }:
            require_action_time(180)
            host.create()
        if collector is None:
            raise RegionalFixtureError("collector case has no owned probe")
        if settings.case_id == "GF-REGIONAL-COLLECT-008":
            result.update(
                run_collect008(
                    settings,
                    regional,
                    host,
                    case_dir,
                    attempt,
                    collector=collector,
                    cleanup=cleanup,
                )
            )
        elif settings.case_id == "GF-REGIONAL-COLLECT-013":
            result.update(
                run_collect013(
                    settings,
                    regional,
                    host,
                    case_dir,
                    attempt,
                    collector=collector,
                    cleanup=cleanup,
                )
            )
        elif settings.case_id == "GF-REGIONAL-COLLECT-004" and collector is not None:
            result.update(
                run_collect004(
                    settings,
                    regional,
                    collector,
                    case_dir,
                    attempt,
                    cleanup=cleanup,
                    expected_scope=preflight["reboot_scope"],
                )
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
                    profile_version,
                    cleanup=cleanup,
                )
            )
        elif settings.case_id == "GF-REGIONAL-COLLECT-015" and collector is not None:
            result.update(
                run_collect015(
                    settings, regional, collector, case_dir, attempt, cleanup=cleanup
                )
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
        result["verdict"] = "FAIL"
    except BaseException as exc:
        result["error"] = f"case interrupted: {type(exc).__name__}"
        result["verdict"] = "FAIL"
        raise
    finally:
        released = cleanup.finish(
            profile_version=profile_version,
            reason=f"{settings.case_id} cleanup after case end",
        )
        if released["errors"]:
            result.setdefault("cleanup_errors", []).extend(released["errors"])
            result["verdict"] = "FAIL"
        if released["restore_workflows"]:
            result["cleanup_restore_workflows"] = released["restore_workflows"]
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
    value.add_argument(
        "--xid",
        type=int,
        choices=COLLECT013_XIDS,
        default=DEFAULT_COLLECT013_XID,
        help="COLLECT-013: which single-GPU reset XID to inject (one cycle)",
    )
    value.add_argument(
        "--debounce-tolerance",
        type=float,
        default=0.5,
        help=(
            "COLLECT-004: allowed extra fraction of the first-mismatch to threshold "
            "collector interval"
        ),
    )
    return value


CASE = CaseSurface(
    parser=parser,
    configure=configure,
    read_only_preflight=read_only_preflight,
    plan_details=plan_details,
    execute_case=execute_case,
)


def main() -> int:
    return run_selected_case(CASE)


if __name__ == "__main__":
    raise SystemExit(run_case_main(main))
