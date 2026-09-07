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
    select_workflow,
)
from scripts.e2e.regional.host_probe_fixture import (  # noqa: E402
    HostProbeError,
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
from scripts.e2e.regional.regional_live_fixture import (  # noqa: E402
    RegionalFixtureError,
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
}, sort_keys=True, default=str))
"""


HOST_INVENTORY_EVIDENCE = r"""
import json
import sys
from datetime import datetime

from gpu_fault.app import ApplicationContext

cluster_id, node_id, observed_after_text = sys.argv[1:]
observed_after = datetime.fromisoformat(observed_after_text.replace("Z", "+00:00"))
store = ApplicationContext.from_environment().store
records = []
for item in store.list_raw_evidence(cluster_id, node_id=node_id, limit=500):
    if item.observed_at < observed_after:
        continue
    if str(getattr(item.kind, "value", item.kind)) != "HOST_TELEMETRY":
        continue
    samples = [
        sample
        for sample in (item.payload.get("samples") or [])
        if str(sample.get("name")) == "gpu_inventory_mismatch"
    ]
    if not samples:
        continue
    records.append({
        "record_id": item.record_id,
        "observed_at": item.observed_at.isoformat(),
        "samples": samples,
    })
records.sort(key=lambda item: item["observed_at"])
print(json.dumps({"records": records}, sort_keys=True, default=str))
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
    # COLLECT-004: the finding's latency may exceed samples x interval by this
    # fraction (collector restart, ingestion) before it stops being "about
    # two collection periods".
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
        debounce_tolerance=float(getattr(arguments, "debounce_tolerance", 0.5)),
    )


FOCUSED_TESTS = {
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
    return regional.wait_for_workflow(
        node=node or settings.node,
        marker=marker,
        observed_after=injected_at,
        case_dir=case_dir,
        timeout_seconds=1800,
    )


def stop_sampler(host: HostProbeFixture, run_id: str, errors: list[str]) -> None:
    """Stop the detached nvidia-smi sampler; a failure is recorded, never raised.

    The sampler polls nvidia-smi four times a second for up to 30 minutes.
    Left running by an exception it outlives the case and sits on the GPUs the
    next reset has to take, so every path stops it -- and a stop that fails
    must not replace the exception that brought us here.
    """

    try:
        host.execute("stop-reset-sampler", "--run-id", run_id, timeout=60)
    except Exception as exc:
        errors.append(f"reset sampler stop failed: {type(exc).__name__}: {exc}")


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
    xid: int,
    marker: str,
    run_id: str,
    node: str | None = None,
    expected_steps: list[str] | None = None,
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
    errors: list[str] = []
    try:
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
        )
    )
    errors.extend(boot_id_errors(baseline, after))
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
    errors: list[str] = []
    try:
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
        )
    )
    errors.extend(boot_id_errors(baseline, after))
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


def restore_collector_env(collector: Any, run_id: str) -> dict[str, Any]:
    """Undo the collector env override, through a fresh probe Pod if needed.

    The first attempt goes through the Pod the case has been using; after a
    real RESTART_NODE that Pod is Failed and `kubectl exec` refuses it, which
    used to leave the override in place until the on-node deadman timer
    expired. One recreate-and-retry is what the reboot costs.
    """

    try:
        return cast(
            dict[str, Any],
            collector.execute(
                "restore-collector-env",
                "--run-id",
                run_id,
                timeout=300,
            ),
        )
    except HostProbeError:
        collector.recreate()
        return cast(
            dict[str, Any],
            collector.execute(
                "restore-collector-env",
                "--run-id",
                run_id,
                timeout=300,
            ),
        )


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


def mismatch_finding(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The first HOST_TELEMETRY record whose ``gpu_inventory_mismatch`` fired."""

    for record in records:
        for sample in record.get("samples") or []:
            if str(sample.get("name")) != "gpu_inventory_mismatch":
                continue
            if float(sample.get("value") or 0) >= 1:
                return {**record, "sample": sample}
    return None


def debounce_errors(
    records: list[dict[str, Any]],
    *,
    interval: int,
    required_samples: int,
    started_at: datetime,
    tolerance: float,
) -> list[str]:
    """COLLECT-004's actual claim: the finding fires on sample N, not sample 1.

    The host collector labels every inventory sample with the consecutive
    mismatch count it has seen and the count it requires; the persisted
    finding (``gpu_inventory_mismatch`` = 1) therefore says on which sample
    it fired. The latency from the override to that record must be about
    ``required_samples`` collection intervals.
    """

    errors = []
    finding = mismatch_finding(records)
    if finding is None:
        errors.append("no gpu_inventory_mismatch finding reached the control plane")
        return errors
    labels = finding["sample"].get("labels") or {}
    consecutive = int(labels.get("consecutive_mismatch_samples") or 0)
    required_label = int(labels.get("required_consecutive_samples") or 0)
    if required_samples < 2:
        errors.append(
            "GPU_FAULT_INVENTORY_MISMATCH_CONSECUTIVE_SAMPLES is below 2; the "
            "debounce this case exists to prove is switched off"
        )
    if consecutive != required_samples:
        errors.append(
            f"finding fired on consecutive sample {consecutive}, the node is "
            f"configured for {required_samples}"
        )
    if consecutive < 2:
        errors.append("finding fired on the first mismatching sample")
    if required_label and required_label != required_samples:
        errors.append(
            f"collector reports required_consecutive_samples={required_label}, "
            f"collector.env says {required_samples}"
        )
    latency = (
        datetime.fromisoformat(str(finding["observed_at"]).replace("Z", "+00:00"))
        - started_at
    ).total_seconds()
    expected = interval * required_samples
    lower = interval * (required_samples - 1)
    upper = expected * (1 + tolerance)
    if not lower <= latency <= upper:
        errors.append(
            f"finding latency {latency:.1f}s is not about {expected}s "
            f"({required_samples} x {interval}s; accepted {lower}..{upper:.0f}s)"
        )
    return errors


def wait_mismatch_finding(
    regional: RegionalLiveFixture,
    settings: Settings,
    *,
    observed_after: datetime,
    timeout_seconds: int,
) -> list[dict[str, Any]]:
    """Poll the node's HOST_TELEMETRY evidence until the mismatch finding lands."""

    deadline = time.monotonic() + timeout_seconds
    records: list[dict[str, Any]] = []
    while True:
        value = regional.cpu_python(
            HOST_INVENTORY_EVIDENCE,
            settings.regional.cluster_id,
            settings.node,
            observed_after.isoformat(),
        )
        records = list(value.get("records") or [])
        if mismatch_finding(records) is not None or time.monotonic() >= deadline:
            return records
        time.sleep(5)


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
    required_samples = collector_setting(
        env, "GPU_FAULT_INVENTORY_MISMATCH_CONSECUTIVE_SAMPLES"
    )
    baseline_env_sha256 = (baseline.get("collector_env_file") or {}).get("sha256")
    run_id = f"c004-{attempt}"
    started_at = datetime.now(timezone.utc)
    result: dict[str, Any] = {
        "errors": [],
        "collector_env": env,
        "interval_seconds": interval,
        "required_consecutive_samples": required_samples,
    }
    result["override"] = collector.execute(
        "override-expected-gpu-count",
        "--run-id",
        run_id,
        "--value",
        str(expected + 1),
        "--restore-seconds",
        "600",
    )
    restore: dict[str, Any] | None = None
    try:
        # (a) The debounce itself: which sample the finding fired on.
        records = wait_mismatch_finding(
            regional,
            settings,
            observed_after=started_at,
            timeout_seconds=interval * (required_samples + 4) + 60,
        )
        result["mismatch_records"] = records
        result["errors"].extend(
            debounce_errors(
                records,
                interval=interval,
                required_samples=required_samples,
                started_at=started_at,
                tolerance=settings.debounce_tolerance,
            )
        )
        planned = wait_planned_workflow(
            regional,
            settings,
            operation="RESTART_NODE",
            observed_after=started_at,
            timeout_seconds=600,
        )
        result["restart_workflow_id"] = planned.get("request_id")
        # (b) The finding is captured and the reboot is planned: the override
        # comes off the node *now*. Kept through the reboot it makes the
        # post-reboot VALIDATE_GPU see the wrong expected count, the workflow
        # fails and escalates, and the runner used to tolerate that.
        restore = restore_collector_env(collector, run_id)
        result["collector_restore"] = restore
        if restore.get("restored") is not True:
            result["errors"].append(
                f"collector env restore did not restore: {restore.get('reason')}"
            )
        node_after = regional.wait_node_ready(
            settings.node,
            timeout_seconds=1800,
            expected_boot_id=str(baseline["boot_id"]),
        )
        result["node_after"] = node_after
        # The reboot took the probe Pod with it; everything below that touches
        # the host needs a live one.
        collector.recreate()
        if node_after["boot_id"] == baseline["boot_id"]:
            result["errors"].append("inventory mismatch did not reboot the node")
        # (c) The reboot workflow itself, not merely "RESTART_NODE planned".
        workflow_state = latest_node_workflow(
            regional,
            settings,
            observed_after=started_at,
        )
        result["workflow_state"] = workflow_state
        matches = workflow_state.get("matches") or []
        if len(matches) != 1:
            result["errors"].append(
                f"node grew {len(matches)} workflows after the injection, expected "
                "exactly the inventory-mismatch reboot"
            )
        restart_workflow = workflow_planning(workflow_state, "RESTART_NODE")
        if restart_workflow is None:
            result["errors"].append("inventory mismatch did not plan RESTART_NODE")
        elif restart_workflow.get("status") != "SUCCEEDED":
            result["errors"].append(
                f"reboot workflow is {restart_workflow.get('status')!r}, not SUCCEEDED"
            )
        after = collector.snapshot()
        result["collector_env_after"] = after.get("collector_env_file")
        after_sha256 = (after.get("collector_env_file") or {}).get("sha256")
        if not baseline_env_sha256 or after_sha256 != baseline_env_sha256:
            result["errors"].append(
                "collector.env digest after restore differs from the baseline"
            )
        reboot = regional.wait_provider_events(
            started_at,
            event_names=set(REBOOT_EVENTS),
            expected_count=1,
        )
        result["provider_events"] = reboot
        if len(reboot) != 1:
            result["errors"].append("CloudTrail does not contain exactly one reboot")
        elif not provider_event_actor_matches_role(
            reboot[0],
            settings.executor_role_arn,
        ):
            result["errors"].append("reboot actor is not the executor role")
    finally:
        if restore is None or restore.get("restored") is not True:
            try:
                result["collector_restore"] = restore_collector_env(collector, run_id)
                if result["collector_restore"].get("restored") is not True:
                    result["errors"].append(
                        "collector env restore did not restore: "
                        f"{result['collector_restore'].get('reason')}"
                    )
            except Exception as exc:
                result["errors"].append(
                    f"collector env restore failed: {type(exc).__name__}: {exc}"
                )
    result["verdict"] = "PASS" if not result["errors"] else "FAIL"
    return result


def fail_closed_timing_errors(
    inventory: dict[str, Any],
    *,
    event_time: datetime,
) -> list[str]:
    """Refuse the fail-closed post unless no stored inventory can vouch for it.

    ``inventory`` is the node's newest ``GpuInventorySnapshot`` as the store
    holds it. Ingest treats a sample as fresh for the event while
    ``event.observed_at - sample.observed_at >= -30s``; the back-dated event is
    safe only when the sample is newer than that by a margin. A node with no
    snapshot at all cannot produce evidence, so it is safe by construction.
    On 2026-09-06 the sample lagged 2-3 minutes, the two-minute back-date
    looked fresh, and two real full-fabric resets ran under a "fail-closed"
    label.
    """

    if not inventory.get("present"):
        return []
    observed_at = datetime.fromisoformat(
        str(inventory.get("observed_at")).replace("Z", "+00:00")
    )
    margin = observed_at - event_time
    needed = INGEST_FUTURE_TOLERANCE + FAIL_CLOSED_SAFETY_MARGIN
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
) -> dict[str, Any]:
    baseline = reset_host.execute("snapshot")
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
        "source": "/var/log/fabricmanager.log",
        "runtime_profile_version": profile_version,
    }
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
    after_fail = reset_host.execute("snapshot")
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
        collector.restore_incidents(
            fail_state,
            profile_version=profile_version,
            reason="COLLECT-014 restore before the positive SXID",
        )
    ]
    if errors:
        # A fail-closed direction that reached the node has already spent
        # the one full fabric reset this case may perform. Stop here rather
        # than reset the machine a second time for a verdict that is FAIL.
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
    state: dict[str, Any] = {}
    try:
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
        # FABRIC_MANAGER_LOG events do not carry the marker into the
        # workflow, so match the positive workflow by injection time as
        # COLLECT-011 does.
        state = collector.wait_marker(
            marker,
            case_dir=case_dir / "positive",
            timeout_seconds=1800,
            terminal_workflow=True,
            observed_after=injected_at,
        )
        workflow = select_workflow(
            state.get("workflows") or [],
            operation="RESET_ALL_GPUS_NVSWITCHES",
        )
        if workflow is None:
            errors.append("no workflow planned RESET_ALL_GPUS_NVSWITCHES")
            workflow = {}
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
        # The probe returns the node agent's whole results ledger (only the
        # kernel journal honours --since-epoch), so earlier full resets on
        # this node -- COLLECT-014's own reruns included -- are still in it.
        # Count the rows this injection added, not every row ever written.
        known = {item.get("command_id") for item in baseline["ledger"]}
        rows = [
            item
            for item in after["ledger"]
            if item.get("operation") == "RESET_ALL_GPUS_NVSWITCHES"
            and item.get("command_id") not in known
        ]
        if len(rows) != 1:
            errors.append("full fabric reset ledger count is not one")
        if len(after["gpu_inventory"]) != len(baseline["gpu_inventory"]):
            errors.append("GPU inventory changed after full fabric reset")
        errors.extend(boot_id_errors(baseline, after))
    finally:
        stop_sampler(reset_host, run_id, errors)
    # A SUCCEEDED reset returns the node to service, but restore through the
    # validated path regardless so a partial run never leaves it quarantined.
    restores.append(
        collector.restore_incidents(
            state,
            profile_version=profile_version,
            reason="COLLECT-014 validated cleanup",
        )
    )
    return {
        "verdict": "PASS" if not errors else "FAIL",
        "errors": errors,
        "fail_closed_marker": fail_marker,
        "positive_marker": marker,
        "inventory_snapshot": inventory,
        "fail_closed": fail_state,
        "positive": state,
        "restore_workflows": restores,
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
) -> dict[str, Any]:
    baseline_node = regional.node_snapshot(settings.node)
    provider = WarmSpareLiveFixture(regional, settings.hyperpod_cluster)
    baseline_provider = provider.provider_inventory()
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
            if self.token.lower() not in encoded:
                continue
            # The HyperPod dependencies chart ships a sibling
            # `...-nvidia-device-plugin-mps-control-daemon` DaemonSet that
            # carries the plugin token but schedules nowhere (desired 0). Only
            # a DaemonSet that actually places Pods can be the one whose
            # exclusion changes the node's allocatable, so the idle sibling is
            # not a candidate (live 2026-09-06 09:52Z: "found 2").
            desired = int((item.get("status") or {}).get("desiredNumberScheduled") or 0)
            if desired <= 0:
                continue
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
    details = {
        "risk": "destructive",
        "case_id": settings.case_id,
        "predecessor": preflight["predecessor"],
        "target_node": settings.node,
        "mutation": {
            "GF-REGIONAL-COLLECT-004": "temporary expected-count mismatch and real reboot",
            "GF-REGIONAL-COLLECT-008": "real kmsg XID63+48 and one GPU reset",
            "GF-REGIONAL-COLLECT-013": (
                f"real kmsg XID{settings.xid} single-GPU reset (one cycle)"
            ),
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
    record_focused_tests(details, preflight["focused_tests"])
    return details


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
    )
    result: dict[str, Any] = {
        "case_id": settings.case_id,
        "attempt": attempt,
        "verdict": "FAIL",
        **regional.evidence_identity(),
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
        help="COLLECT-004: allowed fraction above samples x interval for the finding",
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
