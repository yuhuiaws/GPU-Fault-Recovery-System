"""Pure verdict functions and constants of GF-REGIONAL-DESTR-021.

The case runs COLLECT-017 A's real EFA remediation workflow on one idle node
whose metadata is adversarial (ARCH-A2/A3/A8):

* a stale *foreign* device-plugin-restart annotation set is written before the
  injection, so ``RESTART_EFA_DEVICE_PLUGIN`` has to take it over and clear it
  instead of refusing forever (A3);
* a bounded local writer keeps patching a harmless annotation on the node, so
  ``MARK_UNSCHEDULABLE`` and ``RESTORE_SCHEDULING`` race real 409 conflicts and
  must still end SUCCEEDED through ``patch_node_with_retry`` (A2);
* both scheduling steps must record the observed ``before``/``after`` node
  baselines on their remote command (A8), and the plugin restart must have been
  handed a positive ``expected_count`` (A3 fail-closed input).

Every function here judges plain dicts: the remote-command bundle, node
snapshots and the writer's report. Nothing touches a cluster.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from scripts.e2e.regional.acceptance_runner_common import processor_queue_backlog

CASE_ID = "GF-REGIONAL-DESTR-021"
PREDECESSOR_CASE_ID = "GF-REGIONAL-DESTR-020"
CONFIRMATION = "DESTR021_ADVERSARIAL_NODE_METADATA"
RISK = "live-node-mutation"

EFA_RESOURCE = "vpc.amazonaws.com/efa"
QUARANTINE_TAINT = "gpu-fault.io/quarantined"
TICK_ANNOTATION = "gpu-fault-acceptance.io/destr021-tick"
PLUGIN_RESTART_ANNOTATIONS = (
    "gpu-fault.io/efa-plugin-restart-operation",
    "gpu-fault.io/efa-plugin-restart-incident",
    "gpu-fault.io/efa-plugin-restart-pod-uid",
    "gpu-fault.io/efa-plugin-restart-started-at",
)
ANNOTATION_OPERATION, ANNOTATION_INCIDENT, ANNOTATION_POD_UID, ANNOTATION_STARTED = (
    PLUGIN_RESTART_ANNOTATIONS
)
# The deployed adapter's default ``restart_timeout_seconds``; the pre-seeded
# annotation is dated this far plus a margin in the past so it is provably stale.
RESTART_TIMEOUT_SECONDS = 180
STALE_MARGIN_SECONDS = 600
EXPECTED_STEPS = (
    "FREEZE_EVIDENCE",
    "MARK_UNSCHEDULABLE",
    "REMEDIATE_EFA_DRIVER",
    "RESTART_EFA_DEVICE_PLUGIN",
    "TRIGGER_HEALTH_SNAPSHOT",
    "VALIDATE_FABRIC",
    "RESTORE_SCHEDULING",
)
FORBIDDEN_OPERATIONS = frozenset(
    {
        "QUARANTINE",
        "QUIESCE_GPU_SERVICES",
        "RESTORE_GPU_SERVICES",
        "RESET_GPU",
        "RESET_ALL_GPUS_NVSWITCHES",
        "RESTART_NODE",
        "REPLACE_NODE",
    }
)
BASELINE_KEYS = ("unschedulable", "taint_keys", "resource_version")
WRITER_INTERVAL_SECONDS = 0.5
WRITER_MAX_SECONDS = 900
WRITER_MIN_PATCHES = 60
WORKFLOW_TIMEOUT_SECONDS = 600


def parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def stale_started_at(now: datetime) -> str:
    """The ``started-at`` value that makes a foreign restart provably stale."""

    seconds = RESTART_TIMEOUT_SECONDS + STALE_MARGIN_SECONDS
    return (now - timedelta(seconds=seconds)).isoformat()


def preseed_annotations(
    *, foreign_operation: str, foreign_incident: str, now: datetime
) -> dict[str, str]:
    return {
        ANNOTATION_OPERATION: foreign_operation,
        ANNOTATION_INCIDENT: foreign_incident,
        ANNOTATION_POD_UID: "",
        ANNOTATION_STARTED: stale_started_at(now),
    }


def preseed_errors(annotations: dict[str, Any], *, foreign_incident: str) -> list[str]:
    """The pre-seed landed as written and is old enough to be taken over."""

    errors: list[str] = []
    if annotations.get(ANNOTATION_INCIDENT) != foreign_incident:
        errors.append("the foreign incident annotation did not land on the node")
    if not annotations.get(ANNOTATION_OPERATION):
        errors.append("the foreign operation annotation did not land on the node")
    started = parse_time(annotations.get(ANNOTATION_STARTED))
    if started is None:
        errors.append(
            "the pre-seeded started-at is not parsable; it would never be stale"
        )
    elif (
        datetime.now(timezone.utc) - started
    ).total_seconds() < RESTART_TIMEOUT_SECONDS:
        errors.append("the pre-seeded started-at is not older than the restart timeout")
    return errors


# --------------------------------------------------------------------------- #
# Preflight
# --------------------------------------------------------------------------- #
def preflight_errors(
    *,
    node: dict[str, Any],
    node_annotations: dict[str, Any],
    workloads: list[dict[str, str]],
    queue: dict[str, Any],
    remote_commands: dict[str, Any],
    efa_allocatable: int,
    bound_efa_functions: int,
    predecessor_valid: bool,
    tests_passed: bool,
    site_file_exists: bool,
) -> list[str]:
    errors: list[str] = []
    if node.get("ready") != "True":
        errors.append("target node is not Ready")
    if node.get("unschedulable"):
        errors.append("target node is already unschedulable")
    if node.get("taints"):
        errors.append("target node has pre-existing taints")
    if node.get("ownership_annotations"):
        errors.append("target node has pre-existing workflow ownership")
    present = sorted(
        key for key in PLUGIN_RESTART_ANNOTATIONS if node_annotations.get(key)
    )
    if present:
        errors.append(
            f"target node already carries plugin-restart annotations {present}"
        )
    if node_annotations.get(TICK_ANNOTATION) is not None:
        errors.append("target node carries a tick annotation from an earlier run")
    if workloads:
        errors.append("target node has non-system running Pods")
    if efa_allocatable < 1:
        errors.append("target node reports no allocatable EFA devices")
    if bound_efa_functions < 1:
        errors.append("target node has no bound EFA function to unbind")
    if processor_queue_backlog(queue):
        errors.append("processor queue is not empty")
    if remote_commands.get("open_by_cluster"):
        errors.append("remote command queue is not empty")
    if not predecessor_valid:
        errors.append(f"{PREDECESSOR_CASE_ID} predecessor evidence is not PASS")
    if not tests_passed:
        errors.append("focused regression tests failed")
    if not site_file_exists:
        errors.append("regional site file does not exist")
    return errors


# --------------------------------------------------------------------------- #
# Workflow shape
# --------------------------------------------------------------------------- #
def workflow_errors(
    bundle: dict[str, Any], *, steps: tuple[str, ...] = EXPECTED_STEPS
) -> list[str]:
    errors: list[str] = []
    workflow = bundle.get("workflow") or {}
    incident = bundle.get("incident") or {}
    operations = [
        str(item.get("operation")) for item in workflow.get("official_steps") or []
    ]
    if operations != list(steps):
        errors.append(f"official steps {operations} != {list(steps)}")
    if workflow.get("status") != "SUCCEEDED":
        errors.append(f"workflow is {workflow.get('status')!r}, not SUCCEEDED")
    statuses = {
        str(item.get("operation")): item.get("status")
        for item in workflow.get("step_executions") or []
    }
    for operation in steps:
        if statuses.get(operation) != "SUCCEEDED":
            errors.append(f"{operation} is {statuses.get(operation)!r}, not SUCCEEDED")
    reached = set(statuses)
    forbidden = sorted(reached & FORBIDDEN_OPERATIONS)
    if forbidden:
        errors.append(f"workflow reached forbidden operations: {forbidden}")
    if incident.get("state") != "RECOVERED":
        errors.append(f"incident state is {incident.get('state')!r}, not RECOVERED")
    return errors


def _commands(bundle: dict[str, Any], operation: str) -> list[dict[str, Any]]:
    return [
        item
        for item in bundle.get("remote_commands") or []
        if item.get("operation") == operation
    ]


def _node_result(bundle: dict[str, Any], operation: str, node: str) -> dict[str, Any]:
    commands = _commands(bundle, operation)
    if not commands:
        return {}
    details = commands[-1].get("result_details") or {}
    result = (details.get("node_results") or {}).get(node)
    return dict(result) if isinstance(result, dict) else {}


# --------------------------------------------------------------------------- #
# A3: stale foreign restart takeover and expected_count
# --------------------------------------------------------------------------- #
def takeover_errors(
    bundle: dict[str, Any],
    *,
    node: str,
    foreign_incident: str,
    final_annotations: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    commands = _commands(bundle, "RESTART_EFA_DEVICE_PLUGIN")
    if not commands:
        errors.append("no RESTART_EFA_DEVICE_PLUGIN remote command was recorded")
        return errors
    if commands[-1].get("status") != "SUCCEEDED":
        errors.append(
            "RESTART_EFA_DEVICE_PLUGIN command is "
            f"{commands[-1].get('status')!r}, not SUCCEEDED"
        )
    result = _node_result(bundle, "RESTART_EFA_DEVICE_PLUGIN", node)
    if not result:
        errors.append("RESTART_EFA_DEVICE_PLUGIN carries no node_results for the node")
    elif result.get("took_over_incident") != foreign_incident:
        errors.append(
            "the plugin restart did not take over the stale foreign annotation: "
            f"took_over_incident={result.get('took_over_incident')!r}"
        )
    left = sorted(
        key for key in PLUGIN_RESTART_ANNOTATIONS if final_annotations.get(key)
    )
    if left:
        errors.append(f"plugin-restart annotations were left on the node: {left}")
    return errors


def expected_count_errors(
    bundle: dict[str, Any], *, node: str, baseline_efa: int
) -> list[str]:
    errors: list[str] = []
    commands = _commands(bundle, "RESTART_EFA_DEVICE_PLUGIN")
    parameters: dict[str, Any] = {}
    if commands:
        step = commands[-1].get("step") or {}
        parameters = dict(step.get("parameters") or {})
    if not parameters:
        workflow = bundle.get("workflow") or {}
        for item in workflow.get("official_steps") or []:
            if item.get("operation") == "RESTART_EFA_DEVICE_PLUGIN":
                parameters = dict(item.get("parameters") or {})
    raw = parameters.get("expected_count")
    try:
        expected = int(raw) if raw is not None else 0
    except (TypeError, ValueError):
        expected = 0
    if expected < 1:
        errors.append(
            f"RESTART_EFA_DEVICE_PLUGIN was dispatched with expected_count={raw!r}; "
            "the adapter would fail closed"
        )
    elif expected != baseline_efa:
        errors.append(
            f"expected_count {expected} differs from the node's EFA allocatable "
            f"{baseline_efa}"
        )
    result = _node_result(bundle, "RESTART_EFA_DEVICE_PLUGIN", node)
    if result:
        if result.get("expected") != expected:
            errors.append(
                f"node_results.expected {result.get('expected')!r} != {expected}"
            )
        allocatable = result.get("allocatable")
        if not isinstance(allocatable, int) or allocatable < expected:
            errors.append(
                f"node_results.allocatable {allocatable!r} is below expected {expected}"
            )
    return errors


# --------------------------------------------------------------------------- #
# A8: observed before/after baselines
# --------------------------------------------------------------------------- #
def _baseline(bundle: dict[str, Any], operation: str, node: str) -> dict[str, Any]:
    commands = _commands(bundle, operation)
    if not commands:
        return {}
    details = commands[-1].get("result_details") or {}
    value = (details.get("node_baselines") or {}).get(node)
    return dict(value) if isinstance(value, dict) else {}


def _snapshot_errors(label: str, snapshot: Any) -> list[str]:
    if not isinstance(snapshot, dict):
        return [f"{label} baseline is missing"]
    missing = [key for key in BASELINE_KEYS if key not in snapshot]
    if missing:
        return [f"{label} baseline lacks {missing}"]
    return []


def baseline_errors(bundle: dict[str, Any], *, node: str) -> list[str]:
    errors: list[str] = []
    isolate = _baseline(bundle, "MARK_UNSCHEDULABLE", node)
    restore = _baseline(bundle, "RESTORE_SCHEDULING", node)
    if not isolate:
        errors.append("MARK_UNSCHEDULABLE recorded no node_baselines for the node")
    if not restore:
        errors.append("RESTORE_SCHEDULING recorded no node_baselines for the node")
    if errors:
        return errors
    for label, snapshot in (
        ("isolate before", isolate.get("before")),
        ("isolate after", isolate.get("after")),
        ("restore before", restore.get("before")),
        ("restore after", restore.get("after")),
    ):
        errors.extend(_snapshot_errors(label, snapshot))
    if errors:
        return errors
    if isolate["before"].get("unschedulable") is not False:
        errors.append("isolate before-snapshot does not show a schedulable node")
    if isolate["after"].get("unschedulable") is not True:
        errors.append("isolate after-snapshot does not show the node cordoned")
    if QUARANTINE_TAINT not in (isolate["after"].get("taint_keys") or []):
        errors.append("isolate after-snapshot lacks the quarantine taint")
    if isolate["before"].get("resource_version") == isolate["after"].get(
        "resource_version"
    ):
        errors.append("isolate after-snapshot was not re-read after the patch")
    if restore["before"].get("unschedulable") is not True:
        errors.append("restore before-snapshot does not show the node cordoned")
    if restore["after"].get("unschedulable") is not False:
        errors.append("restore after-snapshot does not show the node schedulable")
    if QUARANTINE_TAINT in (restore["after"].get("taint_keys") or []):
        errors.append("restore after-snapshot still carries the quarantine taint")
    if restore["before"].get("resource_version") == restore["after"].get(
        "resource_version"
    ):
        errors.append("restore after-snapshot was not re-read after the patch")
    return errors


# --------------------------------------------------------------------------- #
# A2: RESTORE_SCHEDULING under a 409 race
# --------------------------------------------------------------------------- #
def conflict_retries_observed(waiting_evidence: list[dict[str, Any]]) -> int:
    count = 0
    for item in waiting_evidence:
        details = item.get("details") or {}
        if details.get("patch_conflict_retry"):
            count += 1
    return count


def restore_errors(
    bundle: dict[str, Any], *, timeline_statuses: list[dict[str, Any]]
) -> list[str]:
    """RESTORE_SCHEDULING ended SUCCEEDED and was never FAILED on the way."""

    errors: list[str] = []
    workflow = bundle.get("workflow") or {}
    executions = [
        item
        for item in workflow.get("step_executions") or []
        if item.get("operation") == "RESTORE_SCHEDULING"
    ]
    if not executions:
        errors.append("RESTORE_SCHEDULING never executed")
        return errors
    if executions[-1].get("status") != "SUCCEEDED":
        errors.append(
            f"RESTORE_SCHEDULING ended {executions[-1].get('status')!r}, not SUCCEEDED"
        )
    for sample in timeline_statuses:
        if sample.get("operation") == "RESTORE_SCHEDULING" and sample.get("status") == (
            "FAILED"
        ):
            errors.append("RESTORE_SCHEDULING was observed FAILED during the race")
            break
    commands = _commands(bundle, "RESTORE_SCHEDULING")
    if commands and commands[-1].get("status") != "SUCCEEDED":
        errors.append(f"RESTORE_SCHEDULING command is {commands[-1].get('status')!r}")
    return errors


def writer_errors(
    report: dict[str, Any], *, min_patches: int = WRITER_MIN_PATCHES
) -> list[str]:
    """The concurrent writer really ran: enough patches, bounded, cleared."""

    errors: list[str] = []
    patches = int(report.get("patches") or 0)
    if patches < min_patches:
        errors.append(
            f"the annotation writer issued only {patches} patches (< {min_patches}); "
            "the race was not real"
        )
    if report.get("stopped") is not True:
        errors.append("the annotation writer did not stop")
    if report.get("cleared") is not True:
        errors.append("the tick annotation was not cleared")
    if report.get("exceeded_max_seconds"):
        errors.append(
            "the annotation writer hit its hard bound before the workflow ended"
        )
    return errors


# --------------------------------------------------------------------------- #
# Final node
# --------------------------------------------------------------------------- #
def node_final_errors(
    baseline: dict[str, Any],
    final: dict[str, Any],
    *,
    final_annotations: dict[str, Any],
    baseline_efa: int,
    final_efa: int,
) -> list[str]:
    errors: list[str] = []
    if final.get("ready") != "True":
        errors.append("target node is not Ready after the case")
    if final.get("unschedulable"):
        errors.append("target node was left cordoned")
    if any(item.get("key") == QUARANTINE_TAINT for item in final.get("taints") or []):
        errors.append("target node was left with the quarantine taint")
    if final.get("ownership_annotations"):
        errors.append(
            f"gpu-fault annotations remain: {sorted(final['ownership_annotations'])}"
        )
    if final_annotations.get(TICK_ANNOTATION) is not None:
        errors.append("the tick annotation remains on the node")
    if final.get("uid") != baseline.get("uid") or final.get("boot_id") != baseline.get(
        "boot_id"
    ):
        errors.append("node identity changed; this was not an in-place remediation")
    if final_efa != baseline_efa:
        errors.append(f"EFA allocatable {final_efa} != baseline {baseline_efa}")
    return errors
