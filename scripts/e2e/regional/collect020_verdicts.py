"""Pure verdict functions and constants of GF-REGIONAL-COLLECT-020.

The case proves ARCH-G6 (and the marker/notification halves of ARCH-I4 and
ARCH-E7) on a real regional deployment: with ``GPU_FAULT_EXPECTED_GPU_COUNT``
unset the inventory batch still carries the instance type's expected count;
a GPU UUID that disappears from one inventory query becomes a CRITICAL
``gpu_inventory_identity_changed`` finding naming the removed UUID, whose
workflow runs a DCGM diagnostic and validation -- never a reboot -- and mails
an operator-review notification plus the DCGM completion; once the incident
is restored its markers carry ``retired_at`` / ``retired_reason``.

The disappearance is a shadow of ``nvidia-smi`` output for exactly one
inventory query of one collector unit. No device is touched, and the shadow
is capped below the host collector's mismatch threshold so the REBOOT_NODE
path is unreachable. Every function here judges documents the runner wrote.
"""

from __future__ import annotations

from typing import Any

from gpu_fault.collectors.gpu.discovery import expected_accelerator_counts
from gpu_fault.models import Severity

CASE_ID = "GF-REGIONAL-COLLECT-020"
CONFIRMATION = "COLLECT020_EXECUTE"
PREDECESSOR_CASE_ID = "GF-REGIONAL-COLLECT-019"
UNIT = "gpu-fault-metrics-collector.service"
IDENTITY_METRIC = "gpu_inventory_identity_changed"
UNKNOWN_COUNT_METRIC = "gpu_expected_count_unknown"
MISMATCH_METRIC = "gpu_inventory_mismatch"
OPERATOR_REVIEW_CATEGORY = "OPERATOR_REVIEW"
DIAGNOSTIC_OPERATIONS = frozenset({"RUN_DCGM_DIAGNOSTIC", "VALIDATE_GPU"})
FORBIDDEN_OPERATIONS = frozenset(
    {"RESTART_NODE", "REPLACE_NODE", "RESET_GPU", "RESET_ALL_GPUS_NVSWITCHES"}
)
DROP_CALLS = 1
WINDOW_RESTORE_SECONDS = 900
FINDING_TIMEOUT_SECONDS = 600
WORKFLOW_TIMEOUT_SECONDS = 1500


def expected_count_for(instance_type: str | None) -> int | None:
    counts = expected_accelerator_counts(instance_type)
    return counts["gpu"] if counts is not None else None


def preconditions_errors(
    *,
    instance_type: str | None,
    gpu_count: int,
    mismatch_samples: int,
) -> list[str]:
    """The node must be one the instance-type table knows and the drop safe."""

    errors: list[str] = []
    expected = expected_count_for(instance_type)
    if expected is None:
        errors.append(
            f"instance type {instance_type!r} is not in the expected-count table; "
            "the unset-env half of the case cannot be judged"
        )
    elif expected != gpu_count:
        errors.append(
            f"the node shows {gpu_count} GPUs but its type is built with {expected}"
        )
    if gpu_count < 2:
        errors.append("a one-GPU node cannot lose a UUID and keep an inventory")
    if DROP_CALLS >= mismatch_samples:
        errors.append(
            f"hiding a GPU for {DROP_CALLS} query reaches the host collector's "
            f"{mismatch_samples}-sample mismatch threshold; the reboot path opens"
        )
    return errors


def window_errors(opened: dict[str, Any], *, dropped_uuid: str) -> list[str]:
    errors: list[str] = []
    if opened.get("unit") != UNIT:
        errors.append(f"window opened on {opened.get('unit')!r}, not {UNIT!r}")
    if "GPU_FAULT_EXPECTED_GPU_COUNT" not in (opened.get("unset") or []):
        errors.append("the window did not unset GPU_FAULT_EXPECTED_GPU_COUNT")
    shadow = opened.get("shadow") or []
    if list(shadow[:2]) != ["drop-uuid", dropped_uuid]:
        errors.append(f"window shadow is {shadow!r}, expected drop-uuid {dropped_uuid}")
    if shadow and int(shadow[2]) > DROP_CALLS:
        errors.append(f"the shadow hides the GPU for {shadow[2]} queries")
    if (opened.get("after") or {}).get("ActiveState") != "active":
        errors.append("the metrics collector is not active after the window opened")
    return errors


def inventory_evidence_errors(
    evidence: list[dict[str, Any]],
    *,
    dropped_uuid: str,
    expected_count: int,
    expected_uuids: set[str] | None = None,
) -> list[str]:
    """The shortened snapshot was kept, and it knows the expected count."""

    shortened = [
        item
        for item in evidence
        if dropped_uuid
        not in {
            str(device.get("gpu_uuid"))
            for device in (item.get("payload") or {}).get("devices") or []
        }
    ]
    if not shortened:
        return ["no GPU_INVENTORY evidence without the dropped UUID was recorded"]
    errors: list[str] = []
    for item in shortened:
        payload = item.get("payload") or {}
        devices = payload.get("devices") or []
        uuids = {str(device.get("gpu_uuid") or "") for device in devices}
        if (
            len(devices) != expected_count - 1
            or len(uuids) != len(devices)
            or "" in uuids
        ):
            errors.append(
                f"snapshot {item.get('record_id')} is not exactly one missing GPU"
            )
        if expected_uuids is not None and uuids != expected_uuids - {dropped_uuid}:
            errors.append(
                f"snapshot {item.get('record_id')} does not match the baseline UUID set"
            )
        if payload.get("expected_gpu_count") != expected_count:
            errors.append(
                f"snapshot {item.get('record_id')} carries expected_gpu_count="
                f"{payload.get('expected_gpu_count')!r}, not the instance type's "
                f"{expected_count}"
            )
    return errors


def _metric_record(record: dict[str, Any], metric: str) -> bool:
    if record.get("metric_name") == metric or record.get("event_type") == metric:
        return True
    for field in ("event_id", "marker_id"):
        value = str(record.get(field) or "")
        if value.endswith(f"-{metric}") or f"-{metric}-" in value:
            return True
    reasons = [*(record.get("reasons") or []), record.get("raw_reason") or ""]
    return any(metric in str(reason) for reason in reasons)


def _identity_incidents(activity: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item
        for item in activity.get("incidents") or []
        if _metric_record(item, IDENTITY_METRIC)
        or any(
            "GPU inventory identity changed" in str(reason)
            for reason in item.get("reasons") or []
        )
    ]


def identity_incident_errors(
    activity: dict[str, Any], *, dropped_uuid: str
) -> list[str]:
    """The finding-phase check: the incident exists and names the UUID.

    The workflow checks belong to the terminal phase; asking for them while
    polling for the finding kept the first live runs waiting the whole finding
    budget on an incident that had already opened.
    """

    incidents = _identity_incidents(activity)
    if not incidents:
        return [f"no incident names {IDENTITY_METRIC}"]
    if not any(dropped_uuid in str(item.get("reasons")) for item in incidents):
        return [f"no identity-changed incident names the removed UUID {dropped_uuid}"]
    return []


def identity_finding_errors(
    activity: dict[str, Any], *, dropped_uuid: str
) -> list[str]:
    """The CRITICAL finding names the UUID; its workflow diagnoses, never reboots."""

    errors: list[str] = []
    incidents = _identity_incidents(activity)
    if not incidents:
        return [f"no incident names {IDENTITY_METRIC}"]
    if not any(dropped_uuid in str(item.get("reasons")) for item in incidents):
        errors.append(
            f"no identity-changed incident names the removed UUID {dropped_uuid}"
        )
    incident_ids = {str(item.get("incident_id")) for item in incidents}
    markers = [
        item
        for item in activity.get("markers") or []
        if str(item.get("incident_id")) in incident_ids
        and _metric_record(item, IDENTITY_METRIC)
        and dropped_uuid in ((item.get("scope") or {}).get("gpu_uuids") or [])
    ]
    if not markers or any(
        item.get("severity") != Severity.CRITICAL.value for item in markers
    ):
        errors.append(
            "the removed UUID has no matching CRITICAL identity-finding marker"
        )
    workflows = [
        item
        for item in activity.get("workflows") or []
        if str(item.get("incident_id")) in incident_ids
    ]
    if not workflows:
        errors.append("the identity-changed incident opened no workflow")
    for workflow in workflows:
        operations = {
            str(step.get("operation")) for step in workflow.get("official_steps") or []
        }
        if not DIAGNOSTIC_OPERATIONS <= operations:
            errors.append(
                f"workflow {workflow.get('request_id')} lacks DCGM diagnostic or "
                f"GPU validation: {sorted(operations)}"
            )
        forbidden = sorted(operations & FORBIDDEN_OPERATIONS)
        if forbidden:
            errors.append(f"workflow {workflow.get('request_id')} compiles {forbidden}")
        if workflow.get("status") not in {"SUCCEEDED", "FAILED", "BLOCKED"}:
            errors.append(
                f"workflow {workflow.get('request_id')} is {workflow.get('status')}, "
                "not terminal"
            )
    review = [
        item
        for item in activity.get("notifications") or []
        if str(item.get("incident_id")) in incident_ids
        and item.get("category") == OPERATOR_REVIEW_CATEGORY
    ]
    if not review:
        errors.append(f"no {OPERATOR_REVIEW_CATEGORY} notification for the finding")
    return errors


def dcgm_notification_errors(activity: dict[str, Any]) -> list[str]:
    """ARCH-E7: the regional DCGM completion mails, whether it passed or failed."""

    incident_ids = {
        str(item.get("incident_id")) for item in _identity_incidents(activity)
    }
    matches = [
        item
        for item in activity.get("notifications") or []
        if str(item.get("incident_id")) in incident_ids
        and (
            "DCGM" in str(item.get("category", "")).upper()
            or "DCGM" in str(item.get("subject", "")).upper()
        )
    ]
    if not matches:
        return ["no DCGM diagnostic completion notification was created"]
    return []


def no_reboot_errors(
    activity: dict[str, Any], *, boot_id_before: str, boot_id_after: str
) -> list[str]:
    errors: list[str] = []
    if boot_id_before != boot_id_after:
        errors.append("the node rebooted during the case")
    records = [
        *(activity.get("incidents") or []),
        *(activity.get("markers") or []),
        *(activity.get("findings") or []),
    ]
    if any(_metric_record(item, MISMATCH_METRIC) for item in records):
        errors.append("the host collector raised gpu_inventory_mismatch")
    if any(_metric_record(item, UNKNOWN_COUNT_METRIC) for item in records):
        errors.append(
            "gpu_expected_count_unknown fired although the instance type is known"
        )
    for workflow in activity.get("workflows") or []:
        operations = {
            str(step.get("operation")) for step in workflow.get("official_steps") or []
        }
        if operations & FORBIDDEN_OPERATIONS:
            errors.append(
                f"workflow {workflow.get('request_id')} compiles a reboot/reset"
            )
    return errors


def marker_retirement_errors(activity: dict[str, Any]) -> list[str]:
    """ARCH-I4: every marker of a RECOVERED incident says when and why it retired."""

    recovered = {
        str(item.get("incident_id"))
        for item in activity.get("incidents") or []
        if item.get("state") == "RECOVERED"
    }
    if not recovered:
        return ["no incident of the case reached RECOVERED"]
    errors: list[str] = []
    markers = [
        item
        for item in activity.get("markers") or []
        if str(item.get("incident_id")) in recovered
    ]
    marked = {str(item.get("incident_id")) for item in markers}
    for incident_id in sorted(recovered - marked):
        errors.append(f"recovered incident {incident_id} has no retirement markers")
    for marker in markers:
        if not marker.get("retired_at"):
            errors.append(f"marker {marker.get('marker_id')} has no retired_at")
        if not marker.get("retired_reason"):
            errors.append(f"marker {marker.get('marker_id')} has no retired_reason")
    return errors


def restored_node_errors(node: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if node.get("ownership_annotations"):
        errors.append("the node still carries workflow ownership")
    if node.get("unschedulable"):
        errors.append("the node is still unschedulable")
    if node.get("taints"):
        errors.append(f"the node still carries taints: {node.get('taints')}")
    return errors


def case_verdict(stages: dict[str, list[str]]) -> str:
    return "PASS" if not any(stages.values()) else "FAIL"
