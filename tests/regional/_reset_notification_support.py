"""The GPU reset completion record a reset case's store snapshot carries.

The store probe lists an incident's notifications as ``{"notification",
"result"}`` pairs; the reset runners require exactly one GPU_RESET_COMPLETED
among them, keyed by the RESET_GPU step's own idempotency key, SKIPPED by the
drill policy for a drill incident and SENT for a real one. The fixtures that
drive those runners build that entry here.
"""

from __future__ import annotations

from typing import Any

_UNSET = object()


def drill_policy_reason(drill_id: str) -> str:
    """The reason ``AdvisoryNotificationService.suppresses`` records for a drill."""

    return (
        f"drill {drill_id}: not delivered because it describes a fault that did "
        "not happen (set GPU_FAULT_NOTIFICATION_DELIVER_DRILLS=true to mail drills)"
    )


def reset_notification_entry(
    *,
    cluster_id: str,
    incident_id: str,
    operation_id: str,
    drill_id: str | None = None,
    status: Any = _UNSET,
    provider_message_id: Any = _UNSET,
    reason: Any = _UNSET,
) -> dict[str, Any]:
    """A SENT record, or -- given ``drill_id`` -- one SKIPPED by the drill policy.

    Explicit ``status``/``provider_message_id``/``reason`` override the branch's
    defaults so a test can describe the defect it wants judged.
    """

    if status is _UNSET:
        status = "SKIPPED" if drill_id else "SENT"
    if provider_message_id is _UNSET:
        provider_message_id = None if drill_id else "unit-message"
    if reason is _UNSET:
        reason = drill_policy_reason(drill_id) if drill_id else None
    notification_id = f"notification-{incident_id}-reset"
    subject = f"[通知][GPU 已自动重置] {cluster_id}: GPU-a"
    return {
        "notification": {
            "notification_id": notification_id,
            "deduplication_key": f"{cluster_id}/{incident_id}/gpu-reset/{operation_id}",
            "cluster_name": cluster_id,
            "incident_id": incident_id,
            "subject": f"[DRILL:{drill_id}] {subject}" if drill_id else subject,
            "body_text": "unit fixture",
            "support_case_draft": "",
            "evidence_refs": [],
            "drill_id": drill_id,
            "category": "ACTION_COMPLETED",
            "priority": 50,
            "not_before": None,
            "created_at": "2026-01-01T00:00:00+00:00",
        },
        "result": {
            "notification_id": notification_id,
            "status": status,
            "provider_message_id": provider_message_id,
            "reason": reason,
        },
    }


def skipped_drill_result(entry: dict[str, Any], drill_id: str) -> dict[str, Any]:
    """Turn a record into the drill-labelled one a site that keeps drills writes."""

    entry["notification"]["drill_id"] = drill_id
    entry["result"] = {
        "notification_id": entry["notification"]["notification_id"],
        "status": "SKIPPED",
        "provider_message_id": None,
        "reason": drill_policy_reason(drill_id),
    }
    return entry
