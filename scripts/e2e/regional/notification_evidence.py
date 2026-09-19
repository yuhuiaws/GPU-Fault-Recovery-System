"""Validate notification acceptance evidence without trusting supplied verdicts."""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from scripts.e2e.regional.regional_live_fixture import predecessor_evidence

NOTIFICATION_KINDS = {
    "gpu-reset": "/gpu-reset/",
    "workload-restart": "/workload-restarted/",
}


class NotificationAcceptanceError(RuntimeError):
    pass


def parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def validate_external_evidence(
    path: Path | None,
    kind: str,
    *,
    record_times: Sequence[datetime] = (),
) -> dict[str, Any]:
    if path is None:
        return {"valid": False, "errors": [f"{kind} evidence is required"]}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"valid": False, "errors": [f"cannot read {kind} evidence: {exc}"]}
    if not isinstance(value, dict):
        return {"valid": False, "errors": ["external evidence is not an object"]}
    errors = []
    for field in ("method", "reference"):
        if not isinstance(value.get(field), str) or not value[field].strip():
            errors.append(f"{field} is required")
    window_start = parse_time(value.get("window_start"))
    window_end = parse_time(value.get("window_end"))
    if window_start is None or window_end is None:
        errors.append("window_start and window_end must be ISO-8601 timestamps")
    elif window_start > window_end:
        errors.append("window_start is after window_end")
    else:
        uncovered = [
            item.isoformat()
            for item in record_times
            if not (window_start <= item <= window_end)
        ]
        if uncovered:
            errors.append(f"window does not cover record times: {uncovered}")
    if kind == "receipt":
        if value.get("received") is not True:
            errors.append("received is not true")
    elif kind == "dedup":
        for field in ("send_count_delta", "duplicate_inbox_count"):
            if type(value.get(field)) is not int or value[field] != 0:
                errors.append(f"{field} must be the integer 0")
    else:
        errors.append(f"unknown evidence kind {kind}")
    fields = (
        "method",
        "reference",
        "window_start",
        "window_end",
        "received",
        "send_count_delta",
        "duplicate_inbox_count",
    )
    return {
        **{key: value[key] for key in fields if key in value},
        "valid": not errors,
        "errors": errors,
    }


def validate_duplicate_evidence(
    path: Path | None, drill: dict[str, Any]
) -> dict[str, Any]:
    initial = parse_time(drill.get("initial_completed_at"))
    start = parse_time(drill.get("duplicate_window_start"))
    end = parse_time(drill.get("duplicate_window_end"))
    if initial is None or start is None or end is None or not initial <= start <= end:
        return {"valid": False, "errors": ["drill has no valid duplicate-only window"]}
    evidence = validate_external_evidence(path, "dedup", record_times=(start, end))
    window_start = parse_time(evidence.get("window_start"))
    if window_start is not None and window_start < initial:
        evidence["errors"].append("duplicate send window includes the initial send")
        evidence["valid"] = False
    return evidence


def notification_kind(deduplication_key: str) -> str | None:
    return next(
        (
            kind
            for kind, marker in NOTIFICATION_KINDS.items()
            if marker in deduplication_key
        ),
        None,
    )


def completion_checks(
    evidence: dict[str, dict[str, Any]],
    dedup_drill: dict[str, Any],
    receipt: dict[str, Any],
    duplicate_window: dict[str, Any],
) -> dict[str, bool]:
    return {
        "gpu_reset_sent": evidence["gpu-reset"]["sent"],
        "workload_restart_sent": evidence["workload-restart"]["sent"],
        "gpu_reset_deduplicated": evidence["gpu-reset"]["deduplicated"],
        "workload_restart_deduplicated": evidence["workload-restart"]["deduplicated"],
        "four_submissions_return_one_provider_id": (
            dedup_drill["statuses"] == ["SENT", "DUPLICATE", "DUPLICATE", "DUPLICATE"]
            and dedup_drill["provider_message_id_present"] is True
            and dedup_drill["provider_message_id_stable"] is True
            and dedup_drill.get("notification_count") == 1
            and dedup_drill.get("notifier_calls") == 1
            and dedup_drill.get("completion_calls") == 4
            and dedup_drill.get("command_statuses") == ["SUCCEEDED"] * 4
            and dedup_drill.get("injection_path")
            == "gpu_fault.app.routes.regional.complete_remote_command"
        ),
        "gpu_reset_mail_keyed_by_the_batched_step": drill_mails_the_batched_reset(
            dedup_drill
        ),
        "receipt_confirmed_outside_the_solution": receipt["valid"],
        "ses_send_count_did_not_increase_for_duplicates": duplicate_window["valid"],
    }


def drill_mails_the_batched_reset(drill: dict[str, Any]) -> bool:
    """The gpu-reset drill replayed a compound carrier and the one mail it
    produced is the batched RESET_GPU step's, keyed by that step's own
    idempotency key; the head QUIESCE_GPU_SERVICES produced none.

    This is the shape production issues on the current protocol (the idle-node
    reset chain of DESTR-001/HA-003/HA-004); a standalone RESET_GPU drill
    proves the head path only and would have passed while every real batched
    reset went unmailed.
    """

    expected = drill.get("expected_operation_id")
    return (
        drill.get("command_shape") == "compound"
        and drill.get("head_operation") == "QUIESCE_GPU_SERVICES"
        and "RESET_GPU" in (drill.get("batched_operations") or [])
        and isinstance(expected, str)
        and expected.endswith("/RESET_GPU")
        and drill.get("notification_operation_ids") == [expected]
    )


def select_live_record(
    records: Sequence[dict[str, Any]], *, kind: str, cluster_id: str
) -> dict[str, Any] | None:
    """Select the newest real SENT record for this cluster and notification kind."""
    matching = [
        item
        for item in records
        if not item.get("missing")
        and item.get("kind") == kind
        and item.get("cluster_name") == cluster_id
        and not item.get("drill_id")
        and item.get("status") == "SENT"
        and item.get("provider_message_id_present")
    ]
    if not matching:
        return None
    return max(matching, key=lambda item: str(item.get("created_at") or ""))


def requeue_route_errors(drill: dict[str, Any]) -> list[str]:
    expected = {
        "suppressed_backlog": 1,
        "unauthorized_status": 403,
        "send_status": "FAILED",
        "status_before_requeue": "DEAD",
        "requeue_status": "QUEUED",
        "status_after_requeue": "PENDING",
        "attempts_after_requeue": 0,
        "dispatch_sent": 1,
        "duplicate_requeue_status": "DUPLICATE",
        "second_dispatch_sent": 0,
        "notifier_calls": 1,
        "store_scope": "isolated-memory",
        "request_path": "/v1/advisory-notifications/{notification_id}/requeue",
        "authorization_bucket": "execution-token",
        "store_io_closed": True,
    }
    return [
        f"public requeue proof differs: {key}"
        for key, value in expected.items()
        if drill.get(key) != value or type(drill.get(key)) is not type(value)
    ]


def _notification_entries(value: Any) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        if isinstance(value.get("notification"), dict) and "result" in value:
            yield value
            return
        for item in value.values():
            yield from _notification_entries(item)
    elif isinstance(value, list):
        for item in value:
            yield from _notification_entries(item)


def action_completed_records_from_evidence(
    run_dir: Path, *, cluster_id: str, release_id: str | None = None
) -> dict[str, list[dict[str, Any]]]:
    records: dict[str, list[dict[str, Any]]] = {kind: [] for kind in NOTIFICATION_KINDS}
    for path in sorted((run_dir / "cases").glob("GF-REGIONAL-*/GF-REGIONAL-*.json")):
        if path.stem != path.parent.name or path.stem.startswith("GF-REGIONAL-NOTIFY-"):
            continue
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for entry in _notification_entries(document):
            notification = entry["notification"]
            if (
                notification.get("category") != "ACTION_COMPLETED"
                or notification.get("cluster_name") != cluster_id
            ):
                continue
            kind = notification_kind(str(notification.get("deduplication_key") or ""))
            if kind is None or not notification.get("notification_id"):
                continue
            proof = predecessor_evidence(
                path, path.stem, cluster_id=cluster_id, release_id=release_id
            )
            if proof.get("valid") is not True:
                raise NotificationAcceptanceError(
                    "ACTION_COMPLETED candidate has failed or unbound case evidence"
                )
            records[kind].append(
                {
                    "case_id": document.get("case_id"),
                    "evidence_path": str(path),
                    "notification_id": str(notification["notification_id"]),
                    "incident_id": notification.get("incident_id"),
                    "deduplication_key": notification.get("deduplication_key"),
                    "created_at": notification.get("created_at"),
                    "recorded_result": entry.get("result"),
                }
            )
    return records


def split_drill_tagged_live_records(
    records: Sequence[dict[str, Any]], *, kind: str, candidate_ids: set[str]
) -> tuple[list[str], list[dict[str, Any]]]:
    """(drill-tagged record ids, records that count as unverified live delivery).

    A real action performed under a drill id (DESTR-001, HA-003 and HA-004
    inject with ``--drill-id``) gets its ACTION_COMPLETED record SKIPPED by the
    drill policy: there was never a live delivery to verify, so such a record
    neither proves nor blocks the kind -- the drill does. Only a non-drill
    record that is missing, not SENT or without a provider id is unverified.
    """

    of_kind = [
        item
        for item in records
        if item.get("kind") == kind or item.get("notification_id") in candidate_ids
    ]
    drill_tagged = sorted(
        str(item.get("notification_id"))
        for item in of_kind
        if item.get("drill_id") and not item.get("missing")
    )
    unusable = [
        item
        for item in of_kind
        if not item.get("drill_id")
        and (
            item.get("missing")
            or item.get("status") != "SENT"
            or not item.get("provider_message_id_present")
        )
    ]
    return drill_tagged, unusable
