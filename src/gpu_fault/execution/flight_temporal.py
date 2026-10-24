"""Compare process-bound FR observations without confusing enqueue with completion."""

from __future__ import annotations

import hashlib
import json
import math
from bisect import bisect_left, bisect_right
from collections import defaultdict
from datetime import datetime
from typing import Any

from gpu_fault.communication_window import common_window


def group_key(entry: dict[str, Any]) -> tuple[str, str, int] | None:
    name, members = entry.get("pg_name"), entry.get("pg_members")
    digest, count = entry.get("pg_membership_sha256"), entry.get("pg_member_count")
    if (
        isinstance(members, list)
        and members
        and all(type(rank) is int and rank >= 0 for rank in members)
    ):
        if len(set(members)) != len(members):
            return None
        actual = membership_digest(members)
        if (
            digest is not None
            and digest != actual
            or count is not None
            and count != len(members)
        ):
            return None
        digest, count = actual, len(members)
    if (
        not name
        or name == ":"
        or not isinstance(digest, str)
        or len(digest) != 64
        or type(count) is not int
        or count <= 0
    ):
        return None
    return str(name), digest, count


def membership_digest(members: list[int]) -> str:
    return hashlib.sha256(
        json.dumps(sorted(members), separators=(",", ":")).encode()
    ).hexdigest()


def snapshot_groups(
    snapshot: dict[str, Any],
) -> dict[tuple[str, str, int], dict[str, Any]]:
    if not isinstance(snapshot, dict):
        return {}
    if (
        snapshot.get("status") != "dumped"
        or snapshot.get("parse_error")
        or snapshot.get("process_groups_truncated")
    ):
        return {}
    entries = snapshot.get("groups")
    if not isinstance(entries, list) or len(entries) > 64:
        return {}
    result = {}
    duplicates = set()
    for entry in entries:
        if not isinstance(entry, dict) or (key := group_key(entry)) is None:
            continue
        if key in result:
            duplicates.add(key)
        result[key] = entry
    return {key: entry for key, entry in result.items() if key not in duplicates}


def paired_groups(signal: dict[str, Any]) -> list[dict[str, Any]]:
    series = signal.get("flight_recorder_series") or {}
    identity = signal.get("process_identity") or {}
    if not isinstance(series, dict) or not isinstance(identity, dict):
        return []
    if (
        series.get("version") != "fr-series/v1"
        or series.get("status") != "PAIRED"
        or identity.get("verified") is not True
        or type(series.get("process_start_ticks")) is not int
        or series["process_start_ticks"] <= 0
        or identity.get("start_time_ticks") != series["process_start_ticks"]
    ):
        return []
    interval = series.get("interval_seconds")
    if (
        isinstance(interval, bool)
        or not isinstance(interval, (int, float))
        or not math.isfinite(interval)
        or not 1 <= interval <= 30
    ):
        return []
    try:
        first_at = datetime.fromisoformat(series["first_observed_at"])
        requested_at = datetime.fromisoformat(series["second_requested_at"])
        second_at = datetime.fromisoformat(series["second_observed_at"])
        if (
            any(stamp.tzinfo is None for stamp in (first_at, requested_at, second_at))
            or not first_at < requested_at <= second_at
            or (second_at - first_at).total_seconds() > 30
        ):
            return []
    except (TypeError, ValueError, KeyError):
        return []
    first, second = (
        snapshot_groups(series.get("first") or {}),
        snapshot_groups(series.get("second") or {}),
    )
    result = []
    for key in first.keys() & second.keys():
        before, after = (
            first[key].get("progress") or {},
            second[key].get("progress") or {},
        )
        if not isinstance(before, dict) or not isinstance(after, dict):
            continue
        if not comparable_progress(before, after):
            continue
        old_pending, new_pending = (
            before.get("oldest_pending") or {},
            after.get("oldest_pending") or {},
        )
        persists = bool(
            old_pending.get("operation_key")
            and old_pending.get("operation_key") == new_pending.get("operation_key")
            and old_pending.get("collective_seq_id")
            == new_pending.get("collective_seq_id")
        )
        old, new = before["last_completed"], after["last_completed"]
        advanced = new is not None and old is not None and new > old
        if old is None and new is not None and old_pending:
            advanced = new >= old_pending["collective_seq_id"]
        result.append(
            {
                "key": key,
                "rank": signal["rank"],
                "pid": signal.get("pid"),
                "node_id": signal.get("node_id"),
                "before": before,
                "after": after,
                "first_at": first_at,
                "requested_at": requested_at,
                "clock_uncertainty_seconds": series.get("clock_uncertainty_seconds"),
                "completion_advanced": advanced,
                "enqueue_advanced": after["last_enqueued"] > before["last_enqueued"],
                "watermark_deltas": {
                    field: after[field] - before[field]
                    if before[field] is not None and after[field] is not None
                    else None
                    for field in ("last_enqueued", "last_started", "last_completed")
                },
                "pending_persists": persists,
                "pending_observed_seconds": interval if persists else None,
                "members": second[key].get("pg_members") or [],
            }
        )
    return result


def comparable_progress(before: dict[str, Any], after: dict[str, Any]) -> bool:
    if not before.get("comparison_supported") or not after.get("comparison_supported"):
        return False
    for name in ("last_enqueued", "last_started", "last_completed"):
        if name not in before or name not in after:
            return False
        old, new = before.get(name), after.get(name)
        if any(
            value is not None and (type(value) is not int or value < 0)
            for value in (old, new)
        ):
            return False
        if old is not None and (new is None or new < old):
            return False
    for point in (before, after):
        pending = point.get("oldest_pending")
        if pending is not None and (
            not isinstance(pending, dict)
            or type(pending.get("collective_seq_id")) is not int
            or type(point.get("last_enqueued")) is not int
            or not 0 <= pending["collective_seq_id"] <= point["last_enqueued"]
        ):
            return False
    return (
        before.get("last_enqueued") is not None
        and after.get("last_enqueued") is not None
    )


def progress_reference(rows: list[dict[str, Any]]) -> dict[str, list[int]]:
    return {
        **{
            field: sorted(
                min(row["before"][field], row["after"][field])
                for row in rows
                if all(
                    type(row[point].get(field)) is int for point in ("before", "after")
                )
            )
            for field in ("last_completed", "last_started")
        },
        "pending": sorted(
            row["after"]["oldest_pending"]["collective_seq_id"]
            for row in rows
            if row["pending_persists"]
        ),
    }


def rank_reasons(
    row: dict[str, Any], reference: dict[str, list[int]], member_count: int
) -> list[str]:
    if row["completion_advanced"]:
        return []
    before, after = row["before"], row["after"]
    pending = after.get("oldest_pending") or {}
    reasons = []
    if row["pending_persists"]:
        seq = pending["collective_seq_id"]
        for field, reason in (
            ("last_completed", "PERSISTENT_COMPLETION_LAG"),
            ("last_started", "PERSISTENT_START_LAG"),
        ):
            if field == "last_started" and (
                pending.get("started")
                or (before.get("oldest_pending") or {}).get("started")
            ):
                continue
            values = reference[field]
            own_vote = all(
                type(row[point].get(field)) is int and row[point][field] >= seq
                for point in ("before", "after")
            )
            votes = len(values) - bisect_left(values, seq) - int(own_vote)
            if votes > member_count / 2:
                reasons.append(reason)
    if before["last_enqueued"] == after["last_enqueued"] and (
        row["pending_persists"]
        or not before.get("oldest_pending")
        and not after.get("oldest_pending")
    ):
        values = reference["pending"]
        votes = len(values) - bisect_right(values, after["last_enqueued"])
        if votes > member_count / 2:
            reasons.append("PERSISTENT_ENQUEUE_LAG")
    return reasons


def temporal_verdict(signals: list[dict[str, Any]]) -> dict[str, Any] | None:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for signal in signals:
        for row in paired_groups(signal):
            groups[row["key"]].append(row)
    if not groups:
        return None
    findings, candidate_groups, stalled_groups = [], [], []
    candidates: set[int] = set()
    controls: set[int] = set()
    for (name, digest, count), rows in sorted(groups.items()):
        members = sorted(row["rank"] for row in rows)
        full = len(members) == count and membership_digest(members) == digest
        window = common_window(
            [
                {
                    **row,
                    "first_observed_at": row["first_at"].isoformat(),
                    "second_observed_at": row["requested_at"].isoformat(),
                }
                for row in rows
            ],
            max_seconds=30,
        )
        overlap = window["common_window_seconds"]
        comparable = window["time_comparable"]
        reference = progress_reference(rows)
        reasons = (
            {str(row["rank"]): rank_reasons(row, reference, len(rows)) for row in rows}
            if comparable and len(rows) >= 3
            else {}
        )
        leads = {int(rank) for rank, values in reasons.items() if values}
        stalled = [
            row["rank"]
            for row in rows
            if row["pending_persists"] and not row["completion_advanced"]
        ]
        finding = {
            "pg_name": name,
            "membership_sha256": digest,
            "expected_member_count": count,
            "expected_members": rows[0]["members"],
            "observed_members": members,
            "membership_verified": full,
            "common_observation_seconds": overlap,
            "time_reason": window["time_reason"],
            "time_comparable": comparable,
            "culprit_ranks": sorted(leads),
            "candidate_reasons": reasons,
            "completion_advanced_ranks": [
                row["rank"] for row in rows if row["completion_advanced"]
            ],
            "enqueue_advanced_ranks": [
                row["rank"] for row in rows if row["enqueue_advanced"]
            ],
            "watermark_deltas_by_rank": {
                str(row["rank"]): row["watermark_deltas"] for row in rows
            },
            "persistent_pending_ranks": stalled,
            "pending_observed_seconds_by_rank": {
                str(row["rank"]): row["pending_observed_seconds"]
                for row in rows
                if row["pending_persists"]
            },
            "group_stall_observed": bool(full and comparable and len(stalled) == count),
        }
        findings.append(finding)
        if finding["group_stall_observed"]:
            stalled_groups.append(name)
        if leads:
            candidates.update(leads)
            controls.update(set(members) - leads)
            candidate_groups.append(finding)
    return {
        "classification": "PLAUSIBLE"
        if candidates or stalled_groups
        else "UNDETERMINED",
        "confidence": "PLAUSIBLE" if candidates or stalled_groups else "NONE",
        "culprit_ranks": sorted(candidates),
        "control_ranks": sorted(controls - candidates)[:1],
        "reason": "process-bound FR time series; persistent pending operations are diagnostic evidence, not proof of root cause",
        "temporal_verified": True,
        "membership_verified": bool(candidate_groups)
        and all(item["membership_verified"] for item in candidate_groups),
        "process_groups": findings,
        "stalled_process_groups": stalled_groups,
        "temporal_rank_count": len(
            {row["rank"] for rows in groups.values() for row in rows}
        ),
        "ranks_without_comparable_fr_pair": sorted(
            {signal["rank"] for signal in signals}
            - {row["rank"] for rows in groups.values() for row in rows}
        ),
        "pending_age_basis": "lower_bound_between_two_observations_not_time_since_enqueue",
    }
