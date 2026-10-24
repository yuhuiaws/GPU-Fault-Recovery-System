"""Observed collective watermarks, not a claim that ring-buffer history is complete."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any


def observed_timestamp(value: Any) -> bool:
    return (
        type(value) is int
        and 0 < value < 2**64
        or type(value) is float
        and math.isfinite(value)
        and value > 0
    )


def completed(entry: dict[str, Any]) -> bool:
    return (
        observed_timestamp(entry.get("time_discovered_completed"))
        or str(entry.get("state") or "").lower() == "completed"
    )


def started(entry: dict[str, Any]) -> bool:
    return (
        completed(entry)
        or observed_timestamp(entry.get("time_discovered_started"))
        or str(entry.get("state") or "").lower() == "started"
    )


def operation_key(entry: dict[str, Any]) -> str | None:
    identity = [entry.get(name) for name in ("record_id", "op_id", "time_created")]
    if all(value is None for value in identity) or any(
        value is not None and type(value) not in (int, str) for value in identity
    ):
        return None
    return hashlib.sha256(
        json.dumps(
            [
                entry["collective_seq_id"],
                str(entry.get("profiling_name") or ""),
                *identity,
            ],
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def is_p2p(entry: dict[str, Any]) -> bool:
    return entry.get("is_p2p") is True or any(
        token in str(entry.get("profiling_name") or "").lower()
        for token in ("send", "recv")
    )


def group_progress(entries: list[dict[str, Any]]) -> dict[str, Any]:
    p2p = any(is_p2p(item) for item in entries)
    known = [
        item
        for item in entries
        if type(item.get("collective_seq_id")) is int and item["collective_seq_id"] >= 0
    ]
    pending = [
        item
        for item in known
        if not completed(item)
        and (
            started(item)
            or str(item.get("state") or "").lower() in {"scheduled", "enqueued"}
        )
    ]
    oldest = min(pending, key=lambda item: item["collective_seq_id"], default=None)
    return {
        "last_enqueued": max(
            (item["collective_seq_id"] for item in known), default=None
        ),
        "last_started": max(
            (item["collective_seq_id"] for item in known if started(item)), default=None
        ),
        "last_completed": max(
            (item["collective_seq_id"] for item in known if completed(item)),
            default=None,
        ),
        "oldest_pending": {
            "collective_seq_id": oldest["collective_seq_id"],
            "operation_key": operation_key(oldest),
            "started": started(oldest),
        }
        if oldest is not None
        else None,
        "observed_pending_count": len(pending),
        "comparison_supported": bool(known) and not p2p,
        "unsupported_reason": "P2P_OUT_OF_SCOPE" if p2p else None,
        "history_complete": False,
        "watermark_source": "VISIBLE_RING_ENTRIES",
    }


def apply_pg_status(
    progress: dict[str, Any], payload: Any, pg_id: Any
) -> dict[str, Any]:
    """PyTorch 2.10 keys pg_status by numeric pg_id, not the PG's name/UID."""
    if not isinstance(payload, dict):
        return progress
    result = {**progress, "dump_version": payload.get("version")}
    if str(payload.get("version")) != "2.10":
        return {**result, "pg_status_availability": "UNSUPPORTED_DUMP_VERSION"}
    statuses = payload.get("pg_status")
    status = statuses.get(str(pg_id)) if isinstance(statuses, dict) else None
    if not isinstance(status, dict) or pg_id is None:
        return {**result, "pg_status_availability": "MISSING_OR_UNMAPPED_PG_ID"}
    counters = {}
    for name in ("last_enqueued", "last_started", "last_completed"):
        raw = status.get(name + "_collective")
        if isinstance(raw, str) and (raw.isdigit() or raw == "-1"):
            raw = int(raw)
        if type(raw) is not int or not -1 <= raw < 2**63:
            return {
                **result,
                "comparison_supported": False,
                "pg_status_availability": "INVALID",
            }
        counters[name] = raw
    if not (
        counters["last_completed"]
        <= counters["last_started"]
        <= counters["last_enqueued"]
    ):
        return {
            **result,
            "comparison_supported": False,
            "pg_status_availability": "INCONSISTENT",
        }
    pending = result.get("oldest_pending")
    if pending and pending["collective_seq_id"] <= counters["last_completed"]:
        pending = None
    return {
        **result,
        **{name: value if value >= 0 else None for name, value in counters.items()},
        "oldest_pending": pending,
        "watermark_source": "PG_STATUS",
        "pg_status_availability": "AVAILABLE",
    }
