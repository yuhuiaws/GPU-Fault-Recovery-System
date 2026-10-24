"""Attach bounded pre-incident history and process-bound NIC evidence to a survey."""

from __future__ import annotations

from typing import Any
import hashlib
import sqlite3
import zlib

from gpu_fault.node_agent.communication_history import CommunicationHistory
from gpu_fault.node_agent.operations.hung_summary_io import read_bounded
from gpu_fault.ras_progress import ras_progress


def cached_communication(agent: Any, command: Any) -> dict[str, Any]:
    path = agent.diagnostic_output_dir / "communication-history.db"
    if not path.exists():
        return {"communication_history": {"status": "UNAVAILABLE"}}
    try:
        boot = read_bounded(agent.proc_root / "sys/kernel/random/boot_id", 128).strip()
        history = CommunicationHistory(path)
        scope = history.scope(
            command.parameters["job_id"], command.parameters["attempt_id"], boot
        )
        now = agent.now().timestamp()
        epochs = command.parameters.get("communication_history_epochs")
        if epochs is not None:
            if not (
                isinstance(epochs, list)
                and len(epochs) == 2
                and all(type(value) is int for value in epochs)
                and epochs[1] == epochs[0] + 1
                and 0 <= now - epochs[1] * 30 <= 1800
            ):
                return {"communication_history": {"status": "INVALID_REQUESTED_EPOCHS"}}
            rows = history.read(
                scope,
                since=epochs[0] * 30,
                until=(epochs[1] + 1) * 30 - 0.000001,
                limit=2,
            )
            rows = [row for row in rows if row["epoch"] in epochs]
        else:
            rows = history.read(scope, since=now - 120, until=now, limit=2)
        if len(rows) != 2:
            return {
                "communication_history": {
                    "status": "INSUFFICIENT_SAMPLES",
                    "samples": len(rows),
                }
            }
        first, second = (row["document"] for row in rows)
        span = rows[1]["observed_at_epoch"] - rows[0]["observed_at_epoch"]
        if not 1 <= span <= 90:
            return {"communication_history": {"status": "INVALID_WINDOW"}}
        return {
            "communication_history": {
                "status": "AVAILABLE",
                "first_epoch": rows[0]["epoch"],
                "second_epoch": rows[1]["epoch"],
                "interval_seconds": span,
                "age_seconds": now - rows[1]["observed_at_epoch"],
                "first_observed_at": first["observed_at"],
                "second_observed_at": second["observed_at"],
                "samples": 2,
                "clock_uncertainty_seconds": command.parameters.get(
                    "clock_uncertainty_seconds"
                ),
                "requested_epochs": epochs,
            },
            "node_addresses": second["node_addresses"],
            "cached_ras": ras_progress(first.get("ras", {}), second.get("ras", {})),
            "cached_fr": cached_flight_pairs(first, second),
        }
    except (OSError, ValueError, KeyError, sqlite3.Error, zlib.error) as exc:
        return {
            "communication_history": {"status": "UNAVAILABLE", "reason": str(exc)[:128]}
        }


def cached_flight_pairs(
    first: dict[str, Any], second: dict[str, Any]
) -> list[dict[str, Any]]:
    before = {
        (row["pid"], row["process_identity"]["start_time_ticks"]): row
        for row in first.get("processes", [])
        if row["process_identity"]["verified"]
    }
    result = []
    for row in second.get("processes", []):
        previous = before.get((row["pid"], row["process_identity"]["start_time_ticks"]))
        old, new = (
            (previous or {}).get("flight_recorder", {}),
            row.get("flight_recorder", {}),
        )
        if (
            previous is None
            or not row["process_identity"]["verified"]
            or old.get("status") != "dumped"
            or new.get("status") != "dumped"
            or old.get("parse_error")
            or new.get("parse_error")
            or not old.get("source_mtime_ns")
            or new.get("source_mtime_ns", 0) <= old["source_mtime_ns"]
        ):
            continue
        result.append(
            {
                "pid": row["pid"],
                "rank": row["rank"],
                "process_identity": row["process_identity"],
                "first": old.get("last_entries_by_pg", []),
                "second": new.get("last_entries_by_pg", []),
                "completion_source": "FR_STATUS_OR_VISIBLE_ENTRIES",
                "scope": "EXISTING_FILES_ONLY",
            }
        )
    return result[:256]


def apply_cached_flights(summary: dict[str, Any]) -> None:
    window = summary.get("communication_history", {})
    if window.get("status") != "AVAILABLE":
        return
    processes = {row["pid"]: row for row in summary["ranks"]}
    for pair in summary.get("cached_fr", []):
        row = processes.get(pair["pid"])
        if (
            row is None
            or row.get("communication")
            or row["process_identity"].get("verified") is not True
            or row["process_identity"].get("start_time_ticks")
            != pair["process_identity"]["start_time_ticks"]
        ):
            continue
        previous = {group["pg_name"]: group for group in pair["first"]}
        for after in pair["second"]:
            before = previous.get(after["pg_name"])
            if (
                before is None
                or any(
                    before.get(field) != after.get(field)
                    for field in ("pg_membership_sha256", "pg_member_count")
                )
                or not after.get("pg_membership_sha256")
            ):
                continue
            left, right = before.get("progress", {}), after.get("progress", {})
            if not left.get("comparison_supported") or not right.get(
                "comparison_supported"
            ):
                continue
            key = (
                "fr-"
                + hashlib.sha256(
                    f"{after['pg_name']}/{after['pg_membership_sha256']}".encode()
                ).hexdigest()
            )
            summary["groups"][key] = {
                "group_key": key,
                "name": after["pg_name"],
                "membership_sha256": after["pg_membership_sha256"],
                "member_count": after["pg_member_count"],
            }
            row["communication"].append(
                {
                    "group": key,
                    "before": [
                        left.get(field) if left.get(field) is not None else -1
                        for field in ("last_enqueued", "last_started", "last_completed")
                    ],
                    "after": [
                        right.get(field) if right.get(field) is not None else -1
                        for field in ("last_enqueued", "last_started", "last_completed")
                    ],
                    "first_observed_at": window["first_observed_at"],
                    "second_observed_at": window["second_observed_at"],
                    "clock_uncertainty_seconds": window["clock_uncertainty_seconds"],
                    "source": "CACHED_FLIGHT_RECORDER",
                }
            )
        if row["communication"]:
            row["communication_status"] = "AVAILABLE"
    summary.pop("cached_fr", None)
