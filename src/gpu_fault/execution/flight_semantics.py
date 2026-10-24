"""Conservative operation evidence; P2P sequence numbers are not collective IDs."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from gpu_fault.communication_window import common_window
from gpu_fault.execution.flight_temporal import snapshot_groups
from gpu_fault.flight_progress import completed, is_p2p, operation_key
from gpu_fault.pytorch_fr_matcher import match_collective


def persistent_operations(signal: dict[str, Any]) -> list[dict[str, Any]]:
    series = signal.get("flight_recorder_series") or {}
    identity = signal.get("process_identity") or {}
    if (
        series.get("status") != "PAIRED"
        or identity.get("verified") is not True
        or type(identity.get("start_time_ticks")) is not int
        or identity.get("start_time_ticks") != series.get("process_start_ticks")
    ):
        return []
    first = snapshot_groups(series.get("first") or {})
    second = snapshot_groups(series.get("second") or {})
    result = []
    for key in first.keys() & second.keys():
        old = {
            operation_key(entry)
            for entry in first[key].get("semantic_entries", [])
            if not completed(entry)
        } - {None}
        for entry in second[key].get("semantic_entries", [])[:64]:
            if completed(entry) or operation_key(entry) not in old:
                continue
            result.append(
                {
                    "key": key,
                    "entry": entry,
                    "rank": signal.get("rank"),
                    "node_id": signal.get("node_id"),
                    "pid": signal.get("pid"),
                    "members": second[key].get("pg_rank_order") or [],
                    "first_observed_at": series.get("first_observed_at"),
                    "second_observed_at": series.get("second_requested_at"),
                    "clock_uncertainty_seconds": series.get(
                        "clock_uncertainty_seconds"
                    ),
                }
            )
    return result


def flight_semantics(signals: list[dict[str, Any]]) -> dict[str, Any]:
    collectives: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    unsupported_p2p = 0
    for signal in signals:
        for row in persistent_operations(signal):
            entry = row["entry"]
            if not is_p2p(entry):
                collectives[(*row["key"], entry["collective_seq_id"])].append(row)
                continue
            unsupported_p2p += 1
    findings = []
    for key, rows in collectives.items():
        window = common_window(rows, max_seconds=30)
        if len({row["rank"] for row in rows}) < 2 or not window["time_comparable"]:
            continue
        entries = [row["entry"] for row in rows]
        states = {match_collective(entries[0], entry, key[2]) for entry in entries[1:]}
        issues = sorted(
            states - {"FULLY_MATCHED", "UNDECIDED", "COLLECTIVE_STATE_MISMATCH"}
        )
        if issues:
            findings.append(
                {
                    "pg_name": key[0],
                    "collective_seq_id": key[-1],
                    "observed_ranks": sorted(
                        {row["rank"] for row in rows if row["rank"] is not None}
                    ),
                    "issues": issues,
                    "confidence": "PLAUSIBLE",
                    **window,
                }
            )
    return {
        "status": "OBSERVED" if findings else "UNDETERMINED",
        "findings": findings[:256],
        "truncated": len(findings) > 256,
        "unsupported_p2p_entries": unsupported_p2p,
        "p2p_analysis": "OUT_OF_SCOPE",
        "root_cause_confirmed": False,
        "matcher": "pytorch-fr/2.10-collective-rules",
        "undecided_requires": "complete archived traces and official offline analyzer",
    }
