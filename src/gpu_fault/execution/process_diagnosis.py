"""Fuse independent observations by process identity, without promoting symptoms to causes."""

from __future__ import annotations

from collections import Counter
from typing import Any

from gpu_fault.execution.flight_temporal import paired_groups
from gpu_fault.execution.process_cohorts import compare_cohorts
from gpu_fault.execution.process_observations import (
    behavior_observations,
    process_key,
    stack_observations,
)


def process_diagnosis(
    signals: list[dict[str, Any]], fr_candidates: set[int] | None = None
) -> dict[str, Any]:
    counts = Counter(key for row in signals if (key := process_key(row)) is not None)
    rows = [
        {**row, "_process_key": key}
        for row in signals
        if (key := process_key(row)) is not None and counts[key] == 1
    ]
    reports = {}
    for row in rows:
        stack, behavior = stack_observations(row), behavior_observations(row)
        local_fr = paired_groups(row)
        reports[row["_process_key"]] = {
            "node_id": row["node_id"],
            "pid": row["pid"],
            "start_time_ticks": row["_process_key"][2],
            "rank": row.get("rank"),
            "stack": stack,
            "behavior": behavior,
            "observations": [*stack["observations"], *behavior["observations"]],
            "fr_completion_advanced_groups": [
                item["key"][0] for item in local_fr if item["completion_advanced"]
            ],
            "fr_persistent_pending_groups": [
                item["key"][0] for item in local_fr if item["pending_persists"]
            ],
            "cross_evidence": "identity_bound_separate_observation_windows_not_a_fault_probability",
            "root_cause_confirmed": False,
        }
        if row.get("rank") in (fr_candidates or set()):
            reports[row["_process_key"]]["observations"].append(
                {
                    "code": "FR_PROGRESS_OUTLIER",
                    "candidate": True,
                    "qualification": "Communication lead, not confirmed cause",
                }
            )
        elif any(item["pending_persists"] for item in local_fr):
            reports[row["_process_key"]]["observations"].append(
                {
                    "code": "FR_PERSISTENT_PENDING_OPERATION",
                    "candidate": True,
                    "qualification": "Process-local FR observation; no rank identity required",
                }
            )
    cohorts = compare_cohorts(rows, reports)
    candidates = [
        {
            "node_id": item["node_id"],
            "pid": item["pid"],
            "rank": item["rank"],
            "start_time_ticks": item["start_time_ticks"],
            "reasons": sorted(
                {
                    observation["code"]
                    for observation in item["observations"]
                    if observation.get("candidate")
                }
            ),
            "confidence": "WEAK",
            "diagnostic_only": True,
        }
        for item in reports.values()
        if any(observation.get("candidate") for observation in item["observations"])
    ]
    return {
        "verified_process_count": len(rows),
        "unknown_rank_process_count": sum(
            type(row.get("rank")) is not int for row in rows
        ),
        "identity_unavailable_count": len(signals) - len(rows),
        "duplicate_processes": [
            list(key) for key, count in counts.items() if count > 1
        ],
        "processes": list(reports.values())[:256],
        "process_details_truncated": len(reports) > 256,
        "cohorts": cohorts,
        "candidate_processes": candidates[:256],
        "candidate_process_count": len(candidates),
        "candidate_processes_truncated": len(candidates) > 256,
    }
