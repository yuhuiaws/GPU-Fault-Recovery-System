"""Keep OS process identity separate from a communication rank hint."""

from __future__ import annotations

from collections import Counter
from typing import Any


def participant(row: dict[str, Any]) -> bool:
    identity = row.get("process_identity") or {}
    return bool(
        identity.get("verified") is True
        and type(identity.get("start_time_ticks")) is int
        and identity["start_time_ticks"] > 0
        and (
            row.get("participant_basis") == "PROCESS_BOUND_COMMUNICATION"
            or row.get("gpu_uuids")
        )
    )


def inventory(processes: list[dict[str, Any]]) -> dict[str, Any]:
    participants = [row for row in processes if participant(row)]
    ranked = [
        row for row in participants if type(row.get("rank")) is int and row["rank"] >= 0
    ]
    counts = Counter(row["rank"] for row in ranked)
    valid = [row for row in ranked if counts[row["rank"]] == 1]
    sizes = {
        row["world_size"]
        for row in participants
        if type(row.get("world_size")) is int and 0 < row["world_size"] <= 65536
    }
    expected = next(iter(sizes)) if len(sizes) == 1 else None
    return {
        "participants": participants,
        "valid": valid,
        "expected": expected,
        "duplicates": sorted(rank for rank, count in counts.items() if count > 1),
        "unknown_rank_count": len(participants) - len(ranked),
        "auxiliary_count": sum(
            row.get("participant_basis") == "NO_OBSERVED_GPU_OR_COMMUNICATION"
            for row in processes
        ),
        "unknown_count": sum(
            not participant(row)
            and row.get("participant_basis") != "NO_OBSERVED_GPU_OR_COMMUNICATION"
            for row in processes
        ),
    }
