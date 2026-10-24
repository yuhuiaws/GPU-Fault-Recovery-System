"""Compare two complete RAS views without calling submission counts completion."""

from __future__ import annotations

from collections import Counter
from typing import Any


def ras_progress(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    if any(
        value.get("status") != "COLLECTED" or value.get("timeouts_count") != 0
        for value in (before, after)
    ):
        return {"status": "UNAVAILABLE", "groups": []}
    previous = {tuple(group["identity"]): group for group in before.get("groups", [])}
    groups = []
    for group in after.get("groups", []):
        old = previous.get(tuple(group["identity"]))
        if old is None or old["member_count"] != group["member_count"]:
            continue
        old_members = {
            (row["rank"], row["host"], row["pid"]): row for row in old["members"]
        }
        paired = [
            (old_members[key], row)
            for row in group["members"]
            if (key := (row["rank"], row["host"], row["pid"])) in old_members
        ]
        complete = (
            old["complete"]
            and group["complete"]
            and len(paired) == group["member_count"]
        )
        candidates = []
        ready = all(
            row["status"].get("init_state") == 0
            and row["status"].get("async_error") == 0
            and not any(
                row["status"].get(key)
                for key in ("finalize_called", "destroy_flag", "abort_flag")
            )
            for pair in paired
            for row in pair
        )
        operations = (
            set.intersection(*(set(row["collective_counts"]) for _, row in paired))
            if paired
            else set()
        )
        for operation in sorted(operations - {"SendRecv", "Send", "Recv"}):
            counts = [row["collective_counts"][operation] for _, row in paired]
            references = Counter(counts).most_common(1)
            old_references = Counter(
                left["collective_counts"].get(operation) for left, _ in paired
            ).most_common(1)
            if (
                not complete
                or not ready
                or not references
                or references[0][1] <= len(paired) / 2
                or not old_references
                or old_references[0][0] is None
                or old_references[0][1] <= len(paired) / 2
                or any(
                    right["collective_counts"][operation]
                    < left["collective_counts"].get(operation, 0)
                    for left, right in paired
                )
            ):
                continue
            reference = references[0][0]
            for left, right in paired:
                first, last = (
                    left["collective_counts"].get(operation),
                    right["collective_counts"][operation],
                )
                if (
                    first is not None
                    and first == last < reference
                    and first < old_references[0][0]
                ):
                    candidates.append(
                        {
                            "communicator_rank": right["rank"],
                            "host": right["host"],
                            "pid": right["pid"],
                            "operation": operation,
                            "submission_count": last,
                            "reason": "PERSISTENT_SUBMISSION_COUNT_LAG",
                            "confidence": "WEAK",
                        }
                    )
        groups.append(
            {
                "identity": group["identity"],
                "expected_members": group["member_count"],
                "paired_members": len(paired),
                "membership_complete": complete,
                "communicator_ready": ready,
                "unchanged_submission_members": sum(
                    left["collective_counts"] == right["collective_counts"]
                    for left, right in paired
                ),
                "candidate_members": candidates[:64],
                "candidates_truncated": len(candidates) > 64,
                "counter_semantics": "SUBMISSION_NOT_COMPLETION",
                "rank_namespace": "COMMUNICATOR_LOCAL",
            }
        )
    return {
        "status": "PAIRED",
        "groups": groups[:256],
        "groups_truncated": len(groups) > 256,
        "completion_progress_available": False,
        "root_cause_confirmed": False,
    }
