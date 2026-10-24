"""Pair two whole-task sweeps by process generation, never by inherited RANK."""

from __future__ import annotations

from typing import Any

from gpu_fault.hung_summary_codec import decode_summary, encode_summary


def recheck_details(
    baseline: dict[str, Any], current: dict[str, Any]
) -> dict[str, Any]:
    results = {}
    for node, encoded in (current.get("node_results") or {}).items():
        try:
            right = decode_summary(encoded)
            left = decode_summary((baseline.get("node_results") or {}).get(node, {}))
            if any(
                left.get(key) != right.get(key) or right.get(key) is None
                for key in ("job_id", "attempt_id", "snapshot_epoch", "node_id")
            ):
                raise ValueError("sweep identity mismatch")
            old_rows = {
                (
                    row.get("pid"),
                    (row.get("process_identity") or {}).get("start_time_ticks"),
                ): row
                for row in left.get("ranks", [])
            }
            for row in right.get("ranks", []):
                old = old_rows.get(
                    (
                        row.get("pid"),
                        (row.get("process_identity") or {}).get("start_time_ticks"),
                    )
                )
                pairs = []
                if old and (old.get("process_identity") or {}).get("verified") is True:
                    old_groups = {
                        (left.get("groups") or {})
                        .get(item["group"], {})
                        .get("group_key"): item
                        for item in old.get("communication", [])
                    }
                    for item in row.get("communication", []):
                        if item.get("source") == "CACHED_FLIGHT_RECORDER":
                            pairs.append(item)
                            continue
                        descriptor = (right.get("groups") or {}).get(item["group"], {})
                        previous = old_groups.get(descriptor.get("group_key"))
                        if previous is not None:
                            pairs.append(
                                {
                                    **item,
                                    "before": previous["after"],
                                    "first_observed_at": previous.get(
                                        "second_observed_at"
                                    ),
                                    "sweep_pair": True,
                                }
                            )
                row["communication"] = pairs
            right["communication_sweep_pair"] = True
            results[node] = encode_summary(right)
        except (TypeError, ValueError, KeyError):
            # Keep OS evidence, but do not turn a failed baseline into a time series.
            try:
                right = decode_summary(encoded)
                for row in right.get("ranks", []):
                    row["communication"] = []
                results[node] = encode_summary(right)
            except (TypeError, ValueError):
                results[node] = encoded
    return {
        **current,
        "node_results": results,
        "communication_sampling": "TWO_SWEEPS_WITH_BARRIER",
    }
