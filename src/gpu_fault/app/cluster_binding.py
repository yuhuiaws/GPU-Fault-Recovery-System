from __future__ import annotations

from typing import Any

CLUSTER_IDENTIFIER_FIELDS = frozenset({"cluster_id", "clusterId", "cluster"})


def payload_cluster_ids(payload: Any) -> set[str]:
    cluster_ids: set[str] = set()
    pending = [payload]
    visited = 0
    while pending:
        current = pending.pop()
        visited += 1
        if visited > 100_000:
            raise ValueError("JSON payload is too structurally complex")
        if isinstance(current, dict):
            for key, value in current.items():
                if key in CLUSTER_IDENTIFIER_FIELDS:
                    if not isinstance(value, str) or not value:
                        raise ValueError(
                            f"payload {key} values must be non-empty strings"
                        )
                    cluster_ids.add(value)
                else:
                    pending.append(value)
        elif isinstance(current, list):
            pending.extend(current)
    return cluster_ids
