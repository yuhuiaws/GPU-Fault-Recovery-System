from __future__ import annotations

from datetime import datetime
from typing import Any


def load_receipt(run_id: str, started: datetime) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "boot_id": "boot-test",
        "intent_sha256": "a" * 64,
        "gpu_index": 0,
        "baseline": [{"index": 0, "uuid": "GPU-test", "power_min_limit_w": 200.0}],
        "load_start": {
            "run_id": run_id,
            "boot_id": "boot-test",
            "intent_sha256": "a" * 64,
            "gpu_index": 0,
            "gpu_uuid": "GPU-test",
            "started_at": started.isoformat(),
            "started_monotonic": 1000.0,
        },
    }


def candidate_record(
    observed_at: datetime, *, cluster_id: str = "cluster-a", node_id: str = "node-a"
) -> dict[str, Any]:
    batch_id = f"dcgm-{node_id}-{int(observed_at.timestamp() * 1_000_000)}"
    payload = {
        "batch_id": batch_id,
        "cluster_id": cluster_id,
        "node_id": node_id,
        "source": "DCGM_EXPORTER",
        "collection_errors": [],
        "observed_at": observed_at.isoformat(),
        "collected_at": observed_at.isoformat(),
        "ingested_at": observed_at.isoformat(),
        "edge_filter_reasons": ["candidate-confirmed"],
        "samples": [
            {
                "canonical_name": name,
                "gpu_uuid": "GPU-test",
                "gpu_index": "0",
                "value": value,
            }
            for name, value in (
                ("power_limit_w", 200.0),
                ("power_usage_w", 206.0),
                ("gpu_utilization_percent", 99.0),
            )
        ],
    }
    return {
        "record_id": f"gpu-metrics/{batch_id}",
        "cluster_id": cluster_id,
        "node_id": node_id,
        "kind": "GPU_METRICS",
        "observed_at": observed_at.isoformat(),
        "ingested_at": observed_at.isoformat(),
        "payload": payload,
    }
