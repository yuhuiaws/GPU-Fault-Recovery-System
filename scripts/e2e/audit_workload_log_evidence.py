from __future__ import annotations

import json
import os
import sys

from gpu_fault.store import PostgresStore
from gpu_fault.telemetry import EvidenceKind


def main() -> None:
    cluster_id, attempt_id = sys.argv[1:3]
    store = PostgresStore(
        os.environ["GPU_FAULT_STORE_URL"],
        pool_min_size=1,
        pool_max_size=2,
        pool_timeout_seconds=5,
    )
    try:
        evidence = store.list_raw_evidence(
            cluster_id,
            attempt_id=attempt_id,
            kind=EvidenceKind.WORKLOAD_LOG,
            limit=100,
        )
        print(
            json.dumps(
                [
                    {
                        "record_id": item.record_id,
                        "expires_at": item.expires_at.isoformat(),
                        "ttl_hours": round(
                            (item.expires_at - item.ingested_at).total_seconds() / 3600,
                            2,
                        ),
                        "tail_bytes": item.payload.get("tail_bytes"),
                        "truncated": item.payload.get("truncated"),
                        "s3_uri": item.payload.get("s3_uri"),
                        "heartbeat": (
                            "Q118_GPU_TRAIN" in str(item.payload.get("tail") or "")
                        ),
                    }
                    for item in evidence
                ],
                indent=2,
            )
        )
    finally:
        store.close()


if __name__ == "__main__":
    main()
