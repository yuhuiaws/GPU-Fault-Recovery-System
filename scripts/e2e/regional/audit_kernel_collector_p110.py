from __future__ import annotations

import os
import sys
import time
from datetime import datetime

from gpu_fault.store import PostgresStore
from gpu_fault.telemetry import EvidenceKind


def _time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def main() -> None:
    marker = sys.argv[1]
    cluster_id = sys.argv[2]
    node_id = sys.argv[3]
    store = PostgresStore(
        os.environ["GPU_FAULT_STORE_URL"],
        pool_min_size=1,
        pool_max_size=2,
        pool_timeout_seconds=5,
    )
    try:
        matching = []
        for _ in range(20):
            evidence = store.list_raw_evidence(
                cluster_id,
                node_id=node_id,
                kind=EvidenceKind.NVIDIA_KERNEL,
                limit=100,
            )
            matching = [
                item
                for item in evidence
                if marker in str(item.payload.get("message") or "")
            ]
            if len(matching) >= 3:
                break
            time.sleep(1)

        assert len(matching) == 3, [item.record_id for item in matching]
        record_ids = {item.record_id for item in matching}
        assert len(record_ids) == 3, record_ids

        delays = []
        for item in matching:
            collected_at = _time(str(item.payload["collected_at"]))
            delay = (collected_at - item.observed_at).total_seconds()
            assert -5 <= delay <= 5, delay
            delays.append(round(delay, 3))

        print("marker", marker)
        print("record_ids", sorted(record_ids))
        print("collection_delays_seconds", sorted(delays))
    finally:
        store.close()


if __name__ == "__main__":
    main()
