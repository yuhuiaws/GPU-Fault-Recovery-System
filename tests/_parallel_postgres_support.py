from __future__ import annotations

import copy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gpu_fault.admin.atomic_json import write_json_atomic
from scripts.postgres_shard_receipts import ShardReceipt, validate_shard_receipt
from tools.pytest_result_identity import PASSED_PHASES, partition_for_nodeid

IDENTITY = "a" * 64
STARTED = datetime(2026, 9, 1, tzinfo=timezone.utc)
TARGETS = ("tests/test_first.py", "tests/test_second.py")
DISCOVERED = [
    f"{target}::test_variant[{index}]" for target in TARGETS for index in range(24)
]


def receipt_payload(index: int, workers: int = 4) -> dict[str, Any]:
    collected = sorted(
        nodeid
        for nodeid in DISCOVERED
        if partition_for_nodeid(nodeid, workers) == index
    )
    return {
        "schema_version": 1,
        "source_identity": IDENTITY,
        "session": {
            "source_identity": IDENTITY,
            "started_at": "2026-09-01T00:00:01+00:00",
            "finished_at": "2026-09-01T00:00:02+00:00",
            "exitstatus": 0,
            "collected_nodeids": collected,
            "discovered_nodeids": sorted(DISCOVERED),
            "collected_files": list(TARGETS),
            "collection_errors": [],
            "collection_skips": [],
            "selection": {
                "targets": list(TARGETS),
                "partition": [workers, index],
                "keyword": "",
                "markexpr": "",
                "deselect": [],
                "numprocesses": 0,
                "requested_numprocesses": 0,
                "stress_workers": "8",
                "stress_rounds": "40",
            },
        },
        "records": {
            nodeid: {
                "status": "PASS",
                "phases": dict(PASSED_PHASES),
                "duration_seconds": 0.25,
                "output": "",
            }
            for nodeid in collected
        },
    }


def checked_receipt(
    directory: Path, value: dict[str, Any], *, index: int = 0, workers: int = 4
) -> ShardReceipt:
    path = directory / f"receipt-{index}.json"
    write_json_atomic(path, copy.deepcopy(value))
    return validate_shard_receipt(
        path,
        root=directory / "repository",
        identity=IDENTITY,
        tests=TARGETS,
        workers=workers,
        index=index,
        stress=("8", "40"),
        started_after=STARTED,
    )
