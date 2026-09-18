"""Structured child evidence for mocked supervised pytest commands only."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tools.pytest_case_reporter import REPORT_ENV, REPORT_SCHEMA_VERSION
from tools.pytest_result_identity import source_identity


def write_focused_receipt(
    command: Sequence[str],
    *,
    environment: Mapping[str, str],
    cwd: Path | None = None,
    nodeids: Sequence[str] | None = None,
    returncode: int = 0,
    defect: str = "complete",
) -> dict[str, Any] | None:
    path = Path(environment[REPORT_ENV])
    assert "-p" in command and "tools.pytest_case_reporter" in command, (
        "the real public command boundary must install its receipt reporter"
    )
    assert not path.exists(), "a mocked child must not reuse an earlier receipt"
    if defect == "missing":
        return None
    selected = (
        list(nodeids)
        if nodeids is not None
        else [value for value in command if "::" in value]
    )
    assert selected and len(selected) == len(set(selected)), (
        "a generic fake selection needs explicit, distinct discovered leaf nodeids"
    )
    identity = source_identity(cwd or Path.cwd())
    timestamp = datetime.now(UTC).isoformat()
    records: dict[str, Any] = {
        nodeid: {
            "status": "PASS",
            "phases": {"setup": "passed", "call": "passed", "teardown": "passed"},
            "duration_seconds": 0.0,
            "output": "",
        }
        for nodeid in selected
    }
    value: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "source_identity": identity,
        "session": {
            "source_identity": identity,
            "started_at": timestamp,
            "finished_at": timestamp,
            "exitstatus": returncode,
            "collected_nodeids": list(selected),
            "discovered_nodeids": list(selected),
            "collection_errors": [],
            "collection_skips": [],
        },
        "records": records,
    }
    if returncode or defect == "failed":
        records[selected[0]]["status"] = "FAIL"
        records[selected[0]]["phases"]["call"] = "failed"
    if defect == "missing-phase":
        del records[selected[0]]["phases"]["teardown"]
    elif defect == "unexecuted-discovery":
        value["session"]["discovered_nodeids"].append(selected[0] + "[not-executed]")
    elif defect == "foreign-source":
        value["source_identity"] = "0" * 64
    else:
        assert defect in {"complete", "failed"}, "unknown mock receipt defect"
    path.write_text(json.dumps(value), encoding="utf-8")
    return value
