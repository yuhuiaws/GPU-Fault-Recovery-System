from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from tools.pytest_result_identity import source_identity


REPORT_ENV = "PYTEST_GPU_FAULT_CASE_REPORT"
REPORT_SCHEMA_VERSION = 1
REPORTS: dict[str, dict[str, Any]] = {}


def _report_path() -> Path | None:
    value = os.getenv(REPORT_ENV, "").strip()
    return Path(value) if value else None


def _worker_report_path(path: Path, worker_id: str) -> Path:
    return path.with_name(f"{path.name}.worker-{worker_id}.json")


def _write_records(path: Path, records: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def output_text(value: object) -> str:
        if isinstance(value, list):
            return "\n".join(str(item) for item in value if item)
        return str(value or "")

    normalized: dict[str, dict[str, Any]] = {
        nodeid: {
            **record,
            "output": output_text(record.get("output")),
        }
        for nodeid, record in sorted(records.items())
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(normalized, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_final_report(path: Path, records: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "source_identity": source_identity(Path.cwd()),
        "records": records,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def pytest_configure(config: Any) -> None:
    REPORTS.clear()
    path = _report_path()
    if path is None or hasattr(config, "workerinput"):
        return
    for worker_path in path.parent.glob(f"{path.name}.worker-*.json"):
        worker_path.unlink()


def pytest_runtest_logreport(report: Any) -> None:
    record = REPORTS.setdefault(
        str(report.nodeid),
        {
            "duration_seconds": 0.0,
            "output": [],
            "phases": {},
            "status": "PASS",
        },
    )
    record["duration_seconds"] = float(record["duration_seconds"]) + float(
        report.duration
    )
    record["phases"][str(report.when)] = str(report.outcome)
    for value in (report.capstdout, report.capstderr):
        if value:
            record["output"].append(str(value).rstrip())
    if report.failed or getattr(report, "skipped", False):
        record["status"] = "FAIL"
        detail = str(report.longrepr)
        if detail:
            record["output"].append(detail)


def pytest_sessionfinish(session: Any) -> None:
    path = _report_path()
    if path is None:
        return
    worker_input = getattr(session.config, "workerinput", None)
    if isinstance(worker_input, dict):
        worker_id = str(worker_input.get("workerid") or "unknown")
        _write_records(_worker_report_path(path, worker_id), REPORTS)
        return
    merged: dict[str, dict[str, Any]] = dict(REPORTS)
    for worker_path in sorted(path.parent.glob(f"{path.name}.worker-*.json")):
        value = json.loads(worker_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise RuntimeError(f"pytest worker report is invalid: {worker_path}")
        for nodeid, record in value.items():
            merged.setdefault(nodeid, record)
        worker_path.unlink()
    normalized_path = path.with_name(path.name + ".normalized")
    _write_records(normalized_path, merged)
    normalized = json.loads(normalized_path.read_text(encoding="utf-8"))
    normalized_path.unlink()
    if not isinstance(normalized, dict):
        raise RuntimeError("normalized pytest report is invalid")
    _write_final_report(path, normalized)
