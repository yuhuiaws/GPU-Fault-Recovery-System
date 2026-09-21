from __future__ import annotations

import json
import os
from collections.abc import Generator, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import pytest

from tools.pytest_result_identity import (
    CI_CONTEXT_ENV,
    partition_for_nodeid as partition_for_nodeid,
    repository_root,
    source_identity,
)

REPORT_ENV = "PYTEST_GPU_FAULT_CASE_REPORT"
# The git tree the receipt is bound to when pytest runs in a copy of it that
# carries no .git (BOOT-018 builds and tests in such copies, live 2026-09-20).
IDENTITY_ROOT_ENV = "PYTEST_GPU_FAULT_IDENTITY_ROOT"
PARTITION_COUNT_ENV = "PYTEST_GPU_FAULT_PARTITION_COUNT"
PARTITION_INDEX_ENV = "PYTEST_GPU_FAULT_PARTITION_INDEX"
REPORT_SCHEMA_VERSION = 1
REPORTS: dict[str, dict[str, Any]] = {}
COLLECTED: set[str] = set()
DISCOVERED: set[str] = set()
COLLECTION_ERRORS: set[str] = set()
SKIPPED_COLLECTORS: set[str] = set()
COLLECTED_FILES: set[str] = set()
STARTED: dict[str, Any] = {}
WORKER_DISCOVERY_KEY = "gpu_fault_pytest_discovery"
REQUESTED_NUMPROCESSES = pytest.StashKey[object]()
SOURCE_ROOT = Path.cwd().resolve()
COLLECTION_ROOT = SOURCE_ROOT
CONTROL_ENVIRONMENT: ContextVar[Mapping[str, str] | None] = ContextVar(
    "pytest_reporter_controls", default=None
)


@contextmanager
def bound_session_controls(environment: Mapping[str, str]) -> Iterator[None]:
    """Keep serial session controls out of nested child environments."""
    if set(environment) != {REPORT_ENV, PARTITION_COUNT_ENV, PARTITION_INDEX_ENV}:
        raise ValueError("pytest session controls must bind the complete control set")
    if any(not isinstance(value, str) for value in environment.values()):
        raise ValueError("pytest session controls must contain string values")
    token = CONTROL_ENVIRONMENT.set(dict(environment))
    try:
        yield
    finally:
        CONTROL_ENVIRONMENT.reset(token)


def _control_value(name: str) -> str:
    environment = CONTROL_ENVIRONMENT.get()
    return (
        os.getenv(name, "") if environment is None else environment.get(name, "")
    ).strip()


@lru_cache(maxsize=None)
def _source_path(path_text: str) -> str:
    # Collection fixes the path before test fixtures can replace filesystem APIs.
    path = Path(path_text)
    actual = (path if path.is_absolute() else COLLECTION_ROOT / path).resolve()
    return Path(os.path.relpath(actual, SOURCE_ROOT)).as_posix()


def _source_nodeid(nodeid: str) -> str:
    path_text, separator, selector = nodeid.partition("::")
    return _source_path(path_text) + (f"::{selector}" if separator else "")


def _report_path() -> Path | None:
    value = _control_value(REPORT_ENV)
    return Path(value) if value else None


def _worker_report_path(path: Path, worker_id: str) -> Path:
    return path.with_name(f"{path.name}.worker-{worker_id}.json")


def _partition() -> tuple[int, int] | None:
    raw_count = _control_value(PARTITION_COUNT_ENV)
    raw_index = _control_value(PARTITION_INDEX_ENV)
    if not raw_count and not raw_index:
        return None
    if not raw_count.isdecimal() or not raw_index.isdecimal():
        raise RuntimeError("pytest partition count and index must be integers")
    count = int(raw_count)
    index = int(raw_index)
    if count < 1 or not 0 <= index < count:
        raise RuntimeError("pytest partition index is outside the configured count")
    return count, index


def _selection(config: Any) -> dict[str, Any]:
    option = getattr(config, "option", None)
    partition = _partition()
    selection = {
        "targets": sorted(
            _source_nodeid(str(arg)) for arg in getattr(config, "args", [])
        ),
        "partition": list(partition) if partition is not None else None,
        "keyword": getattr(option, "keyword", ""),
        "markexpr": getattr(option, "markexpr", ""),
        "deselect": list(getattr(option, "deselect", []) or []),
        "numprocesses": 0,
        "stress_workers": os.getenv("GPU_FAULT_POSTGRES_LOCK_STRESS_WORKERS", ""),
        "stress_rounds": os.getenv("GPU_FAULT_POSTGRES_LOCK_STRESS_ROUNDS", ""),
    }
    stash = getattr(config, "stash", {})
    if REQUESTED_NUMPROCESSES in stash:
        selection["requested_numprocesses"] = stash[REQUESTED_NUMPROCESSES]
    return selection


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


def _write_final_report(
    path: Path, records: dict[str, dict[str, Any]], *, exitstatus: int
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "source_identity": source_identity(SOURCE_ROOT),
        "session": {
            **STARTED,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "exitstatus": exitstatus,
            "collected_nodeids": sorted(COLLECTED),
            "discovered_nodeids": sorted(DISCOVERED),
            "collected_files": sorted(COLLECTED_FILES),
            "collection_errors": sorted(COLLECTION_ERRORS),
            "collection_skips": sorted(SKIPPED_COLLECTORS),
        },
        "records": records,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_cmdline_main(config: pytest.Config) -> Generator[None, Any, Any]:
    # xdist resolves automatic modes and applies worker caps before configure.
    requested = getattr(config.option, "numprocesses", None)
    config.stash[REQUESTED_NUMPROCESSES] = 0 if requested is None else requested
    return (yield)


def pytest_configure(config: Any) -> None:
    global SOURCE_ROOT, COLLECTION_ROOT
    SOURCE_ROOT = Path.cwd().resolve()
    COLLECTION_ROOT = Path(getattr(config, "rootpath", SOURCE_ROOT)).resolve()
    _source_path.cache_clear()
    REPORTS.clear()
    COLLECTED.clear()
    DISCOVERED.clear()
    COLLECTION_ERRORS.clear()
    SKIPPED_COLLECTORS.clear()
    COLLECTED_FILES.clear()
    STARTED.clear()
    _partition()
    path = _report_path()
    if path is None:
        return
    identity_root = os.getenv(IDENTITY_ROOT_ENV, "").strip()
    SOURCE_ROOT = repository_root(
        Path(identity_root).resolve() if identity_root else SOURCE_ROOT
    )
    if hasattr(config, "workerinput"):
        return
    STARTED.update(
        source_identity=source_identity(SOURCE_ROOT),
        started_at=datetime.now(timezone.utc).isoformat(),
        selection=_selection(config),
    )
    context = os.getenv(CI_CONTEXT_ENV)
    if context is not None:
        value = json.loads(context)
        if not isinstance(value, dict):
            raise RuntimeError("pytest CI context must be an object")
        STARTED["ci_context"] = value
    for worker_path in path.parent.glob(f"{path.name}.worker-*.json"):
        worker_path.unlink()


@pytest.hookimpl(optionalhook=True)
def pytest_xdist_setupnodes(config: Any, specs: Sequence[Any]) -> None:
    if STARTED and not hasattr(config, "workerinput"):
        STARTED["selection"]["numprocesses"] = len(specs)


def pytest_collection_modifyitems(config: Any, items: list[Any]) -> None:
    partition = _partition()
    if partition is None:
        return
    count, index = partition
    selected = [
        item
        for item in items
        if partition_for_nodeid(_source_nodeid(str(item.nodeid)), count) == index
    ]
    selected_ids = {id(item) for item in selected}
    deselected = [item for item in items if id(item) not in selected_ids]
    if deselected:
        config.hook.pytest_deselected(items=deselected)
    items[:] = selected


def pytest_collection_finish(session: Any) -> None:
    COLLECTED.update(_source_nodeid(str(item.nodeid)) for item in session.items)


def pytest_collectreport(report: Any) -> None:
    collector = _source_nodeid(str(report.nodeid))
    if collector.endswith(".py"):
        COLLECTED_FILES.add(collector)
    if report.failed:
        COLLECTION_ERRORS.add(_source_nodeid(str(report.nodeid)))
    if getattr(report, "skipped", False):
        SKIPPED_COLLECTORS.add(_source_nodeid(str(report.nodeid)))
    for item in report.result or ():
        if isinstance(item, pytest.Item):
            DISCOVERED.add(_source_nodeid(str(item.nodeid)))


@pytest.hookimpl(wrapper=True)
def pytest_make_collect_report(collector: Any) -> Generator[None, Any, Any]:
    report = yield
    # Explicit parameter selectors can prune the report before pytest_collectreport.
    pytest_collectreport(report)
    return report


@pytest.hookimpl(optionalhook=True)
def pytest_xdist_node_collection_finished(node: Any, ids: list[str]) -> None:
    COLLECTED.update(_source_nodeid(nodeid) for nodeid in ids)


@pytest.hookimpl(optionalhook=True)
def pytest_testnodedown(node: Any, error: object) -> None:
    if _report_path() is None:
        return
    output = getattr(node, "workeroutput", None)
    facts = output.get(WORKER_DISCOVERY_KEY) if isinstance(output, dict) else None
    if not isinstance(facts, dict):
        COLLECTION_ERRORS.add("xdist worker discovery is unavailable")
        return
    ids, errors, skips, files = (
        facts.get("nodeids"),
        facts.get("errors"),
        facts.get("skips", []),
        facts.get("files", []),
    )
    if (
        not isinstance(ids, list)
        or any(not isinstance(nodeid, str) or "::" not in nodeid for nodeid in ids)
        or not isinstance(errors, list)
        or any(not isinstance(message, str) for message in errors)
        or not isinstance(skips, list)
        or any(not isinstance(nodeid, str) for nodeid in skips)
        or not isinstance(files, list)
        or any(not isinstance(path, str) for path in files)
    ):
        COLLECTION_ERRORS.add("xdist worker discovery is invalid")
        return
    DISCOVERED.update(ids)
    COLLECTION_ERRORS.update(errors)
    SKIPPED_COLLECTORS.update(skips)
    COLLECTED_FILES.update(files)


def pytest_runtest_logreport(report: Any) -> None:
    record = REPORTS.setdefault(
        _source_nodeid(str(report.nodeid)),
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
        worker_output = getattr(session.config, "workeroutput", None)
        if isinstance(worker_output, dict):
            worker_output[WORKER_DISCOVERY_KEY] = {
                "nodeids": sorted(DISCOVERED),
                "errors": sorted(COLLECTION_ERRORS),
                "skips": sorted(SKIPPED_COLLECTORS),
                "files": sorted(COLLECTED_FILES),
            }
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
    _write_final_report(path, normalized, exitstatus=int(session.exitstatus))
