from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import lru_cache
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping

import yaml

from gpu_fault.policy import load_xid_policy

if __package__:
    from .case_scheduler import (
        ENVIRONMENT_MODES,
        FAILURE_SCOPES,
        LOCK_MODES,
        ExecutionPolicy,
        ResourceLock,
        run_scheduled_cases,
    )
    from .pytest_result_identity import source_identity
else:
    from case_scheduler import (
        ENVIRONMENT_MODES,
        FAILURE_SCOPES,
        LOCK_MODES,
        ExecutionPolicy,
        ResourceLock,
        run_scheduled_cases,
    )
    from pytest_result_identity import source_identity


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = ROOT / "testcases" / "fault-scenarios.yaml"
DEFAULT_REPORT_DIR = ROOT / "artifacts" / "fault"
REQUIRED_FIELDS = {
    "id",
    "title",
    "category",
    "level",
    "risk",
    "problem",
    "injection",
    "expected",
    "automation",
}
OPTIONAL_CASE_FIELDS = {
    "capture_processing_trace",
    "command",
    "evidence",
    "execution",
    "gate",
    "manifest",
    "operator_case",
    "procedure",
    "pytest_nodeid",
    "related_pytest",
    "superseded_by",
}
AUTOMATION_TYPES = {"pytest", "command", "manual"}
CURRENT_STATUS_VALUES = {
    "PASS",
    "NOT_RUN",
    "BLOCKED",
    "SUPERSEDED",
}
LEGACY_EVIDENCE_FIELDS = {
    "current_status",
    "result",
    "last_result",
    "live_result",
    "report",
    "incidental_findings",
    "validation_limitations",
    "derived_requirements",
    "defects_fixed",
}
EVIDENCE_NOTE_FIELDS = {
    "defects_fixed",
    "derived_requirements",
    "incidental_findings",
    "last_result",
    "live_result",
    "result",
    "validation_limitations",
}
# 风险等级词表。文档 `docs/区域模式端到端验收测试用例.md` §2.2 的
# 「等级」与本目录的 `risk:` 共用这一份取值，双向由
# tests/test_fault_scenario_catalog.py 守卫；新增取值必须先进 §2.2 的表。
# `destructive-provider-replace` 不在其中：该动作在本方案中不存在，
# 只作为否证目标出现，不能作为某条用例自身的等级。
RISK_VALUES = {
    "non-destructive",
    "read-only-signal-replay",
    "live-non-destructive",
    "live-kernel-log-injection",
    "live-isolation",
    "live-node-mutation",
    "live-service-action",
    "live-workload-restart",
    "destructive",
    "destructive-warm-spare",
    "destructive-provider-reboot",
}
LEVEL_VALUES = {
    "unit",
    "component",
    "integration",
    "staging",
    "live",
    "end-to-end",
}
NONEMPTY_STRING_FIELDS = {
    "id",
    "title",
    "category",
    "level",
    "risk",
    "problem",
    "injection",
    "automation",
}
TRACE_ENV = "GPU_FAULT_EMIT_PROCESSING_TRACE"
TRACE_PREFIX = "GPU_FAULT_PROCESSING_TRACE="
PYTEST_BATCH_REPORT_ENV = "PYTEST_GPU_FAULT_CASE_REPORT"
PYTEST_RESULT_SCHEMA_VERSION = 1
EXECUTION_FIELDS = {
    "depends_on",
    "environment",
    "failure_scope",
    "locks",
    "parallel_safe",
}
LOCK_FIELDS = {"mode", "resource"}
ISOLATED_ENVIRONMENT_NAMES = (
    "HOME",
    "PATH",
    "PYTHONPATH",
    "PYTHONPYCACHEPREFIX",
    "TMPDIR",
    "TMP",
    "TEMP",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TZ",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
)


class _UniqueKeyLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        self.flatten_mapping(node)
        mapping = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in mapping:
                raise ValueError(f"duplicate YAML key: {key}")
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


def _validate_nonempty_string(value: Any, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"test case field {field} must be a non-empty string")


def _validate_evidence(case: dict[str, Any]) -> str | None:
    legacy = LEGACY_EVIDENCE_FIELDS.intersection(case)
    if legacy:
        raise ValueError(
            "legacy evidence fields must be nested under evidence: "
            + ", ".join(sorted(legacy))
        )
    evidence = case.get("evidence")
    if evidence is None:
        return None
    if not isinstance(evidence, dict):
        raise ValueError("test case evidence must be a mapping")
    allowed = {"verdict"}
    unknown = set(evidence) - allowed
    if unknown:
        raise ValueError(f"test case evidence has unknown fields: {sorted(unknown)}")
    verdict = evidence.get("verdict")
    _validate_nonempty_string(verdict, "evidence.verdict")
    if verdict not in CURRENT_STATUS_VALUES:
        raise ValueError(f"unsupported test case evidence verdict: {verdict}")
    return verdict


def _validate_execution(case: dict[str, Any]) -> None:
    execution = case.get("execution")
    if execution is None:
        return
    if not isinstance(execution, dict):
        raise ValueError("test case execution must be a mapping")
    unknown = set(execution) - EXECUTION_FIELDS
    if unknown:
        raise ValueError(f"test case execution has unknown fields: {sorted(unknown)}")
    parallel_safe = execution.get("parallel_safe")
    if not isinstance(parallel_safe, bool):
        raise ValueError("test case execution.parallel_safe must be a boolean")
    failure_scope = execution.get("failure_scope", "branch")
    if failure_scope not in FAILURE_SCOPES:
        raise ValueError(f"unsupported test case failure scope: {failure_scope}")
    environment = execution.get("environment", "inherit")
    if environment not in ENVIRONMENT_MODES:
        raise ValueError(f"unsupported test case execution environment: {environment}")
    depends_on = execution.get("depends_on", [])
    if not isinstance(depends_on, list) or any(
        not isinstance(item, str) or not item for item in depends_on
    ):
        raise ValueError("test case execution.depends_on must be a list of case IDs")
    if case["id"] in depends_on:
        raise ValueError("test case cannot depend on itself")
    locks = execution.get("locks", [])
    if not isinstance(locks, list):
        raise ValueError("test case execution.locks must be a list")
    if parallel_safe and not locks:
        raise ValueError("parallel-safe test case execution requires resource locks")
    resources = set()
    for lock in locks:
        if not isinstance(lock, dict):
            raise ValueError("test case resource lock must be a mapping")
        lock_unknown = set(lock) - LOCK_FIELDS
        if lock_unknown:
            raise ValueError(
                f"test case resource lock has unknown fields: {sorted(lock_unknown)}"
            )
        resource = lock.get("resource")
        mode = lock.get("mode")
        _validate_nonempty_string(resource, "execution.locks.resource")
        if mode not in LOCK_MODES:
            raise ValueError(f"unsupported test case resource lock mode: {mode}")
        if resource in resources:
            raise ValueError(f"duplicate test case resource lock: {resource}")
        resources.add(resource)


def _validate_case(case: dict[str, Any]) -> None:
    unknown_fields = set(case) - REQUIRED_FIELDS - OPTIONAL_CASE_FIELDS
    if unknown_fields:
        raise ValueError(f"test case has unknown fields: {sorted(unknown_fields)}")
    missing = REQUIRED_FIELDS.difference(case)
    if missing:
        raise ValueError(f"test case is missing fields: {sorted(missing)}")
    for field in NONEMPTY_STRING_FIELDS:
        _validate_nonempty_string(case[field], field)
    expected = case["expected"]
    if (
        not isinstance(expected, list)
        or not expected
        or any(not isinstance(item, str) or not item.strip() for item in expected)
    ):
        raise ValueError("test case field expected must be a non-empty list of strings")
    risk = case["risk"]
    if risk not in RISK_VALUES:
        raise ValueError(f"unsupported test case risk: {risk}")
    level = case["level"]
    if level not in LEVEL_VALUES:
        raise ValueError(f"unsupported test case level: {level}")
    automation = case["automation"]
    if automation not in AUTOMATION_TYPES:
        raise ValueError(f"unsupported test case automation: {automation}")
    if automation == "pytest":
        nodeid = case.get("pytest_nodeid")
        _validate_nonempty_string(nodeid, "pytest_nodeid")
        if "::" not in nodeid:
            raise ValueError("pytest_nodeid must include a test function")
    if automation == "command":
        command = case.get("command")
        if (
            not isinstance(command, list)
            or not command
            or any(not isinstance(item, str) or not item for item in command)
        ):
            raise ValueError("command automation requires a non-empty argument list")
    if automation == "manual":
        procedure = case.get("procedure")
        if isinstance(procedure, str):
            _validate_nonempty_string(procedure, "procedure")
        elif (
            not isinstance(procedure, list)
            or not procedure
            or any(not isinstance(item, str) or not item.strip() for item in procedure)
        ):
            raise ValueError(
                "manual automation requires a procedure string "
                "or non-empty list of strings"
            )
    if "related_pytest" in case:
        related = case["related_pytest"]
        _validate_nonempty_string(related, "related_pytest")
        if "::" not in related:
            raise ValueError("related_pytest must include a test function")
    if "capture_processing_trace" in case and not isinstance(
        case["capture_processing_trace"], bool
    ):
        raise ValueError("capture_processing_trace must be a boolean")
    for field in ("manifest", "gate"):
        if field not in case:
            continue
        _validate_nonempty_string(case[field], field)
        if not (ROOT / case[field]).is_file():
            raise ValueError(f"test case {field} does not exist: {case[field]}")
    if "operator_case" in case:
        _validate_nonempty_string(case["operator_case"], "operator_case")
    _validate_execution(case)
    current_status = _validate_evidence(case)
    superseded_by = case.get("superseded_by")
    if current_status == "SUPERSEDED":
        if automation != "manual":
            raise ValueError("SUPERSEDED test cases must use manual automation")
        _validate_nonempty_string(superseded_by, "superseded_by")
        if superseded_by == case["id"]:
            raise ValueError("superseded_by must reference another case")
    elif superseded_by is not None:
        raise ValueError("superseded_by requires evidence.verdict=SUPERSEDED")


@lru_cache(maxsize=1)
def _xid_catalog_values() -> tuple[int, ...]:
    values = tuple(rule.xid for rule in load_xid_policy().catalog_rules)
    if len(values) != len(set(values)):
        raise ValueError("pinned XID catalog contains duplicate rule IDs")
    return values


def _expand_generated_families(
    value: dict[str, Any],
) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    families = value.get("generated_test_families", [])
    if not isinstance(families, list):
        raise ValueError("generated_test_families must be a list")
    for family in families:
        if not isinstance(family, dict):
            raise ValueError("every generated family must be a mapping")
        generator = family.get("generator")
        try:
            template = family["template"]
        except KeyError as exc:
            raise ValueError("generated family requires a template") from exc
        if not isinstance(template, dict):
            raise ValueError("generated family template must be a mapping")
        if generator == "integer_range":
            try:
                start = int(family["start"])
                end = int(family["end"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("integer_range family requires start and end") from exc
            if start > end:
                raise ValueError("generated family range is invalid")
            numbers = range(start, end + 1)
        elif generator == "xid_catalog_rules":
            if "start" in family or "end" in family:
                raise ValueError(
                    "xid_catalog_rules derives values and forbids start/end"
                )
            numbers = _xid_catalog_values()
        else:
            raise ValueError(
                "generated family generator must be integer_range or xid_catalog_rules"
            )
        for number in numbers:
            case: dict[str, Any] = {}
            for key, item in template.items():
                if isinstance(item, str):
                    case[key] = item.format(value=number)
                elif isinstance(item, list):
                    case[key] = [
                        (
                            entry.format(value=number)
                            if isinstance(entry, str)
                            else entry
                        )
                        for entry in item
                    ]
                else:
                    case[key] = item
            expanded.append(case)
    return expanded


def load_catalog(path: Path) -> list[dict[str, Any]]:
    value = yaml.load(
        path.read_text(encoding="utf-8"),
        Loader=_UniqueKeyLoader,
    )
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("fault scenario catalog schema_version must be 1")
    cases = value.get("test_cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("fault scenario catalog has no test_cases")
    cases = [*cases, *_expand_generated_families(value)]
    seen: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("every test case must be a mapping")
        _validate_case(case)
        case_id = case["id"]
        if case_id in seen:
            raise ValueError(f"duplicate test case id: {case_id}")
        seen.add(case_id)
    for case in cases:
        evidence = case.get("evidence") or {}
        if evidence.get("verdict") == "SUPERSEDED":
            if case["superseded_by"] not in seen:
                raise ValueError(
                    f"superseded_by references unknown case: {case['superseded_by']}"
                )
        execution = case.get("execution") or {}
        unknown_dependencies = set(execution.get("depends_on", ())) - seen
        if unknown_dependencies:
            raise ValueError(
                "test case execution depends_on references unknown cases: "
                + ", ".join(sorted(unknown_dependencies))
            )
    return cases


def select_cases(
    cases: list[dict[str, Any]],
    *,
    case_ids: Iterable[str],
    categories: set[str],
    levels: set[str],
    include_manual: bool,
    include_live: bool,
    also_case_ids: Iterable[str] = (),
) -> list[dict[str, Any]]:
    requested_ids = list(case_ids)
    requested = set(requested_ids)
    additional_ids = list(also_case_ids)
    additional = set(additional_ids)
    selected = []
    for case in cases:
        if requested and case["id"] not in requested:
            continue
        if categories and case["category"] not in categories:
            continue
        if levels and case["level"] not in levels:
            continue
        if case["automation"] == "manual" and not include_manual:
            continue
        if case["automation"] == "command":
            if not include_live:
                continue
            if case["id"] not in requested:
                continue
        selected.append(case)
    by_id = {case["id"]: case for case in cases}
    unknown = (requested | additional).difference(by_id)
    if unknown:
        raise ValueError(f"unknown test case IDs: {sorted(unknown)}")
    if requested_ids and not isinstance(case_ids, (set, frozenset)):
        position = {case_id: index for index, case_id in enumerate(requested_ids)}
        selected.sort(key=lambda case: position[case["id"]])
    for case_id in additional_ids:
        case = by_id[case_id]
        if case["automation"] != "command":
            raise ValueError("--also-case only accepts command automation cases")
        if not include_live:
            raise ValueError("--also-case requires --include-live")
        if case_id not in {item["id"] for item in selected}:
            selected.append(case)
    return selected


def _extract_processing_trace(
    output: str,
) -> tuple[dict[str, Any] | None, str]:
    trace = None
    retained = []
    for line in output.splitlines():
        if not line.startswith(TRACE_PREFIX):
            retained.append(line)
            continue
        if trace is not None:
            raise ValueError("test emitted multiple processing traces")
        value = json.loads(line[len(TRACE_PREFIX) :])
        if not isinstance(value, dict):
            raise ValueError("processing trace must be a JSON object")
        trace = value
    return trace, "\n".join(retained).strip()


def build_isolated_environment(
    source: Mapping[str, str] | None = None,
) -> dict[str, str]:
    source_environment = os.environ if source is None else source
    environment = {
        name: value
        for name in ISOLATED_ENVIRONMENT_NAMES
        if (value := source_environment.get(name)) is not None
    }
    environment.setdefault("PATH", os.defpath)
    environment["GPU_FAULT_STORE_URL"] = ""
    environment["GPU_FAULT_TEST_POSTGRES_URL"] = ""
    return environment


def run_case(
    case: dict[str, Any],
    *,
    environment: Mapping[str, str] | None = None,
    environment_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    start = time.monotonic()
    if case["automation"] == "manual":
        evidence = case.get("evidence") or {}
        result = {
            "id": case["id"],
            "title": case["title"],
            "category": case["category"],
            "level": case["level"],
            "risk": case["risk"],
            "problem": case["problem"],
            "injection": case["injection"],
            "expected": case["expected"],
            "status": "NOT_RUN",
            "reason": (
                f"superseded by {case['superseded_by']}"
                if evidence.get("verdict") == "SUPERSEDED"
                else ("manual test case is not executed by the runner")
            ),
            "procedure": case["procedure"],
            "started_at": started.isoformat(),
            "duration_seconds": 0.0,
        }
        for field in (
            "evidence",
            "superseded_by",
            "operator_case",
        ):
            if field in case:
                result[field] = case[field]
        return result
    if case["automation"] == "pytest":
        command = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
        ]
        if case.get("capture_processing_trace"):
            command.append("-s")
        command.append(str(case["pytest_nodeid"]))
    else:
        command = [str(item) for item in case["command"]]
    child_environment = dict(os.environ if environment is None else environment)
    if environment_overrides:
        child_environment.update(environment_overrides)
    if case.get("capture_processing_trace"):
        child_environment[TRACE_ENV] = "1"
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=child_environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    trace = None
    output = completed.stdout.strip()
    trace_error = None
    try:
        trace, output = _extract_processing_trace(output)
    except (json.JSONDecodeError, ValueError) as exc:
        trace_error = str(exc)
    trace_required = bool(case.get("capture_processing_trace"))
    passed = (
        completed.returncode == 0
        and trace_error is None
        and (not trace_required or trace is not None)
    )
    result = {
        "id": case["id"],
        "title": case["title"],
        "category": case["category"],
        "level": case["level"],
        "risk": case["risk"],
        "problem": case["problem"],
        "injection": case["injection"],
        "expected": case["expected"],
        "status": "PASS" if passed else "FAIL",
        "executor": (
            case["pytest_nodeid"] if case["automation"] == "pytest" else case["command"]
        ),
        "started_at": started.isoformat(),
        "duration_seconds": round(time.monotonic() - start, 3),
        "output": output,
    }
    if trace is not None:
        result["processing_trace"] = trace
    elif trace_required:
        result["trace_error"] = (
            trace_error or "required processing trace was not emitted"
        )
    return result


def pytest_batch_eligible(
    case: dict[str, Any],
    policy: ExecutionPolicy,
) -> bool:
    return (
        case["automation"] == "pytest"
        and not case.get("capture_processing_trace")
        and policy.parallel_safe
        and policy.environment == "isolated"
        and not policy.depends_on
        and policy.failure_scope == "branch"
        and all(lock.mode == "shared" for lock in policy.effective_locks())
    )


def _normalized_pytest_nodeid(value: str) -> str:
    path_text, separator, selector = value.partition("::")
    path = Path(path_text)
    resolved = (path if path.is_absolute() else ROOT / path).resolve()
    return str(resolved) + (f"::{selector}" if separator else "")


def _results_from_pytest_records(
    requested: Mapping[str, dict[str, Any]],
    *,
    normalized_records: Mapping[str, object],
    started: datetime,
    batch_output: str,
) -> dict[str, dict[str, Any]]:
    normalized = {
        _normalized_pytest_nodeid(nodeid): case for nodeid, case in requested.items()
    }
    if len(normalized) != len(requested):
        raise ValueError("pytest cases must use unique nodeids")
    results: dict[str, dict[str, Any]] = {}
    for nodeid, case in requested.items():
        normalized_nodeid = _normalized_pytest_nodeid(nodeid)
        exact = normalized_records.get(normalized_nodeid)
        records_for_case = (
            [exact]
            if isinstance(exact, dict)
            else [
                record
                for reported_nodeid, record in normalized_records.items()
                if reported_nodeid.startswith(normalized_nodeid + "[")
                and isinstance(record, dict)
            ]
        )
        if records_for_case:
            status = (
                "PASS"
                if all(record.get("status") == "PASS" for record in records_for_case)
                else "FAIL"
            )
            output = "\n".join(
                str(record.get("output") or "").strip()
                for record in records_for_case
                if str(record.get("output") or "").strip()
            )
            duration = round(
                sum(
                    float(record.get("duration_seconds") or 0.0)
                    for record in records_for_case
                ),
                3,
            )
        else:
            status = "FAIL"
            output = (
                f"pytest produced no result for {nodeid}; "
                f"available={sorted(normalized_records)}"
                + (f"\n{batch_output}" if batch_output else "")
            )
            duration = 0.0
        results[str(case["id"])] = {
            "id": case["id"],
            "title": case["title"],
            "category": case["category"],
            "level": case["level"],
            "risk": case["risk"],
            "problem": case["problem"],
            "injection": case["injection"],
            "expected": case["expected"],
            "status": status,
            "executor": nodeid,
            "started_at": started.isoformat(),
            "duration_seconds": duration,
            "output": output,
        }
    return results


def run_pytest_batch(
    cases: Iterable[dict[str, Any]],
    *,
    environment: Mapping[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    selected = tuple(cases)
    if not selected:
        return {}
    started = datetime.now(timezone.utc)
    requested = {str(case["pytest_nodeid"]): case for case in selected}
    if len(requested) != len(selected):
        raise ValueError("batched pytest cases must use unique nodeids")
    with tempfile.TemporaryDirectory(prefix="gpu-fault-pytest-batch-") as directory:
        report_path = Path(directory) / "report.json"
        child_environment = dict(os.environ if environment is None else environment)
        child_environment[PYTEST_BATCH_REPORT_ENV] = str(report_path)
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-p",
                "tools.pytest_case_reporter",
                *requested,
            ],
            cwd=ROOT,
            env=child_environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        try:
            records = _load_pytest_result_records(report_path)
        except ValueError:
            records = {}
    batch_output = (completed.stdout or "").strip()
    normalized_records = (
        {
            _normalized_pytest_nodeid(str(nodeid)): record
            for nodeid, record in records.items()
        }
        if isinstance(records, dict)
        else {}
    )
    return _results_from_pytest_records(
        requested,
        normalized_records=normalized_records,
        started=started,
        batch_output=batch_output,
    )


def load_pytest_results(
    cases: Iterable[dict[str, Any]],
    *,
    path: Path,
) -> dict[str, dict[str, Any]]:
    records = _load_pytest_result_records(path)
    selected = tuple(cases)
    started = datetime.now(timezone.utc)
    requested = {str(case["pytest_nodeid"]): case for case in selected}
    normalized_records = {
        _normalized_pytest_nodeid(str(nodeid)): record
        for nodeid, record in records.items()
    }
    return _results_from_pytest_records(
        requested,
        normalized_records=normalized_records,
        started=started,
        batch_output="",
    )


def _load_pytest_result_records(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"pytest case results are invalid: {path}") from exc
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != PYTEST_RESULT_SCHEMA_VERSION
        or value.get("source_identity") != source_identity(ROOT)
        or not isinstance(value.get("records"), dict)
    ):
        raise ValueError("pytest case result identity does not match current source")
    return dict(value["records"])


def execution_policy(case: dict[str, Any]) -> ExecutionPolicy:
    configured = case.get("execution")
    if configured is not None:
        return ExecutionPolicy(
            parallel_safe=bool(configured["parallel_safe"]),
            locks=tuple(
                ResourceLock(
                    resource=str(lock["resource"]),
                    mode=str(lock["mode"]),
                )
                for lock in configured.get("locks", ())
            ),
            depends_on=tuple(str(item) for item in configured.get("depends_on", ())),
            failure_scope=str(configured.get("failure_scope", "branch")),
            environment=str(configured.get("environment", "inherit")),
        )
    if case["automation"] == "pytest" and case["risk"] == "non-destructive":
        return ExecutionPolicy(
            parallel_safe=True,
            locks=(ResourceLock("local-test", "shared"),),
            environment="isolated",
        )
    return ExecutionPolicy(parallel_safe=False, failure_scope="global")


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Run recorded GPU fault simulation test cases and emit JSON.")
    )
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        dest="case_ids",
    )
    parser.add_argument(
        "--also-case",
        action="append",
        default=[],
        dest="also_case_ids",
        help=(
            "Add an explicitly selected command case to the default or "
            "filtered pytest selection. Requires --include-live."
        ),
    )
    parser.add_argument("--category", action="append", default=[])
    parser.add_argument("--level", action="append", default=[])
    parser.add_argument(
        "--include-manual",
        action="store_true",
        help="Include destructive manual cases as NOT_RUN records.",
    )
    parser.add_argument(
        "--include-live",
        action="store_true",
        help="Allow explicitly selected live non-destructive cases.",
    )
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--workers",
        type=_positive_integer,
        default=1,
        help=(
            "Run explicitly parallel-safe cases concurrently. "
            "Defaults to 1 and preserves serial execution."
        ),
    )
    parser.add_argument(
        "--batch-pytest",
        action="store_true",
        help=(
            "Run a compatible all-pytest selection in one pytest process while "
            "retaining per-case results."
        ),
    )
    parser.add_argument(
        "--pytest-results",
        type=Path,
        help="Reuse per-node pytest results emitted by tools.pytest_case_reporter.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cases = load_catalog(args.catalog)
    selected = select_cases(
        cases,
        case_ids=args.case_ids,
        categories=set(args.category),
        levels=set(args.level),
        include_manual=args.include_manual,
        include_live=args.include_live,
        also_case_ids=args.also_case_ids,
    )
    if args.list:
        for case in selected:
            print(f"{case['id']}\t{case['level']}\t{case['risk']}\t{case['title']}")
        return 0
    if not selected:
        print("ERROR: no test cases selected", file=sys.stderr)
        return 2

    run_started = time.monotonic()
    policies = {str(case["id"]): execution_policy(case) for case in selected}
    if args.batch_pytest and args.pytest_results is not None:
        print(
            "ERROR: --batch-pytest and --pytest-results are mutually exclusive",
            file=sys.stderr,
        )
        return 2
    batch_enabled = bool(args.batch_pytest or args.pytest_results is not None)
    if batch_enabled:
        incompatible = [
            str(case["id"])
            for case in selected
            if not pytest_batch_eligible(case, policies[str(case["id"])])
        ]
        if incompatible:
            print(
                "ERROR: --batch-pytest selection contains incompatible cases: "
                + ", ".join(incompatible),
                file=sys.stderr,
            )
            return 2

    def execute(case: dict[str, Any], policy: ExecutionPolicy) -> dict[str, Any]:
        environment = (
            build_isolated_environment() if policy.environment == "isolated" else None
        )
        return run_case(case, environment=environment)

    def execute_batch(
        cases: Iterable[dict[str, Any]],
        policies: Iterable[ExecutionPolicy],
    ) -> dict[str, dict[str, Any]]:
        case_values = tuple(cases)
        policy_values = tuple(policies)
        if args.pytest_results is not None:
            return load_pytest_results(
                case_values,
                path=args.pytest_results.resolve(),
            )
        environment = (
            build_isolated_environment()
            if policy_values and policy_values[0].environment == "isolated"
            else None
        )
        return run_pytest_batch(case_values, environment=environment)

    def on_start(case: dict[str, Any], _policy: ExecutionPolicy) -> None:
        print(f"RUN  {case['id']} {case['title']}", flush=True)

    def on_finish(case: dict[str, Any], result: dict[str, Any]) -> None:
        print(f"{result['status']:7} {case['id']}", flush=True)

    results = run_scheduled_cases(
        selected,
        policy_for=execution_policy,
        execute=execute,
        batch_key_for=(
            (lambda _case, _policy: "isolated-pytest") if batch_enabled else None
        ),
        execute_batch=execute_batch if batch_enabled else None,
        max_batch_size=len(selected) if batch_enabled else 1,
        max_workers=max(args.workers, len(selected)) if batch_enabled else args.workers,
        on_start=on_start,
        on_finish=on_finish,
    )

    counts = {
        status: sum(item["status"] == status for item in results)
        for status in ("PASS", "FAIL", "BLOCKED", "NOT_RUN")
    }
    executed_at = datetime.now(timezone.utc).isoformat()
    verdict = (
        "FAIL"
        if counts["FAIL"] or counts["BLOCKED"]
        else "NOT_RUN"
        if counts["PASS"] == 0
        else "PASS_WITH_LIMITATIONS"
        if counts["NOT_RUN"]
        else "PASS"
    )
    report = {
        "schema_version": 2,
        "report_type": "fault-test-run",
        "executed_at": executed_at,
        "verdict": verdict,
        "limitations": (
            [
                "Report contains BLOCKED or NOT_RUN cases and is not "
                "complete acceptance evidence."
            ]
            if counts["BLOCKED"] or counts["NOT_RUN"]
            else []
        ),
        "catalog": str(args.catalog),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "summary": {
            "total": len(results),
            **{key.lower(): value for key, value in counts.items()},
            "workers": args.workers,
            "wall_duration_seconds": round(time.monotonic() - run_started, 3),
        },
        "results": results,
    }
    report_path = args.report
    if report_path is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        report_path = DEFAULT_REPORT_DIR / f"fault-tests-{stamp}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"REPORT {report_path}")
    print(
        "SUMMARY "
        f"pass={counts['PASS']} fail={counts['FAIL']} "
        f"blocked={counts['BLOCKED']} not_run={counts['NOT_RUN']}"
    )
    return 1 if counts["FAIL"] or counts["BLOCKED"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
