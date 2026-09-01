"""Compile the regional acceptance catalog into a deterministic execution plan.

Reviewed overrides use this schema::

    schema_version: 1
    reviewed: true
    reviewed_by: acceptance-reviewer
    reviewed_at: 2026-08-28T12:00:00Z
    order_sha256: <sha256 of regional-execution-order.yaml>
    catalog_sha256: <sha256 of fault-scenarios.yaml>
    proposal_sha256: <optional sha256 of the reviewed proposal>
    cases:
      GF-REGIONAL-BOOT-013:
        executor: codex-manual
        depends_on:
          - GF-REGIONAL-BOOT-011
        locks:
          - resource: acceptance-evidence
            mode: shared

Overrides are additive. They cannot replace catalog or formal-order dependencies,
remove locks, select another executor, or make a retired case runnable.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import hashlib
import json
from pathlib import Path
import re
from typing import Any, cast

import yaml  # type: ignore[import-untyped]


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ORDER = ROOT / "testcases" / "regional-execution-order.yaml"
DEFAULT_CATALOG = ROOT / "testcases" / "fault-scenarios.yaml"
REGIONAL_CASE_ID = re.compile(r"^GF-REGIONAL-[A-Z0-9]+-\d{3}$")
SHA256_HEX = re.compile(r"^[0-9a-fA-F]{64}$")
RFC3339_TIMESTAMP = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
    r"(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$"
)


class PlanMode(str, Enum):
    FORMAL = "formal"
    LOCAL_PREACCEPTANCE = "local-preacceptance"
    COLLECT_ALL = "collect-all"


class AutomationKind(str, Enum):
    PYTEST = "pytest"
    COMMAND = "command"
    MANUAL = "manual"


class ExecutorKind(str, Enum):
    PYTEST = "pytest"
    COMMAND = "command"
    CODEX_MANUAL = "codex-manual"
    HUMAN = "human"
    DO_NOT_RUN = "do-not-run"


class LockMode(str, Enum):
    SHARED = "shared"
    EXCLUSIVE = "exclusive"


class FailureScope(str, Enum):
    CASE = "case"
    BRANCH = "branch"
    GLOBAL = "global"


class EnvironmentMode(str, Enum):
    INHERIT = "inherit"
    ISOLATED = "isolated"


LOCAL_PROXY_RISKS = frozenset(
    {
        "non-destructive",
        "read-only-signal-replay",
        "live-non-destructive",
    }
)


@dataclass(frozen=True, order=True)
class ResourceLock:
    resource: str
    mode: LockMode

    def as_dict(self) -> dict[str, str]:
        return {"resource": self.resource, "mode": self.mode.value}


@dataclass(frozen=True)
class ExecutionPolicy:
    parallel_safe: bool
    locks: tuple[ResourceLock, ...] = ()
    depends_on: tuple[str, ...] = ()
    failure_scope: FailureScope = FailureScope.GLOBAL
    environment: EnvironmentMode = EnvironmentMode.INHERIT
    configured_in_catalog: bool = False
    collect_all: bool = False
    read_only: bool = False
    repair_allowed: bool = False

    def __post_init__(self) -> None:
        if self.collect_all and self.failure_scope is not FailureScope.CASE:
            raise ValueError("collect-all execution requires failure_scope=case")
        if self.read_only and self.repair_allowed:
            raise ValueError("read-only execution cannot allow repair")

    def as_dict(self) -> dict[str, object]:
        return {
            "parallel_safe": self.parallel_safe,
            "locks": [lock.as_dict() for lock in self.locks],
            "depends_on": list(self.depends_on),
            "failure_scope": self.failure_scope.value,
            "environment": self.environment.value,
            "configured_in_catalog": self.configured_in_catalog,
            "collect_all": self.collect_all,
            "read_only": self.read_only,
            "repair_allowed": self.repair_allowed,
        }


@dataclass(frozen=True)
class DependencyNode:
    case_id: str
    depends_on: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {"case_id": self.case_id, "depends_on": list(self.depends_on)}


@dataclass(frozen=True)
class ReviewMetadata:
    reviewed_by: str
    reviewed_at: str
    order_sha256: str
    catalog_sha256: str
    proposal_sha256: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "reviewed_by": self.reviewed_by,
            "reviewed_at": self.reviewed_at,
            "order_sha256": self.order_sha256,
            "catalog_sha256": self.catalog_sha256,
            "proposal_sha256": self.proposal_sha256,
        }


@dataclass(frozen=True)
class RegionalAcceptanceCase:
    id: str
    ordinal: int
    phase_sequence: int | None
    phase_name: str | None
    maintenance_window: str | None
    title: str
    category: str
    level: str
    risk: str
    problem: str
    injection: str
    expected: tuple[str, ...]
    procedure: str | None
    automation: AutomationKind
    related_pytest: str | None
    gate: str | None
    catalog_execution: ExecutionPolicy
    executor: ExecutorKind
    pytest_nodeids: tuple[str, ...]
    command: tuple[str, ...]
    execution: ExecutionPolicy
    mandatory_depends_on: tuple[str, ...]
    override_depends_on: tuple[str, ...]
    local_proxy: bool
    blocked_reason: str | None
    do_not_run_reason: str | None

    @property
    def blocked(self) -> bool:
        return self.blocked_reason is not None

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "ordinal": self.ordinal,
            "phase_sequence": self.phase_sequence,
            "phase_name": self.phase_name,
            "maintenance_window": self.maintenance_window,
            "title": self.title,
            "category": self.category,
            "level": self.level,
            "risk": self.risk,
            "problem": self.problem,
            "injection": self.injection,
            "expected": list(self.expected),
            "procedure": self.procedure,
            "automation": self.automation.value,
            "related_pytest": self.related_pytest,
            "gate": self.gate,
            "catalog_execution": self.catalog_execution.as_dict(),
            "executor": self.executor.value,
            "pytest_nodeids": list(self.pytest_nodeids),
            "command": list(self.command),
            "execution": self.execution.as_dict(),
            "mandatory_depends_on": list(self.mandatory_depends_on),
            "override_depends_on": list(self.override_depends_on),
            "local_proxy": self.local_proxy,
            "blocked": self.blocked,
            "blocked_reason": self.blocked_reason,
            "do_not_run_reason": self.do_not_run_reason,
        }


@dataclass(frozen=True)
class RegionalAcceptancePlan:
    mode: PlanMode
    collect_all: bool
    read_only: bool
    repair_allowed: bool
    review_metadata: ReviewMetadata | None
    cases: tuple[RegionalAcceptanceCase, ...]
    execution_order: tuple[str, ...]
    do_not_run_case_ids: tuple[str, ...]
    dependency_graph: tuple[DependencyNode, ...]

    @property
    def case_ids(self) -> tuple[str, ...]:
        return tuple(case.id for case in self.cases)

    @property
    def graph(self) -> dict[str, tuple[str, ...]]:
        return {node.case_id: node.depends_on for node in self.dependency_graph}

    def case(self, case_id: str) -> RegionalAcceptanceCase:
        for item in self.cases:
            if item.id == case_id:
                return item
        raise KeyError(case_id)

    def blocked_case_ids(
        self,
        failed_case_ids: set[str] | frozenset[str],
    ) -> tuple[str, ...]:
        unknown = sorted(failed_case_ids.difference(self.case_ids))
        if unknown:
            raise ValueError(f"unknown failed case IDs: {unknown}")
        failed_or_blocked = set(failed_case_ids)
        blocked: set[str] = set()
        changed = True
        while changed:
            changed = False
            for case in self.cases:
                if case.id in failed_or_blocked:
                    continue
                if any(
                    dependency in failed_or_blocked
                    for dependency in case.execution.depends_on
                ):
                    blocked.add(case.id)
                    failed_or_blocked.add(case.id)
                    changed = True
        return tuple(case.id for case in self.cases if case.id in blocked)

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "mode": self.mode.value,
            "collect_all": self.collect_all,
            "read_only": self.read_only,
            "repair_allowed": self.repair_allowed,
            "review_metadata": (
                self.review_metadata.as_dict()
                if self.review_metadata is not None
                else None
            ),
            "case_ids": list(self.case_ids),
            "execution_order": list(self.execution_order),
            "do_not_run_case_ids": list(self.do_not_run_case_ids),
            "cases": [case.as_dict() for case in self.cases],
            "dependency_graph": [node.as_dict() for node in self.dependency_graph],
        }


@dataclass(frozen=True)
class _OrderedCase:
    id: str
    phase_sequence: int | None
    phase_name: str | None
    maintenance_window: str | None
    do_not_run_reason: str | None = None


@dataclass(frozen=True)
class _CatalogCase:
    id: str
    title: str
    category: str
    level: str
    risk: str
    problem: str
    injection: str
    expected: tuple[str, ...]
    procedure: str | None
    automation: AutomationKind
    pytest_nodeid: str | None
    command: tuple[str, ...]
    related_pytest: str | None
    gate: str | None
    execution: ExecutionPolicy
    evidence_verdict: str | None


@dataclass(frozen=True)
class _CaseOverride:
    executor: ExecutorKind | None = None
    depends_on: tuple[str, ...] = ()
    locks: tuple[ResourceLock, ...] = ()


@dataclass(frozen=True)
class _ReviewedOverride:
    metadata: ReviewMetadata
    cases: Mapping[str, _CaseOverride]


class _UniqueKeyLoader(yaml.SafeLoader):  # type: ignore[misc]
    def construct_mapping(
        self,
        node: Any,
        deep: bool = False,
    ) -> dict[object, object]:
        self.flatten_mapping(node)
        result: dict[object, object] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in result:
                raise ValueError(f"duplicate YAML key: {key}")
            result[key] = self.construct_object(value_node, deep=deep)
        return result


def _load_yaml(path: Path) -> object:
    try:
        return cast(
            object,
            yaml.load(
                path.read_text(encoding="utf-8"),
                Loader=_UniqueKeyLoader,
            ),
        )
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot load YAML {path}: {exc}") from exc


def _file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ValueError(f"cannot hash {path}: {exc}") from exc


def _canonical_sha256(value: object, context: str) -> str:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be JSON-serializable") from exc
    return hashlib.sha256(encoded).hexdigest()


def _mapping(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{context} keys must be strings")
        result[key] = item
    return result


def _list(value: object, context: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list")
    return cast(list[object], value)


def _string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _integer(value: object, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{context} must be an integer")
    return value


def _known_fields(
    value: Mapping[str, object],
    allowed: set[str],
    context: str,
) -> None:
    unknown = set(value).difference(allowed)
    if unknown:
        raise ValueError(f"{context} has unknown fields: {sorted(unknown)}")


def _case_id(value: object, context: str) -> str:
    result = _string(value, context)
    if REGIONAL_CASE_ID.fullmatch(result) is None:
        raise ValueError(f"{context} is not a regional case ID: {result}")
    return result


def _optional_string(
    value: Mapping[str, object],
    field: str,
    context: str,
) -> str | None:
    raw = value.get(field)
    if raw is None:
        return None
    return _string(raw, f"{context}.{field}")


def _sha256(value: object, context: str) -> str:
    text = _string(value, context)
    if SHA256_HEX.fullmatch(text) is None:
        raise ValueError(f"{context} must be a 64-character SHA-256 hex digest")
    return text.lower()


def _rfc3339(value: object, context: str) -> str:
    text = _string(value, context)
    if RFC3339_TIMESTAMP.fullmatch(text) is None:
        raise ValueError(f"{context} must be an RFC3339 timestamp with timezone")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(
            f"{context} must be an RFC3339 timestamp with timezone"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{context} must include a timezone")
    return text


def _enum_value[E: Enum](
    enum_type: type[E],
    value: object,
    context: str,
) -> E:
    text = _string(value, context)
    try:
        return enum_type(text)
    except ValueError as exc:
        allowed = sorted(str(item.value) for item in enum_type)
        raise ValueError(f"{context} must be one of {allowed}, got {text!r}") from exc


def _unique_strings(value: object, context: str) -> tuple[str, ...]:
    items = tuple(
        _case_id(item, f"{context}[{index}]")
        for index, item in enumerate(_list(value, context))
    )
    if len(items) != len(set(items)):
        raise ValueError(f"{context} contains duplicate case IDs")
    return items


def _text_sequence(value: object, context: str) -> tuple[str, ...]:
    if isinstance(value, str):
        return (_string(value, context),)
    items = tuple(
        _string(item, f"{context}[{index}]")
        for index, item in enumerate(_list(value, context))
    )
    if not items:
        raise ValueError(f"{context} must not be empty")
    return items


def _parse_locks(value: object, context: str) -> tuple[ResourceLock, ...]:
    result: list[ResourceLock] = []
    resources: set[str] = set()
    for index, raw_lock in enumerate(_list(value, context)):
        lock_context = f"{context}[{index}]"
        lock = _mapping(raw_lock, lock_context)
        _known_fields(lock, {"resource", "mode"}, lock_context)
        resource = _string(lock.get("resource"), f"{lock_context}.resource")
        mode = _enum_value(
            LockMode,
            lock.get("mode"),
            f"{lock_context}.mode",
        )
        if resource in resources:
            raise ValueError(f"{context} contains duplicate lock {resource!r}")
        resources.add(resource)
        result.append(ResourceLock(resource=resource, mode=mode))
    return tuple(result)


def _parse_execution(
    raw: object,
    *,
    case_id: str,
) -> ExecutionPolicy:
    if raw is None:
        return ExecutionPolicy(parallel_safe=False)
    context = f"catalog case {case_id}.execution"
    value = _mapping(raw, context)
    _known_fields(
        value,
        {
            "parallel_safe",
            "locks",
            "depends_on",
            "failure_scope",
            "environment",
        },
        context,
    )
    parallel_safe = value.get("parallel_safe")
    if not isinstance(parallel_safe, bool):
        raise ValueError(f"{context}.parallel_safe must be a boolean")
    locks = _parse_locks(value.get("locks", []), f"{context}.locks")
    if parallel_safe and not locks:
        raise ValueError(f"{context} requires locks when parallel_safe is true")
    depends_on = _unique_strings(
        value.get("depends_on", []),
        f"{context}.depends_on",
    )
    if case_id in depends_on:
        raise ValueError(f"catalog case {case_id} cannot depend on itself")
    return ExecutionPolicy(
        parallel_safe=parallel_safe,
        locks=locks,
        depends_on=depends_on,
        failure_scope=_enum_value(
            FailureScope,
            value.get("failure_scope", FailureScope.BRANCH.value),
            f"{context}.failure_scope",
        ),
        environment=_enum_value(
            EnvironmentMode,
            value.get("environment", EnvironmentMode.INHERIT.value),
            f"{context}.environment",
        ),
        configured_in_catalog=True,
    )


def _expand_order(
    path: Path,
) -> tuple[tuple[_OrderedCase, ...], tuple[_OrderedCase, ...]]:
    root = _mapping(_load_yaml(path), f"regional order {path}")
    _known_fields(
        root,
        {
            "schema_version",
            "source_document",
            "ordering_rule",
            "phases",
            "do_not_run",
        },
        f"regional order {path}",
    )
    if _integer(root.get("schema_version"), "regional order schema_version") != 1:
        raise ValueError("regional order schema_version must be 1")

    parsed_phases: list[tuple[int, str, str, tuple[str, ...]]] = []
    seen_sequences: set[int] = set()
    for phase_index, raw_phase in enumerate(
        _list(root.get("phases"), "regional order phases")
    ):
        context = f"regional order phases[{phase_index}]"
        phase = _mapping(raw_phase, context)
        _known_fields(
            phase,
            {
                "sequence",
                "name",
                "maintenance_window",
                "entries",
                "notes",
                "prerequisite",
            },
            context,
        )
        sequence = _integer(phase.get("sequence"), f"{context}.sequence")
        if sequence < 0:
            raise ValueError(f"{context}.sequence must not be negative")
        if sequence in seen_sequences:
            raise ValueError(f"duplicate regional phase sequence: {sequence}")
        seen_sequences.add(sequence)
        name = _string(phase.get("name"), f"{context}.name")
        maintenance_window = _string(
            phase.get("maintenance_window"),
            f"{context}.maintenance_window",
        )
        case_ids: list[str] = []
        for entry_index, raw_entry in enumerate(
            _list(phase.get("entries"), f"{context}.entries")
        ):
            entry_context = f"{context}.entries[{entry_index}]"
            entry = _mapping(raw_entry, entry_context)
            if set(entry) == {"case"}:
                case_ids.append(_case_id(entry["case"], f"{entry_context}.case"))
                continue
            if set(entry) != {"range"}:
                raise ValueError(f"{entry_context} must contain exactly case or range")
            range_value = _mapping(
                entry["range"],
                f"{entry_context}.range",
            )
            _known_fields(
                range_value,
                {"prefix", "start", "end"},
                f"{entry_context}.range",
            )
            prefix = _string(
                range_value.get("prefix"),
                f"{entry_context}.range.prefix",
            )
            if re.fullmatch(r"[A-Z0-9]+", prefix) is None:
                raise ValueError(f"{entry_context}.range.prefix is invalid: {prefix}")
            start = _integer(
                range_value.get("start"),
                f"{entry_context}.range.start",
            )
            end = _integer(
                range_value.get("end"),
                f"{entry_context}.range.end",
            )
            if start < 1 or end < start:
                raise ValueError(f"{entry_context}.range is invalid")
            case_ids.extend(
                f"GF-REGIONAL-{prefix}-{number:03d}" for number in range(start, end + 1)
            )
        parsed_phases.append((sequence, name, maintenance_window, tuple(case_ids)))

    parsed_phases.sort(key=lambda item: item[0])
    sequences = [item[0] for item in parsed_phases]
    if sequences != list(range(len(parsed_phases))):
        raise ValueError("regional phase sequences must be contiguous starting at zero")

    ordered: list[_OrderedCase] = []
    for sequence, name, maintenance_window, phase_case_ids in parsed_phases:
        ordered.extend(
            _OrderedCase(
                id=case_id,
                phase_sequence=sequence,
                phase_name=name,
                maintenance_window=maintenance_window,
            )
            for case_id in phase_case_ids
        )

    retired: list[_OrderedCase] = []
    for index, raw_entry in enumerate(
        _list(root.get("do_not_run"), "regional order do_not_run")
    ):
        context = f"regional order do_not_run[{index}]"
        entry = _mapping(raw_entry, context)
        _known_fields(entry, {"case", "reason"}, context)
        retired.append(
            _OrderedCase(
                id=_case_id(entry.get("case"), f"{context}.case"),
                phase_sequence=None,
                phase_name=None,
                maintenance_window=None,
                do_not_run_reason=_string(
                    entry.get("reason"),
                    f"{context}.reason",
                ),
            )
        )

    all_ids = [item.id for item in (*ordered, *retired)]
    duplicates = sorted(
        case_id for case_id in set(all_ids) if all_ids.count(case_id) > 1
    )
    if duplicates:
        raise ValueError(f"duplicate regional execution order case IDs: {duplicates}")
    if not retired:
        raise ValueError("regional order must declare its DO_NOT_RUN cases")
    return tuple(ordered), tuple(retired)


def _load_catalog(path: Path) -> dict[str, _CatalogCase]:
    root = _mapping(_load_yaml(path), f"fault catalog {path}")
    if _integer(root.get("schema_version"), "fault catalog schema_version") != 1:
        raise ValueError("fault catalog schema_version must be 1")
    raw_cases = _list(root.get("test_cases"), "fault catalog test_cases")
    result: dict[str, _CatalogCase] = {}
    for index, raw_case in enumerate(raw_cases):
        context = f"fault catalog test_cases[{index}]"
        value = _mapping(raw_case, context)
        case_id = _string(value.get("id"), f"{context}.id")
        if case_id in result:
            raise ValueError(f"duplicate fault catalog case ID: {case_id}")
        automation = _enum_value(
            AutomationKind,
            value.get("automation"),
            f"catalog case {case_id}.automation",
        )
        pytest_nodeid = _optional_string(
            value,
            "pytest_nodeid",
            f"catalog case {case_id}",
        )
        raw_command = value.get("command")
        command: tuple[str, ...] = ()
        if raw_command is not None:
            command = tuple(
                _string(item, f"catalog case {case_id}.command[{position}]")
                for position, item in enumerate(
                    _list(raw_command, f"catalog case {case_id}.command")
                )
            )
            if not command:
                raise ValueError(f"catalog case {case_id}.command is empty")
        if automation is AutomationKind.PYTEST and pytest_nodeid is None:
            raise ValueError(f"catalog pytest case {case_id} lacks pytest_nodeid")
        if automation is AutomationKind.COMMAND and not command:
            raise ValueError(f"catalog command case {case_id} lacks command")
        related_pytest = _optional_string(
            value,
            "related_pytest",
            f"catalog case {case_id}",
        )
        gate = _optional_string(value, "gate", f"catalog case {case_id}")
        raw_procedure = value.get("procedure")
        procedure = (
            _string(raw_procedure, f"catalog case {case_id}.procedure")
            if isinstance(raw_procedure, str)
            else None
        )
        if (
            case_id.startswith("GF-REGIONAL-")
            and automation is AutomationKind.MANUAL
            and procedure is None
        ):
            raise ValueError(f"regional catalog manual case {case_id} lacks procedure")
        if related_pytest is not None and gate is not None:
            raise ValueError(
                f"catalog case {case_id} has ambiguous related_pytest and gate"
            )
        evidence_verdict: str | None = None
        raw_evidence = value.get("evidence")
        if raw_evidence is not None:
            evidence = _mapping(
                raw_evidence,
                f"catalog case {case_id}.evidence",
            )
            evidence_verdict = _optional_string(
                evidence,
                "verdict",
                f"catalog case {case_id}.evidence",
            )
        result[case_id] = _CatalogCase(
            id=case_id,
            title=_string(value.get("title"), f"catalog case {case_id}.title"),
            category=_string(
                value.get("category"),
                f"catalog case {case_id}.category",
            ),
            level=_string(value.get("level"), f"catalog case {case_id}.level"),
            risk=_string(value.get("risk"), f"catalog case {case_id}.risk"),
            problem=_string(
                value.get("problem"),
                f"catalog case {case_id}.problem",
            ),
            injection=_string(
                value.get("injection"),
                f"catalog case {case_id}.injection",
            ),
            expected=_text_sequence(
                value.get("expected"),
                f"catalog case {case_id}.expected",
            ),
            procedure=procedure,
            automation=automation,
            pytest_nodeid=pytest_nodeid,
            command=command,
            related_pytest=related_pytest,
            gate=gate,
            execution=_parse_execution(value.get("execution"), case_id=case_id),
            evidence_verdict=evidence_verdict,
        )
    return result


def _validate_coverage(
    ordered: tuple[_OrderedCase, ...],
    retired: tuple[_OrderedCase, ...],
    catalog: Mapping[str, _CatalogCase],
) -> None:
    indexed = {item.id for item in (*ordered, *retired)}
    unknown = sorted(indexed.difference(catalog))
    if unknown:
        raise ValueError(f"unknown regional case IDs in execution order: {unknown}")
    regional_catalog = {
        case_id for case_id in catalog if case_id.startswith("GF-REGIONAL-")
    }
    missing = sorted(regional_catalog.difference(indexed))
    if missing:
        raise ValueError(
            f"regional catalog cases missing from execution order: {missing}"
        )
    retired_ids = {item.id for item in retired}
    catalog_retired = {
        case_id
        for case_id in regional_catalog
        if catalog[case_id].evidence_verdict == "SUPERSEDED"
    }
    if retired_ids != catalog_retired:
        raise ValueError(
            "DO_NOT_RUN cases must exactly match regional catalog "
            f"SUPERSEDED cases; order={sorted(retired_ids)}, "
            f"catalog={sorted(catalog_retired)}"
        )
    invalid = sorted(
        case_id
        for case_id in retired_ids
        if catalog[case_id].automation is not AutomationKind.MANUAL
    )
    if invalid:
        raise ValueError(f"DO_NOT_RUN cases must be manual: {invalid}")


def load_reviewed_override(
    path: Path,
    *,
    known_cases: Mapping[str, AutomationKind],
    do_not_run_case_ids: set[str],
    order_sha256: str,
    catalog_sha256: str,
) -> _ReviewedOverride:
    root = _mapping(_load_yaml(path), f"reviewed override {path}")
    _known_fields(
        root,
        {
            "schema_version",
            "reviewed",
            "reviewed_by",
            "reviewed_at",
            "order_sha256",
            "catalog_sha256",
            "proposal_sha256",
            "cases",
        },
        f"reviewed override {path}",
    )
    if _integer(root.get("schema_version"), "override schema_version") != 1:
        raise ValueError("override schema_version must be 1")
    if root.get("reviewed") is not True:
        raise ValueError("override must explicitly declare reviewed: true")
    metadata = ReviewMetadata(
        reviewed_by=_string(
            root.get("reviewed_by"),
            "override reviewed_by",
        ).strip(),
        reviewed_at=_rfc3339(
            root.get("reviewed_at"),
            "override reviewed_at",
        ),
        order_sha256=_sha256(
            root.get("order_sha256"),
            "override order_sha256",
        ),
        catalog_sha256=_sha256(
            root.get("catalog_sha256"),
            "override catalog_sha256",
        ),
        proposal_sha256=(
            _sha256(
                root["proposal_sha256"],
                "override proposal_sha256",
            )
            if root.get("proposal_sha256") is not None
            else None
        ),
    )
    if metadata.order_sha256 != order_sha256:
        raise ValueError(
            "reviewed override order_sha256 does not match the current "
            "regional execution order"
        )
    if metadata.catalog_sha256 != catalog_sha256:
        raise ValueError(
            "reviewed override catalog_sha256 does not match the current fault catalog"
        )
    raw_cases = _mapping(root.get("cases"), "override cases")
    result: dict[str, _CaseOverride] = {}
    for case_id, raw_override in raw_cases.items():
        _case_id(case_id, f"override case {case_id}")
        if case_id not in known_cases:
            raise ValueError(f"override references unknown case ID: {case_id}")
        if case_id in do_not_run_case_ids:
            raise ValueError(f"override cannot target DO_NOT_RUN case {case_id}")
        context = f"override case {case_id}"
        value = _mapping(raw_override, context)
        _known_fields(value, {"executor", "depends_on", "locks"}, context)
        if not value:
            raise ValueError(f"{context} must add at least one constraint")
        executor: ExecutorKind | None = None
        if "executor" in value:
            executor = _enum_value(
                ExecutorKind,
                value["executor"],
                f"{context}.executor",
            )
            if executor is not ExecutorKind.CODEX_MANUAL:
                raise ValueError(f"{context}.executor may only be codex-manual")
            if known_cases[case_id] is not AutomationKind.MANUAL:
                raise ValueError(
                    f"{context} can select codex-manual only for manual cases"
                )
        result[case_id] = _CaseOverride(
            executor=executor,
            depends_on=_unique_strings(
                value.get("depends_on", []),
                f"{context}.depends_on",
            ),
            locks=_parse_locks(
                value.get("locks", []),
                f"{context}.locks",
            ),
        )
    return _ReviewedOverride(metadata=metadata, cases=result)


def untrusted_dependency_proposal_to_override_template(
    proposal: Mapping[str, object],
    *,
    order_path: Path = DEFAULT_ORDER,
    catalog_path: Path = DEFAULT_CATALOG,
) -> dict[str, object]:
    """Return a YAML-safe, explicitly unreviewed override candidate.

    The result is deliberately rejected by ``load_reviewed_override`` until a
    human supplies review metadata and changes ``reviewed`` to ``true``.
    Executor suggestions capable of mutation are never copied into the
    candidate override.
    """

    root = _mapping(proposal, "dependency proposal")
    if root.get("trusted") is not False:
        raise ValueError("dependency proposal must explicitly declare trusted: false")
    raw_payload = root.get("proposal")
    payload = (
        _mapping(raw_payload, "dependency proposal.proposal")
        if raw_payload is not None
        else root
    )
    if payload.get("trusted") is not False:
        raise ValueError(
            "dependency proposal payload must explicitly declare trusted: false"
        )

    cases: dict[str, object] = {}
    allowed_executors = {
        "pytest",
        "command",
        "codex-read-only",
        "codex-manual",
        "human",
        "controlled-live-executor",
        "do-not-run",
    }
    for index, raw_case in enumerate(
        _list(payload.get("results"), "dependency proposal results")
    ):
        context = f"dependency proposal results[{index}]"
        value = _mapping(raw_case, context)
        case_id = _case_id(value.get("case_id"), f"{context}.case_id")
        if case_id in cases:
            raise ValueError(f"dependency proposal has duplicate case ID: {case_id}")
        executor = _string(value.get("executor"), f"{context}.executor")
        if executor not in allowed_executors:
            raise ValueError(f"{context}.executor has unsupported value: {executor!r}")
        candidate: dict[str, object] = {
            "depends_on": list(
                _unique_strings(
                    value.get("depends_on", []),
                    f"{context}.depends_on",
                )
            ),
            "locks": [
                lock.as_dict()
                for lock in _parse_locks(
                    value.get("locks", []),
                    f"{context}.locks",
                )
            ],
        }
        if executor in {"codex-read-only", "codex-manual"}:
            candidate["executor"] = ExecutorKind.CODEX_MANUAL.value
        cases[case_id] = candidate

    return {
        "schema_version": 1,
        "reviewed": False,
        "reviewed_by": "",
        "reviewed_at": "",
        "order_sha256": _file_sha256(order_path),
        "catalog_sha256": _file_sha256(catalog_path),
        "proposal_sha256": _canonical_sha256(
            proposal,
            "dependency proposal",
        ),
        "cases": cases,
    }


dependency_proposal_to_override_template = (
    untrusted_dependency_proposal_to_override_template
)


def _ordered_union(
    first: tuple[str, ...],
    second: tuple[str, ...],
) -> tuple[str, ...]:
    result = list(first)
    seen = set(first)
    for item in second:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return tuple(result)


def _merge_locks(
    base: tuple[ResourceLock, ...],
    added: tuple[ResourceLock, ...],
    *,
    case_id: str,
) -> tuple[ResourceLock, ...]:
    result = list(base)
    by_resource = {lock.resource: lock for lock in base}
    for lock in added:
        current = by_resource.get(lock.resource)
        if current is not None:
            if current.mode is not lock.mode:
                raise ValueError(
                    f"override for {case_id} cannot replace lock "
                    f"{lock.resource!r} mode {current.mode.value!r}"
                )
            continue
        result.append(lock)
        by_resource[lock.resource] = lock
    return tuple(result)


def _local_default_policy() -> ExecutionPolicy:
    return ExecutionPolicy(
        parallel_safe=True,
        locks=(ResourceLock("local-test", LockMode.SHARED),),
        failure_scope=FailureScope.CASE,
        environment=EnvironmentMode.ISOLATED,
    )


def _codex_default_policy() -> ExecutionPolicy:
    return ExecutionPolicy(
        parallel_safe=True,
        locks=(ResourceLock("acceptance-evidence", LockMode.SHARED),),
        failure_scope=FailureScope.CASE,
        environment=EnvironmentMode.ISOLATED,
    )


def _formal_executor(
    catalog_case: _CatalogCase,
    override: _CaseOverride,
) -> tuple[ExecutorKind, tuple[str, ...], tuple[str, ...], bool, str | None]:
    if override.executor is ExecutorKind.CODEX_MANUAL:
        return ExecutorKind.CODEX_MANUAL, (), (), False, None
    if catalog_case.automation is AutomationKind.PYTEST:
        assert catalog_case.pytest_nodeid is not None
        return (
            ExecutorKind.PYTEST,
            (catalog_case.pytest_nodeid,),
            (),
            False,
            None,
        )
    if catalog_case.automation is AutomationKind.COMMAND:
        return ExecutorKind.COMMAND, (), catalog_case.command, False, None
    return ExecutorKind.HUMAN, (), (), False, None


def _local_executor(
    catalog_case: _CatalogCase,
    override: _CaseOverride,
) -> tuple[ExecutorKind, tuple[str, ...], tuple[str, ...], bool, str | None]:
    if override.executor is ExecutorKind.CODEX_MANUAL:
        return (
            ExecutorKind.CODEX_MANUAL,
            (),
            (),
            False,
            "codex-manual execution is reserved for the formal reviewed plan",
        )
    if (
        catalog_case.risk == "non-destructive"
        and catalog_case.automation is AutomationKind.PYTEST
    ):
        assert catalog_case.pytest_nodeid is not None
        return (
            ExecutorKind.PYTEST,
            (catalog_case.pytest_nodeid,),
            (),
            False,
            None,
        )
    if (
        catalog_case.risk == "non-destructive"
        and catalog_case.automation is AutomationKind.COMMAND
    ):
        return ExecutorKind.COMMAND, (), catalog_case.command, False, None
    if (
        catalog_case.automation is AutomationKind.MANUAL
        and catalog_case.risk in LOCAL_PROXY_RISKS
    ):
        if catalog_case.related_pytest is not None:
            return (
                ExecutorKind.PYTEST,
                (catalog_case.related_pytest,),
                (),
                True,
                None,
            )
        if catalog_case.gate is not None:
            gate_command = (
                ("python3", catalog_case.gate)
                if catalog_case.gate.endswith(".py")
                else (catalog_case.gate,)
            )
            return ExecutorKind.COMMAND, (), gate_command, True, None
    return (
        ExecutorKind.HUMAN,
        (),
        (),
        False,
        "case is outside local-preacceptance automated scope",
    )


def _collect_all_executor(
    catalog_case: _CatalogCase,
    override: _CaseOverride,
) -> tuple[ExecutorKind, tuple[str, ...], tuple[str, ...], bool, str | None]:
    if override.executor is ExecutorKind.CODEX_MANUAL:
        return ExecutorKind.CODEX_MANUAL, (), (), False, None
    return _local_executor(catalog_case, override)


def _validate_dependencies(
    cases: tuple[RegionalAcceptanceCase, ...],
    do_not_run_case_ids: set[str],
) -> None:
    known = {case.id for case in cases}
    graph = {case.id: case.execution.depends_on for case in cases}
    for case_id, dependencies in graph.items():
        unknown = sorted(set(dependencies).difference(known))
        if unknown:
            raise ValueError(f"case {case_id} has unknown dependencies: {unknown}")
        retired = sorted(set(dependencies).intersection(do_not_run_case_ids))
        if retired:
            raise ValueError(f"case {case_id} depends on DO_NOT_RUN cases: {retired}")
        if case_id in dependencies:
            raise ValueError(f"case {case_id} cannot depend on itself")

    state: dict[str, int] = {}
    stack: list[str] = []

    def visit(case_id: str) -> None:
        current = state.get(case_id, 0)
        if current == 2:
            return
        if current == 1:
            start = stack.index(case_id)
            cycle = [*stack[start:], case_id]
            raise ValueError(f"regional acceptance dependency cycle: {cycle}")
        state[case_id] = 1
        stack.append(case_id)
        for dependency in graph[case_id]:
            visit(dependency)
        stack.pop()
        state[case_id] = 2

    for case in cases:
        visit(case.id)


def _load_plan_overrides(
    override_path: Path | None,
    *,
    catalog: Mapping[str, _CatalogCase],
    retired_ids: set[str],
    order_digest: str,
    catalog_digest: str,
) -> tuple[Mapping[str, _CaseOverride], ReviewMetadata | None]:
    if override_path is None:
        return {}, None
    reviewed = load_reviewed_override(
        override_path,
        known_cases={
            case_id: catalog_case.automation
            for case_id, catalog_case in catalog.items()
        },
        do_not_run_case_ids=retired_ids,
        order_sha256=order_digest,
        catalog_sha256=catalog_digest,
    )
    return reviewed.cases, reviewed.metadata


def _plan_mode(value: PlanMode | str) -> PlanMode:
    try:
        return PlanMode(value)
    except ValueError as exc:
        raise ValueError(f"unsupported regional acceptance mode: {value}") from exc


def compile_regional_acceptance_plan(
    *,
    mode: PlanMode | str = PlanMode.FORMAL,
    order_path: Path = DEFAULT_ORDER,
    catalog_path: Path = DEFAULT_CATALOG,
    override_path: Path | None = None,
) -> RegionalAcceptancePlan:
    """Compile and validate the complete regional acceptance plan."""

    selected_mode = _plan_mode(mode)

    order_digest = _file_sha256(order_path)
    catalog_digest = _file_sha256(catalog_path)
    ordered, retired = _expand_order(order_path)
    catalog = _load_catalog(catalog_path)
    if _file_sha256(order_path) != order_digest:
        raise ValueError("regional execution order changed while compiling")
    if _file_sha256(catalog_path) != catalog_digest:
        raise ValueError("fault catalog changed while compiling")
    _validate_coverage(ordered, retired, catalog)
    retired_ids = {item.id for item in retired}
    overrides, review_metadata = _load_plan_overrides(
        override_path,
        catalog=catalog,
        retired_ids=retired_ids,
        order_digest=order_digest,
        catalog_digest=catalog_digest,
    )

    index = {item.id: position for position, item in enumerate((*ordered, *retired))}
    cases: list[RegionalAcceptanceCase] = []
    previous_case_id: str | None = None
    for order_case in ordered:
        catalog_case = catalog[order_case.id]
        override = overrides.get(order_case.id, _CaseOverride())
        mandatory = catalog_case.execution.depends_on
        if selected_mode is PlanMode.FORMAL and previous_case_id is not None:
            mandatory = _ordered_union(mandatory, (previous_case_id,))
        dependencies = _ordered_union(mandatory, override.depends_on)

        if selected_mode is PlanMode.FORMAL:
            executor, pytest_nodeids, command, local_proxy, blocked_reason = (
                _formal_executor(catalog_case, override)
            )
            base_policy = catalog_case.execution
            effective_parallel_safe = False
        elif selected_mode is PlanMode.LOCAL_PREACCEPTANCE:
            executor, pytest_nodeids, command, local_proxy, blocked_reason = (
                _local_executor(catalog_case, override)
            )
            eligible = (
                executor in {ExecutorKind.PYTEST, ExecutorKind.COMMAND}
                and blocked_reason is None
            )
            if eligible and catalog_case.execution.configured_in_catalog:
                base_policy = catalog_case.execution
            elif eligible:
                base_policy = _local_default_policy()
            else:
                base_policy = catalog_case.execution
            effective_parallel_safe = eligible and base_policy.parallel_safe
        else:
            executor, pytest_nodeids, command, local_proxy, blocked_reason = (
                _collect_all_executor(catalog_case, override)
            )
            eligible = (
                executor
                in {
                    ExecutorKind.PYTEST,
                    ExecutorKind.COMMAND,
                    ExecutorKind.CODEX_MANUAL,
                }
                and blocked_reason is None
            )
            if (
                executor in {ExecutorKind.PYTEST, ExecutorKind.COMMAND}
                and catalog_case.execution.configured_in_catalog
            ):
                base_policy = catalog_case.execution
            elif executor in {ExecutorKind.PYTEST, ExecutorKind.COMMAND}:
                base_policy = _local_default_policy()
            elif executor is ExecutorKind.CODEX_MANUAL:
                base_policy = _codex_default_policy()
            else:
                base_policy = catalog_case.execution
            effective_parallel_safe = eligible and base_policy.parallel_safe

        diagnostic_mode = selected_mode is not PlanMode.FORMAL
        executor_read_only = diagnostic_mode or executor is ExecutorKind.CODEX_MANUAL

        execution = ExecutionPolicy(
            parallel_safe=effective_parallel_safe,
            locks=_merge_locks(
                base_policy.locks,
                override.locks,
                case_id=order_case.id,
            ),
            depends_on=dependencies,
            failure_scope=(
                FailureScope.CASE if diagnostic_mode else FailureScope.GLOBAL
            ),
            environment=base_policy.environment,
            configured_in_catalog=base_policy.configured_in_catalog,
            collect_all=diagnostic_mode,
            read_only=executor_read_only,
            repair_allowed=False,
        )
        cases.append(
            RegionalAcceptanceCase(
                id=order_case.id,
                ordinal=index[order_case.id],
                phase_sequence=order_case.phase_sequence,
                phase_name=order_case.phase_name,
                maintenance_window=order_case.maintenance_window,
                title=catalog_case.title,
                category=catalog_case.category,
                level=catalog_case.level,
                risk=catalog_case.risk,
                problem=catalog_case.problem,
                injection=catalog_case.injection,
                expected=catalog_case.expected,
                procedure=catalog_case.procedure,
                automation=catalog_case.automation,
                related_pytest=catalog_case.related_pytest,
                gate=catalog_case.gate,
                catalog_execution=catalog_case.execution,
                executor=executor,
                pytest_nodeids=pytest_nodeids,
                command=command,
                execution=execution,
                mandatory_depends_on=mandatory,
                override_depends_on=override.depends_on,
                local_proxy=local_proxy,
                blocked_reason=blocked_reason,
                do_not_run_reason=None,
            )
        )
        previous_case_id = order_case.id

    for order_case in retired:
        catalog_case = catalog[order_case.id]
        cases.append(
            RegionalAcceptanceCase(
                id=order_case.id,
                ordinal=index[order_case.id],
                phase_sequence=None,
                phase_name=None,
                maintenance_window=None,
                title=catalog_case.title,
                category=catalog_case.category,
                level=catalog_case.level,
                risk=catalog_case.risk,
                problem=catalog_case.problem,
                injection=catalog_case.injection,
                expected=catalog_case.expected,
                procedure=catalog_case.procedure,
                automation=catalog_case.automation,
                related_pytest=catalog_case.related_pytest,
                gate=catalog_case.gate,
                catalog_execution=catalog_case.execution,
                executor=ExecutorKind.DO_NOT_RUN,
                pytest_nodeids=(),
                command=(),
                execution=ExecutionPolicy(
                    parallel_safe=False,
                    failure_scope=(
                        FailureScope.CASE
                        if selected_mode is not PlanMode.FORMAL
                        else FailureScope.GLOBAL
                    ),
                    collect_all=selected_mode is not PlanMode.FORMAL,
                    read_only=selected_mode is not PlanMode.FORMAL,
                    repair_allowed=False,
                ),
                mandatory_depends_on=(),
                override_depends_on=(),
                local_proxy=False,
                blocked_reason=order_case.do_not_run_reason,
                do_not_run_reason=order_case.do_not_run_reason,
            )
        )

    compiled_cases = tuple(cases)
    _validate_dependencies(compiled_cases, retired_ids)
    graph = tuple(
        DependencyNode(case.id, case.execution.depends_on) for case in compiled_cases
    )
    return RegionalAcceptancePlan(
        mode=selected_mode,
        collect_all=selected_mode is not PlanMode.FORMAL,
        read_only=selected_mode is not PlanMode.FORMAL,
        repair_allowed=False,
        review_metadata=review_metadata,
        cases=compiled_cases,
        execution_order=tuple(item.id for item in ordered),
        do_not_run_case_ids=tuple(item.id for item in retired),
        dependency_graph=graph,
    )


compile_plan = compile_regional_acceptance_plan
