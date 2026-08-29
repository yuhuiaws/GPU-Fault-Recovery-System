#!/usr/bin/env python3
"""Select pytest and regional acceptance cases from changed repository paths."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import fnmatch
from functools import lru_cache
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Callable, Sequence

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tools.run_fault_test_cases import load_catalog  # noqa: E402


DEFAULT_MATRIX = ROOT / "testcases" / "change-impact.yaml"
CATALOG = ROOT / "testcases" / "fault-scenarios.yaml"
ORDER = ROOT / "testcases" / "regional-execution-order.yaml"
ALLOWED_CHECKS = {
    "architecture-check",
    "artifact-check",
    "case-index-check",
    "check",
    "config-check",
    "deployment-contracts-check",
    "docs-check",
    "impact-check",
    "mypy-check",
    "runtime-image-check",
    "xid-catalog-check",
}


class ImpactError(RuntimeError):
    pass


class UniqueKeyLoader(yaml.SafeLoader):
    def construct_mapping(self, node: yaml.Node, deep: bool = False) -> Any:
        self.flatten_mapping(node)
        mapping: dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in mapping:
                raise ImpactError(f"duplicate YAML key: {key}")
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


@dataclass(frozen=True)
class Rule:
    id: str
    description: str
    paths: tuple[str, ...]
    pytest: tuple[str, ...]
    regional_cases: tuple[str, ...]
    approval_cases: frozenset[str]
    checks: tuple[str, ...]
    full: bool
    postgres: bool
    fallback: bool
    counts_as_domain: bool


@dataclass(frozen=True)
class Settings:
    max_domains_before_full: int
    safe_regional_risks: frozenset[str]
    full_checks: tuple[str, ...]
    postgres_command: tuple[str, ...]
    rules: tuple[Rule, ...]


@dataclass(frozen=True)
class Plan:
    changed_files: tuple[str, ...]
    domains: tuple[str, ...]
    pytest_targets: tuple[str, ...]
    checks: tuple[str, ...]
    safe_cases: tuple[str, ...]
    approval_cases: tuple[str, ...]
    not_selected_families: tuple[str, ...]
    full: bool
    postgres: bool
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "changed_files": list(self.changed_files),
            "domains": list(self.domains),
            "pytest_targets": list(self.pytest_targets),
            "checks": list(self.checks),
            "regional_safe_cases": list(self.safe_cases),
            "regional_approval_cases": list(self.approval_cases),
            "not_selected_regional_families": list(self.not_selected_families),
            "full": self.full,
            "postgres": self.postgres,
            "reasons": list(self.reasons),
        }


def _mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ImpactError(f"{field} must be a mapping")
    return dict(value)


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ImpactError(f"{field} must be a non-empty string")
    return value.strip()


def _strings(
    value: object,
    field: str,
    *,
    allow_empty: bool = True,
) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ImpactError(f"{field} must be a string list")
    if not allow_empty and not value:
        raise ImpactError(f"{field} must not be empty")
    return tuple(item.strip() for item in value)


def _boolean(value: object, field: str, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ImpactError(f"{field} must be a boolean")
    return value


def _case_expression(value: str) -> tuple[str, ...]:
    prefix = "GF-REGIONAL-" if not value.startswith("GF-REGIONAL-") else ""
    normalized = prefix + value
    if ".." not in normalized:
        return (normalized,)
    start_text, end_text = normalized.split("..", 1)
    head, separator, start_number = start_text.rpartition("-")
    if (
        not separator
        or not start_number.isdigit()
        or not end_text.isdigit()
        or len(start_number) != len(end_text)
    ):
        raise ImpactError(f"invalid regional case range: {value}")
    start = int(start_number)
    end = int(end_text)
    if start > end:
        raise ImpactError(f"descending regional case range: {value}")
    return tuple(
        f"{head}-{number:0{len(start_number)}d}" for number in range(start, end + 1)
    )


def expand_case_expressions(values: Sequence[str]) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        result.extend(_case_expression(value))
    return tuple(dict.fromkeys(result))


@lru_cache(maxsize=4)
def ordered_regional_cases(
    path: Path = ORDER,
) -> tuple[tuple[str, ...], frozenset[str]]:
    value = _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), "order")
    phases = sorted(value.get("phases") or [], key=lambda item: item["sequence"])
    ordered: list[str] = []
    for phase in phases:
        for entry in phase["entries"]:
            if "case" in entry:
                ordered.append(str(entry["case"]))
                continue
            item = entry["range"]
            ordered.extend(
                f"GF-REGIONAL-{item['prefix']}-{number:03d}"
                for number in range(int(item["start"]), int(item["end"]) + 1)
            )
    do_not_run = frozenset(str(item["case"]) for item in value.get("do_not_run") or [])
    return tuple(ordered), do_not_run


@lru_cache(maxsize=4)
def regional_catalog(path: Path = CATALOG) -> dict[str, dict[str, Any]]:
    return {
        case["id"]: case
        for case in load_catalog(path)
        if case["id"].startswith("GF-REGIONAL-")
    }


def _validate_pattern(value: str, field: str) -> str:
    if value.startswith("/") or ".." in Path(value).parts:
        raise ImpactError(f"{field} must stay within the repository: {value}")
    return value


def _validate_pytest_target(root: Path, value: str, field: str) -> str:
    path_text = value.split("::", 1)[0]
    path = root / path_text
    if not path.exists():
        raise ImpactError(f"{field} does not exist: {value}")
    return value


def load_settings(
    path: Path = DEFAULT_MATRIX,
    *,
    root: Path = ROOT,
) -> Settings:
    value = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
    document = _mapping(value, "change impact document")
    allowed_top = {
        "version",
        "max_domains_before_full",
        "safe_regional_risks",
        "full",
        "rules",
    }
    unknown_top = set(document) - allowed_top
    if unknown_top:
        raise ImpactError(f"change impact has unknown fields: {sorted(unknown_top)}")
    if document.get("version") != 1:
        raise ImpactError("change impact version must be 1")
    maximum = document.get("max_domains_before_full")
    if not isinstance(maximum, int) or maximum < 1:
        raise ImpactError("max_domains_before_full must be a positive integer")
    safe_risks = frozenset(
        _strings(
            document.get("safe_regional_risks"),
            "safe_regional_risks",
            allow_empty=False,
        )
    )
    full = _mapping(document.get("full"), "full")
    full_checks = _strings(full.get("checks"), "full.checks", allow_empty=False)
    postgres_command = _strings(
        full.get("postgres_command"),
        "full.postgres_command",
        allow_empty=False,
    )
    catalog = regional_catalog()
    ordered, do_not_run = ordered_regional_cases()
    ordered_set = set(ordered)
    rules_value = document.get("rules")
    if not isinstance(rules_value, list) or not rules_value:
        raise ImpactError("rules must be a non-empty list")
    rules: list[Rule] = []
    seen: set[str] = set()
    allowed_rule = {
        "id",
        "description",
        "paths",
        "pytest",
        "regional_cases",
        "approval_cases",
        "checks",
        "full",
        "postgres",
        "fallback",
        "counts_as_domain",
    }
    for index, raw in enumerate(rules_value):
        item = _mapping(raw, f"rules[{index}]")
        unknown = set(item) - allowed_rule
        if unknown:
            raise ImpactError(f"rules[{index}] has unknown fields: {sorted(unknown)}")
        rule_id = _string(item.get("id"), f"rules[{index}].id")
        if rule_id in seen:
            raise ImpactError(f"duplicate impact rule id: {rule_id}")
        seen.add(rule_id)
        patterns = tuple(
            _validate_pattern(value, f"{rule_id}.paths")
            for value in _strings(
                item.get("paths"),
                f"{rule_id}.paths",
                allow_empty=False,
            )
        )
        pytest = tuple(
            _validate_pytest_target(root, target, f"{rule_id}.pytest")
            for target in _strings(item.get("pytest"), f"{rule_id}.pytest")
        )
        regional = expand_case_expressions(
            _strings(item.get("regional_cases"), f"{rule_id}.regional_cases")
        )
        approval = frozenset(
            expand_case_expressions(
                _strings(item.get("approval_cases") or [], f"{rule_id}.approval_cases")
            )
        )
        unknown_cases = sorted(set(regional) - set(catalog))
        if unknown_cases:
            raise ImpactError(f"{rule_id} references unknown cases: {unknown_cases}")
        unordered = sorted(set(regional) - ordered_set)
        if unordered:
            raise ImpactError(
                f"{rule_id} cases are absent from regional execution order: {unordered}"
            )
        forbidden = sorted(set(regional) & do_not_run)
        if forbidden:
            raise ImpactError(f"{rule_id} selects DO_NOT_RUN cases: {forbidden}")
        if not approval.issubset(regional):
            raise ImpactError(f"{rule_id}.approval_cases must be selected cases")
        checks = _strings(item.get("checks") or [], f"{rule_id}.checks")
        invalid_checks = sorted(set(checks) - ALLOWED_CHECKS)
        if invalid_checks:
            raise ImpactError(f"{rule_id} has unsupported checks: {invalid_checks}")
        rules.append(
            Rule(
                id=rule_id,
                description=_string(
                    item.get("description"),
                    f"rules[{index}].description",
                ),
                paths=patterns,
                pytest=pytest,
                regional_cases=regional,
                approval_cases=approval,
                checks=checks,
                full=_boolean(item.get("full"), f"{rule_id}.full", False),
                postgres=_boolean(
                    item.get("postgres"),
                    f"{rule_id}.postgres",
                    False,
                ),
                fallback=_boolean(
                    item.get("fallback"),
                    f"{rule_id}.fallback",
                    False,
                ),
                counts_as_domain=_boolean(
                    item.get("counts_as_domain"),
                    f"{rule_id}.counts_as_domain",
                    True,
                ),
            )
        )
    invalid_full_checks = sorted(set(full_checks) - ALLOWED_CHECKS)
    if invalid_full_checks:
        raise ImpactError(f"full has unsupported checks: {invalid_full_checks}")
    return Settings(
        max_domains_before_full=maximum,
        safe_regional_risks=safe_risks,
        full_checks=full_checks,
        postgres_command=postgres_command,
        rules=tuple(rules),
    )


def _matches(path: str, pattern: str) -> bool:
    return fnmatch.fnmatchcase(path, pattern)


def _case_family(case_id: str) -> str:
    body = case_id.removeprefix("GF-REGIONAL-")
    return body.rsplit("-", 1)[0]


def build_plan(
    changed_files: Sequence[str],
    settings: Settings,
    *,
    root: Path = ROOT,
) -> Plan:
    catalog = regional_catalog()
    ordered, _do_not_run = ordered_regional_cases()
    order = {case_id: index for index, case_id in enumerate(ordered)}
    selected_rules: dict[str, Rule] = {}
    unmatched: list[str] = []
    fallback_files: list[str] = []
    normalized = tuple(dict.fromkeys(sorted(changed_files)))
    for changed in normalized:
        primary = [
            rule
            for rule in settings.rules
            if not rule.fallback and any(_matches(changed, item) for item in rule.paths)
        ]
        fallback = next(
            (
                rule
                for rule in settings.rules
                if rule.fallback and any(_matches(changed, item) for item in rule.paths)
            ),
            None,
        )
        matches = primary or ([fallback] if fallback is not None else [])
        if not matches:
            unmatched.append(changed)
            continue
        if not primary and matches[0].full:
            fallback_files.append(changed)
        for rule in matches:
            selected_rules[rule.id] = rule
    counted_domains = sorted(
        rule.id for rule in selected_rules.values() if rule.counts_as_domain
    )
    reasons: list[str] = []
    full = False
    if unmatched:
        full = True
        reasons.append("unmatched files: " + ", ".join(unmatched))
    if fallback_files:
        reasons.append("fail-closed fallback: " + ", ".join(fallback_files))
    forcing = sorted(rule.id for rule in selected_rules.values() if rule.full)
    if forcing:
        full = True
        reasons.append("full-test domains: " + ", ".join(forcing))
    if len(counted_domains) > settings.max_domains_before_full:
        full = True
        reasons.append(
            f"changed domains {len(counted_domains)} exceed "
            f"limit {settings.max_domains_before_full}"
        )
    pytest_targets: set[str] = set()
    checks: set[str] = set()
    regional: set[str] = set()
    explicit_approval: set[str] = set()
    postgres = False
    for rule in selected_rules.values():
        pytest_targets.update(rule.pytest)
        checks.update(rule.checks)
        regional.update(rule.regional_cases)
        explicit_approval.update(rule.approval_cases)
        postgres |= rule.postgres
    for changed in normalized:
        if changed.startswith("tests/") and changed.endswith(".py"):
            if (root / changed).is_file():
                pytest_targets.add(changed)
    if full:
        regional = set(ordered)
        checks = set(settings.full_checks)
        pytest_targets.clear()
    if not full:
        for case_id in regional:
            case = catalog[case_id]
            for field in ("pytest_nodeid", "related_pytest"):
                target = case.get(field)
                if isinstance(target, str):
                    pytest_targets.add(target)
    safe = sorted(
        (
            case_id
            for case_id in regional
            if case_id not in explicit_approval
            and catalog[case_id]["risk"] in settings.safe_regional_risks
        ),
        key=order.__getitem__,
    )
    approval = sorted(set(regional) - set(safe), key=order.__getitem__)
    all_families = {_case_family(case_id) for case_id in ordered}
    selected_families = {_case_family(case_id) for case_id in regional}
    return Plan(
        changed_files=normalized,
        domains=tuple(sorted(selected_rules)),
        pytest_targets=tuple(sorted(pytest_targets)),
        checks=tuple(sorted(checks)),
        safe_cases=tuple(safe),
        approval_cases=tuple(approval),
        not_selected_families=tuple(sorted(all_families - selected_families)),
        full=full,
        postgres=postgres,
        reasons=tuple(reasons),
    )


def _git(
    root: Path,
    arguments: Sequence[str],
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> list[str]:
    completed = runner(
        ["git", "-c", "core.quotePath=false", *arguments],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        raise ImpactError(completed.stderr.strip() or "git command failed")
    return [item.strip() for item in completed.stdout.splitlines() if item.strip()]


def changed_files_from_git(
    base: str,
    *,
    root: Path = ROOT,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> tuple[str, ...]:
    _git(root, ["rev-parse", "--verify", f"{base}^{{commit}}"], runner)
    values: set[str] = set()
    for arguments in (
        ["diff", "--name-only", "--diff-filter=ACMRD", f"{base}...HEAD"],
        ["diff", "--name-only", "--diff-filter=ACMRD"],
        ["diff", "--cached", "--name-only", "--diff-filter=ACMRD"],
        ["ls-files", "--others", "--exclude-standard"],
    ):
        values.update(_git(root, arguments, runner))
    return tuple(sorted(values))


def pytest_command(plan: Plan) -> tuple[str, ...]:
    if plan.full:
        return ("make", "check", f"PYTHON={sys.executable}")
    if not plan.pytest_targets:
        return ()
    return (
        sys.executable,
        "-m",
        "pytest",
        "-q",
        *plan.pytest_targets,
    )


def render_text(plan: Plan, *, regional_only: bool = False) -> str:
    lines = ["Changed files:"]
    lines.extend(f"  {item}" for item in plan.changed_files)
    if not plan.changed_files:
        lines.append("  none")
    lines.extend(["", "Changed domains:"])
    lines.extend(f"  {item}" for item in plan.domains)
    if not plan.domains:
        lines.append("  none")
    lines.extend(
        [
            "",
            f"Full pytest/static gate: {'yes' if plan.full else 'no'}",
            f"Full regional acceptance: {'yes' if plan.full else 'no'}",
            f"PostgreSQL stress required: {'yes' if plan.postgres else 'no'}",
        ]
    )
    if plan.reasons:
        lines.append("Escalation reasons:")
        lines.extend(f"  {item}" for item in plan.reasons)
    if not regional_only:
        lines.extend(["", "Run static checks:"])
        lines.extend(f"  make {item}" for item in plan.checks)
        if not plan.checks:
            lines.append("  none")
        lines.extend(["", "Run pytest:"])
        command = pytest_command(plan)
        lines.append("  " + (shlex.join(command) if command else "none"))
    lines.extend(["", "Regional safe cases:"])
    lines.extend(f"  {item}" for item in plan.safe_cases)
    if not plan.safe_cases:
        lines.append("  none")
    lines.extend(["", "Require staging/live approval:"])
    lines.extend(f"  {item}" for item in plan.approval_cases)
    if not plan.approval_cases:
        lines.append("  none")
    lines.extend(["", "Not selected regional families:"])
    lines.append(
        "  "
        + (
            ", ".join(plan.not_selected_families)
            if plan.not_selected_families
            else "none"
        )
    )
    return "\n".join(lines) + "\n"


def execute_plan(plan: Plan, settings: Settings, *, root: Path = ROOT) -> None:
    if plan.full:
        subprocess.run(
            ["make", "check", f"PYTHON={sys.executable}"],
            cwd=root,
            check=True,
        )
    else:
        for check in plan.checks:
            subprocess.run(
                ["make", check, f"PYTHON={sys.executable}"],
                cwd=root,
                check=True,
            )
        command = pytest_command(plan)
        if command:
            subprocess.run(command, cwd=root, check=True)
    if plan.postgres:
        if not os.environ.get("GPU_FAULT_TEST_POSTGRES_URL"):
            raise ImpactError(
                "PostgreSQL stress is required; set GPU_FAULT_TEST_POSTGRES_URL"
            )
        subprocess.run(
            [*settings.postgres_command, f"PYTHON={sys.executable}"],
            cwd=root,
            check=True,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=os.environ.get("BASE", "origin/main"))
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--changed-file", action="append", default=[])
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--regional-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--check", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        settings = load_settings(arguments.matrix.resolve())
        if arguments.check:
            print(
                f"change impact check passed: {len(settings.rules)} rules",
                file=sys.stderr,
            )
            return 0
        changed = (
            tuple(arguments.changed_file)
            if arguments.changed_file
            else changed_files_from_git(arguments.base)
        )
        plan = build_plan(changed, settings)
        if arguments.format == "json":
            print(json.dumps(plan.as_dict(), indent=2, sort_keys=True))
        else:
            print(render_text(plan, regional_only=arguments.regional_only), end="")
        if arguments.execute:
            if arguments.regional_only:
                raise ImpactError("--execute cannot run regional cases")
            execute_plan(plan, settings)
        return 0
    except (
        ImpactError,
        OSError,
        subprocess.SubprocessError,
        ValueError,
        yaml.YAMLError,
    ) as exc:
        print(f"test impact selection failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
