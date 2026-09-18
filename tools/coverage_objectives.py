"""Report the same source inventories with separate statement and branch goals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.ci_coverage_floors import (
    COVERAGE_SCOPES,
    OBJECTIVE_TARGET,
    CoverageCounts,
    module_floor_violations,
)
from scripts.ci_gate_artifacts import CoverageGateError

ROOT = Path(__file__).resolve().parents[1]
SCOPES = COVERAGE_SCOPES
TARGET = OBJECTIVE_TARGET


def measured_files(root: Path, scope: str) -> set[str]:
    result: set[str] = set()
    for relative in SCOPES[scope]:
        directory = root / relative
        if not directory.is_dir() or directory.is_symlink():
            raise CoverageGateError(f"coverage source root is unavailable: {relative}")
        for path in directory.rglob("*"):
            if path.is_symlink():
                raise CoverageGateError(f"unsupported coverage source: {path}")
            if path.is_file() and path.suffix == ".py":
                result.add(path.relative_to(root).as_posix())
    return result


def objective_report(
    report: Mapping[str, Any], *, root: Path, scope: str
) -> dict[str, Any]:
    metadata = report.get("meta")
    if not isinstance(metadata, dict) or metadata.get("branch_coverage") is not True:
        raise CoverageGateError("coverage objectives require real branch coverage")
    files = report.get("files")
    if not isinstance(files, dict):
        raise CoverageGateError("coverage file inventory is invalid")
    expected = measured_files(root, scope)
    missing = expected - files.keys()
    if missing:
        raise CoverageGateError(
            "unmeasured source files: " + ", ".join(sorted(missing))
        )
    totals = [0, 0, 0, 0]
    gaps = []
    excluded: dict[str, list[int]] = {}
    for name in sorted(expected):
        value = files[name]
        counts = CoverageCounts.from_summary(
            value.get("summary") if isinstance(value, dict) else None, source=name
        )
        for index, count in enumerate(
            (
                counts.statements,
                counts.covered_lines,
                counts.branches,
                counts.covered_branches,
            )
        ):
            totals[index] += count
        if value.get("excluded_lines"):
            excluded[name] = value["excluded_lines"]
        if counts.measured > counts.covered:
            gaps.append(
                {
                    "file": name,
                    "missing_statements": counts.statements - counts.covered_lines,
                    "missing_branches": counts.branches - counts.covered_branches,
                }
            )
    statements, covered_lines, branches, covered_branches = totals
    if not statements or not branches:
        raise CoverageGateError("coverage scope has no statements or branches")
    line_percent = 100 * covered_lines / statements
    branch_percent = 100 * covered_branches / branches
    return {
        "scope": scope,
        "source_roots": list(SCOPES[scope]),
        "source_files": len(expected),
        "statements": statements,
        "covered_statements": covered_lines,
        "branches": branches,
        "covered_branches": covered_branches,
        "statement_percent": line_percent,
        "branch_percent": branch_percent,
        "combined_percent": 100
        * (covered_lines + covered_branches)
        / (statements + branches),
        "target_percent": TARGET,
        "target_met": line_percent >= TARGET and branch_percent >= TARGET,
        "excluded_lines": excluded,
        "gaps": sorted(
            gaps,
            key=lambda item: -(item["missing_statements"] + item["missing_branches"]),
        ),
    }


def require_ci_coverage(
    coverage_json: Path, *, root: Path, config: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    try:
        value = json.loads(coverage_json.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CoverageGateError("coverage JSON report is unreadable") from exc
    if not isinstance(value, dict):
        raise CoverageGateError("coverage report is not a mapping")
    reports = {
        scope: objective_report(value, root=root, scope=scope) for scope in SCOPES
    }
    violations = module_floor_violations(coverage_json, config=config, root=root)
    if reports["production"]["combined_percent"] < config["coverage"]["floor"]:
        violations.append("production combined coverage is below the existing floor")
    for scope, report in reports.items():
        if not report["target_met"]:
            violations.append(
                f"{scope} requires >= {TARGET}% statements AND branches: "
                f"{report['statement_percent']:.2f}% statements, "
                f"{report['branch_percent']:.2f}% branches"
            )
    if violations:
        raise CoverageGateError("coverage floors failed:\n- " + "\n- ".join(violations))
    return reports


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage-json", type=Path, required=True)
    parser.add_argument("--scope", choices=SCOPES, required=True)
    parser.add_argument("--require-target", action="store_true")
    options = parser.parse_args(argv)
    try:
        raw = json.loads(options.coverage_json.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise CoverageGateError("coverage report is not a mapping")
        result = objective_report(raw, root=ROOT, scope=options.scope)
    except (OSError, ValueError, CoverageGateError) as exc:
        parser.exit(2, f"coverage objective report refused: {exc}\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return int(options.require_target and not result["target_met"])


if __name__ == "__main__":
    raise SystemExit(main())
