from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts.ci_coverage_floors import (
    CoverageCounts,
    CoverageGateError,
    module_floor_violations,
    validate_module_floors,
)

SOURCE = "src/gpu_fault/example.py"
SUMMARY = {
    "num_statements": 10,
    "covered_lines": 5,
    "missing_lines": 5,
    "num_branches": 10,
    "covered_branches": 0,
    "missing_branches": 10,
    "num_partial_branches": 0,
}
FLOOR = {
    "id": "example",
    "description": "Both completely and partially missing paths matter.",
    "globs": [SOURCE],
    "group_floor": 60,
    "file_floor": 60,
}


@pytest.fixture
def source_root(tmp_path: Path) -> Path:
    subprocess.run(
        ["git", "init", "-q", str(tmp_path)], check=True, capture_output=True
    )
    source = tmp_path / SOURCE
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    return tmp_path


def test_missing_group_member_cannot_raise_module_coverage(tmp_path: Path) -> None:
    source = "src/gpu_fault/admin/bootstrap_load_balancer.py"
    path = tmp_path / "partial.json"
    path.write_text(
        json.dumps(
            {
                "files": {
                    source: {
                        "summary": {
                            **SUMMARY,
                            "covered_lines": 10,
                            "missing_lines": 0,
                            "covered_branches": 10,
                            "missing_branches": 0,
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    violations = module_floor_violations(
        path,
        config={
            "coverage": {
                "module_floors": [{**FLOOR, "globs": ["src/gpu_fault/admin/*.py"]}]
            }
        },
    )
    assert any("unmeasured" in violation for violation in violations), (
        "one measured sibling cannot stand in for every current module in its group"
    )


def test_completely_unexecuted_branches_are_not_counted_as_covered(
    source_root: Path,
) -> None:
    path = source_root / "coverage.json"
    path.write_text(
        json.dumps({"files": {SOURCE: {"summary": SUMMARY}}}), encoding="utf-8"
    )
    failures = module_floor_violations(
        path, config={"coverage": {"module_floors": [FLOOR]}}, root=source_root
    )
    assert failures == [
        f"{SOURCE} covers 25.0% of 20 measurable points, below the example file floor of 60%",
        "module group example covers 25.0%, below its group floor of 60%",
    ], "unexecuted branches must not inherit coverage from untouched branch lines"


@pytest.mark.parametrize("extra", ["missing-source", "unknown-reported-source", "none"])
def test_module_inventory_matches_the_current_source_tree(source_root: Path, extra):
    complete = {
        **SUMMARY,
        "covered_lines": 10,
        "missing_lines": 0,
        "covered_branches": 10,
        "missing_branches": 0,
    }
    files = {SOURCE: {"summary": complete}}
    other = "src/gpu_fault/other.py"
    if extra == "missing-source":
        (source_root / other).write_text("VALUE = 2\n", encoding="utf-8")
    elif extra == "unknown-reported-source":
        files[other] = {"summary": complete}
    report = source_root / "coverage.json"
    report.write_text(json.dumps({"files": files}), encoding="utf-8")
    violations = module_floor_violations(
        report,
        config={
            "coverage": {"module_floors": [{**FLOOR, "globs": ["src/gpu_fault/*.py"]}]}
        },
        root=source_root,
    )
    if extra == "none":
        assert violations == [], "a complete passing report must remain accepted"
    else:
        expected = "unmeasured" if extra == "missing-source" else "unknown"
        assert len(violations) == 1 and expected in violations[0], (
            "missing or fictional siblings must not inflate group coverage"
        )


@pytest.mark.parametrize("partial", [0, 1, 3])
def test_branch_line_diagnostic_does_not_change_executed_branch_counts(
    partial: int,
) -> None:
    counts = CoverageCounts.from_summary(
        {**SUMMARY, "num_partial_branches": partial}, source=SOURCE
    )
    assert counts.covered == 5 and counts.measured == 20, (
        "num_partial_branches counts branch lines, not executed branch outcomes"
    )


@pytest.mark.parametrize("field", list(SUMMARY)[:-1])
@pytest.mark.parametrize("value", [None, True, False, -1, 1.5, "0"])
def test_invalid_counters_cannot_be_coerced_into_coverage(
    field: str, value: object
) -> None:
    with pytest.raises(CoverageGateError, match="counters are invalid"):
        CoverageCounts.from_summary({**SUMMARY, field: value}, source=SOURCE)


@pytest.mark.parametrize("field", list(SUMMARY)[:-1])
def test_missing_counters_fail_closed(field: str) -> None:
    summary = {key: value for key, value in SUMMARY.items() if key != field}
    with pytest.raises(CoverageGateError, match="counters are invalid"):
        CoverageCounts.from_summary(summary, source=SOURCE)


@pytest.mark.parametrize("field", ["covered_lines", "missing_branches"])
def test_inconsistent_totals_fail_closed(field: str) -> None:
    with pytest.raises(CoverageGateError, match="counters disagree"):
        CoverageCounts.from_summary({**SUMMARY, field: 11}, source=SOURCE)


@pytest.mark.parametrize("summary", [None, [], "unavailable"])
def test_unknown_summary_is_not_zero_measurable_points(summary: object) -> None:
    with pytest.raises(CoverageGateError, match="summary is missing"):
        CoverageCounts.from_summary(summary, source=SOURCE)


@pytest.mark.parametrize("field", ["file_floor", "group_floor"])
def test_boolean_floor_is_not_a_percentage(field: str) -> None:
    with pytest.raises(CoverageGateError, match="entry is invalid"):
        validate_module_floors([{**FLOOR, field: True}])


@pytest.mark.parametrize("files", [None, [], "unavailable"])
def test_unknown_file_inventory_is_not_an_empty_report(
    tmp_path: Path, files: object
) -> None:
    path = tmp_path / "coverage.json"
    path.write_text(json.dumps({"files": files}), encoding="utf-8")
    with pytest.raises(CoverageGateError, match="inventory is invalid"):
        module_floor_violations(path, config={"coverage": {"module_floors": [FLOOR]}})
