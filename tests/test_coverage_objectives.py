from __future__ import annotations

import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

from tools import coverage_objectives as objectives


@pytest.fixture
def measured(tmp_path: Path) -> tuple[Path, dict]:
    files = {}
    for directory in objectives.SCOPES["runner"]:
        path = tmp_path / directory / "sample.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("if enabled:\n    action()\n", encoding="utf-8")
        files[path.relative_to(tmp_path).as_posix()] = {
            "summary": {
                "num_statements": 100,
                "covered_lines": 100,
                "missing_lines": 0,
                "num_branches": 100,
                "covered_branches": 90,
                "missing_branches": 10,
            },
            "excluded_lines": [],
        }
    return tmp_path, {"meta": {"branch_coverage": True}, "files": files}


def test_statement_success_cannot_hide_branch_shortfall(measured) -> None:
    root, raw = measured
    report = objectives.objective_report(raw, root=root, scope="runner")
    assert report["combined_percent"] == 95, "the old combined goal would pass"
    assert report["statement_percent"] == 100, "all statements were executed"
    assert report["branch_percent"] == 90, "unexecuted alternatives remain visible"
    assert report["target_met"] is False, "both dimensions independently require 95"
    assert len(report["gaps"]) == 2, "neither source root may disappear"


def test_exact_threshold_is_accepted_and_exclusions_are_disclosed(measured) -> None:
    root, raw = measured
    for value in raw["files"].values():
        value["summary"].update(covered_branches=95, missing_branches=5)
    first = next(iter(raw["files"]))
    raw["files"][first]["excluded_lines"] = [3]
    report = objectives.objective_report(raw, root=root, scope="runner")
    assert report["target_met"] is True, "95 is inclusive for statements and branches"
    assert report["excluded_lines"] == {first: [3]}, (
        "existing exclusions are not hidden"
    )


@pytest.mark.parametrize(
    "metadata", [None, {}, {"branch_coverage": False}, {"branch_coverage": 1}]
)
def test_line_only_coverage_cannot_be_used_as_branch_proof(measured, metadata) -> None:
    root, raw = measured
    raw["meta"] = metadata
    with pytest.raises(objectives.CoverageGateError, match="real branch"):
        objectives.objective_report(raw, root=root, scope="runner")


@pytest.mark.parametrize(
    "problem", ["inventory", "missing-file", "missing-root", "empty", "symlink"]
)
def test_incomplete_or_changed_inventory_is_rejected(measured, problem) -> None:
    root, raw = measured
    first = next(iter(raw["files"]))
    if problem == "inventory":
        raw["files"] = []
    elif problem == "missing-file":
        raw["files"].pop(first)
    elif problem == "missing-root":
        (root / first).unlink()
        (root / first).parent.rmdir()
    elif problem == "empty":
        for value in raw["files"].values():
            value["summary"] = {field: 0 for field in value["summary"]}
    else:
        (root / first).unlink()
        (root / first).symlink_to(root / "tools" / "sample.py")
    with pytest.raises(objectives.CoverageGateError):
        objectives.objective_report(raw, root=root, scope="runner")


@pytest.mark.parametrize("require", [False, True])
def test_cli_reports_unmet_target_without_changing_the_existing_floor(
    measured, monkeypatch, capsys, require
) -> None:
    root, raw = measured
    path = root / "coverage.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    monkeypatch.setattr(objectives, "ROOT", root)
    arguments = ["--coverage-json", str(path), "--scope", "runner"]
    if require:
        arguments.append("--require-target")
    assert objectives.main(arguments) == int(require), (
        "explicit target enforcement is reliable"
    )
    result = json.loads(capsys.readouterr().out)
    assert result["target_percent"] == 95, (
        "the requested goal is not a configurable escape"
    )


@pytest.mark.parametrize("payload", ["[]", "{"])
def test_cli_rejects_unreadable_report(tmp_path, monkeypatch, payload) -> None:
    path = tmp_path / "coverage.json"
    path.write_text(payload, encoding="utf-8")
    monkeypatch.setattr(objectives, "ROOT", tmp_path)
    with pytest.raises(SystemExit) as error:
        objectives.main(["--coverage-json", str(path), "--scope", "runner"])
    assert error.value.code == 2, (
        "invalid input must not look like an unmet numeric goal"
    )


def test_a_symlink_directory_cannot_hide_source_from_the_inventory(measured) -> None:
    root, raw = measured
    directory = root / "tools/alias"
    directory.symlink_to(root / "scripts/e2e/regional", target_is_directory=True)
    with pytest.raises(
        objectives.CoverageGateError, match="unsupported coverage source"
    ):
        objectives.objective_report(raw, root=root, scope="runner")


def test_inventory_ignores_non_python_assets_without_omitting_subpackages(
    measured,
) -> None:
    root, raw = measured
    (root / "tools/data").mkdir()
    (root / "tools/data/notes.txt").write_text("fixture", encoding="utf-8")
    assert (
        objectives.objective_report(raw, root=root, scope="runner")["source_files"] == 2
    ), "non-executable assets do not add artificial statement coverage"


def test_module_entrypoint_has_the_same_invalid_input_exit(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "coverage-objectives",
            "--coverage-json",
            str(tmp_path / "missing.json"),
            "--scope",
            "runner",
        ],
    )
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(Path(objectives.__file__)), run_name="__main__")
    assert error.value.code == 2, (
        "the executable entrypoint fails closed on missing evidence"
    )


@pytest.fixture
def ci_report(tmp_path: Path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True, capture_output=True)
    files = {}
    for roots in objectives.SCOPES.values():
        for directory in roots:
            path = tmp_path / directory / "sample.py"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("if enabled:\n    action()\n", encoding="utf-8")
            files[path.relative_to(tmp_path).as_posix()] = {
                "summary": {
                    "num_statements": 100,
                    "covered_lines": 100,
                    "missing_lines": 0,
                    "num_branches": 100,
                    "covered_branches": 100,
                    "missing_branches": 0,
                }
            }
    path = tmp_path / "coverage.json"
    report = {
        "meta": {"branch_coverage": True},
        "files": files,
        "totals": {"percent_covered": 100},
    }
    return tmp_path, path, report, {"coverage": {"floor": 78, "module_floors": []}}


@pytest.mark.parametrize("scope", ["production", "runner"])
@pytest.mark.parametrize("dimension", ["statements", "branches"])
@pytest.mark.parametrize("percent", [94, 95])
def test_ci_requires_95_for_each_dimension_of_each_scope(
    ci_report, scope, dimension, percent
) -> None:
    root, path, report, config = ci_report
    for name, value in report["files"].items():
        if not any(
            name.startswith(source + "/") for source in objectives.SCOPES[scope]
        ):
            continue
        covered, missing = (
            ("covered_lines", "missing_lines")
            if dimension == "statements"
            else ("covered_branches", "missing_branches")
        )
        value["summary"].update({covered: percent, missing: 100 - percent})
    path.write_text(json.dumps(report), encoding="utf-8")
    if percent == 94:
        with pytest.raises(
            objectives.CoverageGateError, match=scope + " requires >= 95"
        ):
            objectives.require_ci_coverage(path, root=root, config=config)
    else:
        actual = objectives.require_ci_coverage(path, root=root, config=config)
        assert all(value["target_met"] for value in actual.values()), (
            "95 is inclusive in both independent scopes"
        )


def test_runner_cannot_raise_the_original_production_floor(ci_report) -> None:
    root, path, report, config = ci_report
    for name, value in report["files"].items():
        if any(
            name.startswith(source + "/") for source in objectives.SCOPES["production"]
        ):
            value["summary"].update(
                covered_lines=77,
                missing_lines=23,
                covered_branches=77,
                missing_branches=23,
            )
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(
        objectives.CoverageGateError, match="production combined coverage"
    ):
        objectives.require_ci_coverage(path, root=root, config=config)


@pytest.mark.parametrize("defect", ["no-runner", "no-branch-counter", "contradictory"])
def test_ci_rejects_incomplete_coverage_even_with_a_passing_total(
    ci_report, defect
) -> None:
    root, path, report, config = ci_report
    first = next(iter(report["files"].values()))
    if defect == "no-runner":
        report["files"] = {
            name: value
            for name, value in report["files"].items()
            if not name.startswith(("tools/", "scripts/e2e/regional/"))
        }
    elif defect == "no-branch-counter":
        first["summary"].pop("covered_branches")
    else:
        first["summary"]["covered_branches"] = 101
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(objectives.CoverageGateError):
        objectives.require_ci_coverage(path, root=root, config=config)
