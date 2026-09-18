"""Check declared scenario variants against isolated pytest discovery."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from tools.run_fault_test_cases import build_isolated_environment, load_catalog
from tools.scenario_requirements import load_requirements

ROOT = Path(__file__).resolve().parents[1]


def test_scenario_check_parameter_counts_match_actual_collection(
    tmp_path: Path,
) -> None:
    catalog = {
        case["id"]: case
        for case in load_catalog(ROOT / "testcases/fault-scenarios.yaml")
    }
    requirements = load_requirements(
        ROOT / "testcases/scenario-requirements.yaml", root=ROOT, catalog=catalog
    )
    expected: dict[str, int] = {}
    for requirement in requirements:
        for check in requirement.checks:
            previous = expected.setdefault(check.nodeid, check.expected)
            assert previous == check.expected, (
                f"requirements disagree on the parameter count for {check.nodeid}"
            )

    report_path = tmp_path / "collection.json"
    environment = build_isolated_environment()
    environment.update(
        PYTHONPATH=os.pathsep.join((str(ROOT / "src"), str(ROOT))),
        PYTHONDONTWRITEBYTECODE="1",
        PYTEST_ADDOPTS="",
        PYTEST_GPU_FAULT_CASE_REPORT=str(report_path),
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-n",
            "0",
            "-o",
            "addopts=",
            "-p",
            "tools.pytest_case_reporter",
            *sorted(expected),
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    session = report["session"]
    assert session["source_identity"] == report["source_identity"], (
        "scenario discovery must bind one unchanged source snapshot"
    )
    assert session["collection_errors"] == session["collection_skips"] == []
    assert report["records"] == {}, "discovery must not execute any test"
    collected = set(session["collected_nodeids"])
    discovered = set(session["discovered_nodeids"])
    assert collected <= discovered, "selected tests must belong to discovery"
    for selector, count in expected.items():
        generic = "[" not in selector.partition("::")[2]
        matches = {
            nodeid
            for nodeid in discovered
            if nodeid == selector or (generic and nodeid.startswith(selector + "["))
        }
        assert len(matches) == count, (
            f"{selector}: declared {count} variants, discovered {len(matches)}; "
            "update the reviewed scenario binding when parameterization changes"
        )
        assert matches <= collected, f"scenario selection omitted part of {selector}"
