#!/usr/bin/env python3
"""Run the 32 general-fault software injection groups without live credentials."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.general_fault_injections import INJECTIONS  # noqa: E402
from tools.run_fault_test_cases import load_catalog, main as run_cases  # noqa: E402


def select(numbers):
    wanted = set()
    for value in numbers:
        value = value.removeprefix("GF-GENERAL-E2E-")
        try:
            number = int(value)
        except ValueError as error:
            raise ValueError(f"invalid case: {value}") from error
        if not 1 <= number <= 32:
            raise ValueError(f"case must be between 001 and 032: {value}")
        wanted.add(number)
    return [item for item in INJECTIONS if not wanted or item.number in wanted]


def build_catalog(selected, manual):
    checks = {}
    bindings = {}
    for injection in selected:
        parent = manual[f"GF-GENERAL-E2E-{injection.number:03}"]
        bindings[injection.number] = []
        for nodeid in injection.checks:
            if nodeid not in checks:
                identifier = f"GF-INJECT-CHECK-{len(checks) + 1:03}"
                checks[nodeid] = {
                    "id": identifier,
                    "title": parent["title"],
                    "category": "general-fault-injection",
                    "level": "integration",
                    "risk": "non-destructive",
                    "automation": "pytest",
                    "problem": parent["problem"],
                    "injection": injection.source,
                    "expected": [
                        "All selected behavioral assertions pass; skips are not success.",
                        "Local fixtures only; this does not certify physical recovery.",
                    ],
                    "pytest_nodeid": nodeid,
                }
            bindings[injection.number].append(checks[nodeid]["id"])
    return list(checks.values()), bindings


def summarize(selected, manual, bindings, result):
    by_id = {item["id"]: item for item in result["results"]}
    groups = []
    for injection in selected:
        children = [by_id[identifier] for identifier in bindings[injection.number]]
        passed = all(child["status"] == "PASS" for child in children)
        groups.append(
            {
                "case_id": f"GF-GENERAL-E2E-{injection.number:03}",
                "title": manual[f"GF-GENERAL-E2E-{injection.number:03}"]["title"],
                "status": "PASS" if passed else "FAIL",
                "proof_scope": "software",
                "coverage": "continuous-primary-variant"
                if injection.continuous
                else "segmented-checks",
                "raw_injection_boundary": injection.source,
                "checks": [child["id"] for child in children],
                "physical_acceptance": "NOT_RUN",
                "limitations": (
                    "Primary variants only, not all variants in the manual acceptance specification."
                    if injection.continuous
                    else "Separate source/policy/adapter tests; NOT a continuous end-to-end recovery proof."
                ),
            }
        )
    return {
        "schema_version": 1,
        "report_type": "general-fault-injection",
        "source": result.get("environment"),
        "summary": {
            "total": len(groups),
            "pass": sum(item["status"] == "PASS" for item in groups),
        },
        "live_actions_executed": False,
        "cases": groups,
        "checks_report": "checks.json",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", action="append", default=[], help="001..032; repeatable; default all"
    )
    parser.add_argument("--list", action="store_true")
    parser.add_argument(
        "--output", type=Path, help="new output directory; existing directories refused"
    )
    parser.add_argument(
        "--require-continuous",
        action="store_true",
        help="fail if selected groups only have segmented checks",
    )
    args = parser.parse_args(argv)
    try:
        selected = select(args.case)
    except ValueError as error:
        parser.error(str(error))
    manual = {
        case["id"]: case
        for case in load_catalog(ROOT / "testcases/fault-scenarios.yaml")
    }
    if args.list:
        for item in selected:
            label = (
                "continuous-primary-variant" if item.continuous else "segmented-checks"
            )
            print(
                f"{item.number:03} {label}: {manual[f'GF-GENERAL-E2E-{item.number:03}']['title']}"
            )
        return 0
    output = (
        args.output.resolve()
        if args.output
        else Path(tempfile.mkdtemp(prefix="gfrs-general-injection-"))
    )
    if args.output:
        output.mkdir(parents=True, exist_ok=False)
    cases, bindings = build_catalog(selected, manual)
    catalog = output / "checks.yaml"
    # JSON is a YAML subset and avoids a second catalog serializer.
    catalog.write_text(
        json.dumps(
            {"schema_version": 1, "test_cases": cases}, indent=2, ensure_ascii=True
        )
        + "\n"
    )
    original = Path.cwd()
    try:
        os.chdir(ROOT)
        exit_code = run_cases(
            [
                "--catalog",
                str(catalog),
                "--batch-pytest",
                "--report",
                str(output / "checks.json"),
            ]
        )
    finally:
        os.chdir(original)
    result = json.loads((output / "checks.json").read_text())
    report = summarize(selected, manual, bindings, result)
    report["strict_coverage_satisfied"] = all(item.continuous for item in selected)
    (output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n"
    )
    print(f"Report: {output / 'report.json'}")
    if args.require_continuous and not report["strict_coverage_satisfied"]:
        return 1
    return exit_code or (0 if report["summary"]["pass"] == len(selected) else 1)


if __name__ == "__main__":
    raise SystemExit(main())
