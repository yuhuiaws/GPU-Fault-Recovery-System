"""Audit requirement-to-case coverage; never execute an acceptance runner."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from gpu_fault.policy import load_xid_policy
from tools.pytest_result_identity import source_identity
from tools.run_fault_test_cases import case_definition_digest, load_catalog
from tools.scenario_requirements import Check, Requirement, load_requirements
from tools.scenario_test_evidence import PytestEvidence, load_test_evidence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REQUIREMENTS = ROOT / "testcases/scenario-requirements.yaml"
TARGET = 95
STAGES = ("designed", "implemented", "local_verified")


def _rate(rows: Sequence[Mapping[str, Any]], stage: str) -> dict[str, Any]:
    if not rows:
        raise ValueError("scenario coverage denominator is empty")
    covered = sum(row[stage] is True for row in rows)
    return {
        "covered": covered,
        "required": len(rows),
        "percent": 100 * covered / len(rows),
    }


def _requirement_row(
    requirement: Requirement,
    catalog: Mapping[str, Mapping[str, Any]],
    evidence: PytestEvidence,
) -> dict[str, Any]:
    implemented = requirement.gap is None
    return {
        **requirement.model_dump(mode="json"),
        "case_digests": {
            case_id: case_definition_digest(catalog[case_id])
            for case_id in requirement.cases
        },
        "designed": bool(requirement.cases),
        "implemented": implemented,
        "local_verified": implemented
        and all(evidence.confirms(check) for check in requirement.checks),
        "unverified_checks": [
            check.model_dump()
            for check in requirement.checks
            if not evidence.confirms(check)
        ],
        "live_verified": None,
    }


def fault_rule_coverage(
    catalog: Mapping[str, Mapping[str, Any]], evidence: PytestEvidence
) -> dict[str, Any]:
    rows = []
    for rule in load_xid_policy().catalog_rules:
        for prefix in ("GF-XID-KMSG-", "GF-XID-KMSG-B200-"):
            case_id = f"{prefix}{rule.xid:03d}"
            case = catalog.get(case_id)
            designed = case is not None
            implemented = case is not None and case["automation"] == "pytest"
            rows.append(
                {
                    "case_id": case_id,
                    "designed": designed,
                    "implemented": implemented,
                    "local_verified": case is not None
                    and implemented
                    and evidence.confirms(Check(nodeid=case["pytest_nodeid"])),
                }
            )
    return {**{stage: _rate(rows, stage) for stage in STAGES}, "rules": rows}


def build_report(
    requirements: Sequence[Requirement],
    *,
    catalog: Mapping[str, Mapping[str, Any]],
    evidence: PytestEvidence,
    identity: str,
) -> dict[str, Any]:
    rows = [_requirement_row(item, catalog, evidence) for item in requirements]
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        families[row["family"]].append(row)
    critical = [row for row in rows if row["critical"]]
    if not critical:
        raise ValueError("scenario requirements omit critical safety checks")
    family_rates = {
        family: {stage: _rate(items, stage) for stage in STAGES}
        for family, items in sorted(families.items())
    }
    overall = {stage: _rate(rows, stage) for stage in STAGES}
    critical_rates = {stage: _rate(critical, stage) for stage in STAGES}
    fault_rules = fault_rule_coverage(catalog, evidence)
    target_met = {
        stage: overall[stage]["percent"] >= TARGET
        and critical_rates[stage]["percent"] == 100
        and fault_rules[stage]["percent"] >= TARGET
        and all(value[stage]["percent"] >= TARGET for value in family_rates.values())
        for stage in STAGES
    }
    return {
        "schema_version": 1,
        "source_identity": identity,
        "target_percent": TARGET,
        "target_met": target_met,
        "mechanisms": overall,
        "families": family_rates,
        "critical": critical_rates,
        "fault_rules": fault_rules,
        "live_verification": {
            "status": "NOT_MEASURED",
            "reason": "Local pytest results do not prove a deployed or physical outcome.",
            "regional_requirements": sum(row["level"] == "regional" for row in rows),
        },
        "requirements": rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requirements", type=Path, default=DEFAULT_REQUIREMENTS)
    parser.add_argument("--pytest-results", type=Path, action="append", default=[])
    parser.add_argument("--require-stage", choices=STAGES)
    options = parser.parse_args(argv)
    try:
        identity = source_identity(ROOT)
        catalog = {
            case["id"]: case
            for case in load_catalog(ROOT / "testcases/fault-scenarios.yaml")
        }
        requirements = load_requirements(
            options.requirements, root=ROOT, catalog=catalog
        )
        evidence = load_test_evidence(
            options.pytest_results,
            expected_identity=identity,
            now=datetime.now(timezone.utc),
            root=ROOT,
        )
        result = build_report(
            requirements, catalog=catalog, evidence=evidence, identity=identity
        )
        if source_identity(ROOT) != identity:
            raise ValueError("source changed while auditing scenario coverage")
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"scenario coverage refused: {exc}\n")
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return int(
        options.require_stage is not None
        and not result["target_met"][options.require_stage]
    )


if __name__ == "__main__":
    raise SystemExit(main())
