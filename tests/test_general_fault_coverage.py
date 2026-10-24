"""Traceability checks, not evidence that the fault workflows ran."""

from pathlib import Path

import pytest

from tools.run_fault_test_cases import load_catalog
from tools.scenario_requirements import load_requirements

ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = ROOT / "docs/components/general-fault-e2e-audit.md"
AUDIT = AUDIT_PATH.read_text(encoding="utf-8")
CATALOG = {
    case["id"]: case for case in load_catalog(ROOT / "testcases/fault-scenarios.yaml")
}
REQUIREMENTS = {
    requirement.id: requirement
    for requirement in load_requirements(
        ROOT / "testcases/scenario-requirements.yaml", root=ROOT, catalog=CATALOG
    )
}


def table_rows(section):
    rows = [
        [cell.strip() for cell in line.split("|")[1:-1]]
        for line in section.splitlines()
        if line.startswith("|")
    ]
    return rows[2:]


def test_general_fault_audit_matches_every_documented_category():
    source = (ROOT / "docs/故障类别与处置动作总表.md").read_text(encoding="utf-8")
    general = source.split("## 一、通用故障类别（非 XID/SXID）", 1)[1].split(
        "### RecoveryAction", 1
    )[0]
    matrix = AUDIT.split("## 逐项清单", 1)[1].split("## 通用验收协议", 1)[0]
    documented = table_rows(general)
    audited = table_rows(matrix)
    assert len(documented) == len(audited) == 32, "do not shrink the denominator"
    assert [row[0] for row in documented] == [row[1] for row in audited], (
        "every documented category needs its own audit row"
    )
    assert [row[0] for row in audited] == [f"{index:03}" for index in range(1, 33)]
    assert {
        case_id for case_id in CATALOG if case_id.startswith("GF-GENERAL-E2E-")
    } == {f"GF-GENERAL-E2E-{index:03}" for index in range(1, 33)}


@pytest.mark.parametrize("number", range(1, 33), ids=lambda number: f"{number:03}")
def test_each_general_fault_has_an_explicit_unexecuted_e2e_requirement(number):
    suffix = f"{number:03}"
    case_id = f"GF-GENERAL-E2E-{suffix}"
    case = CATALOG[case_id]
    requirement = REQUIREMENTS[f"GF-REQ-GENERAL-{suffix}"]
    assert case["level"] == "end-to-end", case
    assert case["automation"] == "manual", "a specification is not an executed test"
    assert case["evidence"] == {"verdict": "NOT_RUN"}, case
    assert case["procedure"] == (
        f"docs/components/general-fault-e2e-audit.md#{case_id.lower()}"
    )
    assert requirement.cases == [case_id], requirement
    assert requirement.covers == [f"fault:general-{suffix}"], requirement
    assert requirement.gap, "unimplemented or unexecuted acceptance stays visible"
    section = AUDIT.split(f"### {case_id}\n", 1)[1].split("\n##", 1)[0]
    assert len(case["expected"]) >= 3, "detection alone is not the full contract"
    assert "缺口" in section or "整改" in section, section


def test_software_cases_have_separate_exact_behavioral_checks():
    requirement = REQUIREMENTS["GF-REQ-GENERAL-HOST-SOFTWARE"]
    assert requirement.gap is None, requirement
    assert requirement.cases == [
        "GF-GENERAL-SW-001",
        "GF-GENERAL-SW-002",
        "GF-GENERAL-SW-003",
    ]
    assert [check.expected for check in requirement.checks] == [6, 6, 5]
    assert [CATALOG[case]["pytest_nodeid"] for case in requirement.cases] == [
        check.nodeid for check in requirement.checks
    ]


def test_all_twelve_policy_changes_have_independent_software_requirements():
    changed = {
        "016",
        "017",
        "018",
        "019",
        "020",
        "022",
        "023",
        "024",
        "025",
        "027",
        "030",
        "031",
    }
    requirements = {
        key.removeprefix("GF-REQ-GENERAL-FIX-"): value
        for key, value in REQUIREMENTS.items()
        if key.startswith("GF-REQ-GENERAL-FIX-")
    }
    assert set(requirements) == changed, "no pending category may disappear"
    for requirement in requirements.values():
        assert requirement.gap is None and requirement.checks, requirement
        assert all(
            CATALOG[case]["automation"] == "pytest" for case in requirement.cases
        ), requirement
    matrix = AUDIT.split("## 逐项清单", 1)[1].split("## 通用验收协议", 1)[0]
    assert {row[0] for row in table_rows(matrix) if row[2] == "已改/待真机"} == changed
