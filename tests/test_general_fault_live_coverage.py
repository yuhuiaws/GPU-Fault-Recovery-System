"""Live case traceability, not a claim that any hardware experiment ran."""

from pathlib import Path

import pytest
import yaml

from tools.run_fault_test_cases import UniqueKeyLoader, load_catalog, select_cases
from tools.scenario_requirements import load_requirements

ROOT = Path(__file__).resolve().parents[1]
MATRIX = yaml.load(
    (ROOT / "testcases/general-live-coverage.yaml").read_text(), Loader=UniqueKeyLoader
)
CASES = {
    item["id"]: item for item in load_catalog(ROOT / "testcases/fault-scenarios.yaml")
}
PROCEDURE_PATH = "docs/components/general-fault-live-acceptance.md"
PROCEDURE = (ROOT / PROCEDURE_PATH).read_text()


def test_live_matrix_keeps_exactly_32_categories_and_reuses_environment_cases():
    assert set(MATRIX) == {"schema_version", "scope", "execution_verdict", "rows"}
    assert MATRIX["schema_version"] == 1
    assert MATRIX["scope"] == "general-fault-categories-32"
    assert MATRIX["execution_verdict"] == "NOT_RUN"
    assert [row["number"] for row in MATRIX["rows"]] == list(range(1, 33))
    assert sum(len(row["variants"]) for row in MATRIX["rows"]) >= 260, (
        "the reviewed required-variant denominator must not silently shrink"
    )
    expected = {
        f"GF-GENERAL-LIVE-{number:03}"
        for number in range(1, 33)
        if number not in {18, 23}
    }
    assert {key for key in CASES if key.startswith("GF-GENERAL-LIVE-")} == expected
    used = {case for row in MATRIX["rows"] for case in row["cases"]}
    assert used == expected | {f"GF-ENV-LIVE-{number:03}" for number in range(1, 13)}


@pytest.mark.parametrize("row", MATRIX["rows"], ids=lambda row: f"{row['number']:03}")
def test_every_category_has_bound_live_procedures_and_required_variants(row):
    assert set(row) == {"number", "cases", "proof_modes", "variants"}
    assert type(row["number"]) is int
    assert row["proof_modes"] and set(row["proof_modes"]) <= {
        "native",
        "live-replay",
        "vendor-lab",
    }, row
    assert len(row["proof_modes"]) == len(set(row["proof_modes"]))
    assert len(row["variants"]) >= 6, "do not collapse a fault class to one happy path"
    assert len(row["variants"]) == len(set(row["variants"]))
    assert all(isinstance(item, str) and item for item in row["variants"]), row
    assert len(row["cases"]) == len(set(row["cases"]))
    for identifier in row["cases"]:
        case = CASES[identifier]
        assert case["level"] == "end-to-end"
        assert case["automation"] == "manual"
        assert case["evidence"] == {"verdict": "NOT_RUN"}
        assert len(case["expected"]) >= 3
        assert case["risk"] not in {"non-destructive", "read-only-signal-replay"}, (
            "live prerequisites and effects must remain visible"
        )
        filename, anchor = case["procedure"].split("#")
        assert anchor == identifier.lower()
        assert f"### {identifier}\n" in (ROOT / filename).read_text()
    if row["number"] in {18, 23}:
        assert all(item.startswith("GF-ENV-LIVE-") for item in row["cases"]), row
    else:
        identifier = f"GF-GENERAL-LIVE-{row['number']:03}"
        assert row["cases"] == [identifier]
        section = PROCEDURE.split(f"### {identifier}\n", 1)[1].split("\n##", 1)[0]
        for variant in row["variants"]:
            assert f"`{variant}`" in section, f"{identifier} missing {variant}"


def test_high_risk_variants_and_requirement_gap_cannot_disappear():
    rows = {row["number"]: row for row in MATRIX["rows"]}
    assert {
        "fr_enabled",
        "fr_disabled",
        "simultaneous_nodes",
        "unreachable_node",
        "large_job",
        "few_candidates",
        "ambiguous_candidates",
        "different_process_groups",
    } <= set(rows[11]["variants"])
    assert {"query_failed", "proc_unreadable", "reboot_verified"} <= set(
        rows[12]["variants"]
    )
    assert {"hardware_chain_continues", "restart_withheld", "concurrent_claim"} <= set(
        rows[29]["variants"]
    )
    requirements = {
        item.id: item
        for item in load_requirements(
            ROOT / "testcases/scenario-requirements.yaml", root=ROOT, catalog=CASES
        )
    }
    requirement = requirements["GF-REQ-GENERAL-LIVE"]
    assert requirement.gap and requirement.level == "regional", requirement
    assert set(requirement.cases) == {
        case for row in rows.values() for case in row["cases"]
    }
    assert set(requirement.covers) == {
        f"fault:general-live-{number:03}" for number in range(1, 33)
    }


def test_live_specifications_are_not_selected_for_automatic_execution():
    categories = {"general-fault-live", "gpu-environment-live"}
    automatic = select_cases(
        list(CASES.values()),
        case_ids=[],
        categories=categories,
        levels=set(),
        include_manual=False,
        include_live=False,
    )
    assert automatic == [], "adding specifications must not dispatch live mutations"
    listed = select_cases(
        list(CASES.values()),
        case_ids=[],
        categories=categories,
        levels=set(),
        include_manual=True,
        include_live=False,
    )
    assert len(listed) == 42, (
        "thirty new procedures plus twelve reused environment cases"
    )
