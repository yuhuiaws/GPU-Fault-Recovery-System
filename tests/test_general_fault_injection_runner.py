"""Runner receipts must not turn partial coverage or skipped checks into E2E PASS."""

from pathlib import Path

import pytest

from scripts.e2e.run_general_fault_injections import build_catalog, select, summarize
from tools.general_fault_injections import INJECTIONS
from tools.run_fault_test_cases import load_catalog

ROOT = Path(__file__).resolve().parents[1]


def manual_catalog():
    return {
        case["id"]: case
        for case in load_catalog(ROOT / "testcases/fault-scenarios.yaml")
    }


def test_injection_catalog_covers_exactly_32_categories():
    assert [item.number for item in INJECTIONS] == list(range(1, 33))
    assert all(item.checks and item.source for item in INJECTIONS), (
        "every category needs executable injections, not just a title"
    )
    cases, bindings = build_catalog(INJECTIONS, manual_catalog())
    assert set(bindings) == set(range(1, 33))
    assert len({case["pytest_nodeid"] for case in cases}) == len(cases), (
        "shared checks run once, with explicit bindings to each category"
    )
    assert all(case["risk"] == "non-destructive" for case in cases), cases


@pytest.mark.parametrize("value", ["0", "33", "foo", "../main"])
def test_invalid_case_is_rejected_before_execution(value):
    with pytest.raises(ValueError):
        select([value])


def test_selection_is_stable_and_deduplicated():
    assert [item.number for item in select(["017", "1", "GF-GENERAL-E2E-017"])] == [
        1,
        17,
    ]


@pytest.mark.parametrize("status", ["FAIL", "BLOCKED", "NOT_RUN", "SKIPPED"])
def test_unsuccessful_receipt_cannot_be_counted_as_pass(status):
    selected = select(["001"])
    cases, bindings = build_catalog(selected, manual_catalog())
    result = {"results": [{"id": case["id"], "status": status} for case in cases]}
    report = summarize(selected, manual_catalog(), bindings, result)
    assert report["summary"] == {"total": 1, "pass": 0}
    assert report["cases"][0]["physical_acceptance"] == "NOT_RUN"


def test_segmented_success_does_not_claim_continuous_recovery():
    selected = select(["012"])
    cases, bindings = build_catalog(selected, manual_catalog())
    result = {"results": [{"id": case["id"], "status": "PASS"} for case in cases]}
    report = summarize(selected, manual_catalog(), bindings, result)
    assert report["cases"][0]["coverage"] == "segmented-checks"
    assert report["live_actions_executed"] is False, report


def test_nccl_continuous_primary_remains_software_only_and_requires_its_receipts():
    selected = select(["011"])
    cases, bindings = build_catalog(selected, manual_catalog())
    chain = [
        case
        for case in cases
        if "test_nccl_continuous_e2e.py::" in case["pytest_nodeid"]
    ]
    assert len(chain) == 14, chain
    result = {"results": [{"id": case["id"], "status": "PASS"} for case in cases]}
    report = summarize(selected, manual_catalog(), bindings, result)
    assert report["cases"][0]["coverage"] == "continuous-primary-variant", report
    assert report["cases"][0]["physical_acceptance"] == "NOT_RUN", report
    assert report["live_actions_executed"] is False, report
    result["results"][0]["status"] = "FAIL"
    assert (
        summarize(selected, manual_catalog(), bindings, result)["summary"]["pass"] == 0
    )
