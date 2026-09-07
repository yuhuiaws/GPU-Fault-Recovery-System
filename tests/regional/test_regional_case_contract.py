"""The formal predecessor chain has one source: the order file plus the catalog.

Every acceptance runner that hard-codes ``PREDECESSOR_CASE_ID`` (or the
``PREDECESSORS`` dict the collector runners use) must agree with
``regional_case_contract.formal_predecessor``. When the runner's evidence
predecessor is not its positional neighbour, the order file says so with an
explicit ``predecessor:``; the runner constant never wins on its own.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest
import yaml

from scripts.e2e.regional import regional_case_contract as contract

ROOT = Path(__file__).resolve().parents[2]
REGIONAL = ROOT / "scripts/e2e/regional"
CATALOG = ROOT / "testcases/fault-scenarios.yaml"
ORDER = ROOT / "testcases/regional-execution-order.yaml"
CASE_LITERAL = re.compile(
    r'^(CASE_ID|PREDECESSOR_CASE_ID)\s*=\s*"(GF-REGIONAL-[A-Z0-9-]+)"', re.MULTILINE
)
PREDECESSORS_DICT = re.compile(r"^PREDECESSORS\s*=\s*\{", re.MULTILINE)
# Runner constants that name a pytest wrapper as their predecessor. A pytest
# wrapper never writes ``cases/<id>/<id>.json``, so in formal scope these
# constants can never be satisfied; the contract already skips the wrapper.
# The runners are owned elsewhere -- when one is corrected, delete its row here
# and this test starts holding it to the contract like every other runner.
KNOWN_RUNNER_PREDECESSOR_DEFECTS = {
    "GF-REGIONAL-COLLECT-009": ("GF-REGIONAL-COLLECT-007", "GF-REGIONAL-COLLECT-005"),
    "GF-REGIONAL-NOTIFY-007": ("GF-REGIONAL-NOTIFY-006", "GF-REGIONAL-NOTIFY-005"),
}


def _catalog() -> dict[str, dict]:
    return {
        case["id"]: case
        for case in yaml.safe_load(CATALOG.read_text(encoding="utf-8"))["test_cases"]
        if case["id"].startswith("GF-REGIONAL-")
    }


def _runner_predecessors() -> dict[str, tuple[str, str]]:
    """``{case_id: (predecessor_id, source file)}`` from every runner module."""
    found: dict[str, tuple[str, str]] = {}
    for path in sorted(REGIONAL.glob("*.py")):
        if not (
            path.name.startswith("run_")
            or path.name.startswith("audit_")
            or path.name.endswith("_verdicts.py")
        ):
            continue
        text = path.read_text(encoding="utf-8")
        literals = dict(CASE_LITERAL.findall(text))
        if "CASE_ID" in literals and "PREDECESSOR_CASE_ID" in literals:
            found[literals["CASE_ID"]] = (literals["PREDECESSOR_CASE_ID"], path.name)
        if PREDECESSORS_DICT.search(text) is None:
            continue
        for node in ast.parse(text, filename=str(path)).body:
            if not isinstance(node, ast.Assign):
                continue
            if any(
                isinstance(target, ast.Name) and target.id == "PREDECESSORS"
                for target in node.targets
            ):
                for case_id, predecessor in ast.literal_eval(node.value).items():
                    found[str(case_id)] = (str(predecessor), path.name)
    return found


def test_every_runner_predecessor_matches_the_formal_contract() -> None:
    runners = _runner_predecessors()
    assert len(runners) >= 40, sorted(runners)
    mismatches = {
        case_id: (runner_value, contract.formal_predecessor(case_id), source)
        for case_id, (runner_value, source) in runners.items()
        if case_id not in KNOWN_RUNNER_PREDECESSOR_DEFECTS
        and contract.formal_predecessor(case_id) != runner_value
    }
    assert mismatches == {}, (
        "runner predecessor constants disagree with formal_predecessor(); give the "
        "order file an explicit `predecessor:` or fix the runner"
    )


def test_known_runner_predecessor_defects_are_still_present() -> None:
    runners = _runner_predecessors()
    for case_id, (stale, formal) in KNOWN_RUNNER_PREDECESSOR_DEFECTS.items():
        assert runners[case_id][0] == stale, (
            f"{case_id} runner no longer says {stale}; drop it from "
            "KNOWN_RUNNER_PREDECESSOR_DEFECTS"
        )
        assert contract.formal_predecessor(case_id) == formal
        assert contract.is_pytest_wrapper(_catalog()[stale]), (
            f"{stale} is no longer a pytest wrapper; the defect row is stale"
        )


def test_explicit_predecessors_are_the_documented_anchor_cases() -> None:
    # Each override names the case whose evidence the runner reads; the order
    # notes have said so in prose for every one of them.
    assert contract.explicit_predecessors() == {
        "GF-REGIONAL-BOOT-023": "GF-REGIONAL-BOOT-020",
        "GF-REGIONAL-HA-010": "GF-REGIONAL-HA-001",
        "GF-REGIONAL-DESTR-016": "GF-REGIONAL-DESTR-002",
        "GF-REGIONAL-DESTR-017": "GF-REGIONAL-DESTR-002",
        "GF-REGIONAL-DESTR-018": "GF-REGIONAL-DESTR-010",
        "GF-REGIONAL-DESTR-019": "GF-REGIONAL-DESTR-010",
        "GF-REGIONAL-DESTR-020": "GF-REGIONAL-DESTR-001",
        "GF-REGIONAL-DESTR-003": "GF-REGIONAL-DESTR-012",
        "GF-REGIONAL-DESTR-014": "GF-REGIONAL-DESTR-008",
        "GF-REGIONAL-HA-003": "GF-REGIONAL-DESTR-008",
        "GF-REGIONAL-E2E-002": "GF-REGIONAL-ISO-001",
        "GF-REGIONAL-COLLECT-015": "GF-REGIONAL-COLLECT-017",
    }
    ordered = contract.ordered_case_ids()
    for case_id, predecessor in contract.explicit_predecessors().items():
        assert ordered.index(predecessor) < ordered.index(case_id)
        assert not contract.is_pytest_wrapper(_catalog()[predecessor]), predecessor


def test_positional_fallback_skips_pytest_wrappers() -> None:
    catalog = _catalog()
    wrappers = set(contract.pytest_wrapper_case_ids())
    by_automation = {
        case_id for case_id, case in catalog.items() if case["automation"] == "pytest"
    }
    by_command = {
        case_id
        for case_id, case in catalog.items()
        if case["automation"] == "command"
        and case["command"][:3] == ["python3", "-m", "pytest"]
    }
    assert by_automation <= wrappers
    assert by_command <= wrappers
    assert wrappers == by_automation | by_command
    # The thirteen command cases that only wrap pytest, by name, so a
    # reclassification is a visible diff rather than a silent count change.
    assert by_command == {
        "GF-REGIONAL-BOOT-022",
        *(
            f"GF-REGIONAL-PREEMPT-{number:03d}"
            for number in (17, 21, 24, 25, 26, 28, 29, 31, 32, 33, 35)
        ),
    }
    # A live case behind a wrapper reads the last real evidence producer.
    assert contract.formal_predecessor("GF-REGIONAL-NOTIFY-007") == (
        "GF-REGIONAL-NOTIFY-005"
    )
    assert contract.formal_predecessor("GF-REGIONAL-COLLECT-009") == (
        "GF-REGIONAL-COLLECT-005"
    )
    assert contract.formal_predecessor("GF-REGIONAL-PREEMPT-037") == (
        "GF-REGIONAL-PREEMPT-036"
    )
    # Plain positional neighbours are untouched.
    assert contract.formal_predecessor("GF-REGIONAL-BOOT-011") is None
    assert contract.formal_predecessor("GF-REGIONAL-DESTR-002") == (
        "GF-REGIONAL-DESTR-001"
    )
    # The wrapper itself still has a predecessor, so its metadata resolves;
    # PREEMPT-014..033 are all wrappers, so the last evidence producer before
    # PREEMPT-035 is the live PREEMPT-012.
    assert contract.case_metadata("GF-REGIONAL-PREEMPT-035").predecessor == (
        "GF-REGIONAL-PREEMPT-012"
    )


def test_do_not_run_cases_are_outside_the_formal_chain() -> None:
    order = yaml.safe_load(ORDER.read_text(encoding="utf-8"))
    retired = {item["case"] for item in order["do_not_run"]}
    assert contract.do_not_run_case_ids() == retired
    assert retired.isdisjoint(contract.ordered_case_ids()), (
        f"DO_NOT_RUN cases leaked into the formal chain: "
        f"{sorted(retired & set(contract.ordered_case_ids()))}"
    )
    for case_id in sorted(retired):
        with pytest.raises(contract.RegionalCaseContractError, match="DO_NOT_RUN"):
            contract.formal_predecessor(case_id)
    # Nothing in the formal order depends on a retired case, positionally or
    # explicitly, so retiring a case never strands the one behind it.
    for case_id in contract.ordered_case_ids():
        predecessor = contract.formal_predecessor(case_id)
        assert predecessor not in retired, (case_id, predecessor)


def test_order_file_rejects_malformed_predecessors(tmp_path: Path) -> None:
    order = yaml.safe_load(ORDER.read_text(encoding="utf-8"))
    for mutate, message in (
        (
            lambda phases: phases[0]["entries"].append(
                {"case": "GF-REGIONAL-BOOT-011", "predecessor": "GF-REGIONAL-BOOT-023"}
            ),
            "duplicate",
        ),
        (
            lambda phases: phases[0]["entries"].__setitem__(
                1,
                {
                    "case": "GF-REGIONAL-BOOT-023",
                    "predecessor": "GF-REGIONAL-COLLECT-015",
                },
            ),
            "does not run before",
        ),
        (
            lambda phases: phases[0]["entries"].__setitem__(
                1,
                {"case": "GF-REGIONAL-BOOT-023", "predecessor": "GF-REGIONAL-NOPE-001"},
            ),
            "not in the formal",
        ),
        (
            lambda phases: phases[0]["entries"].__setitem__(
                0,
                {
                    "range": {"prefix": "BOOT", "start": 11, "end": 22},
                    "predecessor": "GF-REGIONAL-BOOT-001",
                },
            ),
            "range entries cannot",
        ),
    ):
        broken = yaml.safe_load(yaml.safe_dump(order, allow_unicode=True))
        mutate(broken["phases"])
        path = tmp_path / "order.yaml"
        path.write_text(yaml.safe_dump(broken, allow_unicode=True), encoding="utf-8")
        original = contract.ORDER_PATH
        contract.ORDER_PATH = path
        contract.expanded_order.cache_clear()
        try:
            with pytest.raises(contract.RegionalCaseContractError, match=message):
                contract.ordered_case_ids()
        finally:
            contract.ORDER_PATH = original
            contract.expanded_order.cache_clear()
