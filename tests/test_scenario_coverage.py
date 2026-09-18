from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from pydantic import ValidationError

from tools import pytest_result_identity
from tools import scenario_coverage as coverage
from tools.scenario_requirements import (
    Check,
    Requirement,
    load_requirements,
    local_file,
    validate_check,
)
from tools.scenario_test_evidence import PytestEvidence

CASE_ID = "GF-SAMPLE-001"
SELECTOR = "tests/test_sample.py::test_result"
CATALOG = {CASE_ID: {"id": CASE_ID, "automation": "pytest", "pytest_nodeid": SELECTOR}}
PHASES = {"setup": "passed", "call": "passed", "teardown": "passed"}


def passing_evidence(*nodeids: str) -> PytestEvidence:
    records = {
        pytest_result_identity.normalized_pytest_nodeid(nodeid, root=coverage.ROOT): {
            "status": "PASS",
            "phases": PHASES,
        }
        for nodeid in nodeids
    }
    return PytestEvidence(
        pytest_result_identity.PytestReceipt(records, frozenset(records)), coverage.ROOT
    )


@pytest.fixture
def requirement() -> dict:
    return {
        "id": "GF-REQ-SAMPLE",
        "family": "security",
        "title": "The denied request cannot mutate another cluster.",
        "source": ["src/sample.py"],
        "covers": ["operation:SAMPLE"],
        "conditions": ["normal", "safety"],
        "assertions": ["A foreign identity changes no state."],
        "cases": [CASE_ID],
        "checks": [{"nodeid": SELECTOR}],
        "critical": True,
    }


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    for filename, contents in {
        "src/sample.py": "STATE = 1\n",
        "tests/test_sample.py": "def test_result():\n    assert True\n",
        "scripts/e2e/run_sample.py": "def main():\n    return 0\n",
    }.items():
        path = tmp_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")
    return tmp_path


def load(root: Path, requirements: list[dict], catalog=None):
    path = root / "requirements.yaml"
    path.write_text(
        yaml.safe_dump({"schema_version": 1, "requirements": requirements}),
        encoding="utf-8",
    )
    return load_requirements(
        path,
        root=root,
        catalog=CATALOG if catalog is None else catalog,
        inventory={"operation:SAMPLE"},
    )


def test_bound_requirement_remains_distinct_from_execution(
    repository, requirement
) -> None:
    parsed = load(repository, [requirement])
    result = coverage.build_report(
        parsed, catalog=CATALOG, evidence=passing_evidence(), identity="source"
    )
    assert result["mechanisms"]["implemented"]["percent"] == 100, (
        "reviewed code bindings exist"
    )
    assert result["target_met"]["local_verified"] is False, (
        "files are not test execution"
    )
    assert result["live_verification"]["status"] == "NOT_MEASURED", (
        "local data never claims LIVE"
    )
    assert result["requirements"][0]["case_digests"], (
        "case definition changes invalidate bindings"
    )


@pytest.mark.parametrize(
    "case", [{**CATALOG[CASE_ID], "evidence": None}, CATALOG[CASE_ID]]
)
def test_optional_catalog_evidence_is_not_required_for_a_design_mapping(
    repository, requirement, case
) -> None:
    assert load(repository, [requirement], catalog={CASE_ID: case}), (
        "a requirement may exist before any live evidence"
    )


@pytest.mark.parametrize(
    "problem", ["id", "scope", "source", "case", "test", "registry"]
)
def test_duplicate_and_dangling_bindings_are_rejected(
    repository, requirement, problem
) -> None:
    rows = [requirement]
    if problem == "id":
        rows.append(dict(requirement))
    elif problem == "scope":
        rows.append({**requirement, "id": "GF-REQ-RENAMED"})
    elif problem == "source":
        requirement["source"] = ["src/missing.py"]
    elif problem == "case":
        requirement["cases"] = ["GF-UNKNOWN"]
    elif problem == "test":
        requirement["checks"] = [{"nodeid": SELECTOR + "_missing"}]
    else:
        requirement["covers"] = ["operation:UNKNOWN"]
    with pytest.raises(ValueError):
        load(repository, rows)


@pytest.mark.parametrize("case_id", [CASE_ID, "GF-REGIONAL-DESTR-004"])
def test_retired_or_do_not_run_cases_never_count_as_implementation(
    repository, requirement, case_id
) -> None:
    requirement["cases"] = [case_id]
    case = {**CATALOG[CASE_ID], "id": case_id, "evidence": {"verdict": "SUPERSEDED"}}
    with pytest.raises(ValueError, match="retired"):
        load(repository, [requirement], catalog={case_id: case})


@pytest.mark.parametrize(
    "change",
    [
        {"checks": []},
        {"cases": []},
        {"level": "regional"},
        {"critical": 1},
        {"unexpected": "claim"},
        {"conditions": ["normal", "normal"]},
        {"checks": [{"nodeid": SELECTOR}, {"nodeid": SELECTOR}]},
    ],
)
def test_incomplete_implementation_requires_an_explicit_gap(
    requirement, change
) -> None:
    with pytest.raises(ValidationError):
        Requirement.model_validate({**requirement, **change})


def test_gap_stays_in_the_denominator_even_when_tests_pass(
    repository, requirement
) -> None:
    requirement["gap"] = "The deployed recovery safeguard has not been implemented."
    parsed = load(repository, [requirement])
    evidence = passing_evidence(SELECTOR)
    result = coverage.build_report(
        parsed, catalog=CATALOG, evidence=evidence, identity="source"
    )
    assert result["mechanisms"]["implemented"] == {
        "covered": 0,
        "required": 1,
        "percent": 0,
    }, "passing local proxy tests cannot erase an implementation gap"
    assert result["critical"]["local_verified"]["covered"] == 0, (
        "critical gaps cannot pass"
    )


def test_critical_and_family_gaps_cannot_hide_in_a_large_majority(
    requirement, monkeypatch
) -> None:
    monkeypatch.setattr(
        coverage,
        "load_xid_policy",
        lambda: SimpleNamespace(catalog_rules=[SimpleNamespace(xid=1)]),
    )
    good = Requirement.model_validate(requirement)
    gap = Requirement.model_validate(
        {
            **requirement,
            "id": "GF-REQ-MISSING",
            "family": "recovery",
            "gap": "No independent recovery after reboot.",
        }
    )
    evidence = passing_evidence(SELECTOR)
    result = coverage.build_report(
        [good] * 99 + [gap], catalog=CATALOG, evidence=evidence, identity="source"
    )
    assert result["mechanisms"]["local_verified"]["percent"] == 99, (
        "the raw percentage alone passes"
    )
    assert result["target_met"]["local_verified"] is False, (
        "family and critical gates detect the gap"
    )
    assert result["fault_rules"]["designed"]["percent"] == 0, (
        "missing XID cases stay in their inventory"
    )


def test_pinned_fault_rule_inventory_does_not_depend_on_case_count(monkeypatch) -> None:
    monkeypatch.setattr(
        coverage,
        "load_xid_policy",
        lambda: SimpleNamespace(catalog_rules=[SimpleNamespace(xid=31)]),
    )
    case = {**CATALOG[CASE_ID], "id": "GF-XID-KMSG-031"}
    evidence = passing_evidence(SELECTOR)
    report = coverage.fault_rule_coverage({case["id"]: case}, evidence)
    assert report["local_verified"] == {"covered": 1, "required": 2, "percent": 50}, (
        "the missing B200 requirement is counted instead of disappearing"
    )


@pytest.mark.parametrize("rows", [[], [{"critical": False}]])
def test_empty_or_safety_free_requirements_are_refused(rows, requirement) -> None:
    items = [Requirement.model_validate({**requirement, **row}) for row in rows]
    with pytest.raises(ValueError, match="critical safety"):
        coverage.build_report(
            items, catalog=CATALOG, evidence=passing_evidence(), identity="source"
        )


@pytest.mark.parametrize("reference", ["../outside.py", "/absolute.py", "missing.py"])
def test_external_or_missing_references_are_refused(repository, reference) -> None:
    with pytest.raises(ValueError):
        local_file(repository, reference)


def test_symlinks_and_invalid_selectors_are_refused(repository) -> None:
    path = repository / "src/alias.py"
    path.symlink_to(repository / "src/sample.py")
    with pytest.raises(ValueError, match="regular file"):
        local_file(repository, "src/alias.py")
    with pytest.raises(ValueError, match="selector"):
        validate_check(repository, Check(nodeid="src/sample.py"))
    with pytest.raises(ValueError, match="exactly one"):
        validate_check(repository, Check(nodeid=SELECTOR + "[a]", expected=2))


def test_regional_runner_reference_is_checked(repository, requirement) -> None:
    requirement.update(level="regional", runners=["scripts/e2e/run_sample.py"])
    assert load(repository, [requirement]), (
        "an explicit existing runner satisfies the binding"
    )
    requirement["runners"] = ["src/sample.py"]
    with pytest.raises(ValueError, match="outside acceptance"):
        load(repository, [requirement])


def test_duplicate_yaml_keys_and_boolean_version_are_refused(repository) -> None:
    path = repository / "requirements.yaml"
    for contents in (
        "schema_version: 1\nschema_version: 1\nrequirements: []\n",
        "schema_version: true\nrequirements: []\n",
    ):
        path.write_text(contents, encoding="utf-8")
        with pytest.raises(ValueError):
            load_requirements(path, root=repository, catalog=CATALOG, inventory=set())


def test_unparseable_test_source_is_a_controlled_mapping_failure(repository) -> None:
    (repository / "tests/test_sample.py").write_text(
        "def test_result(:\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="cannot be parsed"):
        validate_check(repository, Check(nodeid=SELECTOR))


def test_an_empty_fault_rule_inventory_cannot_report_full_coverage(monkeypatch) -> None:
    monkeypatch.setattr(
        coverage, "load_xid_policy", lambda: SimpleNamespace(catalog_rules=[])
    )
    with pytest.raises(ValueError, match="denominator is empty"):
        coverage.fault_rule_coverage({}, passing_evidence())


def test_module_entrypoint_refuses_invalid_requirements_without_running_cases(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        pytest_result_identity, "source_identity", lambda root: "source"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["scenario-coverage", "--requirements", str(tmp_path / "missing.yaml")],
    )
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(Path(coverage.__file__)), run_name="__main__")
    assert error.value.code == 2, (
        "an invalid requirements path does not start an acceptance runner"
    )


@pytest.mark.parametrize("enforce", [None, "local_verified"])
def test_cli_only_reports_and_never_runs_acceptance(
    repository, requirement, monkeypatch, capsys, enforce
) -> None:
    parsed = load(repository, [requirement])
    monkeypatch.setattr(coverage, "ROOT", repository)
    monkeypatch.setattr(coverage, "source_identity", lambda root: "source")
    monkeypatch.setattr(coverage, "load_catalog", lambda path: list(CATALOG.values()))
    monkeypatch.setattr(coverage, "load_requirements", lambda *args, **kwargs: parsed)
    arguments = [] if enforce is None else ["--require-stage", enforce]
    assert coverage.main(arguments) == int(enforce is not None), (
        "explicit stage gates retain missing execution"
    )
    result = json.loads(capsys.readouterr().out)
    assert result["live_verification"]["status"] == "NOT_MEASURED", (
        "reporting never executes LIVE"
    )


def test_cli_rejects_source_drift_during_audit(
    repository, requirement, monkeypatch
) -> None:
    parsed = load(repository, [requirement])
    identities = iter(["before", "after"])
    monkeypatch.setattr(coverage, "ROOT", repository)
    monkeypatch.setattr(coverage, "source_identity", lambda root: next(identities))
    monkeypatch.setattr(coverage, "load_catalog", lambda path: list(CATALOG.values()))
    monkeypatch.setattr(coverage, "load_requirements", lambda *args, **kwargs: parsed)
    with pytest.raises(SystemExit) as error:
        coverage.main([])
    assert error.value.code == 2, (
        "mixed source cannot produce a current scenario report"
    )
