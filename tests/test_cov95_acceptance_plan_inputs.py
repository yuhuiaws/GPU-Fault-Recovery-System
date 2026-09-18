from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from tests._cov95_acceptance_plan import RETIRED, A, B, PlanInputs, catalog_case
from tools.regional_acceptance_plan import ExecutionPolicy, FailureScope


def replace_field(document: Any, path: tuple[Any, ...], value: Any) -> None:
    for key in path[:-1]:
        document = document[key]
    document[path[-1]] = value


def test_valid_plan_retains_order_retirement_and_transitive_failure_scope(
    tmp_path: Path,
) -> None:
    plan = PlanInputs(tmp_path).compile()
    assert plan.case_ids == (A, B, RETIRED)
    assert plan.execution_order == (A, B)
    assert plan.graph[B] == (A,)
    assert plan.blocked_case_ids({A}) == (B,)
    assert plan.case(RETIRED).blocked is True
    assert plan.as_dict()["do_not_run_case_ids"] == [RETIRED]
    with pytest.raises(KeyError):
        plan.case("unknown")
    with pytest.raises(ValueError, match="unknown failed case IDs"):
        plan.blocked_case_ids({"unknown"})


@pytest.mark.parametrize(
    "path,value,problem",
    [
        (("schema_version",), 2, "schema_version must be 1"),
        (("schema_version",), True, "must be an integer"),
        (("phases",), {}, "must be a list"),
        (("phases", 0), [], "must be a mapping"),
        (("phases", 0, "sequence"), -1, "must not be negative"),
        (("phases", 0, "sequence"), 1, "contiguous"),
        (("phases", 0, "name"), "", "non-empty string"),
        (
            ("phases", 0, "entries"),
            [{"case": "not-regional"}],
            "not a regional case ID",
        ),
        (("phases", 0, "entries"), [{"unknown": A}], "exactly case or range"),
        (
            ("phases", 0, "entries"),
            [{"case": A, "predecessor": B}, {"case": B}],
            "does not run before",
        ),
        (
            ("phases", 0, "entries"),
            [{"case": A, "predecessor": "GF-REGIONAL-CAP-099"}],
            "not an ordered case",
        ),
        (
            ("phases", 0, "entries"),
            [{"range": {"prefix": "bad-prefix", "start": 1, "end": 2}}],
            "prefix is invalid",
        ),
        (
            ("phases", 0, "entries"),
            [{"range": {"prefix": "BOOT", "start": 0, "end": 2}}],
            "range is invalid",
        ),
        (
            ("phases", 0, "entries"),
            [{"range": {"prefix": "BOOT", "start": 2, "end": 1}}],
            "range is invalid",
        ),
        (("do_not_run",), [], "must declare its DO_NOT_RUN"),
    ],
)
def test_invalid_order_is_rejected_before_any_plan_can_be_used(
    tmp_path: Path, path: tuple[Any, ...], value: Any, problem: str
) -> None:
    inputs = PlanInputs(tmp_path)
    replace_field(inputs.order, path, value)
    inputs.write()
    with pytest.raises(ValueError, match=problem):
        inputs.compile()


@pytest.mark.parametrize(
    "fault,problem",
    [
        ("duplicate-phase", "duplicate regional phase sequence"),
        ("duplicate-case", "duplicate regional execution order"),
        ("unknown-field", "unknown fields"),
        ("nonstring-key", "keys must be strings"),
        ("unknown-case", "unknown regional case IDs"),
        ("unindexed-case", "missing from execution order"),
        ("retirement-mismatch", "exactly match"),
        ("retired-pytest", "must be manual"),
    ],
)
def test_order_and_catalog_must_cover_one_unambiguous_inventory(
    tmp_path: Path, fault: str, problem: str
) -> None:
    inputs = PlanInputs(tmp_path)
    if fault == "duplicate-phase":
        inputs.order["phases"].append(copy.deepcopy(inputs.order["phases"][0]))
    elif fault == "duplicate-case":
        inputs.order["phases"][0]["entries"].append({"case": A})
    elif fault == "unknown-field":
        inputs.order["extra"] = True
    elif fault == "nonstring-key":
        inputs.order[17] = "invalid"
    elif fault == "unknown-case":
        inputs.catalog["test_cases"].pop(0)
    elif fault == "unindexed-case":
        inputs.catalog["test_cases"].append(catalog_case("GF-REGIONAL-CAP-099"))
    elif fault == "retirement-mismatch":
        inputs.catalog["test_cases"][0]["evidence"] = {"verdict": "SUPERSEDED"}
    else:
        inputs.catalog["test_cases"][-1].update(
            automation="pytest",
            pytest_nodeid="tests/test_case_scheduler.py::test_results_remain_in_declared_order",
        )
    inputs.write()
    with pytest.raises(ValueError, match=problem):
        inputs.compile()


@pytest.mark.parametrize(
    "field,value,problem",
    [
        ("automation", "unknown", "must be one of"),
        ("automation", "pytest", "lacks pytest_nodeid"),
        ("automation", "command", "lacks command"),
        ("command", [], "command is empty"),
        ("command", "python", "must be a list"),
        ("procedure", None, "lacks procedure"),
        ("expected", [], "must not be empty"),
        ("expected", " ", "non-empty string"),
        ("execution", {"parallel_safe": 1}, "must be a boolean"),
        ("execution", {"parallel_safe": True}, "requires locks"),
        ("execution", {"parallel_safe": False, "depends_on": [A]}, "depend on itself"),
        (
            "execution",
            {"parallel_safe": False, "depends_on": [B, B]},
            "duplicate case IDs",
        ),
        (
            "execution",
            {"parallel_safe": False, "failure_scope": "unknown"},
            "must be one of",
        ),
    ],
)
def test_catalog_execution_metadata_is_not_inferred_from_malformed_values(
    tmp_path: Path, field: str, value: Any, problem: str
) -> None:
    inputs = PlanInputs(tmp_path)
    inputs.catalog["test_cases"][0][field] = value
    inputs.write()
    with pytest.raises(ValueError, match=problem):
        inputs.compile()


@pytest.mark.parametrize("fault", ["schema", "duplicate", "ambiguous-proxy"])
def test_catalog_schema_and_proxy_identity_are_unambiguous(
    tmp_path: Path, fault: str
) -> None:
    inputs = PlanInputs(tmp_path)
    if fault == "schema":
        inputs.catalog["schema_version"] = 2
    elif fault == "duplicate":
        inputs.catalog["test_cases"].append(
            copy.deepcopy(inputs.catalog["test_cases"][0])
        )
    else:
        inputs.catalog["test_cases"][0].update(
            related_pytest="tests/example.py::test_case", gate="scripts/example.py"
        )
    inputs.write()
    with pytest.raises(ValueError, match="schema_version|duplicate|ambiguous"):
        inputs.compile()


@pytest.mark.parametrize("target", ["order", "catalog"])
def test_compiler_refuses_input_drift_between_read_and_final_hash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, target: str
) -> None:
    inputs = PlanInputs(tmp_path)
    selected = inputs.order_path if target == "order" else inputs.catalog_path
    read = Path.read_bytes
    reads = 0

    def changing(path: Path) -> bytes:
        nonlocal reads
        data = read(path)
        if path == selected:
            reads += 1
            if reads == 2:
                return data + b"\n# changed input\n"
        return data

    monkeypatch.setattr(Path, "read_bytes", changing)
    with pytest.raises(ValueError, match="changed while compiling"):
        inputs.compile()
    assert reads == 2


@pytest.mark.parametrize(
    "text", ["[broken", "schema_version: 1\nschema_version: 1\n", "[]"]
)
def test_invalid_or_duplicate_yaml_does_not_compile(tmp_path: Path, text: str) -> None:
    inputs = PlanInputs(tmp_path)
    inputs.order_path.write_text(text)
    with pytest.raises(ValueError, match="YAML|mapping"):
        inputs.compile()


def test_missing_plan_inputs_and_unknown_modes_fail_closed(tmp_path: Path) -> None:
    inputs = PlanInputs(tmp_path)
    with pytest.raises(ValueError, match="unsupported regional acceptance mode"):
        inputs.compile(mode="unknown")
    inputs.order_path.unlink()
    with pytest.raises(ValueError, match="cannot hash"):
        inputs.compile()


def test_execution_flags_cannot_combine_read_only_and_repair() -> None:
    with pytest.raises(ValueError, match="read-only"):
        ExecutionPolicy(parallel_safe=True, read_only=True, repair_allowed=True)
    with pytest.raises(ValueError, match="failure_scope=case"):
        ExecutionPolicy(
            parallel_safe=True, collect_all=True, failure_scope=FailureScope.GLOBAL
        )
