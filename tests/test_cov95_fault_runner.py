from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml

from tools import run_fault_test_cases as runner


@pytest.fixture(autouse=True)
def fixed_platform(monkeypatch):
    monkeypatch.setattr(runner.platform, "platform", lambda: "unit-test-platform")
    monkeypatch.setattr(runner, "source_identity", lambda root: "a" * 64)


def case(**changes):
    value = {
        "id": "GF-UNIT-001",
        "title": "Unit scenario",
        "category": "unit",
        "level": "unit",
        "risk": "non-destructive",
        "problem": "Unit condition",
        "injection": "Unit fixture",
        "expected": ["No external mutation"],
        "automation": "pytest",
        "pytest_nodeid": "tests/test_unit.py::test_case",
    }
    value.update(changes)
    return value


def write_receipt(environment, records, *, exitstatus=0):
    Path(environment[runner.PYTEST_BATCH_REPORT_ENV]).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_identity": "a" * 64,
                "session": {
                    "source_identity": "a" * 64,
                    "exitstatus": exitstatus,
                    "collected_nodeids": list(records),
                    "discovered_nodeids": list(records),
                },
                "records": records,
            }
        ),
        encoding="utf-8",
    )


def passed_record():
    return {
        "status": "PASS",
        "phases": {"setup": "passed", "call": "passed", "teardown": "passed"},
    }


@pytest.mark.parametrize("automation", ["pytest", "command"])
@pytest.mark.parametrize("summary", ["1 skipped", "1 xfailed", "1 passed, 1 skipped"])
def test_zero_exit_from_skipped_pytest_does_not_prove_case_success(
    monkeypatch, automation, summary
) -> None:
    selected = case(automation=automation)
    if automation == "command":
        selected["command"] = [
            runner.sys.executable,
            "-m",
            "pytest",
            "-q",
            selected["pytest_nodeid"],
        ]
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 0, stdout=summary
        ),
    )
    result = runner.run_case(selected, environment={})
    assert result["status"] == "FAIL", (
        "pytest exit zero cannot turn unexecuted case assertions into PASS"
    )


@pytest.mark.parametrize("automation", ["pytest", "command"])
@pytest.mark.parametrize("outcome", ["passed", "skipped", "failed", "partial"])
def test_fresh_pytest_case_uses_the_complete_structured_receipt(
    monkeypatch, automation, outcome
) -> None:
    selected = case(automation=automation)
    if automation == "command":
        selected["command"] = ["python3", "-m", "pytest", selected["pytest_nodeid"]]
    record = passed_record()
    if outcome == "partial":
        record["phases"].pop("teardown")
    elif outcome != "passed":
        record["status"] = "FAIL"
        record["phases"]["call"] = outcome
    exitstatus = int(outcome == "failed")
    environments = []

    def execute(command, **options):
        environments.append(options["env"])
        assert "-p" in command and "tools.pytest_case_reporter" in command, (
            "the real child must generate its own structured receipt"
        )
        write_receipt(
            options["env"], {selected["pytest_nodeid"]: record}, exitstatus=exitstatus
        )
        return subprocess.CompletedProcess(
            command, exitstatus, stdout="unreliable prose"
        )

    monkeypatch.setattr(runner.subprocess, "run", execute)
    result = runner.run_case(
        selected,
        environment={
            "PYTEST_ADDOPTS": "-k hidden-filter",
            "PYTEST_GPU_FAULT_PARTITION_COUNT": "3",
            "PYTEST_GPU_FAULT_PARTITION_INDEX": "1",
        },
    )
    assert result["status"] == ("PASS" if outcome == "passed" else "FAIL"), (
        "only completed passing setup, call and teardown prove the case"
    )
    assert not {
        "PYTEST_ADDOPTS",
        "PYTEST_GPU_FAULT_PARTITION_COUNT",
        "PYTEST_GPU_FAULT_PARTITION_INDEX",
    }.intersection(environments[0]), "inherited selection must not silently trim a case"


def test_fresh_generic_case_cannot_pass_a_filtered_child_receipt(monkeypatch) -> None:
    selected = case()
    base = selected["pytest_nodeid"]
    variants = [base + "[first]", base + "[second]"]

    def execute(command, **options):
        write_receipt(options["env"], {variants[0]: passed_record()})
        path = Path(options["env"][runner.PYTEST_BATCH_REPORT_ENV])
        value = json.loads(path.read_text())
        value["session"]["discovered_nodeids"] = variants
        path.write_text(json.dumps(value))
        return subprocess.CompletedProcess(command, 0, stdout="")

    monkeypatch.setattr(runner.subprocess, "run", execute)
    result = runner.run_case(selected, environment={})
    assert result["status"] == "FAIL"
    assert "all discovered variants" in result["pytest_error"]


def catalog(tmp_path: Path, cases=None, **changes):
    value = {
        "schema_version": 1,
        "test_cases": cases if cases is not None else [case()],
    }
    value.update(changes)
    path = tmp_path / "catalog.yaml"
    path.write_text(yaml.safe_dump(value), encoding="utf-8")
    return path


@pytest.mark.parametrize("field", sorted(runner.REQUIRED_FIELDS))
def test_catalog_requires_every_contract_field(tmp_path, field) -> None:
    value = case()
    value.pop(field)
    with pytest.raises(ValueError, match="missing fields"):
        runner.load_catalog(catalog(tmp_path, [value]))


@pytest.mark.parametrize(
    "changes",
    [
        {"extra": True},
        {"title": " "},
        {"expected": []},
        {"expected": [""]},
        {"risk": "destructive-provider-replace"},
        {"level": "unknown"},
        {"automation": "shell"},
        {"pytest_nodeid": "tests/test_unit.py"},
        {"related_pytest": "tests/unknown.py"},
        {"capture_processing_trace": "true"},
        {"operator_case": ""},
        {"manifest": "does-not-exist.yaml"},
        {"gate": "missing"},
        {"automation": "command", "command": []},
        {"automation": "command", "command": [1]},
        {"automation": "manual", "procedure": []},
        {"automation": "manual", "procedure": [None]},
        {"automation": "manual", "procedure": " "},
    ],
)
def test_invalid_catalog_vocabulary_and_metadata_are_rejected(
    tmp_path, changes
) -> None:
    with pytest.raises(ValueError):
        runner.load_catalog(catalog(tmp_path, [case(**changes)]))


@pytest.mark.parametrize("procedure", ["manual procedure", ["manual step"]])
def test_manual_cases_preserve_explicit_procedure(tmp_path, procedure) -> None:
    value = case(automation="manual", procedure=procedure)
    assert runner.load_catalog(catalog(tmp_path, [value])) == [value], (
        "a manual plan remains a plan, not executable code"
    )


@pytest.mark.parametrize(
    "document",
    [[], {}, {"schema_version": 2}, {"schema_version": True, "test_cases": [case()]}],
)
def test_bad_catalog_document_is_not_version_one(tmp_path, document) -> None:
    path = tmp_path / "catalog.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        runner.load_catalog(path)


@pytest.mark.parametrize("items", [None, [], [None], [case(), case()]])
def test_missing_invalid_or_duplicate_case_inventory_is_refused(
    tmp_path, items
) -> None:
    path = (
        catalog(tmp_path, test_cases=items)
        if items is None
        else catalog(tmp_path, items)
    )
    if items is None:
        path.write_text("schema_version: 1\ntest_cases: null\n", encoding="utf-8")
    with pytest.raises(ValueError):
        runner.load_catalog(path)


@pytest.mark.parametrize(
    "family",
    [
        None,
        {},
        {"generator": "integer_range", "template": []},
        {"generator": "integer_range", "template": case()},
        {"generator": "integer_range", "start": 3, "end": 1, "template": case()},
        {"generator": "integer_range", "start": None, "end": 1, "template": case()},
        {"generator": "unknown", "template": case()},
        {"generator": "xid_catalog_rules", "start": 1, "template": case()},
    ],
)
def test_generated_family_cannot_hide_invalid_definition(tmp_path, family) -> None:
    with pytest.raises(ValueError):
        runner.load_catalog(catalog(tmp_path, generated_test_families=[family]))


def test_generated_families_expand_without_changing_value_types(tmp_path) -> None:
    template = case(
        id="GF-UNIT-{value:03d}",
        title="Case {value}",
        expected=["condition {value}"],
        capture_processing_trace=False,
    )
    values = runner.load_catalog(
        catalog(
            tmp_path,
            [case(id="GF-BASE")],
            generated_test_families=[
                {
                    "generator": "integer_range",
                    "start": 2,
                    "end": 3,
                    "template": template,
                }
            ],
        )
    )
    assert [item["id"] for item in values] == [
        "GF-BASE",
        "GF-UNIT-002",
        "GF-UNIT-003",
    ], "expansion is ordered and complete"
    assert (
        values[-1]["expected"] == ["condition 3"]
        and values[-1]["capture_processing_trace"] is False
    ), "only templated strings are formatted"


@pytest.mark.parametrize("family_list", [None, {}])
def test_family_inventory_must_be_a_list(tmp_path, family_list) -> None:
    with pytest.raises(ValueError, match="must be a list"):
        runner.load_catalog(catalog(tmp_path, generated_test_families=family_list))


@pytest.mark.parametrize(
    "execution",
    [
        "invalid",
        {"extra": True},
        {"parallel_safe": "true"},
        {"depends_on": "case"},
        {"depends_on": ["missing"]},
        {"depends_on": ["GF-UNIT-001"]},
        {"environment": "shared"},
        {"failure_scope": "ignored"},
        {"locks": "lock"},
        {"locks": [None]},
        {"locks": [{"resource": "r", "mode": "invalid"}]},
    ],
)
def test_execution_policy_refuses_unknown_authority(tmp_path, execution) -> None:
    if isinstance(execution, dict):
        execution = {"parallel_safe": False, **execution}
    with pytest.raises(ValueError):
        runner.load_catalog(catalog(tmp_path, [case(execution=execution)]))


@pytest.mark.parametrize(
    "evidence",
    [
        "PASS",
        {},
        {"verdict": "UNKNOWN"},
        {"verdict": "PASS"},
        {"verdict": "NOT_RUN", "verified": {}},
        {"verdict": "NOT_RUN", "extra": "data"},
    ],
)
def test_evidence_cannot_claim_an_unbound_pass(tmp_path, evidence) -> None:
    with pytest.raises(ValueError):
        runner.load_catalog(catalog(tmp_path, [case(evidence=evidence)]))


def test_verified_evidence_is_bound_to_case_definition_and_components(tmp_path) -> None:
    value = case()
    verified = {
        "case_digest": runner.case_definition_digest(value),
        "components": {name: "a" * 64 for name in runner.VERIFIED_COMPONENTS},
    }
    value["evidence"] = {"verdict": "PASS", "verified": verified}
    assert runner.load_catalog(catalog(tmp_path, [value])) == [value], (
        "a fully bound proof remains loadable"
    )
    value["expected"] = ["Changed claim"]
    with pytest.raises(ValueError, match="does not describe this case"):
        runner.load_catalog(catalog(tmp_path, [value]))


@pytest.mark.parametrize("defect", ["missing-components", "bad-sha", "extra-field"])
def test_verified_evidence_rejects_incomplete_identity(tmp_path, defect) -> None:
    value = case()
    verified = {
        "case_digest": runner.case_definition_digest(value),
        "components": {name: "a" * 64 for name in runner.VERIFIED_COMPONENTS},
    }
    if defect == "missing-components":
        verified.pop("components")
    elif defect == "bad-sha":
        verified["case_digest"] = "invalid"
    else:
        verified["extra"] = "claim"
    value["evidence"] = {"verdict": "PASS", "verified": verified}
    with pytest.raises(ValueError):
        runner.load_catalog(catalog(tmp_path, [value]))


def test_retired_case_is_non_executable_and_points_to_a_real_successor(
    tmp_path, monkeypatch
) -> None:
    retired = case(
        automation="manual",
        procedure="retired",
        evidence={"verdict": "SUPERSEDED"},
        superseded_by="GF-UNIT-002",
        operator_case="operator",
    )
    current = case(id="GF-UNIT-002")
    selected = runner.load_catalog(catalog(tmp_path, [retired, current]))
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("manual case executed"),
    )
    report = runner.run_case(selected[0])
    assert report["status"] == "NOT_RUN" and "superseded" in report["reason"], (
        "retirement never becomes a synthetic execution"
    )
    assert report["operator_case"] == "operator", "manual ownership is preserved"
    retired["superseded_by"] = "missing"
    with pytest.raises(ValueError, match="unknown case"):
        runner.load_catalog(catalog(tmp_path, [retired, current]))


def test_selection_requires_explicit_command_ids_and_preserves_requested_order() -> (
    None
):
    values = [
        case(),
        case(id="GF-UNIT-002"),
        case(id="GF-LIVE", automation="command", command=["unit-tool"]),
        case(id="GF-MANUAL", automation="manual", procedure="manual"),
    ]

    def select(**options):
        arguments = {
            "case_ids": [],
            "categories": set(),
            "levels": set(),
            "include_live": False,
            "include_manual": False,
        }
        arguments.update(options)
        return runner.select_cases(values, **arguments)

    assert [item["id"] for item in select(include_live=True)] == [
        "GF-UNIT-001",
        "GF-UNIT-002",
    ], "including live capability without naming a command cannot execute it"
    assert [item["id"] for item in select(case_ids=["GF-UNIT-002", "GF-UNIT-001"])] == [
        "GF-UNIT-002",
        "GF-UNIT-001",
    ], "operator order is retained"
    assert select(categories={"other"}) == [] and select(levels={"live"}) == [], (
        "filters only narrow selection"
    )
    assert select(case_ids=["GF-LIVE"], include_live=True)[0]["id"] == "GF-LIVE", (
        "a named command may be selected with live enabled"
    )
    with pytest.raises(ValueError, match="unknown test case"):
        select(case_ids=["unknown"])
    with pytest.raises(ValueError, match="only accepts command"):
        select(also_case_ids=["GF-UNIT-001"], include_live=True)
    with pytest.raises(ValueError, match="requires --include-live"):
        select(also_case_ids=["GF-LIVE"])


@pytest.mark.parametrize("kind", ["pytest", "command"])
@pytest.mark.parametrize("trace", ["none", "valid", "missing", "malformed"])
def test_run_case_requires_success_and_requested_processing_trace(
    monkeypatch, kind, trace
) -> None:
    value = case(automation=kind)
    if kind == "command":
        value["command"] = ["unit-tool", "argument"]
    value["capture_processing_trace"] = trace != "none"
    output = "observed"
    if trace == "valid":
        output += "\n" + runner.TRACE_PREFIX + json.dumps({"event": "unit"})
    elif trace == "malformed":
        output += "\n" + runner.TRACE_PREFIX + "{"
    calls = []

    def run(arguments, **options):
        calls.append((arguments, options))
        if kind == "pytest":
            write_receipt(options["env"], {value["pytest_nodeid"]: passed_record()})
        return subprocess.CompletedProcess(arguments, 0, output)

    monkeypatch.setattr(runner.subprocess, "run", run)
    result = runner.run_case(
        value, environment={"PATH": "/unit"}, environment_overrides={"EXAMPLE": "1"}
    )
    assert result["status"] == ("PASS" if trace in {"none", "valid"} else "FAIL"), (
        "missing evidence cannot be hidden by process exit zero"
    )
    assert calls[0][1]["env"]["EXAMPLE"] == "1", (
        "explicit controlled environment overrides are retained"
    )
    if trace == "valid":
        assert result["processing_trace"] == {"event": "unit"}, (
            "structured trace is kept outside raw output"
        )
    elif trace != "none":
        assert "trace_error" in result, "incomplete processing proof is explained"


@pytest.mark.parametrize(
    ("manual", "failed", "expected"),
    [
        (False, False, "PASS"),
        (False, True, "FAIL"),
        (True, False, "PASS_WITH_LIMITATIONS"),
    ],
)
def test_main_publishes_actual_case_outcomes_and_never_executes_manual_cases(
    tmp_path, monkeypatch, capsys, manual, failed, expected
) -> None:
    values = [case()]
    if manual:
        values.append(
            case(id="GF-MANUAL", automation="manual", procedure="manual review")
        )
    path = catalog(tmp_path, values)
    calls = []

    def run(arguments, **options):
        calls.append((arguments, options))
        record = passed_record()
        if failed:
            record["status"] = "FAIL"
            record["phases"]["call"] = "failed"
        write_receipt(
            options["env"], {values[0]["pytest_nodeid"]: record}, exitstatus=int(failed)
        )
        return subprocess.CompletedProcess(arguments, int(failed), "unit result")

    monkeypatch.setattr(runner.subprocess, "run", run)
    monkeypatch.setattr(runner, "DEFAULT_REPORT_DIR", tmp_path / "reports")
    arguments = ["--catalog", str(path)]
    if manual:
        arguments.append("--include-manual")
    assert runner.main(arguments) == int(failed), (
        "runner exit status reflects test failure"
    )
    reports = list((tmp_path / "reports").glob("*.json"))
    assert len(reports) == 1 and len(calls) == 1, (
        "manual entries do not add a process execution"
    )
    report = json.loads(reports[0].read_text())
    assert report["verdict"] == expected, (
        "mixed execution cannot be called unconditional PASS"
    )
    assert report["summary"]["not_run"] == int(manual), (
        "manual work stays visible as NOT_RUN"
    )
    assert "SUMMARY" in capsys.readouterr().out, (
        "the operator receives a completion summary"
    )


def test_main_with_only_manual_work_reports_not_run(tmp_path, monkeypatch) -> None:
    path = catalog(tmp_path, [case(automation="manual", procedure="manual review")])
    report_path = tmp_path / "report.json"
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("manual execution"),
    )
    assert (
        runner.main(
            ["--catalog", str(path), "--include-manual", "--report", str(report_path)]
        )
        == 0
    ), "recording a manual plan is not an execution failure"
    assert json.loads(report_path.read_text())["verdict"] == "NOT_RUN", (
        "manual-only work never reports PASS"
    )


def test_list_empty_selection_and_bad_worker_count_do_not_start_a_case(
    tmp_path, monkeypatch, capsys
) -> None:
    path = catalog(tmp_path)
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("unexpected process"),
    )
    assert runner.main(["--catalog", str(path), "--list"]) == 0, (
        "listing remains read-only"
    )
    assert "GF-UNIT-001" in capsys.readouterr().out, (
        "the selected catalog identity is printed"
    )
    assert runner.main(["--catalog", str(path), "--category", "missing"]) == 2, (
        "empty selection does not pass silently"
    )
    with pytest.raises(SystemExit) as error:
        runner.main(["--catalog", str(path), "--workers", "0"])
    assert error.value.code == 2, "worker count must be positive"


@pytest.mark.parametrize("problem", ["incompatible", "two-result-modes"])
def test_invalid_batch_modes_are_rejected_before_execution(
    tmp_path, monkeypatch, problem
) -> None:
    value = case(capture_processing_trace=problem == "incompatible")
    path = catalog(tmp_path, [value])
    arguments = ["--catalog", str(path), "--batch-pytest"]
    if problem == "two-result-modes":
        arguments.extend(["--pytest-results", str(tmp_path / "stored.json")])
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("invalid batch ran"),
    )
    assert runner.main(arguments) == 2, (
        "trace-sensitive or conflicting batches cannot run"
    )


@pytest.mark.parametrize("reuse", [False, True])
def test_main_batches_use_bound_per_test_results(tmp_path, monkeypatch, reuse) -> None:
    values = [
        case(),
        case(id="GF-UNIT-002", pytest_nodeid="tests/test_unit.py::test_other"),
    ]
    path = catalog(tmp_path, values)
    record = {
        "status": "PASS",
        "duration_seconds": 0.1,
        "output": "unit",
        "phases": {"setup": "passed", "call": "passed", "teardown": "passed"},
    }
    payload = {
        "schema_version": 1,
        "source_identity": "a" * 64,
        "session": {
            "source_identity": "a" * 64,
            "exitstatus": 0,
            "collected_nodeids": [value["pytest_nodeid"] for value in values],
            "discovered_nodeids": [value["pytest_nodeid"] for value in values],
        },
        "records": {value["pytest_nodeid"]: record for value in values},
    }
    monkeypatch.setattr(runner, "source_identity", lambda root: "a" * 64)
    calls = []

    def run(arguments, **options):
        calls.append(arguments)
        assert not reuse, "reused proof must not run tests again"
        Path(options["env"][runner.PYTEST_BATCH_REPORT_ENV]).write_text(
            json.dumps(payload)
        )
        return subprocess.CompletedProcess(arguments, 0, "unit batch")

    monkeypatch.setattr(runner.subprocess, "run", run)
    destination = tmp_path / "result.json"
    arguments = ["--catalog", str(path), "--report", str(destination), "--workers", "2"]
    if reuse:
        stored = tmp_path / "stored.json"
        stored.write_text(json.dumps(payload))
        arguments.extend(["--pytest-results", str(stored)])
    else:
        arguments.append("--batch-pytest")
    assert runner.main(arguments) == 0, "all concrete test records passed"
    report = json.loads(destination.read_text())
    assert report["summary"]["pass"] == 2 and report["summary"]["total"] == 2, (
        "batching retains both case outcomes"
    )
    assert len(calls) == int(not reuse), (
        "batch execution happens once, reuse not at all"
    )


@pytest.mark.parametrize("exitstatus", [0, 1, 2])
def test_batch_exit_status_cannot_hide_a_missing_or_failed_execution_proof(
    monkeypatch, exitstatus
) -> None:
    values = [case(), case(id="GF-OTHER", pytest_nodeid="tests/test_unit.py::other")]
    records = {item["pytest_nodeid"]: passed_record() for item in values}
    if exitstatus == 1:
        records[values[1]["pytest_nodeid"]] = {
            "status": "FAIL",
            "phases": {"setup": "passed", "call": "failed", "teardown": "passed"},
        }

    def execute(command, **options):
        write_receipt(options["env"], records, exitstatus=exitstatus)
        return subprocess.CompletedProcess(command, exitstatus, stdout="")

    monkeypatch.setattr(runner.subprocess, "run", execute)
    result = runner.run_pytest_batch(values, environment={})
    assert result["GF-UNIT-001"]["status"] == ("FAIL" if exitstatus == 2 else "PASS"), (
        "collection failure cannot be promoted, but independent test outcomes survive"
    )
    assert result["GF-OTHER"]["status"] == ("PASS" if exitstatus == 0 else "FAIL"), (
        "a failed test remains failed within the batch"
    )


def test_nonzero_pytest_exit_without_a_failed_record_is_not_a_passing_batch(
    monkeypatch,
) -> None:
    selected = case()

    def execute(command, **options):
        write_receipt(
            options["env"], {selected["pytest_nodeid"]: passed_record()}, exitstatus=1
        )
        return subprocess.CompletedProcess(command, 1, stdout="")

    monkeypatch.setattr(runner.subprocess, "run", execute)
    result = runner.run_pytest_batch([selected], environment={})
    assert result[selected["id"]]["status"] == "FAIL", (
        "an unaccounted process failure invalidates the purported passing receipt"
    )


def test_a_missing_batch_report_is_failure_even_when_process_returns_zero(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda arguments, **kwargs: subprocess.CompletedProcess(
            arguments, 0, "no report"
        ),
    )
    result = runner.run_pytest_batch([case()], environment={})
    assert result["GF-UNIT-001"]["status"] == "FAIL", (
        "missing per-test proof cannot inherit process success"
    )
    assert runner.run_pytest_batch([]) == {}, (
        "an explicitly empty batch has no invented result"
    )
    with pytest.raises(ValueError, match="unique nodeids"):
        runner.run_pytest_batch([case(), case(id="GF-UNIT-002")], environment={})


@pytest.mark.parametrize(
    "output",
    [
        runner.TRACE_PREFIX + "[]",
        runner.TRACE_PREFIX + "{}\n" + runner.TRACE_PREFIX + "{}",
    ],
)
def test_nonobject_or_duplicate_trace_is_failure(monkeypatch, output) -> None:
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda arguments, **kwargs: subprocess.CompletedProcess(arguments, 0, output),
    )
    result = runner.run_case(case(capture_processing_trace=True), environment={})
    assert result["status"] == "FAIL" and result.get("trace_error"), (
        "one complete processing trace is required"
    )


@pytest.mark.parametrize(
    "locks",
    [
        [],
        [{"resource": "r", "mode": "shared", "unknown": True}],
        [{"resource": "r", "mode": "shared"}, {"resource": "r", "mode": "exclusive"}],
        [{"resource": "", "mode": "shared"}],
    ],
)
def test_parallel_execution_requires_complete_unique_locks(tmp_path, locks) -> None:
    value = case(execution={"parallel_safe": True, "locks": locks})
    with pytest.raises(ValueError):
        runner.load_catalog(catalog(tmp_path, [value]))


def test_configured_execution_preserves_authority_and_dependencies(tmp_path) -> None:
    first = case(id="GF-FIRST")
    second = case(
        execution={
            "parallel_safe": True,
            "environment": "isolated",
            "failure_scope": "branch",
            "depends_on": ["GF-FIRST"],
            "locks": [{"resource": "unit-resource", "mode": "shared"}],
        }
    )
    values = runner.load_catalog(catalog(tmp_path, [first, second]))
    policy = runner.execution_policy(values[1])
    assert policy.depends_on == ("GF-FIRST",) and policy.parallel_safe, (
        "declared dependency is not lost during scheduling"
    )
    assert policy.locks[0].resource == "unit-resource", (
        "declared lock scope is retained"
    )
    assert not runner.pytest_batch_eligible(second, policy), (
        "dependent cases cannot be collapsed into an unordered batch"
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"superseded_by": "GF-OTHER"},
        {"evidence": {"verdict": "SUPERSEDED"}, "superseded_by": "GF-OTHER"},
        {
            "automation": "manual",
            "procedure": "retired",
            "evidence": {"verdict": "SUPERSEDED"},
            "superseded_by": "GF-UNIT-001",
        },
        {"current_status": "PASS"},
    ],
)
def test_retirement_and_legacy_evidence_fields_cannot_change_execution(
    tmp_path, changes
) -> None:
    with pytest.raises(ValueError):
        runner.load_catalog(catalog(tmp_path, [case(**changes)]))


def test_an_unbound_historical_pass_cannot_be_retroactively_given_current_components(
    tmp_path,
) -> None:
    identifier = next(iter(runner.EVIDENCE_UNBOUND_PASS_CASES))
    value = case(id=identifier, automation="manual", procedure="historical only")
    value["evidence"] = {
        "verdict": "PASS",
        "verified": {
            "case_digest": runner.case_definition_digest(value),
            "components": {name: "a" * 64 for name in runner.VERIFIED_COMPONENTS},
        },
    }
    with pytest.raises(ValueError, match="cannot be backfilled"):
        runner.load_catalog(catalog(tmp_path, [value]))


def test_duplicate_yaml_keys_are_not_last_value_wins(tmp_path) -> None:
    path = catalog(tmp_path)
    with path.open("a", encoding="utf-8") as output:
        output.write("schema_version: 1\n")
    with pytest.raises(ValueError, match="duplicate YAML key"):
        runner.load_catalog(path)


def test_additional_explicit_command_is_selected_once() -> None:
    value = case(id="GF-COMMAND", automation="command", command=["unit-command"])
    selected = runner.select_cases(
        [value],
        case_ids=["GF-COMMAND"],
        also_case_ids=["GF-COMMAND", "GF-COMMAND"],
        categories=set(),
        levels=set(),
        include_manual=False,
        include_live=True,
    )
    assert selected == [value], "additional selection cannot duplicate a command"
    selected = runner.select_cases(
        [value],
        case_ids=[],
        also_case_ids=["GF-COMMAND"],
        categories=set(),
        levels=set(),
        include_manual=False,
        include_live=True,
    )
    assert selected == [value], "an additional command still requires its explicit ID"
