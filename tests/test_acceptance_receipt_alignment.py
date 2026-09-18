from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from scripts.e2e.regional import run_preemption_contracts as contracts
from tests.test_run_regional_acceptance import execute, plan_for
from tools import run_fault_test_cases as fault_runner
from tools.regional_acceptance_plan import PlanMode, compile_regional_acceptance_plan

IDENTITY = "a" * 64


def complete_report(selectors):
    return {
        "schema_version": 1,
        "source_identity": IDENTITY,
        "session": {
            "source_identity": IDENTITY,
            "exitstatus": 0,
            "started_at": "2026-09-15T00:00:00+00:00",
            "finished_at": "2026-09-15T00:00:01+00:00",
            "collected_nodeids": list(selectors),
            "discovered_nodeids": list(selectors),
            "collection_errors": [],
            "collection_skips": [],
            "selection": {
                "targets": list(selectors),
                "keyword": "",
                "markexpr": "",
                "deselect": [],
                "partition": None,
                "numprocesses": 0,
            },
        },
        "records": {
            item: {
                "status": "PASS",
                "phases": {"setup": "passed", "call": "passed", "teardown": "passed"},
            }
            for item in selectors
        },
    }


@pytest.mark.parametrize(
    "defect",
    [
        "none",
        "missing",
        "skipped",
        "deselected",
        "filtered",
        "wrong-target",
        "source",
        "teardown",
        "extra",
        "collector-skip",
        "partition",
    ],
)
def test_preemption_contract_pass_requires_a_complete_unfiltered_current_receipt(
    tmp_path, monkeypatch, capsys, defect
):
    case_id = "GF-REGIONAL-PREEMPT-009"
    selectors = contracts.CASE_NODEIDS[case_id]
    report = complete_report(selectors)
    monkeypatch.setattr(fault_runner, "source_identity", lambda root: IDENTITY)
    monkeypatch.setenv(
        "GPU_FAULT_TEST_POSTGRES_URL", "postgresql://production.invalid/db"
    )
    monkeypatch.setenv("PYTEST_ADDOPTS", "-k filtered")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "inherited-not-a-real-credential")
    if defect == "skipped":
        report["records"][selectors[0]]["phases"] = {
            "setup": "skipped",
            "teardown": "passed",
        }
    elif defect == "deselected":
        report["session"]["collected_nodeids"].remove(selectors[0])
        report["records"].pop(selectors[0])
    elif defect == "filtered":
        report["session"]["selection"]["keyword"] = "partial"
    elif defect == "partition":
        report["session"]["selection"]["partition"] = [2, 0]
    elif defect == "wrong-target":
        report["session"]["selection"]["targets"] = ["tests/other.py::test_other"]
    elif defect == "source":
        report["source_identity"] = "b" * 64
    elif defect == "teardown":
        report["records"][selectors[0]]["phases"].pop("teardown")
    elif defect == "extra":
        report["records"]["tests/other.py::test_other"] = report["records"][
            selectors[0]
        ]
    elif defect == "collector-skip":
        report["session"]["collection_skips"] = ["tests/skip.py"]

    def run(command, **kwargs):
        environment = kwargs["env"]
        assert environment["GPU_FAULT_TEST_POSTGRES_URL"] == ""
        assert environment["KUBECONFIG"] == "/dev/null"
        assert (
            "AWS_ACCESS_KEY_ID" not in environment
            and "PYTEST_ADDOPTS" not in environment
        )
        assert command[-2:] == ["-o", "addopts="]
        assert command[command.index("-n") + 1] == "0"
        if defect != "missing":
            Path(environment["PYTEST_GPU_FAULT_CASE_REPORT"]).write_text(
                json.dumps(report)
            )
        return subprocess.CompletedProcess(command, 0, stdout="1 passed\n")

    code = contracts.run_case(case_id, run_dir=tmp_path, runner=run)
    evidence = json.loads(
        (tmp_path / "cases" / case_id / f"{case_id}.json").read_text()
    )
    assert (code == 0) is (defect == "none")
    assert evidence["verdict"] == ("PASS" if defect == "none" else "FAIL")
    assert evidence["proof_scope"] == "local-contract"
    if defect == "none":
        assert evidence["source_identity"] == IDENTITY
        assert len(evidence["pytest_records"]) == len(selectors)
    else:
        assert evidence["receipt_errors"]
    assert (
        json.loads(capsys.readouterr().out.splitlines()[-1])["verdict"]
        == evidence["verdict"]
    )


@pytest.mark.parametrize("text", ["1 skipped", "1 passed, 3 deselected"])
def test_zero_exit_without_a_pytest_receipt_is_never_accepted(text):
    evidence = contracts.case_evidence(
        "GF-REGIONAL-PREEMPT-009", subprocess.CompletedProcess([], 0, stdout=text)
    )
    assert evidence["verdict"] == "FAIL"


@pytest.mark.parametrize(
    "case_id", ["GF-REGIONAL-PREEMPT-033", "GF-REGIONAL-NOTIFY-008"]
)
def test_native_contract_is_blocked_before_any_subprocess_or_database_connection(
    monkeypatch, case_id
):
    case = compile_regional_acceptance_plan(mode=PlanMode.LOCAL_PREACCEPTANCE).case(
        case_id
    )
    case = replace(
        case,
        mandatory_depends_on=(),
        override_depends_on=(),
        execution=replace(case.execution, depends_on=()),
    )
    calls = []
    monkeypatch.setattr(
        fault_runner.subprocess,
        "run",
        lambda *args, **kwargs: calls.append(args)
        or pytest.fail("ungranted native prerequisite reached a subprocess"),
    )
    monkeypatch.setenv(
        "GPU_FAULT_TEST_POSTGRES_URL", "postgresql://production.invalid/db"
    )
    results = execute(
        plan_for(case, mode=PlanMode.LOCAL_PREACCEPTANCE), workers=1, review_workers=1
    )
    assert results[0]["status"] == "BLOCKED"
    assert "owned isolated PostgreSQL gate" in results[0]["reason"]
    assert calls == []
