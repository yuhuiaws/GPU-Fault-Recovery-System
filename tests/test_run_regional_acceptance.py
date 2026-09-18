from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from tools import run_regional_acceptance as runner
from tools.codex_acceptance import CaseAnalysis, CodexAcceptanceError, EvidenceReview
from tools.regional_acceptance_plan import (
    AutomationKind,
    DependencyNode,
    EnvironmentMode,
    ExecutionPolicy,
    ExecutorKind,
    FailureScope,
    PlanMode,
    RegionalAcceptanceCase,
    RegionalAcceptancePlan,
    compile_regional_acceptance_plan,
)


@pytest.fixture(scope="module")
def template() -> RegionalAcceptanceCase:
    return compile_regional_acceptance_plan().case("GF-REGIONAL-BOOT-011")


def planned_case(
    template: RegionalAcceptanceCase, index: int, **kwargs: Any
) -> RegionalAcceptanceCase:
    return replace(
        template,
        id=f"GF-REGIONAL-TEST-{index:03d}",
        ordinal=index - 1,
        executor=ExecutorKind.PYTEST,
        automation=AutomationKind.PYTEST,
        pytest_nodeids=("tests/example.py::test_example",),
        command=(),
        local_proxy=False,
        risk="non-destructive",
        blocked_reason=None,
        do_not_run_reason=None,
        execution=ExecutionPolicy(
            parallel_safe=False,
            environment=EnvironmentMode.ISOLATED,
            failure_scope=FailureScope.GLOBAL,
        ),
        mandatory_depends_on=(),
        override_depends_on=(),
        **kwargs,
    )


def plan_for(
    *cases: RegionalAcceptanceCase, mode: PlanMode = PlanMode.FORMAL
) -> RegionalAcceptancePlan:
    return RegionalAcceptancePlan(
        mode=mode,
        collect_all=mode is not PlanMode.FORMAL,
        read_only=mode is not PlanMode.FORMAL,
        repair_allowed=False,
        review_metadata=None,
        cases=cases,
        execution_order=tuple(
            case.id for case in cases if case.executor is not ExecutorKind.DO_NOT_RUN
        ),
        do_not_run_case_ids=tuple(
            case.id for case in cases if case.executor is ExecutorKind.DO_NOT_RUN
        ),
        dependency_graph=tuple(
            DependencyNode(case.id, case.execution.depends_on) for case in cases
        ),
    )


@pytest.fixture
def executions(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def execute(case: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        calls.append({"case": case, **kwargs})
        return {"id": case["id"], "status": "PASS"}

    monkeypatch.setattr(runner, "run_case", execute)
    return calls


def execute(plan: RegionalAcceptancePlan, **kwargs: Any) -> list[dict[str, Any]]:
    options = dict(
        workers=4,
        codex_batch_size=3,
        backend=None,
        review_with_codex=False,
        review_workers=2,
        approved_inherited_cases=set(),
    )
    options.update(kwargs)
    return runner.execute_plan(plan, **options)


def report(
    plan: RegionalAcceptancePlan, results: list[dict[str, Any]]
) -> dict[str, object]:
    return runner.build_report(
        plan,
        results,
        started=time.monotonic(),
        workers=4,
        codex_batch_size=3,
        reviewed_plan=None,
        order_path=Path("order.yaml"),
        catalog_path=Path("catalog.yaml"),
    )


class AnalysisBackend:
    def __init__(
        self, status: str = "PASS", *, review_status: str = "PASS", fail: bool = False
    ) -> None:
        self.status, self.review_status, self.fail = status, review_status, fail

    def analyze_manual_cases(
        self, cases: list[dict[str, Any]]
    ) -> dict[str, CaseAnalysis]:
        if self.fail:
            raise CodexAcceptanceError("analysis unavailable")
        return {
            case["id"]: CaseAnalysis(
                case["id"],
                self.status,
                "read-only analysis",
                (),
                (),
                (),
                (),
                (),
                (),
                (),
            )
            for case in cases
        }

    def review_evidence(self, case: dict[str, Any], evidence: Any) -> EvidenceReview:
        return EvidenceReview(case["id"], self.review_status, "review", (), (), (), ())


@pytest.mark.parametrize("executor", [ExecutorKind.HUMAN, ExecutorKind.CODEX_MANUAL])
def test_formal_not_run_stops_later_cases(
    template: RegionalAcceptanceCase,
    executions: list[dict[str, Any]],
    executor: ExecutorKind,
) -> None:
    first = replace(
        planned_case(template, 1), executor=executor, automation=AutomationKind.MANUAL
    )
    second = planned_case(template, 2)
    second = replace(
        second, execution=replace(second.execution, depends_on=(first.id,))
    )
    plan = plan_for(first, second)
    results = execute(plan, backend=AnalysisBackend())
    assert [item["status"] for item in results] == ["NOT_RUN", "BLOCKED"]
    assert not executions, executions
    assert report(plan, results)["verdict"] == "PARTIAL"


def test_local_proxy_does_not_claim_live_pass(
    template: RegionalAcceptanceCase, executions: list[dict[str, Any]]
) -> None:
    case = replace(
        planned_case(template, 1), local_proxy=True, automation=AutomationKind.MANUAL
    )
    plan = plan_for(case, mode=PlanMode.LOCAL_PREACCEPTANCE)
    results = execute(plan)
    assert len(executions) == 1
    assert results[0]["status"] == results[0]["execution_status"] == "NOT_RUN"
    assert results[0]["proxy_check"]["status"] == "PASS"
    assert report(plan, results)["verdict"] == "PARTIAL"


@pytest.mark.parametrize("status", ["FAIL", "BLOCKED", "NEEDS_HUMAN"])
def test_independent_review_changes_public_verdict(
    template: RegionalAcceptanceCase, executions: list[dict[str, Any]], status: str
) -> None:
    case = replace(
        planned_case(template, 1),
        executor=ExecutorKind.CODEX_MANUAL,
        automation=AutomationKind.MANUAL,
    )
    plan = plan_for(case, mode=PlanMode.COLLECT_ALL)
    results = execute(
        plan, backend=AnalysisBackend(review_status=status), review_with_codex=True
    )
    assert results[0]["status"] == ("FAIL" if status == "FAIL" else "BLOCKED")
    assert report(plan, results)["verdict"] != "PASS"
    assert not executions, executions


@pytest.mark.parametrize("status", ["PASS", "FAIL", "BLOCKED", "NEEDS_HUMAN"])
def test_analysis_never_promotes_execution(
    template: RegionalAcceptanceCase, executions: list[dict[str, Any]], status: str
) -> None:
    case = replace(planned_case(template, 1), executor=ExecutorKind.CODEX_MANUAL)
    result = execute(plan_for(case), backend=AnalysisBackend(status))[0]
    assert result["execution_status"] == "NOT_RUN"
    assert result["analysis_status"] == status
    assert not executions, executions


@pytest.mark.parametrize("backend", [None, AnalysisBackend(fail=True)])
def test_missing_or_failed_analysis_backend_is_blocked(
    template: RegionalAcceptanceCase, executions: list[dict[str, Any]], backend: Any
) -> None:
    case = replace(planned_case(template, 1), executor=ExecutorKind.CODEX_MANUAL)
    results = execute(plan_for(case), backend=backend)
    assert results[0]["status"] == "BLOCKED"
    assert not executions, executions


@pytest.mark.parametrize(
    "bad_results",
    [
        [],
        [{"id": "GF-REGIONAL-TEST-001", "status": "PASS", "execution_status": "PASS"}]
        * 2,
        [{"id": "GF-REGIONAL-WRONG-001", "status": "PASS", "execution_status": "PASS"}],
        [{"id": "GF-REGIONAL-TEST-001", "status": "UNKNOWN"}],
        [
            {
                "id": "GF-REGIONAL-TEST-001",
                "status": "PASS",
                "execution_status": "NOT_RUN",
            }
        ],
    ],
)
def test_incomplete_or_malformed_reports_never_pass(
    template: RegionalAcceptanceCase, bad_results: list[dict[str, Any]]
) -> None:
    assert report(plan_for(planned_case(template, 1)), bad_results)["verdict"] == "FAIL"


def test_retired_cases_are_accounted_for_without_failing_active_success(
    template: RegionalAcceptanceCase, executions: list[dict[str, Any]]
) -> None:
    first = planned_case(template, 1)
    retired = replace(
        planned_case(template, 2),
        executor=ExecutorKind.DO_NOT_RUN,
        do_not_run_reason="retired",
        phase_sequence=None,
    )
    plan = plan_for(first, retired)
    results = execute(plan)
    assert [item["status"] for item in results] == ["PASS", "NOT_RUN"]
    assert [item["case"]["id"] for item in executions] == [first.id]
    assert report(plan, results)["verdict"] == "PASS"
    results[1]["status"] = "PASS"
    assert report(plan, results)["verdict"] == "FAIL"


def test_empty_or_retired_only_plan_cannot_claim_acceptance(
    template: RegionalAcceptanceCase, executions: list[dict[str, Any]]
) -> None:
    retired = replace(planned_case(template, 1), executor=ExecutorKind.DO_NOT_RUN)
    plan = plan_for(retired)
    assert report(plan, execute(plan))["verdict"] == "PARTIAL"
    assert report(plan_for(), [])["verdict"] == "PARTIAL"
    assert not executions, executions


@pytest.mark.parametrize(
    "raw",
    [
        [],
        {"id": "wrong", "status": "PASS"},
        {"id": "GF-REGIONAL-TEST-001", "status": "unexpected"},
    ],
)
def test_bad_executor_result_stops_formal_run(
    template: RegionalAcceptanceCase, monkeypatch: pytest.MonkeyPatch, raw: Any
) -> None:
    monkeypatch.setattr(runner, "run_case", lambda *_args, **_kwargs: raw)
    results = execute(plan_for(planned_case(template, 1), planned_case(template, 2)))
    assert [item["status"] for item in results] == ["FAIL", "BLOCKED"]


@pytest.mark.parametrize("executor", [ExecutorKind.COMMAND, ExecutorKind.PYTEST])
def test_formal_plan_is_not_live_authorization(
    template: RegionalAcceptanceCase,
    executions: list[dict[str, Any]],
    executor: ExecutorKind,
) -> None:
    case = replace(planned_case(template, 1), risk="destructive", executor=executor)
    assert execute(plan_for(case))[0]["status"] == "BLOCKED"
    assert not executions, executions


@pytest.mark.parametrize(
    "problem",
    [
        "missing-nodeid",
        "two-nodeids",
        "missing-command",
        "unreviewed-inherit",
        "unapproved-inherit",
        "missing-env",
    ],
)
def test_invalid_executor_inputs_fail_before_launch(
    template: RegionalAcceptanceCase,
    executions: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    problem: str,
) -> None:
    case = planned_case(template, 1)
    if problem == "missing-nodeid":
        case = replace(case, pytest_nodeids=())
    elif problem == "two-nodeids":
        case = replace(case, pytest_nodeids=("a", "b"))
    elif problem == "missing-command":
        case = replace(case, executor=ExecutorKind.COMMAND)
    else:
        case = replace(
            case, execution=replace(case.execution, environment=EnvironmentMode.INHERIT)
        )
        if problem != "unreviewed-inherit":
            case = replace(case, id="GF-REGIONAL-CAP-005")
    monkeypatch.delenv("GPU_FAULT_STORE_URL", raising=False)
    results = execute(
        plan_for(case),
        approved_inherited_cases={case.id} if problem == "missing-env" else set(),
    )
    assert results[0]["status"] in {"FAIL", "BLOCKED"}
    assert not executions, executions


def test_reviewed_environment_inherits_only_required_values(
    template: RegionalAcceptanceCase,
    executions: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = replace(
        planned_case(template, 1),
        id="GF-REGIONAL-CAP-005",
        executor=ExecutorKind.COMMAND,
        command=("python3", "local-probe.py"),
    )
    case = replace(
        case, execution=replace(case.execution, environment=EnvironmentMode.INHERIT)
    )
    monkeypatch.setenv("GPU_FAULT_STORE_URL", "postgresql://test@127.0.0.1/probe")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "never-inherit")
    assert (
        execute(plan_for(case), approved_inherited_cases={case.id})[0]["status"]
        == "PASS"
    )
    environment = executions[0]["environment"]
    assert environment["GPU_FAULT_STORE_URL"].endswith("/probe"), (
        "reviewed inheritance must retain the fixture's /probe database target"
    )
    assert "AWS_SECRET_ACCESS_KEY" not in environment
    assert environment["KUBECONFIG"] == environment["AWS_CONFIG_FILE"] == "/dev/null"


@pytest.mark.parametrize(
    ("mode", "status", "expected"),
    [
        (PlanMode.FORMAL, "NOT_RUN", 1),
        (PlanMode.FORMAL, "PASS", 0),
        (PlanMode.LOCAL_PREACCEPTANCE, "NOT_RUN", 0),
        (PlanMode.COLLECT_ALL, "FAIL", 1),
    ],
)
def test_main_exit_matches_mode_and_report(
    tmp_path: Path,
    template: RegionalAcceptanceCase,
    monkeypatch: pytest.MonkeyPatch,
    mode: PlanMode,
    status: str,
    expected: int,
) -> None:
    plan = plan_for(planned_case(template, 1), mode=mode)
    monkeypatch.setattr(
        runner, "compile_regional_acceptance_plan", lambda **_kwargs: plan
    )
    monkeypatch.setattr(
        runner,
        "execute_plan",
        lambda *_args, **_kwargs: [
            {"id": plan.case_ids[0], "status": status, "execution_status": status}
        ],
    )
    output = tmp_path / "report.json"
    assert runner.main(["--mode", mode.value, "--report", str(output)]) == expected
    assert output.stat().st_mode & 0o077 == 0
    assert json.loads(output.read_text())["mode"] == mode.value


def test_list_plan_does_not_launch_cases(
    tmp_path: Path,
    template: RegionalAcceptanceCase,
    executions: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: Any,
) -> None:
    plan = plan_for(planned_case(template, 1))
    monkeypatch.setattr(
        runner, "compile_regional_acceptance_plan", lambda **_kwargs: plan
    )
    output = tmp_path / "plan.json"
    assert runner.main(["--list-plan", "--write-plan", str(output)]) == 0
    assert plan.case_ids[0] in capsys.readouterr().out
    assert json.loads(output.read_text())["case_ids"] == list(plan.case_ids)
    assert not executions, executions


def test_main_requires_selected_analysis_backend(
    template: RegionalAcceptanceCase,
    executions: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = plan_for(
        replace(planned_case(template, 1), executor=ExecutorKind.CODEX_MANUAL)
    )
    monkeypatch.setattr(
        runner, "compile_regional_acceptance_plan", lambda **_kwargs: plan
    )
    assert runner.main([]) == 2
    assert not executions, executions
