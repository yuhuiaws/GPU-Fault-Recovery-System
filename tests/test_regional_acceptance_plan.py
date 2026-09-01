from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, cast

import pytest
import yaml  # type: ignore[import-untyped]

from tools.regional_acceptance_plan import (
    ExecutorKind,
    FailureScope,
    LockMode,
    PlanMode,
    compile_regional_acceptance_plan,
)

ROOT = Path(__file__).resolve().parents[1]
ORDER = ROOT / "testcases" / "regional-execution-order.yaml"
CATALOG = ROOT / "testcases" / "fault-scenarios.yaml"


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict), path
    return cast(dict[str, Any], value)


def _write_yaml(path: Path, value: object) -> Path:
    path.write_text(
        yaml.safe_dump(value, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return path


def _write_override(tmp_path: Path, cases: dict[str, object]) -> Path:
    return _write_yaml(
        tmp_path / "override.yaml",
        {
            "schema_version": 1,
            "reviewed": True,
            "reviewed_by": "test-acceptance-reviewer",
            "reviewed_at": "2026-09-01T00:00:00Z",
            "order_sha256": hashlib.sha256(ORDER.read_bytes()).hexdigest(),
            "catalog_sha256": hashlib.sha256(CATALOG.read_bytes()).hexdigest(),
            "cases": cases,
        },
    )


def test_formal_plan_expands_complete_order_and_serial_dependency_chain() -> None:
    plan = compile_regional_acceptance_plan()

    assert plan.mode is PlanMode.FORMAL
    assert plan.collect_all is False
    assert plan.read_only is False
    assert plan.repair_allowed is False
    assert len(plan.execution_order) == 152
    assert len(plan.cases) == 153
    assert plan.execution_order[:2] == ("GF-REGIONAL-BOOT-011", "GF-REGIONAL-BOOT-012")
    assert plan.execution_order[-1] == "GF-REGIONAL-COLLECT-015"
    assert plan.do_not_run_case_ids == ("GF-REGIONAL-DESTR-004",)
    assert plan.case_ids == (*plan.execution_order, "GF-REGIONAL-DESTR-004")

    first = plan.case(plan.execution_order[0])
    assert first.execution.depends_on == ()
    for previous, case_id in zip(plan.execution_order, plan.execution_order[1:]):
        case = plan.case(case_id)
        assert case.mandatory_depends_on == (previous,)
        assert case.execution.depends_on == (previous,)
        assert case.execution.parallel_safe is False
        assert case.execution.failure_scope is FailureScope.GLOBAL
        assert case.execution.collect_all is False

    retired = plan.case("GF-REGIONAL-DESTR-004")
    assert retired.executor is ExecutorKind.DO_NOT_RUN
    assert retired.blocked is True
    assert retired.execution.depends_on == ()
    assert plan.graph["GF-REGIONAL-DESTR-004"] == ()


def test_formal_plan_merges_catalog_metadata_and_execution() -> None:
    plan = compile_regional_acceptance_plan()
    cap005 = plan.case("GF-REGIONAL-CAP-005")

    assert cap005.executor is ExecutorKind.COMMAND
    assert cap005.risk == "non-destructive"
    assert cap005.command == (
        "python3",
        "scripts/e2e/regional/run_cap005_postgres_suite.py",
    )
    assert cap005.catalog_execution.parallel_safe is True
    assert cap005.execution.parallel_safe is False
    assert {(lock.resource, lock.mode) for lock in cap005.execution.locks} == {
        ("postgres-server", LockMode.EXCLUSIVE),
        ("cap005-workdir", LockMode.EXCLUSIVE),
    }

    related = plan.case("GF-REGIONAL-BOOT-004")
    assert related.related_pytest == (
        "tests/regional/test_regional_control_plane.py::"
        "test_regional_context_rejects_local_kubernetes_adapter"
    )
    assert related.executor is ExecutorKind.HUMAN


def test_local_preacceptance_parallelizes_only_safe_or_proxy_work() -> None:
    plan = compile_regional_acceptance_plan(mode=PlanMode.LOCAL_PREACCEPTANCE)
    assert plan.collect_all is True
    assert plan.read_only is True
    assert plan.repair_allowed is False

    pytest_case = plan.case("GF-REGIONAL-PREEMPT-010")
    assert pytest_case.executor is ExecutorKind.PYTEST
    assert pytest_case.execution.parallel_safe is True
    assert pytest_case.execution.depends_on == ()
    assert pytest_case.local_proxy is False

    command_case = plan.case("GF-REGIONAL-BOOT-022")
    assert command_case.executor is ExecutorKind.COMMAND
    assert command_case.execution.parallel_safe is True
    assert command_case.command[:3] == ("python3", "-m", "pytest")

    related = plan.case("GF-REGIONAL-BOOT-004")
    assert related.executor is ExecutorKind.PYTEST
    assert related.local_proxy is True
    assert related.execution.parallel_safe is True
    assert related.pytest_nodeids == (related.related_pytest,)

    gate = plan.case("GF-REGIONAL-BOOT-017")
    assert gate.executor is ExecutorKind.COMMAND
    assert gate.local_proxy is True
    assert gate.command == ("python3", "scripts/check-manual-command-order.py")

    human = plan.case("GF-REGIONAL-BOOT-001")
    assert human.executor is ExecutorKind.HUMAN
    assert human.blocked is True
    destructive = plan.case("GF-REGIONAL-DESTR-001")
    assert destructive.executor is ExecutorKind.HUMAN
    assert destructive.blocked is True

    automated = [
        case
        for case in plan.cases
        if case.executor in {ExecutorKind.PYTEST, ExecutorKind.COMMAND}
        and not case.blocked
    ]
    assert len(automated) == 62
    assert all(
        case.risk == "non-destructive" or case.local_proxy for case in automated
    ), [case.id for case in automated]
    assert all(
        case.execution.failure_scope is FailureScope.CASE
        and case.execution.collect_all
        and case.execution.read_only
        and not case.execution.repair_allowed
        for case in plan.cases
    ), plan.as_dict()


def test_collect_all_blocks_only_real_dependents_and_keeps_subagents_read_only(
    tmp_path: Path,
) -> None:
    override = _write_override(
        tmp_path,
        {
            "GF-REGIONAL-BOOT-011": {"executor": "codex-manual"},
            "GF-REGIONAL-BOOT-012": {
                "executor": "codex-manual",
                "depends_on": ["GF-REGIONAL-BOOT-011"],
            },
            "GF-REGIONAL-BOOT-013": {
                "executor": "codex-manual",
                "depends_on": ["GF-REGIONAL-BOOT-012"],
            },
            "GF-REGIONAL-BOOT-014": {"executor": "codex-manual"},
        },
    )

    plan = compile_regional_acceptance_plan(
        mode=PlanMode.COLLECT_ALL, override_path=override
    )

    assert plan.collect_all is True
    assert plan.read_only is True
    assert plan.repair_allowed is False
    first = plan.case("GF-REGIONAL-BOOT-011")
    dependent = plan.case("GF-REGIONAL-BOOT-012")
    transitive = plan.case("GF-REGIONAL-BOOT-013")
    independent = plan.case("GF-REGIONAL-BOOT-014")
    assert first.execution.depends_on == ()
    assert dependent.execution.depends_on == ("GF-REGIONAL-BOOT-011",)
    assert transitive.execution.depends_on == ("GF-REGIONAL-BOOT-012",)
    assert independent.execution.depends_on == ()
    assert plan.blocked_case_ids({"GF-REGIONAL-BOOT-011"}) == (
        "GF-REGIONAL-BOOT-012",
        "GF-REGIONAL-BOOT-013",
    )

    for case in (first, dependent, transitive, independent):
        assert case.executor is ExecutorKind.CODEX_MANUAL
        assert case.blocked is False
        assert case.pytest_nodeids == ()
        assert case.command == ()
        assert case.problem, case.id
        assert case.injection, case.id
        assert case.expected, case.id
        assert case.procedure is not None
        assert case.execution.parallel_safe is True
        assert case.execution.failure_scope is FailureScope.CASE
        assert case.execution.collect_all is True
        assert case.execution.read_only is True
        assert case.execution.repair_allowed is False

    codex_payload = first.as_dict()
    assert codex_payload["automation"] == "manual"
    assert codex_payload["problem"] == first.problem
    assert codex_payload["injection"] == first.injection
    assert codex_payload["expected"] == list(first.expected)
    assert codex_payload["procedure"] == first.procedure

    formal = compile_regional_acceptance_plan(override_path=override)
    assert formal.case("GF-REGIONAL-BOOT-014").execution.depends_on == (
        "GF-REGIONAL-BOOT-013",
    )


def test_reviewed_override_only_adds_constraints_and_codex_manual(
    tmp_path: Path,
) -> None:
    override = _write_override(
        tmp_path,
        {
            "GF-REGIONAL-BOOT-013": {
                "executor": "codex-manual",
                "depends_on": ["GF-REGIONAL-BOOT-011"],
                "locks": [{"resource": "acceptance-evidence", "mode": "shared"}],
            },
            "GF-REGIONAL-BOOT-014": {"depends_on": []},
        },
    )

    plan = compile_regional_acceptance_plan(override_path=override)
    selected = plan.case("GF-REGIONAL-BOOT-013")
    assert selected.executor is ExecutorKind.CODEX_MANUAL
    assert selected.execution.read_only is True
    assert selected.execution.repair_allowed is False
    assert selected.mandatory_depends_on == ("GF-REGIONAL-BOOT-012",)
    assert selected.override_depends_on == ("GF-REGIONAL-BOOT-011",)
    assert selected.execution.depends_on == (
        "GF-REGIONAL-BOOT-012",
        "GF-REGIONAL-BOOT-011",
    )
    assert [(lock.resource, lock.mode) for lock in selected.execution.locks] == [
        ("acceptance-evidence", LockMode.SHARED)
    ]

    no_op_addition = plan.case("GF-REGIONAL-BOOT-014")
    assert no_op_addition.execution.depends_on == ("GF-REGIONAL-BOOT-013",)

    local = compile_regional_acceptance_plan(
        mode=PlanMode.LOCAL_PREACCEPTANCE, override_path=override
    )
    local_selected = local.case("GF-REGIONAL-BOOT-013")
    assert local_selected.executor is ExecutorKind.CODEX_MANUAL
    assert local_selected.blocked is True


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "reviewed", [False, "true", None]
)
def test_override_requires_explicit_boolean_review(
    tmp_path: Path, reviewed: object
) -> None:
    override = _write_yaml(
        tmp_path / "override.yaml",
        {"schema_version": 1, "reviewed": reviewed, "cases": {}},
    )

    with pytest.raises(ValueError, match="reviewed: true"):
        compile_regional_acceptance_plan(override_path=override)


def test_override_cannot_promote_cases_to_command_or_reclassify_pytest(
    tmp_path: Path,
) -> None:
    command_override = _write_override(
        tmp_path, {"GF-REGIONAL-DESTR-001": {"executor": "command"}}
    )
    with pytest.raises(ValueError, match="may only be codex-manual"):
        compile_regional_acceptance_plan(override_path=command_override)

    codex_override = _write_override(
        tmp_path, {"GF-REGIONAL-PREEMPT-010": {"executor": "codex-manual"}}
    )
    with pytest.raises(ValueError, match="only for manual cases"):
        compile_regional_acceptance_plan(override_path=codex_override)


def test_override_rejects_unknown_and_do_not_run_cases(tmp_path: Path) -> None:
    unknown = _write_override(tmp_path, {"GF-REGIONAL-FAKE-001": {"depends_on": []}})
    with pytest.raises(ValueError, match="unknown case ID"):
        compile_regional_acceptance_plan(override_path=unknown)

    retired = _write_override(
        tmp_path, {"GF-REGIONAL-DESTR-004": {"executor": "codex-manual"}}
    )
    with pytest.raises(ValueError, match="DO_NOT_RUN"):
        compile_regional_acceptance_plan(override_path=retired)


def test_override_rejects_unknown_dependencies_and_lock_replacement(
    tmp_path: Path,
) -> None:
    unknown_dependency = _write_override(
        tmp_path, {"GF-REGIONAL-BOOT-013": {"depends_on": ["GF-REGIONAL-FAKE-001"]}}
    )
    with pytest.raises(ValueError, match="unknown dependencies"):
        compile_regional_acceptance_plan(
            mode=PlanMode.LOCAL_PREACCEPTANCE, override_path=unknown_dependency
        )

    replacement = _write_override(
        tmp_path,
        {
            "GF-REGIONAL-BOOT-022": {
                "locks": [{"resource": "local-test", "mode": "exclusive"}]
            }
        },
    )
    with pytest.raises(ValueError, match="cannot replace lock"):
        compile_regional_acceptance_plan(override_path=replacement)


def test_dependency_cycle_is_rejected(tmp_path: Path) -> None:
    override = _write_override(
        tmp_path,
        {
            "GF-REGIONAL-BOOT-011": {"depends_on": ["GF-REGIONAL-BOOT-012"]},
            "GF-REGIONAL-BOOT-012": {"depends_on": ["GF-REGIONAL-BOOT-011"]},
        },
    )

    with pytest.raises(ValueError, match="dependency cycle"):
        compile_regional_acceptance_plan(
            mode=PlanMode.LOCAL_PREACCEPTANCE, override_path=override
        )


def test_order_rejects_duplicate_unknown_and_missing_cases(tmp_path: Path) -> None:
    duplicate_value = _read_yaml(ORDER)
    duplicate_phases = cast(list[dict[str, Any]], duplicate_value["phases"])
    duplicate_entries = cast(list[dict[str, Any]], duplicate_phases[0]["entries"])
    duplicate_entries.append({"case": "GF-REGIONAL-BOOT-011"})
    duplicate_order = _write_yaml(tmp_path / "duplicate.yaml", duplicate_value)
    with pytest.raises(ValueError, match="duplicate regional"):
        compile_regional_acceptance_plan(order_path=duplicate_order)

    unknown_value = _read_yaml(ORDER)
    unknown_phases = cast(list[dict[str, Any]], unknown_value["phases"])
    unknown_entries = cast(list[dict[str, Any]], unknown_phases[0]["entries"])
    unknown_entries.append({"case": "GF-REGIONAL-FAKE-001"})
    unknown_order = _write_yaml(tmp_path / "unknown.yaml", unknown_value)
    with pytest.raises(ValueError, match="unknown regional case IDs"):
        compile_regional_acceptance_plan(order_path=unknown_order)

    missing_value = _read_yaml(ORDER)
    missing_phases = cast(list[dict[str, Any]], missing_value["phases"])
    final_entries = cast(list[dict[str, Any]], missing_phases[-1]["entries"])
    assert final_entries.pop() == {"case": "GF-REGIONAL-COLLECT-015"}
    missing_order = _write_yaml(tmp_path / "missing.yaml", missing_value)
    with pytest.raises(ValueError, match="missing from execution order"):
        compile_regional_acceptance_plan(order_path=missing_order)


def test_order_requires_do_not_run_and_plan_output_is_stable(tmp_path: Path) -> None:
    value = _read_yaml(ORDER)
    value["do_not_run"] = []
    no_retired = _write_yaml(tmp_path / "no-retired.yaml", value)
    with pytest.raises(ValueError, match="must declare its DO_NOT_RUN"):
        compile_regional_acceptance_plan(order_path=no_retired)

    first = compile_regional_acceptance_plan().as_dict()
    second = compile_regional_acceptance_plan().as_dict()
    assert first == second
