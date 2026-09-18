from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests._cov95_acceptance_plan import RETIRED, A, B, PlanInputs
from tools.regional_acceptance_plan import ExecutorKind, PlanMode


@pytest.mark.parametrize(
    "field,value,problem",
    [
        ("schema_version", 2, "schema_version must be 1"),
        ("reviewed", False, "reviewed: true"),
        ("reviewed_by", None, "non-empty string"),
        ("reviewed_at", "2026-09-12", "RFC3339"),
        ("reviewed_at", "2026-99-12T00:00:00Z", "RFC3339"),
        ("order_sha256", "bad", "SHA-256"),
        ("order_sha256", "0" * 64, "order_sha256 does not match"),
        ("catalog_sha256", "0" * 64, "catalog_sha256 does not match"),
        ("proposal_sha256", "bad", "SHA-256"),
        ("cases", [], "must be a mapping"),
    ],
)
def test_review_metadata_must_bind_valid_current_inputs(
    tmp_path: Path, field: str, value: Any, problem: str
) -> None:
    inputs = PlanInputs(tmp_path)
    inputs.override({A: {"executor": "codex-manual"}}, **{field: value})
    with pytest.raises(ValueError, match=problem):
        inputs.compile(override_path=inputs.override_path)


@pytest.mark.parametrize(
    "constraints,problem",
    [
        ({A: {}}, "at least one constraint"),
        ({A: {"parallel_safe": True}}, "unknown fields"),
        ({A: {"executor": "command"}}, "may only be codex-manual"),
        ({RETIRED: {"executor": "codex-manual"}}, "DO_NOT_RUN"),
        ({"GF-REGIONAL-CAP-099": {"executor": "codex-manual"}}, "unknown case ID"),
        ({A: {"depends_on": [RETIRED]}}, "depends on DO_NOT_RUN"),
        ({A: {"depends_on": ["GF-REGIONAL-CAP-099"]}}, "unknown dependencies"),
        ({A: {"depends_on": [A]}}, "depend on itself"),
        ({A: {"depends_on": [B]}}, "dependency cycle"),
        (
            {
                A: {
                    "locks": [
                        {"resource": "evidence", "mode": "shared"},
                        {"resource": "evidence", "mode": "shared"},
                    ]
                }
            },
            "duplicate lock",
        ),
    ],
)
def test_reviewed_override_cannot_widen_execution_or_corrupt_dependencies(
    tmp_path: Path, constraints: dict[str, Any], problem: str
) -> None:
    inputs = PlanInputs(tmp_path)
    inputs.override(constraints)
    with pytest.raises(ValueError, match=problem):
        inputs.compile(override_path=inputs.override_path)


def test_review_cannot_redirect_an_automated_case_to_a_manual_backend(
    tmp_path: Path,
) -> None:
    inputs = PlanInputs(tmp_path)
    inputs.catalog["test_cases"][0].update(
        automation="pytest",
        pytest_nodeid="tests/test_case_scheduler.py::test_results_remain_in_declared_order",
    )
    inputs.write()
    inputs.override({A: {"executor": "codex-manual"}})
    with pytest.raises(ValueError, match="only for manual cases"):
        inputs.compile(override_path=inputs.override_path)


@pytest.mark.parametrize("mode", list(PlanMode))
def test_reviewed_manual_execution_stays_read_only_and_mode_scoped(
    tmp_path: Path, mode: PlanMode
) -> None:
    inputs = PlanInputs(tmp_path)
    metadata = inputs.override(
        {A: {"executor": "codex-manual"}},
        reviewed_by=" reviewer ",
        proposal_sha256="a" * 64,
    )
    plan = inputs.compile(mode=mode, override_path=inputs.override_path)
    selected = plan.case(A)
    assert selected.executor is ExecutorKind.CODEX_MANUAL
    assert selected.execution.read_only is True
    assert selected.execution.repair_allowed is False
    assert selected.blocked is (mode is PlanMode.LOCAL_PREACCEPTANCE)
    assert selected.execution.parallel_safe is (mode is PlanMode.COLLECT_ALL)
    assert plan.review_metadata is not None
    assert plan.review_metadata.as_dict() == {
        "reviewed_by": "reviewer",
        "reviewed_at": metadata["reviewed_at"],
        "order_sha256": metadata["order_sha256"],
        "catalog_sha256": metadata["catalog_sha256"],
        "proposal_sha256": "a" * 64,
    }
    assert plan.case(RETIRED).executor is ExecutorKind.DO_NOT_RUN


@pytest.mark.parametrize("replace_mode", [False, True])
def test_override_adds_locks_without_replacing_a_catalog_lock(
    tmp_path: Path, replace_mode: bool
) -> None:
    inputs = PlanInputs(tmp_path)
    inputs.catalog["test_cases"][0]["execution"] = {
        "parallel_safe": True,
        "locks": [{"resource": "evidence", "mode": "shared"}],
    }
    inputs.write()
    additions = [
        {"resource": "evidence", "mode": "exclusive" if replace_mode else "shared"},
        {"resource": "local-ledger", "mode": "exclusive"},
    ]
    inputs.override({A: {"locks": additions}})
    if replace_mode:
        with pytest.raises(ValueError, match="cannot replace lock"):
            inputs.compile(override_path=inputs.override_path)
    else:
        plan = inputs.compile(override_path=inputs.override_path)
        assert [lock.as_dict() for lock in plan.case(A).execution.locks] == additions


@pytest.mark.parametrize(
    "field,value,expected",
    [
        (
            "related_pytest",
            "tests/example.py::test_case",
            ("tests/example.py::test_case",),
        ),
        ("gate", "scripts/example.py", ("python3", "scripts/example.py")),
        ("gate", "local-gate", ("local-gate",)),
    ],
)
def test_local_proxy_plan_preserves_its_declared_test_or_gate(
    tmp_path: Path, field: str, value: str, expected: tuple[str, ...]
) -> None:
    inputs = PlanInputs(tmp_path)
    inputs.catalog["test_cases"][0][field] = value
    inputs.write()
    selected = inputs.compile(mode=PlanMode.LOCAL_PREACCEPTANCE).case(A)
    assert selected.local_proxy is True
    assert selected.blocked is False
    assert selected.execution.read_only is True
    assert selected.execution.repair_allowed is False
    assert (
        selected.pytest_nodeids if field == "related_pytest" else selected.command
    ) == expected
