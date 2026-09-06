from __future__ import annotations

from dataclasses import replace

import pytest

from gpu_fault.models import WorkflowOperation
from gpu_fault.operation_registry import (
    MULTI_NODE_BARRIER_OPERATIONS,
    OPERATION_CAPABILITY,
    OPERATION_REGISTRY,
    RECOVERY_OPERATION_RANK,
    OperationAdapter,
    operations_for_adapter,
    validate_operation_registry,
)


def test_operation_registry_is_exhaustive() -> None:
    validate_operation_registry()
    assert set(OPERATION_REGISTRY) == set(WorkflowOperation)
    assert set(OPERATION_CAPABILITY) == set(WorkflowOperation)


def test_every_executable_operation_has_an_adapter() -> None:
    for operation, semantics in OPERATION_REGISTRY.items():
        assert semantics.scope is not None, operation
        assert semantics.capability is not None, operation
        assert semantics.recovery_rank >= 0, operation
        assert semantics.adapters or semantics.planning_only


def test_adapter_operation_sets_are_derived_from_registry() -> None:
    assert WorkflowOperation.RESET_GPU in operations_for_adapter(
        OperationAdapter.NODE_ACTION
    )
    assert WorkflowOperation.MARK_UNSCHEDULABLE in operations_for_adapter(
        OperationAdapter.KUBERNETES
    )
    assert WorkflowOperation.VALIDATE_GPU in operations_for_adapter(
        OperationAdapter.GPU_VALIDATION
    )
    assert WorkflowOperation.RESTART_NODE in operations_for_adapter(
        OperationAdapter.HYPERPOD
    )


def test_rank_and_barrier_semantics_are_explicit() -> None:
    assert RECOVERY_OPERATION_RANK[WorkflowOperation.REPLACE_NODE] == 75
    assert MULTI_NODE_BARRIER_OPERATIONS == {
        WorkflowOperation.RESET_GPU,
        WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES,
    }
    assert OPERATION_REGISTRY[WorkflowOperation.RESTART_VM].planning_only


@pytest.mark.parametrize(
    ("operation", "field"),
    [(WorkflowOperation.RUN_DCGM_DIAGNOSTIC, "hardware_escalation_relevant")],
)
def test_risk_semantics_must_be_explicit(monkeypatch, operation, field) -> None:
    monkeypatch.setitem(
        OPERATION_REGISTRY,
        operation,
        replace(OPERATION_REGISTRY[operation], **{field: None}),
    )

    with pytest.raises(
        RuntimeError, match=f"{operation.value} must explicitly declare.*{field}"
    ):
        validate_operation_registry()


def _dominance() -> dict[WorkflowOperation, frozenset[WorkflowOperation]]:
    return {
        operation: semantics.dominates
        for operation, semantics in OPERATION_REGISTRY.items()
        if semantics.dominates
    }


def test_dominance_implies_a_strictly_higher_recovery_rank() -> None:
    """The arbiter compares ranks first and dominance second (F-C5 / P0-67A).

    A tie between the two ends of a declared dominance let ``_winner`` pick
    either side, so the incident's declared action and the action actually
    executed could disagree.
    """

    violations = [
        (operation.value, dominated.value)
        for operation, dominated_set in _dominance().items()
        for dominated in dominated_set
        if not (
            OPERATION_REGISTRY[operation].recovery_rank
            > OPERATION_REGISTRY[dominated].recovery_rank
        )
    ]

    assert violations == []


def test_dominance_is_transitively_closed() -> None:
    dominance = _dominance()
    missing = [
        (operation.value, grandchild.value)
        for operation, dominated_set in dominance.items()
        for dominated in dominated_set
        for grandchild in dominance.get(dominated, frozenset())
        if grandchild not in dominated_set
    ]

    assert missing == []


def test_dominance_has_no_cycle() -> None:
    dominance = _dominance()
    for start in dominance:
        frontier = set(dominance[start])
        seen: set[WorkflowOperation] = set()
        while frontier:
            current = frontier.pop()
            assert current is not start, f"{start.value} dominates itself transitively"
            if current in seen:
                continue
            seen.add(current)
            frontier |= dominance.get(current, frozenset())


def test_registry_validation_rejects_a_dominance_rank_tie(monkeypatch) -> None:
    monkeypatch.setitem(
        OPERATION_REGISTRY,
        WorkflowOperation.RESET_GPU,
        replace(
            OPERATION_REGISTRY[WorkflowOperation.RESET_GPU],
            recovery_rank=OPERATION_REGISTRY[
                WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES
            ].recovery_rank,
        ),
    )

    with pytest.raises(RuntimeError, match="RESET_ALL_GPUS_NVSWITCHES.*RESET_GPU"):
        validate_operation_registry()
