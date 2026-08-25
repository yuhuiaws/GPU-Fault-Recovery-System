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
    assert RECOVERY_OPERATION_RANK[WorkflowOperation.REPLACE_NODE] == 70
    assert MULTI_NODE_BARRIER_OPERATIONS == {
        WorkflowOperation.RESET_GPU,
        WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES,
    }
    assert OPERATION_REGISTRY[WorkflowOperation.RESTART_VM].planning_only


@pytest.mark.parametrize(
    ("operation", "field"),
    [
        (WorkflowOperation.RESET_GPU, "preemption_non_cancelable"),
        (WorkflowOperation.RUN_DCGM_DIAGNOSTIC, "hardware_escalation_relevant"),
    ],
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
