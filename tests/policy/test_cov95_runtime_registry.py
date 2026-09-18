from __future__ import annotations

from dataclasses import replace
from enum import StrEnum

import pytest

from gpu_fault import operation_registry as registry
from gpu_fault.models import WorkflowOperation
from gpu_fault.operation_registry import OperationAdapter, OperationResourceClaim


class UnregisteredOperation(StrEnum):
    NEW_ACTION = "NEW_ACTION"


@pytest.mark.parametrize("change", ["missing", "extra"])
def test_registry_rejects_an_incomplete_or_unregistered_operation_set(
    change: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = dict(registry.OPERATION_REGISTRY)
    if change == "missing":
        values.pop(WorkflowOperation.FREEZE_EVIDENCE)
    else:
        values[UnregisteredOperation.NEW_ACTION] = values[
            WorkflowOperation.FREEZE_EVIDENCE
        ]
    monkeypatch.setattr(registry, "OPERATION_REGISTRY", values)
    with pytest.raises(RuntimeError, match="WorkflowOperation registry mismatch"):
        registry.validate_operation_registry()


@pytest.mark.parametrize(
    ("operation", "updates", "reason"),
    [
        (
            WorkflowOperation.FREEZE_EVIDENCE,
            {"adapters": frozenset()},
            "has no adapter and is not planning-only",
        ),
        (
            WorkflowOperation.FREEZE_EVIDENCE,
            {"recovery_rank": -1},
            "negative recovery rank",
        ),
        (
            WorkflowOperation.RESTART_NODE,
            {"hardware_escalation_relevant": None},
            "must explicitly declare hardware_escalation_relevant",
        ),
        (
            WorkflowOperation.FREEZE_EVIDENCE,
            {"generation_stable_command_id": True},
            "but is not a node action",
        ),
        (
            WorkflowOperation.REMEDIATE_DRIVER,
            {"maintenance_generation_scoped": True},
            "already pins its command_id",
        ),
        (
            WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE,
            {"generation_stable_command_id": True},
            "but is not node-mutating",
        ),
        (
            WorkflowOperation.MARK_UNSCHEDULABLE,
            {
                "adapters": frozenset({OperationAdapter.NODE_ACTION}),
                "generation_stable_command_id": True,
                "resource_claims": frozenset(
                    {OperationResourceClaim.SCHEDULER_MUTATION}
                ),
            },
            "but is not node-mutating",
        ),
        (
            WorkflowOperation.FREEZE_EVIDENCE,
            {"dominates": frozenset({UnregisteredOperation.NEW_ACTION})},
            "dominates unknown operations",
        ),
    ],
)
def test_registry_rejects_missing_risk_ownership_and_incompatible_idempotency_semantics(
    operation, updates, reason: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(
        registry.OPERATION_REGISTRY,
        operation,
        replace(registry.OPERATION_REGISTRY[operation], **updates),
    )
    with pytest.raises(RuntimeError, match=reason):
        registry.validate_operation_registry()


def test_planning_only_operation_can_omit_an_adapter_but_is_not_dispatchable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operation = WorkflowOperation.FREEZE_EVIDENCE
    monkeypatch.setitem(
        registry.OPERATION_REGISTRY,
        operation,
        replace(
            registry.OPERATION_REGISTRY[operation],
            adapters=frozenset(),
            planning_only=True,
        ),
    )
    registry.validate_operation_registry()
    assert all(
        operation not in registry.operations_for_adapter(adapter)
        for adapter in OperationAdapter
    ), "a planning-only placeholder must not be advertised by an executable adapter"


def test_dominance_requires_every_transitive_recovery_step_to_be_declared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operation = WorkflowOperation.RESTART_NODE
    current = registry.OPERATION_REGISTRY[operation]
    monkeypatch.setitem(
        registry.OPERATION_REGISTRY,
        operation,
        replace(current, dominates=current.dominates - {WorkflowOperation.RESET_GPU}),
    )
    with pytest.raises(RuntimeError, match="dominance is not transitively closed"):
        registry.validate_operation_registry()
