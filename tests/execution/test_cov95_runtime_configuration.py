from __future__ import annotations

from uuid import UUID

import pytest

from gpu_fault.execution.config import ProductionExecutorConfig
from gpu_fault.execution.models import WorkflowExecutionError
from gpu_fault.models import WorkflowOperation


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        (
            "GPU_FAULT_ALLOWED_OPERATIONS",
            "FREEZE_EVIDENCE,UNKNOWN_OPERATION",
            "invalid GPU_FAULT_ALLOWED_OPERATIONS",
        ),
        ("GPU_FAULT_WORKFLOW_LEASE_DURATION_SECONDS", "29", "at least 30"),
        (
            "GPU_FAULT_JOB_WORKFLOW_MAX_LIFETIME_SECONDS",
            "0",
            "lifetimes must be positive",
        ),
        (
            "GPU_FAULT_NODE_WORKFLOW_MAX_LIFETIME_SECONDS",
            "0",
            "lifetimes must be positive",
        ),
        (
            "GPU_FAULT_WORKFLOW_STEP_TIMEOUT_SECONDS",
            "0",
            "step timeout must be positive",
        ),
        (
            "GPU_FAULT_WORKFLOW_STEP_WARNING_SECONDS",
            "0",
            "warning threshold must be positive",
        ),
    ],
)
def test_executor_mapping_refuses_invalid_operations_and_unbounded_lifecycle_values(
    key, value, message
) -> None:
    with pytest.raises((WorkflowExecutionError, RuntimeError), match=message):
        ProductionExecutorConfig.from_mapping({key: value})


@pytest.mark.parametrize(
    ("values", "identity"),
    [
        ({}, "unknown-host"),
        ({"HOSTNAME": "host-a"}, "host-a"),
        ({"HOSTNAME": "host-a", "POD_UID": "pod-a"}, "pod-a"),
    ],
)
def test_generated_executor_identity_is_per_process_and_uses_the_strongest_available_identity(
    values, identity
) -> None:
    first = ProductionExecutorConfig.from_mapping(values)
    second = ProductionExecutorConfig.from_mapping(values)
    prefix = f"gpu-fault-control-plane/{identity}/"
    assert first.executor_id.startswith(prefix), first.executor_id
    assert (
        str(UUID(first.executor_id[len(prefix) :])) == first.executor_id[len(prefix) :]
    ), first.executor_id
    assert first.executor_id != second.executor_id, (
        "two replicas must not share a generated lease identity"
    )
    assert first.enabled is False and first.allowed_operations == frozenset(), first


def test_explicit_executor_mapping_normalizes_whitespace_without_broadening_permissions() -> (
    None
):
    config = ProductionExecutorConfig.from_mapping(
        {
            "GPU_FAULT_EXECUTOR_ID": " unit-executor ",
            "GPU_FAULT_EXECUTOR_MODE": " ACTIVE ",
            "GPU_FAULT_ALLOWED_OPERATIONS": " FREEZE_EVIDENCE, , FREEZE_EVIDENCE ",
            "GPU_FAULT_ENABLE_WORKFLOW_PREEMPTION": "false",
        }
    )
    assert config.executor_id == "unit-executor" and config.enabled is True, config
    assert config.allowed_operations == frozenset(
        {WorkflowOperation.FREEZE_EVIDENCE}
    ), config
    assert config.workflow_preemption_enabled is False, config
