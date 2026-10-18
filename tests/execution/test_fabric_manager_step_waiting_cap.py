"""The steps that may start Fabric Manager wait for the Agent's 780 s (batch 131).

The Agent now gives ``nvidia-fabricmanager`` 780 s to start or restart (the
R580 unit's ``TimeoutStartSec=720`` plus a minute; on NVL5 the unit starts the
NVLink Subnet Manager first) where every other unit keeps 120 s. A window's
RESTORE starts it (its restore loop, or the start after a full GPU + NVSwitch
reset that found it down) and RESTART_FABRIC_MANAGER restarts it; the control
plane polls those steps WAITING while the Agent works, and the default 600 s cap
would fail them on the generic "past the per-step cap" path while the unit is
still legitimately starting. Their cap is 1200 s: the 780 s plus the rest of a
restore (container restore wait 180 s, settle 30 s, the other units). It never
exceeds the node workflow lifetime, so a compression drill keeps a bootable set,
and it fits the executor's per-command limit.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from gpu_fault.cluster_executor import DEFAULT_MAX_EXECUTION_SECONDS
from gpu_fault.cluster_scope import FABRIC_MANAGER_START_TIMEOUT_SECONDS
from gpu_fault.execution import step_bounds
from gpu_fault.execution.config import (
    FABRIC_MANAGER_START_OPERATIONS,
    FABRIC_MANAGER_STEP_TIMEOUT_SECONDS,
    ProductionExecutorConfig,
    validate_timing_from_environment,
)
from gpu_fault.execution.models import WorkflowStepOutcome
from gpu_fault.models import WorkflowOperation, WorkflowStepStatus
from tests._builders import workflow_request, workflow_step, workflow_step_execution

STARTS_FABRIC_MANAGER = [
    WorkflowOperation.RESTORE_GPU_SERVICES,
    WorkflowOperation.RESTART_FABRIC_MANAGER,
]
CONTAINER_RESTORE_SECONDS = 180
RESTORE_SETTLE_SECONDS = 30


def _executor(**values: str) -> SimpleNamespace:
    return SimpleNamespace(config=ProductionExecutorConfig.from_mapping(values))


def _pending_for(operation: WorkflowOperation, seconds: int):
    workflow = workflow_request(
        "wf-fabric",
        "inc-fabric",
        official_steps=[workflow_step(operation, node_ids=["node-a"])],
        step_executions=[
            workflow_step_execution(
                0,
                operation,
                WorkflowStepStatus.WAITING,
                details={
                    "node_action_command_id": "workflow/fabric/node-a",
                    "node_action_state": "PENDING",
                },
                started_at=datetime.now(timezone.utc) - timedelta(seconds=seconds),
            )
        ],
    )
    return workflow, workflow.official_steps[0]


def _pending_poll() -> WorkflowStepOutcome:
    return WorkflowStepOutcome.waiting(
        operation_id="workflow/fabric",
        details={
            "node_action_command_id": "workflow/fabric/node-a",
            "node_action_state": "PENDING",
        },
    )


def test_the_cap_clears_the_agent_s_fabric_manager_start_and_a_restore() -> None:
    config = ProductionExecutorConfig.from_mapping({})

    assert list(FABRIC_MANAGER_START_OPERATIONS) == STARTS_FABRIC_MANAGER
    assert FABRIC_MANAGER_STEP_TIMEOUT_SECONDS == 1200
    assert FABRIC_MANAGER_STEP_TIMEOUT_SECONDS > (
        FABRIC_MANAGER_START_TIMEOUT_SECONDS
        + CONTAINER_RESTORE_SECONDS
        + RESTORE_SETTLE_SECONDS
    ), "the Fabric Manager start plus the rest of a restore"
    for operation in STARTS_FABRIC_MANAGER:
        assert config.step_waiting_limit(operation) == 1200, operation
        assert config.step_waiting_warning_limit(operation) == 900, (
            "the warning keeps the shipped 300 s lead"
        )
    for operation in (
        WorkflowOperation.QUIESCE_GPU_SERVICES,
        WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
        WorkflowOperation.RESET_GPU,
        WorkflowOperation.RESET_ALL_GPUS_NVSWITCHES,
        WorkflowOperation.VALIDATE_GPU,
    ):
        assert config.step_waiting_limit(operation) == 600, operation


def test_the_cap_fits_the_executor_s_per_command_limit_and_the_lifetime() -> None:
    config = ProductionExecutorConfig.from_mapping({})

    assert DEFAULT_MAX_EXECUTION_SECONDS == 1800
    assert FABRIC_MANAGER_STEP_TIMEOUT_SECONDS < DEFAULT_MAX_EXECUTION_SECONDS, (
        "one claimed command may run for the whole step"
    )
    assert config.node_workflow_lifetime_seconds == 3600
    assert validate_timing_from_environment({}) == [], (
        "the shipped set stays consistent with the raised cap"
    )


@pytest.mark.parametrize("operation", STARTS_FABRIC_MANAGER, ids=lambda op: op.value)
def test_a_fabric_manager_still_starting_at_ten_minutes_is_still_waiting(
    operation: WorkflowOperation,
) -> None:
    workflow, step = _pending_for(operation, 700)

    bounded = step_bounds.bounded_waiting_outcome(
        _executor(), workflow, step, 0, _pending_poll()
    )

    assert bounded.status is WorkflowStepStatus.WAITING, (
        f"the control plane gave up on a Fabric Manager still starting: {bounded.error}"
    )
    assert "step_waiting_timeout_seconds" not in bounded.details, bounded.details

    workflow, step = _pending_for(operation, 1201)
    bounded = step_bounds.bounded_waiting_outcome(
        _executor(), workflow, step, 0, _pending_poll()
    )
    assert bounded.status is WorkflowStepStatus.FAILED, bounded
    assert "past the 1200s per-step cap" in (bounded.error or ""), bounded.error


def test_a_compressed_lifetime_keeps_the_cap_at_or_below_it() -> None:
    """DESTR-018's window lowers the lifetime and the step cap together; the
    derived ceiling follows instead of refusing the control plane's boot."""

    compressed = {
        "GPU_FAULT_NODE_WORKFLOW_MAX_LIFETIME_SECONDS": "180",
        "GPU_FAULT_WORKFLOW_EXECUTION_TIMEOUT_SECONDS": "180",
        "GPU_FAULT_WORKFLOW_STEP_TIMEOUT_SECONDS": "180",
        "GPU_FAULT_WORKFLOW_STEP_WARNING_SECONDS": "90",
        "GPU_FAULT_HYPERPOD_MANAGED_RECOVERY_TIMEOUT_SECONDS": "180",
        "GPU_FAULT_WORKFLOW_INSTALL_STEP_TIMEOUT_SECONDS": "180",
        "GPU_FAULT_WORKFLOW_INSTALL_CONTAINMENT_SECONDS": "0",
        "GPU_FAULT_WORKFLOW_LEASE_DURATION_SECONDS": "90",
    }
    config = ProductionExecutorConfig.from_mapping(compressed)

    for operation in STARTS_FABRIC_MANAGER:
        assert config.step_waiting_limit(operation) == 180, operation
    assert validate_timing_from_environment(compressed) == [], "the window boots"

    raised = ProductionExecutorConfig.from_mapping(
        {"GPU_FAULT_WORKFLOW_STEP_TIMEOUT_SECONDS": "1500"}
    )
    for operation in STARTS_FABRIC_MANAGER:
        assert raised.step_waiting_limit(operation) == 1500, (
            "never below the default cap"
        )


def test_a_directly_built_config_keeps_the_raised_cap() -> None:
    config = ProductionExecutorConfig(
        enabled=True, executor_id="direct", allowed_operations=frozenset()
    )

    for operation in STARTS_FABRIC_MANAGER:
        assert config.step_waiting_limit(operation) == 1200, operation
