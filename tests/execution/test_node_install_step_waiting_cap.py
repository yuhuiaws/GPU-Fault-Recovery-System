"""The per-step waiting cap against a node install that runs for 20 minutes.

REMEDIATE_DRIVER, UPDATE_SOFTWARE_FIRMWARE and REMEDIATE_EFA_DRIVER hand the
node agent an install whose subprocess timeout is 1800 s
(``node_agent/operations/remediation.py``) and whose unit is allowed 1900 s to
stop (``deploy/systemd/gpu-fault-node-agent.service``). The control plane polls
the step WAITING for as long as the agent works, and
``step_bounds.bounded_waiting_outcome`` measures that wait from the step's
FIRST record -- ``record_attempt`` deliberately keeps ``started_at`` across
polls -- so a cap below the install's own timeout fails the step on the
generic "past the per-step cap" path while the install is still running.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from gpu_fault.execution import step_bounds
from gpu_fault.execution.config import (
    ProductionExecutorConfig,
    WorkflowDispatcherConfig,
    WorkflowExecutionError,
    validate_timing_relationships,
)
from gpu_fault.execution.models import WorkflowStepOutcome
from gpu_fault.models import WorkflowOperation, WorkflowStepStatus
from tests._builders import workflow_request, workflow_step, workflow_step_execution

INSTALLS = [
    WorkflowOperation.REMEDIATE_DRIVER,
    WorkflowOperation.UPDATE_SOFTWARE_FIRMWARE,
    WorkflowOperation.REMEDIATE_EFA_DRIVER,
]
AGENT_INSTALL_TIMEOUT_SECONDS = 1800
AGENT_UNIT_STOP_TIMEOUT_SECONDS = 1900
KNOB = "GPU_FAULT_WORKFLOW_INSTALL_STEP_TIMEOUT_SECONDS"
TWENTY_MINUTES = 20 * 60


def _executor(**values: str) -> SimpleNamespace:
    return SimpleNamespace(config=ProductionExecutorConfig.from_mapping(values))


def _pending_for(operation: WorkflowOperation, seconds: int):
    """A single-node step of ``operation`` whose agent accepted it ``seconds`` ago."""

    workflow = workflow_request(
        "wf-install",
        "inc-install",
        official_steps=[workflow_step(operation, node_ids=["node-a"])],
        step_executions=[
            workflow_step_execution(
                0,
                operation,
                WorkflowStepStatus.WAITING,
                details={
                    "node_action_command_id": "workflow/install/node-a",
                    "node_action_state": "PENDING",
                },
                started_at=datetime.now(timezone.utc) - timedelta(seconds=seconds),
            )
        ],
    )
    return workflow, workflow.official_steps[0]


def _pending_poll() -> WorkflowStepOutcome:
    return WorkflowStepOutcome.waiting(
        operation_id="workflow/install",
        details={
            "node_action_command_id": "workflow/install/node-a",
            "node_action_state": "PENDING",
        },
    )


# --- how the cap measures ---------------------------------------------------------


def test_a_waiting_poll_keeps_the_steps_first_start_time() -> None:
    """The evidence behind the cap: polling does not move the step's clock."""

    workflow, step = _pending_for(WorkflowOperation.REMEDIATE_DRIVER, 900)
    first_start = workflow.step_executions[0].started_at

    polled = step_bounds.record_attempt(workflow, step, 0, _pending_poll())

    assert [item.started_at for item in polled.step_executions] == [first_start], (
        "a WAITING poll must not refresh started_at, or no step could ever be capped"
    )
    bounded = step_bounds.bounded_waiting_outcome(
        _executor(), polled, step, 0, _pending_poll()
    )
    assert bounded.details["step_waiting_seconds"] >= 900, bounded.details


# --- the installs ---------------------------------------------------------------------


@pytest.mark.parametrize("operation", INSTALLS, ids=lambda op: op.value)
def test_a_twenty_minute_install_is_still_waiting(operation: WorkflowOperation) -> None:
    workflow, step = _pending_for(operation, TWENTY_MINUTES)

    bounded = step_bounds.bounded_waiting_outcome(
        _executor(), workflow, step, 0, _pending_poll()
    )

    assert bounded.status is WorkflowStepStatus.WAITING, (
        f"the control plane gave up on a running install: {bounded.error}"
    )
    assert bounded.details["node_action_state"] == "PENDING", bounded.details
    assert "step_waiting_timeout_seconds" not in bounded.details, bounded.details


@pytest.mark.parametrize("operation", INSTALLS, ids=lambda op: op.value)
def test_the_install_ceiling_clears_the_agents_own_timeouts(
    operation: WorkflowOperation,
) -> None:
    """A cap below the agent's 1800 s install timeout fails the step first, on
    the generic path, so the agent's own verdict for the install never lands."""

    config = ProductionExecutorConfig.from_mapping({})

    assert config.step_waiting_limit(operation) == AGENT_UNIT_STOP_TIMEOUT_SECONDS
    assert config.step_waiting_limit(operation) > AGENT_INSTALL_TIMEOUT_SECONDS
    assert config.step_waiting_warning_limit(operation) == (
        AGENT_UNIT_STOP_TIMEOUT_SECONDS - 300
    ), "the warning lead time carries over to the raised ceiling"


def test_a_diagnostic_step_still_fails_at_the_default_cap() -> None:
    """Only the three installs are raised; a bundle collection is not."""

    workflow, step = _pending_for(
        WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE, TWENTY_MINUTES
    )

    bounded = step_bounds.bounded_waiting_outcome(
        _executor(), workflow, step, 0, _pending_poll()
    )

    assert bounded.status is WorkflowStepStatus.FAILED, bounded
    assert bounded.details["step_waiting_timeout_seconds"] == 600, bounded.details
    assert "past the 600s per-step cap" in (bounded.error or ""), bounded.error


# --- the knob -------------------------------------------------------------------------


def test_the_install_ceiling_is_one_knob_for_the_three_installs() -> None:
    raised = ProductionExecutorConfig.from_mapping({KNOB: "2400"})

    assert [raised.step_waiting_limit(op) for op in INSTALLS] == [2400, 2400, 2400]
    assert raised.step_waiting_limit(WorkflowOperation.RESET_GPU) == 600
    assert raised.step_waiting_limit(WorkflowOperation.REPLACE_NODE) == 1800, (
        "the managed recovery window is a different knob"
    )


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        ("0", "must be positive"),
        ("-5", "must be positive"),
        ("300", "below the default"),
    ],
)
def test_an_install_ceiling_that_cannot_hold_an_install_is_refused(
    raw: str, match: str
) -> None:
    with pytest.raises(WorkflowExecutionError, match=match):
        ProductionExecutorConfig.from_mapping({KNOB: raw})


def test_a_directly_built_config_carries_the_install_ceiling() -> None:
    """A test fixture or the simulation context must not cap an install at
    the default either -- the same reason the managed window is defaulted."""

    config = ProductionExecutorConfig(
        enabled=True,
        executor_id="executor-direct",
        allowed_operations=frozenset(INSTALLS),
    )

    assert [config.step_waiting_limit(op) for op in INSTALLS] == [1900, 1900, 1900]


def test_the_shipped_install_ceiling_sits_inside_the_node_lifetime() -> None:
    executor = ProductionExecutorConfig.from_mapping({})
    dispatcher = WorkflowDispatcherConfig.from_mapping({}, executor_enabled=True)

    assert (
        validate_timing_relationships(
            executor, dispatcher, verify_max_attempts=60, branch_max_rungs=2
        )
        == []
    )
    assert executor.step_waiting_limit(WorkflowOperation.REMEDIATE_DRIVER) <= (
        executor.node_workflow_lifetime_seconds
    )
