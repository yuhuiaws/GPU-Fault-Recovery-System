"""A step that waits for a human is bounded by the operator's clock.

CHECK_MECHANICALS means "wait for the mechanical inspection and its explicit
confirmation" -- hours, sometimes a day. It was capped by the default
ten-minute step ceiling and by the workflow's 30-minute execution deadline
and one-hour lifetime, so COLLECT-009 could never pass the step. The ceiling
is now one working day by default, still finite and still tunable, and the
deadlines of a workflow that contains the step are floored to it.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.execution.config import (
    DEFAULT_OPERATOR_ACKNOWLEDGEMENT_TIMEOUT_SECONDS,
    ProductionExecutorConfig,
    WorkflowExecutionError,
)
from gpu_fault.execution.restart_budget_preflight import claim_deadlines
from gpu_fault.models import WorkflowOperation
from tests._builders import workflow_request, workflow_step

NOW = datetime(2026, 9, 6, 14, 0, tzinfo=timezone.utc)
INSPECT = WorkflowOperation.CHECK_MECHANICALS
RESET = WorkflowOperation.RESET_GPU


def test_the_inspection_step_waits_a_working_day_by_default_and_is_tunable():
    default = ProductionExecutorConfig.from_mapping({})
    tightened = ProductionExecutorConfig.from_mapping(
        {"GPU_FAULT_OPERATOR_ACKNOWLEDGEMENT_TIMEOUT_SECONDS": "7200"}
    )

    assert (
        default.step_waiting_limit(INSPECT)
        == DEFAULT_OPERATOR_ACKNOWLEDGEMENT_TIMEOUT_SECONDS
        == 86400
    )
    assert default.step_waiting_limit(RESET) == default.step_waiting_timeout_seconds
    assert tightened.step_waiting_limit(INSPECT) == 7200
    assert tightened.operator_acknowledgement_timeout_seconds == 7200


def test_an_acknowledgement_ceiling_below_the_default_step_cap_is_refused():
    with pytest.raises(WorkflowExecutionError, match="below the default step timeout"):
        ProductionExecutorConfig.from_mapping(
            {"GPU_FAULT_OPERATOR_ACKNOWLEDGEMENT_TIMEOUT_SECONDS": "300"}
        )
    with pytest.raises(WorkflowExecutionError, match="must be positive"):
        ProductionExecutorConfig.from_mapping(
            {"GPU_FAULT_OPERATOR_ACKNOWLEDGEMENT_TIMEOUT_SECONDS": "0"}
        )


def test_a_workflow_waiting_on_an_inspection_is_not_killed_by_the_hour_deadlines():
    inspecting = workflow_request(
        "wf-inspect",
        "inc",
        official_steps=[
            workflow_step(WorkflowOperation.MARK_UNSCHEDULABLE, node_ids=["node-a"]),
            workflow_step(INSPECT, node_ids=["node-a"], depends_on_step_indexes=[0]),
        ],
    )
    plain = workflow_request("wf-reset", "inc", official_steps=[workflow_step(RESET)])
    kwargs = dict(
        timeout_seconds=1800,
        job_lifetime_seconds=3600,
        node_lifetime_seconds=3600,
        operator_acknowledgement_seconds=86400,
    )

    execution, lifetime = claim_deadlines(inspecting, NOW, **kwargs)
    plain_execution, plain_lifetime = claim_deadlines(plain, NOW, **kwargs)

    assert execution == NOW + timedelta(days=1)
    assert lifetime == NOW + timedelta(days=1)
    # A workflow without the step keeps the ordinary bounds.
    assert plain_execution == NOW + timedelta(seconds=1800)
    assert plain_lifetime == NOW + timedelta(hours=1)


def test_an_inherited_shorter_lifetime_is_lifted_to_the_acknowledgement_floor():
    """A replacement workflow inherits its predecessor's lifetime (F-N1); if
    it now waits on an inspection, that lifetime is extended to the floor
    rather than failing the inspection at the inherited hour."""

    inherited = workflow_request(
        "wf-support",
        "inc",
        official_steps=[workflow_step(INSPECT, node_ids=["node-a"])],
        lifetime_deadline_at=NOW + timedelta(minutes=20),
    )

    execution, lifetime = claim_deadlines(
        inherited,
        NOW,
        timeout_seconds=1800,
        job_lifetime_seconds=3600,
        node_lifetime_seconds=3600,
        operator_acknowledgement_seconds=86400,
    )

    assert lifetime == NOW + timedelta(days=1)
    assert execution == NOW + timedelta(days=1)
