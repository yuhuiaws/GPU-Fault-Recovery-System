"""Which claim a workflow makes: admission for its pending budgeted steps, or a
hold of what it already holds once its own budgeted work has begun."""

from __future__ import annotations

from gpu_fault.execution.remediation_budget import (
    RemediationBudgetPolicy,
    budget_work_begun,
    remediation_budget_claims,
)
from gpu_fault.models import WorkflowStatus, WorkflowStepStatus
from tests._builders import (
    fault_incident,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)
from tests.execution._budget_hold_support import REBOOT, RESET, RESTORE_GPU, STOP


def _incident():
    return fault_incident("incident-x", "event-x", node_ids=["node-x"])


def _plan(**values):
    return workflow_request(
        "workflow-x",
        "incident-x",
        official_steps=[
            workflow_step(STOP, node_ids=["node-x"]),
            workflow_step(RESET, node_ids=["node-x"]),
            workflow_step(REBOOT, node_ids=["node-x"]),
        ],
        **values,
    )


def test_a_fresh_workflow_is_admitted_for_its_pending_budgeted_steps():
    workflow = _plan()
    claims = remediation_budget_claims(
        RemediationBudgetPolicy(), workflow, _incident(), workflow.official_steps
    )

    assert budget_work_begun(workflow, workflow.official_steps) is False
    assert sorted(claims) == [
        "class:cluster-a:GPU_RUNTIME_MUTATION",
        "class:cluster-a:NODE_LIFECYCLE_MUTATION",
        "class:cluster-a:WORKLOAD_MUTATION",
        "cluster:cluster-a",
        "node:cluster-a:node-x",
        "region",
    ]


def test_a_started_workflow_holds_what_it_holds_plus_its_own_plan():
    workflow = _plan(
        status=WorkflowStatus.RUNNING,
        completed_step_indexes=[0],
        remediation_budget_claims=["cluster:cluster-a", "region"],
        remediation_budget_limits={"cluster:cluster-a": 5, "region": 20},
    )
    claims = remediation_budget_claims(
        RemediationBudgetPolicy(), workflow, _incident(), workflow.official_steps
    )

    assert budget_work_begun(workflow, workflow.official_steps) is True
    # Completed STOP keeps its class: the hold is the whole plan's, until the end.
    assert "class:cluster-a:WORKLOAD_MUTATION" in claims
    assert "class:cluster-a:NODE_LIFECYCLE_MUTATION" in claims
    assert claims["cluster:cluster-a"] == 5


def test_inherited_containment_does_not_make_a_successor_started():
    """A preempting successor inherits STOP; its own work has not begun."""

    workflow = _plan(
        completed_step_indexes=[0],
        inherited_step_indexes=[0],
        step_executions=[workflow_step_execution(0, STOP)],
    )
    claims = remediation_budget_claims(
        RemediationBudgetPolicy(), workflow, _incident(), workflow.official_steps
    )

    assert budget_work_begun(workflow, workflow.official_steps) is False
    assert "class:cluster-a:WORKLOAD_MUTATION" not in claims
    assert "class:cluster-a:NODE_LIFECYCLE_MUTATION" in claims


def test_superseded_budgeted_steps_claim_nothing():
    workflow = _plan(superseded_step_indexes=[1, 2])
    claims = remediation_budget_claims(
        RemediationBudgetPolicy(), workflow, _incident(), workflow.official_steps
    )

    assert "class:cluster-a:GPU_RUNTIME_MUTATION" not in claims
    assert "class:cluster-a:NODE_LIFECYCLE_MUTATION" not in claims
    assert "class:cluster-a:WORKLOAD_MUTATION" in claims


def test_owed_compensation_is_started_work_even_without_a_budgeted_step():
    workflow = workflow_request(
        "workflow-restore",
        "incident-x",
        official_steps=[workflow_step(RESTORE_GPU, node_ids=["node-x"])],
        pending_failure_step_index=0,
        pending_failure_error="reset failed",
    )

    assert budget_work_begun(workflow, workflow.official_steps) is True
    # Nothing budgeted is left and nothing is held: an empty hold, never refused.
    assert (
        remediation_budget_claims(
            RemediationBudgetPolicy(), workflow, _incident(), workflow.official_steps
        )
        == {}
    )


def test_an_attempted_budgeted_step_is_started_work():
    """A WAITING or FAILED execution of its own means a node may be mutating."""

    for status in (WorkflowStepStatus.WAITING, WorkflowStepStatus.FAILED):
        workflow = _plan(
            status=WorkflowStatus.RUNNING,
            step_executions=[workflow_step_execution(0, STOP, status)],
        )
        assert budget_work_begun(workflow, workflow.official_steps) is True, status


def test_a_record_without_claims_heals_to_its_whole_plan_once_started():
    """An older release cleared the claims after STOP: the hold is rebuilt."""

    workflow = _plan(status=WorkflowStatus.RUNNING, completed_step_indexes=[0])
    claims = remediation_budget_claims(
        RemediationBudgetPolicy(), workflow, _incident(), workflow.official_steps
    )

    assert sorted(claims) == [
        "class:cluster-a:GPU_RUNTIME_MUTATION",
        "class:cluster-a:NODE_LIFECYCLE_MUTATION",
        "class:cluster-a:WORKLOAD_MUTATION",
        "cluster:cluster-a",
        "node:cluster-a:node-x",
        "region",
    ]
