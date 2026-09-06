"""Escalating a branch settles its budget and its quiesce per branch.

F-N1 remainder (docs/review/IMPLEMENTATION-LOG.md §29 未做). A workflow
claims its remediation concurrency budget once, for the plan it was born
with. When a node branch escalated in place from RESET_GPU to RESTART_NODE
the workflow started a node lifecycle mutation the cluster budget had never
counted, and the quiesce that branch had completed was judged "unrestored"
forever because its RESTORE_GPU_SERVICES was superseded with the rest of
the branch -- even though the reboot that replaced it restores the services.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.execution.executor import ProductionWorkflowExecutor
from gpu_fault.execution.models import WorkflowExecutionRequest
from gpu_fault.execution.remediation_budget import (
    RemediationBudgetPolicy,
    escalation_budget_claims,
)
from gpu_fault.models import IncidentState, WorkflowOperation, WorkflowStatus
from gpu_fault.store import SqliteStore
from gpu_fault.store.shared.errors import RemediationBudgetError
from tests._builders import (
    active_workflow_executor,
    build_store,
    copy_model,
    fault_incident,
    workflow_request,
    workflow_step,
)
from tests.execution._support import FakeAdapter, WorkflowStepOutcome, workflow_state
from tests.execution.test_branch_escalation import (
    ALL_OPERATIONS,
    REBOOT,
    RESET,
    RESTART_JOB,
    VALIDATIONS,
    _escalator,
    _job_dag,
    _ok,
)
from tests.store._postgres_processor_claim_support import (
    _truncate,
    postgres_store_instance,
)

QUIESCE = WorkflowOperation.QUIESCE_GPU_SERVICES
RESTORE = WorkflowOperation.RESTORE_GPU_SERVICES
REPLACE = WorkflowOperation.REPLACE_NODE
LIFECYCLE_SCOPE = "class:cluster-a:NODE_LIFECYCLE_MUTATION"


@pytest.fixture(params=["memory", "sqlite", "postgres"])
def store(request, tmp_path):
    if request.param == "memory":
        yield build_store()
        return
    if request.param == "sqlite":
        sqlite = SqliteStore(str(tmp_path / "settle.db"))
        try:
            yield sqlite
        finally:
            sqlite.close()
        return
    if not os.getenv("GPU_FAULT_TEST_POSTGRES_URL"):
        pytest.skip("GPU_FAULT_TEST_POSTGRES_URL is required")
    for postgres in postgres_store_instance():
        yield postgres
    _truncate()


# ---------------------------------------------------------------- quiesce


def test_a_completed_reboot_restores_the_quiesce_of_its_node():
    _, workflow = workflow_state(build_store(), [QUIESCE, RESTORE, REBOOT])
    escalated = copy_model(
        workflow,
        official_steps=[
            workflow_step(QUIESCE, node_ids=["node-b"]),
            workflow_step(RESTORE, node_ids=["node-b"]),  # superseded with the branch
            workflow_step(REBOOT, node_ids=["node-b"]),  # the rung that replaced it
        ],
        completed_step_indexes=[0, 2],
        completed_operations=[QUIESCE, REBOOT],
        superseded_step_indexes=[1],
    )
    still_pending = copy_model(
        escalated, completed_step_indexes=[0], completed_operations=[QUIESCE]
    )
    other_node = copy_model(
        escalated,
        official_steps=[
            workflow_step(QUIESCE, node_ids=["node-b"]),
            workflow_step(RESTORE, node_ids=["node-b"]),
            workflow_step(REBOOT, node_ids=["node-c"]),
        ],
    )

    assert ProductionWorkflowExecutor._has_unrestored_quiesce(escalated) is False
    assert ProductionWorkflowExecutor._has_unrestored_quiesce(still_pending) is True
    assert ProductionWorkflowExecutor._has_unrestored_quiesce(other_node) is True


# ---------------------------------------------------------------- budget claims


def test_escalation_claims_only_the_scopes_the_workflow_does_not_hold():
    policy = RemediationBudgetPolicy()
    incident = fault_incident("inc-a", "event-a", cluster_id="cluster-a")
    held = workflow_request(
        "wf-a",
        "inc-a",
        status=WorkflowStatus.RUNNING,
        official_steps=[workflow_step(RESET, node_ids=["node-b"])],
        remediation_budget_claims=[
            "region",
            "cluster:cluster-a",
            "node:cluster-a:node-b",
            "class:cluster-a:GPU_RUNTIME_MUTATION",
        ],
    )
    rung = [
        workflow_step(REBOOT, node_ids=["node-b"]),
        *[workflow_step(op, node_ids=["node-b"]) for op in VALIDATIONS],
    ]

    claims = escalation_budget_claims(policy, held, incident, rung)

    # Only node-mutating rungs are budgeted; the validations and the
    # scheduler release that follow them never were.
    assert claims == {LIFECYCLE_SCOPE: policy.resource_class_limit}
    assert (
        escalation_budget_claims(
            policy,
            copy_model(
                held,
                remediation_budget_claims=[*held.remediation_budget_claims, *claims],
            ),
            incident,
            rung,
        )
        == {}
    )


def _leased(
    store, request_id: str, scopes: list[str], *, owner: str = "executor-x"
) -> None:
    incident = fault_incident(
        f"inc-{request_id}", f"event-{request_id}", workflow_request_id=request_id
    )
    workflow = workflow_request(
        request_id,
        incident.incident_id,
        status=WorkflowStatus.RUNNING,
        official_steps=[workflow_step(REBOOT, node_ids=[f"node-{request_id}"])],
        execution_owner_id=owner,
        execution_lease_expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        remediation_budget_claims=scopes,
        remediation_budget_limits={scope: 2 for scope in scopes},
    )
    store.save_incident_and_workflow(incident, workflow)


def test_the_store_extends_a_leased_workflow_budget_or_refuses(store):
    _leased(
        store,
        "mine",
        ["region", "class:cluster-a:GPU_RUNTIME_MUTATION"],
        owner="executor-a",
    )
    _leased(store, "other-1", [LIFECYCLE_SCOPE])

    extended = store.extend_remediation_budget(
        "mine", "executor-a", {LIFECYCLE_SCOPE: 2}
    )

    assert LIFECYCLE_SCOPE in extended.remediation_budget_claims
    assert "class:cluster-a:GPU_RUNTIME_MUTATION" in extended.remediation_budget_claims
    assert extended.remediation_budget_limits[LIFECYCLE_SCOPE] == 2
    assert (
        store.get_workflow("mine").remediation_budget_claims
        == extended.remediation_budget_claims
    )

    _leased(store, "other-2", [LIFECYCLE_SCOPE])
    _leased(store, "late", ["region"], owner="executor-a")
    with pytest.raises(RemediationBudgetError):
        store.extend_remediation_budget("late", "executor-a", {LIFECYCLE_SCOPE: 2})
    # A refusal leaves the record and its lease untouched.
    late = store.get_workflow("late")
    assert late.remediation_budget_claims == ["region"]
    assert late.execution_owner_id == "executor-a"


# ---------------------------------------------------------------- executor


def test_a_rung_the_cluster_budget_cannot_take_exhausts_the_branch_instead():
    store = build_store()
    incident, workflow = _job_dag(
        store, node_c_operation=WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE
    )
    # Two other reboots already run in this cluster: the class limit is 2.
    _leased(store, "other-1", [LIFECYCLE_SCOPE])
    _leased(store, "other-2", [LIFECYCLE_SCOPE])
    outcomes = {
        RESET: WorkflowStepOutcome.failed("reset refused"),
        **_ok(
            REBOOT,
            WorkflowOperation.COLLECT_DIAGNOSTIC_BUNDLE,
            RESTART_JOB,
            *VALIDATIONS,
        ),
    }
    adapter = FakeAdapter(outcomes)
    executor = active_workflow_executor(store, [adapter], ALL_OPERATIONS)
    executor.branch_escalator = _escalator()

    result = executor.execute(
        workflow.request_id,
        WorkflowExecutionRequest(expected_fencing_token=workflow.fencing_token),
    )

    saved = store.get_workflow(workflow.request_id)
    assert result.status is WorkflowStatus.FAILED
    assert not any(call.endswith("/RESTART_NODE") for call in adapter.calls), (
        'expected any(call.endswith("/RESTART_NODE") for call in adapter.calls) to be false'
    )
    assert len(saved.exhausted_branch_ids) == 1
    assert saved.branch_escalation_counts == {}
    assert (
        "workflow-active/2/COLLECT_DIAGNOSTIC_BUNDLE" in adapter.calls
    )  # node-c finished
    assert executor.branch_escalation_budget_refusals_total == 1
    assert store.get_incident(incident.incident_id).state is IncidentState.ESCALATED


def test_an_escalated_rung_is_counted_against_the_cluster_budget():
    store = build_store()
    _, workflow = _job_dag(store)
    outcomes = {
        RESET: WorkflowStepOutcome.failed("reset refused"),
        **_ok(REBOOT, RESTART_JOB, *VALIDATIONS),
    }
    adapter = FakeAdapter(outcomes)
    executor = active_workflow_executor(store, [adapter], ALL_OPERATIONS)
    executor.branch_escalator = _escalator()
    seen: list[list[str]] = []
    original = executor._save_leased

    def spy(workflow, epoch):
        seen.append(list(workflow.remediation_budget_claims))
        return original(workflow, epoch)

    executor._save_leased = spy

    result = executor.execute(
        workflow.request_id,
        WorkflowExecutionRequest(expected_fencing_token=workflow.fencing_token),
    )

    assert result.status is WorkflowStatus.SUCCEEDED
    # From the escalation on, the lifecycle class was part of the held budget.
    assert any(LIFECYCLE_SCOPE in claims for claims in seen), (
        "expected any(LIFECYCLE_SCOPE in claims for claims in seen) to be true"
    )
