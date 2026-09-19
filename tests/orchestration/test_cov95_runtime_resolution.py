from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace

import pytest

from gpu_fault.models import (
    IncidentState,
    PlanStatus,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
)
from gpu_fault.remote_command_models import RemoteCommandStatus
from gpu_fault.retired_generation import retired_generation_records
from gpu_fault.store.shared.errors import StaleWriteError
from gpu_fault.workflow_resolution import (
    reconciled_restore_records,
    restore_reconciliation_reasons,
)
from tests._builders import workflow_step_execution
from tests.orchestration._cov95_runtime_resolution import (
    NOW,
    restore_state,
    retired_state,
)


@pytest.mark.parametrize(
    ("target", "changes", "reason"),
    [
        ("workflow", {"status": WorkflowStatus.PENDING}, "not BLOCKED"),
        ("workflow", {"execution_owner_id": "live-owner"}, "execution owner"),
        (
            "workflow",
            {"execution_lease_expires_at": NOW + timedelta(seconds=1)},
            "lease has not expired",
        ),
        # A plan-driven record must link to the plan it names; a record that
        # was never plan-driven is reconciled by its verified successor and
        # is pinned in tests/test_workflow_reconcile_non_plan_successor.py.
        ("workflow", {"source_plan_id": "other-plan"}, "plan linkage is invalid"),
        ("incident", {"state": IncidentState.ACTION_PENDING}, "not RECOVERED"),
        (
            "incident",
            {"workflow_request_id": "other-successor"},
            "successor is no longer valid",
        ),
        ("incident", {"fencing_token": 8}, "successor is no longer valid"),
        (
            "successor",
            {"incident_id": "other-incident"},
            "successor is no longer valid",
        ),
        ("successor", {"fencing_token": 8}, "successor is no longer valid"),
        (
            "successor",
            {"status": WorkflowStatus.FAILED},
            "successor is no longer valid",
        ),
        ("successor", {"completed_operations": []}, "successor is no longer valid"),
        ("plan", {"plan_id": "other-plan"}, "plan linkage is invalid"),
        ("plan", {"incident_id": "other-incident"}, "plan linkage is invalid"),
        ("plan", {"workflow_request_id": "other-workflow"}, "plan linkage is invalid"),
        ("plan", {"status": PlanStatus.PENDING}, "plan is not FAILED"),
    ],
)
def test_restore_reconciliation_rechecks_every_link_and_liveness_guard(
    target, changes, reason
) -> None:
    state = restore_state()
    setattr(state, target, getattr(state, target).model_copy(update=changes))
    before = [
        value.model_copy(deep=True)
        for value in (state.workflow, state.incident, state.successor, state.plan)
    ]
    reasons = restore_reconciliation_reasons(
        state.workflow,
        state.incident,
        state.successor,
        state.plan,
        [],
        evaluated_at=NOW,
    )
    assert any(reason in value for value in reasons), reasons
    with pytest.raises(ValueError, match=reason):
        reconciled_restore_records(
            state.workflow,
            state.incident,
            state.successor,
            state.plan,
            [],
            expected_fencing_token=7,
            expected_workflow_updated_at=state.workflow.updated_at,
            reference="CHG-unit",
            reconciled_at=NOW,
        )
    assert [state.workflow, state.incident, state.successor, state.plan] == before, (
        "a refused reconciliation must leave every input snapshot unchanged"
    )


@pytest.mark.parametrize(
    ("missing", "reason"),
    [
        ("incident", "incident is missing"),
        ("plan", "source recovery plan is missing"),
        ("successor", "no verified restore successor"),
    ],
)
def test_missing_reconciliation_evidence_is_not_treated_as_success(
    missing: str, reason: str
) -> None:
    state = restore_state()
    setattr(state, missing, None)
    reasons = restore_reconciliation_reasons(
        state.workflow,
        state.incident,
        state.successor,
        state.plan,
        [],
        evaluated_at=NOW,
    )
    assert any(reason in value for value in reasons), reasons


@pytest.mark.parametrize("active", ["command", "provider"])
def test_unsettled_effects_prevent_restore_reconciliation(active: str) -> None:
    state = restore_state()
    if active == "command":
        state.commands = [
            SimpleNamespace(
                command_id="pending",
                workflow_request_id=state.workflow.request_id,
                status=RemoteCommandStatus.LEASED,
            )
        ]
        reason = "open remote commands"
    else:
        state.workflow = state.workflow.model_copy(
            update={
                "step_executions": [
                    workflow_step_execution(
                        1,
                        WorkflowOperation.RESTART_NODE,
                        WorkflowStepStatus.WAITING,
                        adapter_operation_id="provider-owned",
                    )
                ]
            }
        )
        reason = "unknown provider action"
    reasons = restore_reconciliation_reasons(
        state.workflow,
        state.incident,
        state.successor,
        state.plan,
        state.commands,
        evaluated_at=NOW,
    )
    assert any(reason in value for value in reasons), reasons


@pytest.mark.parametrize("field", ["fence", "epoch", "timestamp"])
def test_restore_compare_and_set_rejects_stale_approval_identity(field: str) -> None:
    state = restore_state()
    with pytest.raises(StaleWriteError, match="changed"):
        reconciled_restore_records(
            state.workflow,
            state.incident,
            state.successor,
            state.plan,
            [],
            expected_fencing_token=6 if field == "fence" else 7,
            expected_execution_epoch=1 if field == "epoch" else 2,
            expected_workflow_updated_at=(
                NOW if field == "timestamp" else state.workflow.updated_at
            ),
            reference="CHG-unit",
            reconciled_at=NOW,
        )
    assert state.workflow.status is WorkflowStatus.BLOCKED, state.workflow


@pytest.mark.parametrize("successor_present", [False, True])
def test_verified_restore_reconciliation_preserves_failure_and_audits_its_resolution(
    successor_present: bool,
) -> None:
    state = restore_state()
    if not successor_present:
        state.successor = None
        state.workflow = state.workflow.model_copy(
            update={"completed_operations": [WorkflowOperation.QUARANTINE]}
        )
        state.incident = state.incident.model_copy(
            update={"state": IncidentState.ESCALATED}
        )
    workflow, incident, plan = reconciled_restore_records(
        state.workflow,
        state.incident,
        state.successor,
        state.plan,
        [],
        expected_fencing_token=7,
        expected_execution_epoch=2,
        expected_workflow_updated_at=state.workflow.updated_at,
        reference="CHG-unit",
        reconciled_at=NOW,
    )
    assert workflow.status is WorkflowStatus.SUPERSEDED, workflow
    assert workflow.blocked_reasons == ["validation failed"], workflow
    assert (
        plan.status is PlanStatus.FAILED and plan.reconciliation_reference == "CHG-unit"
    ), plan
    assert plan.resolved_by_restore_workflow_id == (
        "restored-workflow" if successor_present else None
    ), plan
    assert "CHG-unit" in incident.reasons[-1], incident
    assert state.workflow.status is WorkflowStatus.BLOCKED, (
        "record transformation must not mutate its input"
    )


@pytest.mark.parametrize(
    ("target", "updates", "reason"),
    [
        ("incident", {"workflow_request_id": "other"}, "no longer names"),
        ("successor", {"incident_id": "foreign"}, "another incident"),
        ("successor", {"fencing_token": 9}, "not at the incident generation"),
        ("workflow", {"fencing_token": 8}, "not behind its incident"),
    ],
)
def test_retired_generation_write_revalidates_the_incident_successor_binding(
    target, updates, reason
) -> None:
    state = retired_state()
    setattr(state, target, getattr(state, target).model_copy(update=updates))
    with pytest.raises(ValueError, match=reason):
        retired_generation_records(
            state.workflow,
            state.incident,
            state.successor,
            [],
            expected_fencing_token=state.workflow.fencing_token,
            reference="CHG-unit",
            reconciled_at=NOW,
        )
    assert state.workflow.status is WorkflowStatus.PENDING, state.workflow
    assert state.workflow.execution_owner_id == "retired-executor", state.workflow


@pytest.mark.parametrize("reference", [None, "CHG-unit"])
def test_retired_generation_releases_only_the_retired_lease_and_preserves_the_successor(
    reference,
) -> None:
    state = retired_state()
    workflow, incident = retired_generation_records(
        state.workflow,
        state.incident,
        state.successor,
        [],
        expected_fencing_token=7,
        reference=reference,
        reconciled_at=NOW,
        actor="unit-operator",
        approval={"plan_sha256": "a" * 64},
    )
    assert workflow.status is WorkflowStatus.SUPERSEDED, workflow
    assert (
        workflow.execution_owner_id is None
        and workflow.execution_lease_expires_at is None
    ), workflow
    assert incident.workflow_request_id == state.successor.request_id, incident
    assert incident.fencing_token == 8 and workflow.fencing_token == 7, (
        workflow,
        incident,
    )
    assert any("revoked retired generation" in reason for reason in incident.reasons), (
        incident
    )
    assert state.workflow.execution_owner_id == "retired-executor", state.workflow
