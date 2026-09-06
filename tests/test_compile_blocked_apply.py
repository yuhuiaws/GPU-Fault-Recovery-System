"""The compile-blocked reconcile: a BLOCKED-at-compile no-op is closed, and
nothing that was dispatched, planned, or still waiting on a workflow is."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from gpu_fault.compile_blocked import (
    APPLY_MODE,
    CLOSE_MARKER,
    PLAN_MODE,
    apply_compile_blocked_plan,
    build_compile_blocked_plan,
)
from gpu_fault.models import (
    IncidentState,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
)
from tests._builders import (
    build_store,
    fault_incident,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)

WORKFLOW = "workflow-924abfb5-be1d-481d-a0f6-72801da4007c"
INCIDENT = "inc-host-node-a-efa_inventory_mismatch-node"
REFERENCE = "pre-deploy-efa-owner"


def _compile_blocked_pair(
    store, *, incident_state=IncidentState.ESCALATED, **overrides
):
    incident = fault_incident(
        INCIDENT,
        "event-efa",
        event_type="HOST",
        node_ids=["node-a"],
        state=incident_state,
        # The incident moved on to a (failed) validated restore; the BLOCKED
        # record is no longer the one it names.
        workflow_request_id="workflow-validated-restore-failed",
    )
    values = {
        "official_action": "REMEDIATE_EFA_DRIVER",
        "official_steps": [
            workflow_step(WorkflowOperation.FREEZE_EVIDENCE),
            workflow_step(WorkflowOperation.MARK_UNSCHEDULABLE),
            workflow_step(WorkflowOperation.RESTORE_SCHEDULING),
        ],
        "blocked_reasons": ["no executable owner for efaDriverRemediation"],
    }
    values.update(overrides)
    workflow = workflow_request(
        WORKFLOW, INCIDENT, status=WorkflowStatus.BLOCKED, fencing_token=1, **values
    )
    store.save_incident(incident)
    store.save_workflow(workflow)
    return incident, workflow


def test_a_compile_time_blocked_no_op_is_eligible_and_closed() -> None:
    store = build_store()
    _compile_blocked_pair(store)

    plan = build_compile_blocked_plan(store, [WORKFLOW])
    (item,) = plan["items"]
    assert plan["mode"] == PLAN_MODE
    assert item["eligible"] is True, item["reasons"]
    assert item["already_closed"] is False
    assert item["node_ids"] == ["node-a"]
    assert item["blocked_reasons"] == ["no executable owner for efaDriverRemediation"]

    result = apply_compile_blocked_plan(
        store,
        workflow_ids=[WORKFLOW],
        expected_plan_sha256=plan["plan_sha256"],
        reference=REFERENCE,
        now=datetime(2026, 9, 6, 14, 0, tzinfo=timezone.utc),
    )

    assert result["mode"] == APPLY_MODE
    assert result["applied_workflow_ids"] == [WORKFLOW]
    assert result["failed_workflow_ids"] == []
    assert result["records_deleted"] == 0
    closed = store.get_workflow(WORKFLOW)
    assert closed.status is WorkflowStatus.SUPERSEDED
    assert REFERENCE in str(closed.preemption_reason)
    assert CLOSE_MARKER in str(closed.preemption_reason)
    assert "efaDriverRemediation" in str(closed.preemption_reason)
    assert closed.superseded_at == datetime(2026, 9, 6, 14, 0, tzinfo=timezone.utc)
    # The incident is deliberately not written.
    assert store.get_incident(INCIDENT).state is IncidentState.ESCALATED


def test_a_rerun_skips_what_an_earlier_apply_closed() -> None:
    store = build_store()
    _compile_blocked_pair(store)
    plan = build_compile_blocked_plan(store, [WORKFLOW])
    apply_compile_blocked_plan(
        store,
        workflow_ids=[WORKFLOW],
        expected_plan_sha256=plan["plan_sha256"],
        reference=REFERENCE,
    )

    again = build_compile_blocked_plan(store, [WORKFLOW])
    (item,) = again["items"]
    assert item["already_closed"] is True
    assert item["eligible"] is False
    result = apply_compile_blocked_plan(
        store,
        workflow_ids=[WORKFLOW],
        expected_plan_sha256=again["plan_sha256"],
        reference=REFERENCE,
    )
    assert result["already_closed_workflow_ids"] == [WORKFLOW]
    assert result["applied_workflow_ids"] == []


@pytest.mark.parametrize(
    "name,overrides,incident_state,expected",
    [
        (
            "it was dispatched",
            {
                "step_executions": [
                    workflow_step_execution(
                        0,
                        WorkflowOperation.FREEZE_EVIDENCE,
                        WorkflowStepStatus.SUCCEEDED,
                    )
                ]
            },
            IncidentState.ESCALATED,
            "was dispatched",
        ),
        (
            "it completed an operation",
            {"completed_operations": [WorkflowOperation.MARK_UNSCHEDULABLE]},
            IncidentState.ESCALATED,
            "changed state",
        ),
        (
            "it is plan-driven",
            {"source_plan_id": "plan-a"},
            IncidentState.ESCALATED,
            "use --mode restore",
        ),
        (
            "it did not block at compile time",
            {"blocked_reasons": []},
            IncidentState.ESCALATED,
            "did not block at compile time",
        ),
        (
            "its incident is still working",
            {},
            IncidentState.ACTION_PENDING,
            "still waiting on a workflow",
        ),
    ],
)
def test_anything_that_touched_or_may_touch_a_node_is_refused(
    name: str, overrides: dict, incident_state: IncidentState, expected: str
) -> None:
    store = build_store()
    _compile_blocked_pair(store, incident_state=incident_state, **overrides)

    plan = build_compile_blocked_plan(store, [WORKFLOW])
    (item,) = plan["items"]
    assert item["eligible"] is False, name
    assert any(expected in reason for reason in item["reasons"]), (
        name,
        item["reasons"],
    )
    with pytest.raises(ValueError, match="ineligible records"):
        apply_compile_blocked_plan(
            store,
            workflow_ids=[WORKFLOW],
            expected_plan_sha256=plan["plan_sha256"],
            reference=REFERENCE,
        )
    assert store.get_workflow(WORKFLOW).status is WorkflowStatus.BLOCKED


def test_a_non_blocked_workflow_is_named_with_its_real_status() -> None:
    store = build_store()
    _compile_blocked_pair(store)
    store.save_workflow(
        store.get_workflow(WORKFLOW).model_copy(
            update={"status": WorkflowStatus.RUNNING}
        )
    )

    (item,) = build_compile_blocked_plan(store, [WORKFLOW])["items"]
    assert item["eligible"] is False
    assert "workflow is RUNNING, not BLOCKED" in item["reasons"]


def test_apply_refuses_a_plan_that_changed_under_it() -> None:
    store = build_store()
    _compile_blocked_pair(store)
    plan = build_compile_blocked_plan(store, [WORKFLOW])
    # A re-plan bumps the fencing token: the approval no longer describes it.
    store.save_workflow(
        store.get_workflow(WORKFLOW).model_copy(update={"fencing_token": 2})
    )

    with pytest.raises(ValueError, match="changed before apply"):
        apply_compile_blocked_plan(
            store,
            workflow_ids=[WORKFLOW],
            expected_plan_sha256=plan["plan_sha256"],
            reference=REFERENCE,
        )
    assert store.get_workflow(WORKFLOW).status is WorkflowStatus.BLOCKED


def test_plan_requires_explicit_workflow_ids_and_reports_unknown_ones() -> None:
    store = build_store()
    with pytest.raises(ValueError, match="explicit workflow IDs"):
        build_compile_blocked_plan(store, [])
    (item,) = build_compile_blocked_plan(store, ["workflow-missing"])["items"]
    assert item == {
        "request_id": "workflow-missing",
        "eligible": False,
        "already_closed": False,
        "reasons": ["workflow does not exist"],
    }
