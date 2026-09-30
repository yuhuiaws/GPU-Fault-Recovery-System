"""A never-changed BLOCKED record without a source plan is closable.

On 2026-09-30 an ``efa_kubernetes_allocatable_mismatch`` workflow had its plan
replaced in place by a merge (``PLAN_REWRITE``, ``source_plan_id`` left empty),
attempted its device-plugin restart, was reaped at the execution deadline with
the outcome unknown and parked BLOCKED / NEEDS_OPERATOR; HyperPod then replaced
the node. The record completed no node-mutating operation, so it was a
``never-changed`` shape -- yet the planner refused it for "workflow has no
source recovery plan" and ``_close_never_changed`` raised on the same, leaving
no admin path to close it and ``uninstall`` failing closed on it.

The plan gate now applies to plan-driven records only. The never-changed close
without a plan writes the workflow exactly as it does with one (``amend_workflow``
with the ``OPERATOR_RECONCILED`` event, ``preemption_reason`` as the audit) and
skips the plan write; the incident is still never written here. The admin side
adds the node evidence (restored, or gone from the provider) before applying.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import pytest

from gpu_fault.models import (
    BlockedKind,
    IncidentState,
    WorkflowEventCode,
    WorkflowEventKind,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
)
from gpu_fault.store import InMemoryStore, SqliteStore
from gpu_fault.store.shared.errors import StaleWriteError
from gpu_fault.workflow_reconcile import (
    apply_workflow_reconcile_plan,
    build_workflow_reconcile_plan,
)
from gpu_fault.workflow_resolution import reconciled_restore_records
from tests._builders import (
    fault_incident,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)

NOW = datetime(2026, 9, 30, 9, 0, tzinfo=timezone.utc)
PLUGIN = WorkflowOperation.RESTART_GPU_DEVICE_PLUGIN
NODE = "hyperpod-i-00000000000000001"
INCIDENT_ID = "incident-efa-mismatch"
BLOCKED_ID = "workflow-b2cf18a64127092442f07935"
REFERENCE = "CHG-2026-0930-01"
ACTOR = "arn:aws:sts::123456789012:assumed-role/Admin/ops"


def _store_factories(tmp_path: Path) -> tuple[Callable[[], Any], ...]:
    return (
        InMemoryStore,
        lambda: SqliteStore(str(tmp_path / "never-changed-non-plan.sqlite")),
    )


def _rewritten_state(
    store: Any, *, completed_operations: list[WorkflowOperation] | None = None
) -> None:
    """The reaped record: plan rewritten in place, step outcome unknown."""

    blocked = workflow_request(
        BLOCKED_ID,
        INCIDENT_ID,
        status=WorkflowStatus.BLOCKED,
        blocked_kind=BlockedKind.NEEDS_OPERATOR,
        fencing_token=2,
        execution_epoch=1,
        source_plan_id=None,
        official_action=PLUGIN.value,
        official_steps=[workflow_step(PLUGIN, node_ids=[NODE])],
        completed_operations=list(completed_operations or []),
        step_executions=[
            workflow_step_execution(
                0, PLUGIN, WorkflowStepStatus.FAILED, details={"outcome_unknown": True}
            )
        ],
        blocked_reasons=["NODE_ACTION_OUTCOME_UNRESOLVED: execution deadline"],
        updated_at=NOW - timedelta(hours=1),
    )
    incident = fault_incident(
        INCIDENT_ID,
        "event-efa-mismatch",
        node_ids=[NODE],
        state=IncidentState.ESCALATED,
        workflow_request_id=BLOCKED_ID,
        fencing_token=2,
        updated_at=NOW - timedelta(hours=1),
    )
    store.save_workflow(blocked)
    store.save_incident(incident)


@pytest.mark.parametrize("factory_index", [0, 1])
def test_the_plan_marks_the_record_eligible_as_never_changed(
    tmp_path: Path, factory_index: int
) -> None:
    store = _store_factories(tmp_path)[factory_index]()
    try:
        _rewritten_state(store)

        item = build_workflow_reconcile_plan(store, [BLOCKED_ID], now=NOW)["items"][0]

        assert item["eligible"] is True, item["reasons"]
        assert item["terminalization"] == "never-changed"
        assert item["source_plan_id"] is None
        assert item["never_changed_a_node"] is True
        assert item["successor_workflow_id"] is None
    finally:
        close = getattr(store, "close", None)
        if close is not None:
            close()


@pytest.mark.parametrize("factory_index", [0, 1])
def test_applying_the_plan_supersedes_the_record_without_touching_the_incident(
    tmp_path: Path, factory_index: int
) -> None:
    store = _store_factories(tmp_path)[factory_index]()
    try:
        _rewritten_state(store)
        before = store.get_workflow(BLOCKED_ID)
        incident_before = store.get_incident(INCIDENT_ID)
        plan = build_workflow_reconcile_plan(store, [BLOCKED_ID], now=NOW)

        result = apply_workflow_reconcile_plan(
            store,
            workflow_ids=[BLOCKED_ID],
            expected_plan_sha256=plan["plan_sha256"],
            reference=REFERENCE,
            now=NOW,
            actor=ACTOR,
            admin_plan_sha256="d" * 64,
        )

        assert result["applied_workflow_ids"] == [BLOCKED_ID]
        assert result["failures"] == {}
        assert result["resolved_plan_ids"] == [], "there is no plan to resolve"
        assert result["archive_eligible_incident_ids"] == [INCIDENT_ID]
        assert result["records_deleted"] == 0
        closed = store.get_workflow(BLOCKED_ID)
        assert closed.status is WorkflowStatus.SUPERSEDED
        assert closed.preempted_by_workflow_id is None
        assert closed.superseded_at == NOW
        assert closed.merge_revision == before.merge_revision + 1, (
            "the out-of-lease amend bumps the merge revision"
        )
        assert closed.preemption_reason == (
            f"operator reconciliation {REFERENCE}: closed {BLOCKED_ID}, which "
            "completed no node-mutating operation, with its incident ESCALATED"
        )
        event = closed.events[-1]
        assert event.kind is WorkflowEventKind.OPERATOR_RECONCILED
        assert event.code == WorkflowEventCode.OPERATOR_RECONCILED.value
        assert event.actor == ACTOR
        assert event.details["terminalization"] == "never-changed"
        assert event.details["expected_fencing_token"] == 2
        assert event.details["expected_execution_epoch"] == 1
        assert event.details["plan_sha256"] == plan["plan_sha256"]
        assert event.details["admin_plan_sha256"] == "d" * 64
        assert event.details["previous_status"] == "BLOCKED"
        assert store.get_incident(INCIDENT_ID) == incident_before, (
            "the never-changed close never writes the incident"
        )
        assert (
            store.list_active_workflow_incidents(
                incident_before.cluster_id, node_ids={NODE}
            )
            == []
        ), "the superseded record no longer holds the node"
    finally:
        close = getattr(store, "close", None)
        if close is not None:
            close()


def test_a_fencing_or_epoch_move_refuses_the_close_as_stale() -> None:
    """The compare-and-set the plan-less path inherits from the resolver."""

    store = InMemoryStore()
    _rewritten_state(store)
    workflow = store.get_workflow(BLOCKED_ID)
    incident = store.get_incident(INCIDENT_ID)

    with pytest.raises(StaleWriteError, match="fencing token changed: expected 3"):
        reconciled_restore_records(
            workflow,
            incident,
            None,
            None,
            [],
            expected_fencing_token=3,
            expected_execution_epoch=1,
            expected_workflow_updated_at=None,
            reference=REFERENCE,
            reconciled_at=NOW,
        )
    with pytest.raises(StaleWriteError, match="execution epoch changed: expected 0"):
        reconciled_restore_records(
            workflow,
            incident,
            None,
            None,
            [],
            expected_fencing_token=2,
            expected_execution_epoch=0,
            expected_workflow_updated_at=None,
            reference=REFERENCE,
            reconciled_at=NOW,
        )
    assert store.get_workflow(BLOCKED_ID).status is WorkflowStatus.BLOCKED


def test_a_record_re_planned_after_the_approval_is_not_applied() -> None:
    store = InMemoryStore()
    _rewritten_state(store)
    plan = build_workflow_reconcile_plan(store, [BLOCKED_ID], now=NOW)
    workflow = store.get_workflow(BLOCKED_ID)
    store.save_workflow(
        workflow.model_copy(update={"fencing_token": 3, "updated_at": NOW}),
        expected=workflow,
    )

    with pytest.raises(ValueError, match="plan changed"):
        apply_workflow_reconcile_plan(
            store,
            workflow_ids=[BLOCKED_ID],
            expected_plan_sha256=plan["plan_sha256"],
            reference=REFERENCE,
            now=NOW,
        )

    assert store.get_workflow(BLOCKED_ID).status is WorkflowStatus.BLOCKED


def test_a_record_that_changed_the_node_still_needs_a_successor_and_a_plan() -> None:
    """Dropping the plan gate is scoped to never-changed: a record that did
    complete a node-mutating operation without a plan is still neither shape."""

    store = InMemoryStore()
    _rewritten_state(store, completed_operations=[PLUGIN])

    item = build_workflow_reconcile_plan(store, [BLOCKED_ID], now=NOW)["items"][0]

    assert item["eligible"] is False
    assert item["never_changed_a_node"] is False
    assert "workflow has no verified restore successor" in item["reasons"]
    assert "workflow has no source recovery plan" in item["reasons"]


def test_an_incident_still_driving_a_workflow_keeps_the_record_open() -> None:
    store = InMemoryStore()
    _rewritten_state(store)
    incident = store.get_incident(INCIDENT_ID)
    store.save_incident(
        incident.model_copy(update={"state": IncidentState.ACTION_PENDING}),
        expected=incident,
    )

    item = build_workflow_reconcile_plan(store, [BLOCKED_ID], now=NOW)["items"][0]

    assert item["eligible"] is False
    assert item["reasons"] == [
        "incident is ACTION_PENDING, still waiting on a workflow"
    ]
