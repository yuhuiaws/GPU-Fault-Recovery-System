"""A BLOCKED record that was never plan-driven is reconciled by its successor.

The restore reconcile used to require a source recovery plan for every record
so the plan could carry the reconciliation audit. A job DAG's node-branch
workflow is not plan-driven (``source_plan_id`` is empty), and when its branch
escalation is exhausted it parks BLOCKED / NEEDS_OPERATOR under an incident
that a later validated restore then recovers. The successor proves the node
was restored, the incident is RECOVERED, and the ``OPERATOR_RECONCILED`` event
on the workflow already carries the audit -- yet the record stayed ineligible
("workflow has no source recovery plan") and kept both nodes out of fault
handling (GF-REGIONAL-DESTR-014, unknown-reboot). The gate now applies only
where it means something: to a plan-driven record (whose plan is written).
A record without a plan is judged by its successor or, on the never-changed
path, by having completed no node-mutating operation.
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
from gpu_fault.workflow_reconcile import (
    apply_workflow_reconcile_plan,
    build_workflow_reconcile_plan,
)
from gpu_fault.workflow_resolution import restore_reconciliation_reasons
from tests._builders import (
    fault_incident,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)

NOW = datetime(2026, 9, 19, 9, 0, tzinfo=timezone.utc)
RESTART = WorkflowOperation.RESTART_NODE
RESTORE = WorkflowOperation.RESTORE_SCHEDULING


def _store_factories(tmp_path: Path) -> tuple[Callable[[], Any], ...]:
    return (
        InMemoryStore,
        lambda: SqliteStore(str(tmp_path / "non-plan-reconcile.sqlite")),
    )


def _non_plan_state(store: Any, *, successor: bool = True) -> tuple[str, str, str]:
    incident_id = "inc-xid-79"
    blocked_id = "workflow-branch-exhausted"
    successor_id = "workflow-validated-restore-1"
    blocked = workflow_request(
        blocked_id,
        incident_id,
        status=WorkflowStatus.BLOCKED,
        blocked_kind=BlockedKind.NEEDS_OPERATOR,
        fencing_token=4,
        source_plan_id=None,
        dag_enabled=True,
        official_steps=[
            workflow_step(WorkflowOperation.MARK_UNSCHEDULABLE, node_ids=["node-a"]),
            workflow_step(RESTART, node_ids=["node-a"]),
        ],
        completed_step_indexes=[0],
        completed_operations=[WorkflowOperation.MARK_UNSCHEDULABLE],
        step_executions=[
            workflow_step_execution(
                1,
                RESTART,
                WorkflowStepStatus.FAILED,
                phase="official",
                details={
                    "outcome_unknown": False,
                    "manual_confirmation_required": False,
                    "operator_confirmed": {
                        "actor": "ops",
                        "reference": "CHG-1",
                        "confirmed_at": NOW.isoformat(),
                        "node_id": "node-a",
                        "operation": "RESTART_NODE",
                    },
                },
            )
        ],
        blocked_reasons=["node branch escalation exhausted"],
        updated_at=NOW - timedelta(hours=1),
    )
    restore = workflow_request(
        successor_id,
        incident_id,
        status=WorkflowStatus.SUCCEEDED,
        fencing_token=4,
        official_action=RESTORE.value,
        completed_operations=[
            WorkflowOperation.VALIDATE_GPU,
            WorkflowOperation.VALIDATE_HOST,
            WorkflowOperation.VALIDATE_FABRIC,
            RESTORE,
        ],
        updated_at=NOW - timedelta(minutes=30),
    )
    incident = fault_incident(
        incident_id,
        "event-79",
        node_ids=["node-a", "node-b"],
        state=IncidentState.RECOVERED if successor else IncidentState.ESCALATED,
        workflow_request_id=successor_id if successor else blocked_id,
        fencing_token=4,
        updated_at=NOW - timedelta(minutes=30),
    )
    store.save_workflow(blocked)
    if successor:
        store.save_workflow(restore)
    store.save_incident(incident)
    return incident_id, blocked_id, successor_id


@pytest.mark.parametrize("factory_index", [0, 1])
def test_a_non_plan_record_with_a_verified_restore_successor_is_eligible(
    tmp_path: Path, factory_index: int
) -> None:
    store = _store_factories(tmp_path)[factory_index]()
    _incident_id, blocked_id, successor_id = _non_plan_state(store)

    plan = build_workflow_reconcile_plan(store, [blocked_id], now=NOW)
    item = plan["items"][0]

    assert item["eligible"] is True, item["reasons"]
    assert item["terminalization"] == "verified-restore"
    assert item["successor_workflow_id"] == successor_id
    assert item["source_plan_id"] is None


@pytest.mark.parametrize("factory_index", [0, 1])
def test_applying_the_plan_supersedes_the_record_and_names_no_plan(
    tmp_path: Path, factory_index: int
) -> None:
    store = _store_factories(tmp_path)[factory_index]()
    incident_id, blocked_id, successor_id = _non_plan_state(store)
    plan = build_workflow_reconcile_plan(store, [blocked_id], now=NOW)

    result = apply_workflow_reconcile_plan(
        store,
        workflow_ids=[blocked_id],
        expected_plan_sha256=plan["plan_sha256"],
        reference="CHG-2026-0919-02",
        now=NOW,
        actor="arn:aws:sts::123456789012:assumed-role/Admin/ops",
    )

    assert result["applied_workflow_ids"] == [blocked_id]
    assert result["failures"] == {}
    assert result["resolved_plan_ids"] == [], "there is no plan to resolve"
    assert result["archive_eligible_incident_ids"] == [incident_id]
    closed = store.get_workflow(blocked_id)
    assert closed.status is WorkflowStatus.SUPERSEDED
    assert closed.preempted_by_workflow_id == successor_id
    assert "after verified restore" in (closed.preemption_reason or "")
    event = closed.events[-1]
    assert event.kind is WorkflowEventKind.OPERATOR_RECONCILED
    assert event.code == WorkflowEventCode.OPERATOR_RECONCILED.value
    assert event.details["successor_workflow_id"] == successor_id
    incident = store.get_incident(incident_id)
    assert incident.state is IncidentState.RECOVERED
    assert any("CHG-2026-0919-02" in reason for reason in incident.reasons), (
        incident.reasons
    )
    assert (
        store.list_active_workflow_incidents(
            incident.cluster_id, node_ids={"node-a", "node-b"}
        )
        == []
    ), "the superseded record no longer holds either node"


def test_a_non_plan_record_without_a_successor_is_closable_as_never_changed() -> None:
    """The never-changed path no longer needs a plan either.

    Only a containment completed and the incident is settled: there is nothing
    on the node to restore, and no plan to carry the audit was never a reason
    to keep the record open (2026-09-30: a plan replaced in place under a merge
    left ``source_plan_id`` empty on a record no admin path could then close).
    The admin side still demands node evidence before applying it.
    """

    store = InMemoryStore()
    _incident_id, blocked_id, _successor_id = _non_plan_state(store, successor=False)

    item = build_workflow_reconcile_plan(store, [blocked_id], now=NOW)["items"][0]

    assert item["eligible"] is True, item["reasons"]
    assert item["terminalization"] == "never-changed"
    assert item["source_plan_id"] is None


def test_a_plan_driven_record_still_needs_its_plan_beside_the_successor() -> None:
    store = InMemoryStore()
    _incident_id, blocked_id, _successor_id = _non_plan_state(store)
    store.save_workflow(
        store.get_workflow(blocked_id).model_copy(
            update={"source_plan_id": "plan-missing"}
        )
    )

    item = build_workflow_reconcile_plan(store, [blocked_id], now=NOW)["items"][0]

    assert item["eligible"] is False
    assert "source recovery plan is missing" in item["reasons"], item["reasons"]


def test_restore_reconciliation_reasons_accepts_a_none_plan_only_for_a_non_plan_record() -> (
    None
):
    store = InMemoryStore()
    _incident_id, blocked_id, successor_id = _non_plan_state(store)
    workflow = store.get_workflow(blocked_id)
    incident = store.get_incident(workflow.incident_id)
    successor = store.get_workflow(successor_id)

    assert (
        restore_reconciliation_reasons(
            workflow, incident, successor, None, [], evaluated_at=NOW
        )
        == []
    )
    assert restore_reconciliation_reasons(
        workflow.model_copy(update={"source_plan_id": "plan-1"}),
        incident,
        successor,
        None,
        [],
        evaluated_at=NOW,
    ) == ["source recovery plan is missing"]
