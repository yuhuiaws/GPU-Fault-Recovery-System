"""The operator exit for a BLOCKED record must be winnable (F-K1 / P0-72A).

``workflow-reconcile`` is the only path that closes a BLOCKED workflow after its
successor restored the node. Its approval digest hashed ``workflow_updated_at``,
a field every merge into the record restamps, so the plan an operator had just
reviewed was refused with "plan changed before apply" for exactly the records
the tool exists to close. The sibling ``retired_generation`` module had already
learned this lesson; these cases pin it here too, along with the rest of the
operator-exit fixes: the epoch joins the approval, refusals name the field that
moved, discovery is filtered and bounded instead of refused, and the audit line
stops being appended to ``blocked_reasons``.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.models import (
    BlockedKind,
    IncidentState,
    PlanStatus,
    RecoveryPlan,
    WorkflowOperation,
    WorkflowStatus,
)
from gpu_fault.store import InMemoryStore
from gpu_fault.workflow_reconcile import (
    apply_workflow_reconcile_plan,
    build_workflow_reconcile_plan,
)
from gpu_fault.workflow_resolution import reconciled_restore_records
from tests._builders import fault_incident, workflow_request, workflow_step

NOW = datetime(2026, 9, 6, 7, 0, tzinfo=timezone.utc)


def _restored_state(
    store: InMemoryStore, *, suffix: str = "a", blocked_kind: BlockedKind | None = None
) -> tuple[str, str, str]:
    """One BLOCKED predecessor whose successor already restored the node."""

    incident_id = f"incident-{suffix}"
    blocked_id = f"workflow-blocked-{suffix}"
    successor_id = f"workflow-restored-{suffix}"
    plan_id = f"plan-{suffix}"
    store.save_plan(
        RecoveryPlan(
            plan_id=plan_id,
            incident_id=incident_id,
            attempt_id="attempt-a",
            trigger="test",
            runtime_profile_version="profile-v1",
            steps=[],
            workflow_request_id=blocked_id,
            status=PlanStatus.FAILED,
            created_at=NOW - timedelta(hours=2),
        )
    )
    store.save_workflow(
        workflow_request(
            blocked_id,
            incident_id,
            status=WorkflowStatus.BLOCKED,
            fencing_token=7,
            source_plan_id=plan_id,
            blocked_kind=blocked_kind,
            blocked_reasons=["policy requires an operator"],
            official_action=WorkflowOperation.QUARANTINE.value,
            official_steps=[workflow_step(WorkflowOperation.QUARANTINE)],
            updated_at=NOW - timedelta(hours=1),
        )
    )
    store.save_workflow(
        workflow_request(
            successor_id,
            incident_id,
            status=WorkflowStatus.SUCCEEDED,
            fencing_token=7,
            predecessor_workflow_id=blocked_id,
            completed_operations=[WorkflowOperation.RESTORE_SCHEDULING],
            updated_at=NOW - timedelta(minutes=30),
        )
    )
    store.save_incident(
        fault_incident(
            incident_id,
            f"event-{suffix}",
            state=IncidentState.RECOVERED,
            workflow_request_id=successor_id,
            fencing_token=7,
            updated_at=NOW - timedelta(minutes=30),
        )
    )
    return incident_id, blocked_id, successor_id


def test_a_restamped_updated_at_does_not_invalidate_the_reviewed_plan() -> None:
    """A merge restamps ``updated_at`` and changes nothing the verdict reads."""

    store = InMemoryStore()
    _incident_id, blocked_id, successor_id = _restored_state(store)
    plan = build_workflow_reconcile_plan(store, [blocked_id], now=NOW)
    workflow = store.get_workflow(blocked_id)
    store.save_workflow(workflow.model_copy(update={"updated_at": NOW}))

    again = build_workflow_reconcile_plan(store, [blocked_id], now=NOW)
    assert (
        again["items"][0]["workflow_updated_at"]
        != plan["items"][0]["workflow_updated_at"]
    ), "the fixture did not actually restamp the record"
    assert again["plan_sha256"] == plan["plan_sha256"], (
        "a timestamp the verdict never reads invalidated the reviewed plan"
    )

    result = apply_workflow_reconcile_plan(
        store,
        workflow_ids=[blocked_id],
        expected_plan_sha256=plan["plan_sha256"],
        reference="CHG-1",
        now=NOW + timedelta(minutes=1),
    )

    assert result["applied_workflow_ids"] == [blocked_id]
    assert store.get_workflow(blocked_id).preempted_by_workflow_id == successor_id


def test_the_execution_epoch_is_part_of_what_the_approval_binds() -> None:
    """A claim bumps the epoch; a merge does not. The digest must see the former."""

    store = InMemoryStore()
    _incident_id, blocked_id, _successor_id = _restored_state(store)
    plan = build_workflow_reconcile_plan(store, [blocked_id], now=NOW)
    assert plan["items"][0]["execution_epoch"] == 0

    workflow = store.get_workflow(blocked_id)
    # Deliberate out-of-band claim; ``expected`` names the row as read (store
    # review 2026-09-07, item B).
    store.save_workflow(
        workflow.model_copy(update={"execution_epoch": 1}), expected=workflow
    )

    assert (
        build_workflow_reconcile_plan(store, [blocked_id], now=NOW)["plan_sha256"]
        != plan["plan_sha256"]
    ), "an epoch change went unnoticed by the approval digest"


def test_the_transactional_compare_names_the_field_and_both_values() -> None:
    store = InMemoryStore()
    incident_id, blocked_id, successor_id = _restored_state(store)
    workflow = store.get_workflow(blocked_id)

    with pytest.raises(ValueError, match=r"fencing token.*expected 8.*found 7"):
        reconciled_restore_records(
            workflow,
            store.get_incident(incident_id),
            store.get_workflow(successor_id),
            store.get_plan("plan-a"),
            [],
            expected_fencing_token=8,
            expected_workflow_updated_at=workflow.updated_at,
            reference="CHG-1",
            reconciled_at=NOW,
        )
    with pytest.raises(ValueError, match=r"execution epoch.*expected 3.*found 0"):
        reconciled_restore_records(
            workflow,
            store.get_incident(incident_id),
            store.get_workflow(successor_id),
            store.get_plan("plan-a"),
            [],
            expected_fencing_token=7,
            expected_workflow_updated_at=workflow.updated_at,
            expected_execution_epoch=3,
            reference="CHG-1",
            reconciled_at=NOW,
        )


def test_the_audit_line_is_not_appended_to_blocked_reasons() -> None:
    """``blocked_reasons`` is a record of why it blocked, not a second audit log.

    The retired-generation sibling refuses to touch it and says why; the restore
    reconcile appended the same audit it had already written to
    ``preemption_reason`` (P1-61D).
    """

    store = InMemoryStore()
    _incident_id, blocked_id, _successor_id = _restored_state(store)
    before = store.get_workflow(blocked_id).blocked_reasons
    plan = build_workflow_reconcile_plan(store, [blocked_id], now=NOW)

    apply_workflow_reconcile_plan(
        store,
        workflow_ids=[blocked_id],
        expected_plan_sha256=plan["plan_sha256"],
        reference="CHG-1",
        now=NOW + timedelta(minutes=1),
    )

    closed = store.get_workflow(blocked_id)
    assert closed.status is WorkflowStatus.SUPERSEDED
    assert closed.blocked_reasons == before
    assert "CHG-1" in (closed.preemption_reason or "")


def test_discovery_can_be_batched_by_incident_and_by_blocked_kind() -> None:
    store = InMemoryStore()
    incident_a, blocked_a, _ = _restored_state(
        store, suffix="a", blocked_kind=BlockedKind.NEEDS_OPERATOR
    )
    _incident_b, blocked_b, _ = _restored_state(
        store, suffix="b", blocked_kind=BlockedKind.INTERNAL_ERROR
    )
    _incident_c, blocked_c, _ = _restored_state(store, suffix="c")

    everything = build_workflow_reconcile_plan(store, now=NOW)
    by_incident = build_workflow_reconcile_plan(
        store, incident_ids=[incident_a], now=NOW
    )
    by_kind = build_workflow_reconcile_plan(
        store, blocked_kinds=[BlockedKind.INTERNAL_ERROR], now=NOW
    )

    assert [item["request_id"] for item in everything["items"]] == sorted(
        [blocked_a, blocked_b, blocked_c]
    )
    assert [item["request_id"] for item in by_incident["items"]] == [blocked_a]
    assert [item["request_id"] for item in by_kind["items"]] == [blocked_b]
    assert by_kind["items"][0]["blocked_kind"] == "INTERNAL_ERROR"
    assert everything["items"][2]["blocked_kind"] is None


def test_discovery_bounds_the_batch_instead_of_refusing_a_large_backlog() -> None:
    """More than a thousand BLOCKED rows is the storm this tool is for.

    Refusing to plan at that size, while the only way to learn the ids is the
    very discovery that refused, left the operator with no exit at all (P1-58G).
    """

    store = InMemoryStore()
    for index in range(1001):
        store.save_workflow(
            workflow_request(
                f"workflow-blocked-{index:04d}",
                f"incident-{index:04d}",
                status=WorkflowStatus.BLOCKED,
                fencing_token=1,
                updated_at=NOW - timedelta(minutes=1001 - index),
            )
        )

    plan = build_workflow_reconcile_plan(store, max_items=5, now=NOW)

    assert len(plan["items"]) == 5
    assert plan["discovery"] == {
        "scanned": 1001,
        "selected": 5,
        "remaining": 996,
        "scan_truncated": False,
    }
    assert [item["request_id"] for item in plan["items"]] == [
        f"workflow-blocked-{index:04d}" for index in range(5)
    ], "the oldest records come first, so repeated batches walk the backlog"


def test_a_record_that_never_changed_a_node_is_eligible_without_a_successor() -> None:
    """F-B4 (4): the second eligible path.

    A plan-driven workflow BLOCKED before it touched the node has no restore
    successor and never will -- nothing needed restoring -- so the single path
    that required one left it BLOCKED forever. It still needs its incident
    settled (escalated to the operator, or recovered elsewhere) and its source
    plan; the ops-manual rule that a record with no source plan stays ineligible
    is untouched.
    """

    store = InMemoryStore()
    incident_id, blocked_id, successor_id = _restored_state(
        store, blocked_kind=BlockedKind.INTERNAL_ERROR
    )
    # No successor: the incident escalated and still points at the record.
    store.save_incident(
        store.get_incident(incident_id).model_copy(
            update={"state": IncidentState.ESCALATED, "workflow_request_id": blocked_id}
        )
    )
    store.save_workflow(
        store.get_workflow(blocked_id).model_copy(
            update={"completed_operations": [WorkflowOperation.MARK_UNSCHEDULABLE]}
        )
    )

    plan = build_workflow_reconcile_plan(store, [blocked_id], now=NOW)
    item = plan["items"][0]
    assert item["reasons"] == []
    assert item["eligible"] is True
    assert item["successor_workflow_id"] is None
    assert item["terminalization"] == "never-changed"
    assert item["never_changed_a_node"] is True
    assert item["completed_containment_operations"] == ["MARK_UNSCHEDULABLE"]

    result = apply_workflow_reconcile_plan(
        store,
        workflow_ids=[blocked_id],
        expected_plan_sha256=plan["plan_sha256"],
        reference="CHG-2",
        now=NOW + timedelta(minutes=1),
    )

    closed = store.get_workflow(blocked_id)
    assert result["applied_workflow_ids"] == [blocked_id]
    assert closed.status is WorkflowStatus.SUPERSEDED
    assert closed.preempted_by_workflow_id is None
    assert "CHG-2" in (closed.preemption_reason or "")
    assert store.get_plan("plan-a").reconciliation_reference == "CHG-2"


def test_a_record_that_changed_a_node_still_needs_a_verified_restore() -> None:
    store = InMemoryStore()
    incident_id, blocked_id, _successor_id = _restored_state(store)
    store.save_incident(
        store.get_incident(incident_id).model_copy(
            update={"state": IncidentState.ESCALATED, "workflow_request_id": blocked_id}
        )
    )
    store.save_workflow(
        store.get_workflow(blocked_id).model_copy(
            update={"completed_operations": [WorkflowOperation.RESTART_NODE]}
        )
    )

    item = build_workflow_reconcile_plan(store, [blocked_id], now=NOW)["items"][0]

    assert item["eligible"] is False
    assert item["never_changed_a_node"] is False
    assert "workflow has no verified restore successor" in item["reasons"]


def test_never_changed_still_waits_for_an_incident_that_is_driving_a_workflow() -> None:
    store = InMemoryStore()
    incident_id, blocked_id, _successor_id = _restored_state(store)
    store.save_incident(
        store.get_incident(incident_id).model_copy(
            update={
                "state": IncidentState.ACTION_PENDING,
                "workflow_request_id": blocked_id,
            }
        )
    )

    item = build_workflow_reconcile_plan(store, [blocked_id], now=NOW)["items"][0]

    assert item["eligible"] is False
    assert any("ACTION_PENDING" in reason for reason in item["reasons"]), item[
        "reasons"
    ]


def test_a_record_without_a_source_plan_stays_ineligible_by_design() -> None:
    """The ops manual (§10) rule: not a defect, and this pin keeps it that way."""

    store = InMemoryStore()
    incident_id, blocked_id, _successor_id = _restored_state(store)
    store.save_incident(
        store.get_incident(incident_id).model_copy(
            update={"state": IncidentState.ESCALATED, "workflow_request_id": blocked_id}
        )
    )
    store.save_workflow(
        store.get_workflow(blocked_id).model_copy(update={"source_plan_id": None})
    )

    item = build_workflow_reconcile_plan(store, [blocked_id], now=NOW)["items"][0]

    assert item["eligible"] is False
    assert item["reasons"] == ["workflow has no source recovery plan"]
