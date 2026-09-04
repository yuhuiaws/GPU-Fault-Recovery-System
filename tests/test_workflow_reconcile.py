from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import pytest

from gpu_fault.models import (
    IncidentState,
    PlanStatus,
    RecoveryPlan,
    WorkflowOperation,
    WorkflowStatus,
    WorkflowStepStatus,
)
from gpu_fault.store import InMemoryStore, SqliteStore
from gpu_fault.workflow_reconcile import (
    apply_workflow_reconcile_plan,
    build_workflow_reconcile_plan,
)
from tests._builders import (
    fault_incident,
    workflow_request,
    workflow_step,
    workflow_step_execution,
)
from tests.regional._regional_support import enqueue_remote_command

NOW = datetime(2026, 9, 3, 7, 0, tzinfo=timezone.utc)


def _store_factories(tmp_path: Path) -> tuple[Callable[[], Any], ...]:
    return (
        InMemoryStore,
        lambda: SqliteStore(str(tmp_path / "workflow-reconcile.sqlite")),
    )


def _restored_state(store: Any) -> tuple[str, str, str, str]:
    incident_id = "incident-restored"
    blocked_id = "workflow-blocked"
    successor_id = "workflow-restored"
    plan_id = "plan-failed"
    plan = RecoveryPlan(
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
    blocked = workflow_request(
        blocked_id,
        incident_id,
        status=WorkflowStatus.BLOCKED,
        fencing_token=7,
        source_plan_id=plan_id,
        official_action=WorkflowOperation.QUARANTINE.value,
        official_steps=[workflow_step(WorkflowOperation.QUARANTINE)],
        updated_at=NOW - timedelta(hours=1),
    )
    successor = workflow_request(
        successor_id,
        incident_id,
        status=WorkflowStatus.SUCCEEDED,
        fencing_token=7,
        predecessor_workflow_id=blocked_id,
        completed_operations=[
            WorkflowOperation.VALIDATE_GPU,
            WorkflowOperation.VALIDATE_HOST,
            WorkflowOperation.VALIDATE_FABRIC,
            WorkflowOperation.RESTORE_SCHEDULING,
        ],
        updated_at=NOW - timedelta(minutes=30),
    )
    incident = fault_incident(
        incident_id,
        "event-restored",
        state=IncidentState.RECOVERED,
        workflow_request_id=successor_id,
        fencing_token=7,
        updated_at=NOW - timedelta(minutes=30),
    )
    store.save_plan(plan)
    store.save_workflow(blocked)
    store.save_workflow(successor)
    store.save_incident(incident)
    return incident_id, blocked_id, successor_id, plan_id


@pytest.mark.parametrize("factory_index", [0, 1])
def test_reconcile_supersedes_only_verified_predecessor(
    tmp_path: Path, factory_index: int
) -> None:
    store = _store_factories(tmp_path)[factory_index]()
    incident_id, blocked_id, successor_id, plan_id = _restored_state(store)
    try:
        plan = build_workflow_reconcile_plan(store, [blocked_id], now=NOW)

        result = apply_workflow_reconcile_plan(
            store,
            workflow_ids=[blocked_id],
            expected_plan_sha256=plan["plan_sha256"],
            reference="CHG-12345",
            now=NOW + timedelta(minutes=1),
        )

        predecessor = store.get_workflow(blocked_id)
        incident = store.get_incident(incident_id)
        source_plan = store.get_plan(plan_id)
        assert predecessor.status is WorkflowStatus.SUPERSEDED
        assert predecessor.preempted_by_workflow_id == successor_id
        assert incident.workflow_request_id == successor_id
        assert source_plan.status is PlanStatus.FAILED
        assert source_plan.resolved_by_restore_workflow_id == successor_id
        assert source_plan.reconciliation_reference == "CHG-12345"
        assert result["records_deleted"] == 0
        assert result["archive_eligible_incident_ids"] == [incident_id]
        assert result["resolved_plan_ids"] == [plan_id]
    finally:
        close = getattr(store, "close", None)
        if close is not None:
            close()


@pytest.mark.parametrize(
    ("update", "reason"),
    [
        (
            {
                "execution_owner_id": "executor-a",
                "execution_lease_expires_at": NOW + timedelta(minutes=5),
            },
            "execution owner",
        ),
        (
            {
                "step_executions": [
                    workflow_step_execution(
                        0,
                        WorkflowOperation.RESTART_NODE,
                        WorkflowStepStatus.WAITING,
                        adapter_operation_id="provider/reboot-a",
                    )
                ]
            },
            "unknown provider action",
        ),
    ],
)
def test_reconcile_plan_rejects_active_or_unknown_actions(
    update: dict[str, Any], reason: str
) -> None:
    store = InMemoryStore()
    _incident_id, blocked_id, _successor_id, _plan_id = _restored_state(store)
    workflow = store.get_workflow(blocked_id)
    store.save_workflow(workflow.model_copy(update=update))

    plan = build_workflow_reconcile_plan(store, [blocked_id], now=NOW)

    assert plan["items"][0]["eligible"] is False
    assert any(reason in item for item in plan["items"][0]["reasons"]), (
        "reconcile plan omitted the blocking safety reason"
    )


def test_reconcile_apply_fails_when_fencing_identity_drifts() -> None:
    store = InMemoryStore()
    _incident_id, blocked_id, _successor_id, _plan_id = _restored_state(store)
    plan = build_workflow_reconcile_plan(store, [blocked_id], now=NOW)
    workflow = store.get_workflow(blocked_id)
    store.save_workflow(
        workflow.model_copy(
            update={"fencing_token": workflow.fencing_token + 1, "updated_at": NOW}
        )
    )

    with pytest.raises(ValueError, match="plan changed"):
        apply_workflow_reconcile_plan(
            store,
            workflow_ids=[blocked_id],
            expected_plan_sha256=plan["plan_sha256"],
            reference="CHG-12345",
            now=NOW + timedelta(minutes=1),
        )


def test_the_plan_reads_only_the_commands_of_the_workflows_it_reconciles() -> None:
    """The command read used to be the whole table.

    Every use of that list, here and in ``restore_reconciliation_reasons``,
    filters on ``workflow_request_id``, so a plan covering one workflow decoded
    every command ever issued in the region to answer a question about one.
    """

    store = InMemoryStore()
    _incident_id, blocked_id, _successor_id, _plan_id = _restored_state(store)
    seen: list[set[str] | None] = []
    underlying = store.list_remote_commands

    def recording(*, workflow_request_ids=None):
        seen.append(None if workflow_request_ids is None else set(workflow_request_ids))
        return underlying(workflow_request_ids=workflow_request_ids)

    store.list_remote_commands = recording  # type: ignore[method-assign]

    build_workflow_reconcile_plan(store, [blocked_id], now=NOW)

    assert seen == [{blocked_id}]


def test_a_command_from_another_workflow_is_not_returned() -> None:
    """The narrowing has to be a filter, not just a hint."""

    store = InMemoryStore()
    _incident_id, blocked_id, _successor_id, _plan_id = _restored_state(store)
    # The helper derives ``workflow_request_id`` as ``workflow-<command id>``,
    # which is why the ids are named for the workflows they belong to.
    for index, suffix in enumerate(("blocked", "elsewhere")):
        enqueue_remote_command(
            store, suffix, created_at=NOW - timedelta(minutes=10 - index)
        )
    assert blocked_id == "workflow-blocked", (
        "the command fixture is keyed off the blocked workflow's id"
    )

    assert [
        item.command_id
        for item in store.list_remote_commands(workflow_request_ids=[blocked_id])
    ] == ["blocked"]
    assert len(store.list_remote_commands()) == 2
