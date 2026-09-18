from __future__ import annotations

from typing import Any

import pytest

from gpu_fault.models import WorkflowStatus
from gpu_fault.remote_command_models import RemoteCommandStatus
from tests.regional.test_destr008_cancellation_probe import armed_submission, command


def test_20221_unrelated_workflows_never_enter_guard_inventory_or_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bound, _, _, store, watchdog, incident, workflow = armed_submission(
        status=WorkflowStatus.FAILED
    )
    foreign_pairs = []
    for index in range(20_221):
        foreign_incident = incident.model_copy(
            update={
                "incident_id": f"foreign-inc-{index}",
                "event_id": f"foreign-event-{index}",
                "job_id": f"foreign-job-{index}",
                "attempt_id": f"foreign-attempt-{index}",
                "workflow_request_id": f"foreign-workflow-{index}",
            }
        )
        foreign_workflow = workflow.model_copy(
            update={
                "request_id": f"foreign-workflow-{index}",
                "incident_id": foreign_incident.incident_id,
            }
        )
        store.save_incident_and_workflow(foreign_incident, foreign_workflow)
        if index in {0, 20_220}:
            foreign_pairs.append((foreign_incident, foreign_workflow))
    foreign_commands = [
        command(store, *foreign_pairs[0], command_id="foreign-pending"),
        command(
            store,
            *foreign_pairs[1],
            command_id="foreign-leased",
            status=RemoteCommandStatus.LEASED,
        ),
    ]
    owned_command = command(store, incident, workflow)
    assert sum(store.workflow_status_counts().values()) == 20_222
    query = store.list_job_recovery_workflow_incidents
    get = store.get_workflow
    cancel = store.cancel_remote_commands_for_workflow
    queries: list[tuple[tuple[Any, ...], dict[str, Any], int]] = []
    read_ids: list[str] = []
    cancelled_ids: list[str] = []

    def scoped(*args: Any, **kwargs: Any) -> Any:
        result = query(*args, **kwargs)
        queries.append((args, kwargs, len(result)))
        return result

    def get_owned(key: str) -> Any:
        read_ids.append(key)
        return get(key)

    def cancel_owned(key: str, *, reason: str) -> Any:
        cancelled_ids.append(key)
        return cancel(key, reason=reason)

    monkeypatch.setattr(store, "list_job_recovery_workflow_incidents", scoped)
    monkeypatch.setattr(store, "get_workflow", get_owned)
    monkeypatch.setattr(store, "cancel_remote_commands_for_workflow", cancel_owned)
    monkeypatch.setattr(
        store,
        "list_workflows",
        lambda *args, **kwargs: pytest.fail(
            "a watchdog must never scan global history"
        ),
    )
    assert watchdog.tick(bound.deadline_at).state == "REVOKED"
    done = watchdog.tick(bound.deadline_at + 5)
    assert done.state == "QUIESCENT"
    assert done.workflow_ids == [workflow.request_id]
    assert done.command_ids == [owned_command.command_id]
    assert (
        queries
        == [
            (
                (bound.cluster_id, bound.job_id, bound.attempt_id),
                {"limit": 129, "include_terminal": True},
                1,
            )
        ]
        * 4
    ), "probe payload work must depend only on its scoped inventory"
    assert read_ids == [workflow.request_id] * 6
    assert cancelled_ids == [workflow.request_id] * 2
    for original in foreign_commands:
        assert store.get_remote_command(original.command_id) == original
    for original_incident, original_workflow in foreign_pairs:
        assert store.get_incident(original_incident.incident_id) == original_incident
        assert get(original_workflow.request_id) == original_workflow


def test_late_current_successor_is_loaded_by_exact_pointer_not_global_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bound, _, _, store, watchdog, incident, workflow = armed_submission(
        status=WorkflowStatus.FAILED
    )
    child = workflow.model_copy(
        update={
            "request_id": "late-child",
            "status": WorkflowStatus.BLOCKED,
            "predecessor_workflow_id": workflow.request_id,
        }
    )
    store.save_workflow(child)
    current_incident = incident.model_copy(
        update={"workflow_request_id": child.request_id}
    )
    store.save_incident(current_incident, expected=incident)
    query = store.list_job_recovery_workflow_incidents
    first = True

    def before_child_committed(*args: Any, **kwargs: Any) -> Any:
        nonlocal first
        if first:
            first = False
            return [(current_incident, workflow)]
        return query(*args, **kwargs)

    monkeypatch.setattr(
        store, "list_job_recovery_workflow_incidents", before_child_committed
    )
    monkeypatch.setattr(
        store, "list_workflows", lambda **_: pytest.fail("no global fallback")
    )
    result = watchdog.tick(bound.deadline_at)
    assert set(result.workflow_ids) == {workflow.request_id, child.request_id}
    assert store.get_workflow(child.request_id).workload_withdrawn_at is not None
    assert watchdog.tick(bound.deadline_at + 5).state == "QUIESCENT"
