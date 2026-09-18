from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta, timezone

import pytest

from gpu_fault.models import WorkflowOperation, WorkflowRequest
from gpu_fault.regional import RemoteActionCommand
from gpu_fault.store.postgres.state_table_payload import (
    REMOTE_COMMAND_LAYOUT,
    WORKFLOW_LAYOUT,
    join_state_record,
    split_state_record,
    state_update_columns,
)
from tests._builders import fault_incident, workflow_request, workflow_step


def command(at: datetime) -> RemoteActionCommand:
    incident = fault_incident("incident-state-table", "event-state-table")
    workflow = workflow_request("workflow-state-table", incident.incident_id)
    step = workflow_step(
        WorkflowOperation.RESTART_NODE, parameters={"snapshot": "x" * 100_000}
    )
    return RemoteActionCommand(
        command_id="command-state-table",
        cluster_id="cluster-state-table",
        workflow_request_id=workflow.request_id,
        incident_id=incident.incident_id,
        fencing_token=workflow.fencing_token,
        step_index=0,
        step=step,
        idempotency_key="command-state-table/0",
        workflow=workflow,
        incident=incident,
        created_at=at,
        updated_at=at,
        lease_expires_at=at + timedelta(minutes=2),
    )


@pytest.mark.parametrize("offset", [None, UTC, timezone(timedelta(hours=8))])
@pytest.mark.parametrize("microsecond", [0, 123456])
def test_remote_command_column_round_trip_preserves_exact_model_json(
    offset: timezone | None, microsecond: int
) -> None:
    value = command(datetime(2026, 9, 11, 12, 0, 0, microsecond, tzinfo=offset))
    columns = split_state_record(REMOTE_COMMAND_LAYOUT, value)
    restored = RemoteActionCommand.model_validate(
        join_state_record(REMOTE_COMMAND_LAYOUT, columns)
    )
    assert restored.model_dump(mode="json") == value.model_dump(mode="json")
    assert "batched_steps" not in columns["snapshot"], (
        "empty batches changed the wire contract"
    )
    assert not set(REMOTE_COMMAND_LAYOUT.fields).intersection(columns["snapshot"]), (
        "mutable columns were duplicated inside the large snapshot"
    )


@pytest.mark.parametrize("offset", [None, UTC])
def test_workflow_column_round_trip_preserves_nulls_and_datetime_style(
    offset: timezone | None,
) -> None:
    at = datetime(2026, 9, 11, 12, 0, tzinfo=offset)
    value = workflow_request("workflow-columns", "incident-columns").model_copy(
        update={
            "created_at": at,
            "updated_at": at,
            "not_before": at,
            "execution_lease_expires_at": at + timedelta(minutes=3),
        }
    )
    restored = WorkflowRequest.model_validate(
        join_state_record(WORKFLOW_LAYOUT, split_state_record(WORKFLOW_LAYOUT, value))
    )
    assert restored.model_dump(mode="json") == value.model_dump(mode="json")


def test_lease_projection_omits_the_large_command_snapshots() -> None:
    value = command(datetime(2026, 9, 11, tzinfo=UTC))
    columns = state_update_columns(
        REMOTE_COMMAND_LAYOUT, value, frozenset({"lease_expires_at", "updated_at"})
    )
    assert set(columns) == {
        "lease_expires_at",
        "updated_at",
        "lease_expires_at_naive",
        "updated_at_naive",
    }
    assert len(json.dumps(columns, default=str)) < 500
    assert len(value.model_dump_json()) > 100_000


def test_partial_projection_does_not_traverse_the_command_snapshot() -> None:
    value = command(datetime.now(UTC))
    value.step.parameters["opaque"] = object()
    columns = state_update_columns(
        REMOTE_COMMAND_LAYOUT, value, frozenset({"lease_expires_at"})
    )
    assert set(columns) == {"lease_expires_at", "lease_expires_at_naive"}


@pytest.mark.parametrize(
    "fields", [frozenset(), frozenset({"command_id"}), frozenset({"workflow"})]
)
def test_partial_update_cannot_replace_identity_or_snapshot(
    fields: frozenset[str],
) -> None:
    with pytest.raises(ValueError, match="invalid columns"):
        state_update_columns(REMOTE_COMMAND_LAYOUT, command(datetime.now(UTC)), fields)


def test_join_refuses_a_shadow_copy_of_mutable_state() -> None:
    columns = split_state_record(REMOTE_COMMAND_LAYOUT, command(datetime.now(UTC)))
    columns["snapshot"]["lease_owner"] = "stale-owner"
    with pytest.raises(ValueError, match="separately stored field"):
        join_state_record(REMOTE_COMMAND_LAYOUT, columns)
