from __future__ import annotations

import json
import os
import random
from datetime import UTC, datetime, timedelta

import pytest

from gpu_fault.regional import RemoteCommandResult, RemoteCommandStatus
from gpu_fault.store import NotFoundError, PostgresStore, WorkflowLeaseError
from tests.store import test_postgres_state_tables as state_tables
from tests.store.test_postgres_workflow_state_tables import select_mode
from tests.store.test_state_table_payload import command

database = state_tables.database
migration_database = state_tables.migration_database
POSTGRES_URL = os.getenv("GPU_FAULT_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(
    not POSTGRES_URL, reason="GPU_FAULT_TEST_POSTGRES_URL is required"
)


@pytest.fixture(params=["legacy", "dual", "dedicated"])
def mode(request):
    return request.param


@pytest.fixture
def store(mode, migration_database):
    select_mode(migration_database, "remote_command", mode)
    assert POSTGRES_URL is not None
    instance = PostgresStore(POSTGRES_URL, initialize_schema=False)
    try:
        yield instance
    finally:
        instance.close()


def test_cleanup_skips_locked_rows_before_counting_its_limit(store) -> None:
    import psycopg

    now = datetime.now(UTC)
    oldest = command(now - timedelta(days=3)).model_copy(
        update={
            "command_id": "cleanup-locked",
            "status": RemoteCommandStatus.SUCCEEDED,
            "lease_expires_at": None,
        }
    )
    available = oldest.model_copy(
        update={
            "command_id": "cleanup-available",
            "updated_at": now - timedelta(days=2),
        }
    )
    for value in (oldest, available):
        store.ensure_remote_command(value)
    assert POSTGRES_URL is not None
    with psycopg.connect(POSTGRES_URL) as locker:
        locker.execute(
            "SELECT payload FROM gpu_fault_lock_control_state('remote_command', %s)",
            (oldest.command_id,),
        ).fetchone()
        assert store.cleanup_terminal_remote_commands(older_than=now, limit=1) == 1
        assert store.get_remote_command(oldest.command_id).status is oldest.status
        with pytest.raises(NotFoundError):
            store.get_remote_command(available.command_id)
    assert store.cleanup_terminal_remote_commands(older_than=now, limit=1) == 1
    assert store.list_remote_commands() == []


def test_reclaim_changes_the_lease_and_rejects_the_previous_owner(store) -> None:
    value = command(datetime.now(UTC))
    store.ensure_remote_command(value)
    first = store.claim_remote_commands(
        value.cluster_id, "first-executor", limit=1, lease_seconds=-1
    )[0]
    second = store.claim_remote_commands(
        value.cluster_id, "second-executor", limit=1, lease_seconds=60
    )[0]
    assert second.lease_token != first.lease_token
    with pytest.raises(WorkflowLeaseError):
        store.renew_remote_command_lease(
            value.cluster_id,
            value.command_id,
            "first-executor",
            first.lease_token,
            lease_seconds=60,
        )
    with pytest.raises(WorkflowLeaseError):
        store.complete_remote_command(
            value.cluster_id,
            value.command_id,
            RemoteCommandResult(
                lease_token=first.lease_token, status=RemoteCommandStatus.SUCCEEDED
            ),
        )
    completed = store.complete_remote_command(
        value.cluster_id,
        value.command_id,
        RemoteCommandResult(
            lease_token=second.lease_token,
            status=RemoteCommandStatus.SUCCEEDED,
            details={"owner": "second-executor"},
        ),
    )
    assert completed.last_lease_owner == "second-executor"
    assert completed.result_details == {"owner": "second-executor"}


def test_cancellation_survives_renewal_and_completion(store) -> None:
    value = command(datetime.now(UTC))
    store.ensure_remote_command(value)
    leased = store.claim_remote_commands(
        value.cluster_id, "executor", limit=1, lease_seconds=60
    )[0]
    assert store.cancel_remote_commands_for_workflow(
        value.workflow_request_id, reason="workflow stopped"
    ) == {"cancelled": 0, "cancellation_requested": 1}
    renewed = store.renew_remote_command_lease(
        value.cluster_id,
        value.command_id,
        "executor",
        leased.lease_token,
        lease_seconds=60,
    )
    assert renewed.cancellation_reason == "workflow stopped"
    completed = store.complete_remote_command(
        value.cluster_id,
        value.command_id,
        RemoteCommandResult(
            lease_token=leased.lease_token,
            status=RemoteCommandStatus.SUCCEEDED,
            details={"result": "executor evidence"},
        ),
    )
    assert completed.status is RemoteCommandStatus.FAILED
    assert completed.status_source == "completed-after-cancellation"
    assert completed.result_details["post_cancellation_status"] == "SUCCEEDED"
    assert completed.result_details["result"] == "executor evidence"
    assert store.get_remote_command(value.command_id) == completed
    assert store.cancel_remote_commands_for_workflow(
        value.workflow_request_id, reason="retry"
    ) == {"cancelled": 0, "cancellation_requested": 0}


@pytest.mark.parametrize("mode", ["dedicated"], indirect=True)
def test_full_transitions_preserve_the_unchanged_toast_snapshot(
    store, migration_database
) -> None:
    value = command(datetime.now(UTC))
    value.step.parameters["snapshot"] = random.Random(23).randbytes(120_000).hex()
    store.ensure_remote_command(value)
    size_query = (
        "SELECT pg_total_relation_size(reltoastrelid) FROM pg_class "
        "WHERE oid='gpu_fault_remote_commands'::regclass"
    )
    toast_before = migration_database.execute(size_query).fetchone()[0]
    assert toast_before > 0, "the fixture must exercise a toasted command snapshot"
    (leased,) = store.claim_remote_commands(
        value.cluster_id, "executor", limit=1, lease_seconds=600
    )
    for index in range(20):
        store.record_remote_command_progress(
            value.cluster_id,
            value.command_id,
            "executor",
            leased.lease_token,
            batched_results={str(index): {"status": "SUCCEEDED"}},
        )
    assert store.cancel_remote_commands_for_workflow(
        value.workflow_request_id, reason="workflow stopped"
    ) == {"cancelled": 0, "cancellation_requested": 1}
    completed = store.complete_remote_command(
        value.cluster_id,
        value.command_id,
        RemoteCommandResult(
            lease_token=leased.lease_token,
            status=RemoteCommandStatus.SUCCEEDED,
            details={"evidence": "completed"},
        ),
    )
    assert completed.status is RemoteCommandStatus.FAILED
    assert completed.step == value.step
    assert completed.workflow == value.workflow
    assert completed.incident == value.incident
    assert store.get_remote_command(value.command_id) == completed
    toast_after = migration_database.execute(size_query).fetchone()[0]
    assert toast_after == toast_before, (
        "full transitions rewrote the unchanged embedded snapshot into TOAST"
    )


@pytest.mark.parametrize("mode", ["dual"], indirect=True)
def test_cleanup_deletes_a_legacy_row_that_differs_from_its_projection(
    store, migration_database
) -> None:
    now = datetime.now(UTC)
    value = command(now - timedelta(days=3)).model_copy(
        update={"status": RemoteCommandStatus.SUCCEEDED, "lease_expires_at": None}
    )
    payload = value.model_dump(mode="json")
    payload.pop("last_lease_owner")
    payload["created_at"] = value.created_at.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    migration_database.execute(
        "INSERT INTO gpu_fault_objects(kind, key, payload) "
        "VALUES ('remote_command', %s, %s::jsonb)",
        (value.command_id, json.dumps(payload)),
    )
    projected = migration_database.execute(
        "SELECT payload FROM gpu_fault_remote_command_records WHERE key=%s",
        (value.command_id,),
    ).fetchone()[0]
    assert projected != payload, "the fixture did not exercise a reconstructed payload"
    assert store.get_remote_command(value.command_id) == value
    assert store.cleanup_terminal_remote_commands(older_than=now, limit=1) == 1
    assert store.list_remote_commands() == []
    assert migration_database.execute(
        "SELECT count(*) FROM gpu_fault_objects WHERE kind='remote_command' AND key=%s",
        (value.command_id,),
    ).fetchone() == (0,), "cleanup left the authoritative legacy row behind"
