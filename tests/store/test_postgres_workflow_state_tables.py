from __future__ import annotations

import json
import os
import random
import time
from datetime import UTC, datetime, timedelta

import pytest

from gpu_fault.models import WorkflowOperation, WorkflowStatus
from gpu_fault.regional import RemoteCommandStatus
from gpu_fault.state_table_migrate import (
    StateTableMigrationError,
    backfill_state_table,
    purge_legacy_state,
    set_state_table_mode,
    state_table_status,
)
from gpu_fault.store import PostgresStore
from gpu_fault.store.shared.errors import StaleFencingTokenError, StaleWriteError
from tests._builders import fault_incident, workflow_request, workflow_step
from tests.store import test_postgres_state_tables as state_tables
from tests.store.test_state_table_payload import command

database = state_tables.database
migration_database = state_tables.migration_database

POSTGRES_URL = os.getenv("GPU_FAULT_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(
    not POSTGRES_URL, reason="GPU_FAULT_TEST_POSTGRES_URL is not configured"
)


def select_mode(connection, kind: str, mode: str) -> None:
    if mode == "legacy":
        return
    set_state_table_mode(connection, kind, "dual", expected_mode="legacy")
    if mode == "dedicated":
        backfill_state_table(connection, kind)
        set_state_table_mode(
            connection, kind, "dedicated", expected_mode="dual", confirm_dedicated=True
        )


@pytest.mark.parametrize("workflow_mode", ["legacy", "dual", "dedicated"])
@pytest.mark.parametrize("command_mode", ["legacy", "dual", "dedicated"])
def test_workflow_and_remote_command_modes_keep_fencing_and_lease_contracts(
    migration_database, workflow_mode: str, command_mode: str
) -> None:
    connection = migration_database
    select_mode(connection, "remote_command", command_mode)
    select_mode(connection, "workflow", workflow_mode)
    assert POSTGRES_URL is not None, "mixed-mode contracts require local PostgreSQL"
    store = PostgresStore(POSTGRES_URL, initialize_schema=False)
    try:
        at = datetime.now(UTC)
        flow = workflow_request("mixed-mode-workflow", "mixed-mode-incident")
        store.save_workflow(flow)
        assert store.get_workflow(flow.request_id).model_dump(
            mode="json"
        ) == flow.model_dump(mode="json")
        assert store.workflow_status_counts()[flow.status] == 1
        assert [item.request_id for item in store.list_workflows({flow.status})] == [
            flow.request_id
        ]
        leased = store.claim_workflow(
            flow.request_id,
            "workflow-executor",
            flow.fencing_token,
            now=at,
            lease_duration=timedelta(minutes=3),
            remediation_budget_claims={"cluster-test/node/node-test": 1},
        )
        renewed = store.renew_workflow_lease(
            flow.request_id,
            "workflow-executor",
            leased.execution_epoch,
            now=at + timedelta(seconds=120),
        )
        assert renewed.execution_lease_expires_at > leased.execution_lease_expires_at
        merged = renewed.model_copy(
            update={"merge_revision": renewed.merge_revision + 1}
        )
        store.save_workflow(merged, expected=renewed)
        with pytest.raises(StaleWriteError, match="merge_revision"):
            store.save_workflow(renewed)
        remote = command(at).model_copy(
            update={
                "workflow_request_id": flow.request_id,
                "incident_id": flow.incident_id,
                "workflow": merged,
                "fencing_token": merged.fencing_token,
            }
        )
        store.ensure_remote_command(remote)
        claimed = store.claim_remote_commands(
            remote.cluster_id, "command-executor", limit=1, lease_seconds=60
        )
        assert len(claimed) == 1
        assert claimed[0].lease_token is not None, (
            "the command claim omitted its lease token"
        )
        current = store.get_workflow(flow.request_id)
        store.save_workflow(
            current.model_copy(update={"fencing_token": current.fencing_token + 1}),
            expected=current,
        )
        with pytest.raises(StaleFencingTokenError, match="fencing token"):
            store.save_workflow_if_leased(
                current, "workflow-executor", current.execution_epoch
            )
        cancellation = store.renew_remote_command_lease(
            remote.cluster_id,
            remote.command_id,
            "command-executor",
            claimed[0].lease_token,
            lease_seconds=60,
        )
        assert cancellation.status is RemoteCommandStatus.LEASED
        assert cancellation.cancellation_requested_at is not None, (
            "a command from an old workflow fence was not marked for cancellation"
        )
        assert cancellation.workflow.fencing_token == remote.workflow.fencing_token
    finally:
        store.close()


def test_workflow_lease_only_updates_are_hot_and_preserve_updated_at(
    migration_database,
) -> None:
    connection = migration_database
    select_mode(connection, "workflow", "dedicated")
    assert POSTGRES_URL is not None, "workflow HOT checks require local PostgreSQL"
    store = PostgresStore(POSTGRES_URL, initialize_schema=False)
    at = datetime.now(UTC)
    try:
        flow = workflow_request(
            "workflow-hot", "incident-hot", status=WorkflowStatus.RUNNING
        ).model_copy(
            update={
                "execution_owner_id": "hot-owner",
                "execution_epoch": 1,
                "execution_lease_expires_at": at + timedelta(seconds=60),
                "created_at": at,
                "updated_at": at,
                "official_steps": [
                    workflow_step(
                        WorkflowOperation.RESTART_NODE,
                        parameters={
                            "snapshot": random.Random(19).randbytes(120_000).hex()
                        },
                    )
                ],
            }
        )
        store.save_workflow(flow)
        toast_before = connection.execute(
            "SELECT pg_total_relation_size(reltoastrelid) FROM pg_class "
            "WHERE oid='gpu_fault_workflows'::regclass"
        ).fetchone()[0]
        connection.execute(
            "SELECT pg_stat_reset_single_table_counters('gpu_fault_workflows'::regclass)"
        )
        early = store.renew_workflow_lease(
            flow.request_id,
            "hot-owner",
            1,
            now=at + timedelta(seconds=1),
            lease_duration=timedelta(seconds=60),
        )
        assert early.execution_lease_expires_at == flow.execution_lease_expires_at
        for index in range(1, 21):
            renewed = store.renew_workflow_lease(
                flow.request_id,
                "hot-owner",
                1,
                now=at + timedelta(seconds=31 * index),
                lease_duration=timedelta(seconds=60),
            )
            assert renewed.updated_at == at
    finally:
        store.close()
    deadline = time.monotonic() + 5
    while True:
        connection.execute("SELECT pg_stat_clear_snapshot()")
        updates, hot = connection.execute(
            "SELECT n_tup_upd,n_tup_hot_upd FROM pg_stat_user_tables WHERE relname='gpu_fault_workflows'"
        ).fetchone()
        if updates >= 20 or time.monotonic() >= deadline:
            break
        time.sleep(0.05)
    assert updates == hot == 20, (
        "workflow renewal either rewrote indexed fields or lost the half-lease optimization"
    )
    toast_after = connection.execute(
        "SELECT pg_total_relation_size(reltoastrelid) FROM pg_class WHERE oid='gpu_fault_workflows'::regclass"
    ).fetchone()[0]
    assert toast_after == toast_before, (
        "workflow renewal rewrote the large execution payload"
    )


@pytest.mark.parametrize("mode", ["legacy", "dual", "dedicated"])
def test_workflow_dispatch_cursor_and_predecessor_filters_survive_migration(
    migration_database, mode: str
) -> None:
    select_mode(migration_database, "workflow", mode)
    assert POSTGRES_URL is not None, "workflow dispatch checks require local PostgreSQL"
    store = PostgresStore(POSTGRES_URL, initialize_schema=False)
    at = datetime.now(UTC)
    try:
        first = workflow_request("cursor-first", "incident-first").model_copy(
            update={
                "created_at": at - timedelta(seconds=30),
                "not_before": at - timedelta(seconds=10),
            }
        )
        second = workflow_request("cursor-second", "incident-second").model_copy(
            update={
                "created_at": at - timedelta(seconds=20),
                "not_before": at - timedelta(seconds=5),
            }
        )
        held = workflow_request("cursor-held", "incident-held").model_copy(
            update={
                "predecessor_workflow_id": first.request_id,
                "preempt_predecessor": True,
                "created_at": at - timedelta(seconds=40),
            }
        )
        future = workflow_request("cursor-future", "incident-future").model_copy(
            update={"not_before": at + timedelta(minutes=1)}
        )
        for flow in (first, second, held, future):
            store.save_workflow(flow)
        statuses = {
            WorkflowStatus.PENDING,
            WorkflowStatus.RUNNING,
            WorkflowStatus.SAFETY_PENDING,
        }
        page = store.list_workflows(statuses, dispatchable_at=at, limit=1)
        assert [item.request_id for item in page] == [first.request_id]
        following = store.list_workflows(
            statuses, dispatchable_at=at, after=page[0], limit=10
        )
        assert [item.request_id for item in following] == [second.request_id]
        assert store.count_held_workflows(statuses, dispatchable_at=at) == {
            "not_before": 1,
            "predecessor": 1,
        }
        assert store.has_workflow_successor(first.request_id), (
            "predecessor lookup lost a migrated successor"
        )
        assert (
            store.get_preempting_successor(first.request_id).request_id
            == held.request_id
        )
    finally:
        store.close()


@pytest.mark.parametrize("mode", ["legacy", "dual", "dedicated"])
def test_reconcile_queries_read_workflows_from_the_selected_storage(
    migration_database, mode: str
) -> None:
    select_mode(migration_database, "workflow", mode)
    assert POSTGRES_URL is not None, "workflow reconcile reads require local PostgreSQL"
    store = PostgresStore(POSTGRES_URL, initialize_schema=False)
    at = datetime.now(UTC)
    try:
        incident = fault_incident("orphan-incident", "orphan-event")
        store.save_incident(incident)
        orphan = workflow_request("orphan-flow", incident.incident_id).model_copy(
            update={"created_at": at - timedelta(minutes=5)}
        )
        store.save_workflow(orphan)
        missing = fault_incident(
            "missing-workflow-incident", "missing-workflow-event"
        ).model_copy(update={"workflow_request_id": "nonexistent-workflow"})
        store.save_incident(missing)
        failed = workflow_request(
            "failed-flow", "failed-incident", status=WorkflowStatus.FAILED
        )
        store.save_workflow(failed)
        assert [
            item.request_id for item in store.list_orphan_workflows(created_before=at)
        ] == [orphan.request_id]
        assert [
            item.incident_id for item in store.list_incidents_with_missing_workflow()
        ] == [missing.incident_id]
        assert [
            item.request_id for item in store.list_unhandled_failed_workflows()
        ] == [failed.request_id]
    finally:
        store.close()


def test_workflow_backfill_canonicalizes_and_retires_only_workflow_legacy_state(
    migration_database,
) -> None:
    connection = migration_database
    flow = workflow_request(
        "workflow-migrate", "incident-migrate", status=WorkflowStatus.SUCCEEDED
    )
    payload = flow.model_dump(mode="json")
    del payload["merge_revision"]
    connection.execute(
        "INSERT INTO gpu_fault_objects(kind,key,payload) VALUES ('workflow',%s,%s::jsonb)",
        (flow.request_id, json.dumps(payload)),
    )
    set_state_table_mode(connection, "workflow", "dual", expected_mode="legacy")
    before = state_table_status(connection, "workflow")
    assert before["missing_rows"] == before["noncanonical_records"] == 1
    result = backfill_state_table(connection, "workflow", batch_size=1, max_batches=1)
    assert result["status"]["verification_performed"] is False, (
        "an unfinished backfill must not scan and validate the entire payload history"
    )
    assert "verified" not in result["status"], (
        "an unverified batch must not claim a complete verification result"
    )
    assert result["status"]["backfill_complete"] is False, (
        "a full batch still needs a subsequent scan to prove the end of the table"
    )
    assert state_table_status(connection, "workflow")["verified"] is True, (
        "explicit status must still verify all copied records"
    )
    with pytest.raises(StateTableMigrationError, match="incomplete"):
        set_state_table_mode(
            connection,
            "workflow",
            "dedicated",
            expected_mode="dual",
            confirm_dedicated=True,
        )
    backfill_state_table(connection, "workflow")
    set_state_table_mode(
        connection,
        "workflow",
        "dedicated",
        expected_mode="dual",
        confirm_dedicated=True,
    )
    retired = purge_legacy_state(connection, "workflow", confirm=True)
    assert retired["deleted"] == 1
    assert retired["retired_indexes"], "no workflow legacy indexes were retired"
    assert state_table_status(connection, "remote_command")["mode"] == "legacy"
    assert POSTGRES_URL is not None, (
        "workflow retirement checks require local PostgreSQL"
    )
    store = PostgresStore(POSTGRES_URL)
    try:
        assert store.get_workflow(flow.request_id).model_dump(
            mode="json"
        ) == flow.model_dump(mode="json")
    finally:
        store.close()
    assert (
        connection.execute(
            "SELECT indexname FROM pg_indexes WHERE indexname=ANY(%s)",
            (retired["retired_indexes"],),
        ).fetchall()
        == []
    )
    assert (
        connection.execute(
            "SELECT to_regclass('gpu_fault_remote_command_claim')"
        ).fetchone()[0]
        is not None
    ), "workflow retirement dropped a remote-command index"


def test_workflow_cutover_requires_drained_leases(migration_database) -> None:
    connection = migration_database
    assert POSTGRES_URL is not None, "workflow cutover checks require local PostgreSQL"
    store = PostgresStore(POSTGRES_URL, initialize_schema=False)
    try:
        flow = workflow_request(
            "workflow-drain", "incident-drain", status=WorkflowStatus.RUNNING
        ).model_copy(
            update={
                "execution_owner_id": "workflow-owner",
                "execution_epoch": 1,
                "execution_lease_expires_at": datetime.now(UTC) + timedelta(minutes=5),
            }
        )
        store.save_workflow(flow)
        set_state_table_mode(connection, "workflow", "dual", expected_mode="legacy")
        backfill_state_table(connection, "workflow")
        with pytest.raises(StateTableMigrationError, match="drained workflow leases"):
            set_state_table_mode(
                connection,
                "workflow",
                "dedicated",
                expected_mode="dual",
                confirm_dedicated=True,
            )
        assert state_table_status(connection, "workflow")["mode"] == "dual"
        store.save_workflow(
            flow.model_copy(
                update={"execution_owner_id": None, "execution_lease_expires_at": None}
            ),
            expected=flow,
        )
        set_state_table_mode(
            connection,
            "workflow",
            "dedicated",
            expected_mode="dual",
            confirm_dedicated=True,
        )
    finally:
        store.close()


def test_reentering_dual_cannot_read_the_previous_staging_copy(
    migration_database,
) -> None:
    connection = migration_database
    assert POSTGRES_URL is not None, "dual reentry checks require local PostgreSQL"
    store = PostgresStore(POSTGRES_URL, initialize_schema=False)
    try:
        flow = workflow_request("workflow-retry", "incident-retry")
        store.save_workflow(flow)
        set_state_table_mode(connection, "workflow", "dual", expected_mode="legacy")
        backfill_state_table(connection, "workflow")
        set_state_table_mode(connection, "workflow", "legacy", expected_mode="dual")
        current = flow.model_copy(update={"status": WorkflowStatus.SUCCEEDED})
        store.save_workflow(current, expected=flow)
        set_state_table_mode(connection, "workflow", "dual", expected_mode="legacy")
        assert store.get_workflow(flow.request_id).status is WorkflowStatus.SUCCEEDED
        assert state_table_status(connection, "workflow")["missing_rows"] == 1
        backfill_state_table(connection, "workflow")
        assert state_table_status(connection, "workflow")["verified"] is True, (
            "reentering dual must rebuild and verify the current legacy record"
        )
    finally:
        store.close()


@pytest.mark.parametrize(
    ("query", "expected_index"),
    [
        ("dispatch", "gpu_fault_workflows_dispatch"),
        ("failed", "gpu_fault_workflows_unhandled_failed"),
        ("blocked", "gpu_fault_workflows_blocked_updated"),
    ],
)
def test_dedicated_workflow_hot_queries_use_indexes_without_reading_legacy(
    migration_database, query: str, expected_index: str
) -> None:
    connection = migration_database
    at = datetime.now(UTC)
    template = workflow_request(
        "index-template", "index-incident", status=WorkflowStatus.SUCCEEDED
    )
    connection.execute(
        "INSERT INTO gpu_fault_objects(kind,key,payload) "
        "SELECT 'workflow', 'wf-indexed-' || n, "
        "jsonb_set(%s::jsonb, '{request_id}', to_jsonb('wf-indexed-' || n)) "
        "FROM generate_series(1,500) n",
        (template.model_dump_json(),),
    )
    select_mode(connection, "workflow", "dedicated")
    assert POSTGRES_URL is not None, "workflow index checks require local PostgreSQL"
    store = PostgresStore(POSTGRES_URL, initialize_schema=False)
    try:
        for status in (
            WorkflowStatus.PENDING,
            WorkflowStatus.FAILED,
            WorkflowStatus.BLOCKED,
        ):
            for index in range(2):
                store.save_workflow(
                    workflow_request(
                        f"index-{status.value}-{index}", "index-incident", status=status
                    ).model_copy(
                        update={
                            "created_at": at - timedelta(seconds=30 - index),
                            "updated_at": at - timedelta(seconds=10 - index),
                        }
                    )
                )
        if query == "dispatch":
            statuses = {
                WorkflowStatus.PENDING,
                WorkflowStatus.RUNNING,
                WorkflowStatus.SAFETY_PENDING,
            }
            (anchor,) = store.list_workflows(statuses, dispatchable_at=at, limit=1)
            statement, parameters = store.workflow_scan_query(
                statuses, dispatchable_at=at, after=anchor, limit=1
            )
        elif query == "failed":
            statement, parameters = store.unhandled_failed_workflows_query(limit=1)
        else:
            statement, parameters = store.workflow_scan_query(
                {WorkflowStatus.BLOCKED}, limit=1, newest_first=False
            )
        connection.execute("ANALYZE gpu_fault_objects")
        connection.execute("ANALYZE gpu_fault_workflows")
        plan = connection.execute(
            "EXPLAIN (ANALYZE, FORMAT JSON, COSTS OFF) " + statement, parameters
        ).fetchone()[0]
        nodes = list(state_tables.plan_nodes(plan))
        assert any(
            node.get("Index Name") == expected_index and node.get("Actual Loops", 0) > 0
            for node in nodes
        ), plan
        assert all(
            node.get("Actual Loops") == 0
            for node in nodes
            if node.get("Relation Name") == "gpu_fault_objects"
        ), "dedicated workflow queries still accessed legacy rows"
    finally:
        store.close()
