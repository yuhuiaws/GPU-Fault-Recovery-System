from __future__ import annotations

import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta, timezone

import pytest

from gpu_fault.models import datetime_json_text
from gpu_fault.regional import RemoteCommandResult, RemoteCommandStatus
from gpu_fault.state_table_migrate import (
    StateTableMigrationError,
    backfill_state_table,
    purge_legacy_state,
    purge_legacy_state_rows,
    set_state_table_mode,
    state_table_status,
)
from gpu_fault.store import PostgresStore
from gpu_fault.store.postgres.state_table_payload import (
    REMOTE_COMMAND_LAYOUT,
    split_state_record,
    state_update_columns,
)
from tests.store._postgres_processor_claim_support import _truncate
from tests.store.test_state_table_payload import command

POSTGRES_URL = os.getenv("GPU_FAULT_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(
    not POSTGRES_URL, reason="GPU_FAULT_TEST_POSTGRES_URL is not configured"
)


@pytest.fixture(scope="module")
def database():
    import psycopg

    assert POSTGRES_URL is not None
    _truncate()
    store = PostgresStore(POSTGRES_URL)
    store.close()
    with psycopg.connect(POSTGRES_URL, autocommit=True) as connection:
        yield connection


def test_state_schema_starts_in_legacy_without_backfilling(migration_database) -> None:
    database = migration_database
    rows = database.execute(
        "SELECT kind, mode FROM gpu_fault_control_state_modes ORDER BY kind"
    ).fetchall()
    assert rows == [("remote_command", "legacy"), ("workflow", "legacy")]
    assert (
        database.execute("SELECT count(*) FROM gpu_fault_remote_commands").fetchone()[0]
        == 0
    )
    attributes = database.execute(
        "SELECT attname FROM pg_attribute "
        "WHERE attrelid='gpu_fault_remote_commands'::regclass "
        "AND attnum>0 AND NOT attisdropped"
    ).fetchall()
    assert {row[0] for row in attributes} == set(REMOTE_COMMAND_LAYOUT.column_names)


@pytest.mark.parametrize("offset", [None, UTC])
def test_database_projection_preserves_the_complete_command_json(
    migration_database, offset: timezone | None
) -> None:
    database = migration_database
    payload = command(datetime(2026, 9, 11, 12, 0, tzinfo=offset)).model_dump(
        mode="json"
    )
    result = database.execute(
        "SELECT gpu_fault_remote_command_payload(gpu_fault_remote_command_columns(%s::jsonb))",
        (json.dumps(payload),),
    ).fetchone()[0]
    assert result == payload


def test_native_writes_cannot_bypass_legacy_authority(migration_database) -> None:
    import psycopg

    database = migration_database
    payload = command(datetime.now(UTC)).model_dump_json()
    with pytest.raises(psycopg.Error, match="not authoritative"):
        database.execute(
            "INSERT INTO gpu_fault_remote_commands "
            "SELECT (gpu_fault_remote_command_columns(%s::jsonb)).*",
            (payload,),
        )
    assert (
        database.execute("SELECT count(*) FROM gpu_fault_remote_commands").fetchone()[0]
        == 0
    )


@pytest.mark.parametrize("mode", ["legacy", "dual", "dedicated"])
def test_database_routes_full_writes_and_lease_only_patches(
    migration_database, mode: str
) -> None:
    database = migration_database
    value = command(datetime(2026, 9, 11, tzinfo=UTC))
    with database.transaction(force_rollback=True):
        database.execute(
            "UPDATE gpu_fault_control_state_modes SET mode=%s, "
            "dedicated_at=CASE WHEN %s='dedicated' THEN now() ELSE NULL END "
            "WHERE kind='remote_command'",
            (mode, mode),
        )
        columns = split_state_record(REMOTE_COMMAND_LAYOUT, value)
        written = database.execute(
            "SELECT gpu_fault_put_control_state('remote_command', %s, %s::jsonb)",
            (value.command_id, json.dumps(columns, default=datetime_json_text)),
        ).fetchone()[0]
        assert written is True
        actual = database.execute(
            "SELECT payload FROM gpu_fault_remote_command_records WHERE key=%s",
            (value.command_id,),
        ).fetchone()[0]
        assert actual == value.model_dump(mode="json")
        legacy = database.execute(
            "SELECT count(*) FROM gpu_fault_objects WHERE kind='remote_command'"
        ).fetchone()[0]
        native = database.execute(
            "SELECT count(*) FROM gpu_fault_remote_commands"
        ).fetchone()[0]
        assert (legacy, native) == {
            "legacy": (1, 0),
            "dual": (1, 1),
            "dedicated": (0, 1),
        }[mode]
        updated = value.model_copy(
            update={"updated_at": datetime(2026, 9, 12, tzinfo=UTC)}
        )
        patch = state_update_columns(
            REMOTE_COMMAND_LAYOUT, updated, frozenset({"updated_at"})
        )
        assert (
            database.execute(
                "SELECT gpu_fault_patch_control_state('remote_command', %s, %s::jsonb)",
                (value.command_id, json.dumps(patch, default=datetime_json_text)),
            ).fetchone()[0]
            is True
        )
        actual = database.execute(
            "SELECT payload FROM gpu_fault_remote_command_records WHERE key=%s",
            (value.command_id,),
        ).fetchone()[0]
        assert actual == updated.model_dump(mode="json")


@pytest.mark.parametrize("mode", ["legacy", "dual", "dedicated"])
def test_store_command_lifecycle_uses_the_authoritative_state(
    migration_database, mode: str
) -> None:
    database = migration_database
    assert POSTGRES_URL is not None
    database.execute(
        "UPDATE gpu_fault_control_state_modes SET mode=%s, "
        "dedicated_at=CASE WHEN %s='dedicated' THEN now() ELSE NULL END "
        "WHERE kind='remote_command'",
        (mode, mode),
    )
    store = PostgresStore(POSTGRES_URL, initialize_schema=False)
    try:
        value = command(datetime.now(UTC))
        store.save_workflow(value.workflow)
        store.ensure_remote_command(value)
        assert [item.command_id for item in store.list_remote_commands()] == [
            value.command_id
        ]
        claimed = store.claim_remote_commands(
            value.cluster_id, "executor-state-table", limit=1, lease_seconds=60
        )
        assert len(claimed) == 1
        assert claimed[0].lease_token is not None
        renewed = store.renew_remote_command_lease(
            value.cluster_id,
            value.command_id,
            "executor-state-table",
            claimed[0].lease_token,
            lease_seconds=60,
        )
        assert renewed.status is RemoteCommandStatus.LEASED
        completed = store.complete_remote_command(
            value.cluster_id,
            value.command_id,
            RemoteCommandResult(
                lease_token=claimed[0].lease_token, status=RemoteCommandStatus.SUCCEEDED
            ),
        )
        assert completed.status is RemoteCommandStatus.SUCCEEDED
        assert store.remote_command_stats()["by_status"]["SUCCEEDED"] == 1
        assert (
            store.cleanup_terminal_remote_commands(
                older_than=datetime.now(UTC) + timedelta(days=1), limit=10
            )
            == 1
        )
        assert store.list_remote_commands() == []
    finally:
        store.close()
        database.execute(
            "TRUNCATE gpu_fault_objects, gpu_fault_remote_commands, gpu_fault_workflows"
        )
        database.execute(
            "UPDATE gpu_fault_control_state_modes SET mode='legacy', dedicated_at=NULL "
            "WHERE kind='remote_command'"
        )


@pytest.fixture
def migration_database(database):
    def reset() -> None:
        database.execute(
            "TRUNCATE gpu_fault_objects, gpu_fault_remote_commands, gpu_fault_workflows"
        )
        database.execute(
            "UPDATE gpu_fault_control_state_modes SET mode='legacy', revision=0, "
            "dedicated_at=NULL, legacy_purged=FALSE, backfill_after_key=NULL, backfill_complete=FALSE"
        )
        assert POSTGRES_URL is not None
        initialized = PostgresStore(POSTGRES_URL)
        initialized.close()

    reset()
    try:
        yield database
    finally:
        reset()


def terminal_command(command_id: str):
    return command(datetime(2026, 9, 11, tzinfo=UTC)).model_copy(
        update={
            "command_id": command_id,
            "status": RemoteCommandStatus.SUCCEEDED,
            "lease_expires_at": None,
        }
    )


def test_migration_resumes_and_captures_new_rows_behind_its_cursor(
    migration_database,
) -> None:
    import psycopg

    connection = migration_database
    kind = "remote_command"
    for name in ("migration-a", "migration-b"):
        value = terminal_command(name)
        connection.execute(
            "INSERT INTO gpu_fault_objects(kind, key, payload) VALUES (%s, %s, %s::jsonb)",
            (kind, name, value.model_dump_json()),
        )
    set_state_table_mode(connection, kind, "dual", expected_mode="legacy")
    assert state_table_status(connection, kind)["missing_rows"] == 2
    first = backfill_state_table(connection, kind, batch_size=1, max_batches=1)
    assert first["copied"] == 1
    assert first["status"]["backfill_complete"] is False
    with pytest.raises(StateTableMigrationError, match="incomplete"):
        set_state_table_mode(
            connection, kind, "dedicated", expected_mode="dual", confirm_dedicated=True
        )
    value = terminal_command("before-cursor")
    connection.execute(
        "INSERT INTO gpu_fault_objects(kind, key, payload) VALUES (%s, %s, %s::jsonb)",
        (kind, value.command_id, value.model_dump_json()),
    )
    finished = backfill_state_table(connection, kind, batch_size=1, max_batches=4)
    assert finished["status"]["verified"] is True
    assert finished["status"]["backfill_complete"] is True
    assert finished["status"]["dedicated_rows"] == 3
    set_state_table_mode(
        connection, kind, "dedicated", expected_mode="dual", confirm_dedicated=True
    )
    with pytest.raises(psycopg.Error, match="fenced after cutover"):
        connection.execute(
            "UPDATE gpu_fault_objects SET payload=payload WHERE kind=%s", (kind,)
        )
    with pytest.raises(StateTableMigrationError, match="irreversible"):
        set_state_table_mode(connection, kind, "legacy", expected_mode="dedicated")
    first_purge = purge_legacy_state_rows(
        connection, kind, confirm=True, batch_size=1, max_batches=1
    )
    assert first_purge["deleted"] == 1
    assert first_purge["has_more"] is True
    final = purge_legacy_state_rows(connection, kind, confirm=True)
    assert final["status"]["legacy_rows"] == 0
    assert final["status"]["dedicated_rows"] == 3
    assert final["status"]["legacy_purged"] is True


def test_cutover_refuses_open_commands_and_unconfirmed_changes(
    migration_database,
) -> None:
    connection = migration_database
    kind = "remote_command"
    value = command(datetime.now(UTC))
    connection.execute(
        "INSERT INTO gpu_fault_objects(kind, key, payload) VALUES (%s, %s, %s::jsonb)",
        (kind, value.command_id, value.model_dump_json()),
    )
    with pytest.raises(StateTableMigrationError, match="dual mode"):
        backfill_state_table(connection, kind)
    set_state_table_mode(connection, kind, "dual", expected_mode="legacy")
    backfill_state_table(connection, kind)
    with pytest.raises(StateTableMigrationError, match="confirmation"):
        set_state_table_mode(connection, kind, "dedicated", expected_mode="dual")
    with pytest.raises(StateTableMigrationError, match="drained remote commands"):
        set_state_table_mode(
            connection, kind, "dedicated", expected_mode="dual", confirm_dedicated=True
        )
    assert state_table_status(connection, kind)["mode"] == "dual"


def activate_empty_dedicated(connection) -> None:
    set_state_table_mode(connection, "remote_command", "dual", expected_mode="legacy")
    backfill_state_table(connection, "remote_command")
    set_state_table_mode(
        connection,
        "remote_command",
        "dedicated",
        expected_mode="dual",
        confirm_dedicated=True,
    )


def test_dedicated_lease_renewals_are_hot_and_do_not_grow_toast(
    migration_database,
) -> None:
    connection = migration_database
    activate_empty_dedicated(connection)
    assert POSTGRES_URL is not None
    store = PostgresStore(POSTGRES_URL, initialize_schema=False)
    try:
        value = command(datetime.now(UTC)).model_copy(
            update={
                "status": RemoteCommandStatus.LEASED,
                "lease_owner": "hot-executor",
                "lease_token": "local-test-lease",
                "lease_expires_at": datetime.now(UTC) + timedelta(minutes=5),
            }
        )
        value.step.parameters["snapshot"] = random.Random(17).randbytes(120_000).hex()
        store.ensure_remote_command(value)
        toast_before = connection.execute(
            "SELECT pg_total_relation_size(reltoastrelid) FROM pg_class "
            "WHERE oid='gpu_fault_remote_commands'::regclass"
        ).fetchone()[0]
        connection.execute(
            "SELECT pg_stat_reset_single_table_counters('gpu_fault_remote_commands'::regclass)"
        )
        for _ in range(20):
            store.renew_remote_command_lease(
                value.cluster_id,
                value.command_id,
                "hot-executor",
                "local-test-lease",
                lease_seconds=60,
            )
    finally:
        store.close()
    deadline = time.monotonic() + 5
    while True:
        connection.execute("SELECT pg_stat_clear_snapshot()")
        updates, hot = connection.execute(
            "SELECT n_tup_upd, n_tup_hot_upd FROM pg_stat_user_tables "
            "WHERE relname='gpu_fault_remote_commands'"
        ).fetchone()
        if updates >= 20 or time.monotonic() >= deadline:
            break
        time.sleep(0.05)
    assert updates >= 20
    assert hot == updates, "lease-only updates maintained indexes instead of using HOT"
    toast_after = connection.execute(
        "SELECT pg_total_relation_size(reltoastrelid) FROM pg_class "
        "WHERE oid='gpu_fault_remote_commands'::regclass"
    ).fetchone()[0]
    assert toast_before > 0
    assert toast_after == toast_before, (
        "renewals rewrote the embedded snapshots into TOAST"
    )


def test_retired_legacy_indexes_are_not_recreated_or_required(
    migration_database,
) -> None:
    connection = migration_database
    activate_empty_dedicated(connection)
    result = purge_legacy_state(connection, "remote_command", confirm=True)
    assert result["retired_indexes"], "the legacy index retirement selected no indexes"
    assert POSTGRES_URL is not None
    store = PostgresStore(POSTGRES_URL)
    store.close()
    remaining = connection.execute(
        "SELECT indexname FROM pg_indexes WHERE schemaname=current_schema() AND indexname=ANY(%s)",
        (result["retired_indexes"],),
    ).fetchall()
    assert remaining == [], "schema ensure recreated retired legacy indexes"
    store = PostgresStore(POSTGRES_URL, initialize_schema=False)
    store.close()


def plan_nodes(value):
    if isinstance(value, dict):
        yield value
        for nested in value.values():
            yield from plan_nodes(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from plan_nodes(nested)


def test_dedicated_read_plan_prunes_legacy_and_uses_the_claim_index(
    migration_database,
) -> None:
    connection = migration_database
    value = terminal_command("index-template")
    value.step.parameters.clear()
    connection.execute(
        "INSERT INTO gpu_fault_objects(kind, key, payload) "
        "SELECT 'remote_command', 'indexed-' || n, "
        "jsonb_set(%s::jsonb, '{command_id}', to_jsonb('indexed-' || n)) "
        "FROM generate_series(1, 500) n",
        (value.model_dump_json(),),
    )
    set_state_table_mode(connection, "remote_command", "dual", expected_mode="legacy")
    backfill_state_table(connection, "remote_command", batch_size=100, max_batches=10)
    set_state_table_mode(
        connection,
        "remote_command",
        "dedicated",
        expected_mode="dual",
        confirm_dedicated=True,
    )
    assert POSTGRES_URL is not None
    store = PostgresStore(POSTGRES_URL, initialize_schema=False)
    pending = command(datetime.now(UTC))
    try:
        store.ensure_remote_command(pending)
    finally:
        store.close()
    connection.execute("ANALYZE gpu_fault_objects")
    connection.execute("ANALYZE gpu_fault_remote_commands")
    plan = connection.execute(
        "EXPLAIN (ANALYZE, FORMAT JSON, COSTS OFF) "
        "SELECT key FROM gpu_fault_remote_command_records "
        "WHERE cluster_id=%s AND status IN ('PENDING', 'LEASED', 'WAITING') "
        "AND execution_owner=%s ORDER BY created_at, key LIMIT 5",
        (pending.cluster_id, pending.step.execution_owner),
    ).fetchone()[0]
    nodes = list(plan_nodes(plan))
    assert any(
        node.get("Index Name") == "gpu_fault_remote_commands_claim" for node in nodes
    ), "the dedicated claim predicates did not use their column index"
    assert all(
        node.get("Actual Loops") == 0
        for node in nodes
        if node.get("Relation Name") == "gpu_fault_objects"
    ), "a dedicated claim still read legacy heap rows"


@pytest.mark.parametrize(
    ("isolation", "expected_sqlstate"),
    [("READ COMMITTED", "55000"), ("REPEATABLE READ", "40001")],
)
def test_a_waiting_old_writer_cannot_cross_cutover(
    migration_database, isolation: str, expected_sqlstate: str
) -> None:
    import psycopg

    connection = migration_database
    value = terminal_command("cutover-writer")
    connection.execute(
        "INSERT INTO gpu_fault_objects(kind, key, payload) VALUES ('remote_command', %s, %s::jsonb)",
        (value.command_id, value.model_dump_json()),
    )
    set_state_table_mode(connection, "remote_command", "dual", expected_mode="legacy")
    backfill_state_table(connection, "remote_command")
    ready = threading.Event()
    writer_pid: list[int] = []

    def old_writer() -> str | None:
        assert POSTGRES_URL is not None
        try:
            with psycopg.connect(POSTGRES_URL) as writer:
                writer.execute(f"SET TRANSACTION ISOLATION LEVEL {isolation}")
                writer_pid.append(
                    writer.execute("SELECT pg_backend_pid()").fetchone()[0]
                )
                ready.set()
                writer.execute(
                    "UPDATE gpu_fault_objects SET payload=payload "
                    "WHERE kind='remote_command' AND key=%s",
                    (value.command_id,),
                )
        except psycopg.Error as exc:
            return exc.sqlstate
        return None

    with ThreadPoolExecutor(max_workers=1) as executor:
        with connection.transaction():
            for kind in ("remote_command", "workflow"):
                connection.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended(%s,0))",
                    (f"gpu_fault_control_state/{kind}",),
                )
            pending = executor.submit(old_writer)
            assert ready.wait(5), (
                "old writer did not establish its pre-cutover snapshot"
            )
            deadline = time.monotonic() + 5
            while True:
                connection.execute("SELECT pg_stat_clear_snapshot()")
                wait = connection.execute(
                    "SELECT wait_event FROM pg_stat_activity WHERE pid=%s",
                    (writer_pid[0],),
                ).fetchone()
                if wait and wait[0] == "advisory":
                    break
                assert time.monotonic() < deadline, (
                    "old writer never waited behind the cutover fence"
                )
                time.sleep(0.02)
            set_state_table_mode(
                connection,
                "remote_command",
                "dedicated",
                expected_mode="dual",
                confirm_dedicated=True,
            )
        assert pending.result(timeout=5) == expected_sqlstate
    assert state_table_status(connection, "remote_command")["mode"] == "dedicated"


@pytest.mark.parametrize(
    ("mutation", "restore", "inspection", "message"),
    [
        (
            "ALTER FUNCTION gpu_fault_control_state_lock_mode(text) STABLE",
            "ALTER FUNCTION gpu_fault_control_state_lock_mode(text) VOLATILE",
            "SELECT pg_get_functiondef('gpu_fault_control_state_lock_mode(text)'::regprocedure)",
            "function definitions differ",
        ),
        (
            "ALTER FUNCTION gpu_fault_control_state_lock_mode(text) SECURITY DEFINER",
            "ALTER FUNCTION gpu_fault_control_state_lock_mode(text) SECURITY INVOKER",
            "SELECT pg_get_functiondef('gpu_fault_control_state_lock_mode(text)'::regprocedure)",
            "ownership or overload differs",
        ),
        (
            "ALTER TABLE gpu_fault_workflows DISABLE TRIGGER gpu_fault_workflows_fence",
            "ALTER TABLE gpu_fault_workflows ENABLE TRIGGER gpu_fault_workflows_fence",
            "SELECT tgenabled FROM pg_trigger WHERE tgname='gpu_fault_workflows_fence'",
            "triggers are disabled",
        ),
    ],
)
def test_business_startup_refuses_state_schema_drift_without_repairing_it(
    migration_database, mutation: str, restore: str, inspection: str, message: str
) -> None:
    connection = migration_database
    connection.execute(mutation)
    changed = connection.execute(inspection).fetchone()
    try:
        assert POSTGRES_URL is not None
        with pytest.raises(RuntimeError, match=message):
            PostgresStore(POSTGRES_URL, initialize_schema=False)
        assert connection.execute(inspection).fetchone() == changed, (
            "business startup repaired schema outside the ensure Job"
        )
    finally:
        connection.execute(restore)


def test_business_startup_rejects_a_view_that_hides_dedicated_records(
    migration_database,
) -> None:
    connection = migration_database
    inspection = "SELECT pg_get_viewdef('gpu_fault_control_records'::regclass, true)"
    original = connection.execute(inspection).fetchone()[0]
    connection.execute(
        "CREATE OR REPLACE VIEW gpu_fault_control_records AS "
        "SELECT kind,key,payload FROM gpu_fault_objects"
    )
    changed = connection.execute(inspection).fetchone()
    try:
        assert POSTGRES_URL is not None
        with pytest.raises(RuntimeError, match="control-state view.*differs"):
            PostgresStore(POSTGRES_URL, initialize_schema=False)
        assert connection.execute(inspection).fetchone() == changed, (
            "business startup rewrote the logical read view"
        )
    finally:
        connection.execute(
            "CREATE OR REPLACE VIEW gpu_fault_control_records AS " + original
        )


def test_ignored_old_insert_does_not_mirror_a_phantom_record(
    migration_database,
) -> None:
    connection = migration_database
    value = terminal_command("ignored-insert")
    connection.execute(
        "INSERT INTO gpu_fault_objects(kind,key,payload) VALUES ('remote_command',%s,%s::jsonb)",
        (value.command_id, value.model_dump_json()),
    )
    set_state_table_mode(connection, "remote_command", "dual", expected_mode="legacy")
    conflicting = value.model_copy(update={"status": RemoteCommandStatus.FAILED})
    connection.execute(
        "INSERT INTO gpu_fault_objects(kind,key,payload) VALUES ('remote_command',%s,%s::jsonb) "
        "ON CONFLICT(kind,key) DO NOTHING",
        (conflicting.command_id, conflicting.model_dump_json()),
    )
    assert (
        connection.execute("SELECT count(*) FROM gpu_fault_remote_commands").fetchone()[
            0
        ]
        == 0
    ), "an ignored insert mirrored its rejected candidate"
    assert connection.execute(
        "SELECT payload FROM gpu_fault_remote_command_records WHERE key=%s",
        (value.command_id,),
    ).fetchone()[0] == value.model_dump(mode="json")


@pytest.mark.parametrize("kind", ["remote_command", "workflow"])
def test_state_table_cli_performs_each_explicit_migration_stage(
    migration_database, monkeypatch, capsys, kind: str
) -> None:
    from gpu_fault import store_migrate

    assert POSTGRES_URL is not None
    monkeypatch.setenv("GPU_FAULT_STORE_URL", POSTGRES_URL)

    def run(*arguments: str) -> dict:
        monkeypatch.setattr(
            "sys.argv",
            ["gpu-fault-store-migrate", "--state-table-kind", kind, *arguments],
        )
        store_migrate.main()
        return json.loads(capsys.readouterr().out)

    assert run("--state-table-status")["mode"] == "legacy"
    assert (
        run("--set-state-table-mode", "dual", "--expected-state-table-mode", "legacy")[
            "mode"
        ]
        == "dual"
    )
    backfill = run("--backfill-state-table")
    assert backfill["status"]["verified"] is True
    assert backfill["status"]["backfill_complete"] is True
    with pytest.raises(SystemExit, match="explicit confirmation"):
        run(
            "--set-state-table-mode", "dedicated", "--expected-state-table-mode", "dual"
        )
    assert run("--state-table-status")["mode"] == "dual"
    assert (
        run(
            "--set-state-table-mode",
            "dedicated",
            "--expected-state-table-mode",
            "dual",
            "--confirm-state-table-change",
            "DEDICATED",
        )["mode"]
        == "dedicated"
    )
    retired = run(
        "--purge-legacy-state-table", "--confirm-state-table-change", "PURGE_LEGACY"
    )
    assert retired["status"]["legacy_purged"] is True
    assert retired["retired_indexes"], "CLI retirement left all legacy indexes active"


def test_online_index_builder_cannot_race_legacy_retirement(migration_database) -> None:
    import psycopg

    from gpu_fault.store.postgres.index_builder import (
        build_missing_indexes_concurrently,
    )

    connection = migration_database
    activate_empty_dedicated(connection)
    assert POSTGRES_URL is not None
    with psycopg.connect(POSTGRES_URL, autocommit=True) as builder:
        connection.execute(
            "SELECT pg_advisory_lock_shared(hashtextextended('gpu_fault_schema_bootstrap', 0))"
        )
        try:
            with pytest.raises(RuntimeError, match="maintenance.*retry"):
                build_missing_indexes_concurrently(builder)
            retired = purge_legacy_state(connection, "remote_command", confirm=True)
        finally:
            connection.execute(
                "SELECT pg_advisory_unlock_shared(hashtextextended('gpu_fault_schema_bootstrap', 0))"
            )
        result = build_missing_indexes_concurrently(builder)
    assert not set(retired["retired_indexes"]) & set(result["built"]), (
        "the builder used a pre-retirement required-index inventory"
    )
    assert (
        connection.execute(
            "SELECT indexname FROM pg_indexes WHERE indexname=ANY(%s)",
            (retired["retired_indexes"],),
        ).fetchall()
        == []
    ), "a concurrent builder recreated a retired legacy index"


@pytest.mark.parametrize("kind", ["remote_command", "workflow"])
def test_ensure_does_not_reseed_a_lost_migration_mode(migration_database, kind: str):
    connection = migration_database
    connection.execute(
        "DELETE FROM gpu_fault_control_state_modes WHERE kind=%s", (kind,)
    )
    try:
        assert POSTGRES_URL is not None
        with pytest.raises(RuntimeError, match="migration metadata is missing"):
            PostgresStore(POSTGRES_URL).close()
        assert (
            connection.execute(
                "SELECT mode FROM gpu_fault_control_state_modes WHERE kind=%s", (kind,)
            ).fetchone()
            is None
        ), "ensure silently reset a lost migration mode to legacy"
    finally:
        connection.execute(
            "INSERT INTO gpu_fault_control_state_modes(kind) VALUES (%s)", (kind,)
        )


def test_ensure_refuses_a_lost_mode_registry_when_native_tables_exist(
    migration_database,
):
    connection = migration_database
    connection.execute(
        "ALTER TABLE gpu_fault_control_state_modes RENAME TO gf_test_modes_backup"
    )
    try:
        assert POSTGRES_URL is not None
        with pytest.raises(RuntimeError, match="migration metadata is missing"):
            PostgresStore(POSTGRES_URL).close()
        assert (
            connection.execute(
                "SELECT to_regclass('gpu_fault_control_state_modes')"
            ).fetchone()[0]
            is None
        ), "ensure invented a new mode registry beside existing state"
    finally:
        connection.execute(
            "ALTER TABLE gf_test_modes_backup RENAME TO gpu_fault_control_state_modes"
        )


@pytest.mark.parametrize(
    ("kind", "table"),
    [
        ("remote_command", "gpu_fault_remote_commands"),
        ("workflow", "gpu_fault_workflows"),
    ],
)
def test_ensure_does_not_replace_a_missing_authoritative_table_with_an_empty_one(
    migration_database, kind: str, table: str
):
    connection = migration_database
    set_state_table_mode(connection, kind, "dual", expected_mode="legacy")
    backfill_state_table(connection, kind)
    set_state_table_mode(
        connection, kind, "dedicated", expected_mode="dual", confirm_dedicated=True
    )
    connection.execute(f"ALTER TABLE {table} RENAME TO gf_test_state_backup")
    try:
        assert POSTGRES_URL is not None
        with pytest.raises(
            RuntimeError, match="dedicated control-state table is missing"
        ):
            PostgresStore(POSTGRES_URL).close()
        assert (
            connection.execute("SELECT to_regclass(%s)", (table,)).fetchone()[0] is None
        ), "ensure replaced a missing dedicated authority with an empty table"
    finally:
        connection.execute(f"ALTER TABLE gf_test_state_backup RENAME TO {table}")
