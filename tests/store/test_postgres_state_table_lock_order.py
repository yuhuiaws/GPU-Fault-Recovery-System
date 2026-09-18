"""Mode activation must coexist with Store readers and older SQL writers."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import timedelta
from queue import Queue
from threading import Event

import pytest
from pydantic import BaseModel

from gpu_fault.models import WorkflowStatus
from gpu_fault.schema_migrations import LATEST_POSTGRES_SCHEMA_VERSION
from gpu_fault.state_table_migrate import (
    StateTableMigrationError,
    backfill_state_table,
    set_state_table_mode,
    state_table_status,
)
from gpu_fault.store import PostgresStore
from gpu_fault.store.postgres.state_table_payload import STATE_LAYOUTS
from gpu_fault.store.postgres.state_table_storage import get_state_payload
from tests.store import test_postgres_state_tables as state_tables

database = state_tables.database
migration_database = state_tables.migration_database
POSTGRES_URL = state_tables.POSTGRES_URL
pytestmark = pytest.mark.skipif(
    not POSTGRES_URL, reason="requires an isolated PostgreSQL test database"
)


def record(kind: str, key: str):
    value = state_tables.terminal_command(key)
    if kind == "workflow":
        value = value.workflow.model_copy(
            update={"request_id": key, "status": WorkflowStatus.SUCCEEDED}
        )
    return value


def stale_copy(connection, kind: str):
    current = record(kind, "retained")
    orphan = record(kind, "orphan")
    set_state_table_mode(connection, kind, "dual", expected_mode="legacy")
    for value in (current, orphan):
        connection.execute(
            "INSERT INTO gpu_fault_objects(kind,key,payload) VALUES (%s,%s,%s::jsonb)",
            (
                kind,
                getattr(value, STATE_LAYOUTS[kind].key_field),
                value.model_dump_json(),
            ),
        )
    set_state_table_mode(connection, kind, "legacy", expected_mode="dual")
    connection.execute(
        "DELETE FROM gpu_fault_objects WHERE kind=%s AND key='orphan'", (kind,)
    )
    updated = current.model_copy(
        update={"updated_at": current.updated_at + timedelta(hours=1)}
    )
    connection.execute(
        "UPDATE gpu_fault_objects SET payload=%s::jsonb WHERE kind=%s AND key='retained'",
        (updated.model_dump_json(), kind),
    )
    return updated


def lock_modes(connection) -> None:
    connection.execute(
        "SELECT pg_advisory_xact_lock_shared(hashtextextended('gpu_fault_schema_bootstrap',0))"
    )
    for kind in sorted(STATE_LAYOUTS):
        connection.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended(%s,0))",
            (f"gpu_fault_control_state/{kind}",),
        )


def assert_view_lock(connection, pid: int, kind: str) -> None:
    assert connection.execute(
        "SELECT EXISTS(SELECT 1 FROM pg_locks WHERE pid=%s "
        "AND relation=to_regclass(%s) AND mode='AccessShareLock' AND granted)",
        (pid, STATE_LAYOUTS[kind].table),
    ).fetchone()[0], (
        "the read-before-write transaction must retain the native view relation lock"
    )


def wait_for_mode_fence(connection, pid: int) -> None:
    deadline = time.monotonic() + 5
    while not connection.execute(
        "SELECT pg_backend_pid()=ANY(pg_blocking_pids(%s))", (pid,)
    ).fetchone()[0]:
        assert time.monotonic() < deadline, "writer never reached the held mode fence"
        time.sleep(0.01)


class PausedCommandStore(PostgresStore):
    def pause_before_command_write(self, pids: Queue[int], proceed: Event) -> None:
        self.write_pids = pids
        self.write_proceed = proceed

    def _put(
        self,
        kind: str,
        key: str,
        value: BaseModel,
        *,
        expected: BaseModel | None = None,
    ) -> None:
        if kind == "remote_command":
            with self._db.cursor() as cursor:
                cursor.execute("SELECT pg_backend_pid()")
                self.write_pids.put(cursor.fetchone()[0])
            assert self.write_proceed.wait(5), "test did not release the command writer"
        super()._put(kind, key, value, expected=expected)


def test_ensure_remote_command_read_does_not_deadlock_legacy_to_dual(
    migration_database,
) -> None:
    connection = migration_database
    previous = stale_copy(connection, "remote_command")
    assert POSTGRES_URL is not None
    with closing(
        PausedCommandStore(
            POSTGRES_URL, initialize_schema=False, pool_min_size=1, pool_max_size=1
        )
    ) as store:
        pids: Queue[int] = Queue()
        proceed = Event()
        store.pause_before_command_write(pids, proceed)
        value = record("remote_command", "new-command")
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(store.ensure_remote_command, value)
            try:
                pid = pids.get(timeout=5)
                assert_view_lock(connection, pid, "remote_command")
                with connection.transaction():
                    lock_modes(connection)
                    proceed.set()
                    wait_for_mode_fence(connection, pid)
                    switched = set_state_table_mode(
                        connection, "remote_command", "dual", expected_mode="legacy"
                    )
                    assert switched["dedicated_rows"] == 0, (
                        "a cancelled migration's stale copy must not survive activation"
                    )
                assert pending.result(timeout=5) == value
            finally:
                proceed.set()
        assert store.get_remote_command(previous.command_id) == previous
        assert store.get_remote_command(value.command_id) == value
    status = backfill_state_table(connection, "remote_command")["status"]
    assert status["verified"] is True
    assert status["legacy_rows"] == status["dedicated_rows"] == 2


@pytest.mark.parametrize("kind", ["remote_command", "workflow"])
@pytest.mark.parametrize(
    ("isolation", "sqlstate"),
    [("READ COMMITTED", None), ("REPEATABLE READ", "40001"), ("SERIALIZABLE", "40001")],
)
def test_read_before_legacy_sql_write_preserves_atomic_mode_activation(
    migration_database, kind: str, isolation: str, sqlstate: str | None
) -> None:
    import psycopg

    connection = migration_database
    previous = stale_copy(connection, kind)
    changed = previous.model_copy(
        update={"updated_at": previous.updated_at + timedelta(hours=1)}
    )
    pids: Queue[int] = Queue()
    proceed = Event()

    def old_writer():
        assert POSTGRES_URL is not None
        try:
            with psycopg.connect(POSTGRES_URL) as writer:
                writer.execute(f"SET TRANSACTION ISOLATION LEVEL {isolation}")
                writer.execute("SET LOCAL statement_timeout='10s'")
                assert get_state_payload(
                    writer, kind, "retained"
                ) == previous.model_dump(mode="json")
                pids.put(writer.info.backend_pid)
                assert proceed.wait(5), "test did not release the legacy writer"
                writer.execute(
                    "UPDATE gpu_fault_objects SET payload=%s::jsonb WHERE kind=%s AND key='retained'",
                    (changed.model_dump_json(), kind),
                )
        except psycopg.Error as error:
            return error.sqlstate
        return None

    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(old_writer)
        try:
            pid = pids.get(timeout=5)
            assert_view_lock(connection, pid, kind)
            with connection.transaction():
                lock_modes(connection)
                proceed.set()
                wait_for_mode_fence(connection, pid)
                switched = set_state_table_mode(
                    connection, kind, "dual", expected_mode="legacy"
                )
                assert switched["dedicated_rows"] == 0, "stale copies survived reset"
            assert pending.result(timeout=5) == sqlstate, (
                "old writers must mirror the new mode or reject their stale snapshot, never deadlock"
            )
        finally:
            proceed.set()
    assert get_state_payload(connection, kind, "retained") == (
        changed if sqlstate is None else previous
    ).model_dump(mode="json")
    checked = backfill_state_table(connection, kind)["status"]
    assert checked["verified"] is True
    assert checked["legacy_rows"] == checked["dedicated_rows"] == 1


@pytest.mark.parametrize("kind", ["remote_command", "workflow"])
def test_locked_native_copy_aborts_reset_atomically_and_can_be_retried(
    migration_database, kind: str
) -> None:
    import psycopg
    from psycopg import sql

    connection = migration_database
    stale_copy(connection, kind)
    before = state_table_status(connection, kind, verify=False)
    layout = STATE_LAYOUTS[kind]
    assert POSTGRES_URL is not None
    with psycopg.connect(POSTGRES_URL) as locker:
        locker.execute(
            sql.SQL("SELECT {} FROM {} WHERE {}='retained' FOR UPDATE").format(
                sql.Identifier(layout.key_field),
                sql.Identifier(layout.table),
                sql.Identifier(layout.key_field),
            )
        )
        with pytest.raises(StateTableMigrationError, match="locked rows; retry"):
            set_state_table_mode(connection, kind, "dual", expected_mode="legacy")
        assert state_table_status(connection, kind, verify=False) == before, (
            "a skipped row must roll back every deletion and mode/checkpoint change"
        )
        assert connection.execute(
            "SELECT current_setting('gpu_fault.control_state_reset', true)"
        ).fetchone()[0] in (None, ""), (
            "the reset permission leaked outside its transaction"
        )
    switched = set_state_table_mode(connection, kind, "dual", expected_mode="legacy")
    assert switched["dedicated_rows"] == 0
    checked = backfill_state_table(connection, kind)["status"]
    assert checked["verified"] is True
    assert checked["dedicated_rows"] == 1


@pytest.mark.parametrize("busy_kind", ["remote_command", "workflow"])
def test_mode_change_does_not_queue_ahead_of_legacy_tuple_writers(
    migration_database, busy_kind: str
) -> None:
    import psycopg

    connection = migration_database
    stale_copy(connection, "remote_command")
    before = state_table_status(connection, "remote_command", verify=False)
    assert POSTGRES_URL is not None
    with psycopg.connect(POSTGRES_URL) as writer:
        writer.execute(
            "SELECT pg_advisory_xact_lock_shared(hashtextextended(%s,0))",
            (f"gpu_fault_control_state/{busy_kind}",),
        )
        with pytest.raises(StateTableMigrationError, match="writers are busy; retry"):
            set_state_table_mode(
                connection, "remote_command", "dual", expected_mode="legacy"
            )
        assert state_table_status(connection, "remote_command", verify=False) == before
        # Even a refusal on the second fence must release the first one.
        with writer.transaction():
            writer.execute("SET LOCAL lock_timeout='1s'")
            writer.execute(
                "UPDATE gpu_fault_objects SET payload=payload "
                "WHERE kind='remote_command' AND key='retained'"
            )
    switched = set_state_table_mode(
        connection, "remote_command", "dual", expected_mode="legacy"
    )
    assert switched["dedicated_rows"] == 0


@pytest.mark.parametrize("kind", ["remote_command", "workflow"])
def test_v18_schema_upgrade_preserves_modes_checkpoints_and_both_copies(
    migration_database, kind: str
) -> None:
    connection = migration_database
    current = stale_copy(connection, kind)
    before = state_table_status(connection, kind, verify=False)
    history = connection.execute(
        "SELECT version,name,checksum FROM gpu_fault_schema_migrations WHERE version<=17 ORDER BY version"
    ).fetchall()
    with connection.transaction():
        connection.execute("DELETE FROM gpu_fault_schema_migrations WHERE version=18")
        connection.execute("UPDATE gpu_fault_schema_version SET version=17")
        connection.execute(
            """
            CREATE OR REPLACE FUNCTION gpu_fault_native_control_state_fence()
            RETURNS trigger LANGUAGE plpgsql AS $$
            DECLARE current_mode TEXT; expected_table TEXT;
            BEGIN
                SELECT table_name INTO expected_table
                FROM gpu_fault_control_state_descriptor(TG_ARGV[0]);
                IF expected_table IS DISTINCT FROM TG_TABLE_NAME THEN
                    RAISE EXCEPTION 'control-state trigger identity differs'
                        USING ERRCODE='55000';
                END IF;
                current_mode := gpu_fault_control_state_lock_mode(TG_ARGV[0]);
                IF current_mode='legacy' OR
                   (current_mode='dual' AND pg_trigger_depth()<2) THEN
                    RAISE EXCEPTION 'dedicated control-state write is not authoritative'
                        USING ERRCODE='55000';
                END IF;
                RETURN CASE WHEN TG_OP='DELETE' THEN OLD ELSE NEW END;
            END
            $$
            """
        )
    assert POSTGRES_URL is not None
    initialized = PostgresStore(POSTGRES_URL)
    initialized.close()
    assert connection.execute(
        "SELECT version FROM gpu_fault_schema_version WHERE singleton"
    ).fetchone() == (LATEST_POSTGRES_SCHEMA_VERSION,)
    assert (
        connection.execute(
            "SELECT version,name,checksum FROM gpu_fault_schema_migrations WHERE version<=17 ORDER BY version"
        ).fetchall()
        == history
    ), "schema ensure rewrote historical migrations"
    assert state_table_status(connection, kind, verify=False) == before, (
        "schema ensure activated, reset, or backfilled data without explicit maintenance"
    )
    assert get_state_payload(connection, kind, "retained") == current.model_dump(
        mode="json"
    )
    activated = set_state_table_mode(connection, kind, "dual", expected_mode="legacy")
    assert activated["dedicated_rows"] == 0, (
        "ensure did not install the v18 reset fence"
    )


@pytest.mark.parametrize("kind", ["remote_command", "workflow"])
@pytest.mark.parametrize("mode", ["legacy", "dual"])
@pytest.mark.parametrize("operation", ["INSERT", "UPDATE", "DELETE"])
def test_reset_marker_does_not_bypass_native_authority(
    migration_database, kind: str, mode: str, operation: str
) -> None:
    import psycopg
    from psycopg import sql

    connection = migration_database
    stale_copy(connection, kind)
    if mode == "dual":
        set_state_table_mode(connection, kind, mode, expected_mode="legacy")
        backfill_state_table(connection, kind)
    layout = STATE_LAYOUTS[kind]
    with connection.transaction(force_rollback=True):
        # A matching marker only allows DELETE while the native copy is not
        # authoritative; even that exception cannot cross to another kind.
        marker = (
            ("workflow" if kind == "remote_command" else "remote_command")
            if (mode == "legacy" and operation == "DELETE")
            else kind
        )
        connection.execute(
            "SELECT set_config('gpu_fault.control_state_reset', %s, true)", (marker,)
        )
        with pytest.raises(psycopg.Error, match="not authoritative"):
            with connection.transaction():
                if operation == "INSERT":
                    connection.execute(
                        sql.SQL("INSERT INTO {} SELECT ({}(%s::jsonb)).*").format(
                            sql.Identifier(layout.table),
                            sql.Identifier(f"gpu_fault_{kind}_columns"),
                        ),
                        (record(kind, "rejected").model_dump_json(),),
                    )
                else:
                    statement = (
                        sql.SQL("DELETE FROM {} WHERE {}='retained'").format(
                            sql.Identifier(layout.table),
                            sql.Identifier(layout.key_field),
                        )
                        if operation == "DELETE"
                        else sql.SQL("UPDATE {} SET {}={} WHERE {}='retained'").format(
                            sql.Identifier(layout.table),
                            sql.Identifier(layout.payload_column),
                            sql.Identifier(layout.payload_column),
                            sql.Identifier(layout.key_field),
                        )
                    )
                    connection.execute(statement)
        assert get_state_payload(connection, kind, "retained") is not None
