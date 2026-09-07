"""A no-op ``--ensure-schema`` takes no table lock.

Store review 2026-09-07, item J. The idempotent DDL runs on every deploy and on
every process start that bootstraps; Postgres takes the lock before it finds
out that ``CREATE INDEX ... IF NOT EXISTS`` / ``ADD COLUMN IF NOT EXISTS`` /
``DROP TRIGGER`` + ``CREATE TRIGGER`` / ``ENABLE TRIGGER`` have nothing to do,
and holds it until commit. On an already-current database the whole run must
therefore leave nothing stronger than AccessShareLock behind, the counter seed
must not run, and the index declaration must still be exactly what the schema
check demands and what the online builder builds.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest

from gpu_fault.store import PostgresStore
from gpu_fault.store.postgres.ddl import create_postgres_schema, declared_index_names
from gpu_fault.store.postgres.index_builder import declared_index_statements
from tests.store._postgres_processor_claim_support import _truncate

POSTGRES_URL = os.getenv("GPU_FAULT_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(
    not POSTGRES_URL, reason="GPU_FAULT_TEST_POSTGRES_URL is not configured"
)
QUEUE = "gpu_fault_processor_queue"
NOTIFY_TRIGGER = "gpu_fault_processor_queue_notify_pending_trigger"
SPOOL_TRIGGER = "gpu_fault_telemetry_spool_notify_available_trigger"
SEEDED_AT = datetime(2026, 9, 1, tzinfo=timezone.utc)

RELATION_LOCKS_SQL = """
    SELECT l.mode, c.relname
    FROM pg_locks l JOIN pg_class c ON c.oid=l.relation
    WHERE l.pid=pg_backend_pid()
      AND l.locktype='relation'
      AND c.relname LIKE 'gpu\\_fault%'
    ORDER BY c.relname, l.mode
"""


@pytest.fixture(autouse=True)
def current_schema():
    assert POSTGRES_URL is not None
    PostgresStore(POSTGRES_URL).close()
    _truncate()
    yield
    _truncate()


def _connect(**kwargs):
    import psycopg

    assert POSTGRES_URL is not None
    return psycopg.connect(POSTGRES_URL, **kwargs)


def _seed_counters() -> None:
    """Counter rows as a live region has them, so the seed step has nothing
    to do and its LOCK TABLE / TRUNCATE must not run."""

    with _connect(autocommit=True) as connection:
        with connection.cursor() as cursor:
            # ``_truncate`` empties the mode singleton too; a live database
            # always has it.
            cursor.execute(
                """
                INSERT INTO gpu_fault_processor_counter_mode(
                    singleton, mode, updated_at
                ) VALUES (TRUE, 'dual', %s)
                ON CONFLICT(singleton) DO NOTHING
                """,
                (SEEDED_AT,),
            )
            cursor.execute(
                """
                INSERT INTO gpu_fault_processor_queue_counts(
                    cluster_id, incomplete_count, updated_at
                ) VALUES ('cluster-a', 0, %s)
                """,
                (SEEDED_AT,),
            )
            cursor.execute(
                """
                INSERT INTO gpu_fault_processor_priority_count_shards(
                    cluster_id, priority_bucket, shard_id, incomplete_count,
                    updated_at
                ) VALUES ('cluster-a', 0, 0, 0, %s)
                """,
                (SEEDED_AT,),
            )


def test_a_no_op_ensure_schema_holds_only_access_share_locks() -> None:
    _seed_counters()

    with _connect() as connection:
        with connection.cursor() as cursor:
            create_postgres_schema(cursor)
            cursor.execute(RELATION_LOCKS_SQL)
            locks = cursor.fetchall()
        connection.commit()

    assert locks, "expected the catalog reads to show up as relation locks"
    assert {mode for mode, _ in locks} == {"AccessShareLock"}, locks
    with _connect(autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT updated_at FROM gpu_fault_processor_queue_counts"
                " WHERE cluster_id='cluster-a'"
            )
            counts_updated_at = cursor.fetchone()[0]
            cursor.execute(
                "SELECT updated_at FROM gpu_fault_processor_priority_count_shards"
                " WHERE cluster_id='cluster-a'"
            )
            shards_updated_at = cursor.fetchone()[0]
    # The seed step rewrites the counter rows with now(); untouched means
    # it did not run.
    assert counts_updated_at == SEEDED_AT
    assert shards_updated_at == SEEDED_AT


def test_a_second_bootstrap_does_not_recreate_a_matching_trigger() -> None:
    with _connect(autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT t.oid FROM pg_trigger t WHERE t.tgname = ANY(%s)"
                " ORDER BY t.tgname",
                ([NOTIFY_TRIGGER, SPOOL_TRIGGER],),
            )
            before = [row[0] for row in cursor.fetchall()]

    assert POSTGRES_URL is not None
    PostgresStore(POSTGRES_URL, initialize_schema=True).close()

    with _connect(autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT t.oid FROM pg_trigger t WHERE t.tgname = ANY(%s)"
                " ORDER BY t.tgname",
                ([NOTIFY_TRIGGER, SPOOL_TRIGGER],),
            )
            after = [row[0] for row in cursor.fetchall()]
    # A DROP + CREATE allocates a new oid; the same oid means the trigger
    # was left alone.
    assert after == before


def test_a_drifted_notify_trigger_is_recreated_to_the_declared_definition() -> None:
    with _connect(autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute(f"DROP TRIGGER {NOTIFY_TRIGGER} ON {QUEUE}")
            # Same name, narrower event list: present but wrong.
            cursor.execute(
                f"""
                CREATE TRIGGER {NOTIFY_TRIGGER}
                AFTER INSERT ON {QUEUE}
                FOR EACH ROW
                EXECUTE FUNCTION gpu_fault_processor_queue_notify_pending()
                """
            )

    assert POSTGRES_URL is not None
    PostgresStore(POSTGRES_URL, initialize_schema=True).close()

    with _connect(autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT pg_get_triggerdef(oid) FROM pg_trigger WHERE tgname=%s",
                (NOTIFY_TRIGGER,),
            )
            definition = cursor.fetchone()[0]

    assert "AFTER INSERT OR UPDATE OF status" in definition, definition


def test_a_disabled_counter_trigger_is_re_enabled_by_the_bootstrap() -> None:
    name = "gpu_fault_processor_priority_count_insert"
    with _connect(autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute(f"ALTER TABLE {QUEUE} DISABLE TRIGGER {name}")

    assert POSTGRES_URL is not None
    PostgresStore(POSTGRES_URL, initialize_schema=True).close()

    with _connect(autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT tgenabled FROM pg_trigger WHERE tgname=%s", (name,))
            enabled = cursor.fetchone()[0]

    assert enabled == "O"


def test_declared_indexes_match_the_statements_and_the_database() -> None:
    """The two source scrapers see the same set, and after a bootstrap that
    set is exactly the non-constraint gpu_fault indexes in the database --
    so routing every declaration through ``_declare_index`` changed how the
    DDL runs, not what it declares."""

    names = set(declared_index_names())

    assert names == set(declared_index_statements())
    with _connect(autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT c.relname
                FROM pg_class c
                JOIN pg_index i ON i.indexrelid=c.oid
                WHERE c.relname LIKE 'gpu\\_fault%'
                  AND NOT EXISTS (
                      SELECT 1 FROM pg_constraint k WHERE k.conindid=c.oid
                  )
                """
            )
            present = {row[0] for row in cursor.fetchall()}

    assert present == names, sorted(present ^ names)


def test_a_dropped_index_is_redeclared_by_the_bootstrap() -> None:
    """The existence check must not turn ``_declare_index`` into a no-op for
    an index that is actually missing."""

    with _connect(autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute("DROP INDEX gpu_fault_workflow_status_count")

    assert POSTGRES_URL is not None
    PostgresStore(POSTGRES_URL, initialize_schema=True).close()

    with _connect(autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT to_regclass('gpu_fault_workflow_status_count')")
            assert cursor.fetchone()[0] is not None, "index was not redeclared"
