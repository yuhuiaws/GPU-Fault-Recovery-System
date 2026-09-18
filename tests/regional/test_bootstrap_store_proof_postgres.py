"""Serial tests against a dedicated local PostgreSQL instance, never a site DB."""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Iterator

import pytest

from gpu_fault.schema_migrations import LATEST_POSTGRES_SCHEMA_VERSION
from gpu_fault.store import PostgresStore
from gpu_fault_release.regional_release_store_probe import (
    connection_arguments,
    inspect_database,
)

URL = os.environ.get("GPU_FAULT_TEST_POSTGRES_URL", "")
pytestmark = pytest.mark.skipif(
    not URL, reason="requires a dedicated local PostgreSQL test URL"
)


@pytest.fixture
def database() -> Iterator[str]:
    import psycopg
    from psycopg import sql
    from psycopg.conninfo import conninfo_to_dict, make_conninfo

    parsed = conninfo_to_dict(URL)
    assert parsed.get("host") == "127.0.0.1", (
        "these tests only accept a local dedicated database"
    )
    name = f"release_proof_{uuid.uuid4().hex}"
    with psycopg.connect(URL, autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    try:
        yield make_conninfo(URL, dbname=name)
    finally:
        with psycopg.connect(URL, autocommit=True) as connection:
            connection.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(name)))


def initialize(url: str) -> None:
    store = PostgresStore(url, pool_min_size=1, pool_max_size=1)
    store.close()


def inspect(url: str) -> dict:
    import psycopg

    with psycopg.connect(url, autocommit=True) as connection:
        return inspect_database(connection, LATEST_POSTGRES_SCHEMA_VERSION)


def test_new_database_is_proved_empty_by_a_fresh_read(database: str) -> None:
    proof = inspect(database)
    assert proof["database_state"] == "uninitialized_empty", (
        "fresh database was not inspected"
    )
    assert proof["safe"] is True, "a genuinely empty database was rejected"


def test_partial_schema_can_only_bootstrap_when_every_real_table_is_empty(
    database: str,
) -> None:
    import psycopg

    with psycopg.connect(database, autocommit=True) as connection:
        connection.execute(
            "CREATE TABLE gpu_fault_objects (kind text, key text, payload jsonb)"
        )
    assert inspect(database)["database_state"] == "uninitialized_empty", (
        "empty partial schema could not resume initialization"
    )
    with psycopg.connect(database, autocommit=True) as connection:
        connection.execute(
            "INSERT INTO gpu_fault_objects VALUES ('workflow','pending','{}')"
        )
    with pytest.raises(ValueError, match="incomplete schema contains rows"):
        inspect(database)


def test_unknown_schema_never_counts_as_empty(database: str) -> None:
    import psycopg

    with psycopg.connect(database, autocommit=True) as connection:
        connection.execute("CREATE TABLE unrelated (id integer)")
    with pytest.raises(ValueError, match="unrecognized database schema"):
        inspect(database)


@pytest.mark.parametrize("kind", ["VIEW", "MATERIALIZED VIEW"])
def test_uninitialized_views_are_refused_without_executing_them(
    database: str, kind: str
) -> None:
    import psycopg
    from psycopg import sql

    with psycopg.connect(database, autocommit=True) as connection:
        connection.execute(
            """CREATE FUNCTION public.forbidden_probe() RETURNS SETOF integer
               LANGUAGE plpgsql AS $$ BEGIN RAISE EXCEPTION 'view was executed'; END $$"""
        )
        suffix = " WITH NO DATA" if kind == "MATERIALIZED VIEW" else ""
        connection.execute(
            sql.SQL(
                "CREATE {} gpu_fault_unknown AS SELECT * FROM forbidden_probe(){}"
            ).format(sql.SQL(kind), sql.SQL(suffix))
        )
    with pytest.raises(ValueError, match="unrecognized views or tables"):
        inspect(database)


def test_prefix_alone_does_not_authorize_partial_table_read(database: str) -> None:
    import psycopg

    with psycopg.connect(database, autocommit=True) as connection:
        connection.execute("CREATE TABLE gpu_fault_unrecognized(id integer)")
    with pytest.raises(ValueError, match="unrecognized views or tables"):
        inspect(database)


def test_empty_schema_metadata_without_version_rows_is_a_partial_schema(
    database: str,
) -> None:
    import psycopg

    with psycopg.connect(database, autocommit=True) as connection:
        connection.execute(
            "CREATE TABLE gpu_fault_schema_version(singleton boolean, version integer)"
        )
        connection.execute(
            "CREATE TABLE gpu_fault_schema_migrations(version integer, name text, checksum text)"
        )
    assert inspect(database)["database_state"] == "uninitialized_empty", (
        "empty metadata tables were mistaken for an initialized schema"
    )


@pytest.mark.parametrize("status", ["PENDING", "RUNNING", "BLOCKED", "UNKNOWN"])
def test_initialized_database_rows_are_checked_on_every_retry(
    database: str, status: str
) -> None:
    import psycopg

    initialize(database)
    assert inspect(database)["safe"] is True, "initialized empty schema was rejected"
    with psycopg.connect(database, autocommit=True) as connection:
        connection.execute(
            "INSERT INTO gpu_fault_objects(kind,key,payload) VALUES ('workflow',%s,%s::jsonb)",
            (
                "workflow-proof",
                json.dumps({"request_id": "workflow-proof", "status": status}),
            ),
        )
    proof = inspect(database)
    assert proof["safe"] is False and proof["blockers"]["workflow"] == 1, (
        "a previous successful proof hid current workflow rows"
    )


def test_terminal_history_is_not_mistaken_for_an_empty_database(database: str) -> None:
    import psycopg

    initialize(database)
    with psycopg.connect(database, autocommit=True) as connection:
        connection.execute(
            "INSERT INTO gpu_fault_objects(kind,key,payload) VALUES ('workflow',%s,%s::jsonb)",
            (
                "workflow-proof",
                json.dumps({"request_id": "workflow-proof", "status": "SUCCEEDED"}),
            ),
        )
    proof = inspect(database)
    assert proof["database_state"] == "initialized" and proof["safe"] is True, (
        "terminal history was either ignored as an empty DB or incorrectly blocked"
    )


def test_migration_history_drift_fails_even_without_active_rows(database: str) -> None:
    import psycopg

    initialize(database)
    with psycopg.connect(database, autocommit=True) as connection:
        connection.execute(
            "UPDATE gpu_fault_schema_migrations SET checksum=%s WHERE version=1",
            ("0" * 64,),
        )
    with pytest.raises(ValueError, match="migration history differs"):
        inspect(database)


def test_probe_never_switches_its_session_to_read_write(database: str) -> None:
    import psycopg

    initialize(database)
    with psycopg.connect(
        database, autocommit=True, options="-c default_transaction_read_only=on"
    ) as connection:
        assert (
            inspect_database(connection, LATEST_POSTGRES_SCHEMA_VERSION)["safe"] is True
        ), "read-only inspection failed"
        assert connection.execute("SHOW default_transaction_read_only").fetchone() == (
            "on",
        ), "the proof widened the session to read-write"
        with pytest.raises(psycopg.errors.ReadOnlySqlTransaction):
            connection.execute("CREATE TABLE gpu_fault_forbidden(id integer)")


def test_connection_is_bound_to_database_user_host_and_verified_tls() -> None:
    identity = {
        "endpoint": "db.example",
        "port": 5432,
        "database": "gpu_fault",
        "username": "owner",
    }
    dsn = (
        "postgresql://owner:test-only@db.example:5432/gpu_fault"
        "?sslmode=verify-full&sslrootcert=/etc/gpu-fault/rds/ca-bundle.pem"
    )
    arguments = connection_arguments(dsn, identity)
    assert arguments["sslmode"] == "verify-full", "proof weakened authenticated TLS"
    assert "default_transaction_read_only=on" in arguments["options"], (
        "the connection can begin a read-write transaction"
    )
    for bad in (
        dsn.replace("db.example", "other.example"),
        dsn.replace("/gpu_fault?", "/other?"),
        dsn.replace("owner:", "other:"),
        dsn.replace("verify-full", "require"),
    ):
        with pytest.raises(ValueError, match="binding differs"):
            connection_arguments(bad, identity)
