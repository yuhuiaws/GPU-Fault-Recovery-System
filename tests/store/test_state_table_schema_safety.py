from __future__ import annotations

import os
from collections.abc import Iterator
from uuid import uuid4

import pytest

from gpu_fault.store import PostgresStore
from gpu_fault.store.postgres.state_table_schema import validate_state_table_schema
from tests.store._postgres_processor_claim_support import _truncate

POSTGRES_URL = os.environ.get("GPU_FAULT_TEST_POSTGRES_URL", "")
pytestmark = pytest.mark.skipif(
    not POSTGRES_URL, reason="requires an isolated PostgreSQL test database"
)


@pytest.fixture(scope="module")
def database() -> Iterator[object]:
    import psycopg

    _truncate()
    store = PostgresStore(POSTGRES_URL, pool_min_size=1, pool_max_size=1)
    store.close()
    with psycopg.connect(POSTGRES_URL, autocommit=True) as connection:
        yield connection


@pytest.mark.parametrize(
    ("table", "trigger", "function", "arguments"),
    [
        (
            "gpu_fault_objects",
            "gpu_fault_objects_control_state_fence_trigger",
            "gpu_fault_objects_control_state_fence",
            "",
        ),
        (
            "gpu_fault_remote_commands",
            "gpu_fault_remote_commands_fence",
            "gpu_fault_native_control_state_fence",
            "'remote_command'",
        ),
        (
            "gpu_fault_workflows",
            "gpu_fault_workflows_fence",
            "gpu_fault_native_control_state_fence",
            "'workflow'",
        ),
    ],
)
def test_other_schema_trigger_cannot_hide_a_disabled_local_fence(
    database, table: str, trigger: str, function: str, arguments: str
) -> None:
    from psycopg import sql

    namespace = "state_shadow_" + uuid4().hex
    schema = database.execute("SELECT current_schema()").fetchone()[0]
    with database.transaction(force_rollback=True):
        database.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(namespace)))
        database.execute(
            sql.SQL("CREATE TABLE {}.{} (kind text, key text, payload jsonb)").format(
                sql.Identifier(namespace), sql.Identifier(table)
            )
        )
        database.execute(
            sql.SQL(
                "CREATE TRIGGER {} BEFORE INSERT ON {}.{} "
                "FOR EACH ROW EXECUTE FUNCTION {}.{}({})"
            ).format(
                sql.Identifier(trigger),
                sql.Identifier(namespace),
                sql.Identifier(table),
                sql.Identifier(schema),
                sql.Identifier(function),
                sql.SQL(arguments),
            )
        )
        database.execute(
            sql.SQL("ALTER TABLE {}.{} DISABLE TRIGGER {}").format(
                sql.Identifier(schema), sql.Identifier(table), sql.Identifier(trigger)
            )
        )
        with pytest.raises(RuntimeError, match="missing or disabled"):
            validate_state_table_schema(database)


@pytest.mark.parametrize(
    ("name", "signature"),
    [
        ("gpu_fault_put_control_state", "text,text,jsonb,jsonb,boolean"),
        ("gpu_fault_put_native_control_state", "text,text,jsonb,jsonb,boolean"),
        ("gpu_fault_patch_control_state", "text,text,jsonb"),
    ],
)
def test_state_writer_return_type_drift_is_not_a_valid_schema(
    database, name: str, signature: str
) -> None:
    from psycopg import sql

    schema = database.execute("SELECT current_schema()").fetchone()[0]
    original = database.execute(
        "SELECT pg_get_functiondef(to_regprocedure(%s))", (f"{name}({signature})",)
    ).fetchone()[0]
    assert "RETURNS boolean" in original, "fixture must select a boolean writer"
    changed = original.replace("RETURNS boolean", "RETURNS text", 1)
    with database.transaction(force_rollback=True):
        database.execute(
            sql.SQL("DROP FUNCTION {}.{}({})").format(
                sql.Identifier(schema), sql.Identifier(name), sql.SQL(signature)
            )
        )
        database.execute(changed)
        with pytest.raises(RuntimeError, match="function definitions differ"):
            validate_state_table_schema(database)


@pytest.mark.parametrize("attribute", ["STRICT", "PARALLEL SAFE"])
def test_state_writer_execution_attributes_must_match_the_release(
    database, attribute: str
) -> None:
    from psycopg import sql

    with database.transaction(force_rollback=True):
        database.execute(
            sql.SQL(
                "ALTER FUNCTION gpu_fault_put_control_state(text,text,jsonb,jsonb,boolean) {}"
            ).format(sql.SQL(attribute))
        )
        with pytest.raises(RuntimeError, match="function definitions differ"):
            validate_state_table_schema(database)


def test_state_writer_defaults_must_match_the_release(database) -> None:
    original = database.execute(
        "SELECT pg_get_functiondef("
        "'gpu_fault_delete_control_state(text,text,jsonb,boolean)'::regprocedure)"
    ).fetchone()[0]
    assert "DEFAULT false" in original, (
        "fixture must select the guarded delete defaults"
    )
    with database.transaction(force_rollback=True):
        database.execute(original.replace("DEFAULT false", "DEFAULT true"))
        with pytest.raises(RuntimeError, match="function definitions differ"):
            validate_state_table_schema(database)
