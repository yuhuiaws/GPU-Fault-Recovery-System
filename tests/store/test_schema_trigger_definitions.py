"""The schema check verifies the queue counter triggers beyond ``tgenabled``.

FINAL-建议汇总 F-D10 (P1-75D, P1-75F). A counter trigger that exists and is
enabled can still be wrong: recreated ``FOR EACH ROW`` without its transition
table, bound to the other counter function, or driving a function whose body
was replaced. Each one drifts the admission counters silently, and the
``IF NOT EXISTS`` in the DDL means ``--ensure-schema`` never repairs a trigger
that is present. The check compares every one of the six counter triggers and
both counter functions with what the DDL declares.
"""

from __future__ import annotations

import os

import pytest

from gpu_fault.store import PostgresStore
from tests.store._postgres_processor_claim_support import _truncate

POSTGRES_URL = os.getenv("GPU_FAULT_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(
    not POSTGRES_URL, reason="GPU_FAULT_TEST_POSTGRES_URL is not configured"
)

QUEUE = "gpu_fault_processor_queue"
PRIORITY_SYNC = "gpu_fault_processor_priority_count_sync"
LEGACY_SYNC = "gpu_fault_processor_queue_count_sync"


@pytest.fixture(autouse=True)
def pristine_schema():
    assert POSTGRES_URL is not None
    PostgresStore(POSTGRES_URL).close()
    _truncate()
    yield
    _truncate()


def _execute(*statements: str) -> None:
    import psycopg

    assert POSTGRES_URL is not None
    with psycopg.connect(POSTGRES_URL, autocommit=True) as connection:
        with connection.cursor() as cursor:
            for statement in statements:
                cursor.execute(statement)


def _restore() -> None:
    """Drop whatever this test left behind and let the DDL recreate it."""

    assert POSTGRES_URL is not None
    _execute(
        f"DROP TRIGGER IF EXISTS gpu_fault_processor_priority_count_insert ON {QUEUE}",
        f"DROP TRIGGER IF EXISTS gpu_fault_processor_priority_count_delete ON {QUEUE}",
        f"DROP TRIGGER IF EXISTS gpu_fault_processor_queue_count_update ON {QUEUE}",
    )
    PostgresStore(POSTGRES_URL, initialize_schema=True).close()


def _validate() -> None:
    assert POSTGRES_URL is not None
    PostgresStore(POSTGRES_URL, initialize_schema=False).close()


def test_a_pristine_schema_passes_the_trigger_definition_check() -> None:
    _validate()


def test_a_counter_trigger_recreated_per_row_is_rejected() -> None:
    _execute(
        f"DROP TRIGGER gpu_fault_processor_priority_count_insert ON {QUEUE}",
        f"""
        CREATE TRIGGER gpu_fault_processor_priority_count_insert
        AFTER INSERT ON {QUEUE}
        FOR EACH ROW EXECUTE FUNCTION {PRIORITY_SYNC}()
        """,
    )
    try:
        with pytest.raises(
            RuntimeError, match="gpu_fault_processor_priority_count_insert"
        ):
            _validate()
    finally:
        _restore()


def test_a_counter_trigger_bound_to_the_other_counter_function_is_rejected() -> None:
    _execute(
        f"DROP TRIGGER gpu_fault_processor_priority_count_delete ON {QUEUE}",
        f"""
        CREATE TRIGGER gpu_fault_processor_priority_count_delete
        AFTER DELETE ON {QUEUE}
        REFERENCING OLD TABLE AS removed
        FOR EACH STATEMENT EXECUTE FUNCTION {LEGACY_SYNC}()
        """,
    )
    try:
        with pytest.raises(
            RuntimeError, match="gpu_fault_processor_priority_count_delete"
        ):
            _validate()
    finally:
        _restore()


def test_a_missing_legacy_counter_trigger_is_rejected() -> None:
    _execute(f"DROP TRIGGER gpu_fault_processor_queue_count_update ON {QUEUE}")
    try:
        with pytest.raises(
            RuntimeError, match="gpu_fault_processor_queue_count_update"
        ):
            _validate()
    finally:
        _restore()


def test_a_replaced_counter_function_body_is_rejected() -> None:
    _execute(
        f"""
        CREATE OR REPLACE FUNCTION {PRIORITY_SYNC}()
        RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
            RETURN NULL;
        END
        $$
        """
    )
    try:
        with pytest.raises(RuntimeError, match=PRIORITY_SYNC):
            _validate()
    finally:
        _restore()
