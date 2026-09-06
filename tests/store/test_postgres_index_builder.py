"""Operators build declared indexes CONCURRENTLY and gate a release on a
read-only schema preflight, both from the migrate CLI.

FINAL F-J3 three-step method, operationalised (2026-09-06). The idempotent DDL
runs inside one transaction and therefore cannot ``CREATE INDEX
CONCURRENTLY``; left to the ensure Job, a new hot-table index is built with a
write lock on ``gpu_fault_objects`` / ``gpu_fault_processor_queue``. The
index builder does it online first, drops and rebuilds an invalid leftover,
and the preflight tells a FULL release whether the database is ready: schema
version, index presence and validity, and in-flight safety workflows the
``safety_only`` field cannot yet describe.
"""

from __future__ import annotations

import os

import pytest

from gpu_fault.schema_migrations import LATEST_POSTGRES_SCHEMA_VERSION
from gpu_fault.store import PostgresStore
from gpu_fault.store.postgres.ddl import declared_index_names
from gpu_fault.store.postgres.index_builder import (
    build_missing_indexes_concurrently,
    declared_index_statements,
    index_health,
    schema_preflight,
)
from tests.store._postgres_processor_claim_support import _truncate

POSTGRES_URL = os.getenv("GPU_FAULT_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(
    not POSTGRES_URL, reason="GPU_FAULT_TEST_POSTGRES_URL is not configured"
)
PROBE_INDEX = "gpu_fault_processor_queue_priority_claim"


@pytest.fixture
def connection():
    psycopg = pytest.importorskip("psycopg")
    assert POSTGRES_URL is not None
    PostgresStore(POSTGRES_URL).close()  # schema present
    _truncate()
    with psycopg.connect(POSTGRES_URL, autocommit=True) as conn:
        yield conn
        # Leave the schema whole for the next test.
        build_missing_indexes_concurrently(conn)
    _truncate()


def test_every_declared_index_has_a_statement():
    statements = declared_index_statements()
    assert set(statements) == set(declared_index_names())
    assert all(name in sql for name, sql in statements.items()), (
        "expected all(name in sql for name, sql in statements.items()) to be true"
    )
    assert all("CONCURRENTLY" not in sql for sql in statements.values()), (
        'expected all("CONCURRENTLY" not in sql for sql in statements.values()) to be true'
    )


def test_a_missing_index_is_reported_then_built_online(connection):
    with connection.cursor() as cursor:
        cursor.execute(f"DROP INDEX IF EXISTS {PROBE_INDEX}")

    before = {row["name"]: row for row in index_health(connection)}
    assert before[PROBE_INDEX]["present"] is False

    report = build_missing_indexes_concurrently(connection)

    assert PROBE_INDEX in report["built"]
    after = {row["name"]: row for row in index_health(connection)}
    assert after[PROBE_INDEX] == {"name": PROBE_INDEX, "present": True, "valid": True}
    # Second run is a no-op.
    assert build_missing_indexes_concurrently(connection)["built"] == []


def test_the_preflight_passes_on_a_ready_database_and_names_what_blocks(connection):
    ready = schema_preflight(connection)
    assert ready["ok"] is True
    assert ready["schema_version"]["registered"] == LATEST_POSTGRES_SCHEMA_VERSION
    assert ready["indexes"]["missing"] == [] and ready["indexes"]["invalid"] == []
    assert ready["in_flight_safety_workflows_without_flag"] == 0

    with connection.cursor() as cursor:
        cursor.execute(f"DROP INDEX IF EXISTS {PROBE_INDEX}")
        cursor.execute(
            """
            INSERT INTO gpu_fault_objects(kind, key, payload) VALUES (
                'workflow', 'wf-legacy-safety',
                '{"request_id":"wf-legacy-safety","incident_id":"inc-x","status":"RUNNING",
                  "fencing_token":1,"blocked_reasons":["policy"],"official_steps":[],
                  "safety_steps":[]}'::jsonb
            )
            """
        )

    blocked = schema_preflight(connection)

    assert blocked["ok"] is False
    assert blocked["indexes"]["missing"] == [PROBE_INDEX]
    assert blocked["in_flight_safety_workflows_without_flag"] == 1
    assert any("index" in reason for reason in blocked["blocking_reasons"]), (
        'expected any("index" in reason for reason in blocked["blocking_reasons"]) to be true'
    )
    assert any("safety" in reason for reason in blocked["blocking_reasons"]), (
        'expected any("safety" in reason for reason in blocked["blocking_reasons"]) to be true'
    )
