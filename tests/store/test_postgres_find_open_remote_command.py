"""The Postgres ``find_open_remote_command`` and the index that serves it.

Architecture review 2026-09-07, item D5. Behaviour is the contract
``test_find_open_remote_command.py`` pins on memory and SQLite; this adds the
Postgres backend and asks EXPLAIN which index the runtime lookup walks in each
storage mode. The workflow prefix must still narrow the authoritative records,
and dedicated reads must not scan the retained legacy rows.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import timedelta

import pytest

from gpu_fault.regional import RemoteCommandResult
from gpu_fault.remote_command_models import RemoteCommandStatus
from gpu_fault.store.postgres.pool import PooledPostgresDatabase
from tests._builders import copy_model
from tests.regional._regional_support import NOW
from tests.store import test_postgres_remote_runtime as remote_runtime
from tests.store._postgres_processor_claim_support import _command
from tests.store.test_postgres_state_tables import plan_nodes

database = remote_runtime.database
migration_database = remote_runtime.migration_database
mode = remote_runtime.mode
store = remote_runtime.store

pytestmark = pytest.mark.skipif(
    not os.getenv("GPU_FAULT_TEST_POSTGRES_URL"),
    reason="GPU_FAULT_TEST_POSTGRES_URL is required",
)


class _QueryRecorder:
    def __init__(self, cursor, statements) -> None:
        self.cursor = cursor
        self.statements = statements

    def execute(self, query, parameters=None):
        self.statements.append((query, parameters))
        return self.cursor.execute(query, parameters)

    def fetchall(self):
        return self.cursor.fetchall()


def test_open_sibling_is_found_and_a_terminal_one_is_not(store) -> None:
    command = _command(store, "remote-a", request_id="wf-open")

    found = store.find_open_remote_command("wf-open", 0, "official")
    assert found is not None and found.command_id == "remote-a"
    assert store.find_open_remote_command("wf-open", 1, "official") is None
    assert store.find_open_remote_command("wf-open", 0, "safety") is None
    assert (
        store.find_open_remote_command(
            "wf-open", 0, "official", exclude_command_id="remote-a"
        )
        is None
    )

    claimed = store.claim_remote_commands(
        command.cluster_id, "executor-a", limit=1, lease_seconds=60
    )
    leased = store.find_open_remote_command("wf-open", 0, "official")
    assert leased is not None and leased.status is RemoteCommandStatus.LEASED
    store.complete_remote_command(
        command.cluster_id,
        "remote-a",
        RemoteCommandResult(
            lease_token=claimed[0].lease_token, status=RemoteCommandStatus.SUCCEEDED
        ),
    )

    assert store.find_open_remote_command("wf-open", 0, "official") is None


def test_the_oldest_open_sibling_wins(store) -> None:
    later = _command(store, "remote-later", request_id="wf-order")
    store.ensure_remote_command(
        copy_model(
            later, command_id="remote-earlier", created_at=NOW - timedelta(minutes=1)
        )
    )

    found = store.find_open_remote_command("wf-order", 0, "official")

    assert found is not None and found.command_id == "remote-earlier"


def test_the_lookup_walks_the_workflow_index(store, mode, monkeypatch) -> None:
    """Explain the public lookup's statement, not a parallel SQL approximation."""

    import psycopg

    base = _command(store, "remote-plan", request_id="wf-plan")
    for index in range(120):
        store.ensure_remote_command(
            copy_model(
                base,
                command_id=f"remote-fill-{index:04d}",
                workflow_request_id=f"wf-fill-{index % 20}",
                created_at=NOW + timedelta(seconds=index),
            )
        )
    statements = []
    original_cursor = PooledPostgresDatabase.cursor

    @contextmanager
    def recording_cursor(database):
        with original_cursor(database) as cursor:
            yield _QueryRecorder(cursor, statements)

    with monkeypatch.context() as scoped:
        scoped.setattr(PooledPostgresDatabase, "cursor", recording_cursor)
        found = store.find_open_remote_command("wf-plan", 0, "official")
    assert found is not None and found.command_id == base.command_id
    assert len(statements) == 1
    query, parameters = statements[0]
    with psycopg.connect(os.environ["GPU_FAULT_TEST_POSTGRES_URL"]) as conn:
        with conn.cursor() as cursor:
            cursor.execute("ANALYZE gpu_fault_objects")
            cursor.execute("ANALYZE gpu_fault_remote_commands")
            cursor.execute("SET enable_seqscan = off")
            cursor.execute(
                "EXPLAIN (ANALYZE, FORMAT JSON, COSTS OFF) " + query, parameters
            )
            nodes = list(plan_nodes(cursor.fetchone()[0]))
    index = (
        "gpu_fault_remote_command_workflow_all"
        if mode == "legacy"
        else "gpu_fault_remote_commands_workflow"
    )
    assert any(node.get("Index Name") == index for node in nodes), nodes
    if mode == "dedicated":
        assert all(
            node.get("Actual Loops") == 0
            for node in nodes
            if node.get("Relation Name") == "gpu_fault_objects"
        ), "dedicated lookup scanned legacy storage"
