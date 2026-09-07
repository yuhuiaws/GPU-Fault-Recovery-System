"""The Postgres ``find_open_remote_command`` and the index that serves it.

Architecture review 2026-09-07, item D5. Behaviour is the contract
``test_find_open_remote_command.py`` pins on memory and SQLite; this adds the
Postgres backend and asks EXPLAIN which index the lookup walks. The query is
keyed on ``workflow_request_id`` first, which is exactly the prefix of
``gpu_fault_remote_command_workflow_all`` (v12, store review item H1), so no
new index and no schema migration are needed for it; this test is what turns
that claim into a check.
"""

from __future__ import annotations

import os
from datetime import timedelta

import pytest

from gpu_fault.regional import RemoteCommandResult
from gpu_fault.remote_command_models import RemoteCommandStatus
from tests._builders import copy_model
from tests.regional._regional_support import NOW
from tests.store._postgres_processor_claim_support import (
    _command,
    _truncate,
    postgres_store_instance,
)

pytestmark = pytest.mark.skipif(
    not os.getenv("GPU_FAULT_TEST_POSTGRES_URL"),
    reason="GPU_FAULT_TEST_POSTGRES_URL is required",
)

LOOKUP_SQL = """
    SELECT payload FROM gpu_fault_objects
    WHERE kind='remote_command'
      AND payload->>'workflow_request_id'=%s
      AND (payload->>'step_index')::int=%s
      AND payload->>'status' IN ('PENDING', 'LEASED', 'WAITING')
      AND key IS DISTINCT FROM %s
    ORDER BY payload->>'created_at', key
"""


@pytest.fixture
def store():
    for postgres in postgres_store_instance():
        yield postgres
    _truncate()


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


def test_the_lookup_walks_the_workflow_index(store) -> None:
    """Planner-neutral check like ``test_postgres_workflow_indexes.py``: with
    sequential scans priced out, the plan must name the v12 workflow index."""

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
    with psycopg.connect(os.environ["GPU_FAULT_TEST_POSTGRES_URL"]) as conn:
        with conn.cursor() as cursor:
            cursor.execute("ANALYZE gpu_fault_objects")
            cursor.execute("SET enable_seqscan = off")
            cursor.execute("EXPLAIN " + LOOKUP_SQL, ("wf-plan", 0, None))
            plan = "\n".join(str(row[0]) for row in cursor.fetchall())

    assert "gpu_fault_remote_command_workflow_all" in plan, plan
