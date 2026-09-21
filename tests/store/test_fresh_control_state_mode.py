"""Fresh PostgreSQL databases may start the control-state tables in dedicated mode.

Ordinary deployment never activates a migration on an existing database; a
brand-new database has nothing to migrate, so the schema release Job may seed
``dedicated`` directly (user decision 2026-09-20). Existing databases keep the
explicit legacy -> dual -> dedicated path whatever the caller asks for.
"""

from __future__ import annotations

import json
import os

import pytest

from gpu_fault.state_table_migrate import state_table_status
from gpu_fault.store import PostgresStore

POSTGRES_URL = os.getenv("GPU_FAULT_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(
    not POSTGRES_URL, reason="GPU_FAULT_TEST_POSTGRES_URL is not configured"
)


def _fresh_database() -> None:
    import psycopg  # optional driver: never at module scope

    assert POSTGRES_URL is not None
    with psycopg.connect(POSTGRES_URL, autocommit=True) as connection:
        connection.execute("DROP SCHEMA public CASCADE")
        connection.execute("CREATE SCHEMA public")


def _modes() -> list[tuple]:
    import psycopg

    assert POSTGRES_URL is not None
    with psycopg.connect(POSTGRES_URL, autocommit=True) as connection:
        return connection.execute(
            "SELECT kind, mode, revision, backfill_complete, legacy_purged, "
            "dedicated_at IS NOT NULL FROM gpu_fault_control_state_modes ORDER BY kind"
        ).fetchall()


def test_fresh_database_seeds_dedicated_when_the_schema_job_asks_for_it() -> None:
    _fresh_database()
    PostgresStore(POSTGRES_URL, fresh_control_state_mode="dedicated").close()

    assert _modes() == [
        ("remote_command", "dedicated", 1, True, False, True),
        ("workflow", "dedicated", 1, True, False, True),
    ]
    import psycopg

    with psycopg.connect(POSTGRES_URL, autocommit=True) as connection:
        for kind in ("remote_command", "workflow"):
            status = state_table_status(connection, kind)
            assert status["mode"] == "dedicated", status
            assert status["backfill_complete"] is True, status
            assert (status["legacy_rows"], status["dedicated_rows"]) == (0, 0), status

    # Reopening -- with or without the request -- is a no-op: the database is
    # no longer fresh, so nothing is re-seeded and the revision stays put.
    PostgresStore(POSTGRES_URL, fresh_control_state_mode="dedicated").close()
    PostgresStore(POSTGRES_URL).close()
    assert [row[1:3] for row in _modes()] == [("dedicated", 1), ("dedicated", 1)]


def test_fresh_database_keeps_legacy_by_default() -> None:
    _fresh_database()
    PostgresStore(POSTGRES_URL).close()
    assert [row[1:3] for row in _modes()] == [("legacy", 0), ("legacy", 0)]


def test_existing_database_is_never_switched_by_the_fresh_install_request() -> None:
    _fresh_database()
    PostgresStore(POSTGRES_URL).close()
    import psycopg

    with psycopg.connect(POSTGRES_URL, autocommit=True) as connection:
        connection.execute(
            "INSERT INTO gpu_fault_objects(kind, key, payload) VALUES (%s, %s, %s)",
            (
                "marker",
                "fresh-mode-guard",
                json.dumps({"marker_id": "fresh-mode-guard"}),
            ),
        )

    PostgresStore(POSTGRES_URL, fresh_control_state_mode="dedicated").close()

    assert [row[1:3] for row in _modes()] == [("legacy", 0), ("legacy", 0)], (
        "an existing database must keep the explicit migration path"
    )


@pytest.mark.parametrize("value", ["dual", "", "Dedicated ", "later"])
def test_fresh_control_state_mode_accepts_only_legacy_or_dedicated(value: str) -> None:
    if value.strip().lower() == "dedicated":
        _fresh_database()
        PostgresStore(POSTGRES_URL, fresh_control_state_mode=value).close()
        assert [row[1] for row in _modes()] == ["dedicated", "dedicated"]
        return
    with pytest.raises(ValueError, match="fresh_control_state_mode"):
        PostgresStore(POSTGRES_URL, fresh_control_state_mode=value)
