"""Serial, local PostgreSQL coverage for uninstall's read-only drain evidence."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from tests._script_loader import lazy_script_module

PROBE = lazy_script_module(
    Path(__file__).resolve().parents[2]
    / "deploy/control-plane/tools/cleanup_activity.py"
)
POSTGRES_URL = os.environ.get("GPU_FAULT_TEST_POSTGRES_URL", "")
pytestmark = pytest.mark.skipif(
    not POSTGRES_URL, reason="requires a dedicated local PostgreSQL test URL"
)


@pytest.fixture
def connection() -> Iterator[Any]:
    import psycopg
    from psycopg import sql
    from psycopg.conninfo import conninfo_to_dict, make_conninfo

    assert conninfo_to_dict(POSTGRES_URL).get("host") == "127.0.0.1"
    schema = "uninstall_probe_" + uuid4().hex
    with psycopg.connect(POSTGRES_URL, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        try:
            with psycopg.connect(
                make_conninfo(POSTGRES_URL, options=f"-c search_path={schema}"),
                autocommit=True,
            ) as target:
                target.execute(
                    "CREATE TABLE gpu_fault_objects(kind text, key text, payload jsonb)"
                )
                target.execute(
                    "CREATE TABLE gpu_fault_processor_queue(status text, cluster_id text)"
                )
                target.execute(
                    "CREATE TABLE gpu_fault_telemetry_spool(cluster_id text)"
                )
                yield target
        finally:
            admin.execute(
                sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema))
            )


def insert(
    connection: Any, kind: str, payload: dict[str, Any], key: str = "record"
) -> None:
    from psycopg.types.json import Jsonb

    connection.execute(
        "INSERT INTO gpu_fault_objects VALUES (%s,%s,%s)", (kind, key, Jsonb(payload))
    )


@pytest.mark.parametrize("scope", ["all", "gpu"])
@pytest.mark.parametrize("status", ["RUNNING", "BLOCKED", "UNKNOWN", None])
def test_orphan_and_unknown_workflows_block_cleanup(
    connection: Any, scope: str, status: str | None
) -> None:
    insert(connection, "workflow", {"status": status, "incident_id": "missing"})
    assert PROBE.counts(connection, scope=scope, cluster_ids=["gpu-a"]) == (1, 0, 0, 0)


def test_scoped_cleanup_excludes_known_other_clusters_but_not_unknown_commands(
    connection: Any,
) -> None:
    insert(connection, "incident", {"cluster_id": "gpu-b"}, key="incident-b")
    insert(connection, "workflow", {"status": "RUNNING", "incident_id": "incident-b"})
    insert(connection, "remote_command", {"status": "UNKNOWN"})
    assert PROBE.counts(connection, scope="gpu", cluster_ids=["gpu-a"]) == (0, 1, 0, 0)
    assert PROBE.counts(connection, scope="all", cluster_ids=["gpu-a"]) == (1, 1, 0, 0)


def test_terminal_records_do_not_block_but_unknown_queue_status_does(
    connection: Any,
) -> None:
    for index, status in enumerate(("SUCCEEDED", "FAILED", "SUPERSEDED")):
        insert(connection, "workflow", {"status": status}, key=str(index))
    for index, status in enumerate(("SUCCEEDED", "FAILED")):
        insert(connection, "remote_command", {"status": status}, key=str(index))
    connection.execute(
        "INSERT INTO gpu_fault_processor_queue VALUES ('UNKNOWN','gpu-a'),('COMPLETED','gpu-a')"
    )
    connection.execute("INSERT INTO gpu_fault_telemetry_spool VALUES ('gpu-a')")
    assert PROBE.counts(connection, scope="gpu", cluster_ids=["gpu-a"]) == (0, 0, 1, 1)


def test_missing_required_table_is_not_an_empty_queue(connection: Any) -> None:
    import psycopg

    connection.execute("DROP TABLE gpu_fault_processor_queue")
    with pytest.raises(psycopg.errors.UndefinedTable):
        PROBE.counts(connection, scope="all", cluster_ids=[])


def test_control_records_view_is_used_when_present(connection: Any) -> None:
    connection.execute("CREATE TABLE current_records(kind text,key text,payload jsonb)")
    connection.execute(
        "CREATE VIEW gpu_fault_control_records AS SELECT * FROM current_records"
    )
    connection.execute(
        """INSERT INTO current_records VALUES
        ('remote_command','current','{"cluster_id":"gpu-a","status":"LEASED"}')"""
    )
    assert PROBE.counts(connection, scope="gpu", cluster_ids=["gpu-a"]) == (0, 1, 0, 0)


def test_unknown_control_state_mode_fails_even_with_empty_tables(
    connection: Any,
) -> None:
    connection.execute(
        "CREATE TABLE gpu_fault_control_state_modes(kind text,mode text)"
    )
    connection.execute(
        "INSERT INTO gpu_fault_control_state_modes VALUES ('workflow','UNKNOWN')"
    )
    with pytest.raises(ValueError, match="storage is not known"):
        PROBE.counts(connection, scope="all", cluster_ids=[])


def test_probe_transaction_is_read_only(connection: Any) -> None:
    import psycopg

    connection.execute(
        """CREATE FUNCTION unexpected_write() RETURNS jsonb LANGUAGE plpgsql AS $$
        BEGIN INSERT INTO gpu_fault_processor_queue VALUES ('PENDING','gpu-a');
        RETURN '{"status":"RUNNING"}'::jsonb; END $$"""
    )
    connection.execute(
        """CREATE VIEW gpu_fault_control_records AS
        SELECT 'workflow'::text AS kind, 'row'::text AS key, unexpected_write() AS payload"""
    )
    with pytest.raises(psycopg.errors.ReadOnlySqlTransaction):
        PROBE.counts(connection, scope="all", cluster_ids=[])
    assert connection.execute(
        "SELECT count(*) FROM gpu_fault_processor_queue"
    ).fetchone() == (0,)
