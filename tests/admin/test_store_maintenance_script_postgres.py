"""The store-maintenance script against the test database: plan, execute,
refusals, idempotence, and the stdin/``python -c`` channel the verb uses."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from datetime import UTC, datetime, timedelta

import pytest

from gpu_fault.admin import store_maintenance_script as script
from gpu_fault.admin.store_maintenance import maintenance_source
from gpu_fault.store.postgres.ddl import declared_index_names
from tests.store._postgres_processor_claim_support import (
    _ensure_postgres_schema,
    _truncate,
)

POSTGRES_URL = os.getenv("GPU_FAULT_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(
    not POSTGRES_URL, reason="requires GPU_FAULT_TEST_POSTGRES_URL"
)
HOT_TABLES = set(script.COMPACT_TABLES) | set(script.REINDEX_TABLES)
SIZE_KEYS = (
    "live_rows",
    "dead_rows",
    "heap_pages",
    "heap_bytes",
    "index_bytes",
    "toast_bytes",
    "total_bytes",
)


@pytest.fixture(autouse=True)
def current_schema():
    _truncate()
    _ensure_postgres_schema()
    yield
    _truncate()


def _seed_churn() -> None:
    """Rows inserted and mostly deleted: dead tuples in heap and indexes."""

    import psycopg

    now = datetime.now(UTC)
    with psycopg.connect(POSTGRES_URL, autocommit=True) as connection:
        connection.execute(
            "INSERT INTO gpu_fault_processor_lanes "
            "(ordering_key, owner_id, epoch, lease_token, lease_expires_at, updated_at) "
            "SELECT 'lane-' || n, 'owner', 1, 'token', %s, %s FROM generate_series(1, 300) n",
            (now, now),
        )
        connection.execute(
            "DELETE FROM gpu_fault_processor_lanes WHERE ordering_key <> 'lane-1'"
        )
        connection.execute(
            "INSERT INTO gpu_fault_gpu_metric_latest "
            "(key, cluster_id, node_id, observed_at, payload) "
            "SELECT 'metric-' || n, 'hp-cluster', 'worker-1', %s, '{}'::jsonb "
            "FROM generate_series(1, 300) n",
            (now,),
        )
        connection.execute(
            "DELETE FROM gpu_fault_gpu_metric_latest WHERE key <> 'metric-1'"
        )


def test_plan_lists_every_hot_table_with_sizes_and_actions() -> None:
    with script.connect(POSTGRES_URL) as connection, connection.cursor() as cursor:
        plan = script.plan(cursor)

    assert {item["table"] for item in plan["tables"]} == HOT_TABLES
    assert plan["missing_tables"] == []
    assert len(plan["plan_sha256"]) == 64
    assert plan["server"]["reindex_concurrently_supported"] is True
    assert plan["server"]["version_num"] >= 120000
    for item in plan["tables"]:
        assert set(SIZE_KEYS) <= set(item), item["table"]
        assert all(isinstance(item[key], int) for key in SIZE_KEYS), (
            "sizes are whole numbers of rows, pages and bytes"
        )
        assert item["reloptions"], "v19 storage parameters are reported"
        expected = (
            script.COMPACT_ACTION
            if item["table"] in script.COMPACT_TABLES
            else script.REINDEX_ACTION
        )
        assert item["action"] == expected
        assert {index["name"] for index in item["indexes"]} >= {f"{item['table']}_pkey"}


def test_execute_reclaims_reports_before_after_and_is_idempotent() -> None:
    _seed_churn()
    declared = set(declared_index_names())
    with script.connect(POSTGRES_URL) as connection:
        with connection.cursor() as cursor:
            plan = script.plan(cursor)
        result = script.execute(connection, plan["plan_sha256"])

        assert result["status"] == "SUCCEEDED", result["errors"]
        assert result["plan_sha256"] == plan["plan_sha256"]
        assert {item["table"] for item in result["actions"]} == HOT_TABLES
        by_table = {item["table"]: item for item in result["actions"]}
        for table in script.COMPACT_TABLES:
            action = by_table[table]
            assert action["status"] == "SUCCEEDED"
            assert [step["statement"] for step in action["steps"]] == [
                f"VACUUM (FULL, ANALYZE) {table}"
            ]
            assert action["after"]["dead_rows"] == 0
        assert by_table["gpu_fault_processor_lanes"]["before"]["dead_rows"] >= 299
        metric = by_table["gpu_fault_gpu_metric_latest"]
        statements = [step["statement"] for step in metric["steps"]]
        assert statements[-1] == "VACUUM (ANALYZE) gpu_fault_gpu_metric_latest"
        assert statements[:-1] == [
            f"REINDEX INDEX CONCURRENTLY {index['name']}"
            for index in plan_index_names(plan, "gpu_fault_gpu_metric_latest")
        ]
        assert set(metric["before"]) == set(SIZE_KEYS) == set(metric["after"])

        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT c.relname, i.indisvalid
                FROM pg_class c JOIN pg_index i ON i.indexrelid=c.oid
                JOIN pg_class t ON t.oid=i.indrelid
                WHERE c.relname LIKE 'gpu\\_fault%'
                  AND NOT t.relispartition
                  AND NOT EXISTS (SELECT 1 FROM pg_constraint k WHERE k.conindid=c.oid)
                """
            )
            rows = cursor.fetchall()
        assert {name for name, _ in rows} == declared, "reindex keeps every index"
        assert all(valid for _, valid in rows), "reindex leaves no invalid index"

        with connection.cursor() as cursor:
            again = script.plan(cursor)
        assert again["plan_sha256"] == plan["plan_sha256"], (
            "the plan identity is stable"
        )
        assert script.execute(connection, again["plan_sha256"])["status"] == "SUCCEEDED"


def plan_index_names(plan: dict, table: str) -> list[dict]:
    (item,) = [entry for entry in plan["tables"] if entry["table"] == table]
    return item["indexes"]


def test_execute_refuses_a_drifted_plan_before_touching_anything() -> None:
    with script.connect(POSTGRES_URL) as connection:
        with pytest.raises(script.MaintenanceRefused, match="plan again"):
            script.execute(connection, "0" * 64)


def test_execute_refuses_while_a_schema_ensure_holds_the_lock() -> None:
    import psycopg

    held = threading.Event()
    release = threading.Event()

    def hold() -> None:
        with psycopg.connect(POSTGRES_URL, autocommit=True) as other:
            other.execute(
                "SELECT pg_advisory_lock(hashtextextended(%s, 0))",
                (script.ADVISORY_LOCK_NAME,),
            )
            held.set()
            release.wait(timeout=30)

    holder = threading.Thread(target=hold, daemon=True)
    holder.start()
    assert held.wait(timeout=10), "the holder must take the advisory lock first"
    try:
        with script.connect(POSTGRES_URL) as connection:
            with connection.cursor() as cursor:
                plan = script.plan(cursor)
            with pytest.raises(
                script.MaintenanceRefused, match="holds the store maintenance lock"
            ):
                script.execute(connection, plan["plan_sha256"])
    finally:
        release.set()
        holder.join(timeout=10)


def test_execute_needs_an_idle_autocommit_connection() -> None:
    import psycopg

    with psycopg.connect(POSTGRES_URL) as transactional:
        with pytest.raises(script.MaintenanceRefused, match="autocommit"):
            script.execute(transactional, "0" * 64)


def test_run_dispatches_modes_and_refuses_the_rest() -> None:
    with script.connect(POSTGRES_URL) as connection:
        assert script.run({"mode": "plan"}, connection)["mode"] == "plan"
        with pytest.raises(script.MaintenanceRefused, match="reviewed plan identity"):
            script.run({"mode": "execute"}, connection)
        with pytest.raises(script.MaintenanceRefused, match="unsupported"):
            script.run({"mode": "vacuum-everything"}, connection)
    assert script.reindex_concurrently_supported(110000) is False
    assert script.reindex_concurrently_supported(120000) is True


def test_the_script_runs_standalone_over_stdin_like_the_pod_exec(tmp_path) -> None:
    """``python -c <source>`` with the request on stdin and the DSN from the
    mounted file, exactly as the verb execs it into the Pod -- the source text
    is the one the verb ships (``maintenance_source``)."""

    dsn_file = tmp_path / "postgres-url"
    dsn_file.write_text(POSTGRES_URL + "\n", encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, "-c", maintenance_source()],
        input=json.dumps({"mode": "plan"}),
        env={
            "PATH": os.environ.get("PATH", ""),
            "HOME": os.environ.get("HOME", str(tmp_path)),
            "GPU_FAULT_STORE_URL_FILE": str(dsn_file),
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["mode"] == "plan"
    assert {item["table"] for item in payload["tables"]} == HOT_TABLES

    refused = subprocess.run(
        [sys.executable, "-c", maintenance_source()],
        input=json.dumps({"mode": "execute"}),
        env={
            "PATH": os.environ.get("PATH", ""),
            "HOME": os.environ.get("HOME", str(tmp_path)),
            "GPU_FAULT_STORE_URL_FILE": str(dsn_file),
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert refused.returncode == 1
    assert json.loads(refused.stdout)["error_kind"] == "refused"


def test_store_dsn_prefers_the_mounted_file(tmp_path, monkeypatch) -> None:
    dsn_file = tmp_path / "postgres-url"
    dsn_file.write_text("postgresql://file\n", encoding="utf-8")
    monkeypatch.setenv("GPU_FAULT_STORE_URL_FILE", str(dsn_file))
    monkeypatch.setenv("GPU_FAULT_STORE_URL", "postgresql://env")
    assert script.store_dsn() == "postgresql://file"
    monkeypatch.setenv("GPU_FAULT_STORE_URL_FILE", "")
    monkeypatch.setattr(script, "DEFAULT_STORE_URL_FILE", str(tmp_path / "absent"))
    assert script.store_dsn() == "postgresql://env"
    monkeypatch.setenv("GPU_FAULT_STORE_URL", "")
    with pytest.raises(script.MaintenanceRefused, match="unavailable"):
        script.store_dsn()
    _ = timedelta  # the churn helper's import stays used in this module
