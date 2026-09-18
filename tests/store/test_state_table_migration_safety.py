from __future__ import annotations

import json
import os
import time
from contextlib import nullcontext
from datetime import UTC, datetime, timedelta

import pytest

from gpu_fault import state_table_migrate, store_migrate
from gpu_fault.state_table_migrate import (
    backfill_state_table,
    set_state_table_mode,
    state_table_status,
)
from gpu_fault.store.postgres.state_table_payload import STATE_LAYOUTS
from gpu_fault.store.postgres.state_table_storage import (
    get_state_payload,
    put_state_record,
)
from gpu_fault.store.shared.errors import StaleWriteError
from tests.store import test_postgres_state_tables as state_tables
from tests.store.test_state_table_payload import command

database = state_tables.database
migration_database = state_tables.migration_database
POSTGRES_URL = os.environ.get("GPU_FAULT_TEST_POSTGRES_URL", "")
pytestmark = pytest.mark.skipif(
    not POSTGRES_URL, reason="requires an isolated PostgreSQL test database"
)


def legacy_record(connection, kind: str, mode: str):
    at = datetime(2026, 9, 11, tzinfo=UTC)
    value = command(at)
    if kind == "workflow":
        value = value.workflow.model_copy(update={"created_at": at, "updated_at": at})
    key = getattr(value, STATE_LAYOUTS[kind].key_field)
    payload = value.model_dump(mode="json")
    payload["created_at"] = "2026-09-11T00:00:00Z"
    payload.pop("lease_owner" if kind == "remote_command" else "blocked_kind")
    if mode == "dual":
        set_state_table_mode(connection, kind, "dual", expected_mode="legacy")
    connection.execute(
        "INSERT INTO gpu_fault_objects(kind,key,payload) VALUES (%s,%s,%s::jsonb)",
        (kind, key, json.dumps(payload)),
    )
    return key, type(value)


@pytest.mark.parametrize("kind", ["remote_command", "workflow"])
@pytest.mark.parametrize("mode", ["legacy", "dual"])
def test_conditional_write_accepts_the_unchanged_legacy_record(
    migration_database, kind: str, mode: str
) -> None:
    connection = migration_database
    key, model = legacy_record(connection, kind, mode)
    expected = model.model_validate(get_state_payload(connection, kind, key))
    updated = expected.model_copy(
        update={"updated_at": expected.updated_at + timedelta(seconds=1)}
    )
    assert put_state_record(connection, kind, key, updated, expected=expected), (
        "conditional writes must accept the same record before backfill canonicalizes it"
    )
    assert get_state_payload(connection, kind, key) == updated.model_dump(mode="json")


@pytest.mark.parametrize("kind", ["remote_command", "workflow"])
def test_conditional_delete_accepts_the_unchanged_dual_view(
    migration_database, kind: str
) -> None:
    connection = migration_database
    key, _model = legacy_record(connection, kind, "dual")
    expected = get_state_payload(connection, kind, key)
    removed = connection.execute(
        "SELECT gpu_fault_delete_control_state(%s,%s,%s::jsonb)",
        (kind, key, json.dumps(expected)),
    ).fetchone()[0]
    assert removed is True, (
        "a mirror projection must still bind its authoritative legacy row"
    )
    assert connection.execute(
        "SELECT count(*) FROM gpu_fault_control_records WHERE kind=%s AND key=%s",
        (kind, key),
    ).fetchone() == (0,)


@pytest.mark.parametrize("kind", ["remote_command", "workflow"])
@pytest.mark.parametrize("mode", ["legacy", "dual"])
def test_legacy_compatibility_does_not_accept_a_stale_write(
    migration_database, kind: str, mode: str
) -> None:
    connection = migration_database
    key, model = legacy_record(connection, kind, mode)
    previous = model.model_validate(get_state_payload(connection, kind, key))
    current = previous.model_copy(
        update={"updated_at": previous.updated_at + timedelta(seconds=1)}
    )
    put_state_record(connection, kind, key, current, expected=previous)
    with pytest.raises(StaleWriteError):
        put_state_record(connection, kind, key, previous, expected=previous)
    assert get_state_payload(connection, kind, key) == current.model_dump(mode="json")


@pytest.mark.parametrize("kind", ["remote_command", "workflow"])
def test_dual_projection_does_not_accept_a_stale_delete(
    migration_database, kind: str
) -> None:
    connection = migration_database
    key, _model = legacy_record(connection, kind, "dual")
    previous = get_state_payload(connection, kind, key)
    connection.execute(
        "UPDATE gpu_fault_objects SET payload=payload || %s::jsonb WHERE kind=%s AND key=%s",
        (json.dumps({"status": "FAILED"}), kind, key),
    )
    removed = connection.execute(
        "SELECT gpu_fault_delete_control_state(%s,%s,%s::jsonb)",
        (kind, key, json.dumps(previous)),
    ).fetchone()[0]
    assert removed is False, "normalization must not erase a real state change"
    assert get_state_payload(connection, kind, key)["status"] == "FAILED"


@pytest.mark.parametrize("kind", ["remote_command", "workflow"])
def test_incomplete_backfill_defers_full_payload_verification(
    migration_database, kind: str
) -> None:
    connection = migration_database
    legacy_record(connection, kind, "dual")
    result = backfill_state_table(connection, kind, batch_size=1, max_batches=1)
    assert result["status"]["backfill_complete"] is False
    assert result["status"]["verification_performed"] is False
    assert "verified" not in result["status"], (
        "an incomplete batch must not claim full verification"
    )
    checked = state_table_status(connection, kind)
    assert checked["verification_performed"] is True
    assert checked["verified"] is True
    finished = backfill_state_table(connection, kind, batch_size=1, max_batches=1)
    assert finished["status"]["backfill_complete"] is True
    assert finished["status"]["verification_performed"] is True
    assert finished["status"]["verified"] is True


def test_status_has_a_database_wait_deadline_and_restores_connection_settings(
    migration_database, monkeypatch: pytest.MonkeyPatch
) -> None:
    import psycopg

    connection = migration_database
    previous = connection.execute("SHOW statement_timeout").fetchone()
    monkeypatch.setattr(state_table_migrate, "STATE_TABLE_VALIDATION_SECONDS", 0.1)
    with psycopg.connect(state_tables.POSTGRES_URL) as blocker:
        blocker.execute("LOCK gpu_fault_objects IN ACCESS EXCLUSIVE MODE")
        started = time.monotonic()
        with pytest.raises(psycopg.errors.QueryCanceled):
            state_table_status(connection, "workflow")
        assert time.monotonic() - started < 2, (
            "status must not wait indefinitely behind DDL"
        )
    assert connection.execute("SHOW statement_timeout").fetchone() == previous


@pytest.mark.parametrize(
    "corruption",
    [
        "ALTER TABLE gpu_fault_objects DISABLE TRIGGER gpu_fault_objects_control_state_mirror_trigger",
        "UPDATE gpu_fault_schema_version SET version=version+1",
        "UPDATE gpu_fault_schema_migrations SET checksum=repeat('0',64) WHERE version=16",
    ],
)
def test_cli_refuses_schema_drift_before_changing_migration_mode(
    migration_database, monkeypatch: pytest.MonkeyPatch, corruption: str
) -> None:
    import psycopg

    connection = migration_database
    monkeypatch.setenv("GPU_FAULT_STORE_URL", state_tables.POSTGRES_URL)
    monkeypatch.setattr(
        "sys.argv",
        [
            "gpu-fault-store-migrate",
            "--state-table-kind",
            "workflow",
            "--set-state-table-mode",
            "dual",
            "--expected-state-table-mode",
            "legacy",
        ],
    )
    monkeypatch.setattr(
        psycopg, "connect", lambda *_args, **_kwargs: nullcontext(connection)
    )
    with connection.transaction(force_rollback=True):
        connection.execute(corruption)
        with pytest.raises(
            SystemExit, match="missing or disabled|schema version|history differs"
        ):
            store_migrate.main()
        assert connection.execute(
            "SELECT mode FROM gpu_fault_control_state_modes WHERE kind='workflow'"
        ).fetchone() == ("legacy",)
