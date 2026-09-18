from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.models import WorkflowStatus
from gpu_fault.regional import RemoteCommandStatus
from gpu_fault.schema_migrations import (
    LATEST_POSTGRES_SCHEMA_VERSION,
    POSTGRES_SCHEMA_MIGRATIONS,
)
from gpu_fault.state_table_migrate import (
    backfill_state_table,
    set_state_table_mode,
    state_table_status,
)
from gpu_fault.store import PostgresStore
from scripts.e2e.regional.probes import state_table_snapshot as probe
from scripts.e2e.regional.run_state_table_acceptance import state_errors
from tests._builders import workflow_request
from tests.store import test_postgres_state_tables as state_tables
from tests.store.test_state_table_payload import command

POSTGRES_URL = os.getenv("GPU_FAULT_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(
    not POSTGRES_URL, reason="GPU_FAULT_TEST_POSTGRES_URL is not configured"
)
database = state_tables.database
migration_database = state_tables.migration_database


@pytest.fixture
def probe_database(migration_database, monkeypatch: pytest.MonkeyPatch):
    assert POSTGRES_URL, (
        "this test requires the explicitly configured isolated PostgreSQL"
    )
    monkeypatch.setenv("GPU_FAULT_STORE_URL", POSTGRES_URL)
    monkeypatch.delenv("GPU_FAULT_STORE_URL_FILE", raising=False)
    return migration_database


def seed_state_table(connection, kind: str, mode: str) -> dict[str, Any]:
    assert POSTGRES_URL, "only the explicitly configured fixture database may be seeded"
    store = PostgresStore(POSTGRES_URL, initialize_schema=False)
    try:
        if kind == "remote_command":
            record = command(datetime.now(timezone.utc)).model_copy(
                update={
                    "status": RemoteCommandStatus.SUCCEEDED,
                    "lease_expires_at": None,
                }
            )
            store.ensure_remote_command(record)
        else:
            record = workflow_request("audit-workflow", "audit-incident").model_copy(
                update={"status": WorkflowStatus.SUCCEEDED}
            )
            store.save_workflow(record)
    finally:
        store.close()
    if mode != "legacy":
        set_state_table_mode(connection, kind, "dual", expected_mode="legacy")
        backfill_state_table(connection, kind)
    if mode == "dedicated":
        set_state_table_mode(
            connection, kind, mode, expected_mode="dual", confirm_dedicated=True
        )
    return record.model_dump(mode="json")


@pytest.mark.parametrize("kind", ["remote_command", "workflow"])
@pytest.mark.parametrize("mode", ["legacy", "dual", "dedicated"])
def test_deployed_state_probe_uses_readonly_real_postgres(
    probe_database, monkeypatch: pytest.MonkeyPatch, kind: str, mode: str
) -> None:
    seed_state_table(probe_database, kind, mode)
    validate_schema = probe.validate_state_table_schema

    def check_schema(connection) -> None:
        validate_schema(connection)
        assert connection.execute(
            "SELECT current_setting('transaction_read_only'), pg_my_temp_schema()"
        ).fetchone() == ("on", 0), (
            "schema validation must not create even temporary objects"
        )

    monkeypatch.setattr(probe, "validate_state_table_schema", check_schema)
    before = state_table_status(probe_database, kind, verify=False)
    for verify in (False, True):
        report = probe.snapshot(kind, verify=verify)
        assert state_errors(report, kind, mode, verify=verify) == [], report
        assert report["state"]["legacy_rows"] == 1, report
        assert report["state"]["dedicated_rows"] == (0 if mode == "legacy" else 1), (
            report
        )
        assert report["read_only"] is True and report["writer"] is True, report
        assert report["state"]["verification_performed"] is (verify and mode == "dual")
        assert state_table_status(probe_database, kind, verify=False) == before


@pytest.mark.parametrize(
    "mutation",
    [
        "DELETE FROM gpu_fault_control_state_modes",
        "CREATE TEMP VIEW unexpected_probe_mutation AS SELECT 1",
    ],
)
def test_probe_session_cannot_mutate_even_if_a_future_validator_tries(
    probe_database, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    import psycopg

    def attempted_write(connection) -> None:
        connection.execute(mutation)

    monkeypatch.setattr(probe, "validate_state_table_schema", attempted_write)
    with pytest.raises(psycopg.errors.ReadOnlySqlTransaction):
        probe.snapshot("workflow", verify=True)
    assert (
        probe_database.execute(
            "SELECT count(*) FROM gpu_fault_control_state_modes"
        ).fetchone()[0]
        == 2
    ), "read-only audit must preserve both migration metadata rows"


@pytest.mark.parametrize("drift", ["version", "checksum", "missing-history"])
def test_probe_rejects_release_history_drift(probe_database, drift: str) -> None:
    from psycopg.types.json import Jsonb

    migration = POSTGRES_SCHEMA_MIGRATIONS[-1]
    original = probe_database.execute(
        "SELECT to_jsonb(m) FROM gpu_fault_schema_migrations m WHERE version=%s",
        (migration.version,),
    ).fetchone()[0]
    try:
        if drift == "version":
            probe_database.execute(
                "UPDATE gpu_fault_schema_version SET version=version+1"
            )
        elif drift == "checksum":
            probe_database.execute(
                "UPDATE gpu_fault_schema_migrations SET checksum=repeat('0',64) WHERE version=%s",
                (migration.version,),
            )
        else:
            probe_database.execute(
                "DELETE FROM gpu_fault_schema_migrations WHERE version=%s",
                (migration.version,),
            )
        with pytest.raises(RuntimeError, match="schema version|migration history"):
            probe.snapshot("workflow", verify=True)
    finally:
        probe_database.execute(
            "UPDATE gpu_fault_schema_version SET version=%s",
            (LATEST_POSTGRES_SCHEMA_VERSION,),
        )
        probe_database.execute(
            "INSERT INTO gpu_fault_schema_migrations "
            "SELECT * FROM jsonb_populate_record(NULL::gpu_fault_schema_migrations, %s) "
            "ON CONFLICT (version) DO UPDATE SET checksum=EXCLUDED.checksum",
            (Jsonb(original),),
        )


@pytest.mark.parametrize(
    "view",
    [
        "gpu_fault_remote_command_records",
        "gpu_fault_workflow_records",
        "gpu_fault_control_records",
    ],
)
@pytest.mark.parametrize("drift", ["predicate", "column-name"])
def test_probe_rejects_read_view_drift_without_repairing_it(
    probe_database, view: str, drift: str
) -> None:
    from psycopg import sql

    original = probe_database.execute(
        "SELECT pg_get_viewdef(to_regclass(%s), true)", (view,)
    ).fetchone()[0]
    try:
        if drift == "predicate":
            probe_database.execute(
                sql.SQL(
                    "CREATE OR REPLACE VIEW {} AS SELECT * FROM ({}) drifted WHERE false"
                ).format(
                    sql.Identifier(view), sql.SQL(original.rstrip().removesuffix(";"))
                )
            )
        else:
            probe_database.execute(
                sql.SQL(
                    "ALTER VIEW {} RENAME COLUMN payload TO drifted_payload"
                ).format(sql.Identifier(view))
            )
        with pytest.raises(RuntimeError, match="control-state view .* differs"):
            probe.snapshot("workflow", verify=True)
        current = probe_database.execute(
            "SELECT pg_get_viewdef(to_regclass(%s), true)", (view,)
        ).fetchone()[0]
        assert current != original, "an audit must not repair drifted schema"
    finally:
        if drift == "column-name":
            probe_database.execute(
                sql.SQL(
                    "ALTER VIEW {} RENAME COLUMN drifted_payload TO payload"
                ).format(sql.Identifier(view))
            )
        probe_database.execute(
            sql.SQL("CREATE OR REPLACE VIEW {} AS {}").format(
                sql.Identifier(view), sql.SQL(original)
            )
        )


def test_probe_rejects_disabled_mirroring_without_enabling_it(probe_database) -> None:
    try:
        probe_database.execute(
            "ALTER TABLE gpu_fault_objects DISABLE TRIGGER gpu_fault_objects_control_state_mirror_trigger"
        )
        with pytest.raises(RuntimeError, match="missing or disabled"):
            probe.snapshot("remote_command", verify=True)
        assert probe_database.execute(
            "SELECT tgenabled FROM pg_trigger "
            "WHERE tgrelid='gpu_fault_objects'::regclass "
            "AND tgname='gpu_fault_objects_control_state_mirror_trigger'"
        ).fetchone() == ("D",), "an audit must not enable a disabled trigger"
    finally:
        probe_database.execute(
            "ALTER TABLE gpu_fault_objects ENABLE TRIGGER gpu_fault_objects_control_state_mirror_trigger"
        )


@pytest.mark.parametrize("kind", ["remote_command", "workflow"])
def test_probe_rejects_missing_mode_without_reseeding_it(
    probe_database, kind: str
) -> None:
    from psycopg.types.json import Jsonb

    original = probe_database.execute(
        "SELECT to_jsonb(m) FROM gpu_fault_control_state_modes m WHERE kind=%s", (kind,)
    ).fetchone()[0]
    try:
        probe_database.execute(
            "DELETE FROM gpu_fault_control_state_modes WHERE kind=%s", (kind,)
        )
        with pytest.raises(RuntimeError, match="migration modes"):
            probe.snapshot(kind, verify=True)
        assert (
            probe_database.execute(
                "SELECT kind FROM gpu_fault_control_state_modes WHERE kind=%s", (kind,)
            ).fetchone()
            is None
        ), "an audit must not recreate missing migration metadata"
    finally:
        probe_database.execute(
            "INSERT INTO gpu_fault_control_state_modes "
            "SELECT * FROM jsonb_populate_record(NULL::gpu_fault_control_state_modes, %s)",
            (Jsonb(original),),
        )


@pytest.mark.parametrize("kind", ["remote_command", "workflow"])
def test_dual_probe_compares_full_payload_not_just_counts(
    probe_database, kind: str
) -> None:
    from psycopg.types.json import Jsonb

    payload = seed_state_table(probe_database, kind, "dual")
    if kind == "remote_command":
        payload["step"]["parameters"]["snapshot"] = "changed-tail-" + "y" * 100_000
    else:
        payload["blocked_reasons"] = ["changed-full-payload"]
    with probe_database.transaction():
        probe_database.execute(
            "ALTER TABLE gpu_fault_objects DISABLE TRIGGER gpu_fault_objects_control_state_mirror_trigger"
        )
        probe_database.execute(
            "UPDATE gpu_fault_objects SET payload=%s WHERE kind=%s",
            (Jsonb(payload), kind),
        )
        probe_database.execute(
            "ALTER TABLE gpu_fault_objects ENABLE TRIGGER gpu_fault_objects_control_state_mirror_trigger"
        )
    report = probe.snapshot(kind, verify=True)
    assert report["state"]["legacy_rows"] == report["state"]["dedicated_rows"] == 1
    assert report["state"]["missing_rows"] == report["state"]["extra_rows"] == 0
    assert (
        report["state"]["invalid_records"]
        == report["state"]["noncanonical_records"]
        == 0
    )
    assert (
        report["state"]["mismatched_rows"] == 1 and report["state"]["verified"] is False
    )
    assert state_errors(report, kind, "dual", verify=True), (
        "equal counts must not hide payload drift"
    )


def test_probe_uses_current_mounted_credentials_instead_of_stale_environment(
    probe_database, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    assert POSTGRES_URL, (
        "this test requires the explicitly configured isolated PostgreSQL"
    )
    credentials = tmp_path / "store-url"
    credentials.write_text(POSTGRES_URL)
    credentials.chmod(0o600)
    monkeypatch.setenv("GPU_FAULT_STORE_URL_FILE", str(credentials))
    monkeypatch.setenv(
        "GPU_FAULT_STORE_URL", "postgresql://obsolete@127.0.0.1:1/obsolete"
    )
    report = probe.snapshot("workflow", verify=False)
    assert state_errors(report, "workflow", "legacy", verify=False) == [], report
