from __future__ import annotations

import hashlib
from datetime import timedelta

import pytest

from gpu_fault import schema_migrations, state_table_migrate
from gpu_fault.control_record_archive import ControlRecordArchiver
from gpu_fault.schema_migrations import SchemaMigration
from gpu_fault.state_table_migrate import (
    StateTableMigrationError,
    backfill_state_table,
    purge_legacy_state_rows,
    retire_legacy_state_indexes,
    set_state_table_mode,
    state_table_maintenance,
    state_table_status,
)
from tests._builders import workflow_request
from tests.store import _cov95_runtime_postgres as postgres
from tests.store._cov95_runtime_schema import (
    CatalogConnection,
    MigrationRecorder,
    migration_callback,
)


@pytest.mark.parametrize("mode", ["", "unknown", "LEGACY"])
def test_unknown_modes_are_rejected_before_database_access(mode):
    connection = CatalogConnection()
    with pytest.raises(StateTableMigrationError, match="invalid state-table mode"):
        set_state_table_mode(connection, "workflow", mode, expected_mode="legacy")
    assert connection.statements == [], "invalid modes must not reach SQL"


@pytest.mark.parametrize("operation", ["status", "backfill", "purge", "retire", "mode"])
def test_unknown_state_kind_cannot_reach_any_maintenance_writer(operation):
    connection = CatalogConnection()
    with pytest.raises(StateTableMigrationError, match="not enabled"):
        if operation == "status":
            state_table_status(connection, "unknown-kind")
        elif operation == "backfill":
            backfill_state_table(connection, "unknown-kind")
        elif operation == "purge":
            purge_legacy_state_rows(connection, "unknown-kind", confirm=True)
        elif operation == "retire":
            retire_legacy_state_indexes(connection, "unknown-kind")
        else:
            set_state_table_mode(
                connection, "unknown-kind", "dual", expected_mode="legacy"
            )
    assert connection.statements == [], "unknown kinds must not reach SQL"


@pytest.mark.parametrize("operation", [backfill_state_table, purge_legacy_state_rows])
@pytest.mark.parametrize(
    "budgets",
    [
        {"batch_size": 0},
        {"batch_size": 1001},
        {"max_batches": 0},
        {"max_batches": 1001},
    ],
)
def test_backfill_and_retirement_require_bounded_budgets(operation, budgets):
    connection = CatalogConnection()
    kwargs = {"confirm": True} if operation is purge_legacy_state_rows else {}
    with pytest.raises(StateTableMigrationError, match="budgets are out of range"):
        operation(connection, "workflow", **budgets, **kwargs)
    assert connection.statements == [], (
        "invalid budgets cannot start a write transaction"
    )


def test_retirement_requires_confirmation_and_autocommit():
    connection = CatalogConnection()
    with pytest.raises(StateTableMigrationError, match="explicit confirmation"):
        purge_legacy_state_rows(connection, "workflow", confirm=False)
    connection.autocommit = False
    with pytest.raises(StateTableMigrationError, match="requires autocommit"):
        retire_legacy_state_indexes(connection, "workflow")
    assert connection.statements == [], "refused retirement must not touch SQL"


def test_missing_mode_metadata_is_not_assumed_legacy():
    connection = CatalogConnection(missing_mode=True)
    with pytest.raises(StateTableMigrationError, match="metadata is missing"):
        state_table_status(connection, "workflow")


def test_mode_compare_and_set_rejects_drift_before_mutation():
    connection = CatalogConnection(mode="legacy")
    with pytest.raises(StateTableMigrationError, match="mode changed"):
        set_state_table_mode(connection, "workflow", "dedicated", expected_mode="dual")


@pytest.mark.parametrize(
    ("current", "target"),
    [("legacy", "dedicated"), ("dedicated", "dual"), ("dedicated", "legacy")],
)
def test_irreversible_mode_transitions_are_rejected(current, target):
    connection = CatalogConnection(mode=current)
    with pytest.raises(StateTableMigrationError, match="unsupported or irreversible"):
        set_state_table_mode(
            connection,
            "workflow",
            target,
            expected_mode=current,
            confirm_dedicated=True,
        )


def test_dedicated_cutover_requires_confirmation():
    connection = CatalogConnection(mode="dual")
    with pytest.raises(StateTableMigrationError, match="explicit confirmation"):
        set_state_table_mode(connection, "workflow", "dedicated", expected_mode="dual")


def test_repeated_mode_selection_returns_status_without_a_write():
    connection = CatalogConnection()
    report = set_state_table_mode(
        connection, "workflow", "legacy", expected_mode="legacy"
    )
    assert report["mode"] == "legacy"
    assert report["legacy_rows"] == report["dedicated_rows"] == 0
    assert report["verification_performed"] is False, (
        "a no-op mode selection cannot claim a full legacy/native verification"
    )


@pytest.mark.parametrize("mode", ["legacy", "dedicated"])
def test_backfill_cannot_run_outside_dual_mode(mode):
    with pytest.raises(StateTableMigrationError, match="backfill requires dual"):
        backfill_state_table(CatalogConnection(mode=mode), "workflow")


def test_validation_deadline_expires_before_reading_unbounded_state(monkeypatch):
    monkeypatch.setattr(state_table_migrate, "STATE_TABLE_VALIDATION_SECONDS", 0)
    connection = CatalogConnection()
    with pytest.raises(StateTableMigrationError, match="validation exceeded"):
        state_table_status(connection, "workflow")
    assert not any("count(" in query for query, _ in connection.statements), (
        "an expired validation budget must not start a table scan"
    )


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"version": []}, "schema version"),
        ({"version": [(999,)]}, "schema version"),
        ({"history": []}, "history differs"),
        ({}, "schema is incomplete"),
    ],
)
def test_maintenance_releases_its_lock_after_schema_validation_refusal(
    monkeypatch, options, message
):
    import psycopg

    connection = CatalogConnection(**options)
    monkeypatch.setattr(psycopg, "connect", lambda *args, **kwargs: connection)
    with pytest.raises(StateTableMigrationError, match=message):
        with state_table_maintenance("postgresql://fake.invalid/postgres"):
            pytest.fail("invalid schema cannot enter the maintenance body")
    assert connection.closed, "the rejected maintenance connection must be closed"
    assert "pg_advisory_unlock_shared" in connection.statements[-1][0], (
        "schema refusal must release the session maintenance lock"
    )


def test_registered_migration_callbacks_use_the_public_cursor_boundary():
    recorder = MigrationRecorder()
    statements_by_version = {}
    for migration in schema_migrations.POSTGRES_SCHEMA_MIGRATIONS:
        if migration.apply is not None:
            before = len(recorder.statements)
            migration.apply(recorder)
            statements_by_version[migration.version] = recorder.statements[before:]
    for version in (4, 5, 6, 9, 10, 13, 14, 15, 16, 17):
        assert statements_by_version[version] == [("SELECT 1", None)], (
            f"migration {version} must retain its published no-op callback"
        )
    assert statements_by_version[11] == [
        ("DROP INDEX IF EXISTS gpu_fault_processor_queue_claim_order", None)
    ]
    assert statements_by_version[12] == [
        ("DROP INDEX IF EXISTS gpu_fault_remote_command_workflow", None)
    ]
    for column in ("not_before", "retry_count", "lane_policy"):
        assert any(
            query.startswith("ALTER TABLE gpu_fault_processor_queue ADD COLUMN")
            and column in query
            for query, _ in statements_by_version[7]
        ), f"retry-schedule migration must still add {column}"
    assert any(
        "CREATE OR REPLACE FUNCTION" in query for query, _ in statements_by_version[8]
    ), "priority-tier migration must update its counter function"


def test_migration_checksum_binds_metadata_callback_and_legacy_pin():
    empty = SchemaMigration(1, "example")
    assert empty.checksum == hashlib.sha256("1\0example\0\0".encode()).hexdigest()
    callback = SchemaMigration(1, "example", apply=migration_callback)
    assert callback.checksum != empty.checksum, (
        "the apply callback must affect the digest"
    )
    assert (
        SchemaMigration(1, "example", ddl_checksum="different").checksum
        != empty.checksum
    )
    assert (
        SchemaMigration(1, "example", legacy_checksum="legacy-pin").checksum
        == "legacy-pin"
    )


@pytest.mark.parametrize(
    ("registry", "message"),
    [
        ((SchemaMigration(2, "not-first"),), "contiguous"),
        (
            (SchemaMigration(1, "duplicate"), SchemaMigration(2, "duplicate")),
            "names must be unique",
        ),
        ((SchemaMigration(1, "bad-checksum", legacy_checksum="short"),), "SHA-256"),
        (
            (SchemaMigration(1, "unversioned-ddl", ddl_checksum="0" * 64),),
            "DDL changed without a new schema migration",
        ),
    ],
)
def test_invalid_migration_registry_is_rejected_without_modifying_published_entries(
    monkeypatch, registry, message
):
    monkeypatch.setattr(schema_migrations, "POSTGRES_SCHEMA_MIGRATIONS", registry)
    with pytest.raises(RuntimeError, match=message):
        schema_migrations.validate_migration_registry()


def test_ddl_checksum_binds_filenames_contents_and_only_ddl_sources(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(schema_migrations, "DDL_ROOT", tmp_path)
    (tmp_path / "ddl_b.py").write_text("second", encoding="utf-8")
    (tmp_path / "ddl_a.py").write_text("first", encoding="utf-8")
    (tmp_path / "unrelated.py").write_text("ignored", encoding="utf-8")
    expected = hashlib.sha256(b"ddl_a.py\0firstddl_b.py\0second").hexdigest()
    assert schema_migrations.postgres_ddl_source_checksum() == expected
    (tmp_path / "ddl_a.py").write_text("changed", encoding="utf-8")
    assert schema_migrations.postgres_ddl_source_checksum() != expected, (
        "DDL changes must change the release's source binding"
    )


@pytest.mark.parametrize(
    ("uri", "options", "message"),
    [
        ("https://example.invalid/archive", {}, "must use s3"),
        ("s3:///archive", {}, "must use s3"),
        ("s3://fake-bucket/archive", {"retention": timedelta(0)}, ">=1 day"),
        ("s3://fake-bucket/archive", {"batch_size": 0}, "must be positive"),
    ],
)
def test_archive_configuration_is_validated_with_a_fake_storage_transport(
    uri, options, message
):
    with pytest.raises(ValueError, match=message):
        ControlRecordArchiver(
            "postgresql://fake.invalid/postgres", uri, s3_client=object(), **options
        )


def test_archive_default_storage_factory_is_replaceable_without_cloud_access(
    monkeypatch,
):
    import boto3

    transport = object()
    services = []

    def fake_client(service):
        services.append(service)
        return transport

    monkeypatch.setattr(boto3, "client", fake_client)
    archiver = ControlRecordArchiver(
        "postgresql://fake.invalid/postgres", "s3://fake-bucket"
    )
    assert archiver.s3 is transport, (
        "the archive must use the provided public client factory"
    )
    assert archiver.bucket == "fake-bucket"
    assert archiver.prefix == ""
    assert services == ["s3"], "no other provider client is needed for archive storage"


def test_missing_pg_configuration_skips_before_the_worker_guard(monkeypatch):
    monkeypatch.delenv("GPU_FAULT_TEST_POSTGRES_URL", raising=False)
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw-example")
    with pytest.raises(pytest.skip.Exception, match="separately allocated"):
        postgres.validated_url()


def test_configured_pg_worker_is_rejected_before_server_access(monkeypatch):
    monkeypatch.setenv(
        "GPU_FAULT_TEST_POSTGRES_URL", "postgresql://fake.invalid/postgres"
    )
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw-example")
    with pytest.raises(pytest.fail.Exception, match="serial -n0"):
        postgres.validated_url()


def test_row_retirement_requires_a_dedicated_authoritative_copy():
    with pytest.raises(StateTableMigrationError, match="requires dedicated"):
        purge_legacy_state_rows(
            CatalogConnection(mode="dual"), "workflow", confirm=True
        )


def test_index_retirement_refusal_restores_the_session_timeout():
    connection = CatalogConnection()
    with pytest.raises(
        StateTableMigrationError, match="completed legacy row retirement"
    ):
        retire_legacy_state_indexes(connection, "workflow")
    assert connection.statements[-1][1] == ("10s",), (
        "a refused retirement must restore the caller's session timeout"
    )
    assert "pg_advisory_unlock_shared" in connection.statements[-2][0]


def test_index_retirement_cannot_drop_indexes_while_legacy_rows_remain():
    connection = CatalogConnection(mode="dedicated", remaining=True)
    connection.mode_row = ("dedicated", 0, None, True, True)
    with pytest.raises(
        StateTableMigrationError, match="legacy control-state rows remain"
    ):
        retire_legacy_state_indexes(connection, "workflow")
    assert connection.statements[-1][1] == ("10s",)


def test_failed_retirement_lock_does_not_unlock_a_lock_it_never_acquired():
    connection = CatalogConnection(fail_query="pg_advisory_lock_shared")
    with pytest.raises(RuntimeError, match="synthetic catalog"):
        retire_legacy_state_indexes(connection, "workflow")
    assert not any(
        "pg_advisory_unlock_shared" in query for query, _ in connection.statements
    ), "failed acquisition must not release another session's lock"
    assert connection.statements[-1][1] == ("10s",), (
        "even lock-acquisition failure must restore session settings"
    )


def test_backfill_rejects_a_legacy_payload_bound_to_the_wrong_key():
    workflow = workflow_request("payload-identity", "incident")
    connection = CatalogConnection(
        mode="dual", records=[("different-key", workflow.model_dump(mode="json"))]
    )
    with pytest.raises(StateTableMigrationError, match="invalid legacy control record"):
        backfill_state_table(connection, "workflow")
