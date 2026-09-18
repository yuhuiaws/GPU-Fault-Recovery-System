"""Serial migration guards on owned rows; schema and migration sources stay unchanged."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import UTC, datetime, timedelta

import pytest

from gpu_fault.control_record_archive import ArchiveSafetyError, ControlRecordArchiver
from gpu_fault.models import WorkflowStatus
from gpu_fault.schema_migrations import POSTGRES_SCHEMA_MIGRATIONS
from gpu_fault.state_table_migrate import (
    StateTableMigrationError,
    backfill_state_table,
    purge_legacy_state,
    purge_legacy_state_rows,
    set_state_table_mode,
    state_table_status,
)
from gpu_fault.store import PostgresStore
from gpu_fault.store.postgres.index_builder import schema_preflight
from tests._builders import fault_incident, workflow_request
from tests.store import _cov95_runtime_postgres as postgres
from tests.store._postgres_processor_claim_support import postgres_store_instance
from tests.store.test_postgres_workflow_state_tables import select_mode


@pytest.fixture
def database(monkeypatch):
    import psycopg

    url = postgres.validated_url()
    monkeypatch.setenv("GPU_FAULT_POSTGRES_HOT_STATE_MODE", "legacy")
    with closing(postgres_store_instance()) as setup:
        store = next(setup)
        with psycopg.connect(url, autocommit=True) as connection:
            yield store, connection


@pytest.mark.parametrize(
    ("current", "target", "expected", "message"),
    [
        ("legacy", "dual", "dual", "mode changed"),
        ("legacy", "dedicated", "legacy", "unsupported or irreversible"),
        ("dual", "dedicated", "dual", "explicit confirmation"),
        ("dedicated", "legacy", "dedicated", "unsupported or irreversible"),
        ("dedicated", "dual", "dedicated", "unsupported or irreversible"),
    ],
)
def test_mode_refusal_preserves_catalog_state(
    database, current, target, expected, message
):
    _, connection = database
    select_mode(connection, "workflow", current)
    before = state_table_status(connection, "workflow")
    with pytest.raises(StateTableMigrationError, match=message):
        set_state_table_mode(connection, "workflow", target, expected_mode=expected)
    assert state_table_status(connection, "workflow") == before, (
        "a refused transition cannot advance migration mode or checkpoint state"
    )


def test_backfill_resumes_then_restarts_only_on_explicit_request(database):
    store, connection = database
    values = [
        workflow_request(name, f"incident-{name}", status=WorkflowStatus.SUCCEEDED)
        for name in ("a-workflow", "b-workflow")
    ]
    for value in values:
        store.save_workflow(value)
    select_mode(connection, "workflow", "dual")
    partial = backfill_state_table(connection, "workflow", batch_size=1, max_batches=1)
    assert partial["copied"] == 1
    assert partial["status"]["backfill_complete"] is False, (
        "one full batch is not proof that the input scan reached its end"
    )
    resumed = backfill_state_table(connection, "workflow", batch_size=1, max_batches=4)
    assert resumed["copied"] == 1, (
        "resume must not copy the first checkpointed row twice"
    )
    assert resumed["status"]["verified"] is True
    assert backfill_state_table(connection, "workflow")["copied"] == 0
    restarted = backfill_state_table(connection, "workflow", restart=True)
    assert restarted["copied"] == 2, "explicit restart must scan from the first key"
    assert restarted["status"]["verified"] is True
    assert [store.get_workflow(value.request_id) for value in values] == values


def test_invalid_legacy_payload_rolls_back_backfill_progress(database):
    store, connection = database
    value = workflow_request("invalid-workflow", "incident-example")
    store.save_workflow(value)
    select_mode(connection, "workflow", "dual")
    with connection.transaction(force_rollback=True):
        connection.execute(
            "UPDATE gpu_fault_objects SET payload=payload || %s::jsonb "
            "WHERE kind='workflow' AND key=%s",
            (json.dumps({"unexpected_legacy_field": True}), value.request_id),
        )
        query = (
            "SELECT backfill_after_key,backfill_complete "
            "FROM gpu_fault_control_state_modes WHERE kind='workflow'"
        )
        before = connection.execute(query).fetchone()
        with pytest.raises(
            StateTableMigrationError, match="invalid legacy control record"
        ):
            backfill_state_table(connection, "workflow", restart=True)
        assert connection.execute(query).fetchone() == before, (
            "a rejected payload must not advance the durable resume checkpoint"
        )
    assert store.get_workflow(value.request_id) == value, (
        "the test corruption must be rolled back before the next Store operation"
    )


def test_legacy_retirement_is_bounded_and_keeps_native_records(database):
    store, connection = database
    values = [
        workflow_request(
            f"flow-{index}", f"incident-{index}", status=WorkflowStatus.SUCCEEDED
        )
        for index in range(3)
    ]
    for value in values:
        store.save_workflow(value)
    select_mode(connection, "workflow", "dedicated")
    partial = purge_legacy_state(
        connection, "workflow", confirm=True, batch_size=1, max_batches=1
    )
    assert partial["deleted"] == 1
    assert partial["has_more"] is True
    assert "retired_indexes" not in partial, (
        "indexes must remain while legacy rows still exist"
    )
    assert partial["status"]["legacy_purged"] is False
    completed = purge_legacy_state_rows(
        connection, "workflow", confirm=True, batch_size=1, max_batches=4
    )
    assert completed["deleted"] == 2
    assert completed["has_more"] is False
    assert completed["status"]["legacy_purged"] is True
    assert completed["status"]["dedicated_rows"] == 3
    assert [store.get_workflow(value.request_id) for value in values] == values
    assert purge_legacy_state_rows(connection, "workflow", confirm=True)["deleted"] == 0


@pytest.mark.parametrize("drift", ["future-version", "checksum"])
def test_release_preflight_refuses_migration_history_drift_and_restores_read_mode(
    database, drift
):
    _, connection = database
    migration = POSTGRES_SCHEMA_MIGRATIONS[-1]
    target_version = (
        migration.version + 1 if drift == "future-version" else migration.version
    )
    before_mode = connection.execute("SHOW default_transaction_read_only").fetchone()
    try:
        connection.execute(
            "UPDATE gpu_fault_schema_migrations SET version=%s,checksum=%s WHERE version=%s",
            (
                target_version,
                "0" * 64 if drift == "checksum" else migration.checksum,
                migration.version,
            ),
        )
        report = schema_preflight(connection)
        assert report["ok"] is False, (
            "a different migration history cannot pass readiness"
        )
        assert report["schema_version"]["history_ok"] is False
        assert any("history" in reason for reason in report["blocking_reasons"]), (
            "the preflight must identify the invalid history rather than hide its cause"
        )
        if drift == "future-version":
            assert any("ahead" in reason for reason in report["blocking_reasons"]), (
                "a wheel older than the recorded schema must fail closed"
            )
        assert (
            connection.execute("SHOW default_transaction_read_only").fetchone()
            == before_mode
        )
    finally:
        connection.execute(
            "UPDATE gpu_fault_schema_migrations SET version=%s,name=%s,checksum=%s "
            "WHERE version=%s",
            (migration.version, migration.name, migration.checksum, target_version),
        )
    assert connection.execute(
        "SELECT name,checksum FROM gpu_fault_schema_migrations WHERE version=%s",
        (migration.version,),
    ).fetchone() == (migration.name, migration.checksum), (
        "the owned test database must retain the original migration history after refusal"
    )


@pytest.mark.parametrize("concurrency", ["0", "9", "2"])
def test_completion_concurrency_cannot_exceed_the_owned_pool(
    database, monkeypatch, concurrency
):
    monkeypatch.setenv(
        "GPU_FAULT_PROCESSOR_COMPLETION_CLUSTER_CONCURRENCY", concurrency
    )
    with pytest.raises(ValueError, match="must be between"):
        PostgresStore(
            postgres.validated_url(),
            initialize_schema=False,
            hot_state_mode="legacy",
            pool_max_size=1,
        )


@pytest.mark.parametrize(("total", "lock"), [("0", "60"), ("30", "60"), ("540", "0")])
def test_bootstrap_timeout_refusal_closes_the_configured_completion_executor(
    database, monkeypatch, total, lock
):
    from psycopg_pool import ConnectionPool

    shutdowns, closes = [], []
    shutdown = ThreadPoolExecutor.shutdown
    close = ConnectionPool.close

    def record_shutdown(executor, wait=True, *, cancel_futures=False):
        shutdowns.append((wait, cancel_futures))
        return shutdown(executor, wait=wait, cancel_futures=cancel_futures)

    def record_close(pool, *args, **kwargs):
        closes.append(True)
        return close(pool, *args, **kwargs)

    monkeypatch.setattr(ThreadPoolExecutor, "shutdown", record_shutdown)
    monkeypatch.setattr(ConnectionPool, "close", record_close)
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_COMPLETION_CLUSTER_CONCURRENCY", "2")
    monkeypatch.setenv("GPU_FAULT_POSTGRES_BOOTSTRAP_TIMEOUT_SECONDS", total)
    monkeypatch.setenv("GPU_FAULT_POSTGRES_BOOTSTRAP_LOCK_TIMEOUT_SECONDS", lock)
    with pytest.raises(ValueError, match="bootstrap timeouts"):
        PostgresStore(
            postgres.validated_url(), hot_state_mode="legacy", pool_max_size=2
        )
    assert shutdowns == [(True, True)], "initialization failure must drain its executor"
    assert closes == [True], "initialization failure must close its owned database pool"


def test_optional_pool_settings_can_use_driver_defaults(database, monkeypatch):
    for name in (
        "GPU_FAULT_POSTGRES_STATEMENT_TIMEOUT_SECONDS",
        "GPU_FAULT_POSTGRES_POOL_MAX_IDLE_SECONDS",
        "GPU_FAULT_POSTGRES_POOL_MAX_LIFETIME_SECONDS",
    ):
        monkeypatch.setenv(name, "0")
    with closing(
        PostgresStore(
            postgres.validated_url(), initialize_schema=False, hot_state_mode="legacy"
        )
    ) as store:
        assert store.list_workflows(set()) == [], "empty status scope must remain empty"
        assert store.list_notifications(limit=0) == [], (
            "zero scan budget must return no rows"
        )


class ArchiveSink:
    def __init__(self, on_put=None):
        self.puts = []
        self.on_put = on_put

    def put_object(self, **kwargs):
        self.puts.append(kwargs)
        if self.on_put is not None:
            self.on_put()


def test_archive_missing_incident_cannot_publish_an_empty_bundle(database):
    store, _ = database
    sink = ArchiveSink()
    archive = ControlRecordArchiver(
        postgres.validated_url(), "s3://fake-bucket", s3_client=sink, store=store
    )
    with pytest.raises(ArchiveSafetyError, match="does not exist"):
        archive.archive_one("missing")
    assert sink.puts == [], "a missing incident cannot produce archive evidence"


def test_archive_prefixless_legacy_incident_without_links_can_be_retired(database):
    store, connection = database
    old = datetime.now(UTC) - timedelta(days=366)
    incident = fault_incident(
        "archive-no-links", "source-event", created_at=old, updated_at=old
    )
    connection.execute(
        "INSERT INTO gpu_fault_objects(kind,key,payload) VALUES ('incident',%s,%s::jsonb)",
        (incident.incident_id, incident.model_dump_json()),
    )
    sink = ArchiveSink()
    archive = ControlRecordArchiver(
        postgres.validated_url(), "s3://fake-bucket", s3_client=sink, store=store
    )
    uri = archive.archive_one(incident.incident_id)
    assert uri.startswith("s3://fake-bucket/archive-no-links-"), (
        "a prefixless archive URI must retain the incident-bound object name"
    )
    assert len(sink.puts) == 1, "one exact incident bundle must be uploaded"
    assert store.get_incident_by_event(incident.event_id) is None
    assert connection.execute(
        "SELECT count(*) FROM gpu_fault_objects WHERE kind='incident' AND key=%s",
        (incident.incident_id,),
    ).fetchone() == (0,), "retirement must remove the archived unlinked incident"


def test_archive_upload_race_keeps_the_new_incident_state(database):
    store, _ = database
    old = datetime.now(UTC) - timedelta(days=366)
    incident = fault_incident(
        "archive-drift", "source-event", created_at=old, updated_at=old
    )
    store.save_incident(incident)
    changed = incident.model_copy(update={"reasons": ["new evidence after snapshot"]})
    sink = ArchiveSink(lambda: store.save_incident(changed, expected=incident))
    archive = ControlRecordArchiver(
        postgres.validated_url(), "s3://fake-bucket", s3_client=sink, store=store
    )
    with pytest.raises(ArchiveSafetyError, match="changed after archive upload"):
        archive.archive_one(incident.incident_id)
    assert len(sink.puts) == 1, "the race occurs after an immutable bundle upload"
    assert store.get_incident(incident.incident_id) == changed, (
        "an uploaded old snapshot cannot authorize deleting newer incident state"
    )
