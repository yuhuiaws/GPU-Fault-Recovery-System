from __future__ import annotations

import json
import runpy
import sqlite3
import sys
from contextlib import closing, nullcontext
from datetime import datetime, timezone
from pathlib import Path

import pytest

from gpu_fault import state_table_migrate, store_migrate
from gpu_fault.gpu_metrics import GpuMetricLatest, GpuMetricSample, GpuMetricSource
from gpu_fault.store.postgres import index_builder
from tests.store._cov95_migration_support import (
    INCIDENT,
    LINKS,
    OBJECTS,
    WORKFLOW,
    Store,
)

URL = "postgresql://destination.invalid/control"
SOURCE = "postgresql://source.invalid/control"
OPERATIONS = [
    "--backfill-hot-state",
    "--hot-state-status",
    "--backfill-processor-queue-state",
    "--processor-queue-state-status",
    "--processor-queue-count-status",
    "--finalize-processor-counter-shards",
    "--restore-legacy-processor-counters",
    "--processor-counter-shard-status",
    "--purge-legacy-hot-state",
]


@pytest.fixture
def store_factory(monkeypatch):
    import psycopg

    events = []
    stores = []

    def factory(url, **options):
        store = Store(url, events=events, **options)
        stores.append(store)
        return store

    monkeypatch.setattr(store_migrate, "PostgresStore", factory)
    monkeypatch.setattr(
        psycopg,
        "connect",
        lambda *args, **kwargs: pytest.fail(
            "unit test attempted an external PostgreSQL connection"
        ),
    )
    return stores, events


@pytest.mark.parametrize("operation", OPERATIONS)
def test_only_explicit_schema_initialization_can_create_schema(
    store_factory, monkeypatch, capsys, operation
) -> None:
    stores, events = store_factory
    monkeypatch.setattr(
        sys, "argv", ["store-migrate", operation, "--postgres-url", URL]
    )
    store_migrate.main()
    assert stores[0].options["initialize_schema"] is False, (
        "status and data operations must not bootstrap DDL"
    )
    assert stores[0].closed, "the CLI always releases its connection pool"
    assert isinstance(json.loads(capsys.readouterr().out), dict), (
        "operation output remains structured"
    )
    assert events[0][0] == "open" and events[-1] == ("close", URL), (
        "resource lifetime brackets the command"
    )


@pytest.mark.parametrize("operation", OPERATIONS)
def test_data_operation_failure_still_closes_the_store(monkeypatch, operation) -> None:
    events = []
    store = Store(URL, events=events, invalid=True)
    monkeypatch.setattr(store_migrate, "PostgresStore", lambda *args, **kwargs: store)
    monkeypatch.setattr(
        sys, "argv", ["store-migrate", operation, "--postgres-url", URL]
    )
    with pytest.raises(ValueError, match="operation refused"):
        store_migrate.main()
    assert store.closed, "failed migration/status must not leak a pool"


def test_postgres_copy_preserves_control_state_and_closes_in_reverse_order(
    store_factory,
) -> None:
    stores, events = store_factory
    result = store_migrate.migrate_postgres_to_postgres(SOURCE, URL)
    assert (result.objects, result.links) == (2, 1), (
        "copy counts describe logical records"
    )
    assert all(store.closed for store in stores), "both connection pools are closed"
    assert events[-2:] == [("close", URL), ("close", SOURCE)], (
        "destination cleanup does not precede source acquisition"
    )
    assert all(store.options["initialize_schema"] is False for store in stores), (
        "copy cannot initialize either database"
    )
    writes = stores[1].writes
    assert writes[0][1] == [OBJECTS[0]], (
        "workflow must not bypass its dedicated-state write function"
    )
    assert writes[1][1][0][:2] == ("workflow", WORKFLOW.request_id), (
        "native state retains kind and key"
    )
    columns = json.loads(writes[1][1][0][2])
    assert columns["request_id"] == WORKFLOW.request_id, (
        "projected identity is lossless"
    )
    assert (
        columns["status"] == WORKFLOW.status.value
        and "status" not in columns["payload"]
    ), "mutable state is not duplicated in immutable payload"
    assert writes[2][1] == LINKS, "event-to-incident links are retained"


@pytest.mark.parametrize("problem", ["existing", "invalid"])
def test_copy_refuses_nonempty_or_invalid_inputs_before_inserting(
    store_factory, monkeypatch, problem
) -> None:
    stores, events = store_factory

    def factory(url, **options):
        store = Store(
            url, events=events, existing=int(problem == "existing"), **options
        )
        store.invalid_decode = problem == "invalid"
        stores.append(store)
        return store

    monkeypatch.setattr(store_migrate, "PostgresStore", factory)
    with pytest.raises((RuntimeError, ValueError)):
        store_migrate.migrate_postgres_to_postgres(SOURCE, URL)
    assert all(store.closed and not store.writes for store in stores), (
        "refusal cannot leave copied data or an open pool"
    )


def test_source_pool_closes_when_destination_constructor_fails(monkeypatch) -> None:
    source = Store(SOURCE, events=[])

    def factory(url, **options):
        if url == URL:
            raise RuntimeError("destination unavailable")
        return source

    monkeypatch.setattr(store_migrate, "PostgresStore", factory)
    with pytest.raises(RuntimeError, match="destination unavailable"):
        store_migrate.migrate_postgres_to_postgres(SOURCE, URL)
    assert source.closed, (
        "failure before the migration body must still close the source"
    )


def test_source_closes_even_if_destination_close_raises(monkeypatch) -> None:
    stores = []

    def factory(url, **options):
        store = Store(url, events=[], close_error=url == URL, **options)
        stores.append(store)
        return store

    monkeypatch.setattr(store_migrate, "PostgresStore", factory)
    with pytest.raises(RuntimeError, match="destination close failed"):
        store_migrate.migrate_postgres_to_postgres(SOURCE, URL)
    assert all(store.closed for store in stores), (
        "one cleanup error cannot skip the other cleanup"
    )


def legacy_database(path):
    with closing(sqlite3.connect(path)) as database, database:
        database.executescript(
            "CREATE TABLE objects(kind TEXT,key TEXT,payload TEXT);"
            "CREATE TABLE links(kind TEXT,key TEXT,value TEXT);"
        )
        database.executemany("INSERT INTO objects VALUES(?,?,?)", OBJECTS)
        database.executemany("INSERT INTO links VALUES(?,?,?)", LINKS)


def test_legacy_import_uses_an_encoded_read_only_filename(
    tmp_path, store_factory
) -> None:
    path = tmp_path / "legacy?backup.db"
    legacy_database(path)
    stores, _ = store_factory
    result = store_migrate.migrate_sqlite_to_postgres(str(path), URL)
    assert (result.objects, result.links) == (2, 1), (
        "URI metacharacters are filename data, not connection options"
    )
    assert stores[0].closed, "legacy compatibility does not leak the destination"
    with closing(sqlite3.connect(path)) as database:
        assert database.execute("SELECT count(*) FROM objects").fetchone() == (2,), (
            "the legacy source is unchanged"
        )
    assert INCIDENT.incident_id in stores[0].writes[0][1][0], (
        "legacy rows keep their identity"
    )


@pytest.mark.parametrize("use_env", [False, True])
def test_postgres_source_url_can_be_selected_without_echoing_it(
    monkeypatch, capsys, use_env
) -> None:
    calls = []
    monkeypatch.setenv("GPU_FAULT_STORE_URL", URL)
    monkeypatch.setenv("UNIT_SOURCE_DATABASE", SOURCE)
    arguments = (
        ["--source-postgres-url-env", "UNIT_SOURCE_DATABASE"]
        if use_env
        else ["--source-postgres-url", SOURCE]
    )
    monkeypatch.setattr(sys, "argv", ["store-migrate", *arguments])
    monkeypatch.setattr(
        store_migrate,
        "migrate_postgres_to_postgres",
        lambda source, target: calls.append((source, target))
        or store_migrate.MigrationResult(2, 1),
    )
    store_migrate.main()
    output = capsys.readouterr().out
    assert calls == [(SOURCE, URL)], (
        "the selected source and destination are not interchanged"
    )
    assert (
        "objects=2 links=1" in output and SOURCE not in output and URL not in output
    ), "normal output contains counts, not DSNs"


@pytest.mark.parametrize("missing", ["destination", "source-environment"])
def test_missing_connection_reference_fails_before_opening_any_store(
    store_factory, monkeypatch, missing
) -> None:
    stores, _ = store_factory
    monkeypatch.delenv("GPU_FAULT_STORE_URL", raising=False)
    monkeypatch.delenv("UNIT_SOURCE_DATABASE", raising=False)
    arguments = (
        ["--hot-state-status"]
        if missing == "destination"
        else [
            "--source-postgres-url-env",
            "UNIT_SOURCE_DATABASE",
            "--postgres-url",
            URL,
        ]
    )
    monkeypatch.setattr(sys, "argv", ["store-migrate", *arguments])
    with pytest.raises(SystemExit) as error:
        store_migrate.main()
    assert error.value.code == 2 and not stores, (
        "missing connection authority is rejected before I/O"
    )


@pytest.mark.parametrize(
    ("argument", "function", "report", "code"),
    [
        (
            "--build-indexes-concurrently",
            "build_missing_indexes_concurrently",
            {"missing_after": [], "invalid_after": []},
            0,
        ),
        (
            "--build-indexes-concurrently",
            "build_missing_indexes_concurrently",
            {"missing_after": ["index"], "invalid_after": []},
            1,
        ),
        (
            "--build-indexes-concurrently",
            "build_missing_indexes_concurrently",
            {"missing_after": [], "invalid_after": ["index"]},
            1,
        ),
        ("--schema-preflight", "schema_preflight", {"ok": True}, 0),
        ("--schema-preflight", "schema_preflight", {"ok": False}, 1),
    ],
)
def test_explicit_schema_commands_propagate_failed_proof(
    store_factory, monkeypatch, capsys, argument, function, report, code
) -> None:
    import psycopg

    connection = object()
    calls = []
    monkeypatch.setattr(
        psycopg,
        "connect",
        lambda url, **options: calls.append((url, options)) or nullcontext(connection),
    )
    monkeypatch.setattr(
        index_builder,
        function,
        lambda database: report
        if database is connection
        else pytest.fail("wrong database"),
    )
    monkeypatch.setattr(sys, "argv", ["store-migrate", argument, "--postgres-url", URL])
    if code:
        with pytest.raises(SystemExit) as error:
            store_migrate.main()
        assert error.value.code == code, "an incomplete schema/index proof is failure"
    else:
        store_migrate.main()
    assert json.loads(capsys.readouterr().out) == report, (
        "the complete schema report reaches the caller"
    )
    assert calls == [(URL, {"autocommit": True})], (
        "concurrent index building requires autocommit"
    )


@pytest.mark.parametrize(
    ("arguments", "function", "expected"),
    [
        (["--state-table-status"], "state_table_status", {}),
        (
            ["--set-state-table-mode", "dual", "--expected-state-table-mode", "legacy"],
            "set_state_table_mode",
            {"expected_mode": "legacy", "confirm_dedicated": False},
        ),
        (
            [
                "--set-state-table-mode",
                "dedicated",
                "--expected-state-table-mode",
                "dual",
                "--confirm-state-table-change",
                "DEDICATED",
            ],
            "set_state_table_mode",
            {"expected_mode": "dual", "confirm_dedicated": True},
        ),
        (
            [
                "--backfill-state-table",
                "--restart-state-table-backfill",
                "--state-table-batch-size",
                "7",
                "--state-table-max-batches",
                "2",
            ],
            "backfill_state_table",
            {"batch_size": 7, "max_batches": 2, "restart": True},
        ),
        (
            [
                "--purge-legacy-state-table",
                "--confirm-state-table-change",
                "PURGE_LEGACY",
            ],
            "purge_legacy_state",
            {"batch_size": 100, "max_batches": 25, "confirm": True},
        ),
        (
            ["--purge-legacy-state-table"],
            "purge_legacy_state",
            {"batch_size": 100, "max_batches": 25, "confirm": False},
        ),
    ],
)
@pytest.mark.parametrize("kind", ["remote_command", "workflow"])
def test_state_table_cli_preserves_phase_scope_and_confirmation(
    store_factory, monkeypatch, capsys, arguments, function, expected, kind
) -> None:
    database = object()
    calls = []
    monkeypatch.setattr(
        state_table_migrate,
        "state_table_maintenance",
        lambda url: nullcontext(database)
        if url == URL
        else pytest.fail("wrong database"),
    )

    def operation(*args, **kwargs):
        calls.append((args, kwargs))
        return {"verified": True, "kind": kind}

    monkeypatch.setattr(state_table_migrate, function, operation)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "store-migrate",
            *arguments,
            "--state-table-kind",
            kind,
            "--postgres-url",
            URL,
        ],
    )
    store_migrate.main()
    assert calls[0][0][:2] == (database, kind), (
        "phase operates only on its selected table"
    )
    assert calls[0][1] == expected, (
        "CLI cannot drop confirmation, expected mode or batch bounds"
    )
    if function == "set_state_table_mode":
        assert calls[0][0][2] == arguments[1], (
            "the requested destination mode is retained"
        )
    assert json.loads(capsys.readouterr().out) == {"verified": True, "kind": kind}, (
        "structured phase result is preserved"
    )


@pytest.mark.parametrize(
    "arguments",
    [
        ["--state-table-status"],
        ["--set-state-table-mode", "dual", "--state-table-kind", "workflow"],
    ],
)
def test_state_phase_requires_kind_and_expected_mode_before_connection(
    store_factory, monkeypatch, arguments
) -> None:
    monkeypatch.setattr(
        state_table_migrate,
        "state_table_maintenance",
        lambda url: pytest.fail("missing phase authority reached the database"),
    )
    monkeypatch.setattr(
        sys, "argv", ["store-migrate", *arguments, "--postgres-url", URL]
    )
    with pytest.raises(SystemExit) as error:
        store_migrate.main()
    assert error.value.code == 2, (
        "incomplete migration intent is rejected before mutation"
    )


@pytest.mark.parametrize("failure", ["unverified", "migration-error", "database-error"])
def test_state_status_reports_failure_and_redacts_database_exception(
    store_factory, monkeypatch, capsys, failure
) -> None:
    import psycopg

    monkeypatch.setattr(
        state_table_migrate,
        "state_table_maintenance",
        lambda url: nullcontext(object()),
    )

    def status(*args):
        if failure == "migration-error":
            raise state_table_migrate.StateTableMigrationError("state schema mismatch")
        if failure == "database-error":
            raise psycopg.OperationalError("example-sensitive-driver-detail")
        return {"verified": False}

    monkeypatch.setattr(state_table_migrate, "state_table_status", status)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "store-migrate",
            "--state-table-status",
            "--state-table-kind",
            "workflow",
            "--postgres-url",
            URL,
        ],
    )
    with pytest.raises(SystemExit) as error:
        store_migrate.main()
    if failure == "unverified":
        assert error.value.code == 1, "mismatched state is not a successful status"
        assert json.loads(capsys.readouterr().out) == {"verified": False}, (
            "failed verification still has a report"
        )
    elif failure == "migration-error":
        assert str(error.value) == "state schema mismatch", (
            "domain refusal stays actionable"
        )
    else:
        assert (
            str(error.value) == "state-table database operation failed (SQLSTATE None)"
        ), "database diagnostics must not expose a DSN or driver input"


def test_legacy_cli_refuses_a_nonempty_destination(
    tmp_path, store_factory, monkeypatch
) -> None:
    source = tmp_path / "legacy.db"
    legacy_database(source)
    target = Store(URL, events=[], existing=1)
    monkeypatch.setattr(store_migrate, "PostgresStore", lambda *args, **kwargs: target)
    monkeypatch.setattr(
        sys,
        "argv",
        ["store-migrate", "--sqlite-path", str(source), "--postgres-url", URL],
    )
    with pytest.raises(RuntimeError, match="destination is not empty"):
        store_migrate.main()
    assert target.closed and not target.writes, (
        "legacy import cannot merge into existing data"
    )


def test_legacy_source_connection_closes_when_destination_is_unavailable(
    tmp_path, store_factory, monkeypatch
) -> None:
    path = tmp_path / "legacy.db"
    legacy_database(path)
    opened = []
    connect = sqlite3.connect

    def track(*args, **kwargs):
        connection = connect(*args, **kwargs)
        opened.append(connection)
        return connection

    def unavailable(*args, **kwargs):
        raise RuntimeError("destination unavailable")

    monkeypatch.setattr(sqlite3, "connect", track)
    monkeypatch.setattr(store_migrate, "PostgresStore", unavailable)
    with pytest.raises(RuntimeError, match="destination unavailable"):
        store_migrate.migrate_sqlite_to_postgres(str(path), URL)
    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        opened[0].execute("SELECT 1")


def test_diagnostics_does_not_misclassify_non_permission_failure(
    store_factory, monkeypatch
) -> None:
    import psycopg

    class Cursor:
        def execute(self, statement):
            raise psycopg.errors.UndefinedTable("missing extension catalog")

    class Connection:
        def cursor(self):
            return nullcontext(Cursor())

    monkeypatch.setattr(
        psycopg, "connect", lambda *args, **kwargs: nullcontext(Connection())
    )
    with pytest.raises(psycopg.errors.UndefinedTable):
        store_migrate.ensure_diagnostics(URL)


def test_module_entrypoint_refuses_missing_destination_without_connecting(
    store_factory, monkeypatch
) -> None:
    monkeypatch.delenv("GPU_FAULT_STORE_URL", raising=False)
    monkeypatch.setattr(sys, "argv", ["store-migrate", "--hot-state-status"])
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(Path(store_migrate.__file__)), run_name="__main__")
    assert error.value.code == 2, "an incomplete invocation never reaches a database"


@pytest.mark.parametrize("side", ["source", "destination"])
def test_logical_copy_refuses_unmapped_native_telemetry_before_any_insert(
    store_factory, monkeypatch, side
) -> None:
    stores, events = store_factory

    def factory(url, **options):
        store = Store(url, events=events, **options)
        if url == (SOURCE if side == "source" else URL):
            store.hot_status["gpu_metric_latest"]["dedicated"] = 1
        stores.append(store)
        return store

    monkeypatch.setattr(store_migrate, "PostgresStore", factory)
    with pytest.raises(RuntimeError, match="dedicated telemetry state"):
        store_migrate.migrate_postgres_to_postgres(SOURCE, URL)
    assert all(store.closed and not store.writes for store in stores), (
        "logical copy cannot report success after omitting authoritative native data"
    )


@pytest.mark.parametrize(
    "status",
    [
        {},
        {"sample": {}},
        {"sample": {"dedicated": True}},
        {"sample": {"dedicated": -1}},
    ],
)
def test_unknown_native_inventory_does_not_authorize_copy(
    store_factory, monkeypatch, status
) -> None:
    stores, events = store_factory

    def factory(url, **options):
        store = Store(url, events=events, **options)
        store.hot_status = status
        stores.append(store)
        return store

    monkeypatch.setattr(store_migrate, "PostgresStore", factory)
    with pytest.raises(RuntimeError, match="inventory is incomplete"):
        store_migrate.migrate_postgres_to_postgres(SOURCE, URL)
    assert all(store.closed and not store.writes for store in stores), (
        "unknown inventory is not empty inventory"
    )


@pytest.mark.parametrize("legacy_sqlite", [False, True])
def test_legacy_hot_rows_are_backfilled_before_copy_commit(
    tmp_path, store_factory, monkeypatch, legacy_sqlite
) -> None:
    stores, events = store_factory
    value = GpuMetricLatest(
        cluster_id="unit-cluster",
        node_id="unit-node",
        observed_at=datetime(2026, 9, 12, tzinfo=timezone.utc),
        source=GpuMetricSource.DCGM_EXPORTER,
        sample=GpuMetricSample(
            metric_name="temperature", canonical_name="temperature", value=70.0
        ),
    )
    row = ("gpu_metric_latest", "unit-metric", value.model_dump_json())

    def factory(url, **options):
        store = Store(url, events=events, **options)
        if url == SOURCE:
            store.objects.append(row)
        stores.append(store)
        return store

    monkeypatch.setattr(store_migrate, "PostgresStore", factory)
    if legacy_sqlite:
        path = tmp_path / "legacy.db"
        legacy_database(path)
        with closing(sqlite3.connect(path)) as database, database:
            database.execute("INSERT INTO objects VALUES(?,?,?)", row)
        result = store_migrate.migrate_sqlite_to_postgres(str(path), URL)
    else:
        result = store_migrate.migrate_postgres_to_postgres(SOURCE, URL)
    assert result.objects == 3, (
        "the imported hot row remains in the logical record count"
    )
    backfill = ("method", URL, "backfill_hot_state_tables")
    assert events.count(backfill) == 1, (
        "dedicated readers need one in-transaction backfill"
    )
    assert events.index(backfill) < events.index(("close", URL)), (
        "backfill precedes successful return and resource closure"
    )
    assert all(
        store.options.get("hot_state_mode") == "legacy"
        for store in stores
        if store.url == URL
    ), "an empty copy target cannot refuse before the legacy rows are backfilled"
