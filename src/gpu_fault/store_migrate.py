from __future__ import annotations

import argparse
import json
import os
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gpu_fault.models import datetime_json_text
from gpu_fault.store import PostgresStore
from gpu_fault.store.postgres.state_table_payload import (
    STATE_LAYOUTS,
    split_state_record,
)
from gpu_fault.store.postgres.state_table_storage import ENABLED_STATE_KINDS


@dataclass(frozen=True)
class MigrationResult:
    objects: int
    links: int


def _legacy_copy_kinds(store: PostgresStore, *, label: str) -> frozenset[str]:
    status = store.hot_state_migration_status()
    if not status or any(
        type(item.get("dedicated")) is not int or item["dedicated"] < 0
        for item in status.values()
    ):
        raise RuntimeError(f"{label} dedicated hot-state inventory is incomplete")
    if any(item["dedicated"] for item in status.values()):
        raise RuntimeError(
            f"{label} contains dedicated telemetry state that logical copy cannot "
            "preserve; use a reviewed native PostgreSQL backup/restore"
        )
    return frozenset(status)


def _insert_objects(
    cursor: Any,
    rows: Sequence[tuple[str, str, str]],
    store: PostgresStore,
) -> None:
    cursor.executemany(
        "INSERT INTO gpu_fault_objects(kind, key, payload) VALUES (%s, %s, %s::jsonb)",
        (row for row in rows if row[0] not in ENABLED_STATE_KINDS),
    )

    def state_rows() -> Iterator[tuple[str, str, str]]:
        for kind, key, payload in rows:
            if kind in ENABLED_STATE_KINDS:
                columns = split_state_record(
                    STATE_LAYOUTS[kind], store._decode(kind, payload)
                )
                yield kind, key, json.dumps(columns, default=datetime_json_text)

    cursor.executemany(
        "SELECT gpu_fault_put_control_state(%s, %s, %s::jsonb)",
        state_rows(),
    )


def migrate_sqlite_to_postgres(sqlite_path: str, postgres_url: str) -> MigrationResult:
    source_uri = Path(sqlite_path).resolve().as_uri() + "?mode=ro"
    with (
        closing(sqlite3.connect(source_uri, uri=True)) as source,
        closing(
            PostgresStore(
                postgres_url, initialize_schema=False, hot_state_mode="legacy"
            )
        ) as destination,
    ):
        hot_kinds = _legacy_copy_kinds(destination, label="destination")
        objects = source.execute(
            "SELECT kind, key, payload FROM objects ORDER BY kind, key"
        ).fetchall()
        links = source.execute(
            "SELECT kind, key, value FROM links ORDER BY kind, key"
        ).fetchall()

        for kind, _, payload in objects:
            destination._decode(kind, payload)

        with destination._db.transaction():
            with destination._db.cursor() as cursor:
                cursor.execute("SELECT count(*) FROM gpu_fault_control_records")
                existing_objects = cursor.fetchone()[0]
                cursor.execute("SELECT count(*) FROM gpu_fault_links")
                existing_links = cursor.fetchone()[0]
                if existing_objects or existing_links:
                    raise RuntimeError("PostgreSQL destination is not empty")
                _insert_objects(cursor, objects, destination)
                cursor.executemany(
                    """
                    INSERT INTO gpu_fault_links(kind, key, value)
                    VALUES (%s, %s, %s)
                    """,
                    links,
                )
            if any(kind in hot_kinds for kind, _, _ in objects):
                destination.backfill_hot_state_tables()
        return MigrationResult(objects=len(objects), links=len(links))


def migrate_postgres_to_postgres(
    source_url: str, destination_url: str
) -> MigrationResult:
    with (
        closing(PostgresStore(source_url, initialize_schema=False)) as source,
        closing(
            PostgresStore(
                destination_url, initialize_schema=False, hot_state_mode="legacy"
            )
        ) as destination,
    ):
        _legacy_copy_kinds(source, label="source")
        hot_kinds = _legacy_copy_kinds(destination, label="destination")
        with source._db.cursor() as cursor:
            cursor.execute(
                """
                SELECT kind, key, payload::text
                FROM gpu_fault_control_records ORDER BY kind, key
                """
            )
            objects = cursor.fetchall()
            cursor.execute(
                """
                SELECT kind, key, value
                FROM gpu_fault_links ORDER BY kind, key
                """
            )
            links = cursor.fetchall()

        for kind, _, payload in objects:
            destination._decode(kind, payload)

        with destination._db.transaction():
            with destination._db.cursor() as cursor:
                cursor.execute("SELECT count(*) FROM gpu_fault_control_records")
                existing_objects = cursor.fetchone()[0]
                cursor.execute("SELECT count(*) FROM gpu_fault_links")
                existing_links = cursor.fetchone()[0]
                if existing_objects or existing_links:
                    raise RuntimeError("PostgreSQL destination is not empty")
                _insert_objects(cursor, objects, destination)
                cursor.executemany(
                    """
                    INSERT INTO gpu_fault_links(kind, key, value)
                    VALUES (%s, %s, %s)
                    """,
                    links,
                )
            if any(kind in hot_kinds for kind, _, _ in objects):
                destination.backfill_hot_state_tables()
        return MigrationResult(objects=len(objects), links=len(links))


def ensure_diagnostics(postgres_url: str) -> int:
    """Install pg_stat_statements and print the diagnostics report (item K2).

    ``CREATE EXTENSION`` needs the library preloaded and a role allowed to
    create it; on Aurora that is the parameter group and the master user, so
    an insufficient-privilege failure (sqlstate 42501) is reported as such
    instead of a stack trace. Runs on an autocommit connection: the deploy
    Job runs this once and the extension is not transactional state.
    """

    import psycopg

    from gpu_fault.store.postgres.index_builder import (
        database_diagnostics,
        diagnostics_warnings,
    )

    with psycopg.connect(postgres_url, autocommit=True) as connection:
        with connection.cursor() as cursor:
            try:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS pg_stat_statements")
            except psycopg.Error as exc:
                if exc.sqlstate != "42501":
                    raise
                print(
                    "cannot install pg_stat_statements: insufficient privilege "
                    f"({str(exc).strip()}); run --ensure-diagnostics as "
                    "the database master user or create the extension by hand"
                )
                return 1
            diagnostics = database_diagnostics(cursor)
    print(
        json.dumps(
            {
                "diagnostics": diagnostics,
                "warnings": diagnostics_warnings(diagnostics),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _run_state_table_command(
    parser: argparse.ArgumentParser,
    arguments: argparse.Namespace,
) -> None:
    import psycopg

    from gpu_fault.state_table_migrate import (
        StateTableMigrationError,
        backfill_state_table,
        purge_legacy_state,
        set_state_table_mode,
        state_table_maintenance,
        state_table_status,
    )

    if not arguments.state_table_kind:
        parser.error("--state-table-kind is required for state-table operations")
    if arguments.set_state_table_mode and not arguments.expected_state_table_mode:
        parser.error("--expected-state-table-mode is required for a mode transition")
    try:
        with state_table_maintenance(arguments.postgres_url) as connection:
            if arguments.state_table_status:
                report = state_table_status(connection, arguments.state_table_kind)
            elif arguments.set_state_table_mode:
                report = set_state_table_mode(
                    connection,
                    arguments.state_table_kind,
                    arguments.set_state_table_mode,
                    expected_mode=arguments.expected_state_table_mode,
                    confirm_dedicated=arguments.confirm_state_table_change
                    == "DEDICATED",
                )
            elif arguments.backfill_state_table:
                report = backfill_state_table(
                    connection,
                    arguments.state_table_kind,
                    batch_size=arguments.state_table_batch_size,
                    max_batches=arguments.state_table_max_batches,
                    restart=arguments.restart_state_table_backfill,
                )
            else:
                report = purge_legacy_state(
                    connection,
                    arguments.state_table_kind,
                    batch_size=arguments.state_table_batch_size,
                    max_batches=arguments.state_table_max_batches,
                    confirm=arguments.confirm_state_table_change == "PURGE_LEGACY",
                )
    except StateTableMigrationError as exc:
        raise SystemExit(str(exc)) from None
    except psycopg.Error as exc:
        raise SystemExit(
            f"state-table database operation failed (SQLSTATE {exc.sqlstate})"
        ) from None
    print(json.dumps(report, indent=2, sort_keys=True))
    if arguments.state_table_status and report.get("verified") is False:
        raise SystemExit(1)


def _state_table_arguments(
    source: argparse._MutuallyExclusiveGroup,
    parser: argparse.ArgumentParser,
) -> None:
    source.add_argument("--state-table-status", action="store_true")
    source.add_argument("--backfill-state-table", action="store_true")
    source.add_argument(
        "--set-state-table-mode", choices=("legacy", "dual", "dedicated")
    )
    source.add_argument("--purge-legacy-state-table", action="store_true")
    parser.add_argument("--state-table-kind", choices=("remote_command", "workflow"))
    parser.add_argument(
        "--expected-state-table-mode", choices=("legacy", "dual", "dedicated")
    )
    parser.add_argument("--state-table-batch-size", type=int, default=100)
    parser.add_argument("--state-table-max-batches", type=int, default=25)
    parser.add_argument("--restart-state-table-backfill", action="store_true")
    parser.add_argument(
        "--confirm-state-table-change", choices=("DEDICATED", "PURGE_LEGACY")
    )


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect PostgreSQL state or migrate a stopped control-plane store."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--sqlite-path")
    source.add_argument("--source-postgres-url")
    source.add_argument(
        "--source-postgres-url-env",
        metavar="NAME",
        help="read the source PostgreSQL URL from environment variable NAME "
        "(keeps the DSN off the process argv)",
    )
    source.add_argument("--ensure-schema", action="store_true")
    parser.add_argument(
        "--fresh-control-state-mode",
        choices=("legacy", "dedicated"),
        default="legacy",
        help="with --ensure-schema only: the control-state mode a database that "
        "had no schema starts in (an existing database keeps its recorded modes)",
    )
    source.add_argument("--build-indexes-concurrently", action="store_true")
    source.add_argument("--schema-preflight", action="store_true")
    source.add_argument("--ensure-diagnostics", action="store_true")
    source.add_argument("--backfill-hot-state", action="store_true")
    source.add_argument("--hot-state-status", action="store_true")
    source.add_argument(
        "--backfill-processor-queue-state",
        action="store_true",
    )
    source.add_argument(
        "--processor-queue-state-status",
        action="store_true",
    )
    source.add_argument(
        "--processor-queue-count-status",
        action="store_true",
    )
    source.add_argument(
        "--finalize-processor-counter-shards",
        action="store_true",
    )
    source.add_argument(
        "--restore-legacy-processor-counters",
        action="store_true",
    )
    source.add_argument(
        "--processor-counter-shard-status",
        action="store_true",
    )
    source.add_argument("--purge-legacy-hot-state", action="store_true")
    _state_table_arguments(source, parser)
    parser.add_argument(
        "--postgres-url",
        help="explicit destination; otherwise use the projected Store DSN or environment",
    )
    return parser


def resolve_postgres_url(
    parser: argparse.ArgumentParser, arguments: argparse.Namespace
) -> str:
    path = os.getenv("GPU_FAULT_STORE_URL_FILE", "").strip()
    explicit = arguments.postgres_url
    if explicit is not None and not explicit.strip():
        parser.error("--postgres-url must not be empty")
    try:
        # Maintenance needs a successful current read, not the pool's last-good DSN.
        current = (
            Path(path).read_text(encoding="utf-8").strip()
            if path
            else os.getenv("GPU_FAULT_STORE_URL", "").strip()
        )
    except UnicodeError:
        parser.error("GPU_FAULT_STORE_URL_FILE is not valid UTF-8")
    except OSError:
        parser.error("current GPU_FAULT_STORE_URL_FILE credentials are unavailable")
    if path:
        if not current:
            parser.error("current GPU_FAULT_STORE_URL_FILE credentials are unavailable")
        if explicit and explicit != current:
            parser.error("--postgres-url conflicts with GPU_FAULT_STORE_URL_FILE")
        if arguments.source_postgres_url or arguments.source_postgres_url_env:
            # PostgresStore uses the process-wide projection for every pool.
            # It must never silently redirect the explicit source to the target.
            parser.error(
                "logical PostgreSQL copy requires a dedicated process without "
                "GPU_FAULT_STORE_URL_FILE; use explicit source/destination references"
            )
    value = str(explicit or current).strip()
    if not value:
        parser.error(
            "--postgres-url, GPU_FAULT_STORE_URL_FILE or GPU_FAULT_STORE_URL is required"
        )
    return str(value)


def main() -> None:
    parser = _argument_parser()
    arguments = parser.parse_args()
    arguments.postgres_url = resolve_postgres_url(parser, arguments)
    if any(
        (
            arguments.state_table_status,
            arguments.backfill_state_table,
            arguments.set_state_table_mode,
            arguments.purge_legacy_state_table,
        )
    ):
        _run_state_table_command(parser, arguments)
        return
    if arguments.fresh_control_state_mode != "legacy" and not arguments.ensure_schema:
        parser.error("--fresh-control-state-mode requires --ensure-schema")
    if arguments.ensure_schema:
        store = PostgresStore(
            arguments.postgres_url,
            hot_state_mode="legacy",
            fresh_control_state_mode=arguments.fresh_control_state_mode,
        )
        store.close()
        print("schema initialization complete")
        return
    if arguments.build_indexes_concurrently or arguments.schema_preflight:
        # Deliberately not a PostgresStore: opening one validates the schema
        # and fails closed on the very indexes this is about to build.
        import psycopg

        from gpu_fault.store.postgres.index_builder import (
            build_missing_indexes_concurrently,
            schema_preflight,
        )

        with psycopg.connect(arguments.postgres_url, autocommit=True) as connection:
            if arguments.build_indexes_concurrently:
                report = build_missing_indexes_concurrently(connection)
                print(json.dumps(report, indent=2, sort_keys=True))
                if report["missing_after"] or report["invalid_after"]:
                    raise SystemExit(1)
                return
            report = schema_preflight(connection)
            print(json.dumps(report, indent=2, sort_keys=True))
            if not report["ok"]:
                raise SystemExit(1)
            return
    if arguments.ensure_diagnostics:
        raise SystemExit(ensure_diagnostics(arguments.postgres_url))
    if arguments.backfill_hot_state:
        store = PostgresStore(
            arguments.postgres_url, hot_state_mode="dual", initialize_schema=False
        )
        try:
            result = store.backfill_hot_state_tables()
            status = store.hot_state_migration_status()
        finally:
            store.close()
        print(
            json.dumps(
                {"backfill": result, "status": status},
                indent=2,
                sort_keys=True,
            )
        )
        return
    if arguments.hot_state_status:
        store = PostgresStore(
            arguments.postgres_url, hot_state_mode="dual", initialize_schema=False
        )
        try:
            status = store.hot_state_migration_status()
        finally:
            store.close()
        print(json.dumps(status, indent=2, sort_keys=True))
        return
    if arguments.backfill_processor_queue_state:
        store = PostgresStore(arguments.postgres_url, initialize_schema=False)
        try:
            updated = store.backfill_processor_queue_state_columns()
            status = store.processor_queue_state_status()
        finally:
            store.close()
        print(
            json.dumps(
                {"updated": updated, "status": status},
                indent=2,
                sort_keys=True,
            )
        )
        return
    if arguments.processor_queue_state_status:
        store = PostgresStore(arguments.postgres_url, initialize_schema=False)
        try:
            status = store.processor_queue_state_status()
        finally:
            store.close()
        print(json.dumps(status, indent=2, sort_keys=True))
        return
    if arguments.processor_queue_count_status:
        store = PostgresStore(arguments.postgres_url, initialize_schema=False)
        try:
            status = store.processor_queue_count_status()
        finally:
            store.close()
        print(json.dumps(status, indent=2, sort_keys=True))
        return
    if arguments.finalize_processor_counter_shards:
        store = PostgresStore(arguments.postgres_url, initialize_schema=False)
        try:
            status = store.finalize_processor_counter_shards()
        finally:
            store.close()
        print(json.dumps(status, indent=2, sort_keys=True))
        return
    if arguments.restore_legacy_processor_counters:
        store = PostgresStore(arguments.postgres_url, initialize_schema=False)
        try:
            status = store.restore_legacy_processor_counters()
        finally:
            store.close()
        print(json.dumps(status, indent=2, sort_keys=True))
        return
    if arguments.processor_counter_shard_status:
        store = PostgresStore(arguments.postgres_url, initialize_schema=False)
        try:
            status = {
                "mode": store.processor_counter_mode(),
                **store.processor_queue_count_status(),
            }
        finally:
            store.close()
        print(json.dumps(status, indent=2, sort_keys=True))
        return
    if arguments.purge_legacy_hot_state:
        store = PostgresStore(
            arguments.postgres_url, hot_state_mode="dual", initialize_schema=False
        )
        try:
            deleted = store.purge_legacy_hot_state()
        finally:
            store.close()
        print(json.dumps(deleted, indent=2, sort_keys=True))
        return
    if arguments.sqlite_path:
        result = migrate_sqlite_to_postgres(
            arguments.sqlite_path, arguments.postgres_url
        )
    else:
        source_url = arguments.source_postgres_url or os.environ.get(
            arguments.source_postgres_url_env or "", ""
        )
        if not source_url:
            parser.error(
                "--source-postgres-url or a non-empty --source-postgres-url-env "
                "is required"
            )
        result = migrate_postgres_to_postgres(
            source_url,
            arguments.postgres_url,
        )
    print(f"migration complete: objects={result.objects} links={result.links}")


if __name__ == "__main__":
    main()
