from __future__ import annotations

import argparse
import json
import os
import sqlite3
from dataclasses import dataclass

from gpu_fault.store import PostgresStore


@dataclass(frozen=True)
class MigrationResult:
    objects: int
    links: int


def migrate_sqlite_to_postgres(sqlite_path: str, postgres_url: str) -> MigrationResult:
    source = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    destination = PostgresStore(postgres_url)
    try:
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
                cursor.execute("SELECT count(*) FROM gpu_fault_objects")
                existing_objects = cursor.fetchone()[0]
                cursor.execute("SELECT count(*) FROM gpu_fault_links")
                existing_links = cursor.fetchone()[0]
                if existing_objects or existing_links:
                    raise RuntimeError("PostgreSQL destination is not empty")
                cursor.executemany(
                    """
                    INSERT INTO gpu_fault_objects(kind, key, payload)
                    VALUES (%s, %s, %s::jsonb)
                    """,
                    objects,
                )
                cursor.executemany(
                    """
                    INSERT INTO gpu_fault_links(kind, key, value)
                    VALUES (%s, %s, %s)
                    """,
                    links,
                )
        return MigrationResult(objects=len(objects), links=len(links))
    finally:
        source.close()
        destination.close()


def migrate_postgres_to_postgres(
    source_url: str, destination_url: str
) -> MigrationResult:
    source = PostgresStore(source_url)
    destination = PostgresStore(destination_url)
    try:
        with source._db.cursor() as cursor:
            cursor.execute(
                """
                SELECT kind, key, payload::text
                FROM gpu_fault_objects ORDER BY kind, key
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
                cursor.execute("SELECT count(*) FROM gpu_fault_objects")
                existing_objects = cursor.fetchone()[0]
                cursor.execute("SELECT count(*) FROM gpu_fault_links")
                existing_links = cursor.fetchone()[0]
                if existing_objects or existing_links:
                    raise RuntimeError("PostgreSQL destination is not empty")
                cursor.executemany(
                    """
                    INSERT INTO gpu_fault_objects(kind, key, payload)
                    VALUES (%s, %s, %s::jsonb)
                    """,
                    objects,
                )
                cursor.executemany(
                    """
                    INSERT INTO gpu_fault_links(kind, key, value)
                    VALUES (%s, %s, %s)
                    """,
                    links,
                )
        return MigrationResult(objects=len(objects), links=len(links))
    finally:
        source.close()
        destination.close()


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate a stopped SQLite store to PostgreSQL."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--sqlite-path")
    source.add_argument("--source-postgres-url")
    source.add_argument("--ensure-schema", action="store_true")
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
    parser.add_argument(
        "--postgres-url",
        default=os.getenv("GPU_FAULT_STORE_URL"),
    )
    arguments = parser.parse_args()
    if not arguments.postgres_url:
        parser.error("--postgres-url or GPU_FAULT_STORE_URL is required")
    if arguments.ensure_schema:
        store = PostgresStore(arguments.postgres_url, hot_state_mode="legacy")
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
        store = PostgresStore(arguments.postgres_url, hot_state_mode="dual")
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
        store = PostgresStore(arguments.postgres_url, hot_state_mode="dual")
        try:
            status = store.hot_state_migration_status()
        finally:
            store.close()
        print(json.dumps(status, indent=2, sort_keys=True))
        return
    if arguments.backfill_processor_queue_state:
        store = PostgresStore(arguments.postgres_url)
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
        store = PostgresStore(arguments.postgres_url)
        try:
            status = store.processor_queue_state_status()
        finally:
            store.close()
        print(json.dumps(status, indent=2, sort_keys=True))
        return
    if arguments.processor_queue_count_status:
        store = PostgresStore(arguments.postgres_url)
        try:
            status = store.processor_queue_count_status()
        finally:
            store.close()
        print(json.dumps(status, indent=2, sort_keys=True))
        return
    if arguments.finalize_processor_counter_shards:
        store = PostgresStore(arguments.postgres_url)
        try:
            status = store.finalize_processor_counter_shards()
        finally:
            store.close()
        print(json.dumps(status, indent=2, sort_keys=True))
        return
    if arguments.restore_legacy_processor_counters:
        store = PostgresStore(arguments.postgres_url)
        try:
            status = store.restore_legacy_processor_counters()
        finally:
            store.close()
        print(json.dumps(status, indent=2, sort_keys=True))
        return
    if arguments.processor_counter_shard_status:
        store = PostgresStore(arguments.postgres_url)
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
        store = PostgresStore(arguments.postgres_url, hot_state_mode="dual")
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
        result = migrate_postgres_to_postgres(
            arguments.source_postgres_url,
            arguments.postgres_url,
        )
    print(f"migration complete: objects={result.objects} links={result.links}")


if __name__ == "__main__":
    main()
