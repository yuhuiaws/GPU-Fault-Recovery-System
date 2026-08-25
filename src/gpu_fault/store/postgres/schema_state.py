from __future__ import annotations

from typing import Any, Callable

import json
import os

from gpu_fault.schema_migrations import (
    LATEST_POSTGRES_SCHEMA_VERSION,
    POSTGRES_SCHEMA_MIGRATIONS,
)
from gpu_fault.store.postgres.ddl import (
    create_postgres_schema,
)


POSTGRES_SCHEMA_VERSION = LATEST_POSTGRES_SCHEMA_VERSION


class PostgresSchemaMixin:
    # Attributes supplied by the composed concrete implementation.
    _db: Any
    hot_state_migration_status: Callable[..., Any]
    hot_state_mode: Any

    def _initialize_schema_state(self, initialize_schema: bool) -> None:
        if initialize_schema:
            self._bootstrap_schema()
        else:
            self._validate_existing_schema()
        if self.hot_state_mode == "dedicated":
            self._validate_dedicated_hot_state()

    def _bootstrap_schema(self) -> None:
        with self._db.transaction():
            with self._db.cursor() as cursor:
                bootstrap_timeout_seconds = float(
                    os.getenv(
                        "GPU_FAULT_POSTGRES_BOOTSTRAP_TIMEOUT_SECONDS",
                        "540",
                    )
                )
                bootstrap_lock_timeout_seconds = float(
                    os.getenv(
                        "GPU_FAULT_POSTGRES_BOOTSTRAP_LOCK_TIMEOUT_SECONDS",
                        "60",
                    )
                )
                if (
                    bootstrap_timeout_seconds <= 0
                    or bootstrap_lock_timeout_seconds <= 0
                    or bootstrap_lock_timeout_seconds > bootstrap_timeout_seconds
                ):
                    raise ValueError(
                        "PostgreSQL bootstrap timeouts must be positive "
                        "and lock timeout must not exceed the total timeout"
                    )
                cursor.execute(
                    """
                    SELECT set_config(
                        'statement_timeout', %s, true
                    )
                    """,
                    (f"{int(bootstrap_timeout_seconds * 1000)}ms",),
                )
                cursor.execute(
                    """
                    SELECT set_config('lock_timeout', %s, true)
                    """,
                    (f"{int(bootstrap_lock_timeout_seconds * 1000)}ms",),
                )
                cursor.execute(
                    """
                    SELECT pg_advisory_xact_lock(
                        hashtextextended(%s, 0)
                    )
                    """,
                    ("gpu_fault_schema_bootstrap",),
                )
                self._create_schema(cursor)
                self._record_schema_migrations(cursor)
                cursor.execute(
                    """
                    INSERT INTO gpu_fault_schema_version(
                        singleton, version, updated_at
                    ) VALUES (TRUE, %s, now())
                    ON CONFLICT(singleton) DO UPDATE SET
                        version=excluded.version,
                        updated_at=excluded.updated_at
                    """,
                    (POSTGRES_SCHEMA_VERSION,),
                )

    def _validate_existing_schema(self) -> None:
        with self._db.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    to_regclass('gpu_fault_objects'),
                    to_regclass('gpu_fault_links'),
                    to_regclass('gpu_fault_processor_queue'),
                    to_regclass('gpu_fault_processor_lanes'),
                    to_regclass('gpu_fault_schema_version'),
                    to_regclass('gpu_fault_schema_migrations'),
                    to_regclass('gpu_fault_telemetry_spool'),
                    to_regclass(
                        'gpu_fault_telemetry_spool_path_available'
                    ),
                    to_regclass('gpu_fault_processor_counter_mode'),
                    to_regclass(
                        'gpu_fault_processor_priority_count_shards'
                    ),
                    to_regprocedure(
                        'gpu_fault_processor_queue_notify_pending()'
                    ),
                    to_regprocedure(
                        'gpu_fault_telemetry_spool_notify_available()'
                    ),
                    to_regprocedure(
                        'gpu_fault_processor_priority_count_sync()'
                    )
                """
            )
            required_relations = cursor.fetchone()
        if any(item is None for item in required_relations):
            raise RuntimeError(
                "PostgreSQL schema is not initialized; run "
                "gpu-fault-store-migrate --ensure-schema first"
            )
        trigger_names = (
            "gpu_fault_processor_queue_notify_pending_trigger",
            "gpu_fault_telemetry_spool_notify_available_trigger",
        )
        with self._db.cursor() as cursor:
            cursor.execute(
                """
                SELECT tgname
                FROM pg_trigger
                WHERE tgname=ANY(%s) AND NOT tgisinternal
                """,
                (list(trigger_names),),
            )
            present_triggers = {row[0] for row in cursor.fetchall()}
            cursor.execute(
                """
                SELECT count(*)=3
                FROM pg_trigger
                WHERE tgname IN (
                    'gpu_fault_processor_priority_count_insert',
                    'gpu_fault_processor_priority_count_update',
                    'gpu_fault_processor_priority_count_delete'
                )
                  AND NOT tgisinternal
                """
            )
            fault_counter_triggers_exist = cursor.fetchone()[0]
        missing = set(trigger_names) - present_triggers
        if missing:
            label = (
                "processor queue" if trigger_names[0] in missing else "telemetry spool"
            )
            raise RuntimeError(
                f"PostgreSQL {label} notification trigger is missing; "
                "run gpu-fault-store-migrate --ensure-schema"
            )
        if not fault_counter_triggers_exist:
            raise RuntimeError(
                "PostgreSQL fault counter shard triggers are missing; "
                "run gpu-fault-store-migrate --ensure-schema"
            )
        with self._db.cursor() as cursor:
            cursor.execute(
                """
                SELECT version
                FROM gpu_fault_schema_version
                WHERE singleton=TRUE
                """
            )
            row = cursor.fetchone()
            self._validate_schema_migrations(cursor)
        actual_schema_version = row[0] if row else None
        if actual_schema_version != POSTGRES_SCHEMA_VERSION:
            raise RuntimeError(
                "PostgreSQL schema version mismatch: expected "
                f"{POSTGRES_SCHEMA_VERSION}, got {actual_schema_version}; "
                "run gpu-fault-store-migrate --ensure-schema"
            )

    def _validate_dedicated_hot_state(self) -> None:
        with self._db.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    to_regclass('gpu_fault_gpu_metric_latest'),
                    to_regclass('gpu_fault_gpu_metrics_batches'),
                    to_regclass('gpu_fault_attempt_observations'),
                    to_regclass('gpu_fault_training_progress')
                """
            )
            dedicated_relations = cursor.fetchone()
        if any(item is None for item in dedicated_relations):
            raise RuntimeError(
                "dedicated hot-state tables are not initialized; run "
                "gpu-fault-store-migrate --ensure-schema and "
                "--backfill-hot-state first"
            )
        unsafe = {
            kind: item["missing_or_mismatched"]
            for kind, item in self.hot_state_migration_status().items()
            if item["missing_or_mismatched"]
        }
        if unsafe:
            raise RuntimeError(
                "dedicated hot-state backfill is incomplete: "
                + json.dumps(unsafe, sort_keys=True)
            )

    @staticmethod
    def _record_schema_migrations(cursor) -> None:
        for migration in POSTGRES_SCHEMA_MIGRATIONS:
            cursor.execute(
                """
                SELECT name, checksum
                FROM gpu_fault_schema_migrations
                WHERE version=%s
                """,
                (migration.version,),
            )
            row = cursor.fetchone()
            if row is not None and row != (
                migration.name,
                migration.checksum,
            ):
                raise RuntimeError(
                    "PostgreSQL schema migration checksum mismatch "
                    f"at version {migration.version}"
                )
            if row is None and migration.apply is not None:
                migration.apply(cursor)
            cursor.execute(
                """
                INSERT INTO gpu_fault_schema_migrations(
                    version, name, checksum, applied_at
                ) VALUES (%s, %s, %s, now())
                ON CONFLICT(version) DO NOTHING
                """,
                (
                    migration.version,
                    migration.name,
                    migration.checksum,
                ),
            )

    @staticmethod
    def _validate_schema_migrations(cursor) -> None:
        cursor.execute(
            """
            SELECT version, name, checksum
            FROM gpu_fault_schema_migrations
            ORDER BY version
            """
        )
        actual = cursor.fetchall()
        expected = [
            (
                migration.version,
                migration.name,
                migration.checksum,
            )
            for migration in POSTGRES_SCHEMA_MIGRATIONS
        ]
        if actual != expected:
            raise RuntimeError(
                "PostgreSQL schema migration history mismatch: "
                "run gpu-fault-store-migrate --ensure-schema"
            )

    @staticmethod
    def _create_schema(cursor) -> None:
        """Apply the latest idempotent PostgreSQL schema."""
        create_postgres_schema(cursor)
