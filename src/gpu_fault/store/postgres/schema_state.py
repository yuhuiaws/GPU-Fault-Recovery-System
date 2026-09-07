from __future__ import annotations

from typing import Any, Callable

import os
import re
from pathlib import Path

from gpu_fault.schema_migrations import (
    LATEST_POSTGRES_SCHEMA_VERSION,
    POSTGRES_SCHEMA_MIGRATIONS,
)
from gpu_fault.store.postgres.ddl import (
    declared_index_names,
    create_postgres_schema,
)


POSTGRES_SCHEMA_VERSION = LATEST_POSTGRES_SCHEMA_VERSION

# ``pg_trigger.tgtype`` bits (utils/reltrigger.h). A statement-level AFTER
# trigger sets only its event bit; ROW / BEFORE / INSTEAD stay clear.
_TGTYPE_AFTER_STATEMENT_INSERT = 1 << 2
_TGTYPE_AFTER_STATEMENT_DELETE = 1 << 3
_TGTYPE_AFTER_STATEMENT_UPDATE = 1 << 4

_COUNTER_TRIGGER_TABLE = "gpu_fault_processor_queue"

# What the DDL declares for the six queue counter triggers (F-D10):
# name -> (function, tgtype, OLD transition table, NEW transition table).
# A counter trigger recreated ``FOR EACH ROW``, without its transition table,
# or bound to the other counter function is enabled and present -- and drifts
# the admission counters on every write. The DDL creates these ``IF NOT
# EXISTS`` so ``--ensure-schema`` cannot repair a present-but-wrong one.
_COUNTER_TRIGGERS: dict[str, tuple[str, int, str | None, str | None]] = {
    "gpu_fault_processor_queue_count_insert": (
        "gpu_fault_processor_queue_count_sync",
        _TGTYPE_AFTER_STATEMENT_INSERT,
        None,
        "added",
    ),
    "gpu_fault_processor_queue_count_update": (
        "gpu_fault_processor_queue_count_sync",
        _TGTYPE_AFTER_STATEMENT_UPDATE,
        "removed",
        "added",
    ),
    "gpu_fault_processor_queue_count_delete": (
        "gpu_fault_processor_queue_count_sync",
        _TGTYPE_AFTER_STATEMENT_DELETE,
        "removed",
        None,
    ),
    "gpu_fault_processor_priority_count_insert": (
        "gpu_fault_processor_priority_count_sync",
        _TGTYPE_AFTER_STATEMENT_INSERT,
        None,
        "added",
    ),
    "gpu_fault_processor_priority_count_update": (
        "gpu_fault_processor_priority_count_sync",
        _TGTYPE_AFTER_STATEMENT_UPDATE,
        "removed",
        "added",
    ),
    "gpu_fault_processor_priority_count_delete": (
        "gpu_fault_processor_priority_count_sync",
        _TGTYPE_AFTER_STATEMENT_DELETE,
        "removed",
        None,
    ),
}

_COUNTER_FUNCTIONS = (
    "gpu_fault_processor_queue_count_sync",
    "gpu_fault_processor_priority_count_sync",
)

_TRIGGER_FUNCTION_BODY = re.compile(
    r"CREATE\s+OR\s+REPLACE\s+FUNCTION\s+(\w+)\(\)\s+RETURNS\s+trigger\s+"
    r"LANGUAGE\s+plpgsql\s+AS\s+\$\$(.*?)\$\$",
    re.DOTALL,
)


def declared_trigger_function_bodies() -> dict[str, str]:
    """The plpgsql body of every trigger function the DDL declares, verbatim.

    Read from the DDL source like ``declared_index_names`` so the schema check
    cannot drift from the DDL: Postgres stores the body between the dollar
    quotes character for character in ``pg_proc.prosrc``, and ``CREATE OR
    REPLACE`` on every ``--ensure-schema`` keeps a live database on exactly
    this text -- so anything else is a body somebody replaced by hand.
    """

    bodies: dict[str, str] = {}
    for path in sorted(Path(__file__).parent.glob("ddl*.py")):
        for name, body in _TRIGGER_FUNCTION_BODY.findall(
            path.read_text(encoding="utf-8")
        ):
            bodies[name] = body
    return bodies


class PostgresSchemaMixin:
    # Attributes supplied by the composed concrete implementation.
    _db: Any
    hot_state_backfill_gaps: Callable[..., Any]
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
        self._validate_declared_indexes()
        self._validate_triggers_enabled()
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
        # Present -- now check they are the triggers the DDL declares (F-D10).
        self._validate_counter_trigger_definitions()
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

    def _validate_declared_indexes(self) -> None:
        """Every index the DDL declares must exist (F-J3, three-step method).

        The DDL runs in one transaction and therefore cannot ``CREATE INDEX
        CONCURRENTLY``; large indexes are built by an operator first and the
        DDL only declares them ``IF NOT EXISTS``. Without this check a skipped
        build degraded silently into a sequential scan per dispatcher tick.
        """

        expected = declared_index_names()
        with self._db.cursor() as cursor:
            cursor.execute(
                "SELECT indexname FROM pg_indexes WHERE indexname = ANY(%s)",
                (sorted(expected),),
            )
            present = {row[0] for row in cursor.fetchall()}
        missing = sorted(expected - present)
        if missing:
            raise RuntimeError(
                "PostgreSQL indexes are missing: "
                + ", ".join(missing)
                + "; build them (CREATE INDEX CONCURRENTLY on a live database) "
                "and run gpu-fault-store-migrate --ensure-schema"
            )

    def _validate_triggers_enabled(self) -> None:
        """A disabled gpu-fault trigger silently drifts the queue counters
        (F-D10); the schema check treats it like a missing one."""

        with self._db.cursor() as cursor:
            cursor.execute(
                """
                SELECT tgname
                FROM pg_trigger
                WHERE tgname LIKE 'gpu\\_fault\\_%'
                  AND NOT tgisinternal
                  AND tgenabled = 'D'
                ORDER BY tgname
                """
            )
            disabled = [row[0] for row in cursor.fetchall()]
        if disabled:
            raise RuntimeError(
                "PostgreSQL triggers are disabled: "
                + ", ".join(disabled)
                + "; re-enable them before starting the control plane"
            )

    def _validate_counter_trigger_definitions(self) -> None:
        """Every queue counter trigger must be the one the DDL declares, and
        must drive the function body the DDL declares (F-D10).

        ``tgenabled`` alone passes a trigger that was recreated per row, lost
        its transition table, or was pointed at the other counter function;
        each drifts the counters silently and the ingress then 429s fault
        events by the depth of the drift.
        """

        with self._db.cursor() as cursor:
            cursor.execute(
                """
                SELECT t.tgname, c.relname, p.proname, t.tgtype,
                       t.tgoldtable, t.tgnewtable
                FROM pg_trigger t
                JOIN pg_class c ON c.oid = t.tgrelid
                JOIN pg_proc p ON p.oid = t.tgfoid
                WHERE t.tgname = ANY(%s) AND NOT t.tgisinternal
                """,
                (sorted(_COUNTER_TRIGGERS),),
            )
            actual = {
                row[0]: (row[1], row[2], int(row[3]), row[4], row[5])
                for row in cursor.fetchall()
            }
            cursor.execute(
                "SELECT proname, prosrc FROM pg_proc WHERE proname = ANY(%s)",
                (list(_COUNTER_FUNCTIONS),),
            )
            bodies = {row[0]: row[1] for row in cursor.fetchall()}
        problems: list[str] = []
        for name, (function, tgtype, old_table, new_table) in sorted(
            _COUNTER_TRIGGERS.items()
        ):
            expected = (_COUNTER_TRIGGER_TABLE, function, tgtype, old_table, new_table)
            found = actual.get(name)
            if found is None:
                problems.append(f"{name} is missing")
            elif found != expected:
                problems.append(
                    f"{name} is defined as (table, function, tgtype, old, new)="
                    f"{found!r}, expected {expected!r}"
                )
        declared = declared_trigger_function_bodies()
        for function in _COUNTER_FUNCTIONS:
            if function not in declared:
                problems.append(f"{function} is not declared by the DDL source")
            elif function not in bodies:
                problems.append(f"{function} is missing")
            elif bodies[function] != declared[function]:
                problems.append(f"{function} body differs from the DDL")
        if problems:
            raise RuntimeError(
                "PostgreSQL queue counter triggers drifted from the DDL: "
                + "; ".join(problems)
                + "; drop the drifted trigger and run "
                "gpu-fault-store-migrate --ensure-schema"
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
        # EXISTS per kind, not the counting status: this runs on every worker
        # process start (store review 2026-09-07, item J).
        unsafe = sorted(
            kind for kind, gap in self.hot_state_backfill_gaps().items() if gap
        )
        if unsafe:
            raise RuntimeError(
                "dedicated hot-state backfill is incomplete for: "
                + ", ".join(unsafe)
                + "; run gpu-fault-store-migrate --hot-state-status "
                "and --backfill-hot-state"
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
