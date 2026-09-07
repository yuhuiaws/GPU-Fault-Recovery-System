"""Online index builds and the read-only schema preflight for a release.

The three-step method (F-J3): operators build a new index ``CONCURRENTLY``
before the rolling upgrade, the transactional DDL then finds it already
present, and the schema check fails a replica closed if it is missing. This
module is the operator side of that method plus the gate a FULL release runs
before it rolls anything: schema version, index presence and validity, and
in-flight safety workflows written before ``safety_only`` existed. The report
also carries the server's diagnostic settings (item K2) as non-blocking
warnings.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from gpu_fault.store.postgres import ddl

from gpu_fault.schema_migrations import (
    LATEST_POSTGRES_SCHEMA_VERSION,
    POSTGRES_SCHEMA_MIGRATIONS,
)
from gpu_fault.store.postgres.ddl import declared_index_names

# The DDL source is checksummed into the migration registry, so the statement
# scraper lives here, next to its only consumer, rather than in ddl.py.
_CREATE_INDEX_STATEMENT = re.compile(
    r'"""\s*(CREATE\s+(?:UNIQUE\s+)?INDEX\s+IF\s+NOT\s+EXISTS\s+(\w+)[\s\S]*?)"""'
)


def declared_index_statements() -> dict[str, str]:
    """Every declared ``CREATE INDEX IF NOT EXISTS`` statement, by index name.

    Read from the DDL source like ``declared_index_names`` so the online
    builder (``CREATE INDEX CONCURRENTLY``, which the transactional DDL cannot
    issue) always builds exactly what the schema check will demand.
    """

    statements: dict[str, str] = {}
    for path in sorted(Path(ddl.__file__).parent.glob("ddl*.py")):
        for statement, name in _CREATE_INDEX_STATEMENT.findall(
            path.read_text(encoding="utf-8")
        ):
            if "{" in statement:
                raise RuntimeError(
                    f"index {name} is declared with an interpolated statement; "
                    "the online builder needs a literal one"
                )
            statements[name] = " ".join(statement.split())
    return statements


def index_health(connection: Any) -> list[dict[str, Any]]:
    """Presence and validity of every declared index, sorted by name."""

    names = sorted(declared_index_names())
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT c.relname, i.indisvalid
            FROM pg_class AS c
            JOIN pg_index AS i ON i.indexrelid = c.oid
            WHERE c.relname = ANY(%s)
            """,
            (names,),
        )
        found = {row[0]: bool(row[1]) for row in cursor.fetchall()}
    return [
        {"name": name, "present": name in found, "valid": found.get(name, False)}
        for name in names
    ]


def build_missing_indexes_concurrently(connection: Any) -> dict[str, Any]:
    """Build every declared index that is missing or invalid, online.

    ``connection`` must be in autocommit mode: ``CREATE INDEX CONCURRENTLY``
    cannot run inside a transaction. An invalid index (a failed earlier
    concurrent build) is dropped and rebuilt. Returns what was done and the
    indexes still not valid afterwards.
    """

    if not getattr(connection, "autocommit", False):
        raise RuntimeError("concurrent index builds need an autocommit connection")
    statements = declared_index_statements()
    built: list[str] = []
    dropped: list[str] = []
    with connection.cursor() as cursor:
        for row in index_health(connection):
            name = row["name"]
            if row["present"] and row["valid"]:
                continue
            if row["present"] and not row["valid"]:
                cursor.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {name}")
                dropped.append(name)
            statement = statements[name].replace(
                "INDEX IF NOT EXISTS", "INDEX CONCURRENTLY IF NOT EXISTS", 1
            )
            cursor.execute(statement)
            built.append(name)
    after = index_health(connection)
    return {
        "built": built,
        "dropped_invalid": dropped,
        "present": [row["name"] for row in after if row["present"]],
        "invalid_after": [
            row["name"] for row in after if row["present"] and not row["valid"]
        ],
        "missing_after": [row["name"] for row in after if not row["present"]],
    }


def schema_preflight(connection: Any) -> dict[str, Any]:
    """Read-only readiness report for a FULL release against this database."""

    reasons: list[str] = []
    with connection.cursor() as cursor:
        # Read-only for the duration of the report only; the same autocommit
        # session may go on to build indexes.
        cursor.execute("SET default_transaction_read_only = on")
        try:
            return _schema_preflight_report(connection, cursor, reasons)
        finally:
            cursor.execute("RESET default_transaction_read_only")


def _schema_preflight_report(
    connection: Any, cursor: Any, reasons: list[str]
) -> dict[str, Any]:
    cursor.execute("SELECT to_regclass('gpu_fault_schema_migrations') IS NOT NULL")
    has_history = bool(cursor.fetchone()[0])
    registered = 0
    history_ok = True
    if has_history:
        cursor.execute(
            "SELECT version, name, checksum FROM gpu_fault_schema_migrations"
            " ORDER BY version"
        )
        rows = cursor.fetchall()
        registered = max((int(row[0]) for row in rows), default=0)
        expected = [
            (migration.version, migration.name, migration.checksum)
            for migration in POSTGRES_SCHEMA_MIGRATIONS
            if migration.version <= registered
        ]
        history_ok = [tuple(row) for row in rows] == expected
    cursor.execute(
        """
        SELECT count(*) FROM gpu_fault_objects
        WHERE kind='workflow'
          AND payload->>'status' IN ('RUNNING', 'SAFETY_PENDING')
          AND jsonb_array_length(coalesce(payload->'blocked_reasons', '[]'::jsonb)) > 0
          AND coalesce(payload->>'safety_only', 'false') <> 'true'
        """
    )
    unflagged = int(cursor.fetchone()[0])
    health = index_health(connection)
    missing = [row["name"] for row in health if not row["present"]]
    invalid = [row["name"] for row in health if row["present"] and not row["valid"]]
    if registered < LATEST_POSTGRES_SCHEMA_VERSION:
        reasons.append(
            f"schema version {registered} is behind the wheel's "
            f"{LATEST_POSTGRES_SCHEMA_VERSION}; run --ensure-schema"
        )
    elif registered > LATEST_POSTGRES_SCHEMA_VERSION:
        reasons.append(
            f"schema version {registered} is ahead of the wheel's "
            f"{LATEST_POSTGRES_SCHEMA_VERSION}; this wheel is older than the database"
        )
    if not history_ok:
        reasons.append("schema migration history does not match this wheel's registry")
    if missing:
        reasons.append(
            "indexes missing (build with --build-indexes-concurrently): "
            + ", ".join(missing)
        )
    if invalid:
        reasons.append(
            "indexes invalid (rebuild with --build-indexes-concurrently): "
            + ", ".join(invalid)
        )
    if unflagged:
        reasons.append(
            f"{unflagged} in-flight safety workflow(s) predate safety_only; "
            "wait for them to finish before rolling"
        )
    diagnostics = database_diagnostics(cursor)
    return {
        "ok": not reasons,
        "schema_version": {
            "registered": registered,
            "required": LATEST_POSTGRES_SCHEMA_VERSION,
            "history_ok": history_ok,
        },
        "indexes": {"missing": missing, "invalid": invalid, "declared": len(health)},
        "in_flight_safety_workflows_without_flag": unflagged,
        "blocking_reasons": reasons,
        "diagnostics": diagnostics,
        "warnings": diagnostics_warnings(diagnostics),
    }


# The Aurora cluster is not visible to the deploy host's AWS role, so the
# database itself is the only place to see whether lock waits are logged and
# whether pg_stat_statements is loaded (store review 2026-09-07, item K2).
_DIAGNOSTIC_SETTINGS = (
    "log_lock_waits",
    "deadlock_timeout",
    "log_min_duration_statement",
    "shared_preload_libraries",
)


def database_diagnostics(cursor: Any) -> dict[str, Any]:
    """Server settings that decide what the database can tell us afterwards.

    ``current_setting(name, true)`` returns NULL rather than raising for a
    name this server does not know, so the report never blocks on one.
    """

    report: dict[str, Any] = {}
    for name in _DIAGNOSTIC_SETTINGS:
        cursor.execute("SELECT current_setting(%s, true)", (name,))
        report[name] = cursor.fetchone()[0]
    cursor.execute(
        "SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname='pg_stat_statements')"
    )
    report["pg_stat_statements_installed"] = bool(cursor.fetchone()[0])
    return report


def diagnostics_warnings(diagnostics: dict[str, Any]) -> list[str]:
    """Non-blocking: each entry names what the missing diagnostic costs."""

    warnings: list[str] = []
    if str(diagnostics.get("log_lock_waits") or "off").lower() != "on":
        warnings.append(
            "log_lock_waits is off: lock waits longer than deadlock_timeout and "
            "the deadlocks the store retries stay invisible in the database log"
        )
    if not diagnostics.get("pg_stat_statements_installed"):
        warnings.append(
            "pg_stat_statements is not installed: there is no per-statement "
            "profile of the ACU spend; run --ensure-diagnostics (needs "
            "shared_preload_libraries=pg_stat_statements in the parameter group)"
        )
    return warnings
