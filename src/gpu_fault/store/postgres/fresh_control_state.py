"""Seed a brand-new database\'s control-state mode at schema creation.

Ordinary deployment never activates a control-state migration on an existing
database: the mode rows are the single source of truth and change only through
``gpu-fault-store-migrate``. A database that has no schema at all is different:
there is nothing to migrate, so the schema release Job may ask for the tables to
start in ``dedicated`` mode (user decision 2026-09-20). This module lives
outside ``ddl*.py`` on purpose: it changes no DDL, so the migration registry\'s
DDL checksum and the schema version stay put.
"""

from __future__ import annotations

from typing import Any

FRESH_CONTROL_STATE_MODES = ("legacy", "dedicated")
_SEEDED_KINDS = ("remote_command", "workflow")


def normalize_fresh_control_state_mode(value: str | None) -> str:
    mode = (value if value is not None else "legacy").strip().lower()
    if mode not in FRESH_CONTROL_STATE_MODES:
        raise ValueError("fresh_control_state_mode must be legacy or dedicated")
    return mode


def database_is_fresh(cursor: Any) -> bool:
    """No schema at all: neither the migration history nor the object store."""
    cursor.execute(
        "SELECT to_regclass('gpu_fault_schema_migrations') IS NULL "
        "AND to_regclass('gpu_fault_schema_version') IS NULL "
        "AND to_regclass('gpu_fault_control_state_modes') IS NULL "
        "AND to_regclass('gpu_fault_objects') IS NULL"
    )
    return bool(cursor.fetchone()[0])


def seed_fresh_control_state_modes(cursor: Any, mode: str) -> dict[str, Any]:
    """Move the freshly created default rows to ``mode``; fail closed otherwise.

    Runs only after the caller proved the database was fresh before this
    bootstrap. Every guard re-checks what the bootstrap just wrote: both rows
    at their defaults and every control-state table empty, so a database that
    is not what a fresh install produces is left alone with an error instead
    of being silently switched.
    """
    mode = normalize_fresh_control_state_mode(mode)
    if mode == "legacy":
        return {"seeded": False, "mode": "legacy"}
    cursor.execute(
        "SELECT kind, mode, revision, backfill_complete "
        "FROM gpu_fault_control_state_modes ORDER BY kind"
    )
    rows = [tuple(row) for row in cursor.fetchall()]
    if rows != [(kind, "legacy", 0, False) for kind in _SEEDED_KINDS]:
        raise RuntimeError(
            "fresh control-state seeding found a migration registry that is not "
            f"the freshly created default: {rows!r}"
        )
    cursor.execute(
        "SELECT (SELECT count(*) FROM gpu_fault_objects "
        "WHERE kind IN ('remote_command', 'workflow')), "
        "(SELECT count(*) FROM gpu_fault_remote_commands), "
        "(SELECT count(*) FROM gpu_fault_workflows)"
    )
    if any(int(count) for count in cursor.fetchone()):
        raise RuntimeError("fresh control-state seeding found existing control records")
    cursor.execute(
        "UPDATE gpu_fault_control_state_modes SET mode='dedicated', revision=1, "
        "dedicated_at=now(), backfill_complete=TRUE, updated_at=now() "
        "WHERE kind IN ('remote_command', 'workflow') AND mode='legacy'"
    )
    if cursor.rowcount != len(_SEEDED_KINDS):
        raise RuntimeError(
            "fresh control-state seeding updated an unexpected number of rows"
        )
    return {"seeded": True, "mode": "dedicated", "kinds": list(_SEEDED_KINDS)}
