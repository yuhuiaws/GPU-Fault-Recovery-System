"""Catalog-checked DDL helpers shared by the ``ddl*.py`` stage modules.

Idempotent means lock-free when nothing changes (store review 2026-09-07,
item J): Postgres takes the table lock before it finds a ``CREATE INDEX ...
IF NOT EXISTS`` / ``ALTER TABLE ... IF [NOT] EXISTS`` statement is a no-op,
so every stage reads the catalog through these helpers first and issues the
DDL only when the object is missing or different. Kept in its own module so
``ddl.py`` and the stage modules it imports share it without an import cycle.
The file name keeps the ``ddl`` prefix on purpose: the migration registry
checksums every ``ddl*.py`` file, so a change here is a schema change too.
"""

from __future__ import annotations

import re
from typing import Any

_CREATE_INDEX = re.compile(r"CREATE\s+(?:UNIQUE\s+)?INDEX\s+IF\s+NOT\s+EXISTS\s+(\w+)")


def _declare_index(cursor: Any, statement: str) -> None:
    """Run one literal ``CREATE [UNIQUE] INDEX ... IF NOT EXISTS`` only when
    the index is absent (item J).

    The statement stays a literal so ``declared_index_names`` and
    ``declared_index_statements`` keep scraping it from the source; the
    ``IF NOT EXISTS`` stays as the race guard for two bootstraps that both
    saw the index missing. Without the ``to_regclass`` check, the no-op form
    still opens the table in ShareLock until commit.
    """

    match = _CREATE_INDEX.search(statement)
    if match is None:
        raise ValueError("not a CREATE INDEX ... IF NOT EXISTS statement")
    cursor.execute("SELECT to_regclass(%s)", (match.group(1),))
    if cursor.fetchone()[0] is None:
        cursor.execute(statement)


def _add_column_if_missing(
    cursor: Any, table: str, column: str, definition: str
) -> None:
    """``ALTER TABLE ... ADD COLUMN IF NOT EXISTS`` without its
    AccessExclusiveLock when the column is already there (item J)."""

    cursor.execute(
        """
        SELECT 1 FROM information_schema.columns
        WHERE table_schema=current_schema()
          AND table_name=%s AND column_name=%s
        """,
        (table, column),
    )
    if cursor.fetchone() is None:
        cursor.execute(
            f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} {definition}"
        )


def _drop_column_if_present(cursor: Any, table: str, column: str) -> None:
    cursor.execute(
        """
        SELECT 1 FROM information_schema.columns
        WHERE table_schema=current_schema()
          AND table_name=%s AND column_name=%s
        """,
        (table, column),
    )
    if cursor.fetchone() is not None:
        cursor.execute(f"ALTER TABLE {table} DROP COLUMN IF EXISTS {column}")


def _drop_index_if_present(cursor: Any, name: str) -> None:
    cursor.execute("SELECT to_regclass(%s)", (name,))
    if cursor.fetchone()[0] is not None:
        cursor.execute(f"DROP INDEX IF EXISTS {name}")


def _ensure_trigger(cursor: Any, name: str, table: str, statement: str) -> None:
    """Create a row trigger when missing; recreate it only when its stored
    definition differs from ``statement`` (item J).

    ``DROP TRIGGER`` + ``CREATE TRIGGER`` on every bootstrap took an
    AccessExclusiveLock on the table each time. ``pg_get_triggerdef`` renders
    the stored trigger in the same ``CREATE TRIGGER`` grammar the DDL uses, so
    a whitespace-normalized comparison is exact once the schema qualifier it
    always puts on the table name (``ON public.<table>``) is stripped.
    """

    cursor.execute(
        """
        SELECT pg_get_triggerdef(t.oid)
        FROM pg_trigger t
        WHERE t.tgrelid=to_regclass(%s) AND t.tgname=%s AND NOT t.tgisinternal
        """,
        (table, name),
    )
    row = cursor.fetchone()
    if row is not None and _normalize_trigger_definition(
        row[0]
    ) == _normalize_trigger_definition(statement):
        return
    if row is not None:
        cursor.execute(f"DROP TRIGGER IF EXISTS {name} ON {table}")
    cursor.execute(statement)


_TRIGGER_TABLE_QUALIFIER = re.compile(r"\bON\s+\w+\.(\w+)")


def _normalize_trigger_definition(definition: str) -> str:
    return _TRIGGER_TABLE_QUALIFIER.sub(r"ON \1", " ".join(definition.split()))


def _enable_trigger_if_disabled(cursor: Any, table: str, name: str) -> None:
    """``ALTER TABLE ... ENABLE TRIGGER`` takes ShareRowExclusiveLock even when
    the trigger is already enabled ('O' = origin-and-local), so only issue it
    for a trigger somebody disabled (item J)."""

    cursor.execute(
        """
        SELECT tgenabled FROM pg_trigger
        WHERE tgrelid=to_regclass(%s) AND tgname=%s AND NOT tgisinternal
        """,
        (table, name),
    )
    row = cursor.fetchone()
    if row is not None and row[0] != "O":
        cursor.execute(f"ALTER TABLE {table} ENABLE TRIGGER {name}")


def _set_table_options_if_different(
    cursor: Any, table: str, options: dict[str, str]
) -> None:
    """``ALTER TABLE ... SET (option=value, ...)`` only for the options whose
    stored value differs (item J: the ALTER takes ShareUpdateExclusiveLock
    even when nothing changes).

    Control-plane review 2026-09-08, G-9: ``gpu_fault_objects`` is a whole-row
    JSONB upsert table under ~50 expression indexes, so every lease renewal
    leaves a dead tuple that cannot be HOT-pruned; the default 20 % dead-tuple
    vacuum threshold let a 300k-row table accumulate 60k dead tuples before a
    vacuum. Options are validated as identifiers and numerics here because they
    are interpolated into DDL text.
    """

    for name, value in options.items():
        if not name.replace("_", "").isalnum() or not _is_numeric(value):
            raise ValueError(f"unsafe table option: {name}={value}")
    cursor.execute(
        "SELECT reloptions FROM pg_class WHERE oid=to_regclass(%s)", (table,)
    )
    row = cursor.fetchone()
    current: dict[str, str] = {}
    stored = row[0] if row else None
    for item in stored if isinstance(stored, (list, tuple)) else []:
        key, _, value = str(item).partition("=")
        current[key] = value
    pending = {
        name: value for name, value in options.items() if current.get(name) != value
    }
    if not pending:
        return
    rendered = ", ".join(f"{name}={value}" for name, value in sorted(pending.items()))
    cursor.execute(f"ALTER TABLE {table} SET ({rendered})")


def _is_numeric(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def _recorded_schema_version(cursor: Any) -> int | None:
    """The version the last bootstrap recorded, or None on a fresh database.
    ``_create_base_tables`` has created the table by the time this runs."""

    cursor.execute("SELECT version FROM gpu_fault_schema_version WHERE singleton=TRUE")
    row = cursor.fetchone()
    return None if row is None else int(row[0])
