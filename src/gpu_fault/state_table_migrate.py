"""Explicit, resumable control-state migrations; ordinary deployment never activates them."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from uuid import uuid4

from pydantic import ValidationError

from gpu_fault.store.postgres.state_table_payload import STATE_LAYOUTS, StateTableLayout
from gpu_fault.store.postgres.state_table_storage import ENABLED_STATE_KINDS
from gpu_fault.store.shared.record_models import record_models

STATE_TABLE_VALIDATION_SECONDS = 30.0


class StateTableMigrationError(RuntimeError):
    pass


@contextmanager
def state_table_maintenance(postgres_url: str) -> Iterator[Any]:
    import psycopg

    from gpu_fault.schema_migrations import (
        LATEST_POSTGRES_SCHEMA_VERSION,
        POSTGRES_SCHEMA_MIGRATIONS,
    )
    from gpu_fault.store.postgres.state_table_schema import validate_state_table_schema

    with psycopg.connect(
        postgres_url, autocommit=True, connect_timeout=10
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SET lock_timeout='5s'")
            cursor.execute("SET statement_timeout='30s'")
            cursor.execute(
                "SELECT pg_advisory_lock_shared(hashtextextended('gpu_fault_schema_bootstrap', 0))"
            )
            try:
                cursor.execute(
                    "SELECT version FROM gpu_fault_schema_version WHERE singleton=TRUE"
                )
                if cursor.fetchall() != [(LATEST_POSTGRES_SCHEMA_VERSION,)]:
                    raise StateTableMigrationError(
                        "state-table maintenance requires this release's schema version"
                    )
                cursor.execute(
                    "SELECT version,name,checksum FROM gpu_fault_schema_migrations ORDER BY version"
                )
                if cursor.fetchall() != [
                    (item.version, item.name, item.checksum)
                    for item in POSTGRES_SCHEMA_MIGRATIONS
                ]:
                    raise StateTableMigrationError(
                        "state-table maintenance migration history differs from this release"
                    )
                try:
                    validate_state_table_schema(connection)
                except RuntimeError as exc:
                    raise StateTableMigrationError(str(exc)) from None
                yield connection
            finally:
                cursor.execute(
                    "SELECT pg_advisory_unlock_shared(hashtextextended('gpu_fault_schema_bootstrap', 0))"
                )


def _layout(kind: str) -> StateTableLayout:
    if kind not in ENABLED_STATE_KINDS:
        raise StateTableMigrationError(
            "this state table is not enabled in this schema release"
        )
    return STATE_LAYOUTS[kind]


def _lock(cursor: Any, *, exclusive: bool) -> None:
    cursor.execute("SET LOCAL lock_timeout='5s'")
    cursor.execute("SET LOCAL statement_timeout='30s'")
    cursor.execute(
        "SELECT pg_advisory_xact_lock_shared(hashtextextended('gpu_fault_schema_bootstrap', 0))"
    )
    # Old SQL writers can acquire a legacy tuple before their row trigger
    # requests the shared fence. Queuing an exclusive fence behind an existing
    # writer could then block the old writer while it holds that writer's tuple.
    function = (
        "pg_try_advisory_xact_lock" if exclusive else "pg_advisory_xact_lock_shared"
    )
    for kind in sorted(STATE_LAYOUTS):
        cursor.execute(
            f"SELECT {function}(hashtextextended(%s, 0))",
            (f"gpu_fault_control_state/{kind}",),
        )
        if exclusive and not cursor.fetchone()[0]:
            raise StateTableMigrationError("state-table writers are busy; retry")


def _mode(cursor: Any, kind: str) -> dict[str, Any]:
    cursor.execute(
        "SELECT mode, revision, backfill_after_key, backfill_complete, legacy_purged "
        "FROM gpu_fault_control_state_modes WHERE kind=%s",
        (kind,),
    )
    row = cursor.fetchone()
    if row is None:
        raise StateTableMigrationError("state-table migration metadata is missing")
    return dict(
        zip(
            (
                "mode",
                "revision",
                "backfill_after_key",
                "backfill_complete",
                "legacy_purged",
            ),
            row,
        )
    )


def _validation_timeout(cursor: Any, deadline: float) -> None:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise StateTableMigrationError("state-table validation exceeded its deadline")
    cursor.execute(
        "SELECT set_config('statement_timeout', %s, true)",
        (f"{max(1, int(remaining * 1000))}ms",),
    )


def _status(
    cursor: Any,
    layout: StateTableLayout,
    *,
    verify: bool,
    deadline: float | None = None,
) -> dict[str, Any]:
    from psycopg import sql

    if deadline is None:
        deadline = time.monotonic() + STATE_TABLE_VALIDATION_SECONDS
    _validation_timeout(cursor, deadline)
    metadata = _mode(cursor, layout.kind)
    _validation_timeout(cursor, deadline)
    cursor.execute(
        "SELECT count(*) FROM gpu_fault_objects WHERE kind=%s", (layout.kind,)
    )
    legacy_count = int(cursor.fetchone()[0])
    _validation_timeout(cursor, deadline)
    cursor.execute(
        sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(layout.table))
    )
    dedicated_count = int(cursor.fetchone()[0])
    report: dict[str, Any] = {
        "kind": layout.kind,
        "mode": metadata["mode"],
        "revision": metadata["revision"],
        "backfill_complete": metadata["backfill_complete"],
        "legacy_purged": metadata["legacy_purged"],
        "legacy_rows": legacy_count,
        "dedicated_rows": dedicated_count,
        "verification_applicable": metadata["mode"] == "dual",
        "verification_performed": verify and metadata["mode"] == "dual",
    }
    if not verify or metadata["mode"] != "dual":
        return report
    _validation_timeout(cursor, deadline)
    cursor.execute(
        sql.SQL(
            "SELECT count(*) FILTER (WHERE d.{key} IS NULL), "
            "count(*) FILTER (WHERE d.{key} IS NOT NULL AND {decode}(d) IS DISTINCT FROM o.payload) "
            "FROM gpu_fault_objects o LEFT JOIN {table} d ON d.{key}=o.key WHERE o.kind=%s"
        ).format(
            key=sql.Identifier(layout.key_field),
            table=sql.Identifier(layout.table),
            decode=sql.Identifier(f"gpu_fault_{layout.kind}_payload"),
        ),
        (layout.kind,),
    )
    missing, mismatched = cursor.fetchone()
    _validation_timeout(cursor, deadline)
    cursor.execute(
        sql.SQL(
            "SELECT count(*) FROM {table} d WHERE NOT EXISTS (SELECT 1 FROM gpu_fault_objects o "
            "WHERE o.kind=%s AND o.key=d.{key})"
        ).format(
            table=sql.Identifier(layout.table), key=sql.Identifier(layout.key_field)
        ),
        (layout.kind,),
    )
    extra = int(cursor.fetchone()[0])
    invalid = noncanonical = 0
    model = record_models()[layout.kind]
    with cursor.connection.cursor(name=f"gf_state_verify_{uuid4().hex}") as stream:
        _validation_timeout(cursor, deadline)
        stream.execute(
            "SELECT key, payload FROM gpu_fault_objects WHERE kind=%s ORDER BY key",
            (layout.kind,),
        )
        while True:
            _validation_timeout(cursor, deadline)
            rows = stream.fetchmany(100)
            if not rows:
                break
            for key, payload in rows:
                try:
                    canonical = model.model_validate(payload).model_dump(mode="json")
                except (ValueError, TypeError, ValidationError):
                    invalid += 1
                    continue
                if canonical.get(layout.key_field) != key:
                    invalid += 1
                elif canonical != payload:
                    noncanonical += 1
    _validation_timeout(cursor, deadline)
    report.update(
        missing_rows=int(missing),
        mismatched_rows=int(mismatched),
        extra_rows=extra,
        invalid_records=invalid,
        noncanonical_records=noncanonical,
        verified=not any((missing, mismatched, extra, invalid, noncanonical)),
    )
    return report


def state_table_status(
    connection: Any, kind: str, *, verify: bool = True
) -> dict[str, Any]:
    layout = _layout(kind)
    with connection.transaction():
        with connection.cursor() as cursor:
            cursor.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
            cursor.execute("SET LOCAL lock_timeout='5s'")
            return _status(cursor, layout, verify=verify)


def _require_idle_cutover(cursor: Any) -> None:
    cursor.execute(
        "SELECT EXISTS(SELECT 1 FROM gpu_fault_remote_command_records "
        "WHERE status IN ('PENDING', 'LEASED', 'WAITING'))"
    )
    if cursor.fetchone()[0]:
        raise StateTableMigrationError("cutover requires drained remote commands")
    cursor.execute(
        "SELECT EXISTS(SELECT 1 FROM gpu_fault_control_records WHERE kind='workflow' "
        "AND payload->>'execution_owner_id' IS NOT NULL "
        "AND (payload->>'execution_lease_expires_at' IS NULL "
        "OR (payload->>'execution_lease_expires_at')::timestamptz>now()))"
    )
    if cursor.fetchone()[0]:
        raise StateTableMigrationError("cutover requires drained workflow leases")


def _reset_legacy_copy(cursor: Any, layout: StateTableLayout) -> None:
    from psycopg import sql

    # View readers retain AccessShare before taking the writer mode fence.
    # TRUNCATE would wait for those readers while holding that same fence.
    cursor.execute(
        "SELECT set_config('gpu_fault.control_state_reset', %s, true)", (layout.kind,)
    )
    cursor.execute(
        sql.SQL(
            "WITH victims AS (SELECT {key} FROM {table} ORDER BY {key} "
            "FOR UPDATE SKIP LOCKED) "
            "DELETE FROM {table} target USING victims "
            "WHERE target.{key}=victims.{key}"
        ).format(
            key=sql.Identifier(layout.key_field), table=sql.Identifier(layout.table)
        )
    )
    cursor.execute("SELECT set_config('gpu_fault.control_state_reset', '', true)")
    cursor.execute(
        sql.SQL("SELECT EXISTS(SELECT 1 FROM {})").format(sql.Identifier(layout.table))
    )
    if cursor.fetchone()[0]:
        # A native row locker can itself be waiting for the mode fence.
        # Never wait for it or publish dual with a stale copy left behind.
        raise StateTableMigrationError("state-table reset found locked rows; retry")


def set_state_table_mode(
    connection: Any,
    kind: str,
    mode: str,
    *,
    expected_mode: str,
    confirm_dedicated: bool = False,
) -> dict[str, Any]:
    layout = _layout(kind)
    if mode not in {"legacy", "dual", "dedicated"}:
        raise StateTableMigrationError("invalid state-table mode")
    with connection.transaction():
        with connection.cursor() as cursor:
            _lock(cursor, exclusive=True)
            before = _mode(cursor, kind)
            if before["mode"] != expected_mode:
                raise StateTableMigrationError(
                    "state-table mode changed since it was inspected"
                )
            if mode == before["mode"]:
                return _status(cursor, layout, verify=False)
            if (before["mode"], mode) not in {
                ("legacy", "dual"),
                ("dual", "legacy"),
                ("dual", "dedicated"),
            }:
                raise StateTableMigrationError(
                    "unsupported or irreversible state-table transition"
                )
            if mode == "dedicated":
                if not confirm_dedicated:
                    raise StateTableMigrationError(
                        "dedicated cutover requires explicit confirmation"
                    )
                _require_idle_cutover(cursor)
                checked = _status(
                    cursor,
                    layout,
                    verify=True,
                    deadline=time.monotonic() + STATE_TABLE_VALIDATION_SECONDS,
                )
                if not before["backfill_complete"] or not checked.get("verified"):
                    raise StateTableMigrationError(
                        "state-table backfill is incomplete or differs from legacy"
                    )
                cursor.execute(
                    "UPDATE gpu_fault_control_state_modes SET mode='dedicated', dedicated_at=now(), "
                    "revision=revision+1, updated_at=now() WHERE kind=%s",
                    (kind,),
                )
            else:
                if mode == "dual":
                    # Only the legacy table is authoritative here. A cancelled
                    # earlier dual attempt must not become a stale read source.
                    _reset_legacy_copy(cursor, layout)
                cursor.execute(
                    "UPDATE gpu_fault_control_state_modes SET mode=%s, revision=revision+1, "
                    "backfill_after_key=NULL, backfill_complete=FALSE, updated_at=now() WHERE kind=%s",
                    (mode, kind),
                )
            return _status(cursor, layout, verify=False)


def backfill_state_table(
    connection: Any,
    kind: str,
    *,
    batch_size: int = 100,
    max_batches: int = 25,
    restart: bool = False,
) -> dict[str, Any]:
    layout = _layout(kind)
    if not 1 <= batch_size <= 1000 or not 1 <= max_batches <= 1000:
        raise StateTableMigrationError("state-table backfill budgets are out of range")
    copied = 0
    complete = False
    model = record_models()[kind]
    for batch in range(max_batches):
        with connection.transaction():
            with connection.cursor() as cursor:
                _lock(cursor, exclusive=False)
                cursor.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                    (f"gpu_fault_control_state_backfill/{kind}",),
                )
                before = _mode(cursor, kind)
                if before["mode"] != "dual":
                    raise StateTableMigrationError("backfill requires dual mode")
                after = None if restart and batch == 0 else before["backfill_after_key"]
                if before["backfill_complete"] and not (restart and batch == 0):
                    complete = True
                    break
                cursor.execute(
                    "SELECT key, payload FROM gpu_fault_objects WHERE kind=%s "
                    "AND (%s::text IS NULL OR key>%s) ORDER BY key LIMIT %s FOR UPDATE",
                    (kind, after, after, batch_size),
                )
                rows = cursor.fetchall()
                for key, payload in rows:
                    try:
                        canonical = model.model_validate(payload).model_dump(
                            mode="json"
                        )
                        if canonical.get(layout.key_field) != key:
                            raise ValueError("identity differs")
                    except (ValueError, TypeError, ValidationError):
                        raise StateTableMigrationError(
                            "invalid legacy control record; backfill stopped"
                        ) from None
                    text = json.dumps(canonical)
                    cursor.execute(
                        "UPDATE gpu_fault_objects SET payload=CASE WHEN payload=%s::jsonb "
                        "THEN payload ELSE %s::jsonb END WHERE kind=%s AND key=%s",
                        (text, text, kind, key),
                    )
                    copied += 1
                complete = len(rows) < batch_size
                cursor.execute(
                    "UPDATE gpu_fault_control_state_modes SET backfill_after_key=%s, "
                    "backfill_complete=%s, updated_at=now() WHERE kind=%s",
                    (rows[-1][0] if rows else after, complete, kind),
                )
        if complete:
            break
    return {
        "copied": copied,
        "status": state_table_status(connection, kind, verify=complete),
    }


def purge_legacy_state_rows(
    connection: Any,
    kind: str,
    *,
    confirm: bool,
    batch_size: int = 100,
    max_batches: int = 25,
) -> dict[str, Any]:
    _layout(kind)
    if not confirm:
        raise StateTableMigrationError(
            "legacy retirement requires explicit confirmation"
        )
    if not 1 <= batch_size <= 1000 or not 1 <= max_batches <= 1000:
        raise StateTableMigrationError("legacy retirement budgets are out of range")
    deleted = 0
    for _batch in range(max_batches):
        with connection.transaction():
            with connection.cursor() as cursor:
                _lock(cursor, exclusive=False)
                before = _mode(cursor, kind)
                if before["mode"] != "dedicated":
                    raise StateTableMigrationError(
                        "legacy retirement requires dedicated mode"
                    )
                cursor.execute(
                    "SELECT set_config('gpu_fault.control_state_cleanup', %s, true)",
                    (kind,),
                )
                cursor.execute(
                    "WITH victims AS (SELECT key FROM gpu_fault_objects WHERE kind=%s "
                    "ORDER BY key LIMIT %s FOR UPDATE SKIP LOCKED) "
                    "DELETE FROM gpu_fault_objects o USING victims v WHERE o.kind=%s AND o.key=v.key",
                    (kind, batch_size, kind),
                )
                removed = cursor.rowcount
                deleted += removed
                cursor.execute(
                    "SELECT EXISTS(SELECT 1 FROM gpu_fault_objects WHERE kind=%s)",
                    (kind,),
                )
                remaining = bool(cursor.fetchone()[0])
                if not remaining:
                    cursor.execute(
                        "UPDATE gpu_fault_control_state_modes SET legacy_purged=TRUE, updated_at=now() "
                        "WHERE kind=%s",
                        (kind,),
                    )
        if not remaining or not removed:
            break
    return {
        "deleted": deleted,
        "has_more": remaining,
        "status": state_table_status(connection, kind, verify=False),
    }


def retire_legacy_state_indexes(connection: Any, kind: str) -> dict[str, Any]:
    from psycopg import sql

    from gpu_fault.store.postgres.ddl_helpers import CONTROL_STATE_INDEX_KIND
    from gpu_fault.store.postgres.index_builder import declared_index_statements

    _layout(kind)
    if not connection.autocommit:
        raise StateTableMigrationError(
            "concurrent index retirement requires autocommit"
        )
    names = sorted(
        name
        for name, statement in declared_index_statements().items()
        if (match := CONTROL_STATE_INDEX_KIND.search(statement))
        and match.group(1) == kind
    )
    dropped: list[str] = []
    with connection.cursor() as cursor:
        cursor.execute("SELECT current_setting('statement_timeout')")
        previous_timeout = cursor.fetchone()[0]
        cursor.execute("SET statement_timeout='30s'")
        locked = False
        try:
            cursor.execute(
                "SELECT pg_advisory_lock_shared(hashtextextended('gpu_fault_schema_bootstrap', 0))"
            )
            locked = True
            before = _mode(cursor, kind)
            if before["mode"] != "dedicated" or not before["legacy_purged"]:
                raise StateTableMigrationError(
                    "index retirement requires completed legacy row retirement"
                )
            cursor.execute(
                "SELECT EXISTS(SELECT 1 FROM gpu_fault_objects WHERE kind=%s)", (kind,)
            )
            if cursor.fetchone()[0]:
                raise StateTableMigrationError("legacy control-state rows remain")
            for name in names:
                cursor.execute(
                    sql.SQL("DROP INDEX CONCURRENTLY IF EXISTS {}").format(
                        sql.Identifier(name)
                    )
                )
                dropped.append(name)
        finally:
            if locked:
                cursor.execute(
                    "SELECT pg_advisory_unlock_shared(hashtextextended('gpu_fault_schema_bootstrap', 0))"
                )
            cursor.execute(
                "SELECT set_config('statement_timeout', %s, false)", (previous_timeout,)
            )
    return {"retired_indexes": dropped}


def purge_legacy_state(
    connection: Any,
    kind: str,
    *,
    confirm: bool,
    batch_size: int = 100,
    max_batches: int = 25,
) -> dict[str, Any]:
    result = purge_legacy_state_rows(
        connection,
        kind,
        confirm=confirm,
        batch_size=batch_size,
        max_batches=max_batches,
    )
    if not result["has_more"]:
        result.update(retire_legacy_state_indexes(connection, kind))
    return result
