"""Read-only deployed schema/migration proof; never constructs a Store or runs DDL."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Self

from gpu_fault.schema_migrations import (
    LATEST_POSTGRES_SCHEMA_VERSION,
    POSTGRES_SCHEMA_MIGRATIONS,
)
from gpu_fault.state_table_migrate import state_table_status
from gpu_fault.store.postgres.pool import StoreCredentials
from gpu_fault.store.postgres.state_table_schema import validate_state_table_schema

if TYPE_CHECKING:
    import psycopg


@lru_cache(maxsize=1)
def _read_only_cursor_type() -> type[psycopg.Cursor[tuple[Any, ...]]]:
    import psycopg
    from psycopg import sql

    class ReadOnlySchemaCursor(psycopg.Cursor[tuple[Any, ...]]):
        """Replace the shared validator's temporary-view DDL with planner reads."""

        comparison_row: tuple[Any, ...] | None = None
        expected_plan: dict[str, Any] | None = None

        def query_plan(self, query: str) -> dict[str, Any]:
            super().execute(
                "EXPLAIN (ANALYZE FALSE, VERBOSE, FORMAT JSON, COSTS FALSE) " + query
            )
            row = super().fetchone()
            if row is None or not isinstance(row[0], list) or len(row[0]) != 1:
                raise RuntimeError("state-table view plan is unavailable")
            plan = row[0][0].get("Plan")
            if not isinstance(plan, dict):
                raise RuntimeError("state-table view plan is malformed")
            super().execute(
                sql.SQL("SELECT * FROM ({}) AS state_view_shape LIMIT 0").format(
                    sql.SQL(query.rstrip().removesuffix(";"))
                )
            )
            columns = self.description
            if columns is None:
                raise RuntimeError("state-table view columns are unavailable")
            return {
                "plan": plan,
                "columns": [(column.name, column.type_code) for column in columns],
            }

        def execute(
            self,
            query: Any,
            params: Any = None,
            *,
            prepare: bool | None = None,
            binary: bool | None = None,
        ) -> Self:
            self.comparison_row = None
            if query == "SELECT pg_get_viewdef(to_regclass(%s), true)":
                super().execute(query, params, prepare=prepare, binary=binary)
                row = super().fetchone()
                self.comparison_row = (
                    self.query_plan(row[0])
                    if row and isinstance(row[0], str)
                    else None,
                )
            elif isinstance(query, str) and query.startswith(
                "CREATE TEMP VIEW gf_control_state_view_probe AS "
            ):
                self.expected_plan = self.query_plan(
                    query.removeprefix(
                        "CREATE TEMP VIEW gf_control_state_view_probe AS "
                    )
                )
            elif (
                query
                == "SELECT pg_get_viewdef('gf_control_state_view_probe'::regclass, true)"
            ):
                if self.expected_plan is None:
                    raise RuntimeError("state-table view comparison was not prepared")
                self.comparison_row = (self.expected_plan,)
            elif query == "DROP VIEW gf_control_state_view_probe":
                self.expected_plan = None
            else:
                super().execute(query, params, prepare=prepare, binary=binary)
            return self

        def fetchone(self) -> tuple[Any, ...] | None:
            if self.comparison_row is not None:
                row, self.comparison_row = self.comparison_row, None
                return row
            return super().fetchone()

    return ReadOnlySchemaCursor


def __getattr__(name: str) -> type[psycopg.Cursor[tuple[Any, ...]]]:
    if name == "ReadOnlySchemaCursor":
        return _read_only_cursor_type()
    raise AttributeError(name)


def snapshot(kind: str, *, verify: bool) -> dict[str, Any]:
    import psycopg

    if kind not in {"remote_command", "workflow"}:
        raise ValueError("unsupported acceptance state table")
    path = os.getenv("GPU_FAULT_STORE_URL_FILE")
    credentials = StoreCredentials(os.getenv("GPU_FAULT_STORE_URL", ""), path=path)
    url = credentials.conninfo()
    if not url or (path and credentials.source != "file"):
        raise RuntimeError("current database credentials are unavailable")
    with psycopg.connect(
        url,
        autocommit=True,
        connect_timeout=10,
        cursor_factory=_read_only_cursor_type(),
        options="-c default_transaction_read_only=on -c statement_timeout=30000 -c lock_timeout=5000",
    ) as connection:
        row = connection.execute(
            "SELECT current_database(), inet_server_addr()::text, inet_server_port(), "
            "current_schemas(true), pg_is_in_recovery(), "
            "current_setting('transaction_read_only')"
        ).fetchone()
        if (
            row is None
            or len(row) != 6
            or any(not isinstance(value, str) or not value for value in row[:2])
            or type(row[2]) is not int
            or not 0 < row[2] <= 65535
            or not isinstance(row[3], list)
            or not row[3]
            or any(not isinstance(value, str) or not value for value in row[3])
            or row[4] is not False
            or row[5] != "on"
        ):
            raise RuntimeError(
                "acceptance requires a read-only connection to the writer"
            )
        server_digest = hashlib.sha256(json.dumps(row[:4]).encode()).hexdigest()
        # Keep schema ensure from racing the separate read-only status transaction.
        connection.execute(
            "SELECT pg_advisory_lock_shared(hashtextextended('gpu_fault_schema_bootstrap', 0))"
        )
        if connection.execute(
            "SELECT version FROM gpu_fault_schema_version WHERE singleton=TRUE"
        ).fetchall() != [(LATEST_POSTGRES_SCHEMA_VERSION,)]:
            raise RuntimeError(
                "deployed schema version differs from the installed release"
            )
        if connection.execute(
            "SELECT version,name,checksum FROM gpu_fault_schema_migrations ORDER BY version"
        ).fetchall() != [
            (item.version, item.name, item.checksum)
            for item in POSTGRES_SCHEMA_MIGRATIONS
        ]:
            raise RuntimeError(
                "deployed migration history differs from the installed release"
            )
        validate_state_table_schema(connection)
        state = state_table_status(connection, kind, verify=verify)
    return {
        "schema_version": LATEST_POSTGRES_SCHEMA_VERSION,
        "schema_valid": True,
        "read_only": True,
        "writer": True,
        "database_sha256": server_digest,
        "state": state,
    }


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    try:
        if len(arguments) != 2 or arguments[1] not in {"verify", "metadata"}:
            raise ValueError("expected kind and verification mode")
        report = snapshot(arguments[0], verify=arguments[1] == "verify")
    except Exception as exc:
        print(json.dumps({"verdict": "FAIL", "error_type": type(exc).__name__}))
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
