"""The online index builder on a database whose tables do not exist yet.

Live 2026-09-12: the first bootstrap of a fresh site (Aurora reset) ran the
index-build Job before the ensure Job, as the three-step method requires, and
died on ``relation "gpu_fault_objects" does not exist``. A missing table has no
writers to protect, and the ensure Job's DDL creates table and indexes together,
so those indexes are reported as awaiting their table rather than built or
counted as missing.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

from gpu_fault.store.postgres import index_builder as MODULE


class _Cursor:
    def __init__(self, tables: set[str]) -> None:
        self.tables = tables
        self.statements: list[str] = []
        self._result: list[tuple] = []

    def __enter__(self) -> "_Cursor":
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def execute(self, statement: str, parameters: tuple = ()) -> None:
        self.statements.append(statement)
        if statement.startswith("SELECT to_regclass"):
            table = parameters[0]
            self._result = [(table if table in self.tables else None,)]
        elif "pg_index" in statement:
            self._result = []  # no declared index exists
        else:
            self._result = []

    def fetchone(self) -> tuple:
        return self._result[0]

    def fetchall(self) -> list[tuple]:
        return self._result


def _connection(tables: set[str]) -> tuple[SimpleNamespace, _Cursor]:
    cursor = _Cursor(tables)
    return SimpleNamespace(autocommit=True, cursor=lambda: cursor), cursor


def test_indexes_of_absent_tables_are_left_to_the_ensure_job() -> None:
    connection, cursor = _connection(set())

    report = MODULE.build_missing_indexes_concurrently(connection)

    assert report["built"] == []
    assert report["missing_after"] == [], "nothing the ensure Job cannot create"
    assert sorted(report["awaiting_table"]) == sorted(MODULE.declared_index_names())
    assert not any("CREATE INDEX" in s for s in cursor.statements)


def test_indexes_of_existing_tables_are_still_built_online() -> None:
    statements = MODULE.declared_index_statements()
    name = sorted(statements)[0]
    table = re.search(r"\bON\s+(\w+)", statements[name]).group(1)
    connection, cursor = _connection({table})

    report = MODULE.build_missing_indexes_concurrently(connection)

    assert name in report["built"]
    assert name not in report["awaiting_table"]
    assert any(
        s.startswith("CREATE") and "CONCURRENTLY" in s and f" {name} " in s
        for s in cursor.statements
    )
