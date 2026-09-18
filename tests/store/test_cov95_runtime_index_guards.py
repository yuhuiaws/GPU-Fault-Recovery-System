from __future__ import annotations

from types import SimpleNamespace

import pytest

from gpu_fault.store.postgres import ddl, index_builder

STATEMENTS = {
    "ready_index": "CREATE INDEX IF NOT EXISTS ready_index ON gpu_fault_objects (kind)",
    "invalid_index": "CREATE INDEX IF NOT EXISTS invalid_index ON gpu_fault_objects (key)",
    "native_index": "CREATE INDEX IF NOT EXISTS native_index ON gpu_fault_workflows (request_id)",
}


class IndexConnection:
    def __init__(self, *, native_table=False, lock_available=True, fail_name=None):
        from psycopg.pq import TransactionStatus

        self.autocommit = True
        self.info = SimpleNamespace(transaction_status=TransactionStatus.IDLE)
        self.native_table = native_table
        self.lock_available = lock_available
        self.fail_name = fail_name
        self.indexes = {"ready_index": True, "invalid_index": False}
        self.calls = []

    def cursor(self):
        return IndexCursor(self)


class IndexCursor:
    def __init__(self, connection):
        self.connection = connection
        self.rows = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, query, parameters=None):
        sql = " ".join(query.split())
        self.connection.calls.append((sql, parameters))
        if "pg_try_advisory_lock" in sql:
            self.rows = [(self.connection.lock_available,)]
        elif "pg_advisory_unlock" in sql:
            self.rows = [(True,)]
        elif sql == "SELECT to_regclass('gpu_fault_control_state_modes')":
            self.rows = [(None,)]
        elif sql == "SELECT to_regclass(%s)":
            table = parameters[0]
            exists = table == "gpu_fault_objects" or (
                table == "gpu_fault_workflows" and self.connection.native_table
            )
            self.rows = [(table if exists else None,)]
        elif "SELECT c.relname, i.indisvalid" in sql:
            self.rows = [
                (name, valid)
                for name, valid in self.connection.indexes.items()
                if name in parameters[0]
            ]
        else:
            for name, declaration in STATEMENTS.items():
                if sql == f"DROP INDEX CONCURRENTLY IF EXISTS {name}":
                    self.connection.indexes.pop(name, None)
                    break
                if sql == declaration.replace(
                    "INDEX IF NOT EXISTS", "INDEX CONCURRENTLY IF NOT EXISTS"
                ):
                    if self.connection.fail_name == name:
                        raise RuntimeError("synthetic index build failure")
                    self.connection.indexes[name] = True
                    break
            else:
                raise AssertionError(
                    f"unexpected fake index transport statement: {sql}"
                )

    def fetchone(self):
        return self.rows[0]

    def fetchall(self):
        return list(self.rows)


@pytest.fixture
def index_sources(tmp_path, monkeypatch):
    path = tmp_path / "ddl_example.py"
    path.write_text(
        "\n".join(f'"""{statement}"""' for statement in STATEMENTS.values()),
        encoding="utf-8",
    )
    monkeypatch.setattr(ddl, "__file__", str(path))
    return path


@pytest.mark.parametrize("native_table", [False, True])
def test_online_index_build_rebuilds_invalid_and_defers_only_absent_native_tables(
    index_sources, native_table
):
    connection = IndexConnection(native_table=native_table)
    report = index_builder.build_missing_indexes_concurrently(connection)
    assert report["built"] == (
        ["invalid_index", "native_index"] if native_table else ["invalid_index"]
    )
    assert report["dropped_invalid"] == ["invalid_index"]
    assert report["deferred_until_schema"] == ([] if native_table else ["native_index"])
    assert report["awaiting_table"] == report["deferred_until_schema"]
    assert report["invalid_after"] == report["missing_after"] == []
    assert "pg_advisory_unlock" in connection.calls[-1][0], (
        "online build must release its session maintenance lock"
    )
    assert (
        index_builder.build_missing_indexes_concurrently(connection)["built"] == []
    ), "already valid indexes must not be rebuilt"


def test_failed_index_build_unlocks_and_can_be_retried(index_sources):
    connection = IndexConnection(fail_name="invalid_index")
    with pytest.raises(RuntimeError, match="synthetic index"):
        index_builder.build_missing_indexes_concurrently(connection)
    assert "pg_advisory_unlock" in connection.calls[-1][0], (
        "a failed concurrent build must release its maintenance lock"
    )
    connection.fail_name = None
    report = index_builder.build_missing_indexes_concurrently(connection)
    assert report["built"] == ["invalid_index"]
    assert report["invalid_after"] == [], "retry must replace the failed build"


def test_busy_index_maintenance_does_not_build_or_unlock_another_owner(index_sources):
    connection = IndexConnection(lock_available=False)
    with pytest.raises(RuntimeError, match="already running"):
        index_builder.build_missing_indexes_concurrently(connection)
    assert len(connection.calls) == 1, "a refused maintenance lock cannot perform DDL"
    assert connection.indexes == {"ready_index": True, "invalid_index": False}


@pytest.mark.parametrize("active", [False, True])
def test_index_builder_requires_idle_autocommit(index_sources, active):
    from psycopg.pq import TransactionStatus

    connection = IndexConnection()
    if active:
        connection.info.transaction_status = TransactionStatus.INTRANS
    else:
        connection.autocommit = False
    with pytest.raises(RuntimeError, match="idle autocommit"):
        index_builder.build_missing_indexes_concurrently(connection)
    assert connection.calls == [], "connection guards must run before any SQL"


def test_interpolated_index_declarations_fail_closed(index_sources):
    index_sources.write_text(
        '"""CREATE INDEX IF NOT EXISTS broken ON {table} (key)"""', encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="interpolated statement"):
        index_builder.declared_index_statements()
