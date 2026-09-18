from __future__ import annotations

from contextlib import nullcontext

from gpu_fault.schema_migrations import (
    LATEST_POSTGRES_SCHEMA_VERSION,
    POSTGRES_SCHEMA_MIGRATIONS,
)


class CatalogConnection:
    def __init__(
        self,
        *,
        mode="legacy",
        version=None,
        history=None,
        missing_mode=False,
        fail_query=None,
        remaining=False,
        records=(),
    ):
        self.autocommit = True
        self.statements = []
        self.version = (
            [(LATEST_POSTGRES_SCHEMA_VERSION,)] if version is None else version
        )
        self.history = (
            [
                (item.version, item.name, item.checksum)
                for item in POSTGRES_SCHEMA_MIGRATIONS
            ]
            if history is None
            else history
        )
        self.mode_row = None if missing_mode else (mode, 0, None, False, False)
        self.closed = False
        self.fail_query = fail_query
        self.remaining = remaining
        self.records = list(records)

    def cursor(self, **kwargs):
        return CatalogCursor(self)

    def transaction(self):
        return nullcontext()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True


class CatalogCursor:
    def __init__(self, connection):
        self.connection = connection
        self.query = ""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, query, params=None):
        text = query if isinstance(query, str) else query.as_string()
        if text.lstrip().split()[0] in {
            "UPDATE",
            "DELETE",
            "TRUNCATE",
            "CREATE",
            "DROP",
        }:
            raise AssertionError("a refused maintenance operation attempted mutation")
        self.query = text
        self.connection.statements.append((text, params))
        if self.connection.fail_query and self.connection.fail_query in text:
            raise RuntimeError("synthetic catalog transport failure")

    def fetchall(self):
        if self.query.startswith("SELECT version FROM"):
            return self.connection.version
        if self.query.startswith("SELECT version,name,checksum"):
            return self.connection.history
        if self.query.startswith("SELECT key, payload FROM gpu_fault_objects"):
            return self.connection.records
        raise AssertionError(f"unexpected catalog read: {self.query}")

    def fetchone(self):
        if self.query.startswith("SELECT pg_try_advisory_xact_lock("):
            return (True,)
        if self.query.startswith("SELECT mode, revision"):
            return self.connection.mode_row
        if self.query.startswith("SELECT count(*)"):
            return (0,)
        if self.query.startswith("SELECT to_regclass"):
            return (None,) * 6
        if self.query.startswith("SELECT current_setting('statement_timeout')"):
            return ("10s",)
        if self.query.startswith("SELECT EXISTS"):
            return (self.connection.remaining,)
        raise AssertionError(f"unexpected catalog read: {self.query}")


class MigrationRecorder:
    def __init__(self):
        self.statements = []

    def execute(self, query, params=None):
        self.statements.append((query, params))

    def fetchone(self):
        query, _ = self.statements[-1]
        return (None,) if "to_regclass" in query else None


def migration_callback(cursor):
    cursor.execute("SELECT 1")
