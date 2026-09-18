"""Exercise the diagnostics fixture lifecycle without a PostgreSQL connection."""

from __future__ import annotations

from contextlib import contextmanager

import pytest

from gpu_fault.store.postgres import index_builder
from tests.regional import test_bootstrap_store_proof_postgres as temporary
from tests.store import test_store_migrate as migrate

diagnostics_database = migrate.diagnostics_database
without_pg_stat_statements = migrate.without_pg_stat_statements


@pytest.fixture(params=[False, True])
def extension_state(request, monkeypatch):
    psycopg = pytest.importorskip("psycopg")
    from psycopg.conninfo import conninfo_to_dict

    shared = "postgresql://127.0.0.1/shared_test_database"
    statements = []
    states = {"shared_test_database": request.param}
    monkeypatch.setattr(temporary, "URL", shared)
    monkeypatch.setattr(migrate, "POSTGRES_URL", shared)

    class Connection:
        def __init__(self, database):
            self.database = database

        @contextmanager
        def cursor(self):
            yield self

        def execute(self, query):
            text = query if isinstance(query, str) else query.as_string()
            statements.append((self.database, text))
            if text.startswith("DROP EXTENSION"):
                states[self.database] = False
            elif text.startswith("CREATE EXTENSION"):
                states[self.database] = True
            elif not text.startswith(("CREATE DATABASE", "DROP DATABASE")):
                raise AssertionError("unexpected diagnostics fixture statement")

    @contextmanager
    def connect(url, *, autocommit):
        assert autocommit is True
        yield Connection(conninfo_to_dict(url)["dbname"])

    monkeypatch.setattr(psycopg, "connect", connect)
    yield states

    private = set(states) - {"shared_test_database"}
    assert len(private) == 1, "the scenario must use exactly one temporary database"
    name = private.pop()
    assert states["shared_test_database"] is request.param, (
        "the shared database extension state must survive the scenario"
    )
    assert [
        query for database, query in statements if database == "shared_test_database"
    ] == [f'CREATE DATABASE "{name}"', f'DROP DATABASE "{name}"'], (
        "only creation and cleanup of the private database may touch the shared one"
    )


@pytest.mark.parametrize("report_fails", [False, True])
def test_cli_extensions_are_isolated_even_when_reporting_fails(
    extension_state, request, monkeypatch, capsys, report_fails
) -> None:
    from psycopg.conninfo import conninfo_to_dict

    url = request.getfixturevalue("without_pg_stat_statements")
    name = conninfo_to_dict(url)["dbname"]
    assert name != "shared_test_database", "the fixture must return its private DSN"
    assert extension_state[name] is False, "the CLI must start without the extension"

    def diagnostics(cursor):
        assert cursor.database == name, "the CLI must connect to the temporary DB"
        assert extension_state[name] is True, "the CLI must install the extension"
        if report_fails:
            raise RuntimeError("simulated diagnostics report failure")
        return {
            "pg_stat_statements_installed": extension_state[name],
            "log_lock_waits": "on",
            "deadlock_timeout": "1s",
            "log_min_duration_statement": "-1",
            "shared_preload_libraries": "",
        }

    monkeypatch.setattr(index_builder, "database_diagnostics", diagnostics)
    if report_fails:
        with pytest.raises(RuntimeError, match="simulated diagnostics report failure"):
            migrate.test_store_migrate_ensure_diagnostics_installs_pg_stat_statements(
                monkeypatch, capsys, url
            )
    else:
        migrate.test_store_migrate_ensure_diagnostics_installs_pg_stat_statements(
            monkeypatch, capsys, url
        )
