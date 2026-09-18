"""Readonly schema evidence over fake SQL calls, with no PostgreSQL slot."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from scripts.e2e.regional.probes import state_table_snapshot as probe
from tests.regional import _cov95_residual_sql as sql_support
from tests.regional import _cov95_residual_support as support

residual_isolation = support.residual_isolation
readonly_cursor = sql_support.readonly_cursor


def test_view_comparison_translates_temporary_ddl_to_planner_reads(readonly_cursor):
    cursor = readonly_cursor
    assert (
        cursor.execute(
            "CREATE TEMP VIEW gf_control_state_view_probe AS SELECT id FROM fake_state;"
        )
        is cursor
    )
    cursor.execute(
        "SELECT pg_get_viewdef('gf_control_state_view_probe'::regclass, true)"
    )
    expected = {"plan": {"Node Type": "Result"}, "columns": [("id", 25)]}
    assert cursor.fetchone() == (expected,)
    assert cursor.fetchone() is None
    cursor.execute(
        "SELECT pg_get_viewdef(to_regclass(%s), true)",
        ("actual-view",),
        prepare=True,
        binary=False,
    )
    assert cursor.fetchone() == (expected,)
    cursor.execute("DROP VIEW gf_control_state_view_probe")
    with pytest.raises(RuntimeError, match="not prepared"):
        cursor.execute(
            "SELECT pg_get_viewdef('gf_control_state_view_probe'::regclass, true)"
        )
    assert all(
        text.startswith(("SELECT ", "EXPLAIN ")) for text, *_ in cursor.commands
    ), "the readonly cursor must never forward temporary-view DDL"
    assert cursor.commands[2] == (
        "SELECT pg_get_viewdef(to_regclass(%s), true)",
        ("actual-view",),
        True,
        False,
    )


@pytest.mark.parametrize("view_row", [None, (None,), (42,)])
def test_missing_view_definition_cannot_match_an_expected_plan(
    readonly_cursor, view_row
):
    readonly_cursor.view_row = view_row
    readonly_cursor.execute(
        "SELECT pg_get_viewdef(to_regclass(%s), true)", ("missing-view",)
    )
    assert readonly_cursor.fetchone() == (None,)
    assert len(readonly_cursor.commands) == 1


@pytest.mark.parametrize("plan_row", [None, ("not-list",), ([],), ([{}, {}],), ([{}],)])
def test_missing_or_malformed_query_plan_is_rejected(readonly_cursor, plan_row):
    readonly_cursor.plan_row = plan_row
    with pytest.raises(RuntimeError, match="plan"):
        readonly_cursor.query_plan("SELECT id FROM fake_state")
    assert len(readonly_cursor.commands) == 1


def test_missing_columns_and_plain_queries_preserve_driver_contract(readonly_cursor):
    readonly_cursor.columns = None
    with pytest.raises(RuntimeError, match="columns"):
        readonly_cursor.query_plan("SELECT id FROM fake_state")
    assert (
        readonly_cursor.execute("SELECT ordinary", (3,), prepare=False, binary=True)
        is readonly_cursor
    )
    assert readonly_cursor.fetchone() == ("ordinary-result",)
    assert readonly_cursor.commands[-1] == ("SELECT ordinary", (3,), False, True)


@pytest.mark.parametrize(
    "failure", ["schema-version", "migrations", "schema-validator", "status", None]
)
def test_snapshot_rejects_unproven_schema_and_always_closes_fake_connection(
    monkeypatch, failure
):
    calls, closed = [], []
    monkeypatch.setattr(
        probe,
        "StoreCredentials",
        lambda *args, **kwargs: SimpleNamespace(
            conninfo=lambda: "fake-dsn", source="file"
        ),
    )

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            closed.append(True)

        def execute(self, query):
            calls.append(query)
            if query.startswith("SELECT current_database"):
                return SimpleNamespace(
                    fetchone=lambda: (
                        "database",
                        "192.0.2.1",
                        5432,
                        ["public"],
                        False,
                        "on",
                    )
                )
            if "SELECT version FROM" in query:
                rows = [
                    (
                        probe.LATEST_POSTGRES_SCHEMA_VERSION
                        - int(failure == "schema-version"),
                    )
                ]
            elif "SELECT version,name,checksum" in query:
                rows = [
                    (entry.version, entry.name, entry.checksum)
                    for entry in probe.POSTGRES_SCHEMA_MIGRATIONS
                ]
                if failure == "migrations":
                    rows = rows[:-1]
            else:
                rows = []
            return SimpleNamespace(fetchall=lambda: rows)

    def connect(url, **kwargs):
        assert url == "fake-dsn"
        assert kwargs["autocommit"] is True
        assert kwargs["cursor_factory"] is probe.ReadOnlySchemaCursor
        assert "default_transaction_read_only=on" in kwargs["options"]
        return Connection()

    def validate(connection):
        calls.append("validator")
        if failure == "schema-validator":
            raise RuntimeError("schema invalid")

    def status(connection, kind, *, verify):
        calls.append(("status", kind, verify))
        if failure == "status":
            raise RuntimeError("status unavailable")
        return {"kind": kind}

    monkeypatch.setattr("psycopg.connect", connect)
    monkeypatch.setattr(probe, "validate_state_table_schema", validate)
    monkeypatch.setattr(probe, "state_table_status", status)
    if failure:
        with pytest.raises(RuntimeError):
            probe.snapshot("workflow", verify=True)
    else:
        result = probe.snapshot("workflow", verify=False)
        assert result["state"] == {"kind": "workflow"}
        assert result["writer"] is result["read_only"] is result["schema_valid"] is True
        assert len(result["database_sha256"]) == 64
        assert calls[-1] == ("status", "workflow", False)
    assert closed == [True]


@pytest.mark.parametrize("source,url", [("environment", "fake-dsn"), ("file", "")])
def test_current_credential_source_is_required_before_sql(monkeypatch, source, url):
    monkeypatch.setenv("GPU_FAULT_STORE_URL_FILE", "/private/current-dsn")
    monkeypatch.setattr(
        probe,
        "StoreCredentials",
        lambda *args, **kwargs: SimpleNamespace(conninfo=lambda: url, source=source),
    )
    monkeypatch.setattr("psycopg.connect", support.forbidden)
    with pytest.raises(RuntimeError, match="credentials"):
        probe.snapshot("remote_command", verify=True)


def test_main_refuses_invalid_mode_without_sql_details(monkeypatch, capsys):
    monkeypatch.setattr("psycopg.connect", support.forbidden)
    assert probe.main(["workflow", "unknown"]) == 1
    assert json.loads(capsys.readouterr().out) == {
        "verdict": "FAIL",
        "error_type": "ValueError",
    }
