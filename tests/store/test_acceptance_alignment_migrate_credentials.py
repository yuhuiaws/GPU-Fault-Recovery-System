from __future__ import annotations

import json
import sys
from contextlib import nullcontext

import pytest

from gpu_fault import state_table_migrate, store_migrate


def test_state_table_cli_reads_current_projected_credentials_each_invocation(
    tmp_path, monkeypatch, capsys
):
    path = tmp_path / "dsn"
    path.write_text("postgresql://first.invalid/control")
    monkeypatch.setenv("GPU_FAULT_STORE_URL_FILE", str(path))
    monkeypatch.setenv("GPU_FAULT_STORE_URL", "postgresql://stale.invalid/control")
    seen = []
    monkeypatch.setattr(
        state_table_migrate,
        "state_table_maintenance",
        lambda url: seen.append(url) or nullcontext(object()),
    )
    monkeypatch.setattr(
        state_table_migrate, "state_table_status", lambda *_args: {"verified": True}
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["store-migrate", "--state-table-status", "--state-table-kind", "workflow"],
    )
    store_migrate.main()
    assert json.loads(capsys.readouterr().out) == {"verified": True}
    path.write_text("postgresql://rotated.invalid/control")
    store_migrate.main()
    assert seen == [
        "postgresql://first.invalid/control",
        "postgresql://rotated.invalid/control",
    ]
    assert "rotated.invalid" not in capsys.readouterr().out


@pytest.mark.parametrize(
    "problem", ["missing", "empty", "conflicting-explicit", "two-databases"]
)
def test_migration_refuses_ambiguous_projection_before_connection(
    tmp_path, monkeypatch, problem
):
    path = tmp_path / "dsn"
    monkeypatch.setenv("GPU_FAULT_STORE_URL_FILE", str(path))
    monkeypatch.setenv("GPU_FAULT_STORE_URL", "postgresql://stale.invalid/control")
    if problem != "missing":
        path.write_text(
            "" if problem == "empty" else "postgresql://current.invalid/control"
        )
    arguments = ["--schema-preflight"]
    if problem == "conflicting-explicit":
        arguments += ["--postgres-url", "postgresql://other.invalid/control"]
    if problem == "two-databases":
        arguments = ["--source-postgres-url", "postgresql://source.invalid/control"]
    monkeypatch.setattr(sys, "argv", ["store-migrate", *arguments])
    import psycopg

    monkeypatch.setattr(
        psycopg,
        "connect",
        lambda *_a, **_kw: pytest.fail("ambiguous credentials reached I/O"),
    )
    with pytest.raises(SystemExit) as raised:
        store_migrate.main()
    assert raised.value.code == 2
