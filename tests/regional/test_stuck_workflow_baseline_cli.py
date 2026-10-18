"""Command-line surface of the stuck-workflow baseline audit.

The queries themselves are covered elsewhere; this pins how the entry point
finds its DSN (file, env, explicit flag, or a clean exit 2), what it prints
(summary or JSON), the markdown row it appends, and the non-zero exit the
``--fail-on-nonzero`` guard turns a dirty census into. ``run`` is replaced by a
scripted result, so no connection is ever opened.
"""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path
from typing import Any

import pytest

from tests._script_loader import lazy_script_module

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts/e2e/regional/audit_stuck_workflow_baseline.py"
)
# Loaded lazily: the script imports ``psycopg`` at module scope, and the suite
# must still collect when the optional driver is absent.
audit = lazy_script_module(SCRIPT)
FAKE_DSN = "postgresql://audit:placeholder@db.invalid:5432/gpu_fault"
CLEAN = {
    "observed_at": "2026-09-07T15:00:00+00:00",
    "orphan": {"total": 0, "same_generation": 0},
    "zombie": 0,
    "stuck_pending": 0,
}
DIRTY = {
    "observed_at": "2026-09-08T09:30:00+00:00",
    "orphan": {"total": 3, "same_generation": 1},
    "zombie": 0,
    "stuck_pending": 2,
}


@pytest.fixture
def scripted_run(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    calls: dict[str, Any] = {"result": CLEAN}

    def run(store_url: str, **options: Any) -> dict[str, Any]:
        calls["store_url"] = store_url
        calls["options"] = options
        return dict(calls["result"])

    monkeypatch.setattr(audit, "run", run)
    monkeypatch.delenv("GPU_FAULT_STORE_URL_FILE", raising=False)
    monkeypatch.delenv("GPU_FAULT_STORE_URL", raising=False)
    return calls


def test_store_dsn_reads_the_configured_file_and_refuses_empty_input(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dsn_file = tmp_path / "postgres-url"
    dsn_file.write_text(f"{FAKE_DSN}\n", encoding="utf-8")
    monkeypatch.setenv("GPU_FAULT_STORE_URL_FILE", str(dsn_file))
    assert audit.store_dsn() == FAKE_DSN

    dsn_file.write_text("", encoding="utf-8")
    with pytest.raises(RuntimeError, match="file is empty"):
        audit.store_dsn()

    monkeypatch.setenv("GPU_FAULT_STORE_URL_FILE", str(tmp_path / "absent"))
    with pytest.raises(FileNotFoundError):
        audit.store_dsn()

    monkeypatch.setenv("GPU_FAULT_STORE_URL_FILE", "")
    with pytest.raises(RuntimeError, match="file path is empty"):
        audit.store_dsn()


def test_store_dsn_falls_back_to_the_env_without_a_configured_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GPU_FAULT_STORE_URL_FILE", raising=False)
    monkeypatch.setenv("GPU_FAULT_STORE_URL", FAKE_DSN)

    assert audit.store_dsn() == FAKE_DSN


def test_main_exits_2_without_any_dsn_source(
    scripted_run: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    assert audit.main([]) == 2

    captured = capsys.readouterr()
    assert "neither GPU_FAULT_STORE_URL_FILE nor GPU_FAULT_STORE_URL" in captured.err
    assert captured.out == ""
    assert "store_url" not in scripted_run, "no query may run without a DSN"


def test_main_prints_the_summary_and_passes_the_tuning_flags_through(
    scripted_run: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    code = audit.main(
        [
            "--store-url",
            FAKE_DSN,
            "--orphan-cutoff-seconds",
            "120",
            "--stuck-idle-seconds",
            "900",
            "--statement-timeout-seconds",
            "7",
        ]
    )

    assert code == 0
    assert scripted_run["store_url"] == FAKE_DSN
    assert scripted_run["options"] == {
        "orphan_cutoff_seconds": 120,
        "stuck_idle_seconds": 900,
        "statement_timeout_seconds": 7,
    }
    out = capsys.readouterr().out
    assert "Q-ORPHAN         0 (same generation: 0)" in out
    assert "Q-STUCK-PENDING  0" in out
    assert FAKE_DSN not in out


def test_main_json_mode_prints_only_the_result(
    scripted_run: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("GPU_FAULT_STORE_URL", FAKE_DSN)

    assert audit.main(["--json"]) == 0

    assert json.loads(capsys.readouterr().out) == CLEAN
    assert scripted_run["store_url"] == FAKE_DSN


def test_main_appends_a_markdown_row_and_fails_on_a_dirty_census(
    scripted_run: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scripted_run["result"] = DIRTY
    table = tmp_path / "baseline.md"
    table.write_text("| day | orphan | zombie | stuck | note |\n", encoding="utf-8")

    code = audit.main(
        [
            "--store-url",
            FAKE_DSN,
            "--append-markdown",
            str(table),
            "--note",
            "after rollout",
            "--fail-on-nonzero",
        ]
    )

    assert code == 1
    assert table.read_text(encoding="utf-8").splitlines()[-1] == (
        "| 2026-09-08 | 3 / 1 | 0 | 2 | after rollout |"
    )
    assert "Q-ORPHAN         3 (same generation: 1)" in capsys.readouterr().out


def test_fail_on_nonzero_leaves_a_clean_census_at_exit_0(
    scripted_run: dict[str, Any],
) -> None:
    assert audit.main(["--store-url", FAKE_DSN, "--fail-on-nonzero"]) == 0


class FakeCursor:
    def __init__(self, statements: list[str]) -> None:
        self.statements = statements

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def execute(self, sql: str, parameters: Any = None) -> None:
        self.statements.append(" ".join(sql.split()))

    def fetchone(self) -> tuple[int, int]:
        return (0, 0)


class FakeConnection:
    def __init__(self) -> None:
        self.statements: list[str] = []
        self.rolled_back = False

    def __enter__(self) -> FakeConnection:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def cursor(self) -> FakeCursor:
        return FakeCursor(self.statements)

    def rollback(self) -> None:
        self.rolled_back = True


def test_module_entry_point_exits_with_main_result(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import psycopg

    connection = FakeConnection()
    monkeypatch.setattr(psycopg, "connect", lambda url: connection)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--store-url", FAKE_DSN, "--json"])

    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(SCRIPT), run_name="__main__")

    assert exit_info.value.code == 0
    assert connection.statements[0] == "SET TRANSACTION READ ONLY"
    assert connection.rolled_back is True
    result = json.loads(capsys.readouterr().out)
    assert result["stuck_pending"] == 0 and result["zombie"] == 0
