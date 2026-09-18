from __future__ import annotations

import json
import signal
import stat
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.e2e.regional.probes import cap_probe_supervisor as supervisor
from tests.regional._cov95_cap_probes import (
    RecordingDatabase,
    RecordingProcess,
    RecordingStore,
)

BASE_URL = "postgresql://example.invalid/base?sslmode=require"
TEST_URL = "postgresql://example.invalid/cap-test?sslmode=require"


def model(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    *,
    exit_code: int | None = None,
    wait_timeout: bool = False,
    failure: str | None = None,
) -> SimpleNamespace:
    events: list[str] = []
    database = RecordingDatabase(events)
    process = RecordingProcess(events, exit_code=exit_code, wait_timeout=wait_timeout)
    store = RecordingStore(events)
    launches: list[tuple[list[str], dict[str, Any]]] = []
    stores: list[tuple[str, dict[str, Any]]] = []
    signals: list[int] = []

    def open_store(url: str, **kwargs: Any) -> RecordingStore:
        events.append("store-open")
        stores.append((url, kwargs))
        if failure == "store":
            raise RuntimeError("fake store initialization failed")
        return store

    def launch(arguments: list[str], **kwargs: Any) -> RecordingProcess:
        events.append("launch")
        launches.append((arguments, kwargs))
        if failure == "launch":
            raise RuntimeError("fake child launch failed")
        return process

    monkeypatch.setenv("GPU_FAULT_BASE_STORE_URL", BASE_URL)
    monkeypatch.setenv("CAP_DATABASE_NAME", "cap-test")
    monkeypatch.setattr(supervisor, "WORK", root)
    monkeypatch.setattr(supervisor, "STOP", False)
    monkeypatch.setattr("psycopg.connect", database.connect)
    monkeypatch.setattr(supervisor, "PostgresStore", open_store)
    monkeypatch.setattr(
        supervisor,
        "subprocess",
        SimpleNamespace(Popen=launch, TimeoutExpired=subprocess.TimeoutExpired),
    )
    monkeypatch.setattr(
        supervisor,
        "signal",
        SimpleNamespace(
            signal=lambda number, _handler: signals.append(number),
            SIGTERM=signal.SIGTERM,
            SIGINT=signal.SIGINT,
        ),
    )
    monkeypatch.setattr(
        supervisor,
        "time",
        SimpleNamespace(sleep=lambda _seconds: supervisor.stop(signal.SIGTERM, None)),
    )
    if failure == "drop":
        database.fail_statement = "DROP DATABASE"
    elif failure == "create":
        database.fail_statement = "CREATE DATABASE"
    return SimpleNamespace(
        events=events,
        database=database,
        process=process,
        store=store,
        launches=launches,
        stores=stores,
        signals=signals,
    )


@pytest.mark.parametrize("wait_timeout", [False, True])
def test_supervisor_stops_child_before_dropping_its_database(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    wait_timeout: bool,
) -> None:
    fixture = model(monkeypatch, tmp_path, wait_timeout=wait_timeout)
    assert supervisor.main() == 0
    tail = ["terminate", "wait:35"]
    if wait_timeout:
        tail += ["kill", "wait:5"]
    assert fixture.events == [
        "drop",
        "create",
        "store-open",
        "store-close",
        "launch",
        *tail,
        "drop",
    ]
    assert fixture.database.closed == 3
    assert fixture.signals == [signal.SIGTERM, signal.SIGINT]
    assert fixture.stores == [
        (
            TEST_URL,
            {
                "pool_min_size": 1,
                "pool_max_size": 2,
                "pool_timeout_seconds": 30,
                "initialize_schema": True,
            },
        )
    ]
    assert fixture.store.closed is True
    assert (tmp_path / "store-url").read_text() == TEST_URL
    assert stat.S_IMODE((tmp_path / "store-url").stat().st_mode) == 0o600
    assert (tmp_path / "database-name").read_text() == "cap-test"
    arguments, options = fixture.launches[0]
    assert arguments[1:5] == ["-m", "uvicorn", "cap_probe_app:create_app", "--factory"]
    assert "--no-proxy-headers" in arguments
    assert options["env"]["GPU_FAULT_STORE_URL"] == TEST_URL
    assert "GPU_FAULT_BASE_STORE_URL" not in options["env"]
    assert json.loads(capsys.readouterr().out) == {
        "database": "cap-test",
        "database_isolated": True,
        "api_pid": fixture.process.pid,
    }


def test_supervisor_returns_child_failure_and_still_removes_database(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fixture = model(monkeypatch, tmp_path, exit_code=7)
    assert supervisor.main() == 7
    assert fixture.events == [
        "drop",
        "create",
        "store-open",
        "store-close",
        "launch",
        "drop",
    ]
    assert fixture.database.closed == 3


@pytest.mark.parametrize("failure", ["drop", "create", "store", "launch"])
def test_supervisor_cleans_only_after_successful_database_creation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: str
) -> None:
    fixture = model(monkeypatch, tmp_path, failure=failure)
    with pytest.raises(RuntimeError, match="fake"):
        supervisor.main()
    assert fixture.events.count("drop") == (2 if failure in {"store", "launch"} else 1)
    assert "terminate" not in fixture.events
    assert fixture.database.closed == len(fixture.database.connections)
    if failure in {"drop", "create"}:
        assert fixture.stores == []
    if failure != "launch":
        assert fixture.launches == []


def test_database_names_are_sql_identifiers_not_interpolated_programs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = RecordingDatabase([])
    monkeypatch.setattr("psycopg.connect", database.connect)
    name = 'cap-"quoted'
    supervisor.drop_database(BASE_URL, name)
    assert database.connections == [(BASE_URL, {"autocommit": True})]
    assert database.statements[0][1] == (name,)
    assert database.statements[1] == (
        'DROP DATABASE IF EXISTS "cap-""quoted" WITH (FORCE)',
        None,
    )
    assert database.closed == 1
    assert (
        supervisor.database_url(BASE_URL + "#fragment", "cap-test")
        == TEST_URL + "#fragment"
    )
