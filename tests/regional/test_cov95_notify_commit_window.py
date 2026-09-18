from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from gpu_fault.store import InMemoryStore, SqliteStore
from tests.regional._cov95_notify_commit_window import (
    CrashPoint,
    exercise_provider_commit_window,
)


@pytest.fixture(params=["memory", "sqlite-unit-simulator"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> Any:
    if request.param == "memory":
        yield InMemoryStore()
    else:
        value = SqliteStore(str(tmp_path / "unit-notifications.db"))
        try:
            yield value
        finally:
            value.close()


@pytest.mark.parametrize("asynchronous", [False, True], ids=["inline", "outbox"])
@pytest.mark.parametrize("kind", ["gpu-reset", "workload-restart"])
@pytest.mark.parametrize(
    "crash", ["before-provider", "accepted-before-commit", "committed-before-ack"]
)
def test_backend_neutral_provider_commit_crash_contract(
    store: Any,
    monkeypatch: pytest.MonkeyPatch,
    asynchronous: bool,
    kind: str,
    crash: CrashPoint,
) -> None:
    result = exercise_provider_commit_window(
        store, monkeypatch, asynchronous=asynchronous, crash=crash, kind=kind
    )
    assert result["final_status"] == "SENT"
    assert result["duplicate_delivery_possible"] is (crash == "accepted-before-commit")
