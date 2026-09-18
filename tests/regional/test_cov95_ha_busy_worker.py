from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from gpu_fault.store import InMemoryStore, SqliteStore
from tests.regional._cov95_ha_busy_worker import exercise_busy_worker_takeover


@pytest.fixture(params=["memory", "sqlite-unit-simulator"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> Any:
    if request.param == "memory":
        yield InMemoryStore()
    else:
        value = SqliteStore(str(tmp_path / "unit-busy-worker.db"))
        try:
            yield value
        finally:
            value.close()


@pytest.mark.parametrize("role", ["processor", "spool"])
def test_busy_cpu_owner_loss_reclaims_and_fences_late_work(
    store: Any, monkeypatch: pytest.MonkeyPatch, role: str
) -> None:
    result = exercise_busy_worker_takeover(store, monkeypatch, role=role)
    assert result["late_result_fenced"] is True
    assert result["queue_depth"] == 0
    assert result["validation_scope"] == "local-worker-lease-simulation"
