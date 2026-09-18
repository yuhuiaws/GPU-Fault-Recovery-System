"""Serial PostgreSQL lease takeover, not a deployed Pod or saturation test."""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

from gpu_fault.store import PostgresStore
from scripts.e2e.regional.run_cap005_postgres_suite import validate_server
from tests.regional._cov95_ha_busy_worker import exercise_busy_worker_takeover
from tests.store._postgres_processor_claim_support import (
    POSTGRES_URL,
    postgres_store_instance,
)

pytestmark = pytest.mark.skipif(
    not POSTGRES_URL, reason="requires the separately allocated serial PostgreSQL slot"
)


@pytest.fixture
def busy_worker_store() -> Iterator[PostgresStore]:
    if os.environ.get("PYTEST_XDIST_WORKER"):
        pytest.fail("busy-worker PostgreSQL takeover tests must run serially")
    assert POSTGRES_URL is not None, "the PostgreSQL test database must be allocated"
    validate_server(POSTGRES_URL)
    yield from postgres_store_instance()


@pytest.mark.parametrize("role", ["processor", "spool"])
def test_postgres_busy_owner_loss_reclaims_once_and_fences_late_completion(
    busy_worker_store: PostgresStore, monkeypatch: pytest.MonkeyPatch, role: str
) -> None:
    result = exercise_busy_worker_takeover(busy_worker_store, monkeypatch, role=role)
    assert result["handler_attempts"] == 2, (
        "only the original owner and one replacement execute the request"
    )
    assert result["late_result_fenced"] is True, (
        "the PostgreSQL result must reject the old owner's late completion"
    )
    assert result["queue_depth"] == 0, "the owned durable queue must drain"
    assert result["cpu_saturation_tested"] is False, (
        "lease takeover is not a CPU saturation or deployed Pod proof"
    )
