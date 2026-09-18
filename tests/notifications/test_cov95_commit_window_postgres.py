"""Serial PostgreSQL counterpart of the backend-neutral delivery crash contract."""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

from gpu_fault.store import PostgresStore
from scripts.e2e.regional.run_cap005_postgres_suite import validate_server
from tests.regional._cov95_notify_commit_window import (
    CrashPoint,
    exercise_provider_commit_window,
)
from tests.store._postgres_processor_claim_support import (
    POSTGRES_URL,
    postgres_store_instance,
)

pytestmark = pytest.mark.skipif(
    not POSTGRES_URL, reason="requires the separately allocated serial PostgreSQL slot"
)


@pytest.fixture
def postgres_notification_store() -> Iterator[PostgresStore]:
    if os.environ.get("PYTEST_XDIST_WORKER"):
        pytest.fail("notification crash-window PostgreSQL tests must run serially")
    assert POSTGRES_URL is not None, "the test database must be explicitly allocated"
    validate_server(POSTGRES_URL)
    yield from postgres_store_instance()


@pytest.mark.parametrize("asynchronous", [False, True], ids=["inline", "outbox"])
@pytest.mark.parametrize("kind", ["gpu-reset", "workload-restart"])
@pytest.mark.parametrize(
    "crash", ["before-provider", "accepted-before-commit", "committed-before-ack"]
)
def test_postgres_provider_commit_crash_contract(
    postgres_notification_store: PostgresStore,
    monkeypatch: pytest.MonkeyPatch,
    asynchronous: bool,
    kind: str,
    crash: CrashPoint,
) -> None:
    result = exercise_provider_commit_window(
        postgres_notification_store,
        monkeypatch,
        asynchronous=asynchronous,
        crash=crash,
        kind=kind,
    )
    assert result["final_status"] == "SENT", "the real PostgreSQL result is durable"
    assert result["duplicate_delivery_possible"] is (
        crash == "accepted-before-commit"
    ), "PostgreSQL cannot atomically commit an external provider acknowledgement"
