"""A failed withdrawal must not commit a duplicate-suppressing terminal decision."""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

from gpu_fault.models import TerminalEvent
from gpu_fault.store import PostgresStore
from scripts.e2e.regional.run_cap005_postgres_suite import validate_server
from tests._cov95_recovery_services import exercise_failed_withdrawal
from tests.store._postgres_processor_claim_support import (
    POSTGRES_URL,
    postgres_store_instance,
)

pytestmark = pytest.mark.skipif(
    not POSTGRES_URL, reason="requires the separately allocated serial PostgreSQL slot"
)


@pytest.fixture
def withdrawal_store() -> Iterator[PostgresStore]:
    if os.environ.get("PYTEST_XDIST_WORKER"):
        pytest.fail("withdrawal PostgreSQL transaction tests must run serially")
    assert POSTGRES_URL is not None, "the PostgreSQL test database must be allocated"
    validate_server(POSTGRES_URL)
    yield from postgres_store_instance()


@pytest.mark.parametrize(
    "boundary", ["list_active_workflow_incidents", "amend_workflow"]
)
def test_postgres_failed_withdrawal_remains_retryable(
    withdrawal_store: PostgresStore,
    failed_event: TerminalEvent,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    exercise_failed_withdrawal(withdrawal_store, failed_event, monkeypatch, boundary)
