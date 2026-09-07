"""The stuck-workflow baseline audit: read-only for real, terminal like the dispatcher.

Two things the Postgres-only suite in ``tests/store`` cannot see cheaply: that
the read-only guard is the transaction's *first* statement (a session-level
``SET default_transaction_read_only`` inside an open transaction applies to the
next one), and -- with a database -- that a BLOCKED predecessor whose safety
plan settled releases its successor for the stuck count the way the dispatcher
releases it for dispatch.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.models import BlockedKind, IncidentState, WorkflowStatus
from tests._builders import fault_incident, workflow_request
from tests._script_loader import lazy_script_module

ROOT = Path(__file__).resolve().parents[2]
AUDIT = lazy_script_module(
    ROOT / "scripts/e2e/regional/audit_stuck_workflow_baseline.py"
)
POSTGRES_URL = os.getenv("GPU_FAULT_TEST_POSTGRES_URL")
NOW = datetime(2026, 9, 7, 15, 0, tzinfo=timezone.utc)
HOUR_AGO = NOW - timedelta(hours=1)


class FakeCursor:
    def __init__(self, statements: list[str]) -> None:
        self.statements = statements

    def execute(self, sql: str, parameters: Any = None) -> None:
        self.statements.append(" ".join(sql.split()))

    def fetchone(self) -> tuple[int, ...]:
        return (0, 0)

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


class FakeConnection:
    def __init__(self, statements: list[str]) -> None:
        self.statements = statements
        self.rolled_back = False

    def cursor(self) -> FakeCursor:
        return FakeCursor(self.statements)

    def rollback(self) -> None:
        self.rolled_back = True

    def __enter__(self) -> FakeConnection:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def test_read_only_is_the_first_statement_of_the_audit_transaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    statements: list[str] = []
    connection = FakeConnection(statements)
    monkeypatch.setattr(AUDIT.psycopg, "connect", lambda url: connection)

    result = AUDIT.run("postgresql://ignored", now=NOW)

    assert statements[0] == "SET TRANSACTION READ ONLY", statements[:2]
    assert "default_transaction_read_only" not in " ".join(statements), (
        "the session-level default does not apply to the transaction it is set in"
    )
    assert statements[1].startswith("SET statement_timeout"), statements[:2]
    assert connection.rolled_back is True
    assert result["stuck_pending"] == 0


def test_stuck_pending_predicate_releases_a_settled_blocked_predecessor() -> None:
    sql = " ".join(AUDIT.STUCK_PENDING_SQL.split())

    # The dispatcher's reading: open is PENDING/SAFETY_PENDING/RUNNING, plus a
    # BLOCKED row an operator still owns; everything else -- including a
    # BLOCKED row whose safety plan settled, or a predecessor that is gone --
    # releases the successor.
    assert "NOT IN ('PENDING', 'SAFETY_PENDING', 'RUNNING')" in sql
    assert "'blocked_kind' IN ('NEEDS_OPERATOR', 'INTERNAL_ERROR')" in sql
    assert "NOT EXISTS" in sql
    assert "IN ('SUCCEEDED', 'FAILED', 'SUPERSEDED')" not in sql, (
        "an explicit terminal list silently drops BLOCKED and any future status"
    )


@pytest.mark.skipif(
    not POSTGRES_URL, reason="GPU_FAULT_TEST_POSTGRES_URL is not configured"
)
def test_stuck_pending_counts_behind_a_settled_blocked_predecessor_on_postgres() -> (
    None
):
    from tests.store._postgres_processor_claim_support import postgres_store_instance

    def incident(incident_id: str, points_at: str) -> Any:
        return fault_incident(
            incident_id,
            f"event-{incident_id}",
            state=IncidentState.ACTION_PENDING,
            workflow_request_id=points_at,
            fencing_token=1,
            created_at=HOUR_AGO,
            updated_at=HOUR_AGO,
        )

    def pending(request_id: str, incident_id: str, predecessor: str) -> Any:
        return workflow_request(
            request_id,
            incident_id,
            status=WorkflowStatus.PENDING,
            fencing_token=1,
            predecessor_workflow_id=predecessor,
            created_at=HOUR_AGO,
            updated_at=HOUR_AGO,
        )

    def blocked(request_id: str, incident_id: str, kind: BlockedKind | None) -> Any:
        return workflow_request(
            request_id,
            incident_id,
            status=WorkflowStatus.BLOCKED,
            fencing_token=1,
            blocked_kind=kind,
            created_at=HOUR_AGO,
            updated_at=HOUR_AGO,
        )

    for store in postgres_store_instance():
        store.save_incident_and_workflow(
            incident("inc-settled", "wf-after-settled"),
            pending("wf-after-settled", "inc-settled", "wf-settled"),
        )
        store.save_workflow(
            blocked("wf-settled", "inc-settled", BlockedKind.SAFETY_SETTLED)
        )
        store.save_incident_and_workflow(
            incident("inc-operator", "wf-after-operator"),
            pending("wf-after-operator", "inc-operator", "wf-operator"),
        )
        store.save_workflow(
            blocked("wf-operator", "inc-operator", BlockedKind.NEEDS_OPERATOR)
        )
        store.save_incident_and_workflow(
            incident("inc-gone", "wf-after-gone"),
            pending("wf-after-gone", "inc-gone", "wf-gone-predecessor"),
        )

        # Read through the audit's own connection path, as an operator would.
        result = AUDIT.run(POSTGRES_URL, now=NOW)

        # The settled BLOCKED and the missing predecessor release their
        # successors; the operator-owned BLOCKED still holds its own.
        assert result["stuck_pending"] == 2, result
