"""A batch's advisory locks are one statement, in Python ``sorted()`` order.

A control-worker telemetry batch took one ``health_signal_state/*`` advisory
lock round trip per signal key -- about 28 per host item -- inside one long
batch transaction, and the remediation budget paths looped the same way. The
keys now go to the server in one ``unnest`` statement ordered ``COLLATE "C"``:
byte order, which is the order of ``sorted()`` on the key text, so the lock
order rule (F-B2) still holds against single-key callers and against a pod
still running the per-key loop mid-rollout.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

import pytest

from gpu_fault.store.postgres.core import take_advisory_xact_locks
from gpu_fault.store.postgres.workflows import PostgresWorkflowMixin
from gpu_fault.store.postgres.xid import PostgresXidMixin

NOW = datetime(2030, 1, 1, tzinfo=UTC)


class _Stop(Exception):
    """Raised at the first statement after the locks: the test ends there."""


class _Cursor:
    def __init__(self, statements: list[tuple[str, Any]]) -> None:
        self.statements = statements

    def __enter__(self) -> _Cursor:
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def execute(self, sql: str, params: Any = None) -> None:
        if "pg_advisory_xact_lock" not in sql:
            raise _Stop(sql)
        self.statements.append((" ".join(sql.split()), params))


class _Db:
    def __init__(self) -> None:
        self.statements: list[tuple[str, Any]] = []

    @contextmanager
    def transaction(self):
        yield

    def cursor(self) -> _Cursor:
        return _Cursor(self.statements)


class _XidStore(PostgresXidMixin):
    def __init__(self) -> None:
        self.db = _Db()
        self._db = self.db


class _WorkflowStore(PostgresWorkflowMixin):
    def __init__(self) -> None:
        self.db = _Db()
        self._db = self.db

    def _get_for_update(self, kind: str, key: str) -> Any:
        raise _Stop(kind)


def _locked_keys(db: _Db) -> list[list[str]]:
    for sql, _params in db.statements:
        assert 'COLLATE "C"' in sql, sql
    return [list(params[0]) for _sql, params in db.statements]


def test_the_batch_statement_deduplicates_and_sorts_in_byte_order() -> None:
    db = _Db()
    with db.cursor() as cursor:
        take_advisory_xact_locks(cursor, ["b/1", "a/1", "b/1", "C/1"])
        take_advisory_xact_locks(cursor, [])

    assert _locked_keys(db) == [["C/1", "a/1", "b/1"]], (
        "one statement, byte order (upper case first), no duplicate, none for []"
    )


def test_a_health_signal_claim_locks_every_key_in_one_statement() -> None:
    store = _XidStore()
    keys = ["node-b/xid-79", "node-a/xid-79", "Node-c/row-remap", "node-a/xid-79"]

    with pytest.raises(_Stop):
        store.claim_health_signal_transitions([(key, True, NOW, 0.0) for key in keys])

    # The key text of mark_health_signal_notified's _state_transaction.
    assert _locked_keys(store.db) == [
        [f"health_signal_state/{key}" for key in sorted(set(keys))]
    ]


@pytest.mark.parametrize("path", ["claim", "extend"])
def test_remediation_budget_scopes_are_locked_in_one_statement(path: str) -> None:
    store = _WorkflowStore()
    claims = {"region": 2, "node:b": 1, "Rack-7": 1, "node:a": 1}

    with pytest.raises(_Stop):
        if path == "claim":
            store.claim_workflow(
                "workflow/a", "owner", 1, now=NOW, remediation_budget_claims=claims
            )
        else:
            store.extend_remediation_budget("workflow/a", "owner", claims, now=NOW)

    assert _locked_keys(store.db) == [
        [f"remediation_budget/{scope}" for scope in sorted(claims)]
    ]


def test_a_claim_without_budget_scopes_takes_no_advisory_lock() -> None:
    store = _WorkflowStore()

    with pytest.raises(_Stop):
        store.claim_workflow("workflow/a", "owner", 1, now=NOW)

    assert store.db.statements == []
