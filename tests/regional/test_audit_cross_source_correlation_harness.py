"""The cross-source correlation audit harness runs to its PASS line.

``scripts/e2e/regional/audit_cross_source_correlation_postgres.py`` called
``IncidentOrchestrator.correlate_provider_event`` without the keyword-only
``cluster_id`` the provider correlation takes since the regional tenant
boundary (H-14), so it raised ``TypeError`` before its first assertion
(batch 237). Nothing in the repository runs it: the regional acceptance
history lists it as an audit script to run by hand against the site's Aurora.
Here the Postgres store and the cleanup connection are in-memory doubles; the
policy decision and the marker correlation are the product's own.
"""

from __future__ import annotations

from contextlib import contextmanager

import pytest

from gpu_fault.store import InMemoryStore
from scripts.e2e.regional import audit_cross_source_correlation_postgres as audit


class _ClosableStore(InMemoryStore):
    def __init__(self, _dsn: str) -> None:
        super().__init__()

    def close(self) -> None:
        return None


class _Cursor:
    def execute(self, *_args: object) -> None:
        return None


class _Connection:
    @contextmanager
    def cursor(self):
        yield _Cursor()


@contextmanager
def _connect(_dsn: str):
    yield _Connection()


def test_the_audit_correlates_within_its_cluster_and_prints_pass(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(audit, "PostgresStore", _ClosableStore)
    monkeypatch.setattr(audit.psycopg, "connect", _connect)
    monkeypatch.setattr(audit, "store_dsn", lambda: "postgresql://audit.invalid/db")

    audit.main()

    assert capsys.readouterr().out.startswith("PASS"), "the audit reached its end"
