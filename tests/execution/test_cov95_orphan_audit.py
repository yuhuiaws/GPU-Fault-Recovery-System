from __future__ import annotations

import pytest

from gpu_fault.store import InMemoryStore, SqliteStore
from tests.store import _cov95_runtime_orphan_audit as contract


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    value = (
        InMemoryStore()
        if request.param == "memory"
        else SqliteStore(str(tmp_path / "orphan-audit.db"))
    )
    yield value
    if isinstance(value, SqliteStore):
        value.close()


@pytest.mark.parametrize("status", contract.OPEN_STATUSES)
@pytest.mark.parametrize(
    "after_write", [False, True], ids=["before-audit", "after-audit"]
)
def test_orphan_cancellation_cannot_commit_without_its_workflow_audit(
    store, monkeypatch, status, after_write
) -> None:
    contract.assert_audit_rollback(store, monkeypatch, status, after_write=after_write)


@pytest.mark.parametrize(("field", "value"), contract.DRIFT)
def test_orphan_cancellation_rechecks_the_current_workflow(store, field, value) -> None:
    contract.assert_stale_workflow_is_untouched(store, field, value)


def test_orphan_cancellation_requires_an_existing_workflow(store) -> None:
    contract.assert_missing_workflow_is_untouched(store)


def test_orphan_cancellation_audits_only_commands_it_changed(store) -> None:
    contract.assert_exact_changed_ids(store)


def test_orphan_audit_failure_is_isolated_per_workflow(store, monkeypatch) -> None:
    contract.assert_workflow_failure_is_isolated(store, monkeypatch)
