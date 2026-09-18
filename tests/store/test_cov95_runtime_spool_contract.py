from __future__ import annotations

import pytest

from gpu_fault.store import InMemoryStore, SqliteStore
from tests.store._cov95_runtime_queue import (
    assert_rejected_spool_duplicates_are_not_acknowledged,
)


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    value = (
        InMemoryStore()
        if request.param == "memory"
        else SqliteStore(str(tmp_path / "spool-rejection.db"))
    )
    try:
        yield value
    finally:
        if isinstance(value, SqliteStore):
            value.close()


@pytest.mark.parametrize("reason", ["global", "cluster"])
def test_spool_rejection_does_not_acknowledge_a_discarded_duplicate(store, reason):
    assert_rejected_spool_duplicates_are_not_acknowledged(store, reason)
