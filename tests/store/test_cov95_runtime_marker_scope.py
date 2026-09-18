from __future__ import annotations

import pytest

from gpu_fault.store import InMemoryStore, SqliteStore
from tests.store._cov95_runtime_markers import (
    SCOPE_FIELDS,
    assert_cluster_filter_precedes_limit,
)


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    value = (
        InMemoryStore()
        if request.param == "memory"
        else SqliteStore(str(tmp_path / "marker-scope.db"))
    )
    try:
        yield value
    finally:
        if isinstance(value, SqliteStore):
            value.close()


@pytest.mark.parametrize("scope_field", SCOPE_FIELDS)
@pytest.mark.parametrize(
    "own_present", [False, True], ids=["no-own-marker", "own-marker"]
)
def test_cluster_bound_marker_window_filters_before_limit(
    store, scope_field, own_present
):
    assert_cluster_filter_precedes_limit(store, scope_field, own_present=own_present)
