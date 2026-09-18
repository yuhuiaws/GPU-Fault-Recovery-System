from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from gpu_fault.store import InMemoryStore, SqliteStore
from tests.store._cov95_runtime_spool_fencing import (
    ACTIONS,
    TRANSITIONS,
    Action,
    Transition,
    assert_invalid_fence_cannot_mutate,
    assert_latest_wins_keeps_retry_budget,
    assert_old_claim_cannot_mutate_live_replacement,
    assert_reused_lane_does_not_reuse_claim,
)


@pytest.fixture(params=["memory", "sqlite"])
def spool_store(
    request: pytest.FixtureRequest, tmp_path: Path
) -> Iterator[InMemoryStore | SqliteStore]:
    store = (
        InMemoryStore()
        if request.param == "memory"
        else SqliteStore(str(tmp_path / "spool-fencing.db"))
    )
    try:
        yield store
    finally:
        if isinstance(store, SqliteStore):
            store.close()


@pytest.mark.parametrize("action", ACTIONS)
@pytest.mark.parametrize("replacement_owner", ["owner-b", "owner-a"])
def test_old_spool_claim_cannot_mutate_a_still_live_replacement(
    spool_store: InMemoryStore | SqliteStore, action: Action, replacement_owner: str
) -> None:
    assert_old_claim_cannot_mutate_live_replacement(
        spool_store, action, replacement_owner
    )


@pytest.mark.parametrize("action", ACTIONS)
@pytest.mark.parametrize("transition", TRANSITIONS)
def test_spool_claim_is_not_reused_by_lane_lifecycle(
    spool_store: InMemoryStore | SqliteStore, action: Action, transition: Transition
) -> None:
    assert_reused_lane_does_not_reuse_claim(spool_store, action, transition)


@pytest.mark.parametrize("action", ACTIONS)
@pytest.mark.parametrize("invalid", ["missing", "empty", "foreign", "revision"])
def test_spool_rejects_missing_and_mismatched_fences(
    spool_store: InMemoryStore | SqliteStore, action: Action, invalid: str
) -> None:
    assert_invalid_fence_cannot_mutate(spool_store, action, invalid)


def test_spool_claim_tokens_do_not_reset_latest_wins_retry_budget(
    spool_store: InMemoryStore | SqliteStore,
) -> None:
    assert_latest_wins_keeps_retry_budget(spool_store)
