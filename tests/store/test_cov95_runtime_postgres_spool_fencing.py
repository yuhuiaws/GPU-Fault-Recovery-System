"""Serial local PostgreSQL spool fences, not a deployed takeover acceptance."""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from threading import Barrier

import pytest

from gpu_fault.store import PostgresStore
from tests.store._cov95_runtime_postgres import validated_url
from tests.store._cov95_runtime_spool_fencing import (
    ACTIONS,
    LEASE,
    NOW,
    TRANSITIONS,
    Action,
    Transition,
    assert_invalid_fence_cannot_mutate,
    assert_latest_wins_keeps_retry_budget,
    assert_old_claim_cannot_mutate_live_replacement,
    assert_reused_lane_does_not_reuse_claim,
    claim,
    mutate,
    seed,
)
from tests.store._postgres_processor_claim_support import postgres_store_instance


@pytest.fixture
def spool_store() -> Iterator[PostgresStore]:
    validated_url()
    yield from postgres_store_instance()


@pytest.mark.parametrize("action", ACTIONS)
@pytest.mark.parametrize("replacement_owner", ["owner-b", "owner-a"])
def test_postgres_old_spool_claim_cannot_mutate_a_still_live_replacement(
    spool_store: PostgresStore, action: Action, replacement_owner: str
) -> None:
    assert_old_claim_cannot_mutate_live_replacement(
        spool_store, action, replacement_owner
    )


@pytest.mark.parametrize("action", ACTIONS)
@pytest.mark.parametrize("transition", TRANSITIONS)
def test_postgres_spool_claim_is_not_reused_by_lane_lifecycle(
    spool_store: PostgresStore, action: Action, transition: Transition
) -> None:
    assert_reused_lane_does_not_reuse_claim(spool_store, action, transition)


@pytest.mark.parametrize("action", ACTIONS)
@pytest.mark.parametrize("invalid", ["missing", "empty", "foreign", "revision"])
def test_postgres_spool_rejects_missing_and_mismatched_fences(
    spool_store: PostgresStore, action: Action, invalid: str
) -> None:
    assert_invalid_fence_cannot_mutate(spool_store, action, invalid)


def test_postgres_spool_claim_tokens_do_not_reset_latest_wins_retry_budget(
    spool_store: PostgresStore,
) -> None:
    assert_latest_wins_keeps_retry_budget(spool_store)


@pytest.mark.parametrize("action", ACTIONS)
def test_postgres_stale_callback_racing_current_completion_cannot_win(
    spool_store: PostgresStore, action: Action
) -> None:
    seed(spool_store)
    old = claim(spool_store, "owner-a", NOW)
    at = NOW + LEASE
    current = claim(spool_store, "owner-b", at)
    gate = Barrier(2, timeout=5)
    with (
        closing(PostgresStore(validated_url(), initialize_schema=False)) as peer,
        ThreadPoolExecutor(max_workers=2) as executor,
    ):

        def late_callback() -> tuple[int, int]:
            gate.wait()
            return mutate(peer, [old], action, at)

        def finish_current() -> int:
            gate.wait()
            return spool_store.complete_telemetry_spool([current])

        late = executor.submit(late_callback)
        finish = executor.submit(finish_current)
        assert late.result(timeout=10) == (0, 0), (
            "the stale callback cannot win a race against the current claim"
        )
        assert finish.result(timeout=10) == 1, (
            "the valid completion must own the only successful state transition"
        )
    assert spool_store.telemetry_spool_stats(now=at)["depth"] == 0, (
        "racing callbacks must leave no lost or resurrected spool row"
    )


@pytest.mark.parametrize("reuse_owner", [False, True])
def test_postgres_competing_reclaimers_get_only_one_live_fence(
    spool_store: PostgresStore, reuse_owner: bool
) -> None:
    seed(spool_store)
    old = claim(spool_store, "owner-a", NOW)
    at = NOW + LEASE
    gate = Barrier(2, timeout=5)
    with (
        closing(PostgresStore(validated_url(), initialize_schema=False)) as peer,
        ThreadPoolExecutor(max_workers=2) as executor,
    ):

        def take(store: PostgresStore, owner: str):
            gate.wait()
            return store.claim_telemetry_spool(
                owner, now=at, lease_duration=LEASE, limit=1
            )

        first = executor.submit(take, spool_store, "owner-a" if reuse_owner else "b")
        second = executor.submit(take, peer, "owner-a" if reuse_owner else "c")
        rows = first.result(timeout=10) + second.result(timeout=10)
        assert len(rows) == 1, "SKIP LOCKED reclaim must hand the lane to one owner"
        assert rows[0].lease_token != old.lease_token, (
            "even an identical owner ID needs a new fence after lease expiry"
        )
        for action in ACTIONS:
            assert mutate(peer, [old], action, at) == (0, 0), (
                "no old-owner operation may interfere while the race winner is busy"
            )
        assert peer.complete_telemetry_spool(rows) == 1, (
            "the winning claim must remain valid across independent Store connections"
        )
