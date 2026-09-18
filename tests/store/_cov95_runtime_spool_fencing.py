from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Literal

from gpu_fault.store import InMemoryStore, PostgresStore, SqliteStore
from gpu_fault.store.shared.telemetry_models import SpooledTelemetry
from tests.store._cov95_runtime_queue import sample

SpoolStore = InMemoryStore | SqliteStore | PostgresStore
Action = Literal["complete", "retry", "drop", "abandon"]
NOW = datetime(2026, 9, 12, tzinfo=timezone.utc)
LEASE = timedelta(seconds=30)
ACTIONS: tuple[Action, ...] = ("complete", "retry", "drop", "abandon")
Transition = Literal["complete", "retry", "drop", "abandon", "coalesce"]
TRANSITIONS: tuple[Transition, ...] = (*ACTIONS, "coalesce")


def seed(store: SpoolStore, identity: str = "busy-target", value: int = 1) -> None:
    request = sample(identity, value=value)
    admitted = store.try_spool_telemetry_requests(
        [request], max_depth=100, max_cluster_depth=100, now=NOW
    )
    assert admitted[0][0] is not None, "the isolated telemetry sample must be admitted"


def claim(store: SpoolStore, owner: str, at: datetime) -> SpooledTelemetry:
    rows = store.claim_telemetry_spool(owner, now=at, lease_duration=LEASE, limit=1)
    assert len(rows) == 1, "the isolated owner must acquire exactly one spool row"
    return rows[0]


def mutate(
    store: SpoolStore, items: list[SpooledTelemetry], action: Action, at: datetime
) -> tuple[int, int]:
    if action == "complete":
        return store.complete_telemetry_spool(items), 0
    if action == "abandon":
        return store.abandon_telemetry_spool_claims(items, now=at), 0
    return store.release_telemetry_spool(
        items,
        now=at,
        backoff=timedelta(seconds=300),
        max_attempts=1 if action == "drop" else 100,
    )


def assert_old_claim_cannot_mutate_live_replacement(
    store: SpoolStore, action: Action, replacement_owner: str
) -> None:
    seed(store)
    old = claim(store, "owner-a", NOW)
    assert (
        store.claim_telemetry_spool(
            replacement_owner,
            now=NOW + timedelta(seconds=1),
            lease_duration=LEASE,
            limit=1,
        )
        == []
    ), "an unexpired busy claim must not be reclaimed"
    at = NOW + LEASE
    replacement = claim(store, replacement_owner, at)
    assert replacement.spool_key == old.spool_key, "takeover must retain the lane"
    assert replacement.request_id == old.request_id, "takeover must retain the work"
    assert replacement.payload == old.payload, "no new payload may manufacture a fence"
    assert replacement.revision == old.revision, "claiming must not revise the payload"
    assert replacement.lease_token and replacement.lease_token != old.lease_token, (
        "every takeover needs a new claim fence independent of owner and revision"
    )
    assert replacement.attempts == old.attempts + 1, "reclaim must charge one attempt"
    assert store.telemetry_spool_stats(now=at)["leased"] == 1, (
        "the replacement must still own a live row before late completion"
    )

    assert mutate(store, [old], action, at) == (0, 0), (
        f"stale {action} must not delete, reschedule, or debit a live replacement"
    )

    after = store.telemetry_spool_stats(now=at)
    assert (after["depth"], after["leased"]) == (1, 1), (
        "the replacement's durable row and lease must survive the stale operation"
    )
    assert (
        store.claim_telemetry_spool(
            "third-owner", now=at, lease_duration=LEASE, limit=1
        )
        == []
    ), "stale release must not expose the replacement's row to another claimant"
    assert store.complete_telemetry_spool([replacement]) == 1, (
        "the current owner must still be able to complete the original durable work"
    )
    assert store.complete_telemetry_spool([old, replacement]) == 0, (
        "both completions must be idempotent after the current owner drains the row"
    )


def assert_reused_lane_does_not_reuse_claim(
    store: SpoolStore, action: Action, transition: Transition
) -> None:
    seed(store)
    old = claim(store, "same-owner", NOW)
    if transition == "coalesce":
        seed(store, "newest-target", value=2)
    elif transition == "retry":
        assert store.release_telemetry_spool([old, old], now=NOW) == (1, 0), (
            "duplicated retry callbacks must release one claim exactly once"
        )
    else:
        expected = (0, 1) if transition == "drop" else (1, 0)
        assert mutate(store, [old, old], transition, NOW) == expected, (
            "duplicate callbacks must not delete twice or refund two attempts"
        )
        if transition != "abandon":
            seed(store)
    assert mutate(store, [old], action, NOW) == (0, 0), (
        "a resolved claim cannot mutate a pending or recreated lane"
    )
    current = claim(store, "same-owner", NOW)
    assert current.lease_token and current.lease_token != old.lease_token, (
        "owner, timestamp, revision and attempt reuse must not recreate a claim"
    )
    if transition == "coalesce":
        assert current.revision == old.revision + 1, "only new payloads revise a lane"
        assert current.payload["value"] == 2, "coalescing must keep the newest payload"
    else:
        assert current.revision == old.revision, (
            "claim lifecycle is not payload revision"
        )
    expected_attempts = 2 if transition in {"retry", "coalesce"} else 1
    assert current.attempts == expected_attempts, (
        "claim fencing must not change attempt charging or abandonment refunds"
    )
    assert mutate(store, [old], action, NOW) == (0, 0), (
        "an old callback must remain fenced after immediate same-owner reclaim"
    )
    assert store.complete_telemetry_spool([current, current]) == 1, (
        "the replacement must complete exactly once, including duplicate batch items"
    )


def assert_invalid_fence_cannot_mutate(
    store: SpoolStore, action: Action, invalid: str
) -> None:
    seed(store)
    current = claim(store, "owner", NOW)
    tokens = {"missing": None, "empty": "", "foreign": "not-the-claim"}
    forged = (
        replace(current, revision=current.revision + 1)
        if invalid == "revision"
        else replace(current, lease_token=tokens[invalid])
    )
    assert mutate(store, [forged], action, NOW) == (0, 0), (
        "missing, invalid, or mismatched claim and payload fences must fail closed"
    )
    assert store.complete_telemetry_spool([current]) == 1, (
        "rejecting an invalid callback must not damage the valid claim"
    )


def assert_latest_wins_keeps_retry_budget(store: SpoolStore) -> None:
    seed(store)
    previous: SpooledTelemetry | None = None
    for attempt in range(1, 6):
        current = claim(store, "same-owner", NOW)
        assert current.attempts == attempt, "coalescing must not replenish retry budget"
        assert current.revision == attempt - 1, (
            "payload revisions must count coalesces, not claim attempts"
        )
        assert current.lease_token and current.lease_token not in repr(current), (
            "claim fences must not leak through the telemetry model representation"
        )
        if previous is not None:
            assert store.complete_telemetry_spool([previous]) == 0, (
                "the superseded payload must not complete the current sample"
            )
        if attempt < 5:
            seed(store, f"sample-{attempt}", value=attempt + 1)
        previous = current
    assert store.release_telemetry_spool([current], now=NOW) == (0, 1), (
        "the lane must retain its original five-failure drop budget"
    )
    assert store.telemetry_spool_stats(now=NOW)["depth"] == 0, (
        "exhausted telemetry must be removed without carrying claim state forward"
    )
