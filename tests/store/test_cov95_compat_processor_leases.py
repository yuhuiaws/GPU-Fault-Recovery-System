from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from gpu_fault.processor import ProcessorRequestStatus
from gpu_fault.store import NotFoundError
from gpu_fault.store.shared.errors import StaleFencingTokenError
from tests.store._cov95_compat_support import NOW, claim_request, request_model
from tests.store._cov95_compat_support import (
    compat_store_fixture as compat_store_fixture,
)

LEASE = timedelta(seconds=120)
RESPONSE = {
    "response_status": 202,
    "response_content_type": "application/json",
    "response_body_base64": "e30=",
}


def test_periodic_leases_keep_live_owners_and_fence_each_expired_generation(
    compat_store,
):
    store = compat_store
    first = store.acquire_periodic_task_lease(
        "local-task", "owner-a", now=NOW, lease_duration=LEASE
    )
    blocked = store.acquire_periodic_task_lease(
        "local-task", "owner-b", now=NOW + timedelta(seconds=1), lease_duration=LEASE
    )
    assert blocked == first
    renewed = store.acquire_periodic_task_lease(
        "local-task", "owner-a", now=NOW + timedelta(seconds=2), lease_duration=LEASE
    )
    assert renewed.epoch == first.epoch
    assert renewed.lease_expires_at == NOW + LEASE + timedelta(seconds=2)
    reclaimed = store.acquire_periodic_task_lease(
        "local-task", "owner-a", now=renewed.lease_expires_at, lease_duration=LEASE
    )
    assert reclaimed.epoch == first.epoch + 1
    successor = store.acquire_periodic_task_lease(
        "local-task", "owner-b", now=reclaimed.lease_expires_at, lease_duration=LEASE
    )
    assert successor.owner_id == "owner-b"
    assert successor.epoch == first.epoch + 2


def test_leadership_renewal_and_takeover_are_visible_through_public_reads(compat_store):
    store = compat_store
    assert store.get_processor_leadership() is None
    first = store.acquire_processor_leadership("owner-a", now=NOW, lease_duration=LEASE)
    assert store.get_processor_leadership() == first
    assert (
        store.acquire_processor_leadership(
            "owner-b", now=NOW + timedelta(seconds=1), lease_duration=LEASE
        )
        == first
    )
    renewed = store.acquire_processor_leadership(
        "owner-a", now=NOW + timedelta(seconds=1), lease_duration=LEASE
    )
    assert renewed.epoch == first.epoch
    next_owner = store.acquire_processor_leadership(
        "owner-b", now=renewed.lease_expires_at, lease_duration=LEASE
    )
    assert next_owner.epoch == first.epoch + 1
    assert store.get_processor_leadership().owner_id == "owner-b"


@pytest.mark.parametrize("mismatch", ["owner", "epoch", "token", "expired"])
def test_stale_active_lease_cannot_renew_or_complete(compat_store, mismatch):
    store = compat_store
    at = datetime.now(UTC) - (
        timedelta(minutes=5) if mismatch == "expired" else timedelta()
    )
    claimed = claim_request(store, "request", now=at)
    owner, epoch, token = claimed.lease_owner, claimed.leader_epoch, claimed.lease_token
    if mismatch == "owner":
        owner = "other"
    elif mismatch == "epoch":
        epoch += 1
    elif mismatch == "token":
        token = "other-fence"
    assert (
        store.validate_processor_lane(claimed.ordering_key(), owner, epoch, token)
        is False
    )
    assert (
        store.renew_active_processor_request(
            claimed.request_id, owner, epoch, token, lease_duration=LEASE
        )
        is False
    )
    with pytest.raises(StaleFencingTokenError):
        store.complete_active_processor_request(
            claimed.request_id, owner, epoch, token, **RESPONSE
        )
    retained = store.get_processor_request(claimed.request_id)
    assert retained.status is ProcessorRequestStatus.LEASED
    assert retained.lease_owner == claimed.lease_owner
    assert retained.leader_epoch == claimed.leader_epoch


def test_renewal_and_release_leave_a_completed_request_terminal(compat_store):
    store = compat_store
    claimed = claim_request(store, "request")
    key = claimed.ordering_key()
    owner, epoch, token = claimed.lease_owner, claimed.leader_epoch, claimed.lease_token
    assert store.validate_processor_lane(key, owner, epoch, token) is True
    before = datetime.now(UTC)
    assert (
        store.renew_active_processor_request(
            "request", owner, epoch, token, lease_duration=LEASE * 2
        )
        is True
    )
    after = datetime.now(UTC)
    renewed = store.get_processor_request("request")
    assert before + LEASE * 2 <= renewed.lease_expires_at <= after + LEASE * 2
    completed = store.complete_active_processor_request(
        "request", owner, epoch, token, **RESPONSE
    )
    assert completed.status is ProcessorRequestStatus.COMPLETED
    assert store.validate_processor_lane(key, owner, epoch, token) is False
    store.release_active_processor_request(
        "request", owner, epoch, token, retry_count=9
    )
    retained = store.get_processor_request("request")
    assert retained.status is ProcessorRequestStatus.COMPLETED
    assert retained.retry_count == 0
    assert retained.response_body_base64 == RESPONSE["response_body_base64"]


def test_legacy_release_does_not_reopen_a_completed_request(compat_store):
    store = compat_store
    now = datetime.now(UTC)
    store.enqueue_processor_request(
        request_model("request", created_at=now, updated_at=now)
    )
    leadership = store.acquire_processor_leadership(
        "owner", now=now, lease_duration=LEASE
    )
    (claimed,) = store.claim_processor_requests(
        "owner", leadership.epoch, now=now, lease_duration=LEASE, limit=1
    )
    store.complete_processor_request(
        "request", "owner", leadership.epoch, claimed.lease_token, **RESPONSE
    )
    store.release_processor_request(
        "request", "owner", leadership.epoch, claimed.lease_token, retry_count=4
    )
    assert (
        store.get_processor_request("request").status
        is ProcessorRequestStatus.COMPLETED
    )
    assert store.get_processor_request("request").retry_count == 0
    with pytest.raises(StaleFencingTokenError):
        store.complete_processor_request(
            "request", "owner", leadership.epoch, claimed.lease_token, **RESPONSE
        )


def test_release_books_retry_and_respects_not_before(compat_store):
    store = compat_store
    claimed = claim_request(store, "request")
    not_before = datetime.now(UTC) + timedelta(seconds=30)
    store.release_active_processor_request(
        "request",
        claimed.lease_owner,
        claimed.leader_epoch,
        claimed.lease_token,
        not_before=not_before,
        retry_count=3,
    )
    released = store.get_processor_request("request")
    assert released.status is ProcessorRequestStatus.PENDING
    assert released.not_before == not_before
    assert released.retry_count == 3
    assert released.lease_owner is None
    assert (
        store.claim_active_processor_requests(
            "next", now=not_before - timedelta(seconds=1), lease_duration=LEASE, limit=1
        )
        == []
    )
    (next_claim,) = store.claim_active_processor_requests(
        "next", now=not_before, lease_duration=LEASE, limit=1
    )
    assert next_claim.leader_epoch == claimed.leader_epoch + 1
    assert next_claim.retry_count == 3


def test_reclaimer_is_bounded_idempotent_and_releases_each_expired_request(
    compat_store,
):
    store = compat_store
    past = datetime.now(UTC) - timedelta(minutes=5)
    for request_id in ("a", "b"):
        claim_request(store, request_id, node_id=f"node-{request_id}", now=past)
    now = datetime.now(UTC)
    assert store.reclaim_expired_processor_leases(now=now, limit=1) == 1
    assert store.get_processor_request("a").status is ProcessorRequestStatus.PENDING
    assert store.get_processor_request("a").retry_count == 1
    assert store.get_processor_request("b").status is ProcessorRequestStatus.LEASED
    assert store.reclaim_expired_processor_leases(now=now, limit=1) == 1
    assert store.reclaim_expired_processor_leases(now=now, limit=1) == 0
    for request_id in ("a", "b"):
        current = store.get_processor_request(request_id)
        assert current.retry_count == 1
        assert current.lease_owner is None
        assert current.lease_token is None
        assert current.not_before is None


def test_lane_cleanup_resets_epoch_but_never_reuses_an_old_fence(compat_store):
    store = compat_store
    old = claim_request(store, "old")
    store.complete_active_processor_request(
        "old", old.lease_owner, old.leader_epoch, old.lease_token, **RESPONSE
    )
    assert store.cleanup_processor_lanes(older_than=datetime.now(UTC), limit=1) == 1
    assert store.cleanup_processor_lanes(older_than=datetime.now(UTC), limit=1) == 0
    current = claim_request(store, "new", owner_id="new-owner")
    assert current.leader_epoch == old.leader_epoch == 1
    different_fence = current.lease_token != old.lease_token
    assert different_fence, "retired lane epochs must not make old tokens valid again"
    with pytest.raises(StaleFencingTokenError):
        store.complete_active_processor_request(
            "old", old.lease_owner, old.leader_epoch, old.lease_token, **RESPONSE
        )
    assert (
        store.cleanup_processor_lanes(
            older_than=datetime.now(UTC) + timedelta(days=1), limit=10
        )
        == 0
    )
    assert store.get_processor_request("new").lease_owner == "new-owner"
    assert (
        store.cleanup_completed_processor_requests(
            older_than=datetime.now(UTC), limit=1
        )
        == 1
    )
    with pytest.raises(NotFoundError):
        store.get_processor_request("old")
    assert store.get_processor_request("new").status is ProcessorRequestStatus.LEASED


@pytest.mark.parametrize("mode", ["active", "legacy"])
@pytest.mark.parametrize("same_owner", [False, True], ids=["new-owner", "same-owner"])
def test_expired_claim_release_cannot_undo_a_completed_takeover(
    compat_store, mode, same_owner
):
    store = compat_store
    past = datetime.now(UTC) - timedelta(minutes=5)
    store.enqueue_processor_request(
        request_model("takeover", created_at=past, updated_at=past)
    )

    def claim(owner, at):
        if mode == "active":
            return store.claim_active_processor_requests(
                owner, now=at, lease_duration=LEASE, limit=1
            )[0]
        leadership = store.acquire_processor_leadership(
            owner, now=at, lease_duration=LEASE
        )
        return store.claim_processor_requests(
            owner, leadership.epoch, now=at, lease_duration=LEASE, limit=1
        )[0]

    first = claim("owner", past)
    replacement = claim("owner" if same_owner else "replacement", datetime.now(UTC))
    if mode == "legacy" and same_owner:
        assert replacement.leader_epoch == first.leader_epoch, (
            "legacy leadership preserves its epoch when the same owner reacquires it"
        )
    else:
        assert replacement.leader_epoch == first.leader_epoch + 1, (
            "a lane reclaim or a different legacy leader must advance its epoch"
        )
    changed_token = replacement.lease_token != first.lease_token
    assert changed_token, "every reclaimed request must receive a fresh lease token"
    release = (
        store.release_active_processor_request
        if mode == "active"
        else store.release_processor_request
    )
    release(
        first.request_id,
        first.lease_owner,
        first.leader_epoch,
        first.lease_token,
        retry_count=99,
        not_before=datetime.now(UTC) + timedelta(days=1),
    )

    current = store.get_processor_request(replacement.request_id)
    assert current.lease_owner == replacement.lease_owner, (
        "the old release must not clear the replacement's owner"
    )
    assert current.leader_epoch == replacement.leader_epoch, (
        "the old release must not roll back the replacement's epoch"
    )
    assert (
        current.status is ProcessorRequestStatus.LEASED and current.retry_count == 0
    ), "an obsolete release cannot turn leased work back into a delayed retry"
    assert current.not_before is None, (
        "an obsolete release must not install a backoff on the replacement's request"
    )
    complete = (
        store.complete_active_processor_request
        if mode == "active"
        else store.complete_processor_request
    )
    result = complete(
        replacement.request_id,
        replacement.lease_owner,
        replacement.leader_epoch,
        replacement.lease_token,
        **RESPONSE,
    )
    assert result.status is ProcessorRequestStatus.COMPLETED, (
        "the replacement must still be able to complete using its original lease"
    )


@pytest.mark.parametrize(
    "retry_count", [None, 4], ids=["preserve-retries", "set-retries"]
)
def test_legacy_release_keeps_retry_schedule_and_reclaims_only_when_due(
    compat_store, retry_count
):
    store = compat_store
    at = datetime.now(UTC)
    store.enqueue_processor_request(
        request_model("legacy-retry", created_at=at, updated_at=at, retry_count=2)
    )
    leadership = store.acquire_processor_leadership(
        "owner", now=at, lease_duration=LEASE
    )
    (claimed,) = store.claim_processor_requests(
        "owner", leadership.epoch, now=at, lease_duration=LEASE, limit=1
    )
    due = at + timedelta(seconds=30)

    store.release_processor_request(
        claimed.request_id,
        "owner",
        leadership.epoch,
        claimed.lease_token,
        not_before=due,
        retry_count=retry_count,
    )

    pending = store.get_processor_request(claimed.request_id)
    assert pending.status is ProcessorRequestStatus.PENDING, (
        "a valid legacy release must make the request retryable"
    )
    assert pending.retry_count == (2 if retry_count is None else retry_count), (
        "release must preserve the existing retry count unless explicitly replaced"
    )
    assert pending.not_before == due and pending.lease_owner is None, (
        "release must relinquish ownership while preserving the scheduled backoff"
    )
    assert (
        store.claim_processor_requests(
            "owner",
            leadership.epoch,
            now=due - timedelta(microseconds=1),
            lease_duration=LEASE,
            limit=1,
        )
        == []
    ), "the request must not be reclaimed before its retry deadline"
    (retried,) = store.claim_processor_requests(
        "owner", leadership.epoch, now=due, lease_duration=LEASE, limit=1
    )
    assert retried.request_id == claimed.request_id, (
        "the exact retry deadline must make the released request claimable"
    )
    assert retried.retry_count == pending.retry_count, (
        "claiming a scheduled retry must not silently change its retry budget"
    )
