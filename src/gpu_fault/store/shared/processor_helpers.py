from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from typing import Any


def pending_fault_scope_keys(items: Iterable[Any]) -> set[str]:
    return {
        scope_key
        for item in items
        if item.status.value == "PENDING" and item.is_correlated_fault()
        for scope_key in item.correlation_scope_keys
    }


def incomplete_observation_scope_keys(items: Iterable[Any], now: datetime) -> set[str]:
    """Scope keys of observation rows that still block correlated faults.

    A LEASED observation only counts while its lease is live: a crashed
    owner's row used to shadow every fault on its scope until somebody
    happened to reclaim it (F-D3).
    """

    return {
        scope_key
        for item in items
        if item.path == "/v1/workload-observations"
        and (
            item.status.value == "PENDING"
            or (
                item.status.value == "LEASED"
                and item.lease_expires_at is not None
                and item.lease_expires_at > now
            )
        )
        for scope_key in item.correlation_scope_keys
    }


def fault_rows_blocked_by_observation(items: Iterable[Any], now: datetime) -> int:
    """How many claimable fault rows an incomplete observation holds back."""

    blocking = incomplete_observation_scope_keys(items, now)
    if not blocking:
        return 0
    return sum(
        1
        for item in items
        if item.status.value == "PENDING"
        and item.is_correlated_fault()
        and item.available_for_claim(now)
        and item.waits_for_observation(blocking)
    )


class PartialEnqueueError(RuntimeError):
    """A multi-scope admission batch committed some scope groups, then failed.

    ``try_enqueue_processor_requests_batch`` admits one cluster scope per
    transaction, so a batch that mixes clusters can be half-committed when a
    later group raises. The plain exception hid which rows had landed (F-D9);
    ``committed`` lists the request ids that are already in the queue so the
    caller can answer those waiters instead of retrying every row.
    """

    def __init__(self, *, committed: list[str], cause: BaseException) -> None:
        super().__init__(
            f"processor admission batch committed {len(committed)} request(s) "
            f"before failing: {type(cause).__name__}: {cause}"
        )
        self.committed = committed
        self.cause = cause
