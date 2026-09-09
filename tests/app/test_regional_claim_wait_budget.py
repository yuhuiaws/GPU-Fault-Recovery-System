"""The claim long-poll fits inside the request budget it shares with the final claim.

Live 2026-09-09: executors asked for a 20 s wait, the request budget was 15 s,
and every idle claim ended 503 "store I/O capacity exceeded" -- both replicas
failed readiness and the paused release could not resume.
"""

from __future__ import annotations

from gpu_fault.app.routes import regional
from gpu_fault.async_store import RequestDeadlineExceeded, StoreIoCapacityExceeded


def test_the_wait_is_cut_to_what_the_budget_leaves_after_the_reserve() -> None:
    now = 1000.0
    assert (
        regional.bounded_claim_wait_seconds(20.0, deadline=now + 15.0, now=now) == 12.0
    )
    assert (
        regional.bounded_claim_wait_seconds(5.0, deadline=now + 15.0, now=now) == 5.0
    ), "a wait the budget allows is not shortened"
    assert (
        regional.bounded_claim_wait_seconds(20.0, deadline=now + 2.0, now=now) == 0.0
    ), "no room for a wait means a plain claim"
    assert regional.bounded_claim_wait_seconds(20.0, deadline=None, now=now) == 20.0, (
        "without a request deadline the server cap alone bounds the wait"
    )
    assert regional.bounded_claim_wait_seconds(0.0, deadline=now + 15.0, now=now) == 0.0


def test_a_deadline_rejection_is_named_as_such() -> None:
    deadline = regional.store_io_rejection(RequestDeadlineExceeded("late"))
    assert deadline.status_code == 503, deadline
    assert deadline.detail == "request deadline exceeded", deadline.detail
    capacity = regional.store_io_rejection(StoreIoCapacityExceeded("full"))
    assert capacity.detail == "store I/O capacity exceeded", capacity.detail
    assert capacity.headers == {"Retry-After": "2"}, capacity.headers
