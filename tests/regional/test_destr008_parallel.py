"""Independent arming reads run together, in order, and still fail closed."""

from __future__ import annotations

import threading
import time

import pytest

from scripts.e2e.regional import destr008_parallel as parallel


def test_results_keep_call_order_while_calls_overlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(parallel, "workers", 4)
    started = threading.Barrier(3, timeout=5)

    def call(index: int) -> int:
        started.wait()  # every call is in flight before any returns
        time.sleep(0.05 * (3 - index))
        return index

    assert parallel.gather([lambda i=i: call(i) for i in range(3)]) == [0, 1, 2], (
        "results must follow call order, not completion order"
    )


def test_the_first_failure_in_call_order_propagates_after_every_call_ran(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(parallel, "workers", 4)
    ran: list[int] = []

    def call(index: int) -> int:
        ran.append(index)
        if index == 1:
            raise ValueError(f"probe {index}")
        return index

    with pytest.raises(ValueError, match="probe 1"):
        parallel.gather([lambda i=i: call(i) for i in range(3)])
    assert sorted(ran) == [0, 1, 2], (
        "a failure must not leave sibling probes unobserved"
    )


def test_a_single_worker_runs_calls_sequentially(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(parallel, "workers", 1)
    ran: list[int] = []

    def call(index: int) -> int:
        ran.append(index)
        if index == 0:
            raise RuntimeError("first")
        return index

    with pytest.raises(RuntimeError, match="first"):
        parallel.gather([lambda i=i: call(i) for i in range(3)])
    assert ran == [0], "sequential mode stops at the first failure"
