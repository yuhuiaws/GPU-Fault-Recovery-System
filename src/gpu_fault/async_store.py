from __future__ import annotations

import asyncio
import contextvars
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Callable, TypeVar

from gpu_fault.store.shared.errors import (
    is_retryable_store_unavailable,
)

T = TypeVar("T")


# Monotonic instant by which the request being served must be answered,
# set once per request at the edge. Every queue a request passes through
# used to carry its own independent timeout - ingress backpressure 30s,
# store I/O admission 30s, pool checkout 10s - so a request could occupy
# capacity for over a minute while the client that sent it gave up after
# fifteen seconds. Clamping each wait to what is left of the budget is
# what turns those stacked timeouts into one end-to-end deadline.
REQUEST_DEADLINE: contextvars.ContextVar[float | None] = contextvars.ContextVar(
    "gpu_fault_request_deadline", default=None
)


def remaining_budget(default: float) -> float:
    """Seconds left for the current request, or ``default`` if unbounded."""
    deadline = REQUEST_DEADLINE.get()
    if deadline is None:
        return default
    return min(default, deadline - time.monotonic())


class StoreIoCapacityExceeded(RuntimeError):
    pass


class RequestDeadlineExceeded(StoreIoCapacityExceeded):
    """The caller's own deadline passed before store I/O admission (F-E1).

    A subclass so every existing ``except StoreIoCapacityExceeded`` keeps
    working while the HTTP layer can tell "you ran out of time" from "the
    store is out of capacity" -- they call for different client behaviour.
    """


class AsyncStoreExecutor:
    """Runs synchronous store transactions outside the API event loop."""

    def __init__(
        self,
        *,
        workers: int,
        max_in_flight: int,
        admission_timeout_seconds: float,
        thread_name_prefix: str = "gpu-fault-store-io",
    ) -> None:
        if workers < 1 or max_in_flight < workers:
            raise ValueError("store I/O capacity must cover at least every worker")
        if admission_timeout_seconds <= 0:
            raise ValueError("store I/O admission timeout must be positive")
        self.workers = workers
        self.max_in_flight = max_in_flight
        self.admission_timeout_seconds = admission_timeout_seconds
        self._executor = ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix=thread_name_prefix,
        )
        self._slots = asyncio.Semaphore(max_in_flight)
        self.in_flight = 0
        self.rejected_total = 0
        self.rejected_by_reason: dict[str, int] = {
            "deadline": 0,
            "capacity": 0,
            "writer_unavailable": 0,
        }
        self.admission_wait_count = 0
        self.admission_wait_sum_seconds = 0.0
        self.admission_wait_max_seconds = 0.0

    async def run(self, function: Callable[..., T], /, *args, **kwargs) -> T:
        wait_started = time.monotonic()
        budget = remaining_budget(self.admission_timeout_seconds)
        if budget <= 0:
            # The caller's deadline already passed, so the work would be
            # thrown away on return. Reject without taking a slot.
            self.rejected_total += 1
            self.rejected_by_reason["deadline"] += 1
            raise RequestDeadlineExceeded(
                "request deadline exceeded before store I/O admission"
            )
        try:
            await asyncio.wait_for(
                self._slots.acquire(),
                timeout=budget,
            )
        except TimeoutError as exc:
            self.rejected_total += 1
            self.rejected_by_reason["capacity"] += 1
            raise StoreIoCapacityExceeded(
                "store I/O executor capacity exceeded"
            ) from exc
        waited = time.monotonic() - wait_started
        self.admission_wait_count += 1
        self.admission_wait_sum_seconds += waited
        self.admission_wait_max_seconds = max(self.admission_wait_max_seconds, waited)
        try:
            self.in_flight += 1
            loop = asyncio.get_running_loop()
            context = contextvars.copy_context()
            future = loop.run_in_executor(
                self._executor,
                partial(
                    context.run,
                    function,
                    *args,
                    **kwargs,
                ),
            )
            release_in_finally = True
            try:
                return await asyncio.shield(future)
            except asyncio.CancelledError:
                # A running database transaction cannot be cancelled by
                # Future.cancel(). Keep its admission slot until the
                # thread actually returns, but do not keep waking the
                # cancelled request task every millisecond.
                release_in_finally = False
                future.add_done_callback(lambda _future: self._release_slot())
                raise
            except Exception as exc:
                if not is_retryable_store_unavailable(exc):
                    raise
                self.rejected_total += 1
                self.rejected_by_reason["writer_unavailable"] += 1
                raise StoreIoCapacityExceeded(
                    "PostgreSQL writer is temporarily unavailable"
                ) from exc
        finally:
            if "release_in_finally" not in locals() or release_in_finally:
                self._release_slot()

    def _release_slot(self) -> None:
        self.in_flight -= 1
        self._slots.release()

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
