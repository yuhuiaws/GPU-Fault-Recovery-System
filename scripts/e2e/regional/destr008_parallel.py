"""Bounded concurrency for independent reads and probes on the DESTR-008 arming path.

Every control-plane kubectl round trip costs ~1.2 s from the acceptance host and
the watchdog arming makes dozens of them, so a sequential arming (~6 min live,
attempt 14) ate the independent cancellation window. Reads that do not depend
on one another run together; nothing is skipped, and the first failure still
fails closed. Test harnesses pin ``workers = 1`` so their fakes stay
deterministic.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

T = TypeVar("T")

workers = 8


def gather(calls: Sequence[Callable[[], T]]) -> list[T]:
    """Run ``calls`` together and return their results in call order.

    The first call that raised (in call order) re-raises after every call has
    finished, so a failure never leaves a probe running unobserved.
    """

    if workers <= 1 or len(calls) <= 1:
        return [call() for call in calls]
    with ThreadPoolExecutor(max_workers=min(workers, len(calls))) as pool:
        futures = [pool.submit(call) for call in calls]
        return [future.result() for future in futures]
