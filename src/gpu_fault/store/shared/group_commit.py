"""Leader-elected group commit that drains until the queue is empty (F-D6).

The first caller to find no active leader becomes one, waits a few
milliseconds for company, then flushes the queue in batches until it is
empty. Everyone else waits for their entry's event; a waiter that finds no
active leader and a non-empty queue promotes itself, so a leader that
raised or exited while others were still queuing does not leave them
leaderless. A waiter that times out removes its own entry so one stall
cannot wedge every later caller behind it.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from threading import Condition
from typing import Any

Entry = dict[str, Any]


def submit_group_commit(
    entry: Entry,
    *,
    condition: Condition,
    queue: list[Entry],
    host: object,
    active_attr: str,
    flush: Callable[[list[Entry]], None],
    batch_size: int = 64,
    first_wait: float = 0.01,
    timeout: float = 30.0,
) -> None:
    """Queue ``entry`` and return once ``flush`` has set its event.

    ``flush`` receives up to ``batch_size`` entries and must set each entry's
    ``event`` (after filling ``result`` / ``error``); it must not raise.
    """

    with condition:
        queue.append(entry)
        leader = not getattr(host, active_attr, False)
        if leader:
            setattr(host, active_attr, True)
        else:
            condition.notify()
    if leader:
        _drain(condition, queue, host, active_attr, flush, batch_size, first_wait)
    deadline = time.monotonic() + timeout
    while not entry["event"].is_set():
        with condition:
            if entry["event"].is_set():
                break
            promote = not getattr(host, active_attr, False) and bool(queue)
            if promote:
                setattr(host, active_attr, True)
        if promote:
            _drain(condition, queue, host, active_attr, flush, batch_size, 0.0)
            continue
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            with condition:
                for index, queued in enumerate(queue):
                    if queued is entry:
                        del queue[index]
                        break
            raise TimeoutError("group commit batch did not flush")
        entry["event"].wait(min(0.05, remaining))


def _drain(
    condition: Condition,
    queue: list[Entry],
    host: object,
    active_attr: str,
    flush: Callable[[list[Entry]], None],
    batch_size: int,
    first_wait: float,
) -> None:
    try:
        if first_wait > 0:
            with condition:
                condition.wait(first_wait)
        while True:
            with condition:
                batch = queue[:batch_size]
                del queue[: len(batch)]
            if not batch:
                return
            flush(batch)
    finally:
        with condition:
            setattr(host, active_attr, False)
            condition.notify_all()
