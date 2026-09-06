"""The observation group commit must flush everything that queues behind it.

FINAL-建议汇总 F-D6 (P1-34A, P1-18C, P2-13D, P2-47K). The first completion to
find the queue empty becomes leader, waits 10 ms, cuts ``[:64]`` and flushes it.
Anything past the 64th entry has no leader: later entrants see a non-empty
queue and wait as followers, so the remainder times out after 30 s - and,
because a timed-out entry is never removed, every later completion in the
process is a follower behind it. One burst over 64 wedges the process's
observation completions for good.

The logic is pure Python inside ``PostgresProcessorCompletionMixin``; these
tests drive it with a stand-in host and no database. The fix belongs to
``store/postgres/processor_completion.py`` (drain until empty, promote a
waiter when no leader is active), so both tests are strict-xfail until then.
"""

from __future__ import annotations

from threading import Condition, Event, Thread

from gpu_fault.store.postgres.processor_completion import (
    PostgresProcessorCompletionMixin,
    _CompletionEntry,
)

OBSERVATION_PATH = "/v1/workload-observations"


class _Host(PostgresProcessorCompletionMixin):
    """Just the state the group commit reads; nothing touches ``_db``."""

    def __init__(self) -> None:
        self._processor_completion_condition = Condition()
        self._processor_completion_queue = []
        self.flushed_batches: list[int] = []

    def _complete_active_processor_requests_batch(self, batch) -> dict[str, object]:
        self.flushed_batches.append(len(batch))
        return {item["args"][0]: {"request_id": item["args"][0]} for item in batch}


def _entry(request_id: str) -> _CompletionEntry:
    return {
        "args": (request_id, "pod-a:1", 1, "token"),
        "kwargs": {
            "response_status": 200,
            "response_content_type": "application/json",
            "response_body_base64": "e30=",
        },
        "event": Event(),
        "result": None,
        "error": None,
    }


def _complete(host: _Host, request_id: str):
    return host.complete_active_processor_request(
        request_id,
        "pod-a:1",
        1,
        "token",
        response_status=200,
        response_content_type="application/json",
        response_body_base64="e30=",
        path=OBSERVATION_PATH,
    )


def test_group_commit_flushes_more_than_sixty_four_pending() -> None:
    host = _Host()
    stragglers = [_entry(f"req-{index:03d}") for index in range(65)]
    with host._processor_completion_condition:
        host._processor_completion_queue.extend(stragglers)

    result: dict[str, object] = {}
    errors: list[BaseException] = []

    def newcomer() -> None:
        try:
            result["value"] = _complete(host, "req-new")
        except BaseException as exc:  # noqa: BLE001 - the timeout is the finding
            errors.append(exc)

    thread = Thread(target=newcomer)
    thread.start()
    thread.join(timeout=5)

    assert not thread.is_alive(), "the newcomer is still waiting for a leader"
    assert errors == []
    assert result["value"] == {"request_id": "req-new"}
    assert all(entry["event"].is_set() for entry in stragglers), (
        "stragglers past the 64th were never flushed"
    )
    assert host._processor_completion_queue == []
    assert sum(host.flushed_batches) == 66
    assert max(host.flushed_batches) <= 64


def test_group_commit_leader_keeps_draining_arrivals() -> None:
    host = _Host()
    late = [_entry(f"late-{index}") for index in range(3)]
    pending = list(late)
    original = host._complete_active_processor_requests_batch

    def flush_with_arrivals(batch):
        # Rows that queue while the leader is inside its flush.
        if pending:
            with host._processor_completion_condition:
                host._processor_completion_queue.extend(pending)
            pending.clear()
        return original(batch)

    host._complete_active_processor_requests_batch = flush_with_arrivals  # type: ignore[method-assign]

    assert _complete(host, "req-leader") == {"request_id": "req-leader"}

    assert all(entry["event"].is_set() for entry in late), (
        "rows that arrived during the leader's flush were left leaderless"
    )
    assert host._processor_completion_queue == []
    assert host.flushed_batches == [1, 3]
