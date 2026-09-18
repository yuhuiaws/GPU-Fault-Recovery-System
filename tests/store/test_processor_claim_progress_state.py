"""Pure coverage of disposable claim hints; no PostgreSQL connection."""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

import pytest

from gpu_fault.channel_registry import NVIDIA_KERNEL_PATH
from gpu_fault.processor.models import ProcessorRequest
from gpu_fault.store import PostgresStore

AT = datetime(2026, 9, 14, tzinfo=timezone.utc)
REQUEST_LEASE = timedelta(seconds=120)


class Cursor:
    def __init__(self, database):
        self.database = database
        self.rows = []

    def execute(self, query, parameters):
        self.database.parameters.append(dict(parameters))
        self.rows = self.database.results.popleft()
        callback, self.database.callback = self.database.callback, None
        if callback is not None:
            callback()

    def fetchall(self):
        return self.rows


class Database:
    def __init__(self):
        self.results = deque()
        self.parameters = []
        self.callback = None
        self.fail_commit = False

    @contextmanager
    def cursor(self):
        yield Cursor(self)

    @contextmanager
    def transaction(self):
        yield
        if self.fail_commit:
            self.fail_commit = False
            raise RuntimeError("modeled commit failure")


class HintStore(PostgresStore):
    def __init__(self, database):
        self._db = database
        self._models = {"processor_request": ProcessorRequest}
        self.claim_window_multiplier = 8
        self.processor_queue_state_mode = "dedicated"


def metadata(key="boundary", *, eligible=0, payload=None):
    return [(payload, 10, AT, key, eligible)]


def query_parameters(store, **filters):
    _, parameters = store.claim_active_processor_query(
        "owner", now=AT, lease_duration=REQUEST_LEASE, limit=1, **filters
    )
    return parameters


def claim(store, **filters):
    return store.claim_active_processor_requests(
        "owner", now=AT, lease_duration=REQUEST_LEASE, limit=1, **filters
    )


@pytest.mark.parametrize("rows", [[], [(None, None, None, None, 0)]])
def test_empty_query_metadata_never_leaks_into_the_claim_api(rows):
    database = Database()
    database.results.append(rows)
    store = HintStore(database)

    assert claim(store) == [], "empty metadata is not a processor request"
    assert query_parameters(store)["after_request_id"] is None, (
        "an exhausted window must not invent a continuation"
    )


def test_progress_is_committed_only_after_the_database_transaction():
    database = Database()
    database.results.append(metadata())
    database.fail_commit = True
    store = HintStore(database)

    with pytest.raises(RuntimeError, match="modeled commit failure"):
        claim(store)

    assert query_parameters(store)["after_request_id"] is None, (
        "a rolled-back claim must not advance its scan hint"
    )


def test_a_concurrent_newer_hint_is_not_overwritten_by_an_older_result():
    database = Database()
    database.results.extend([metadata("older"), metadata("newer")])
    store = HintStore(database)
    database.callback = lambda: claim(store)

    assert claim(store) == [], "both modeled windows contain only held work"
    assert query_parameters(store)["after_request_id"] == "newer", (
        "a stale query result must not replace another completed scan hint"
    )


def test_exhaustion_clears_only_its_own_filter_hint():
    database = Database()
    store = HintStore(database)
    first = {"include_paths": {"/v1/first"}}
    second = {"exclude_paths": {"/v1/first"}}
    database.results.extend(
        [metadata("first"), metadata("second"), [(None, None, None, None, 0)]]
    )
    claim(store, **first)
    claim(store, **second)
    claim(store, **first)

    assert query_parameters(store, **first)["after_request_id"] is None, (
        "exhaustion must wrap this stream to its head"
    )
    assert query_parameters(store, **second)["after_request_id"] == "second", (
        "include and exclude streams must not share a continuation"
    )


def test_a_full_raw_claim_returns_to_the_head_without_a_seek():
    database = Database()
    store = HintStore(database)
    item = ProcessorRequest(request_id="ready", method="POST", path=NVIDIA_KERNEL_PATH)
    database.results.extend(
        [metadata(), metadata("following", eligible=1, payload=item.model_dump())]
    )
    claim(store)

    rows = claim(store)

    assert [row.request_id for row in rows] == ["ready"], (
        "the API must return only decoded processor requests"
    )
    assert query_parameters(store)["after_request_id"] is None, (
        "healthy raw-window throughput should not pay for a continuing seek"
    )


def test_aging_only_completions_do_not_hide_a_full_blocked_raw_prefix():
    database = Database()
    store = HintStore(database)
    routine = ProcessorRequest(
        request_id="routine", method="POST", path="/v1/training-progress"
    )
    database.results.append(metadata(payload=routine.model_dump(), eligible=0))

    rows = claim(store)

    assert [row.request_id for row in rows] == ["routine"], (
        "the aging window may still supply a claim"
    )
    assert query_parameters(store)["after_request_id"] == "boundary", (
        "aging completions must not repeatedly hide the held raw prefix"
    )


def test_filter_hint_memory_is_bounded_and_new_store_instances_start_fresh():
    database = Database()
    store = HintStore(database)
    for index in range(66):
        database.results.append(metadata(f"boundary-{index}"))
        claim(store, include_paths={f"/v1/filter-{index}"})

    assert (
        query_parameters(store, include_paths={"/v1/filter-0"})["after_request_id"]
        is None
    ), "the oldest unused hint must be evicted"
    assert (
        query_parameters(store, include_paths={"/v1/filter-2"})["after_request_id"]
        == "boundary-2"
    ), "the bounded cache should retain recent hints"
    assert (
        query_parameters(HintStore(Database()), include_paths={"/v1/filter-2"})[
            "after_request_id"
        ]
        is None
    ), "scan hints must not be shared across Store instances"
