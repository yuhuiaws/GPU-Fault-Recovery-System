from __future__ import annotations

from threading import Condition, Event, Thread
from types import SimpleNamespace

import pytest

from gpu_fault.store.shared.group_commit import submit_group_commit
from tests.store._cov95_compat_support import request_model


class WaitingEvent(Event):
    def __init__(self):
        super().__init__()
        self.waiting = Event()

    def wait(self, timeout=None):
        self.waiting.set()
        return super().wait(timeout)


class CommitGroup:
    def __init__(self, store):
        self.store = store
        self.condition = Condition()
        self.queue = []
        self.host = SimpleNamespace(active=False)
        self.entries = []
        self.threads = []
        self.gates = []
        self.errors = {}
        self.batches = []

    def entry(self, request_id, event=None):
        entry = {
            "request": request_model(request_id, node_id=request_id),
            "event": WaitingEvent() if event is None else event,
            "result": None,
            "error": None,
        }
        self.entries.append(entry)
        return entry

    def gate(self):
        event = Event()
        self.gates.append(event)
        return event

    def persist(self, batch):
        self.batches.append([item["request"].request_id for item in batch])
        for item in batch:
            item["result"] = self.store.enqueue_processor_request(item["request"])
            item["event"].set()

    def submit(self, entry, *, flush=None, timeout=10, first_wait=0.001, batch_size=1):
        submit_group_commit(
            entry,
            condition=self.condition,
            queue=self.queue,
            host=self.host,
            active_attr="active",
            flush=self.persist if flush is None else flush,
            timeout=timeout,
            first_wait=first_wait,
            batch_size=batch_size,
        )

    def start(self, entry, **options):
        def worker():
            try:
                self.submit(entry, **options)
            except BaseException as error:
                self.errors[entry["request"].request_id] = error

        thread = Thread(target=worker, name=f"compat-{entry['request'].request_id}")
        self.threads.append(thread)
        thread.start()
        return thread

    def join(self, thread):
        thread.join(timeout=12)
        assert not thread.is_alive(), (
            "group-commit participant did not stop within its test budget"
        )

    def close(self):
        for gate in self.gates:
            gate.set()
        for entry in self.entries:
            entry["event"].set()
        for thread in self.threads:
            self.join(thread)


@pytest.fixture(name="commit_group")
def commit_group_fixture(compat_store):
    group = CommitGroup(compat_store)
    try:
        yield group
    finally:
        group.close()
