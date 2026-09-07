"""Owning a notification shard is tied to consuming what it delivers.

FINAL-建议汇总 F-D11 (P2-75K, P1-75A, P1-14A). The shard advisory lock lives on
the LISTEN connection, so a process whose consumer loop is wedged but whose
listener thread is fine kept its shard and swallowed every wakeup for it;
the other processes filtered those payloads out. The listener now watches a
progress counter the consumer advances and gives the shard up when payloads
were forwarded but nothing moved. The factory can derive the shard count
from the consumer process count instead of defaulting to 8 against 24.
"""

from __future__ import annotations

import time
from threading import Event, Thread
from types import SimpleNamespace

import pytest

from gpu_fault.processor import ProcessorCoordinator, ProcessorLeaseSettings
from gpu_fault.store.postgres.processor_admin import PostgresProcessorAdminMixin
from tests._builders import build_store


class _Result:
    def __init__(self, row: tuple[object, ...]) -> None:
        self._row = row

    def fetchone(self) -> tuple[object, ...]:
        return self._row


class FakeListenConnection:
    """Enough of a psycopg connection for the listener loop."""

    def __init__(self, *, payloads_per_cycle: int) -> None:
        self.statements: list[tuple[str, object]] = []
        self.payloads_per_cycle = payloads_per_cycle
        self.cycles = 0

    def __enter__(self) -> "FakeListenConnection":
        return self

    def __exit__(self, *_args: object) -> bool:
        return False

    def execute(self, sql: str, params: object = None) -> _Result:
        self.statements.append((" ".join(sql.split()), params))
        return _Result((True,))

    def notifies(self, *, timeout: float, stop_after: int):
        self.cycles += 1
        for index in range(self.payloads_per_cycle):
            yield SimpleNamespace(payload=f'{{"request_id":"r-{self.cycles}-{index}"}}')
        time.sleep(timeout)

    def unlocks(self) -> list[tuple[str, object]]:
        return [entry for entry in self.statements if "pg_advisory_unlock" in entry[0]]


def _listen(connection: FakeListenConnection, monkeypatch, **kwargs):
    import psycopg

    monkeypatch.setattr(psycopg, "connect", lambda *_a, **_k: connection)
    store = PostgresProcessorAdminMixin()
    store.url = "postgresql://fake"
    stop = Event()
    states: list[tuple[bool, int | None]] = []
    payloads: list[str] = []
    thread = Thread(
        target=store.listen_processor_queue_notifications,
        args=(stop, "pod-a:1", 4, payloads.append, lambda e, s: states.append((e, s))),
        kwargs={"timeout_seconds": 0.01, **kwargs},
    )
    thread.start()
    return stop, thread, states, payloads


def test_shard_owner_without_progress_loses_the_shard(monkeypatch) -> None:
    connection = FakeListenConnection(payloads_per_cycle=1)
    stop, thread, states, payloads = _listen(
        connection, monkeypatch, on_progress=lambda: 0, stall_seconds=0.05
    )
    try:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and (True, None) not in states:
            time.sleep(0.01)
    finally:
        stop.set()
        thread.join(timeout=3)

    assert not thread.is_alive(), "the listener should stop when asked"
    assert states and states[0][0] is True and states[0][1] is not None, (
        "the first state report should be an owned shard"
    )
    assert (True, None) in states, "a stalled owner must report itself shardless"
    assert len(connection.unlocks()) >= 1
    unlock_sql, unlock_params = connection.unlocks()[0]
    assert unlock_params == (f"gpu_fault_processor_notify/{states[0][1]}",)
    assert payloads, "payloads were forwarded before the stall was judged"


def test_a_progressing_owner_keeps_its_shard(monkeypatch) -> None:
    connection = FakeListenConnection(payloads_per_cycle=1)
    counter = {"claims": 0}

    def progress() -> int:
        counter["claims"] += 1
        return counter["claims"]

    stop, thread, states, _payloads = _listen(
        connection, monkeypatch, on_progress=progress, stall_seconds=0.05
    )
    try:
        time.sleep(0.4)
    finally:
        stop.set()
        thread.join(timeout=3)

    assert connection.unlocks() == []
    assert (True, None) not in states, "a consuming owner never goes shardless"


def test_a_silent_shard_is_not_a_stalled_one(monkeypatch) -> None:
    # No payloads forwarded: there is nothing the consumer failed to act on.
    connection = FakeListenConnection(payloads_per_cycle=0)
    stop, thread, states, _payloads = _listen(
        connection, monkeypatch, on_progress=lambda: 0, stall_seconds=0.05
    )
    try:
        time.sleep(0.4)
    finally:
        stop.set()
        thread.join(timeout=3)

    assert connection.unlocks() == []
    assert (True, None) not in states, "silence is not a stall"


def test_the_coordinator_hands_its_claim_counter_to_the_listener(monkeypatch) -> None:
    store = build_store()
    captured: dict[str, object] = {}

    def listen(stop, owner_id, shard_count, notify, on_state, **kwargs):
        captured.update(kwargs)
        on_state(True, 0)
        stop.wait()

    monkeypatch.setattr(
        store, "listen_processor_queue_notifications", listen, raising=False
    )
    processor = ProcessorCoordinator(
        store,
        owner_id="pod-a:1",
        internal_token="token-" + "x" * 32,
        active_consumers=True,
    )
    thread = Thread(target=processor.run_queue_notifications)
    thread.start()
    try:
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and "on_progress" not in captured:
            time.sleep(0.01)
    finally:
        processor.stop()
        thread.join(timeout=2)

    on_progress = captured.get("on_progress")
    assert callable(on_progress), "the listener needs a progress probe"
    before = on_progress()
    processor.claim_rounds_total += 3
    assert on_progress() == before + 3


def test_factory_derives_shards_from_the_consumer_process_count(monkeypatch) -> None:
    monkeypatch.delenv("GPU_FAULT_PROCESSOR_NOTIFICATION_SHARDS", raising=False)
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_CONSUMER_PROCESSES", "24")

    settings = ProcessorLeaseSettings.from_environment()

    assert settings.processor_notification_shard_count == 24


def test_factory_refuses_fewer_shards_than_consumer_processes(monkeypatch) -> None:
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_NOTIFICATION_SHARDS", "8")
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_CONSUMER_PROCESSES", "24")

    with pytest.raises(RuntimeError, match="GPU_FAULT_PROCESSOR_NOTIFICATION_SHARDS"):
        ProcessorLeaseSettings.from_environment()


def test_factory_keeps_an_explicit_larger_shard_count(monkeypatch) -> None:
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_NOTIFICATION_SHARDS", "32")
    monkeypatch.setenv("GPU_FAULT_PROCESSOR_CONSUMER_PROCESSES", "24")

    settings = ProcessorLeaseSettings.from_environment()

    assert settings.processor_notification_shard_count == 32


def test_factory_default_is_unchanged_without_a_process_count(monkeypatch) -> None:
    monkeypatch.delenv("GPU_FAULT_PROCESSOR_NOTIFICATION_SHARDS", raising=False)
    monkeypatch.delenv("GPU_FAULT_PROCESSOR_CONSUMER_PROCESSES", raising=False)

    settings = ProcessorLeaseSettings.from_environment()

    assert settings.processor_notification_shard_count == 8
