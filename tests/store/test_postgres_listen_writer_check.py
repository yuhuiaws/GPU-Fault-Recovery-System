"""The LISTEN connection re-checks that it is still on the writer (G-11).

Control-plane review 2026-09-08. After an Aurora failover the pooled
connections are probed by ``WriterCheck`` on checkout, but the processor's
LISTEN connection -- opened outside the pool -- stayed on the demoted
instance, which does not forward NOTIFY. It reported connected, received
nothing, and every fault-path wakeup fell back to the 5 s poll. The loop now
asks ``pg_is_in_recovery()`` after each idle ``notifies`` timeout, at most once
per ``writer_check_seconds``, and treats a reader like a dropped connection:
close, report disconnected, reconnect.
"""

from __future__ import annotations

import time
from threading import Event, Thread

from gpu_fault.store.postgres.processor_admin import PostgresProcessorAdminMixin


class _Result:
    def __init__(self, row):
        self._row = row

    def fetchone(self):
        return self._row


class DemotedListenConnection:
    """A LISTEN connection whose server answers the writer probe as a reader."""

    def __init__(self, *, in_recovery: bool) -> None:
        self.in_recovery = in_recovery
        self.statements: list[str] = []
        self.closed = False
        self.autocommit = True

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, sql: str, params=None) -> _Result:
        flat = " ".join(sql.split())
        self.statements.append(flat)
        if "pg_is_in_recovery()" in flat:
            return _Result((self.in_recovery, "on" if self.in_recovery else "off"))
        return _Result((True,))

    def notifies(self, *, timeout: float, stop_after: int):
        time.sleep(timeout)
        return iter(())

    def close(self) -> None:
        self.closed = True


def _listen(connections: list[DemotedListenConnection], monkeypatch, **kwargs):
    import psycopg

    handed_out: list[DemotedListenConnection] = []

    def connect(*_args, **_kwargs):
        connection = connections[min(len(handed_out), len(connections) - 1)]
        handed_out.append(connection)
        return connection

    monkeypatch.setattr(psycopg, "connect", connect)
    store = PostgresProcessorAdminMixin()
    store.url = "postgresql://fake"
    stop = Event()
    states: list[tuple[bool, int | None]] = []
    thread = Thread(
        target=store.listen_processor_queue_notifications,
        args=(
            stop,
            "pod-a:1",
            1,
            lambda _payload: None,
            lambda e, s: states.append((e, s)),
        ),
        kwargs={"timeout_seconds": 0.01, **kwargs},
    )
    thread.start()
    return stop, thread, states, handed_out


def test_a_listener_on_a_demoted_writer_reconnects(monkeypatch) -> None:
    demoted = DemotedListenConnection(in_recovery=True)
    healthy = DemotedListenConnection(in_recovery=False)
    stop, thread, states, handed_out = _listen(
        [demoted, healthy], monkeypatch, writer_check_seconds=0.05
    )
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and len(handed_out) < 2:
            time.sleep(0.01)
    finally:
        stop.set()
        thread.join(timeout=5)

    assert demoted.closed is True, "the demoted connection was kept"
    assert any("pg_is_in_recovery()" in sql for sql in demoted.statements)
    assert (False, None) in states, states
    assert len(handed_out) >= 2, "the loop did not reconnect after the reader probe"


def test_the_writer_probe_is_throttled_to_the_configured_interval(monkeypatch) -> None:
    healthy = DemotedListenConnection(in_recovery=False)
    stop, thread, states, _handed_out = _listen(
        [healthy], monkeypatch, writer_check_seconds=10.0
    )
    try:
        time.sleep(0.3)
    finally:
        stop.set()
        thread.join(timeout=5)

    probes = [sql for sql in healthy.statements if "pg_is_in_recovery()" in sql]
    # Dozens of idle cycles at 10 ms, one probe interval of 10 s: at most the
    # first probe fires, never one per cycle.
    assert len(probes) <= 1, probes
    assert (False, None) not in states[:-1], states


def test_a_probe_that_raises_is_treated_like_a_dropped_connection(monkeypatch) -> None:
    class BrokenProbe(DemotedListenConnection):
        def execute(self, sql: str, params=None) -> _Result:
            if "pg_is_in_recovery()" in sql:
                raise RuntimeError("server closed the connection unexpectedly")
            return super().execute(sql, params)

    broken = BrokenProbe(in_recovery=False)
    healthy = DemotedListenConnection(in_recovery=False)
    stop, thread, states, handed_out = _listen(
        [broken, healthy], monkeypatch, writer_check_seconds=0.05
    )
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and len(handed_out) < 2:
            time.sleep(0.01)
    finally:
        stop.set()
        thread.join(timeout=5)

    assert len(handed_out) >= 2
    assert (False, None) in states
