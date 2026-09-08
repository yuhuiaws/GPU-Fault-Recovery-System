"""The kernel collector must not lose kmsg records to its own failures (ARCH-G4/G9).

Any exception used to reopen ``/dev/kmsg`` and seek to the end, so every record
written between the failure and the reopen was gone: a rejected event (a 4xx
nobody buffered), a kernel ring overflow (``EPIPE`` on read), even a health
summary that could not be delivered. The boot time was also estimated once at
open, so an NTP step after that skewed every ``observed_at``.
"""

from __future__ import annotations

import errno
import signal
import threading
import time
from datetime import timedelta
from pathlib import Path

from gpu_fault.channel_registry import COLLECTOR_HEALTH_PATH

from ._support import (
    NOW,
    CollectorError,
    KernelLogCollector,
    RecordingSink,
    context,
    io,
    logging,
    pytest,
)

FIRST = "3,42,2000001,-;NVRM: Xid (PCI:0000:b9:00): 94\n"
SECOND = "3,43,2000002,-;NVRM: Xid (PCI:0000:b9:00): 79\n"
THIRD = "3,44,2000003,-;NVRM: Xid (PCI:0000:b9:00): 74\n"
FOURTH = "3,45,2000004,-;NVRM: Xid (PCI:0000:b9:00): 31\n"
FIFTH = "3,46,2000005,-;NVRM: Xid (PCI:0000:b9:00): 13\n"


class _RejectFirstSink:
    """The control plane rejects the first event outright; nothing buffers it."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, dict]] = []

    def post(self, path, payload):
        self.requests.append((path, payload))
        if len(self.requests) == 1:
            raise CollectorError("collector event rejected (422)", status_code=422)
        return {"accepted": True}


def test_a_rejected_event_does_not_reopen_the_kmsg_stream(monkeypatch) -> None:
    opens = 0

    def fake_open(*_args, **_kwargs):
        nonlocal opens
        opens += 1
        return io.StringIO(FIRST + SECOND)

    monkeypatch.setattr("builtins.open", fake_open)

    def stop(_seconds: float) -> None:
        raise KeyboardInterrupt

    sink = _RejectFirstSink()
    collector = KernelLogCollector(
        sink,
        context(),
        node_id="worker-1",
        boot_id="boot-123",
        now=lambda: NOW,
        start_at_end=False,
        sleep=stop,
    )

    with pytest.raises(KeyboardInterrupt):
        collector.run()

    assert opens == 1, "a delivery failure reopened the stream"
    assert [item[1]["record_id"] for item in sink.requests] == [
        "kmsg-boot-123-42",
        "kmsg-boot-123-43",
    ], "the record after the rejected one was lost"
    assert collector.health_counters["delivery_failures"] == 1, (
        "the rejected delivery was not counted"
    )


def test_kmsg_overflow_continues_on_the_same_stream_and_is_counted(
    monkeypatch, caplog
) -> None:
    class OverflowingStream:
        def __init__(self) -> None:
            self.reads = 0

        def fileno(self) -> int:
            return 42

        def readline(self) -> str:
            self.reads += 1
            if self.reads == 1:
                return FIRST
            if self.reads == 2:
                raise OSError(errno.EPIPE, "Broken pipe")
            if self.reads == 3:
                return SECOND
            return ""

    ready = iter([([42], [], [])] * 4)
    monkeypatch.setattr(
        "gpu_fault.collectors.logs.kernel.select.select", lambda *_args: next(ready)
    )
    sink = RecordingSink()
    collector = KernelLogCollector(
        sink, context(), node_id="worker-1", boot_id="boot-123", now=lambda: NOW
    )
    stream = OverflowingStream()

    with caplog.at_level(logging.WARNING, logger="gpu_fault.collectors.logs.kernel"):
        collector._collect_live_stream(stream)

    assert stream.reads == 4, "the stream was abandoned after the overflow"
    assert [item[1]["record_id"] for item in sink.requests] == [
        "kmsg-boot-123-42",
        "kmsg-boot-123-43",
    ], "records after the overflow were not read"
    assert collector.health_counters["kmsg_overflow"] == 1, "overflow not counted"
    assert "overflow" in caplog.text.lower(), "overflow was not logged"


def test_other_read_errors_still_leave_the_live_stream(monkeypatch) -> None:
    class BrokenStream:
        def fileno(self) -> int:
            return 42

        def readline(self) -> str:
            raise OSError(errno.EIO, "I/O error")

    monkeypatch.setattr(
        "gpu_fault.collectors.logs.kernel.select.select", lambda *_args: ([42], [], [])
    )
    collector = KernelLogCollector(
        RecordingSink(),
        context(),
        node_id="worker-1",
        boot_id="boot-123",
        now=lambda: NOW,
    )

    with pytest.raises(OSError, match="I/O error"):
        collector._collect_live_stream(BrokenStream())


def test_health_summary_carries_failure_counters_only_when_non_zero() -> None:
    sink = RecordingSink()
    collector = KernelLogCollector(
        sink, context(), node_id="worker-1", boot_id="boot-123", now=lambda: NOW
    )

    collector._send_health_summary(NOW)
    collector.health_counters["delivery_failures"] += 2
    collector.health_counters["kmsg_overflow"] += 1
    collector._send_health_summary(NOW + timedelta(seconds=300))

    quiet, noisy = (payload for _path, payload in sink.requests)
    assert quiet["edge_filter_reasons"] == ["health-summary"], (
        "a healthy summary must keep the exact routine reason"
    )
    assert noisy["edge_filter_reasons"] == [
        "health-summary",
        "delivery-failures:2",
        "kmsg-overflow:1",
    ], "failure counters were not carried in the summary"


def test_boot_time_is_re_estimated_when_the_wall_clock_steps(tmp_path) -> None:
    uptime = tmp_path / "uptime"
    uptime.write_text("1000.00 4000.00\n")
    clock = [NOW]
    sink = RecordingSink()
    collector = KernelLogCollector(
        sink,
        context(),
        node_id="worker-1",
        boot_id="boot-123",
        now=lambda: clock[0],
        uptime_path=str(uptime),
    )
    collector.refresh_boot_time()

    # NTP steps the wall clock forward by 300 s while the host's uptime
    # advances by only 300 s: 600 s of wall time have passed.
    clock[0] = NOW + timedelta(seconds=600)
    uptime.write_text("1300.00 5200.00\n")
    collector.collect_lines(["3,50,1300000000,-;NVRM: Xid (PCI:0000:b9:00): 94\n"])

    observed_at = sink.requests[0][1]["observed_at"]
    assert observed_at == (NOW + timedelta(seconds=600)).isoformat(), (
        "the stale boot time skewed observed_at after the clock step"
    )
    assert collector.health_counters["boot_time_reestimates"] == 1, (
        "the re-estimate was not counted"
    )


class _BlockingSink:
    """A control plane that answers only when the test lets it.

    ``HttpEventSink`` spends up to ~47 s per record on retries and backoff
    before its outbox takes over, which is what made the inline post in the
    read loop a loss path: the kernel ring keeps overwriting records while the
    reader waits on the sink.
    """

    def __init__(self) -> None:
        self.requests: list[tuple[str, dict]] = []
        self.entered = threading.Event()
        self.released = threading.Event()

    def post(self, path, payload):
        self.requests.append((path, payload))
        self.entered.set()
        self.released.wait(10)
        return {"accepted": True}


def test_the_kmsg_reader_does_not_wait_for_the_sink() -> None:
    sink = _BlockingSink()
    collector = KernelLogCollector(
        sink, context(), node_id="worker-1", boot_id="boot-123", now=lambda: NOW
    )
    collector.start_delivery()
    try:
        stats = collector.collect_lines([FIRST, SECOND, THIRD])
        assert sink.entered.wait(5), "the delivery thread never reached the sink"
        assert stats.observed == 3, (
            "the reader did not finish all three records while the sink was blocked"
        )
        assert not sink.released.is_set(), (
            "the test released the sink before checking that the reader was free"
        )
    finally:
        sink.released.set()
        collector.stop_delivery()

    assert [item[1]["record_id"] for item in sink.requests] == [
        "kmsg-boot-123-42",
        "kmsg-boot-123-43",
        "kmsg-boot-123-44",
    ], f"queued records were lost instead of draining on shutdown: {sink.requests}"


def test_the_delivery_queue_drops_the_oldest_record_and_counts_it(caplog) -> None:
    sink = _BlockingSink()
    collector = KernelLogCollector(
        sink,
        context(),
        node_id="worker-1",
        boot_id="boot-123",
        now=lambda: NOW,
        delivery_queue_size=2,
    )
    collector.start_delivery()
    try:
        with caplog.at_level(
            logging.WARNING, logger="gpu_fault.collectors.logs.kernel"
        ):
            collector.collect_lines([FIRST])
            assert sink.entered.wait(5), "the delivery thread never took the first"
            # The delivery thread is now held inside ``post``, so the queue can
            # only grow: two fit, and the two after that push the oldest out.
            collector.collect_lines([SECOND, THIRD, FOURTH, FIFTH])
    finally:
        sink.released.set()
        collector.stop_delivery()

    assert collector.health_counters["delivery_queue_drops"] == 2, (
        "the records dropped by the bounded queue were not counted: "
        f"{collector.health_counters}"
    )
    assert [item[1]["record_id"] for item in sink.requests] == [
        "kmsg-boot-123-42",
        "kmsg-boot-123-45",
        "kmsg-boot-123-46",
    ], f"the queue dropped the newest records instead of the oldest: {sink.requests}"
    assert "delivery queue" in caplog.text.lower(), (
        "a dropped kernel record was not logged"
    )
    assert "delivery-queue-drops:2" in collector.health_summary_reasons(), (
        f"the drops are invisible in the health summary: "
        f"{collector.health_summary_reasons()}"
    )


class _UnexpectedErrorFirstSink:
    """A proxy answers 200 with an HTML body; ``json.loads`` raises ValueError."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, dict]] = []

    def post(self, path, payload):
        self.requests.append((path, payload))
        if len(self.requests) == 1:
            raise ValueError("Expecting value: line 1 column 1 (char 0)")
        return {"accepted": True}


def test_an_unexpected_delivery_error_does_not_reopen_the_kmsg_stream(
    monkeypatch,
) -> None:
    """``deliver_event`` converts ``CollectorError`` only (ARCH-G4).

    Anything else propagated out of ``collect_lines``, and ``run()``'s generic
    handler reopened ``/dev/kmsg`` and seeked to the live tail, so every record
    written in between was gone.
    """

    opens = 0

    def fake_open(*_args, **_kwargs):
        nonlocal opens
        opens += 1
        return io.StringIO(FIRST + SECOND)

    monkeypatch.setattr("builtins.open", fake_open)

    def stop(_seconds: float) -> None:
        raise KeyboardInterrupt

    sink = _UnexpectedErrorFirstSink()
    collector = KernelLogCollector(
        sink,
        context(),
        node_id="worker-1",
        boot_id="boot-123",
        now=lambda: NOW,
        start_at_end=False,
        sleep=stop,
    )

    with pytest.raises(KeyboardInterrupt):
        collector.run()

    assert opens == 1, "an unexpected delivery error reopened the stream"
    assert [item[1]["record_id"] for item in sink.requests] == [
        "kmsg-boot-123-42",
        "kmsg-boot-123-43",
    ], f"the record after the failed one was lost: {sink.requests}"
    assert collector.health_counters["delivery_failures"] == 1, (
        f"the failure was not counted: {collector.health_counters}"
    )


def test_collectors_without_a_readable_boot_id_do_not_share_record_ids(
    monkeypatch,
) -> None:
    """``kmsg-unknown-boot-42`` recurs after every reboot.

    kmsg sequence numbers restart at boot and the control plane's ``event_id``
    is cluster-scoped, so the first XID after a reboot was dedupped against the
    one before it whenever ``/proc/sys/kernel/random/boot_id`` was unreadable.
    """

    real_read_text = Path.read_text

    def unreadable(self, *args, **kwargs):
        if str(self) == "/proc/sys/kernel/random/boot_id":
            raise OSError("permission denied")
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", unreadable)
    record_ids = []
    for _ in range(2):
        sink = RecordingSink()
        KernelLogCollector(
            sink, context(), node_id="worker-1", now=lambda: NOW
        ).collect_lines([FIRST])
        record_ids.append(sink.requests[0][1]["record_id"])

    assert all(item.startswith("kmsg-unknown-") for item in record_ids), (
        f"an unreadable boot id must still say so in the record id: {record_ids}"
    )
    assert record_ids[0] != record_ids[1], (
        f"the same kmsg sequence after a reboot kept one record id: {record_ids}"
    )


def _live_delivery_threads() -> list[threading.Thread]:
    """Every live delivery thread, by the name the collector gives it."""

    return [
        item
        for item in threading.enumerate()
        if item.name == "kernel-collector-delivery" and item.is_alive()
    ]


def _wait_until_no_new_delivery_thread(
    known: set[threading.Thread], timeout_seconds: float = 5.0
) -> list[threading.Thread]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        extra = [item for item in _live_delivery_threads() if item not in known]
        if not extra or time.monotonic() >= deadline:
            return extra
        time.sleep(0.02)


class _SlowSink:
    """A control plane that takes its time, the way a retrying one does."""

    def __init__(self, delay_seconds: float) -> None:
        self.requests: list[tuple[str, dict]] = []
        self.delay_seconds = delay_seconds
        self.entered = threading.Event()

    def post(self, path, payload):
        self.requests.append((path, payload))
        self.entered.set()
        time.sleep(self.delay_seconds)
        return {"accepted": True}


class _ReplayBufferingSink(_SlowSink):
    """A sink that can persist a record without posting it (Task 20's method)."""

    def __init__(self, delay_seconds: float) -> None:
        super().__init__(delay_seconds)
        self.buffered: list[tuple[str, dict]] = []

    def buffer_for_replay(self, path, payload):
        self.buffered.append((path, payload))
        return True


def test_a_stop_signal_drains_the_delivery_queue_before_the_reader_exits(
    monkeypatch,
) -> None:
    """systemd stops the unit with SIGTERM on every deploy.

    Nothing installed a handler, so CPython died on the OS default and the
    ``finally`` that drains the delivery queue never ran: up to 2048 queued
    XID/SXID payloads went away with no post, no outbox record and no counter.
    """

    def fake_open(*_args, **_kwargs):
        return io.StringIO(FIRST + SECOND + THIRD)

    monkeypatch.setattr("builtins.open", fake_open)
    sink = _ReplayBufferingSink(delay_seconds=0.3)

    def stop(_seconds: float) -> None:
        # Deliver the signal the way systemd does, through whatever handler the
        # collector installed. Raising instead of calling an absent handler
        # keeps a red run from killing the test process.
        handler = signal.getsignal(signal.SIGTERM)
        assert callable(handler), (
            f"no SIGTERM handler was installed, so systemd's stop kills the "
            f"process with records still queued: {handler}"
        )
        handler(signal.SIGTERM, None)

    before = signal.getsignal(signal.SIGTERM)
    collector = KernelLogCollector(
        sink,
        context(),
        node_id="worker-1",
        boot_id="boot-123",
        now=lambda: NOW,
        start_at_end=False,
        sleep=stop,
    )

    collector.run()

    posted = [payload["record_id"] for _path, payload in sink.requests]
    buffered = [payload["record_id"] for _path, payload in sink.buffered]
    assert sorted(posted + buffered) == [
        "kmsg-boot-123-42",
        "kmsg-boot-123-43",
        "kmsg-boot-123-44",
    ], f"the stop signal lost records: posted={posted} buffered={buffered}"
    # Each post takes 0.3 s and the reader gets through the stream in
    # microseconds, so at most one record can have been posted already.
    assert len(buffered) >= 2, (
        f"the stop signal did not hand the queue to the outbox: {buffered}"
    )
    assert collector.health_counters["delivery_buffered_at_shutdown"] == len(
        buffered
    ), f"the records handed to the outbox were not counted: {collector.health_counters}"
    assert signal.getsignal(signal.SIGTERM) is before, (
        "the collector kept the process's SIGTERM handler after run() returned"
    )


def test_queued_records_are_buffered_for_replay_at_shutdown() -> None:
    """The drain is a single-shot buffer, not a retry storm.

    2048 records times ~47 s of retries fits no stop timeout, so the drain hands
    each record to the sink's durable outbox once. Delivered, buffered or
    counted -- there is no fourth state (ARCH-G3).
    """

    sink = _ReplayBufferingSink(delay_seconds=0.3)
    collector = KernelLogCollector(
        sink, context(), node_id="worker-1", boot_id="boot-123", now=lambda: NOW
    )
    collector.start_delivery()
    collector.collect_lines([FIRST, SECOND, THIRD, FOURTH])
    assert sink.entered.wait(5), "the delivery thread never reached the sink"

    collector.stop_delivery()

    assert [payload["record_id"] for _path, payload in sink.buffered] == [
        "kmsg-boot-123-43",
        "kmsg-boot-123-44",
        "kmsg-boot-123-45",
    ], f"the queue was not handed to the outbox on shutdown: {sink.buffered}"
    assert collector.health_counters["delivery_buffered_at_shutdown"] == 3, (
        f"the buffered records were not counted: {collector.health_counters}"
    )
    assert "delivery-buffered-at-shutdown:3" in collector.health_summary_reasons(), (
        f"the shutdown buffer is invisible in the health summary: "
        f"{collector.health_summary_reasons()}"
    )


def test_a_shutdown_without_a_replay_buffer_counts_what_the_budget_left(caplog) -> None:
    """A sink with no outbox gets one attempt per record inside a budget."""

    sink = _SlowSink(delay_seconds=0.2)
    collector = KernelLogCollector(
        sink, context(), node_id="worker-1", boot_id="boot-123", now=lambda: NOW
    )
    collector.start_delivery()
    collector.collect_lines([FIRST, SECOND, THIRD, FOURTH])
    assert sink.entered.wait(5), "the delivery thread never reached the sink"

    with caplog.at_level(logging.WARNING, logger="gpu_fault.collectors.logs.kernel"):
        collector.stop_delivery(drain_budget_seconds=0.05)

    assert [payload["record_id"] for _path, payload in sink.requests] == [
        "kmsg-boot-123-42",
        "kmsg-boot-123-43",
    ], f"the drain kept retrying past its budget: {sink.requests}"
    assert collector.health_counters["delivery_dropped_at_shutdown"] == 2, (
        f"the records the budget left were not counted: {collector.health_counters}"
    )
    assert "delivery-dropped-at-shutdown:2" in collector.health_summary_reasons(), (
        f"records dropped at shutdown are invisible: "
        f"{collector.health_summary_reasons()}"
    )
    assert "kmsg-boot-123-45" in caplog.text, (
        f"the records lost at shutdown were not named in the log: {caplog.text}"
    )


def test_a_delivery_thread_that_outlived_its_join_stops_consuming() -> None:
    """A join timeout must not resurrect the thread it gave up on.

    ``stop_delivery`` reset the shared stop flag and the thread handle while the
    old thread was still inside ``post``, so that thread looped forever and a
    later ``start_delivery`` left two consumers on one queue -- the second of
    which nothing ever joins.
    """

    known = set(_live_delivery_threads())
    sink = _SlowSink(delay_seconds=0.4)
    collector = KernelLogCollector(
        sink, context(), node_id="worker-1", boot_id="boot-123", now=lambda: NOW
    )
    collector.start_delivery()
    collector.collect_lines([FIRST])
    assert sink.entered.wait(5), "the delivery thread never reached the sink"

    # The join gives up while that post is still in flight.
    collector.stop_delivery(timeout_seconds=0.02, drain_budget_seconds=0.0)
    stranded = _wait_until_no_new_delivery_thread(known)
    assert stranded == [], (
        f"the thread the join gave up on is still consuming the queue: {stranded}"
    )

    collector.start_delivery()
    collector.collect_lines([SECOND])
    collector.stop_delivery()

    assert [payload["record_id"] for _path, payload in sink.requests] == [
        "kmsg-boot-123-42",
        "kmsg-boot-123-43",
    ], f"a record was delivered twice or not at all: {sink.requests}"
    assert _wait_until_no_new_delivery_thread(known) == [], (
        "a delivery thread outlived the collector"
    )


class _ThreadKillingSink:
    """The first post raises a ``BaseException``, which kills its thread.

    ``_deliver_one`` converts every ``Exception``, so only something worse can
    end the delivery loop -- and then the queue has no consumer at all.
    """

    def __init__(self) -> None:
        self.requests: list[tuple[str, dict]] = []
        self.entered = threading.Event()

    def post(self, path, payload):
        self.requests.append((path, payload))
        self.entered.set()
        if len(self.requests) == 1:
            raise SystemExit("the delivery thread dies here")
        return {"accepted": True}


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_a_dead_delivery_thread_is_reported_and_replaced(caplog) -> None:
    """A queue nobody drains is a fourth state.

    ``_submit`` fell back to inline delivery without a counter, a warning or a
    restart, and everything already queued stayed there until the process
    exited.
    """

    known = set(_live_delivery_threads())
    sink = _ThreadKillingSink()
    collector = KernelLogCollector(
        sink, context(), node_id="worker-1", boot_id="boot-123", now=lambda: NOW
    )
    collector.start_delivery()
    collector.collect_lines([FIRST, SECOND, THIRD])
    assert sink.entered.wait(5), "the delivery thread never reached the sink"
    assert _wait_until_no_new_delivery_thread(known) == [], (
        "the sink did not kill the delivery thread, so the case is untested"
    )

    with caplog.at_level(logging.WARNING, logger="gpu_fault.collectors.logs.kernel"):
        collector.collect_lines([FOURTH])
        collector.stop_delivery()

    assert collector.health_counters["delivery_thread_deaths"] == 1, (
        f"the dead delivery thread was not counted: {collector.health_counters}"
    )
    assert [payload["record_id"] for _path, payload in sink.requests] == [
        "kmsg-boot-123-42",
        "kmsg-boot-123-43",
        "kmsg-boot-123-44",
        "kmsg-boot-123-45",
    ], f"the records stranded in the queue were never delivered: {sink.requests}"
    assert "delivery thread" in caplog.text.lower(), (
        "the delivery thread died without a word"
    )


class _ThreadNamingSink:
    """Records which thread each post came from."""

    def __init__(self) -> None:
        self.posts: list[tuple[str, str]] = []
        self.summary_posted = threading.Event()

    def post(self, path, payload):
        self.posts.append((path, threading.current_thread().name))
        if path == COLLECTOR_HEALTH_PATH:
            self.summary_posted.set()
        return {"accepted": True}


def test_the_health_summary_is_not_posted_from_the_read_loop(monkeypatch) -> None:
    """The summary posted synchronously from the reader.

    An unreachable control plane costs ~47 s per post, so once every 300 s the
    kmsg reader stopped reading for that long while the kernel ring kept
    overwriting records -- the loss the queue exists to prevent. The summary
    carries its own channel, so it goes through the same queue.
    """

    def fake_open(*_args, **_kwargs):
        return io.StringIO(FIRST + SECOND)

    monkeypatch.setattr("builtins.open", fake_open)
    sink = _ThreadNamingSink()
    # The health summary is due: the collector phased it from NOW, and the read
    # loop reaches it ten minutes later.
    clock = [NOW]

    def stop(_seconds: float) -> None:
        assert sink.summary_posted.wait(5), "the health summary never reached the sink"
        handler = signal.getsignal(signal.SIGTERM)
        assert callable(handler), f"no SIGTERM handler was installed: {handler}"
        handler(signal.SIGTERM, None)

    collector = KernelLogCollector(
        sink,
        context(),
        node_id="worker-1",
        boot_id="boot-123",
        now=lambda: clock[0],
        start_at_end=False,
        sleep=stop,
    )
    clock[0] = NOW + timedelta(seconds=600)

    collector.run()

    summary_posts = [name for path, name in sink.posts if path == COLLECTOR_HEALTH_PATH]
    assert summary_posts == ["kernel-collector-delivery"], (
        f"the health summary was posted from the read loop: {sink.posts}"
    )
    assert [name for path, name in sink.posts if path != COLLECTOR_HEALTH_PATH] == [
        "kernel-collector-delivery",
        "kernel-collector-delivery",
    ], f"the kmsg records did not go through the delivery thread: {sink.posts}"


class _UnwritableOutboxSink(_SlowSink):
    """The outbox write fails: a read-only volume, a full disk, or none configured.

    ``HttpEventSink.buffer_for_replay`` returns ``False`` and never raises in
    that case (``sinks.py``), which is exactly the case the drain has to notice.
    """

    def __init__(self, delay_seconds: float) -> None:
        super().__init__(delay_seconds)
        self.buffer_attempts: list[str] = []

    def buffer_for_replay(self, path, payload):
        self.buffer_attempts.append(payload["record_id"])
        return False


def test_a_failed_outbox_write_is_not_reported_as_buffered(caplog) -> None:
    """``False`` from ``buffer_for_replay`` means the record is still ours.

    The return value was ignored, so a full disk or a read-only volume -- the
    case the sink's own comment names -- lost the whole queue while the health
    summary said it had been buffered. Each unwritten record gets one delivery
    attempt inside the remaining budget, and what the budget leaves is counted.
    """

    sink = _UnwritableOutboxSink(delay_seconds=0.2)
    collector = KernelLogCollector(
        sink, context(), node_id="worker-1", boot_id="boot-123", now=lambda: NOW
    )
    collector.start_delivery()
    collector.collect_lines([FIRST, SECOND, THIRD, FOURTH])
    assert sink.entered.wait(5), "the delivery thread never reached the sink"

    with caplog.at_level(logging.ERROR, logger="gpu_fault.collectors.logs.kernel"):
        collector.stop_delivery(drain_budget_seconds=0.05)

    assert sink.buffer_attempts, "the drain never tried the outbox at all"
    assert collector.health_counters["delivery_buffered_at_shutdown"] == 0, (
        f"a failed outbox write was counted as buffered: {collector.health_counters}"
    )
    assert [payload["record_id"] for _path, payload in sink.requests] == [
        "kmsg-boot-123-42",
        "kmsg-boot-123-43",
    ], f"the record the outbox refused was never posted either: {sink.requests}"
    assert collector.health_counters["delivery_dropped_at_shutdown"] == 2, (
        f"the records neither buffered nor delivered were not counted: "
        f"{collector.health_counters}"
    )
    assert "kmsg-boot-123-45" in caplog.text, (
        f"the records lost at shutdown were not named in the log: {caplog.text}"
    )


class _SlowOutboxSink(_SlowSink):
    """Buffering itself blocks: the outbox flock is held by the replay thread."""

    def __init__(self, delay_seconds: float, buffer_delay_seconds: float) -> None:
        super().__init__(delay_seconds)
        self.buffered: list[str] = []
        self.buffer_delay_seconds = buffer_delay_seconds

    def buffer_for_replay(self, path, payload):
        time.sleep(self.buffer_delay_seconds)
        self.buffered.append(payload["record_id"])
        return True


def test_the_shutdown_drain_stops_buffering_when_the_budget_is_gone() -> None:
    """The buffer branch ignored the deadline entirely.

    ``buffer_for_replay`` takes the outbox lock, which the sink's own replay
    thread may hold, so a queue of 2048 records could sit past the stop timeout
    and be SIGKILLed with no verdict at all.
    """

    sink = _SlowOutboxSink(delay_seconds=0.2, buffer_delay_seconds=0.2)
    collector = KernelLogCollector(
        sink, context(), node_id="worker-1", boot_id="boot-123", now=lambda: NOW
    )
    collector.start_delivery()
    collector.collect_lines([FIRST, SECOND, THIRD, FOURTH])
    assert sink.entered.wait(5), "the delivery thread never reached the sink"

    collector.stop_delivery(drain_budget_seconds=0.05)

    assert sink.buffered == ["kmsg-boot-123-43"], (
        f"the drain kept buffering past its budget: {sink.buffered}"
    )
    assert collector.health_counters["delivery_buffered_at_shutdown"] == 1, (
        f"the buffered record was not counted: {collector.health_counters}"
    )
    assert collector.health_counters["delivery_dropped_at_shutdown"] == 2, (
        f"the records the budget left were not counted: {collector.health_counters}"
    )


class _InterruptedDrainSink(_SlowSink):
    """The operator's second Ctrl-C lands while the drain is inside the sink.

    A ``KeyboardInterrupt`` raised out of a blocking call is what a second
    SIGINT looks like from here: the first record is buffered, the sink blocks on
    the next one and the signal arrives.
    """

    def __init__(self, delay_seconds: float) -> None:
        super().__init__(delay_seconds)
        self.buffered: list[str] = []
        self.interrupted = threading.Event()

    def buffer_for_replay(self, path, payload):
        if self.buffered:
            self.interrupted.set()
            raise KeyboardInterrupt
        self.buffered.append(payload["record_id"])
        return True


def test_a_second_stop_signal_during_the_drain_counts_what_is_left(caplog) -> None:
    """A drain that is interrupted must still say what it could not account for.

    The first Ctrl-C starts the drain; a second one raises ``KeyboardInterrupt``
    out of whatever the drain is blocked in. The records still in hand then had
    no verdict, no counter and no log line at all -- exactly the silent loss the
    drain exists to remove, reachable by pressing Ctrl-C twice or by a
    supervisor that repeats its stop signal.
    """

    sink = _InterruptedDrainSink(delay_seconds=0.3)
    collector = KernelLogCollector(
        sink, context(), node_id="worker-1", boot_id="boot-123", now=lambda: NOW
    )
    collector.start_delivery()
    collector.collect_lines([FIRST, SECOND, THIRD, FOURTH])
    assert sink.entered.wait(5), "the delivery thread never reached the sink"

    with caplog.at_level(logging.ERROR, logger="gpu_fault.collectors.logs.kernel"):
        with pytest.raises(KeyboardInterrupt):
            collector.stop_delivery()

    assert sink.interrupted.is_set(), "the drain never reached the interrupted record"
    assert sink.buffered == ["kmsg-boot-123-43"], (
        f"the drain did not buffer the record before the interrupt: {sink.buffered}"
    )
    assert collector.health_counters["delivery_dropped_at_shutdown"] == 2, (
        "the records the interrupted drain still held were not counted: "
        f"{collector.health_counters}"
    )
    assert "kmsg-boot-123-45" in caplog.text, (
        f"the records the interrupt left unaccounted for were not named: {caplog.text}"
    )
    assert "buffered 1" in caplog.text, (
        f"the interrupted drain did not report what it had already saved: {caplog.text}"
    )


class _WedgedSink:
    """Blocks inside ``post`` until it is released, then buffers on request."""

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.requests: list[str] = []
        self.buffered: list[str] = []

    def post(self, path, payload):
        self.requests.append(payload["record_id"])
        self.entered.set()
        assert self.release.wait(10), "the wedged sink was never released"
        return {"accepted": True}

    def buffer_for_replay(self, path, payload):
        self.buffered.append(payload["record_id"])
        return True


def test_the_record_in_flight_at_a_join_timeout_still_gets_a_verdict() -> None:
    """The in-flight record was the one record nobody accounted for.

    It had been popped, so the drain never saw it, and its thread was retired
    after the join and killed with the process mid-``deliver_event`` -- ~47 s of
    retries that a stopping process does not have. It goes back into the drain,
    where a duplicate delivery is deduped by ``event_id`` and a loss is not.
    """

    sink = _WedgedSink()
    collector = KernelLogCollector(
        sink, context(), node_id="worker-1", boot_id="boot-123", now=lambda: NOW
    )
    collector.start_delivery()
    try:
        collector.collect_lines([FIRST])
        assert sink.entered.wait(5), "the delivery thread never reached the sink"
        collector.collect_lines([SECOND])

        collector.stop_delivery(timeout_seconds=0.05, drain_budget_seconds=1.0)
    finally:
        sink.release.set()

    assert sorted(sink.buffered) == ["kmsg-boot-123-42", "kmsg-boot-123-43"], (
        f"the record in flight at the join timeout was lost: {sink.buffered}"
    )
    assert collector.health_counters["delivery_buffered_at_shutdown"] == 2, (
        f"the in-flight record was not counted either: {collector.health_counters}"
    )


class _SignalWatchingSink(_SlowSink):
    """Records the process's SIGTERM disposition at each shutdown buffer call."""

    def __init__(self) -> None:
        super().__init__(delay_seconds=0.0)
        self.handlers_during_drain: list[object] = []
        self.buffered: list[str] = []

    def buffer_for_replay(self, path, payload):
        self.handlers_during_drain.append(signal.getsignal(signal.SIGTERM))
        self.buffered.append(payload["record_id"])
        return True


def test_a_second_stop_signal_does_not_kill_the_drain(monkeypatch) -> None:
    """The handlers were restored before the drain ran.

    The drain can take ~10 s; systemd sends SIGTERM again on a slow stop, and
    with the default disposition back in place that second signal killed the
    process in the middle of handing records to the outbox.
    """

    def fake_open(*_args, **_kwargs):
        return io.StringIO(FIRST + SECOND + THIRD)

    monkeypatch.setattr("builtins.open", fake_open)
    sink = _SignalWatchingSink()

    def stop(_seconds: float) -> None:
        handler = signal.getsignal(signal.SIGTERM)
        assert callable(handler), f"no SIGTERM handler was installed: {handler}"
        handler(signal.SIGTERM, None)

    before = signal.getsignal(signal.SIGTERM)
    collector = KernelLogCollector(
        sink,
        context(),
        node_id="worker-1",
        boot_id="boot-123",
        now=lambda: NOW,
        start_at_end=False,
        sleep=stop,
    )

    collector.run()

    assert sink.buffered, "the drain never reached the outbox, so nothing is proven"
    assert all(callable(item) for item in sink.handlers_during_drain), (
        f"the stop handler was restored before the drain finished: "
        f"{sink.handlers_during_drain}"
    )
    assert signal.getsignal(signal.SIGTERM) is before, (
        "the collector kept the process's SIGTERM handler after run() returned"
    )


def test_an_ignored_stop_signal_stays_ignored(monkeypatch) -> None:
    """A signal the parent set to ``SIG_IGN`` must not be re-armed.

    Installing a handler over an inherited ``SIG_IGN`` overrides a deliberate
    decision of whoever started the process (a supervisor that stops its
    children itself), and turns a signal it expects to be swallowed into a
    shutdown.
    """

    def fake_open(*_args, **_kwargs):
        return io.StringIO(FIRST)

    monkeypatch.setattr("builtins.open", fake_open)
    seen: list[object] = []

    def stop(_seconds: float) -> None:
        seen.append(signal.getsignal(signal.SIGINT))
        handler = signal.getsignal(signal.SIGTERM)
        assert callable(handler), f"no SIGTERM handler was installed: {handler}"
        handler(signal.SIGTERM, None)

    collector = KernelLogCollector(
        RecordingSink(),
        context(),
        node_id="worker-1",
        boot_id="boot-123",
        now=lambda: NOW,
        start_at_end=False,
        sleep=stop,
    )
    previous_int = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        collector.run()
    finally:
        signal.signal(signal.SIGINT, previous_int)

    assert seen == [signal.SIG_IGN], (
        f"an inherited SIG_IGN was replaced by the collector's handler: {seen}"
    )


def test_a_dropped_health_summary_is_not_counted_as_a_lost_record(
    monkeypatch, caplog
) -> None:
    """``delivery_queue_drops`` says XID records are lost, so it must be true.

    The summary shares the queue with the records, and the oldest entry is the
    one dropped when the queue is full. Counting a dropped summary there says
    the node lost fault evidence when it lost one liveness report.
    """

    streams = [io.StringIO(FIRST), io.StringIO(""), io.StringIO(SECOND)]

    def fake_open(*_args, **_kwargs):
        return streams.pop(0) if streams else io.StringIO("")

    monkeypatch.setattr("builtins.open", fake_open)
    sink = _WedgedSink()
    clock = [NOW]
    rounds: list[int] = []

    def stop(_seconds: float) -> None:
        rounds.append(len(rounds))
        if len(rounds) == 1:
            # The delivery thread is inside the wedged post, so the queue keeps
            # everything the reader hands it from here on.
            assert sink.entered.wait(5), "the delivery thread never reached the sink"
            clock[0] = NOW + timedelta(seconds=600)
            return
        if len(rounds) == 2:
            # Round 2 read nothing and queued the health summary, which is now
            # the oldest entry in a queue of one.
            return
        sink.release.set()
        handler = signal.getsignal(signal.SIGTERM)
        assert callable(handler), f"no SIGTERM handler was installed: {handler}"
        handler(signal.SIGTERM, None)

    collector = KernelLogCollector(
        sink,
        context(),
        node_id="worker-1",
        boot_id="boot-123",
        now=lambda: clock[0],
        start_at_end=False,
        sleep=stop,
        delivery_queue_size=1,
    )

    with caplog.at_level(logging.WARNING, logger="gpu_fault.collectors.logs.kernel"):
        collector.run()

    assert collector.health_counters["health_summary_queue_drops"] == 1, (
        f"the dropped health summary was not counted anywhere: "
        f"{collector.health_counters}"
    )
    assert collector.health_counters["delivery_queue_drops"] == 0, (
        f"a dropped health summary was reported as lost XID records: "
        f"{collector.health_counters}"
    )
    assert "health summary" in caplog.text.lower(), (
        f"the dropped health summary was not logged as one: {caplog.text}"
    )


class _RetryLadderSink:
    """``HttpEventSink``'s shape: ``post`` walks the retry ladder, ``deliver_once`` does not.

    The control plane is gone and the outbox cannot be written (ENOSPC/EROFS),
    so ``buffer_for_replay`` answers ``False`` and a live ``post`` spends every
    attempt plus every backoff -- the ~47 s ladder in production, scaled here.
    """

    def __init__(self, attempt_seconds: float, attempts: int) -> None:
        self.attempt_seconds = attempt_seconds
        self.attempts = attempts
        self.posts: list[str] = []
        self.single_attempts: list[str] = []
        self.entered = threading.Event()

    def post(self, path, payload):
        self.entered.set()
        for _attempt in range(self.attempts):
            self.posts.append(payload["record_id"])
            time.sleep(self.attempt_seconds)
        raise CollectorError("control-plane delivery failed after 4 attempts")

    def deliver_once(self, path, payload):
        self.entered.set()
        self.single_attempts.append(payload["record_id"])
        time.sleep(self.attempt_seconds)
        raise CollectorError("control-plane delivery failed after 1 attempts")

    def buffer_for_replay(self, path, payload):
        return False


def test_the_shutdown_drain_gives_each_unbuffered_record_one_attempt_not_the_ladder(
    caplog,
) -> None:
    """A full disk at shutdown must not turn the drain into ``records x 47 s``.

    The budget was checked only *between* records, and a record the outbox
    refused went through ``deliver_event`` -- the full retry ladder. Task 16
    set the unit's ``TimeoutStopSec`` believing the drain was bounded at about
    ten seconds, so systemd SIGKILLed the process mid-ladder: the ``finally``
    never ran, the lost-records ERROR never appeared, and the remaining queue
    was lost in silence.
    """

    sink = _RetryLadderSink(attempt_seconds=0.2, attempts=4)
    collector = KernelLogCollector(
        sink, context(), node_id="worker-1", boot_id="boot-123", now=lambda: NOW
    )
    collector.start_delivery()
    collector.collect_lines([FIRST, SECOND, THIRD, FOURTH])
    assert sink.entered.wait(5), "the delivery thread never reached the sink"

    started = time.monotonic()
    with caplog.at_level(logging.ERROR, logger="gpu_fault.collectors.logs.kernel"):
        collector.stop_delivery(timeout_seconds=0.05, drain_budget_seconds=0.3)
    elapsed = time.monotonic() - started

    drained = sink.single_attempts
    # The record in flight at the join timeout (42) is requeued first, then
    # the queue in order; each gets exactly one bounded attempt.
    assert drained[:2] == ["kmsg-boot-123-42", "kmsg-boot-123-43"], (
        "the drain did not give the unbuffered records a single attempt each: "
        f"single_attempts={drained} posts={sink.posts}"
    )
    assert all(drained.count(record_id) == 1 for record_id in drained), (
        f"a record was attempted more than once during the drain: {drained}"
    )
    assert "kmsg-boot-123-43" not in sink.posts, (
        f"the drain sent an unbuffered record down the full retry ladder: {sink.posts}"
    )
    assert elapsed < 1.5, (
        f"the drain ran {elapsed:.1f}s against a 0.3s budget: the retry ladder "
        "ran past the shutdown budget"
    )
    lost = collector.health_counters["delivery_dropped_at_shutdown"]
    assert lost >= 1, (
        f"the records the budget left were not counted as lost: "
        f"{collector.health_counters}"
    )
    assert "kmsg-boot-123-45" in caplog.text, (
        f"the records lost at shutdown were not named in the log: {caplog.text}"
    )


def test_a_health_summary_never_evicts_a_queued_xid(monkeypatch, caplog) -> None:
    """A full queue drops the *summary*, not the oldest XID record.

    Drop-oldest kept the newest record, which is right when the newcomer is an
    XID. When the newcomer is the 300 s health summary it evicted real fault
    evidence to make room for a liveness report; the summary drops itself and
    is counted as ``health_summary_queue_drops``.
    """

    streams = [io.StringIO(FIRST), io.StringIO(SECOND), io.StringIO("")]

    def fake_open(*_args, **_kwargs):
        return streams.pop(0) if streams else io.StringIO("")

    monkeypatch.setattr("builtins.open", fake_open)
    sink = _WedgedSink()
    clock = [NOW]
    rounds: list[int] = []

    def stop(_seconds: float) -> None:
        rounds.append(len(rounds))
        if len(rounds) == 1:
            # The delivery thread is inside the wedged post with FIRST, so the
            # queue keeps everything the reader hands it from here on.
            assert sink.entered.wait(5), "the delivery thread never reached the sink"
            return
        if len(rounds) == 2:
            # Round 2 queued SECOND: the queue of one is full of fault evidence.
            clock[0] = NOW + timedelta(seconds=600)
            return
        # Round 3 read nothing and owed a health summary.
        sink.release.set()
        handler = signal.getsignal(signal.SIGTERM)
        assert callable(handler), f"no SIGTERM handler was installed: {handler}"
        handler(signal.SIGTERM, None)

    collector = KernelLogCollector(
        sink,
        context(),
        node_id="worker-1",
        boot_id="boot-123",
        now=lambda: clock[0],
        start_at_end=False,
        sleep=stop,
        delivery_queue_size=1,
    )

    with caplog.at_level(logging.WARNING, logger="gpu_fault.collectors.logs.kernel"):
        collector.run()

    assert collector.health_counters["delivery_queue_drops"] == 0, (
        f"a health summary evicted a queued XID record: {collector.health_counters}"
    )
    assert collector.health_counters["health_summary_queue_drops"] == 1, (
        f"the summary that yielded to the XID was not counted: "
        f"{collector.health_counters}"
    )
    assert "kmsg-boot-123-43" in sink.requests + sink.buffered, (
        f"the queued XID never got a verdict: posted={sink.requests} "
        f"buffered={sink.buffered}"
    )
    assert not any(item.startswith("kernel-health-") for item in sink.buffered), (
        f"the summary took the XID's place in the queue: {sink.buffered}"
    )
    assert "health summary" in caplog.text.lower(), (
        f"the dropped health summary was not logged as one: {caplog.text}"
    )
