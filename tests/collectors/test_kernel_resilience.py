"""The kernel collector must not lose kmsg records to its own failures (ARCH-G4/G9).

Any exception used to reopen ``/dev/kmsg`` and seek to the end, so every record
written between the failure and the reopen was gone: a rejected event (a 4xx
nobody buffered), a kernel ring overflow (``EPIPE`` on read), even a health
summary that could not be delivered. The boot time was also estimated once at
open, so an NTP step after that skewed every ``observed_at``.
"""

from __future__ import annotations

import errno
import threading
from datetime import timedelta
from pathlib import Path

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
