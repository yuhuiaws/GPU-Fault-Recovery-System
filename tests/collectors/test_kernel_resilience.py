"""The kernel collector must not lose kmsg records to its own failures (ARCH-G4/G9).

Any exception used to reopen ``/dev/kmsg`` and seek to the end, so every record
written between the failure and the reopen was gone: a rejected event (a 4xx
nobody buffered), a kernel ring overflow (``EPIPE`` on read), even a health
summary that could not be delivered. The boot time was also estimated once at
open, so an NTP step after that skewed every ``observed_at``.
"""

from __future__ import annotations

import errno
from datetime import timedelta

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
