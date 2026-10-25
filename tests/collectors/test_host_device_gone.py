"""A block device that left ``/proc/diskstats`` is reported as gone, not
silently dropped (batch 160, live GF-GENERAL-LIVE-006).

The collector rated a dm-delay device's ``disk_io_await_ms`` above the
threshold, the device was torn down, and nothing ever told the control plane:
the signal stayed notified and a same-named successor two hours later was
silenced by the latch. The collector now sends one ``device_state=GONE``
sample per ``disk_io_*`` metric for a device present on the previous read and
absent now (or whose counters restarted), repeated until a batch carries them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gpu_fault.host_device_signals import (
    DEVICE_STATE_GONE,
    DEVICE_STATE_LABEL,
    device_gone,
)
from gpu_fault.host_health import HostMetricSample

from ._support import (
    NOW,
    CollectorError,
    HostTelemetryCollector,
    RecordingSink,
    RejectingSink,
    context,
    timedelta,
)

NVME = "259 0 nvme0n1 {reads} 0 0 {read_ms} {writes} 0 0 {write_ms} 0 {io_ms} 0\n"
DM0 = "253 0 dm-0 {reads} 0 0 {read_ms} {writes} 0 0 {write_ms} 0 {io_ms} 0\n"
GONE_PAIR = [("disk_io_util_percent", "dm-0"), ("disk_io_await_ms", "dm-0")]
GONE_LABELS = {DEVICE_STATE_LABEL: DEVICE_STATE_GONE}


def nvme(step: int) -> str:
    return NVME.format(
        reads=10 * step,
        read_ms=20 * step,
        writes=5 * step,
        write_ms=10 * step,
        io_ms=50 * step,
    )


def dm0(step: int) -> str:
    # 100 ms per I/O: 20 I/Os and 4000 ms per step, 200 ms await per interval.
    return DM0.format(
        reads=10 * step,
        read_ms=2000 * step,
        writes=10 * step,
        write_ms=2000 * step,
        io_ms=5 * step,
    )


def feed(monkeypatch: pytest.MonkeyPatch, snapshots: list[str]) -> None:
    reads = iter(snapshots)
    original = Path.read_text

    def read_text(path, *args, **kwargs):
        if str(path) == "/proc/diskstats":
            return next(reads)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)


def only_diskstats(
    monkeypatch: pytest.MonkeyPatch, host: HostTelemetryCollector
) -> None:
    for name in HostTelemetryCollector.CONTRIBUTORS:
        if name != "_diskstats":
            monkeypatch.setattr(host, name, lambda _observed_at: [])


def posted(sink) -> list[HostMetricSample]:
    """The samples of the last batch the sink saw, as the wire carried them."""

    return [HostMetricSample.model_validate(s) for s in sink.requests[-1][1]["samples"]]


def rated(samples) -> dict[tuple[str, str | None], float]:
    return {(s.name, s.device): s.value for s in samples if not device_gone(s)}


def gone(samples) -> list[tuple[str, str | None]]:
    return [(s.name, s.device) for s in samples if device_gone(s)]


def test_a_device_that_left_diskstats_is_reported_gone_until_a_batch_is_delivered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rejecting = RejectingSink()
    ticks = iter(NOW + timedelta(seconds=15 * tick) for tick in range(8))
    host = HostTelemetryCollector(
        rejecting,
        context(),
        node_id="worker-1",
        now=lambda: next(ticks),
        edge_filter_enabled=False,
    )
    only_diskstats(monkeypatch, host)
    feed(
        monkeypatch,
        [
            nvme(1) + dm0(1),
            nvme(2) + dm0(2),
            nvme(3),  # dm-0 torn down
            nvme(4),
            nvme(5),
            nvme(6),
            nvme(7) + dm0(1),  # a same-named successor appears
            nvme(8) + dm0(2),
        ],
    )

    def tick_rejected() -> list[HostMetricSample]:
        with pytest.raises(CollectorError):
            host.collect_once()
        return posted(rejecting)

    tick_rejected()
    second = tick_rejected()
    assert rated(second)[("disk_io_await_ms", "dm-0")] == 200.0
    assert gone(second) == []

    vanished = tick_rejected()
    assert gone(vanished) == GONE_PAIR
    assert [(s.name, s.device) for s in vanished[:2]] == GONE_PAIR, (
        "the GONE samples come first in the tick"
    )
    assert [(s.value, s.labels) for s in vanished[:2]] == [(0.0, GONE_LABELS)] * 2
    assert ("disk_io_await_ms", "nvme0n1") in rated(vanished), (
        "the devices still present are rated as before"
    )

    # Not delivered (the control plane rejected the batch): repeated next tick.
    assert gone(tick_rejected()) == GONE_PAIR

    # Delivered: the batch carries them one last time, then they stop.
    host.sink = RecordingSink()
    assert gone(host.collect_once().samples) == GONE_PAIR
    assert gone(host.collect_once().samples) == []

    # The successor's first read is its own baseline (the vanished device's
    # counters were forgotten, so its lower counters are not a restart): no
    # GONE and no rating yet; from the second read on it is rated afresh.
    appeared = host.collect_once().samples
    assert gone(appeared) == []
    assert ("disk_io_await_ms", "dm-0") not in rated(appeared)
    successor = host.collect_once().samples
    assert gone(successor) == []
    assert rated(successor)[("disk_io_await_ms", "dm-0")] == 200.0


def test_a_counter_restart_reports_the_old_instance_gone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ticks = iter(NOW + timedelta(seconds=15 * tick) for tick in range(4))
    host = HostTelemetryCollector(
        RecordingSink(),
        context(),
        node_id="worker-1",
        now=lambda: next(ticks),
        edge_filter_enabled=False,
    )
    only_diskstats(monkeypatch, host)
    feed(
        monkeypatch,
        [
            dm0(5),
            dm0(6),
            dm0(1),  # recreated under the same name: every counter restarted
            dm0(2),
        ],
    )
    host.collect_once()
    assert gone(host.collect_once().samples) == []

    restarted = host.collect_once().samples
    assert gone(restarted) == GONE_PAIR
    assert rated(restarted) == {}, "the interval spanning the restart is not rated"

    successor = host.collect_once().samples
    assert gone(successor) == []
    assert rated(successor)[("disk_io_await_ms", "dm-0")] == 200.0, (
        "the successor is rated from the restarted baseline on the next tick"
    )


def test_gone_samples_ride_the_recovered_edge_and_are_not_an_edge_themselves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = RecordingSink()
    ticks = iter(
        NOW + timedelta(seconds=seconds) for seconds in (0, 15, 30, 45, 60, 345)
    )
    host = HostTelemetryCollector(
        sink,
        context(),
        node_id="worker-1",
        now=lambda: next(ticks),
        health_summary_seconds=300,
    )
    only_diskstats(monkeypatch, host)
    feed(
        monkeypatch,
        [
            nvme(1) + dm0(1),
            nvme(2) + dm0(2),  # dm-0 at 200 ms: a threshold edge
            nvme(3),  # torn down: the threshold reason drops out
            nvme(4),
            nvme(5),
            nvme(6),
        ],
    )
    for _ in range(6):
        host.collect_once()

    reasons = [payload["edge_filter_reasons"] for _, payload in sink.requests]
    assert reasons == [
        ["baseline"],
        ["threshold:disk_io_await_ms"],
        ["recovered"],
        ["health-summary"],
    ]
    recovered = [
        HostMetricSample.model_validate(s) for s in sink.requests[2][1]["samples"]
    ]
    assert gone(recovered) == GONE_PAIR
    assert [s.value for s in recovered if device_gone(s)] == [0.0, 0.0]
    assert gone(posted(sink)) == [], "delivered once, the GONE samples are not repeated"
