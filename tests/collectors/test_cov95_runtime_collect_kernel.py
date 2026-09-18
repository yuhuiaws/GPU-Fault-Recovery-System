"""Kernel stream lifecycle, identity and shutdown accounting with fake I/O."""

from __future__ import annotations

import io
import signal
from datetime import timedelta
from types import SimpleNamespace

import pytest

from gpu_fault.channel_registry import COLLECTOR_HEALTH_PATH, NVIDIA_KERNEL_PATH
from gpu_fault.collectors.logs import kernel
from gpu_fault.collectors.sinks import CollectorError
from tests.collectors import _cov95_runtime_collect as common
from tests.collectors import _cov95_runtime_collect_kernel as support

isolated_runtime = common.isolated_runtime
kernel_case = support.kernel_case


@pytest.mark.parametrize(
    "error", [PermissionError("denied"), FileNotFoundError("absent")]
)
def test_default_device_open_failure_still_stops_delivery_and_restores_signals(
    kernel_case, monkeypatch, error
):
    calls = []

    def open_device(path, *args, **kwargs):
        calls.append(path)
        raise error

    monkeypatch.setattr(kernel, "open", open_device, raising=False)
    with pytest.raises(CollectorError, match="CAP_SYSLOG|device not found"):
        kernel_case.build().run()
    assert calls == ["/dev/kmsg"]
    assert kernel_case.sink.requests == []
    assert all(thread.joined for thread in kernel_case.threads), (
        "reader startup failure must join the delivery lifecycle"
    )
    assert kernel_case.handlers == {
        signal.SIGTERM: signal.SIG_DFL,
        signal.SIGINT: signal.SIG_DFL,
    }


@pytest.mark.parametrize("start_at_end", [False, True])
def test_string_stream_eof_reopens_then_drains_selected_records(
    kernel_case, monkeypatch, start_at_end, caplog
):
    monkeypatch.setattr(
        kernel,
        "open",
        lambda *args, **kwargs: io.StringIO(support.line()),
        raising=False,
    )
    collector = kernel_case.build(start_at_end=start_at_end)
    with pytest.raises(common.StopLoop):
        collector.run()
    records = [
        payload
        for path, payload in kernel_case.sink.requests
        if path == NVIDIA_KERNEL_PATH
    ]
    assert [record["record_id"] for record in records] == (
        [] if start_at_end else ["kmsg-private-boot-1"]
    ), "tailing must exclude history while replay mode drains the selected XID"
    summaries = [
        payload
        for path, payload in kernel_case.sink.requests
        if path == COLLECTOR_HEALTH_PATH
    ]
    assert len(summaries) == 1, "the first-round health report must not be discarded"
    assert summaries[0]["collector"] == "NVIDIA_KERNEL", (
        "the summary must identify the kernel collector"
    )
    assert summaries[0]["edge_filter_reasons"] == ["health-summary"], (
        "a healthy EOF transition must not invent hardware faults or delivery losses"
    )
    assert "kernel message stream ended" in caplog.text, (
        "EOF must remain observable before the reader reopens"
    )
    assert kernel_case.clock.sleeps == [1], "EOF must use the configured reopen delay"
    assert collector.health_counters["delivery_dropped_at_shutdown"] == 0, (
        "shutdown must drain both fault records and health reports"
    )


def test_failed_tail_seek_refuses_replaying_historical_ring(
    kernel_case, monkeypatch, caplog
):
    class Unseekable(io.StringIO):
        def seek(self, *args):
            raise OSError("fake device cannot seek")

    monkeypatch.setattr(
        kernel,
        "open",
        lambda *args, **kwargs: Unseekable(support.line()),
        raising=False,
    )
    with pytest.raises(common.StopLoop):
        kernel_case.build().run()
    assert kernel_case.sink.requests == []
    assert "refusing to replay the ring buffer" in caplog.text


@pytest.mark.parametrize("prior", ["callable", "ignored", "none", "worker"])
def test_signal_registration_respects_previous_owner_and_drains_on_exit(
    kernel_case, monkeypatch, prior
):
    forwarded = []
    if prior == "callable":
        kernel_case.handlers[signal.SIGTERM] = lambda number, frame: forwarded.append(
            number
        )
    elif prior == "ignored":
        kernel_case.handlers[signal.SIGTERM] = signal.SIG_IGN
    elif prior == "none":
        kernel_case.handlers[signal.SIGTERM] = None
    else:
        kernel_case.main = False
    original = dict(kernel_case.handlers)

    def opened(*args, **kwargs):
        if prior not in {"ignored", "worker"}:
            kernel_case.handlers[signal.SIGTERM](signal.SIGTERM, None)
        return io.StringIO("")

    monkeypatch.setattr(kernel, "open", opened, raising=False)
    if prior in {"ignored", "worker"}:
        with pytest.raises(common.StopLoop):
            kernel_case.build().run()
    else:
        kernel_case.build().run()
    if prior == "none":
        original[signal.SIGTERM] = signal.SIG_DFL
    assert kernel_case.handlers == original
    assert forwarded == ([signal.SIGTERM] if prior == "callable" else [])
    if prior == "worker":
        assert kernel_case.signal_calls == []


@pytest.mark.parametrize("when", ["install", "restore"])
def test_signal_boundary_failure_cannot_prevent_reader_cleanup(
    kernel_case, monkeypatch, when
):
    calls = []
    original = kernel.signal.signal

    def signal_boundary(number, handler):
        calls.append((number, handler))
        if (when == "install" and callable(handler)) or (
            when == "restore" and not callable(handler)
        ):
            raise ValueError("fake signal boundary unavailable")
        return original(number, handler)

    monkeypatch.setattr(kernel.signal, "signal", signal_boundary)
    monkeypatch.setattr(
        kernel, "open", lambda *args, **kwargs: io.StringIO(""), raising=False
    )
    with pytest.raises(common.StopLoop):
        kernel_case.build().run()
    assert kernel_case.threads[0].joined, "signal failure must not bypass queue cleanup"
    assert len(calls) == (2 if when == "install" else 4)


def test_health_summary_never_evicts_fault_when_delivery_queue_is_full(
    kernel_case, monkeypatch
):
    def opened(*args, **kwargs):
        kernel_case.clock.sleep(301)
        return io.StringIO(support.line())

    monkeypatch.setattr(kernel, "open", opened, raising=False)
    collector = kernel_case.build(start_at_end=False, delivery_queue_size=1)
    with pytest.raises(common.StopLoop):
        collector.run()
    assert collector.health_counters["health_summary_queue_drops"] == 1
    assert collector.health_counters["delivery_queue_drops"] == 0
    assert len(kernel_case.sink.requests) == 1
    assert kernel_case.sink.requests[0][1]["record_id"] == "kmsg-private-boot-1"


def test_repeated_delivery_thread_death_falls_back_inline_and_accounts_for_deaths(
    kernel_case, caplog
):
    kernel_case.start_alive = False
    collector = kernel_case.build()
    collector.start_delivery()
    stats = collector.collect_lines([support.line(index) for index in range(1, 4)])
    collector.stop_delivery()
    assert stats.delivered == 3
    assert collector.health_counters["delivery_thread_deaths"] == 4
    assert len(kernel_case.sink.requests) == 3
    assert "delivering inline from now on" in caplog.text


def test_exhausted_thread_restart_budget_drains_already_queued_records(
    kernel_case, monkeypatch
):
    collector = kernel_case.build()
    collector.start_delivery()
    assert collector.collect_lines([support.line(1)]).delivered == 0
    kernel_case.threads[0].alive = False
    monkeypatch.setattr(kernel, "DELIVERY_THREAD_MAX_RESTARTS", 0)
    assert collector.collect_lines([support.line(2)]).delivered == 1
    collector.stop_delivery()
    assert [payload["record_id"] for _, payload in kernel_case.sink.requests] == [
        "kmsg-private-boot-1",
        "kmsg-private-boot-2",
    ]


def test_restarted_delivery_retains_stop_on_previous_unfinished_thread(
    kernel_case, caplog
):
    kernel_case.survive_join = True
    collector = kernel_case.build()
    collector.start_delivery()
    collector.start_delivery()
    assert len(kernel_case.threads) == 1
    collector.stop_delivery()
    previous = kernel_case.threads[0]
    collector.start_delivery()
    assert len(kernel_case.threads) == 2
    assert previous.args[0].is_set(), "retired worker must never consume new records"
    collector.stop_delivery()
    assert "earlier kernel delivery thread(s) are still finishing" in caplog.text


@pytest.mark.parametrize("buffer_error", [False, True])
def test_shutdown_uses_one_live_attempt_when_private_outbox_cannot_accept(
    kernel_case, buffer_error
):
    attempts = []

    def buffer(path, payload):
        if buffer_error:
            raise OSError("fake outbox full")
        return False

    sink = SimpleNamespace(
        post=common.forbidden,
        buffer_for_replay=buffer,
        deliver_once=lambda path, payload: attempts.append(payload["record_id"]),
    )
    collector = kernel_case.build()
    collector.sink = sink
    collector.start_delivery()
    collector.collect_lines([support.line(1), support.line(2)])
    collector.stop_delivery()
    assert attempts == ["kmsg-private-boot-1", "kmsg-private-boot-2"]
    assert collector.health_counters["delivery_failures"] == 0


@pytest.mark.parametrize("uptime", ["", "invalid", None])
def test_unreadable_uptime_uses_collection_time_without_claiming_clock_correction(
    kernel_case, uptime
):
    if uptime is None:
        kernel_case.uptime.unlink()
    else:
        kernel_case.uptime.write_text(uptime)
    collector = kernel_case.build()
    collector.refresh_boot_time()
    assert collector.collect_lines([support.line()]).delivered == 1
    payload = kernel_case.sink.requests[0][1]
    assert payload["observed_at"] == common.NOW.isoformat()
    assert collector.health_counters["boot_time_reestimates"] == 0


@pytest.mark.parametrize("second_uptime", ["160", "bad"])
def test_old_record_does_not_create_a_clock_step_when_reestimate_is_unchanged_or_unknown(
    kernel_case, second_uptime
):
    collector = kernel_case.build()
    collector.refresh_boot_time()
    kernel_case.clock.sleep(60)
    kernel_case.uptime.write_text(second_uptime)
    collector.collect_lines([support.line()])
    payload = kernel_case.sink.requests[0][1]
    assert payload["observed_at"] == (common.NOW - timedelta(seconds=99)).isoformat()
    assert collector.health_counters["boot_time_reestimates"] == 0


def test_overflowed_monotonic_stamp_and_bounded_dedup_are_observable(kernel_case):
    collector = kernel_case.build(deduplication_window=1)
    collector.refresh_boot_time()
    stats = collector.collect_lines(
        [
            support.line(1, 10**40),
            support.line(2),
            support.line(1, 10**40),
            support.line(3),
        ],
        limit=3,
    )
    assert (stats.observed, stats.delivered, stats.duplicates) == (3, 3, 0)
    assert kernel_case.sink.requests[0][1]["observed_at"] == common.NOW.isoformat()


@pytest.mark.parametrize(
    "options", [{"reopen_delay_seconds": 0}, {"delivery_queue_size": 0}]
)
def test_reader_rejects_disabled_safety_bounds(kernel_case, options):
    with pytest.raises(ValueError, match="must be"):
        kernel_case.build(**options)
