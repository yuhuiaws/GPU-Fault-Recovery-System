"""Replay diagnostics over fake clocks/signals and private process files."""

from __future__ import annotations

import io
import json
import signal
from types import SimpleNamespace

import pytest

from gpu_fault import processor_diagnostics as diagnostics
from tests.collectors import _cov95_runtime_collect as support

isolated_runtime = support.isolated_runtime


def test_replay_tracker_reports_phase_age_and_releases_context_binding(monkeypatch):
    clock = support.Clock()
    thread = SimpleNamespace(name="private-ingress")
    monkeypatch.setattr(diagnostics, "time", SimpleNamespace(monotonic=clock.monotonic))
    monkeypatch.setattr(diagnostics, "current_thread", lambda: thread)
    monkeypatch.setattr(diagnostics, "get_native_id", lambda: 7)
    tracker = diagnostics.ProcessorReplayTracker()
    tracker.update("missing", "ignored")
    tracker.start(
        "first", owner_id="owner-a", path="/events", lane_epoch=1, phase="queued"
    )
    clock.sleep(2)
    tracker.start(
        "second", owner_id="owner-a", path="/events", lane_epoch=2, phase="queued"
    )
    clock.sleep(1)
    token = diagnostics.bind_processor_replay(tracker, "first")
    try:
        thread.name = "private-worker"
        diagnostics.report_processor_replay_phase("persisting")
    finally:
        diagnostics.reset_processor_replay(token)
    diagnostics.report_processor_replay_phase("outside-context")

    assert tracker.phases() == {
        "persisting": {"count": 1, "oldest_seconds": 0},
        "queued": {"count": 1, "oldest_seconds": 1},
    }
    first = tracker.snapshot()[0]
    assert (first["thread_name"], first["native_thread_id"]) == ("private-worker", 7)
    assert first["elapsed_seconds"] == 3
    tracker.finish("first")
    tracker.finish("second")
    tracker.finish("missing")
    assert tracker.snapshot() == []
    assert tracker.phases() == {}


@pytest.mark.parametrize(
    "options", [{"interval_seconds": 0}, {"interval_seconds": 2, "stale_seconds": 3}]
)
def test_diagnostic_publisher_rejects_invalid_liveness_windows(tmp_path, options):
    directory = tmp_path / "publisher"
    with pytest.raises(ValueError, match="processor diagnostics"):
        diagnostics.ProcessorDiagnosticsPublisher(
            str(directory), support.forbidden, **options
        )
    assert not directory.exists(), (
        "invalid bounds must fail before creating runtime files"
    )


@pytest.mark.parametrize("already_stopped", [False, True])
def test_publisher_retries_failed_snapshot_and_removes_only_its_process_file(
    monkeypatch, tmp_path, already_stopped, caplog
):
    calls = []
    sibling = tmp_path / "other.json"
    sibling.write_text("{}")

    def snapshot():
        calls.append("snapshot")
        if len(calls) == 1:
            raise OSError("fake snapshot unavailable")
        return {"process": {"pid": 7}, "in_flight_requests": []}

    publisher = diagnostics.ProcessorDiagnosticsPublisher(str(tmp_path), snapshot)
    waits = []

    def wait(seconds):
        waits.append(seconds)
        if len(waits) == 2:
            assert json.loads(publisher.path.read_text())["process"]["pid"] == 7
            return True
        return False

    stop = SimpleNamespace(is_set=lambda: already_stopped, wait=wait)
    publisher.run(stop)
    assert not publisher.path.exists(), "publisher must remove its own final snapshot"
    assert sibling.read_text() == "{}"
    assert waits == ([] if already_stopped else [1, 1])
    if not already_stopped:
        assert "fake snapshot unavailable" in caplog.text


def test_publisher_cleanup_failure_is_not_reported_as_success(monkeypatch, tmp_path):
    publisher = diagnostics.ProcessorDiagnosticsPublisher(str(tmp_path), lambda: {})
    publisher.path.write_text("{}")
    original = type(publisher.path).unlink

    def fail(path, *args, **kwargs):
        if path == publisher.path:
            raise PermissionError("fake cleanup denied")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(type(publisher.path), "unlink", fail)
    with pytest.raises(PermissionError, match="cleanup denied"):
        publisher.run(SimpleNamespace(is_set=lambda: True))
    assert publisher.path.exists(), (
        "failed cleanup must retain evidence of the residual file"
    )


def test_snapshot_aggregation_filters_invalid_stale_future_and_unreadable_files(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(diagnostics, "time", SimpleNamespace(time=lambda: 100))
    payloads = [
        {"published_at": 99, "process": {"pid": 9}},
        {"published_at": 98, "process": {"pid": 2}},
        {"published_at": 94, "process": {"pid": 1}},
        {"published_at": 101, "process": {"pid": 3}},
        {"published_at": "bad", "process": {"pid": 4}},
        {"published_at": 99, "process": {"pid": "bad"}},
        {"published_at": 99},
        [],
    ]
    for index, payload in enumerate(payloads):
        (tmp_path / f"{index}.json").write_text(json.dumps(payload))
    (tmp_path / "invalid.json").write_text("{")
    unreadable = tmp_path / "unreadable.json"
    unreadable.write_text("{}")
    original = type(unreadable).read_text

    def read(path, *args, **kwargs):
        if path == unreadable:
            raise OSError("fake read denied")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(type(unreadable), "read_text", read)
    publisher = diagnostics.ProcessorDiagnosticsPublisher(str(tmp_path), lambda: {})
    assert [
        (item["process"]["pid"], item["snapshot_age_seconds"])
        for item in publisher.read_all()
    ] == [(2, 2), (9, 1)]


def test_snapshot_listing_failure_yields_no_claimed_worker_observation(
    monkeypatch, tmp_path
):
    publisher = diagnostics.ProcessorDiagnosticsPublisher(str(tmp_path), lambda: {})

    def fail(*args, **kwargs):
        raise OSError("fake listing unavailable")

    monkeypatch.setattr(type(tmp_path), "glob", fail)
    assert publisher.read_all() == []


@pytest.mark.parametrize("unreadable", [False, True])
def test_process_memory_diagnostics_read_only_fake_proc_files(monkeypatch, unreadable):
    calls = []

    def open_memory(path, **kwargs):
        calls.append(path)
        if unreadable:
            raise OSError("fake proc unavailable")
        return io.StringIO(
            "invalid line\nUnknown: not exposed\nVmRSS: 8 kB\nThreads: 3\n"
            "Swap: unavailable\nPrivate_Clean: 4 KB\nAnonymous:\n"
        )

    monkeypatch.setattr(diagnostics, "open", open_memory, raising=False)
    monkeypatch.setattr(diagnostics, "sys", SimpleNamespace())
    result = diagnostics.process_runtime_snapshot()
    assert calls == ["/proc/self/status", "/proc/self/smaps_rollup"]
    assert "allocated_blocks" not in result
    assert "unknown" not in result
    if unreadable:
        assert "vm_rss_bytes" not in result
    else:
        assert result["vm_rss_bytes"] == 8192
        assert result["threads"] == 3
        assert result["swap"] == "unavailable"
        assert result["smaps_private_clean_bytes"] == 4096
        assert result["anonymous"] == ""


@pytest.mark.parametrize(
    "name,expected",
    [("", None), (" sigusr1 ", signal.SIGUSR1), ("SIGUSR2", signal.SIGUSR2)],
)
def test_thread_dump_registration_uses_only_fake_faulthandler(
    monkeypatch, name, expected
):
    calls = []
    monkeypatch.setattr(
        diagnostics,
        "faulthandler",
        SimpleNamespace(
            register=lambda signum, **kwargs: calls.append(
                ("register", signum, kwargs)
            ),
            unregister=lambda signum: calls.append(("unregister", signum)),
        ),
    )
    signum = diagnostics.register_thread_dump_signal(name)
    diagnostics.unregister_thread_dump_signal(signum)
    assert signum == expected
    assert calls == (
        []
        if expected is None
        else [
            ("register", int(expected), {"all_threads": True, "chain": False}),
            ("unregister", int(expected)),
        ]
    )
