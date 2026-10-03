"""Outbox failures preserve private data and expose degraded durability."""

from __future__ import annotations

import errno
import os
from types import SimpleNamespace

import pytest

from gpu_fault.collectors import outbox_file
from tests.collectors import _cov95_runtime_collect as support

isolated_runtime = support.isolated_runtime


@pytest.mark.parametrize("failure", ["open-directory", "sync-directory"])
def test_outbox_directory_sync_failure_reports_degradation_after_atomic_write(
    monkeypatch, tmp_path, failure, caplog
):
    outbox = outbox_file.OutboxFile(tmp_path / "events.ndjson")
    original_open, original_sync = os.open, os.fsync
    directory_handles = []

    def open_file(path, *args, **kwargs):
        if path == outbox.path.parent and failure == "open-directory":
            raise OSError("private directory open denied")
        handle = original_open(path, *args, **kwargs)
        if path == outbox.path.parent:
            directory_handles.append(handle)
        return handle

    def sync(handle):
        if handle in directory_handles and failure == "sync-directory":
            raise OSError("private directory sync denied")
        return original_sync(handle)

    # outbox_file.os is the global os module, which pytest's own tmp_path
    # teardown also uses; keep the fakes to the call under test.
    with monkeypatch.context() as patch:
        patch.setattr(outbox_file.os, "open", open_file)
        patch.setattr(outbox_file.os, "fsync", sync)
        outbox.write([{"event_id": "private-event"}])
    assert outbox.read() == [{"event_id": "private-event"}]
    assert "directory" in caplog.text
    assert "denied" in caplog.text


def test_failed_atomic_replace_preserves_old_outbox_and_removes_its_temporary(
    monkeypatch, tmp_path
):
    outbox = outbox_file.OutboxFile(tmp_path / "events.ndjson")
    outbox.write([{"event_id": "original"}])

    def fail(*args):
        raise OSError("private replace failed")

    monkeypatch.setattr(outbox_file.os, "replace", fail)
    with pytest.raises(OSError, match="replace failed"):
        outbox.write([{"event_id": "replacement"}])
    assert outbox.read() == [{"event_id": "original"}]
    assert list(tmp_path.glob("*.tmp")) == []


@pytest.mark.parametrize("required", [False, True])
def test_unavailable_lock_is_strict_for_maintenance_and_visible_for_collector(
    monkeypatch, tmp_path, required, caplog
):
    outbox = outbox_file.OutboxFile(tmp_path / "events.ndjson")
    entered = []

    def unsupported(*args):
        raise OSError(errno.ENOSYS, "private filesystem has no flock")

    monkeypatch.setattr(outbox_file.fcntl, "flock", unsupported)
    if required:
        with pytest.raises(outbox_file.OutboxLockUnavailable, match="cannot take"):
            with outbox.locked(required=True):
                entered.append(True)
        assert entered == []
        assert not outbox.path.exists(), "refused maintenance must not create an outbox"
    else:
        with outbox.locked():
            entered.append(True)
            outbox.append({"event_id": "durable-but-unlocked"})
        assert entered == [True]
        assert outbox.stats()["unlocked_writes_total"] == 1
        assert "on the in-process lock" in caplog.text


def test_forced_local_lock_fallback_has_a_bounded_poll_and_explicit_warning(
    monkeypatch, tmp_path, caplog
):
    outbox = outbox_file.OutboxFile(tmp_path / "events.ndjson")
    waits = []

    def held(handle, mode):
        raise BlockingIOError(errno.EAGAIN, "private lock held")

    monkeypatch.setattr(outbox_file.fcntl, "flock", held)
    monkeypatch.setattr(outbox_file, "OUTBOX_LOCK_ATTEMPTS", 2)
    monkeypatch.setattr(outbox_file, "time", SimpleNamespace(sleep=waits.append))
    with outbox.locked(required=True, forced=True):
        outbox.append({"event_id": "explicit-local-force"})
    assert waits == [outbox_file.OUTBOX_LOCK_RETRY_SECONDS]
    assert outbox.stats()["unlocked_writes_total"] == 1
    assert "because --force was passed" in caplog.text
