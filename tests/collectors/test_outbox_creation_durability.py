"""A buffered acknowledgement must survive first-file and directory creation."""

from __future__ import annotations

import errno
import os
from contextlib import nullcontext
from pathlib import Path
from urllib.error import URLError

import pytest

from gpu_fault.collectors import sinks
from gpu_fault.collectors.outbox_file import OutboxFile
from gpu_fault.collectors.sinks import DeliveryStatus, HttpEventSink

EVENT_PATH = "/v1/collector-events/nvidia-kernel"


def sync_paths(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    paths: list[Path] = []
    original = os.fsync

    def sync(handle: int) -> None:
        paths.append(Path(os.readlink(f"/proc/self/fd/{handle}")))
        original(handle)

    monkeypatch.setattr(os, "fsync", sync)
    return paths


def sink_for(path: Path, *, max_records: int = 1000) -> HttpEventSink:
    return HttpEventSink(
        "http://127.0.0.1:1",
        outbox_path=str(path),
        outbox_max_records=max_records,
        outbox_replay_background_interval_seconds=0,
    )


@pytest.mark.parametrize("parent_depth", [0, 1, 3])
@pytest.mark.parametrize("locked", [False, True])
def test_first_append_syncs_the_directory_chain_before_data_and_then_only_data(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, parent_depth: int, locked: bool
) -> None:
    path = tmp_path.joinpath(
        *(f"parent-{index}" for index in range(parent_depth)), "events.ndjson"
    )
    outbox = OutboxFile(path)
    calls = sync_paths(monkeypatch)
    first = {"record_id": "first"}
    second = {"record_id": "second"}

    with outbox.locked() if locked else nullcontext():
        assert outbox.append(first) == 1

    assert calls == [path.parent, *path.parent.parents, path], (
        "the file and every potentially new ancestor must be durable before ACK"
    )
    assert outbox.read() == [first]
    calls.clear()

    # No process-local cache is needed to recognize an already published file.
    another_writer = OutboxFile(path)
    with another_writer.locked() if locked else nullcontext():
        assert another_writer.append(second) == 1

    assert calls == [path], "a normal append must not fsync directories again"
    assert another_writer.read() == [first, second]


def test_an_existing_nonempty_outbox_only_needs_one_file_sync_per_append(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "events.ndjson"
    OutboxFile(path).write([{"record_id": "original"}])
    calls = sync_paths(monkeypatch)
    for index in range(3):
        assert sink_for(path).buffer_for_replay(
            EVENT_PATH, {"record_id": str(index)}
        ), f"append {index} to an existing outbox must be durably buffered"
    assert calls == [path] * 3
    assert OutboxFile(path).read()[0] == {"record_id": "original"}


def test_live_delivery_does_not_report_buffered_when_directory_sync_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "events.ndjson"
    payload = {"record_id": "pending"}
    sink = HttpEventSink("http://127.0.0.1:1", outbox_path=str(path), max_attempts=1)

    def offline(*args, **kwargs):
        raise URLError("simulated offline collector")

    def sync(handle: int) -> None:
        raise OSError(errno.EIO, "simulated directory sync failure")

    monkeypatch.setattr(sinks, "urlopen", offline)
    monkeypatch.setattr(os, "fsync", sync)
    result = sink.deliver(EVENT_PATH, payload)
    assert result.status is DeliveryStatus.FAILED
    assert not result.buffered, (
        "failed directory sync must not produce a buffered delivery result"
    )
    assert result.error is not None
    assert not result.error.buffered, (
        "CollectorError must not claim durable buffering after directory sync failure"
    )
    assert path.stat().st_size == 0
    assert payload == {"record_id": "pending"}


@pytest.mark.parametrize("failure", ["open", "sync"])
@pytest.mark.parametrize("directory_level", [0, 1, 2])
def test_failed_directory_publication_is_not_acknowledged_and_a_new_writer_retries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: str, directory_level: int
) -> None:
    path = tmp_path / "new-parent" / "nested" / "events.ndjson"
    failed_directory = path.parents[directory_level]
    payload = {"record_id": "pending", "message": "retained evidence"}
    original_open, original_sync = os.open, os.fsync

    def open_file(name, flags, *args, **kwargs):
        if failure == "open" and Path(name) == failed_directory:
            raise OSError(errno.EIO, "simulated directory open failure")
        return original_open(name, flags, *args, **kwargs)

    def sync(handle: int) -> None:
        target = Path(os.readlink(f"/proc/self/fd/{handle}"))
        if failure == "sync" and target == failed_directory:
            raise OSError(errno.EIO, "simulated directory sync failure")
        original_sync(handle)

    with monkeypatch.context() as patch:
        patch.setattr(os, "open", open_file)
        patch.setattr(os, "fsync", sync)
        assert not sink_for(path).buffer_for_replay(EVENT_PATH, payload), (
            f"directory {failure} failure at level {directory_level} must not "
            "acknowledge buffering"
        )
    assert path.stat().st_size == 0, (
        "failed publication must remain recognizable without an in-memory flag"
    )
    assert payload == {"record_id": "pending", "message": "retained evidence"}

    calls = sync_paths(monkeypatch)
    assert sink_for(path).buffer_for_replay(EVENT_PATH, payload), (
        "a fresh writer must buffer the event once directory publication succeeds"
    )
    assert calls == [path.parent, *path.parent.parents, path]
    assert [record["payload"] for record in OutboxFile(path).read()] == [payload]


@pytest.mark.parametrize("created_before_failure", [False, True])
def test_failed_parent_creation_leaves_delivery_unacknowledged_and_retryable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, created_before_failure: bool
) -> None:
    path = tmp_path / "new-parent" / "nested" / "events.ndjson"
    payload = {"record_id": "pending"}
    original = Path.mkdir

    def mkdir(directory: Path, *args, **kwargs) -> None:
        if created_before_failure:
            original(directory, *args, **kwargs)
        raise OSError(errno.ENOSPC, "simulated directory creation failure")

    with monkeypatch.context() as patch:
        # Creating just the ancestor simulates mkdir(parents=True) failing midway.
        if created_before_failure:
            path.parent.parent.mkdir()
        patch.setattr(Path, "mkdir", mkdir)
        assert not sink_for(path).buffer_for_replay(EVENT_PATH, payload), (
            "failed parent-directory creation must leave the event unacknowledged"
        )
    assert not path.exists(), (
        "the outbox file must not be created when parent-directory setup fails"
    )
    assert payload == {"record_id": "pending"}

    calls = sync_paths(monkeypatch)
    assert sink_for(path).buffer_for_replay(EVENT_PATH, payload), (
        "buffering must succeed after the parent-directory creation failure is resolved"
    )
    assert calls == [path.parent, *path.parent.parents, path]


@pytest.mark.parametrize("existing", [False, True])
def test_file_sync_failure_preserves_records_but_does_not_acknowledge_them(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, existing: bool
) -> None:
    path = tmp_path / "events.ndjson"
    sink = sink_for(path, max_records=2)
    if existing:
        assert sink.buffer_for_replay(EVENT_PATH, {"record_id": "older"}), (
            "the fixture must durably buffer the older record before failing a later fsync"
        )
    original_sync = os.fsync
    payload = {"record_id": "not-acknowledged"}

    def sync(handle: int) -> None:
        if Path(os.readlink(f"/proc/self/fd/{handle}")) == path:
            raise OSError(errno.EIO, "simulated file sync failure")
        original_sync(handle)

    with monkeypatch.context() as patch:
        patch.setattr(os, "fsync", sync)
        assert not sink.buffer_for_replay(EVENT_PATH, payload), (
            "a failed outbox file fsync must not acknowledge buffering"
        )
    assert OutboxFile(path).read()[-1]["payload"] == payload
    assert sink.buffer_for_replay(EVENT_PATH, {"record_id": "next"}), (
        "the next event must buffer successfully after file fsync is restored"
    )
    records = OutboxFile(path).read()
    assert records[-1]["payload"] == {"record_id": "next"}
    if existing:
        assert len(records) == 1, "a failed append must be included in the next recount"
        assert sink.outbox_stats()["evictions_total"] == 2
    else:
        assert [record["payload"] for record in records] == [
            payload,
            {"record_id": "next"},
        ]
