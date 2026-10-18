"""Outbox paths where there is nothing to do, and one where the depth lies.

A buffered failure with the retry schedule disabled arms no worker; an inline
replay after a delivered post with an empty or all-dead outbox rewrites
nothing; and a compaction that finds fewer records on disk than the cached
depth promised (the file was trimmed by hand) evicts nothing and says so by
staying quiet.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

import pytest

from gpu_fault.collectors.sinks import CollectorError, HttpEventSink

WORKER_NAME = "gpu-fault-collector-outbox-replay"


class _Response:
    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return b'{"accepted":true}'


class _ControlPlane:
    def __init__(self, *, down: bool) -> None:
        self.down = down
        self.posted: list[dict[str, Any]] = []

    def __call__(self, request: Any, **_kwargs: Any) -> _Response:
        if self.down:
            raise OSError("network unavailable")
        self.posted.append(json.loads(request.data))
        return _Response()


def _sink(outbox: Path, **overrides: Any) -> HttpEventSink:
    settings: dict[str, Any] = {
        "max_attempts": 1,
        "sleep": lambda _seconds: None,
        "outbox_path": str(outbox),
        "outbox_replay_retry_max_seconds": 0,
    }
    settings.update(overrides)
    return HttpEventSink("https://control", **settings)


def _records(outbox: Path) -> list[dict[str, Any]]:
    if not outbox.exists():
        return []
    return [json.loads(line) for line in outbox.read_text().splitlines() if line]


def _workers() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name == WORKER_NAME and t.is_alive()]


def test_a_buffered_failure_arms_no_worker_when_the_retry_schedule_is_disabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("gpu_fault.collectors.sinks.urlopen", _ControlPlane(down=True))
    outbox = tmp_path / "events.ndjson"
    sink = _sink(outbox)

    with pytest.raises(CollectorError) as failed:
        sink.post("/events", {"event_id": "e-1"})

    assert failed.value.buffered and failed.value.replayable, (
        "a transient failure with an outbox has to be buffered and replayable"
    )
    assert [record["payload"]["event_id"] for record in _records(outbox)] == ["e-1"]
    assert sink.wait_for_outbox_replay(0) is True, "a replay worker was started"
    assert _workers() == []


def test_a_delivered_post_with_an_empty_outbox_rewrites_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plane = _ControlPlane(down=False)
    monkeypatch.setattr("gpu_fault.collectors.sinks.urlopen", plane)
    outbox = tmp_path / "events.ndjson"
    outbox.write_text("", encoding="utf-8")
    sink = _sink(outbox)

    sink.post("/events", {"event_id": "live"})

    assert sink.wait_for_outbox_replay(2) is True
    assert [item["event_id"] for item in plane.posted] == ["live"]
    assert outbox.read_text() == "", "an empty outbox was rewritten"
    assert sink.outbox_stats()["depth"] == 0


def test_a_delivered_post_leaves_an_outbox_of_dead_records_untouched(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plane = _ControlPlane(down=False)
    monkeypatch.setattr("gpu_fault.collectors.sinks.urlopen", plane)
    outbox = tmp_path / "events.ndjson"
    dead = {
        "path": "/events",
        "payload": {"event_id": "dead"},
        "replayable": False,
        "error": "HTTP 422",
        "failed_at": "2026-08-30T00:00:00+00:00",
    }
    outbox.write_text(json.dumps(dead) + "\n", encoding="utf-8")
    sink = _sink(outbox)

    sink.post("/events", {"event_id": "live"})

    assert sink.wait_for_outbox_replay(2) is True
    assert [item["event_id"] for item in plane.posted] == ["live"], (
        "a dead-lettered record was posted again"
    )
    assert _records(outbox) == [dead], "the dead record was rewritten or dropped"
    stats = sink.outbox_stats()
    assert (stats["depth"], stats["replayable"]) == (1, 0)


def test_compaction_evicts_nothing_when_the_file_is_shorter_than_the_cached_depth(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr("gpu_fault.collectors.sinks.urlopen", _ControlPlane(down=True))
    outbox = tmp_path / "events.ndjson"
    sink = _sink(outbox, outbox_max_records=10)

    def buffer(event_id: str) -> None:
        with pytest.raises(CollectorError):
            sink.post("/events", {"event_id": event_id})

    with caplog.at_level(logging.WARNING, logger="gpu_fault.collectors.sinks"):
        for index in range(11):
            buffer(f"e-{index}")
    assert sink.outbox_evictions_total == 2
    assert len(_records(outbox)) == 9
    assert sum("evicted" in record.message for record in caplog.records) == 1

    # An operator trims the file by hand; the sink's cached depth still says 9.
    outbox.write_text("", encoding="utf-8")
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="gpu_fault.collectors.sinks"):
        buffer("e-11")
        buffer("e-12")

    assert [record["payload"]["event_id"] for record in _records(outbox)] == [
        "e-11",
        "e-12",
    ], "a compaction over a trimmed file dropped records that were under the floor"
    assert sink.outbox_evictions_total == 2, "nothing was evicted, yet it was counted"
    assert not any("evicted" in record.message for record in caplog.records), (
        "an eviction warning fired for a compaction that evicted nothing"
    )
    assert sink.outbox_stats()["depth"] == 2


def test_a_second_buffered_failure_while_the_worker_is_live_is_drained_by_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The worker already owns the outbox, so the second failure only marks it
    pending: no second worker, and the record it buffered goes out when the
    first worker's schedule finds the control plane back."""

    plane = _ControlPlane(down=True)
    monkeypatch.setattr("gpu_fault.collectors.sinks.urlopen", plane)
    outbox = tmp_path / "events.ndjson"
    sink = _sink(
        outbox,
        outbox_replay_retry_initial_seconds=0.02,
        outbox_replay_retry_max_seconds=0.08,
        outbox_replay_background_interval_seconds=0,
        jitter=lambda _low, high: high,
    )

    for event_id in ("e-1", "e-2"):
        with pytest.raises(CollectorError):
            sink.post("/events", {"event_id": event_id})

    assert sink.wait_for_outbox_replay(0) is False, "no worker owns the backlog"
    assert len(_workers()) == 1, "a second buffered failure started a second worker"
    plane.down = False

    assert sink.wait_for_outbox_replay(5) is True, "the backlog did not drain"
    assert sorted(item["event_id"] for item in plane.posted) == ["e-1", "e-2"]
    assert _records(outbox) == []
