"""Time-based outbox replay retry (GF-REGIONAL-NET-001, 2026-10-01).

Replay used to be woken only by a delivered live post, and a worker that
made no progress stopped. A channel that posts nothing live for minutes --
the fabric-manager receipt and the host summary run every 300 s -- kept a
backlog buffered for up to its full cadence after connectivity returned, so
the acceptance case's 300 s convergence budget raced the collector's cadence
and failed by construction. The sink now retries a stalled backlog on a
bounded, jittered backoff that a delivered live post can still cut short.
"""

from __future__ import annotations

import io
import json
import threading
from email.message import Message
from pathlib import Path
from threading import Event
from typing import Any
from urllib.error import HTTPError

import pytest

from gpu_fault.collectors import sinks as collector_sinks
from gpu_fault.collectors.sinks import CollectorError, HttpEventSink

WORKER_NAME = "gpu-fault-collector-outbox-replay"


class _Response:
    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return b'{"accepted":true}'


def _http_error(url: str, status: int) -> HTTPError:
    return HTTPError(url, status, "error", Message(), io.BytesIO(b"detail"))


def _seed(path: Path, sequences: list[int]) -> None:
    path.write_text(
        "".join(
            json.dumps(
                {
                    "path": "/events",
                    "payload": {"sequence": sequence, "event_id": f"e-{sequence}"},
                    "replayable": True,
                    "error": "seeded",
                    "failed_at": "2026-08-30T00:00:00+00:00",
                }
            )
            + "\n"
            for sequence in sequences
        ),
        encoding="utf-8",
    )


def _remaining(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _worker_threads() -> list[threading.Thread]:
    """Replay workers still alive once given a moment to leave their last batch."""

    workers = [t for t in threading.enumerate() if t.name == WORKER_NAME]
    for worker in workers:
        worker.join(1)
    return [worker for worker in workers if worker.is_alive()]


def _sink(outbox: Path, **overrides: Any) -> HttpEventSink:
    settings: dict[str, Any] = {
        "max_attempts": 1,
        "sleep": lambda _seconds: None,
        "outbox_path": str(outbox),
        "outbox_replay_batch_size": 2,
        "outbox_replay_background_interval_seconds": 0,
        "outbox_replay_retry_initial_seconds": 0.02,
        "outbox_replay_retry_max_seconds": 0.08,
        # Deterministic: the worker waits exactly the schedule's upper bound.
        "jitter": lambda _low, high: high,
    }
    settings.update(overrides)
    return HttpEventSink("https://control", **settings)


class _ControlPlane:
    """``urlopen`` stand-in whose reachability a test flips without a sleep."""

    def __init__(self) -> None:
        self.down = True
        self.delivered: list[object] = []
        self.attempts: list[object] = []
        self.first_failure = Event()
        self.failures_before_recovery: int | None = None

    def __call__(self, request: Any, **_kwargs: Any) -> _Response:
        sequence = json.loads(request.data)["sequence"]
        self.attempts.append(sequence)
        if self.down:
            self.first_failure.set()
            failures = len(self.attempts) - len(self.delivered)
            if (
                self.failures_before_recovery is not None
                and failures >= self.failures_before_recovery
            ):
                self.down = False
            raise OSError("network unavailable")
        self.delivered.append(sequence)
        return _Response()


@pytest.fixture
def control_plane(monkeypatch: pytest.MonkeyPatch) -> _ControlPlane:
    plane = _ControlPlane()
    monkeypatch.setattr("gpu_fault.collectors.sinks.urlopen", plane)
    return plane


def test_backoff_retry_drains_the_backlog_after_connectivity_returns_without_a_post(
    control_plane: _ControlPlane, tmp_path: Path
) -> None:
    outbox = tmp_path / "fabric-manager.ndjson"
    _seed(outbox, [0, 1, 2])
    sink = _sink(outbox)

    with pytest.raises(CollectorError) as failed:
        sink.post("/events", {"sequence": "live", "event_id": "live"})
    assert failed.value.buffered and failed.value.replayable

    assert control_plane.first_failure.wait(2), "the live post never reached urlopen"
    assert not sink.wait_for_outbox_replay(0.2), (
        "a worker with an undeliverable backlog reported itself idle"
    )
    control_plane.down = False

    assert sink.wait_for_outbox_replay(5), "the backlog did not drain on the schedule"
    assert control_plane.delivered == [0, 1, 2, "live"], control_plane.delivered
    assert _remaining(outbox) == [], "the drained outbox still holds records"
    assert _worker_threads() == [], "the worker outlived its empty outbox"


def test_backoff_doubles_from_its_start_and_caps_at_its_maximum(
    control_plane: _ControlPlane, tmp_path: Path
) -> None:
    waits: list[tuple[float, float]] = []

    def recording_jitter(low: float, high: float) -> float:
        waits.append((low, high))
        return high

    outbox = tmp_path / "host.ndjson"
    _seed(outbox, [0])
    # The live post fails once; six one-record batches then fail before the
    # control plane answers again.
    control_plane.failures_before_recovery = 7
    sink = _sink(
        outbox,
        outbox_replay_batch_size=1,
        outbox_replay_retry_initial_seconds=0.01,
        outbox_replay_retry_max_seconds=0.04,
        jitter=recording_jitter,
    )

    with pytest.raises(CollectorError):
        sink.post("/events", {"sequence": "live", "event_id": "live"})

    assert sink.wait_for_outbox_replay(5), "the backlog did not drain on the schedule"
    assert control_plane.delivered == [0, "live"]
    # One wait when the failed post arms the worker, then one after each of
    # the six zero-progress batches: doubling from the start, pinned at the cap.
    assert [high for _low, high in waits] == [
        0.01,
        0.02,
        0.04,
        0.04,
        0.04,
        0.04,
        0.04,
    ], waits
    assert all(low == pytest.approx(high / 2) for low, high in waits), (
        "the jitter window is not the upper half of the interval"
    )


def test_a_record_dead_lettered_at_replay_is_not_retried(
    control_plane: _ControlPlane, tmp_path: Path
) -> None:
    rejected = Event()
    reachable = control_plane

    def verdict_then_reachable(request: Any, **kwargs: Any) -> _Response:
        if json.loads(request.data)["sequence"] == "poisoned":
            rejected.set()
            raise _http_error(request.full_url, 422)
        return reachable(request, **kwargs)

    outbox = tmp_path / "kernel.ndjson"
    outbox.write_text(
        json.dumps(
            {
                "path": "/events",
                "payload": {"sequence": "poisoned", "event_id": "poisoned"},
                "replayable": True,
                "error": "seeded",
                "failed_at": "2026-08-30T00:00:00+00:00",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    sink = _sink(outbox)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("gpu_fault.collectors.sinks.urlopen", verdict_then_reachable)
        with pytest.raises(CollectorError):
            sink.post("/events", {"sequence": "live", "event_id": "live"})
        control_plane.down = False
        assert sink.wait_for_outbox_replay(5), "the worker did not settle"

    assert control_plane.delivered == ["live"]
    remaining = _remaining(outbox)
    assert [item["payload"]["sequence"] for item in remaining] == ["poisoned"]
    assert remaining[0]["replayable"] is False, "the 422 verdict stayed replayable"
    assert control_plane.attempts.count("poisoned") == 0 and rejected.is_set(), (
        "the dead-lettered record was posted again"
    )
    assert _worker_threads() == [], "a worker kept running for a dead letter"


def test_a_live_dead_letter_does_not_arm_the_retry_worker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def refuse(request: Any, **_kwargs: Any) -> _Response:
        raise _http_error(request.full_url, 400)

    monkeypatch.setattr("gpu_fault.collectors.sinks.urlopen", refuse)
    outbox = tmp_path / "dcgm.ndjson"
    sink = _sink(outbox)

    with pytest.raises(CollectorError) as failed:
        sink.post("/events", {"sequence": "bad", "event_id": "bad"})

    assert failed.value.buffered and not failed.value.replayable
    assert sink.wait_for_outbox_replay(0), "a dead letter started a replay worker"
    assert _worker_threads() == []
    assert [item["replayable"] for item in _remaining(outbox)] == [False]


def test_the_retry_stops_on_an_empty_outbox_and_re_arms_for_the_next_outage(
    control_plane: _ControlPlane, tmp_path: Path
) -> None:
    outbox = tmp_path / "kernel.ndjson"
    sink = _sink(outbox)

    for outage in ("first", "second"):
        control_plane.down = True
        with pytest.raises(CollectorError):
            sink.post("/events", {"sequence": outage, "event_id": outage})
        assert _worker_threads(), f"the {outage} outage did not arm the worker"
        control_plane.down = False
        assert sink.wait_for_outbox_replay(5), f"{outage} backlog did not drain"
        assert _remaining(outbox) == []
        assert _worker_threads() == [], f"the worker outlived the {outage} drain"

    assert control_plane.delivered == ["first", "second"]


def test_a_delivered_live_post_wakes_a_worker_out_of_its_backoff(
    control_plane: _ControlPlane, tmp_path: Path
) -> None:
    outbox = tmp_path / "kernel.ndjson"
    # A 30-60 s schedule: only an interruptible wait lets this test finish.
    sink = _sink(
        outbox,
        outbox_replay_retry_initial_seconds=30,
        outbox_replay_retry_max_seconds=60,
    )

    with pytest.raises(CollectorError):
        sink.post("/events", {"sequence": "buffered", "event_id": "buffered"})
    workers = _worker_threads()
    assert len(workers) == 1 and workers[0].daemon, (
        "the retry worker must be a daemon so process exit never waits on it"
    )
    control_plane.down = False

    assert sink.post("/events", {"sequence": "live", "event_id": "live"}), (
        "the live post after recovery must be delivered"
    )
    assert sink.wait_for_outbox_replay(5), "the live post did not wake the worker"
    assert control_plane.delivered == ["live", "buffered"]
    assert _remaining(outbox) == []
    assert _worker_threads() == []


def test_a_zero_progress_inline_batch_hands_the_backlog_to_the_worker(
    control_plane: _ControlPlane, tmp_path: Path
) -> None:
    """A live post that succeeds while the seeded head still fails."""

    head_failures = [0]
    reachable = control_plane

    def flaky_head(request: Any, **kwargs: Any) -> _Response:
        if json.loads(request.data)["sequence"] == 0 and head_failures[0] < 2:
            head_failures[0] += 1
            raise OSError("head still unreachable")
        return reachable(request, **kwargs)

    outbox = tmp_path / "kernel.ndjson"
    _seed(outbox, [0, 1])
    control_plane.down = False
    sink = _sink(outbox, outbox_replay_batch_size=1)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("gpu_fault.collectors.sinks.urlopen", flaky_head)
        sink.post("/events", {"sequence": "live", "event_id": "live"})
        assert sink.wait_for_outbox_replay(5), "the worker did not drain the head"

    assert control_plane.delivered == ["live", 0, 1]
    assert head_failures[0] == 2, "the head was not retried on the schedule"
    assert _remaining(outbox) == []


def test_a_disabled_schedule_leaves_the_backlog_to_the_next_live_post(
    control_plane: _ControlPlane, tmp_path: Path
) -> None:
    outbox = tmp_path / "kernel.ndjson"
    sink = _sink(outbox, outbox_replay_retry_max_seconds=0)

    with pytest.raises(CollectorError):
        sink.post("/events", {"sequence": "buffered", "event_id": "buffered"})

    assert sink.wait_for_outbox_replay(0), "a disabled schedule started a worker"
    assert _worker_threads() == []
    assert [item["payload"]["sequence"] for item in _remaining(outbox)] == ["buffered"]


@pytest.mark.collector_outbox_retry
def test_the_production_schedule_starts_at_ten_seconds_and_caps_at_sixty() -> None:
    sink = HttpEventSink("https://control")

    assert sink.outbox_replay_retry_initial_seconds == 10.0
    assert sink.outbox_replay_retry_max_seconds == 60.0
    assert collector_sinks.OUTBOX_REPLAY_RETRY_INITIAL_SECONDS == 10.0
    assert collector_sinks.OUTBOX_REPLAY_RETRY_MAX_SECONDS == 60.0


def test_the_suite_default_keeps_test_sinks_from_retrying_on_their_own() -> None:
    assert HttpEventSink("https://control").outbox_replay_retry_max_seconds == 0, (
        "tests/conftest.py no longer disables the retry schedule by default"
    )
