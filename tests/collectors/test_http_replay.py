from __future__ import annotations

from threading import Event

from gpu_fault.collectors.sinks import DeliveryStatus, OutboxFile

from ._support import CollectorError, HttpEventSink, json, pytest


class _Response:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self):
        return b'{"accepted":true}'


def _seed_outbox(path, count: int) -> None:
    path.write_text(
        "".join(
            json.dumps(
                {
                    "path": "/events",
                    "payload": {"sequence": index},
                    "replayable": True,
                    "error": "seeded",
                    "failed_at": "2026-08-30T00:00:00+00:00",
                }
            )
            + "\n"
            for index in range(count)
        ),
        encoding="utf-8",
    )


def test_background_replay_drains_every_batch_without_another_live_post(
    monkeypatch, tmp_path
) -> None:
    calls = []

    def probe(request, **_kwargs):
        calls.append(json.loads(request.data)["sequence"])
        return _Response()

    monkeypatch.setattr("gpu_fault.collectors.sinks.urlopen", probe)
    outbox = tmp_path / "outbox.ndjson"
    _seed_outbox(outbox, 23)
    sink = HttpEventSink(
        "https://control",
        outbox_path=str(outbox),
        outbox_replay_batch_size=4,
        outbox_replay_background_interval_seconds=0,
    )

    sink.post("/events", {"sequence": "live"})

    assert sink.wait_for_outbox_replay(2), "background replay did not become idle"
    assert calls == ["live", *range(23)]
    assert outbox.read_text() == ""


def test_background_replay_clears_net001_backlog_and_preserves_dead_letters(
    monkeypatch, tmp_path
) -> None:
    calls = []

    def probe(request, **_kwargs):
        calls.append(json.loads(request.data)["sequence"])
        return _Response()

    monkeypatch.setattr("gpu_fault.collectors.sinks.urlopen", probe)
    outbox = tmp_path / "outbox.ndjson"
    records = [
        {
            "path": "/events",
            "payload": {"sequence": f"dead-{index}"},
            "replayable": False,
            "error": "HTTP 403: historical token mismatch",
            "failed_at": "2026-08-29T00:00:00+00:00",
        }
        for index in range(8)
    ]
    records.extend(
        {
            "path": "/events",
            "payload": {"sequence": index},
            "replayable": True,
            "error": "network unavailable",
            "failed_at": "2026-08-30T00:00:00+00:00",
        }
        for index in range(39)
    )
    outbox.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    sink = HttpEventSink(
        "https://control",
        outbox_path=str(outbox),
        outbox_replay_batch_size=10,
        outbox_replay_background_interval_seconds=0,
    )

    sink.post("/events", {"sequence": "live"})

    assert sink.wait_for_outbox_replay(2), "NET-001 backlog did not drain"
    assert calls == ["live", *range(39)]
    remaining = [json.loads(line) for line in outbox.read_text().splitlines() if line]
    assert len(remaining) == 8
    assert all(not record["replayable"] for record in remaining), (
        "historical dead letters changed replayability"
    )


def test_live_post_is_not_blocked_by_background_replay(monkeypatch, tmp_path) -> None:
    calls = []
    background_started = Event()
    release_background = Event()

    def probe(request, **_kwargs):
        sequence = json.loads(request.data)["sequence"]
        calls.append(sequence)
        if sequence == 1:
            background_started.set()
            assert release_background.wait(2), "test did not release background replay"
        return _Response()

    monkeypatch.setattr("gpu_fault.collectors.sinks.urlopen", probe)
    outbox = tmp_path / "outbox.ndjson"
    _seed_outbox(outbox, 3)
    sink = HttpEventSink(
        "https://control",
        outbox_path=str(outbox),
        outbox_replay_batch_size=1,
        outbox_replay_background_interval_seconds=0,
    )

    sink.post("/events", {"sequence": "live-1"})
    assert background_started.wait(1), "background replay did not start"

    sink.post("/events", {"sequence": "live-2"})

    assert calls[:4] == ["live-1", 0, 1, "live-2"]
    release_background.set()
    assert sink.wait_for_outbox_replay(2), "background replay did not finish"
    assert calls == ["live-1", 0, 1, "live-2", 2]
    assert outbox.read_text() == ""


def test_background_replay_stops_on_zero_progress_and_can_be_reawakened(
    monkeypatch, tmp_path
) -> None:
    calls = []
    recovered = [False]

    def probe(request, **_kwargs):
        sequence = json.loads(request.data)["sequence"]
        calls.append(sequence)
        if sequence == 1 and not recovered[0]:
            raise OSError("network unavailable")
        return _Response()

    monkeypatch.setattr("gpu_fault.collectors.sinks.urlopen", probe)
    outbox = tmp_path / "outbox.ndjson"
    _seed_outbox(outbox, 3)
    sink = HttpEventSink(
        "https://control",
        outbox_path=str(outbox),
        outbox_replay_batch_size=1,
        outbox_replay_background_interval_seconds=0,
    )

    sink.post("/events", {"sequence": "live-1"})

    assert sink.wait_for_outbox_replay(2), "zero-progress replay did not stop"
    assert calls == ["live-1", 0, 1]
    remaining = [
        json.loads(line)["payload"]["sequence"]
        for line in outbox.read_text().splitlines()
    ]
    assert remaining == [1, 2]

    recovered[0] = True
    sink.post("/events", {"sequence": "live-2"})

    assert sink.wait_for_outbox_replay(2), "reawakened replay did not finish"
    assert calls == ["live-1", 0, 1, "live-2", 1, 2]
    assert outbox.read_text() == ""


def test_a_failed_replay_write_never_surfaces_through_deliver(
    monkeypatch, tmp_path
) -> None:
    """F2: ``/var/lib`` full must not turn a delivered event into an ``OSError``.

    The replay runs behind the live post, so its rewrite failing is
    catch-up work that failed -- not a verdict on the event that just
    went out. It used to propagate out of ``post``/``deliver`` and
    reopen ``/dev/kmsg`` mid-batch.
    """

    calls: list[object] = []

    def probe(request, **_kwargs):
        calls.append(json.loads(request.data)["sequence"])
        return _Response()

    def full_disk(_self, _records) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr("gpu_fault.collectors.sinks.urlopen", probe)
    monkeypatch.setattr(OutboxFile, "write", full_disk)
    outbox = tmp_path / "outbox.ndjson"
    _seed_outbox(outbox, 1)
    sink = HttpEventSink(
        "https://control",
        outbox_path=str(outbox),
        outbox_replay_background_interval_seconds=0,
    )

    result = sink.deliver("/events", {"sequence": "live", "event_id": "live"})

    assert result.status is DeliveryStatus.DELIVERED, result
    assert calls == ["live", 0], calls
    remaining = [
        json.loads(line)["payload"]["sequence"]
        for line in outbox.read_text().splitlines()
        if line
    ]
    assert remaining == [0], "the record the sink could not rewrite was lost"
    assert sink.wait_for_outbox_replay(2), (
        "replay state stayed active after the failed write"
    )


def test_a_buffering_live_post_does_not_wait_for_the_replay_network_call(
    monkeypatch, tmp_path
) -> None:
    """F13: replay held ``_outbox_lock`` across its request.

    A live post that fails needs that lock to buffer, so the collector
    thread stalled for the replay budget plus one client timeout.
    """

    replay_reached = Event()
    buffering_done = Event()
    observed: list[str] = []

    def probe(request, **_kwargs):
        sequence = json.loads(request.data)["sequence"]
        if sequence == 1:
            replay_reached.set()
            observed.append("released" if buffering_done.wait(3) else "timeout")
            raise OSError("network unavailable")
        if sequence == "live-2":
            raise OSError("network unavailable")
        return _Response()

    monkeypatch.setattr("gpu_fault.collectors.sinks.urlopen", probe)
    outbox = tmp_path / "outbox.ndjson"
    _seed_outbox(outbox, 2)
    sink = HttpEventSink(
        "https://control",
        max_attempts=1,
        outbox_path=str(outbox),
        outbox_replay_batch_size=1,
        outbox_replay_background_interval_seconds=0,
        sleep=lambda _seconds: None,
    )

    sink.post("/events", {"sequence": "live-1", "event_id": "live-1"})
    assert replay_reached.wait(2), "background replay never reached its network call"

    with pytest.raises(CollectorError) as captured:
        sink.post("/events", {"sequence": "live-2", "event_id": "live-2"})
    buffering_done.set()

    assert captured.value.buffered is True, "the live failure was not buffered"
    assert sink.wait_for_outbox_replay(4), "background replay did not finish"
    assert observed == ["released"], (
        "the buffering post waited for the replay's in-flight request"
    )
    remaining = [
        json.loads(line)["payload"]["sequence"]
        for line in outbox.read_text().splitlines()
        if line
    ]
    assert remaining == [1, "live-2"], (
        "the replay's rewrite lost the record buffered while it was in flight"
    )


def test_outbox_background_replay_inputs_are_bounded() -> None:
    with pytest.raises(ValueError, match="batch size"):
        HttpEventSink("https://control", outbox_replay_batch_size=0)
    with pytest.raises(ValueError, match="interval"):
        HttpEventSink("https://control", outbox_replay_background_interval_seconds=-0.1)
