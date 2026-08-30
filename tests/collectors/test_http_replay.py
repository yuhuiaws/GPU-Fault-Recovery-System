from __future__ import annotations

from threading import Event

from ._support import HttpEventSink, json, pytest


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


def test_outbox_background_replay_inputs_are_bounded() -> None:
    with pytest.raises(ValueError, match="batch size"):
        HttpEventSink("https://control", outbox_replay_batch_size=0)
    with pytest.raises(ValueError, match="interval"):
        HttpEventSink("https://control", outbox_replay_background_interval_seconds=-0.1)
