"""Collector outbox dead-letter semantics (ARCH-G2).

A 401/403 used to be a permanent verdict, but token rotation windows are
transient and a live 403 drift incident held every node's fault stream for the
length of the drift. Conversely a record that the control plane rejects with a
schema verdict at replay time used to stay ``replayable`` forever, and ten of
them at the head blocked the whole ``outbox_replay_batch_size`` window. And the
outbox evicted its oldest records silently.
"""

from __future__ import annotations

import hashlib
import sys
from email.message import Message
from threading import Event, Thread
from types import SimpleNamespace
from urllib.error import HTTPError

from gpu_fault.collectors.sinks import OutboxFile

from ._support import (
    CollectorError,
    HttpEventSink,
    collectors_cli,
    io,
    json,
    logging,
    pytest,
)


class _Response:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self):
        return b'{"accepted":true}'


def _http_error(url: str, status: int, detail: bytes = b"detail") -> HTTPError:
    return HTTPError(url, status, "error", Message(), io.BytesIO(detail))


def _seed(path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )


def _record(sequence, *, replayable: bool = True, error: str = "seeded") -> dict:
    return {
        "path": "/events",
        "payload": {"sequence": sequence, "event_id": f"e-{sequence}"},
        "replayable": replayable,
        "error": error,
        "failed_at": "2026-08-30T00:00:00+00:00",
    }


def test_a_live_403_is_retried_and_buffered_as_replayable(monkeypatch, tmp_path):
    attempts = 0

    def probe(request, **_kwargs):
        nonlocal attempts
        attempts += 1
        raise _http_error(request.full_url, 403, b"token not accepted")

    monkeypatch.setattr("gpu_fault.collectors.sinks.urlopen", probe)
    outbox = tmp_path / "outbox.ndjson"
    sink = HttpEventSink(
        "https://control",
        max_attempts=3,
        outbox_path=str(outbox),
        sleep=lambda _seconds: None,
    )

    with pytest.raises(CollectorError) as captured:
        sink.post("/events", {"event_id": "e-1"})

    assert attempts == 3, "a 403 was treated as a verdict instead of retried"
    assert captured.value.buffered is True, "the 403 was not buffered"
    assert captured.value.replayable is True, "the 403 was dead-lettered"
    record = json.loads(outbox.read_text())
    assert record["replayable"] is True, "outbox record is not replayable"


def test_a_replay_time_422_dead_letters_the_head_and_the_rest_drains(
    monkeypatch, tmp_path, caplog
) -> None:
    calls = []

    def probe(request, **_kwargs):
        payload = json.loads(request.data)
        calls.append(payload["sequence"])
        if payload["sequence"] == 0:
            raise _http_error(request.full_url, 422, b"unknown field")
        return _Response()

    monkeypatch.setattr("gpu_fault.collectors.sinks.urlopen", probe)
    outbox = tmp_path / "outbox.ndjson"
    _seed(outbox, [_record(0), _record(1), _record(2)])
    sink = HttpEventSink(
        "https://control",
        outbox_path=str(outbox),
        outbox_replay_batch_size=10,
        outbox_replay_background_interval_seconds=0,
    )

    with caplog.at_level(logging.WARNING, logger="gpu_fault.collectors.sinks"):
        sink.post("/events", {"sequence": "live", "event_id": "live"})
        assert sink.wait_for_outbox_replay(2), "replay did not finish"

    assert calls == ["live", 0, 1, 2], "records behind the poisoned head stalled"
    remaining = [json.loads(line) for line in outbox.read_text().splitlines() if line]
    assert [item["payload"]["sequence"] for item in remaining] == [0], remaining
    assert remaining[0]["replayable"] is False, "the 422 record stayed replayable"
    assert "422" in caplog.text, "the dead-letter was not logged with its status"
    assert '"sequence"' not in caplog.text, "the payload body leaked into the log"


def test_a_replay_time_403_stays_replayable(monkeypatch, tmp_path) -> None:
    def probe(request, **_kwargs):
        raise _http_error(request.full_url, 403, b"rotating")

    monkeypatch.setattr("gpu_fault.collectors.sinks.urlopen", probe)
    outbox = tmp_path / "outbox.ndjson"
    _seed(outbox, [_record(0)])
    sink = HttpEventSink(
        "https://control",
        max_attempts=1,
        outbox_path=str(outbox),
        outbox_replay_background_interval_seconds=0,
        sleep=lambda _seconds: None,
    )

    with pytest.raises(CollectorError):
        sink.post("/events", {"sequence": "live", "event_id": "live"})
    sink.wait_for_outbox_replay(2)

    remaining = [json.loads(line) for line in outbox.read_text().splitlines() if line]
    assert all(item["replayable"] for item in remaining), (
        "a transient auth failure at replay time was dead-lettered"
    )


def test_outbox_eviction_is_logged_and_counted(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(
        "gpu_fault.collectors.sinks.urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("unreachable")),
    )
    outbox = tmp_path / "outbox.ndjson"
    sink = HttpEventSink(
        "https://control", max_attempts=1, outbox_path=str(outbox), outbox_max_records=2
    )

    with caplog.at_level(logging.WARNING, logger="gpu_fault.collectors.sinks"):
        for sequence in range(3):
            with pytest.raises(CollectorError):
                sink.post("/events", {"event_id": f"e-{sequence}"})

    remaining = [json.loads(line) for line in outbox.read_text().splitlines() if line]
    assert [item["payload"]["event_id"] for item in remaining] == ["e-1", "e-2"]
    assert sink.outbox_evictions_total == 1, "eviction was not counted"
    assert "evict" in caplog.text.lower(), "eviction was not logged"
    stats = sink.outbox_stats()
    assert stats["depth"] == 2, stats
    assert stats["replayable"] == 2, stats
    assert stats["dead"] == 0, stats
    assert stats["evictions_total"] == 1, stats
    assert stats["oldest_failed_at"] is not None, stats


def test_outbox_cli_lists_stats_and_requeues_dead_records(
    monkeypatch, tmp_path, capsys
) -> None:
    outbox = tmp_path / "kernel.ndjson"
    _seed(
        outbox,
        [
            _record(0, replayable=False, error="HTTP 422: unknown field"),
            _record(1),
            _record(2, replayable=False, error="HTTP 404: gone"),
        ],
    )
    monkeypatch.delenv("GPU_FAULT_CONTROL_PLANE_URL", raising=False)

    monkeypatch.setattr(
        sys,
        "argv",
        ["gpu-fault-collector", "outbox", "--outbox-path", str(outbox), "list"],
    )
    collectors_cli.main()
    output = capsys.readouterr().out
    listed = output.splitlines()
    assert len(listed) == 3, listed
    assert "e-0" not in output, "payload leaked into list output"
    assert all("/events" in line for line in listed), listed

    monkeypatch.setattr(
        sys,
        "argv",
        ["gpu-fault-collector", "outbox", "--outbox-path", str(outbox), "stats"],
    )
    collectors_cli.main()
    stats = json.loads(capsys.readouterr().out)
    assert stats["depth"] == 3 and stats["dead"] == 2 and stats["replayable"] == 1

    monkeypatch.setattr(
        sys,
        "argv",
        ["gpu-fault-collector", "outbox", "--outbox-path", str(outbox), "requeue-dead"],
    )
    with pytest.raises(SystemExit) as refused:
        collectors_cli.main()
    assert refused.value.code not in {0, None}, "requeue ran without --yes"
    assert OutboxFile(outbox).stats()["dead"] == 2, "requeue happened without --yes"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gpu-fault-collector",
            "outbox",
            "--outbox-path",
            str(outbox),
            "requeue-dead",
            "--yes",
        ],
    )
    collectors_cli.main()
    remaining = [json.loads(line) for line in outbox.read_text().splitlines() if line]
    assert all(item["replayable"] for item in remaining), "dead records not requeued"
    assert "2" in capsys.readouterr().out, "requeue did not report a count"


def test_an_oversize_413_dead_record_keeps_a_digest_not_the_body(
    monkeypatch, tmp_path
) -> None:
    """F4: a rejected oversize payload was stored whole and re-parsed forever.

    The control plane refused it for its size, so the outbox keeps what an
    operator can act on -- a digest, the byte count and a bounded excerpt.
    """

    def probe(request, **_kwargs):
        raise _http_error(request.full_url, 413, b"payload too large")

    monkeypatch.setattr("gpu_fault.collectors.sinks.urlopen", probe)
    outbox = tmp_path / "outbox.ndjson"
    sink = HttpEventSink(
        "https://control",
        max_attempts=1,
        outbox_path=str(outbox),
        sleep=lambda _seconds: None,
    )
    payload = {"event_id": "e-1", "blob": "x" * 200_000}

    with pytest.raises(CollectorError) as captured:
        sink.post("/events", payload)

    assert captured.value.buffered is True, "the 413 was not recorded at all"
    record = json.loads(outbox.read_text())
    assert record["replayable"] is False, "an oversize payload is not replayable"
    assert record["payload_truncated"] is True, "the record was not marked truncated"
    stored = record["payload"]
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    assert stored["payload_sha256"] == hashlib.sha256(body).hexdigest(), stored
    assert stored["payload_bytes"] == len(body), stored
    assert len(stored["payload_excerpt"]) <= 4096, (
        f"the excerpt is {len(stored['payload_excerpt'])} characters, not bounded"
    )
    assert outbox.stat().st_size < 8192, (
        f"the whole oversize body landed in the outbox ({outbox.stat().st_size} bytes)"
    )


def test_a_422_dead_record_keeps_its_payload_whole(monkeypatch, tmp_path) -> None:
    """Only an oversize verdict truncates: every other dead letter is intact."""

    def probe(request, **_kwargs):
        raise _http_error(request.full_url, 422, b"unknown field")

    monkeypatch.setattr("gpu_fault.collectors.sinks.urlopen", probe)
    outbox = tmp_path / "outbox.ndjson"
    sink = HttpEventSink(
        "https://control",
        max_attempts=1,
        outbox_path=str(outbox),
        sleep=lambda _seconds: None,
    )

    with pytest.raises(CollectorError):
        sink.post("/events", {"event_id": "e-1", "detail": "keep me"})

    record = json.loads(outbox.read_text())
    assert record["payload"] == {"event_id": "e-1", "detail": "keep me"}, record
    assert record.get("payload_truncated") is not True, (
        "a schema verdict truncated a payload an operator may still need"
    )


def test_requeue_dead_refuses_a_record_whose_payload_was_truncated(tmp_path) -> None:
    """A truncated record cannot be replayed: only a digest of it is left."""

    outbox_path = tmp_path / "outbox.ndjson"
    truncated = _record(0, replayable=False, error="HTTP 413: too large")
    truncated["payload"] = {
        "payload_sha256": "0" * 64,
        "payload_bytes": 200_000,
        "payload_excerpt": "{...}",
    }
    truncated["payload_truncated"] = True
    _seed(
        outbox_path, [truncated, _record(1, replayable=False, error="HTTP 422: nope")]
    )

    requeued = OutboxFile(outbox_path).requeue_dead()

    assert requeued == 1, f"requeue-dead did not skip the truncated record: {requeued}"
    remaining = [
        json.loads(line) for line in outbox_path.read_text().splitlines() if line
    ]
    assert remaining[0]["replayable"] is False, (
        "a record whose body is only a digest was made replayable"
    )
    assert remaining[1]["replayable"] is True, remaining[1]


def test_the_requeue_command_waits_for_the_outbox_lock_and_loses_no_record(
    tmp_path, capsys
) -> None:
    """F7: ``requeue-dead`` raced the live collector with no inter-process lock.

    The flock is what makes the operator command and the sink's rewrite take
    turns: neither the requeue nor the record appended during it is lost.
    """

    outbox_path = tmp_path / "outbox.ndjson"
    _seed(outbox_path, [_record(0, replayable=False, error="HTTP 422: nope")])
    outbox = OutboxFile(outbox_path)
    finished = Event()
    arguments = SimpleNamespace(
        outbox_path=str(outbox_path),
        collector=None,
        outbox_action="requeue-dead",
        yes=True,
        path=None,
    )

    def requeue() -> None:
        collectors_cli.run_outbox_command(arguments)
        finished.set()

    worker = Thread(target=requeue, daemon=True, name="requeue-dead")
    with outbox.locked():
        worker.start()
        assert not finished.wait(0.5), (
            "requeue-dead rewrote the outbox while the sink held the lock"
        )
        outbox.write([*outbox.read(), _record(1)])
    worker.join(5)
    capsys.readouterr()

    assert finished.is_set(), "requeue-dead never finished after the lock was released"
    remaining = [
        json.loads(line) for line in outbox_path.read_text().splitlines() if line
    ]
    assert [item["payload"]["sequence"] for item in remaining] == [0, 1], (
        f"a record was lost to the concurrent requeue: {remaining}"
    )
    assert all(item["replayable"] for item in remaining), (
        f"the requeue was overwritten by the sink's rewrite: {remaining}"
    )
