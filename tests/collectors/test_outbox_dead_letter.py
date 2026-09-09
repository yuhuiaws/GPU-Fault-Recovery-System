"""Collector outbox dead-letter semantics (ARCH-G2).

A 401/403 used to be a permanent verdict, but token rotation windows are
transient and a live 403 drift incident held every node's fault stream for the
length of the drift. Conversely a record that the control plane rejects with a
schema verdict at replay time used to stay ``replayable`` forever, and ten of
them at the head blocked the whole ``outbox_replay_batch_size`` window. And the
outbox evicted its oldest records silently.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import os
import subprocess
import sys
from email.message import Message
from threading import Event, Thread
from types import SimpleNamespace
from urllib.error import HTTPError

from gpu_fault.collectors import outbox_file as collector_outbox
from gpu_fault.collectors import sinks as collector_sinks
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
        "https://control", max_attempts=1, outbox_path=str(outbox), outbox_max_records=3
    )

    with caplog.at_level(logging.WARNING, logger="gpu_fault.collectors.sinks"):
        for sequence in range(5):
            with pytest.raises(CollectorError):
                sink.post("/events", {"event_id": f"e-{sequence}"})

    # Passing the ceiling on the fourth record compacts down to 90% of it (two
    # records), so the fifth is a free append rather than another rewrite.
    remaining = [json.loads(line) for line in outbox.read_text().splitlines() if line]
    assert [item["payload"]["event_id"] for item in remaining] == ["e-2", "e-3", "e-4"]
    assert sink.outbox_evictions_total == 2, "eviction was not counted"
    assert "evict" in caplog.text.lower(), "eviction was not logged"
    assert len([line for line in caplog.text.splitlines() if "evict" in line]) == 1, (
        f"eviction logged one line per record instead of one per compaction: "
        f"{caplog.text!r}"
    )
    stats = sink.outbox_stats()
    assert stats["depth"] == 3, stats
    assert stats["replayable"] == 3, stats
    assert stats["dead"] == 0, stats
    assert stats["evictions_total"] == 2, stats
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
    assert stored["payload_event_key"] == "e-1", (
        f"the truncated record lost the event's identity: {stored}"
    )
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
        force=False,
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


def _requeue_arguments(outbox_path, *, force: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        outbox_path=str(outbox_path),
        collector=None,
        outbox_action="requeue-dead",
        yes=True,
        path=None,
        force=force,
    )


def test_requeue_dead_refuses_to_run_when_the_outbox_lock_cannot_be_taken(
    monkeypatch, tmp_path, capsys
) -> None:
    """I-2: the operator process has no in-process lock to fall back to.

    A ``.lock`` the unprivileged operator cannot open (the collector created it
    as root) must stop the command, not silently reinstate the F7 lost update.
    """

    outbox_path = tmp_path / "outbox.ndjson"
    _seed(outbox_path, [_record(0, replayable=False, error="HTTP 422: nope")])
    real_open = os.open

    def refuse_lock_files(path, flags, mode=0o777, **kwargs):
        if str(path).endswith(".lock"):
            raise PermissionError(13, "Permission denied")
        return real_open(path, flags, mode, **kwargs)

    monkeypatch.setattr(os, "open", refuse_lock_files)

    with pytest.raises(SystemExit) as refused:
        collectors_cli.run_outbox_command(_requeue_arguments(outbox_path))

    assert refused.value.code not in {0, None}, "requeue-dead exited successfully"
    message = str(refused.value)
    assert "lock" in message.lower(), f"the refusal did not name the lock: {message!r}"
    assert "--force" in message, f"the refusal did not offer a way through: {message!r}"
    record = json.loads(outbox_path.read_text())
    assert record["replayable"] is False, (
        "requeue-dead changed the outbox although it could not take the lock"
    )
    capsys.readouterr()


def test_requeue_dead_gives_up_on_a_held_lock_instead_of_waiting_for_ever(
    monkeypatch, tmp_path, capsys
) -> None:
    """N-5: strict mode took the lock with a blocking ``flock``.

    The holder an operator actually meets is the live collector's outbox replay,
    and a saturated backlog keeps it busy for as long as the control plane is
    unreachable. ``requeue-dead`` then printed nothing, made no progress and
    could not be told from a wedged command; there was no timeout and no hint
    that ``--force`` exists. A bounded poll turns that into an answer.
    """

    monkeypatch.setattr(collector_outbox, "OUTBOX_LOCK_RETRY_SECONDS", 0.02)
    outbox_path = tmp_path / "outbox.ndjson"
    _seed(outbox_path, [_record(0, replayable=False, error="HTTP 422: nope")])
    lock_path = OutboxFile(outbox_path).lock_path
    outcome: list[BaseException | None] = []

    def requeue() -> None:
        try:
            collectors_cli.run_outbox_command(_requeue_arguments(outbox_path))
        except BaseException as exc:  # the CLI reports a refusal as SystemExit
            outcome.append(exc)
        else:
            outcome.append(None)

    handle = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    worker = Thread(target=requeue, daemon=True, name="requeue-dead-blocked")
    try:
        # A second open file description, which is what another process's lock
        # looks like to ``flock``.
        fcntl.flock(handle, fcntl.LOCK_EX)
        worker.start()
        worker.join(10)
        blocked = worker.is_alive()
    finally:
        fcntl.flock(handle, fcntl.LOCK_UN)
        os.close(handle)
        worker.join(10)
    capsys.readouterr()

    assert not blocked, (
        "requeue-dead was still waiting for a held outbox lock after 10s; a "
        "strict lock must be bounded, not blocking"
    )
    assert isinstance(outcome[0], SystemExit), (
        f"a lock it could not take did not stop requeue-dead: {outcome[0]!r}"
    )
    message = str(outcome[0])
    assert str(lock_path) in message, (
        f"the refusal did not name the lock file to look at: {message!r}"
    )
    assert "--force" in message, f"the refusal did not offer a way through: {message!r}"
    record = json.loads(outbox_path.read_text())
    assert record["replayable"] is False, (
        "requeue-dead rewrote the outbox although another holder had the lock"
    )


def test_the_outbox_file_layer_is_still_reachable_through_the_sink() -> None:
    """The file layer moved to its own module; the old names must not move.

    ``sinks.OutboxFile`` is what the CLI imports, what the runbooks name and what
    every existing test patches, so the split re-exports the same objects rather
    than a copy of them -- a second class would give a monkeypatch on one no
    effect on the other.
    """

    for name in (
        "OutboxFile",
        "OutboxLockUnavailable",
        "OUTBOX_LOCK_ATTEMPTS",
        "OUTBOX_LOCK_RETRY_SECONDS",
        "OUTBOX_LOCK_FORCE_ADVICE",
        "UNLOCKED_WRITE_WARN_INTERVAL",
        "unlocked_outbox_writes",
    ):
        assert getattr(collector_sinks, name) is getattr(collector_outbox, name), (
            f"sinks.{name} is no longer the object outbox_file defines, so the "
            "two layers can disagree"
        )


def test_the_bounded_outbox_lock_wait_is_short_enough_to_answer() -> None:
    """The bound exists to answer an operator, so it has to stay human-sized."""

    waited = (
        collector_outbox.OUTBOX_LOCK_ATTEMPTS
        * collector_outbox.OUTBOX_LOCK_RETRY_SECONDS
    )
    assert 1 <= waited <= 30, (
        f"the strict outbox lock waits {waited}s, which is either too short to "
        "outlast one read-modify-write or too long to read as an answer"
    )


def test_requeue_dead_force_proceeds_without_the_lock_and_says_so(
    monkeypatch, tmp_path, capsys, caplog
) -> None:
    outbox_path = tmp_path / "outbox.ndjson"
    _seed(outbox_path, [_record(0, replayable=False, error="HTTP 422: nope")])
    real_open = os.open

    def refuse_lock_files(path, flags, mode=0o777, **kwargs):
        if str(path).endswith(".lock"):
            raise PermissionError(13, "Permission denied")
        return real_open(path, flags, mode, **kwargs)

    monkeypatch.setattr(os, "open", refuse_lock_files)

    with caplog.at_level(logging.WARNING, logger="gpu_fault.collectors.sinks"):
        collectors_cli.run_outbox_command(_requeue_arguments(outbox_path, force=True))

    record = json.loads(outbox_path.read_text())
    assert record["replayable"] is True, "--force did not requeue the dead record"
    assert "lock" in caplog.text.lower(), (
        f"an unlocked requeue was not warned about: {caplog.text!r}"
    )
    capsys.readouterr()


def test_force_does_not_walk_back_into_the_block_it_escapes(
    monkeypatch, tmp_path, capsys, caplog
) -> None:
    """The refusal's own advice must not block the way the refusal did.

    ``--force`` mapped to the collector's ``required=False`` path: a single
    blocking ``LOCK_EX``. Against the holder an operator actually meets -- a
    live collector replaying a saturated backlog -- that waits for as long as
    the control plane stays unreachable, so following the advice printed at the
    end of the refusal led straight back into the hang it was printed to
    escape. ``--force`` takes the same bounded poll and then works unlocked.
    """

    monkeypatch.setattr(collector_outbox, "OUTBOX_LOCK_RETRY_SECONDS", 0.02)
    outbox_path = tmp_path / "outbox.ndjson"
    _seed(outbox_path, [_record(0, replayable=False, error="HTTP 422: nope")])
    lock_path = OutboxFile(outbox_path).lock_path
    outcome: list[BaseException | None] = []

    def requeue() -> None:
        try:
            collectors_cli.run_outbox_command(
                _requeue_arguments(outbox_path, force=True)
            )
        except BaseException as exc:
            outcome.append(exc)
        else:
            outcome.append(None)

    handle = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    worker = Thread(target=requeue, daemon=True, name="requeue-dead-forced")
    try:
        # A second open file description: what another process's lock looks
        # like to ``flock``.
        fcntl.flock(handle, fcntl.LOCK_EX)
        with caplog.at_level(logging.WARNING, logger="gpu_fault.collectors.sinks"):
            worker.start()
            worker.join(10)
            blocked = worker.is_alive()
            warnings = caplog.text
    finally:
        fcntl.flock(handle, fcntl.LOCK_UN)
        os.close(handle)
        worker.join(10)
    capsys.readouterr()

    assert not blocked, (
        "--force was still waiting for the held outbox lock after 10s, which is "
        "the block the refusal offers --force as the way out of"
    )
    assert outcome[0] is None, (
        f"--force did not get through without the lock: {outcome[0]!r}"
    )
    record = json.loads(outbox_path.read_text())
    assert record["replayable"] is True, (
        "--force answered but did not requeue the dead record"
    )
    assert "--force" in warnings and str(outbox_path) in warnings, (
        f"a forced unlocked rewrite was not warned about: {warnings!r}"
    )
    assert "pass --force to work without the lock" not in warnings, (
        "the forced warning embeds the strict refusal verbatim, so it still "
        f"advises passing --force after --force was passed: {warnings!r}"
    )


def test_force_over_an_unopenable_lock_does_not_claim_an_in_process_lock(
    tmp_path, capsys, caplog
) -> None:
    """An operator command has no in-process lock, so it must not name one.

    ``--force`` also covers the case where ``.lock`` cannot be opened at all --
    a directory in its place, a read-only volume -- and the rewrite still runs.
    That path went through the collector's own wording, which told the operator
    the write had run "on the in-process lock": for a CLI process that lock
    does not exist, and it reads as a guarantee nothing is providing.
    """

    outbox_path = tmp_path / "outbox.ndjson"
    _seed(outbox_path, [_record(0, replayable=False, error="HTTP 422: nope")])
    # EISDIR from os.open(O_RDWR): a lock that cannot be taken at all, not one
    # somebody else is holding.
    OutboxFile(outbox_path).lock_path.mkdir()

    with caplog.at_level(logging.WARNING, logger="gpu_fault.collectors.sinks"):
        collectors_cli.run_outbox_command(_requeue_arguments(outbox_path, force=True))
    warnings = caplog.text
    capsys.readouterr()

    assert json.loads(outbox_path.read_text())["replayable"] is True, (
        "--force must still requeue when the lock file cannot be opened"
    )
    assert "because --force was passed" in warnings, (
        f"the warning does not say why the write ran unlocked: {warnings!r}"
    )
    assert "on the in-process lock" not in warnings, (
        "the CLI has no in-process lock to fall back to, so claiming one "
        f"overstates what protected this rewrite: {warnings!r}"
    )


def test_a_concurrent_rewrite_does_not_take_this_ones_temporary_file(
    monkeypatch, tmp_path
) -> None:
    """Two writers, one ``.tmp`` inode: the second ``os.replace`` found nothing.

    ``--force`` rewrites without the lock, so an operator's ``requeue-dead``
    can now land inside the collector's own compaction or replay rewrite. Both
    truncated the same ``outbox.ndjson.tmp``, so whichever renamed second raised
    FileNotFoundError -- the collector reporting as failed an event it had
    really buffered -- and the file that survived could hold a blend of the two.
    """

    outbox = OutboxFile(tmp_path / "outbox.ndjson")
    entered: list[str] = []
    monkeypatch.setattr(os, "getpid", lambda: 4243 if entered else 4242)
    real_fsync = os.fsync

    def fsync(descriptor: int) -> None:
        real_fsync(descriptor)
        if entered:
            return
        entered.append("nested")
        # The forced operator rewrite, arriving between this writer's data and
        # its rename -- the window the lock used to close.
        OutboxFile(outbox.path).write([_record(99)])

    monkeypatch.setattr(os, "fsync", fsync)
    try:
        outbox.write([_record(0)])
    except OSError as exc:
        pytest.fail(
            "a concurrent rewrite took this one's temporary file, so a buffered "
            f"event would be reported as failed: {exc!r}"
        )
    finally:
        monkeypatch.undo()

    sequences = [record["payload"]["sequence"] for record in outbox.read()]
    assert sequences == [0], (
        f"the writer that renamed last must own the file, got {sequences}"
    )
    leftovers = sorted(path.name for path in tmp_path.glob("*.tmp"))
    assert not leftovers, f"a temporary file outlived its rewrite: {leftovers}"


def test_the_refusal_names_the_wait_it_took_and_offers_force_once(
    monkeypatch, tmp_path, capsys
) -> None:
    """The poll sleeps *between* attempts, so the promised wait was too long.

    The message said "after 5s" (10 attempts x 0.5 s) but returns after 9
    sleeps, so an operator timing the command sees it give up half a second
    early -- with a held lock the one measurement they can take disagreed with
    the message. The CLI also repeated the sink's own ``--force`` advice in the
    same line.
    """

    slept: list[float] = []
    monkeypatch.setattr(collector_outbox.time, "sleep", slept.append)
    outbox_path = tmp_path / "outbox.ndjson"
    _seed(outbox_path, [_record(0, replayable=False, error="HTTP 422: nope")])
    lock_path = OutboxFile(outbox_path).lock_path

    handle = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(handle, fcntl.LOCK_EX)
        with pytest.raises(SystemExit) as refusal:
            collectors_cli.run_outbox_command(_requeue_arguments(outbox_path))
    finally:
        fcntl.flock(handle, fcntl.LOCK_UN)
        os.close(handle)
    capsys.readouterr()

    message = str(refusal.value)
    assert f"{sum(slept):.1f}s" in message, (
        f"the refusal names a wait it did not take ({sum(slept)}s slept over "
        f"{len(slept)} sleep(s)): {message!r}"
    )
    assert message.count("--force") == 1, (
        f"the refusal repeats the advice it already carries: {message!r}"
    )


def _requeue_against_held_lock(
    outbox_path, *, hold, force: bool, monkeypatch, caplog
) -> tuple[BaseException | None, str]:
    """Run ``requeue-dead`` in a thread while ``hold()`` keeps the lock.

    Returns what stopped the command (``None`` when it ran) and the WARNING
    text it logged. The bounded poll is shortened so the test answers fast.
    """

    monkeypatch.setattr(collector_outbox, "OUTBOX_LOCK_RETRY_SECONDS", 0.02)
    outcome: list[BaseException | None] = []

    def requeue() -> None:
        try:
            collectors_cli.run_outbox_command(
                _requeue_arguments(outbox_path, force=force)
            )
        except BaseException as exc:
            outcome.append(exc)
        else:
            outcome.append(None)

    worker = Thread(target=requeue, daemon=True, name="requeue-dead-named-holder")
    with hold, caplog.at_level(logging.WARNING, logger="gpu_fault.collectors.sinks"):
        worker.start()
        worker.join(10)
        assert not worker.is_alive(), "requeue-dead did not answer in 10s"
        warnings = caplog.text
    worker.join(10)
    return outcome[0], warnings


def test_the_strict_refusal_names_the_collector_holding_the_lock(
    monkeypatch, tmp_path, capsys, caplog
) -> None:
    """R3: ``flock`` carries no owner, so the refusal could not say who held it.

    "Another process" left the operator guessing between a live collector's
    replay and a colleague's ``requeue-dead``; the answer decides whether
    ``--force`` is safe. The taker now writes ``pid``/``role``/``since`` into
    ``.lock`` and the refusal reads it back.
    """

    outbox_path = tmp_path / "outbox.ndjson"
    _seed(outbox_path, [_record(0, replayable=False, error="HTTP 422: nope")])
    # A second open file description in this process: what the live sink looks
    # like to the CLI's flock, with the sink's own role in the file.
    stopped, _warnings = _requeue_against_held_lock(
        outbox_path,
        hold=OutboxFile(outbox_path).locked(),
        force=False,
        monkeypatch=monkeypatch,
        caplog=caplog,
    )
    capsys.readouterr()

    assert isinstance(stopped, SystemExit), f"the held lock did not refuse: {stopped!r}"
    message = str(stopped)
    assert f"held by pid {os.getpid()} (collector) since 20" in message, (
        f"the refusal does not name the holder's pid and role: {message!r}"
    )
    assert "holder unknown" not in message, message
    assert json.loads(outbox_path.read_text())["replayable"] is False, (
        "requeue-dead rewrote the outbox although the collector held the lock"
    )


def test_the_forced_warning_names_the_collector_holding_the_lock(
    monkeypatch, tmp_path, capsys, caplog
) -> None:
    """``--force`` past a live holder is the F7 race; the log must say whose."""

    outbox_path = tmp_path / "outbox.ndjson"
    _seed(outbox_path, [_record(0, replayable=False, error="HTTP 422: nope")])
    stopped, warnings = _requeue_against_held_lock(
        outbox_path,
        hold=OutboxFile(outbox_path).locked(),
        force=True,
        monkeypatch=monkeypatch,
        caplog=caplog,
    )
    capsys.readouterr()

    assert stopped is None, f"--force did not get through: {stopped!r}"
    assert json.loads(outbox_path.read_text())["replayable"] is True, (
        "--force answered but did not requeue the dead record"
    )
    assert f"held by pid {os.getpid()} (collector) since 20" in warnings, (
        f"the forced WARNING does not name who held the lock: {warnings!r}"
    )
    assert "because --force was passed" in warnings, warnings


@pytest.mark.parametrize(
    "contents",
    [
        pytest.param(b"", id="empty"),
        pytest.param(b"\x00garbage{not json", id="garbage"),
        pytest.param(b'{"role":"collector"}', id="no-pid"),
        pytest.param(b'{"pid":"12","role":"collector"}', id="pid-not-int"),
        pytest.param(b'["pid", 12]', id="not-an-object"),
        pytest.param(b"{" + b"x" * 8192 + b"}", id="oversize"),
    ],
)
def test_an_unreadable_lock_file_reads_as_holder_unknown(
    monkeypatch, tmp_path, capsys, caplog, contents
) -> None:
    """A foreign or torn ``.lock`` is a missing hint, never a failure or a guess.

    A pre-R3 collector leaves the file empty; a truncate-then-write torn by a
    concurrent reader leaves it partial; the bounded 4 KB read cuts an oversize
    file mid-JSON. All of them must land on the same three words.
    """

    outbox_path = tmp_path / "outbox.ndjson"
    _seed(outbox_path, [_record(0, replayable=False, error="HTTP 422: nope")])
    lock_path = OutboxFile(outbox_path).lock_path
    lock_path.write_bytes(contents)

    @contextlib.contextmanager
    def hold_without_recording():
        # A raw flock: the holder that does not write its identity.
        handle = os.open(lock_path, os.O_RDWR)
        try:
            fcntl.flock(handle, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
            os.close(handle)

    stopped, _warnings = _requeue_against_held_lock(
        outbox_path,
        hold=hold_without_recording(),
        force=False,
        monkeypatch=monkeypatch,
        caplog=caplog,
    )
    capsys.readouterr()

    assert isinstance(stopped, SystemExit), f"the held lock did not refuse: {stopped!r}"
    message = str(stopped)
    assert "holder unknown" in message, (
        f"an unreadable lock file did not read as unknown: {message!r}"
    )
    assert "held by pid" not in message and "stale holder" not in message, message
    assert lock_path.read_bytes() == contents, (
        "a refused taker rewrote the lock file it does not hold"
    )


def test_a_recorded_holder_that_is_gone_reads_as_stale(
    monkeypatch, tmp_path, capsys, caplog
) -> None:
    """The flock dies with its process, so a dead recorded pid is not the holder.

    Somebody else holds the lock without having overwritten the line -- an
    older collector, or a child that inherited the parent's descriptor. Saying
    "held by pid N" would send the operator to ``kill`` a pid that is gone.
    """

    outbox_path = tmp_path / "outbox.ndjson"
    _seed(outbox_path, [_record(0, replayable=False, error="HTTP 422: nope")])
    lock_path = OutboxFile(outbox_path).lock_path
    # A process that has exited and been reaped: its pid is certainly gone.
    finished = subprocess.Popen([sys.executable, "-c", "pass"])
    assert finished.wait(30) == 0
    dead_pid = finished.pid
    lock_path.write_text(
        json.dumps(
            {"pid": dead_pid, "role": "collector", "since": "2026-09-09T00:00:00+00:00"}
        )
    )

    @contextlib.contextmanager
    def hold_without_recording():
        handle = os.open(lock_path, os.O_RDWR)
        try:
            fcntl.flock(handle, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
            os.close(handle)

    stopped, _warnings = _requeue_against_held_lock(
        outbox_path,
        hold=hold_without_recording(),
        force=False,
        monkeypatch=monkeypatch,
        caplog=caplog,
    )
    capsys.readouterr()

    assert isinstance(stopped, SystemExit), f"the held lock did not refuse: {stopped!r}"
    message = str(stopped)
    assert f"stale holder pid {dead_pid} (collector, gone)" in message, (
        f"a dead recorded pid was not reported as gone: {message!r}"
    )
    assert "held by pid" not in message, message


def test_every_lock_take_records_its_role_and_a_write_failure_is_harmless(
    monkeypatch, tmp_path, capsys
) -> None:
    """The sink writes ``collector``, the CLI ``cli:<subcommand>``; neither may fail.

    The identity line is a hint for the *next* taker, so a lock file that
    cannot be written (a read-only ``.lock`` left by root, a full volume) must
    not fail the lock and, behind it, the post the lock protects.
    """

    outbox_path = tmp_path / "outbox.ndjson"
    _seed(outbox_path, [_record(0, replayable=False, error="HTTP 422: nope")])
    lock_path = OutboxFile(outbox_path).lock_path
    # A longer identity left by an earlier taker. The take must never truncate
    # to zero first (ext4 auto_da_alloc turns truncate-to-zero + rewrite +
    # close into a ~1 ms flush), so it writes over the old line and trims to
    # its own length; the reader takes the first JSON value either way.
    lock_path.write_text(json.dumps({"pid": 1, "role": "x" * 300}) + "\n")

    with OutboxFile(outbox_path).locked():
        recorded = json.loads(lock_path.read_text().splitlines()[0])
    assert recorded["pid"] == os.getpid() and recorded["role"] == "collector", recorded
    assert recorded["since"].startswith("20") and "+00:00" in recorded["since"], (
        f"the timestamp is not an ISO UTC instant: {recorded!r}"
    )
    assert set(recorded) == {"pid", "role", "since"}, (
        f"the lock file carries more than the holder's identity: {recorded!r}"
    )
    handle = os.open(lock_path, os.O_RDONLY)
    try:
        assert collector_outbox.describe_lock_holder(handle).startswith(
            f"held by pid {os.getpid()} (collector) since 20"
        ), "the reader did not take the first line over the old tail"
    finally:
        os.close(handle)

    collectors_cli.run_outbox_command(_requeue_arguments(outbox_path))
    capsys.readouterr()
    recorded = json.loads(lock_path.read_text().splitlines()[0])
    assert recorded["role"] == "cli:requeue-dead", (
        f"the CLI did not record its subcommand as the role: {recorded!r}"
    )

    def refuse_writes(*_args, **_kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(os, "pwrite", refuse_writes)
    with OutboxFile(outbox_path).locked(required=True):
        pass  # the lock was taken: the body ran and nothing was raised
    assert lock_path.read_bytes() == b"", (
        "the failed identity write left stale contents behind instead of an "
        "empty file, so the next refusal would name a holder that is gone"
    )
