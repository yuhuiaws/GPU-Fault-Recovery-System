"""The collector outbox *file*: one NDJSON file, its lock, and its rewrites.

Split out of :mod:`gpu_fault.collectors.sinks` along the seam the reviews kept
drawing anyway. This module owns how the file is read, appended to, rewritten
and locked; the sink above it keeps the policy -- what to buffer, when to
compact, what to replay and what to dead-letter. Nothing here knows about HTTP.

The logger is deliberately the sink's own name: an operator's log filters, the
runbooks and the tests all grep ``gpu_fault.collectors.sinks``, and splitting a
file must not move a single log line.
"""

from __future__ import annotations

import contextlib
import errno
import fcntl
import json
import logging
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

LOGGER = logging.getLogger("gpu_fault.collectors.sinks")

#: One WARNING per this many writes that ran without the cross-process lock:
#: one per path hid every later unlocked write, including a permanent failure.
UNLOCKED_WRITE_WARN_INTERVAL = 100

#: Writes per lock path that ran without the lock: bounded logs, plus how long
#: a degradation has lasted. The lock also guards the sink's own counters, so
#: both tables stay under one mutex.
_UNLOCKED_WRITES: dict[str, int] = {}
_OUTBOX_COUNTER_LOCK = Lock()

#: How long an operator command polls for the outbox lock before giving up: a
#: blocking ``flock`` behind a live replay reads as a wedged command.
OUTBOX_LOCK_ATTEMPTS = 10
#: The strict refusal's tail, named so the forced warning -- which embeds that
#: refusal -- can strip the flag the operator has already passed.
OUTBOX_LOCK_FORCE_ADVICE = (
    ": retry, stop the collector, or pass --force to work without the lock,"
    " which can lose one side's update"
)
OUTBOX_LOCK_RETRY_SECONDS = 0.5


def _bump(counters: dict[str, int], key: str) -> int:
    with _OUTBOX_COUNTER_LOCK:
        total = counters.get(key, 0) + 1
        counters[key] = total
        return total


def unlocked_outbox_writes(lock_path: Path) -> int:
    """How many writes to ``lock_path``'s outbox ran without the flock."""

    with _OUTBOX_COUNTER_LOCK:
        return _UNLOCKED_WRITES.get(str(lock_path), 0)


class OutboxLockUnavailable(OSError):
    """Raised for ``required=True`` only: a command with no lock to degrade to."""


@dataclass(frozen=True)
class OutboxFile:
    """The durable NDJSON outbox one collector writes (ARCH-G2).

    Shared by the sink and the ``gpu-fault-collector outbox`` command so an
    operator reads exactly the records the sink will replay. Every
    read-modify-write on either side runs inside :meth:`locked`, an
    ``fcntl.flock`` on ``<outbox>.lock``: the sink's in-process lock cannot see
    the operator's process, and a ``requeue-dead`` that interleaved with a
    replay's rewrite silently lost one side's update (F7).
    """

    path: Path

    @property
    def lock_path(self) -> Path:
        return self.path.with_name(self.path.name + ".lock")

    @contextlib.contextmanager
    def locked(self, *, required: bool = False, forced: bool = False) -> Iterator[None]:
        """Hold the cross-process outbox lock for one read-modify-write.

        Not re-entrant (``flock`` is per open file description, so two nested
        blocks deadlock): take it once around the whole rewrite. ``required=False``
        is the collector's write path: a filesystem without
        ``flock``, or a ``.lock`` it cannot open, must not fail a post, so the
        body runs on the in-process lock alone, counted and warned about.
        ``required=True`` is an operator command, which has no in-process lock:
        that failure and a lock still held after the bounded poll both raise
        :class:`OutboxLockUnavailable` and the body never runs. ``forced=True``
        is ``--force``: the same bounded poll, then the body *without* the lock,
        never ``required=False``'s blocking wait -- the block it exists to escape.

        A failure to create the outbox *directory* is not a lock problem and is
        left to the caller: the append that follows fails with the same error,
        and reporting it as a missing lock hid a full or read-only volume.
        """

        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle: int | None = None
        try:
            handle = self._take_lock(required=required or forced)
        except OutboxLockUnavailable as exc:
            if not forced:
                raise
            self._warn_unlocked_write(exc, forced=True)
        except OSError as exc:
            if required:
                raise OutboxLockUnavailable(
                    exc.errno or 0,
                    f"cannot take the collector outbox lock {self.lock_path}: {exc}",
                ) from exc
            self._warn_unlocked_write(exc, forced=forced)
        try:
            yield
        finally:
            if handle is not None:
                with contextlib.suppress(OSError):
                    fcntl.flock(handle, fcntl.LOCK_UN)
                with contextlib.suppress(OSError):
                    os.close(handle)

    def _take_lock(self, *, required: bool) -> int:
        """Open ``<outbox>.lock`` and hold ``flock`` on it, or raise ``OSError``.

        A collector waits (the holder is one short read-modify-write and its own
        writes must not fail); an operator command polls, so a ``requeue-dead``
        behind a saturated replay gets an answer rather than a block.
        """

        handle = os.open(self.lock_path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
        mode = fcntl.LOCK_EX | (fcntl.LOCK_NB if required else 0)
        attempts = OUTBOX_LOCK_ATTEMPTS if required else 1
        waited = (attempts - 1) * OUTBOX_LOCK_RETRY_SECONDS  # sleeps go between
        try:
            while True:
                try:
                    fcntl.flock(handle, mode)
                    return handle
                except OSError as exc:
                    attempts -= 1
                    if exc.errno not in {errno.EACCES, errno.EAGAIN}:
                        raise
                    if attempts <= 0:
                        raise OutboxLockUnavailable(
                            exc.errno or 0,
                            f"another process still holds the collector"
                            f" outbox lock {self.lock_path} after {waited:.1f}s"
                            " (a live collector's outbox replay)"
                            f"{OUTBOX_LOCK_FORCE_ADVICE}",
                        ) from exc
                time.sleep(OUTBOX_LOCK_RETRY_SECONDS)
        except BaseException:
            with contextlib.suppress(OSError):
                os.close(handle)
            raise

    def _warn_unlocked_write(self, exc: OSError, *, forced: bool = False) -> None:
        total = _bump(_UNLOCKED_WRITES, str(self.lock_path))
        if not forced and total > 1 and total % UNLOCKED_WRITE_WARN_INTERVAL:
            return
        LOGGER.warning(
            "collector outbox lock %s is unavailable (%s); %d write(s) so far have "
            "run without it%s, so a concurrent append or 'outbox requeue-dead' can "
            "lose one side's update",
            self.lock_path,
            str(exc).replace(OUTBOX_LOCK_FORCE_ADVICE, "") if forced else exc,
            total,
            " because --force was passed" if forced else " on the in-process lock",
        )

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        unparseable = 0
        text = self.path.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                # A crash between an append and its fsync can tear the last
                # line (one merely missing its newline still parses). Skipping
                # it loses that record; crashing loses the whole outbox.
                unparseable += 1
                continue
            if isinstance(value, dict):
                records.append(value)
            else:
                unparseable += 1
        if unparseable:
            LOGGER.warning(
                "collector outbox %s has %d unparseable line(s), skipped "
                "(a torn append is expected after an unclean shutdown)",
                self.path,
                unparseable,
            )
        return records

    def count_lines(self) -> int:
        """How many lines the file holds, without parsing any of them.

        The append path needs a depth to compare against ``outbox_max_records``
        without re-parsing the whole backlog per buffered event (F4). A final
        line with no newline of its own is not counted, so this can read one
        short of :meth:`read`, which revives a complete but unterminated record.
        """

        if not self.path.exists():
            return 0
        total = 0
        with open(self.path, "rb") as handle:
            while chunk := handle.read(65536):
                total += chunk.count(b"\n")
        return total

    def ends_mid_line(self) -> bool:
        """Whether the file ends in a fragment with no newline of its own."""

        try:
            with open(self.path, "rb") as handle:
                handle.seek(0, os.SEEK_END)
                if handle.tell() == 0:
                    return False
                handle.seek(-1, os.SEEK_END)
                return handle.read(1) != b"\n"
        except FileNotFoundError:
            return False

    def append(self, record: dict[str, Any]) -> None:
        """Add one record with a single append and one ``fsync``.

        Append-only is what makes buffering during an outage O(1) instead of a
        full re-parse and rewrite per event (F4). A crash between the write and
        the ``fsync`` may lose at most this record, and only if it was torn
        mid-JSON (see :meth:`read`).

        The torn tail gets a newline of its own first: appending straight onto a
        fragment concatenated the two into one unparseable line, so ``read()``
        dropped the *new* record too -- after the collector had been told it was
        buffered and had moved its cursor past a real event.
        """

        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, separators=(",", ":"), default=str) + "\n"
        with open(self.path, "a", encoding="utf-8") as handle:
            if self.ends_mid_line():
                LOGGER.warning(
                    "collector outbox %s ends in a partial record; closing that "
                    "line so the record appended now is readable (the fragment "
                    "itself is skipped)",
                    self.path,
                )
                handle.write("\n")
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())

    def write(self, records: list[dict[str, Any]]) -> None:
        """Replace the file with ``records``, durably (F9).

        ``os.replace`` is atomic against a kill, but without the two ``fsync``
        calls a hard node reset -- an action this product performs -- can
        persist the rename before the data and leave an empty outbox.

        The temporary carries the pid because ``--force`` rewrites unlocked: on
        one shared name an operator's ``requeue-dead`` inside the collector's own
        compaction left the second rename with ``FileNotFoundError``.
        """

        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f"{self.path.name}.{os.getpid()}.tmp")
        try:
            with open(temporary, "w", encoding="utf-8") as handle:
                handle.write(
                    "".join(
                        json.dumps(item, separators=(",", ":"), default=str) + "\n"
                        for item in records
                    )
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        except BaseException:
            # Nobody else can, now that the name is this process's alone.
            with contextlib.suppress(OSError):
                temporary.unlink()
            raise
        self._fsync_directory()

    def _fsync_directory(self) -> None:
        """Persist the rename itself; a failure here is not a failed write."""

        try:
            descriptor = os.open(self.path.parent, os.O_RDONLY)
        except OSError as exc:
            LOGGER.warning(
                "cannot open collector outbox directory %s to fsync the rename: %s",
                self.path.parent,
                exc,
            )
            return
        try:
            os.fsync(descriptor)
        except OSError as exc:
            LOGGER.warning(
                "collector outbox directory %s could not be fsynced: %s",
                self.path.parent,
                exc,
            )
        finally:
            os.close(descriptor)

    @staticmethod
    def summarize(
        records: list[dict[str, Any]],
        *,
        evictions_total: int = 0,
        unlocked_writes_total: int = 0,
    ) -> dict[str, Any]:
        replayable = sum(1 for record in records if record.get("replayable"))
        failed_at = sorted(
            str(record["failed_at"])
            for record in records
            if isinstance(record.get("failed_at"), str)
        )
        return {
            "depth": len(records),
            "replayable": replayable,
            "dead": len(records) - replayable,
            "evictions_total": evictions_total,
            # Non-zero: written without the cross-process lock at least
            # once, so an 'outbox requeue-dead' can lose an update (F7).
            "unlocked_writes_total": unlocked_writes_total,
            "oldest_failed_at": failed_at[0] if failed_at else None,
        }

    def stats(self) -> dict[str, Any]:
        return self.summarize(
            self.read(),
            unlocked_writes_total=unlocked_outbox_writes(self.lock_path),
        )

    @staticmethod
    def describe(index: int, record: dict[str, Any]) -> str:
        """One line per record for an operator; never the payload."""

        error = str(record.get("error") or "")[:120].replace("\n", " ")
        state = "replayable" if record.get("replayable") else "dead"
        return (
            f"{index}\t{record.get('path')}\t{state}\t"
            f"{record.get('failed_at')}\t{error}"
        )

    def requeue_dead(
        self, *, path_filter: str | None = None, require_lock: bool = True
    ) -> int:
        """Mark dead-lettered records replayable again; returns how many.

        The whole read-modify-write runs under :meth:`locked`, so a live
        collector's appends and this requeue cannot lose each other (F7). The lock
        is *required* by default because this process has none of its own;
        ``require_lock=False`` is ``--force``: the same poll, then an unlocked
        rewrite.
        """

        with self.locked(required=require_lock, forced=not require_lock):
            records = self.read()
            requeued = 0
            skipped_truncated = 0
            for record in records:
                if record.get("replayable"):
                    continue
                if path_filter is not None and record.get("path") != path_filter:
                    continue
                if record.get("payload_truncated"):
                    # Only a digest and a 4 KB excerpt were kept, so this
                    # would post a body that is not the event.
                    skipped_truncated += 1
                    continue
                record["replayable"] = True
                previous = str(record.get("error") or "")
                record["error"] = f"requeued by operator: {previous}"[:500]
                requeued += 1
            if requeued:
                self.write(records)
        if skipped_truncated:
            LOGGER.warning(
                "left %d oversize record(s) dead in %s: only a digest of the "
                "payload was kept, so it cannot be replayed",
                skipped_truncated,
                self.path,
            )
        return requeued
