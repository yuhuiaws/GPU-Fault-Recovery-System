from __future__ import annotations

import contextlib
import fcntl
import gzip
import hashlib
import json
import logging
import os
import random
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from threading import Lock, Thread
from typing import Any, Callable, Protocol
from urllib.error import HTTPError
from urllib.request import (
    Request,
)


from gpu_fault import __version__


from gpu_fault.transport.http_client import urlopen

LOGGER = logging.getLogger(__name__)


def is_retryable_delivery_status(status_code: int | None) -> bool:
    """Whether a control-plane response means "try the same event again later".

    The one rule both delivery layers use (F-G1): no status (network failure),
    any 5xx, and the three 4xx codes that mean "not now" rather than "never"
    (408 request timeout, 425 too early, 429 too many requests). Every other
    4xx is a verdict on the event itself; replaying it forever only blocks the
    records behind it.
    """

    return status_code is None or status_code >= 500 or status_code in {408, 425, 429}


#: Authentication verdicts a collector treats as "not now" (ARCH-G2). A cluster
#: token rotates, and for the length of the rotation window every node gets a
#: 401/403 from a control plane that will accept the same event a minute later.
#: A live token-drift incident held every node's fault stream for exactly that
#: long because these were dead-lettered. The completion outbox keeps the
#: stricter shared rule above: its records are node-action results whose
#: token is per workflow, not per cluster.
AUTH_TRANSIENT_STATUSES = frozenset({401, 403})

#: Upper bound on a ``Retry-After`` the client will honour. Any proxy or load
#: balancer between a node and the control plane can answer 429 with an hour,
#: and a collector that sleeps that long inside ``post()`` stops reading its
#: source (a live 429 with ``Retry-After: 3600`` made ``post`` sleep 3600 s).
#: Past the cap the record is buffered and replayed instead of held in memory.
RETRY_AFTER_CAP_SECONDS = 30.0

#: Payload fields that already identify one event, in preference order. The
#: value becomes the ``Idempotency-Key`` the control plane dedupes on, and it
#: is also what makes a request safe for the transport pool to retry on a
#: stale keep-alive connection. A payload with none of them gets one attempt.
IDEMPOTENCY_ID_FIELDS = (
    "event_id",
    "request_id",
    "batch_id",
    "observation_id",
    "summary_id",
    "record_id",
    # gpu-inventory snapshots and CloudWatch HMA log events used to fall
    # through to the single-attempt path, so every idle-closed connection
    # sent them straight to the outbox with no server-side dedup key.
    "snapshot_id",
    "log_event_id",
)

#: Timestamp fields that make an ``attempt_id`` unique for one observation.
_IDEMPOTENCY_TIME_FIELDS = ("observed_at", "detected_at", "ended_at")

#: Body headers a bodiless receipt GET must not carry.
_RECEIPT_GET_STRIPPED_HEADERS = frozenset(
    {"content-type", "content-encoding", "idempotency-key"}
)


def is_retryable_collector_status(status_code: int | None) -> bool:
    """Whether a collector may try the same event again later.

    The shared rule plus the two authentication codes: a token rotation
    window is transient, and the event behind it is still real.
    """

    return (
        is_retryable_delivery_status(status_code)
        or status_code in AUTH_TRANSIENT_STATUSES
    )


class CollectorError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        buffered: bool = False,
        replayable: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.buffered = buffered
        self.replayable = replayable


class EventSink(Protocol):
    """The one method every sink implements, plus two optional ones.

    ``post`` is required. A sink may also offer ``deliver`` (a
    :class:`DeliveryResult` instead of an exception, see :func:`deliver_event`)
    and ``buffer_for_replay`` (persist a record straight to the durable outbox
    with no network attempt, for a collector draining its in-memory queue on
    SIGTERM). Both are optional by design so that a test double or an SQS sink
    stays a valid ``EventSink``; callers discover them with
    ``getattr(sink, "buffer_for_replay", None)`` rather than ``isinstance``.
    """

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]: ...


def event_idempotency_key(payload: dict[str, Any]) -> str | None:
    """The stable id of one event, or ``None`` if the payload carries none.

    One derivation shared by the request header and the operator-facing log
    line, so a record named in a warning is the record the control plane
    deduped on.
    """

    for name in IDEMPOTENCY_ID_FIELDS:
        if payload.get(name):
            return str(payload[name])
    if payload.get("attempt_id"):
        event_time = next(
            (payload[name] for name in _IDEMPOTENCY_TIME_FIELDS if payload.get(name)),
            None,
        )
        if event_time is not None:
            return f"{payload['attempt_id']}/{event_time}"
    # An HMA node event carries no id field of its own; the Kubernetes object's
    # name plus its ``resourceVersion`` names exactly one observed revision.
    node = payload.get("node")
    metadata = node.get("metadata") if isinstance(node, dict) else None
    if isinstance(metadata, dict):
        node_name = metadata.get("name")
        resource_version = metadata.get("resourceVersion")
        if node_name and resource_version:
            return f"node/{node_name}/{resource_version}"
    return None


class _UnparseableBody(CollectorError):
    """A 2xx whose body is not a JSON object: the outcome is unknown.

    Raised inside the request loops and never allowed to escape them. It is a
    ``CollectorError`` only so that an unforeseen escape is still a delivery
    verdict rather than the ``json.JSONDecodeError`` (a ``ValueError``) that
    used to slip past :func:`deliver_event` and tear the kernel collector's
    ``/dev/kmsg`` reader down mid-batch (ARCH-G4).
    """

    def __init__(self, status: int, body: bytes) -> None:
        excerpt = body[:120].decode("utf-8", errors="replace")
        super().__init__(
            f"control-plane returned a non-JSON-object body with HTTP {status}: "
            f"{excerpt!r}",
            status_code=status,
        )


def _parse_json_body(body: bytes, status: int) -> dict[str, Any]:
    """Decode a control-plane JSON object, or report an unknown outcome.

    Only an object counts: an array or a scalar is as unusable to the callers
    as an HTML maintenance page, and both take the same "ask again" path, so
    the ``dict`` the callers are typed for is the ``dict`` they get.
    """

    if not body:
        return {}
    try:
        parsed = json.loads(body)
    except ValueError as exc:
        raise _UnparseableBody(status, body) from exc
    if not isinstance(parsed, dict):
        raise _UnparseableBody(status, body)
    return parsed


def _read_error_detail(exc: HTTPError) -> str:
    """Drain and close an ``HTTPError`` body, returning a short detail.

    The transport's proxy fallback wraps a live socket, so a body that is not
    read and closed leaks a connection for every retried poll.
    """

    detail = str(exc)
    try:
        body = exc.read()
    except Exception:  # noqa: BLE001 - a consumed or dead body is not fatal
        body = b""
    if body:
        detail = body.decode(errors="replace")
    with contextlib.suppress(Exception):
        exc.close()
    return detail


class DeliveryStatus(StrEnum):
    """What became of one event handed to a sink (ARCH-G3)."""

    #: The control plane accepted it.
    DELIVERED = "DELIVERED"
    #: The live post failed for a transient reason and the durable outbox took
    #: the record; it will be replayed. For a collector's cursor this is as good
    #: as delivered -- re-reading the source would only duplicate the record.
    BUFFERED = "BUFFERED"
    #: Neither delivered nor buffered: the event is lost unless the caller keeps
    #: its own cursor pinned.
    FAILED = "FAILED"


@dataclass(frozen=True)
class DeliveryResult:
    status: DeliveryStatus
    response: dict[str, Any] | None = None
    error: CollectorError | None = None

    @property
    def delivered(self) -> bool:
        return self.status is DeliveryStatus.DELIVERED

    @property
    def buffered(self) -> bool:
        return self.status is DeliveryStatus.BUFFERED

    @property
    def failed(self) -> bool:
        return self.status is DeliveryStatus.FAILED

    def raise_for_failure(self) -> None:
        """Re-raise the sink's error when the event went nowhere."""

        if not self.failed:
            return
        if self.error is not None:
            raise self.error
        raise CollectorError("collector event delivery failed")


def _result_for_error(exc: CollectorError) -> DeliveryResult:
    if exc.buffered and exc.replayable:
        return DeliveryResult(DeliveryStatus.BUFFERED, error=exc)
    return DeliveryResult(DeliveryStatus.FAILED, error=exc)


def deliver_event(
    sink: EventSink, path: str, payload: dict[str, Any]
) -> DeliveryResult:
    """Hand ``payload`` to ``sink`` and say what became of it, without raising.

    "Buffered means delivered" is the sink's contract, owned here rather than
    re-derived by each collector (ARCH-G3): only the kernel collector used to
    read ``exc.buffered and exc.replayable``, the node log collector rolled its
    journal cursor back and re-read the same window every poll, and the Fabric
    Manager collector pinned its offset on the first SXID the outbox had
    already taken. Sinks that implement ``deliver`` answer directly; any other
    ``post``-only sink is wrapped. Exceptions other than ``CollectorError``
    still propagate: they are the caller's failure, not a delivery verdict.
    """

    deliver = getattr(sink, "deliver", None)
    if callable(deliver):
        result = deliver(path, payload)
        if isinstance(result, DeliveryResult):
            return result
    try:
        response = sink.post(path, payload)
    except CollectorError as exc:
        return _result_for_error(exc)
    return DeliveryResult(DeliveryStatus.DELIVERED, response=response)


def deliver_or_raise(
    sink: EventSink,
    path: str,
    payload: dict[str, Any],
    *,
    logger: logging.Logger,
    what: str,
) -> DeliveryResult:
    """:func:`deliver_event` plus the one reaction every collector shares.

    ``FAILED`` re-raises the sink's error, so a caller that pins a cursor on
    an exception keeps doing so. ``BUFFERED`` is a delivery for the caller's
    purposes but not silent: it logs one warning naming ``what``, the event's
    id and the transport error, which is the only place an operator learns
    that a node is running on its durable outbox.
    """

    result = deliver_event(sink, path, payload)
    result.raise_for_failure()
    if result.buffered:
        logger.warning(
            "%s could not be delivered live and was persisted to the collector "
            "outbox (id=%s): %s",
            what,
            event_idempotency_key(payload) or "unkeyed",
            result.error,
        )
    return result


#: How much of an oversize payload the outbox keeps. A 413 is the control
#: plane refusing the body for its size, so keeping it whole re-parses and
#: re-serialises megabytes on every later rewrite for a record that can never
#: go out as it stands (F4). The digest identifies the event, the byte count
#: says how far over it was, and the excerpt is what an operator reads.
OVERSIZE_PAYLOAD_EXCERPT_BYTES = 4096

#: What a compaction leaves behind, as a fraction of ``outbox_max_records``.
#: Compacting back to exactly the ceiling made every append past it rewrite the
#: whole file, which is F4's headline scenario (a saturated 1000-record
#: backlog) and was slower than the code it replaced. Dropping to 90% buys
#: ``max/10`` free appends per rewrite.
OUTBOX_COMPACTION_FLOOR_RATIO = 0.9

#: One WARNING per this many writes that ran without the cross-process lock.
#: A single warning per path hid every later unlocked write, including a
#: transient failure that has since become permanent.
UNLOCKED_WRITE_WARN_INTERVAL = 100

#: Writes per outbox lock path that ran without the lock, and outbox paths whose
#: write failed, both counted so the logs stay bounded and an operator can see
#: how long a degradation has lasted.
_UNLOCKED_WRITES: dict[str, int] = {}
_OUTBOX_WRITE_FAILURES: dict[str, int] = {}
_OUTBOX_COUNTER_LOCK = Lock()


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
    """The cross-process outbox lock could not be taken.

    Raised only for callers that asked for ``required=True``: a collector
    degrades to its in-process lock rather than dropping an event, but an
    operator command has no in-process lock to degrade to, so for it the
    missing lock is the F7 race itself and must stop the command.
    """


def digest_oversize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """A bounded stand-in for a payload the control plane refused as too large.

    Never the whole body: a digest an operator can correlate with the source,
    the size that was rejected, and the first
    :data:`OVERSIZE_PAYLOAD_EXCERPT_BYTES` bytes of it.
    """

    body = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode()
    return {
        # The event's own id, kept so a truncated record is still identifiable
        # in the control plane's logs even though its body is gone.
        "payload_event_key": event_idempotency_key(payload),
        "payload_sha256": hashlib.sha256(body).hexdigest(),
        "payload_bytes": len(body),
        "payload_excerpt": body[:OVERSIZE_PAYLOAD_EXCERPT_BYTES].decode(
            "utf-8", errors="replace"
        ),
    }


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
    def locked(self, *, required: bool = False) -> Iterator[None]:
        """Hold the cross-process outbox lock for one read-modify-write.

        Not re-entrant: ``flock`` is held per open file description, so two
        nested :meth:`locked` blocks in one process would deadlock. Callers
        take it once around the whole read-modify-write.

        With ``required=False`` (the collector's write path) a filesystem with
        no ``flock`` support, or a ``.lock`` this process cannot open, must not
        fail a post: the body runs anyway, serialised by the caller's
        in-process lock alone, counted in :func:`unlocked_outbox_writes` and
        warned about every :data:`UNLOCKED_WRITE_WARN_INTERVAL` writes.

        With ``required=True`` (an operator command, which has no in-process
        lock to fall back on) the same failure raises
        :class:`OutboxLockUnavailable` and the body never runs.

        A failure to create the outbox *directory* is not a lock problem and is
        left to the caller: the append or rewrite that follows would fail with
        the same error, and reporting it as a missing lock hid a full or
        read-only volume behind a warning about concurrency.
        """

        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle: int | None = None
        try:
            handle = os.open(
                self.lock_path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600
            )
            fcntl.flock(handle, fcntl.LOCK_EX)
        except OSError as exc:
            if handle is not None:
                with contextlib.suppress(OSError):
                    os.close(handle)
                handle = None
            if required:
                raise OutboxLockUnavailable(
                    exc.errno or 0,
                    f"cannot take the collector outbox lock {self.lock_path}: {exc}",
                ) from exc
            self._warn_unlocked_write(exc)
        try:
            yield
        finally:
            if handle is not None:
                with contextlib.suppress(OSError):
                    fcntl.flock(handle, fcntl.LOCK_UN)
                with contextlib.suppress(OSError):
                    os.close(handle)

    def _warn_unlocked_write(self, exc: OSError) -> None:
        total = _bump(_UNLOCKED_WRITES, str(self.lock_path))
        if total > 1 and total % UNLOCKED_WRITE_WARN_INTERVAL:
            return
        LOGGER.warning(
            "collector outbox lock %s is unavailable (%s); %d write(s) so far have "
            "run on the in-process lock alone, so a concurrent 'outbox "
            "requeue-dead' can lose an update",
            self.lock_path,
            exc,
            total,
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
                # A crash between an append and its fsync leaves a torn last
                # line. Skipping it loses that one record; crashing here would
                # take the whole outbox and the collector with it.
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

        The append path needs a depth to compare against
        ``outbox_max_records``, and counting newlines is what keeps a buffered
        event from re-parsing the whole backlog (F4). A torn final line with
        no newline is not counted; :meth:`read` drops it too.
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
        full re-parse and rewrite per event (F4). A crash between the write
        and the ``fsync`` may lose at most this record, and a torn line is
        skipped by :meth:`read`; the whole backlog behind it survives, which a
        rewrite of the entire file could not promise.

        "At most this record" is why the torn tail gets a newline of its own
        first: appending straight onto a fragment left by an unclean shutdown
        concatenated the two into one unparseable line, so ``read()`` dropped
        the *new* record too while the collector had already been told it was
        buffered and had advanced its cursor past a real event.
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
        """

        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
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
            # Non-zero means this outbox has been written without the
            # cross-process lock, so an 'outbox requeue-dead' can lose an
            # update (F7 degraded).
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

        The whole read-modify-write runs under :meth:`locked` so an operator
        running this against a live collector's outbox cannot lose the sink's
        appends, and the sink's next rewrite cannot lose this requeue (F7).
        The lock is *required* by default: without it this is the F7 race, and
        an operator process has no in-process lock to fall back on.
        ``require_lock=False`` is the ``--force`` escape hatch.
        """

        with self.locked(required=require_lock):
            records = self.read()
            requeued = 0
            skipped_truncated = 0
            for record in records:
                if record.get("replayable"):
                    continue
                if path_filter is not None and record.get("path") != path_filter:
                    continue
                if record.get("payload_truncated"):
                    # Only a digest and a 4 KB excerpt of this payload were
                    # kept, so replaying it would post a body that is not the
                    # event. It stays dead and inspectable.
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


@dataclass(frozen=True)
class _OutboxReplayResult:
    attempted: int = 0
    delivered: int = 0
    replayable_remaining: int = 0


#: Identity of one buffered record: path, when it was buffered and its payload.
#: Used to reconcile a replay's outcome with a file another writer may have
#: appended to while the replay's requests were in flight, because a record's
#: position is not stable across a concurrent append or eviction.
_OutboxRecordKey = tuple[str, str, str]


def _outbox_record_key(record: dict[str, Any]) -> _OutboxRecordKey:
    return (
        str(record.get("path")),
        str(record.get("failed_at")),
        json.dumps(
            record.get("payload"),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ),
    )


def _apply_replay_outcome(
    records: list[dict[str, Any]],
    removals: list[_OutboxRecordKey],
    updates: list[tuple[_OutboxRecordKey, dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Drop the delivered records and re-stamp the failed ones, in file order.

    Anything not named by ``removals``/``updates`` is kept verbatim, which is
    how a record buffered while the replay was in flight survives the rewrite.
    Counts are honoured rather than keys alone so two identical records are
    not both dropped for one delivery.
    """

    remaining_removals: dict[_OutboxRecordKey, int] = {}
    for key in removals:
        remaining_removals[key] = remaining_removals.get(key, 0) + 1
    pending_updates: dict[_OutboxRecordKey, list[dict[str, Any]]] = {}
    for key, update in updates:
        pending_updates.setdefault(key, []).append(update)
    kept: list[dict[str, Any]] = []
    for record in records:
        key = _outbox_record_key(record)
        if remaining_removals.get(key):
            remaining_removals[key] -= 1
            continue
        queued = pending_updates.get(key)
        if queued:
            record.update(queued.pop(0))
        kept.append(record)
    return kept


class HttpEventSink:
    """Small dependency-free JSON client with bounded retry."""

    def __init__(
        self,
        base_url: str,
        *,
        bearer_token: str | None = None,
        timeout_seconds: float = 10,
        max_attempts: int = 4,
        sleep: Callable[[float], None] = time.sleep,
        jitter: Callable[[float, float], float] = random.uniform,
        outbox_path: str | None = None,
        outbox_max_records: int = 1000,
        outbox_replay_batch_size: int = 10,
        outbox_replay_budget_seconds: float | None = None,
        outbox_replay_background_interval_seconds: float = 0.25,
        processor_receipt_timeout_seconds: float = 120,
        processor_receipt_poll_seconds: float = 0.25,
        gzip_min_bytes: int | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.bearer_token = bearer_token
        if timeout_seconds <= 0:
            raise ValueError("collector HTTP timeout must be positive")
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        self.sleep = sleep
        self.jitter = jitter
        self.outbox_path = Path(outbox_path) if outbox_path else None
        self.outbox_max_records = outbox_max_records
        if outbox_replay_batch_size <= 0:
            raise ValueError("collector outbox replay batch size must be positive")
        self.outbox_replay_batch_size = outbox_replay_batch_size
        # Replaying the backlog is best-effort catch-up work and must
        # never become the reason a live fault is late, so it runs under
        # a wall-clock budget with one attempt per record.
        self.outbox_replay_budget_seconds = (
            outbox_replay_budget_seconds
            if outbox_replay_budget_seconds is not None
            else float(
                os.getenv(
                    "GPU_FAULT_COLLECTOR_OUTBOX_REPLAY_BUDGET_SECONDS",
                    "5",
                )
            )
        )
        if self.outbox_replay_budget_seconds <= 0:
            raise ValueError("collector outbox replay budget must be positive")
        if outbox_replay_background_interval_seconds < 0:
            raise ValueError(
                "collector outbox background replay interval cannot be negative"
            )
        self.outbox_replay_background_interval_seconds = (
            outbox_replay_background_interval_seconds
        )
        if (
            processor_receipt_timeout_seconds <= 0
            or processor_receipt_poll_seconds <= 0
        ):
            raise ValueError(
                "processor receipt timeout and poll interval must be positive"
            )
        self.processor_receipt_timeout_seconds = processor_receipt_timeout_seconds
        self.processor_receipt_poll_seconds = processor_receipt_poll_seconds
        # Telemetry payloads are highly repetitive JSON and compress by
        # roughly an order of magnitude. Bodies below the floor are left
        # alone because the CPU cost outweighs a sub-packet saving.
        self.gzip_min_bytes = (
            gzip_min_bytes
            if gzip_min_bytes is not None
            else int(os.getenv("GPU_FAULT_COLLECTOR_GZIP_MIN_BYTES", "4096"))
        )
        if self.gzip_min_bytes < 0:
            raise ValueError("collector gzip threshold must not be negative")
        self.outbox_evictions_total = 0
        self._outbox_lock = Lock()
        # Depth of the outbox as this process last saw it, so an append does
        # not have to parse the backlog to know whether it must compact.
        # ``None`` means "not known yet"; it is recomputed from the file on the
        # first append and refreshed by every full read.
        self._outbox_line_count: int | None = None
        self._outbox_replay_state_lock = Lock()
        self._outbox_replay_active = False
        self._outbox_replay_requested = False
        self._outbox_replay_thread: Thread | None = None

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Deliver ``payload``, then catch the backlog up if there is time.

        The live event goes first. Replaying before it meant a fresh XID
        queued behind up to ``outbox_replay_batch_size`` stale records,
        each of which could still burn ``max_attempts`` timeouts and --
        for the two receipt-bearing paths -- a
        ``processor_receipt_timeout_seconds`` poll, so a node that had
        just come back from a network partition could delay its own
        fault report by minutes while the kernel collector's /dev/kmsg
        reader fell behind. Replay only runs after a successful
        delivery: if the fresh post failed, the link is still down and
        replaying would only waste the caller's time.
        """

        result = self._post_with_retry(path, payload, buffer_failure=True)
        self._kick_outbox_replay()
        return result

    def deliver(self, path: str, payload: dict[str, Any]) -> DeliveryResult:
        """``post`` that answers with a :class:`DeliveryResult` instead of raising.

        A record the outbox took is ``BUFFERED``; anything else that failed
        is ``FAILED`` with the error attached. See :func:`deliver_event`.
        """

        try:
            response = self.post(path, payload)
        except CollectorError as exc:
            return _result_for_error(exc)
        return DeliveryResult(DeliveryStatus.DELIVERED, response=response)

    def outbox_stats(self) -> dict[str, Any]:
        """Depth, replayable/dead split, evictions and the oldest failure."""

        if self.outbox_path is None:
            return OutboxFile.summarize([], evictions_total=self.outbox_evictions_total)
        with self._outbox_lock:
            return OutboxFile.summarize(
                self._read_outbox(),
                evictions_total=self.outbox_evictions_total,
                unlocked_writes_total=self.outbox_unlocked_writes_total,
            )

    def wait_for_outbox_replay(self, timeout_seconds: float = 5) -> bool:
        """Wait for an already-started background replay worker.

        Collectors do not call this on their hot path. It exists for
        acceptance probes and orderly embedders that need to prove the
        durable queue has converged before they exit.
        """

        if timeout_seconds < 0:
            raise ValueError("outbox replay wait timeout cannot be negative")
        with self._outbox_replay_state_lock:
            active = self._outbox_replay_active
            thread = self._outbox_replay_thread
        if not active:
            return True
        if thread is None:
            return False
        thread.join(timeout=timeout_seconds)
        with self._outbox_replay_state_lock:
            return not self._outbox_replay_active

    def _kick_outbox_replay(self) -> None:
        if self.outbox_path is None:
            return
        with self._outbox_replay_state_lock:
            if self._outbox_replay_active:
                self._outbox_replay_requested = True
                return
            self._outbox_replay_active = True
            self._outbox_replay_requested = False
        try:
            outcome = self._replay_outbox()
        except Exception:
            # Catch-up work that failed is never a verdict on the event that
            # just went out: the live post already succeeded. A full
            # ``/var/lib`` used to raise ``OSError`` out of ``post()`` and
            # ``deliver()``, which reopened ``/dev/kmsg`` and dropped
            # everything written in between (ARCH-G4); a reconcile bug must
            # not fail a delivered post either. The next successful post
            # kicks a fresh replay.
            LOGGER.exception(
                "collector outbox replay failed after a delivered post (path=%s)",
                self.outbox_path,
            )
            with self._outbox_replay_state_lock:
                self._outbox_replay_active = False
                self._outbox_replay_thread = None
            return
        if not self._continue_outbox_replay(outcome):
            return
        thread = Thread(
            target=self._background_outbox_replay,
            name="gpu-fault-collector-outbox-replay",
            daemon=True,
        )
        with self._outbox_replay_state_lock:
            self._outbox_replay_thread = thread
        try:
            thread.start()
        except Exception:
            with self._outbox_replay_state_lock:
                self._outbox_replay_active = False
                self._outbox_replay_thread = None
            raise

    def _continue_outbox_replay(self, outcome: _OutboxReplayResult) -> bool:
        with self._outbox_replay_state_lock:
            requested = self._outbox_replay_requested
            self._outbox_replay_requested = False
            if outcome.replayable_remaining and (outcome.delivered or requested):
                return True
            self._outbox_replay_active = False
            self._outbox_replay_thread = None
            return False

    def _background_outbox_replay(self) -> None:
        try:
            while True:
                if self.outbox_replay_background_interval_seconds:
                    time.sleep(self.outbox_replay_background_interval_seconds)
                outcome = self._replay_outbox()
                if not self._continue_outbox_replay(outcome):
                    return
        except Exception:
            LOGGER.exception("collector background outbox replay failed")
            with self._outbox_replay_state_lock:
                self._outbox_replay_active = False
                self._outbox_replay_thread = None

    def _post_with_retry(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        buffer_failure: bool,
        max_attempts: int | None = None,
        poll_receipt: bool = True,
    ) -> dict[str, Any]:
        body = json.dumps(payload, separators=(",", ":"), default=str).encode()
        headers = {
            "Content-Type": "application/json",
            "User-Agent": f"gpu-fault-collector/{__version__}",
        }
        idempotency_key = event_idempotency_key(payload)
        if idempotency_key is not None:
            headers["Idempotency-Key"] = idempotency_key
        if len(body) >= self.gzip_min_bytes:
            body = gzip.compress(body, compresslevel=6)
            headers["Content-Encoding"] = "gzip"
        cluster_id = payload.get("cluster_id")
        if cluster_id:
            headers["X-GPU-Fault-Cluster-ID"] = str(cluster_id)
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"

        attempts = self.max_attempts if max_attempts is None else max(1, max_attempts)
        if idempotency_key is None:
            attempts = 1
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            retry_after = 0.0
            request = Request(
                f"{self.base_url}{path}",
                data=body,
                headers=headers,
                method="POST",
            )
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    result = response.read()
                    status = int(getattr(response, "status", 200))
                    parsed: dict[str, Any] = _parse_json_body(result, status)
                    if (
                        status == 202
                        and parsed.get("processor_request_id")
                        and self._requires_processor_receipt(path)
                        and poll_receipt
                    ):
                        return self._poll_processor_receipt(
                            parsed,
                            headers=headers,
                        )
                    return parsed
            except _UnparseableBody as exc:
                # An unparseable 2xx is an unknown outcome, not a verdict on
                # the event: the control plane may well have accepted it. It
                # walks the same ladder as a network failure and, if the body
                # never becomes usable, lands in the outbox as replayable --
                # the retry and the replay carry the same Idempotency-Key, so
                # the server dedupes a request that did get through. Treating
                # it as terminal gave a garbage 200 a harsher verdict than a
                # 503 and left the event with no persistent record at all.
                last_error = exc
                LOGGER.warning(
                    "collector event will be retried: %s (path=%s)",
                    exc,
                    path,
                )
            except HTTPError as exc:
                last_error = exc
                if not is_retryable_collector_status(exc.code):
                    detail = exc.read().decode(errors="replace")
                    buffered = False
                    if buffer_failure:
                        buffered = self._buffer_event(
                            path,
                            payload,
                            replayable=False,
                            error=f"HTTP {exc.code}: {detail}",
                            # 413 is a verdict on the body's size: keeping it
                            # whole costs a re-parse and a rewrite of megabytes
                            # for a record that can never be posted as it
                            # stands (F4).
                            oversize=exc.code == 413,
                        )
                    raise CollectorError(
                        f"collector event rejected ({exc.code}): {detail}",
                        status_code=exc.code,
                        buffered=buffered,
                    ) from exc
                try:
                    retry_after = float(exc.headers.get("Retry-After", "0"))
                except (TypeError, ValueError):
                    retry_after = 0.0
            except OSError as exc:
                last_error = exc
            if attempt < attempts:
                backoff = min(2 ** (attempt - 1), 8)
                if retry_after > 0:
                    # Cap first, then jitter: the cap bounds how long a
                    # collector stops reading its source, and the jitter still
                    # spreads a fleet-wide 429 across nodes.
                    delay = min(retry_after, RETRY_AFTER_CAP_SECONDS) + self.jitter(
                        0.0, min(backoff, 1.0)
                    )
                else:
                    delay = self.jitter(0.0, backoff)
                self.sleep(delay)
        error = f"control-plane delivery failed after {attempts} attempts: {last_error}"
        buffered = False
        if buffer_failure:
            buffered = self._buffer_event(
                path,
                payload,
                replayable=True,
                error=error,
            )
        raise CollectorError(
            error,
            buffered=buffered,
            replayable=buffered,
        )

    @staticmethod
    def _requires_processor_receipt(path: str) -> bool:
        return path in {
            "/v1/attempts/failure-detected",
            "/v1/attempts/terminal",
        }

    def _poll_processor_receipt(
        self,
        accepted: dict[str, Any],
        *,
        headers: dict[str, str],
    ) -> dict[str, Any]:
        status_url = accepted.get("status_url")
        if not isinstance(status_url, str) or not status_url:
            status_url = f"/v1/processor/requests/{accepted['processor_request_id']}"
        # The GET carries no body, so the POST's ``Content-Type``,
        # ``Content-Encoding: gzip`` and ``Idempotency-Key`` describe nothing:
        # authentication and cluster routing headers are all it needs.
        get_headers = {
            name: value
            for name, value in headers.items()
            if name.lower() not in _RECEIPT_GET_STRIPPED_HEADERS
        }
        deadline = time.monotonic() + self.processor_receipt_timeout_seconds
        delay = self.processor_receipt_poll_seconds
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            request = Request(
                f"{self.base_url}{status_url}",
                headers=get_headers,
                method="GET",
            )
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    payload = response.read()
                    status = int(getattr(response, "status", 200))
                    if status == 202:
                        self.sleep(delay)
                        delay = min(delay * 2, 2.0)
                        continue
                    receipt: dict[str, Any] = _parse_json_body(payload, status)
                    return receipt
            except _UnparseableBody as exc:
                # An unreadable receipt is not a completed request: ask again
                # until the deadline rather than failing a post whose 202 the
                # control plane already accepted.
                last_error = exc
                self.sleep(delay)
                delay = min(delay * 2, 2.0)
            except HTTPError as exc:
                detail = _read_error_detail(exc)
                if is_retryable_collector_status(exc.code):
                    # The 202 was accepted; a 503 from the store or a 429 on
                    # the status route says "ask again", not that the request
                    # failed. Polling stays bounded by the same deadline.
                    last_error = CollectorError(
                        f"processor receipt poll got HTTP {exc.code}: {detail}",
                        status_code=exc.code,
                    )
                    self.sleep(delay)
                    delay = min(delay * 2, 2.0)
                    continue
                raise CollectorError(
                    f"processor request completed with HTTP {exc.code}: {detail}",
                    status_code=exc.code,
                ) from exc
            except OSError as exc:
                last_error = exc
                self.sleep(delay)
                delay = min(delay * 2, 2.0)
        raise CollectorError(
            "processor receipt timed out for "
            f"{accepted['processor_request_id']}: {last_error}"
        )

    def buffer_for_replay(self, path: str, payload: dict[str, Any]) -> bool:
        """Persist one record for later replay without attempting a post.

        For a collector draining its in-memory queue on SIGTERM: a live post
        per record would spend the retry ladder (and, on the two
        receipt-bearing paths, a receipt poll) against a control plane that may
        already be gone, so a shutdown budget of a second buys a handful of
        records instead of the whole queue. Each record costs one appended
        line, and the next successful post replays them in file order.

        Returns ``False`` -- never raises -- when there is no outbox
        configured or the write failed, so a drain loop can count what it
        could not save.
        """

        return self._buffer_event(
            path,
            payload,
            replayable=True,
            error="buffered at shutdown",
        )

    def _buffer_event(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        replayable: bool,
        error: str,
        oversize: bool = False,
    ) -> bool:
        if self.outbox_path is None:
            return False
        record: dict[str, Any] = {
            "path": path,
            "payload": digest_oversize_payload(payload) if oversize else payload,
            "replayable": replayable,
            "error": error,
            "failed_at": datetime.now(timezone.utc).isoformat(),
        }
        if oversize:
            # Read by ``requeue_dead`` and by the replay guard: what is left of
            # this payload is a digest, so it must never be posted as an event.
            record["payload_truncated"] = True
        outbox = OutboxFile(self.outbox_path)
        try:
            # Two locks: ``_outbox_lock`` serialises this process's collector
            # and replay threads, ``locked()`` serialises this process against
            # ``gpu-fault-collector outbox requeue-dead`` (F7). Always in this
            # order, everywhere.
            with self._outbox_lock, outbox.locked():
                if self._outbox_line_count is None:
                    self._outbox_line_count = outbox.count_lines()
                outbox.append(record)
                self._outbox_line_count += 1
                if self._outbox_line_count > self.outbox_max_records:
                    self._compact_outbox_locked(outbox)
            return True
        except OSError:
            failures = _bump(_OUTBOX_WRITE_FAILURES, str(self.outbox_path))
            if failures == 1:
                LOGGER.exception("cannot persist collector outbox event")
            else:
                # A full or read-only volume fails for every record; one
                # traceback per outbox is diagnosis, thousands are noise.
                LOGGER.warning(
                    "cannot persist collector outbox event (%d failure(s) for %s)",
                    failures,
                    self.outbox_path,
                )
            return False

    @property
    def outbox_unlocked_writes_total(self) -> int:
        """Writes that ran without the cross-process lock (F7 degraded)."""

        if self.outbox_path is None:
            return 0
        return unlocked_outbox_writes(OutboxFile(self.outbox_path).lock_path)

    def _compact_outbox_locked(self, outbox: OutboxFile) -> None:
        """Rewrite the outbox down to 90% of ``outbox_max_records``.

        The only place a buffered event pays for the whole file, and only once
        the append pushed it past the ceiling. Compacting back to exactly the
        ceiling meant a saturated outbox -- F4's own scenario -- rewrote itself
        on every append; leaving 90% behind makes the next ``max/10`` appends
        free. Callers hold both locks.
        """

        records = outbox.read()
        floor = max(1, int(self.outbox_max_records * OUTBOX_COMPACTION_FLOOR_RATIO))
        evicted = len(records) - floor
        if evicted > 0:
            # The oldest records go first; they are the least likely to still
            # be wanted, but they are still loss and used to be dropped
            # silently (ARCH-G2).
            self.outbox_evictions_total += evicted
            LOGGER.warning(
                "collector outbox is full; evicted %d oldest record(s) "
                "(max_records=%d, kept=%d, evictions_total=%d, path=%s)",
                evicted,
                self.outbox_max_records,
                floor,
                self.outbox_evictions_total,
                self.outbox_path,
            )
        kept = records[-floor:]
        outbox.write(kept)
        self._outbox_line_count = len(kept)

    def _read_outbox(self) -> list[dict[str, Any]]:
        if self.outbox_path is None:
            return []
        records = OutboxFile(self.outbox_path).read()
        self._outbox_line_count = len(records)
        return records

    def _write_outbox(self, records: list[dict[str, Any]]) -> None:
        if self.outbox_path is None:
            return
        OutboxFile(self.outbox_path).write(records)
        self._outbox_line_count = len(records)

    def _replay_outbox(self) -> _OutboxReplayResult:
        """Deliver one batch of buffered records, then reconcile the file.

        The file is read under ``_outbox_lock``, every request is made with
        the lock released, and the outcome is applied to a *fresh* read under
        the lock again. Holding the lock across the network meant a live post
        that failed in the collector thread waited for the whole replay budget
        plus one in-flight client timeout before it could buffer (F13); the
        re-read is what keeps that concurrent ``_buffer_event`` from being
        overwritten by a rewrite built from a stale snapshot.
        """

        if self.outbox_path is None:
            return _OutboxReplayResult()
        outbox = OutboxFile(self.outbox_path)
        with self._outbox_lock, outbox.locked():
            records = self._read_outbox()
        if not records:
            return _OutboxReplayResult()
        attempted = 0
        delivered = 0
        removals: list[_OutboxRecordKey] = []
        updates: list[tuple[_OutboxRecordKey, dict[str, Any]]] = []
        deadline = time.monotonic() + self.outbox_replay_budget_seconds
        for record in records:
            if (
                not record.get("replayable")
                # Only a digest of this payload survives, so posting it would
                # send a body that is not the event. ``requeue_dead`` refuses
                # to resurrect one; this guard covers a hand-edited file.
                or record.get("payload_truncated")
                or attempted >= self.outbox_replay_batch_size
                or time.monotonic() >= deadline
            ):
                continue
            key = _outbox_record_key(record)
            try:
                # One attempt and no receipt poll: a record that
                # cannot go through right now stays in the outbox
                # for the next cycle, which is cheaper than
                # spending the retry ladder plus a 120s receipt
                # wait on it while fresh events pile up behind.
                self._post_with_retry(
                    str(record["path"]),
                    dict(record["payload"]),
                    buffer_failure=False,
                    max_attempts=1,
                    poll_receipt=False,
                )
            except Exception as exc:
                update: dict[str, Any] = {"error": f"{type(exc).__name__}: {exc}"[:500]}
                status = getattr(exc, "status_code", None)
                if (
                    isinstance(exc, CollectorError)
                    and isinstance(status, int)
                    and 400 <= status < 500
                    and not is_retryable_collector_status(status)
                ):
                    # A verdict on the record itself (ARCH-G2): keeping it
                    # replayable let ten poisoned heads fill every replay
                    # window forever. It stays in the file, dead, for
                    # ``gpu-fault-collector outbox`` to inspect or requeue.
                    update["replayable"] = False
                    LOGGER.warning(
                        "collector outbox record dead-lettered at replay: "
                        "path=%s status=%s failed_at=%s",
                        record.get("path"),
                        status,
                        record.get("failed_at"),
                    )
                updates.append((key, update))
            else:
                delivered += 1
                removals.append(key)
            attempted += 1
        if not removals and not updates:
            # Nothing was attempted (all dead, or the budget was already
            # spent), so the file on disk is still correct as it stands.
            return _OutboxReplayResult(
                replayable_remaining=sum(
                    1 for record in records if record.get("replayable")
                )
            )
        with self._outbox_lock, outbox.locked():
            kept = _apply_replay_outcome(self._read_outbox(), removals, updates)
            replayable_remaining = sum(1 for record in kept if record.get("replayable"))
            if attempted:
                LOGGER.info(
                    "collector outbox replay: %d attempted, %d delivered, "
                    "%d replayable and %d total still buffered",
                    attempted,
                    delivered,
                    replayable_remaining,
                    len(kept),
                )
            self._write_outbox(kept)
        return _OutboxReplayResult(
            attempted=attempted,
            delivered=delivered,
            replayable_remaining=replayable_remaining,
        )


class SqsEventSink:
    """Queues normalized collector requests for private in-cluster delivery."""

    def __init__(self, queue_url: str, client: Any | None = None) -> None:
        self.queue_url = queue_url
        if client is None:
            try:
                import boto3
            except ImportError as exc:
                raise CollectorError(
                    "install gpu-fault-control-plane[hyperpod] for SQS support"
                ) from exc
            client = boto3.client("sqs")
        self.client = client

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = self.client.send_message(
            QueueUrl=self.queue_url,
            MessageBody=json.dumps(
                {"path": path, "payload": payload},
                separators=(",", ":"),
                default=str,
            ),
        )
        return {"message_id": response.get("MessageId")}
