from __future__ import annotations

import gzip
import json
import logging
import os
import random
import time
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
    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]: ...


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


@dataclass(frozen=True)
class OutboxFile:
    """The durable NDJSON outbox one collector writes (ARCH-G2).

    Shared by the sink and the ``gpu-fault-collector outbox`` command so an
    operator reads exactly the records the sink will replay.
    """

    path: Path

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records = []
        text = self.path.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                records.append(value)
        return records

    def write(self, records: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            "".join(
                json.dumps(item, separators=(",", ":"), default=str) + "\n"
                for item in records
            ),
            encoding="utf-8",
        )
        os.replace(temporary, self.path)

    @staticmethod
    def summarize(
        records: list[dict[str, Any]], *, evictions_total: int = 0
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
            "oldest_failed_at": failed_at[0] if failed_at else None,
        }

    def stats(self) -> dict[str, Any]:
        return self.summarize(self.read())

    @staticmethod
    def describe(index: int, record: dict[str, Any]) -> str:
        """One line per record for an operator; never the payload."""

        error = str(record.get("error") or "")[:120].replace("\n", " ")
        state = "replayable" if record.get("replayable") else "dead"
        return (
            f"{index}\t{record.get('path')}\t{state}\t"
            f"{record.get('failed_at')}\t{error}"
        )

    def requeue_dead(self, *, path_filter: str | None = None) -> int:
        """Mark dead-lettered records replayable again; returns how many."""

        records = self.read()
        requeued = 0
        for record in records:
            if record.get("replayable"):
                continue
            if path_filter is not None and record.get("path") != path_filter:
                continue
            record["replayable"] = True
            previous = str(record.get("error") or "")
            record["error"] = f"requeued by operator: {previous}"[:500]
            requeued += 1
        if requeued:
            self.write(records)
        return requeued


@dataclass(frozen=True)
class _OutboxReplayResult:
    attempted: int = 0
    delivered: int = 0
    replayable_remaining: int = 0


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
                self._read_outbox(), evictions_total=self.outbox_evictions_total
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
            with self._outbox_replay_state_lock:
                self._outbox_replay_active = False
            raise
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
        idempotency_key = next(
            (
                str(payload[name])
                for name in (
                    "event_id",
                    "request_id",
                    "batch_id",
                    "observation_id",
                    "summary_id",
                    "record_id",
                )
                if payload.get(name)
            ),
            None,
        )
        if idempotency_key is None and payload.get("attempt_id"):
            event_time = next(
                (
                    payload[name]
                    for name in (
                        "observed_at",
                        "detected_at",
                        "ended_at",
                    )
                    if payload.get(name)
                ),
                None,
            )
            if event_time is not None:
                idempotency_key = f"{payload['attempt_id']}/{event_time}"
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
                    parsed = json.loads(result) if result else {}
                    if (
                        getattr(response, "status", 200) == 202
                        and isinstance(parsed, dict)
                        and parsed.get("processor_request_id")
                        and self._requires_processor_receipt(path)
                        and poll_receipt
                    ):
                        return self._poll_processor_receipt(
                            parsed,
                            headers=headers,
                        )
                    return parsed
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
                    delay = retry_after + self.jitter(0.0, min(backoff, 1.0))
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
        deadline = time.monotonic() + self.processor_receipt_timeout_seconds
        delay = self.processor_receipt_poll_seconds
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            request = Request(
                f"{self.base_url}{status_url}",
                headers=headers,
                method="GET",
            )
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    payload = response.read()
                    if getattr(response, "status", 200) == 202:
                        self.sleep(delay)
                        delay = min(delay * 2, 2.0)
                        continue
                    return json.loads(payload) if payload else {}
            except HTTPError as exc:
                detail = exc.read().decode(errors="replace")
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

    def _buffer_event(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        replayable: bool,
        error: str,
    ) -> bool:
        if self.outbox_path is None:
            return False
        record = {
            "path": path,
            "payload": payload,
            "replayable": replayable,
            "error": error,
            "failed_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            with self._outbox_lock:
                records = self._read_outbox()
                records.append(record)
                evicted = len(records) - self.outbox_max_records
                if evicted > 0:
                    # The oldest records go first; they are the least likely
                    # to still be wanted, but they are still loss and used to
                    # be dropped silently (ARCH-G2).
                    self.outbox_evictions_total += evicted
                    LOGGER.warning(
                        "collector outbox is full; evicted %d oldest record(s) "
                        "(max_records=%d, evictions_total=%d, path=%s)",
                        evicted,
                        self.outbox_max_records,
                        self.outbox_evictions_total,
                        self.outbox_path,
                    )
                self._write_outbox(records[-self.outbox_max_records :])
            return True
        except OSError:
            LOGGER.exception("cannot persist collector outbox event")
            return False

    def _read_outbox(self) -> list[dict[str, Any]]:
        if self.outbox_path is None:
            return []
        return OutboxFile(self.outbox_path).read()

    def _write_outbox(self, records: list[dict[str, Any]]) -> None:
        if self.outbox_path is None:
            return
        OutboxFile(self.outbox_path).write(records)

    def _replay_outbox(self) -> _OutboxReplayResult:
        if self.outbox_path is None:
            return _OutboxReplayResult()
        with self._outbox_lock:
            records = self._read_outbox()
            if not records:
                return _OutboxReplayResult()
            kept = []
            attempted = 0
            delivered = 0
            deadline = time.monotonic() + self.outbox_replay_budget_seconds
            for record in records:
                if (
                    not record.get("replayable")
                    or attempted >= self.outbox_replay_batch_size
                    or time.monotonic() >= deadline
                ):
                    kept.append(record)
                    continue
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
                    record["error"] = f"{type(exc).__name__}: {exc}"[:500]
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
                        record["replayable"] = False
                        LOGGER.warning(
                            "collector outbox record dead-lettered at replay: "
                            "path=%s status=%s failed_at=%s",
                            record.get("path"),
                            status,
                            record.get("failed_at"),
                        )
                    kept.append(record)
                else:
                    delivered += 1
                attempted += 1
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
