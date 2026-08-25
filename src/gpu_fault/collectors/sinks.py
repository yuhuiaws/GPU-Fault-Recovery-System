from __future__ import annotations

import gzip
import json
import logging
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Protocol
from urllib.error import HTTPError
from urllib.request import (
    Request,
)


from gpu_fault import __version__


from gpu_fault.transport.http_client import urlopen

LOGGER = logging.getLogger(__name__)


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
        self._outbox_lock = Lock()

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
        self._replay_outbox()
        return result

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
                if exc.code < 500 and exc.code != 429:
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
                self._write_outbox(records[-self.outbox_max_records :])
            return True
        except OSError:
            LOGGER.exception("cannot persist collector outbox event")
            return False

    def _read_outbox(self) -> list[dict[str, Any]]:
        if self.outbox_path is None or not self.outbox_path.exists():
            return []
        records = []
        for line in self.outbox_path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                records.append(value)
        return records

    def _write_outbox(self, records: list[dict[str, Any]]) -> None:
        if self.outbox_path is None:
            return
        self.outbox_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.outbox_path.with_suffix(self.outbox_path.suffix + ".tmp")
        temporary.write_text(
            "".join(
                json.dumps(item, separators=(",", ":"), default=str) + "\n"
                for item in records
            ),
            encoding="utf-8",
        )
        os.replace(temporary, self.outbox_path)

    def _replay_outbox(self) -> None:
        if self.outbox_path is None:
            return
        with self._outbox_lock:
            records = self._read_outbox()
            if not records:
                return
            kept = []
            replayed = 0
            deadline = time.monotonic() + self.outbox_replay_budget_seconds
            for record in records:
                if (
                    not record.get("replayable")
                    or replayed >= self.outbox_replay_batch_size
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
                    record["error"] = f"{type(exc).__name__}: {exc}"
                    kept.append(record)
                replayed += 1
            if replayed:
                LOGGER.info(
                    "collector outbox replay: %d attempted, %d still buffered",
                    replayed,
                    len(kept),
                )
            self._write_outbox(kept)


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
