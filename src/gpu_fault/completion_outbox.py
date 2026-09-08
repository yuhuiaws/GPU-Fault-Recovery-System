from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections.abc import Callable
from typing import Any

from gpu_fault.collectors.sinks import (
    CollectorError,
    EventSink,
    HttpEventSink,
    is_retryable_delivery_status,
)

CRITICAL_COMPLETION_PATHS = frozenset(
    {
        "/v1/attempts/failure-detected",
        "/v1/attempts/terminal",
    }
)
WORKLOAD_OBSERVATION_PATH = "/v1/workload-observations"
LOGGER = logging.getLogger(__name__)
#: Per-snapshot cap on the log tail kept in the write-ahead record. A live
#: failure event carries up to ``workload_log_max_bytes`` (262 144) per rank,
#: so four ranks alone exceed ``max_bytes`` (900 000) and used to make the
#: whole event undeliverable (F1).
DEFAULT_BUFFERED_TAIL_BYTES = 8192
#: Bound on the per-key "already reported" set behind the log-once rule.
_MAX_LOGGED_APPEND_FAILURES = 1024


#: Marker in the ``post`` result: nothing was delivered live because the record
#: is still buffered, so ``replay`` owns it (F8).
DEFERRED_TO_REPLAY = "deferred_to_replay"


class CompletionOutboxFull(RuntimeError):
    pass


def completion_delivery_deferred(response: Any) -> bool:
    """``True`` when ``post`` left this event to the outbox replay.

    The caller has *not* had the event accepted by the control plane: any
    protection that depends on delivery (the watcher's emergency workload
    stop) has to stay armed.
    """

    return isinstance(response, dict) and bool(response.get(DEFERRED_TO_REPLAY))


def pointer_sized_completion_payload(
    payload: dict[str, Any],
    *,
    max_tail_bytes: int = DEFAULT_BUFFERED_TAIL_BYTES,
) -> dict[str, Any]:
    """The write-ahead copy of a critical event: pointers, not log tails.

    The live POST carries the full ``workload_log_snapshots`` -- the control
    plane is what archives the tails -- but the ConfigMap copy only has to be
    enough to *identify* the evidence after a watcher restart, so each
    snapshot keeps its ``record_id`` / ``s3_uri`` / ``sha256`` / ``truncated``
    pointers and at most ``max_tail_bytes`` of tail (measured on the encoded
    bytes, per snapshot). Without this a 4-rank attempt with ordinary logs is
    ~1 MB, ``_write_data`` raises ``CompletionOutboxFull`` before the POST is
    ever attempted, and the failure never reaches the control plane at all.

    Consequence to know about: what ``replay`` re-sends is this pointer-sized
    record, so an event delivered by replay (rather than live) reaches the
    control plane with truncated tails and ``buffered_tail_truncated`` set on
    each trimmed snapshot. That is deliberate -- the full tail is in the S3
    archive named by ``s3_uri`` and in the Pod annotation written at capture
    time -- and it is the reason the field is not simply dropped.

    The argument is never mutated; a payload with nothing to trim is returned
    as-is.
    """

    snapshots = payload.get("workload_log_snapshots")
    if not isinstance(snapshots, list):
        return payload
    trimmed: list[Any] = []
    changed = False
    for snapshot in snapshots:
        tail = snapshot.get("tail") if isinstance(snapshot, dict) else None
        if not isinstance(tail, str):
            trimmed.append(snapshot)
            continue
        raw = tail.encode("utf-8", errors="replace")
        if len(raw) <= max_tail_bytes:
            trimmed.append(snapshot)
            continue
        kept = raw[-max_tail_bytes:]
        # Cut forward to a character boundary: a slice starting inside a
        # multi-byte character would decode to U+FFFD (3 bytes each) and could
        # push the record back over the cap it is here to respect.
        start = 0
        while start < len(kept) and kept[start] & 0xC0 == 0x80:
            start += 1
        record = dict(snapshot)
        record["tail"] = kept[start:].decode("utf-8", errors="replace")
        record["buffered_tail_bytes"] = len(kept) - start
        record["buffered_tail_truncated"] = True
        trimmed.append(record)
        changed = True
    if not changed:
        return payload
    return {**payload, "workload_log_snapshots": trimmed}


def completion_delivery_disposition(exc: BaseException) -> str:
    """``"retry"`` or ``"quarantine"`` for a failed critical delivery.

    Uses the same status-code rule as the collector sink
    (``is_retryable_delivery_status``) so that the two layers cannot disagree
    about a 409 or a 422 again (P0-64B). An exception that carries no status
    -- a socket error, an unexpected bug -- is retried; the attempt counter
    bounds it.
    """

    if isinstance(exc, CollectorError):
        if exc.replayable:
            return "retry"
        return (
            "retry" if is_retryable_delivery_status(exc.status_code) else "quarantine"
        )
    code = getattr(exc, "code", None)
    if isinstance(code, int):
        return "retry" if is_retryable_delivery_status(code) else "quarantine"
    return "retry"


class KubernetesCompletionOutbox:
    """ConfigMap-backed write-ahead delivery for critical watcher events.

    The write-ahead record is durability, not permission to deliver: it is
    pointer-sized (see ``pointer_sized_completion_payload``) and a write that
    fails is counted in ``append_failures_total`` and logged once per key,
    after which the live POST is attempted anyway. Only when both the buffer
    and the delivery fail does ``post`` raise, and then it raises the buffer
    error -- an unwritable outbox is the actionable cause -- with the delivery
    error chained onto it.
    """

    DATA_KEY = "events.json"
    ATTEMPTS_KEY = "active-attempts.json"

    def __init__(
        self,
        core_api: Any,
        sink: EventSink,
        *,
        namespace: str = "gpu-fault-system",
        name: str = "gpu-fault-completion-watcher-outbox",
        max_records: int = 256,
        max_bytes: int = 900_000,
        max_buffered_tail_bytes: int = DEFAULT_BUFFERED_TAIL_BYTES,
        replay_batch_size: int = 32,
        replay_budget_seconds: float = 10.0,
        max_replay_attempts: int = 20,
        monotonic: Callable[[], float] = time.monotonic,
        now: Callable[[], float] = time.time,
    ) -> None:
        if max_records < 1 or max_bytes < 1024 or replay_batch_size < 1:
            raise ValueError("completion outbox bounds must be positive")
        if max_buffered_tail_bytes < 1:
            raise ValueError("completion outbox buffered tail bound must be positive")
        if replay_budget_seconds <= 0 or max_replay_attempts < 1:
            raise ValueError("completion outbox replay bounds must be positive")
        self.core_api = core_api
        self.sink = sink
        self.namespace = namespace
        self.name = name
        self.max_records = max_records
        self.max_bytes = max_bytes
        self.max_buffered_tail_bytes = max_buffered_tail_bytes
        self.replay_batch_size = replay_batch_size
        self.replay_budget_seconds = replay_budget_seconds
        self.max_replay_attempts = max_replay_attempts
        self.monotonic = monotonic
        self.now = now
        self.last_replay: dict[str, int] = {
            "replayed": 0,
            "deferred": 0,
            "quarantined": 0,
        }
        self._attempt_digests: dict[str, str] = {}
        # Critical events whose write-ahead copy could not be written. Exported
        # as ``gpu_fault_completion_outbox_append_failures_total``: since a
        # failed buffer no longer vetoes the POST, this counter is the only
        # signal that the outbox ConfigMap is full or unwritable (F1/F12).
        self.append_failures_total = 0
        self._append_failures_logged: set[str] = set()
        # Depth gauges as of the last replay pass, exported as
        # ``gpu_fault_completion_outbox_depth`` /
        # ``..._quarantined_depth``. A quarantined record is one no replay will
        # ever pick up again, so it has to be visible without a ConfigMap read
        # per scrape (F12).
        self.last_depth = 0
        self.last_quarantined_depth = 0

    @staticmethod
    def _record_key(path: str, payload: dict[str, Any]) -> str:
        cluster_id = str(payload.get("cluster_id") or "")
        attempt_id = str(payload.get("attempt_id") or "")
        if not cluster_id or not attempt_id:
            raise ValueError(
                "critical completion event requires cluster_id and attempt_id"
            )
        return f"{cluster_id}/{attempt_id}/{path.rsplit('/', 1)[-1]}"

    @staticmethod
    def _data(value: Any) -> dict[str, str]:
        raw = value.get("data", {}) if isinstance(value, dict) else value.data
        return dict(raw or {})

    @staticmethod
    def _resource_version(value: Any) -> str | None:
        metadata = (
            value.get("metadata", {}) if isinstance(value, dict) else value.metadata
        )
        if isinstance(metadata, dict):
            return metadata.get("resourceVersion") or metadata.get("resource_version")
        return getattr(metadata, "resource_version", None)

    def _read_data(self) -> tuple[dict[str, str], str | None]:
        value = self.core_api.read_namespaced_config_map(self.name, self.namespace)
        return self._data(value), self._resource_version(value)

    def _events(self, data: dict[str, str]) -> list[dict[str, Any]]:
        document = json.loads(data.get(self.DATA_KEY, "[]"))
        if not isinstance(document, list) or any(
            not isinstance(item, dict) for item in document
        ):
            raise RuntimeError("completion outbox ConfigMap contains invalid JSON")
        return document

    def _attempts(self, data: dict[str, str]) -> dict[str, dict[str, Any]]:
        document = json.loads(data.get(self.ATTEMPTS_KEY, "{}"))
        if not isinstance(document, dict) or any(
            not isinstance(key, str) or not isinstance(item, dict)
            for key, item in document.items()
        ):
            raise RuntimeError("completion attempt state contains invalid JSON")
        return document

    def _read(self) -> tuple[list[dict[str, Any]], str | None]:
        data, resource_version = self._read_data()
        return self._events(data), resource_version

    def _write_data(
        self,
        data: dict[str, str],
        resource_version: str | None,
    ) -> None:
        records = self._events(data)
        attempts = self._attempts(data)
        payload_bytes = sum(
            len(key.encode()) + len(value.encode()) for key, value in data.items()
        )
        if (
            len(records) > self.max_records
            or len(attempts) > self.max_records
            or payload_bytes > self.max_bytes
        ):
            raise CompletionOutboxFull(
                "completion outbox or active attempt state exceeds its bounds"
            )
        self.core_api.replace_namespaced_config_map(
            self.name,
            self.namespace,
            {
                "metadata": {
                    "name": self.name,
                    **(
                        {"resourceVersion": resource_version}
                        if resource_version is not None
                        else {}
                    ),
                },
                "data": data,
            },
        )

    def _mutate_data(
        self,
        function: Callable[[dict[str, str]], dict[str, str]],
    ) -> None:
        for attempt in range(3):
            data, resource_version = self._read_data()
            updated = function(data)
            if updated is data:
                # A mutation that changes nothing (a key that is already
                # buffered, a removal of a key that is gone) must not spend a
                # whole-document replace on this hot path.
                return
            try:
                self._write_data(updated, resource_version)
                return
            except Exception as exc:
                if getattr(exc, "status", None) != 409 or attempt == 2:
                    raise

    def _mutate(
        self,
        function: Callable[
            [list[dict[str, Any]]],
            list[dict[str, Any]],
        ],
    ) -> None:
        def update(data: dict[str, str]) -> dict[str, str]:
            document = json.dumps(
                function(self._events(data)),
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
            if document == data.get(self.DATA_KEY):
                return data
            result = dict(data)
            result[self.DATA_KEY] = document
            return result

        self._mutate_data(update)

    def _append(self, key: str, path: str, payload: dict[str, Any]) -> bool:
        """Buffer the pointer-sized copy; ``True`` when the caller must post.

        A key that is already in the ConfigMap is a record an earlier pass
        failed to deliver, and ``replay`` owns it from then on, so the caller
        must not post it live a second time in the same pass (F8) -- *unless*
        that record is quarantined. ``replay`` skips a quarantined record for
        ever (no production caller passes ``include_quarantined``), and it
        quarantines after ``max_replay_attempts`` even for a transient outage,
        so deferring to a replay that will never come would make the event
        permanently undeliverable. For a quarantined key the live POST is the
        only path left: the record is kept as it is -- attempts, quarantine
        flag and last error stay readable -- and the caller removes it once
        the control plane accepts the event.
        """

        buffered = pointer_sized_completion_payload(
            payload, max_tail_bytes=self.max_buffered_tail_bytes
        )
        deliver_live = True

        def append(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
            nonlocal deliver_live
            existing = next((item for item in records if item.get("key") == key), None)
            if existing is not None:
                deliver_live = bool(existing.get("quarantined", False))
                return records
            return [*records, self._record(key, path, buffered)]

        self._mutate(append)
        return deliver_live

    def _count_append_failure(
        self, key: str, exc: BaseException, *, stage: str = "buffer"
    ) -> None:
        """Count every outbox bookkeeping failure; log the first one per key.

        ``stage`` is ``"buffer"`` for the write-ahead append and ``"clear"``
        for the removal that follows a successful delivery. Both mean the
        ConfigMap could not be written, and both are counted the same way; only
        the consequence differs, so only the message does. The watcher
        reconciles every 30 s, so logging each pass would bury the cause under
        its own repetition (the same rule as F9).
        """

        self.append_failures_total += 1
        if key in self._append_failures_logged:
            LOGGER.debug(
                "completion outbox write (%s) still failing: key=%s (%s: %s)",
                stage,
                key,
                type(exc).__name__,
                exc,
            )
            return
        if len(self._append_failures_logged) >= _MAX_LOGGED_APPEND_FAILURES:
            self._append_failures_logged.clear()
        self._append_failures_logged.add(key)
        if stage == "clear":
            LOGGER.error(
                "cannot clear the delivered critical completion event key=%s "
                "(%s: %s); the control plane has already accepted it, so the "
                "record stays buffered and the replay will re-send it (the "
                "control plane deduplicates by event_key)",
                key,
                type(exc).__name__,
                exc,
            )
            return
        LOGGER.error(
            "cannot write ahead critical completion event key=%s (%s: %s); "
            "delivering it live without a buffered copy -- a watcher restart "
            "before the control plane accepts it would lose the event",
            key,
            type(exc).__name__,
            exc,
        )

    def _record(self, key: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "key": key,
            "path": path,
            "payload": payload,
            "buffered_at": self.now(),
            "attempts": 0,
            "quarantined": False,
        }

    def _upsert_latest(self, path: str, payload: dict[str, Any]) -> str:
        key = self._record_key(path, payload)

        def upsert(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
            retained = [item for item in records if item.get("key") != key]
            return [*retained, self._record(key, path, payload)]

        self._mutate(upsert)
        return key

    def _remove(self, key: str) -> None:
        self._mutate(
            lambda records: [item for item in records if item.get("key") != key]
        )

    def _update(self, key: str, **fields: Any) -> None:
        def update(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return [
                {**item, **fields} if item.get("key") == key else item
                for item in records
            ]

        self._mutate(update)

    @staticmethod
    def _attempt_key(payload: dict[str, Any]) -> str:
        cluster_id = str(payload.get("cluster_id") or "")
        attempt_id = str(payload.get("attempt_id") or "")
        if not cluster_id or not attempt_id:
            raise ValueError("attempt observation requires cluster_id and attempt_id")
        return f"{cluster_id}/{attempt_id}"

    @staticmethod
    def _attempt_digest(payload: dict[str, Any]) -> str:
        structural = dict(payload)
        structural.pop("observed_at", None)
        return hashlib.sha256(
            json.dumps(
                structural,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode()
        ).hexdigest()

    def load_attempt_observations(self) -> list[dict[str, Any]]:
        data, _resource_version = self._read_data()
        attempts = self._attempts(data)
        self._attempt_digests = {
            key: self._attempt_digest(payload) for key, payload in attempts.items()
        }
        return [dict(attempts[key]) for key in sorted(attempts)]

    def save_attempt_observation(self, payload: dict[str, Any]) -> bool:
        key = self._attempt_key(payload)
        digest = self._attempt_digest(payload)
        if self._attempt_digests.get(key) == digest:
            return False

        def update(data: dict[str, str]) -> dict[str, str]:
            attempts = self._attempts(data)
            attempts[key] = dict(payload)
            result = dict(data)
            result[self.ATTEMPTS_KEY] = json.dumps(
                attempts,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
            return result

        self._mutate_data(update)
        self._attempt_digests[key] = digest
        return True

    def remove_attempt_observation(self, payload: dict[str, Any]) -> None:
        key = self._attempt_key(payload)

        def update(data: dict[str, str]) -> dict[str, str]:
            attempts = self._attempts(data)
            if key not in attempts:
                return data
            attempts.pop(key)
            result = dict(data)
            result[self.ATTEMPTS_KEY] = json.dumps(
                attempts,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
            return result

        self._mutate_data(update)
        self._attempt_digests.pop(key, None)

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        if path == WORKLOAD_OBSERVATION_PATH:
            try:
                return self.sink.post(path, payload)
            except Exception:
                self._upsert_latest(path, payload)
                raise
        if path not in CRITICAL_COMPLETION_PATHS:
            return self.sink.post(path, payload)
        # A malformed payload has no key and is a caller bug, not a durability
        # problem: it still fails hard, before anything is attempted.
        key = self._record_key(path, payload)
        try:
            deliver_live = self._append(key, path, payload)
        except Exception as append_error:
            self._count_append_failure(key, append_error)
            try:
                return self.sink.post(path, payload)
            except Exception as delivery_error:
                # Nothing buffered and nothing delivered: stay fail-closed and
                # name the buffer failure, which is what has to be fixed.
                raise append_error from delivery_error
        self._append_failures_logged.discard(key)
        if not deliver_live:
            LOGGER.info(
                "critical completion event already buffered; leaving delivery "
                "to the outbox replay: key=%s",
                key,
            )
            return {DEFERRED_TO_REPLAY: True}
        result = self.sink.post(path, payload)
        try:
            self._remove(key)
        except Exception as removal_error:
            # The control plane has the event; a ConfigMap that cannot be
            # written must not turn that into a failed delivery, or the caller
            # would arm its emergency fallback over an accepted event. The
            # record stays and replay re-sends it; the control plane
            # deduplicates by event_key.
            self._count_append_failure(key, removal_error, stage="clear")
        return result

    def replay(self, *, include_quarantined: bool = False) -> int:
        """Deliver buffered records, isolating each one (F-G1).

        A record whose delivery fails is kept and counted, never allowed to
        stop the records behind it: a permanent rejection (or too many
        attempts) quarantines it, a transient failure defers it to the next
        cycle. Quarantined records are skipped unless ``include_quarantined``
        is set, which is how an operator retries them after fixing the cause
        (for example registering the runtime profile the control plane did
        not know). The whole pass is bounded by ``replay_budget_seconds``.

        What is re-sent is the buffered, pointer-sized record: its log tails
        are capped at ``max_buffered_tail_bytes`` per snapshot and flagged
        ``buffered_tail_truncated``. See
        ``pointer_sized_completion_payload``.
        """

        records, _resource_version = self._read()
        candidates = [
            record
            for record in records
            if include_quarantined or not record.get("quarantined", False)
        ]
        batch = candidates[: self.replay_batch_size]
        deferred = len(candidates) - len(batch)
        replayed = quarantined = 0
        drained: set[str] = set()
        isolated: set[str] = set()
        deadline = self.monotonic() + self.replay_budget_seconds
        for index, record in enumerate(batch):
            if index > 0 and self.monotonic() >= deadline:
                deferred += len(batch) - index
                LOGGER.warning(
                    "completion outbox replay budget exhausted: "
                    "%d records deferred to the next cycle",
                    len(batch) - index,
                )
                break
            key = str(record["key"])
            path = str(record["path"])
            payload = dict(record["payload"])
            try:
                self.sink.post(path, payload)
            except Exception as exc:
                attempts = int(record.get("attempts", 0)) + 1
                disposition = completion_delivery_disposition(exc)
                status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
                if disposition == "quarantine" or attempts >= self.max_replay_attempts:
                    quarantined += 1
                    isolated.add(key)
                    LOGGER.warning(
                        "completion outbox record quarantined after %d attempts: "
                        "key=%s status=%s error=%s",
                        attempts,
                        key,
                        status,
                        exc,
                    )
                    self._update(
                        key,
                        attempts=attempts,
                        quarantined=True,
                        quarantined_at=self.now(),
                        last_status=status,
                        last_error=str(exc)[:500],
                    )
                else:
                    deferred += 1
                    self._update(
                        key,
                        attempts=attempts,
                        last_status=status,
                        last_error=str(exc)[:500],
                    )
                continue
            if path == WORKLOAD_OBSERVATION_PATH and str(
                payload.get("workload_phase") or ""
            ) not in {"PENDING", "RUNNING"}:
                self.remove_attempt_observation(payload)
            self._remove(key)
            drained.add(key)
            replayed += 1
        self.last_replay = {
            "replayed": replayed,
            "deferred": deferred,
            "quarantined": quarantined,
        }
        self._publish_depth(records, drained=drained, isolated=isolated)
        return replayed

    def _publish_depth(
        self,
        records: list[dict[str, Any]],
        *,
        drained: set[str],
        isolated: set[str],
    ) -> None:
        """Refresh the exported depth gauges from the pass we just walked.

        Derived from the snapshot ``replay`` already read rather than from a
        second ConfigMap GET: the watcher reconciles every 30 s and the gauge
        does not justify an extra API call per pass. It therefore reports the
        depth as of the start of the pass minus what the pass drained, so a
        record buffered later in the same reconcile shows up one pass later.
        """

        remaining = [
            record for record in records if str(record.get("key")) not in drained
        ]
        self.last_depth = len(remaining)
        self.last_quarantined_depth = sum(
            1
            for record in remaining
            if record.get("quarantined", False) or str(record.get("key")) in isolated
        )

    def depth(self) -> int:
        records, _resource_version = self._read()
        return len(records)

    def quarantined_depth(self) -> int:
        records, _resource_version = self._read()
        return sum(1 for record in records if record.get("quarantined", False))

    def stats(self) -> dict[str, float]:
        """Depth, quarantined depth and the age of the oldest buffered record."""

        records, _resource_version = self._read()
        now = self.now()
        ages = [
            max(0.0, now - float(record["buffered_at"]))
            for record in records
            if isinstance(record.get("buffered_at"), (int, float))
        ]
        return {
            "depth": len(records),
            "quarantined": sum(1 for r in records if r.get("quarantined", False)),
            "oldest_age_seconds": max(ages, default=0.0),
        }


def completion_sink_from_environment(core_api: Any) -> KubernetesCompletionOutbox:
    return KubernetesCompletionOutbox(
        core_api,
        HttpEventSink(
            os.environ["GPU_FAULT_CONTROL_PLANE_URL"],
            bearer_token=os.getenv("GPU_FAULT_CONTROL_PLANE_TOKEN"),
            timeout_seconds=10,
            processor_receipt_timeout_seconds=float(
                os.getenv("GPU_FAULT_PROCESSOR_RECEIPT_TIMEOUT_SECONDS", "120")
            ),
            processor_receipt_poll_seconds=float(
                os.getenv("GPU_FAULT_PROCESSOR_RECEIPT_POLL_SECONDS", "0.25")
            ),
        ),
        namespace=os.getenv("GPU_FAULT_NAMESPACE", "gpu-fault-system"),
        max_records=int(os.getenv("GPU_FAULT_COMPLETION_OUTBOX_MAX_RECORDS", "256")),
        max_bytes=int(os.getenv("GPU_FAULT_COMPLETION_OUTBOX_MAX_BYTES", "900000")),
        # ``max_buffered_tail_bytes`` is deliberately not an environment knob:
        # the write-ahead copy has to fit whatever ``max_bytes`` allows, and an
        # operator raising it would re-create F1.
        max_buffered_tail_bytes=DEFAULT_BUFFERED_TAIL_BYTES,
        replay_batch_size=int(
            os.getenv("GPU_FAULT_COMPLETION_OUTBOX_REPLAY_BATCH_SIZE", "32")
        ),
    )


def replay_completion_outbox(sink: Any, logger: logging.Logger) -> None:
    replay = getattr(sink, "replay", None)
    if replay is None:
        return
    try:
        replay()
    except Exception:
        logger.exception("cannot replay Completion Watcher outbox")
