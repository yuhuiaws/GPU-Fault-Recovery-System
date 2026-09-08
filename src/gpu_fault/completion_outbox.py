from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections.abc import Callable
from datetime import datetime, timezone
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
#: Suffix of the second ConfigMap. ``<name>`` carries the critical write-ahead
#: log and ``<name>-active`` the routine per-attempt state, because a ConfigMap
#: is capped at 1 MiB *per object*: with both documents in one object N running
#: Pods of routine state could fill it and the terminal event of a failing
#: attempt could no longer be written ahead at all (F6).
ACTIVE_STATE_SUFFIX = "-active"
#: Key both objects use for attempt state: the one the WAL object carried
#: before F6 split them, and the only key of ``<name>-active`` after it.
ATTEMPT_STATE_KEY = "active-attempts.json"
#: Key of the write-ahead log inside the WAL object.
EVENT_LOG_KEY = "events.json"
#: Statuses on ``<name>-active`` that mean "not there yet" rather than broken:
#: 404 is the upgrade window before the manifest that adds the object, 403 the
#: window before the Role that names it. Both are survivable -- attempt state is
#: a restart optimisation -- and neither may be fatal, because
#: ``load_attempt_observations`` runs from the controller's constructor (M2).
ACTIVE_STATE_UNAVAILABLE_STATUSES = frozenset({403, 404})
#: How long the object is left alone after it refused. Without this every
#: ``save_attempt_observation`` reads before it writes, so a watcher watching N
#: attempts spent N refused GETs per reconcile pass for as long as the outage
#: lasted (I1). One minute is short enough that an operator who applies the
#: manifest sees state persisted again within a pass or two.
ACTIVE_STATE_RETRY_SECONDS = 60.0
#: How often the same degradation is restated at ERROR. Logging it once hid an
#: outage that started hours before the operator looked; logging it per pass is
#: the F9 storm.
ACTIVE_STATE_REPEAT_LOG_SECONDS = 3600.0


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


# Pure helpers, module level rather than static methods: the outbox class is
# at its architecture size limit, and none of them reads the instance.
def _record_key(path: str, payload: dict[str, Any]) -> str:
    cluster_id = str(payload.get("cluster_id") or "")
    attempt_id = str(payload.get("attempt_id") or "")
    if not cluster_id or not attempt_id:
        raise ValueError("critical completion event requires cluster_id and attempt_id")
    return f"{cluster_id}/{attempt_id}/{path.rsplit('/', 1)[-1]}"


def _data(value: Any) -> dict[str, str]:
    raw = value.get("data", {}) if isinstance(value, dict) else value.data
    return dict(raw or {})


def _resource_version(value: Any) -> str | None:
    metadata = value.get("metadata", {}) if isinstance(value, dict) else value.metadata
    if isinstance(metadata, dict):
        return metadata.get("resourceVersion") or metadata.get("resource_version")
    return getattr(metadata, "resource_version", None)


def _attempt_key(payload: dict[str, Any]) -> str:
    cluster_id = str(payload.get("cluster_id") or "")
    attempt_id = str(payload.get("attempt_id") or "")
    if not cluster_id or not attempt_id:
        raise ValueError("attempt observation requires cluster_id and attempt_id")
    return f"{cluster_id}/{attempt_id}"


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


def _remaining_depth(
    records: list[dict[str, Any]],
    *,
    drained: set[str],
    isolated: set[str],
) -> tuple[int, int]:
    """``(depth, quarantined_depth)`` of the pass ``replay`` just walked.

    Derived from the snapshot ``replay`` already read rather than from a second
    ConfigMap GET: the watcher reconciles every 30 s and the gauge does not
    justify an extra API call per pass. It therefore reports the depth as of the
    start of the pass minus what the pass drained, so a record buffered later in
    the same reconcile shows up one pass later.
    """

    remaining = [record for record in records if str(record.get("key")) not in drained]
    return len(remaining), sum(
        1
        for record in remaining
        if record.get("quarantined", False) or str(record.get("key")) in isolated
    )


def _observed_instant(record: dict[str, Any]) -> datetime | None:
    """When ``record`` was observed, or ``None`` if it does not say.

    Parsed rather than compared as text: pydantic renders a whole second as
    ``...T10:00:00Z`` and anything else as ``...T10:00:00.500000Z``, and ``Z``
    sorts *after* ``.``, so string order puts the earlier instant last. The
    ``Z`` is spelled out for ``fromisoformat`` because Python before 3.11 does
    not accept it.
    """

    text = str(record.get("observed_at") or "")
    try:
        instant = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    # A record without a zone is one an older release (or a hand edit) wrote,
    # and UTC is the only zone anything in this system writes. Without this the
    # comparison below raises ``TypeError`` inside the migration's own
    # try/except, so the migration never completed and every load and every
    # owed probe printed the same traceback again.
    return instant if instant.tzinfo else instant.replace(tzinfo=timezone.utc)


def _merge_legacy_attempt_state(
    legacy: dict[str, dict[str, Any]],
    current: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Adopt legacy records, keeping the newer ``observed_at`` per key (I2).

    "The object that already holds it wins" is wrong in one direction and "the
    legacy object wins" in the other: the migration can run again after a drop
    that failed, and by then the live process may have written a fresher record
    for the same attempt. A record whose timestamp cannot be read loses to one
    that has a readable timestamp, and to nothing else.
    """

    merged = dict(current)
    for key, record in legacy.items():
        existing = merged.get(key)
        if existing is None:
            merged[key] = record
            continue
        candidate = _observed_instant(record)
        held = _observed_instant(existing)
        if candidate is not None and (held is None or candidate > held):
            merged[key] = record
    return merged


def _event_records(data: dict[str, str]) -> list[dict[str, Any]]:
    """The write-ahead log of the WAL object, validated."""

    document = json.loads(data.get(EVENT_LOG_KEY, "[]"))
    if not isinstance(document, list) or any(
        not isinstance(item, dict) for item in document
    ):
        raise RuntimeError("completion outbox ConfigMap contains invalid JSON")
    return document


def _buffered_record(
    key: str,
    path: str,
    payload: dict[str, Any],
    buffered_at: Any,
) -> dict[str, Any]:
    """A fresh write-ahead record: never delivered, never quarantined."""

    return {
        "key": key,
        "path": path,
        "payload": payload,
        "buffered_at": buffered_at,
        "attempts": 0,
        "quarantined": False,
    }


def _record_field_update(
    key: str, fields: dict[str, Any]
) -> Callable[[list[dict[str, Any]]], list[dict[str, Any]]]:
    """Mutation that sets ``fields`` on the record with ``key``, if it is there."""

    def update(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {**item, **fields} if item.get("key") == key else item for item in records
        ]

    return update


def _attempt_records(data: dict[str, str]) -> dict[str, dict[str, Any]]:
    """The attempt-state document of either object, validated."""

    document = json.loads(data.get(ATTEMPT_STATE_KEY, "{}"))
    if not isinstance(document, dict) or any(
        not isinstance(key, str) or not isinstance(item, dict)
        for key, item in document.items()
    ):
        raise RuntimeError("completion attempt state contains invalid JSON")
    return document


def _without_attempt_state(current: dict[str, str]) -> dict[str, str]:
    """The WAL document with the migrated attempt-state key dropped."""

    if ATTEMPT_STATE_KEY not in current:
        return current
    result = dict(current)
    result.pop(ATTEMPT_STATE_KEY)
    return result


def _legacy_adoption(
    legacy: dict[str, dict[str, Any]],
) -> Callable[[dict[str, str]], dict[str, str]]:
    """Mutation that merges ``legacy`` into ``<name>-active``."""

    def adopt(current: dict[str, str]) -> dict[str, str]:
        attempts = _merge_legacy_attempt_state(legacy, _attempt_records(current))
        document = _attempt_document(attempts)
        if document == current.get(ATTEMPT_STATE_KEY):
            return current
        result = dict(current)
        result[ATTEMPT_STATE_KEY] = document
        return result

    return adopt


def _attempt_removal(key: str) -> Callable[[dict[str, str]], dict[str, str]]:
    """Mutation that drops ``key`` from the attempt-state document."""

    def update(data: dict[str, str]) -> dict[str, str]:
        attempts = _attempt_records(data)
        if key not in attempts:
            return data
        attempts.pop(key)
        result = dict(data)
        result[ATTEMPT_STATE_KEY] = _attempt_document(attempts)
        return result

    return update


def _sorted_records(attempts: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(attempts[key]) for key in sorted(attempts)]


def _attempt_document(attempts: dict[str, dict[str, Any]]) -> str:
    return json.dumps(
        attempts,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


class ActiveStateHealth:
    """Whether ``<name>-active`` can be used, and when to try it again (I1/M2).

    Its own object because the answer is three pieces of state that only make
    sense together -- the exported 0/1 gauge, the instant the next probe is
    allowed and the instant the outage was last reported -- and because the
    outbox class is at its architecture size limit.
    """

    def __init__(
        self,
        namespace: str,
        name: str,
        now: Callable[[], float],
        *,
        retry_seconds: float = ACTIVE_STATE_RETRY_SECONDS,
        repeat_log_seconds: float = ACTIVE_STATE_REPEAT_LOG_SECONDS,
    ) -> None:
        self.namespace = namespace
        self.name = name
        self.now = now
        self.retry_seconds = retry_seconds
        self.repeat_log_seconds = repeat_log_seconds
        #: Exported as ``gpu_fault_completion_active_state_unavailable``.
        self.unavailable = 0
        self._retry_at: float | None = None
        self._logged_at: float | None = None

    def degraded(self) -> bool:
        """``True`` while the object must not be touched at all.

        The window expires by itself, so an operator who applies the manifest
        needs no restart. Expiry is not recovery: the gauge stays up until a
        call actually succeeds, and a still-refused object degrades again on
        that one probe.
        """

        retry_at = self._retry_at
        if retry_at is None:
            return False
        if self.now() < retry_at:
            return True
        self._retry_at = None
        return False

    def refused(self, exc: BaseException) -> bool:
        """``True`` when ``exc`` is the object (or the permission) not existing.

        The watcher keeps watching either way, and the Role deliberately grants
        no ``create`` (a ``create`` cannot be scoped by ``resourceNames``, so it
        would let this pod make any ConfigMap). Anything else -- an API server
        that is down, a malformed document -- is not this and propagates.
        """

        if getattr(exc, "status", None) not in ACTIVE_STATE_UNAVAILABLE_STATUSES:
            return False
        self.unavailable = 1
        now = self.now()
        self._retry_at = now + self.retry_seconds
        last = self._logged_at
        if last is None or now - last >= self.repeat_log_seconds:
            self._logged_at = now
            LOGGER.error(
                "ConfigMap %s/%s cannot be used (%s: %s), so active attempt "
                "state is not persisted and a watcher restart will re-derive "
                "its attempts from live Pods; apply "
                "deploy/dataplane/completion-watcher.yaml -- both its ConfigMap "
                "and its Role. Retrying in %.0fs",
                self.namespace,
                self.name,
                type(exc).__name__,
                exc,
                self.retry_seconds,
            )
        return True

    def owes_a_probe(self) -> bool:
        """``True`` when the window has expired and nothing has probed yet.

        The gauge only comes back down when a call succeeds, and on an idle
        cluster no attempt-state call is ever made: without this the watcher
        reports the outage for ever after the operator fixed it.
        """

        return bool(self.unavailable) and not self.degraded()

    def defer(self) -> None:
        """Hold the next probe for one window, leaving the report as it is.

        For a probe that failed for a reason that is not "the object is not
        there": the outage stands, and retrying it once a minute rather than
        once a pass is what keeps a broken API server from being a log storm.
        """

        self._retry_at = self.now() + self.retry_seconds

    def usable(self) -> None:
        """Record that the object answered, so the gauge comes back down."""

        if self.unavailable:
            LOGGER.info(
                "ConfigMap %s/%s is usable again; active attempt state is "
                "persisted from now on",
                self.namespace,
                self.name,
            )
        self.unavailable = 0
        self._retry_at = None
        self._logged_at = None


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

    DATA_KEY = EVENT_LOG_KEY
    ATTEMPTS_KEY = ATTEMPT_STATE_KEY

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
        #: Routine attempt state; see ``ACTIVE_STATE_SUFFIX``. Both objects are
        #: shipped by ``deploy/dataplane/completion-watcher.yaml`` and both are
        #: named in the Role's ``resourceNames``.
        self.active_name = f"{name}{ACTIVE_STATE_SUFFIX}"
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
        # What this process believes the WAL holds, so that the replay at the
        # top of every reconcile pass (and every debounced flush) does not spend
        # a whole-document GET on a document it emptied itself (F6). ``None`` is
        # "unknown" -- a fresh process, or any write that raised -- and always
        # forces the read; only a known zero may skip it.
        self._known_depth: int | None = None
        self._known_quarantined: int | None = None
        # Replay passes that skipped the GET. Not a published metric -- the
        # depth gauges already say what the WAL holds -- but the number that
        # explains a drop in this pod's ConfigMap read rate.
        self.replay_reads_skipped_total = 0
        # Whether attempt state left in the WAL object by the previous release
        # has been moved to ``active_name`` yet; done once per process, on the
        # first read of the active state.
        self._legacy_state_migrated = False
        # Whether the drop half of that migration is still owed. It is retried
        # from the next save/remove: an adopt that succeeded and a drop that did
        # not leaves the record in both objects, and the next restart would
        # resurrect an attempt that has since finished (I2).
        self._legacy_drop_pending = False
        self._legacy_drop_failure_logged = False
        # ``monotonic``, not ``now``: a wall clock that steps backwards (NTP, a
        # suspended node) would hold the probe for as long as the step.
        self._active_state = ActiveStateHealth(
            namespace, self.active_name, self.monotonic
        )

    def _read_data(self, name: str) -> tuple[dict[str, str], str | None]:
        value = self.core_api.read_namespaced_config_map(name, self.namespace)
        return _data(value), _resource_version(value)

    def _read(self) -> tuple[list[dict[str, Any]], str | None]:
        data, resource_version = self._read_data(self.name)
        records = _event_records(data)
        self._note_depth(records)
        return records, resource_version

    def _write_data(
        self,
        name: str,
        data: dict[str, str],
        resource_version: str | None,
    ) -> None:
        payload_bytes = sum(
            len(key.encode()) + len(value.encode()) for key, value in data.items()
        )
        if payload_bytes > self.max_bytes:
            raise CompletionOutboxFull(
                f"{name} exceeds its byte bound "
                f"({payload_bytes} > {self.max_bytes} bytes)"
            )
        # ``max_records`` bounds the write-ahead log only. Routine attempt state
        # lives in its own object (F6), where the bound that matters is the
        # 1 MiB one every ConfigMap has: capping it at 256 attempts would refuse
        # to remember the 257th running job on a large cluster for no reason.
        if len(_event_records(data)) > self.max_records:
            raise CompletionOutboxFull(
                f"completion outbox exceeds its record bound of {self.max_records}"
            )
        self.core_api.replace_namespaced_config_map(
            name,
            self.namespace,
            {
                "metadata": {
                    "name": name,
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
        name: str,
        function: Callable[[dict[str, str]], dict[str, str]],
    ) -> None:
        for attempt in range(3):
            data, resource_version = self._read_data(name)
            updated = function(data)
            if updated is data:
                # A mutation that changes nothing (a key that is already
                # buffered, a removal of a key that is gone) must not spend a
                # whole-document replace on this hot path.
                return
            try:
                self._write_data(name, updated, resource_version)
                return
            except Exception as exc:
                if getattr(exc, "status", None) != 409 or attempt == 2:
                    raise

    def _note_depth(self, records: list[dict[str, Any]]) -> None:
        """Remember what the WAL object holds after a read or a write we own."""

        self._known_depth = len(records)
        self._known_quarantined = sum(
            1 for record in records if record.get("quarantined", False)
        )

    def _forget_depth(self) -> None:
        """Drop the in-process belief; the next replay must read the object.

        Called whenever a write raised: the ConfigMap may hold what we tried to
        write, what was there before, or a concurrent writer's document, and a
        skipped read would then hide a buffered record for ever.
        """

        self._known_depth = None
        self._known_quarantined = None

    def _mutate(
        self,
        function: Callable[
            [list[dict[str, Any]]],
            list[dict[str, Any]],
        ],
    ) -> None:
        written: list[dict[str, Any]] = []

        def update(data: dict[str, str]) -> dict[str, str]:
            records = function(_event_records(data))
            written[:] = records
            document = json.dumps(
                records,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
            if document == data.get(self.DATA_KEY):
                return data
            result = dict(data)
            result[self.DATA_KEY] = document
            return result

        try:
            self._mutate_data(self.name, update)
        except Exception:
            self._forget_depth()
            raise
        self._note_depth(written)

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
            return [*records, _buffered_record(key, path, buffered, self.now())]

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
                "record is flagged as delivered for the next replay pass to "
                "drop without re-sending it",
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

    def _upsert_latest(self, path: str, payload: dict[str, Any]) -> str:
        key = _record_key(path, payload)

        def upsert(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
            retained = [item for item in records if item.get("key") != key]
            return [*retained, _buffered_record(key, path, payload, self.now())]

        self._mutate(upsert)
        return key

    def _remove(self, key: str) -> None:
        self._mutate(
            lambda records: [item for item in records if item.get("key") != key]
        )

    def _update(self, key: str, **fields: Any) -> None:
        self._mutate(_record_field_update(key, fields))

    def _reclaim_delivered(self, key: str, removal_error: BaseException) -> None:
        """Hand a delivered-but-unremovable record back to ``replay``.

        The control plane has accepted the event and the removal that should
        have cleared the record failed. Leaving it as it is would strand it:
        a quarantined record is one ``replay`` skips for ever, and the live
        path is finished with the key, so nothing would ever drop it and
        ``..._quarantined_depth`` would never come back down. Flagging it
        ``delivered`` (and lifting the quarantine) makes the next replay pass
        remove it without re-sending it -- a smaller write than the removal,
        so it usually survives whatever broke the removal.
        """

        try:
            self._update(key, delivered=True, quarantined=False, attempts=0)
        except Exception as flag_error:
            LOGGER.error(
                "a delivered completion record could not be cleared (%s: %s) "
                "nor flagged as delivered (%s: %s): key=%s stays buffered "
                "until a pass that can write the ConfigMap picks it up, or "
                "until an operator removes it",
                type(removal_error).__name__,
                removal_error,
                type(flag_error).__name__,
                flag_error,
                key,
            )

    @property
    def active_state_unavailable(self) -> int:
        """0/1: is ``<name>-active`` refusing (I1)? Exported as a gauge."""

        return self._active_state.unavailable

    def _retry_legacy_drop(self) -> None:
        """Finish a migration whose adopt succeeded and whose drop did not (I2).

        Called from the routine save/remove path, so the retry costs nothing
        until it is owed and needs no timer of its own. Its own failure is
        logged once and swallowed: the records are readable from either object,
        which is the state this arrived in, and the next write tries again.
        """

        try:
            self._mutate_data(self.name, _without_attempt_state)
        except Exception as exc:
            if self._legacy_drop_failure_logged:
                LOGGER.debug(
                    "the migrated attempt state still cannot be dropped from "
                    "%s (%s: %s)",
                    self.name,
                    type(exc).__name__,
                    exc,
                )
                return
            self._legacy_drop_failure_logged = True
            LOGGER.warning(
                "cannot drop the migrated attempt state from %s (%s: %s); it "
                "stays in both objects and the next attempt-state write will "
                "try again -- a restart before that would restore attempts "
                "that have since finished",
                self.name,
                type(exc).__name__,
                exc,
            )
            return
        self._legacy_drop_pending = False
        self._legacy_drop_failure_logged = False

    def _before_active_state_write(self) -> bool:
        """``False`` when ``<name>-active`` must not be touched at all (I1).

        Also the one place the owed legacy drop is retried from, so every
        routine write finishes a migration that a failed drop left half done.
        """

        if self._active_state.degraded():
            return False
        if self._legacy_drop_pending:
            self._retry_legacy_drop()
        return True

    def _migrate_legacy_attempt_state(self) -> None:
        """Adopt attempt state a previous release left in the WAL object (F6).

        The new object is written first and the old key dropped only after, so a
        crash in between repeats the migration rather than losing an attempt;
        the adoption merges by key, which makes the repeat a no-op. A failure
        leaves the flag down so the next call tries again.
        """

        if self._legacy_state_migrated or self._active_state.degraded():
            return
        self._legacy_state_migrated = True
        try:
            data, _resource_version = self._read_data(self.name)
            legacy = _attempt_records(data)
            if not legacy:
                return

            self._mutate_data(self.active_name, _legacy_adoption(legacy))
            self._legacy_drop_pending = True
            LOGGER.info(
                "migrated %d persisted attempt observations from %s to %s",
                len(legacy),
                self.name,
                self.active_name,
            )
        except Exception as exc:
            if not self._active_state.refused(exc):
                LOGGER.exception(
                    "cannot migrate persisted attempt observations from %s to %s",
                    self.name,
                    self.active_name,
                )
            self._legacy_state_migrated = False
            return
        # Only after the adoption is durable, and never fatal: the records are
        # readable from either object, so an owed drop is a repeat, not a loss.
        self._retry_legacy_drop()

    def probe_active_state(self) -> None:
        """Spend the one GET a recovered ``<name>-active`` is owed (I1).

        Called from the top of every reconcile pass, because ``usable()`` is
        only reached from a save/remove/load and an idle cluster makes none of
        those: the operator applied the manifest and the gauge stayed at 1.

        The records this reads are deliberately thrown away: this is a flip of
        the health flag (and the migration it drags along), not a restore.
        Restoring is constructor-shaped -- it seeds the controller's trackers
        before any pass has run -- so replaying it mid-life would overwrite
        observations the live passes have since published with whatever the
        object held, and re-arm the missing-attempt grace (F7) for attempts the
        watcher is already watching. The state that matters after a recovery is
        written back by the very next ``save_attempt_observation``.
        """

        if not self._active_state.owes_a_probe():
            return
        try:
            self.load_attempt_observations()
        except Exception as exc:
            self._active_state.defer()
            LOGGER.warning(
                "the owed probe of %s failed (%s: %s); retrying in one window",
                self.active_name,
                type(exc).__name__,
                exc,
            )

    def _degraded_attempt_observations(self) -> list[dict[str, Any]]:
        """What a restart can still restore while ``<name>-active`` refuses.

        A release that upgrades into a cluster whose new object (or Role) is not
        applied yet has the previous release's records right there in the WAL
        object, unmigrated. Returning nothing threw away the restart memory
        this whole path exists for; they are read, not dropped, so the
        migration still owns moving them.
        """

        try:
            attempts = _attempt_records(self._read_data(self.name)[0])
        except Exception as exc:
            LOGGER.warning(
                "cannot read the legacy attempt state from %s either (%s: %s)",
                self.name,
                type(exc).__name__,
                exc,
            )
            return []
        return _sorted_records(attempts)

    def load_attempt_observations(self) -> list[dict[str, Any]]:
        self._migrate_legacy_attempt_state()
        if self._active_state.degraded():
            return self._degraded_attempt_observations()
        try:
            data, _resource_version = self._read_data(self.active_name)
        except Exception as exc:
            if not self._active_state.refused(exc):
                raise
            return self._degraded_attempt_observations()
        self._active_state.usable()
        attempts = _attempt_records(data)
        self._attempt_digests = {
            key: _attempt_digest(payload) for key, payload in attempts.items()
        }
        return _sorted_records(attempts)

    def save_attempt_observation(self, payload: dict[str, Any]) -> bool:
        key = _attempt_key(payload)
        digest = _attempt_digest(payload)
        if self._attempt_digests.get(key) == digest:
            return False
        if not self._before_active_state_write():
            return False

        def update(data: dict[str, str]) -> dict[str, str]:
            attempts = _attempt_records(data)
            stored = attempts.get(key)
            if stored is not None and _attempt_digest(stored) == digest:
                # The same test the digest cache applies, applied to what the
                # object actually holds -- the cache itself cannot answer here
                # because it is empty after a restart, after a migration and
                # after every recovery from a degraded object, and the mutation
                # always returns a new document, so ``_mutate_data``'s
                # unchanged-mutation shortcut never fires on its own. One pass
                # then rewrote the whole document once per attempt -- 125
                # attempts of ~830 KB each. Comparing the records instead would
                # be no guard at all: ``observed_at`` is stamped every pass and
                # is exactly what ``_attempt_digest`` leaves out.
                return data
            attempts[key] = dict(payload)
            result = dict(data)
            result[self.ATTEMPTS_KEY] = _attempt_document(attempts)
            return result

        try:
            self._mutate_data(self.active_name, update)
        except Exception as exc:
            if not self._active_state.refused(exc):
                raise
            return False
        self._active_state.usable()
        self._attempt_digests[key] = digest
        return True

    def remove_attempt_observation(self, payload: dict[str, Any]) -> None:
        key = _attempt_key(payload)
        if not self._before_active_state_write():
            return

        try:
            self._mutate_data(self.active_name, _attempt_removal(key))
        except Exception as exc:
            if not self._active_state.refused(exc):
                raise
            return
        self._active_state.usable()
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
        key = _record_key(path, payload)
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
            # record stays, flagged delivered so that replay drops it instead
            # of sending it again.
            self._count_append_failure(key, removal_error, stage="clear")
            self._reclaim_delivered(key, removal_error)
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

        A record flagged ``delivered`` was already accepted by the control
        plane and only failed to be cleared, so it is dropped here without
        being sent again; it is not counted in the returned number, which
        stays "records this pass delivered".
        """

        if self._known_empty(include_quarantined=include_quarantined):
            # The WAL is empty and this process emptied it, so there is nothing
            # to read: the watcher replays at the top of every reconcile pass and
            # on every debounced flush, and on a healthy cluster that was a
            # whole-document GET per pass for an empty document (F6).
            self.replay_reads_skipped_total += 1
            self.last_depth = 0
            self.last_quarantined_depth = 0
            self.last_replay = {"replayed": 0, "deferred": 0, "quarantined": 0}
            return 0
        try:
            records, _resource_version = self._read()
        except Exception:
            # Nothing was read, so there is no snapshot to publish: stale
            # gauges are still true of the last pass, a fabricated 0 would not
            # be true of anything.
            LOGGER.warning(
                "completion outbox replay could not read its ConfigMap; the "
                "depth gauges keep their last known values "
                "(depth=%d quarantined=%d)",
                self.last_depth,
                self.last_quarantined_depth,
            )
            raise
        drained: set[str] = set()
        isolated: set[str] = set()
        try:
            return self._replay_pass(
                records,
                include_quarantined=include_quarantined,
                drained=drained,
                isolated=isolated,
            )
        finally:
            # Even a pass that died on an unwritable ConfigMap has to leave the
            # depth behind: that is exactly when an operator needs it.
            self.last_depth, self.last_quarantined_depth = _remaining_depth(
                records, drained=drained, isolated=isolated
            )

    def _known_empty(self, *, include_quarantined: bool) -> bool:
        """Whether this process can prove the WAL holds nothing to replay.

        Only a known zero counts. ``None`` -- a fresh process, or any write that
        raised -- reads, because another watcher generation, an operator or a
        half-applied write of our own may have left a record behind. A pass that
        was asked for quarantined records also refuses to skip unless the
        quarantined count is a known zero.
        """

        if self._known_depth != 0:
            return False
        return not include_quarantined or self._known_quarantined == 0

    def _replay_pass(
        self,
        records: list[dict[str, Any]],
        *,
        include_quarantined: bool,
        drained: set[str],
        isolated: set[str],
    ) -> int:
        """One replay pass over an already read snapshot.

        ``drained`` and ``isolated`` are filled in as the pass goes so that
        ``replay`` can publish the depth gauges even when this raises half way
        through.
        """

        candidates = [
            record
            for record in records
            if include_quarantined or not record.get("quarantined", False)
        ]
        batch = candidates[: self.replay_batch_size]
        deferred = len(candidates) - len(batch)
        replayed = quarantined = 0
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
            if record.get("delivered", False):
                # The control plane accepted this event already; only the
                # write that should have cleared it failed. Dropping it is the
                # whole job -- re-sending would be a duplicate.
                LOGGER.info(
                    "dropping a delivered completion record whose removal "
                    "failed earlier: key=%s",
                    key,
                )
                self._remove(key)
                drained.add(key)
                continue
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
        return replayed

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
    if replay is not None:
        try:
            replay()
        except Exception:
            logger.exception("cannot replay Completion Watcher outbox")
    # The same "top of every pass" hook pays for the one probe a recovered
    # attempt-state object is owed: an idle cluster saves no attempt state, so
    # nothing else would ever notice that the manifest was applied (I1).
    probe = getattr(sink, "probe_active_state", None)
    if probe is None:
        return
    try:
        probe()
    except Exception:
        logger.exception("cannot probe the Completion Watcher attempt state")
