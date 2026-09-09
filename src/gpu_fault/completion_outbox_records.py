"""The ConfigMap documents of the completion outbox, and pure mutations of them.

Split out of ``completion_outbox`` because that module and its outbox class
are both at their architecture size limits: everything here reads and writes
records and documents and nothing here logs, counts or touches the API server.
The write-ahead log is a JSON list of records under ``EVENT_LOG_KEY``; the
routine attempt state is a JSON object under ``ATTEMPT_STATE_KEY``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

#: Key both objects use for attempt state: the one the WAL object carried
#: before F6 split them, and the only key of ``<name>-active`` after it.
ATTEMPT_STATE_KEY = "active-attempts.json"
#: Key of the write-ahead log inside the WAL object.
EVENT_LOG_KEY = "events.json"


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


def _buffered_attempt_ids(records: list[dict[str, Any]]) -> frozenset[str]:
    """The attempt ids the write-ahead log still names.

    Read off the record keys (``<cluster>/<attempt>/<kind>``) so the controller
    can hold an attempt's state for as long as any event of it is buffered:
    once it has evicted the attempt, a quarantined record has no live path
    left and only the operator lever can deliver it.
    """

    ids = set()
    for record in records:
        parts = str(record.get("key") or "").split("/")
        if len(parts) >= 2 and parts[1]:
            ids.add(parts[1])
    return frozenset(ids)


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


def _event_document(records: list[dict[str, Any]]) -> str:
    """The write-ahead log as the ConfigMap stores it."""

    return json.dumps(records, sort_keys=True, separators=(",", ":"), default=str)


def _event_log_byte_budget(data: dict[str, str], max_bytes: int) -> int:
    """How many encoded bytes the write-ahead document may take in ``data``.

    The byte bound is on the whole object -- every key and value -- so the
    budget of the log is what the other keys (a not yet dropped legacy
    attempt-state document) leave of it, less the log's own key.
    """

    others = sum(
        len(key.encode()) + len(value.encode())
        for key, value in data.items()
        if key != EVENT_LOG_KEY
    )
    return max_bytes - others - len(EVENT_LOG_KEY.encode())


def _payload_digest(payload: Any) -> str:
    """sha256 of a record's payload, canonical JSON; the evidence an ERROR
    keeps of a record that is about to leave the log."""

    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _record_age_key(record: dict[str, Any]) -> float:
    buffered_at = record.get("buffered_at")
    return float(buffered_at) if isinstance(buffered_at, (int, float)) else 0.0


def _oldest_quarantined(records: list[dict[str, Any]]) -> int | None:
    """Index of the quarantined record that has been buffered longest.

    A record without a readable ``buffered_at`` sorts *first* (its age key is
    0, older than any real instant): a hand-edited record with no timestamp is
    the one with the least claim on a slot. Ties fall back to list order,
    which is insertion order.
    """

    candidates = [
        index for index, record in enumerate(records) if record.get("quarantined")
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda index: (_record_age_key(records[index]), index))


def _evict_quarantined_for_room(
    records: list[dict[str, Any]],
    *,
    max_records: int,
    byte_budget: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """``(kept, evicted)``: drop quarantined records, oldest first, until it fits.

    R5: a quarantined record is the operator's evidence and the one-shot's
    input, so it stays for as long as the log has room -- but never at the
    price of a critical event's slot. Only quarantined records are ever
    evicted, and only as many as the bound needs; a log that is full of
    live records is left as it is, and the caller's write then fails with
    ``CompletionOutboxFull`` exactly as before. The caller applies this
    inside the same mutation as its append, so eviction and append are one
    ConfigMap replace and a 409 retry re-derives both from the fresh read.
    """

    kept = list(records)
    evicted: list[dict[str, Any]] = []
    while len(kept) > max_records or len(_event_document(kept).encode()) > byte_budget:
        victim = _oldest_quarantined(kept)
        if victim is None:
            break
        evicted.append(kept.pop(victim))
    return kept, evicted
