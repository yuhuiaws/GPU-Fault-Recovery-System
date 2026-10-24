"""Bounded physical lines and CRI records, with JSON-serializable continuations."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO

CRI = re.compile(r"^(\S+) (stdout|stderr) ([PF]) (.*)$", re.S)
MARKER = b"\n...[node-log-entry-truncated]...\n"


def journal_fields(item: dict[str, Any], boot_id: str) -> dict[str, str]:
    return {
        "_COLLECTOR_BOOT_ID": boot_id,
        **{
            name: str(item[name])
            for name in (
                "__CURSOR",
                "_PID",
                "_TRANSPORT",
                "_SYSTEMD_CGROUP",
                "_BOOT_ID",
                "CONTAINER_ID_FULL",
                "CONTAINER_NAME",
            )
            if name in item and isinstance(item[name], (str, int))
        },
    }


def bounded_bytes(raw: bytes, limit: int) -> bytes:
    if len(raw) <= limit:
        return raw
    available = max(0, limit - len(MARKER))
    head = available // 2
    tail = available - head
    return raw[:head] + MARKER + (raw[-tail:] if tail else b"")


def read_line(
    stream: BinaryIO,
    pending: dict[str, Any],
    budget: Any,
    limit: int,
) -> tuple[int, bytes] | None:
    """Never read more than the remaining raw-byte budget, even without LF."""
    start = int(pending.get("start", stream.tell()))
    buffer = bytes.fromhex(pending.get("hex", ""))
    while not budget.spent():
        remaining = max(0, budget.max_bytes - budget.bytes_read)
        raw = stream.readline(min(16384, remaining))
        if not raw:
            break
        budget.consume_bytes(len(raw))
        buffer = bounded_bytes(buffer + raw, limit)
        if raw.endswith(b"\n"):
            pending.clear()
            return start, buffer
    if buffer:
        pending.update(start=start, hex=buffer.hex())
    return None


def pod_identity(path: str, root: str) -> dict[str, str]:
    try:
        relative = Path(path).resolve().relative_to(Path(root).resolve())
    except (OSError, ValueError):
        return {}
    if len(relative.parts) != 3:
        return {}
    pod, container, _ = relative.parts
    parts = pod.split("_", 2)
    if len(parts) != 3 or not all(parts) or not container:
        return {}
    return {"pod_uid": parts[2], "container_name": container}


def decode_record(
    raw: bytes,
    start: int,
    observed_at: datetime,
    fragments: dict[str, Any],
    limit: int,
) -> tuple[int, datetime, str, dict[str, str]] | None:
    text = raw.decode("utf-8", errors="replace").rstrip("\n")
    match = CRI.match(text)
    if match is None:
        return start, observed_at, text, {"timestamp_source": "collection"}
    stamp, stream, flag, message = match.groups()
    try:
        at = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        if at.tzinfo is None:
            raise ValueError("CRI timestamps require a timezone")
    except ValueError:
        return start, observed_at, text, {"timestamp_source": "invalid-cri"}
    previous = fragments.pop(stream, None)
    if previous:
        start = int(previous["start"])
        at = datetime.fromisoformat(previous["at"])
        message = previous["message"] + message
    message = bounded_bytes(message.encode("utf-8"), limit).decode(
        "utf-8", errors="replace"
    )
    if flag == "P":
        fragments[stream] = {"start": start, "at": at.isoformat(), "message": message}
        return None
    return start, at, message, {"timestamp_source": "cri", "stream": stream}
