"""Durable log positions and bounded continuation decoding."""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


def save_state(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(value, separators=(",", ":")))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


@dataclass(frozen=True)
class LoadedState:
    """What the state file said, already checked -- or why there is nothing.

    ``cold_start_reason`` is ``None`` only when a journal position was loaded;
    otherwise it is the sentence the first batch carries to say how long the
    collector may have been blind (see ``NodeLogCollector._journal_window``).
    """

    cold_start_reason: str | None
    journal_since: datetime | None = None
    journal_cursor: str | None = None
    files: dict[str, dict[str, int]] = field(default_factory=dict)
    pending: dict[str, dict[str, Any]] = field(default_factory=dict)
    resume_after: str | None = None


def load_state(path: Path | None, *, boot_id: str, max_entry_bytes: int) -> LoadedState:
    """Read the collector's state file, or say why it starts cold.

    A file that cannot be read as a whole yields a cold start rather than a
    partial state: a cursor without its offsets, or offsets without their
    continuations, would be read as a position this collector never held.
    """

    if path is None:
        return LoadedState("this collector keeps no state file")
    if not path.exists():
        return LoadedState(f"no state file at {path}")
    try:
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError("node log collector state must be an object")
        since = value.get("journal_since")
        files = decode_files(
            value.get("training_log_files") or value.get("training_log_offsets") or {}
        )
        resume_after = value.get("training_log_resume_after")
        return LoadedState(
            cold_start_reason=(
                None if since else "the state file held no journal cursor"
            ),
            journal_since=datetime.fromisoformat(since) if since else None,
            journal_cursor=resumable_cursor(value, boot_id=boot_id),
            files=files,
            pending=decode_pending(
                value.get("training_log_pending", {}), files, max_entry_bytes
            ),
            resume_after=str(resume_after) if resume_after else None,
        )
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        LOGGER.exception("cannot load node log collector state")
        return LoadedState("the state file could not be read")


def resumable_cursor(value: dict[str, Any], *, boot_id: str) -> str | None:
    """The saved journal cursor, when this boot may resume from it.

    A cursor is resumed only by the boot that saved it (``boot_id`` is written
    next to it). After a reboot the position it names may be gone: on a
    volatile journal ``--after-cursor`` then seeks to the closest entry in time
    and steps past it -- the first line of the new boot -- while the time window
    from ``journal_since`` reads everything the new boot wrote. On a persistent
    journal the two reads cover the same entries, so the window is the safe
    choice in both cases; it is not a loss and not a collection error, only the
    boundary entry can repeat under its own id. A state file written before
    the boot was recorded cannot prove the boot and is treated the same way.

    The boot is recorded rather than read out of the cursor: the cursor string
    is opaque to clients (sd_journal_get_cursor(3)), see ``node_journal``.
    """

    cursor = value.get("journal_cursor")
    if not isinstance(cursor, str) or not cursor:
        return None
    saved_boot = value.get("boot_id")
    if saved_boot == boot_id:
        return cursor
    LOGGER.warning(
        "journal cursor saved %s is not resumed on boot %s; the next poll reads by "
        "time window from %s",
        f"on boot {saved_boot}" if isinstance(saved_boot, str) else "without a boot",
        boot_id,
        value.get("journal_since"),
    )
    return None


def decode_files(stored: dict[str, Any]) -> dict[str, dict[str, int]]:
    """Reads both the offset-only and the file-identity state form.

    An agent that is already running has `{path: offset}` on disk under the
    old key, and it keeps it: rejecting that would send the collector back to
    `initial_tail_bytes` on the upgrade, which at the default of 0 means
    skipping every line written since the last poll. A legacy record is kept
    without a device or inode rather than with a placeholder, so the first
    poll afterwards adopts the file it finds instead of reporting that every
    training log rotated.
    """

    if not isinstance(stored, dict):
        raise ValueError("node log file checkpoints must be an object")
    files: dict[str, dict[str, int]] = {}
    for key, item in stored.items():
        record = (
            {
                "device": int(item.get("device", 0)),
                "inode": int(item.get("inode", 0)),
                "offset": int(item.get("offset", 0)),
            }
            if isinstance(item, dict)
            else {"offset": int(item)}
        )
        if record["offset"] >= 0:
            files[str(key)] = record
    return files


def decode_pending(
    value: Any, files: dict[str, Any], max_bytes: int
) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict):
        raise ValueError("log continuations must be an object")
    result = {}
    for key, item in value.items():
        if key not in files:
            continue
        if not isinstance(item, dict):
            raise ValueError("invalid log continuation")
        physical = item.get("physical", {})
        raw = bytes.fromhex(physical.get("hex", ""))
        if len(raw) > max_bytes or int(physical.get("start", 0)) < 0:
            raise ValueError("invalid physical line continuation")
        fragments = item.get("cri", {})
        for name, fragment in fragments.items():
            if name not in {"stdout", "stderr"}:
                raise ValueError("invalid CRI stream")
            if len(fragment["message"].encode("utf-8")) > max_bytes + 4:
                raise ValueError("oversize CRI continuation")
            if (
                int(fragment["start"]) < 0
                or datetime.fromisoformat(fragment["at"]).tzinfo is None
            ):
                raise ValueError("invalid CRI continuation position")
        result[key] = item
    return result
