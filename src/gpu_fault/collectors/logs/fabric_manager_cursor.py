"""Durable journal/file positions for one serial Fabric Manager reader."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from gpu_fault.collectors.logs.fabric_manager_receipts import identity_sha256


class FabricManagerCursorState:
    """Own checkpoint identity, rotation inheritance and batch durability."""

    def __init__(
        self,
        path: Path | None,
        *,
        max_tracked_files: int,
        logger: logging.Logger,
    ) -> None:
        if max_tracked_files < 1:
            raise ValueError("Fabric Manager tracked file limit must be at least 1")
        self.path = path
        self.max_tracked_files = max_tracked_files
        self.logger = logger
        self.journal_cursor: str | None = None
        self._files: dict[str, dict[str, int]] = {}
        self._dirty = False
        self._bound_warned = False
        self._load()

    def clear_journal_cursor(self) -> None:
        self.journal_cursor = None
        self._dirty = True

    def generation(self, path: str) -> int:
        return self._files[path].get("generation", 0)

    def resume_offsets(
        self, live: list[tuple[str, os.stat_result]]
    ) -> list[tuple[str, os.stat_result, int]]:
        plan: list[tuple[str, os.stat_result, int]] = []
        baselined: set[str] = set()
        checkpoints = dict(self._files)
        # Resolve all renamed inodes against the same pre-rotation snapshot.
        for key, stat in live:
            previous = checkpoints.get(key)
            if previous is None or previous.get("inode") != stat.st_ino:
                inherited = self._inherit_rotated_offset(
                    key, stat, live=live, checkpoints=checkpoints
                )
                if inherited is None and previous is None:
                    # Lost checkpoints do not authorize replaying historical logs.
                    self._files[key] = {
                        "device": stat.st_dev,
                        "inode": stat.st_ino,
                        "offset": stat.st_size,
                        "generation": uuid4().int,
                    }
                    self._dirty = True
                    baselined.add(key)
        for key, stat in live:
            if key in baselined:
                continue
            previous = self._files[key]
            recorded = int(previous.get("offset", 0))
            if previous.get("inode") != stat.st_ino or stat.st_size < recorded:
                self._files[key] = {
                    "device": stat.st_dev,
                    "inode": stat.st_ino,
                    "offset": 0,
                    "generation": previous.get("generation", 0) + 1,
                }
                self._dirty = True
                plan.append((key, stat, 0))
                continue
            if previous.get("device") != stat.st_dev:
                # A root device number can change at boot without a new file.
                previous["device"] = stat.st_dev
                self._dirty = True
            plan.append((key, stat, min(recorded, stat.st_size)))
        return plan

    def _inherit_rotated_offset(
        self,
        key: str,
        stat: os.stat_result,
        *,
        live: list[tuple[str, os.stat_result]],
        checkpoints: dict[str, dict[str, int]],
    ) -> int | None:
        current = {name: (item.st_dev, item.st_ino) for name, item in live}
        identity = (stat.st_dev, stat.st_ino)
        for other, record in checkpoints.items():
            if other == key:
                continue
            if (record.get("device"), record.get("inode")) != identity:
                continue
            if current.get(other) == identity:
                # A still-present hard link is not a renamed source.
                continue
            offset = int(record.get("offset", 0))
            generation = record.get("generation", 0)
            if stat.st_size < offset:
                offset = 0
                generation += 1
            if other not in current:
                self._files.pop(other, None)
            self._files[key] = {
                "device": stat.st_dev,
                "inode": stat.st_ino,
                "offset": offset,
                "generation": generation,
            }
            self._dirty = True
            self.logger.warning(
                "Fabric Manager log file_sha256=%s was rotated to file_sha256=%s; "
                "resuming its tail at byte %d instead of baselining at the end",
                identity_sha256(other),
                identity_sha256(key),
                offset,
            )
            return offset
        return None

    def file_checkpoint(
        self, key: str, stat: os.stat_result, offset: int
    ) -> dict[str, Any]:
        return {
            "kind": "file",
            "path": key,
            "device": stat.st_dev,
            "inode": stat.st_ino,
            "offset": offset,
            "generation": self.generation(key),
        }

    def bound_files(self, seen: set[str]) -> None:
        """Evict only disappeared paths; live files must not be rebaselined."""

        excess = len(self._files) - self.max_tracked_files
        if excess <= 0:
            return
        stale = [key for key in self._files if key not in seen]
        evicted = stale[:excess]
        for key in evicted:
            del self._files[key]
        if evicted:
            self._dirty = True
        if not self._bound_warned:
            self._bound_warned = True
            self.logger.warning(
                "Fabric Manager tracked file table exceeded %d entries; "
                "forgot %d stale offset(s), %d live file(s) kept",
                self.max_tracked_files,
                len(evicted),
                len(self._files),
            )

    def commit_record(self, record: dict[str, Any]) -> None:
        checkpoint = record.pop("_checkpoint", None)
        if not checkpoint:
            return
        if checkpoint["kind"] == "journal":
            cursor = str(checkpoint.get("cursor") or "")
            if cursor:
                self.journal_cursor = cursor
        elif checkpoint["kind"] == "file":
            self._files[str(checkpoint["path"])] = {
                "device": int(checkpoint["device"]),
                "inode": int(checkpoint["inode"]),
                "offset": int(checkpoint["offset"]),
                "generation": int(checkpoint.get("generation", 0)),
            }
        else:
            raise ValueError(
                "unknown Fabric Manager checkpoint kind: " + str(checkpoint["kind"])
            )
        self._dirty = True

    def flush(self) -> None:
        """One durability point per batch; generation changes flush before send."""

        if not self._dirty:
            return
        self._save()
        self._dirty = False

    def _load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("Fabric Manager collector state must be an object")
            self.journal_cursor = value.get("journal_cursor")
            raw_files = value.get("files") or {}
            if not isinstance(raw_files, dict):
                raise ValueError("Fabric Manager file checkpoints must be an object")
            self._files = {
                str(path): {
                    "device": int(state["device"]),
                    "inode": int(state["inode"]),
                    "offset": int(state["offset"]),
                    "generation": int(state.get("generation", 0)),
                }
                for path, state in raw_files.items()
                if int(state["offset"]) >= 0 and int(state.get("generation", 0)) >= 0
            }
        except (KeyError, OSError, ValueError, TypeError, json.JSONDecodeError):
            self.logger.error("cannot load Fabric Manager collector state")
            self.journal_cursor = None
            self._files = {}

    def _save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {"journal_cursor": self.journal_cursor, "files": self._files},
                    separators=(",", ":"),
                )
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.path)
        directory_fd = os.open(
            self.path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
