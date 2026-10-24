"""Durable log positions and bounded continuation decoding."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


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
