"""Durable refusal after losing control of an approved command process tree."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import threading

from gpu_fault.admin.atomic_json import write_json_atomic

_LOCK = threading.Lock()
_MARKER: Path | None = None


def require_supervision_clear(run_dir: Path) -> None:
    marker = run_dir.resolve() / "command-supervision-lost.json"
    if marker.exists() or marker.is_symlink():
        raise RuntimeError(
            "this acceptance run lost command supervision; independent recovery "
            "and cleanup verification are required before any retry"
        )


def bind_command_supervision(run_dir: Path) -> None:
    global _MARKER
    with _LOCK:
        require_supervision_clear(run_dir)
        _MARKER = run_dir.resolve() / "command-supervision-lost.json"


def record_supervision_loss() -> None:
    with _LOCK:
        if _MARKER is not None:
            write_json_atomic(
                _MARKER,
                {
                    "schema_version": 1,
                    "status": "RECOVERY_REQUIRED",
                    "recorded_at": datetime.now(timezone.utc).isoformat(),
                    "reason": "command process-tree termination could not be verified",
                },
            )
