"""Attempt-bound notification observations; unknown ACKs never trigger another send."""

from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from scripts.e2e.regional.acceptance_runner_common import write_json_atomic
from scripts.e2e.regional.notification_evidence import NotificationAcceptanceError


def cached_drill(
    path: Path, identity: Mapping[str, Any], execute: Callable[[], dict[str, Any]]
) -> dict[str, Any]:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(
        path.with_suffix(".lock"), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600
    )
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise NotificationAcceptanceError(
                "notification drill is already running"
            ) from exc
        if path.is_symlink():
            raise NotificationAcceptanceError(
                "notification observation cache is a symlink"
            )
        if path.exists():
            document = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(document, dict) or document.get("identity") != dict(
                identity
            ):
                raise NotificationAcceptanceError(
                    "notification observation identity changed"
                )
            if document.get("state") != "COMPLETED" or not isinstance(
                document.get("result"), dict
            ):
                raise NotificationAcceptanceError(
                    "notification drill outcome is unconfirmed; do not send again in this attempt"
                )
            return document["result"]
        document = {"identity": dict(identity), "state": "STARTED"}
        write_json_atomic(path, document)
        result = execute()
        if not isinstance(result, dict):
            raise NotificationAcceptanceError(
                "notification drill returned no observation"
            )
        write_json_atomic(path, {**document, "state": "COMPLETED", "result": result})
        return result
    finally:
        os.close(fd)
