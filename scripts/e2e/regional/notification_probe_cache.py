"""Attempt-bound notification observations; unknown ACKs never trigger another send."""

from __future__ import annotations

import fcntl
import hashlib
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


# The code a completed drill is bound to. A drill is an observed fact about
# the release (mails sent, provider ids); it is re-judged -- never re-sent --
# while the code that ran and interprets it is unchanged. Binding the whole
# tree instead (the focused-test rule) made a later receipt impossible: the
# evidence record, unrelated runner fixes and the very plan the receipt flags
# alter all changed the identity (2026-09-18).
DRILL_SOURCE_MODULES = (
    "run_notification_acceptance.py",
    "notification_probe_cache.py",
    "notification_evidence.py",
)


def drill_source_digest() -> str:
    digest = hashlib.sha256()
    here = Path(__file__).resolve().parent
    for name in DRILL_SOURCE_MODULES:
        digest.update(b"\0" + name.encode() + b"\0")
        digest.update((here / name).read_bytes())
    return digest.hexdigest()


def drill_plan_binding(plan_path: Path) -> str | None:
    """The plan's ``details_sha256`` -- the drill inputs -- not the whole
    plan, whose ``arguments_sha256`` changes when ``--receipt-evidence`` and
    ``--ses-window-evidence`` are added for the re-judgement."""
    if not plan_path.is_file():
        return None
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    value = plan.get("details_sha256") if isinstance(plan, dict) else None
    return str(value) if value else None
