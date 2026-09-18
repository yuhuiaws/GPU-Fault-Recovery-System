"""Pinned activation ownership and clocks at the actual Secret write boundary."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from gpu_fault.admin.deadlines import Deadline, current_deadline
from gpu_fault.admin.node_key_custody_activation_state import (
    require_owned_wave,
    validate_activation_state,
)
from gpu_fault.admin.node_key_custody_chain import authorize_now
from gpu_fault.admin.node_key_custody_crypto import read_regular
from gpu_fault.admin.node_key_custody_models import Authorization, CustodyError, Signed


class ActivationWriterGuard:
    def __init__(
        self,
        path: Path,
        expected_sha256: str,
        authorization: Signed[Authorization],
        *,
        now: Callable[[], datetime],
    ) -> None:
        self.path = path
        self.sha256 = expected_sha256
        self.authorization = authorization
        self.now = now
        raw = self.read_bound_state()
        self.state = validate_activation_state(raw, authorization)
        if (
            authorization.statement.purpose != "rotate"
            or "FENCED" not in self.state.completed
            or "KEYS_PROVISIONED" not in self.state.started
            or "KEYS_PROVISIONED" in self.state.completed
            or "UNFENCED" in self.state.started
        ):
            raise CustodyError("custody writer activation phase is not authorized")
        remaining = (self.state.deadline - self.now()).total_seconds()
        parent = current_deadline()
        expires = time.monotonic() + remaining
        self.deadline = Deadline(
            "custody activation key writer",
            min(expires, parent.expires) if parent is not None else expires,
        )
        self.last_wall = max(
            (
                datetime.fromisoformat(item["completed_at"])
                for item in self.state.completed.values()
            ),
            default=self.state.started_at,
        )
        self.check()

    def read_bound_state(self) -> bytes:
        raw = read_regular(self.path, private=True)
        if hashlib.sha256(raw).hexdigest() != self.sha256:
            raise CustodyError("custody writer activation state identity changed")
        return raw

    def check(self) -> float:
        self.read_bound_state()
        now = self.now()
        authorize_now(self.authorization.statement, now)
        if now < self.last_wall or now >= self.state.deadline:
            raise CustodyError(
                "custody writer activation deadline expired or clock regressed"
            )
        remaining = min(
            (self.state.deadline - now).total_seconds(),
            self.deadline.remaining(),
        )
        if (parent := current_deadline()) is not None:
            remaining = min(remaining, parent.remaining())
        self.last_wall = now
        return remaining

    def before_write(self, read_wave: Callable[[str], dict[str, Any]]) -> float:
        self.check()
        wave = self.state.snapshot["wave"]
        require_owned_wave(read_wave(wave["name"]), wave, self.authorization.statement)
        return self.check()
