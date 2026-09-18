"""Phase-attributed claim samples for the AUTH-016 token overlap window."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from scripts.e2e.regional.identity_acceptance_common import (
    IdentityAcceptanceError,
    utc_now,
)


class TokenRotationSampler:
    def __init__(
        self, old_token: str, new_token: str, sample: Callable[[str], int | str]
    ) -> None:
        self.old_token = old_token
        self.new_token = new_token
        self.sample = sample
        self.token_lock = threading.Lock()
        self.token = old_token
        self.phase = "baseline"
        self.stop = threading.Event()
        self.samples: list[dict[str, Any]] = []

    def run(self) -> None:
        while not self.stop.is_set():
            with self.token_lock:
                token, phase = self.token, self.phase
            if self.stop.is_set():
                return
            credentials = [("old" if token == self.old_token else "new", token)]
            if phase == "overlap":
                credentials = [("old", self.old_token), ("new", self.new_token)]
            for slot, credential in credentials:
                self.samples.append(
                    {
                        "observed_at": utc_now(),
                        "phase": phase,
                        "credential_slot": slot,
                        "status": self.sample(credential),
                    }
                )
            self.stop.wait(2)

    def enter(self, phase: str, *, token: str | None = None) -> None:
        with self.token_lock:
            self.phase = phase
            if token is not None:
                self.token = token

    def require_phase(self, phase: str, slots: set[str]) -> None:
        measured = [item for item in self.samples if item["phase"] == phase]
        if (
            not measured
            or any(
                type(item["status"]) is not int or item["status"] != 200
                for item in measured
            )
            or {item["credential_slot"] for item in measured} != slots
        ):
            raise IdentityAcceptanceError(
                f"rotation {phase} contains failed or missing samples"
            )
