"""Owned local process helpers and explicit in-memory orchestration doubles."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Condition, Event
from typing import Any

import pytest

from gpu_fault.store import InMemoryStore
from scripts.e2e.regional.ha011_contracts import LEASE_SECONDS, SPOOL_PATH, ProofError
from scripts.e2e.regional.probes import ha011_probe as probe
from tests.regional._cov95_ha011_support import POD_UID, RUN_ID


@pytest.fixture(name="isolated")
def isolated_fixture(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("GPU_FAULT_", "PG", "AWS_", "KUBE"))
        and key.lower() not in {"http_proxy", "https_proxy", "all_proxy"}
    }
    environment.update(
        {
            "AWS_CONFIG_FILE": os.devnull,
            "AWS_SHARED_CREDENTIALS_FILE": os.devnull,
            "AWS_EC2_METADATA_DISABLED": "true",
            "KUBECONFIG": os.devnull,
            "HA011_ISOLATION_ID": RUN_ID,
            "POD_UID": POD_UID,
            "POD_NAMESPACE": "gf-regional-ha011-" + RUN_ID,
            "HA011_BOUNDARY": probe.BOUNDARY,
        }
    )
    monkeypatch.setattr(os, "environ", environment)
    monkeypatch.setattr(probe, "ARM_FILE", tmp_path / "armed.json")
    monkeypatch.setattr(probe, "SERVICE_ACCOUNT", tmp_path / "must-not-exist")
    monkeypatch.setattr(probe, "PASSWORD_FILE", tmp_path / "private-password")
    probe.PASSWORD_FILE.write_text(
        "public-fake-private-password-for-isolated-tests", encoding="utf-8"
    )
    # Inline probe entrypoints disable logging in what would be a separate process.
    saved_disable = logging.root.manager.disable
    try:
        yield tmp_path
    finally:
        logging.disable(saved_disable)


class Capture:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.changed = Condition()
        self.closed = False

    def send(self, event: dict[str, Any]) -> None:
        with self.changed:
            self.events.append(event)
            self.changed.notify_all()

    def close(self) -> None:
        self.closed = True

    def wait(self, kind: str, timeout: float = 5) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        with self.changed:
            while True:
                for event in self.events:
                    if event["kind"] == kind:
                        return event
                remaining = deadline - time.monotonic()
                assert remaining > 0, (
                    f"owned runtime did not emit {kind} before timeout"
                )
                self.changed.wait(remaining)


def owned_idle_child(role, request_id, connection, stop, release) -> None:
    connection.send(
        {"kind": "ready", "pid": os.getpid(), "role": role, "request_id": request_id}
    )
    stop.wait(10)
    connection.close()


def owned_exit_child(role, request_id, connection, stop, release) -> None:
    connection.close()


class ReleasingEvent:
    def __init__(self, on_release) -> None:
        self.event = Event()
        self.on_release = on_release

    def is_set(self) -> bool:
        return self.event.is_set()

    def set(self) -> None:
        if not self.event.is_set():
            self.event.set()
            self.on_release()


class MemoryWorkers:
    def __init__(self, store: InMemoryStore, *, fail: str = "") -> None:
        self.store = store
        self.fail = fail
        self.workers: list[MemoryWorker] = []

    def __call__(self, target, *, role: str, request_id: str):
        worker = MemoryWorker(self, len(self.workers) % 3, role, request_id)
        self.workers.append(worker)
        return worker


class MemoryWorker:
    def __init__(
        self, factory: MemoryWorkers, stage: int, role: str, target: str
    ) -> None:
        self.factory, self.stage, self.role, self.target = factory, stage, role, target
        self.owner = f"{POD_UID}:{101 + stage}"
        self.events: list[dict[str, Any]] = [{"kind": "ready", "owner": self.owner}]
        self.running = True
        self.current = None
        self.release = ReleasingEvent(self.complete_current)
        now = datetime.now(timezone.utc)
        if stage == 2:
            while rows := self.claim(now):
                self.complete(rows[0])
            now += timedelta(seconds=LEASE_SECONDS + 1)
        rows = self.claim(now)
        if rows:
            self.current = rows[0]
            self.events.extend(
                [{"kind": "busy", "request_id": target, "cpu_seconds": 0.1}]
            )
            if stage == 2 and factory.fail == "duplicate-claim":
                self.events.append(
                    {
                        "kind": "claim",
                        "owner": self.owner,
                        "request_id": target,
                        "item": self.current,
                    }
                )
        else:
            self.events.append({"kind": "idle"})
        if stage == 2 and factory.fail == "not-busy":
            self.running = False

    def claim(self, now):
        method = (
            self.factory.store.claim_telemetry_spool
            if self.role == "spool"
            else self.factory.store.claim_active_processor_requests
        )
        kwargs = {"path": SPOOL_PATH} if self.role == "spool" else {}
        rows = method(
            self.owner,
            now=now,
            lease_duration=timedelta(seconds=LEASE_SECONDS),
            limit=1,
            **kwargs,
        )
        for item in rows:
            self.events.append(
                {
                    "kind": "claim",
                    "owner": self.owner,
                    "request_id": item.request_id,
                    "item": item,
                }
            )
        return rows

    def complete(self, item) -> None:
        if self.role == "spool":
            assert self.factory.store.complete_telemetry_spool([item]) == 1, (
                "fake current owner lost its spool claim"
            )
        else:
            self.factory.store.complete_active_processor_request(
                item.request_id,
                self.owner,
                item.leader_epoch,
                item.lease_token,
                response_status=200,
                response_content_type="application/json",
                response_body_base64="e30=",
                path=item.path,
            )
        self.events.append({"kind": "complete", "request_id": item.request_id})

    def complete_current(self) -> None:
        if self.current is not None and self.running:
            self.complete(self.current)
            if self.factory.fail == "duplicate-completion":
                self.events.append(
                    {"kind": "complete", "request_id": self.current.request_id}
                )

    def wait(self, kind: str, deadline: float, request_id: str | None = None):
        for event in self.events:
            if event["kind"] == kind and (
                request_id is None or event.get("request_id") == request_id
            ):
                return event
        raise ProofError("fake runtime did not produce the required event")

    def crash(self) -> int:
        self.running = False
        return -9

    def finish(self) -> bool:
        self.running = False
        return True

    def close(self) -> None:
        self.running = False

    def alive(self) -> bool:
        return self.running
