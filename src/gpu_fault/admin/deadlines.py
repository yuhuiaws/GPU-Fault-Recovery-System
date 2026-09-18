"""Shared deployment clocks for native work and isolated command admission."""

from __future__ import annotations

import contextlib
import contextvars
import math
import os
import threading
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Protocol

DEADLINE_ENV = "GPU_FAULT_DEPLOY_DEADLINE_MONOTONIC"
DEADLINE_LABEL_ENV = "GPU_FAULT_DEPLOY_DEADLINE_LABEL"
HARD_DEADLINE_ENV = "GPU_FAULT_DEPLOY_HARD_DEADLINE_MONOTONIC"
RECOVERY_ACTIVE_ENV = "GPU_FAULT_DEPLOY_RECOVERY_ACTIVE"
OVERALL_DEPLOY_SECONDS = 21600
RECOVERY_SECONDS = 7200
MAX_HTTP_RESPONSE_BYTES = 8 * 1024 * 1024
_CURRENT: contextvars.ContextVar[Deadline | None] = contextvars.ContextVar(
    "deployment_deadline", default=None
)
_RECOVERING: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "deployment_recovery", default=False
)


class DeploymentDeadlineExceeded(TimeoutError):
    pass


class HttpResponseTooLarge(ValueError):
    pass


class HttpResponseReader(Protocol):
    def read1(self, size: int = -1, /) -> bytes: ...


@dataclass(frozen=True)
class Deadline:
    label: str
    expires: float

    def remaining(self) -> float:
        remaining = self.expires - time.monotonic()
        if remaining <= 0:
            raise DeploymentDeadlineExceeded(
                f"deployment deadline exceeded: {self.label}"
            )
        return remaining

    def cap(self, seconds: float) -> float:
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("command timeout must be finite and positive")
        return min(seconds, self.remaining())


def current_deadline() -> Deadline | None:
    active = _CURRENT.get()
    if active is not None:
        return active
    raw = os.environ.get(DEADLINE_ENV)
    if raw is None:
        return None
    try:
        expires = float(raw)
    except ValueError:
        raise ValueError("invalid inherited deployment deadline") from None
    if (
        not math.isfinite(expires)
        or expires > time.monotonic() + OVERALL_DEPLOY_SECONDS
    ):
        raise ValueError("invalid inherited deployment deadline")
    return Deadline(os.environ.get(DEADLINE_LABEL_ENV, "parent deployment"), expires)


def hard_deadline() -> float | None:
    raw = os.environ.get(HARD_DEADLINE_ENV)
    if raw is None:
        return None
    try:
        value = float(raw)
    except ValueError:
        raise ValueError("invalid inherited deployment hard deadline") from None
    if not math.isfinite(value) or value > (
        time.monotonic() + OVERALL_DEPLOY_SECONDS + RECOVERY_SECONDS
    ):
        raise ValueError("invalid inherited deployment hard deadline")
    return value


def recovery_active() -> bool:
    return _RECOVERING.get() or os.environ.get(RECOVERY_ACTIVE_ENV) == "true"


def remaining_timeout(seconds: float) -> float:
    """Cap one native or subprocess operation at the task and hard deadlines."""
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("command timeout must be finite and positive")
    deadline = current_deadline()
    if deadline is not None:
        seconds = deadline.cap(seconds)
    if (hard := hard_deadline()) is not None:
        seconds = Deadline("deployment hard stop", hard).cap(seconds)
    return seconds


def read_http_response(response: HttpResponseReader) -> bytes:
    """Bound body bytes inside the isolated HTTP worker.

    Framing, trailers, DNS and TLS may block within a library call. The worker's
    independent process supervisor, not these between-read checks, enforces the
    absolute deadline.
    """
    body = bytearray()
    while True:
        remaining_timeout(OVERALL_DEPLOY_SECONDS)
        chunk = response.read1(min(65536, MAX_HTTP_RESPONSE_BYTES + 1 - len(body)))
        remaining_timeout(OVERALL_DEPLOY_SECONDS)
        if not chunk:
            return bytes(body)
        if len(body) + len(chunk) > MAX_HTTP_RESPONSE_BYTES:
            raise HttpResponseTooLarge("native HTTP response exceeds its size limit")
        body.extend(chunk)


def deadline_environment(
    environment: Mapping[str, str] | None,
) -> dict[str, str] | None:
    """Export thread-local task/recovery state without changing process globals."""
    deadline = current_deadline()
    if deadline is None:
        return dict(environment) if environment is not None else None
    values = dict(os.environ if environment is None else environment)
    values[DEADLINE_ENV] = str(deadline.expires)
    values[DEADLINE_LABEL_ENV] = deadline.label
    if (hard := hard_deadline()) is not None:
        values[HARD_DEADLINE_ENV] = str(hard)
    values[RECOVERY_ACTIVE_ENV] = str(recovery_active()).lower()
    return values


def _hard_cap(expires: float) -> float:
    inherited = hard_deadline()
    return min(expires, inherited) if inherited is not None else expires


@contextlib.contextmanager
def _export_deadline(deadline: Deadline, hard: float) -> Iterator[None]:
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    values = {
        DEADLINE_ENV: str(deadline.expires),
        DEADLINE_LABEL_ENV: deadline.label,
        HARD_DEADLINE_ENV: str(hard),
        RECOVERY_ACTIVE_ENV: str(recovery_active()).lower(),
    }
    old = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in old.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@contextlib.contextmanager
def deadline_scope(
    label: str, seconds: float, *, parent: Deadline | None = None
) -> Iterator[Deadline]:
    if not math.isfinite(seconds) or not 0 < seconds <= OVERALL_DEPLOY_SECONDS:
        raise ValueError("deployment time budget is outside its limits")
    parent = parent or current_deadline()
    expires = time.monotonic() + seconds
    deadline = Deadline(label, min(expires, parent.expires) if parent else expires)
    deadline.remaining()
    token = _CURRENT.set(deadline)
    try:
        yield deadline
    finally:
        _CURRENT.reset(token)


@contextlib.contextmanager
def deployment_deadline(
    label: str, seconds: float, *, recovery_seconds: float = RECOVERY_SECONDS
) -> Iterator[Deadline]:
    """Export the root budget for legacy children and nested worker pools."""
    if threading.current_thread() is not threading.main_thread():
        raise ValueError("root deployment deadline must be owned by the CLI thread")
    if not 0 <= recovery_seconds <= RECOVERY_SECONDS:
        raise ValueError("invalid deployment recovery reserve")
    with deadline_scope(label, seconds) as deadline:
        hard = _hard_cap(
            deadline.expires + (0 if recovery_active() else recovery_seconds)
        )
        with _export_deadline(deadline, hard):
            yield deadline


@contextlib.contextmanager
def cleanup_deadline(label: str, seconds: float = 90) -> Iterator[None]:
    if not 0 < seconds <= 120:
        raise ValueError("cleanup grace is outside its bounded budget")
    parent = current_deadline() if recovery_active() else None
    expires = _hard_cap(time.monotonic() + seconds)
    deadline = Deadline(label, min(expires, parent.expires) if parent else expires)
    deadline.remaining()
    token = _CURRENT.set(deadline)
    recovering = _RECOVERING.set(True)
    try:
        yield
    finally:
        _RECOVERING.reset(recovering)
        _CURRENT.reset(token)


@contextlib.contextmanager
def recovery_deadline(label: str) -> Iterator[None]:
    """Compensation keeps its own bounded window after deployment time expires."""
    parent = current_deadline() if recovery_active() else None
    hard = _hard_cap(time.monotonic() + RECOVERY_SECONDS)
    deadline = Deadline(
        label,
        min(hard, parent.expires if parent is not None else math.inf),
    )
    deadline.remaining()
    token = _CURRENT.set(deadline)
    recovering = _RECOVERING.set(True)
    try:
        with _export_deadline(deadline, hard):
            yield
    finally:
        _RECOVERING.reset(recovering)
        _CURRENT.reset(token)
