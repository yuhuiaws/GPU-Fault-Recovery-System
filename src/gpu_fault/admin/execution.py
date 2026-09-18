"""Deployment deadlines, process lifetime and identity-bound proof reuse."""

from __future__ import annotations

import contextlib
import copy
import os
import subprocess
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TypeVar, cast

from gpu_fault.admin.api_budget import api_environment, deployment_api_budget
from gpu_fault.admin.deadlines import (
    DEADLINE_ENV as DEADLINE_ENV,
    DEADLINE_LABEL_ENV as DEADLINE_LABEL_ENV,
    HARD_DEADLINE_ENV as HARD_DEADLINE_ENV,
    OVERALL_DEPLOY_SECONDS as OVERALL_DEPLOY_SECONDS,
    RECOVERY_ACTIVE_ENV as RECOVERY_ACTIVE_ENV,
    RECOVERY_SECONDS as RECOVERY_SECONDS,
    Deadline as Deadline,
    DeploymentDeadlineExceeded as DeploymentDeadlineExceeded,
    cleanup_deadline as cleanup_deadline,
    current_deadline as current_deadline,
    deadline_environment as deadline_environment,
    deadline_scope as deadline_scope,
    deployment_deadline as _deployment_deadline,
    hard_deadline as hard_deadline,
    recovery_active as recovery_active,
    recovery_deadline as recovery_deadline,
    remaining_timeout as remaining_timeout,
)
from gpu_fault.admin.diagnostics import DriverDiagnostics, diagnostic_command
from gpu_fault.admin.process_supervisor import (
    ensure_supervision_safe,
    interruption_scope,
    run_owned_command,
)


def _hard_cap(expires: float) -> float:
    inherited = hard_deadline()
    return min(expires, inherited) if inherited is not None else expires


def command_environment(environment: Mapping[str, str] | None) -> dict[str, str] | None:
    return api_environment(deadline_environment(environment))


@contextlib.contextmanager
def deployment_deadline(
    label: str, seconds: float, *, recovery_seconds: float = RECOVERY_SECONDS
) -> Iterator[Deadline]:
    with (
        _deployment_deadline(
            label, seconds, recovery_seconds=recovery_seconds
        ) as deadline,
        interruption_scope(wait_all=True),
    ):
        yield deadline


@contextlib.contextmanager
def operation_budget(label: str, *, readonly: bool = False) -> Iterator[None]:
    with (
        deployment_api_budget(),
        deployment_deadline(
            label,
            900 if readonly else OVERALL_DEPLOY_SECONDS,
            recovery_seconds=0 if readonly else RECOVERY_SECONDS,
        ),
    ):
        yield


def command_timeout(arguments: Sequence[str], requested: float | None) -> float:
    executable = os.path.basename(arguments[0]) if arguments else ""
    default = (
        7200
        if executable == "make" or executable.startswith("python")
        else 3600
        if executable in {"docker", "bash", "sh"} or executable.endswith(".sh")
        else 900
        if "exec" in arguments or "wait" in arguments or "rollout" in arguments
        else 120
    )
    return remaining_timeout(requested if requested is not None else float(default))


def run_command(
    arguments: Sequence[str],
    *,
    input_text: str | None = None,
    capture: bool = True,
    environment: Mapping[str, str] | None = None,
    cwd: os.PathLike[str] | None = None,
    timeout_seconds: float | None = None,
    pass_fds: Sequence[int] = (),
) -> subprocess.CompletedProcess[str]:
    ensure_supervision_safe(allow_interrupted=recovery_active())
    expires = time.monotonic() + command_timeout(arguments, timeout_seconds)
    prepared_environment = command_environment(environment)
    timeout = min(
        command_timeout(arguments, timeout_seconds),
        Deadline("command preparation", expires).remaining(),
    )
    return run_owned_command(
        arguments,
        timeout=timeout,
        expires_at=expires,
        allow_interrupted=recovery_active(),
        input_text=input_text,
        capture=capture,
        environment=prepared_environment,
        cwd=cwd,
        pass_fds=pass_fds,
    )


def run_driver(
    arguments: Sequence[str],
    *,
    cwd: os.PathLike[str] | None = None,
    env: Mapping[str, str] | None = None,
    check: bool = False,
    capture_output: bool = False,
    stdout: int | None = None,
    text: bool = True,
    encoding: str | None = None,
    pass_fds: Sequence[int] = (),
) -> subprocess.CompletedProcess[str]:
    """Bound lifecycle driver hops while preserving their descriptor/lock contract."""
    ensure_supervision_safe(allow_interrupted=recovery_active())
    if stdout not in {None, subprocess.PIPE} or encoding not in {None, "utf-8"}:
        raise ValueError("deployment driver uses an unsupported output mode")
    del text
    captured = capture_output or stdout == subprocess.PIPE
    normal = time.monotonic() + command_timeout(arguments, OVERALL_DEPLOY_SECONDS)
    hard = _hard_cap(normal + (0 if recovery_active() else RECOVERY_SECONDS))
    environment = command_environment(env) or dict(os.environ)
    environment.setdefault(DEADLINE_ENV, str(normal))
    environment.setdefault(DEADLINE_LABEL_ENV, "deployment driver")
    environment[HARD_DEADLINE_ENV] = str(hard)
    diagnostics = DriverDiagnostics() if captured and not capture_output else None
    try:
        # A lifecycle driver must remain alive to receive the child's failure and
        # finish compensation. Only leaf operations use the normal deadline; all
        # drivers share this hard ceiling, so nested hops cannot add more reserves.
        result = run_owned_command(
            arguments,
            capture=captured,
            environment=environment,
            cwd=cwd,
            timeout=Deadline("deployment driver hard stop", hard).remaining(),
            expires_at=hard,
            allow_interrupted=recovery_active(),
            pass_fds=pass_fds,
            diagnostics=diagnostics,
        )
    except subprocess.TimeoutExpired:
        raise DeploymentDeadlineExceeded(
            "deployment driver exceeded its deadline: "
            + diagnostic_command(arguments[:2])
        ) from None
    finally:
        if diagnostics is not None:
            diagnostics.finish()
    if check and result.returncode:
        raise subprocess.CalledProcessError(
            result.returncode, diagnostic_command(arguments)
        )
    return result


class ProofSubject(StrEnum):
    TOOLCHAIN = "toolchain"
    ARTIFACT = "artifact"
    CREDENTIAL_SHAPE = "credential-shape"


@dataclass(frozen=True)
class ProofKey:
    subject: ProofSubject
    scope: str
    identity_sha256: str
    verifier_sha256: str
    resource_version: str = ""

    def __post_init__(self) -> None:
        import re

        if not self.scope or any(
            re.fullmatch(r"[0-9a-f]{64}", value) is None
            for value in (self.identity_sha256, self.verifier_sha256)
        ):
            raise ValueError("proof identity is incomplete")
        if self.subject is ProofSubject.CREDENTIAL_SHAPE and not self.resource_version:
            raise ValueError(
                "credential proof requires a freshly observed resource version"
            )


@dataclass
class _Proof:
    value: object = field(repr=False)
    expires: float


_T = TypeVar("_T")


class ProofCache:
    """Only bounded positive proofs; callers recompute the identity before lookup."""

    def __init__(self, *, maximum: int = 256) -> None:
        if maximum <= 0:
            raise ValueError("proof cache capacity must be positive")
        self._entries: dict[ProofKey, _Proof] = {}
        self._lock = threading.RLock()
        self._inflight: dict[ProofKey, threading.Event] = {}
        self._generations: dict[str, int] = {}
        self.maximum = maximum

    def verify(self, key: ProofKey, verify: Callable[[], _T], *, max_age: float) -> _T:
        if not 0 < max_age <= 300:
            raise ValueError("proof age must be between zero and 300 seconds")
        while True:
            with self._lock:
                now = time.monotonic()
                if (
                    entry := self._entries.get(key)
                ) is not None and entry.expires > now:
                    return cast(_T, copy.deepcopy(entry.value))
                pending = self._inflight.get(key)
                if pending is None:
                    self._inflight[key] = pending = threading.Event()
                    generation = self._generations.get(key.scope, 0)
                    break
            deadline = current_deadline()
            pending.wait(timeout=deadline.cap(1) if deadline else 1)
        try:
            value = verify()
            with self._lock:
                if generation == self._generations.get(key.scope, 0):
                    if len(self._entries) >= self.maximum:
                        self._entries = {
                            item: proof
                            for item, proof in self._entries.items()
                            if proof.expires > time.monotonic()
                        }
                        if len(self._entries) >= self.maximum:
                            self._entries.pop(next(iter(self._entries)))
                    self._entries[key] = _Proof(
                        copy.deepcopy(value), time.monotonic() + max_age
                    )
            return value
        finally:
            with self._lock:
                self._inflight.pop(key).set()

    def invalidate(self, scope: str) -> None:
        with self._lock:
            self._generations[scope] = self._generations.get(scope, 0) + 1
            self._entries = {
                key: value for key, value in self._entries.items() if key.scope != scope
            }


PROOFS = ProofCache()
