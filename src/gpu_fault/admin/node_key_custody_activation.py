"""Bounded, fail-forward activation of one independently authorized node key."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Protocol

from gpu_fault.admin.atomic_json import write_json_atomic
from gpu_fault.admin.bootstrap_common import BootstrapMutationRequired, CommandRunner
from gpu_fault.admin.execution import deadline_scope
from gpu_fault.admin.node_key_custody_activation_state import (
    ACTIVATION_STEPS as ACTIVATION_STEPS,
    ACTIVATION_TIMEOUT_SECONDS as ACTIVATION_TIMEOUT_SECONDS,
    ActivationState as ActivationState,
    validate_activation_state,
)
from gpu_fault.admin.node_key_custody_admin_probe import AdminNodeKeyContext
from gpu_fault.admin.node_key_custody_chain import authorize_now
from gpu_fault.admin.node_key_custody_crypto import (
    private_directory,
    read_regular,
)
from gpu_fault.admin.node_key_custody_models import (
    Authorization,
    Completed,
    CustodyError,
    Signed,
    statement_sha256,
)


class ActivationIO(Protocol):
    def bind_completed(self, completed: Completed) -> None: ...
    def keys_provisioned(self, snapshot: dict[str, Any]) -> dict[str, Any]: ...
    def capture(self) -> dict[str, Any]: ...
    def verify(self, snapshot: dict[str, Any]) -> None: ...
    def require_owned_wave(self, snapshot: dict[str, Any]) -> None: ...
    def guard(self, snapshot: dict[str, Any]) -> dict[str, Any]: ...
    def fence(self, snapshot: dict[str, Any]) -> dict[str, Any]: ...
    def refresh_executor(self, snapshot: dict[str, Any]) -> dict[str, Any]: ...
    def install(self, snapshot: dict[str, Any]) -> dict[str, Any]: ...
    def refresh_cpu(self, snapshot: dict[str, Any]) -> dict[str, Any]: ...
    def observe(self, snapshot: dict[str, Any]) -> dict[str, Any]: ...
    def unfence(self, snapshot: dict[str, Any]) -> dict[str, Any]: ...
    def current(self, snapshot: dict[str, Any]) -> dict[str, Any]: ...


def activation_path(context: AdminNodeKeyContext, authorization: Authorization) -> Path:
    return (
        context.state_dir
        / "node-key-custody"
        / "activation"
        / (authorization.transaction_id + ".json")
    )


def activation_io(
    runner: CommandRunner,
    context: AdminNodeKeyContext,
    authorization: Authorization,
) -> ActivationIO:
    from gpu_fault.admin.node_key_custody_activation_io import CustodyActivationIO

    return CustodyActivationIO(runner, context, authorization)


class CustodyActivation:
    """Operational progress is not an independent signed Activated receipt."""

    def __init__(
        self,
        io: ActivationIO,
        context: AdminNodeKeyContext,
        authorization: Signed[Authorization],
        *,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self.io = io
        self.context = context
        self.authorization = authorization
        self.now = now
        self.path = activation_path(context, authorization.statement)
        self.state: ActivationState | None = None

    def load(self) -> ActivationState | None:
        if not self.path.exists():
            return None
        return validate_activation_state(
            read_regular(self.path, private=True), self.authorization
        )

    def check_window(self) -> None:
        approve = self.authorization.statement
        now = self.now()
        authorize_now(approve, now)
        if self.state is not None:
            last = max(
                (
                    datetime.fromisoformat(item["completed_at"])
                    for item in self.state.completed.values()
                ),
                default=self.state.started_at,
            )
            if now < self.state.started_at or now < last:
                raise CustodyError("node key activation clock regressed")
            if now >= self.state.deadline:
                raise CustodyError(
                    "node key activation deadline expired; retain state for reconciliation"
                )

    def probe(self) -> dict[str, str]:
        self.state = self.load()
        if self.state is None or set(self.state.completed) != set(ACTIVATION_STEPS):
            raise BootstrapMutationRequired("authorized node key activation is pending")
        self.io.current(self.state.snapshot)
        return {
            "runtime_activation": "DEPLOYED_NOT_WITNESSED",
            "custody_activation": str(self.path),
        }

    def prepare(self) -> None:
        self.state = self.load()
        if self.state is not None and set(self.state.completed) == set(
            ACTIVATION_STEPS
        ):
            self.probe()
            return
        self.check_window()
        if self.state is None:
            started_at = self.now()
            deadline = min(
                self.authorization.statement.expires_at,
                started_at + timedelta(seconds=ACTIVATION_TIMEOUT_SECONDS),
            )
            with deadline_scope(
                "node key activation preparation",
                (deadline - started_at).total_seconds(),
            ) as preparation:
                snapshot = self.io.capture()
                preparation.remaining()
            authorize_now(self.authorization.statement, self.now())
            if self.now() >= deadline:
                raise CustodyError("node key activation preparation deadline expired")
            self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            private_directory(self.path.parent)
            self.state = ActivationState(
                authorization_sha256=statement_sha256(self.authorization),
                binding_sha256=statement_sha256(self.authorization.statement.binding),
                transaction_id=self.authorization.statement.transaction_id,
                started_at=started_at,
                deadline=deadline,
                snapshot=snapshot,
            )
            self.save()
        self.io.verify(self.state.snapshot)
        if "FENCED" in self.state.completed and "UNFENCED" not in self.state.started:
            self.io.require_owned_wave(self.state.snapshot)
        self.step("GUARDED", self.io.guard)
        self.step("FENCED", self.io.fence)

    @contextmanager
    def key_writer(self) -> Iterator[tuple[Path, str]]:
        """The real helper shares this budget, plus its own pinned write guard."""
        if self.state is None or "FENCED" not in self.state.completed:
            raise CustodyError("node key writer requires a durable owned wave")
        if "KEYS_PROVISIONED" in self.state.completed:
            raise CustodyError("node key writer has already completed")
        self.check_window()
        self.io.verify(self.state.snapshot)
        self.io.require_owned_wave(self.state.snapshot)
        if "KEYS_PROVISIONED" not in self.state.started:
            self.state = self.state.model_copy(
                update={"started": [*self.state.started, "KEYS_PROVISIONED"]}
            )
            self.save()
        digest = hashlib.sha256(read_regular(self.path, private=True)).hexdigest()
        with deadline_scope(
            "node key activation key writer",
            (self.state.deadline - self.now()).total_seconds(),
        ) as deadline:
            yield self.path, digest
            deadline.remaining()
            self.check_window()
            self.io.require_owned_wave(self.state.snapshot)

    def save(self) -> None:
        if self.state is None:
            raise CustodyError("node key activation has no durable intent")
        write_json_atomic(self.path, self.state.model_dump(mode="json"))

    def step(
        self, name: str, operation: Callable[[dict[str, Any]], dict[str, Any]]
    ) -> None:
        if self.state is None:
            raise CustodyError("node key activation has no prepared state")
        if name in self.state.completed:
            return
        if name != ACTIVATION_STEPS[len(self.state.completed)]:
            raise CustodyError("node key activation phase is out of order")
        self.check_window()
        self.io.verify(self.state.snapshot)
        if "FENCED" in self.state.completed and not (
            name == "UNFENCED" and name in self.state.started
        ):
            self.io.require_owned_wave(self.state.snapshot)
        if name not in self.state.started:
            self.state = self.state.model_copy(
                update={"started": [*self.state.started, name]}
            )
            self.save()
        with deadline_scope(
            "node key activation " + name,
            (self.state.deadline - self.now()).total_seconds(),
        ) as deadline:
            evidence = operation(self.state.snapshot)
            deadline.remaining()
            self.check_window()
        self.state = self.state.model_copy(
            update={
                "completed": {
                    **self.state.completed,
                    name: {
                        "completed_at": self.now().isoformat(),
                        "evidence": evidence,
                    },
                }
            }
        )
        self.save()

    def finish(self) -> dict[str, str]:
        self.step("KEYS_PROVISIONED", self.io.keys_provisioned)
        self.step("EXECUTOR_REFRESHED", self.io.refresh_executor)
        self.step("NODE_INSTALLED", self.io.install)
        self.step("CPU_REFRESHED", self.io.refresh_cpu)
        self.step("OBSERVED", self.io.observe)
        self.step("UNFENCED", self.io.unfence)
        return self.probe()
