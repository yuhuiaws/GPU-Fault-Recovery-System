"""One-use ownership challenges issued after queueing, at the physical boundary."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import secrets
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from threading import Event, RLock
from typing import TYPE_CHECKING, Annotated, Any, Final, Literal, TypeVar

from pydantic import ConfigDict, Field, PrivateAttr, field_validator, model_validator

from gpu_fault.models import StrictModel, WorkflowOperation
from gpu_fault.operation_registry import (
    GENERATION_STABLE_COMMAND_OPERATIONS,
    MULTI_NODE_BARRIER_OPERATIONS,
    NODE_MUTATING_OPERATIONS,
)

if TYPE_CHECKING:
    from gpu_fault.node_agent.protocol import NodeActionCommand, SignedNodeAction

LOGGER = logging.getLogger(__name__)

OWNERSHIP_PROTOCOL: Final = "node-final-ownership/v1"
CHALLENGE_SECONDS = 90
PERMIT_SECONDS = 3
_QUIESCE_RECHECK_OPERATIONS = MULTI_NODE_BARRIER_OPERATIONS | (
    GENERATION_STABLE_COMMAND_OPERATIONS - {WorkflowOperation.REMEDIATE_EFA_DRIVER}
)
_CLIENT_RECHECK_OPERATIONS = _QUIESCE_RECHECK_OPERATIONS | {
    WorkflowOperation.RESTART_FABRIC_MANAGER
}
Digest = Annotated[str, Field(strict=True, pattern=r"^[0-9a-f]{64}$")]
Identity = Annotated[str, Field(strict=True, min_length=1, max_length=512)]


def aware_timestamp(value: Any) -> datetime:
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError:
            raise ValueError("ownership timestamp is invalid") from None
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError("ownership timestamp must be timezone aware")
    return value


class OwnershipChallenge(StrictModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    protocol: Literal["node-final-ownership/v1"] = OWNERSHIP_PROTOCOL
    command_id: Identity
    workflow_id: Identity
    incident_id: Identity
    node_id: Identity
    boot_id: Identity
    agent_generation: Annotated[int, Field(strict=True, ge=1)]
    fencing_token: Annotated[int, Field(strict=True, ge=1)]
    command_sha256: Digest
    nonce: Digest
    sequence: Annotated[int, Field(strict=True, ge=1)]
    boundary: Literal["AGENT_PRE_SPAWN", "AGENT_HANDLER_ENTRY"]
    expires_at: datetime

    @field_validator("expires_at", mode="before")
    @classmethod
    def expiry_timestamp(cls, value: Any) -> datetime:
        return aware_timestamp(value)


class OwnershipPermit(StrictModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    challenge: OwnershipChallenge
    allowed: bool
    reason: Annotated[str, Field(strict=True, pattern=r"^[A-Z][A-Z0-9_]{0,63}$")]
    checked_at: datetime
    expires_at: datetime
    signature: Digest
    # Delivery authority is node-local; it never changes the signed wire document.
    _delivery_check: Callable[[], None] | None = PrivateAttr(default=None)

    @field_validator("checked_at", "expires_at", mode="before")
    @classmethod
    def permit_timestamp(cls, value: Any) -> datetime:
        return aware_timestamp(value)

    @model_validator(mode="after")
    def decision_matches_reason(self) -> OwnershipPermit:
        if self.allowed != (self.reason == "OK"):
            raise ValueError("ownership permit decision contradicts its reason")
        return self

    def validate_delivery(self) -> None:
        if self._delivery_check is not None:
            self._delivery_check()


def command_identity(command: NodeActionCommand) -> str:
    document = command.model_dump(mode="json", exclude={"issued_at", "expires_at"})
    return hashlib.sha256(
        json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def permit_signature(permit: OwnershipPermit, secret: str) -> str:
    payload = permit.model_dump(mode="json", exclude={"signature"})
    return hmac.new(
        secret.encode(),
        OWNERSHIP_PROTOCOL.encode()
        + b"\x00"
        + json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(),
        hashlib.sha256,
    ).hexdigest()


def sign_permit(
    challenge: OwnershipChallenge,
    secret: str,
    *,
    allowed: bool,
    reason: str,
    now: datetime,
) -> OwnershipPermit:
    permit = OwnershipPermit(
        challenge=challenge,
        allowed=allowed,
        reason=reason,
        checked_at=now,
        expires_at=min(challenge.expires_at, now + timedelta(seconds=PERMIT_SECONDS)),
        signature="0" * 64,
    )
    return permit.model_copy(update={"signature": permit_signature(permit, secret)})


#: Holder attribution carried by an ``OWNERSHIP_FINAL_CLIENTS_CHANGED``
#: refusal: at most this many holders, each identity field cut to this many
#: printable characters. A ``comm`` is 15 bytes by kernel contract and a GPU
#: UUID or ``/dev/nvidia*`` path is shorter than the cap, so the cap only bites
#: on an injected finder. Nothing beyond these four fields is ever copied: a
#: command line or environment could carry a token into the journal.
FINAL_CLIENT_ATTRIBUTION_LIMIT: Final = 8
FINAL_CLIENT_FIELD_LIMIT: Final = 64
_HOLDER_FIELDS: Final = ("gpu_uuid", "pid", "device", "process_name")


def _printable(value: Any) -> str:
    text = "".join(char if char.isprintable() else "?" for char in str(value))
    return text[:FINAL_CLIENT_FIELD_LIMIT]


def device_holder_attribution(
    holders: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, str]], int]:
    """Bounded, sanitized identity of persistent device holders.

    Returns the attributed entries and how many holders were left out, so a
    truncated list is visible as such instead of reading as the whole set.
    """

    attributed = [
        {field: _printable(item.get(field, "")) for field in _HOLDER_FIELDS}
        for item in holders[:FINAL_CLIENT_ATTRIBUTION_LIMIT]
    ]
    return attributed, len(holders) - len(attributed)


def device_holder_summary(attributed: Sequence[Mapping[str, str]], omitted: int) -> str:
    """``<gpu>:<pid>:<comm>`` per holder, the shape the sampled preflight uses."""

    text = ", ".join(
        f"{item['gpu_uuid']}:{item['pid']}:{item['process_name']}"
        for item in attributed
    )
    return f"{text} (+{omitted} more)" if omitted else text


class OwnershipRefused(ValueError):
    """A safety refusal, never a failed GPU or an authorization to escalate."""

    def __init__(
        self,
        reason: str,
        *,
        boundary: str = "AGENT_PRE_SPAWN",
        cause: str | None = None,
        device_clients: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        attributed, omitted = device_holder_attribution(device_clients or ())
        # The refusal code stays the message prefix for every existing matcher;
        # the holders follow it so the one line an operator reads first (the
        # step's ``node_failures``, the journal) already names who held the
        # device. Without it a lingering workload and a platform daemon (DCGM,
        # the exporter, a health monitor) were the same opaque refusal (live
        # 2026-10-01, a freshly provisioned node).
        message = reason
        if attributed:
            message = f"{reason}: {device_holder_summary(attributed, omitted)}"
        super().__init__(message)
        self.action_details: dict[str, Any] = {
            "reason": reason,
            "safety_rejection": True,
            "manual_confirmation_required": True,
            "node_action_not_started": True,
            "ownership_check_boundary": boundary,
            "agent_queue_ownership_checked": boundary == "AGENT_PRE_SPAWN",
        }
        if cause:
            # The exception *class* of a foreign failure behind an opaque
            # safety code: enough to tell a bug from a moved fence, never the
            # message (which may carry host process names or paths).
            self.action_details["refusal_cause"] = cause
        if attributed:
            self.action_details["persistent_device_clients"] = attributed
            self.action_details["persistent_device_client_count"] = (
                len(attributed) + omitted
            )


@dataclass
class _Pending:
    challenge: OwnershipChallenge
    deadline: float = 0.0
    event: Event = field(default_factory=Event)
    permit: OwnershipPermit | None = None
    permit_deadline: float | None = None


class NodeOwnershipGate:
    def __init__(
        self,
        *,
        secret: str,
        boot_id: str,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if not boot_id or len(secret) < 32:
            raise ValueError("ownership gate requires node identity and a node key")
        self.secret = secret
        self.boot_id = boot_id
        self.now = now
        self.monotonic = monotonic
        self._lock = RLock()
        self._pending: dict[str, _Pending] = {}
        self._sequence = 0
        self._closed = False

    def challenge(self, command_id: str) -> OwnershipChallenge | None:
        with self._lock:
            pending = self._pending.get(command_id)
            return (
                pending.challenge
                if pending is not None and pending.permit is None
                else None
            )

    def authorize(self, permit: OwnershipPermit) -> None:
        now = self.now()
        received_at = self.monotonic()
        if (
            permit.checked_at.tzinfo is None
            or permit.expires_at.tzinfo is None
            or permit.challenge.expires_at.tzinfo is None
            or not permit.checked_at
            <= now
            < permit.expires_at
            <= permit.challenge.expires_at
            or (permit.expires_at - permit.checked_at).total_seconds() > PERMIT_SECONDS
            or not hmac.compare_digest(
                permit_signature(permit, self.secret), permit.signature
            )
        ):
            raise OwnershipRefused("OWNERSHIP_PERMIT_INVALID")
        with self._lock:
            pending = self._pending.get(permit.challenge.command_id)
            if self._closed or pending is None or pending.challenge != permit.challenge:
                raise OwnershipRefused("OWNERSHIP_PERMIT_STALE")
            if pending.permit is not None:
                if (
                    pending.permit.allowed == permit.allowed
                    and pending.permit.reason == permit.reason
                ):
                    return
                if not pending.permit.allowed:
                    raise OwnershipRefused("OWNERSHIP_PERMIT_STALE")
            pending.permit = permit
            pending.permit_deadline = min(
                pending.deadline,
                received_at + (permit.expires_at - now).total_seconds(),
            )
            pending.event.set()

    def require(
        self,
        command: NodeActionCommand,
        *,
        boundary: Literal["AGENT_PRE_SPAWN", "AGENT_HANDLER_ENTRY"] = "AGENT_PRE_SPAWN",
    ) -> OwnershipPermit:
        if (
            command.agent_generation is None
            or command.ownership_guard != OWNERSHIP_PROTOCOL
        ):
            raise OwnershipRefused("OWNERSHIP_PROTOCOL_REQUIRED")
        now = self.now()
        if command.expires_at.tzinfo is None or now >= command.expires_at:
            raise OwnershipRefused("OWNERSHIP_COMMAND_EXPIRED")
        with self._lock:
            if self._closed:
                raise OwnershipRefused("OWNERSHIP_CALLER_LOST")
            if command.command_id in self._pending:
                raise OwnershipRefused("OWNERSHIP_BOUNDARY_ALREADY_PENDING")
            self._sequence += 1
            challenge = OwnershipChallenge(
                command_id=command.command_id,
                workflow_id=command.workflow_request_id,
                incident_id=command.incident_id,
                node_id=command.node_id,
                boot_id=self.boot_id,
                agent_generation=command.agent_generation,
                fencing_token=command.fencing_token,
                command_sha256=command_identity(command),
                nonce=secrets.token_hex(32),
                sequence=self._sequence,
                boundary=boundary,
                expires_at=min(
                    command.expires_at, now + timedelta(seconds=CHALLENGE_SECONDS)
                ),
            )
            pending = _Pending(challenge)
            self._pending[command.command_id] = pending
            deadline = self.monotonic() + (challenge.expires_at - now).total_seconds()
            pending.deadline = deadline
        try:
            if not pending.event.wait(max(0, deadline - self.monotonic())):
                raise OwnershipRefused("OWNERSHIP_RECHECK_TIMEOUT")
            with self._lock:
                permit = pending.permit
                if (
                    permit is None
                    or self._closed
                    or pending.permit_deadline is None
                    or self.monotonic() >= pending.permit_deadline
                    or not permit.checked_at <= self.now() < permit.expires_at
                ):
                    raise OwnershipRefused("OWNERSHIP_PERMIT_EXPIRED")
                if not permit.allowed:
                    raise OwnershipRefused(permit.reason)
                self._bind_delivery(permit, pending.permit_deadline)
                self._pending.pop(command.command_id)
                return permit
        finally:
            with self._lock:
                if self._pending.get(command.command_id) is pending:
                    self._pending.pop(command.command_id)

    def _bind_delivery(self, permit: OwnershipPermit, deadline: float) -> None:
        used = False

        def check() -> None:
            nonlocal used
            with self._lock:
                if self._closed:
                    raise OwnershipRefused("OWNERSHIP_CALLER_LOST")
                if used:
                    raise OwnershipRefused("OWNERSHIP_PERMIT_USED")
                if (
                    self.monotonic() >= deadline
                    or not permit.checked_at <= self.now() < permit.expires_at
                ):
                    raise OwnershipRefused("OWNERSHIP_PERMIT_EXPIRED")
                used = True

        permit._delivery_check = check

    def cancel_all(self) -> None:
        with self._lock:
            self._closed = True
            for pending in self._pending.values():
                pending.event.set()


_physical_check: ContextVar[Callable[[], None] | None] = ContextVar(
    "gpu_fault_physical_ownership_check", default=None
)
_recheck_challenge: ContextVar[OwnershipChallenge | None] = ContextVar(
    "gpu_fault_ownership_recheck_challenge", default=None
)


@contextmanager
def physical_ownership_scope(check: Callable[[], None] | None) -> Iterator[None]:
    token = _physical_check.set(check)
    try:
        yield
    finally:
        _physical_check.reset(token)


def require_physical_ownership() -> None:
    check = _physical_check.get()
    if check is not None:
        check()


@contextmanager
def ownership_recheck_scope(challenge: OwnershipChallenge) -> Iterator[None]:
    token = _recheck_challenge.set(challenge)
    try:
        yield
    finally:
        _recheck_challenge.reset(token)


def current_ownership_challenge() -> OwnershipChallenge | None:
    return _recheck_challenge.get()


def ownership_required(operation: WorkflowOperation) -> bool:
    return (
        operation in NODE_MUTATING_OPERATIONS
        and operation is not WorkflowOperation.RESTORE_GPU_SERVICES
    )


Handler = TypeVar("Handler", bound=Callable[..., Any])


def final_ownership_boundary(handler: Handler) -> Handler:
    """Mark a handler whose first physical mutation calls the final checkpoint."""
    setattr(handler, "__gpu_fault_final_ownership__", True)
    return handler


def supports_final_ownership(handler: Callable[..., Any]) -> bool:
    return (
        getattr(
            getattr(handler, "__func__", handler),
            "__gpu_fault_final_ownership__",
            False,
        )
        is True
    )


def _validate_local_authority(agent: Any, envelope: SignedNodeAction) -> None:
    command = envelope.command
    agent.validate_submission(envelope)
    if not agent.ledger.accept_fencing(command.incident_id, command.fencing_token):
        raise OwnershipRefused("OWNERSHIP_FINAL_FENCE_CHANGED")
    if (
        command.operation in _QUIESCE_RECHECK_OPERATIONS
        and agent.service_quiesce_enabled
        and agent.quiesce_manager is not None
    ):
        agent.quiesce_manager.assert_quiesced(
            incident_id=command.incident_id, for_reset=False
        )


#: The final device-client recheck samples twice, briefly: a holder present in
#: both samples is persistent; one seen once is a transient query.
FINAL_CLIENT_SAMPLES: Final = 2
FINAL_CLIENT_SAMPLE_INTERVAL_SECONDS: Final = 0.5


def persistent_final_device_clients(
    agent: Any, targets: set[str]
) -> list[dict[str, str]]:
    """Device holders that survive every short final sample.

    The sampled preflight (``_persistent_device_clients``) already treats a
    holder seen in a single sample as transient -- a monitoring ``nvidia-smi``
    query holds every ``/dev/nvidia*`` node for a few hundred milliseconds,
    ``nvidia-persistenced`` respawns on a timer. The final recheck used a
    single scan, so any such query at that instant aborted the reset with an
    opaque refusal and quarantined a healthy node (live 2026-09-17). The same
    rule applies here, bounded far inside the permit window: an empty sample
    ends the check early, a holder present in every sample still refuses.
    """
    sleep = getattr(agent, "sleep", None) or time.sleep
    shared: set[tuple[str, str, str]] | None = None
    last: dict[tuple[str, str, str], dict[str, str]] = {}
    for index in range(FINAL_CLIENT_SAMPLES):
        if index:
            sleep(FINAL_CLIENT_SAMPLE_INTERVAL_SECONDS)
        last = {
            (
                str(item.get("gpu_uuid", "")),
                str(item.get("pid", "")),
                str(item.get("device", "")),
            ): item
            for item in agent.device_client_finder(targets)
        }
        shared = set(last) if shared is None else shared & set(last)
        if not shared:
            return []
    return [last[key] for key in sorted(shared or ())]


def execute_with_final_ownership(
    agent: Any,
    envelope: SignedNodeAction,
    handler: Callable[[NodeActionCommand], dict[str, Any]],
) -> dict[str, Any]:
    command = envelope.command
    gate: NodeOwnershipGate | None = agent.ownership_gate
    guarded = gate is not None and ownership_required(command.operation)
    if guarded and not supports_final_ownership(handler):
        raise OwnershipRefused(
            "OWNERSHIP_BOUNDARY_UNSUPPORTED", boundary="AGENT_ADMISSION"
        )
    receipts: list[dict[str, Any]] = []

    def physical_check() -> None:
        assert gate is not None
        permit = gate.require(command)
        try:
            _validate_local_authority(agent, envelope)
            if command.operation in _CLIENT_RECHECK_OPERATIONS:
                # Retain the handler's full sampled preflight, then reject any
                # new client seen after the potentially long ownership wait.
                agent._verify_no_clients(
                    command.gpu_uuids, include_device_clients=False
                )
                if command.operation is not WorkflowOperation.RESTART_FABRIC_MANAGER:
                    with agent._device_path_cache_window():
                        agent._require_resolvable_targets(command.gpu_uuids)
                        holders = persistent_final_device_clients(
                            agent, set(command.gpu_uuids)
                        )
                        if holders:
                            refusal = OwnershipRefused(
                                "OWNERSHIP_FINAL_CLIENTS_CHANGED",
                                device_clients=holders,
                            )
                            # The executor's failure line only carries the
                            # error class; this is the journal's one record of
                            # who held the device when the reset was refused.
                            LOGGER.warning(
                                "final ownership recheck refused command_id=%s "
                                "operation=%s node_id=%s "
                                "persistent_device_clients=%s",
                                command.command_id,
                                command.operation.value,
                                command.node_id,
                                " ".join(
                                    f"{item['gpu_uuid']}:{item['pid']}:"
                                    f"{item['device']}:{item['process_name']}"
                                    for item in refusal.action_details[
                                        "persistent_device_clients"
                                    ]
                                ),
                            )
                            raise refusal
            _validate_local_authority(agent, envelope)
        except OwnershipRefused:
            # A fence that moved names itself (fence, clients); it is not a
            # foreign failure of the local reads.
            raise
        except Exception as exc:
            LOGGER.warning(
                "final ownership safety check failed before the physical call: %s",
                type(exc).__name__,
            )
            raise OwnershipRefused(
                "OWNERSHIP_FINAL_SAFETY_CHANGED", cause=type(exc).__name__
            ) from None
        if not permit.checked_at <= agent.now() < permit.expires_at:
            raise OwnershipRefused("OWNERSHIP_PERMIT_EXPIRED")
        permit.validate_delivery()
        receipts.append(
            {
                "boundary": permit.challenge.boundary,
                "challenge_nonce": permit.challenge.nonce,
                "sequence": permit.challenge.sequence,
                "checked_at": permit.checked_at.isoformat(),
                "expires_at": permit.expires_at.isoformat(),
            }
        )

    try:
        with physical_ownership_scope(physical_check if guarded else None):
            details = handler(command)
    except Exception as exc:
        action_details = getattr(exc, "action_details", None)
        if (
            guarded
            and isinstance(action_details, dict)
            and action_details.get("safety_rejection") is True
        ):
            action_details["physical_ownership_checks"] = [
                dict(receipt) for receipt in receipts
            ]
            action_details["node_action_command_id"] = command.command_id
            if receipts:
                # A granted checkpoint is not completion proof, but rules out
                # claiming that the entire command was denied before dispatch.
                action_details.pop("node_action_not_started", None)
                action_details["outcome_unknown"] = True
                action_details["manual_confirmation_required"] = True
        raise
    if receipts:
        details = {**details, "physical_ownership_checks": receipts}
    if gate is not None:
        details = {**details, "node_action_command_id": command.command_id}
    return details
