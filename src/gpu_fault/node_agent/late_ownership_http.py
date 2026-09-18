"""Ownership extensions for the existing authenticated Agent HTTP contract."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from gpu_fault.node_agent.late_ownership import (
    NodeOwnershipGate,
    OwnershipChallenge,
    OwnershipRefused,
    command_identity,
)
from gpu_fault.node_agent.protocol import (
    NodeActionCommand,
    NodeActionStatus,
    NodeActionSubmission,
    SignedNodeAction,
)


def awaiting_action_retry(state: NodeActionSubmission) -> bool:
    result = state.result
    return (
        result is not None
        and result.status is NodeActionStatus.FAILED
        and result.retryable
    )


def authorize_pending_ownership(
    gate: NodeOwnershipGate | None,
    envelope: SignedNodeAction,
    command: NodeActionCommand,
    submission_state: Callable[[str], NodeActionSubmission | None],
) -> NodeActionSubmission | None:
    permit = envelope.ownership_permit
    if permit is None:
        return None
    if gate is None:
        raise OwnershipRefused("OWNERSHIP_PROTOCOL_UNAVAILABLE")
    if (
        permit.challenge.command_id != command.command_id
        or permit.challenge.command_sha256 != command_identity(command)
    ):
        raise OwnershipRefused("OWNERSHIP_PERMIT_INVALID")
    gate.authorize(permit)
    state = submission_state(command.command_id)
    if state is None:
        raise OwnershipRefused("OWNERSHIP_PERMIT_STALE")
    return state


def queued_ownership_failure(
    command: NodeActionCommand, error: Exception
) -> dict[str, Any]:
    if isinstance(error, OwnershipRefused):
        return error.action_details
    if command.ownership_guard is None:
        return {}
    return OwnershipRefused(
        "OWNERSHIP_QUEUE_ADMISSION_REJECTED", boundary="AGENT_ADMISSION"
    ).action_details


def pending_ownership_challenge(
    gate: NodeOwnershipGate | None, command_id: str
) -> OwnershipChallenge | None:
    return gate.challenge(command_id) if gate is not None else None


def result_query_refusal(
    verify: Callable[..., None],
    command_id: str,
    issued_at: str | None,
    signature: str | None,
    secret: str,
    *,
    required: bool,
    max_skew_seconds: int,
) -> dict[str, Any] | None:
    if not required and signature is None:
        return None
    try:
        verify(
            command_id, issued_at, signature, secret, max_skew_seconds=max_skew_seconds
        )
    except ValueError as exc:
        return {
            "code": "INVALID_SIGNATURE",
            "message": str(exc),
            "retryable": False,
            "requires_new_command": False,
        }
    return None
