from __future__ import annotations

import hashlib
import hmac
import json
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import Field

from gpu_fault.models import StrictModel, WorkflowOperation

RESULT_QUERY_MAX_SKEW_SECONDS = 300


def canonical_command(command: NodeActionCommand) -> bytes:
    return json.dumps(
        command.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def sign_node_action(command: NodeActionCommand, secret: str) -> str:
    return hmac.new(
        secret.encode(), canonical_command(command), hashlib.sha256
    ).hexdigest()


def canonical_result_query(command_id: str, issued_at: str) -> bytes:
    return json.dumps(
        {"command_id": command_id, "issued_at": issued_at},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def sign_result_query(command_id: str, issued_at: str, secret: str) -> str:
    return hmac.new(
        secret.encode(),
        canonical_result_query(command_id, issued_at),
        hashlib.sha256,
    ).hexdigest()


def verify_result_query(
    command_id: str,
    issued_at: str | None,
    signature: str | None,
    secret: str,
    *,
    now: datetime | None = None,
    max_skew_seconds: int = RESULT_QUERY_MAX_SKEW_SECONDS,
) -> None:
    """Authenticate a node action result query.

    ``GET /v1/node-actions/result`` used to take only ``command_id``, so
    anything that could reach the agent's port could read every recovery
    action's result: node IDs, GPU UUIDs, held-open device clients, the
    diagnostics that ran and their output paths. The query is now signed
    with the same shared secret as the command itself.

    Raises ``ValueError`` with a message the HTTP layer maps to 401.
    """
    if not issued_at or not signature:
        raise ValueError("node action result query requires issued_at and signature")
    try:
        stamp = datetime.fromisoformat(issued_at)
    except ValueError as exc:
        raise ValueError("node action result query issued_at is not ISO-8601") from exc
    if stamp.tzinfo is None:
        raise ValueError("node action result query issued_at must be timezone aware")
    reference = now or datetime.now(timezone.utc)
    if abs((reference - stamp).total_seconds()) > max_skew_seconds:
        raise ValueError(
            "node action result query issued_at is outside the "
            f"{max_skew_seconds}s window"
        )
    expected = sign_result_query(command_id, issued_at, secret)
    if not hmac.compare_digest(expected, signature):
        raise ValueError("invalid node action result query signature")


class NodeActionStatus(StrEnum):
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    INTERRUPTED = "INTERRUPTED"


class NodeActionExecutionState(StrEnum):
    PENDING = "PENDING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    INTERRUPTED = "INTERRUPTED"


class UnsupportedOperationError(RuntimeError):
    """Raised when an allow-listed action has no executor branch."""


class NodeActionCommand(StrictModel):
    command_id: str
    workflow_request_id: str
    incident_id: str
    fencing_token: int = Field(ge=1)
    operation: WorkflowOperation
    node_id: str
    agent_generation: int | None = Field(default=None, ge=1)
    gpu_uuids: list[str] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)
    issued_at: datetime
    expires_at: datetime


class SignedNodeAction(StrictModel):
    command: NodeActionCommand
    signature: str


class NodeActionResult(StrictModel):
    command_id: str
    operation: WorkflowOperation
    status: NodeActionStatus
    details: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    retryable: bool = False
    attempt: int = Field(default=1, ge=1)
    completed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class NodeActionSubmission(StrictModel):
    command_id: str
    state: NodeActionExecutionState
    result: NodeActionResult | None = None
