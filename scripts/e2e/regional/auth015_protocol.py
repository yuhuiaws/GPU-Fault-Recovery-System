"""Non-dispatching challenges against an already bound Node Agent endpoint."""

from __future__ import annotations

import json
import secrets
import ssl
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request

from gpu_fault.fleet import AgentRecord
from gpu_fault.models import WorkflowOperation
from gpu_fault.node_agent.protocol import (
    NodeActionCommand,
    SignedNodeAction,
    sign_node_action,
    sign_result_query,
)
from scripts.e2e.regional.auth015_http import HttpDeadlineExceeded, bounded_request

MAX_RESPONSE_BYTES = 8192
REQUEST_TIMEOUT_SECONDS = 5
PROOF_TIMEOUT_SECONDS = 30
CHALLENGE_OPERATION = WorkflowOperation.FREEZE_EVIDENCE


class Auth015ProofError(RuntimeError):
    pass


@dataclass(frozen=True, repr=False)
class SignatureTarget:
    agent: AgentRecord
    key: str


def rejected_command(
    agent: AgentRecord, command_id: str, now: datetime
) -> NodeActionCommand:
    # For every server clock, issued_at is either > now+30s or expires_at <= now.
    # This remains non-dispatching if the operation gains a Node Agent handler.
    return NodeActionCommand(
        command_id=command_id,
        workflow_request_id=command_id,
        incident_id=command_id,
        fencing_token=1,
        operation=CHALLENGE_OPERATION,
        node_id=agent.node_id,
        agent_generation=agent.generation,
        issued_at=now,
        expires_at=now - timedelta(seconds=31),
    )


def rejection_body(code: str, message: str) -> dict[str, Any]:
    return {
        "detail": {
            "code": code,
            "message": message,
            "retryable": False,
            "requires_new_command": False,
        }
    }


def _response(request: Request, certificate: str, deadline: float) -> tuple[int, Any]:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise Auth015ProofError("signature challenge deadline expired")
    body = request.data
    if body is not None and not isinstance(body, bytes):
        raise Auth015ProofError("signature challenge requires a bounded byte body")
    try:
        status, payload = bounded_request(
            request.full_url,
            method=request.get_method(),
            headers=dict(request.header_items()),
            data=body,
            certificate=certificate,
            timeout=min(REQUEST_TIMEOUT_SECONDS, remaining),
            max_response_bytes=MAX_RESPONSE_BYTES,
        )
        if len(payload) > MAX_RESPONSE_BYTES:
            raise Auth015ProofError("signature challenge response exceeds limit")
        if time.monotonic() > deadline:
            raise Auth015ProofError("signature challenge deadline expired")
        return status, json.loads(payload)
    except Auth015ProofError:
        raise
    except HttpDeadlineExceeded:
        raise Auth015ProofError("signature challenge deadline expired") from None
    except Exception:
        # Transport errors can contain signed query URLs or response bodies.
        raise Auth015ProofError(
            "signature challenge transport or JSON failed"
        ) from None


def prove_signatures(
    source: SignatureTarget,
    destination: SignatureTarget,
    *,
    retired_key: str | None = None,
) -> dict[str, Any]:
    agents = (source.agent, destination.agent)
    if (
        agents[0].cluster_id != agents[1].cluster_id
        or agents[0].node_id == agents[1].node_id
        or source.key == destination.key
        or retired_key is not None
        and (
            len(retired_key) < 32
            or retired_key.strip() != retired_key
            or retired_key in {source.key, destination.key}
        )
        or any(
            len(target.key) < 32
            or target.key != target.key.strip()
            or not target.agent.endpoint.startswith("https://")
            or not target.agent.tls_certificate_pem
            or target.agent.node_action_key_version != 2
            or CHALLENGE_OPERATION in target.agent.allowed_operations
            for target in (source, destination)
        )
    ):
        raise Auth015ProofError("signature challenge targets or key scope are invalid")
    try:
        for agent in agents:
            ssl.create_default_context(cadata=agent.tls_certificate_pem)
    except (ValueError, ssl.SSLError):
        raise Auth015ProofError(
            "signature challenge TLS certificate is invalid"
        ) from None
    now = datetime.now(timezone.utc)
    command_id = "auth015-" + secrets.token_hex(24)
    command = rejected_command(destination.agent, command_id, now)
    deadline = time.monotonic() + PROOF_TIMEOUT_SECONDS
    observations: list[dict[str, Any]] = []

    def query(target: SignatureTarget, key: str) -> Request:
        issued_at = datetime.now(timezone.utc).isoformat()
        query_string = urlencode(
            {
                "command_id": command_id,
                "issued_at": issued_at,
                "signature": sign_result_query(command_id, issued_at, key),
            }
        )
        return Request(
            target.agent.endpoint + "/v1/node-actions/result?" + query_string
        )

    def submit(key: str) -> Request:
        envelope = SignedNodeAction(
            command=command, signature=sign_node_action(command, key)
        )
        return Request(
            destination.agent.endpoint + "/v1/node-actions/submit",
            data=envelope.model_dump_json().encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

    def expect(
        name: str,
        request: Request,
        agent: AgentRecord,
        status: int,
        body: dict[str, Any],
    ) -> None:
        observed_status, observed_body = _response(
            request, str(agent.tls_certificate_pem), deadline
        )
        matched = observed_status == status and observed_body == body
        if isinstance(observed_body, dict) and isinstance(
            observed_body.get("detail"), dict
        ):
            detail = observed_body["detail"]
            matched = (
                matched
                and detail.get("retryable") is False
                and detail.get("requires_new_command") is False
            )
        observations.append(
            {"check": name, "status": observed_status, "matched": matched}
        )
        if not matched:
            raise Auth015ProofError(
                f"signature challenge {name} returned an unexpected response"
            )

    unknown = {"detail": "node action command is unknown"}
    expect("source_key_active", query(source, source.key), source.agent, 404, unknown)
    expect(
        "destination_result_absent_before",
        query(destination, destination.key),
        destination.agent,
        404,
        unknown,
    )
    expect(
        "destination_key_active_without_dispatch",
        submit(destination.key),
        destination.agent,
        403,
        rejection_body(
            "OPERATION_NOT_ALLOWED",
            f"operation {CHALLENGE_OPERATION.value} is not allowed",
        ),
    )
    expect(
        "sibling_command_signature_rejected",
        submit(source.key),
        destination.agent,
        401,
        rejection_body("INVALID_SIGNATURE", "invalid node action signature"),
    )
    expect(
        "sibling_result_signature_rejected",
        query(destination, source.key),
        destination.agent,
        401,
        rejection_body(
            "INVALID_SIGNATURE", "invalid node action result query signature"
        ),
    )
    if retired_key is not None:
        expect(
            "retired_command_signature_rejected",
            submit(retired_key),
            destination.agent,
            401,
            rejection_body("INVALID_SIGNATURE", "invalid node action signature"),
        )
        expect(
            "retired_result_signature_rejected",
            query(destination, retired_key),
            destination.agent,
            401,
            rejection_body(
                "INVALID_SIGNATURE", "invalid node action result query signature"
            ),
        )
    expect(
        "destination_result_absent_after",
        query(destination, destination.key),
        destination.agent,
        404,
        unknown,
    )
    return {
        "command_id": command_id,
        "issued_at": now.isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "observations": observations,
        "operation": command.operation.value,
        "non_admissible_time_window": True,
        "supplied_retired_key_denied": retired_key is not None,
        "rotated_key_activation_proved": False,
    }
