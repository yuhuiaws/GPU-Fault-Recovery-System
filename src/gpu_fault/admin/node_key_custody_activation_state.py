"""Shared durable activation schema for the coordinator and real key writer."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from pydantic import Field

from gpu_fault.admin.node_key_custody_crypto import parse
from gpu_fault.admin.node_key_custody_models import (
    Authorization,
    CustodyError,
    Signed,
    Statement,
    statement_sha256,
)

ACTIVATION_TIMEOUT_SECONDS = 1800
ACTIVATION_STEPS = (
    "GUARDED",
    "FENCED",
    "KEYS_PROVISIONED",
    "EXECUTOR_REFRESHED",
    "NODE_INSTALLED",
    "CPU_REFRESHED",
    "OBSERVED",
    "UNFENCED",
)


class ActivationState(Statement):
    schema_version: int = 1
    authorization_sha256: str
    binding_sha256: str
    transaction_id: str
    started_at: datetime
    deadline: datetime
    snapshot: dict[str, Any]
    started: list[str] = Field(default_factory=list)
    completed: dict[str, dict[str, Any]] = Field(default_factory=dict)


def validate_activation_state(
    raw: bytes, authorization: Signed[Authorization]
) -> ActivationState:
    state = parse(ActivationState, raw)
    approved = authorization.statement
    completed = set(state.completed)
    count = len(completed)
    if (
        state.schema_version != 1
        or state.authorization_sha256 != statement_sha256(authorization)
        or state.binding_sha256 != statement_sha256(approved.binding)
        or state.transaction_id != approved.transaction_id
        or state.started_at.tzinfo is None
        or state.deadline.tzinfo is None
        or not approved.not_before <= state.started_at < approved.expires_at
        or state.deadline
        != min(
            state.started_at + timedelta(seconds=ACTIVATION_TIMEOUT_SECONDS),
            approved.expires_at,
        )
        or completed != set(ACTIVATION_STEPS[:count])
        or len(state.started) != len(set(state.started))
        or not completed <= set(state.started) <= set(ACTIVATION_STEPS[: count + 1])
    ):
        raise CustodyError("node key activation journal is inconsistent or foreign")
    try:
        times = [
            datetime.fromisoformat(state.completed[name]["completed_at"])
            for name in ACTIVATION_STEPS[:count]
        ]
        if any(
            stamp.tzinfo is None or not state.started_at <= stamp < state.deadline
            for stamp in times
        ) or times != sorted(times):
            raise ValueError
    except (ValueError, TypeError, KeyError):
        raise CustodyError(
            "node key activation completion timing is unproven"
        ) from None
    return state


def owned_wave_data(
    wave: dict[str, Any], authorization: Authorization
) -> dict[str, str]:
    return {
        **wave["data"],
        "allowed-nodes": str(authorization.rotate_node),
        "max-unavailable": "1",
        "generation": "node-key-" + authorization.transaction_id,
    }


def require_owned_wave(
    document: dict[str, Any], wave: dict[str, Any], authorization: Authorization
) -> None:
    metadata = document.get("metadata") or {}
    if (
        document.get("kind") != "ConfigMap"
        or document.get("apiVersion") != "v1"
        or metadata.get("name") != wave["name"]
        or metadata.get("namespace") != authorization.binding.site.namespace
        or metadata.get("uid") != wave["uid"]
        or not metadata.get("resourceVersion")
        or metadata.get("deletionTimestamp")
        or document.get("data") != owned_wave_data(wave, authorization)
    ):
        raise CustodyError("node key activation lost its owned installer wave")
