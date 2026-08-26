from __future__ import annotations

import re
from typing import Any

from pydantic import Field, field_validator

from gpu_fault.models import StrictModel, WorkflowOperation


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
CURRENT_AGENT_PROTOCOL_VERSION = 3
NODE_ACTION_KEY_VERSION_SHARED = 1
NODE_ACTION_KEY_VERSION_DERIVED = 2


class FleetCompatibilityPolicy(StrictModel):
    required_agent_protocol_version: int = Field(
        default=CURRENT_AGENT_PROTOCOL_VERSION,
        ge=1,
    )
    compatible_agent_protocol_versions: frozenset[int] = Field(
        default_factory=frozenset
    )
    required_node_action_key_version: int | None = Field(
        default=None,
        ge=NODE_ACTION_KEY_VERSION_SHARED,
        le=NODE_ACTION_KEY_VERSION_DERIVED,
    )
    required_agent_version: str | None = None
    required_artifact_sha256: str | None = None
    compatible_artifact_sha256s: frozenset[str] = Field(default_factory=frozenset)
    required_compatibility_digest: str | None = None
    compatible_compatibility_digests: frozenset[str] = Field(default_factory=frozenset)
    required_policy_version: str | None = None
    required_runtime_profile_version: str | None = None
    required_config_digest: str | None = None
    compatible_config_digests: frozenset[str] = Field(default_factory=frozenset)
    required_operations: list[WorkflowOperation] = Field(
        default_factory=lambda: [
            WorkflowOperation.VERIFY_NO_GPU_CLIENTS,
            WorkflowOperation.RESET_GPU,
        ]
    )
    max_heartbeat_age_seconds: int = Field(default=90, ge=10, le=3600)

    @field_validator(
        "required_artifact_sha256",
        "required_compatibility_digest",
        "required_config_digest",
    )
    @classmethod
    def validate_optional_digest(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.lower()
        if not SHA256_PATTERN.fullmatch(normalized):
            raise ValueError("digest must be a SHA-256 hex value")
        return normalized

    @field_validator("compatible_agent_protocol_versions")
    @classmethod
    def validate_compatible_protocols(cls, values: frozenset[int]) -> frozenset[int]:
        if any(value < 1 for value in values):
            raise ValueError("compatible agent protocol versions must be positive")
        return values

    @field_validator(
        "compatible_artifact_sha256s",
        "compatible_compatibility_digests",
        "compatible_config_digests",
    )
    @classmethod
    def validate_compatible_artifacts(cls, values: frozenset[str]) -> frozenset[str]:
        normalized = frozenset(value.lower() for value in values)
        if any(not SHA256_PATTERN.fullmatch(value) for value in normalized):
            raise ValueError("compatible digests must be SHA-256 hex values")
        return normalized


def rollout_compatibility_reasons(
    policy: FleetCompatibilityPolicy,
    record: Any,
) -> list[str]:
    reasons = []
    accepted_protocols = {
        policy.required_agent_protocol_version,
        *policy.compatible_agent_protocol_versions,
    }
    if record.agent_protocol_version not in accepted_protocols:
        expected = (
            str(policy.required_agent_protocol_version)
            if len(accepted_protocols) == 1
            else "one of "
            + ", ".join(str(value) for value in sorted(accepted_protocols))
        )
        reasons.append(
            "agent protocol version mismatch: "
            f"expected {expected}, got {record.agent_protocol_version}"
        )
    accepted_artifacts = {
        value
        for value in {
            policy.required_artifact_sha256,
            *policy.compatible_artifact_sha256s,
        }
        if value is not None
    }
    if accepted_artifacts and record.artifact_sha256 not in accepted_artifacts:
        expected = (
            next(iter(accepted_artifacts))
            if len(accepted_artifacts) == 1
            else "one of " + ", ".join(sorted(accepted_artifacts))
        )
        reasons.append(
            "artifact SHA-256 mismatch: "
            f"expected {expected}, got {record.artifact_sha256}"
        )
    accepted_compatibility = {
        value
        for value in {
            policy.required_compatibility_digest,
            *policy.compatible_compatibility_digests,
        }
        if value is not None
    }
    actual_compatibility = record.compatibility_digest or record.artifact_sha256
    if accepted_compatibility and actual_compatibility not in accepted_compatibility:
        expected = (
            next(iter(accepted_compatibility))
            if len(accepted_compatibility) == 1
            else "one of " + ", ".join(sorted(accepted_compatibility))
        )
        reasons.append(
            "compatibility digest mismatch: "
            f"expected {expected}, got {actual_compatibility}"
        )
    accepted_configs = {
        value
        for value in {
            policy.required_config_digest,
            *policy.compatible_config_digests,
        }
        if value is not None
    }
    if accepted_configs and record.config_digest not in accepted_configs:
        expected = (
            next(iter(accepted_configs))
            if len(accepted_configs) == 1
            else "one of " + ", ".join(sorted(accepted_configs))
        )
        reasons.append(
            f"config digest mismatch: expected {expected}, got {record.config_digest}"
        )
    return reasons


def pin_value_is_accepted(
    policy: FleetCompatibilityPolicy,
    attribute: str,
    actual: str,
    required: str,
) -> bool:
    if attribute == "artifact_sha256":
        return actual in {
            required,
            *policy.compatible_artifact_sha256s,
        }
    if attribute == "config_digest":
        return actual in {
            required,
            *policy.compatible_config_digests,
        }
    if attribute == "compatibility_digest":
        return actual in {
            required,
            *policy.compatible_compatibility_digests,
        }
    return actual == required
