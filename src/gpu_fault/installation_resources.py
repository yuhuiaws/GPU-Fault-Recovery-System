from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Self

from pydantic import Field, field_validator, model_validator

from gpu_fault.models import StrictModel


RESOURCE_KEY_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,511}$")
RESOURCE_ID_PATTERN = re.compile(r"^[^\s\x00-\x1f\x7f]{1,2048}$")
SENSITIVE_KEY_PATTERN = re.compile(
    r"secret|token|password|private.?key|credential",
    re.IGNORECASE,
)


class InstallationResourceOwnership(StrEnum):
    CREATED = "CREATED"
    REUSED = "REUSED"
    EXTERNAL = "EXTERNAL"


class InstallationResourceDeletePolicy(StrEnum):
    DELETE = "DELETE"
    DETACH = "DETACH"
    PRESERVE = "PRESERVE"


class InstallationResourceStatus(StrEnum):
    PLANNED = "PLANNED"
    CREATING = "CREATING"
    ACTIVE = "ACTIVE"
    DELETE_PENDING = "DELETE_PENDING"
    DELETED = "DELETED"
    DETACHED = "DETACHED"
    PRESERVED = "PRESERVED"
    FAILED = "FAILED"


TERMINAL_INSTALLATION_RESOURCE_STATUSES = frozenset(
    {
        InstallationResourceStatus.DELETED,
        InstallationResourceStatus.DETACHED,
        InstallationResourceStatus.PRESERVED,
    }
)


class InstallationResource(StrictModel):
    site_id: str
    resource_key: str
    provider: str = "aws"
    resource_type: str
    resource_id: str
    resource_arn: str | None = None
    region: str | None = None
    account_id: str | None = None
    ownership: InstallationResourceOwnership
    delete_policy: InstallationResourceDeletePolicy
    status: InstallationResourceStatus = InstallationResourceStatus.ACTIVE
    dependencies: list[str] = Field(default_factory=list)
    attributes: dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    error: str | None = None

    @field_validator(  # type: ignore[untyped-decorator]
        "site_id",
        "resource_key",
        "provider",
        "resource_type",
    )
    @classmethod
    def validate_identifier(cls, value: str) -> str:
        normalized = value.strip()
        if not RESOURCE_KEY_PATTERN.fullmatch(normalized):
            raise ValueError("installation resource identifier is invalid")
        return normalized

    @field_validator("resource_id")  # type: ignore[untyped-decorator]
    @classmethod
    def validate_resource_id(cls, value: str) -> str:
        normalized = value.strip()
        if not RESOURCE_ID_PATTERN.fullmatch(normalized):
            raise ValueError("installation resource ID is invalid")
        return normalized

    @field_validator("dependencies")  # type: ignore[untyped-decorator]
    @classmethod
    def normalize_dependencies(cls, value: list[str]) -> list[str]:
        normalized = sorted({item.strip() for item in value if item.strip()})
        if any(not RESOURCE_KEY_PATTERN.fullmatch(item) for item in normalized):
            raise ValueError("installation resource dependency is invalid")
        return normalized

    @field_validator("attributes")  # type: ignore[untyped-decorator]
    @classmethod
    def reject_sensitive_attributes(cls, value: dict[str, str]) -> dict[str, str]:
        sensitive = sorted(key for key in value if SENSITIVE_KEY_PATTERN.search(key))
        if sensitive:
            raise ValueError(
                "installation resource attributes may not contain sensitive keys: "
                + ", ".join(sensitive)
            )
        return {str(key): str(item) for key, item in value.items()}

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_policy(self) -> Self:
        if (
            self.delete_policy is InstallationResourceDeletePolicy.PRESERVE
            and self.ownership is InstallationResourceOwnership.CREATED
            and self.resource_type not in {"cpu_eks", "cpu_hyperpod", "rds_snapshot"}
        ):
            raise ValueError("created resources require DELETE or DETACH policy")
        return self

    def immutable_identity(self) -> tuple[Any, ...]:
        return (
            self.site_id,
            self.resource_key,
            self.provider,
            self.resource_type,
            self.resource_id,
            self.resource_arn,
            self.region,
            self.account_id,
            self.ownership,
            self.delete_policy,
            tuple(self.dependencies),
        )


class InstallationResourceSnapshot(StrictModel):
    site_id: str
    resources: list[InstallationResource]
    source_sha256: str | None = None

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_resources(self) -> Self:
        keys = [item.resource_key for item in self.resources]
        if len(keys) != len(set(keys)):
            raise ValueError("installation resource keys must be unique")
        if any(item.site_id != self.site_id for item in self.resources):
            raise ValueError("installation resource site_id mismatch")
        if self.source_sha256 is not None and self.source_sha256 != self.digest():
            raise ValueError("installation resource snapshot digest mismatch")
        return self

    def digest(self) -> str:
        payload = {
            "site_id": self.site_id,
            "resources": [
                item.model_dump(mode="json")
                for item in sorted(self.resources, key=lambda value: value.resource_key)
            ],
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def require_source_binding(self) -> Self:
        """Reject a snapshot that is not bound to its own content digest.

        ``source_sha256`` is the provenance seal a persisted snapshot carries.
        It is optional so a freshly built snapshot can be minted and then sealed
        (the builders ``model_copy`` the digest in, which bypasses validation),
        but any record read back from disk and trusted as installed state must
        carry a seal that matches its content. A ``None`` seal (an unsealed or
        stripped snapshot) or a mismatched one has no verifiable origin, so it is
        refused here rather than trusted as authoritative.
        """
        if self.source_sha256 is None:
            raise ValueError("installation resource snapshot is not sealed")
        if self.source_sha256 != self.digest():
            raise ValueError("installation resource snapshot digest mismatch")
        return self
