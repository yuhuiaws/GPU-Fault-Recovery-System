"""Strict, non-secret statements for prospective node-key custody evidence."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Annotated, Any, Generic, Literal, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    StringConstraints,
    ValidationInfo,
    field_validator,
    model_serializer,
)

Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Identity = Annotated[
    str, StringConstraints(min_length=1, max_length=512, pattern=r"^[!-~]+$")
]
NodeName = Annotated[
    str, StringConstraints(pattern=r"^[a-z0-9](?:[a-z0-9.-]{0,251}[a-z0-9])?$")
]
Generation = Annotated[int, Field(ge=1)]


class CustodyError(RuntimeError):
    """A custody failure with no credential-bearing diagnostics."""


class Statement(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    @field_validator(
        "schema_version",
        "master_source_verified",
        "runtime_activation_proved",
        "sibling_key_denied",
        "non_dispatching",
        mode="before",
        check_fields=False,
    )
    @classmethod
    def strict_literal_types(cls, value: Any, info: ValidationInfo) -> Any:
        expected = int if info.field_name == "schema_version" else bool
        if type(value) is not expected:
            raise ValueError("custody literal field has an invalid type")
        return value


class ReleaseBinding(Statement):
    release_id: Identity
    manifest_sha256: Digest
    delivery_sha256: Digest
    node_wheel_sha256: Digest
    node_digest: Digest
    bundle_sha256: Digest
    template_sha256: Digest
    config_digest: Digest
    runtime_profile_version: Identity


class SiteBinding(Statement):
    site_name: Identity
    region: Identity
    cpu_eks_arn: Identity
    gpu_eks_arn: Identity
    cluster_id: Identity
    hyperpod_cluster: Identity
    namespace: NodeName
    cpu_cluster_uid: Identity
    gpu_cluster_uid: Identity
    cpu_namespace_uid: Identity
    gpu_namespace_uid: Identity


class MasterSource(Statement):
    secret_name: NodeName
    secret_uid: Identity
    data_key: Identity
    sha256: Digest


class CustodyBinding(Statement):
    release: ReleaseBinding
    site: SiteBinding
    nodes: Annotated[dict[NodeName, Identity], Field(min_length=1, max_length=1024)]
    master: MasterSource


class KeyState(Statement):
    node_uid: Identity
    generation: Generation
    sha256: Digest


class KeyMap(Statement):
    uid: Identity
    resource_version: Identity
    keys: Annotated[dict[NodeName, KeyState], Field(min_length=1, max_length=1024)]


class Authorization(Statement):
    schema_version: Literal[1] = 1
    kind: Literal["node-key-custody-authorization"] = "node-key-custody-authorization"
    transaction_id: Digest
    binding: CustodyBinding
    purpose: Literal["install", "rotate"]
    rotate_node: NodeName | None
    previous_receipt_sha256: Digest | None
    producer_sha256: Digest
    witness_sha256: Digest
    not_before: datetime
    expires_at: datetime


class Started(Statement):
    schema_version: Literal[1] = 1
    kind: Literal["node-key-custody-started"] = "node-key-custody-started"
    authorization_sha256: Digest
    observed_at: datetime
    before_gpu: KeyMap | None
    before_cpu_uid: Identity | None
    cpu_other_keys_sha256: Digest
    planned_keys: dict[NodeName, KeyState]
    #: Live GPU keys of nodes outside the signed predecessor, each proved at
    #: start to be ``derive_node_action_secret`` of its node under the bound
    #: master -- what the executor's node-key sync adds for a node that joined
    #: between deploys: node -> SHA-256 of the key bytes, never the key.
    derived_extra_keys: dict[NodeName, Digest] = Field(default_factory=dict)
    master_source_verified: Literal[True] = True
    custody: Literal["cpu-secret-and-trusted-host-file"] = (
        "cpu-secret-and-trusted-host-file"
    )

    @model_serializer(mode="wrap")
    def _omit_absent_derived_extra_keys(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, Any]:
        # Left out while empty, so a start signed before the field existed
        # canonicalizes -- and verifies -- to exactly the bytes it was signed as.
        data: dict[str, Any] = handler(self)
        if not self.derived_extra_keys:
            data.pop("derived_extra_keys", None)
        return data


class Completed(Statement):
    schema_version: Literal[1] = 1
    kind: Literal["node-key-custody-provisioned"] = "node-key-custody-provisioned"
    started_sha256: Digest
    observed_at: datetime
    gpu: KeyMap
    cpu: KeyMap
    cpu_other_keys_sha256: Digest
    transport: Literal["private-stdio-node-keys-only"] = "private-stdio-node-keys-only"
    runtime_activation_proved: Literal[False] = False


class RuntimeNode(Statement):
    node_uid: Identity
    boot_id: Identity
    agent_incarnation_id: Identity
    agent_generation: Generation
    endpoint: Identity
    certificate_sha256: Digest


class Activated(Statement):
    schema_version: Literal[1] = 1
    kind: Literal["node-key-custody-activated"] = "node-key-custody-activated"
    completed_sha256: Digest
    binding_sha256: Digest
    observed_at: datetime
    nodes: Annotated[dict[NodeName, RuntimeNode], Field(min_length=2, max_length=2)]
    runtime_identity_sha256: Digest
    protocol_sha256: Digest
    retired_key_denied: bool
    sibling_key_denied: Literal[True] = True
    non_dispatching: Literal[True] = True


_StatementT = TypeVar("_StatementT", bound=Statement)


class Signed(Statement, Generic[_StatementT]):
    statement: _StatementT
    signer_sha256: Digest
    signature: Annotated[str, StringConstraints(min_length=64, max_length=512)]


class Transaction(Statement):
    authorization: Signed[Authorization]
    started: Signed[Started]
    completed: Signed[Completed]
    activated: Signed[Activated] | None = None


class Chain(Statement):
    schema_version: Literal[1] = 1
    transactions: Annotated[list[Transaction], Field(min_length=1, max_length=64)]


class PublicAuthority(Statement):
    public_key: Identity
    public_key_sha256: Digest
    kms_key_arn: Annotated[
        str,
        StringConstraints(
            pattern=r"^arn:aws(?:-us-gov|-cn)?:kms:[a-z0-9-]+:[0-9]{12}:key/"
            r"[a-f0-9-]{36}$"
        ),
    ]


class Trust(Statement):
    schema_version: Literal[1] = 1
    approval: PublicAuthority
    provisioner: PublicAuthority
    witness: PublicAuthority


def canonical(value: Statement) -> bytes:
    return json.dumps(
        value.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def statement_sha256(value: Statement) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()
