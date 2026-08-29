from __future__ import annotations

import re
from datetime import datetime
from enum import StrEnum
from typing import Any, cast
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from gpu_fault.fleet_compatibility import (
    CURRENT_AGENT_PROTOCOL_VERSION,
    SHA256_PATTERN,
)
from gpu_fault.models import StrictModel


DEPLOYMENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


class DeploymentStatus(StrEnum):
    PLANNED = "PLANNED"
    IN_PROGRESS = "IN_PROGRESS"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


class DeploymentNodeStatus(StrEnum):
    PENDING = "PENDING"
    INSTALLING = "INSTALLING"
    READY = "READY"
    FAILED = "FAILED"


class FleetDeploymentRequest(StrictModel):
    deployment_id: str | None = None
    cluster_id: str
    node_ids: list[str] = Field(min_length=1)
    desired_agent_protocol_version: int = Field(
        default=CURRENT_AGENT_PROTOCOL_VERSION,
        ge=1,
    )
    desired_agent_version: str
    desired_artifact_sha256: str
    desired_compatibility_digest: str | None = None
    desired_bundle_sha256: str | None = None
    desired_template_sha256: str | None = None
    desired_policy_version: str
    desired_runtime_profile_version: str
    desired_config_digest: str
    max_unavailable: int = Field(default=1, ge=1)

    @field_validator(  # type: ignore[untyped-decorator]
        "desired_artifact_sha256",
        "desired_compatibility_digest",
        "desired_bundle_sha256",
        "desired_template_sha256",
        "desired_config_digest",
    )
    @classmethod
    def validate_digest(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.lower()
        if not SHA256_PATTERN.fullmatch(normalized):
            raise ValueError("digest must be a SHA-256 hex value")
        return normalized

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_nodes(self) -> FleetDeploymentRequest:
        if self.deployment_id is not None and not DEPLOYMENT_ID_PATTERN.fullmatch(
            self.deployment_id
        ):
            raise ValueError("deployment_id contains unsupported characters")
        if len(set(self.node_ids)) != len(self.node_ids):
            raise ValueError("deployment node_ids must be unique")
        if self.max_unavailable > len(self.node_ids):
            raise ValueError("max_unavailable cannot exceed node count")
        return self


class DeploymentNode(StrictModel):
    node_id: str
    status: DeploymentNodeStatus = DeploymentNodeStatus.PENDING
    reason: str | None = None
    updated_at: datetime


class FleetDeployment(StrictModel):
    deployment_id: str = Field(default_factory=lambda: f"fleet-deployment-{uuid4()}")
    cluster_id: str
    desired_agent_protocol_version: int = Field(
        default=CURRENT_AGENT_PROTOCOL_VERSION,
        ge=1,
    )
    desired_agent_version: str
    desired_artifact_sha256: str
    desired_compatibility_digest: str | None = None
    desired_bundle_sha256: str | None = None
    desired_template_sha256: str | None = None
    desired_policy_version: str
    desired_runtime_profile_version: str
    desired_config_digest: str
    max_unavailable: int
    waves: list[list[str]]
    nodes: list[DeploymentNode]
    status: DeploymentStatus = DeploymentStatus.PLANNED
    created_at: datetime
    updated_at: datetime


class DeploymentNodeUpdate(StrictModel):
    status: DeploymentNodeStatus
    reason: str | None = None


class DeploymentWaveLease(StrictModel):
    deployment_id: str
    wave_index: int = Field(ge=0)
    node_ids: list[str]
    issued_at: datetime


def deployment_status(
    nodes: list[DeploymentNode],
) -> DeploymentStatus:
    statuses = {item.status for item in nodes}
    if DeploymentNodeStatus.FAILED in statuses:
        return DeploymentStatus.FAILED
    if statuses == {DeploymentNodeStatus.READY}:
        return DeploymentStatus.SUCCEEDED
    if statuses == {DeploymentNodeStatus.PENDING}:
        return DeploymentStatus.PLANNED
    return DeploymentStatus.IN_PROGRESS


def active_deployment_wave(
    deployment: FleetDeployment,
) -> list[str] | None:
    by_node = {item.node_id: item.status for item in deployment.nodes}
    for wave in deployment.waves:
        if any(by_node[node_id] is not DeploymentNodeStatus.READY for node_id in wave):
            return wave
    return None


def create_deployment_record(
    registry: Any,
    request: FleetDeploymentRequest,
) -> FleetDeployment:
    now = registry.now()
    nodes = sorted(request.node_ids)
    waves = [
        nodes[index : index + request.max_unavailable]
        for index in range(0, len(nodes), request.max_unavailable)
    ]
    expected_identity = (
        request.desired_agent_protocol_version,
        request.desired_agent_version,
        request.desired_artifact_sha256,
        request.desired_compatibility_digest or request.desired_artifact_sha256,
        request.desired_bundle_sha256,
        request.desired_template_sha256,
        request.desired_policy_version,
        request.desired_runtime_profile_version,
        request.desired_config_digest,
    )
    agents = {
        record.node_id: record
        for record in registry.store.list_agents(request.cluster_id)
    }
    deployment_id = request.deployment_id or (f"fleet-deployment-{uuid4()}")
    existing = next(
        (
            item
            for item in registry.store.list_fleet_deployments()
            if item.deployment_id == deployment_id
        ),
        None,
    )
    if existing is not None:
        contract = {
            "cluster_id": request.cluster_id,
            "desired_agent_protocol_version": (request.desired_agent_protocol_version),
            "desired_agent_version": request.desired_agent_version,
            "desired_artifact_sha256": request.desired_artifact_sha256,
            "desired_compatibility_digest": (
                request.desired_compatibility_digest or request.desired_artifact_sha256
            ),
            "desired_bundle_sha256": request.desired_bundle_sha256,
            "desired_template_sha256": request.desired_template_sha256,
            "desired_policy_version": request.desired_policy_version,
            "desired_runtime_profile_version": (
                request.desired_runtime_profile_version
            ),
            "desired_config_digest": request.desired_config_digest,
            "max_unavailable": request.max_unavailable,
            "waves": waves,
        }
        if any(
            getattr(existing, field) != expected for field, expected in contract.items()
        ):
            raise ValueError("deployment_id is already bound to a different rollout")
        return cast(FleetDeployment, existing)
    deployment_nodes = []
    for node_id in nodes:
        record = agents.get(node_id)
        ready = (
            record is not None
            and getattr(
                record.lifecycle_state,
                "value",
                record.lifecycle_state,
            )
            == "ACTIVE"
            and record.identity == expected_identity
        )
        deployment_nodes.append(
            DeploymentNode(
                node_id=node_id,
                status=(
                    DeploymentNodeStatus.READY
                    if ready
                    else DeploymentNodeStatus.PENDING
                ),
                updated_at=now,
            )
        )
    deployment = FleetDeployment(
        deployment_id=deployment_id,
        cluster_id=request.cluster_id,
        desired_agent_protocol_version=request.desired_agent_protocol_version,
        desired_agent_version=request.desired_agent_version,
        desired_artifact_sha256=request.desired_artifact_sha256,
        desired_compatibility_digest=(
            request.desired_compatibility_digest or request.desired_artifact_sha256
        ),
        desired_bundle_sha256=request.desired_bundle_sha256,
        desired_template_sha256=request.desired_template_sha256,
        desired_policy_version=request.desired_policy_version,
        desired_runtime_profile_version=(request.desired_runtime_profile_version),
        desired_config_digest=request.desired_config_digest,
        max_unavailable=request.max_unavailable,
        waves=waves,
        nodes=deployment_nodes,
        status=registry._deployment_status(deployment_nodes),
        created_at=now,
        updated_at=now,
    )
    registry.store.save_fleet_deployment(deployment)
    return deployment
