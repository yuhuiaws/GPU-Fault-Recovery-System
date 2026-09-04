from __future__ import annotations

import re
from datetime import datetime
from enum import StrEnum
from typing import Any, cast
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from gpu_fault.digests import normalized_sha256
from gpu_fault.fleet_compatibility import CURRENT_AGENT_PROTOCOL_VERSION
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
    first_wave_max_unavailable: int | None = Field(default=None, ge=1)
    max_unavailable_per_failure_domain: int = Field(default=1, ge=1)
    node_failure_domains: dict[str, str] = Field(default_factory=dict)

    @field_validator(  # type: ignore[untyped-decorator]
        "desired_artifact_sha256",
        "desired_compatibility_digest",
        "desired_bundle_sha256",
        "desired_template_sha256",
        "desired_config_digest",
    )
    @classmethod
    def validate_digest(cls, value: str | None) -> str | None:
        return normalized_sha256(value)

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
        if (
            self.first_wave_max_unavailable is not None
            and self.first_wave_max_unavailable > self.max_unavailable
        ):
            raise ValueError("first_wave_max_unavailable cannot exceed max_unavailable")
        if self.max_unavailable_per_failure_domain > self.max_unavailable:
            raise ValueError(
                "max_unavailable_per_failure_domain cannot exceed max_unavailable"
            )
        if self.node_failure_domains and set(self.node_failure_domains) != set(
            self.node_ids
        ):
            raise ValueError(
                "node_failure_domains must contain exactly the deployment nodes"
            )
        if any(
            not value.strip() or len(value) > 128
            for value in self.node_failure_domains.values()
        ):
            raise ValueError("node failure domains must be non-empty and bounded")
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

    @model_validator(mode="before")  # type: ignore[untyped-decorator]
    @classmethod
    def normalize_candidate_only_fields(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        for field in (
            "attempt_generation",
            "first_wave_max_unavailable",
            "max_unavailable_per_failure_domain",
            "node_failure_domains",
        ):
            normalized.pop(field, None)
        if normalized.get("status") == "CANCELLED":
            normalized["status"] = DeploymentStatus.FAILED.value
            normalized["nodes"] = [
                {
                    **item,
                    "status": (
                        item.get("status")
                        if item.get("status") == DeploymentNodeStatus.READY.value
                        else DeploymentNodeStatus.FAILED.value
                    ),
                }
                for item in normalized.get("nodes", [])
                if isinstance(item, dict)
            ]
        return normalized


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


def deployment_waves(request: FleetDeploymentRequest) -> list[list[str]]:
    remaining = sorted(request.node_ids)
    domains = {
        node_id: request.node_failure_domains.get(node_id, node_id)
        for node_id in remaining
    }
    waves: list[list[str]] = []
    while remaining:
        capacity = (
            request.first_wave_max_unavailable
            if not waves and request.first_wave_max_unavailable is not None
            else request.max_unavailable
        )
        counts: dict[str, int] = {}
        wave: list[str] = []
        for node_id in remaining:
            domain = domains[node_id]
            if (
                len(wave) >= capacity
                or counts.get(domain, 0) >= request.max_unavailable_per_failure_domain
            ):
                continue
            wave.append(node_id)
            counts[domain] = counts.get(domain, 0) + 1
        if not wave:
            raise ValueError("failure-domain constraints produced an empty wave")
        selected = set(wave)
        remaining = [node_id for node_id in remaining if node_id not in selected]
        waves.append(wave)
    return waves


def create_deployment_record(
    registry: Any,
    request: FleetDeploymentRequest,
) -> FleetDeployment:
    now = registry.now()
    nodes = sorted(request.node_ids)
    waves = deployment_waves(request)
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
    # Keyed lookup rather than a scan for one id. ``deployment_id`` is the
    # store key, and this is the idempotency check on the rollout entry point,
    # so the scan it replaces read and decoded every deployment ever recorded
    # in the region on every call -- including the common case of a fresh
    # ``uuid4`` id that cannot possibly be there.
    # ``NotFoundError`` derives from ``KeyError``, so this catches every store's
    # miss without importing the store package into a model module.
    try:
        existing = registry.store.get_fleet_deployment(deployment_id)
    except KeyError:
        existing = None
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
            and record.lease_expires_at is not None
            and record.lease_expires_at > now
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
    supersede_never_started_deployments(registry, deployment)
    return deployment


def supersede_never_started_deployments(
    registry: Any,
    successor: FleetDeployment,
) -> list[str]:
    """Terminalize the cluster's never-started rollouts behind a newer one.

    ``DeploymentStatus`` has no terminal "superseded" value, and nothing else
    closes a record whose creator died before starting a wave. That is not a
    cosmetic leak: ``execution/fleet_preflight.py`` fences every destructive
    remediation for a cluster behind any non-terminal deployment, and the
    retention drain only collects ``SUCCEEDED``/``FAILED`` records, so an
    abandoned one both fences the cluster and never ages out. On 2026-09-03 a
    ``PLANNED`` record did exactly that for 36 hours.

    ``PLANNED`` is derived state -- ``deployment_status`` returns it only when
    every node is still ``PENDING`` -- so a record matched here has never
    started a wave, has nothing draining and nothing mid-install. Combined with
    a strictly newer deployment for the same cluster, which is being saved right
    now, there is no reading under which it is still going to run: a rollout
    waiting on its first wave is not overtaken by a later rollout of the same
    cluster.

    Resolving it here rather than in the fence is what makes it stick. The
    record becomes a real terminal record, so the fence needs no exception for
    it and retention collects it on the normal schedule.
    """

    superseded = []
    for deployment in registry.store.list_active_fleet_deployments(
        successor.cluster_id
    ):
        if deployment.deployment_id == successor.deployment_id:
            continue
        if deployment.status is not DeploymentStatus.PLANNED:
            continue
        if deployment.created_at >= successor.created_at:
            continue
        registry.cancel_deployment(
            deployment.deployment_id,
            reason=(
                "superseded by fleet deployment "
                f"{successor.deployment_id} before starting a wave"
            ),
        )
        superseded.append(deployment.deployment_id)
    return superseded
