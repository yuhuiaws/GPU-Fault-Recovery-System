from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import ipaddress
import json
import re
from datetime import datetime, timedelta, timezone
from enum import StrEnum
from threading import RLock
from typing import Any
from uuid import uuid4

from pydantic import Field, field_validator

from gpu_fault.collector_requirements import (
    CollectorServices,
    ReportedCollectorServices,
)
from gpu_fault.digests import normalized_sha256
from gpu_fault.fleet_compatibility import (
    CURRENT_AGENT_PROTOCOL_VERSION as CURRENT_AGENT_PROTOCOL_VERSION,
)
from gpu_fault.fleet_compatibility import (
    NODE_ACTION_KEY_VERSION_DERIVED as NODE_ACTION_KEY_VERSION_DERIVED,
)
from gpu_fault.fleet_compatibility import (
    NODE_ACTION_KEY_VERSION_SHARED as NODE_ACTION_KEY_VERSION_SHARED,
)
from gpu_fault.fleet_compatibility import (
    FleetCompatibilityPolicy as FleetCompatibilityPolicy,
)
from gpu_fault.fleet_compatibility import (
    pin_value_is_accepted,
    rollout_compatibility_reasons,
)
from gpu_fault.fleet_deployment import (
    DeploymentNode as DeploymentNode,
)
from gpu_fault.fleet_deployment import (
    DeploymentNodeStatus as DeploymentNodeStatus,
)
from gpu_fault.fleet_deployment import (
    DeploymentNodeUpdate as DeploymentNodeUpdate,
)
from gpu_fault.fleet_deployment import (
    DeploymentStatus as DeploymentStatus,
)
from gpu_fault.fleet_deployment import (
    DeploymentWaveLease as DeploymentWaveLease,
)
from gpu_fault.fleet_deployment import (
    FleetDeployment as FleetDeployment,
)
from gpu_fault.fleet_deployment import (
    FleetDeploymentRequest as FleetDeploymentRequest,
)
from gpu_fault.fleet_deployment import (
    active_deployment_wave as active_deployment_wave,
)
from gpu_fault.fleet_deployment import (
    create_deployment_record as create_deployment_record,
)
from gpu_fault.fleet_deployment import (
    deployment_status as deployment_status,
)
from gpu_fault.fleet_endpoint import (
    DEFAULT_AGENT_ENDPOINT_PORTS,
    endpoint_networks_for_cluster,
    validate_agent_endpoint,
)
from gpu_fault.fleet_endpoint import (
    parse_endpoint_networks as parse_endpoint_networks,
)
from gpu_fault.fleet_registry_deployments import (
    cancel_deployment as cancel_fleet_deployment,
)
from gpu_fault.fleet_registry_deployments import (
    reconcile_deployment as reconcile_fleet_deployment,
)
from gpu_fault.fleet_registry_deployments import (
    reconcile_deployments as reconcile_fleet_deployments,
)
from gpu_fault.fleet_registry_deployments import (
    retry_failed_deployment as retry_fleet_deployment,
)
from gpu_fault.fleet_registry_deployments import (
    start_next_wave as start_fleet_wave,
)
from gpu_fault.fleet_registry_deployments import (
    update_deployment_node as update_fleet_deployment_node,
)
from gpu_fault.installation_inventory import (
    InstalledUnitInventory,
    InstalledUnitReport,
    resolve_agent_inventory,
)
from gpu_fault.models import StrictModel, WorkflowOperation
from gpu_fault.node_action_keys import (
    derive_node_action_secret as derive_node_action_secret,
)
from gpu_fault.node_action_keys import (
    node_action_secrets_from_environment as node_action_secrets_from_environment,
)
from gpu_fault.node_action_keys import (
    resolve_node_action_secret,
)
from gpu_fault.store import NotFoundError
from gpu_fault.store.contracts import ControlPlaneStore

# A fleet node identifier is a bare host label (Kubernetes node name, HyperPod
# instance id, or EC2 private DNS name). Anchor it so a client-supplied value
# cannot smuggle whitespace, control characters, path traversal, URL userinfo,
# or scheme/port punctuation into anything that later treats it as an identity
# or a connection host.
NODE_ID_PATTERN = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,251}[A-Za-z0-9])?$")

# One PEM certificate block. Matched non-greedily so a bundle yields one match
# per certificate and every block in the chain gets validated, not just the
# first one.
CERTIFICATE_PEM_BLOCK = re.compile(
    r"-----BEGIN CERTIFICATE-----(?P<body>.*?)-----END CERTIFICATE-----",
    re.DOTALL,
)


def validate_node_identifier(value: str) -> str:
    if not NODE_ID_PATTERN.fullmatch(value):
        raise ValueError(
            "node_id must be a bare host label of letters, digits, '.', '-' or "
            "'_' (no whitespace, control characters, or path separators)"
        )
    return value


class AgentHeartbeat(StrictModel):
    heartbeat_id: str = Field(default_factory=lambda: f"heartbeat-{uuid4()}")
    cluster_id: str
    node_id: str
    endpoint: str
    tls_certificate_pem: str | None = None
    agent_protocol_version: int = Field(default=1, ge=1)
    node_action_key_version: int = Field(
        default=NODE_ACTION_KEY_VERSION_SHARED,
        ge=NODE_ACTION_KEY_VERSION_SHARED,
        le=NODE_ACTION_KEY_VERSION_DERIVED,
    )
    agent_version: str
    artifact_sha256: str
    compatibility_digest: str | None = None
    installer_bundle_sha256: str | None = None
    installer_template_sha256: str | None = None
    policy_version: str
    runtime_profile_version: str
    config_digest: str
    allowed_operations: list[WorkflowOperation]
    collector_services: ReportedCollectorServices = Field(default_factory=dict)
    installed_unit_report: InstalledUnitReport | None = None
    boot_id: str | None = None
    node_instance_id: str | None = None
    agent_incarnation_id: str | None = None
    observed_at: datetime

    @field_validator("node_id")
    @classmethod
    def validate_node_id(cls, value: str) -> str:
        return validate_node_identifier(value)

    @field_validator("endpoint")
    @classmethod
    def validate_endpoint(cls, value: str) -> str:
        if not value.startswith(("http://", "https://")):
            raise ValueError("agent endpoint must use http or https")
        return value.rstrip("/")

    @field_validator("tls_certificate_pem")  # type: ignore[untyped-decorator]
    @classmethod
    def validate_tls_certificate(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if (
            not normalized.startswith("-----BEGIN CERTIFICATE-----")
            or not normalized.endswith("-----END CERTIFICATE-----")
            or len(normalized) > 65536
        ):
            raise ValueError(
                "agent TLS certificate must be one or more PEM certificates"
            )
        # Validate the whole chain, not just the first block: a bundle can hide
        # a malformed trailing certificate behind a well-formed leaf. Every
        # BEGIN/END block must carry a non-empty base64 body, and the gaps
        # between and around the blocks must be whitespace only, so a bundle
        # cannot smuggle arbitrary bytes past the shape check.
        blocks = list(CERTIFICATE_PEM_BLOCK.finditer(normalized))
        if not blocks:
            raise ValueError(
                "agent TLS certificate must be one or more PEM certificates"
            )
        cursor = 0
        for block in blocks:
            if normalized[cursor : block.start()].strip():
                raise ValueError(
                    "agent TLS certificate has content outside a PEM block"
                )
            body = "".join(block.group("body").split())
            try:
                decoded = base64.b64decode(body, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise ValueError(
                    "agent TLS certificate PEM block is not valid base64"
                ) from exc
            if not decoded:
                raise ValueError("agent TLS certificate PEM block is empty")
            cursor = block.end()
        if normalized[cursor:].strip():
            raise ValueError("agent TLS certificate has content outside a PEM block")
        return normalized + "\n"

    @field_validator(
        "artifact_sha256",
        "compatibility_digest",
        "config_digest",
        "installer_bundle_sha256",
        "installer_template_sha256",
    )
    @classmethod
    def validate_digest(cls, value: str | None) -> str | None:
        return normalized_sha256(value)


class SignedAgentHeartbeat(StrictModel):
    heartbeat: AgentHeartbeat
    signature: str


def canonical_heartbeat(heartbeat: AgentHeartbeat) -> bytes:
    return json.dumps(
        heartbeat.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def sign_agent_heartbeat(heartbeat: AgentHeartbeat, secret: str) -> str:
    return hmac.new(
        secret.encode(),
        canonical_heartbeat(heartbeat),
        hashlib.sha256,
    ).hexdigest()


def validate_agent_tls_policy(
    policy: FleetCompatibilityPolicy,
    heartbeat: AgentHeartbeat,
) -> None:
    if policy.require_tls and (
        not heartbeat.endpoint.startswith("https://")
        or heartbeat.tls_certificate_pem is None
    ):
        raise ValueError(
            "agent endpoint must use HTTPS and publish its signed TLS certificate"
        )


class AgentLifecycleState(StrEnum):
    ACTIVE = "ACTIVE"
    DRAINING = "DRAINING"
    REVOKED = "REVOKED"


class AgentRecord(StrictModel):
    cluster_id: str
    node_id: str
    endpoint: str
    tls_certificate_pem: str | None = None
    agent_protocol_version: int = Field(default=1, ge=1)
    node_action_key_version: int = Field(
        default=NODE_ACTION_KEY_VERSION_SHARED,
        ge=NODE_ACTION_KEY_VERSION_SHARED,
        le=NODE_ACTION_KEY_VERSION_DERIVED,
    )
    agent_version: str
    artifact_sha256: str
    compatibility_digest: str | None = None
    installer_bundle_sha256: str | None = None
    installer_template_sha256: str | None = None
    policy_version: str
    runtime_profile_version: str
    config_digest: str
    allowed_operations: list[WorkflowOperation]
    collector_services: CollectorServices = Field(default_factory=dict)
    installed_unit_inventory: InstalledUnitInventory | None = None
    boot_id: str | None = None
    node_instance_id: str | None = None
    agent_incarnation_id: str | None = None
    retired_incarnation_ids: list[str] = Field(default_factory=list)
    first_seen_at: datetime
    last_seen_at: datetime
    lease_expires_at: datetime | None = None
    generation: int = Field(default=1, ge=1)
    lifecycle_state: AgentLifecycleState = AgentLifecycleState.ACTIVE
    transition_id: str | None = None
    transition_reason: str | None = None
    transition_started_at: datetime | None = None

    @property
    def identity(
        self,
    ) -> tuple[int, str, str, str, str | None, str | None, str, str, str]:
        return (
            self.agent_protocol_version,
            self.agent_version,
            self.artifact_sha256,
            self.compatibility_digest or self.artifact_sha256,
            self.installer_bundle_sha256,
            self.installer_template_sha256,
            self.policy_version,
            self.runtime_profile_version,
            self.config_digest,
        )


class AgentTransitionRequest(StrictModel):
    expected_generation: int = Field(ge=1)
    transition_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class FleetNodeReadiness(StrictModel):
    node_id: str
    ready: bool
    generation: int | None = None
    endpoint: str | None = None
    reasons: list[str] = Field(default_factory=list)


class FleetReadinessRequest(StrictModel):
    cluster_id: str
    node_ids: list[str] = Field(min_length=1)


class FleetReadinessReport(StrictModel):
    cluster_id: str
    ready: bool
    evaluated_at: datetime
    nodes: list[FleetNodeReadiness]


class BarrierState(StrEnum):
    PREPARING = "PREPARING"
    PREPARED = "PREPARED"
    COMMITTING = "COMMITTING"
    COMMITTED = "COMMITTED"
    ABORTED = "ABORTED"
    FAILED = "FAILED"


class BarrierParticipantState(StrEnum):
    PENDING = "PENDING"
    PREPARED = "PREPARED"
    COMMITTED = "COMMITTED"
    FAILED = "FAILED"


class BarrierParticipant(StrictModel):
    node_id: str
    agent_generation: int = Field(ge=1)
    state: BarrierParticipantState = BarrierParticipantState.PENDING
    prepare_details: dict[str, Any] = Field(default_factory=dict)
    commit_details: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    updated_at: datetime


class MultiNodeBarrier(StrictModel):
    barrier_id: str
    cluster_id: str
    workflow_request_id: str
    incident_id: str
    fencing_token: int = Field(ge=1)
    operation: WorkflowOperation
    state: BarrierState
    participants: list[BarrierParticipant]
    created_at: datetime
    updated_at: datetime


# The two verdicts ``FleetRegistry._annotate_pin_drift`` can reach for a
# mismatched pin: the node is behind the fleet, or the pin is ahead of it.
PIN_DRIFT_KINDS = ("NODE_STALE", "PIN_AHEAD_OF_FLEET")


class FleetRegistry:
    _deployment_status = staticmethod(deployment_status)
    _active_wave = staticmethod(active_deployment_wave)

    def __init__(
        self,
        store: ControlPlaneStore,
        secret: str,
        policy: FleetCompatibilityPolicy | None = None,
        *,
        now=None,
        endpoint_allowed_ports: frozenset[int] = (DEFAULT_AGENT_ENDPOINT_PORTS),
        endpoint_allowed_host_suffixes: tuple[str, ...] = (),
        endpoint_allowed_networks: tuple[
            ipaddress.IPv4Network | ipaddress.IPv6Network,
            ...,
        ] = (),
        endpoint_allowed_networks_by_cluster: dict[
            str,
            tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...],
        ]
        | None = None,
        node_secrets: dict[str, str] | None = None,
    ) -> None:
        if len(secret) < 32:
            raise ValueError("agent registration secret must be at least 32 characters")
        if not endpoint_allowed_ports:
            raise ValueError("at least one agent endpoint port must be allowed")
        self.store = store
        self.secret = secret
        self.node_secrets = dict(node_secrets or {})
        self.endpoint_allowed_ports = endpoint_allowed_ports
        self.endpoint_allowed_host_suffixes = endpoint_allowed_host_suffixes
        self.endpoint_allowed_networks = endpoint_allowed_networks
        self.endpoint_allowed_networks_by_cluster = dict(
            endpoint_allowed_networks_by_cluster or {}
        )
        self.policy = policy or FleetCompatibilityPolicy()
        self.now = now or (lambda: datetime.now(timezone.utc))
        self._lock = RLock()
        self._node_locks = tuple(RLock() for _ in range(64))
        # ARCH-E E5: drifted nodes by cluster and kind as of that cluster's
        # latest readiness evaluation. PIN_AHEAD_OF_FLEET used to exist only
        # as a readiness reason, so a pin nobody had shipped stalled every
        # node action on the cluster without a single exported signal.
        self.pin_drift_nodes: dict[str, dict[str, int]] = {}

    def register(self, envelope: SignedAgentHeartbeat) -> AgentRecord:
        heartbeat = envelope.heartbeat
        node_lock = self._node_locks[
            hash((heartbeat.cluster_id, heartbeat.node_id)) % len(self._node_locks)
        ]
        with node_lock:
            return self._register(envelope)

    def _register(self, envelope: SignedAgentHeartbeat) -> AgentRecord:
        heartbeat = envelope.heartbeat
        required_key_version = self.policy.required_node_action_key_version
        if (
            required_key_version is not None
            and heartbeat.node_action_key_version != required_key_version
        ):
            # A pinned v2 fleet must never fall back to its shared secret.
            raise ValueError("node action key version is not accepted")
        heartbeat_secret = resolve_node_action_secret(
            self.secret,
            self.node_secrets,
            heartbeat.cluster_id,
            heartbeat.node_id,
            heartbeat.node_action_key_version,
        )
        expected = sign_agent_heartbeat(heartbeat, heartbeat_secret)
        if not hmac.compare_digest(expected, envelope.signature):
            raise ValueError("invalid agent heartbeat signature")
        # Authenticate before exposing endpoint policy; recheck every heartbeat.
        validate_agent_endpoint(
            heartbeat.node_id,
            heartbeat.endpoint,
            allowed_ports=self.endpoint_allowed_ports,
            allowed_host_suffixes=(self.endpoint_allowed_host_suffixes),
            allowed_networks=endpoint_networks_for_cluster(
                heartbeat.cluster_id,
                self.endpoint_allowed_networks_by_cluster,
                self.endpoint_allowed_networks,
            ),
        )
        validate_agent_tls_policy(self.policy, heartbeat)
        now = self.now()
        if heartbeat.observed_at > now + timedelta(seconds=30):
            raise ValueError("agent heartbeat is from the future")
        if heartbeat.observed_at < now - timedelta(minutes=5):
            raise ValueError("agent heartbeat is too old")
        for _ in range(5):
            try:
                existing = self.store.get_agent(heartbeat.cluster_id, heartbeat.node_id)
            except NotFoundError:
                existing = None
            record = self._next_agent_record(heartbeat, existing, now)
            if self.store.replace_agent_if_matches(record, existing):
                lease_was_stale = existing is not None and (
                    existing.lease_expires_at is None
                    or existing.lease_expires_at <= now
                )
                if (
                    existing is None
                    or existing.identity != record.identity
                    or existing.lifecycle_state is not record.lifecycle_state
                    or lease_was_stale
                ):
                    self._reconcile_deployments(record)
                return record
        raise ValueError("agent heartbeat conflicted with concurrent updates")

    def _next_agent_record(
        self,
        heartbeat: AgentHeartbeat,
        existing: AgentRecord | None,
        now: datetime,
    ) -> AgentRecord:
        if existing is not None and heartbeat.observed_at < existing.last_seen_at:
            raise ValueError("agent heartbeat is older than the stored heartbeat")
        incarnation = (
            heartbeat.agent_incarnation_id or heartbeat.boot_id or heartbeat.endpoint
        )
        existing_incarnation = (
            (existing.agent_incarnation_id if existing is not None else None)
            or (existing.boot_id if existing is not None else None)
            or (existing.endpoint if existing is not None else None)
        )
        if existing is not None and incarnation in existing.retired_incarnation_ids:
            raise ValueError("agent incarnation has been retired")
        incarnation_changed = (
            existing is not None and existing_incarnation != incarnation
        )
        lease_active = (
            existing is not None
            and existing.lease_expires_at is not None
            and existing.lease_expires_at > now
        )
        same_instance_reboot = (
            incarnation_changed
            and heartbeat.boot_id != existing.boot_id
            and (
                (
                    heartbeat.node_instance_id is not None
                    and heartbeat.node_instance_id == existing.node_instance_id
                )
                or (
                    heartbeat.node_instance_id is None
                    and existing.node_instance_id is None
                    and heartbeat.endpoint == existing.endpoint
                )
            )
        )
        if (
            incarnation_changed
            and lease_active
            and not same_instance_reboot
            and existing.lifecycle_state is AgentLifecycleState.ACTIVE
        ):
            raise ValueError("another agent incarnation holds the node lease")
        identity = (
            heartbeat.agent_protocol_version,
            heartbeat.agent_version,
            heartbeat.artifact_sha256,
            heartbeat.compatibility_digest or heartbeat.artifact_sha256,
            heartbeat.installer_bundle_sha256,
            heartbeat.installer_template_sha256,
            heartbeat.policy_version,
            heartbeat.runtime_profile_version,
            heartbeat.config_digest,
        )
        installed_inventory, inventory_changed = resolve_agent_inventory(
            heartbeat.installed_unit_report, existing
        )
        changed = existing is not None and (
            incarnation_changed
            or existing.identity != identity
            or existing.node_action_key_version != heartbeat.node_action_key_version
            or existing.endpoint != heartbeat.endpoint
            or existing.tls_certificate_pem != heartbeat.tls_certificate_pem
            or existing.boot_id != heartbeat.boot_id
            or existing.node_instance_id != heartbeat.node_instance_id
            or inventory_changed
        )
        retired = list(existing.retired_incarnation_ids) if existing is not None else []
        if incarnation_changed and existing_incarnation:
            retired = [
                *[item for item in retired if item != existing_incarnation],
                existing_incarnation,
            ][-16:]
        preserve_transition = (
            existing is not None
            and not incarnation_changed
            and existing.lifecycle_state is AgentLifecycleState.DRAINING
        )
        return AgentRecord(
            cluster_id=heartbeat.cluster_id,
            node_id=heartbeat.node_id,
            endpoint=heartbeat.endpoint,
            tls_certificate_pem=heartbeat.tls_certificate_pem,
            agent_protocol_version=heartbeat.agent_protocol_version,
            node_action_key_version=(heartbeat.node_action_key_version),
            agent_version=heartbeat.agent_version,
            artifact_sha256=heartbeat.artifact_sha256,
            compatibility_digest=(
                heartbeat.compatibility_digest or heartbeat.artifact_sha256
            ),
            installer_bundle_sha256=heartbeat.installer_bundle_sha256,
            installer_template_sha256=heartbeat.installer_template_sha256,
            policy_version=heartbeat.policy_version,
            runtime_profile_version=(heartbeat.runtime_profile_version),
            config_digest=heartbeat.config_digest,
            allowed_operations=heartbeat.allowed_operations,
            collector_services=heartbeat.collector_services,
            installed_unit_inventory=installed_inventory,
            boot_id=heartbeat.boot_id,
            node_instance_id=heartbeat.node_instance_id,
            agent_incarnation_id=incarnation,
            retired_incarnation_ids=retired,
            first_seen_at=(
                existing.first_seen_at
                if existing is not None
                else heartbeat.observed_at
            ),
            last_seen_at=heartbeat.observed_at,
            lease_expires_at=(
                heartbeat.observed_at
                + timedelta(seconds=self.policy.max_heartbeat_age_seconds)
            ),
            generation=(
                1 if existing is None else existing.generation + (1 if changed else 0)
            ),
            lifecycle_state=(
                existing.lifecycle_state
                if preserve_transition
                else AgentLifecycleState.ACTIVE
            ),
            transition_id=(existing.transition_id if preserve_transition else None),
            transition_reason=(
                existing.transition_reason if preserve_transition else None
            ),
            transition_started_at=(
                existing.transition_started_at if preserve_transition else None
            ),
        )

    def readiness(self, cluster_id: str, node_ids: list[str]) -> FleetReadinessReport:
        evaluated_at = self.now()
        records: dict[str, AgentRecord] = {}
        results: dict[str, FleetNodeReadiness] = {}
        for node_id in node_ids:
            try:
                record = self.store.get_agent(cluster_id, node_id)
                records[node_id] = record
                reasons = self._policy_reasons(record, evaluated_at)
                results[node_id] = FleetNodeReadiness(
                    node_id=node_id,
                    ready=not reasons,
                    generation=record.generation,
                    endpoint=record.endpoint,
                    reasons=reasons,
                )
            except NotFoundError:
                results[node_id] = FleetNodeReadiness(
                    node_id=node_id,
                    ready=False,
                    reasons=["agent is not registered"],
                )
        eligible = [
            records[node_id]
            for node_id in node_ids
            if node_id in records and results[node_id].ready
        ]
        identities = {record.identity for record in eligible}
        if len(identities) > 1:
            for record in eligible:
                result = results[record.node_id]
                results[record.node_id] = result.model_copy(
                    update={
                        "ready": False,
                        "reasons": [
                            *result.reasons,
                            "agent version/config identity differs "
                            "within the selected node set",
                        ],
                    }
                )
        self._annotate_pin_drift(cluster_id, records, results)
        nodes = [results[node_id] for node_id in node_ids]
        return FleetReadinessReport(
            cluster_id=cluster_id,
            ready=all(item.ready for item in nodes),
            evaluated_at=evaluated_at,
            nodes=nodes,
        )

    def _pinned_fields(
        self,
    ) -> tuple[tuple[str, str, str], ...]:
        """Pins whose mismatch can be blamed on one side or the other."""
        candidates = (
            (
                "agent version",
                "agent_version",
                self.policy.required_agent_version,
            ),
            (
                "artifact SHA-256",
                "artifact_sha256",
                self.policy.required_artifact_sha256,
            ),
            (
                "policy version",
                "policy_version",
                self.policy.required_policy_version,
            ),
            (
                "runtime profile",
                "runtime_profile_version",
                self.policy.required_runtime_profile_version,
            ),
            (
                "config digest",
                "config_digest",
                self.policy.required_config_digest,
            ),
        )
        return tuple(
            (label, attribute, required)
            for label, attribute, required in candidates
            if required is not None
        )

    def _annotate_pin_drift(
        self,
        cluster_id: str,
        records: dict[str, AgentRecord],
        results: dict[str, FleetNodeReadiness],
    ) -> None:
        """Say whether a pin mismatch means a bad pin or a stale node.

        "expected X, got Y" does not tell an operator which side to
        fix. Compare against the rest of the cluster: if some agent
        already runs the pinned value the node is behind, and if none
        does the pin itself was never shipped.

        Only agents that are still alive count as evidence that the
        pin shipped. Records outlive the machines they describe --
        a retired or reclaimed node keeps its row until the retention
        sweep -- so counting every row would call a pin that no
        running agent has ever served NODE_STALE and send the operator
        off to "restart the agent" on a node that is already current.
        """
        pins = self._pinned_fields()
        drift_counts = {kind: 0 for kind in PIN_DRIFT_KINDS}
        self.pin_drift_nodes[cluster_id] = drift_counts
        if not pins or not records:
            return
        drifted = {
            node_id: [
                (label, attribute, required)
                for label, attribute, required in pins
                if not pin_value_is_accepted(
                    self.policy,
                    attribute,
                    getattr(record, attribute),
                    required,
                )
            ]
            for node_id, record in records.items()
        }
        drifted = {node_id: items for node_id, items in drifted.items() if items}
        if not drifted:
            return
        fleet = self._live_fleet_agents(self._fleet_agents(cluster_id, records))
        for node_id, items in drifted.items():
            hints = []
            kinds: set[str] = set()
            for label, attribute, required in items:
                aligned = sorted(
                    agent.node_id
                    for agent in fleet
                    if getattr(agent, attribute) == required
                    and agent.node_id != node_id
                )
                if aligned:
                    kinds.add("NODE_STALE")
                    shown = ", ".join(aligned[:3])
                    hints.append(
                        f"{label} drift is NODE_STALE: "
                        f"{len(aligned)} other live agent(s) already run "
                        f"the pinned value ({shown}); upgrade or restart "
                        f"the agent on {node_id}"
                    )
                else:
                    kinds.add("PIN_AHEAD_OF_FLEET")
                    hints.append(
                        f"{label} drift is PIN_AHEAD_OF_FLEET: no live "
                        f"agent in {cluster_id} runs the pinned value; "
                        f"correct the control-plane pin or roll out the "
                        f"build first"
                    )
            for kind in kinds:
                drift_counts[kind] += 1
            result = results[node_id]
            results[node_id] = result.model_copy(
                update={"reasons": [*result.reasons, *hints]}
            )

    def _fleet_agents(
        self,
        cluster_id: str,
        records: dict[str, AgentRecord],
    ) -> list[AgentRecord]:
        """Widen the comparison set beyond the requested nodes if possible."""
        lister = getattr(self.store, "list_agents", None)
        if lister is None:
            return list(records.values())
        try:
            agents = list(lister(cluster_id))
        except Exception:  # noqa: BLE001 - the hint is advisory only
            return list(records.values())
        if not agents:
            return list(records.values())
        return agents

    def _live_fleet_agents(self, agents: list[AgentRecord]) -> list[AgentRecord]:
        """Keep only agents whose lease still proves they are running.

        Falls back to the full set when nothing is live, so the hint
        degrades to the old (over-broad) comparison rather than to a
        silent "no agent runs the pin" on a cluster that is merely
        mid-restart.
        """
        now = self.now()
        live = [
            agent
            for agent in agents
            if agent.lifecycle_state is AgentLifecycleState.ACTIVE
            and agent.lease_expires_at is not None
            and agent.lease_expires_at > now
        ]
        return live or agents

    def endpoint(self, cluster_id: str, node_id: str) -> tuple[str, int]:
        report = self.readiness(cluster_id, [node_id])
        node = report.nodes[0]
        if not node.ready or node.endpoint is None:
            raise ValueError(
                f"node {node_id} is not fleet-ready: " + ", ".join(node.reasons)
            )
        return node.endpoint, node.generation or 0

    def maintenance_endpoint(
        self,
        cluster_id: str,
        node_id: str,
        expected_generation: int,
    ) -> str:
        """Resolve a quiesced agent without requiring a fresh heartbeat."""
        record = self.store.get_agent(cluster_id, node_id)
        if record.generation != expected_generation:
            raise ValueError(
                f"agent generation changed from "
                f"{expected_generation} to {record.generation}"
            )
        if record.lifecycle_state is not AgentLifecycleState.ACTIVE:
            raise ValueError(f"agent lifecycle state is {record.lifecycle_state.value}")
        reasons = self._policy_reasons(record, self.now(), check_liveness=False)
        if reasons:
            raise ValueError(
                f"node {node_id} is not maintenance-ready: " + ", ".join(reasons)
            )
        return record.endpoint

    def drain_agent(
        self,
        cluster_id: str,
        node_id: str,
        request: AgentTransitionRequest,
    ) -> AgentRecord:
        with self._lock:
            for _ in range(5):
                current = self.store.get_agent(cluster_id, node_id)
                if (
                    current.transition_id == request.transition_id
                    and current.lifecycle_state
                    in {
                        AgentLifecycleState.DRAINING,
                        AgentLifecycleState.REVOKED,
                    }
                ):
                    return current
                if current.lifecycle_state is not AgentLifecycleState.ACTIVE:
                    raise ValueError("agent already belongs to another transition")
                if current.generation != request.expected_generation:
                    raise ValueError("stale agent generation for drain")
                now = self.now()
                drained = current.model_copy(
                    update={
                        "generation": current.generation + 1,
                        "lifecycle_state": (AgentLifecycleState.DRAINING),
                        "transition_id": request.transition_id,
                        "transition_reason": request.reason,
                        "transition_started_at": now,
                    }
                )
                if self.store.replace_agent_if_matches(drained, current):
                    return drained
            raise ValueError("agent drain conflicted with concurrent updates")

    def revoke_agent(
        self,
        cluster_id: str,
        node_id: str,
        request: AgentTransitionRequest,
    ) -> AgentRecord:
        with self._lock:
            for _ in range(5):
                current = self.store.get_agent(cluster_id, node_id)
                if (
                    current.lifecycle_state is AgentLifecycleState.REVOKED
                    and current.transition_id == request.transition_id
                ):
                    return current
                if (
                    current.lifecycle_state is not AgentLifecycleState.DRAINING
                    or current.transition_id != request.transition_id
                ):
                    raise ValueError("agent must be drained by the same transition")
                retired = list(current.retired_incarnation_ids)
                if (
                    current.agent_incarnation_id
                    and current.agent_incarnation_id not in retired
                ):
                    retired = [
                        *retired,
                        current.agent_incarnation_id,
                    ][-16:]
                revoked = current.model_copy(
                    update={
                        "lifecycle_state": (AgentLifecycleState.REVOKED),
                        "retired_incarnation_ids": retired,
                        "lease_expires_at": self.now(),
                    }
                )
                if self.store.replace_agent_if_matches(revoked, current):
                    return revoked
            raise ValueError("agent revoke conflicted with concurrent updates")

    def reactivate_agent(
        self,
        cluster_id: str,
        node_id: str,
        request: AgentTransitionRequest,
    ) -> AgentRecord:
        """Explicitly admit a revoked agent without bypassing fencing."""
        with self._lock:
            for _ in range(5):
                current = self.store.get_agent(cluster_id, node_id)
                if current.generation != request.expected_generation:
                    raise ValueError("stale agent generation")
                if current.lifecycle_state is not AgentLifecycleState.REVOKED:
                    raise ValueError("agent is not revoked")
                if current.transition_id != request.transition_id:
                    raise ValueError(
                        "reactivation must reference the revoking transition"
                    )
                retired = [
                    item
                    for item in current.retired_incarnation_ids
                    if item != current.agent_incarnation_id
                ]
                reactivated = current.model_copy(
                    update={
                        "lifecycle_state": AgentLifecycleState.ACTIVE,
                        "retired_incarnation_ids": retired,
                        "lease_expires_at": self.now(),
                        "transition_id": None,
                        "transition_reason": None,
                        "transition_started_at": None,
                    }
                )
                if self.store.replace_agent_if_matches(reactivated, current):
                    return reactivated
            raise ValueError("agent reactivation conflicted with concurrent updates")

    def _policy_reasons(
        self,
        record: AgentRecord,
        now: datetime,
        *,
        check_liveness: bool = True,
    ) -> list[str]:
        reasons = []
        if record.lifecycle_state is not AgentLifecycleState.ACTIVE:
            reasons.append(f"agent lifecycle state is {record.lifecycle_state.value}")
        if check_liveness:
            if record.lease_expires_at is None or record.lease_expires_at <= now:
                reasons.append("agent lease is expired; a fresh heartbeat is required")
            age = (now - record.last_seen_at).total_seconds()
            if age > self.policy.max_heartbeat_age_seconds:
                reasons.append(f"heartbeat is stale ({int(age)} seconds old)")
        reasons.extend(rollout_compatibility_reasons(self.policy, record))
        for actual, required, label in (
            (
                record.node_action_key_version,
                self.policy.required_node_action_key_version,
                "node action key version",
            ),
            (
                record.agent_version,
                self.policy.required_agent_version,
                "agent version",
            ),
            (
                record.policy_version,
                self.policy.required_policy_version,
                "policy version",
            ),
            (
                record.runtime_profile_version,
                self.policy.required_runtime_profile_version,
                "runtime profile",
            ),
        ):
            if required is not None and actual != required:
                reasons.append(f"{label} mismatch: expected {required}, got {actual}")
        missing = set(self.policy.required_operations) - set(record.allowed_operations)
        if missing:
            reasons.append(
                "missing operations: "
                + ",".join(sorted(item.value for item in missing))
            )
        return reasons

    def create_deployment(self, request: FleetDeploymentRequest) -> FleetDeployment:
        with self._lock:
            return self._create_deployment(request)

    _create_deployment = create_deployment_record

    def update_deployment_node(
        self,
        deployment_id: str,
        node_id: str,
        update: DeploymentNodeUpdate,
    ) -> FleetDeployment:
        with self._lock:
            return self._update_deployment_node(deployment_id, node_id, update)

    def _update_deployment_node(
        self,
        deployment_id: str,
        node_id: str,
        update: DeploymentNodeUpdate,
        *,
        from_heartbeat: bool = False,
    ) -> FleetDeployment:
        return update_fleet_deployment_node(
            self,
            deployment_id,
            node_id,
            update,
            from_heartbeat=from_heartbeat,
        )

    def start_next_wave(self, deployment_id: str) -> DeploymentWaveLease:
        with self._lock:
            return self._start_next_wave(deployment_id)

    def retry_failed_deployment(self, deployment_id: str) -> FleetDeployment:
        with self._lock:
            return self._retry_failed_deployment(deployment_id)

    def cancel_deployment(
        self,
        deployment_id: str,
        *,
        reason: str,
    ) -> FleetDeployment:
        with self._lock:
            return cancel_fleet_deployment(self, deployment_id, reason=reason)

    def _retry_failed_deployment(self, deployment_id: str) -> FleetDeployment:
        return retry_fleet_deployment(self, deployment_id)

    def _start_next_wave(self, deployment_id: str) -> DeploymentWaveLease:
        return start_fleet_wave(self, deployment_id)

    def _reconcile_deployments(self, record: AgentRecord) -> None:
        reconcile_fleet_deployments(self, record)

    def _reconcile_deployment(
        self,
        deployment: FleetDeployment,
        record: AgentRecord,
    ) -> None:
        reconcile_fleet_deployment(self, deployment, record)


class BarrierCoordinator:
    def __init__(self, store: ControlPlaneStore, *, now=None) -> None:
        self.store = store
        self.now = now or (lambda: datetime.now(timezone.utc))
        self._lock = RLock()

    def create(
        self,
        *,
        barrier_id: str,
        cluster_id: str,
        workflow_request_id: str,
        incident_id: str,
        fencing_token: int,
        operation: WorkflowOperation,
        generations: dict[str, int],
    ) -> MultiNodeBarrier:
        with self._lock:
            return self._create(
                barrier_id=barrier_id,
                cluster_id=cluster_id,
                workflow_request_id=workflow_request_id,
                incident_id=incident_id,
                fencing_token=fencing_token,
                operation=operation,
                generations=generations,
            )

    def _create(
        self,
        *,
        barrier_id: str,
        cluster_id: str,
        workflow_request_id: str,
        incident_id: str,
        fencing_token: int,
        operation: WorkflowOperation,
        generations: dict[str, int],
    ) -> MultiNodeBarrier:
        try:
            existing = self.store.get_barrier(barrier_id)
        except NotFoundError:
            pass
        else:
            expected = {
                "cluster_id": cluster_id,
                "workflow_request_id": workflow_request_id,
                "incident_id": incident_id,
                "fencing_token": fencing_token,
                "operation": operation,
                "node_ids": sorted(generations),
            }
            actual = {
                "cluster_id": existing.cluster_id,
                "workflow_request_id": (existing.workflow_request_id),
                "incident_id": existing.incident_id,
                "fencing_token": existing.fencing_token,
                "operation": existing.operation,
                "node_ids": sorted(item.node_id for item in existing.participants),
            }
            if actual != expected:
                raise ValueError("barrier id is already bound to a different contract")
            return existing
        now = self.now()
        barrier = MultiNodeBarrier(
            barrier_id=barrier_id,
            cluster_id=cluster_id,
            workflow_request_id=workflow_request_id,
            incident_id=incident_id,
            fencing_token=fencing_token,
            operation=operation,
            state=BarrierState.PREPARING,
            participants=[
                BarrierParticipant(
                    node_id=node_id,
                    agent_generation=generation,
                    updated_at=now,
                )
                for node_id, generation in sorted(generations.items())
            ],
            created_at=now,
            updated_at=now,
        )
        self.store.save_barrier(barrier)
        return barrier

    def record_prepare(
        self,
        barrier_id: str,
        node_id: str,
        *,
        details: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> MultiNodeBarrier:
        with self._lock:
            return self._record_prepare(
                barrier_id,
                node_id,
                details=details,
                error=error,
            )

    def _record_prepare(
        self,
        barrier_id: str,
        node_id: str,
        *,
        details: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> MultiNodeBarrier:
        barrier = self.store.get_barrier(barrier_id)
        if barrier.state not in {
            BarrierState.PREPARING,
            BarrierState.PREPARED,
        }:
            return barrier
        now = self.now()
        participants = self._update_participant(
            barrier,
            node_id,
            state=(
                BarrierParticipantState.FAILED
                if error
                else BarrierParticipantState.PREPARED
            ),
            prepare_details=details or {},
            error=error,
            updated_at=now,
        )
        states = {item.state for item in participants}
        state = (
            BarrierState.ABORTED
            if BarrierParticipantState.FAILED in states
            else BarrierState.PREPARED
            if states == {BarrierParticipantState.PREPARED}
            else BarrierState.PREPARING
        )
        return self._save(barrier, participants, state, now)

    def begin_commit(
        self,
        barrier_id: str,
        generations: dict[str, int],
    ) -> MultiNodeBarrier:
        with self._lock:
            return self._begin_commit(barrier_id, generations)

    def _begin_commit(
        self,
        barrier_id: str,
        generations: dict[str, int],
    ) -> MultiNodeBarrier:
        barrier = self.store.get_barrier(barrier_id)
        if barrier.state is not BarrierState.PREPARED:
            return barrier
        expected = {
            item.node_id: item.agent_generation for item in barrier.participants
        }
        if expected != generations:
            return self.abort(
                barrier_id,
                "agent generation changed after PREPARE",
            )
        now = self.now()
        barrier = barrier.model_copy(
            update={
                "state": BarrierState.COMMITTING,
                "updated_at": now,
            }
        )
        self.store.save_barrier(barrier)
        return barrier

    def record_commit(
        self,
        barrier_id: str,
        node_id: str,
        *,
        details: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> MultiNodeBarrier:
        with self._lock:
            return self._record_commit(
                barrier_id,
                node_id,
                details=details,
                error=error,
            )

    def _record_commit(
        self,
        barrier_id: str,
        node_id: str,
        *,
        details: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> MultiNodeBarrier:
        barrier = self.store.get_barrier(barrier_id)
        if barrier.state not in {
            BarrierState.COMMITTING,
            BarrierState.FAILED,
        }:
            return barrier
        now = self.now()
        participants = self._update_participant(
            barrier,
            node_id,
            state=(
                BarrierParticipantState.FAILED
                if error
                else BarrierParticipantState.COMMITTED
            ),
            commit_details=details or {},
            error=error,
            updated_at=now,
        )
        states = {item.state for item in participants}
        state = (
            BarrierState.FAILED
            if BarrierParticipantState.FAILED in states
            else BarrierState.COMMITTED
            if states == {BarrierParticipantState.COMMITTED}
            else BarrierState.COMMITTING
        )
        return self._save(barrier, participants, state, now)

    def abort(self, barrier_id: str, reason: str) -> MultiNodeBarrier:
        with self._lock:
            return self._abort(barrier_id, reason)

    def _abort(self, barrier_id: str, reason: str) -> MultiNodeBarrier:
        barrier = self.store.get_barrier(barrier_id)
        now = self.now()
        participants = [
            item.model_copy(
                update={
                    "error": item.error or reason,
                    "updated_at": now,
                }
            )
            for item in barrier.participants
        ]
        return self._save(barrier, participants, BarrierState.ABORTED, now)

    @staticmethod
    def _update_participant(
        barrier: MultiNodeBarrier,
        node_id: str,
        **updates,
    ) -> list[BarrierParticipant]:
        if node_id not in {item.node_id for item in barrier.participants}:
            raise ValueError("node is not a barrier participant")
        return [
            item.model_copy(update=updates) if item.node_id == node_id else item
            for item in barrier.participants
        ]

    def _save(
        self,
        barrier: MultiNodeBarrier,
        participants: list[BarrierParticipant],
        state: BarrierState,
        now: datetime,
    ) -> MultiNodeBarrier:
        barrier = barrier.model_copy(
            update={
                "participants": participants,
                "state": state,
                "updated_at": now,
            }
        )
        self.store.save_barrier(barrier)
        return barrier
