from __future__ import annotations

import hashlib
import ipaddress
import json
import secrets
from datetime import datetime, timedelta, timezone
from enum import StrEnum
from typing import Any

from pydantic import Field, field_validator, model_validator

import gpu_fault.execution.restart_budget_preflight as restart_budget_preflight
from gpu_fault.execution import (
    WorkflowStepContext,
    WorkflowStepOutcome,
)
from gpu_fault.hyperpod import HyperPodSubmissionRecord
from gpu_fault.models import (
    AdvisoryNotification,
    FaultIncident,
    RestartAuthorization,
    StrictModel,
    WorkflowOperation,
    WorkflowRequest,
    WorkflowStepSpec,
)
from gpu_fault.regional_compatibility import (
    LEGACY_REGIONAL_EXECUTOR_PROTOCOL_VERSION,
)
from gpu_fault.remote_command_models import (
    RemoteCommandStatus as RemoteCommandStatus,
)
from gpu_fault.remote_command_models import (
    lease_deadline as lease_deadline,
)
from gpu_fault.store import NotFoundError
from gpu_fault.store.shared.remote_helpers import workflow_step_space
from gpu_fault.telemetry import EvidenceKind

TOKEN_SLOT_CURRENT = "current"
TOKEN_SLOT_RETIRING = "retiring"

# A rotation window is an interval in which a withdrawn credential still opens
# the door, so it is bounded by construction rather than by operator discipline.
MAX_TOKEN_ROTATION_WINDOW = timedelta(days=7)

# Keys dropped from the registry content digest while unset. Registry revisions
# are durable and their digest is re-verified on every load, so adding a field
# to RegionalClusterRegistration would otherwise invalidate every revision
# published before the field existed. A revision that actually arms rotation
# carries the keys and therefore digests them.
_ROTATION_DIGEST_KEYS = ("retiring_token_sha256", "token_rotation_expires_at")


class RegionalClusterLifecycle(StrEnum):
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    FAILED = "FAILED"
    ROLLED_BACK = "ROLLED_BACK"
    DRAINING = "DRAINING"
    REVOKED = "REVOKED"


class RegionalClusterRegistration(StrictModel):
    cluster_id: str
    region: str
    hyperpod_cluster_name: str
    eks_cluster_arn: str
    token_sha256: str = Field(min_length=64, max_length=64)
    # The credential being retired, not the one being introduced. Rotation sets
    # `token_sha256` to the new token immediately and parks the old digest here
    # with a deadline, so the control plane accepts the new token before any
    # executor presents it and the deadline lapsing converges on the intended
    # end state instead of cutting off executors that already moved.
    retiring_token_sha256: str | None = Field(
        default=None, min_length=64, max_length=64
    )
    token_rotation_expires_at: datetime | None = None
    enabled: bool = True
    lifecycle_state: RegionalClusterLifecycle = RegionalClusterLifecycle.ACTIVE
    synthetic: bool = False
    synthetic_run_id: str | None = Field(default=None, min_length=1, max_length=128)
    synthetic_expires_at: datetime | None = None
    allowed_namespaces: list[str] = Field(default_factory=list)
    agent_endpoint_allowed_cidrs: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator(  # type: ignore[untyped-decorator]
        "agent_endpoint_allowed_cidrs"
    )
    @classmethod
    def validate_agent_endpoint_cidrs(cls, values: list[str]) -> list[str]:
        try:
            networks = {
                str(ipaddress.ip_network(value.strip(), strict=False))
                for value in values
                if value.strip()
            }
        except ValueError as exc:
            raise ValueError("agent endpoint CIDRs must be valid networks") from exc
        return sorted(networks)

    @model_validator(mode="after")
    def validate_token_digest(self) -> RegionalClusterRegistration:
        try:
            bytes.fromhex(self.token_sha256)
        except ValueError as exc:
            raise ValueError("token_sha256 must be hexadecimal") from exc
        self._validate_rotation()
        if self.enabled and not self.agent_endpoint_allowed_cidrs:
            raise ValueError("enabled regional cluster requires agent endpoint CIDRs")
        if self.synthetic:
            if self.synthetic_run_id is None or self.synthetic_expires_at is None:
                raise ValueError(
                    "synthetic regional cluster requires run ID and expiration"
                )
            if self.synthetic_expires_at.tzinfo is None:
                raise ValueError("synthetic expiration must include timezone")
        elif self.synthetic_run_id is not None or self.synthetic_expires_at is not None:
            raise ValueError(
                "non-synthetic regional cluster cannot declare synthetic metadata"
            )
        return self

    def is_active(self, now: datetime | None = None) -> bool:
        observed = now or datetime.now(timezone.utc)
        if observed.tzinfo is None:
            raise ValueError("regional cluster activity check requires timezone")
        return (
            self.enabled
            and self.lifecycle_state is RegionalClusterLifecycle.ACTIVE
            and (
                not self.synthetic
                or (
                    self.synthetic_expires_at is not None
                    and self.synthetic_expires_at > observed
                )
            )
        )

    def accepts_token(self, now: datetime | None = None) -> bool:
        observed = now or datetime.now(timezone.utc)
        if observed.tzinfo is None:
            raise ValueError("regional cluster activity check requires timezone")
        return (
            self.enabled
            and self.lifecycle_state
            not in {
                RegionalClusterLifecycle.REVOKED,
                RegionalClusterLifecycle.ROLLED_BACK,
            }
            and (
                not self.synthetic
                or (
                    self.synthetic_expires_at is not None
                    and self.synthetic_expires_at > observed
                )
            )
        )

    def _validate_rotation(self) -> None:
        digest = self.retiring_token_sha256
        deadline = self.token_rotation_expires_at
        if (digest is None) != (deadline is None):
            raise ValueError(
                "cluster token rotation requires both retiring_token_sha256 "
                "and token_rotation_expires_at"
            )
        if digest is None or deadline is None:
            return
        try:
            bytes.fromhex(digest)
        except ValueError as exc:
            raise ValueError("retiring_token_sha256 must be hexadecimal") from exc
        if secrets.compare_digest(digest, self.token_sha256):
            raise ValueError(
                "retiring_token_sha256 must differ from token_sha256; "
                "rotation has not issued a new token"
            )
        if deadline.tzinfo is None:
            raise ValueError("token_rotation_expires_at must include timezone")
        if self.updated_at.tzinfo is None:
            raise ValueError("token rotation requires a timezone-aware updated_at")
        # The window is measured against the stored `updated_at`, never against
        # the wall clock: a durable registry revision is re-validated every time
        # it is loaded, so a clock-dependent bound would make old revisions
        # unloadable and take the control plane down long after the fact.
        if deadline <= self.updated_at:
            raise ValueError(
                "token_rotation_expires_at must be after updated_at; drop the "
                "retiring token instead of recording an expired window"
            )
        if deadline - self.updated_at > MAX_TOKEN_ROTATION_WINDOW:
            raise ValueError(
                "cluster token rotation window must not exceed "
                f"{MAX_TOKEN_ROTATION_WINDOW.days} days"
            )

    def accepted_token_slots(self, now: datetime | None = None) -> tuple[str, ...]:
        """Name the credential slots that can authenticate this cluster now."""

        observed = now or datetime.now(timezone.utc)
        if observed.tzinfo is None:
            raise ValueError("regional cluster activity check requires timezone")
        if (
            self.retiring_token_sha256 is None
            or self.token_rotation_expires_at is None
            or self.token_rotation_expires_at <= observed
        ):
            return (TOKEN_SLOT_CURRENT,)
        return (TOKEN_SLOT_CURRENT, TOKEN_SLOT_RETIRING)

    def _matched_slot(self, token: str, observed: datetime) -> str | None:
        digest = hashlib.sha256(token.encode()).hexdigest()
        candidates = {
            TOKEN_SLOT_CURRENT: self.token_sha256,
            TOKEN_SLOT_RETIRING: self.retiring_token_sha256,
        }
        matched: str | None = None
        for slot in self.accepted_token_slots(observed):
            candidate = candidates[slot]
            if candidate is None:
                continue
            # Every accepted slot is compared instead of returning on the first
            # hit: short-circuiting would let the response time say which token
            # the caller holds. How many slots are live is already visible to
            # execution-token holders, so only the ordering is hidden here.
            if secrets.compare_digest(digest, candidate):
                matched = slot
        return matched

    def matched_token_slot(
        self,
        token: str,
        now: datetime | None = None,
    ) -> str | None:
        """Return which credential slot ``token`` matched, or ``None``.

        Callers use the slot name to report that an executor is still holding
        the retiring token: while that is reported, the rotation is not finished
        and the retiring digest must not be dropped yet.
        """

        observed = now or datetime.now(timezone.utc)
        if not self.accepts_token(observed):
            return None
        return self._matched_slot(token, observed)

    def token_matches(self, token: str, now: datetime | None = None) -> bool:
        return self.matched_token_slot(token, now) is not None

    def authenticates(self, token: str, now: datetime | None = None) -> bool:
        observed = now or datetime.now(timezone.utc)
        return (
            self.is_active(observed) and self._matched_slot(token, observed) is not None
        )


def _registry_digest_payload(
    registration: RegionalClusterRegistration,
) -> dict[str, Any]:
    payload = registration.model_dump(mode="json")
    for key in _ROTATION_DIGEST_KEYS:
        if payload.get(key) is None:
            payload.pop(key, None)
    return payload


def regional_registry_content_sha256(
    registrations: list[RegionalClusterRegistration],
) -> str:
    payload = [
        _registry_digest_payload(item)
        for item in sorted(registrations, key=lambda value: value.cluster_id)
    ]
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


class RegionalRegistryRevision(StrictModel):
    generation: int = Field(ge=1)
    content_sha256: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    registrations: list[RegionalClusterRegistration] = Field(default_factory=list)
    previous_generation: int | None = Field(default=None, ge=1)
    required_member_ids: list[str] = Field(default_factory=list)
    reason: str = Field(min_length=1, max_length=512)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_revision(self) -> RegionalRegistryRevision:
        cluster_ids = [item.cluster_id for item in self.registrations]
        if len(cluster_ids) != len(set(cluster_ids)):
            raise ValueError("regional registry cluster IDs must be unique")
        if len(self.required_member_ids) != len(set(self.required_member_ids)):
            raise ValueError("regional registry required member IDs must be unique")
        if (
            self.previous_generation is not None
            and self.previous_generation >= self.generation
        ):
            raise ValueError("regional registry generation must increase")
        expected = regional_registry_content_sha256(self.registrations)
        if not secrets.compare_digest(self.content_sha256, expected):
            raise ValueError("regional registry content digest mismatch")
        return self

    @classmethod
    def build(
        cls,
        *,
        generation: int,
        registrations: list[RegionalClusterRegistration],
        previous_generation: int | None,
        required_member_ids: list[str],
        reason: str,
        created_at: datetime | None = None,
    ) -> RegionalRegistryRevision:
        normalized = sorted(registrations, key=lambda item: item.cluster_id)
        return cls(
            generation=generation,
            content_sha256=regional_registry_content_sha256(normalized),
            registrations=normalized,
            previous_generation=previous_generation,
            required_member_ids=sorted(set(required_member_ids)),
            reason=reason,
            created_at=created_at or datetime.now(timezone.utc),
        )


class RegionalRegistryHead(StrictModel):
    generation: int = Field(ge=1)
    content_sha256: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RegionalRegistryMember(StrictModel):
    member_id: str = Field(min_length=1, max_length=256)
    service_role: str = Field(min_length=1, max_length=64)
    release_id: str = Field(min_length=1, max_length=256)
    generation: int = Field(ge=0)
    content_sha256: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    ready: bool
    error: str | None = Field(default=None, max_length=1024)
    started_at: datetime
    last_seen_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RegionalRegistryPublishRequest(StrictModel):
    expected_generation: int = Field(ge=0)
    registrations: list[RegionalClusterRegistration] = Field(default_factory=list)
    required_member_ids: list[str] | None = None
    reason: str = Field(min_length=1, max_length=512)


class RegionalRegistryClusterTransitionRequest(StrictModel):
    registration: RegionalClusterRegistration
    lifecycle_state: RegionalClusterLifecycle
    reason: str = Field(min_length=1, max_length=512)


class RegionalRegistryRollbackRequest(StrictModel):
    expected_generation: int = Field(ge=1)
    target_generation: int = Field(ge=1)
    reason: str = Field(min_length=1, max_length=512)


class RegionalRegistryStatus(StrictModel):
    generation: int = Field(ge=1)
    content_sha256: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    cluster_states: dict[str, RegionalClusterLifecycle]
    required_member_ids: list[str]
    acked_member_ids: list[str]
    missing_member_ids: list[str]
    active_member_ids: list[str]
    members: list[RegionalRegistryMember]
    converged: bool


def cluster_token_sha256(token: str) -> str:
    if len(token) < 32:
        raise ValueError("cluster token must contain at least 32 characters")
    return hashlib.sha256(token.encode()).hexdigest()


class RemoteActionCommand(StrictModel):
    command_id: str
    cluster_id: str
    workflow_request_id: str
    incident_id: str
    step_index: int = Field(ge=0)
    fencing_token: int = Field(ge=1)
    idempotency_key: str
    step: WorkflowStepSpec
    workflow: WorkflowRequest
    incident: FaultIncident
    restart_authorization: RestartAuthorization | None = None
    status: RemoteCommandStatus = RemoteCommandStatus.PENDING
    lease_owner: str | None = None
    last_lease_owner: str | None = None
    lease_token: str | None = None
    lease_expires_at: datetime | None = None
    cancellation_requested_at: datetime | None = None
    cancellation_reason: str | None = None
    result_details: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    status_source: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RemoteCommandClaimRequest(StrictModel):
    executor_id: str
    executor_protocol_version: int = Field(
        default=LEGACY_REGIONAL_EXECUTOR_PROTOCOL_VERSION,
        ge=1,
    )
    executor_artifact_sha256: str | None = Field(
        default=None,
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    executor_compatibility_digest: str | None = Field(
        default=None,
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    execution_owners: list[str] = Field(default_factory=list, max_length=32)
    max_commands: int = Field(default=1, ge=1, le=25)
    lease_seconds: int = Field(default=60, ge=10, le=7200)

    @model_validator(mode="after")
    def validate_execution_owners(
        self,
    ) -> RemoteCommandClaimRequest:
        if any(
            not owner.strip() or owner != owner.strip()
            for owner in self.execution_owners
        ):
            raise ValueError("execution_owners must contain non-empty trimmed values")
        if len(set(self.execution_owners)) != len(self.execution_owners):
            raise ValueError("execution_owners must be unique")
        return self


class RemoteCommandClaim(StrictModel):
    commands: list[RemoteActionCommand] = Field(default_factory=list)


class RegionalExecutorReadinessRequest(StrictModel):
    """What an executor claims about itself when probing readiness.

    The Kubernetes readiness probe used to be an anonymous GET /healthz
    against the control plane, which proves only that some control plane
    somewhere answers. It cannot fail when the executor's own token is
    wrong, when its trust chain is broken, or when it advertises fewer
    execution owners than the queued steps need -- all of which leave
    the Pod Ready and claiming nothing forever.
    """

    executor_id: str = Field(min_length=1)
    executor_protocol_version: int = Field(
        default=LEGACY_REGIONAL_EXECUTOR_PROTOCOL_VERSION,
        ge=1,
    )
    executor_artifact_sha256: str | None = Field(
        default=None,
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    executor_compatibility_digest: str | None = Field(
        default=None,
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    execution_owners: list[str] = Field(default_factory=list, max_length=32)
    # None means "this executor has not completed a claim cycle yet",
    # which is only acceptable before the first poll interval elapses.
    last_successful_claim_age_seconds: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_execution_owners(
        self,
    ) -> RegionalExecutorReadinessRequest:
        if any(
            not owner.strip() or owner != owner.strip()
            for owner in self.execution_owners
        ):
            raise ValueError("execution_owners must contain non-empty trimmed values")
        if len(set(self.execution_owners)) != len(self.execution_owners):
            raise ValueError("execution_owners must be unique")
        return self


class RegionalExecutorReadinessReport(StrictModel):
    ready: bool
    cluster_id: str
    executor_id: str
    executor_protocol_version: int = Field(ge=1)
    executor_artifact_sha256: str | None = None
    executor_compatibility_digest: str | None = None
    registered: bool
    execution_owners: list[str] = Field(default_factory=list)
    # Owners the open backlog needs that this executor did not advertise.
    # Non-empty means those commands can never be claimed here.
    unsupported_execution_owners: list[str] = Field(default_factory=list)
    open_commands: int = 0
    pending_commands: int = 0
    oldest_unclaimed_age_seconds: float = 0.0
    last_successful_claim_age_seconds: float | None = None
    reasons: list[str] = Field(default_factory=list)


class RemoteCommandLeaseRenewal(StrictModel):
    executor_id: str = Field(min_length=1)
    lease_token: str = Field(min_length=1)
    lease_seconds: int = Field(default=60, ge=10, le=7200)


class RemoteCommandResult(StrictModel):
    lease_token: str
    status: RemoteCommandStatus
    # Distinguishes a refused action from an executor-side defect so the
    # control plane and operators do not read "AttributeError" as a
    # legitimate recovery failure.
    status_source: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None

    @model_validator(mode="after")
    def validate_terminal_status(self) -> RemoteCommandResult:
        if self.status not in {
            RemoteCommandStatus.WAITING,
            RemoteCommandStatus.SUCCEEDED,
            RemoteCommandStatus.FAILED,
        }:
            raise ValueError("remote result must be WAITING, SUCCEEDED, or FAILED")
        if self.status is RemoteCommandStatus.FAILED and not self.error:
            raise ValueError("failed remote result requires error")
        return self


class RemoteHyperPodSubmissionRequest(StrictModel):
    """Executor-side request to reserve or record a HyperPod submission.

    The regional executor runs with GPU_FAULT_CLUSTER_EXECUTOR_REMOTE_
    STATE=true and therefore has no database of its own, so provider
    submission idempotency has to be held by the control plane. Without
    it the record lives only in the executor process, which reboot and
    replace routinely restart -- exactly the case it must survive.
    """

    cluster_id: str
    record: HyperPodSubmissionRecord


class RemoteHyperPodSubmissionReservation(StrictModel):
    reserved: bool
    record: HyperPodSubmissionRecord


class RemoteSpareHealthRequest(StrictModel):
    cluster_id: str
    node_aliases: list[str] = Field(min_length=1)
    incident_id: str | None = None
    observed_after: datetime | None = None


class RemoteSpareHealthReport(StrictModel):
    ready: bool
    reasons: list[str] = Field(default_factory=list)


class RemoteIncidentOwnershipReport(StrictModel):
    """Whether a node's previous owning incident is finished.

    The Kubernetes adapter refuses to isolate a node that still carries
    another incident's ownership annotations, and only takes it over once
    that incident's workflow reached a terminal status. In the default
    regional deployment the executor owns no store
    (GPU_FAULT_CLUSTER_EXECUTOR_REMOTE_STATE=true), so the takeover check
    could never be answered locally and every node left annotated by a
    failed workflow stayed permanently unusable. This is that answer,
    computed by the control plane for the executor's own cluster only.
    """

    incident_id: str
    known: bool = False
    workflow_request_id: str | None = None
    workflow_status: str | None = None
    terminal: bool = False
    incident_state: str | None = None
    quarantine_hold: bool = False


class RemoteFleetRolloutFence(StrictModel):
    """Which fleet rollouts currently fence destructive work on one cluster.

    The verdict, not the evidence. The executor only needs to know whether to
    hold, so shipping the ids keeps the answer bounded and keeps the
    supersession rule on the control plane, where the store is: the data plane
    re-deriving it from a deployment list would be a second copy of a safety
    predicate, free to drift. Cross-cluster records never leave the control
    plane, because the caller is authenticated as exactly one cluster.
    """

    cluster_id: str
    fencing_deployment_ids: list[str] = Field(default_factory=list)


class RemoteAdvisoryNotificationRequest(StrictModel):
    """Lets the data-plane executor page an operator.

    The warm-spare coordinator is the only supported node replacement
    path, so when it cannot find capacity the operator MUST be told --
    the workflow stays blocked and the fault node stays isolated until
    someone acts. In the default regional deployment the executor owns
    no store, so the notification has to be persisted by the control
    plane on its behalf.
    """

    cluster_id: str
    notification: AdvisoryNotification


class RemoteEvidenceCaptureRequest(StrictModel):
    cluster_id: str
    record_id: str
    node_id: str
    kind: EvidenceKind
    observed_at: datetime
    attempt_ids: list[str] = Field(default_factory=list)
    payload: dict[str, Any]


class RegionalRemoteWorkflowAdapter:
    """Delegates cluster mutations to a cluster-scoped pull executor."""

    def __init__(
        self,
        store,
        *,
        owners: set[str],
        operations: set[WorkflowOperation] | None = None,
    ) -> None:
        self.store = store
        self.owners = owners
        self.operations = operations or set(WorkflowOperation)
        # How many dispatches were held because another command for the same
        # (workflow, step index, step space) was still open (item D5). Read by
        # the metrics family like the dispatcher's counters.
        self.open_sibling_holds_total = 0

    def supports(self, step: WorkflowStepSpec) -> bool:
        return step.execution_owner in self.owners and step.operation in self.operations

    def execute(self, context: WorkflowStepContext) -> WorkflowStepOutcome:
        try:
            registration = self.store.get_regional_cluster(context.incident.cluster_id)
        except NotFoundError:
            return WorkflowStepOutcome.failed(
                "incident cluster is not registered for remote execution"
            )
        allowed_namespaces = set(registration.allowed_namespaces)
        if context.step.workload_ids and not allowed_namespaces:
            return WorkflowStepOutcome.failed(
                "regional cluster registration has no allowed workload namespaces"
            )
        for workload_id in context.step.workload_ids:
            namespace = workload_id.split("/", 1)[0]
            if allowed_namespaces and namespace not in allowed_namespaces:
                return WorkflowStepOutcome.failed(
                    "workflow targets a namespace outside the cluster "
                    f"registration: {namespace}"
                )
        # The command_id covers action semantics, not DAG scheduling
        # metadata. A health branch may rewrite branch/dependency fields
        # while a physical command is in flight; treating that rewrite
        # as a new command can submit the same mutation twice. Target,
        # workload and parameter changes remain part of the identity.
        identity_step = context.step.model_dump(mode="json")
        identity_step["branch_id"] = None
        identity_step["depends_on_step_indexes"] = []
        if context.workflow.executes_safety_steps:
            identity_step["command_step_space"] = "safety"
        digest = hashlib.sha256(
            "\x1f".join(
                (
                    context.workflow.request_id,
                    str(context.step_index),
                    str(context.workflow.fencing_token),
                    context.idempotency_key,
                    json.dumps(
                        identity_step,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                )
            ).encode()
        ).hexdigest()[:24]
        command_id = f"remote-{digest}"
        # The open-command invariant (architecture review 2026-09-07, item D5):
        # the digest above changes whenever a merge rewrites the step's targets
        # or parameters, but the physical action the previous digest named may
        # still be executing on the node. One open command per step identity
        # within a generation: while a sibling with another id is PENDING,
        # LEASED or WAITING under this fencing token, hold rather than mint a
        # second command. An older generation's command is the generation
        # fence's business -- the claim and completion paths already refuse it.
        sibling = self.store.find_open_remote_command(
            context.workflow.request_id,
            context.step_index,
            workflow_step_space(context.workflow),
            exclude_command_id=command_id,
        )
        if (
            sibling is not None
            and sibling.fencing_token == context.workflow.fencing_token
        ):
            self.open_sibling_holds_total += 1
            return WorkflowStepOutcome.waiting(
                operation_id=f"remote/{sibling.command_id}",
                details={
                    "reason": "OPEN_SIBLING_COMMAND",
                    "remote_command_id": sibling.command_id,
                    "remote_cluster_id": sibling.cluster_id,
                    "remote_status": sibling.status.value,
                    "held_command_id": command_id,
                    "mutation_submitted_by_control_plane": False,
                },
            )
        current: RemoteActionCommand | None = None
        restart_authorization: RestartAuthorization | None = None
        if context.step.operation is WorkflowOperation.RESTART_WORKLOAD:
            # A command under this id that already SUCCEEDED or FAILED has its
            # verdict, whatever happened to the restart reservation since (a
            # terminal write, an operator restore). The reservation gate is
            # for minting a command, not for reading one back.
            current = self._settled_remote_command(command_id)
            if current is None:
                # The preflight is the only site that reserves restart budget;
                # dispatch reads that reservation back and signs it for the
                # data plane. A step without one fails closed instead of
                # reserving here.
                issued = restart_budget_preflight.issue_restart_authorization(
                    self.store,
                    context.incident,
                    context.step,
                    context.idempotency_key,
                )
                if not isinstance(issued, RestartAuthorization):
                    return issued
                restart_authorization = issued
        if current is None:
            command = RemoteActionCommand(
                command_id=command_id,
                cluster_id=context.incident.cluster_id,
                workflow_request_id=context.workflow.request_id,
                incident_id=context.incident.incident_id,
                step_index=context.step_index,
                fencing_token=context.workflow.fencing_token,
                idempotency_key=context.idempotency_key,
                step=context.step,
                workflow=context.workflow,
                incident=context.incident,
                restart_authorization=restart_authorization,
            )
            current = self.store.ensure_remote_command(command)
        operation_id = f"remote/{current.command_id}"
        if current.status is RemoteCommandStatus.SUCCEEDED:
            return WorkflowStepOutcome.succeeded(
                operation_id=operation_id,
                details=current.result_details,
            )
        if current.status is RemoteCommandStatus.FAILED:
            # ``status_source`` rides along so the executor can tell a command
            # the workflow itself cancelled (``workflow-preempted``,
            # ``workflow-timeout``) from a node refusing the action (D-8).
            return WorkflowStepOutcome.failed(
                current.error or "remote cluster action failed",
                details={
                    **current.result_details,
                    **(
                        {"remote_status_source": current.status_source}
                        if current.status_source is not None
                        else {}
                    ),
                },
            )
        # A RESTART_WORKLOAD hold on the data plane says ``restart_submitted``;
        # carried here so the reservation release can judge a live wait.
        restart_submitted = current.result_details.get("restart_submitted")
        return WorkflowStepOutcome.waiting(
            operation_id=operation_id,
            details={
                "remote_command_id": current.command_id,
                "remote_cluster_id": current.cluster_id,
                "remote_status": current.status.value,
                "mutation_submitted_by_control_plane": False,
                **(
                    {"restart_submitted": restart_submitted}
                    if restart_submitted is not None
                    else {}
                ),
            },
        )

    def _settled_remote_command(self, command_id: str) -> RemoteActionCommand | None:
        """The SUCCEEDED or FAILED command already stored under ``command_id``.

        ``None`` when there is no command yet or it is still open; an open
        command goes through ``ensure_remote_command`` like a new one.
        """

        try:
            existing: RemoteActionCommand = self.store.get_remote_command(command_id)
        except NotFoundError:
            return None
        if existing.status in (
            RemoteCommandStatus.SUCCEEDED,
            RemoteCommandStatus.FAILED,
        ):
            return existing
        return None
