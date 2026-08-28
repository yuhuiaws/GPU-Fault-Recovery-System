from __future__ import annotations

import hashlib
import ipaddress
import json
import secrets
from datetime import datetime, timezone
from typing import Any

from pydantic import Field, field_validator, model_validator

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
from gpu_fault.remote_command_models import (
    RemoteCommandStatus as RemoteCommandStatus,
)
from gpu_fault.regional_compatibility import (
    LEGACY_REGIONAL_EXECUTOR_PROTOCOL_VERSION,
)
from gpu_fault.remote_command_models import (
    lease_deadline as lease_deadline,
)
from gpu_fault.store import NotFoundError
from gpu_fault.telemetry import EvidenceKind


class RegionalClusterRegistration(StrictModel):
    cluster_id: str
    region: str
    hyperpod_cluster_name: str
    eks_cluster_arn: str
    token_sha256: str = Field(min_length=64, max_length=64)
    enabled: bool = True
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
        if self.enabled and not self.agent_endpoint_allowed_cidrs:
            raise ValueError("enabled regional cluster requires agent endpoint CIDRs")
        return self

    def authenticates(self, token: str) -> bool:
        digest = hashlib.sha256(token.encode()).hexdigest()
        return self.enabled and secrets.compare_digest(digest, self.token_sha256)


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
        if context.workflow.blocked_reasons:
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
        restart_authorization = None
        restart_reservation: tuple[str, str, str] | None = None
        if context.step.operation is WorkflowOperation.RESTART_WORKLOAD:
            parameters = context.step.parameters
            required = {
                "cluster_id",
                "job_id",
                "source_attempt_id",
                "source_gpu_count",
                "restart_budget",
            }
            missing = required - set(parameters)
            if missing:
                return WorkflowStepOutcome.failed(
                    "restart safety context is missing: " + ", ".join(sorted(missing))
                )
            state, reserved = self.store.reserve_job_restart(
                str(parameters["cluster_id"]),
                str(parameters["job_id"]),
                int(parameters["restart_budget"]),
                context.idempotency_key,
            )
            if not reserved:
                return WorkflowStepOutcome.failed(
                    "restart budget exhausted for "
                    f"{state.cluster_id}/{state.job_id}: "
                    f"{state.restart_count}/{state.budget}",
                    details={
                        "reason": "RESTART_BUDGET_EXHAUSTED",
                        "restart_count": state.restart_count,
                        "restart_budget": state.budget,
                    },
                )
            restart_reservation = (
                state.cluster_id,
                state.job_id,
                context.idempotency_key,
            )
            restart_authorization = RestartAuthorization(
                cluster_id=state.cluster_id,
                job_id=state.job_id,
                source_attempt_id=str(parameters["source_attempt_id"]),
                source_gpu_count=int(parameters["source_gpu_count"]),
                restart_budget=state.budget,
                restart_count=state.restart_count,
                reservation_id=context.idempotency_key,
            )
        command = RemoteActionCommand(
            command_id=f"remote-{digest}",
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
            if restart_reservation is not None:
                self.store.release_job_restart(*restart_reservation)
            return WorkflowStepOutcome.failed(
                current.error or "remote cluster action failed",
                details=current.result_details,
            )
        return WorkflowStepOutcome.waiting(
            operation_id=operation_id,
            details={
                "remote_command_id": current.command_id,
                "remote_cluster_id": current.cluster_id,
                "remote_status": current.status.value,
                "mutation_submitted_by_control_plane": False,
            },
        )
