from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=False)


class Environment(StrEnum):
    """Environments the control plane knows how to reason about.

    Only these four appear: every recovery decision that consults the environment
    branches on one of them. A member nobody branches on is worse than a missing
    one -- a profile or watcher configured with it would load, then silently take
    the default path -- so bare ``slurm`` and ``ec2`` are deliberately absent and
    are rejected at parse time.
    """

    EKS = "eks"
    HYPERPOD_EKS = "hyperpod-eks"
    HYPERPOD_SLURM = "hyperpod-slurm"
    KUBERNETES = "kubernetes"


class TerminalStatus(StrEnum):
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    STOPPED = "STOPPED"
    TIMED_OUT = "TIMED_OUT"


class Severity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    FATAL = "fatal"


class RecoveryAction(StrEnum):
    NO_ACTION = "NO_ACTION"
    MARK_UNSCHEDULABLE = "MARK_UNSCHEDULABLE"
    DRAIN = "DRAIN"
    STOP_WORKLOAD = "STOP_WORKLOAD"
    COLLECT_EVIDENCE = "COLLECT_EVIDENCE"
    RESTART_WORKLOAD = "RESTART_WORKLOAD"
    RESET_GPU = "RESET_GPU"
    REBOOT_NODE = "REBOOT_NODE"
    REPLACE_NODE = "REPLACE_NODE"
    REMEDIATE_EFA_DRIVER = "REMEDIATE_EFA_DRIVER"
    RESTART_EFA_DEVICE_PLUGIN = "RESTART_EFA_DEVICE_PLUGIN"
    RESTART_GPU_DEVICE_PLUGIN = "RESTART_GPU_DEVICE_PLUGIN"
    RUN_DIAGNOSTICS = "RUN_DIAGNOSTICS"
    VALIDATE_NODE = "VALIDATE_NODE"
    RESTORE_SCHEDULING = "RESTORE_SCHEDULING"
    QUARANTINE = "QUARANTINE"
    ESCALATE_OPERATOR = "ESCALATE_OPERATOR"


RECOVERY_ACTION_RANK: dict[RecoveryAction, int] = {
    RecoveryAction.NO_ACTION: 0,
    RecoveryAction.RESTORE_SCHEDULING: 0,
    RecoveryAction.COLLECT_EVIDENCE: 5,
    RecoveryAction.VALIDATE_NODE: 5,
    RecoveryAction.RUN_DIAGNOSTICS: 10,
    RecoveryAction.RESTART_EFA_DEVICE_PLUGIN: 15,
    RecoveryAction.RESTART_GPU_DEVICE_PLUGIN: 15,
    RecoveryAction.RESTART_WORKLOAD: 20,
    RecoveryAction.MARK_UNSCHEDULABLE: 25,
    RecoveryAction.STOP_WORKLOAD: 25,
    RecoveryAction.RESET_GPU: 30,
    RecoveryAction.REMEDIATE_EFA_DRIVER: 35,
    RecoveryAction.REBOOT_NODE: 50,
    RecoveryAction.DRAIN: 60,
    RecoveryAction.QUARANTINE: 60,
    RecoveryAction.REPLACE_NODE: 70,
    RecoveryAction.ESCALATE_OPERATOR: 80,
}


def recovery_action_sort_key(
    action: RecoveryAction,
) -> tuple[int, str]:
    return RECOVERY_ACTION_RANK[action], action.value


class TriageOutcome(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    INCONCLUSIVE = "INCONCLUSIVE"


class DecisionStatus(StrEnum):
    NO_ACTION = "NO_ACTION"
    PENDING_TRIAGE = "PENDING_TRIAGE"
    PLAN_CREATED = "PLAN_CREATED"


class PlanStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    # A plan whose workflow was replaced by a later generation, or is held for
    # an operator, did not fail; recording it as FAILED alarmed on cleanup
    # and contradicted the dispatcher's own reporting rule (P1-71D).
    SUPERSEDED = "SUPERSEDED"
    BLOCKED = "BLOCKED"


class WorkloadState(StrEnum):
    UNKNOWN = "UNKNOWN"
    IDLE = "IDLE"
    ACTIVE = "ACTIVE"


class XidMetricBaseline(StrictModel):
    cluster_id: str
    node_id: str
    gpu_key: str
    xid: int = Field(ge=0)
    observed_at: datetime


class HealthSignalState(StrictModel):
    signal_key: str
    active: bool
    observed_at: datetime
    active_since: datetime | None = None
    notified: bool | None = None
    # Control-plane time of the last accepted sample (F-M2). Durations and
    # ordering run on it when present; ``observed_at`` is the node's clock.
    clock_at: datetime | None = None


class EfaTrafficSignal(StrEnum):
    WARMUP = "WARMUP"
    NORMAL = "NORMAL"
    SPIKE = "SPIKE"
    DROP = "DROP"
    ZERO_PENDING = "ZERO_PENDING"
    ZERO_WARNING = "ZERO_WARNING"
    HUNG_SUSPECTED = "HUNG_SUSPECTED"
    RECOVERED = "RECOVERED"


class EfaTrafficAdminAction(StrEnum):
    ACKNOWLEDGE_TRANSIENT = "ACKNOWLEDGE_TRANSIENT"
    ACCEPT_NEW_BASELINE = "ACCEPT_NEW_BASELINE"


class EfaTrafficState(StrictModel):
    state_key: str
    cluster_id: str
    node_id: str
    job_id: str
    attempt_id: str
    first_observed_at: datetime
    observed_at: datetime
    baseline_bytes_per_second: float | None = Field(default=None, ge=0)
    last_bytes_per_second: float = Field(ge=0)
    had_active_traffic: bool = False
    zero_since: datetime | None = None
    zero_first_seen_at: datetime | None = None
    consecutive_zero_samples: int = Field(default=0, ge=0)
    signal: EfaTrafficSignal = EfaTrafficSignal.WARMUP
    active_spike_event_id: str | None = None
    spike_acknowledged_at: datetime | None = None
    spike_acknowledged_by: str | None = None
    spike_acknowledgement_reason: str | None = None


class EfaTrafficAdminRequest(StrictModel):
    cluster_id: str = Field(min_length=1)
    node_id: str = Field(min_length=1)
    job_id: str = Field(min_length=1)
    attempt_id: str = Field(min_length=1)
    event_id: str = Field(min_length=1)
    action: EfaTrafficAdminAction
    operator: str = Field(min_length=1, max_length=256)
    reason: str = Field(min_length=1, max_length=2000)


class EfaTrafficAdminDecision(StrictModel):
    decision_id: str
    cluster_id: str
    node_id: str
    job_id: str
    attempt_id: str
    event_id: str
    action: EfaTrafficAdminAction
    operator: str
    reason: str
    decided_at: datetime
    previous_signal: EfaTrafficSignal
    resulting_signal: EfaTrafficSignal
    previous_baseline_bytes_per_second: float | None = Field(default=None, ge=0)
    resulting_baseline_bytes_per_second: float | None = Field(default=None, ge=0)
    accepted_sample_bytes_per_second: float | None = Field(default=None, ge=0)


class IncidentState(StrEnum):
    DETECTED = "DETECTED"
    ACTION_PENDING = "ACTION_PENDING"
    SAFETY_PENDING = "SAFETY_PENDING"
    QUARANTINED = "QUARANTINED"
    RECOVERED = "RECOVERED"
    ESCALATED = "ESCALATED"


class WorkflowStatus(StrEnum):
    PENDING = "PENDING"
    SAFETY_PENDING = "SAFETY_PENDING"
    BLOCKED = "BLOCKED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    SUPERSEDED = "SUPERSEDED"


# The statuses the dispatcher scans and the executor accepts. Everything else
# (SUCCEEDED, FAILED, SUPERSEDED and BLOCKED) will never execute again on its
# own, so nothing may be merged into such a record (F-B4).
EXECUTABLE_WORKFLOW_STATUSES = frozenset(
    {
        WorkflowStatus.PENDING,
        WorkflowStatus.SAFETY_PENDING,
        WorkflowStatus.RUNNING,
    }
)


class BlockedKind(StrEnum):
    """Why a workflow is BLOCKED; set wherever the status is written (F-B4).

    ``SAFETY_SETTLED``: the safety steps ran to completion, the plan ended the
    way it was meant to. ``NEEDS_OPERATOR``: compilation left something an
    operator has to decide (policy block, no executable owner, ...).
    ``INTERNAL_ERROR``: the dispatcher hit an error that proves the record
    itself cannot be executed.
    """

    SAFETY_SETTLED = "SAFETY_SETTLED"
    NEEDS_OPERATOR = "NEEDS_OPERATOR"
    INTERNAL_ERROR = "INTERNAL_ERROR"


# BLOCKED rows that still hold their node and their successors: an operator
# has to act before anything else may touch what they touched (F-A4). A
# settled safety plan releases both; a legacy row without a kind keeps the
# pre-F-B4 semantics (released) so nothing written before the field existed
# can pin a successor forever.
OCCUPYING_BLOCKED_KINDS = frozenset(
    {BlockedKind.NEEDS_OPERATOR, BlockedKind.INTERNAL_ERROR}
)


def workflow_is_open(status: WorkflowStatus, blocked_kind: BlockedKind | None) -> bool:
    """Whether a workflow in ``status`` still gates successors and node use."""

    if status in EXECUTABLE_WORKFLOW_STATUSES:
        return True
    return status is WorkflowStatus.BLOCKED and blocked_kind in OCCUPYING_BLOCKED_KINDS


class WorkflowStepStatus(StrEnum):
    SUCCEEDED = "SUCCEEDED"
    WAITING = "WAITING"
    FAILED = "FAILED"


class WorkflowOperation(StrEnum):
    FREEZE_EVIDENCE = "FREEZE_EVIDENCE"
    COLLECT_HUNG_TRIAGE = "COLLECT_HUNG_TRIAGE"
    COLLECT_DIAGNOSTIC_BUNDLE = "COLLECT_DIAGNOSTIC_BUNDLE"
    RUN_DCGM_DIAGNOSTIC = "RUN_DCGM_DIAGNOSTIC"
    MARK_UNSCHEDULABLE = "MARK_UNSCHEDULABLE"
    CHECKPOINT_WORKLOADS = "CHECKPOINT_WORKLOADS"
    STOP_WORKLOADS = "STOP_WORKLOADS"
    QUIESCE_GPU_SERVICES = "QUIESCE_GPU_SERVICES"
    VERIFY_NO_GPU_CLIENTS = "VERIFY_NO_GPU_CLIENTS"
    TRIGGER_HEALTH_SNAPSHOT = "TRIGGER_HEALTH_SNAPSHOT"
    RESET_GPU = "RESET_GPU"
    RESTORE_GPU_SERVICES = "RESTORE_GPU_SERVICES"
    RESTART_NODE = "RESTART_NODE"
    REPLACE_NODE = "REPLACE_NODE"
    RESTART_VM = "RESTART_VM"
    RESTART_FABRIC_MANAGER = "RESTART_FABRIC_MANAGER"
    REMEDIATE_EFA_DRIVER = "REMEDIATE_EFA_DRIVER"
    RESTART_EFA_DEVICE_PLUGIN = "RESTART_EFA_DEVICE_PLUGIN"
    RESTART_GPU_DEVICE_PLUGIN = "RESTART_GPU_DEVICE_PLUGIN"
    RUN_FIELD_DIAGNOSTIC = "RUN_FIELD_DIAGNOSTIC"
    RUN_NVLINK74_WORKFLOW = "RUN_NVLINK74_WORKFLOW"
    RESET_ALL_GPUS_NVSWITCHES = "RESET_ALL_GPUS_NVSWITCHES"
    CHECK_MECHANICALS = "CHECK_MECHANICALS"
    REMEDIATE_DRIVER = "REMEDIATE_DRIVER"
    UPDATE_SOFTWARE_FIRMWARE = "UPDATE_SOFTWARE_FIRMWARE"
    ESCALATE_SUPPORT = "ESCALATE_SUPPORT"
    VALIDATE_GPU = "VALIDATE_GPU"
    VALIDATE_HOST = "VALIDATE_HOST"
    VALIDATE_FABRIC = "VALIDATE_FABRIC"
    RESTORE_SCHEDULING = "RESTORE_SCHEDULING"
    RESTART_WORKLOAD = "RESTART_WORKLOAD"
    QUARANTINE = "QUARANTINE"


class CapabilityMode(StrEnum):
    OWN = "OWN"
    DELEGATE = "DELEGATE"
    AUGMENT = "AUGMENT"
    OBSERVE = "OBSERVE"
    DISABLED = "DISABLED"


class CapabilityName(StrEnum):
    GPU_DETECTION = "gpuDetection"
    HOST_NETWORK_DETECTION = "hostNetworkDetection"
    NODE_MARKER_WRITER = "nodeMarkerWriter"
    SCHEDULER_DRAIN = "schedulerDrain"
    GPU_RESET = "gpuReset"
    NODE_REBOOT = "nodeReboot"
    NODE_REPLACE = "nodeReplace"
    DEEP_DIAGNOSTICS = "deepDiagnostics"
    WORKLOAD_STOP = "workloadStop"
    WORKLOAD_RESTART = "workloadRestart"
    CHECKPOINT_RESTORE = "checkpointRestore"
    EVIDENCE_CAPTURE = "evidenceCapture"
    DIAGNOSTIC_BUNDLE_CAPTURE = "diagnosticBundleCapture"
    VM_RESTART = "vmRestart"
    FABRIC_MANAGER_RESTART = "fabricManagerRestart"
    NVLINK_DIAGNOSTICS = "nvlinkDiagnostics"
    MEMORY_DIAGNOSTICS = "memoryDiagnostics"
    FABRIC_RESET = "fabricReset"
    MECHANICAL_INSPECTION = "mechanicalInspection"
    DRIVER_REMEDIATION = "driverRemediation"
    EFA_DRIVER_REMEDIATION = "efaDriverRemediation"
    SOFTWARE_FIRMWARE_UPDATE = "softwareFirmwareUpdate"
    SUPPORT_ESCALATION = "supportEscalation"
    PROVIDER_HEALTH_ISOLATION = "providerHealthIsolation"


DESTRUCTIVE_CAPABILITIES = frozenset(
    {
        CapabilityName.SCHEDULER_DRAIN,
        CapabilityName.GPU_RESET,
        CapabilityName.NODE_REBOOT,
        CapabilityName.NODE_REPLACE,
        CapabilityName.WORKLOAD_STOP,
        CapabilityName.WORKLOAD_RESTART,
        CapabilityName.DRIVER_REMEDIATION,
        CapabilityName.EFA_DRIVER_REMEDIATION,
        CapabilityName.VM_RESTART,
        CapabilityName.FABRIC_MANAGER_RESTART,
        CapabilityName.FABRIC_RESET,
        CapabilityName.SOFTWARE_FIRMWARE_UPDATE,
    }
)


class RankExitStatus(StrictModel):
    rank: int = Field(ge=0)
    exit_code: int
    node_id: str
    signal: int | None = Field(default=None, ge=0)
    finished_at: datetime | None = None


class AllocationEntry(StrictModel):
    node_id: str
    instance_id: str | None = None
    rank: int | None = Field(default=None, ge=0)
    gpu_uuids: list[str] = Field(default_factory=list)
    gpu_count: int = Field(default=0, ge=0)
    fabric_partition: str | None = None


class TerminalEvent(StrictModel):
    schema_version: str = "1"
    cluster_id: str
    environment: Environment
    job_id: str
    attempt_id: str
    terminal_status: TerminalStatus
    ended_at: datetime
    rank_exit_status: list[RankExitStatus] = Field(default_factory=list)
    allocation: list[AllocationEntry] = Field(default_factory=list)
    workload_ids: list[str] = Field(default_factory=list)
    checkpoint_manifest_ref: str | None = None
    termination_initiator_incident_id: str | None = None
    runtime_profile_version: str
    restart_budget: int = Field(default=1, ge=0)

    @property
    def event_key(self) -> str:
        return f"{self.cluster_id}/{self.attempt_id}/TrainingAttemptTerminal"

    @property
    def has_nonzero_exit(self) -> bool:
        return any(item.exit_code != 0 for item in self.rank_exit_status)

    @property
    def is_failure(self) -> bool:
        return (
            self.terminal_status
            in {
                TerminalStatus.FAILED,
                TerminalStatus.TIMED_OUT,
            }
            or self.has_nonzero_exit
        )

    @property
    def gpu_count(self) -> int:
        declared = sum(allocation.gpu_count for allocation in self.allocation)
        if declared:
            return declared
        return len(
            {
                gpu_uuid
                for allocation in self.allocation
                for gpu_uuid in allocation.gpu_uuids
            }
        )


class RestartBudgetState(StrictModel):
    cluster_id: str
    job_id: str
    budget: int = Field(ge=0)
    restart_count: int = Field(default=0, ge=0)
    reservation_ids: list[str] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def validate_count(self) -> RestartBudgetState:
        if self.restart_count != len(self.reservation_ids):
            raise ValueError("restart_count must match restart reservations")
        if self.restart_count > self.budget:
            raise ValueError("restart_count exceeds restart budget")
        if len(self.reservation_ids) != len(set(self.reservation_ids)):
            raise ValueError("restart reservation IDs must be unique")
        return self


class MarkerScope(StrictModel):
    node_ids: list[str] = Field(default_factory=list)
    gpu_uuids: list[str] = Field(default_factory=list)
    pci_bdfs: list[str] = Field(default_factory=list)
    fabric_partitions: list[str] = Field(default_factory=list)


class NodeMarker(StrictModel):
    marker_id: str = Field(default_factory=lambda: f"marker-{uuid4()}")
    source: str
    trusted: bool = False
    incident_id: str
    observed_at: datetime
    source_event_time: datetime | None = None
    source_monotonic_us: int | None = Field(default=None, ge=0)
    source_boot_id: str | None = None
    collected_at: datetime | None = None
    ingested_at: datetime | None = None
    event_source: str | None = None
    expires_at: datetime
    scope: MarkerScope
    severity: Severity
    recommended_action: RecoveryAction | None = None
    action_owner: str | None = None
    mapping_version: str
    policy_source: str | None = None
    official_action: str | None = None
    site_safety_action: str | None = None
    investigatory_action: str | None = None
    action_disposition: str | None = None
    fault_class: str | None = None
    correlation_keys: list[str] = Field(default_factory=list)
    raw_reason: str | None = None
    raw_evidence_ref: str | None = None
    drill_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    )
    active: bool = True

    @model_validator(mode="after")
    def validate_window(self) -> NodeMarker:
        if self.expires_at <= self.observed_at:
            raise ValueError("expires_at must be after observed_at")
        return self


class DiagnosticRequest(StrictModel):
    request_id: str = Field(default_factory=lambda: f"diag-{uuid4()}")
    cluster_id: str | None = None
    attempt_id: str
    node_ids: list[str]
    checks: list[str]
    deadline_seconds: int = Field(default=60, ge=1, le=600)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TriageFinding(StrictModel):
    node_id: str
    outcome: TriageOutcome
    failed_checks: list[str] = Field(default_factory=list)
    proposed_action: RecoveryAction | None = None
    evidence_refs: list[str] = Field(default_factory=list)
    reason: str | None = None


class TriageReport(StrictModel):
    request_id: str
    cluster_id: str | None = None
    attempt_id: str
    findings: list[TriageFinding] = Field(min_length=1)
    completed_at: datetime


class PlanStep(StrictModel):
    action: RecoveryAction
    node_ids: list[str] = Field(default_factory=list)
    gpu_uuids: list[str] = Field(default_factory=list)
    execution_owner: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class RecoveryPlan(StrictModel):
    plan_id: str = Field(default_factory=lambda: f"plan-{uuid4()}")
    incident_id: str
    attempt_id: str
    trigger: str
    runtime_profile_version: str
    steps: list[PlanStep]
    avoid_node_ids: list[str] = Field(default_factory=list)
    checkpoint_manifest_ref: str | None = None
    drill_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    )
    workflow_request_id: str | None = None
    status: PlanStatus = PlanStatus.PENDING
    resolved_by_restore_workflow_id: str | None = None
    reconciliation_reference: str | None = None
    reconciled_at: datetime | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CompletionDecision(StrictModel):
    cluster_id: str
    attempt_id: str
    event_key: str
    status: DecisionStatus
    reason: str
    duplicate: bool = False
    matched_marker_ids: list[str] = Field(default_factory=list)
    diagnostic_request_id: str | None = None
    recovery_plan_id: str | None = None


class CapabilityClaim(StrictModel):
    capability: CapabilityName
    mode: CapabilityMode
    owner: str
    adapter: str | None = None


class ObservedCapability(StrictModel):
    capability: CapabilityName
    owner: str
    available: bool
    version: str | None = None


class RuntimeProfile(StrictModel):
    cluster_id: str
    environment: Environment
    claims: list[CapabilityClaim]
    observed: list[ObservedCapability]
    profile_version: str


class EffectiveCapability(StrictModel):
    capability: CapabilityName
    mode: CapabilityMode
    owner: str
    adapter: str | None = None
    observed_version: str | None = None


class EffectiveRuntimeProfile(StrictModel):
    cluster_id: str
    environment: Environment
    profile_version: str
    capabilities: list[EffectiveCapability]
    warnings: list[str] = Field(default_factory=list)


class OperationResult(StrictModel):
    operation_id: str
    plan_id: str
    status: PlanStatus
    completed_steps: int = 0
    error: str | None = None


class WorkflowStepSpec(StrictModel):
    operation: WorkflowOperation
    execution_owner: str
    node_ids: list[str] = Field(default_factory=list)
    gpu_uuids: list[str] = Field(default_factory=list)
    workload_ids: list[str] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)
    depends_on_step_indexes: list[int] = Field(default_factory=list)
    branch_id: str | None = None
    # The nodes a job-workflow branch was created for. Branch identity reads
    # this, not ``node_ids``: widening a host-level step onto another node
    # must not make that node's lookups collapse onto this branch (F-B6 (2)).
    # Empty on rows written before the field existed; callers fall back to
    # ``node_ids`` then.
    branch_node_ids: list[str] = Field(default_factory=list)


# Which step list a record belongs to. Safety and official steps share
# indexes, so a step's identity is (phase, step_index, operation) (F-C2).
StepPhase = Literal["official", "safety"]


class WorkflowStepExecution(StrictModel):
    step_index: int = Field(ge=0)
    operation: WorkflowOperation
    status: WorkflowStepStatus
    # None on records written before the phase existed; such a record answers
    # for either phase (dual read) until it is replaced by a stamped one.
    phase: StepPhase | None = None
    adapter_operation_id: str | None = None
    error: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class FaultIncident(StrictModel):
    incident_id: str
    event_id: str
    event_type: str
    event_source: str | None = None
    source_boot_id: str | None = None
    cluster_id: str
    node_ids: list[str]
    gpu_uuids: list[str] = Field(default_factory=list)
    job_id: str | None = None
    attempt_id: str | None = None
    workload_identity_source: str | None = None
    policy_version: str
    policy_source: str
    policy_reference: str | None = None
    official_action: str | None = None
    effective_action: RecoveryAction | None = None
    safety_action: RecoveryAction | None = None
    state: IncidentState = IncidentState.DETECTED
    workflow_request_id: str | None = None
    fencing_token: int = Field(default=1, ge=1)
    drill_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    )
    reasons: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class WorkflowRequest(StrictModel):
    request_id: str = Field(default_factory=lambda: f"workflow-{uuid4()}")
    incident_id: str
    source_plan_id: str | None = None
    predecessor_workflow_id: str | None = None
    preempt_predecessor: bool = False
    preempted_by_workflow_id: str | None = None
    preemption_reason: str | None = None
    # Set by the merge transaction that created a stronger successor with
    # ``preempt_predecessor`` (F-C1): this row is about to be superseded, so
    # the dispatcher must not start it while the executor works out the safe
    # boundary. Cleared when the row reaches a terminal status.
    preemption_pending_by_workflow_id: str | None = None
    inherited_step_indexes: list[int] = Field(default_factory=list)
    quiesce_handoff_from_workflow_id: str | None = None
    superseded_at: datetime | None = None
    dag_enabled: bool = False
    dag_revision: int = Field(default=0, ge=0)
    runtime_profile_version: str | None = None
    status: WorkflowStatus
    official_action: str | None = None
    fencing_token: int = Field(ge=1)
    # Bumped by every merge into an existing workflow row; the leased writers
    # compare it so an executor copy taken before a merge cannot erase the
    # merged targets (F-B1).
    merge_revision: int = Field(default=0, ge=0)
    safety_steps: list[WorkflowStepSpec] = Field(default_factory=list)
    official_steps: list[WorkflowStepSpec] = Field(default_factory=list)
    blocked_reasons: list[str] = Field(default_factory=list)
    blocked_kind: BlockedKind | None = None
    # F-N1: per-node escalation bookkeeping for a multi-branch job workflow.
    # Keyed by node id; counts the in-place rungs already taken.
    branch_escalation_counts: dict[str, int] = Field(default_factory=dict)
    # Branch ids whose escalation ladder is exhausted. The job-restart join
    # must not run while this is non-empty; the workflow ends FAILED.
    exhausted_branch_ids: list[str] = Field(default_factory=list)
    # Hard lifetime of this remediation (F-N1). Stamped at the first claim
    # from the workflow's kind, inherited along the escalation chain. At the
    # deadline the workflow fails and escalates to an operator; later events
    # for the same scope are recorded on the incident, not re-planned.
    lifetime_deadline_at: datetime | None = None
    # F-N1 §7: the training job this workflow was repairing was stopped by
    # someone else. In-flight work finishes, touched nodes are released,
    # nothing new starts, the job is not restarted.
    workload_withdrawn_at: datetime | None = None
    workload_withdrawn_reason: str | None = None
    # F-N1 §8: when set, the workflow ends FAILED with this reason once its
    # (rewritten) steps complete, instead of SUCCEEDED.
    terminal_failure_reason: str | None = None
    # True when this record was compiled to run its ``safety_steps`` (it was
    # created SAFETY_PENDING). Explicit so that a warning appended to
    # ``blocked_reasons`` can never switch the executor's step set (F-C8).
    safety_only: bool = False
    completed_operations: list[WorkflowOperation] = Field(default_factory=list)
    completed_step_indexes: list[int] = Field(default_factory=list)
    superseded_step_indexes: list[int] = Field(default_factory=list)
    pending_failure_step_index: int | None = Field(default=None, ge=0)
    pending_failure_error: str | None = None
    failure_handled_at: datetime | None = None
    failure_handling_attempts: int = Field(default=0, ge=0)
    step_executions: list[WorkflowStepExecution] = Field(default_factory=list)
    execution_owner_id: str | None = None
    execution_epoch: int = Field(default=0, ge=0)
    execution_lease_expires_at: datetime | None = None
    execution_deadline: datetime | None = None
    remediation_budget_claims: list[str] = Field(default_factory=list)
    remediation_budget_limits: dict[str, int] = Field(default_factory=dict)
    remediation_budget_wait_count: int = Field(default=0, ge=0)
    remediation_budget_last_blocked_reason: str | None = None
    not_before: datetime | None = None
    aggregation_max_deadline: datetime | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def executes_safety_steps(self) -> bool:
        """Whether the executor runs ``safety_steps`` rather than
        ``official_steps``. SAFETY_PENDING implies it for records written
        before ``safety_only`` existed."""

        return self.safety_only or self.status is WorkflowStatus.SAFETY_PENDING


class RestartAuthorization(StrictModel):
    cluster_id: str
    job_id: str
    source_attempt_id: str
    source_gpu_count: int = Field(ge=0)
    restart_budget: int = Field(ge=0)
    restart_count: int = Field(ge=1)
    reservation_id: str


class WorkflowExecutionRequest(StrictModel):
    expected_fencing_token: int = Field(ge=1)
    isolation_verified_nodes: list[str] = Field(default_factory=list)
    confirm_cluster_name: str | None = None
    confirmed_adapter_operation_ids: list[str] = Field(default_factory=list)
    restart_authorization: RestartAuthorization | None = None


class WorkflowExecutionResult(StrictModel):
    operation_id: str = Field(default_factory=lambda: f"workflow-op-{uuid4()}")
    workflow_request_id: str
    incident_id: str
    status: WorkflowStatus
    completed_operations: list[WorkflowOperation]
    simulation_only: bool = True
    waiting_step_index: int | None = Field(default=None, ge=0)
    error: str | None = None


def lifetime_exceeded(workflow: "WorkflowRequest", now: datetime | None = None) -> bool:
    """Whether the workflow's hard lifetime has passed (F-N1).

    An unstamped workflow (never claimed) has no lifetime yet and is never
    exceeded.
    """

    if workflow.lifetime_deadline_at is None:
        return False
    return (now or datetime.now(timezone.utc)) >= workflow.lifetime_deadline_at


# Upper bound on ``FaultIncident.reasons``. Every merged event appends its
# reasons; a long-lived incident (an hour of correlated faults, a node that
# keeps re-reporting) grew the list without bound and squeezed the row (F-J6).
INCIDENT_REASONS_LIMIT = 200


def bounded_reasons(
    values: Iterable[str], *, limit: int = INCIDENT_REASONS_LIMIT
) -> list[str]:
    """Deduplicated reasons within ``limit``: the first half of the budget
    keeps the reasons the incident opened with, the rest the most recent."""

    unique = list(dict.fromkeys(values))
    if len(unique) <= limit:
        return unique
    head = limit // 2
    return [*unique[:head], *unique[-(limit - head) :]]


def resolved_step_indexes(workflow: "WorkflowRequest") -> frozenset[int]:
    """Steps that will never run again: completed or superseded (F-C2).

    A superseded step is skipped by the executor exactly like a completed one,
    so every "is this step still pending?" question must use this set. It is
    *not* the set of steps whose effects took place -- for that (coverage,
    inherited containment, unrestored quiesce) read ``completed_step_indexes``.
    """

    return frozenset(workflow.completed_step_indexes) | frozenset(
        workflow.superseded_step_indexes
    )


def execution_phase(workflow: "WorkflowRequest") -> StepPhase:
    """The phase whose steps this record executes right now (F-C2)."""

    return "safety" if workflow.executes_safety_steps else "official"


def execution_matches_step(
    execution: WorkflowStepExecution,
    index: int,
    operation: WorkflowOperation,
    phase: StepPhase | None = None,
) -> bool:
    """Is this record *this* step's record? Identity is (phase, index,
    operation); a record or a caller without a phase matches either phase."""

    return (
        execution.step_index == index
        and execution.operation is operation
        and (phase is None or execution.phase is None or execution.phase == phase)
    )


class WorkflowDispatchFailure(StrictModel):
    workflow_request_id: str
    error: str


class WorkflowDispatchReport(StrictModel):
    scanned: int = Field(ge=0)
    executed: int = Field(ge=0)
    waiting: int = Field(ge=0)
    completed: int = Field(ge=0)
    failed: int = Field(ge=0)
    failures: list[WorkflowDispatchFailure] = Field(default_factory=list)
    # Rows the eligibility filters held back this tick, by reason (F-A2).
    filtered: dict[str, int] = Field(default_factory=dict)
    # True when the scan hit its ceiling before finding a full batch.
    horizon_exhausted: bool = False
    # True when another process holds the fleet-wide dispatch lease (F-A1).
    lease_held_by_other: bool = False
    # Dispatch attempts that raised something other than a lease/fencing or
    # transient store error; the workflow stayed executable (F-B4 (3)).
    internal_errors: int = Field(default=0, ge=0)
    # Rows scanned but never started because the cycle deadline passed; they
    # stay PENDING for the next cycle (F-C7).
    deferred: int = Field(default=0, ge=0)


class NotificationStatus(StrEnum):
    QUEUED = "QUEUED"
    SENT = "SENT"
    SKIPPED = "SKIPPED"
    DUPLICATE = "DUPLICATE"
    FAILED = "FAILED"


class NotificationDeliveryStatus(StrEnum):
    PENDING = "PENDING"
    LEASED = "LEASED"
    RETRY = "RETRY"
    SENT = "SENT"
    DEAD = "DEAD"


class AdvisoryNotification(StrictModel):
    notification_id: str = Field(default_factory=lambda: f"notification-{uuid4()}")
    deduplication_key: str
    cluster_name: str
    incident_id: str
    subject: str
    body_text: str
    support_case_draft: str
    evidence_refs: list[str] = Field(default_factory=list)
    drill_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    )
    category: str = "GENERAL"
    priority: int = Field(default=50, ge=0, le=1000)
    not_before: datetime | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class NotificationResult(StrictModel):
    notification_id: str
    status: NotificationStatus
    provider_message_id: str | None = None
    reason: str | None = None


class NotificationDelivery(StrictModel):
    notification_id: str
    status: NotificationDeliveryStatus = NotificationDeliveryStatus.PENDING
    attempts: int = Field(default=0, ge=0)
    available_at: datetime
    lease_owner: str | None = None
    lease_epoch: int = Field(default=0, ge=0)
    lease_expires_at: datetime | None = None
    last_error: str | None = None
    created_at: datetime
    updated_at: datetime
    requeued_at: datetime | None = None
    """When a retired delivery was last put back by hand.

    The shelf life is measured from the event, so a notification an
    operator deliberately re-queues would otherwise expire again the
    moment the dispatcher picked it up. Re-queueing restarts the window
    rather than removing it: a stale notification asked for once still
    stops being retried eventually.
    """


class NotificationDispatchWatermark(StrictModel):
    """The point in time the dispatcher started being responsible.

    Notifications created before this are the backlog that accumulated
    while asynchronous delivery was disabled. Mailing them the moment the
    dispatcher is first enabled floods the operator with history, so they
    are suppressed instead of sent.
    """

    owner_id: str = "default"
    established_at: datetime
    established_by: str
    suppressed: int = Field(default=0, ge=0)


class NotificationDispatchRequest(StrictModel):
    limit: int = Field(default=25, ge=1, le=100)


class NotificationDispatchReport(StrictModel):
    attempted: int
    sent: int
    skipped: int
    failed: int
    results: list[NotificationResult]
    suppressed_backlog: int = Field(default=0, ge=0)
    throttled: int = Field(default=0, ge=0)
    """Claimed, then returned to the outbox because the provider said slow down.

    Not a failure of these notifications and not charged an attempt: the
    cycle stops at the first throttle so the process stops adding to the
    rate it is being limited for.
    """

    expired: int = Field(default=0, ge=0)
    """Claimed but retired unsent because they were past their shelf life.

    Separate from ``skipped``, which the provider decided, and from
    ``suppressed_backlog``, which only ever happens once per deployment.
    A non-zero value here every cycle means the outbox is draining slower
    than it fills.
    """
    suppressed_drills: int = Field(default=0, ge=0)
    """Retired unsent because they were labelled as drills.

    Only counts drills that were already in the outbox: ``send`` keeps new
    ones out of it, so a steady non-zero value here means something is
    enqueuing deliveries without going through it.
    """
    scan_truncated: bool = False
    """The synchronous path saw only the newest slice of the notification table.

    ``expired`` and ``suppressed_drills`` are then counted over that slice
    rather than over the whole history, and a deliverable notification older
    than the slice -- only possible for a category with no shelf life -- is not
    reached. Raising ``GPU_FAULT_NOTIFICATION_DISPATCH_SCAN_LIMIT`` moves the
    cost back onto every call; the answer to a persistently truncated scan is
    ``GPU_FAULT_NOTIFICATION_ASYNC_DELIVERY``, which claims from the outbox
    instead of scanning. Always ``False`` on the async path, which never scans.
    """
