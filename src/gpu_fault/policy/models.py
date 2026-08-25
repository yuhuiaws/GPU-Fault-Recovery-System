from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
import re
from uuid import uuid4

from pydantic import Field, model_validator

from gpu_fault.models import (
    AllocationEntry,
    FaultIncident,
    NodeMarker,
    RecoveryAction,
    Severity,
    StrictModel,
    WorkflowRequest,
    WorkloadState,
)


class FaultEventType(StrEnum):
    XID = "XID"
    SXID = "SXID"


class Containment(StrEnum):
    APPLICATION = "APPLICATION"
    ALL_APPLICATIONS = "ALL_APPLICATIONS"
    GPU = "GPU"
    FABRIC_PARTITION = "FABRIC_PARTITION"
    NODE = "NODE"
    UNKNOWN = "UNKNOWN"


class SxidLinkScope(StrEnum):
    ACCESS = "ACCESS"
    TRUNK = "TRUNK"
    UNKNOWN = "UNKNOWN"


# NVIDIA Fabric Manager User Guide, Table 23 (Nov 14, 2025).
NVIDIA_ALWAYS_FATAL_SXIDS = frozenset(
    {
        12020,
        22003,
        22011,
        23001,
        23002,
        23003,
        23004,
        23005,
        23006,
        23007,
        23008,
        23009,
        23010,
        23011,
        23012,
        23013,
        23014,
        23015,
        23016,
        23017,
    }
)
NVIDIA_CODE_SPECIFIC_FULL_RESET_SXIDS = frozenset({10003, 19084})

NVLINK74_REGISTER_RULES = {
    0: {
        "safe_ignore": {0, 23, 30},
        "secondary": {1, 20},
        "ecc_parity": {4, 5},
        "mechanical_or_hardware": {8, 9, 12, 16, 17, 24, 28},
        "marginal_channel": {21, 22},
        "report_if_repeated": {27, 29},
    },
    2: {
        "ecc_parity": {0, 1, 2, 6},
        "unexpected_production": {13},
        "field_diag_if_repeated": {16, 19},
        "report_if_repeated": {17, 18},
    },
    3: {
        "secondary": {16, 17},
        "fabric_reset_required": {18},
    },
    4: {
        "ecc_parity": {18, 19, 21, 22, 24, 25, 27, 28},
        "corrected_threshold": {20, 23, 26, 29},
    },
}


class SxidClassification(StrEnum):
    NON_FATAL = "NON_FATAL"
    FATAL = "FATAL"
    ALWAYS_FATAL = "ALWAYS_FATAL"


class SxidCatalogRule(StrictModel):
    sxid: int = Field(ge=1)
    classification: SxidClassification
    official_action: str = Field(alias="officialAction")
    investigatory_action: str | None = Field(default=None, alias="investigatoryAction")
    applicability: str


class SxidPolicy(StrictModel):
    name: str
    source_url: str
    source_last_updated: str
    source_sha256: str
    coverage: str
    rules: list[SxidCatalogRule]

    @property
    def mapping_version(self) -> str:
        return f"{self.name}/sha256:{self.source_sha256[:16]}"


class DynamicRecoveryAction(StrEnum):
    IGNORE = "IGNORE"
    DRAIN_P2P = "DRAIN_P2P"
    DRAIN_AND_RESET = "DRAIN_AND_RESET"
    RESTART_APP = "RESTART_APP"
    RESET_GPU = "RESET_GPU"
    RESTART_BM = "RESTART_BM"


class ActionSource(StrEnum):
    NVIDIA_CATALOG = "NVIDIA_CATALOG"
    NVIDIA_XID_154 = "NVIDIA_XID_154"
    NVIDIA_FABRIC_MANAGER = "NVIDIA_FABRIC_MANAGER"
    SITE_SAFETY = "SITE_SAFETY"


class ActionDisposition(StrEnum):
    PENDING_CORRELATION = "PENDING_CORRELATION"
    EXECUTABLE = "EXECUTABLE"
    MONITOR_ONLY = "MONITOR_ONLY"
    BLOCKED_WORKFLOW = "BLOCKED_WORKFLOW"
    BLOCKED_MISSING_EVIDENCE = "BLOCKED_MISSING_EVIDENCE"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    SITE_SAFETY = "SITE_SAFETY"


DIRECT_ACTION_MAP = {
    "IGNORE": RecoveryAction.NO_ACTION,
    "RESTART_APP": RecoveryAction.RESTART_WORKLOAD,
    "RESET_GPU": RecoveryAction.RESET_GPU,
    "RESTART_BM": RecoveryAction.REBOOT_NODE,
}

# XID-45 companion selection only. Workflow merge arbitration uses the
# compiled-operation ranking in orchestrator.py; do not reuse this table
# there because companion containment and recovery severity are different
# orderings.
RECOVERY_ACTION_SEVERITY_RANK = {
    RecoveryAction.NO_ACTION: 0,
    RecoveryAction.COLLECT_EVIDENCE: 1,
    RecoveryAction.RUN_DIAGNOSTICS: 2,
    RecoveryAction.VALIDATE_NODE: 2,
    RecoveryAction.RESTART_WORKLOAD: 3,
    RecoveryAction.MARK_UNSCHEDULABLE: 4,
    RecoveryAction.DRAIN: 5,
    RecoveryAction.STOP_WORKLOAD: 5,
    RecoveryAction.RESET_GPU: 6,
    RecoveryAction.REBOOT_NODE: 7,
    RecoveryAction.REPLACE_NODE: 8,
    RecoveryAction.QUARANTINE: 9,
    RecoveryAction.ESCALATE_OPERATOR: 9,
}

CONTAINMENT_SEVERITY_RANK = {
    Containment.APPLICATION: 0,
    Containment.GPU: 1,
    Containment.ALL_APPLICATIONS: 2,
    Containment.FABRIC_PARTITION: 3,
    Containment.NODE: 4,
    Containment.UNKNOWN: 5,
}

# An operator decision is at least as serious as a GPU reset, so an
# unresolvable companion must not lose to an IGNORE companion.
OPERATOR_SEVERITY_RANK = 9

DEFAULT_DECISION_CACHE_SIZE = 4096

XID154_ACTION_LABELS = {
    "none": DynamicRecoveryAction.IGNORE,
    "drain p2p": DynamicRecoveryAction.DRAIN_P2P,
    "drain and reset": DynamicRecoveryAction.DRAIN_AND_RESET,
    "gpu reset required": DynamicRecoveryAction.RESET_GPU,
    "node reboot required": DynamicRecoveryAction.RESTART_BM,
}


def parse_xid154_action(
    message: str | None,
) -> DynamicRecoveryAction | None:
    """Parse only the recovery-action label emitted by XID 154."""
    if not message or not re.search(r"\bXid\b[^,\n]*\b154\b", message, re.I):
        return None
    match = re.search(
        r"GPU\s+recovery\s+action\s+changed\s+from\b.*?\bto\b"
        r"\s*(?:0x[0-9a-f]+\s*)?\(([^)]+)\)",
        message,
        re.IGNORECASE,
    )
    if match is None:
        return None
    label = re.sub(r"\s+", " ", match.group(1)).strip().lower()
    return XID154_ACTION_LABELS.get(label)


class Nvlink74BitOccurrenceState(StrictModel):
    cluster_id: str
    gpu_identity: str
    link_id: int = Field(ge=0)
    register_index: int = Field(ge=0, le=6)
    bit: int = Field(ge=0)
    count: int = Field(ge=1)
    first_observed_at: datetime
    last_observed_at: datetime
    last_event_id: str

    def incremented(
        self, event_id: str, observed_at: datetime
    ) -> Nvlink74BitOccurrenceState:
        return self.model_copy(
            update={
                "count": self.count + 1,
                "last_observed_at": max(self.last_observed_at, observed_at),
                "last_event_id": event_id,
            }
        )


def _normalize_observed_at_to_utc(event) -> None:
    """Force ``observed_at`` to UTC, in place.

    Correlation windows are filtered by lexical comparison on the
    serialized value, so a producer reporting +08:00 would sort outside
    its own window. A naive value is read as UTC rather than rejected,
    because kernel log scrapers routinely omit the offset. Other
    timestamps keep the producer's offset: they are provenance, never
    query bounds.
    """
    value = event.observed_at
    normalized = (
        value.replace(tzinfo=timezone.utc)
        if value.tzinfo is None
        else value.astimezone(timezone.utc)
    )
    if normalized is not value:
        object.__setattr__(event, "observed_at", normalized)


class XidEvent(StrictModel):
    event_id: str = Field(default_factory=lambda: f"xid-{uuid4()}")
    cluster_id: str
    node_id: str
    observed_at: datetime
    source_event_time: datetime | None = None
    source_monotonic_us: int | None = Field(default=None, ge=0)
    source_boot_id: str | None = None
    collected_at: datetime | None = None
    ingested_at: datetime | None = None
    event_source: str | None = None
    xid: int = Field(ge=0)
    gpu_uuid: str | None = None
    pci_bdf: str | None = None
    pod_uid: str | None = None
    container_id: str | None = None
    host_pid: int | None = Field(default=None, ge=1)
    cgroup_path: str | None = None
    product: str | None = None
    driver_branch: int | None = Field(default=None, ge=0)
    cuda_version: str | None = None
    job_id: str | None = None
    attempt_id: str | None = None
    workload_identity_source: str | None = None
    fabric_partition: str | None = None
    runtime_profile_version: str | None = None
    workload_state: WorkloadState = WorkloadState.UNKNOWN
    affected_workload_ids: list[str] = Field(default_factory=list)
    checkpoint_manifest_ref: str | None = None
    intr_info: int | None = Field(default=None, ge=0)
    error_status: int | None = Field(default=None, ge=0)
    registers: list[int] = Field(default_factory=list)
    nvlink_link_id: int | None = Field(default=None, ge=0)
    nvlink_link_identity_source: str | None = None
    nvlink_occurrence_counts: dict[str, int] = Field(default_factory=dict)
    xid_154_action: DynamicRecoveryAction | None = None
    uvm_in_use: bool | None = None
    raw_message: str | None = None
    evidence_ref: str | None = None
    drill_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    )
    synthetic: bool = False

    @model_validator(mode="after")
    def normalize_observed_at(self) -> XidEvent:
        _normalize_observed_at_to_utc(self)
        return self


class DistributedXidBatch(StrictModel):
    batch_id: str = Field(default_factory=lambda: f"xid-batch-{uuid4()}")
    job_id: str
    attempt_id: str
    restart_budget: int = Field(ge=0)
    allocation: list[AllocationEntry] = Field(min_length=2)
    events: list[XidEvent] = Field(min_length=1)
    affected_workload_ids: list[str] = Field(min_length=1)
    checkpoint_manifest_ref: str | None = None

    @model_validator(mode="after")
    def validate_scope(self) -> DistributedXidBatch:
        event_ids = [item.event_id for item in self.events]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("distributed XID event IDs must be unique")
        clusters = {item.cluster_id for item in self.events}
        profiles = {item.runtime_profile_version for item in self.events}
        if len(clusters) != 1:
            raise ValueError("distributed XID events must share one cluster")
        if len(profiles) != 1 or None in profiles:
            raise ValueError("distributed XID events must share one runtime profile")
        if any(item.job_id not in {None, self.job_id} for item in self.events):
            raise ValueError("distributed XID events target another job")
        if any(item.workload_state is not WorkloadState.ACTIVE for item in self.events):
            raise ValueError("distributed XID recovery requires an ACTIVE workload")
        allocation_gpus: dict[str, set[str]] = {}
        for entry in self.allocation:
            allocation_gpus.setdefault(entry.node_id, set()).update(entry.gpu_uuids)
        for item in self.events:
            if item.node_id not in allocation_gpus:
                raise ValueError(f"fault node {item.node_id} is outside allocation")
            if not item.gpu_uuid or item.gpu_uuid not in allocation_gpus[item.node_id]:
                raise ValueError(f"fault GPU for {item.node_id} is not in allocation")
        return self


class SxidEvent(StrictModel):
    event_id: str = Field(default_factory=lambda: f"sxid-{uuid4()}")
    cluster_id: str
    node_id: str
    observed_at: datetime
    source_event_time: datetime | None = None
    source_monotonic_us: int | None = Field(default=None, ge=0)
    source_boot_id: str | None = None
    collected_at: datetime | None = None
    ingested_at: datetime | None = None
    event_source: str | None = None
    sxid: int = Field(ge=0)
    classification: SxidClassification
    classification_source: str
    link_scope: SxidLinkScope = SxidLinkScope.UNKNOWN
    link_scope_source: str | None = None
    product: str | None = None
    switch_id: str | None = None
    pci_bdf: str | None = None
    port: str | None = None
    fabric_partition: str | None = None
    job_id: str | None = None
    attempt_id: str | None = None
    workload_identity_source: str | None = None
    runtime_profile_version: str | None = None
    workload_state: WorkloadState = WorkloadState.UNKNOWN
    affected_workload_ids: list[str] = Field(default_factory=list)
    checkpoint_manifest_ref: str | None = None
    participating_gpu_uuids: list[str] = Field(default_factory=list)
    pod_uid: str | None = None
    container_id: str | None = None
    host_pid: int | None = Field(default=None, ge=1)
    cgroup_path: str | None = None
    raw_message: str | None = None
    evidence_ref: str | None = None
    drill_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    )
    synthetic: bool = False

    @model_validator(mode="after")
    def normalize_observed_at(self) -> SxidEvent:
        _normalize_observed_at_to_utc(self)
        return self


class CatalogRule(StrictModel):
    xid: int
    mnemonic: str | None = None
    description: str | None = None
    products: list[str] = Field(default_factory=list)
    immediate_action: str | None = Field(alias="immediateAction")
    investigatory_action: str | None = Field(default=None, alias="investigatoryAction")
    xid154_linkage: str | None = Field(default=None, alias="xid154Linkage")
    trigger_conditions: str | None = Field(default=None, alias="triggerConditions")


class NvlinkDecodeRule(StrictModel):
    xid: int
    subcode_name: str = Field(alias="subcodeName")
    v1_pattern: str = Field(alias="v1Pattern")
    v2_pattern: str = Field(alias="v2Pattern")
    error_status: str | None = Field(alias="errorStatus")
    recovery_action: str = Field(alias="recoveryAction")
    action2_v1_pattern: str | None = Field(default=None, alias="action2V1Pattern")
    action2_v2_pattern: str | None = Field(default=None, alias="action2V2Pattern")
    action2: str | None = None
    investigatory_action: str | None = Field(default=None, alias="investigatoryAction")
    severity: str | None = None
    fault_origin: str | None = Field(default=None, alias="faultOrigin")
    local_remote: str | None = Field(default=None, alias="localRemote")

    @model_validator(mode="after")
    def validate_decode_patterns(self):
        for name in (
            "v1_pattern",
            "v2_pattern",
            "action2_v1_pattern",
            "action2_v2_pattern",
        ):
            pattern = getattr(self, name)
            if pattern is None:
                continue
            if len(pattern) != 32 or re.fullmatch(r"[01-]{32}", pattern) is None:
                raise ValueError(f"{name} must contain exactly 32 binary/wildcard bits")
        if self.error_status is not None:
            for item in self.error_status.split("/"):
                try:
                    int(item.strip(), 16)
                except ValueError as exc:
                    raise ValueError(
                        "errorStatus must contain slash-separated hexadecimal values"
                    ) from exc
        return self


class Nvlink5Policy(StrictModel):
    driver_boundary: int = Field(alias="driverBoundary")
    decode_rules: list[NvlinkDecodeRule] = Field(alias="decodeRules")


class CatalogProductFamily(StrictModel):
    family: str = Field(min_length=1)
    model_prefixes: list[str] = Field(
        min_length=1,
        alias="modelPrefixes",
    )


class XidPolicy(StrictModel):
    name: str
    catalog_version: str
    coverage: str
    source_url: str
    source_sha256: str
    generated_sha256: str
    product_families: list[CatalogProductFamily]
    marker_ttl_seconds: int
    companion_window_seconds: int
    catalog_rules: list[CatalogRule]
    nvlink5: Nvlink5Policy
    resolution_buckets: dict[str, str | None]

    @property
    def mapping_version(self) -> str:
        return f"{self.name}/sha256:{self.generated_sha256[:16]}"


class FaultPolicyDecision(StrictModel):
    event_id: str
    event_type: FaultEventType
    policy_version: str
    source: ActionSource
    disposition: ActionDisposition
    official_action: str | None = None
    investigatory_action: str | None = None
    action: RecoveryAction | None = None
    safety_action: RecoveryAction | None = None
    severity: Severity
    containment: Containment
    reasons: list[str]
    pre_actions: list[RecoveryAction] = Field(default_factory=list)
    decoded_subcode: int | None = None
    matched_decode_rules: list[str] = Field(default_factory=list)
    nvlink_link_id: int | None = Field(default=None, ge=0)
    nvlink_occurrence_counts: dict[str, int] = Field(default_factory=dict)
    requires_operator: bool = False
    marker: NodeMarker
    incident_id: str | None = None
    workflow_request_id: str | None = None
    advisory_notification_id: str | None = None
    investigatory_notification_id: str | None = None
    duplicate: bool = False
    correlated_event_id: str | None = None


class XidCorrelationStatus(StrEnum):
    PENDING = "PENDING"
    FINALIZED = "FINALIZED"


class XidCorrelationRecord(StrictModel):
    event_id: str
    deadline: datetime
    status: XidCorrelationStatus = XidCorrelationStatus.PENDING
    lease_owner: str | None = None
    lease_expires_at: datetime | None = None
    finalized_at: datetime | None = None


class DistributedXidIngestionResult(StrictModel):
    batch_id: str
    decisions: list[FaultPolicyDecision]
    incident: FaultIncident
    workflow: WorkflowRequest


@dataclass
class _Resolution:
    source: ActionSource
    disposition: ActionDisposition
    official_action: str | None
    investigatory_action: str | None
    action: RecoveryAction | None
    containment: Containment
    reasons: list[str]
    pre_actions: list[RecoveryAction] = field(default_factory=list)
    decoded_subcode: int | None = None
    matched_decode_rules: list[str] = field(default_factory=list)
    requires_operator: bool = False
    correlated_event_id: str | None = None
