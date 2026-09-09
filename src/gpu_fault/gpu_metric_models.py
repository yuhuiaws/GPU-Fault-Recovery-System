from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import Field

from gpu_fault.models import StrictModel, WorkloadState

SITE_METRIC_POLICY_VERSION = "site-dcgm-metric-policy/v3"


class GpuHealthSeverity(StrEnum):
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class GpuHealthFinding(StrictModel):
    finding_id: str
    cluster_id: str
    node_id: str
    observed_at: datetime
    severity: GpuHealthSeverity
    reason: str
    canonical_name: str
    value: float
    delta: float | None = None
    rate_per_minute: float | None = None
    threshold_value: float | None = None
    threshold_source: str | None = None
    gpu_uuid: str | None = None
    pci_bdf: str | None = None
    evidence_ref: str | None = None
    automatic_action: str | None = None
    policy_source: str = "SITE_DCGM_METRIC"
    policy_version: str = SITE_METRIC_POLICY_VERSION
    policy_reference: str | None = None
    official_action: str | None = None
    runtime_profile_version: str | None = None
    workload_state: WorkloadState = WorkloadState.UNKNOWN
    affected_workload_ids: list[str] = Field(default_factory=list)
    finding_kind: str = "METRIC"
    correlation_rule_id: str | None = None
    component_finding_ids: list[str] = Field(default_factory=list)
    component_metrics: list[str] = Field(default_factory=list)
    component_evidence_refs: list[str] = Field(default_factory=list)
    affected_gpu_uuids: list[str] = Field(default_factory=list)
    confidence: str | None = None


class GpuFindingState(StrictModel):
    observed_at: datetime
    finding: GpuHealthFinding | None = None
    consecutive_breaches: int = Field(default=0, ge=0)
