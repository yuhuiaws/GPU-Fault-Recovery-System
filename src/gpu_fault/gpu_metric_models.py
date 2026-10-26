from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import Field

from gpu_fault.models import StrictModel, WorkloadState

SITE_METRIC_POLICY_VERSION = "site-dcgm-metric-policy/v5"


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
    confirmation_count: int = Field(default=0, ge=0)


class GpuFindingState(StrictModel):
    observed_at: datetime
    finding: GpuHealthFinding | None = None
    consecutive_breaches: int = Field(default=0, ge=0)


def gpu_finding_transition(
    previous: GpuFindingState | None,
    finding: GpuHealthFinding | None,
    observed_at: datetime,
) -> tuple[GpuFindingState, bool]:
    """The state that replaces ``previous`` and whether that write is an
    activation -- a finding that warrants an incident of its own.

    The one rule every store applies (memory, SQLite, PostgreSQL) once it has
    rejected a stale observation. A METRIC finding activates when none was
    active or its grade (severity) or automatic action changed; its id is
    stamped with every batch, so the id says nothing about identity. A
    COMPOSITE finding carries one id per episode
    (``gpu_composites.composite_episode_identity``): a new id is a new episode
    -- it activates even at the grade the stale state held and restarts the
    consecutive count -- while the same id with a changed grade is a re-grade
    of the open episode, counted here as an activation (the history keeps the
    latest grade) and told apart by the caller through the unchanged id.
    """

    previous_finding = previous.finding if previous is not None else None
    new_episode = (
        finding is not None
        and previous_finding is not None
        and finding.finding_kind == "COMPOSITE"
        and finding.finding_id != previous_finding.finding_id
    )
    consecutive_breaches = (
        0
        if finding is None
        else 1
        if previous is None or previous_finding is None or new_episode
        else previous.consecutive_breaches + 1
    )
    activated = finding is not None and (
        previous_finding is None
        or new_episode
        or finding.severity != previous_finding.severity
        or finding.automatic_action != previous_finding.automatic_action
    )
    return (
        GpuFindingState(
            observed_at=observed_at,
            finding=finding,
            consecutive_breaches=consecutive_breaches,
        ),
        activated,
    )
