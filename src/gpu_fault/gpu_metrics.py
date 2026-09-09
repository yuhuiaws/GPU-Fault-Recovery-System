from __future__ import annotations

import math
import os
from collections import Counter
from datetime import datetime, timedelta
from enum import StrEnum
from threading import RLock
from uuid import uuid4

from pydantic import Field, model_validator

from gpu_fault.gpu_collection_errors import (
    collection_error_finding,
    collection_error_key,
)
from gpu_fault.gpu_metric_models import (
    SITE_METRIC_POLICY_VERSION,
    GpuHealthSeverity,
)
from gpu_fault.gpu_metric_models import (
    GpuFindingState as GpuFindingState,
)
from gpu_fault.gpu_metric_models import (
    GpuHealthFinding as GpuHealthFinding,
)
from gpu_fault.gpu_power_policy import (
    NVIDIA_DCGM_HEALTH_REFERENCE,
    NVIDIA_DCGM_POLICY_VERSION,
    power_violation_decision,
)
from gpu_fault.models import RecoveryAction, StrictModel, WorkloadState
from gpu_fault.policy import FaultPolicyDecision, XidEvent

NVIDIA_GPU_MEMORY_POLICY_VERSION = "nvidia-gpu-memory-error-management/2026-07-26"
NVIDIA_ROW_REMAP_REFERENCE = (
    "https://docs.nvidia.com/deploy/a100-gpu-mem-error-mgmt/row-remapping.html"
)
NVIDIA_ROW_REMAP_RMA_REFERENCE = (
    "https://docs.nvidia.com/deploy/a100-gpu-mem-error-mgmt/"
    "rma-policy-thresholds-for-row-remapping.html"
)
NVIDIA_NVML_TEMPERATURE_REFERENCE = (
    "https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html"
)
SITE_CORRELATION_POLICY_VERSION = "site-dcgm-correlation/v1"
THERMAL_CLOCK_THROTTLE_MASK = 0x20 | 0x40
# One minute of violation per minute, plus a margin for sampling skew. A
# violation-duration counter cannot exceed this without meaning something other
# than microseconds spent in violation.
MAX_PLAUSIBLE_VIOLATION_US_PER_MINUTE = 60_000_000 * 1.05


class GpuMetricSource(StrEnum):
    DCGM_EXPORTER = "DCGM_EXPORTER"
    NVIDIA_SMI = "NVIDIA_SMI"


class GpuMetricSample(StrictModel):
    metric_name: str
    canonical_name: str
    value: float
    unit: str | None = None
    gpu_index: str | None = None
    gpu_uuid: str | None = None
    pci_bdf: str | None = None
    labels: dict[str, str] = Field(default_factory=dict)


class GpuInventoryDevice(StrictModel):
    gpu_index: int = Field(ge=0)
    gpu_uuid: str = Field(min_length=1)
    pci_bdf: str = Field(min_length=1)
    product: str | None = None


class GpuInventorySnapshot(StrictModel):
    snapshot_id: str = Field(default_factory=lambda: f"gpu-inventory-{uuid4()}")
    cluster_id: str = Field(min_length=1)
    node_id: str = Field(min_length=1)
    observed_at: datetime
    collected_at: datetime | None = None
    ingested_at: datetime | None = None
    source: GpuMetricSource
    source_boot_id: str = Field(min_length=1)
    node_instance_id: str | None = None
    devices: list[GpuInventoryDevice] = Field(min_length=1)
    expected_gpu_count: int | None = Field(default=None, ge=1)
    runtime_profile_version: str | None = None
    evidence_ref: str | None = None

    @model_validator(mode="after")
    def validate_devices(self) -> GpuInventorySnapshot:
        indexes = [item.gpu_index for item in self.devices]
        uuids = [item.gpu_uuid for item in self.devices]
        pci_bdfs = [item.pci_bdf.strip().lower() for item in self.devices]
        if len(indexes) != len(set(indexes)):
            raise ValueError("GPU inventory contains duplicate indexes")
        if len(uuids) != len(set(uuids)):
            raise ValueError("GPU inventory contains duplicate UUIDs")
        if len(pci_bdfs) != len(set(pci_bdfs)):
            raise ValueError("GPU inventory contains duplicate PCI BDFs")
        if (
            self.expected_gpu_count is not None
            and len(self.devices) > self.expected_gpu_count
        ):
            raise ValueError("GPU inventory exceeds expected GPU count")
        return self


class GpuMetricHistoryPoint(StrictModel):
    observed_at: datetime
    samples: list[GpuMetricSample]


class GpuMetricBatch(StrictModel):
    batch_id: str = Field(default_factory=lambda: f"gpu-metrics-{uuid4()}")
    cluster_id: str = Field(min_length=1)
    node_id: str = Field(min_length=1)
    observed_at: datetime
    collected_at: datetime | None = None
    ingested_at: datetime | None = None
    source: GpuMetricSource
    samples: list[GpuMetricSample]
    runtime_profile_version: str | None = None
    product: str | None = None
    driver_branch: int | None = Field(default=None, ge=0)
    cuda_version: str | None = None
    workload_state: WorkloadState = WorkloadState.UNKNOWN
    affected_workload_ids: list[str] = Field(default_factory=list)
    checkpoint_manifest_ref: str | None = None
    evidence_ref: str | None = None
    edge_filter_reasons: list[str] = Field(default_factory=list)
    collection_errors: list[str] = Field(default_factory=list)
    context_history: list[GpuMetricHistoryPoint] = Field(default_factory=list)


class GpuMetricsThresholds(StrictModel):
    gpu_temperature_warning_c: float = Field(default=85, ge=0)
    gpu_temperature_critical_c: float = Field(default=90, ge=0)
    memory_temperature_warning_c: float = Field(default=90, ge=0)
    memory_temperature_critical_c: float = Field(default=95, ge=0)
    gpu_temperature_warning_margin_c: float = Field(default=5, gt=0)
    gpu_temperature_shutdown_margin_c: float = Field(default=3, gt=0)
    memory_temperature_warning_margin_c: float = Field(default=5, gt=0)
    pcie_replay_rate_warning_per_minute: float = Field(default=8, ge=0)
    nvlink_error_delta_critical: float = Field(default=1, gt=0)
    power_violation_delta_warning_us: float = Field(default=1, gt=0)
    thermal_violation_delta_warning_us: float = Field(default=1, gt=0)
    thermal_violation_drain_consecutive_samples: int = Field(default=2, ge=2)
    correlation_window_seconds: int = Field(default=45, ge=15, le=300)
    composite_consecutive_samples: int = Field(default=2, ge=2, le=10)
    power_limit_ratio: float = Field(default=0.95, gt=0, le=1)
    power_correlation_min_utilization_percent: float = Field(default=80, ge=0, le=100)
    ecc_sbe_delta_warning: float = Field(default=1, gt=0)
    retired_pages_sbe_delta_warning: float = Field(default=1, gt=0)
    retired_pages_dbe_delta_critical: float = Field(default=1, gt=0)
    row_remap_correctable_delta_warning: float = Field(default=1, gt=0)
    correctable_memory_drain_consecutive_samples: int = Field(default=3, ge=2, le=10)

    @model_validator(mode="after")
    def validate_temperature_order(
        self,
    ) -> GpuMetricsThresholds:
        if self.gpu_temperature_critical_c <= self.gpu_temperature_warning_c:
            raise ValueError("GPU critical temperature must exceed warning")
        if self.memory_temperature_critical_c <= self.memory_temperature_warning_c:
            raise ValueError("memory critical temperature must exceed warning")
        return self

    @classmethod
    def from_environment(cls) -> GpuMetricsThresholds:
        defaults = cls()
        values = {}
        for field_name, env_name in {
            "gpu_temperature_warning_c": ("GPU_FAULT_GPU_TEMP_WARNING_C"),
            "gpu_temperature_critical_c": ("GPU_FAULT_GPU_TEMP_CRITICAL_C"),
            "memory_temperature_warning_c": ("GPU_FAULT_MEMORY_TEMP_WARNING_C"),
            "memory_temperature_critical_c": ("GPU_FAULT_MEMORY_TEMP_CRITICAL_C"),
            "gpu_temperature_warning_margin_c": ("GPU_FAULT_GPU_TEMP_WARNING_MARGIN_C"),
            "gpu_temperature_shutdown_margin_c": (
                "GPU_FAULT_GPU_TEMP_SHUTDOWN_MARGIN_C"
            ),
            "memory_temperature_warning_margin_c": (
                "GPU_FAULT_MEMORY_TEMP_WARNING_MARGIN_C"
            ),
            "pcie_replay_rate_warning_per_minute": (
                "GPU_FAULT_PCIE_REPLAY_RATE_WARNING_PER_MINUTE"
            ),
            "nvlink_error_delta_critical": ("GPU_FAULT_NVLINK_ERROR_DELTA_CRITICAL"),
            "power_violation_delta_warning_us": (
                "GPU_FAULT_POWER_VIOLATION_DELTA_WARNING_US"
            ),
            "thermal_violation_delta_warning_us": (
                "GPU_FAULT_THERMAL_VIOLATION_DELTA_WARNING_US"
            ),
        }.items():
            raw = os.getenv(env_name)
            values[field_name] = (
                float(raw) if raw is not None else getattr(defaults, field_name)
            )
        raw_consecutive = os.getenv(
            "GPU_FAULT_THERMAL_VIOLATION_DRAIN_CONSECUTIVE_SAMPLES"
        )
        values["thermal_violation_drain_consecutive_samples"] = (
            int(raw_consecutive)
            if raw_consecutive is not None
            else defaults.thermal_violation_drain_consecutive_samples
        )
        raw_window = os.getenv("GPU_FAULT_DCGM_CORRELATION_WINDOW_SECONDS")
        values["correlation_window_seconds"] = (
            int(raw_window)
            if raw_window is not None
            else defaults.correlation_window_seconds
        )
        raw_consecutive = os.getenv("GPU_FAULT_DCGM_COMPOSITE_CONSECUTIVE_SAMPLES")
        values["composite_consecutive_samples"] = (
            int(raw_consecutive)
            if raw_consecutive is not None
            else defaults.composite_consecutive_samples
        )
        raw_power_ratio = os.getenv("GPU_FAULT_DCGM_POWER_LIMIT_RATIO")
        values["power_limit_ratio"] = (
            float(raw_power_ratio)
            if raw_power_ratio is not None
            else defaults.power_limit_ratio
        )
        raw_power_utilization = os.getenv(
            "GPU_FAULT_DCGM_POWER_CORRELATION_MIN_UTILIZATION_PERCENT"
        )
        values["power_correlation_min_utilization_percent"] = (
            float(raw_power_utilization)
            if raw_power_utilization is not None
            else defaults.power_correlation_min_utilization_percent
        )
        for field_name, env_name in {
            "ecc_sbe_delta_warning": ("GPU_FAULT_DCGM_ECC_SBE_DELTA_WARNING"),
            "retired_pages_sbe_delta_warning": (
                "GPU_FAULT_DCGM_RETIRED_PAGES_SBE_DELTA_WARNING"
            ),
            "retired_pages_dbe_delta_critical": (
                "GPU_FAULT_DCGM_RETIRED_PAGES_DBE_DELTA_CRITICAL"
            ),
            "row_remap_correctable_delta_warning": (
                "GPU_FAULT_DCGM_ROW_REMAP_CORRECTABLE_DELTA_WARNING"
            ),
        }.items():
            raw = os.getenv(env_name)
            values[field_name] = (
                float(raw) if raw is not None else getattr(defaults, field_name)
            )
        raw_correctable_consecutive = os.getenv(
            "GPU_FAULT_DCGM_CORRECTABLE_MEMORY_DRAIN_CONSECUTIVE_SAMPLES"
        )
        values["correctable_memory_drain_consecutive_samples"] = (
            int(raw_correctable_consecutive)
            if raw_correctable_consecutive is not None
            else defaults.correctable_memory_drain_consecutive_samples
        )
        return cls(**values)


class GpuMetricLatest(StrictModel):
    cluster_id: str
    node_id: str
    observed_at: datetime
    source: GpuMetricSource
    sample: GpuMetricSample


class GpuMetricsIngestionResult(StrictModel):
    batch_id: str
    accepted_samples: int
    duplicate: bool = False
    findings: list[GpuHealthFinding] = Field(default_factory=list)
    new_findings: list[GpuHealthFinding] = Field(default_factory=list)
    composite_findings: list[GpuHealthFinding] = Field(default_factory=list)
    new_composite_findings: list[GpuHealthFinding] = Field(default_factory=list)
    suppressed_finding_ids: list[str] = Field(default_factory=list)
    xid_events: list[XidEvent] = Field(default_factory=list)
    decisions: list[FaultPolicyDecision] = Field(default_factory=list)


class GpuMetricsService:
    """Keeps latest telemetry and evaluates evidence-only health rules."""

    _COUNTERS = {
        "ecc_sbe_volatile_total",
        "ecc_dbe_volatile_total",
        "ecc_sbe_aggregate_total",
        "ecc_dbe_aggregate_total",
        "retired_pages_sbe_total",
        "retired_pages_dbe_total",
        "pcie_replay_total",
        "nvlink_crc_flit_error_total",
        "nvlink_crc_data_error_total",
        "nvlink_replay_error_total",
        "nvlink_recovery_error_total",
        "nvlink_crc_aggregate_error_total",
        "nvlink_recovery_aggregate_error_total",
        "nvlink_replay_aggregate_error_total",
        "power_violation_total_us",
        "thermal_violation_total_us",
        "row_remap_correctable_total",
        "row_remap_uncorrectable_total",
    }
    _TEMPERATURE_LIMITS = {
        "gpu_slowdown_temperature_c",
        "gpu_shutdown_temperature_c",
        "gpu_max_operating_temperature_c",
        "memory_max_operating_temperature_c",
    }
    _COMPOSITE_RULE_IDS = {
        "THERMAL_STRESS",
        "POWER_LIMIT_THROTTLING",
        "GPU_MEMORY_DEGRADATION",
        "PCIE_XID_LINK_FAILURE",
        "NVLINK_LINK_DEGRADATION",
        "MULTI_GPU_NVLINK_FABRIC_FAILURE",
        "CORRECTABLE_MEMORY_DEGRADATION",
    }

    def __init__(
        self,
        thresholds: GpuMetricsThresholds | None = None,
        *,
        store=None,
    ) -> None:
        if store is None:
            from gpu_fault.store import InMemoryStore

            store = InMemoryStore()
        self.store = store
        self.thresholds = thresholds or GpuMetricsThresholds()
        self._node_locks = tuple(RLock() for _ in range(64))
        # New findings that deliberately end without an incident of their own,
        # by reason. The only reason today is a component activated in the same
        # batch as the composite that explains it (F-M1: a CRITICAL finding
        # either becomes an incident or is counted here).
        self.findings_without_incident: Counter[str] = Counter()

    def ingest(self, batch: GpuMetricBatch) -> GpuMetricsIngestionResult:
        findings: list[GpuHealthFinding] = []
        new_findings: list[GpuHealthFinding] = []
        xid_events: list[XidEvent] = []
        accepted = 0
        touched_gpu_keys: set[str] = set()
        node_lock = self._node_locks[
            hash((batch.cluster_id, batch.node_id)) % len(self._node_locks)
        ]
        with node_lock:
            batch_key = (
                batch.cluster_id,
                batch.node_id,
                batch.batch_id,
            )
            existing = self.store.get_gpu_metrics_batch(batch_key)
            if existing is not None:
                return existing.model_copy(update={"duplicate": True})
            temperature_limits = self._temperature_limits(batch)
            candidates = []
            for sample in batch.samples:
                if not math.isfinite(sample.value):
                    continue
                gpu_key = (
                    sample.gpu_uuid or sample.pci_bdf or sample.gpu_index or "node"
                )
                key = (
                    batch.cluster_id,
                    batch.node_id,
                    gpu_key,
                    sample.canonical_name,
                )
                latest = GpuMetricLatest(
                    cluster_id=batch.cluster_id,
                    node_id=batch.node_id,
                    observed_at=batch.observed_at,
                    source=batch.source,
                    sample=sample,
                )
                candidates.append((sample, gpu_key, key, latest))
            previous_latest_values = self.store.observe_gpu_metrics(
                [(key, latest) for _, _, key, latest in candidates]
            )
            accepted_candidates = [
                (candidate, previous_latest)
                for candidate, previous_latest in zip(
                    candidates,
                    previous_latest_values,
                    strict=True,
                )
                if previous_latest is not False
            ]
            completeness_key = collection_error_key(batch)
            *previous_finding_states, previous_completeness = (
                self.store.get_gpu_finding_states(
                    [
                        key
                        for (
                            _sample,
                            _gpu_key,
                            key,
                            _latest,
                        ), _previous_latest in accepted_candidates
                    ]
                    + [completeness_key]
                )
            )
            finding_updates: list[
                tuple[tuple[str, str, str, str], GpuHealthFinding | None, datetime]
            ] = []
            findings_by_update: list[GpuHealthFinding | None] = []
            collection_error = collection_error_finding(batch)
            if collection_error is not None:
                error_key, error_finding = collection_error
                findings.append(error_finding)
                finding_updates.append((error_key, error_finding, batch.observed_at))
                findings_by_update.append(error_finding)
            elif (
                previous_completeness is not None
                and previous_completeness.finding is not None
            ):
                # A clean scrape clears the collection-error finding. Without
                # this the WARNING a collector raised while dcgm-exporter was
                # still starting after a reboot stayed active for good, and
                # VALIDATE_GPU refused every validated restore of that node
                # with active_gpu_health_findings (DESTR-014 attempt 10,
                # 2026-09-09).
                finding_updates.append((completeness_key, None, batch.observed_at))
                findings_by_update.append(None)
            state_by_key = {}
            for (
                (
                    sample,
                    gpu_key,
                    key,
                    _latest,
                ),
                previous_latest,
            ), initial_finding_state in zip(
                accepted_candidates,
                previous_finding_states,
                strict=True,
            ):
                accepted += 1
                touched_gpu_keys.add(gpu_key)
                previous = (
                    previous_latest.sample.value
                    if isinstance(previous_latest, GpuMetricLatest)
                    else None
                )
                elapsed_seconds = (
                    (batch.observed_at - previous_latest.observed_at).total_seconds()
                    if isinstance(previous_latest, GpuMetricLatest)
                    else None
                )
                delta = (
                    max(0.0, sample.value - previous)
                    if previous is not None and sample.canonical_name in self._COUNTERS
                    else None
                )
                rate_per_minute = (
                    delta * 60.0 / elapsed_seconds
                    if delta is not None
                    and elapsed_seconds is not None
                    and elapsed_seconds > 0
                    else None
                )
                previous_finding_state = state_by_key.get(key, initial_finding_state)
                finding = self._evaluate(
                    batch,
                    sample,
                    delta,
                    rate_per_minute,
                    gpu_key,
                    temperature_limits.get(gpu_key, {}),
                    previous_finding_state,
                )
                if finding is not None:
                    findings.append(finding)
                if finding is not None or (
                    previous_finding_state is not None
                    and previous_finding_state.finding is not None
                ):
                    finding_updates.append((key, finding, batch.observed_at))
                    findings_by_update.append(finding)
                state_by_key[key] = GpuFindingState(
                    observed_at=batch.observed_at,
                    finding=finding,
                    consecutive_breaches=(
                        previous_finding_state.consecutive_breaches + 1
                        if finding is not None
                        and previous_finding_state is not None
                        and previous_finding_state.finding is not None
                        else 1
                        if finding is not None
                        else 0
                    ),
                )
                xid = self._xid_event(batch, sample, gpu_key)
                if xid is not None:
                    xid_events.append(xid)
            activated_values = self.store.update_gpu_findings(finding_updates)
            new_findings.extend(
                finding
                for finding, activated in zip(
                    findings_by_update,
                    activated_values,
                    strict=True,
                )
                if activated and finding is not None
            )
            (
                composite_findings,
                new_composite_findings,
                suppressed_finding_ids,
            ) = self._evaluate_composite_findings(
                batch,
                touched_gpu_keys,
            )
            suppressed = set(suppressed_finding_ids)
            effective_new_findings = [
                finding
                for finding in new_findings
                if finding.finding_id not in suppressed
            ]
            self.findings_without_incident["suppressed_by_composite"] += len(
                new_findings
            ) - len(effective_new_findings)
            effective_new_findings.extend(new_composite_findings)
            result = GpuMetricsIngestionResult(
                batch_id=batch.batch_id,
                accepted_samples=accepted,
                findings=findings,
                new_findings=effective_new_findings,
                composite_findings=composite_findings,
                new_composite_findings=(new_composite_findings),
                suppressed_finding_ids=suppressed_finding_ids,
                xid_events=xid_events,
            )
            return self.store.save_gpu_metrics_batch(batch_key, result)

    def _temperature_limits(
        self,
        batch: GpuMetricBatch,
    ) -> dict[str, dict[str, float]]:
        result: dict[str, dict[str, float]] = {}
        for item in batch.samples:
            if item.canonical_name not in self._TEMPERATURE_LIMITS:
                continue
            gpu_key = item.gpu_uuid or item.pci_bdf or item.gpu_index or "node"
            result.setdefault(gpu_key, {})[item.canonical_name] = item.value
        return result

    @staticmethod
    def _finding_gpu_key(finding: GpuHealthFinding) -> str:
        return finding.gpu_uuid or finding.pci_bdf or "node"

    def _evaluate_composite_findings(
        self,
        batch: GpuMetricBatch,
        touched_gpu_keys: set[str],
    ) -> tuple[
        list[GpuHealthFinding],
        list[GpuHealthFinding],
        list[str],
    ]:
        if not touched_gpu_keys:
            return [], [], []
        composite_scopes = sorted({*touched_gpu_keys, "node"})
        composite_keys = [
            self._composite_key(batch, scope_key, rule_id)
            for scope_key in composite_scopes
            for rule_id in sorted(self._COMPOSITE_RULE_IDS)
        ]
        previous_composite_states = dict(
            zip(
                composite_keys,
                self.store.get_gpu_finding_states(composite_keys),
                strict=True,
            )
        )
        cutoff = batch.observed_at - timedelta(
            seconds=self.thresholds.correlation_window_seconds
        )
        active_components = [
            finding
            for finding in self.store.list_gpu_findings(
                batch.cluster_id,
                batch.node_id,
                active_only=True,
            )
            if finding.finding_kind == "METRIC" and finding.observed_at >= cutoff
        ]
        latest = [
            item
            for item in self.store.list_gpu_metrics_latest(
                batch.cluster_id, batch.node_id
            )
            if item.observed_at >= cutoff
        ]
        findings_by_gpu: dict[str, dict[str, GpuHealthFinding]] = {}
        latest_by_gpu: dict[str, dict[str, GpuMetricLatest]] = {}
        for finding in active_components:
            findings_by_gpu.setdefault(self._finding_gpu_key(finding), {})[
                finding.canonical_name
            ] = finding
        for item in latest:
            gpu_key = (
                item.sample.gpu_uuid
                or item.sample.pci_bdf
                or item.sample.gpu_index
                or "node"
            )
            latest_by_gpu.setdefault(gpu_key, {})[item.sample.canonical_name] = item

        candidates: dict[tuple[str, str], GpuHealthFinding] = {}
        node_nvlink_components = [
            finding
            for finding in active_components
            if finding.canonical_name.startswith("nvlink_")
            and finding.canonical_name.endswith("_error_total")
            and finding.gpu_uuid
        ]
        nvlink_gpu_uuids = sorted(
            {finding.gpu_uuid for finding in node_nvlink_components if finding.gpu_uuid}
        )
        if len(nvlink_gpu_uuids) >= 2:
            candidates[("node", "MULTI_GPU_NVLINK_FABRIC_FAILURE")] = (
                self._composite_finding(
                    batch,
                    scope_key="node",
                    rule_id="MULTI_GPU_NVLINK_FABRIC_FAILURE",
                    components=node_nvlink_components,
                    component_metrics=sorted(
                        {finding.canonical_name for finding in node_nvlink_components}
                    ),
                    severity=GpuHealthSeverity.CRITICAL,
                    action=RecoveryAction.DRAIN,
                    reason=(
                        "correlated NVLink errors affected multiple GPUs; "
                        "treat as a node fabric or NVSwitch failure"
                    ),
                    affected_gpu_uuids=nvlink_gpu_uuids,
                    confidence="HIGH",
                )
            )

        for gpu_key in sorted(touched_gpu_keys):
            self._add_gpu_composite_candidates(
                batch,
                gpu_key,
                findings_by_gpu.get(gpu_key, {}),
                latest_by_gpu.get(gpu_key, {}),
                previous_composite_states,
                nvlink_gpu_uuids,
                candidates,
            )

        composite_findings = []
        new_composite_findings = []
        suppressed_ids: set[str] = set()
        ordered_rule_ids = sorted(self._COMPOSITE_RULE_IDS)
        updates = []
        candidate_updates = []
        for scope_key in composite_scopes:
            for rule_id in ordered_rule_ids:
                key = self._composite_key(batch, scope_key, rule_id)
                candidate = candidates.get((scope_key, rule_id))
                previous = previous_composite_states.get(key)
                if candidate is None and (previous is None or previous.finding is None):
                    continue
                updates.append((key, candidate, batch.observed_at))
                if candidate is not None:
                    candidate_updates.append((key, candidate))
        activated_values = self.store.update_gpu_findings(updates)
        activated_by_key = {
            key: activated
            for (
                key,
                _candidate,
                _observed_at,
            ), activated in zip(updates, activated_values, strict=True)
        }
        for key, candidate in candidate_updates:
            if candidate is None:
                continue
            composite_findings.append(candidate)
            suppressed_ids.update(candidate.component_finding_ids)
            if activated_by_key.get(key, False):
                new_composite_findings.append(candidate)
        return (
            composite_findings,
            new_composite_findings,
            sorted(suppressed_ids),
        )

    def _add_gpu_composite_candidates(
        self,
        batch,
        gpu_key,
        components,
        metrics,
        previous_composite_states,
        nvlink_gpu_uuids,
        candidates,
    ) -> None:
        self._add_thermal_composite(
            batch,
            gpu_key,
            components,
            metrics,
            previous_composite_states,
            candidates,
        )
        self._add_memory_composites(
            batch,
            gpu_key,
            components,
            previous_composite_states,
            candidates,
        )
        self._add_pcie_composite(batch, gpu_key, components, metrics, candidates)
        self._add_power_composite(batch, gpu_key, components, metrics, candidates)
        self._add_nvlink_composite(
            batch,
            gpu_key,
            components,
            nvlink_gpu_uuids,
            candidates,
        )

    def _add_thermal_composite(
        self,
        batch,
        gpu_key,
        components,
        metrics,
        previous_composite_states,
        candidates,
    ) -> None:
        temperature = next(
            (
                components[name]
                for name in {
                    "gpu_temperature_c",
                    "memory_temperature_c",
                }
                if name in components
            ),
            None,
        )
        thermal = components.get("thermal_violation_total_us")
        thermal_throttle = components.get("clock_throttle_reasons")
        thermal_signals = [
            signal for signal in [thermal, thermal_throttle] if signal is not None
        ]
        if temperature is not None and thermal_signals:
            previous = previous_composite_states.get(
                self._composite_key(batch, gpu_key, "THERMAL_STRESS")
            )
            consecutive = (
                previous.consecutive_breaches + 1
                if previous is not None and previous.finding is not None
                else 1
            )
            critical = consecutive >= self.thresholds.composite_consecutive_samples
            candidates[(gpu_key, "THERMAL_STRESS")] = self._composite_finding(
                batch,
                scope_key=gpu_key,
                rule_id="THERMAL_STRESS",
                components=[
                    temperature,
                    *thermal_signals,
                ],
                component_metrics=list(
                    dict.fromkeys(
                        [
                            temperature.canonical_name,
                            *(signal.canonical_name for signal in thermal_signals),
                            *(
                                name
                                for name in {
                                    "sm_clock_mhz",
                                    "memory_clock_mhz",
                                }
                                if name in metrics
                            ),
                        ]
                    )
                ),
                severity=(
                    GpuHealthSeverity.CRITICAL
                    if critical
                    else GpuHealthSeverity.WARNING
                ),
                action=(
                    RecoveryAction.DRAIN if critical else RecoveryAction.RUN_DIAGNOSTICS
                ),
                reason=(
                    "temperature threshold and thermal "
                    "violation or thermal clock throttling "
                    "were correlated for "
                    f"{consecutive} consecutive samples; "
                    "clock context: "
                    + ", ".join(
                        f"{name}={metrics[name].sample.value}"
                        for name in {
                            "sm_clock_mhz",
                            "memory_clock_mhz",
                        }
                        if name in metrics
                    )
                ),
                confidence="HIGH",
            )

    def _add_memory_composites(
        self,
        batch,
        gpu_key,
        components,
        previous_composite_states,
        candidates,
    ) -> None:
        memory_errors = [
            components[name]
            for name in {
                "ecc_dbe_volatile_total",
                "ecc_dbe_aggregate_total",
                "row_remap_uncorrectable_total",
            }
            if name in components
        ]
        memory_repair = [
            components[name]
            for name in {
                "row_remap_pending",
                "row_remap_failure",
                "retired_pages_pending",
                "retired_pages_dbe_total",
            }
            if name in components
        ]
        if memory_errors and memory_repair:
            candidates[(gpu_key, "GPU_MEMORY_DEGRADATION")] = self._composite_finding(
                batch,
                scope_key=gpu_key,
                rule_id="GPU_MEMORY_DEGRADATION",
                components=[
                    *memory_errors,
                    *memory_repair,
                ],
                component_metrics=sorted(
                    {
                        finding.canonical_name
                        for finding in [
                            *memory_errors,
                            *memory_repair,
                        ]
                    }
                ),
                severity=GpuHealthSeverity.CRITICAL,
                action=RecoveryAction.DRAIN,
                reason=(
                    "uncorrectable memory errors and pending or "
                    "failed memory repair were correlated"
                ),
                confidence="HIGH",
            )

        correctable_memory = [
            components[name]
            for name in {
                "ecc_sbe_volatile_total",
                "ecc_sbe_aggregate_total",
                "retired_pages_sbe_total",
                "row_remap_correctable_total",
            }
            if name in components
        ]
        if len({finding.canonical_name for finding in correctable_memory}) >= 2:
            previous = previous_composite_states.get(
                self._composite_key(
                    batch,
                    gpu_key,
                    "CORRECTABLE_MEMORY_DEGRADATION",
                )
            )
            consecutive = (
                previous.consecutive_breaches + 1
                if previous is not None and previous.finding is not None
                else 1
            )
            critical = (
                consecutive
                >= self.thresholds.correctable_memory_drain_consecutive_samples
            )
            candidates[
                (
                    gpu_key,
                    "CORRECTABLE_MEMORY_DEGRADATION",
                )
            ] = self._composite_finding(
                batch,
                scope_key=gpu_key,
                rule_id="CORRECTABLE_MEMORY_DEGRADATION",
                components=correctable_memory,
                component_metrics=sorted(
                    {finding.canonical_name for finding in correctable_memory}
                ),
                severity=(
                    GpuHealthSeverity.CRITICAL
                    if critical
                    else GpuHealthSeverity.WARNING
                ),
                action=(
                    RecoveryAction.DRAIN if critical else RecoveryAction.RUN_DIAGNOSTICS
                ),
                reason=(
                    "multiple correctable memory degradation "
                    "signals increased for "
                    f"{consecutive} consecutive samples"
                ),
                confidence="MEDIUM",
            )

    def _add_pcie_composite(
        self,
        batch,
        gpu_key,
        components,
        metrics,
        candidates,
    ) -> None:
        pcie = components.get("pcie_replay_total")
        xid = metrics.get("xid_last_error")
        if pcie is not None and xid is not None and int(xid.sample.value) in {32, 79}:
            candidates[(gpu_key, "PCIE_XID_LINK_FAILURE")] = self._composite_finding(
                batch,
                scope_key=gpu_key,
                rule_id="PCIE_XID_LINK_FAILURE",
                components=[pcie],
                component_metrics=[
                    "pcie_replay_total",
                    "xid_last_error",
                ],
                severity=GpuHealthSeverity.CRITICAL,
                action=RecoveryAction.DRAIN,
                reason=(
                    "PCIe replay threshold was correlated with "
                    f"XID {int(xid.sample.value)}"
                ),
                confidence="HIGH",
            )

    def _add_power_composite(
        self,
        batch,
        gpu_key,
        components,
        metrics,
        candidates,
    ) -> None:
        power = components.get("power_violation_total_us")
        power_usage = metrics.get("power_usage_w")
        power_limit = metrics.get("power_limit_w")
        utilization = metrics.get("gpu_utilization_percent")
        has_temperature_finding = any(
            name in components
            for name in {
                "gpu_temperature_c",
                "memory_temperature_c",
                "thermal_violation_total_us",
                "clock_throttle_reasons",
            }
        )
        if (
            power is not None
            and power_usage is not None
            and power_limit is not None
            and power_limit.sample.value > 0
            and power_usage.sample.value
            >= power_limit.sample.value * self.thresholds.power_limit_ratio
            and utilization is not None
            and utilization.sample.value
            >= self.thresholds.power_correlation_min_utilization_percent
            and not has_temperature_finding
        ):
            candidates[(gpu_key, "POWER_LIMIT_THROTTLING")] = self._composite_finding(
                batch,
                scope_key=gpu_key,
                rule_id="POWER_LIMIT_THROTTLING",
                components=[power],
                component_metrics=[
                    "power_violation_total_us",
                    "power_usage_w",
                    "power_limit_w",
                    "gpu_utilization_percent",
                ],
                severity=GpuHealthSeverity.WARNING,
                action=RecoveryAction.RUN_DIAGNOSTICS,
                reason=(
                    "power violation occurred near the enforced "
                    "power limit under high utilization without "
                    "a temperature finding"
                ),
                confidence="MEDIUM",
            )

    def _add_nvlink_composite(
        self,
        batch,
        gpu_key,
        components,
        nvlink_gpu_uuids,
        candidates,
    ) -> None:
        nvlink = [
            finding
            for name, finding in components.items()
            if name.startswith("nvlink_") and name.endswith("_error_total")
        ]
        if (
            len({item.canonical_name for item in nvlink}) >= 2
            and len(nvlink_gpu_uuids) < 2
        ):
            candidates[(gpu_key, "NVLINK_LINK_DEGRADATION")] = self._composite_finding(
                batch,
                scope_key=gpu_key,
                rule_id="NVLINK_LINK_DEGRADATION",
                components=nvlink,
                component_metrics=sorted(
                    {finding.canonical_name for finding in nvlink}
                ),
                severity=GpuHealthSeverity.CRITICAL,
                action=RecoveryAction.DRAIN,
                reason=("multiple NVLink error classes increased on the same GPU"),
                confidence="HIGH",
            )

    @staticmethod
    def _composite_key(
        batch: GpuMetricBatch,
        scope_key: str,
        rule_id: str,
    ) -> tuple[str, str, str, str]:
        return (
            batch.cluster_id,
            batch.node_id,
            scope_key,
            f"composite:{rule_id}",
        )

    def _composite_finding(
        self,
        batch: GpuMetricBatch,
        *,
        scope_key: str,
        rule_id: str,
        components: list[GpuHealthFinding],
        component_metrics: list[str],
        severity: GpuHealthSeverity,
        action: RecoveryAction,
        reason: str,
        confidence: str,
        affected_gpu_uuids: list[str] | None = None,
    ) -> GpuHealthFinding:
        gpu_uuids = sorted(
            {
                *(affected_gpu_uuids if affected_gpu_uuids is not None else []),
                *(component.gpu_uuid for component in components if component.gpu_uuid),
            }
        )
        gpu_uuid = gpu_uuids[0] if len(gpu_uuids) == 1 else None
        pci_bdf = next(
            (component.pci_bdf for component in components if component.pci_bdf),
            None,
        )
        evidence_refs = list(
            dict.fromkeys(
                component.evidence_ref
                for component in components
                if component.evidence_ref
            )
        )
        return GpuHealthFinding(
            finding_id=(f"{batch.batch_id}-{scope_key}-composite-{rule_id}"),
            cluster_id=batch.cluster_id,
            node_id=batch.node_id,
            observed_at=batch.observed_at,
            severity=severity,
            reason=reason,
            canonical_name=f"composite:{rule_id}",
            value=1,
            gpu_uuid=gpu_uuid,
            pci_bdf=pci_bdf,
            evidence_ref=(
                evidence_refs[0] if len(evidence_refs) == 1 else batch.evidence_ref
            ),
            automatic_action=action.value,
            policy_source="SITE_DCGM_CORRELATION",
            policy_version=SITE_CORRELATION_POLICY_VERSION,
            policy_reference=NVIDIA_DCGM_HEALTH_REFERENCE,
            runtime_profile_version=batch.runtime_profile_version,
            workload_state=batch.workload_state,
            affected_workload_ids=batch.affected_workload_ids,
            finding_kind="COMPOSITE",
            correlation_rule_id=rule_id,
            component_finding_ids=[component.finding_id for component in components],
            component_metrics=component_metrics,
            component_evidence_refs=evidence_refs,
            affected_gpu_uuids=gpu_uuids,
            confidence=confidence,
        )

    def latest(self, cluster_id: str, node_id: str) -> list[GpuMetricLatest]:
        result = self.store.list_gpu_metrics_latest(cluster_id, node_id)
        return sorted(
            result,
            key=lambda item: (
                item.sample.gpu_index or "",
                item.sample.canonical_name,
            ),
        )

    def findings(
        self,
        cluster_id: str,
        node_id: str,
        *,
        active_only: bool = True,
    ) -> list[GpuHealthFinding]:
        result = self.store.list_gpu_findings(
            cluster_id, node_id, active_only=active_only
        )
        return sorted(
            result,
            key=lambda item: item.observed_at,
            reverse=True,
        )

    def _evaluate(
        self,
        batch,
        sample,
        delta,
        rate_per_minute,
        gpu_key,
        temperature_limits,
        previous_finding_state,
    ) -> GpuHealthFinding | None:
        decision = power_violation_decision(
            self.thresholds, batch, sample, delta, gpu_key
        )
        if sample.canonical_name != "power_violation_total_us":
            for resolver in (
                self._temperature_decision,
                self._memory_decision,
                self._link_decision,
                self._throttle_decision,
            ):
                decision = resolver(
                    sample,
                    delta,
                    rate_per_minute,
                    temperature_limits,
                    previous_finding_state,
                )
                if decision is not None:
                    break
        if decision is None:
            return None
        return GpuHealthFinding(
            finding_id=f"{batch.batch_id}-{gpu_key}-{sample.canonical_name}",
            cluster_id=batch.cluster_id,
            node_id=batch.node_id,
            observed_at=batch.observed_at,
            severity=decision["severity"],
            reason=decision["reason"],
            canonical_name=sample.canonical_name,
            value=sample.value,
            delta=delta,
            rate_per_minute=rate_per_minute,
            threshold_value=decision.get("threshold_value"),
            threshold_source=decision.get("threshold_source"),
            gpu_uuid=sample.gpu_uuid,
            pci_bdf=sample.pci_bdf,
            evidence_ref=batch.evidence_ref,
            automatic_action=decision.get("automatic_action"),
            policy_source=decision.get("policy_source", "SITE_DCGM_METRIC"),
            policy_version=decision.get("policy_version", SITE_METRIC_POLICY_VERSION),
            policy_reference=decision.get("policy_reference"),
            official_action=decision.get("official_action"),
            runtime_profile_version=batch.runtime_profile_version,
            workload_state=batch.workload_state,
            affected_workload_ids=batch.affected_workload_ids,
        )

    def _temperature_decision(self, sample, _delta, _rate, limits, _previous):
        name = sample.canonical_name
        if name not in {
            "gpu_temperature_c",
            "memory_temperature_c",
        }:
            return None
        if name == "gpu_temperature_c":
            slowdown = limits.get("gpu_slowdown_temperature_c")
            shutdown = limits.get("gpu_shutdown_temperature_c")
            maximum = limits.get("gpu_max_operating_temperature_c")
            critical = (
                slowdown
                or maximum
                or (
                    shutdown - self.thresholds.gpu_temperature_shutdown_margin_c
                    if shutdown is not None
                    else None
                )
            )
            fallback_warning = self.thresholds.gpu_temperature_warning_c
            fallback_critical = self.thresholds.gpu_temperature_critical_c
            margin = self.thresholds.gpu_temperature_warning_margin_c
            label = "GPU"
        else:
            maximum = limits.get("memory_max_operating_temperature_c")
            critical = maximum
            fallback_warning = self.thresholds.memory_temperature_warning_c
            fallback_critical = self.thresholds.memory_temperature_critical_c
            margin = self.thresholds.memory_temperature_warning_margin_c
            label = "GPU memory"
        if critical is not None:
            warning = min(
                critical - margin,
                maximum if maximum is not None else critical,
            )
            source = "NVIDIA_DEVICE_LIMIT"
            policy = {
                "policy_source": "SITE_NVIDIA_DEVICE_LIMIT_DERIVED",
                "policy_reference": NVIDIA_NVML_TEMPERATURE_REFERENCE,
            }
        else:
            warning, critical, source, policy = (
                fallback_warning,
                fallback_critical,
                "CONFIGURED_FALLBACK",
                {},
            )
        severity = (
            GpuHealthSeverity.CRITICAL
            if sample.value >= critical
            else GpuHealthSeverity.WARNING
            if sample.value >= warning
            else None
        )
        if severity is None:
            return None
        return {
            "severity": severity,
            "reason": f"{label} temperature exceeded {source.lower()} threshold",
            "automatic_action": RecoveryAction.DRAIN.value
            if severity is GpuHealthSeverity.CRITICAL
            else RecoveryAction.RUN_DIAGNOSTICS.value,
            "threshold_value": critical
            if severity is GpuHealthSeverity.CRITICAL
            else warning,
            "threshold_source": source,
            **policy,
        }

    def _memory_decision(self, sample, delta, _rate, _limits, _previous):
        name = sample.canonical_name
        warning = RecoveryAction.RUN_DIAGNOSTICS.value
        cases = {
            "ecc_sbe_volatile_total": (
                delta is not None and delta >= self.thresholds.ecc_sbe_delta_warning,
                GpuHealthSeverity.WARNING,
                "correctable ECC counter increased",
                warning,
            ),
            "ecc_sbe_aggregate_total": (
                delta is not None and delta >= self.thresholds.ecc_sbe_delta_warning,
                GpuHealthSeverity.WARNING,
                "correctable ECC counter increased",
                warning,
            ),
            "ecc_dbe_volatile_total": (
                sample.value > 0,
                GpuHealthSeverity.CRITICAL,
                "uncorrectable ECC errors detected",
                RecoveryAction.DRAIN.value,
            ),
            "ecc_dbe_aggregate_total": (
                delta is not None and delta > 0,
                GpuHealthSeverity.WARNING,
                "aggregate uncorrectable ECC counter increased",
                warning,
            ),
            "row_remap_failure": (
                sample.value > 0,
                GpuHealthSeverity.CRITICAL,
                "GPU row remap failure reported",
                RecoveryAction.DRAIN.value,
            ),
            "row_remap_uncorrectable_total": (
                delta is not None and delta > 0,
                GpuHealthSeverity.WARNING,
                "uncorrectable GPU row remap errors detected",
                warning,
            ),
            "row_remap_correctable_total": (
                delta is not None
                and delta >= self.thresholds.row_remap_correctable_delta_warning,
                GpuHealthSeverity.WARNING,
                "correctable GPU row remap counter increased",
                warning,
            ),
            "retired_pages_sbe_total": (
                delta is not None
                and delta >= self.thresholds.retired_pages_sbe_delta_warning,
                GpuHealthSeverity.WARNING,
                "correctable ECC retired-page counter increased",
                warning,
            ),
            "retired_pages_dbe_total": (
                delta is not None
                and delta >= self.thresholds.retired_pages_dbe_delta_critical,
                GpuHealthSeverity.CRITICAL,
                "uncorrectable ECC retired-page counter increased",
                RecoveryAction.DRAIN.value,
            ),
            "row_remap_pending": (
                sample.value > 0,
                GpuHealthSeverity.WARNING,
                "GPU memory repair is pending",
                RecoveryAction.RESET_GPU.value,
            ),
            "retired_pages_pending": (
                sample.value > 0,
                GpuHealthSeverity.WARNING,
                "GPU memory repair is pending",
                RecoveryAction.RESET_GPU.value,
            ),
        }
        case = cases.get(name)
        if case is None or not case[0]:
            return None
        result = {
            "severity": case[1],
            "reason": case[2],
            "automatic_action": case[3],
        }
        thresholds = {
            "ecc_sbe_volatile_total": self.thresholds.ecc_sbe_delta_warning,
            "ecc_sbe_aggregate_total": self.thresholds.ecc_sbe_delta_warning,
            "row_remap_correctable_total": self.thresholds.row_remap_correctable_delta_warning,
            "retired_pages_sbe_total": self.thresholds.retired_pages_sbe_delta_warning,
            "retired_pages_dbe_total": self.thresholds.retired_pages_dbe_delta_critical,
        }
        if name in thresholds:
            result.update(
                threshold_value=thresholds[name],
                threshold_source="CONFIGURED_DELTA",
            )
        if name == "ecc_dbe_volatile_total":
            result.update(
                policy_source="NVIDIA_DCGM_HEALTH",
                policy_version=NVIDIA_DCGM_POLICY_VERSION,
                policy_reference=NVIDIA_DCGM_HEALTH_REFERENCE,
                official_action=("TERMINATE_JOB_AND_ANALYZE_GPU_HEALTH"),
            )
        if name == "row_remap_failure":
            result.update(
                policy_source=("NVIDIA_GPU_MEMORY_ERROR_MANAGEMENT"),
                policy_version=NVIDIA_GPU_MEMORY_POLICY_VERSION,
                policy_reference=NVIDIA_ROW_REMAP_RMA_REFERENCE,
                official_action="RUN_FIELD_DIAGNOSTIC_FOR_RMA",
            )
        if name in {
            "row_remap_pending",
            "retired_pages_pending",
        }:
            result.update(
                policy_source=("NVIDIA_GPU_MEMORY_ERROR_MANAGEMENT"),
                policy_version=NVIDIA_GPU_MEMORY_POLICY_VERSION,
                policy_reference=NVIDIA_ROW_REMAP_REFERENCE,
                official_action="RESET_GPU_IN_SERVICE_WINDOW",
            )
        if name in {
            "row_remap_correctable_total",
            "retired_pages_sbe_total",
        }:
            result["policy_reference"] = NVIDIA_ROW_REMAP_REFERENCE
        if name == "retired_pages_dbe_total":
            result["policy_reference"] = NVIDIA_ROW_REMAP_RMA_REFERENCE
        return result

    def _link_decision(self, sample, delta, rate, _limits, _previous):
        name = sample.canonical_name
        if (
            name == "pcie_replay_total"
            and rate is not None
            and rate > self.thresholds.pcie_replay_rate_warning_per_minute
        ):
            source = (
                "NVIDIA_DCGM_HEALTH"
                if self.thresholds.pcie_replay_rate_warning_per_minute == 8
                else "SITE_OVERRIDE_NVIDIA_DCGM_HEALTH"
            )
            return {
                "severity": GpuHealthSeverity.WARNING,
                "reason": "PCIe replay rate exceeded DCGM health threshold",
                "automatic_action": RecoveryAction.RUN_DIAGNOSTICS.value,
                "policy_source": source,
                "policy_version": NVIDIA_DCGM_POLICY_VERSION
                if source == "NVIDIA_DCGM_HEALTH"
                else SITE_METRIC_POLICY_VERSION,
                "policy_reference": NVIDIA_DCGM_HEALTH_REFERENCE,
                "official_action": "EXAMINE_GPU_HEALTH",
            }
        if (
            name.startswith("nvlink_")
            and name.endswith("_error_total")
            and delta is not None
            and delta >= self.thresholds.nvlink_error_delta_critical
        ):
            source = (
                "NVIDIA_DCGM_HEALTH"
                if self.thresholds.nvlink_error_delta_critical == 1
                else "SITE_OVERRIDE_NVIDIA_DCGM_HEALTH"
            )
            return {
                "severity": GpuHealthSeverity.CRITICAL,
                "reason": "NVLink error counter increased",
                "automatic_action": RecoveryAction.DRAIN.value,
                "policy_source": source,
                "policy_version": NVIDIA_DCGM_POLICY_VERSION
                if source == "NVIDIA_DCGM_HEALTH"
                else SITE_METRIC_POLICY_VERSION,
                "policy_reference": NVIDIA_DCGM_HEALTH_REFERENCE,
                "official_action": "TERMINATE_JOB_AND_ANALYZE_GPU_HEALTH",
            }
        return None

    def _throttle_decision(self, sample, delta, rate, _limits, previous):
        name = sample.canonical_name
        if (
            name == "clock_throttle_reasons"
            and int(sample.value) & THERMAL_CLOCK_THROTTLE_MASK
        ):
            return {
                "severity": GpuHealthSeverity.WARNING,
                "reason": "DCGM reports software or hardware thermal clock throttling",
                "automatic_action": RecoveryAction.RUN_DIAGNOSTICS.value,
                "threshold_value": float(THERMAL_CLOCK_THROTTLE_MASK),
                "threshold_source": "NVIDIA_THROTTLE_REASON_BITMASK",
                "policy_source": "SITE_NVIDIA_DEVICE_LIMIT_DERIVED",
                "policy_reference": NVIDIA_DCGM_HEALTH_REFERENCE,
                "official_action": "EXAMINE_GPU_HEALTH",
            }
        limits = {
            "thermal_violation_total_us": self.thresholds.thermal_violation_delta_warning_us
        }
        if name not in limits or delta is None or delta < limits[name]:
            return None
        if rate is not None and rate > MAX_PLAUSIBLE_VIOLATION_US_PER_MINUTE:
            # The counter claims more throttled time than has elapsed, so it is
            # not microseconds on this device. Two consecutive breaches here
            # escalate to DRAIN, so grading an impossible rate evicts healthy
            # nodes on the strength of a counter nobody can interpret.
            return None
        threshold = limits[name]
        severity, reason, action = (
            GpuHealthSeverity.WARNING,
            "GPU thermal violation duration increased",
            RecoveryAction.RUN_DIAGNOSTICS.value,
        )
        breaches = (
            previous.consecutive_breaches
            if previous is not None and previous.finding is not None
            else 0
        ) + 1
        if breaches >= self.thresholds.thermal_violation_drain_consecutive_samples:
            severity, reason, action = (
                GpuHealthSeverity.CRITICAL,
                f"GPU thermal violation duration increased for {breaches} consecutive samples",
                RecoveryAction.DRAIN.value,
            )
        source = (
            "NVIDIA_DCGM_HEALTH"
            if threshold == 1
            else "SITE_OVERRIDE_NVIDIA_DCGM_HEALTH"
        )
        return {
            "severity": severity,
            "reason": reason,
            "automatic_action": action,
            "threshold_value": threshold,
            "threshold_source": "CONFIGURED_DELTA",
            "policy_source": source,
            "policy_version": NVIDIA_DCGM_POLICY_VERSION
            if source == "NVIDIA_DCGM_HEALTH"
            else SITE_METRIC_POLICY_VERSION,
            "policy_reference": NVIDIA_DCGM_HEALTH_REFERENCE,
            "official_action": "EXAMINE_GPU_HEALTH",
        }

    def _xid_event(
        self,
        batch: GpuMetricBatch,
        sample: GpuMetricSample,
        gpu_key: str,
    ) -> XidEvent | None:
        if sample.canonical_name != "xid_last_error":
            return None
        if sample.value < 0 or not sample.value.is_integer():
            return None
        xid = int(sample.value)
        # DCGM exposes the last XID as persistent state. The first scrape
        # establishes a baseline; raw kernel events remain the fresh source.
        # The store claims the transition atomically across API replicas.
        if not self.store.observe_xid_metric(
            batch.cluster_id,
            batch.node_id,
            gpu_key,
            xid,
            batch.observed_at,
        ):
            return None
        return XidEvent(
            event_id=f"{batch.batch_id}-{gpu_key}-xid-{xid}",
            cluster_id=batch.cluster_id,
            node_id=batch.node_id,
            observed_at=batch.observed_at,
            source_event_time=batch.observed_at,
            collected_at=batch.collected_at or batch.observed_at,
            ingested_at=batch.ingested_at,
            event_source=GpuMetricSource.DCGM_EXPORTER.value,
            xid=xid,
            gpu_uuid=sample.gpu_uuid,
            pci_bdf=sample.pci_bdf,
            product=batch.product or sample.labels.get("modelName"),
            driver_branch=batch.driver_branch,
            cuda_version=batch.cuda_version,
            runtime_profile_version=batch.runtime_profile_version,
            workload_state=batch.workload_state,
            affected_workload_ids=batch.affected_workload_ids,
            checkpoint_manifest_ref=batch.checkpoint_manifest_ref,
            raw_message=(f"DCGM_FI_DEV_XID_ERRORS={xid} gpu={gpu_key}"),
            evidence_ref=batch.evidence_ref,
        )
