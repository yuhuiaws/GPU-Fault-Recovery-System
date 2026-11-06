from __future__ import annotations

import logging
import os
import re
from collections.abc import Mapping
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4

from pydantic import (
    Field,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
)

from gpu_fault.device_plugin_liveness import kubernetes_device_plugin_rules
from gpu_fault.efa_traffic_state import traffic_observation
from gpu_fault.env import env_bool
from gpu_fault.efa_stall_episode import episode_parameters
from gpu_fault.gpu_field_rules import GPU_FIELD_FAULT_CLASSES
from gpu_fault.gpu_instance_inventory import gpu_instance_inventory
from gpu_fault.host_capability_markers import CAPABILITY_MARKERS
from gpu_fault.host_device_signals import (
    age_out_vanished_device_signals,
    device_gone,
    device_signal_age_out_seconds,
)
from gpu_fault.host_low_utilization import (
    BatchManagedAttempt,
    UnresolvedAllocationWarnings,
    resolve_managed_attempt,
)
from gpu_fault.hung_lifecycle import cleanup_reason
from gpu_fault.log_recovery_context import (
    log_context,
    log_event_id,
    log_rule_decision,
)
from gpu_fault.log_rules import NODE_LOG_RULES
from gpu_fault.models import (
    EfaTrafficSignal,
    MarkerScope,
    NodeMarker,
    RecoveryAction,
    Severity,
    StrictModel,
    WorkloadState,
    recovery_action_sort_key,
)
from gpu_fault.store.shared.health_signals import (
    finding_health_signal_fingerprint,
    health_signal_key,
)

LOGGER = logging.getLogger(__name__)


class HostMetricSample(StrictModel):
    name: str
    value: float
    unit: str | None = None
    device: str | None = None
    labels: dict[str, str] = Field(default_factory=dict)


class HostTelemetryHistoryPoint(StrictModel):
    observed_at: datetime
    samples: list[HostMetricSample]
    collection_errors: list[str] = Field(default_factory=list)


class HostTelemetryBatch(StrictModel):
    batch_id: str = Field(default_factory=lambda: f"host-telemetry-{uuid4()}")
    # A misconfigured collector that resolves an empty node id would
    # otherwise store evidence and latest metrics under a phantom node
    # that no recovery path can ever act on.
    cluster_id: str = Field(min_length=1)
    node_id: str = Field(min_length=1)
    observed_at: datetime
    # When the control plane accepted this batch (F-M2). Sustained-signal
    # windows are measured on it; ``observed_at`` is the node's clock.
    received_at: datetime | None = None
    samples: list[HostMetricSample]
    collection_errors: list[str] = Field(default_factory=list)
    runtime_profile_version: str | None = None
    workload_state: WorkloadState = WorkloadState.UNKNOWN
    affected_workload_ids: list[str] = Field(default_factory=list)
    evidence_ref: str | None = None
    edge_filter_reasons: list[str] = Field(default_factory=list)
    context_history: list[HostTelemetryHistoryPoint] = Field(default_factory=list)
    # Who read the host. The node's own collector is the only producer whose
    # batches say anything about that collector's health; the control plane's
    # Kubernetes reader posts EFA/GPU allocatable counts on this same channel
    # for the same node, and its clean, fresher batches used to overwrite the
    # node's HOST_TELEMETRY status row -- erasing the collection errors and,
    # because a node tick under a hung nvidia-smi is stamped earlier than the
    # reader's snapshot, dropping the node's own status as a replay. A dead
    # host collector looked alive the same way. Older node collectors omit the
    # field and default to the node.
    producer: Literal["node", "control-plane"] = "node"


class NodeLogEntry(StrictModel):
    entry_id: str
    source: str
    observed_at: datetime
    message: str
    priority: int | None = None
    unit: str | None = None
    fields: dict[str, str] = Field(default_factory=dict)


class NodeLogBatch(StrictModel):
    batch_id: str = Field(default_factory=lambda: f"node-logs-{uuid4()}")
    cluster_id: str = Field(min_length=1)
    node_id: str = Field(min_length=1)
    collected_at: datetime
    entries: list[NodeLogEntry]
    runtime_profile_version: str | None = None
    workload_state: WorkloadState = WorkloadState.UNKNOWN
    affected_workload_ids: list[str] = Field(default_factory=list)
    edge_filter_reasons: list[str] = Field(default_factory=list)
    # What the collector could not read for this batch: a journalctl that exited
    # non-zero, a window it had to move forward, a rotated training log. Without
    # this an empty `entries` means "the node is quiet" and "the node's log
    # collection is broken" equally well, and the second one used to be silent.
    collection_errors: list[str] = Field(default_factory=list)


class NodeHealthCategory(StrEnum):
    GPU = "GPU"
    CPU = "CPU"
    MEMORY = "MEMORY"
    STORAGE = "STORAGE"
    NETWORK = "NETWORK"
    RDMA = "RDMA"
    MCE = "MCE"
    NCCL = "NCCL"
    TRAINING = "TRAINING"
    SYSTEM_LOG = "SYSTEM_LOG"
    BMC = "BMC"


_GPU_METRIC_FAULT_CLASSES = {
    "gpu_temperature_c": "GPU_THERMAL",
    "gpu_temperature_celsius": "GPU_THERMAL",
    "memory_temperature_c": "GPU_THERMAL",
    "gpu_slowdown_temperature_c": "GPU_THERMAL",
    "gpu_shutdown_temperature_c": "GPU_THERMAL",
    "gpu_max_operating_temperature_c": "GPU_THERMAL",
    "memory_max_operating_temperature_c": "GPU_THERMAL",
    # The T.Limit scale (gpu_temperature_policy.py): the reading and its limits.
    "gpu_tlimit_c": "GPU_THERMAL",
    "gpu_slowdown_tlimit_c": "GPU_THERMAL",
    "gpu_shutdown_tlimit_c": "GPU_THERMAL",
    "gpu_max_operating_tlimit_c": "GPU_THERMAL",
    "memory_max_operating_tlimit_c": "GPU_THERMAL",
    "thermal_violation_total_us": "GPU_THERMAL",
    "power_violation_total_us": "GPU",
    "power_usage_w": "GPU",
    "power_limit_w": "GPU",
    "ecc_sbe_volatile_total": "GPU_MEMORY",
    "ecc_dbe_volatile_total": "GPU_MEMORY",
    "ecc_sbe_aggregate_total": "GPU_MEMORY",
    "ecc_dbe_aggregate_total": "GPU_MEMORY",
    "dcgm_ecc_dbe_delta": "GPU_MEMORY",
    "retired_pages_sbe_total": "GPU_MEMORY",
    "retired_pages_dbe_total": "GPU_MEMORY",
    "retired_pages_pending": "GPU_MEMORY",
    "row_remap_correctable_total": "GPU_MEMORY",
    "row_remap_uncorrectable_total": "GPU_MEMORY",
    "row_remap_pending": "GPU_MEMORY",
    "row_remap_failure": "GPU_MEMORY",
    "nvlink_crc_flit_error_total": "GPU_FABRIC",
    "nvlink_crc_data_error_total": "GPU_FABRIC",
    "nvlink_replay_error_total": "GPU_FABRIC",
    "nvlink_recovery_error_total": "GPU_FABRIC",
    "nvlink_crc_aggregate_error_total": "GPU_FABRIC",
    "nvlink_recovery_aggregate_error_total": "GPU_FABRIC",
    "nvlink_replay_aggregate_error_total": "GPU_FABRIC",
    "gpu_inventory_mismatch": "GPU_INVENTORY",
    "gpu_inventory_identity_changed": "GPU_INVENTORY",
    "gpu_expected_count_unknown": "GPU_INVENTORY",
    "gpu_expected_count_contradicted": "GPU_INVENTORY",
    "gpu_kubernetes_allocatable_mismatch": "GPU_INVENTORY",
    "gpu_device_plugin_unresponsive": "GPU_INVENTORY",
    # These signals do not establish a narrower hardware failure domain.
    "dcgm_field_completeness": "GPU",
    "host_gpu_utilization_percent": "GPU",
    "gpu_utilization_percent": "GPU",
    "pcie_replay_total": "GPU",
    **GPU_FIELD_FAULT_CLASSES,
}
_GPU_COMPOSITE_FAULT_CLASSES = {
    "THERMAL_STRESS": "GPU_THERMAL",
    "POWER_LIMIT_THROTTLING": "GPU",
    "GPU_MEMORY_DEGRADATION": "GPU_MEMORY",
    "CORRECTABLE_MEMORY_DEGRADATION": "GPU_MEMORY",
    "NVLINK_LINK_DEGRADATION": "GPU_FABRIC",
    "MULTI_GPU_NVLINK_FABRIC_FAILURE": "GPU_FABRIC",
    "PCIE_XID_LINK_FAILURE": "GPU",
}


class NodeHealthFinding(StrictModel):
    finding_id: str
    event_id: str
    cluster_id: str
    node_id: str
    observed_at: datetime
    category: NodeHealthCategory
    severity: Severity
    reason: str
    recommended_action: RecoveryAction
    metric_name: str | None = None
    value: float | None = None
    device: str | None = None
    pci_bdf: str | None = None
    gpu_uuids: list[str] = Field(default_factory=list)
    pod_uid: str | None = None
    container_id: str | None = None
    host_pid: int | None = Field(default=None, ge=1)
    cgroup_path: str | None = None
    raw_message: str | None = None
    evidence_ref: str | None = None
    runtime_profile_version: str | None = None
    workload_state: WorkloadState = WorkloadState.UNKNOWN
    affected_workload_ids: list[str] = Field(default_factory=list)
    job_id: str | None = None
    attempt_id: str | None = None
    diagnostic_parameters: dict = Field(default_factory=dict)
    policy_version: str = "site-node-health-policy/v1"
    policy_source: str = "SITE_NODE_HEALTH"
    policy_reference: str | None = None
    official_action: str | None = None
    drill_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    )

    def fault_class(self) -> str:
        if self.category is NodeHealthCategory.GPU:
            metric = (self.metric_name or "").strip().lower()
            if metric in _GPU_METRIC_FAULT_CLASSES:
                return _GPU_METRIC_FAULT_CLASSES[metric]
            rule = (
                metric.removeprefix("composite:").upper()
                if metric.startswith("composite:")
                else self.diagnostic_parameters.get("correlation_rule_id")
            )
            if isinstance(rule, str):
                return _GPU_COMPOSITE_FAULT_CLASSES.get(rule.upper(), "GPU")
            return "GPU"
        if self.category in {
            NodeHealthCategory.RDMA,
            NodeHealthCategory.NCCL,
            NodeHealthCategory.NETWORK,
        }:
            return "NETWORK_FABRIC"
        return self.category.value

    def correlation_keys(self) -> list[str]:
        keys = {
            *(f"gpu:{item.lower()}" for item in self.gpu_uuids),
            *([f"pci:{self.pci_bdf.strip().lower()}"] if self.pci_bdf else []),
            *([f"device:{self.device.strip().lower()}"] if self.device else []),
        }
        for name, prefix in (
            ("nvlink_link_id", "link"),
            ("fabric_partition", "fabric"),
            ("switch_id", "switch"),
            ("port", "port"),
        ):
            value = self.diagnostic_parameters.get(name)
            if value is not None and str(value).strip():
                keys.add(f"{prefix}:{str(value).strip().lower()}")
        return sorted(keys)

    def marker(self, ttl_seconds: int = 3600) -> NodeMarker:
        return NodeMarker(
            marker_id=f"marker-{self.event_id}",
            source=("gpu-fault-policy/" + self.policy_source.lower().replace("_", "-")),
            # Stamp the tenant so a scoped marker read cannot cross clusters
            # (H-14). The finding always carries a non-empty cluster_id.
            cluster_id=self.cluster_id,
            trusted=True,
            incident_id=f"inc-{self.event_id}",
            observed_at=self.observed_at,
            expires_at=self.observed_at + timedelta(seconds=ttl_seconds),
            scope=MarkerScope(
                node_ids=[self.node_id],
                gpu_uuids=self.gpu_uuids,
                pci_bdfs=([self.pci_bdf] if self.pci_bdf else []),
            ),
            event_source=self.policy_source,
            severity=self.severity,
            recommended_action=self.recommended_action,
            action_owner="gpu-fault-node-health-policy",
            mapping_version="site-node-health-policy/v1",
            policy_source="SITE_NODE_HEALTH",
            official_action=None,
            site_safety_action=self.recommended_action.value,
            action_disposition=(
                "SITE_SAFETY"
                if self.recommended_action is RecoveryAction.QUARANTINE
                else "MONITOR_ONLY"
            ),
            fault_class=self.fault_class(),
            correlation_keys=self.correlation_keys(),
            raw_reason=self.reason,
            raw_evidence_ref=self.evidence_ref,
            drill_id=self.drill_id,
        )


class NodeHealthIngestionResult(StrictModel):
    batch_id: str
    findings: list[NodeHealthFinding] = Field(default_factory=list)
    marker_ids: list[str] = Field(default_factory=list)
    incident_ids: list[str] = Field(default_factory=list)
    workflow_request_ids: list[str] = Field(default_factory=list)
    notification_ids: list[str] = Field(default_factory=list)
    duplicate: bool = False


class SyntheticNodeReplacementRequest(StrictModel):
    event_id: str
    cluster_id: str
    node_id: str
    observed_at: datetime
    runtime_profile_version: str
    job_id: str
    attempt_id: str
    affected_workload_ids: list[str] = Field(min_length=1)
    gpu_uuids: list[str] = Field(default_factory=list)
    reason: str
    replacement_strategy: Literal["HEALTHY_WARM_SPARE_ONLY"] = "HEALTHY_WARM_SPARE_ONLY"
    synthetic: Literal[True] = True
    activation_forbidden: Literal[True] | None = Field(default=None, frozen=True)

    @field_validator("activation_forbidden", mode="before")
    @classmethod
    def require_literal_inhibition(cls, value: object) -> object:
        if value is not True:
            raise ValueError("activation_forbidden, when present, must be literal true")
        return value

    @model_serializer(mode="wrap")
    def omit_unrequested_inhibition(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, Any]:
        value: dict[str, Any] = handler(self)
        if "activation_forbidden" not in self.model_fields_set:
            value.pop("activation_forbidden", None)
        return value


_HOST_LATEST_METRICS = CAPABILITY_MARKERS | {
    "cpu_usage_percent",
    "load1_per_cpu",
    "memory_used_percent",
    "memory_available_percent",
    "swap_used_percent",
    "page_cache_percent",
    "filesystem_used_percent",
    "filesystem_inodes_used_percent",
    "filesystem_unavailable",
    "local_filesystem_used_percent",
    "shared_filesystem_unavailable",
    "shared_filesystem_used_percent",
    "disk_io_util_percent",
    "disk_io_await_ms",
    "smart_health_failed",
    "network_link_down",
    "bmc_critical_sensor",
    "training_rank_cpu_percent",
    "host_gpu_utilization_percent",
}


#: A configured mount is unusable when either its blocks or its inodes run out:
#: both refuse new files with ENOSPC, and free bytes say nothing about an
#: exhausted inode table. They grade alike -- 98 %, CRITICAL, QUARANTINE, no
#: STOP (GF-GENERAL-LIVE-004 ``capacity_full`` and ``inode_full``) -- as two
#: signals per mount, so one can neither clear nor mask the other.
_FILESYSTEM_EXHAUSTION_RULES = {
    "filesystem_used_percent": (
        98.0,
        NodeHealthCategory.STORAGE,
        Severity.CRITICAL,
        RecoveryAction.QUARANTINE,
        "filesystem capacity exhausted",
    ),
    "filesystem_inodes_used_percent": (
        98.0,
        NodeHealthCategory.STORAGE,
        Severity.CRITICAL,
        RecoveryAction.QUARANTINE,
        "filesystem inodes exhausted",
    ),
}


def efa_inventory_rule(
    sample: HostMetricSample,
) -> tuple[
    float,
    NodeHealthCategory,
    Severity,
    RecoveryAction,
    str,
]:
    """Resolve EFA recovery semantics without policy state or Store access."""
    failure_mode = sample.labels.get("failure_mode", "INVENTORY_UNAVAILABLE")
    action, reason = {
        "DRIVER_UNBOUND": (
            RecoveryAction.REMEDIATE_EFA_DRIVER,
            "EFA PCI device is present but the efa driver is not bound",
        ),
        "LINK_INACTIVE": (
            RecoveryAction.RUN_DIAGNOSTICS,
            "EFA device and driver are present but the RDMA port is not ACTIVE/LinkUp",
        ),
        "EXCESS_DEVICE": (
            RecoveryAction.RUN_DIAGNOSTICS,
            "active EFA device inventory exceeds the configured node invariant",
        ),
        "PCI_DEVICE_MISSING": (
            RecoveryAction.REBOOT_NODE,
            "EFA PCI device inventory is below the configured node invariant",
        ),
    }.get(
        failure_mode,
        (
            RecoveryAction.RUN_DIAGNOSTICS,
            "EFA inventory observation or failure mode is unknown",
        ),
    )
    return (
        1.0,
        NodeHealthCategory.RDMA,
        Severity.CRITICAL,
        action,
        reason,
    )


#: ``bmc_critical_sensor`` >= 1: one or more ``ipmitool sensor`` rows in a
#: critical or non-recoverable state (GF-GENERAL-LIVE-015). One rule for the
#: table, the edge filter and ``bmc_critical_sensor_rule``.
_BMC_CRITICAL_SENSOR_RULE = (
    1.0,
    NodeHealthCategory.BMC,
    Severity.CRITICAL,
    RecoveryAction.QUARANTINE,
    "BMC reports a critical sensor",
)


def bmc_critical_sensor_rule(
    sample: HostMetricSample,
) -> tuple[
    float,
    NodeHealthCategory,
    Severity,
    RecoveryAction,
    str,
]:
    """The BMC rule, with the alerting sensors named in its reason.

    The reason is the one text the incident keeps (``reasons``), the marker
    carries (``raw_reason``) and the notice the completed QUARANTINE plan
    sends lists (``quarantined-node-notice-zh-v1``), so this is where the
    names, readings, statuses and thresholds the collector put in the
    sample's labels (``sensor_alerts``, or the bare ``sensors`` names) reach
    the operator: a multi-sensor alert names its sensors instead of reading
    "2". The labels themselves travel as the
    finding's ``diagnostic_parameters``. A count without labels -- a node
    whose collector predates them -- keeps the table's fixed text. Pure: the
    sample is evidence and is not changed.
    """

    threshold, category, severity, action, reason = _BMC_CRITICAL_SENSOR_RULE
    alerts = sample.labels.get("sensor_alerts") or sample.labels.get("sensors")
    if alerts:
        count = int(sample.value) if float(sample.value).is_integer() else sample.value
        noun = "sensor" if count == 1 else "sensors"
        reason = f"BMC reports {count} critical {noun}: {alerts}"
    return threshold, category, severity, action, reason


#: The host collector's ``gpu_inventory_mismatch`` re-read against the control
#: plane's own instance-type table (GF-GENERAL-LIVE-012 ``expectation_wrong``).
#: ``GPU_FAULT_EXPECTED_GPU_COUNT`` is a node-level setting the installer writes
#: from the instance type; typed by hand it can contradict that type (live:
#: ``expected=9`` on an 8-GPU p5en). The collector compares the listed UUIDs
#: with the configured count alone, so the typo read as a lost GPU and, on the
#: second tick, as CRITICAL REBOOT_NODE for a healthy node. When the type is
#: known here, the node lists exactly the GPUs the type is built with and the
#: driver's own bound count (when the batch carries it) agrees, the configured
#: count is the one reading out of three that disagrees: a configuration
#: finding -- WARNING, evidence only, operator review -- never a reboot. A node
#: short of the type's own count, a bound count that disagrees with the
#: listing, an unknown type or a batch without the labels grade as before.
GPU_EXPECTED_COUNT_CONTRADICTED_METRIC = "gpu_expected_count_contradicted"
_GPU_EXPECTED_COUNT_CONTRADICTED_RULE = (
    1.0,
    NodeHealthCategory.GPU,
    Severity.WARNING,
    RecoveryAction.COLLECT_EVIDENCE,
    "the configured expected GPU count contradicts the instance type's own count",
)


def _count_label(labels: Mapping[str, str], name: str) -> int | None:
    try:
        return int(labels[name])
    except (KeyError, TypeError, ValueError):
        return None


def _instance_type_gpu_count(labels: Mapping[str, str]) -> int | None:
    instance_type = labels.get("node_instance_type")
    if not instance_type:
        return None
    try:
        return gpu_instance_inventory(instance_type)[0]
    except ValueError:
        return None


def gpu_inventory_verdicts(samples: list[HostMetricSample]) -> list[HostMetricSample]:
    """``samples`` with the GPU inventory verdict re-read against the instance type.

    For every ``gpu_inventory_mismatch`` sample whose labels name an instance
    type the table knows and integer ``expected_count`` / ``observed_count``:
    a configured count that differs from the type's GPU count appends a
    ``gpu_expected_count_contradicted`` sample at 1 (0 when they agree, so a
    corrected configuration closes the episode), and a mismatch raised while
    the node lists exactly the type's count -- with the driver-bound
    ``discovered_count`` agreeing whenever the collector reported one -- is
    re-read as 0, because the configured count is the odd reading out, not
    the node. Samples without those labels, on unknown types or of other
    resources are returned as they are; nothing is mutated. A batch that
    already carries the contradiction metric was judged by its producer.
    """

    names = {sample.name for sample in samples}
    if (
        "gpu_inventory_mismatch" not in names
        or GPU_EXPECTED_COUNT_CONTRADICTED_METRIC in names
    ):
        return samples
    verdicts: list[HostMetricSample] = []
    contradictions: list[HostMetricSample] = []
    for sample in samples:
        type_count = _instance_type_gpu_count(sample.labels)
        expected = _count_label(sample.labels, "expected_count")
        observed = _count_label(sample.labels, "observed_count")
        if (
            sample.name != "gpu_inventory_mismatch"
            or type_count is None
            or expected is None
            or observed is None
        ):
            verdicts.append(sample)
            continue
        bound = sample.labels.get("discovered_count")
        complete = observed == type_count and (
            bound is None or _count_label(sample.labels, "discovered_count") == observed
        )
        contradicted = expected != type_count
        labels = {**sample.labels, "instance_type_gpu_count": str(type_count)}
        verdicts.append(
            sample.model_copy(
                update={
                    "labels": labels,
                    "value": 0.0 if contradicted and complete else sample.value,
                }
            )
        )
        contradictions.append(
            HostMetricSample(
                name=GPU_EXPECTED_COUNT_CONTRADICTED_METRIC,
                value=1.0 if contradicted else 0.0,
                device=sample.device,
                labels=labels,
            )
        )
    return [*verdicts, *contradictions]


def gpu_expected_count_contradicted_rule(
    sample: HostMetricSample,
) -> tuple[float, NodeHealthCategory, Severity, RecoveryAction, str]:
    """The contradiction rule with the three counts named in its reason: the
    operator-review notification carries the reason, not the labels."""

    labels = sample.labels
    bound = labels.get("discovered_count")
    reason = (
        f"configured expected GPU count {labels.get('expected_count', 'UNKNOWN')} "
        f"(the node's collector.env) contradicts instance type "
        f"{labels.get('node_instance_type', 'UNKNOWN')}, which carries "
        f"{labels.get('instance_type_gpu_count', 'UNKNOWN')} GPUs; the node lists "
        f"{labels.get('observed_count', 'UNKNOWN')}"
        + (f" and the driver has {bound} bound" if bound is not None else "")
        + ". This is a configuration error to correct on the node, not a lost "
        "GPU: no reboot or isolation is derived from it"
    )
    return (*_GPU_EXPECTED_COUNT_CONTRADICTED_RULE[:4], reason)


def metric_rule(
    sample: HostMetricSample,
    rules: Mapping[
        str, tuple[float, NodeHealthCategory, Severity, RecoveryAction, str]
    ],
) -> tuple[float, NodeHealthCategory, Severity, RecoveryAction, str] | None:
    """The rule a sample is graded by: the label-driven rules (EFA inventory
    failure mode, BMC sensor names, the GPU expected-count contradiction)
    first, the static table otherwise."""

    if sample.name == "efa_inventory_mismatch":
        return efa_inventory_rule(sample)
    if sample.name == "bmc_critical_sensor":
        return bmc_critical_sensor_rule(sample)
    if sample.name == GPU_EXPECTED_COUNT_CONTRADICTED_METRIC:
        return gpu_expected_count_contradicted_rule(sample)
    return rules.get(sample.name)


class NodeHealthPolicy:
    LATEST_METRICS_REQUIRED = _HOST_LATEST_METRICS | {
        "gpu_inventory_expected_count",
        "gpu_inventory_active_count",
        "efa_inventory_expected_count",
        "efa_inventory_active_count",
        "nvswitch_port_topology",
        # VALIDATE_FABRIC consumes these latest values after
        # diagnostic, driver, plugin and reboot recovery steps. Keep
        # the persistence allowlist narrow, but do not omit inputs that
        # make every fabric validation wait until its deadline.
        "network_link_up",
        "network_errors_delta",
        "network_drops_delta",
        "rdma_link_down",
        "rdma_errors_delta",
        "efa_rnr_errors_delta",
        "efa_retry_errors_delta",
        "efa_cq_errors_delta",
    }
    SUSTAINED_METRIC_RULES = {
        # Per GPU, in percent of one core (collectors/host/gpu_rank.py): the
        # host-wide cpu_usage_percent read a healthy 8-rank job as idle on a
        # 192-vCPU host and as busy on a 96-vCPU one.
        "training_rank_cpu_percent": (
            "LOW_CPU_UTILIZATION",
            "lte",
            NodeHealthCategory.CPU,
            Severity.WARNING,
            RecoveryAction.RUN_DIAGNOSTICS,
            "the training processes on this GPU used almost no CPU while their "
            "managed attempt was active",
            True,
            "low_utilization",
        ),
        "host_gpu_utilization_percent": (
            "LOW_GPU_UTILIZATION",
            "lte",
            NodeHealthCategory.GPU,
            Severity.WARNING,
            RecoveryAction.RUN_DIAGNOSTICS,
            "GPU utilization remained near zero while a training workload was active",
            True,
            "low_utilization",
        ),
        "memory_available_percent": (
            "LOW_MEMORY_AVAILABLE",
            "lte",
            NodeHealthCategory.MEMORY,
            Severity.WARNING,
            RecoveryAction.RUN_DIAGNOSTICS,
            "available host memory remained below the warning threshold",
            False,
            "memory",
        ),
        "page_cache_percent": (
            "HIGH_PAGE_CACHE",
            "gte",
            NodeHealthCategory.MEMORY,
            Severity.WARNING,
            RecoveryAction.RUN_DIAGNOSTICS,
            "page cache remained near total physical memory capacity",
            False,
            "page_cache",
        ),
        "local_filesystem_used_percent": (
            "HIGH_LOCAL_FILESYSTEM_USAGE",
            "gte",
            NodeHealthCategory.STORAGE,
            Severity.WARNING,
            RecoveryAction.RUN_DIAGNOSTICS,
            "local filesystem usage remained above the warning threshold",
            False,
            "local_filesystem",
        ),
    }

    # Single-sample rules whose reading is noisy on a healthy node must hold
    # for a window before a finding is minted (the health-signal state
    # machine's minimum active time). Disk saturation and packet drops are the
    # siblings of the TCP-retransmit flapper: one busy flush or one congested
    # second is not a node fault, the same reading held for a window is. They
    # stay in METRIC_RULES rather than SUSTAINED_METRIC_RULES because the
    # host collector's edge filter reads that table to decide which breaches
    # are worth shipping; a rule the collector cannot see is a rule the
    # control plane never gets a sample for.
    DISK_IO_SUSTAIN_SECONDS = 120.0
    NETWORK_DROPS_SUSTAIN_SECONDS = 60.0
    # CPU saturation, run-queue load and memory pressure are the noisiest of
    # all: a data-loader burst or a checkpoint flush crosses 98 % CPU or 95 %
    # memory for a few samples on a perfectly healthy trainer. One sample used
    # to mint a WARNING RUN_DIAGNOSTICS finding -- and with it an incident, a
    # marker and a VALIDATE_HOST workflow -- for every such burst.
    HOST_RESOURCE_SUSTAIN_SECONDS = 120.0
    METRIC_RULE_SUSTAIN_SECONDS: dict[str, float] = {
        "disk_io_util_percent": DISK_IO_SUSTAIN_SECONDS,
        "disk_io_await_ms": DISK_IO_SUSTAIN_SECONDS,
        "network_drops_delta": NETWORK_DROPS_SUSTAIN_SECONDS,
        "cpu_usage_percent": HOST_RESOURCE_SUSTAIN_SECONDS,
        "load1_per_cpu": HOST_RESOURCE_SUSTAIN_SECONDS,
        "memory_used_percent": HOST_RESOURCE_SUSTAIN_SECONDS,
    }

    METRIC_RULES = {
        "cpu_usage_percent": (
            98.0,
            NodeHealthCategory.CPU,
            Severity.WARNING,
            RecoveryAction.RUN_DIAGNOSTICS,
            "CPU saturation threshold exceeded",
        ),
        "load1_per_cpu": (
            2.0,
            NodeHealthCategory.CPU,
            Severity.WARNING,
            RecoveryAction.RUN_DIAGNOSTICS,
            "system load exceeds CPU capacity",
        ),
        "memory_used_percent": (
            95.0,
            NodeHealthCategory.MEMORY,
            Severity.WARNING,
            RecoveryAction.RUN_DIAGNOSTICS,
            "host memory pressure",
        ),
        "swap_used_percent": (
            80.0,
            NodeHealthCategory.MEMORY,
            Severity.WARNING,
            RecoveryAction.RUN_DIAGNOSTICS,
            "host swap pressure",
        ),
        **_FILESYSTEM_EXHAUSTION_RULES,
        # A configured mount that is read-only, gone, or whose ``statvfs``
        # cannot be read is unusable regardless of its free bytes
        # (GF-GENERAL-LIVE-004 ``read_only``/``required_mount_missing``/
        # ``probe_error``). A diagnostic, never a hold: VALIDATE_HOST re-reads
        # the gauge, fails while it reads 1 (incident RECOVERED, ``diagnostic
        # inconclusive``) and passes once it reads 0; nothing cordons, resets,
        # reboots or quarantines the node for an unreadable status.
        "filesystem_unavailable": (
            1.0,
            NodeHealthCategory.STORAGE,
            Severity.CRITICAL,
            RecoveryAction.RUN_DIAGNOSTICS,
            "configured filesystem is not usable",
        ),
        "shared_filesystem_unavailable": (
            1.0,
            NodeHealthCategory.STORAGE,
            Severity.CRITICAL,
            RecoveryAction.RUN_DIAGNOSTICS,
            "shared filesystem is unavailable",
        ),
        "shared_filesystem_used_percent": (
            98.0,
            NodeHealthCategory.STORAGE,
            Severity.WARNING,
            RecoveryAction.RUN_DIAGNOSTICS,
            "shared filesystem capacity exhausted",
        ),
        "disk_io_util_percent": (
            98.0,
            NodeHealthCategory.STORAGE,
            Severity.WARNING,
            RecoveryAction.RUN_DIAGNOSTICS,
            "disk I/O saturation",
        ),
        "disk_io_await_ms": (
            100.0,
            NodeHealthCategory.STORAGE,
            Severity.WARNING,
            RecoveryAction.RUN_DIAGNOSTICS,
            "disk I/O latency threshold exceeded",
        ),
        "smart_health_failed": (
            1.0,
            NodeHealthCategory.STORAGE,
            Severity.CRITICAL,
            RecoveryAction.QUARANTINE,
            "SMART reports disk failure",
        ),
        "network_link_down": (
            1.0,
            NodeHealthCategory.NETWORK,
            Severity.CRITICAL,
            RecoveryAction.QUARANTINE,
            "required network interface is down",
        ),
        "network_errors_delta": (
            1.0,
            NodeHealthCategory.NETWORK,
            Severity.WARNING,
            RecoveryAction.RUN_DIAGNOSTICS,
            "network interface errors increased",
        ),
        "network_drops_delta": (
            1.0,
            NodeHealthCategory.NETWORK,
            Severity.WARNING,
            RecoveryAction.RUN_DIAGNOSTICS,
            "network packet drops increased",
        ),
        # tcp_retransmits_delta deliberately has no rule. Every other entry in
        # this table is an error counter, where a delta of 1 really is
        # abnormal. A TCP retransmit is normal congestion control, not an
        # error, and the collector reports the metric as a delta of
        # /proc/net/snmp RetransSegs -- so a threshold of 1.0 fired on every
        # collection cycle on every node. Measured on a 4-node p5en fleet on
        # 2026-09-04 that minted ~155 RUN_DIAGNOSTICS workflows an hour
        # (~3700/day), each with its own incident, and kept the processor queue
        # non-empty often enough to flake any preflight that requires an idle
        # queue. The samples are still collected and still ride along in any
        # batch that ships for a real reason, which is where they have
        # diagnostic value; what is removed is the finding and the edge-filter
        # trigger (collectors/host/collector.py reads METRIC_RULES for both).
        "rdma_errors_delta": (
            1.0,
            NodeHealthCategory.RDMA,
            Severity.WARNING,
            RecoveryAction.RUN_DIAGNOSTICS,
            "RDMA/IB hardware errors increased",
        ),
        "rdma_link_down": (
            1.0,
            NodeHealthCategory.RDMA,
            Severity.WARNING,
            RecoveryAction.RUN_DIAGNOSTICS,
            "RDMA/EFA/InfiniBand port is not active",
        ),
        "efa_rnr_errors_delta": (
            1.0,
            NodeHealthCategory.RDMA,
            Severity.WARNING,
            RecoveryAction.RUN_DIAGNOSTICS,
            "EFA receiver-not-ready errors increased",
        ),
        "efa_retry_errors_delta": (
            1.0,
            NodeHealthCategory.RDMA,
            Severity.WARNING,
            RecoveryAction.RUN_DIAGNOSTICS,
            "EFA retry errors increased",
        ),
        "efa_cq_errors_delta": (
            1.0,
            NodeHealthCategory.RDMA,
            Severity.WARNING,
            RecoveryAction.RUN_DIAGNOSTICS,
            "EFA completion queue errors increased",
        ),
        "gpu_inventory_mismatch": (
            1.0,
            NodeHealthCategory.GPU,
            Severity.CRITICAL,
            RecoveryAction.REBOOT_NODE,
            "active GPU inventory does not match the configured node invariant",
        ),
        GPU_EXPECTED_COUNT_CONTRADICTED_METRIC: _GPU_EXPECTED_COUNT_CONTRADICTED_RULE,
        "efa_inventory_mismatch": (
            1.0,
            NodeHealthCategory.RDMA,
            Severity.CRITICAL,
            RecoveryAction.REBOOT_NODE,
            "active EFA device inventory does not match the configured node invariant",
        ),
        # The Kubernetes device-plugin availability rules: the control plane's
        # allocatable comparison and the node's socket liveness probe
        # (``device_plugin_liveness.py``), every one a CRITICAL plugin restart.
        **kubernetes_device_plugin_rules(NodeHealthCategory),
        "bmc_critical_sensor": _BMC_CRITICAL_SENSOR_RULE,
    }

    LOG_RULES = NODE_LOG_RULES

    def __init__(self, store) -> None:
        from gpu_fault.efa_stall_episode import EfaRecoveryPolicy

        self.store = store
        self.efa_recovery_policy = EfaRecoveryPolicy.from_env()
        self.unresolved_gpu_allocations = UnresolvedAllocationWarnings()
        # Device signals whose device stopped reporting (host_device_signals.py).
        self.device_signal_age_out_seconds = device_signal_age_out_seconds()
        self.device_signal_age_outs_total = 0
        self.efa_minimum_active_bps = float(
            os.getenv("GPU_FAULT_EFA_TRAFFIC_MIN_ACTIVE_BPS", "1048576")
        )
        self.efa_spike_ratio = float(
            os.getenv("GPU_FAULT_EFA_TRAFFIC_SPIKE_RATIO", "4")
        )
        self.efa_drop_ratio = float(
            os.getenv("GPU_FAULT_EFA_TRAFFIC_DROP_RATIO", "0.25")
        )
        self.efa_zero_bps = float(os.getenv("GPU_FAULT_EFA_TRAFFIC_ZERO_BPS", "1"))
        self.efa_zero_warning_seconds = float(
            os.getenv("GPU_FAULT_EFA_TRAFFIC_ZERO_WARNING_SECONDS", "60")
        )
        # Quiet RDMA is a suspicion, not a diagnosis: long checkpoint or compute
        # intervals can exceed this site threshold without a communication hang.
        self.efa_zero_hung_seconds = float(
            os.getenv("GPU_FAULT_EFA_TRAFFIC_ZERO_HUNG_SECONDS", "600")
        )
        # Deprecated configuration remains accepted for rolling upgrades only.
        from gpu_fault.hung_progress import validate_legacy_progress_configuration

        validate_legacy_progress_configuration()
        self.efa_rank_liveness_enabled = env_bool(
            "GPU_FAULT_EFA_TRAFFIC_RANK_LIVENESS_ENABLED", True
        )
        self.efa_baseline_alpha = float(
            os.getenv("GPU_FAULT_EFA_TRAFFIC_BASELINE_ALPHA", "0.2")
        )
        self.efa_startup_grace_seconds = float(
            os.getenv("GPU_FAULT_EFA_TRAFFIC_STARTUP_GRACE_SECONDS", "120")
        )
        self.hung_strace_sample_count = int(
            os.getenv("GPU_FAULT_HUNG_STRACE_SAMPLE_COUNT", "3")
        )
        self.hung_strace_sample_duration_seconds = int(
            os.getenv("GPU_FAULT_HUNG_STRACE_SAMPLE_DURATION_SECONDS", "3")
        )
        self.hung_strace_sample_interval_seconds = int(
            os.getenv("GPU_FAULT_HUNG_STRACE_SAMPLE_INTERVAL_SECONDS", "2")
        )
        self.hung_pyspy_sample_count = int(
            os.getenv("GPU_FAULT_HUNG_PYSPY_SAMPLE_COUNT", "3")
        )
        self.hung_pyspy_sample_interval_seconds = int(
            os.getenv("GPU_FAULT_HUNG_PYSPY_SAMPLE_INTERVAL_SECONDS", "2")
        )
        self.hung_pyspy_timeout_seconds = int(
            os.getenv("GPU_FAULT_HUNG_PYSPY_TIMEOUT_SECONDS", "10")
        )
        self.low_utilization_threshold_percent = float(
            os.getenv("GPU_FAULT_LOW_UTILIZATION_THRESHOLD_PERCENT", "5")
        )
        self.low_utilization_duration_seconds = float(
            os.getenv("GPU_FAULT_LOW_UTILIZATION_DURATION_SECONDS", "300")
        )
        self.memory_available_warning_percent = float(
            os.getenv("GPU_FAULT_MEMORY_AVAILABLE_WARNING_PERCENT", "5")
        )
        self.memory_pressure_duration_seconds = float(
            os.getenv("GPU_FAULT_MEMORY_PRESSURE_DURATION_SECONDS", "60")
        )
        self.page_cache_warning_percent = float(
            os.getenv("GPU_FAULT_PAGE_CACHE_WARNING_PERCENT", "90")
        )
        self.page_cache_duration_seconds = float(
            os.getenv("GPU_FAULT_PAGE_CACHE_DURATION_SECONDS", "300")
        )
        self.local_filesystem_warning_percent = float(
            os.getenv("GPU_FAULT_LOCAL_FILESYSTEM_WARNING_PERCENT", "90")
        )
        self.local_filesystem_duration_seconds = float(
            os.getenv("GPU_FAULT_LOCAL_FILESYSTEM_DURATION_SECONDS", "60")
        )
        if not 0 < self.efa_baseline_alpha <= 1:
            raise ValueError("GPU_FAULT_EFA_TRAFFIC_BASELINE_ALPHA must be in (0, 1]")
        if not 2 <= self.hung_strace_sample_count <= 5:
            raise ValueError("GPU_FAULT_HUNG_STRACE_SAMPLE_COUNT must be from 2 to 5")
        if not 1 <= self.hung_strace_sample_duration_seconds <= 30:
            raise ValueError(
                "GPU_FAULT_HUNG_STRACE_SAMPLE_DURATION_SECONDS must be from 1 to 30"
            )
        if not 0 <= self.hung_strace_sample_interval_seconds <= 30:
            raise ValueError(
                "GPU_FAULT_HUNG_STRACE_SAMPLE_INTERVAL_SECONDS must be from 0 to 30"
            )
        if not 3 <= self.hung_pyspy_sample_count <= 5:
            raise ValueError("GPU_FAULT_HUNG_PYSPY_SAMPLE_COUNT must be from 3 to 5")
        if not 0 <= self.hung_pyspy_sample_interval_seconds <= 30:
            raise ValueError(
                "GPU_FAULT_HUNG_PYSPY_SAMPLE_INTERVAL_SECONDS must be from 0 to 30"
            )
        if not 1 <= self.hung_pyspy_timeout_seconds <= 60:
            raise ValueError(
                "GPU_FAULT_HUNG_PYSPY_TIMEOUT_SECONDS must be from 1 to 60"
            )
        for name, value in (
            (
                "GPU_FAULT_LOW_UTILIZATION_THRESHOLD_PERCENT",
                self.low_utilization_threshold_percent,
            ),
            (
                "GPU_FAULT_MEMORY_AVAILABLE_WARNING_PERCENT",
                self.memory_available_warning_percent,
            ),
            (
                "GPU_FAULT_PAGE_CACHE_WARNING_PERCENT",
                self.page_cache_warning_percent,
            ),
            (
                "GPU_FAULT_LOCAL_FILESYSTEM_WARNING_PERCENT",
                self.local_filesystem_warning_percent,
            ),
        ):
            if not 0 <= value <= 100:
                raise ValueError(f"{name} must be from 0 to 100")
        for name, value in (
            (
                "GPU_FAULT_LOW_UTILIZATION_DURATION_SECONDS",
                self.low_utilization_duration_seconds,
            ),
            (
                "GPU_FAULT_MEMORY_PRESSURE_DURATION_SECONDS",
                self.memory_pressure_duration_seconds,
            ),
            (
                "GPU_FAULT_PAGE_CACHE_DURATION_SECONDS",
                self.page_cache_duration_seconds,
            ),
            (
                "GPU_FAULT_LOCAL_FILESYSTEM_DURATION_SECONDS",
                self.local_filesystem_duration_seconds,
            ),
        ):
            if value < 15:
                raise ValueError(f"{name} must be at least 15")
        if not 0 < self.efa_drop_ratio < 1:
            raise ValueError("GPU_FAULT_EFA_TRAFFIC_DROP_RATIO must be in (0, 1)")
        if self.efa_spike_ratio <= 1:
            raise ValueError("GPU_FAULT_EFA_TRAFFIC_SPIKE_RATIO must be greater than 1")
        if self.efa_zero_hung_seconds < self.efa_zero_warning_seconds:
            raise ValueError("EFA zero hung threshold must not precede warning")

    def evaluate_metrics(self, batch: HostTelemetryBatch) -> list[NodeHealthFinding]:
        from gpu_fault.telemetry import TelemetryMetricLatest

        findings = []
        samples = gpu_inventory_verdicts(batch.samples)
        related_metrics = {sample.name: sample.value for sample in samples}
        managed_attempt = BatchManagedAttempt(self, batch)
        latest_metrics = []
        transitions = []
        transition_findings = []
        for sample in samples:
            # ``device_state=GONE``: a clear, never a reading (host_device_signals).
            gone = device_gone(sample)
            if sample.name in self.LATEST_METRICS_REQUIRED and not gone:
                latest_metrics.append(
                    TelemetryMetricLatest(
                        cluster_id=batch.cluster_id,
                        node_id=batch.node_id,
                        observed_at=batch.observed_at,
                        name=sample.name,
                        value=sample.value,
                        unit=sample.unit,
                        device=sample.device,
                        labels=sample.labels,
                    )
                )
            sustained_rule = self.SUSTAINED_METRIC_RULES.get(sample.name)
            if sustained_rule is not None:
                transition, finding = self._prepare_sustained_metric(
                    batch,
                    sample,
                    sustained_rule,
                    related_metrics,
                    managed_attempt,
                )
                transitions.append(transition)
                transition_findings.append(finding)
            rule = metric_rule(sample, self.METRIC_RULES)
            if rule is None:
                continue
            threshold, category, severity, action, reason = rule
            sustain = self.METRIC_RULE_SUSTAIN_SECONDS.get(sample.name, 0.0)
            active = not gone and sample.value >= threshold
            signal_key = health_signal_key(
                batch.cluster_id, batch.node_id, sample.name, sample.device
            )
            transitions.append((signal_key, active, batch.observed_at, sustain))
            event_id = (
                f"{batch.batch_id}-{sample.name}-{self._safe(sample.device or 'node')}"
            )
            parameters = dict(sample.labels)
            if sample.name == "efa_inventory_mismatch":
                parameters.setdefault("failure_mode", "INVENTORY_UNAVAILABLE")
            transition_findings.append(
                NodeHealthFinding(
                    finding_id=f"finding-{event_id}",
                    event_id=event_id,
                    cluster_id=batch.cluster_id,
                    node_id=batch.node_id,
                    observed_at=batch.observed_at,
                    category=category,
                    severity=severity,
                    reason=reason,
                    recommended_action=action,
                    metric_name=sample.name,
                    value=sample.value,
                    device=sample.device,
                    evidence_ref=batch.evidence_ref,
                    runtime_profile_version=(batch.runtime_profile_version),
                    workload_state=batch.workload_state,
                    affected_workload_ids=(batch.affected_workload_ids),
                    diagnostic_parameters=parameters,
                )
            )
        self.store.observe_telemetry_metrics(latest_metrics)
        age_out_vanished_device_signals(self, batch, transitions)
        emitted = managed_attempt.claim(
            transitions,
            [finding_health_signal_fingerprint(item) for item in transition_findings],
        )
        findings.extend(
            finding
            for finding, emit in zip(transition_findings, emitted, strict=True)
            if emit
        )
        findings.extend(self._evaluate_efa_traffic(batch, managed_attempt))
        return findings

    def _prepare_sustained_metric(
        self,
        batch: HostTelemetryBatch,
        sample: HostMetricSample,
        rule,
        related_metrics: dict[str, float],
        managed_attempt: BatchManagedAttempt,
    ):
        (
            rule_id,
            comparison,
            category,
            severity,
            action,
            reason,
            active_workload_only,
            threshold_group,
        ) = rule
        thresholds = {
            "low_utilization": (
                self.low_utilization_threshold_percent,
                self.low_utilization_duration_seconds,
            ),
            "memory": (
                self.memory_available_warning_percent,
                self.memory_pressure_duration_seconds,
            ),
            "page_cache": (
                self.page_cache_warning_percent,
                self.page_cache_duration_seconds,
            ),
            "local_filesystem": (
                self.local_filesystem_warning_percent,
                self.local_filesystem_duration_seconds,
            ),
        }
        threshold, duration = thresholds[threshold_group]
        active = not device_gone(sample) and (
            sample.value <= threshold
            if comparison == "lte"
            else sample.value >= threshold
        )
        # ``workload_state``/``affected_workload_ids`` are the collector's
        # static environment, not an observation: a node configured ACTIVE
        # reports ACTIVE while a placeholder job holds it idle. The signal only
        # counts while the control plane resolves one real managed attempt with
        # a live container on this node (the resolver the EFA-traffic rule
        # already trusts), and only on a GPU that attempt holds there
        # (host_low_utilization.py). Anything else is an idle sample and resets
        # the sustain window exactly as a healthy reading would. The window is
        # that attempt's own: a replacement attempt starts from zero.
        graded = False
        if active_workload_only:
            graded = managed_attempt.grades(rule_id, sample, active)
            active = active and graded
        attempt = managed_attempt.resolve() if graded else None
        transition = (
            health_signal_key(
                batch.cluster_id, batch.node_id, sample.name, sample.device, rule_id
            ),
            active,
            batch.observed_at,
            duration,
        )
        event_id = (
            f"{batch.batch_id}-{sample.name}-"
            f"{self._safe(sample.device or 'node')}-{rule_id.lower()}"
        )
        return (
            managed_attempt.scope(transition) if graded else transition,
            NodeHealthFinding(
                finding_id=f"finding-{event_id}",
                event_id=event_id,
                cluster_id=batch.cluster_id,
                node_id=batch.node_id,
                observed_at=batch.observed_at,
                category=category,
                severity=severity,
                reason=reason,
                recommended_action=action,
                metric_name=sample.name,
                value=sample.value,
                device=sample.device,
                evidence_ref=batch.evidence_ref,
                runtime_profile_version=batch.runtime_profile_version,
                workload_state=batch.workload_state,
                affected_workload_ids=batch.affected_workload_ids,
                job_id=attempt.job_id if attempt is not None else None,
                attempt_id=attempt.attempt_id if attempt is not None else None,
                diagnostic_parameters={
                    **sample.labels,
                    "signal": rule_id,
                    "comparison": comparison,
                    "threshold_percent": threshold,
                    "minimum_active_seconds": duration,
                    "related_metrics": related_metrics,
                },
                policy_source="SITE_HOST_RESOURCE_HEALTH",
                policy_reference=("site-configurable sustained host resource policy"),
            ),
        )

    def _active_attempt(self, batch: HostTelemetryBatch):
        return resolve_managed_attempt(self.store, batch)

    def _rank_liveness(self, batch: HostTelemetryBatch) -> dict[str, Any]:
        from gpu_fault.hung_progress import rank_liveness_evidence

        return rank_liveness_evidence(batch, self.efa_rank_liveness_enabled)

    def _evaluate_efa_traffic(
        self,
        batch: HostTelemetryBatch,
        managed_attempt: BatchManagedAttempt | None = None,
    ) -> list[NodeHealthFinding]:
        reading = traffic_observation(batch.samples)
        if reading is None:
            return []
        bytes_per_second, observation_valid = reading
        observation = (
            managed_attempt.resolve()
            if managed_attempt is not None
            else self._active_attempt(batch)
        )
        if observation is None or cleanup_reason(observation, [batch.node_id]):
            return []
        rank_liveness = self._rank_liveness(batch)
        # CPU/writes can continue during a collective hang. Neither these hints
        # nor application step/phase reports reset the communication idle clock.
        rank_liveness["use"] = "CONTEXT_ONLY"
        state_key = self.store.efa_traffic_state_key(batch.cluster_id, batch.node_id)
        state, emit = self.store.observe_efa_traffic(
            state_key=state_key,
            cluster_id=batch.cluster_id,
            node_id=batch.node_id,
            job_id=observation.job_id,
            attempt_id=observation.attempt_id,
            observed_at=batch.observed_at,
            bytes_per_second=bytes_per_second,
            observation_valid=observation_valid,
            minimum_active_bps=self.efa_minimum_active_bps,
            spike_ratio=self.efa_spike_ratio,
            drop_ratio=self.efa_drop_ratio,
            zero_bps=self.efa_zero_bps,
            zero_warning_seconds=self.efa_zero_warning_seconds,
            zero_hung_seconds=self.efa_zero_hung_seconds,
            baseline_alpha=self.efa_baseline_alpha,
            startup_grace_seconds=self.efa_startup_grace_seconds,
            progress_observed_at=None,
            spike_event_id=(f"{batch.batch_id}-efa-traffic-spike"),
            recovery_policy=self.efa_recovery_policy,
        )
        actionable = {
            EfaTrafficSignal.SPIKE: (
                Severity.WARNING,
                "EFA/RDMA traffic increased abruptly relative to the attempt baseline",
            ),
            EfaTrafficSignal.DROP: (
                Severity.WARNING,
                "EFA/RDMA traffic dropped abruptly relative to the attempt baseline",
            ),
            EfaTrafficSignal.ZERO_WARNING: (
                Severity.WARNING,
                "EFA/RDMA traffic remains at zero for an active training attempt",
            ),
            EfaTrafficSignal.HUNG_SUSPECTED: (
                Severity.CRITICAL,
                "EFA/RDMA traffic remains at zero long enough to "
                "suspect a hung training attempt",
            ),
        }
        if not emit or state.signal not in actionable:
            return []
        severity, reason = actionable[state.signal]
        zero_duration = (
            (batch.observed_at - state.zero_since).total_seconds()
            if state.zero_since is not None
            else 0
        )
        event_id = f"{batch.batch_id}-efa-traffic-{state.signal.value.lower()}"
        capture_process_state = state.signal is EfaTrafficSignal.HUNG_SUSPECTED
        active_containers = [
            container
            for container in observation.containers
            if container.node_id and not container.terminated
        ]
        attempt_node_ids = sorted(
            {container.node_id for container in active_containers}
        )
        gpu_uuids_by_node = {
            node_id: sorted(
                {
                    gpu_uuid
                    for container in active_containers
                    if container.node_id == node_id
                    for gpu_uuid in container.gpu_uuids
                }
            )
            for node_id in attempt_node_ids
        }
        complete_gpu_mapping = bool(attempt_node_ids) and all(
            gpu_uuids_by_node[node_id] for node_id in attempt_node_ids
        )
        diagnostic_parameters = {
            "diagnostic_reason": (f"EFA_TRAFFIC_{state.signal.value}"),
            "capture_process_state": capture_process_state,
            "strace_sample_count": self.hung_strace_sample_count,
            "strace_duration_seconds": (self.hung_strace_sample_duration_seconds),
            "strace_sample_interval_seconds": (
                self.hung_strace_sample_interval_seconds
            ),
            "pyspy_sample_count": self.hung_pyspy_sample_count,
            "pyspy_sample_interval_seconds": (self.hung_pyspy_sample_interval_seconds),
            "pyspy_timeout_seconds": self.hung_pyspy_timeout_seconds,
            "job_id": observation.job_id,
            "attempt_id": observation.attempt_id,
            "workload_ids": observation.workload_ids,
            "attempt_node_ids": attempt_node_ids,
            "baseline_bytes_per_second": (state.baseline_bytes_per_second),
            "observed_bytes_per_second": bytes_per_second,
            "zero_duration_seconds": zero_duration,
            "efa_zero_since": (
                state.zero_since.isoformat() if state.zero_since is not None else None
            ),
            "efa_zero_first_seen": (
                state.zero_first_seen_at.isoformat()
                if state.zero_first_seen_at is not None
                else None
            ),
            "rank_liveness": rank_liveness,
            "detection_basis": "ATTEMPT_SCOPED_RDMA_IDLE",
            "root_cause_confirmed": False,
            **episode_parameters(state),
        }
        if capture_process_state and complete_gpu_mapping:
            diagnostic_parameters["gpu_uuids_by_node"] = gpu_uuids_by_node
        return [
            NodeHealthFinding(
                finding_id=f"finding-{event_id}",
                event_id=event_id,
                cluster_id=batch.cluster_id,
                node_id=batch.node_id,
                observed_at=batch.observed_at,
                category=NodeHealthCategory.RDMA,
                severity=severity,
                reason=reason,
                recommended_action=RecoveryAction.RUN_DIAGNOSTICS,
                metric_name="efa_traffic_bytes_per_second",
                value=bytes_per_second,
                evidence_ref=batch.evidence_ref,
                runtime_profile_version=(batch.runtime_profile_version),
                workload_state=batch.workload_state,
                affected_workload_ids=(batch.affected_workload_ids),
                job_id=observation.job_id,
                attempt_id=observation.attempt_id,
                policy_version="site-efa-traffic-policy/v2",
                policy_source="SITE_EFA_TRAFFIC",
                diagnostic_parameters=diagnostic_parameters,
            )
        ]

    def evaluate_logs(self, batch: NodeLogBatch) -> list[NodeHealthFinding]:
        findings = []
        severity_rank = {
            Severity.INFO: 0,
            Severity.WARNING: 1,
            Severity.CRITICAL: 2,
            Severity.FATAL: 3,
        }
        for entry in batch.entries:
            entry_source = getattr(entry, "source", None)
            matches = []
            for rule_index, rule in enumerate(self.LOG_RULES):
                if not rule.pattern.search(entry.message):
                    continue
                # Trusted-source gate (H-5): a hardware-fatal rule may only
                # fire from a kernel/journal origin a workload cannot forge.
                # An entry from a disallowed or absent origin is skipped so it
                # can never produce a trusted isolation/quarantine marker.
                if not rule.allows(entry_source):
                    continue
                decision = log_rule_decision(self.store, batch, entry, rule)
                if decision is None:
                    continue
                matches.append(
                    (
                        NodeHealthCategory(rule.category.value),
                        *decision,
                        rule.reason,
                        rule_index,
                    )
                )
            if not matches:
                continue
            category, severity, action, reason, _ = max(
                matches,
                key=lambda item: (
                    severity_rank[item[1]],
                    recovery_action_sort_key(item[2]),
                    -item[4],
                ),
            )
            event_id = self._log_event_id(batch, entry)
            context = log_context(self.store, batch, entry, action)
            findings.append(
                NodeHealthFinding(
                    finding_id=f"finding-{event_id}",
                    event_id=event_id,
                    cluster_id=batch.cluster_id,
                    node_id=batch.node_id,
                    observed_at=entry.observed_at,
                    category=category,
                    severity=severity,
                    reason=reason,
                    recommended_action=action,
                    raw_message=entry.message,
                    evidence_ref=(f"{entry.source}://{batch.node_id}/{entry.entry_id}"),
                    runtime_profile_version=(batch.runtime_profile_version),
                    **context,
                )
            )
        return findings

    @classmethod
    def _log_event_id(cls, batch: NodeLogBatch, entry: NodeLogEntry) -> str:
        return log_event_id(batch, entry, cls._safe)

    @staticmethod
    def _safe(value: str) -> str:
        return event_id_component(value)


def event_id_component(value: str) -> str:
    """One ``-``-joined component of a metric finding's event id.

    ``<batch>-<metric>-<device|node>`` is the shape every metric rule mints;
    the device (a mount point, an interface) is reduced to this alphabet so a
    reader that has the device in hand can recognise the incident the rule
    opened for it (``shared_storage_failure_domain`` matches the per-node
    ``shared_filesystem_unavailable`` incidents of a peer this way).
    """

    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value)[:160]
