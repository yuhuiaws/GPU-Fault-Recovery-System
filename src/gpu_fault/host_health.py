from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4

from pydantic import Field

from gpu_fault.env import env_bool
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
        diagnostic_reason = self.diagnostic_parameters.get("diagnostic_reason", "")
        description = " ".join(
            item
            for item in (
                self.metric_name or "",
                self.reason,
                str(diagnostic_reason),
            )
            if item
        ).lower()
        if self.category is NodeHealthCategory.GPU:
            if any(token in description for token in ("nvlink", "nvswitch", "sxid")):
                return "GPU_FABRIC"
            if any(
                token in description
                for token in (
                    "ecc",
                    "memory",
                    "retired",
                    "row_remap",
                    "row-remap",
                )
            ):
                return "GPU_MEMORY"
            if any(
                token in description for token in ("temperature", "thermal", "power")
            ):
                return "GPU_THERMAL"
            if any(
                token in description for token in ("inventory", "missing", "fallen off")
            ):
                return "GPU_INVENTORY"
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


_HOST_VALIDATION_LATEST_METRICS = {
    "load1_per_cpu",
    "memory_used_percent",
    "filesystem_used_percent",
}


class _BatchManagedAttempt:
    """Resolve the managed attempt that owns a telemetry batch, once.

    A batch carries eight GPUs times several rules and the resolver walks
    every attempt observation in the cluster; the answer cannot change inside
    one ``evaluate_metrics`` call, so it is memoised here and discarded with
    the call. Nothing lands on the policy, so no state leaks across batches.
    """

    __slots__ = ("_attempt", "_batch", "_explained", "_policy", "_resolved")

    def __init__(self, policy: NodeHealthPolicy, batch: HostTelemetryBatch) -> None:
        self._policy = policy
        self._batch = batch
        self._resolved = False
        self._attempt: Any = None
        self._explained = False

    def resolve(self) -> Any:
        """The one managed attempt on the batch's node, or ``None``; memoized."""

        if not self._resolved:
            self._attempt = self._policy._active_attempt(self._batch)
            self._resolved = True
        return self._attempt

    def explain_inactive(self, rule_id: str, sample: HostMetricSample) -> None:
        """Debug-log, once per batch, why an active-workload rule stays idle."""

        if self._explained:
            return
        self._explained = True
        batch = self._batch
        if (
            batch.workload_state is not WorkloadState.ACTIVE
            or not batch.affected_workload_ids
        ):
            reason = "collector reports no ACTIVE workload"
        else:
            reason = (
                "collector declares an ACTIVE workload but no single managed "
                "attempt with a live container resolves on this node"
            )
        LOGGER.debug(
            "%s on cluster=%s node=%s (first breach %s%s) treated as inactive: %s; "
            "workload_state=%s affected_workload_ids=%s",
            rule_id,
            batch.cluster_id,
            batch.node_id,
            sample.name,
            f"/{sample.device}" if sample.device else "",
            reason,
            batch.workload_state.value,
            batch.affected_workload_ids,
        )


class NodeHealthPolicy:
    LATEST_METRICS_REQUIRED = _HOST_VALIDATION_LATEST_METRICS | {
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
        "cpu_usage_percent": (
            "LOW_CPU_UTILIZATION",
            "lte",
            NodeHealthCategory.CPU,
            Severity.WARNING,
            RecoveryAction.RUN_DIAGNOSTICS,
            "CPU utilization remained near zero while a training workload was active",
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
        "filesystem_used_percent": (
            98.0,
            NodeHealthCategory.STORAGE,
            Severity.CRITICAL,
            RecoveryAction.QUARANTINE,
            "filesystem capacity exhausted",
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
        "efa_inventory_mismatch": (
            1.0,
            NodeHealthCategory.RDMA,
            Severity.CRITICAL,
            RecoveryAction.REBOOT_NODE,
            "active EFA device inventory does not match the configured node invariant",
        ),
        "efa_kubernetes_allocatable_mismatch": (
            1.0,
            NodeHealthCategory.RDMA,
            Severity.CRITICAL,
            RecoveryAction.RESTART_EFA_DEVICE_PLUGIN,
            "Kubernetes EFA allocatable does not match the host EFA inventory",
        ),
        "gpu_kubernetes_allocatable_mismatch": (
            1.0,
            NodeHealthCategory.GPU,
            Severity.CRITICAL,
            RecoveryAction.RESTART_GPU_DEVICE_PLUGIN,
            "Kubernetes GPU allocatable does not match the host GPU inventory",
        ),
        "bmc_critical_sensor": (
            1.0,
            NodeHealthCategory.BMC,
            Severity.CRITICAL,
            RecoveryAction.QUARANTINE,
            "BMC reports a critical sensor",
        ),
    }

    LOG_RULES = NODE_LOG_RULES

    def __init__(self, store) -> None:
        self.store = store
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
        # A quiet fabric is not by itself a hang: checkpoint writes and
        # graph compilation routinely idle the network for minutes on a
        # healthy attempt. The threshold is deliberately longer than the
        # longest of those phases, and progress suppression below
        # restarts the clock whenever the attempt is still stepping.
        self.efa_zero_hung_seconds = float(
            os.getenv("GPU_FAULT_EFA_TRAFFIC_ZERO_HUNG_SECONDS", "600")
        )
        self.efa_progress_suppression_enabled = env_bool(
            "GPU_FAULT_EFA_TRAFFIC_PROGRESS_SUPPRESSION_ENABLED", True
        )
        self.efa_rank_liveness_enabled = env_bool(
            "GPU_FAULT_EFA_TRAFFIC_RANK_LIVENESS_ENABLED", True
        )
        self.efa_progress_suppression_max_seconds = float(
            os.getenv(
                ("GPU_FAULT_EFA_TRAFFIC_PROGRESS_SUPPRESSION_MAX_SECONDS"),
                "1800",
            )
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
        if self.efa_progress_suppression_max_seconds <= 0:
            raise ValueError(
                "GPU_FAULT_EFA_TRAFFIC_PROGRESS_SUPPRESSION_MAX_SECONDS "
                "must be positive"
            )

    def evaluate_metrics(self, batch: HostTelemetryBatch) -> list[NodeHealthFinding]:
        from gpu_fault.telemetry import TelemetryMetricLatest

        findings = []
        related_metrics = {sample.name: sample.value for sample in batch.samples}
        managed_attempt = _BatchManagedAttempt(self, batch)
        latest_metrics = []
        transitions = []
        transition_findings = []
        for sample in batch.samples:
            if sample.name in self.LATEST_METRICS_REQUIRED:
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
            rule = (
                self._efa_inventory_rule(sample)
                if sample.name == "efa_inventory_mismatch"
                else self.METRIC_RULES.get(sample.name)
            )
            if rule is None:
                continue
            threshold, category, severity, action, reason = rule
            minimum_active_seconds = self.METRIC_RULE_SUSTAIN_SECONDS.get(
                sample.name, 0.0
            )
            active = sample.value >= threshold
            signal_key = "/".join(
                [
                    batch.cluster_id,
                    batch.node_id,
                    sample.name,
                    sample.device or "node",
                ]
            )
            transitions.append(
                (signal_key, active, batch.observed_at, minimum_active_seconds)
            )
            event_id = (
                f"{batch.batch_id}-{sample.name}-{self._safe(sample.device or 'node')}"
            )
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
                    diagnostic_parameters=dict(sample.labels),
                )
            )
        self.store.observe_telemetry_metrics(latest_metrics)
        emitted = self.store.claim_health_signal_transitions(
            transitions, received_at=batch.received_at
        )
        findings.extend(
            finding
            for finding, emit in zip(transition_findings, emitted, strict=True)
            if emit
        )
        findings.extend(self._evaluate_efa_traffic(batch, managed_attempt))
        return findings

    @staticmethod
    def _efa_inventory_rule(
        sample: HostMetricSample,
    ) -> tuple[
        float,
        NodeHealthCategory,
        Severity,
        RecoveryAction,
        str,
    ]:
        failure_mode = sample.labels.get("failure_mode", "PCI_DEVICE_MISSING")
        action, reason = {
            "DRIVER_UNBOUND": (
                RecoveryAction.REMEDIATE_EFA_DRIVER,
                "EFA PCI device is present but the efa driver is not bound",
            ),
            "LINK_INACTIVE": (
                RecoveryAction.RUN_DIAGNOSTICS,
                "EFA device and driver are present but the RDMA port is "
                "not ACTIVE/LinkUp",
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
                RecoveryAction.REBOOT_NODE,
                "active EFA inventory mismatch has an unknown failure mode",
            ),
        )
        return (
            1.0,
            NodeHealthCategory.RDMA,
            Severity.CRITICAL,
            action,
            reason,
        )

    def _prepare_sustained_metric(
        self,
        batch: HostTelemetryBatch,
        sample: HostMetricSample,
        rule,
        related_metrics: dict[str, float],
        managed_attempt: _BatchManagedAttempt,
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
        active = (
            sample.value <= threshold
            if comparison == "lte"
            else sample.value >= threshold
        )
        # ``workload_state``/``affected_workload_ids`` are the collector's
        # static environment, not an observation: a node configured ACTIVE
        # reports ACTIVE while a placeholder job holds it idle. The signal only
        # counts while the control plane resolves one real managed attempt with
        # a live container on this node (the resolver the EFA-traffic rule
        # already trusts). Anything else is an idle sample and resets the
        # sustain window exactly as a healthy reading would.
        attempt = managed_attempt.resolve() if active_workload_only else None
        if active_workload_only and attempt is None:
            if active:
                managed_attempt.explain_inactive(rule_id, sample)
            active = False
        signal_key = "/".join(
            [
                batch.cluster_id,
                batch.node_id,
                sample.name,
                sample.device or "node",
                rule_id,
            ]
        )
        event_id = (
            f"{batch.batch_id}-{sample.name}-"
            f"{self._safe(sample.device or 'node')}-{rule_id.lower()}"
        )
        return (
            (
                signal_key,
                active,
                batch.observed_at,
                duration,
            ),
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
        if (
            batch.workload_state is not WorkloadState.ACTIVE
            or not batch.affected_workload_ids
        ):
            return None
        workload_ids = set(batch.affected_workload_ids)
        candidates = []
        for observation in self.store.list_attempt_observations(batch.cluster_id):
            age = (batch.observed_at - observation.observed_at).total_seconds()
            if (
                age < -30
                or age > 120
                or observation.workload_phase.value not in {"PENDING", "RUNNING"}
                or not workload_ids.intersection(observation.workload_ids)
                or not any(
                    container.node_id == batch.node_id and not container.terminated
                    for container in observation.containers
                )
            ):
                continue
            candidates.append(observation)
        identities = {(item.job_id, item.attempt_id) for item in candidates}
        if len(identities) > 1:
            LOGGER.warning(
                "ambiguous managed attempt ownership: cluster=%s node=%s identities=%s",
                batch.cluster_id,
                batch.node_id,
                sorted(identities),
            )
            return None
        return candidates[0] if len(identities) == 1 else None

    def _training_progress_at(
        self, cluster_id: str, attempt_id: str
    ) -> datetime | None:
        """Most recent evidence that the attempt is still advancing.

        A rank whose step counter moved, or that is reporting an active
        checkpoint, proves the attempt is not wedged even while the
        fabric is idle. Absent instrumentation this returns ``None`` and
        zero-traffic detection behaves exactly as before.
        """

        if not self.efa_progress_suppression_enabled:
            return None
        lister = getattr(self.store, "list_training_progress_states", None)
        if lister is None:
            return None
        try:
            states = lister(cluster_id, attempt_id)
        except Exception:
            LOGGER.exception(
                "cannot read training progress for attempt %s",
                attempt_id,
            )
            return None
        timestamps = []
        for state in states:
            timestamps.append(state.last_progress_at)
            if state.heartbeat.checkpoint_ref:
                timestamps.append(state.heartbeat.observed_at)
        return max(timestamps) if timestamps else None

    def _rank_liveness(self, batch: HostTelemetryBatch) -> dict[str, Any]:
        """Read the node's own per-rank liveness verdict from the batch.

        The node samples ``/proc/<pid>/{io,stat}`` and GPU utilization for
        every rank on every collection cycle and reports how long ago it
        last saw one advance. That is the pre-escalation check for a hang:
        a quiet fabric plus a rank that is still writing bytes or still
        burning host CPU with idle GPUs is a checkpoint or a compile, not
        a wedged collective. When the node reports nothing the result is
        ``unavailable`` and escalation proceeds exactly as before, so a
        broken probe cannot silence hang detection.
        """

        values = {
            sample.name: sample.value
            for sample in batch.samples
            if sample.device is None and sample.name.startswith("training_rank_")
        }
        process_count = values.get("training_rank_process_count")
        seconds = values.get("training_rank_seconds_since_progress")
        evidence: dict[str, Any] = {
            "process_count": process_count,
            "advancing_count": values.get("training_rank_advancing_count"),
            "seconds_since_progress": seconds,
            "write_bytes_delta": values.get("training_rank_write_bytes_delta"),
            "write_chars_delta": values.get("training_rank_write_chars_delta"),
            "cpu_ticks_delta": values.get("training_rank_cpu_ticks_delta"),
        }
        if not self.efa_rank_liveness_enabled:
            return {**evidence, "status": "disabled"}
        if not process_count:
            return {**evidence, "status": "unavailable"}
        if seconds is None or seconds < 0 or seconds > 86400:
            return {**evidence, "status": "unavailable"}
        progress_at = batch.observed_at - timedelta(seconds=seconds)
        return {
            **evidence,
            "status": (
                "advancing"
                if values.get("training_rank_advancing_count")
                else "stalled"
            ),
            "progress_observed_at": progress_at.isoformat(),
            "_progress_at": progress_at,
        }

    def _evaluate_efa_traffic(
        self,
        batch: HostTelemetryBatch,
        managed_attempt: _BatchManagedAttempt | None = None,
    ) -> list[NodeHealthFinding]:
        samples = [
            item
            for item in batch.samples
            if item.name == "efa_traffic_bytes_per_second" and item.device is None
        ]
        if len(samples) != 1:
            return []
        observation = (
            managed_attempt.resolve()
            if managed_attempt is not None
            else self._active_attempt(batch)
        )
        if observation is None:
            return []
        sample = samples[0]
        rank_liveness = self._rank_liveness(batch)
        rank_progress_at = rank_liveness.pop("_progress_at", None)
        instrumented_progress_at = self._training_progress_at(
            batch.cluster_id, observation.attempt_id
        )
        progress_observed_at = max(
            [
                item
                for item in (
                    rank_progress_at,
                    instrumented_progress_at,
                )
                if item is not None
            ],
            default=None,
        )
        state_key = self.store.efa_traffic_state_key(
            batch.cluster_id,
            batch.node_id,
            observation.job_id,
            observation.attempt_id,
        )
        state, emit = self.store.observe_efa_traffic(
            state_key=state_key,
            cluster_id=batch.cluster_id,
            node_id=batch.node_id,
            job_id=observation.job_id,
            attempt_id=observation.attempt_id,
            observed_at=batch.observed_at,
            bytes_per_second=max(0.0, sample.value),
            minimum_active_bps=self.efa_minimum_active_bps,
            spike_ratio=self.efa_spike_ratio,
            drop_ratio=self.efa_drop_ratio,
            zero_bps=self.efa_zero_bps,
            zero_warning_seconds=self.efa_zero_warning_seconds,
            zero_hung_seconds=self.efa_zero_hung_seconds,
            baseline_alpha=self.efa_baseline_alpha,
            startup_grace_seconds=self.efa_startup_grace_seconds,
            progress_observed_at=progress_observed_at,
            progress_suppression_max_seconds=(
                self.efa_progress_suppression_max_seconds
            ),
            spike_event_id=(f"{batch.batch_id}-efa-traffic-spike"),
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
            "observed_bytes_per_second": sample.value,
            "zero_duration_seconds": zero_duration,
            "efa_zero_since": (
                state.zero_since.isoformat() if state.zero_since is not None else None
            ),
            "efa_zero_first_seen": (
                state.zero_first_seen_at.isoformat()
                if state.zero_first_seen_at is not None
                else None
            ),
            # Why the escalation was not suppressed. ``unavailable``
            # means the node published no per-rank evidence and the
            # decision rests on the zero-traffic clock alone.
            "rank_liveness": rank_liveness,
            "training_progress_at": (
                instrumented_progress_at.isoformat()
                if instrumented_progress_at is not None
                else None
            ),
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
                metric_name=sample.name,
                value=sample.value,
                evidence_ref=batch.evidence_ref,
                runtime_profile_version=(batch.runtime_profile_version),
                workload_state=batch.workload_state,
                affected_workload_ids=(batch.affected_workload_ids),
                job_id=observation.job_id,
                attempt_id=observation.attempt_id,
                policy_version="site-efa-traffic-policy/v1",
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
                matches.append(
                    (
                        NodeHealthCategory(rule.category.value),
                        rule.severity,
                        rule.action,
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
                    workload_state=batch.workload_state,
                    affected_workload_ids=(batch.affected_workload_ids),
                )
            )
        return findings

    @classmethod
    def _log_event_id(cls, batch: NodeLogBatch, entry: NodeLogEntry) -> str:
        """The identity of one node-log finding, decided here and not on the node.

        ``log-<entry_id>`` trusted the collector's id for the whole identity. A
        training-log entry was ``sha256(path:offset)`` and every rank of a
        distributed job writes the same path on its own node, so two nodes'
        faults at the same offset were one event: the second node hit the first
        node's incident in the duplicate fast path and never got a workflow
        (P0-38A). The cluster and node are part of the digest, and the readable
        prefix is for operators only -- ``_safe`` folds separators, so the
        prefix alone could spell ``a-b``/``c`` and ``a``/``b-c`` the same way.
        """

        digest = hashlib.sha256(
            json.dumps(
                [batch.cluster_id, batch.node_id, entry.entry_id],
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()[:32]
        return f"log-{cls._safe(batch.cluster_id)}-{cls._safe(batch.node_id)}-{digest}"

    @staticmethod
    def _safe(value: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value)[:160]
