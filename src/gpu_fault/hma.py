from __future__ import annotations

import json
import hashlib
import re
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import Field, model_validator

from gpu_fault.models import StrictModel, WorkloadState
from gpu_fault.policy import (
    FaultPolicyDecision,
    NVIDIA_ALWAYS_FATAL_SXIDS,
    SxidClassification,
    SxidEvent,
    SxidLinkScope,
    XidEvent,
    load_sxid_policy,
)


HMA_HEALTH_STATUS = "sagemaker.amazonaws.com/node-health-status"
HMA_FAULT_TYPES = "sagemaker.amazonaws.com/fault-types"
HMA_FAULT_REASONS = "sagemaker.amazonaws.com/fault-reasons"
HMA_FAULT_DETAILS = "sagemaker.amazonaws.com/fault-details"
HMA_DETECTION_EVENT = "HealthMonitoringAgentDetectionEvent"
_SXID_CATALOG_CODES = frozenset(rule.sxid for rule in load_sxid_policy().rules)

_XID_PATTERN = re.compile(
    r"\bXid\b\s*(?:\((?:PCI:)?([0-9a-fA-F:.]+)\))?\s*"
    r"(?::|=)?\s*(\d+)",
    re.IGNORECASE,
)
# The bare tokens the node-side collectors filter on. A line that carries one
# of them but yields no code to ``_XID_PATTERN`` / ``_SXID_PATTERN`` is a
# format drift the ingest side must name, not swallow: no code is invented and
# the reason travels on the provider signal so the fault ingestion service can
# turn it into a finding.
_XID_TOKEN_PATTERN = re.compile(r"\bXid\b", re.IGNORECASE)
_SXID_TOKEN_PATTERN = re.compile(r"\bSXid\b", re.IGNORECASE)
UNPARSED_XID_REASON = "unparsed_xid_line"
UNPARSED_SXID_REASON = "unparsed_sxid_line"
UNCLASSIFIED_SXID_REASON = "unclassified_sxid"
# HMA cordons a node for faults that never name an XID/SXID (``EfaError``,
# ``InstanceUnreachable``). The provider signal used to carry the cordon with
# no reason at all, so the route answered 200 and nothing downstream saw it.
# It is its own kind, not the ``UNCLASSIFIED_SXID_REASON`` catch-all: the kind
# is what the metric label and the finding's ``metric_name`` carry, so filing
# an EFA/reachability cordon under an SXID name would be wrong recovery input,
# and it would share one health-signal key with a real unclassified SXID
# episode on the same node -- hiding a genuine fabric fault behind a cordon
# (or the reverse). Every table that enumerates the kinds must carry it:
# ``UNRESOLVED_SIGNAL_KINDS`` (episode clear) and ``OPERATOR_REVIEW_METRICS``
# (the notification), both in ``gpu_fault.app.ingest``.
UNSCHEDULABLE_WITHOUT_CODE_REASON = "hma_unschedulable_without_code"


def unresolved_reason_kind(reason: str) -> str:
    """The finding kind an unresolved provider-signal reason maps to."""

    for kind in (
        UNPARSED_XID_REASON,
        UNPARSED_SXID_REASON,
        UNSCHEDULABLE_WITHOUT_CODE_REASON,
    ):
        if reason.startswith(kind):
            return kind
    return UNCLASSIFIED_SXID_REASON


def _unparsed_line_reasons(
    message: str,
    xid_matches: list[tuple[str | None, int, str]],
    sxid_matches: list[tuple[str | None, int, str]],
) -> list[str]:
    reasons = []
    if not xid_matches and _XID_TOKEN_PATTERN.search(message):
        reasons.append(
            f"{UNPARSED_XID_REASON}: an Xid token is present but no "
            "code could be extracted"
        )
    if not sxid_matches and _SXID_TOKEN_PATTERN.search(message):
        reasons.append(
            f"{UNPARSED_SXID_REASON}: an SXid token is present but no "
            "code could be extracted"
        )
    return reasons


def _unresolved_health_status_reason(
    *,
    health_status: str | None,
    unschedulable_taint: bool,
    fault_types: list[str],
    fault_reasons: list[str],
) -> str | None:
    """Why a cordoned node yielded no XID/SXID event, or ``None`` (F4).

    HMA marks the node ``Unschedulable`` -- by label, by ``NoSchedule`` taint,
    or both -- for fault types that carry no code at all (``EfaError``,
    ``InstanceUnreachable``) as well as for XID faults. When no code came out
    of the snapshot the cordon is still a fault the operator must see, so the
    signal says so. The text is built only from the node's own labels: it is
    identical for every observation of the same cordon, so the finding it
    opens dedupes into one episode instead of one per resync.
    """

    if (health_status or "").strip().lower() != "unschedulable" and (
        not unschedulable_taint
    ):
        return None
    types = ", ".join(dict.fromkeys(fault_types)) or "none"
    reasons = ", ".join(dict.fromkeys(fault_reasons)) or "none"
    return (
        f"{UNSCHEDULABLE_WITHOUT_CODE_REASON}: HMA marked the node "
        f"Unschedulable (health_status={health_status or 'absent'}, "
        f"taint={'present' if unschedulable_taint else 'absent'}, "
        f"fault_types=[{types}], fault_reasons=[{reasons}]) but no "
        "XID/SXID code could be extracted"
    )


_XID_REGISTER_PATTERN = re.compile(r"\b0x([0-9a-fA-F]+)\b")
_NVLINK5_XIDS = frozenset(range(144, 151))
_NVLINK5_REGISTER_TOKEN = r"(?:0[xX][0-9a-fA-F]{1,8}|[0-9a-fA-F]{8})"
_NVLINK5_REGISTER_PAYLOAD_PATTERN = re.compile(
    r"\(\s*"
    rf"(?P<intr_info>{_NVLINK5_REGISTER_TOKEN})"
    r"(?:\s*,\s*|\s+)"
    rf"(?P<error_status>{_NVLINK5_REGISTER_TOKEN})"
    + "".join(rf"(?:\s*,\s*|\s+){_NVLINK5_REGISTER_TOKEN}" for _ in range(5))
    + r"\s*\)",
)
_SXID_PATTERN = re.compile(
    r"\bSXid\b\s*(?:\(PCI:([0-9a-fA-F:.]+)\))?\s*"
    r"(?::|=)?\s*(\d+)",
    re.IGNORECASE,
)
_TIMESTAMP_PATTERN = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
    r"(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?"
)
_SXID_SWITCH_PATTERN = re.compile(
    r"\bnvidia-nvswitch(?P<switch>\d+)\s*:", re.IGNORECASE
)
_SXID_LINK_PATTERN = re.compile(r"\bLink\s+(?P<link>\d+)\b", re.IGNORECASE)


def _scoped_event_id(cluster_id: str, event_id: str) -> str:
    digest = hashlib.sha256(cluster_id.encode("utf-8")).hexdigest()[:16]
    return f"{event_id}-cluster-{digest}"


class HmaSource(StrEnum):
    KUBERNETES_NODE = "KUBERNETES_NODE"
    CLOUDWATCH_LOG = "CLOUDWATCH_LOG"
    KERNEL_LOG = "KERNEL_LOG"
    FABRIC_MANAGER_LOG = "FABRIC_MANAGER_LOG"


class HmaCondition(StrictModel):
    type: str
    status: str
    reason: str | None = None
    message: str | None = None
    last_transition_time: datetime | None = None


class HmaTaint(StrictModel):
    key: str
    value: str | None = None
    effect: str | None = None


class HmaNodeSnapshot(StrictModel):
    cluster_id: str
    node_id: str
    observed_at: datetime
    collected_at: datetime | None = None
    labels: dict[str, str] = Field(default_factory=dict)
    annotations: dict[str, str] = Field(default_factory=dict)
    conditions: list[HmaCondition] = Field(default_factory=list)
    taints: list[HmaTaint] = Field(default_factory=list)
    runtime_profile_version: str | None = None
    product: str | None = None
    driver_branch: int | None = Field(default=None, ge=0)
    cuda_version: str | None = None
    workload_state: WorkloadState = WorkloadState.UNKNOWN
    affected_workload_ids: list[str] = Field(default_factory=list)
    checkpoint_manifest_ref: str | None = None
    evidence_ref: str | None = None


class HmaKubernetesNodeEvent(StrictModel):
    cluster_id: str
    observed_at: datetime
    collected_at: datetime | None = None
    node: dict[str, Any]
    runtime_profile_version: str | None = None
    product: str | None = None
    driver_branch: int | None = Field(default=None, ge=0)
    cuda_version: str | None = None
    workload_state: WorkloadState = WorkloadState.UNKNOWN
    affected_workload_ids: list[str] = Field(default_factory=list)
    checkpoint_manifest_ref: str | None = None
    evidence_ref: str | None = None


class HmaCloudWatchLogEvent(StrictModel):
    cluster_id: str
    node_id: str
    log_event_id: str
    observed_at: datetime
    collected_at: datetime | None = None
    message: str
    log_stream: str | None = None
    runtime_profile_version: str | None = None
    product: str | None = None
    driver_branch: int | None = Field(default=None, ge=0)
    cuda_version: str | None = None
    workload_state: WorkloadState = WorkloadState.UNKNOWN
    affected_workload_ids: list[str] = Field(default_factory=list)
    checkpoint_manifest_ref: str | None = None
    evidence_ref: str | None = None


class NvidiaKernelLogEvent(StrictModel):
    cluster_id: str = Field(min_length=1)
    node_id: str = Field(min_length=1)
    record_id: str
    observed_at: datetime
    source_monotonic_us: int | None = Field(default=None, ge=0)
    source_boot_id: str | None = None
    collected_at: datetime | None = None
    message: str
    runtime_profile_version: str | None = None
    product: str | None = None
    driver_branch: int | None = Field(default=None, ge=0)
    cuda_version: str | None = None
    workload_state: WorkloadState = WorkloadState.UNKNOWN
    affected_workload_ids: list[str] = Field(default_factory=list)
    checkpoint_manifest_ref: str | None = None
    evidence_ref: str | None = None


class FabricManagerLogEvent(StrictModel):
    cluster_id: str = Field(min_length=1)
    node_id: str = Field(min_length=1)
    record_id: str
    observed_at: datetime
    collected_at: datetime | None = None
    message: str
    source: str
    unit: str | None = None
    fields: dict[str, str] = Field(default_factory=dict)
    runtime_profile_version: str | None = None
    product: str | None = None
    driver_branch: int | None = Field(default=None, ge=0)
    cuda_version: str | None = None
    workload_state: WorkloadState = WorkloadState.UNKNOWN
    affected_workload_ids: list[str] = Field(default_factory=list)
    checkpoint_manifest_ref: str | None = None
    evidence_ref: str | None = None


class HmaProviderSignal(StrictModel):
    signal_id: str
    source: HmaSource
    cluster_id: str
    node_id: str
    observed_at: datetime
    health_status: str | None = None
    fault_types: list[str] = Field(default_factory=list)
    fault_reasons: list[str] = Field(default_factory=list)
    fault_details: list[str] = Field(default_factory=list)
    unschedulable_taint: bool = False
    raw_message: str | None = None
    xid_codes: list[int] = Field(default_factory=list)
    sxid_codes: list[int] = Field(default_factory=list)
    unresolved_reasons: list[str] = Field(default_factory=list)


class HmaNormalizedBatch(StrictModel):
    provider_signals: list[HmaProviderSignal]
    xid_events: list[XidEvent] = Field(default_factory=list)
    sxid_events: list[SxidEvent] = Field(default_factory=list)


class HmaDeploymentDiscovery(StrictModel):
    namespace: str
    name: str
    image: str | None = None
    desired_nodes: int = 0
    ready_nodes: int = 0
    disabled_xid_checks: list[int] = Field(default_factory=list)
    container_ports: list[int] = Field(default_factory=list)
    matching_services: list[str] = Field(default_factory=list)
    metrics_available: bool = False

    def covers_xid(self, xid: int) -> bool:
        return xid not in self.disabled_xid_checks


class HmaDeploymentProbe(StrictModel):
    daemonset: dict[str, Any]
    services: list[dict[str, Any]] = Field(default_factory=list)


class HmaIngestionResult(StrictModel):
    normalized: HmaNormalizedBatch
    decisions: list[FaultPolicyDecision] = Field(default_factory=list)
    #: How many fault lines in this batch could not be resolved into an
    #: XID/SXID event. ``decisions == [] and unresolved == 0`` is the only
    #: shape that means "nothing to act on"; a non-zero count says the
    #: provider reported a fault the control plane could not read, and an
    #: operator-review finding was opened for it. Always derived from the
    #: batch below so no caller can report a different number.
    unresolved: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _count_unresolved(self) -> HmaIngestionResult:
        self.unresolved = sum(
            len(signal.unresolved_reasons)
            for signal in self.normalized.provider_signals
        )
        return self


class HyperPodHmaNormalizer:
    """Normalizes documented HMA Node and CloudWatch contracts."""

    def normalize_kubernetes_node(
        self, event: HmaKubernetesNodeEvent
    ) -> HmaNormalizedBatch:
        metadata = event.node.get("metadata", {})
        spec = event.node.get("spec", {})
        status = event.node.get("status", {})
        node_id = metadata.get("name")
        if not node_id:
            raise ValueError("Kubernetes Node metadata.name is required")

        snapshot = HmaNodeSnapshot(
            cluster_id=event.cluster_id,
            node_id=node_id,
            observed_at=event.observed_at,
            collected_at=event.collected_at or event.observed_at,
            labels=metadata.get("labels") or {},
            annotations=metadata.get("annotations") or {},
            conditions=[
                HmaCondition(
                    type=item["type"],
                    status=str(item["status"]),
                    reason=item.get("reason"),
                    message=item.get("message"),
                    last_transition_time=item.get("lastTransitionTime"),
                )
                for item in status.get("conditions") or []
                if item.get("type") and item.get("status") is not None
            ],
            taints=[
                HmaTaint(
                    key=item["key"],
                    value=item.get("value"),
                    effect=item.get("effect"),
                )
                for item in spec.get("taints") or []
                if item.get("key")
            ],
            runtime_profile_version=event.runtime_profile_version,
            product=event.product,
            driver_branch=event.driver_branch,
            cuda_version=event.cuda_version,
            workload_state=event.workload_state,
            affected_workload_ids=event.affected_workload_ids,
            checkpoint_manifest_ref=event.checkpoint_manifest_ref,
            evidence_ref=event.evidence_ref or f"k8s://nodes/{node_id}",
        )
        return self.normalize_node(snapshot)

    def normalize_node(self, snapshot: HmaNodeSnapshot) -> HmaNormalizedBatch:
        details = self._fault_detail_strings(
            snapshot.annotations.get(HMA_FAULT_DETAILS)
        )
        condition_text = [
            " ".join(
                item
                for item in (
                    condition.type,
                    condition.reason,
                    condition.message,
                    (
                        condition.last_transition_time.isoformat()
                        if condition.last_transition_time
                        else None
                    ),
                )
                if item
            )
            for condition in snapshot.conditions
            if condition.status.lower() == "true"
        ]
        texts = list(
            dict.fromkeys(
                [
                    *details,
                    *condition_text,
                    snapshot.labels.get(HMA_FAULT_REASONS, ""),
                ]
            )
        )
        source_id = self._safe_id(snapshot.node_id)
        xid_matches = self._matches(texts, _XID_PATTERN)
        sxid_matches = self._matches(texts, _SXID_PATTERN)
        signal = HmaProviderSignal(
            signal_id=f"hma-node-{source_id}",
            source=HmaSource.KUBERNETES_NODE,
            cluster_id=snapshot.cluster_id,
            node_id=snapshot.node_id,
            observed_at=snapshot.observed_at,
            health_status=snapshot.labels.get(HMA_HEALTH_STATUS),
            fault_types=self._split_label(snapshot.labels.get(HMA_FAULT_TYPES)),
            fault_reasons=self._split_label(snapshot.labels.get(HMA_FAULT_REASONS)),
            fault_details=details,
            unschedulable_taint=any(
                taint.key == HMA_HEALTH_STATUS
                and taint.value == "Unschedulable"
                and taint.effect == "NoSchedule"
                for taint in snapshot.taints
            ),
            xid_codes=sorted({code for _, code, _ in xid_matches}),
            sxid_codes=sorted({code for _, code, _ in sxid_matches}),
        )
        xids = [
            self._xid_event(
                snapshot,
                event_id=_scoped_event_id(
                    snapshot.cluster_id,
                    (f"{signal.signal_id}-xid-{code}-{self._content_id(text)}"),
                ),
                xid=code,
                pci_bdf=pci_bdf,
                raw_message=text,
                source_event_time=self._text_timestamp(text),
            )
            for pci_bdf, code, text in self._deduplicate_matches(xid_matches)
        ]
        sxids, unresolved = self._sxid_events(
            snapshot,
            signal.signal_id,
            sxid_matches,
        )
        unresolved = [
            *_unparsed_line_reasons("\n".join(texts), xid_matches, sxid_matches),
            *unresolved,
        ]
        if not unresolved and not xid_matches and not sxid_matches:
            # Only when nothing else already explains the missing code: an
            # Unschedulable node whose fault text drifted is reported as the
            # drift (a narrower reason), not twice.
            health_reason = _unresolved_health_status_reason(
                health_status=signal.health_status,
                unschedulable_taint=signal.unschedulable_taint,
                fault_types=signal.fault_types,
                fault_reasons=signal.fault_reasons,
            )
            if health_reason is not None:
                unresolved = [health_reason]
        if unresolved:
            signal = signal.model_copy(update={"unresolved_reasons": unresolved})
        return HmaNormalizedBatch(
            provider_signals=[signal],
            xid_events=xids,
            sxid_events=sxids,
        )

    def discover_deployment(
        self,
        daemonset: dict[str, Any],
        services: list[dict[str, Any]] | None = None,
    ) -> HmaDeploymentDiscovery:
        metadata = daemonset.get("metadata", {})
        spec = daemonset.get("spec", {})
        pod_spec = spec.get("template", {}).get("spec", {})
        containers = pod_spec.get("containers", [])
        container = next(
            (
                item
                for item in containers
                if item.get("name") == "health-monitoring-agent"
            ),
            containers[0] if containers else {},
        )
        env = {
            item.get("name"): item.get("value")
            for item in container.get("env", [])
            if item.get("name")
        }
        disabled = []
        for value in re.split(r"[\s,;]+", env.get("DP_DISABLE_HEALTHCHECKS") or ""):
            if value.isdigit():
                disabled.append(int(value))
        ports = [
            item["containerPort"]
            for item in container.get("ports", [])
            if isinstance(item.get("containerPort"), int)
        ]
        labels = spec.get("template", {}).get("metadata", {}).get("labels", {})
        matching_services = []
        for service in services or []:
            selector = service.get("spec", {}).get("selector", {})
            if selector and all(
                labels.get(key) == value for key, value in selector.items()
            ):
                matching_services.append(service.get("metadata", {}).get("name", ""))
        status = daemonset.get("status", {})
        return HmaDeploymentDiscovery(
            namespace=metadata.get("namespace", "default"),
            name=metadata.get("name", "health-monitoring-agent"),
            image=container.get("image"),
            desired_nodes=status.get("desiredNumberScheduled", 0),
            ready_nodes=status.get("numberReady", 0),
            disabled_xid_checks=sorted(set(disabled)),
            container_ports=ports,
            matching_services=[item for item in matching_services if item],
            metrics_available=bool(ports and matching_services),
        )

    def normalize_cloudwatch(self, event: HmaCloudWatchLogEvent) -> HmaNormalizedBatch:
        payload = self._json_object(event.message)
        details = self._details_object(payload)
        raw = str(details.get("message") or event.message)
        observed_at = self._timestamp(details.get("timestamp"), event.observed_at)
        xid_matches = self._matches([raw], _XID_PATTERN)
        sxid_matches = self._matches([raw], _SXID_PATTERN)
        reasons = [str(details["reason"])] if details.get("reason") else []
        unresolved = []
        if payload and HMA_DETECTION_EVENT not in payload:
            unresolved.append("CloudWatch message is not an HMA detection event")
        # A detection event whose Xid/SXid text drifted is the same silent
        # 200 the kernel route already refuses to answer (F4). The node's
        # health status is not in this contract -- only the log line is -- so
        # the cordon itself is reported by the Node source, not from here.
        unresolved.extend(_unparsed_line_reasons(raw, xid_matches, sxid_matches))
        signal = HmaProviderSignal(
            signal_id=f"hma-log-{self._safe_id(event.log_event_id)}",
            source=HmaSource.CLOUDWATCH_LOG,
            cluster_id=event.cluster_id,
            node_id=event.node_id,
            observed_at=observed_at,
            fault_reasons=reasons,
            fault_details=[raw],
            raw_message=event.message,
            xid_codes=sorted({code for _, code, _ in xid_matches}),
            sxid_codes=sorted({code for _, code, _ in sxid_matches}),
            unresolved_reasons=unresolved,
        )
        xids = [
            self._xid_event(
                event,
                event_id=_scoped_event_id(
                    event.cluster_id,
                    f"{signal.signal_id}-xid-{code}",
                ),
                xid=code,
                pci_bdf=pci_bdf,
                raw_message=text,
                observed_at=observed_at,
                source_event_time=observed_at,
            )
            for pci_bdf, code, text in self._deduplicate_matches(xid_matches)
        ]
        sxids, sxid_unresolved = self._sxid_events(
            event,
            signal.signal_id,
            sxid_matches,
            observed_at=observed_at,
        )
        if sxid_unresolved:
            signal = signal.model_copy(
                update={
                    "unresolved_reasons": [
                        *signal.unresolved_reasons,
                        *sxid_unresolved,
                    ]
                }
            )
        return HmaNormalizedBatch(
            provider_signals=[signal],
            xid_events=xids,
            sxid_events=sxids,
        )

    def normalize_kernel(self, event: NvidiaKernelLogEvent) -> HmaNormalizedBatch:
        xid_matches = self._matches([event.message], _XID_PATTERN)
        sxid_matches = self._matches([event.message], _SXID_PATTERN)
        signal = HmaProviderSignal(
            signal_id=(f"kernel-log-{self._safe_id(event.record_id)}"),
            source=HmaSource.KERNEL_LOG,
            cluster_id=event.cluster_id,
            node_id=event.node_id,
            observed_at=event.observed_at,
            fault_details=[event.message],
            raw_message=event.message,
            xid_codes=sorted({code for _, code, _ in xid_matches}),
            sxid_codes=sorted({code for _, code, _ in sxid_matches}),
        )
        xids = [
            self._xid_event(
                event,
                event_id=_scoped_event_id(
                    event.cluster_id,
                    f"{signal.signal_id}-xid-{code}",
                ),
                xid=code,
                pci_bdf=pci_bdf,
                raw_message=text,
                source_event_time=None,
            )
            for pci_bdf, code, text in self._deduplicate_matches(xid_matches)
        ]
        sxids, unresolved = self._sxid_events(
            event,
            signal.signal_id,
            sxid_matches,
        )
        unresolved = [
            *_unparsed_line_reasons(event.message, xid_matches, sxid_matches),
            *unresolved,
        ]
        if unresolved:
            signal = signal.model_copy(update={"unresolved_reasons": unresolved})
        return HmaNormalizedBatch(
            provider_signals=[signal],
            xid_events=xids,
            sxid_events=sxids,
        )

    def normalize_fabric_manager(
        self, event: FabricManagerLogEvent
    ) -> HmaNormalizedBatch:
        xid_matches = self._matches([event.message], _XID_PATTERN)
        sxid_matches = self._matches([event.message], _SXID_PATTERN)
        signal = HmaProviderSignal(
            signal_id=(f"fabric-manager-log-{self._safe_id(event.record_id)}"),
            source=HmaSource.FABRIC_MANAGER_LOG,
            cluster_id=event.cluster_id,
            node_id=event.node_id,
            observed_at=event.observed_at,
            fault_details=[event.message],
            raw_message=event.message,
            xid_codes=sorted({code for _, code, _ in xid_matches}),
            sxid_codes=sorted({code for _, code, _ in sxid_matches}),
        )
        xids = [
            self._xid_event(
                event,
                event_id=_scoped_event_id(
                    event.cluster_id,
                    f"{signal.signal_id}-xid-{code}",
                ),
                xid=code,
                pci_bdf=pci_bdf,
                raw_message=text,
                source_event_time=self._text_timestamp(text),
            )
            for pci_bdf, code, text in self._deduplicate_matches(xid_matches)
        ]
        sxids, unresolved = self._sxid_events(
            event,
            signal.signal_id,
            sxid_matches,
        )
        unresolved = [
            *_unparsed_line_reasons(event.message, xid_matches, sxid_matches),
            *unresolved,
        ]
        if unresolved:
            signal = signal.model_copy(update={"unresolved_reasons": unresolved})
        return HmaNormalizedBatch(
            provider_signals=[signal],
            xid_events=xids,
            sxid_events=sxids,
        )

    @staticmethod
    def _xid_event(
        source,
        *,
        event_id: str,
        xid: int,
        pci_bdf: str | None,
        raw_message: str,
        observed_at: datetime | None = None,
        source_event_time: datetime | None = None,
    ) -> XidEvent:
        intr_info, error_status = HyperPodHmaNormalizer._xid_nvlink5_registers(
            raw_message, xid
        )
        drill_id = HyperPodHmaNormalizer._drill_id(raw_message)
        return XidEvent(
            event_id=event_id,
            cluster_id=source.cluster_id,
            node_id=source.node_id,
            observed_at=observed_at or source.observed_at,
            source_event_time=source_event_time,
            source_monotonic_us=getattr(source, "source_monotonic_us", None),
            source_boot_id=getattr(source, "source_boot_id", None),
            collected_at=(getattr(source, "collected_at", None) or source.observed_at),
            event_source=(
                HmaSource.KERNEL_LOG.value
                if isinstance(source, NvidiaKernelLogEvent)
                else HmaSource.FABRIC_MANAGER_LOG.value
                if isinstance(source, FabricManagerLogEvent)
                else HmaSource.CLOUDWATCH_LOG.value
                if isinstance(source, HmaCloudWatchLogEvent)
                else HmaSource.KUBERNETES_NODE.value
            ),
            xid=xid,
            pci_bdf=pci_bdf,
            product=source.product,
            driver_branch=source.driver_branch,
            cuda_version=source.cuda_version,
            runtime_profile_version=source.runtime_profile_version,
            workload_state=source.workload_state,
            affected_workload_ids=source.affected_workload_ids,
            checkpoint_manifest_ref=source.checkpoint_manifest_ref,
            intr_info=intr_info,
            error_status=error_status,
            registers=(
                HyperPodHmaNormalizer._xid_registers(raw_message, xid)
                if xid == 74
                else []
            ),
            nvlink_link_id=(
                HyperPodHmaNormalizer._xid_nvlink_id(raw_message) if xid == 74 else None
            ),
            nvlink_link_identity_source=(
                "explicit-kernel-message"
                if xid == 74
                and HyperPodHmaNormalizer._xid_nvlink_id(raw_message) is not None
                else None
            ),
            raw_message=raw_message,
            evidence_ref=source.evidence_ref,
            drill_id=drill_id,
        )

    @staticmethod
    def _drill_id(message: str) -> str | None:
        match = re.search(
            r"\bdrill[_-]id=([A-Za-z0-9][A-Za-z0-9._:-]{0,127})\b",
            message,
            re.IGNORECASE,
        )
        return match.group(1) if match else None

    @staticmethod
    def _xid_registers(message: str, xid: int) -> list[int]:
        for match in _XID_PATTERN.finditer(message):
            if int(match.group(2)) != xid:
                continue
            return [
                int(value, 16)
                for value in _XID_REGISTER_PATTERN.findall(message[match.end() :])
            ]
        return []

    @staticmethod
    def _xid_nvlink5_registers(message: str, xid: int) -> tuple[int | None, int | None]:
        if xid not in _NVLINK5_XIDS:
            return None, None
        matches = list(_XID_PATTERN.finditer(message))
        for index, match in enumerate(matches):
            if int(match.group(2)) != xid:
                continue
            end = (
                matches[index + 1].start() if index + 1 < len(matches) else len(message)
            )
            payload = _NVLINK5_REGISTER_PAYLOAD_PATTERN.search(
                message, match.end(), end
            )
            if payload is None:
                continue
            return (
                int(payload.group("intr_info"), 16),
                int(payload.group("error_status"), 16),
            )
        return None, None

    @staticmethod
    def _xid_nvlink_id(message: str) -> int | None:
        match = re.search(
            r"\b(?:NVLink|Link)\s*(?:ID\s*)?[:=#]?\s*(\d+)\b",
            message,
            re.IGNORECASE,
        )
        return int(match.group(1)) if match else None

    def _sxid_events(
        self,
        source,
        signal_id: str,
        matches: list[tuple[str | None, int, str]],
        *,
        observed_at: datetime | None = None,
    ) -> tuple[list[SxidEvent], list[str]]:
        events = []
        unresolved = []
        for pci_bdf, code, text in self._deduplicate_matches(matches):
            classification = self._sxid_classification(text, code)
            if classification is None:
                unresolved.append(
                    f"SXID {code} classification is absent; raw "
                    "Fabric Manager evidence is required"
                )
                continue
            events.append(
                SxidEvent(
                    event_id=_scoped_event_id(
                        source.cluster_id,
                        (f"{signal_id}-sxid-{code}-{self._content_id(text)}"),
                    ),
                    cluster_id=source.cluster_id,
                    node_id=source.node_id,
                    observed_at=observed_at or source.observed_at,
                    source_event_time=(
                        observed_at
                        if observed_at is not None
                        else self._text_timestamp(text)
                    ),
                    source_monotonic_us=getattr(source, "source_monotonic_us", None),
                    source_boot_id=getattr(source, "source_boot_id", None),
                    collected_at=(
                        getattr(source, "collected_at", None) or source.observed_at
                    ),
                    event_source=(
                        HmaSource.KERNEL_LOG.value
                        if isinstance(source, NvidiaKernelLogEvent)
                        else HmaSource.FABRIC_MANAGER_LOG.value
                        if isinstance(source, FabricManagerLogEvent)
                        else HmaSource.CLOUDWATCH_LOG.value
                        if isinstance(source, HmaCloudWatchLogEvent)
                        else HmaSource.KUBERNETES_NODE.value
                    ),
                    sxid=code,
                    classification=classification,
                    classification_source=(
                        "NVIDIA_FABRIC_MANAGER_CATALOG"
                        if code in _SXID_CATALOG_CODES
                        else "NVIDIA_FABRIC_MANAGER_RUNTIME"
                    ),
                    link_scope=self._sxid_scope(
                        text,
                        product=source.product,
                        classification=classification,
                    ),
                    link_scope_source=None,
                    switch_id=self._sxid_switch(text),
                    pci_bdf=pci_bdf,
                    port=self._sxid_link(text),
                    product=source.product,
                    runtime_profile_version=(source.runtime_profile_version),
                    workload_state=source.workload_state,
                    affected_workload_ids=(source.affected_workload_ids),
                    checkpoint_manifest_ref=(source.checkpoint_manifest_ref),
                    raw_message=text,
                    evidence_ref=source.evidence_ref,
                    drill_id=self._drill_id(text),
                )
            )
        return events, unresolved

    @staticmethod
    def _matches(
        texts: list[str], pattern: re.Pattern[str]
    ) -> list[tuple[str | None, int, str]]:
        matches = []
        for text in texts:
            for match in pattern.finditer(text):
                matches.append((match.group(1), int(match.group(2)), text))
        return matches

    @staticmethod
    def _deduplicate_matches(matches):
        unique = {}
        for pci_bdf, code, text in matches:
            timestamp = _TIMESTAMP_PATTERN.search(text)
            occurrence = (
                HyperPodHmaNormalizer._canonical_timestamp(timestamp.group(0))
                if timestamp
                else HyperPodHmaNormalizer._content_id(text)
            )
            unique.setdefault((pci_bdf, code, occurrence), text)
        return [(pci_bdf, code, text) for (pci_bdf, code, _), text in unique.items()]

    @staticmethod
    def _sxid_classification(
        text: str,
        code: int,
    ) -> SxidClassification | None:
        lowered = text.lower()
        if "always fatal" in lowered:
            return SxidClassification.ALWAYS_FATAL
        if "non-fatal" in lowered or "nonfatal" in lowered:
            return SxidClassification.NON_FATAL
        if "fatal" in lowered:
            return (
                SxidClassification.ALWAYS_FATAL
                if code in NVIDIA_ALWAYS_FATAL_SXIDS
                else SxidClassification.FATAL
            )
        return None

    @staticmethod
    def _sxid_scope(
        text: str,
        *,
        product: str | None,
        classification: SxidClassification,
    ) -> SxidLinkScope:
        # Raw descriptions are evidence, not trusted topology. The API
        # resolves switch/port scope from fresh topology telemetry or a
        # pinned product invariant.
        return SxidLinkScope.UNKNOWN

    @staticmethod
    def _sxid_switch(text: str) -> str | None:
        match = _SXID_SWITCH_PATTERN.search(text)
        return match.group("switch") if match else None

    @staticmethod
    def _sxid_link(text: str) -> str | None:
        match = _SXID_LINK_PATTERN.search(text)
        return match.group("link") if match else None

    @staticmethod
    def _fault_detail_strings(value: str | None) -> list[str]:
        if not value:
            return []
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [value]
        if isinstance(parsed, list):
            return list(
                dict.fromkeys(HyperPodHmaNormalizer._detail(item) for item in parsed)
            )
        if isinstance(parsed, dict):
            if isinstance(parsed.get("faults"), list):
                return list(
                    dict.fromkeys(
                        HyperPodHmaNormalizer._detail(item) for item in parsed["faults"]
                    )
                )
            if any(key in parsed for key in ("message", "reason", "timestamp")):
                return [HyperPodHmaNormalizer._detail(parsed)]
            return list(
                dict.fromkeys(
                    HyperPodHmaNormalizer._detail(item) for item in parsed.values()
                )
            )
        return [str(parsed)]

    @staticmethod
    def _detail(value: Any) -> str:
        if isinstance(value, str):
            return value
        return json.dumps(value, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _json_object(value: str) -> dict[str, Any]:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    @staticmethod
    def _details_object(payload: dict[str, Any]) -> dict[str, Any]:
        for key in ("details: ", "details:", "details"):
            value = payload.get(key)
            if isinstance(value, dict):
                return value
        return {}

    @staticmethod
    def _timestamp(value: Any, fallback: datetime) -> datetime:
        if not isinstance(value, str):
            return fallback
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return fallback
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)

    @staticmethod
    def _text_timestamp(value: str) -> datetime | None:
        match = _TIMESTAMP_PATTERN.search(value)
        if match is None:
            return None
        return HyperPodHmaNormalizer._timestamp(
            match.group(0), datetime.now(timezone.utc)
        )

    @staticmethod
    def _split_label(value: str | None) -> list[str]:
        if not value:
            return []
        return [item.strip() for item in re.split(r"[,;]", value) if item.strip()]

    @staticmethod
    def _safe_id(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]", "-", value)

    @staticmethod
    def _content_id(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _canonical_timestamp(value: str) -> str:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).isoformat()
        except ValueError:
            return value
