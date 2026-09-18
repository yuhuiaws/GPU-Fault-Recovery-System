"""Build GPU composites and join PCIe findings to persisted kernel XID evidence."""

from __future__ import annotations

import math
import re
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Sequence

from gpu_fault.gpu_metric_models import (
    GpuFindingState,
    GpuHealthFinding,
    GpuHealthSeverity,
)
from gpu_fault.gpu_power_policy import NVIDIA_DCGM_HEALTH_REFERENCE
from gpu_fault.models import RecoveryAction
from gpu_fault.policy import ActionDisposition, FaultPolicyDecision, XidEvent
from gpu_fault.store.shared.errors import NotFoundError

if TYPE_CHECKING:
    from gpu_fault.gpu_metrics import (
        GpuMetricBatch,
        GpuMetricLatest,
        GpuMetricSample,
        GpuMetricsService,
    )


PCIE_XID_RULE = "PCIE_XID_LINK_FAILURE"
GPU_COMPOSITE_TRANSACTION = "gpu-composite-correlation"
SITE_CORRELATION_POLICY_VERSION = "site-dcgm-correlation/v1"
_LINK_XIDS = frozenset({32, 79})
_PCI_BDF = re.compile(
    r"(?:(?P<domain>[0-9a-f]{4}|[0-9a-f]{8}):)?"
    r"(?P<bus>[0-9a-f]{2}):(?P<device>[0-9a-f]{2})(?:\.(?P<function>[0-7]))?"
)


def composite_finding(
    batch: GpuMetricBatch | XidEvent,
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
        (component.pci_bdf for component in components if component.pci_bdf), None
    )
    evidence_refs = list(
        dict.fromkeys(
            component.evidence_ref for component in components if component.evidence_ref
        )
    )
    batch_id = (
        f"kernel-xid-{batch.event_id}"
        if isinstance(batch, XidEvent)
        else batch.batch_id
    )
    return GpuHealthFinding(
        finding_id=f"{batch_id}-{scope_key}-composite-{rule_id}",
        cluster_id=batch.cluster_id,
        node_id=batch.node_id,
        observed_at=batch.observed_at,
        severity=severity,
        reason=reason,
        canonical_name=f"composite:{rule_id}",
        value=1,
        gpu_uuid=gpu_uuid,
        pci_bdf=pci_bdf,
        evidence_ref=evidence_refs[0]
        if len(evidence_refs) == 1
        else batch.evidence_ref,
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


def composite_update_time(
    batch: GpuMetricBatch,
    candidate: GpuHealthFinding | None,
    previous: GpuFindingState | None,
    metrics: dict[str, GpuMetricLatest],
    components: dict[str, GpuHealthFinding],
) -> datetime:
    if candidate is not None:
        return candidate.observed_at
    if (
        previous is not None
        and previous.finding is not None
        and previous.finding.correlation_rule_id == PCIE_XID_RULE
        and "kernel_xid" in previous.finding.component_metrics
        and "pcie_replay_total" not in components
        and (pcie := metrics.get("pcie_replay_total")) is not None
        and pcie.observed_at == batch.observed_at
    ):
        # A newly accepted healthy PCIe observation clears the join even if
        # the kernel clock was ahead. Keep the Store's monotonic state fence.
        return max(batch.observed_at, previous.observed_at + timedelta(microseconds=1))
    return batch.observed_at


def metric_counter_delta(
    sample: GpuMetricSample, previous: GpuMetricLatest | None, *, is_counter: bool
) -> float | None:
    if previous is None or not is_counter:
        return None
    if sample.canonical_name == "pcie_replay_total":
        boots = {
            value
            for labels in (sample.labels, previous.sample.labels)
            for name in ("source_boot_id", "boot_id")
            if (value := labels.get(name))
        }
        if len(boots) > 1 or min(sample.value, previous.sample.value) < 0:
            return None
    return max(0.0, sample.value - previous.sample.value)


def pci_bdf_matches(left: str, right: str) -> bool:
    first = _PCI_BDF.fullmatch(left.strip().lower())
    second = _PCI_BDF.fullmatch(right.strip().lower())
    if first is None or second is None:
        return False
    for name in ("domain", "bus", "device", "function"):
        a, b = first.group(name), second.group(name)
        if a is not None and b is not None and int(a, 16) != int(b, 16):
            return False
    return True


def is_pcie_kernel_xid(event: XidEvent) -> bool:
    return (
        event.event_source == "KERNEL_LOG"
        and event.xid in _LINK_XIDS
        and not event.synthetic
    )


def _pcie_threshold(finding: GpuHealthFinding, threshold: float) -> bool:
    return (
        finding.finding_kind == "METRIC"
        and finding.canonical_name == "pcie_replay_total"
        and finding.rate_per_minute is not None
        and math.isfinite(finding.rate_per_minute)
        and finding.rate_per_minute > threshold
    )


def kernel_xid_events(
    service: GpuMetricsService,
    batch: GpuMetricBatch,
    components: Sequence[GpuHealthFinding],
) -> list[XidEvent]:
    window = timedelta(seconds=service.thresholds.correlation_window_seconds)
    pcie = [
        finding
        for finding in components
        if _pcie_threshold(
            finding, service.thresholds.pcie_replay_rate_warning_per_minute
        )
        and abs(finding.observed_at - batch.observed_at) <= window
    ]
    if not pcie:
        return []
    events: list[XidEvent] = service.store.list_xid_events(
        batch.cluster_id,
        batch.node_id,
        observed_after=batch.observed_at - window,
        observed_before=batch.observed_at + window,
    )
    return [
        event
        for event in events
        if is_pcie_kernel_xid(event)
        and any(_same_device(event, finding) for finding in pcie)
    ]


def _trusted_policy(event: XidEvent, decision: FaultPolicyDecision | None) -> bool:
    if decision is None:
        return False
    marker = decision.marker
    return (
        decision.event_id == event.event_id
        and decision.disposition
        in {ActionDisposition.EXECUTABLE, ActionDisposition.MONITOR_ONLY}
        and marker.trusted
        and marker.cluster_id == event.cluster_id
        and marker.scope.node_ids == [event.node_id]
        and marker.observed_at == event.observed_at
        and marker.event_source == event.event_source
        and marker.source_boot_id == event.source_boot_id
        and (not event.gpu_uuid or marker.scope.gpu_uuids == [event.gpu_uuid])
        and (not event.pci_bdf or marker.scope.pci_bdfs == [event.pci_bdf])
    )


def _same_device(event: XidEvent, finding: GpuHealthFinding) -> bool:
    if event.gpu_uuid and finding.gpu_uuid and event.gpu_uuid != finding.gpu_uuid:
        return False
    if event.pci_bdf and finding.pci_bdf:
        if not pci_bdf_matches(event.pci_bdf, finding.pci_bdf):
            return False
    return bool(
        (event.gpu_uuid and event.gpu_uuid == finding.gpu_uuid)
        or (event.pci_bdf and finding.pci_bdf)
    )


def _same_boot(
    service: GpuMetricsService,
    event: XidEvent,
    metric: GpuMetricLatest,
) -> bool:
    boots = {
        value
        for value in (
            event.source_boot_id,
            metric.sample.labels.get("source_boot_id"),
            metric.sample.labels.get("boot_id"),
        )
        if value
    }
    inventory = service.store.get_gpu_inventory_snapshot(
        event.cluster_id, event.node_id
    )
    if inventory is not None:
        if (
            inventory.cluster_id != event.cluster_id
            or inventory.node_id != event.node_id
        ):
            return False
        boots.add(inventory.source_boot_id)
        devices = [
            device
            for device in inventory.devices
            if (not metric.sample.gpu_uuid or device.gpu_uuid == metric.sample.gpu_uuid)
            and (
                not metric.sample.pci_bdf
                or pci_bdf_matches(device.pci_bdf, metric.sample.pci_bdf)
            )
            and (not event.gpu_uuid or device.gpu_uuid == event.gpu_uuid)
            and (not event.pci_bdf or pci_bdf_matches(device.pci_bdf, event.pci_bdf))
        ]
        if len(devices) != 1:
            return False
    try:
        agent = service.store.get_agent(event.cluster_id, event.node_id)
    except NotFoundError:
        agent = None
    if agent is not None:
        if agent.cluster_id != event.cluster_id or agent.node_id != event.node_id:
            return False
        if agent.boot_id:
            boots.add(agent.boot_id)
    return len(boots) <= 1


def _unique_pci_device(
    service: GpuMetricsService, event: XidEvent, finding: GpuHealthFinding
) -> bool:
    if event.gpu_uuid or not event.pci_bdf:
        return True
    window = timedelta(seconds=service.thresholds.correlation_window_seconds)
    latest: list[GpuMetricLatest] = service.store.list_gpu_metrics_latest(
        event.cluster_id, event.node_id
    )
    matches = {
        item.sample.gpu_uuid or item.sample.pci_bdf
        for item in latest
        if item.sample.pci_bdf
        and abs(item.observed_at - event.observed_at) <= window
        and pci_bdf_matches(event.pci_bdf, item.sample.pci_bdf)
    }
    return matches == {finding.gpu_uuid or finding.pci_bdf}


def pcie_xid_link_failure(
    service: GpuMetricsService,
    trigger: GpuMetricBatch | XidEvent,
    gpu_key: str,
    components: dict[str, GpuHealthFinding],
    metrics: dict[str, GpuMetricLatest],
    previous: GpuFindingState | None,
    kernel_events: Sequence[XidEvent],
) -> GpuHealthFinding | None:
    pcie = components.get("pcie_replay_total")
    metric = metrics.get("pcie_replay_total")
    window = timedelta(seconds=service.thresholds.correlation_window_seconds)
    if (
        pcie is None
        or metric is None
        or pcie.cluster_id != trigger.cluster_id
        or pcie.node_id != trigger.node_id
        or metric.cluster_id != trigger.cluster_id
        or metric.node_id != trigger.node_id
        or metric.observed_at != pcie.observed_at
        or not _pcie_threshold(
            pcie, service.thresholds.pcie_replay_rate_warning_per_minute
        )
        or abs(pcie.observed_at - trigger.observed_at) > window
    ):
        return None
    kernel = None
    for event in sorted(
        kernel_events, key=lambda item: (item.observed_at, item.event_id), reverse=True
    ):
        if (
            not is_pcie_kernel_xid(event)
            or event.cluster_id != trigger.cluster_id
            or event.node_id != trigger.node_id
            or abs(event.observed_at - trigger.observed_at) > window
            or abs(event.observed_at - pcie.observed_at) > window
            or (
                event.source_event_time is not None
                and (
                    event.source_event_time.tzinfo is None
                    or abs(event.source_event_time - pcie.observed_at) > window
                )
            )
            or len(
                {
                    version
                    for version in (
                        event.runtime_profile_version,
                        pcie.runtime_profile_version,
                        trigger.runtime_profile_version,
                    )
                    if version
                }
            )
            > 1
            or not _same_device(event, pcie)
            or (
                previous is not None
                and previous.finding is None
                and event.observed_at <= previous.observed_at
            )
        ):
            continue
        decision = service.store.get_xid_policy_decision(event.event_id)
        if (
            _trusted_policy(event, decision)
            and _same_boot(service, event, metric)
            and _unique_pci_device(service, event, pcie)
        ):
            kernel = event
            break
    gauge = metrics.get("xid_last_error")
    if kernel is None:
        if (
            isinstance(trigger, XidEvent)
            or gauge is None
            or gauge.sample.value not in _LINK_XIDS
            or abs(gauge.observed_at - trigger.observed_at) > window
            or abs(gauge.observed_at - pcie.observed_at) > window
        ):
            return None
        xid = int(gauge.sample.value)
    else:
        xid = kernel.xid
    finding = composite_finding(
        trigger,
        scope_key=gpu_key,
        rule_id=PCIE_XID_RULE,
        components=[pcie],
        component_metrics=[
            "pcie_replay_total",
            "kernel_xid" if kernel is not None else "xid_last_error",
        ],
        severity=GpuHealthSeverity.CRITICAL,
        action=RecoveryAction.DRAIN,
        reason=(
            f"PCIe replay threshold was correlated with XID {xid}"
            + (f" from kernel event {kernel.event_id}" if kernel is not None else "")
        ),
        confidence="HIGH",
    )
    if kernel is None:
        return finding
    return finding.model_copy(
        update={
            "observed_at": max(
                trigger.observed_at, pcie.observed_at, kernel.observed_at
            ),
            "component_evidence_refs": list(
                dict.fromkeys(
                    [
                        *finding.component_evidence_refs,
                        kernel.evidence_ref or f"xid-event://{kernel.event_id}",
                    ]
                )
            ),
        }
    )


def correlate_kernel_xid(
    service: GpuMetricsService, event: XidEvent
) -> list[GpuHealthFinding]:
    components: list[GpuHealthFinding] = service.store.list_gpu_findings(
        event.cluster_id, event.node_id, active_only=True
    )
    pcie_components = [
        finding
        for finding in components
        if _pcie_threshold(
            finding, service.thresholds.pcie_replay_rate_warning_per_minute
        )
    ]
    if not pcie_components:
        return []
    # Replays must use the original persisted evidence, not a changed request
    # body with the same event id and a newer timestamp or different device.
    stored: XidEvent = service.store.get_xid_event(event.event_id)
    if (
        not is_pcie_kernel_xid(stored)
        or stored.cluster_id != event.cluster_id
        or stored.node_id != event.node_id
    ):
        return []
    latest: list[GpuMetricLatest] = service.store.list_gpu_metrics_latest(
        stored.cluster_id, stored.node_id
    )
    metrics_by_gpu: dict[str, dict[str, GpuMetricLatest]] = {}
    for item in latest:
        device_key = (
            item.sample.gpu_uuid or item.sample.pci_bdf or item.sample.gpu_index
        )
        if device_key:
            metrics_by_gpu.setdefault(device_key, {})[item.sample.canonical_name] = item
    findings = []
    for pcie in pcie_components:
        gpu_key = pcie.gpu_uuid or pcie.pci_bdf
        if not gpu_key:
            continue
        key = (stored.cluster_id, stored.node_id, gpu_key, f"composite:{PCIE_XID_RULE}")
        previous: GpuFindingState | None = service.store.get_gpu_finding_state(key)
        candidate = pcie_xid_link_failure(
            service,
            stored,
            gpu_key,
            {"pcie_replay_total": pcie},
            metrics_by_gpu.get(gpu_key, {}),
            previous,
            [stored],
        )
        if candidate is None:
            continue
        activated = service.store.update_gpu_findings(
            [(key, candidate, candidate.observed_at)]
        )[0]
        if activated:
            findings.append(candidate)
        elif (
            previous is not None
            and previous.finding is not None
            and previous.finding.finding_id == candidate.finding_id
        ):
            # Nontransactional stores may have persisted activation before an
            # incident write failed. Retry the same idempotent ingestion.
            findings.append(previous.finding)
    return findings
