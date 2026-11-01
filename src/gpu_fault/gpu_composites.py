"""Build GPU composites and join PCIe findings to persisted kernel XID evidence."""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Mapping, Sequence

from gpu_fault.gpu_metric_models import (
    GpuFindingState,
    GpuHealthFinding,
    GpuHealthSeverity,
)
from gpu_fault.gpu_pcie_link_policy import PCIE_CORRECTABLE_ERRORS, PCIE_LINK_DOWNTRAIN
from gpu_fault.gpu_power_policy import NVIDIA_DCGM_HEALTH_REFERENCE
from gpu_fault.models import RecoveryAction
from gpu_fault.pci_identity import pci_bdf_matches
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
SITE_CORRELATION_POLICY_VERSION = "site-dcgm-correlation/v3"
#: ``GpuMetricSource.DCGM_EXPORTER``: the source of the XID event
#: ``GpuMetricsService._xid_event`` makes of a change of the last-XID gauge.
DCGM_XID_GAUGE_SOURCE = "DCGM_EXPORTER"
_LINK_XIDS = frozenset({32, 79})
#: The PCIe evidence a same-GPU XID 32/79 joins into the CRITICAL DRAIN
#: composite, in the order one is picked: the replay rate above its threshold,
#: and from batch 233 a confirmed link downtrain or correctable-error rate
#: (``gpu_pcie_link_policy``; their evidence-only first windows do not join).
PCIE_LINK_COMPONENT_NAMES = (
    "pcie_replay_total",
    PCIE_LINK_DOWNTRAIN,
    PCIE_CORRECTABLE_ERRORS,
)
#: A batch carrying one of these evaluates its GPU's PCIe composite.
PCIE_COMPOSITE_TRIGGER_NAMES = frozenset({*PCIE_LINK_COMPONENT_NAMES, "xid_last_error"})
_PCIE_COMPONENT_TEXT = {
    "pcie_replay_total": "PCIe replay threshold",
    PCIE_LINK_DOWNTRAIN: "PCIe link downtrain",
    PCIE_CORRECTABLE_ERRORS: "PCIe correctable-error rate",
}


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
        official_action=next(
            (
                item.official_action
                for item in components
                if item.automatic_action == action.value and item.official_action
            ),
            None,
        ),
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


def episode_composite_state(
    state: GpuFindingState | None, observed_at: datetime, window: timedelta
) -> GpuFindingState | None:
    """``state`` if a batch observed at ``observed_at`` continues its episode.

    An active composite scope is re-evaluated by every batch that touches its
    GPU, and once a component stops breaching the composite clears within one
    correlation window (``_evaluate_composite_findings`` keeps components no
    older than the window). An active state the window does not reach was
    therefore left by a GPU that stopped reporting: it is the previous
    episode's. It neither counts consecutive samples for nor lends its
    identity to a breach observed now -- the device-signal rule (a vanished
    device's signal must not swallow the next episode), applied to composites;
    the caller still clears or replaces the stored state. A cleared state is
    returned as it is: the PCIe join reads its clear time.
    """

    if (
        state is None
        or state.finding is None
        or observed_at - state.observed_at <= window
    ):
        return state
    return None


def composite_episode_identity(
    candidate: GpuHealthFinding, episode: GpuFindingState | None
) -> GpuHealthFinding:
    """``candidate`` carrying the finding id of the episode it continues.

    A composite's finding id is minted with the batch that opens its episode
    (:func:`composite_finding`) and kept for as long as the composite stays
    active with the same automatic action. The id is what the incident, the
    marker and the workflow are derived from (``app/ingest/gpu_findings``:
    event id ``gpu-<finding id>``), so a re-grade of the open episode --
    CORRECTABLE_MEMORY_DEGRADATION turning CRITICAL at
    ``correctable_memory_drain_consecutive_samples`` -- updates the incident it
    already has instead of minting a second one (GF-GENERAL-LIVE-024
    ``independent_xid``, live 2026-10-26: two incidents 30 s apart for one
    held pair). A changed automatic action is a new recovery decision and keeps
    the candidate's own id: a DRAIN plan has to supersede the diagnostic, which
    only a new incident does. So does a breach after the episode cleared or
    after its GPU stopped reporting (``episode`` is ``None`` or cleared,
    :func:`episode_composite_state`). The per-batch component finding ids are
    left alone: they are the batch's evidence, not the composite's identity.
    """

    if (
        episode is None
        or episode.finding is None
        or episode.finding.automatic_action != candidate.automatic_action
        or episode.finding.finding_id == candidate.finding_id
    ):
        return candidate
    return candidate.model_copy(update={"finding_id": episode.finding.finding_id})


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
    boots = {
        value
        for labels in (sample.labels, previous.sample.labels)
        for name in ("source_boot_id", "boot_id")
        if (value := labels.get(name))
    }
    if (
        len(boots) > 1
        or min(sample.value, previous.sample.value) < 0
        or not math.isfinite(previous.sample.value)
        or sample.value < previous.sample.value
        or sample.gpu_uuid != previous.sample.gpu_uuid
        or sample.pci_bdf != previous.sample.pci_bdf
    ):
        return None
    return sample.value - previous.sample.value


def is_pcie_kernel_xid(event: XidEvent) -> bool:
    return (
        event.event_source == "KERNEL_LOG"
        and event.xid in _LINK_XIDS
        and not event.synthetic
    )


def is_dcgm_link_xid(event: XidEvent) -> bool:
    """A change of the DCGM last-XID gauge to XID 32/79."""

    return (
        event.event_source == DCGM_XID_GAUGE_SOURCE
        and event.xid in _LINK_XIDS
        and not event.synthetic
    )


def _pcie_threshold(finding: GpuHealthFinding, threshold: float) -> bool:
    """Whether ``finding`` is PCIe evidence a link XID joins: the replay rate
    above ``threshold``, or a confirmed downtrain or correctable-error rate."""

    if finding.finding_kind != "METRIC":
        return False
    if finding.canonical_name in {PCIE_LINK_DOWNTRAIN, PCIE_CORRECTABLE_ERRORS}:
        return finding.automatic_action != RecoveryAction.NO_ACTION.value
    return (
        finding.canonical_name == "pcie_replay_total"
        and finding.rate_per_minute is not None
        and math.isfinite(finding.rate_per_minute)
        and finding.rate_per_minute > threshold
    )


def link_xid_events(
    service: GpuMetricsService,
    batch: GpuMetricBatch,
    components: Sequence[GpuHealthFinding],
    batch_gauge_changes: Sequence[XidEvent] = (),
) -> list[XidEvent]:
    """The kernel XID 32/79 events and the last-XID gauge changes a PCIe
    breach of ``components`` may join.

    One bounded same-node query serves both, and only when a touched GPU
    breaches the PCIe rate. ``batch_gauge_changes`` are the gauge changes this
    batch itself made: the API persists them only after this evaluation.
    """

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

    def joinable(event: XidEvent) -> bool:
        return any(_same_device(event, finding) for finding in pcie)

    kernel = [
        event for event in events if is_pcie_kernel_xid(event) and joinable(event)
    ]
    gauge_changes = {
        event.event_id: event
        for event in (*events, *batch_gauge_changes)
        if is_dcgm_link_xid(event) and joinable(event)
    }
    return [*kernel, *gauge_changes.values()]


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


def _gauge_change(
    trigger: GpuMetricBatch,
    pcie: GpuHealthFinding,
    gauge: GpuMetricLatest,
    previous: GpuFindingState | None,
    xid_events: Sequence[XidEvent],
    window: timedelta,
) -> XidEvent | None:
    """The change of the last-XID gauge to its current value, inside the window.

    The gauge holds the last XID for as long as the driver stays loaded and
    every scrape re-reads it, so its latest sample is always as fresh as the
    batch: bounding *that* by the window bounded nothing, and a gauge that went
    to 32/79 long before re-armed the CRITICAL DRAIN composite on every later
    PCIe breach (GF-GENERAL-LIVE-022 ``expired_xid``). What dates the XID is
    the event its change produced (``GpuMetricsService._xid_event``, claimed
    once across replicas), held to the kernel path's rules: same node and GPU,
    inside the window of both the trigger and the PCIe finding, and newer than
    the composite's last clear. A value first seen as a baseline produced no
    event, so its age is unknown and it joins nothing.
    """

    xid = int(gauge.sample.value)
    for event in sorted(
        xid_events, key=lambda item: (item.observed_at, item.event_id), reverse=True
    ):
        if (
            is_dcgm_link_xid(event)
            and event.xid == xid
            and event.cluster_id == trigger.cluster_id
            and event.node_id == trigger.node_id
            and abs(event.observed_at - trigger.observed_at) <= window
            and abs(event.observed_at - pcie.observed_at) <= window
            and _same_device(event, pcie)
            and not (
                previous is not None
                and previous.finding is None
                and event.observed_at <= previous.observed_at
            )
        ):
            return event
    return None


def pcie_xid_link_failure(
    service: GpuMetricsService,
    trigger: GpuMetricBatch | XidEvent,
    gpu_key: str,
    components: dict[str, GpuHealthFinding],
    metrics: dict[str, GpuMetricLatest],
    previous: GpuFindingState | None,
    xid_events: Sequence[XidEvent],
) -> GpuHealthFinding | None:
    """The CRITICAL DRAIN composite of a PCIe breach and a same-GPU XID 32/79.

    ``xid_events`` are the window's candidates (:func:`link_xid_events`): a
    trusted kernel event joins by its policy decision; failing one, the
    last-XID gauge joins only through its change event (:func:`_gauge_change`).
    """
    threshold = service.thresholds.pcie_replay_rate_warning_per_minute
    pcie = next(
        (
            components[name]
            for name in PCIE_LINK_COMPONENT_NAMES
            if name in components and _pcie_threshold(components[name], threshold)
        ),
        None,
    )
    metric = metrics.get(pcie.canonical_name) if pcie is not None else None
    window = timedelta(seconds=service.thresholds.correlation_window_seconds)
    if (
        pcie is None
        or metric is None
        or pcie.cluster_id != trigger.cluster_id
        or pcie.node_id != trigger.node_id
        or metric.cluster_id != trigger.cluster_id
        or metric.node_id != trigger.node_id
        or metric.observed_at != pcie.observed_at
        or abs(pcie.observed_at - trigger.observed_at) > window
    ):
        return None
    kernel = None
    for event in sorted(
        xid_events, key=lambda item: (item.observed_at, item.event_id), reverse=True
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
    joined = kernel
    if joined is None:
        if (
            isinstance(trigger, XidEvent)
            or gauge is None
            or gauge.sample.value not in _LINK_XIDS
            or abs(gauge.observed_at - trigger.observed_at) > window
            or abs(gauge.observed_at - pcie.observed_at) > window
        ):
            return None
        joined = _gauge_change(trigger, pcie, gauge, previous, xid_events, window)
        if joined is None:
            return None
    finding = composite_finding(
        trigger,
        scope_key=gpu_key,
        rule_id=PCIE_XID_RULE,
        components=[pcie],
        component_metrics=[
            pcie.canonical_name,
            "kernel_xid" if kernel is not None else "xid_last_error",
        ],
        severity=GpuHealthSeverity.CRITICAL,
        action=RecoveryAction.DRAIN,
        reason=(
            f"{_PCIE_COMPONENT_TEXT[pcie.canonical_name]} was correlated with XID "
            f"{joined.xid}"
            + (
                f" from kernel event {joined.event_id}"
                if kernel is not None
                else f" from the DCGM last-XID gauge change {joined.event_id}"
            )
        ),
        confidence="HIGH",
    )
    if kernel is None:
        # The gauge change is this or an earlier batch: the trigger stays the
        # composite's time; the change is named as its evidence.
        return finding.model_copy(
            update={
                "component_evidence_refs": list(
                    dict.fromkeys(
                        [
                            *finding.component_evidence_refs,
                            f"xid-event://{joined.event_id}",
                        ]
                    )
                ),
            }
        )
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


def resolve_composite_updates(
    service: GpuMetricsService,
    batch: GpuMetricBatch,
    *,
    scopes: Sequence[str],
    rule_ids: Sequence[str],
    pcie_gpu_keys: set[str],
    candidates: Mapping[tuple[str, str], GpuHealthFinding],
    previous_states: Mapping[tuple[str, str, str, str], GpuFindingState | None],
    episode_states: Mapping[tuple[str, str, str, str], GpuFindingState | None],
    fresh_components: set[str],
    latest_by_gpu: Mapping[str, dict[str, GpuMetricLatest]],
    findings_by_gpu: Mapping[str, dict[str, GpuHealthFinding]],
) -> tuple[
    list[GpuHealthFinding],
    list[GpuHealthFinding],
    list[GpuHealthFinding],
    list[str],
]:
    """Write the batch's composite candidates over the stored states and sort
    the outcome: ``(composite findings, activations, re-grades, suppressed ids)``.

    ``previous_states`` are the stored states (what is cleared or replaced);
    ``episode_states`` the ones this batch continues
    (:func:`episode_composite_state`), which lend the candidates their
    identity (:func:`composite_episode_identity`). An activation is a
    composite whose episode starts with this batch -- it gets an incident of
    its own; a re-grade is an active composite whose severity changed under
    the episode's id -- the store reports the grade change as an activation,
    the unchanged id tells it apart, and the ingest updates the incident the
    id already has.
    """

    composite_findings: list[GpuHealthFinding] = []
    new_composite_findings: list[GpuHealthFinding] = []
    updated_composite_findings: list[GpuHealthFinding] = []
    suppressed_ids: set[str] = set()
    updates: list[
        tuple[tuple[str, str, str, str], GpuHealthFinding | None, datetime]
    ] = []
    candidate_updates: list[tuple[tuple[str, str, str, str], GpuHealthFinding]] = []
    for scope_key in scopes:
        for rule_id in rule_ids:
            # Only PCIe/XID inputs share the cross-replica correlation lock.
            # An unrelated scrape is not evidence that a PCIe episode cleared.
            if rule_id == PCIE_XID_RULE and scope_key not in pcie_gpu_keys:
                continue
            key = (batch.cluster_id, batch.node_id, scope_key, f"composite:{rule_id}")
            candidate = candidates.get((scope_key, rule_id))
            previous = previous_states.get(key)
            if candidate is not None:
                # One finding id per episode: the incident is derived from it.
                candidate = composite_episode_identity(
                    candidate, episode_states.get(key)
                )
            if candidate is not None and previous is not None and previous.finding:
                if set(candidate.component_finding_ids) == set(
                    previous.finding.component_finding_ids
                ) or (
                    rule_id != PCIE_XID_RULE
                    and not set(candidate.component_finding_ids) <= fresh_components
                ):
                    composite_findings.append(previous.finding)
                    suppressed_ids.update(previous.finding.component_finding_ids)
                    continue
            if candidate is None and (previous is None or previous.finding is None):
                continue
            observed_at = composite_update_time(
                batch,
                candidate,
                previous,
                latest_by_gpu.get(scope_key, {}),
                findings_by_gpu.get(scope_key, {}),
            )
            updates.append((key, candidate, observed_at))
            if candidate is not None:
                candidate_updates.append((key, candidate))
    activated_values = service.store.update_gpu_findings(updates)
    activated_by_key = {
        key: activated
        for (key, _candidate, _observed_at), activated in zip(
            updates, activated_values, strict=True
        )
    }
    for key, candidate in candidate_updates:
        composite_findings.append(candidate)
        suppressed_ids.update(candidate.component_finding_ids)
        if not activated_by_key.get(key, False):
            continue
        episode = episode_states.get(key)
        if (
            episode is not None
            and episode.finding is not None
            and episode.finding.finding_id == candidate.finding_id
        ):
            updated_composite_findings.append(candidate)
        else:
            new_composite_findings.append(candidate)
    return (
        composite_findings,
        new_composite_findings,
        updated_composite_findings,
        sorted(suppressed_ids),
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
        minted = pcie_xid_link_failure(
            service,
            stored,
            gpu_key,
            {pcie.canonical_name: pcie},
            metrics_by_gpu.get(gpu_key, {}),
            previous,
            [stored],
        )
        if minted is None:
            continue
        # A kernel event that re-confirms an open PCIe episode continues it
        # under the episode's finding id (``composite_episode_identity``); a
        # second XID 32/79 while the composite is active is not a second
        # incident. Only an episode this event opens keeps the event's own id.
        candidate = composite_episode_identity(
            minted,
            episode_composite_state(
                previous,
                minted.observed_at,
                timedelta(seconds=service.thresholds.correlation_window_seconds),
            ),
        )
        activated = service.store.update_gpu_findings(
            [(key, candidate, candidate.observed_at)]
        )[0]
        if activated:
            findings.append(candidate)
        elif (
            previous is not None
            and previous.finding is not None
            and previous.finding.finding_id == minted.finding_id
        ):
            # An exact replay of the event that opened the episode.
            # Nontransactional stores may have persisted activation before an
            # incident write failed. Retry the same idempotent ingestion.
            findings.append(previous.finding)
    return findings
