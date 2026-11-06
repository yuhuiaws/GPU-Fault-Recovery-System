from __future__ import annotations

import logging
import re
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
from typing import Protocol

from gpu_fault.app.ingest.gpu_findings import gpu_node_health_findings
from gpu_fault.app.ingest.node_health import NodeHealthIngestionService
from gpu_fault.gpu_composites import GPU_COMPOSITE_TRANSACTION, is_pcie_kernel_xid
from gpu_fault.gpu_metrics import GpuInventorySnapshot, GpuMetricLatest
from gpu_fault.host_health import (
    NodeHealthCategory,
    NodeHealthFinding,
    NodeHealthIngestionResult,
)
from gpu_fault.markers import retire_markers_for_incident
from gpu_fault.models import (
    FaultIncident,
    IncidentState,
    NodeMarker,
    RecoveryAction,
    Severity,
    WorkloadState,
)
from gpu_fault.nvidia_logs import (
    UNCLASSIFIED_SXID_REASON,
    UNPARSED_SXID_REASON,
    UNPARSED_XID_REASON,
    UNSCHEDULABLE_WITHOUT_CODE_REASON,
    FaultSignal,
    NormalizedFaultBatch,
    unresolved_reason_kind,
)
from gpu_fault.orchestration.provider_correlation import bind_marker_incident
from gpu_fault.pci_identity import pci_bdf_matches
from gpu_fault.policy import (
    ActionDisposition,
    FaultPolicyDecision,
    SxidEvent,
    XidEvent,
)
from gpu_fault.policy.xid_cross_source import (
    KERNEL_XID_SOURCE,
    candidate_window,
    join_gauge_twin,
    paired_reports,
)
from gpu_fault.processor_diagnostics import report_processor_replay_phase
from gpu_fault.store import NotFoundError
from gpu_fault.store.shared.health_signals import finding_health_signal_key
from gpu_fault.telemetry import WorkloadContext

LOGGER = logging.getLogger(__name__)
#: Who retires the marker of a report that joined an already recovered incident.
JOINED_RECOVERED_INCIDENT_ACTOR = "xid-ingest:joined-recovered-incident"
#: Who retires the marker of an incident ingest recorded RECOVERED (b238).
RECORDED_INCIDENT_ACTOR = "fault-ingest:recorded-incident"
#: NVIDIA Fabric Manager User Guide, "Runtime NVSwitch and GPU Errors" >
#: "NVSwitch SXid Errors" > "Fatal SXid Errors" (page "Last updated on Oct 07,
#: 2026"), verbatim: the scope of a fatal SXID on an access link, the NVSwitch
#: port of one GPU (``_enrich_sxid_scope``).
NVIDIA_FM_FATAL_SXID_REFERENCE = (
    "https://docs.nvidia.com/datacenter/tesla/fabric-manager-user-guide/"
    "index.html#fatal-sxid-errors"
)
NVIDIA_FM_FATAL_ACCESS_SXID_QUOTE = (
    "When a fatal SXid error is reported on a NVSwitch port that connects a GPU "
    "and an NVSwitch, the corresponding error will be propagated to the GPU. "
    "The CUDA jobs that are running on that GPU will be aborted"
)
NVIDIA_FM_FATAL_ACCESS_SXID_RECOVERY_QUOTE = (
    "Reset the specified GPU and all the participating GPUs in the affected workload"
)

#: Every kind ``unresolved_reason_kind`` can return. A provider signal that
#: carries no unresolved reason clears the episode of each one, so a kind that
#: is missing here would open a WARNING finding that never closes.
UNRESOLVED_SIGNAL_KINDS = (
    UNPARSED_XID_REASON,
    UNPARSED_SXID_REASON,
    UNCLASSIFIED_SXID_REASON,
    UNSCHEDULABLE_WITHOUT_CODE_REASON,
)


class GpuInventoryReads(Protocol):
    """The store reads a node-wide SXID's GPU inventory needs."""

    def get_gpu_inventory_snapshot(
        self, cluster_id: str, node_id: str
    ) -> GpuInventorySnapshot | None: ...

    def list_gpu_metrics_latest(
        self, cluster_id: str, node_id: str
    ) -> list[GpuMetricLatest]: ...


def complete_node_gpu_inventory(
    store: GpuInventoryReads,
    event: SxidEvent,
    snapshot: GpuInventorySnapshot | None,
    *,
    legacy_max_age: timedelta,
) -> list[str]:
    """The node's whole GPU set a node-wide SXID acts on, or ``[]``.

    A full GPU + NVSwitch reset acts on every GPU of the node: the Node Agent
    resets them all and refuses a list that is not exactly its own inventory,
    and the plan's client check and validations cover only the GPUs it names.
    The set is complete when it is exactly the node's expected GPU count, which
    its inventory snapshots declare (``expected_gpu_count``: the count the
    installer configures, else the instance type's):

    * the fresh snapshot (``snapshot``) is one nvidia-smi enumeration and
      stands for the node unless it lists fewer GPUs than it declares;
    * without one, the GPUs of the node's latest per-GPU rows around the event
      stand for it only when they are exactly the count the node's latest
      snapshot declares. Those rows are no delivery: each is the last value of
      one metric of one GPU, so for an event older than the newest deliveries
      the window keeps exactly the rows that stopped refreshing. A verdict the
      collector publishes only under load (``pcie_link_downtrain``) stays at
      its last delivery after a collector restart, and that one row of GPU 0
      made a back-dated fatal SXID an executable one-GPU full reset on an
      8-GPU node (batch 244).

    Anything else is ``[]``, and the policy blocks the reset for "complete node
    GPU inventory": evidence, cordon, quarantine and an operator.
    """

    if snapshot is not None:
        gpus = {device.gpu_uuid for device in snapshot.devices}
        expected = snapshot.expected_gpu_count
        source = f"GPU inventory {snapshot.snapshot_id}"
        if expected is None or len(gpus) == expected:
            return sorted(gpus)
    else:
        declared = store.get_gpu_inventory_snapshot(event.cluster_id, event.node_id)
        expected = declared.expected_gpu_count if declared is not None else None
        gpus = set()
        for latest in store.list_gpu_metrics_latest(event.cluster_id, event.node_id):
            age = event.observed_at - latest.observed_at
            if latest.sample.gpu_uuid and -timedelta(seconds=30) <= age <= (
                legacy_max_age
            ):
                gpus.add(latest.sample.gpu_uuid)
        source = "the latest per-GPU telemetry around the event"
        if expected is not None and len(gpus) == expected:
            return sorted(gpus)
    LOGGER.warning(
        "SXID %s on %s/%s: %s names %d GPU(s) and the node declares %s; that is "
        "not its complete GPU inventory, so the event names no participating "
        "GPUs (a full GPU/NVSwitch reset blocks for operator review)",
        event.sxid,
        event.cluster_id,
        event.node_id,
        source,
        len(gpus),
        expected if expected is not None else "no GPU count",
    )
    return []


class FaultIngestionService:
    def __init__(self, context) -> None:
        self.context = context
        context.xid_correlation.finalizer = self.finalize_xid
        # Per-kind count of provider signals whose fault line could not be
        # resolved into an XID/SXID event. Exposed for the metrics snapshot;
        # the finding below is the per-episode signal, this is the per-line
        # one, so a drift that hits every line still shows its full rate.
        self.unresolved_signal_totals: dict[str, int] = {}

    def ingest_unresolved_signals(
        self,
        normalized: NormalizedFaultBatch,
        *,
        batch_id: str,
    ) -> NodeHealthIngestionResult | None:
        """Turn an unparsable fault line into something an operator sees.

        The collectors forward every ``NVRM ... Xid`` and ``SXid`` line; when
        the normalizer cannot extract a code (or an SXID's classification
        word) the line used to land as evidence with ``status=success`` and
        the reason in a response body nobody reads. A catalog or format drift
        could therefore swallow real XIDs silently. Now every such line is a
        WARNING and a counter, and the first one per node per episode opens a
        WARNING node-health finding with ``COLLECT_EVIDENCE`` -- it freezes
        evidence and routes to operator review without inventing a code and
        without cordoning the node. A later line from the same node that does
        parse ends the episode, so the next drift is reported again.
        """

        clears: list[tuple[str, bool, datetime, float]] = []
        candidates: list[tuple[tuple[str, bool, datetime, float], NodeHealthFinding]]
        candidates = []
        for signal in normalized.provider_signals:
            if not signal.unresolved_reasons:
                clears.extend(
                    (
                        f"{signal.cluster_id}/{signal.node_id}/{kind}/node",
                        False,
                        signal.observed_at,
                        0.0,
                    )
                    for kind in UNRESOLVED_SIGNAL_KINDS
                )
                continue
            by_kind: dict[str, list[str]] = {}
            for reason in signal.unresolved_reasons:
                by_kind.setdefault(unresolved_reason_kind(reason), []).append(reason)
            for kind, reasons in sorted(by_kind.items()):
                self.unresolved_signal_totals[kind] = (
                    self.unresolved_signal_totals.get(kind, 0) + 1
                )
                LOGGER.warning(
                    "fault line could not be resolved; no code was invented "
                    "kind=%s cluster=%s node=%s source=%s signal=%s reasons=%s",
                    kind,
                    signal.cluster_id,
                    signal.node_id,
                    signal.source.value,
                    signal.signal_id,
                    reasons,
                )
                finding = self._unresolved_signal_finding(
                    signal, kind=kind, reasons=reasons, batch_id=batch_id
                )
                candidates.append(
                    (
                        (
                            finding_health_signal_key(finding),
                            True,
                            signal.observed_at,
                            0.0,
                        ),
                        finding,
                    )
                )
        # The receive clock orders the episode: two kernel records can carry
        # the same second, and a sample that is not newer on the state
        # machine's clock is dropped as a replay.
        received_at = datetime.now(timezone.utc)
        if clears:
            self.context.store.claim_health_signal_transitions(
                clears, received_at=received_at
            )
        if not candidates:
            return None
        emitted = self.context.store.claim_health_signal_transitions(
            [transition for transition, _ in candidates],
            received_at=received_at,
        )
        findings = [
            finding
            for (_, finding), emit in zip(candidates, emitted, strict=True)
            if emit
        ]
        if not findings:
            return None
        result = NodeHealthIngestionService(self.context).ingest(batch_id, findings)
        notified_at = datetime.now(timezone.utc)
        for finding in findings:
            self.context.store.mark_health_signal_notified(
                finding_health_signal_key(finding), notified_at=notified_at
            )
        return result

    def _unresolved_signal_finding(
        self,
        signal: FaultSignal,
        *,
        kind: str,
        reasons: list[str],
        batch_id: str,
    ) -> NodeHealthFinding:
        event_id = f"{signal.signal_id}-{kind}"
        raw_message = (signal.raw_message or "")[:500] or None
        return NodeHealthFinding(
            finding_id=f"finding-{event_id}",
            event_id=event_id,
            cluster_id=signal.cluster_id,
            node_id=signal.node_id,
            observed_at=signal.observed_at,
            runtime_profile_version=self._unresolved_signal_profile_version(signal),
            category=NodeHealthCategory.GPU,
            severity=Severity.WARNING,
            reason=(
                "a fault line from the node could not be resolved into an "
                "XID/SXID event; operator review is required before the "
                "signal is trusted or dismissed"
            ),
            recommended_action=RecoveryAction.COLLECT_EVIDENCE,
            metric_name=kind,
            value=1.0,
            raw_message=raw_message,
            # Preserve the historical evidence URI namespace across the rename.
            evidence_ref=f"hma://{signal.source.value.lower()}/{signal.signal_id}",
            diagnostic_parameters={
                "unresolved_reasons": list(reasons),
                "signal_source": signal.source.value,
                "record_batch_id": batch_id,
            },
            policy_reference=(
                "unresolved fault lines fail closed into operator review"
            ),
        )

    def _unresolved_signal_profile_version(self, signal: FaultSignal) -> str | None:
        """The runtime profile the unresolved line's finding executes under.

        The health family refuses to compile without one -- even the
        FREEZE_EVIDENCE-only workflow this finding asks for needs the profile
        to name the evidence-capture owner -- so a finding built without it
        blocked as NEEDS_OPERATOR with no steps, no evidence frozen, and the
        operator-review notification pointing at an empty workflow. The
        source record's own version comes first (the kernel collector stamps
        it from its environment); a record without one falls back to the
        workload topology, the same source the XID path uses, and finally to
        the node agent's registration.
        """

        if signal.runtime_profile_version:
            return signal.runtime_profile_version
        context: WorkloadContext = self.context.topology.resolve(
            signal.cluster_id, signal.node_id, signal.observed_at
        )
        if context.runtime_profile_version:
            return context.runtime_profile_version
        try:
            agent = self.context.store.get_agent(signal.cluster_id, signal.node_id)
        except NotFoundError:
            return None
        version = agent.runtime_profile_version
        return str(version) if version else None

    def _enrich_fault_identity(
        self,
        event: XidEvent | SxidEvent,
    ) -> XidEvent | SxidEvent:
        if event.job_id and event.attempt_id:
            return event
        context = self.context.topology.resolve(
            event.cluster_id,
            event.node_id,
            event.observed_at,
            target_gpu_uuids=(
                {event.gpu_uuid}
                if isinstance(event, XidEvent) and event.gpu_uuid
                else set(event.participating_gpu_uuids)
                if isinstance(event, SxidEvent)
                else None
            ),
            pod_uid=event.pod_uid,
            container_id=event.container_id,
            host_pid=event.host_pid,
            cgroup_path=event.cgroup_path,
        )
        identities = set(context.job_attempt_ids)
        if len(identities) != 1:
            source = (
                "NO_ACTIVE_MANAGED_ATTEMPT"
                if not identities
                else "AMBIGUOUS_ACTIVE_ATTEMPTS"
            )
            return event.model_copy(update={"workload_identity_source": source})
        job_id, attempt_id = next(iter(identities))
        updates = {
            "job_id": event.job_id or job_id,
            "attempt_id": event.attempt_id or attempt_id,
            "workload_identity_source": (
                event.workload_identity_source or "SOLE_ACTIVE_ATTEMPT_ON_NODE"
            ),
        }
        if not event.affected_workload_ids:
            updates["affected_workload_ids"] = context.workload_ids
        if event.workload_state is WorkloadState.UNKNOWN:
            updates["workload_state"] = WorkloadState(context.workload_state)
        if (
            event.runtime_profile_version is None
            and context.runtime_profile_version is not None
        ):
            updates["runtime_profile_version"] = context.runtime_profile_version
        return event.model_copy(update=updates)

    def _enrich_gpu_identity(
        self,
        event: XidEvent | SxidEvent,
    ) -> XidEvent | SxidEvent:
        """Fill a product-less fault from the node's own fresh GPU inventory.

        The collectors stamp the GPU product they discovered at start-up. One
        that started before the driver answered -- every reboot, until the
        units were ordered after ``nvidia-persistenced`` -- stamps none, and the
        policy's product gate then reads the XID as "not applicable to product
        UNKNOWN" and quarantines the node instead of resetting the GPU (live,
        ``eks-managed``: one of two same-window XID 46 became a NOT_APPLICABLE
        quarantine record while its sibling started the reset). The node's GPU
        inventory is the same hardware's identity, read by nvidia-smi on the
        node, so it stands in -- only when it is fresh (``gpu_inventory_max_age``),
        from the event's boot and names exactly one product. Anything less
        leaves the event as it came and the gate fails closed as before; either
        way ``gpu_identity_source`` records what happened.
        """

        if event.product is not None:
            return event
        snapshot = self._fresh_gpu_inventory(
            cluster_id=event.cluster_id,
            node_id=event.node_id,
            event_time=event.observed_at,
            source_boot_id=event.source_boot_id,
        )
        products = (
            {device.product for device in snapshot.devices}
            if snapshot is not None
            else set()
        )
        code = (
            f"XID {event.xid}" if isinstance(event, XidEvent) else f"SXID {event.sxid}"
        )
        if snapshot is None or len(products) != 1 or None in products:
            LOGGER.warning(
                "%s on %s/%s carries no GPU product and the node has no fresh "
                "single-product GPU inventory from its boot to stand in (%s); "
                "the product gate fails closed",
                code,
                event.cluster_id,
                event.node_id,
                (
                    "no fresh inventory"
                    if snapshot is None
                    else f"products={sorted(item or 'UNKNOWN' for item in products)}"
                ),
            )
            return event.model_copy(
                update={"gpu_identity_source": "NO_FRESH_GPU_INVENTORY"}
            )
        product = products.pop()
        LOGGER.warning(
            "%s on %s/%s carries no GPU product; using %s from GPU inventory %s "
            "(observed %s, boot %s). The node's collector started before the "
            "driver answered; restart it or wait for its re-discovery",
            code,
            event.cluster_id,
            event.node_id,
            product,
            snapshot.snapshot_id,
            snapshot.observed_at.isoformat(),
            snapshot.source_boot_id,
        )
        return event.model_copy(
            update={
                "product": product,
                "gpu_identity_source": f"GPU_INVENTORY:{snapshot.snapshot_id}",
            }
        )

    def _fresh_gpu_inventory(
        self,
        *,
        cluster_id: str,
        node_id: str,
        event_time: datetime,
        source_boot_id: str | None,
    ) -> GpuInventorySnapshot | None:
        snapshot = self.context.store.get_gpu_inventory_snapshot(cluster_id, node_id)
        if snapshot is None:
            return None
        age = event_time - snapshot.observed_at
        if age < -timedelta(seconds=30) or age > self.context.gpu_inventory_max_age:
            return None
        if (
            source_boot_id
            and snapshot.source_boot_id
            and source_boot_id != snapshot.source_boot_id
        ):
            return None
        return snapshot

    def _pci_slot(
        self,
        value: str,
    ) -> tuple[str, str, str | None] | None:
        match = re.fullmatch(
            r"(?:(?:[0-9a-f]{4}|[0-9a-f]{8}):)?"
            r"([0-9a-f]{2}):([0-9a-f]{2})(?:\.([0-7]))?",
            value.strip().lower(),
        )
        return match.groups() if match is not None else None

    def _enrich_sxid_scope(self, event: SxidEvent) -> SxidEvent:
        event = self.context.nvswitch_topology.resolve(event)
        catalog_rule = self.context.policy.sxid_catalog_rule(event.sxid)
        context = self.context.topology.resolve(
            event.cluster_id, event.node_id, event.observed_at
        )
        updates = {}
        inventory_snapshot = self._fresh_gpu_inventory(
            cluster_id=event.cluster_id,
            node_id=event.node_id,
            event_time=event.observed_at,
            source_boot_id=event.source_boot_id,
        )
        if (
            event.classification.value == "FATAL"
            and event.link_scope.value == "ACCESS"
            and not (
                catalog_rule is not None
                and catalog_rule.official_action == "RESET_ALL_GPUS_AND_NVSWITCHES"
            )
        ):
            # NVIDIA_FM_FATAL_ACCESS_SXID_RECOVERY_QUOTE: the specified GPU --
            # the one the trusted port topology names, the one in the
            # NVSwitch's PCI slot -- and every GPU of the node's active managed
            # jobs, which the same workflow stops (without a named GPU the
            # affected job cannot be told from the others). Merged: the port's
            # GPU alone left the job's other GPUs unreset and uncontained.
            participants = set(event.participating_gpu_uuids) | set(context.gpu_uuids)
            if event.pci_bdf and inventory_snapshot is not None:
                event_slot = self._pci_slot(event.pci_bdf)
                for device in inventory_snapshot.devices:
                    sample_slot = self._pci_slot(device.pci_bdf)
                    if (
                        event_slot is not None
                        and sample_slot is not None
                        and sample_slot[:2] == event_slot[:2]
                        and (event_slot[2] is None or sample_slot[2] == event_slot[2])
                    ):
                        participants.add(device.gpu_uuid)
            elif event.pci_bdf:
                event_slot = self._pci_slot(event.pci_bdf)
                for latest in self.context.store.list_gpu_metrics_latest(
                    event.cluster_id, event.node_id
                ):
                    sample_slot = (
                        self._pci_slot(latest.sample.pci_bdf)
                        if latest.sample.pci_bdf
                        else None
                    )
                    age = event.observed_at - latest.observed_at
                    if (
                        event_slot is not None
                        and sample_slot is not None
                        and latest.sample.gpu_uuid
                        and -timedelta(seconds=30)
                        <= age
                        <= self.context.legacy_gpu_metrics_inventory_max_age
                        and sample_slot[:2] == event_slot[:2]
                        and (event_slot[2] is None or sample_slot[2] == event_slot[2])
                    ):
                        participants.add(latest.sample.gpu_uuid)
            if participants:
                updates["participating_gpu_uuids"] = sorted(participants)
        elif (
            catalog_rule is not None
            and catalog_rule.official_action == "RESET_ALL_GPUS_AND_NVSWITCHES"
        ) or (
            event.classification.value == "ALWAYS_FATAL"
            or (
                event.classification.value == "FATAL"
                and event.link_scope.value != "ACCESS"
            )
        ):
            inventory = complete_node_gpu_inventory(
                self.context.store,
                event,
                inventory_snapshot,
                legacy_max_age=self.context.legacy_gpu_metrics_inventory_max_age,
            )
            if inventory:
                updates["participating_gpu_uuids"] = inventory
                updates["fabric_partition"] = (
                    event.fabric_partition
                    or f"{event.cluster_id}/{event.node_id}/local-nvswitch"
                )
            elif (
                event.participating_gpu_uuids
                and event.classification.value != "ALWAYS_FATAL"
            ):
                # The trusted port topology names the GPU on the SXID's port;
                # a node-wide reset needs every GPU of the node.
                updates["participating_gpu_uuids"] = []
        return event.model_copy(update=updates) if updates else event

    def _resolve_kernel_xid_gpu_uuid(self, event: XidEvent) -> XidEvent:
        if event.gpu_uuid or not event.pci_bdf:
            return event

        event_slot = self._pci_slot(event.pci_bdf)
        if event_slot is None:
            return event
        snapshot = self._fresh_gpu_inventory(
            cluster_id=event.cluster_id,
            node_id=event.node_id,
            event_time=event.observed_at,
            source_boot_id=event.source_boot_id,
        )
        if snapshot is not None:
            candidates = {
                device.gpu_uuid
                for device in snapshot.devices
                if (pci_bdf_matches(event.pci_bdf, device.pci_bdf))
            }
            if len(candidates) == 1:
                return event.model_copy(update={"gpu_uuid": candidates.pop()})
            return event
        candidates = set()
        for latest in self.context.store.list_gpu_metrics_latest(
            event.cluster_id, event.node_id
        ):
            sample = latest.sample
            sample_slot = self._pci_slot(sample.pci_bdf) if sample.pci_bdf else None
            age = event.observed_at - latest.observed_at
            if (
                sample_slot is None
                or not sample.gpu_uuid
                or age < -timedelta(seconds=30)
                or age > self.context.legacy_gpu_metrics_inventory_max_age
                or not pci_bdf_matches(event.pci_bdf, sample.pci_bdf)
            ):
                continue
            candidates.add(sample.gpu_uuid)
        if len(candidates) != 1:
            return event
        return event.model_copy(update={"gpu_uuid": candidates.pop()})

    def _record_event_marker(self, marker: NodeMarker, *, repeated: bool) -> None:
        """Write an event's marker, but never revive one its incident retired.

        ``marker_id`` derives from the event id, so a repeat of the same event
        (a client retry, an outbox replay past the queue's dedupe window)
        rewrites the same row from the stored decision. A marker the incident's
        recovery already retired stays retired: the repeat reports the same
        fault again, not a new one. Only a repeat pays the extra read.
        """

        if repeated and any(
            item.marker_id == marker.marker_id and item.retired_at is not None
            for item in self.context.store.list_markers_for_incident(marker.incident_id)
        ):
            return
        self.context.completion.add_marker(marker)

    def _join_gauge_twin(
        self, event: XidEvent, decision: FaultPolicyDecision
    ) -> FaultPolicyDecision:
        """A kernel line whose DCGM field-230 change was decided first, the same
        way, joins that change's incident (``policy/xid_cross_source``).

        Compared once the decision is final -- after the generation fence and
        the site's XID 45 policy -- so the two reports meet as the incident sees
        them; anything else keeps the line's own decision.
        """

        if event.event_source != KERNEL_XID_SOURCE or decision.correlated_event_id:
            return decision
        reach = candidate_window(event, timedelta(0))
        candidates = self.context.store.list_xid_events(
            event.cluster_id,
            event.node_id,
            observed_after=event.observed_at - reach,
            observed_before=event.observed_at + reach,
        )
        return join_gauge_twin(
            event,
            decision,
            candidates,
            paired_reports(
                event, candidates, self.context.store.get_xid_policy_decision
            ),
            self.context.store.get_xid_policy_decision,
        )

    def _retire_recovered_markers(
        self,
        event: XidEvent | SxidEvent,
        decision: FaultPolicyDecision,
        incident: FaultIncident,
    ) -> None:
        """A RECOVERED incident keeps no live marker (ARCH-I4).

        Two ingest answers end RECOVERED with no workflow that would retire the
        marker later: the other source's report of an XID whose incident had
        already recovered (``xid_cross_source``, batch 237), and an incident
        recorded at ingest -- a stale recovery the attempt restart answered
        (F-B8), a line from a boot the node has left (``cover_fault``). A live
        marker of the second kind kept matching the node's terminal events and,
        as a QUARANTINE, kept the node out of warm-spare reuse until its TTL
        (b238).
        """

        if incident.state is not IncidentState.RECOVERED or not decision.marker.active:
            return
        joined = incident.event_id != event.event_id
        retire_markers_for_incident(
            self.context.store,
            incident.incident_id,
            reason=(
                f"event {event.event_id} joined incident {incident.incident_id} "
                "after it recovered"
                if joined
                else f"event {event.event_id}: incident {incident.incident_id} "
                "was recorded RECOVERED, nothing planned"
            ),
            retired_by=(
                JOINED_RECOVERED_INCIDENT_ACTOR if joined else RECORDED_INCIDENT_ACTOR
            ),
        )

    def finalize_xid(
        self, event: XidEvent, decision: FaultPolicyDecision
    ) -> FaultPolicyDecision:
        # Before the companion merge below marks a new event as a duplicate of
        # its companion: only a repeat of this very event is a repeat here.
        repeated = decision.duplicate
        report_processor_replay_phase("xid_action_fence")
        decision = self.context.orchestrator.apply_fault_action_generation_fence(
            event, decision
        )
        if not repeated:
            decision = self._join_gauge_twin(event, decision)
        if decision.correlated_event_id:
            report_processor_replay_phase("xid_companion_lookup")
            companion_incident = self.context.store.get_incident_by_event(
                decision.correlated_event_id
            )
            if companion_incident is None:
                companion_event = self.context.store.get_xid_event(
                    decision.correlated_event_id
                )
                companion_decision = self.context.store.get_xid_policy_decision(
                    decision.correlated_event_id
                )
                if (
                    companion_decision is not None
                    and companion_decision.disposition
                    is not ActionDisposition.PENDING_CORRELATION
                ):
                    self.finalize_xid(companion_event, companion_decision)
                companion_incident = self.context.store.get_incident_by_event(
                    decision.correlated_event_id
                )
            if companion_incident is not None:
                decision = decision.model_copy(
                    update={
                        "duplicate": True,
                        "marker": decision.marker.model_copy(
                            update={"incident_id": (companion_incident.incident_id)}
                        ),
                    }
                )
        report_processor_replay_phase("xid_provider_correlation")
        decision = self.context.orchestrator.correlate_provider_event(
            decision,
            cluster_id=event.cluster_id,
        )
        report_processor_replay_phase("xid_incident_workflow")
        incident, workflow = self.context.orchestrator.ingest(event, decision)
        decision = bind_marker_incident(decision, incident.incident_id)
        report_processor_replay_phase("xid_marker")
        self._record_event_marker(decision.marker, repeated=repeated)
        self._retire_recovered_markers(event, decision, incident)
        report_processor_replay_phase("xid_notification")
        if decision.disposition is ActionDisposition.NOT_APPLICABLE:
            notification = self.context.advisory_notifications.preview_not_applicable(
                incident.incident_id, event, decision
            )
            self.context.advisory_notifications.send(notification.notification_id)
            investigatory_notification = notification
        else:
            notification = self.context.advisory_notifications.try_preview(
                incident.incident_id
            )
            investigatory_notification = (
                self.context.advisory_notifications.preview_xid_investigatory(
                    incident.incident_id, event, decision
                )
            )
            self.context.advisory_notifications.send(
                investigatory_notification.notification_id
            )
        finalized = decision.model_copy(
            update={
                "incident_id": incident.incident_id,
                "workflow_request_id": (workflow.request_id if workflow else None),
                "advisory_notification_id": (
                    notification.notification_id if notification else None
                ),
                "investigatory_notification_id": (
                    investigatory_notification.notification_id
                ),
            }
        )
        report_processor_replay_phase("xid_decision_persist")
        self.context.store.save_xid_policy_decision(finalized)
        return finalized

    def ingest_xid(self, event: XidEvent) -> FaultPolicyDecision:
        report_processor_replay_phase("xid_identity")
        # xid_154_action is derived server-side from the XID 154 log line
        # (see XidCorrelationCoordinator.prepare_xid154). Drop any inbound
        # value at the ingest boundary so a client can never seed or steer
        # the dynamic recovery action (security review H-10).
        if event.xid_154_action is not None:
            event = event.model_copy(update={"xid_154_action": None})
        event = self._enrich_gpu_identity(self._enrich_fault_identity(event))
        if event.ingested_at is None:
            event = event.model_copy(update={"ingested_at": datetime.now(timezone.utc)})
        transaction = (
            self.context.store.collector_ingestion_transaction(
                event.cluster_id, event.node_id, GPU_COMPOSITE_TRANSACTION
            )
            if is_pcie_kernel_xid(event)
            else nullcontext()
        )
        with transaction:
            report_processor_replay_phase("xid_correlation")
            decision = self.context.xid_correlation.ingest(event)
            if decision.disposition is ActionDisposition.PENDING_CORRELATION:
                return decision
            finalized = self.finalize_xid(event, decision)
            if is_pcie_kernel_xid(event):
                findings = self.context.gpu_metrics.correlate_kernel_xid(event)
                if findings:
                    NodeHealthIngestionService(self.context).ingest(
                        event.event_id,
                        gpu_node_health_findings(self.context, findings),
                    )
            return finalized

    def ingest_sxid(self, event: SxidEvent) -> FaultPolicyDecision:
        report_processor_replay_phase("sxid_identity")
        event = self._enrich_gpu_identity(self._enrich_fault_identity(event))
        if event.ingested_at is None:
            event = event.model_copy(update={"ingested_at": datetime.now(timezone.utc)})
        report_processor_replay_phase("sxid_policy")
        decision = self.context.policy.evaluate_sxid(event)
        report_processor_replay_phase("sxid_action_fence")
        decision = self.context.orchestrator.apply_fault_action_generation_fence(
            event, decision
        )
        report_processor_replay_phase("sxid_provider_correlation")
        decision = self.context.orchestrator.correlate_provider_event(
            decision,
            cluster_id=event.cluster_id,
        )
        report_processor_replay_phase("sxid_incident_workflow")
        incident, workflow = self.context.orchestrator.ingest(event, decision)
        decision = decision.model_copy(
            update={
                "marker": decision.marker.model_copy(
                    update={"incident_id": incident.incident_id}
                )
            }
        )
        report_processor_replay_phase("sxid_marker")
        # The SXID duplicate flag comes from a process-local cache, so any
        # replica may be serving a repeat: every SXID checks.
        self._record_event_marker(decision.marker, repeated=True)
        self._retire_recovered_markers(event, decision, incident)
        report_processor_replay_phase("sxid_notification")
        notification = self.context.advisory_notifications.try_preview(
            incident.incident_id
        )
        sxid_notification = self.context.advisory_notifications.preview_sxid_event(
            incident.incident_id, event, decision
        )
        self.context.advisory_notifications.send(sxid_notification.notification_id)
        finalized = decision.model_copy(
            update={
                "incident_id": incident.incident_id,
                "workflow_request_id": (workflow.request_id if workflow else None),
                "advisory_notification_id": (
                    notification.notification_id if notification else None
                ),
                "investigatory_notification_id": (sxid_notification.notification_id),
            }
        )
        report_processor_replay_phase("sxid_decision_persist")
        self.context.store.save_xid_policy_decision(finalized)
        return finalized
