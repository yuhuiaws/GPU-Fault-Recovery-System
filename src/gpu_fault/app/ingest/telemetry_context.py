from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Literal

from gpu_fault.app.ingest.node_health import NodeHealthIngestionService
from gpu_fault.fleet import AgentLifecycleState
from gpu_fault.gpu_metrics import GpuInventorySnapshot
from gpu_fault.host_health import (
    NodeHealthCategory,
    NodeHealthFinding,
    NodeHealthIngestionResult,
)
from gpu_fault.models import RecoveryAction, Severity, WorkloadState
from gpu_fault.pci_identity import pci_bdf_matches
from gpu_fault.store.shared.errors import NotFoundError
from gpu_fault.store.shared.health_signals import finding_health_signal_key
from gpu_fault.telemetry import (
    CollectorKind,
    CollectorStatus,
    EvidenceKind,
)

LOGGER = logging.getLogger(__name__)

#: A GPU UUID left (or joined) the node's inventory between two snapshots, or
#: a GPU that stayed answers on another PCI address.
GPU_IDENTITY_CHANGED_METRIC = "gpu_inventory_identity_changed"
#: The snapshot carries no expected GPU count, so a loss cannot be judged
#: against the node's invariant; fail closed and say so, once per node.
GPU_EXPECTED_COUNT_UNKNOWN_METRIC = "gpu_expected_count_unknown"


def _pci_bdf_moved(before: str, after: str) -> bool:
    """Whether a GPU answers on another PCI address than it did.

    A 4- or 8-digit domain and the hex case are spellings of one address; a
    value that does not parse as an address on either side is unknown, and
    an unknown bus id is not evidence that the GPU moved.
    """

    if before.strip().lower() == after.strip().lower():
        return False
    # ``pci_bdf_matches`` answers False for an unparsable side, which would
    # read as "another address"; an address matches itself exactly when it
    # parses, so that is asked first. (The check stays here rather than as a
    # helper in ``gpu_composites``: that module is in the node runtime and
    # executor closures, and this rule is the control plane's alone.)
    if not (pci_bdf_matches(before, before) and pci_bdf_matches(after, after)):
        return False
    return not pci_bdf_matches(before, after)


class TelemetryContextService:
    def __init__(self, context) -> None:
        self.context = context

    def _inventory_identity(self, snapshot: GpuInventorySnapshot):
        return tuple(
            sorted(
                (
                    item.gpu_index,
                    item.gpu_uuid,
                    item.pci_bdf.lower(),
                    item.product,
                )
                for item in snapshot.devices
            )
        )

    def _enrich_workload_context(
        self, batch, observed_at: datetime, *, observations=None
    ):
        context = self.context.topology.resolve(
            batch.cluster_id,
            batch.node_id,
            observed_at,
            observations=observations,
        )
        updates = {}
        if batch.workload_state is WorkloadState.UNKNOWN:
            updates["workload_state"] = WorkloadState(context.workload_state)
        if not batch.affected_workload_ids:
            updates["affected_workload_ids"] = context.workload_ids
        if (
            batch.runtime_profile_version is None
            and context.runtime_profile_version is not None
        ):
            updates["runtime_profile_version"] = context.runtime_profile_version
        return batch.model_copy(update=updates) if updates else batch

    def _record_collector_status(
        self,
        collector: CollectorKind,
        cluster_id: str,
        node_id: str,
        observed_at: datetime,
        batch_id: str,
        sample_count: int,
        errors: list[str] | None = None,
    ) -> None:
        now = datetime.now(timezone.utc)
        collection_errors = list(errors or [])
        if collector in {CollectorKind.GPU_METRICS, CollectorKind.GPU_INVENTORY} and (
            sample_count == 0 and not collection_errors
        ):
            collection_errors.append("empty GPU observation cannot establish health")
        status = CollectorStatus(
            cluster_id=cluster_id,
            node_id=node_id,
            collector=collector,
            observed_at=observed_at,
            ingested_at=now,
            last_success_at=(observed_at if not collection_errors else None),
            last_error_at=(observed_at if collection_errors else None),
            batch_id=batch_id,
            sample_count=sample_count,
            errors=collection_errors,
        )
        self.context.store.save_collector_status(status)

    def _capture_evidence(
        self,
        *,
        record_id: str,
        cluster_id: str,
        node_id: str,
        kind: EvidenceKind,
        observed_at: datetime,
        payload: dict,
        observations=None,
    ) -> None:
        context = self.context.topology.resolve(
            cluster_id,
            node_id,
            observed_at,
            observations=observations,
        )
        self.context.evidence.capture(
            record_id=record_id,
            cluster_id=cluster_id,
            node_id=node_id,
            kind=kind,
            observed_at=observed_at,
            attempt_ids=context.attempt_ids,
            payload=payload,
        )

    def _ingest_gpu_inventory_batch(
        self,
        snapshots: list[GpuInventorySnapshot],
    ) -> list[GpuInventorySnapshot]:
        if not snapshots:
            return []
        now = datetime.now(timezone.utc)
        normalized = [
            snapshot
            if snapshot.ingested_at is not None
            else snapshot.model_copy(update={"ingested_at": now})
            for snapshot in snapshots
        ]
        for product in {
            device.product
            for snapshot in normalized
            for device in snapshot.devices
            if device.product
        }:
            self.context.policy.observe_product(product)
        finding_candidates: list[tuple[GpuInventorySnapshot, NodeHealthFinding]] = []
        clears: list[tuple[str, bool, datetime, float]] = []
        with self.context.store.processor_batch_transaction():
            refusals = self._stale_boot_refusals(normalized, now)
            accepted = iter(
                self.context.store.observe_gpu_inventory_snapshots(
                    [
                        snapshot
                        for snapshot, refusal in zip(normalized, refusals, strict=True)
                        if refusal is None
                    ]
                )
            )
            # A refused snapshot reads as one the store dropped (``False``):
            # it is answered with the baseline it did not replace.
            previous_values: list[GpuInventorySnapshot | Literal[False] | None] = [
                False if refusal is not None else next(accepted) for refusal in refusals
            ]
            statuses = []
            persisted_values = []
            evidence_candidates = []
            for snapshot, previous, refusal in zip(
                normalized, previous_values, refusals, strict=True
            ):
                persisted = (
                    self.context.store.get_gpu_inventory_snapshot(
                        snapshot.cluster_id, snapshot.node_id
                    )
                    if previous is False
                    else snapshot
                )
                persisted_values.append(persisted)
                statuses.append(self._inventory_status(snapshot, now, refusal))
                if previous is False:
                    continue
                unknown = self._expected_count_unknown_finding(snapshot)
                if unknown is not None:
                    finding_candidates.append((snapshot, unknown))
                else:
                    clears.append(
                        (
                            self._inventory_signal_key(
                                snapshot, GPU_EXPECTED_COUNT_UNKNOWN_METRIC
                            ),
                            False,
                            snapshot.observed_at,
                            0.0,
                        )
                    )
                if previous is None:
                    continue
                count_mismatch = (
                    snapshot.expected_gpu_count is not None
                    and len(snapshot.devices) != snapshot.expected_gpu_count
                )
                if count_mismatch or self._inventory_identity(
                    previous
                ) != self._inventory_identity(snapshot):
                    evidence_candidates.append(snapshot)
                changed = self._identity_changed_finding(previous, snapshot)
                if changed is not None:
                    finding_candidates.append((snapshot, changed))
                else:
                    clears.append(
                        (
                            self._inventory_signal_key(
                                snapshot, GPU_IDENTITY_CHANGED_METRIC
                            ),
                            False,
                            snapshot.observed_at,
                            0.0,
                        )
                    )
            self.context.store.save_collector_statuses_batch(statuses)
            for snapshot in evidence_candidates:
                self._capture_evidence(
                    record_id=(f"gpu-inventory/{snapshot.snapshot_id}"),
                    cluster_id=snapshot.cluster_id,
                    node_id=snapshot.node_id,
                    kind=EvidenceKind.GPU_INVENTORY,
                    observed_at=snapshot.observed_at,
                    payload=snapshot.model_dump(mode="json"),
                )
        # Findings are opened outside the batch transaction: incident,
        # workflow and marker writes are their own serialized unit.
        self._ingest_inventory_findings(finding_candidates, clears)
        return persisted_values

    def _stale_boot_refusals(
        self, snapshots: list[GpuInventorySnapshot], now: datetime
    ) -> list[str | None]:
        """Why each snapshot may not replace its node's baseline (``None``: it may).

        A snapshot from another boot than the baseline's says the node
        rebooted. That is believed of the node's current boot as its Agent
        reports it, and of nothing else: inventory from any other boot -- a
        previous boot's collector or outbox, a boot id that never existed --
        is refused however fresh its timestamp (GF-GENERAL-LIVE-026
        ``stale_boot``) and the baseline stays. A reboot the Agent has not
        re-registered yet waits for its next heartbeat, as the coordinator's
        generation fence does. With no current boot known (no Agent record,
        none reported, a lapsed lease, an Agent not ACTIVE) observed_at
        ordering alone decides, as before; so it does for a node's first
        baseline and for a newer snapshot of the baseline's own boot. The
        batch is walked in order, each snapshot judged against the baseline
        the ones before it leave.
        """

        baselines: dict[tuple[str, str], GpuInventorySnapshot | None] = {}
        current_boots: dict[tuple[str, str], str | None] = {}
        refusals: list[str | None] = []
        for snapshot in snapshots:
            key = (snapshot.cluster_id, snapshot.node_id)
            if key not in baselines:
                baselines[key] = self.context.store.get_gpu_inventory_snapshot(*key)
            baseline = baselines[key]
            if baseline is None or snapshot.observed_at <= baseline.observed_at:
                # A first baseline, or a stale snapshot / retry the store
                # itself drops: nothing to judge here.
                if baseline is None:
                    baselines[key] = snapshot
                refusals.append(None)
                continue
            refusal = None
            if (
                snapshot.source_boot_id
                and baseline.source_boot_id
                and snapshot.source_boot_id != baseline.source_boot_id
            ):
                if key not in current_boots:
                    current_boots[key] = self._current_boot_id(*key, now=now)
                current = current_boots[key]
                if current is not None and snapshot.source_boot_id != current:
                    refusal = (
                        f"GPU inventory from boot {snapshot.source_boot_id} is not "
                        f"the node's current boot {current} (Agent heartbeat); the "
                        f"baseline from boot {baseline.source_boot_id} was kept"
                    )
                    LOGGER.warning(
                        "GPU inventory %s on %s/%s carries boot %s but the node's "
                        "Agent reports boot %s; baseline %s (boot %s) kept",
                        snapshot.snapshot_id,
                        snapshot.cluster_id,
                        snapshot.node_id,
                        snapshot.source_boot_id,
                        current,
                        baseline.snapshot_id,
                        baseline.source_boot_id,
                    )
            if refusal is None:
                baselines[key] = snapshot
            refusals.append(refusal)
        return refusals

    def _current_boot_id(
        self, cluster_id: str, node_id: str, *, now: datetime
    ) -> str | None:
        """The node's current boot as its Agent reports it; ``None``: unknown.

        The authority the coordinator's generation fence trusts: an ACTIVE
        Agent record with a boot id whose heartbeat lease has not lapsed.
        """

        try:
            agent = self.context.store.get_agent(cluster_id, node_id)
        except NotFoundError:
            return None
        lease = agent.lease_expires_at
        if lease is not None and lease.tzinfo is None:
            lease = lease.replace(tzinfo=timezone.utc)
        if (
            agent.lifecycle_state is not AgentLifecycleState.ACTIVE
            or not agent.boot_id
            or lease is None
            or lease <= now
        ):
            return None
        return str(agent.boot_id)

    @staticmethod
    def _inventory_status(
        snapshot: GpuInventorySnapshot, now: datetime, refusal: str | None
    ) -> CollectorStatus:
        """The GPU_INVENTORY channel status one snapshot leaves.

        A refused snapshot is a delivery whose newest state is a failure: the
        collector answered, but not with the node's current inventory, so it
        is no success -- the channel reads erroring with the reason (and, if
        the node's boot never reports again, silent), not healthy.
        """

        return CollectorStatus(
            cluster_id=snapshot.cluster_id,
            node_id=snapshot.node_id,
            collector=CollectorKind.GPU_INVENTORY,
            observed_at=snapshot.observed_at,
            ingested_at=now,
            last_success_at=(snapshot.observed_at if refusal is None else None),
            last_error_at=(snapshot.observed_at if refusal is not None else None),
            batch_id=snapshot.snapshot_id,
            sample_count=len(snapshot.devices),
            errors=[] if refusal is None else [refusal],
        )

    @staticmethod
    def _inventory_signal_key(snapshot: GpuInventorySnapshot, metric: str) -> str:
        return f"{snapshot.cluster_id}/{snapshot.node_id}/{metric}/node"

    def _identity_changed_finding(
        self,
        previous: GpuInventorySnapshot,
        snapshot: GpuInventorySnapshot,
    ) -> NodeHealthFinding | None:
        """A changed GPU identity is a device lost, swapped or moved.

        The identity is the UUID set and each UUID's PCI address: the
        inventory's bus ids are what place an XID/SXID line or a PCIe metric
        on a GPU, so the same UUIDs answering on other addresses (same count
        included -- GF-GENERAL-LIVE-026 ``bdf_changed``) is a change, not only
        evidence. Another spelling of one address is not a move, and a bus id
        that does not parse on either side is unknown, not changed.

        Fail closed: the finding is CRITICAL whether or not an expected count
        is known, because the node the workload was scheduled on no longer
        has the GPUs it was scheduled with. ``RUN_DIAGNOSTICS`` opens an
        incident and a DCGM diagnostic/validation workflow without a reboot;
        the host collector's ``gpu_inventory_mismatch`` keeps the reboot path
        when a configured invariant is breached.
        """

        before = {device.gpu_uuid: device.pci_bdf for device in previous.devices}
        after = {device.gpu_uuid: device.pci_bdf for device in snapshot.devices}
        removed = sorted(set(before) - set(after))
        added = sorted(set(after) - set(before))
        moved = [
            {
                "gpu_uuid": gpu_uuid,
                "previous_pci_bdf": before[gpu_uuid],
                "observed_pci_bdf": after[gpu_uuid],
            }
            for gpu_uuid in sorted(set(before) & set(after))
            if _pci_bdf_moved(before[gpu_uuid], after[gpu_uuid])
        ]
        if not removed and not added and not moved:
            return None
        moved_text = [
            f"{item['gpu_uuid']} {item['previous_pci_bdf']}->{item['observed_pci_bdf']}"
            for item in moved
        ]
        event_id = f"{snapshot.snapshot_id}-{GPU_IDENTITY_CHANGED_METRIC}"
        LOGGER.warning(
            "GPU inventory identity changed cluster=%s node=%s removed=%s added=%s "
            "pci_bdf_changed=%s observed=%d expected=%s",
            snapshot.cluster_id,
            snapshot.node_id,
            removed,
            added,
            moved_text,
            len(after),
            snapshot.expected_gpu_count,
        )
        return NodeHealthFinding(
            finding_id=f"finding-{event_id}",
            event_id=event_id,
            cluster_id=snapshot.cluster_id,
            node_id=snapshot.node_id,
            observed_at=snapshot.observed_at,
            category=NodeHealthCategory.GPU,
            severity=Severity.CRITICAL,
            reason=(
                "GPU inventory identity changed between snapshots: "
                f"removed={removed} added={added}"
                + (f" pci_bdf_changed={moved_text}" if moved else "")
            ),
            recommended_action=RecoveryAction.RUN_DIAGNOSTICS,
            metric_name=GPU_IDENTITY_CHANGED_METRIC,
            value=float(len(removed) + len(added) + len(moved)),
            gpu_uuids=[*removed, *added, *(item["gpu_uuid"] for item in moved)],
            evidence_ref=snapshot.evidence_ref
            or f"gpu-inventory://{snapshot.node_id}/{snapshot.snapshot_id}",
            runtime_profile_version=snapshot.runtime_profile_version,
            diagnostic_parameters={
                "removed_gpu_uuids": removed,
                "added_gpu_uuids": added,
                "pci_bdf_changes": moved,
                "previous_gpu_count": len(before),
                "observed_gpu_count": len(after),
                "expected_gpu_count": snapshot.expected_gpu_count,
                "source_boot_id": snapshot.source_boot_id,
                "previous_source_boot_id": previous.source_boot_id,
            },
            policy_reference=(
                "a GPU that leaves the inventory or answers on another PCI "
                "address is a fault, not only evidence"
            ),
        )

    @staticmethod
    def _expected_count_unknown_finding(
        snapshot: GpuInventorySnapshot,
    ) -> NodeHealthFinding | None:
        if snapshot.expected_gpu_count is not None:
            return None
        event_id = f"{snapshot.snapshot_id}-{GPU_EXPECTED_COUNT_UNKNOWN_METRIC}"
        return NodeHealthFinding(
            finding_id=f"finding-{event_id}",
            event_id=event_id,
            cluster_id=snapshot.cluster_id,
            node_id=snapshot.node_id,
            observed_at=snapshot.observed_at,
            category=NodeHealthCategory.GPU,
            severity=Severity.WARNING,
            reason=(
                "GPU inventory snapshot carries no expected GPU count: the "
                "instance type is unknown to the collector and no explicit "
                "count was configured, so a lost GPU cannot be judged against "
                "the node invariant"
            ),
            recommended_action=RecoveryAction.COLLECT_EVIDENCE,
            metric_name=GPU_EXPECTED_COUNT_UNKNOWN_METRIC,
            value=float(len(snapshot.devices)),
            evidence_ref=snapshot.evidence_ref
            or f"gpu-inventory://{snapshot.node_id}/{snapshot.snapshot_id}",
            runtime_profile_version=snapshot.runtime_profile_version,
            diagnostic_parameters={
                "observed_gpu_count": len(snapshot.devices),
                "source_boot_id": snapshot.source_boot_id,
            },
            policy_reference=(
                "an unknown node invariant fails closed into operator review"
            ),
        )

    def _ingest_inventory_findings(
        self,
        candidates: list[tuple[GpuInventorySnapshot, NodeHealthFinding]],
        clears: list[tuple[str, bool, datetime, float]],
    ) -> list[NodeHealthIngestionResult]:
        """Open one incident per node per episode through the health signals.

        The identity signal deactivates on the next stable snapshot, so each
        change is its own event; the unknown-count signal stays active while
        the snapshots keep arriving without a count and clears the moment one
        carries it, so a fleet without the invariant configured is told once
        per node, not once per snapshot.
        """

        received_at = datetime.now(timezone.utc)
        if clears:
            self.context.store.claim_health_signal_transitions(
                clears, received_at=received_at
            )
        if not candidates:
            return []
        emitted = self.context.store.claim_health_signal_transitions(
            [
                (finding_health_signal_key(finding), True, finding.observed_at, 0.0)
                for _, finding in candidates
            ],
            received_at=received_at,
        )
        results = []
        service = NodeHealthIngestionService(self.context)
        for (snapshot, finding), emit in zip(candidates, emitted, strict=True):
            if not emit:
                continue
            results.append(service.ingest(snapshot.snapshot_id, [finding]))
            self.context.store.mark_health_signal_notified(
                finding_health_signal_key(finding), notified_at=received_at
            )
        return results
