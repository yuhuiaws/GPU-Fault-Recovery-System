from __future__ import annotations

from datetime import datetime, timezone

from gpu_fault.gpu_metrics import GpuInventorySnapshot
from gpu_fault.models import WorkloadState
from gpu_fault.telemetry import (
    CollectorKind,
    CollectorStatus,
    EvidenceKind,
)


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
        collection_errors = errors or []
        self.context.store.save_collector_status(
            CollectorStatus(
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
        )

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
        with self.context.store.processor_batch_transaction():
            previous_values = self.context.store.observe_gpu_inventory_snapshots(
                normalized
            )
            statuses = []
            persisted_values = []
            evidence_candidates = []
            for snapshot, previous in zip(normalized, previous_values, strict=True):
                persisted = (
                    self.context.store.get_gpu_inventory_snapshot(
                        snapshot.cluster_id, snapshot.node_id
                    )
                    if previous is False
                    else snapshot
                )
                persisted_values.append(persisted)
                statuses.append(
                    CollectorStatus(
                        cluster_id=snapshot.cluster_id,
                        node_id=snapshot.node_id,
                        collector=CollectorKind.GPU_INVENTORY,
                        observed_at=snapshot.observed_at,
                        ingested_at=now,
                        last_success_at=snapshot.observed_at,
                        batch_id=snapshot.snapshot_id,
                        sample_count=len(snapshot.devices),
                    )
                )
                if previous is False or previous is None:
                    continue
                count_mismatch = (
                    snapshot.expected_gpu_count is not None
                    and len(snapshot.devices) != snapshot.expected_gpu_count
                )
                if count_mismatch or self._inventory_identity(
                    previous
                ) != self._inventory_identity(snapshot):
                    evidence_candidates.append(snapshot)
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
        return persisted_values
