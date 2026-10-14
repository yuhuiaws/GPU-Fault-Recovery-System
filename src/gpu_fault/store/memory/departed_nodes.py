"""The departed-node cleanup's records in the in-memory store."""

from __future__ import annotations

from datetime import datetime
from threading import RLock

from gpu_fault.departed_nodes import (
    DepartedNodeCleanupRecord,
    DepartedNodeSpareRelease,
    departed_node_cleanup_key,
)
from gpu_fault.kubernetes_node_inventory import (
    KubernetesNodeInventory,
    KubernetesNodeInventoryReport,
    merge_node_inventory,
)


class MemoryDepartedNodeMixin:
    # Attributes supplied by the composed concrete implementation.
    _lock: RLock
    _kubernetes_node_inventories: dict[str, KubernetesNodeInventory]
    _departed_node_cleanups: dict[str, DepartedNodeCleanupRecord]
    _departed_node_spare_releases: dict[str, DepartedNodeSpareRelease]

    def record_kubernetes_node_inventory(
        self,
        report: KubernetesNodeInventoryReport,
        *,
        received_at: datetime,
    ) -> KubernetesNodeInventory:
        with self._lock:
            previous = self._kubernetes_node_inventories.get(report.cluster_id)
            merged = merge_node_inventory(previous, report, received_at=received_at)
            self._kubernetes_node_inventories[report.cluster_id] = merged
            return merged

    def get_kubernetes_node_inventory(
        self, cluster_id: str
    ) -> KubernetesNodeInventory | None:
        with self._lock:
            return self._kubernetes_node_inventories.get(cluster_id)

    def save_departed_node_cleanup(self, record: DepartedNodeCleanupRecord) -> None:
        with self._lock:
            self._departed_node_cleanups[record.key] = record

    def list_departed_node_cleanups(
        self, cluster_id: str | None = None
    ) -> list[DepartedNodeCleanupRecord]:
        with self._lock:
            records = list(self._departed_node_cleanups.values())
        return sorted(
            (item for item in records if cluster_id in (None, item.cluster_id)),
            key=lambda item: (item.cluster_id, item.node_id),
        )

    def delete_departed_node_cleanup(self, cluster_id: str, node_id: str) -> None:
        with self._lock:
            self._departed_node_cleanups.pop(
                departed_node_cleanup_key(cluster_id, node_id), None
            )

    def save_departed_node_spare_release(
        self, release: DepartedNodeSpareRelease
    ) -> None:
        with self._lock:
            self._departed_node_spare_releases[release.key] = release

    def list_departed_node_spare_releases(
        self, cluster_id: str | None = None
    ) -> list[DepartedNodeSpareRelease]:
        with self._lock:
            releases = list(self._departed_node_spare_releases.values())
        return sorted(
            (item for item in releases if cluster_id in (None, item.cluster_id)),
            key=lambda item: (item.cluster_id, item.node_id),
        )

    def delete_departed_node_spare_release(self, cluster_id: str, node_id: str) -> None:
        with self._lock:
            self._departed_node_spare_releases.pop(
                departed_node_cleanup_key(cluster_id, node_id), None
            )
