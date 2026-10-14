"""The departed-node cleanup's records on the key/value stores.

A cluster's Kubernetes Node inventory is one record merged under that
cluster's lock: the merge reads the previous record, so two executor replicas
reporting at once must not lose each other's absences. The ledger is one
record per departed node, written by the single cleanup that holds the
periodic task lease. A spare release is written by the administrator's
``config spare --release`` alone and only read by the cleanup, so it is a
record of its own rather than a field the cleanup would rewrite; so is the
administrator's acceptance of an unknown provider identity.
"""

from __future__ import annotations

from datetime import datetime
from typing import cast

from gpu_fault.departed_nodes import (
    DepartedNodeCleanupRecord,
    DepartedNodeIdentityAcceptance,
    DepartedNodeSpareRelease,
    departed_node_cleanup_key,
)
from gpu_fault.kubernetes_node_inventory import (
    KubernetesNodeInventory,
    KubernetesNodeInventoryReport,
    merge_node_inventory,
)
from gpu_fault.store.shared.primitives import (
    DeleteRecord,
    GetOptionalRecord,
    ListRecords,
    PutRecord,
    StatementGuard,
    StateTransaction,
)

INVENTORY_KIND = "kubernetes_node_inventory"
CLEANUP_KIND = "departed_node_cleanup"
SPARE_RELEASE_KIND = "departed_node_spare_release"
IDENTITY_ACCEPTANCE_KIND = "departed_node_identity_acceptance"


class SharedDepartedNodeMixin:
    # Attributes supplied by the composed concrete implementation.
    _delete: DeleteRecord
    _get_optional: GetOptionalRecord
    _list: ListRecords
    _put: PutRecord
    _statement_guard: StatementGuard
    _state_transaction: StateTransaction

    def record_kubernetes_node_inventory(
        self,
        report: KubernetesNodeInventoryReport,
        *,
        received_at: datetime,
    ) -> KubernetesNodeInventory:
        with self._state_transaction(f"{INVENTORY_KIND}/{report.cluster_id}"):
            previous = cast(
                KubernetesNodeInventory | None,
                self._get_optional(INVENTORY_KIND, report.cluster_id),
            )
            merged = merge_node_inventory(previous, report, received_at=received_at)
            if merged is not previous:
                self._put(INVENTORY_KIND, report.cluster_id, merged)
            return merged

    def get_kubernetes_node_inventory(
        self, cluster_id: str
    ) -> KubernetesNodeInventory | None:
        return cast(
            KubernetesNodeInventory | None,
            self._get_optional(INVENTORY_KIND, cluster_id),
        )

    def save_departed_node_cleanup(self, record: DepartedNodeCleanupRecord) -> None:
        with self._statement_guard():
            self._put(CLEANUP_KIND, record.key, record)

    def list_departed_node_cleanups(
        self, cluster_id: str | None = None
    ) -> list[DepartedNodeCleanupRecord]:
        records = cast(list[DepartedNodeCleanupRecord], self._list(CLEANUP_KIND))
        return sorted(
            (item for item in records if cluster_id in (None, item.cluster_id)),
            key=lambda item: (item.cluster_id, item.node_id),
        )

    def delete_departed_node_cleanup(self, cluster_id: str, node_id: str) -> None:
        with self._statement_guard():
            self._delete(CLEANUP_KIND, departed_node_cleanup_key(cluster_id, node_id))

    def save_departed_node_spare_release(
        self, release: DepartedNodeSpareRelease
    ) -> None:
        with self._statement_guard():
            self._put(SPARE_RELEASE_KIND, release.key, release)

    def list_departed_node_spare_releases(
        self, cluster_id: str | None = None
    ) -> list[DepartedNodeSpareRelease]:
        releases = cast(list[DepartedNodeSpareRelease], self._list(SPARE_RELEASE_KIND))
        return sorted(
            (item for item in releases if cluster_id in (None, item.cluster_id)),
            key=lambda item: (item.cluster_id, item.node_id),
        )

    def delete_departed_node_spare_release(self, cluster_id: str, node_id: str) -> None:
        with self._statement_guard():
            self._delete(
                SPARE_RELEASE_KIND, departed_node_cleanup_key(cluster_id, node_id)
            )

    def save_departed_node_identity_acceptance(
        self, acceptance: DepartedNodeIdentityAcceptance
    ) -> None:
        with self._statement_guard():
            self._put(IDENTITY_ACCEPTANCE_KIND, acceptance.key, acceptance)

    def list_departed_node_identity_acceptances(
        self, cluster_id: str | None = None
    ) -> list[DepartedNodeIdentityAcceptance]:
        acceptances = cast(
            list[DepartedNodeIdentityAcceptance], self._list(IDENTITY_ACCEPTANCE_KIND)
        )
        return sorted(
            (item for item in acceptances if cluster_id in (None, item.cluster_id)),
            key=lambda item: (item.cluster_id, item.node_id),
        )

    def delete_departed_node_identity_acceptance(
        self, cluster_id: str, node_id: str
    ) -> None:
        with self._statement_guard():
            self._delete(
                IDENTITY_ACCEPTANCE_KIND,
                departed_node_cleanup_key(cluster_id, node_id),
            )
