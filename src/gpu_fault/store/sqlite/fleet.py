from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from datetime import datetime

from gpu_fault.store.shared.cleanup_log import log_cleanup

if TYPE_CHECKING:
    from gpu_fault.fleet_deployment import FleetDeployment


class SqliteFleetMixin:
    # Attributes supplied by the composed concrete implementation.
    _agent_key: Callable[..., Any]
    _db: Any
    _delete: Callable[..., Any]
    _get_optional: Callable[..., Any]
    _list: Callable[..., Any]
    _models: Any
    _put: Callable[..., Any]
    _state_transaction: Callable[..., Any]

    def _regional_cluster_region(self, cluster_id: str) -> str | None:
        row = self._db.execute(
            """
            SELECT json_extract(payload, '$.region')
            FROM objects
            WHERE kind='regional_cluster' AND key=?
            """,
            (cluster_id,),
        ).fetchone()
        return str(row[0]) if row and row[0] is not None else None

    def list_regional_clusters(self):
        return sorted(
            self._list("regional_cluster"),
            key=lambda item: item.cluster_id,
        )

    def list_regional_cluster_ids(self) -> list[str]:
        rows = self._db.execute(
            """
            SELECT key
            FROM objects
            WHERE kind='regional_cluster'
            ORDER BY key
            """
        ).fetchall()
        return [str(row[0]) for row in rows]

    def replace_agent_if_matches(self, replacement, expected) -> bool:
        key = self._agent_key(
            replacement.cluster_id,
            replacement.node_id,
        )
        with self._state_transaction(f"agent/{key}"):
            current = self._get_optional("agent", key)
            if current != expected:
                return False
            self._put("agent", key, replacement)
            return True

    def list_agents(self, cluster_id: str | None = None):
        records = self._list("agent")
        if cluster_id is not None:
            records = [item for item in records if item.cluster_id == cluster_id]
        return sorted(
            records,
            key=lambda item: (
                item.cluster_id,
                item.node_id,
            ),
        )

    def replace_fleet_deployment_if_matches(
        self,
        replacement: FleetDeployment,
        expected: FleetDeployment | None,
    ) -> bool:
        with self._state_transaction(f"fleet_deployment/{replacement.deployment_id}"):
            current = self._get_optional(
                "fleet_deployment",
                replacement.deployment_id,
            )
            if current != expected:
                return False
            self._put(
                "fleet_deployment",
                replacement.deployment_id,
                replacement,
            )
            return True

    def list_fleet_deployments(self):
        return sorted(
            self._list("fleet_deployment"),
            key=lambda item: (
                item.created_at,
                item.deployment_id,
            ),
        )

    def list_active_fleet_deployments(self, cluster_id: str):
        rows = self._db.execute(
            """
            SELECT payload
            FROM objects
            WHERE kind='fleet_deployment'
              AND json_extract(payload, '$.cluster_id')=?
              AND json_extract(payload, '$.status')
                  NOT IN ('SUCCEEDED', 'FAILED')
            ORDER BY json_extract(payload, '$.created_at'), key
            """,
            (cluster_id,),
        ).fetchall()
        model = self._models["fleet_deployment"]
        return [model.model_validate_json(row[0]) for row in rows]

    def cleanup_terminal_fleet_deployments(
        self,
        *,
        older_than: datetime,
        limit: int,
    ) -> int:
        with self._state_transaction("fleet_deployment/cleanup"):
            deployment_ids = [
                item.deployment_id
                for item in sorted(
                    self._list("fleet_deployment"),
                    key=lambda item: (
                        item.updated_at,
                        item.deployment_id,
                    ),
                )
                if item.status.value in {"SUCCEEDED", "FAILED"}
                and item.updated_at <= older_than
            ][:limit]
            for deployment_id in deployment_ids:
                self._delete("fleet_deployment", deployment_id)
            return log_cleanup("fleet_deployment", deployment_ids)

    def list_barriers(self):
        return sorted(
            self._list("barrier"),
            key=lambda item: (
                item.created_at,
                item.barrier_id,
            ),
        )
