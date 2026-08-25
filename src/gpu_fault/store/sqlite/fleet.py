from __future__ import annotations

from typing import Any, Callable

from datetime import datetime


class SqliteFleetMixin:
    # Attributes supplied by the composed concrete implementation.
    _db: Any
    _delete: Callable[..., Any]
    _get: Callable[..., Any]
    _get_optional: Callable[..., Any]
    _list: Callable[..., Any]
    _lock: Any
    _models: Any
    _put: Callable[..., Any]
    _state_transaction: Callable[..., Any]

    def save_regional_cluster(self, registration):
        key = registration.cluster_id
        with self._state_transaction(f"regional_cluster/{key}"):
            existing = self._get_optional("regional_cluster", key)
            if existing is not None and existing.region != registration.region:
                raise ValueError("regional cluster cannot move between regions")
            self._put("regional_cluster", key, registration)
            return registration

    def get_regional_cluster(self, cluster_id: str):
        return self._get("regional_cluster", cluster_id)

    def delete_regional_cluster(self, cluster_id: str) -> None:
        with self._state_transaction(f"regional_cluster/{cluster_id}"):
            self._delete("regional_cluster", cluster_id)

    def list_regional_clusters(self):
        return sorted(
            self._list("regional_cluster"),
            key=lambda item: item.cluster_id,
        )

    @staticmethod
    def _agent_key(cluster_id: str, node_id: str) -> str:
        return f"{cluster_id}/{node_id}"

    def save_agent(self, agent) -> None:
        with self._lock:
            self._put(
                "agent",
                self._agent_key(agent.cluster_id, agent.node_id),
                agent,
            )

    def replace_agent_if_matches(self, replacement, expected) -> bool:
        key = self._agent_key(
            replacement.cluster_id,
            replacement.node_id,
        )
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                current = self._get_optional("agent", key)
                if current != expected:
                    self._db.execute("ROLLBACK")
                    return False
                self._put("agent", key, replacement)
                self._db.execute("COMMIT")
                return True
            except Exception:
                self._db.execute("ROLLBACK")
                raise

    def get_agent(self, cluster_id: str, node_id: str):
        return self._get("agent", self._agent_key(cluster_id, node_id))

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

    def save_fleet_deployment(self, deployment) -> None:
        with self._lock:
            self._put(
                "fleet_deployment",
                deployment.deployment_id,
                deployment,
            )

    def get_fleet_deployment(self, deployment_id: str):
        return self._get("fleet_deployment", deployment_id)

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
            return len(deployment_ids)

    def save_barrier(self, barrier) -> None:
        with self._lock:
            self._put("barrier", barrier.barrier_id, barrier)

    def get_barrier(self, barrier_id: str):
        return self._get("barrier", barrier_id)

    def list_barriers(self):
        return sorted(
            self._list("barrier"),
            key=lambda item: (
                item.created_at,
                item.barrier_id,
            ),
        )
