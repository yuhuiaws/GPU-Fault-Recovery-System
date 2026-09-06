from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, cast

from datetime import datetime

if TYPE_CHECKING:
    from gpu_fault.fleet_deployment import FleetDeployment
    from gpu_fault.regional import (
        RegionalRegistryHead,
        RegionalRegistryMember,
        RegionalRegistryRevision,
    )


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
            existing_region = self._regional_cluster_region(key)
            if existing_region is not None and existing_region != registration.region:
                raise ValueError("regional cluster cannot move between regions")
            self._put("regional_cluster", key, registration)
            return registration

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

    def get_regional_cluster(self, cluster_id: str):
        return self._get("regional_cluster", cluster_id)

    def delete_regional_cluster(self, cluster_id: str) -> None:
        with self._state_transaction(f"regional_cluster/{cluster_id}"):
            self._delete("regional_cluster", cluster_id)
            for agent in self.list_agents(cluster_id):
                self._delete(
                    "agent",
                    self._agent_key(agent.cluster_id, agent.node_id),
                )

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

    def get_regional_registry_head(self) -> RegionalRegistryHead:
        return cast(
            "RegionalRegistryHead",
            self._get("regional_registry_head", "current"),
        )

    def get_regional_registry_revision(
        self, generation: int
    ) -> RegionalRegistryRevision:
        return cast(
            "RegionalRegistryRevision",
            self._get("regional_registry_revision", str(generation)),
        )

    def publish_regional_registry_revision(
        self,
        revision: RegionalRegistryRevision,
        *,
        expected_generation: int,
    ) -> RegionalRegistryHead:
        from gpu_fault.regional import RegionalRegistryHead

        with self._state_transaction("regional_registry/head"):
            current = self._get_optional("regional_registry_head", "current")
            current_generation = current.generation if current is not None else 0
            if (
                current is not None
                and current.generation == revision.generation
                and current.content_sha256 == revision.content_sha256
            ):
                return cast("RegionalRegistryHead", current)
            if current_generation != expected_generation:
                raise ValueError("regional registry generation conflict")
            if revision.generation != expected_generation + 1:
                raise ValueError("regional registry generation must be consecutive")
            for registration in revision.registrations:
                existing = self._get_optional(
                    "regional_cluster",
                    registration.cluster_id,
                )
                if existing is not None and existing.region != registration.region:
                    raise ValueError("regional cluster cannot move between regions")
            configured_ids = {
                registration.cluster_id for registration in revision.registrations
            }
            for cluster_id in self.list_regional_cluster_ids():
                if cluster_id not in configured_ids:
                    self._delete("regional_cluster", cluster_id)
            for registration in revision.registrations:
                self._put(
                    "regional_cluster",
                    registration.cluster_id,
                    registration,
                )
            self._put(
                "regional_registry_revision",
                str(revision.generation),
                revision,
            )
            head = RegionalRegistryHead(
                generation=revision.generation,
                content_sha256=revision.content_sha256,
                updated_at=revision.created_at,
            )
            self._put("regional_registry_head", "current", head)
            return head

    def save_regional_registry_member(
        self, member: RegionalRegistryMember
    ) -> RegionalRegistryMember:
        with self._state_transaction(f"regional_registry_member/{member.member_id}"):
            self._put(
                "regional_registry_member",
                member.member_id,
                member,
            )
            return member

    def list_regional_registry_members(self) -> list[RegionalRegistryMember]:
        members = cast(
            "list[RegionalRegistryMember]",
            self._list("regional_registry_member"),
        )
        return sorted(members, key=lambda item: item.member_id)

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
        with self._state_transaction(f"agent/{key}"):
            current = self._get_optional("agent", key)
            if current != expected:
                return False
            self._put("agent", key, replacement)
            return True

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
