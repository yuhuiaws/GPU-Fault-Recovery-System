"""Fleet-record templates shared by the key/value stores: regional cluster
registrations, the registry head/revision/member rows, agents, fleet
deployments and barriers, each a single row keyed by its own id."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, cast

from gpu_fault.store.shared.cleanup_log import log_cleanup
from gpu_fault.store.shared.primitives import (
    DeleteRecord,
    GetOptionalRecord,
    GetRecord,
    ListRecords,
    PutRecord,
    StatementGuard,
    StateTransaction,
)

if TYPE_CHECKING:
    from gpu_fault.regional import (
        RegionalRegistryHead,
        RegionalRegistryMember,
        RegionalRegistryRevision,
    )


class SharedFleetMixin:
    # Attributes supplied by the composed concrete implementation.
    _regional_cluster_region: Callable[..., Any]
    list_agents: Callable[..., Any]
    list_regional_cluster_ids: Callable[..., Any]

    _delete: DeleteRecord
    _get: GetRecord
    _get_optional: GetOptionalRecord
    _list: ListRecords
    _put: PutRecord
    _state_transaction: StateTransaction
    _statement_guard: StatementGuard

    def save_regional_cluster(self, registration):
        key = registration.cluster_id
        with self._state_transaction(f"regional_cluster/{key}"):
            existing_region = self._regional_cluster_region(key)
            if existing_region is not None and existing_region != registration.region:
                raise ValueError("regional cluster cannot move between regions")
            self._put("regional_cluster", key, registration)
            return registration

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

    def cleanup_stale_regional_registry_members(
        self, *, older_than: datetime, limit: int
    ) -> int:
        """Drop heartbeat rows of processes not seen since ``older_than``
        (F-5 / F-8). PostgreSQL overrides this with one set-based statement."""

        with self._state_transaction("regional_registry_member/cleanup"):
            member_ids = [
                item.member_id
                for item in sorted(
                    cast(
                        "list[RegionalRegistryMember]",
                        self._list("regional_registry_member"),
                    ),
                    key=lambda item: (item.last_seen_at, item.member_id),
                )
                if item.last_seen_at <= older_than
            ][:limit]
            for member_id in member_ids:
                self._delete("regional_registry_member", member_id)
            return log_cleanup("regional_registry_member", member_ids)

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
        with self._statement_guard():
            self._put(
                "agent",
                self._agent_key(agent.cluster_id, agent.node_id),
                agent,
            )

    def get_agent(self, cluster_id: str, node_id: str):
        return self._get("agent", self._agent_key(cluster_id, node_id))

    def save_fleet_deployment(self, deployment) -> None:
        with self._statement_guard():
            self._put(
                "fleet_deployment",
                deployment.deployment_id,
                deployment,
            )

    def get_fleet_deployment(self, deployment_id: str):
        return self._get("fleet_deployment", deployment_id)

    def save_barrier(self, barrier) -> None:
        with self._statement_guard():
            self._put("barrier", barrier.barrier_id, barrier)

    def get_barrier(self, barrier_id: str):
        return self._get("barrier", barrier_id)
