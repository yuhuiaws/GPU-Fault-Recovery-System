from __future__ import annotations

from typing import TYPE_CHECKING, Any

from datetime import datetime

from gpu_fault.store.shared.errors import NotFoundError

if TYPE_CHECKING:
    from gpu_fault.fleet_deployment import FleetDeployment
    from gpu_fault.regional import (
        RegionalRegistryHead,
        RegionalRegistryMember,
        RegionalRegistryRevision,
    )


class MemoryFleetMixin:
    # Attributes supplied by the composed concrete implementation.
    _agents: Any
    _barriers: Any
    _fleet_deployments: Any
    _regional_clusters: Any
    _regional_registry_head: RegionalRegistryHead | None
    _regional_registry_members: dict[str, RegionalRegistryMember]
    _regional_registry_revisions: dict[int, RegionalRegistryRevision]

    _lock: Any

    def save_regional_cluster(self, registration):
        with self._lock:
            existing = self._regional_clusters.get(registration.cluster_id)
            if existing is not None and existing.region != registration.region:
                raise ValueError("regional cluster cannot move between regions")
            self._regional_clusters[registration.cluster_id] = registration
            return registration

    def get_regional_cluster(self, cluster_id: str):
        with self._lock:
            registration = self._regional_clusters.get(cluster_id)
            if registration is None:
                raise NotFoundError(cluster_id)
            return registration

    def delete_regional_cluster(self, cluster_id: str) -> None:
        with self._lock:
            self._regional_clusters.pop(cluster_id, None)
            for key in [key for key in self._agents if key[0] == cluster_id]:
                del self._agents[key]

    def list_regional_clusters(self):
        with self._lock:
            return sorted(
                self._regional_clusters.values(),
                key=lambda item: item.cluster_id,
            )

    def list_regional_cluster_ids(self) -> list[str]:
        with self._lock:
            return sorted(self._regional_clusters)

    def get_regional_registry_head(self) -> RegionalRegistryHead:
        with self._lock:
            if self._regional_registry_head is None:
                raise NotFoundError("regional-registry-head")
            return self._regional_registry_head

    def get_regional_registry_revision(
        self, generation: int
    ) -> RegionalRegistryRevision:
        with self._lock:
            revision = self._regional_registry_revisions.get(generation)
            if revision is None:
                raise NotFoundError(str(generation))
            return revision

    def publish_regional_registry_revision(
        self,
        revision: RegionalRegistryRevision,
        *,
        expected_generation: int,
    ) -> RegionalRegistryHead:
        from gpu_fault.regional import RegionalRegistryHead

        with self._lock:
            current_generation = (
                self._regional_registry_head.generation
                if self._regional_registry_head is not None
                else 0
            )
            if (
                self._regional_registry_head is not None
                and self._regional_registry_head.generation == revision.generation
                and self._regional_registry_head.content_sha256
                == revision.content_sha256
            ):
                return self._regional_registry_head
            if current_generation != expected_generation:
                raise ValueError("regional registry generation conflict")
            if revision.generation != expected_generation + 1:
                raise ValueError("regional registry generation must be consecutive")
            for registration in revision.registrations:
                existing = self._regional_clusters.get(registration.cluster_id)
                if existing is not None and existing.region != registration.region:
                    raise ValueError("regional cluster cannot move between regions")
            self._regional_registry_revisions[revision.generation] = revision
            self._regional_clusters = {
                item.cluster_id: item for item in revision.registrations
            }
            self._regional_registry_head = RegionalRegistryHead(
                generation=revision.generation,
                content_sha256=revision.content_sha256,
                updated_at=revision.created_at,
            )
            return self._regional_registry_head

    def save_regional_registry_member(
        self, member: RegionalRegistryMember
    ) -> RegionalRegistryMember:
        with self._lock:
            self._regional_registry_members[member.member_id] = member
            return member

    def list_regional_registry_members(self) -> list[RegionalRegistryMember]:
        with self._lock:
            members: list[RegionalRegistryMember] = list(
                self._regional_registry_members.values()
            )
            return sorted(members, key=lambda item: item.member_id)

    def save_agent(self, agent) -> None:
        with self._lock:
            self._agents[(agent.cluster_id, agent.node_id)] = agent

    def replace_agent_if_matches(self, replacement, expected) -> bool:
        with self._lock:
            key = (replacement.cluster_id, replacement.node_id)
            if self._agents.get(key) != expected:
                return False
            self._agents[key] = replacement
            return True

    def get_agent(self, cluster_id: str, node_id: str):
        with self._lock:
            record = self._agents.get((cluster_id, node_id))
            if record is None:
                raise NotFoundError(f"{cluster_id}/{node_id}")
            return record

    def list_agents(self, cluster_id: str | None = None):
        with self._lock:
            records = list(self._agents.values())
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
            self._fleet_deployments[deployment.deployment_id] = deployment

    def replace_fleet_deployment_if_matches(
        self,
        replacement: FleetDeployment,
        expected: FleetDeployment | None,
    ) -> bool:
        with self._lock:
            if self._fleet_deployments.get(replacement.deployment_id) != expected:
                return False
            self._fleet_deployments[replacement.deployment_id] = replacement
            return True

    def get_fleet_deployment(self, deployment_id: str):
        with self._lock:
            deployment = self._fleet_deployments.get(deployment_id)
            if deployment is None:
                raise NotFoundError(deployment_id)
            return deployment

    def list_fleet_deployments(self):
        with self._lock:
            return sorted(
                self._fleet_deployments.values(),
                key=lambda item: (
                    item.created_at,
                    item.deployment_id,
                ),
            )

    def list_active_fleet_deployments(self, cluster_id: str):
        with self._lock:
            return sorted(
                (
                    item
                    for item in self._fleet_deployments.values()
                    if item.cluster_id == cluster_id
                    and item.status.value
                    not in {
                        "SUCCEEDED",
                        "FAILED",
                    }
                ),
                key=lambda item: (
                    item.created_at,
                    item.deployment_id,
                ),
            )

    def cleanup_terminal_fleet_deployments(
        self,
        *,
        older_than: datetime,
        limit: int,
    ) -> int:
        """Drop SUCCEEDED/FAILED deployments past the retention window.

        A deployment carries one row per node in the wave plan, so a
        regional fleet roll writes the largest single objects the store
        holds. Only the open ones are ever read again -- every agent
        heartbeat asks for its cluster's active deployments -- so the
        terminal ones grow without bound and widen that lookup.
        """

        with self._lock:
            deployment_ids = [
                item.deployment_id
                for item in sorted(
                    self._fleet_deployments.values(),
                    key=lambda item: (
                        item.updated_at,
                        item.deployment_id,
                    ),
                )
                if item.status.value in {"SUCCEEDED", "FAILED"}
                and item.updated_at <= older_than
            ][:limit]
            for deployment_id in deployment_ids:
                del self._fleet_deployments[deployment_id]
            return len(deployment_ids)

    def save_barrier(self, barrier) -> None:
        with self._lock:
            self._barriers[barrier.barrier_id] = barrier

    def get_barrier(self, barrier_id: str):
        with self._lock:
            barrier = self._barriers.get(barrier_id)
            if barrier is None:
                raise NotFoundError(barrier_id)
            return barrier

    def list_barriers(self):
        with self._lock:
            return sorted(
                self._barriers.values(),
                key=lambda item: (
                    item.created_at,
                    item.barrier_id,
                ),
            )
