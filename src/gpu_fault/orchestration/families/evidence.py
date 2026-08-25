from __future__ import annotations

from datetime import datetime, timedelta, timezone
import logging
import os
from threading import Lock
from typing import Any, Callable

from gpu_fault.host_health import NodeHealthFinding
from gpu_fault.policy import SxidEvent, XidEvent


LOGGER = logging.getLogger(__name__)


class EvidenceOperationService:
    def __init__(
        self,
        store,
        active_node_exclusive_workflow: Callable,
    ) -> None:
        self.store = store
        self.active_node_exclusive_workflow = active_node_exclusive_workflow
        self._metrics_lock = Lock()
        self._ambiguous_attempt_ownership_total = 0

    def ambiguous_attempt_ownership_total(self) -> int:
        with self._metrics_lock:
            return self._ambiguous_attempt_ownership_total

    @staticmethod
    def utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    @staticmethod
    def sample_hung_triage_nodes(
        attempt_node_ids: list[str],
        *,
        reporting_node_id: str,
        lowest_rank_by_node: dict[str, int],
    ) -> tuple[list[str], list[str]]:
        maximum = int(os.getenv("GPU_FAULT_HUNG_TRIAGE_MAX_NODES", "8"))
        if maximum < 0:
            raise ValueError("GPU_FAULT_HUNG_TRIAGE_MAX_NODES must not be negative")
        ordered = sorted(set(attempt_node_ids))
        if maximum == 0 or len(ordered) <= maximum:
            return ordered, []
        required = [
            node_id
            for node_id in (
                reporting_node_id,
                min(
                    lowest_rank_by_node,
                    key=lambda node_id: (
                        lowest_rank_by_node[node_id],
                        node_id,
                    ),
                    default=None,
                ),
            )
            if node_id in ordered
        ]
        selected = list(dict.fromkeys(required))
        remaining = [node_id for node_id in ordered if node_id not in set(selected)]
        budget = maximum - len(selected)
        if budget > 0 and remaining:
            stride = len(remaining) / budget
            selected.extend(
                remaining[min(int(index * stride), len(remaining) - 1)]
                for index in range(budget)
            )
        sampled = sorted(dict.fromkeys(selected))
        return sampled, [node_id for node_id in ordered if node_id not in set(sampled)]

    def attempt_observation(
        self,
        event: XidEvent | SxidEvent | NodeHealthFinding,
    ):
        workload_ids = set(event.affected_workload_ids)
        job_id = getattr(event, "job_id", None)
        attempt_id = getattr(event, "attempt_id", None)
        event_gpu_uuids = (
            {event.gpu_uuid}
            if isinstance(event, XidEvent) and event.gpu_uuid
            else set(event.participating_gpu_uuids)
            if isinstance(event, SxidEvent)
            else set(event.gpu_uuids)
            if isinstance(event, NodeHealthFinding)
            else set()
        )
        pod_uid = getattr(event, "pod_uid", None)
        container_id = getattr(event, "container_id", None)
        host_pid = getattr(event, "host_pid", None)
        cgroup_path = getattr(event, "cgroup_path", None)
        ingested_at = getattr(event, "ingested_at", None)

        def container_id_matches(left: str, right: str) -> bool:
            return (
                left == right
                or left.rsplit("://", 1)[-1] == (right.rsplit("://", 1)[-1])
            )

        def cgroup_matches(left: str, right: str) -> bool:
            left = left.rstrip("/")
            right = right.rstrip("/")
            return (
                left == right
                or left.startswith(right + "/")
                or right.startswith(left + "/")
            )

        candidates = []
        for observation in self.store.list_attempt_observations(event.cluster_id):
            age = (
                self.utc(event.observed_at) - self.utc(observation.observed_at)
            ).total_seconds()
            delayed_explicit_generation = bool(
                job_id
                and attempt_id
                and observation.job_id == job_id
                and observation.attempt_id == attempt_id
                and observation.started_at is not None
                and self.utc(observation.started_at) <= self.utc(event.observed_at)
                and ingested_at is not None
                and self.utc(observation.observed_at) <= self.utc(ingested_at)
            )
            if (
                (age < -30 and not delayed_explicit_generation)
                or age > 120
                or observation.workload_phase.value not in {"PENDING", "RUNNING"}
                or not any(
                    container.node_id == event.node_id and not container.terminated
                    for container in observation.containers
                )
                or (
                    workload_ids
                    and not workload_ids.intersection(observation.workload_ids)
                )
                or (job_id is not None and observation.job_id != job_id)
                or (attempt_id is not None and observation.attempt_id != attempt_id)
            ):
                continue
            containers = [
                container
                for container in observation.containers
                if container.node_id == event.node_id and not container.terminated
            ]
            if event_gpu_uuids:
                containers = [
                    item
                    for item in containers
                    if event_gpu_uuids.intersection(item.gpu_uuids)
                ]
            if pod_uid is not None:
                containers = [item for item in containers if item.pod_uid == pod_uid]
            if container_id is not None:
                containers = [
                    item
                    for item in containers
                    if item.container_id is not None
                    and container_id_matches(item.container_id, container_id)
                ]
            if host_pid is not None:
                containers = [item for item in containers if item.host_pid == host_pid]
            if cgroup_path is not None:
                containers = [
                    item
                    for item in containers
                    if item.cgroup_path is not None
                    and cgroup_matches(item.cgroup_path, cgroup_path)
                ]
            if (
                event_gpu_uuids
                or pod_uid is not None
                or container_id is not None
                or host_pid is not None
                or cgroup_path is not None
            ) and not containers:
                continue
            candidates.append(observation)
        identities = {(item.job_id, item.attempt_id) for item in candidates}
        if len(identities) != 1:
            if len(identities) > 1:
                with self._metrics_lock:
                    self._ambiguous_attempt_ownership_total += 1
                LOGGER.warning(
                    "ambiguous attempt ownership for event %s on "
                    "cluster=%s node=%s: candidates=%s",
                    event.event_id,
                    event.cluster_id,
                    event.node_id,
                    sorted(identities),
                )
            elif not identities:
                recovered = self.active_recovery_attempt_observation(
                    event, event_gpu_uuids
                )
                if recovered is not None:
                    return recovered
            return None
        return max(candidates, key=lambda item: item.observed_at)

    def active_recovery_attempt_observation(
        self,
        event: XidEvent | SxidEvent | NodeHealthFinding,
        event_gpu_uuids: set[str],
    ):
        incumbent = self.active_node_exclusive_workflow(
            event.cluster_id, {event.node_id}
        )
        if incumbent is None:
            return None
        incident = self.store.get_incident(incumbent.incident_id)
        if not incident.job_id or not incident.attempt_id:
            return None
        if (
            event_gpu_uuids
            and incident.gpu_uuids
            and not event_gpu_uuids.intersection(incident.gpu_uuids)
        ):
            return None
        if self.utc(event.observed_at) < (
            self.utc(incumbent.created_at) - timedelta(seconds=30)
        ):
            return None
        candidates = [
            observation
            for observation in self.store.list_attempt_observations(event.cluster_id)
            if observation.job_id == incident.job_id
            and observation.attempt_id == incident.attempt_id
            and any(
                container.node_id == event.node_id
                and (
                    not event_gpu_uuids
                    or not container.gpu_uuids
                    or bool(event_gpu_uuids.intersection(container.gpu_uuids))
                )
                for container in observation.containers
            )
        ]
        if not candidates:
            return None
        selected = max(candidates, key=lambda item: item.observed_at)
        LOGGER.info(
            "recovered attempt ownership for event %s from active "
            "workflow %s: job=%s attempt=%s phase=%s",
            event.event_id,
            incumbent.request_id,
            selected.job_id,
            selected.attempt_id,
            selected.workload_phase.value,
        )
        return selected

    @staticmethod
    def workload_cgroup_paths_by_node(
        observation,
    ) -> dict[str, list[str]]:
        mapping: dict[str, set[str]] = {}
        for container in observation.containers:
            if (
                container.terminated
                or not container.node_id
                or not container.cgroup_path
            ):
                continue
            path = container.cgroup_path.rstrip("/")
            if path:
                mapping.setdefault(container.node_id, set()).add(path)
        return {node_id: sorted(paths) for node_id, paths in sorted(mapping.items())}

    def quiesce_parameters(
        self,
        parameters: dict[str, Any],
        observation,
    ) -> dict[str, Any]:
        paths = self.workload_cgroup_paths_by_node(observation)
        return (
            {
                **parameters,
                "workload_cgroup_paths_by_node": paths,
            }
            if paths
            else parameters
        )
