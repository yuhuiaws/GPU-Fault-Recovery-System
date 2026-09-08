from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from enum import StrEnum

from pydantic import Field

# ``CollectorKind`` and the producer view are rows of the collector registry;
# they keep their historical import path here.
from gpu_fault.collector_registry import (
    COLLECTOR_PRODUCER_BY_CHANNEL as COLLECTOR_PRODUCER_BY_CHANNEL,
)
from gpu_fault.collector_registry import CollectorKind as CollectorKind
from gpu_fault.models import StrictModel
from gpu_fault.telemetry_models import (
    TelemetryMetricLatest as TelemetryMetricLatest,
)
from gpu_fault.telemetry_models import (
    WorkloadObservationState as WorkloadObservationState,
)
from gpu_fault.watcher import AttemptObservation, WorkloadPhase


def collector_producer(channel: CollectorKind) -> str:
    return COLLECTOR_PRODUCER_BY_CHANNEL[channel]


class EvidenceKind(StrEnum):
    GPU_INVENTORY = "GPU_INVENTORY"
    GPU_METRICS = "GPU_METRICS"
    HOST_TELEMETRY = "HOST_TELEMETRY"
    NODE_LOGS = "NODE_LOGS"
    NVIDIA_KERNEL = "NVIDIA_KERNEL"
    FABRIC_MANAGER_LOG = "FABRIC_MANAGER_LOG"
    HMA = "HMA"
    TRAINING_PROGRESS = "TRAINING_PROGRESS"
    WORKLOAD_LOG = "WORKLOAD_LOG"
    ADMIN_ACTION = "ADMIN_ACTION"


class CollectorStatus(StrictModel):
    cluster_id: str
    node_id: str
    collector: CollectorKind
    observed_at: datetime
    ingested_at: datetime
    last_success_at: datetime | None = None
    last_error_at: datetime | None = None
    batch_id: str | None = None
    sample_count: int = Field(default=0, ge=0)
    errors: list[str] = Field(default_factory=list)


class CollectorHealthSummary(StrictModel):
    summary_id: str
    cluster_id: str
    node_id: str
    collector: CollectorKind
    observed_at: datetime
    edge_filter_reasons: list[str] = Field(default_factory=lambda: ["health-summary"])


class CollectorMetricsSnapshotRecord(StrictModel):
    observed_at: datetime
    lines: list[str]
    details: list[dict]


class RawEvidenceRecord(StrictModel):
    record_id: str
    cluster_id: str
    node_id: str
    kind: EvidenceKind
    observed_at: datetime
    ingested_at: datetime
    expires_at: datetime
    attempt_ids: list[str] = Field(default_factory=list)
    payload: dict


class EvidenceService:
    def __init__(
        self,
        store,
        *,
        retention: timedelta = timedelta(hours=24),
        max_records_per_node: int = 10000,
    ) -> None:
        self.store = store
        self.retention = retention
        self.max_records_per_node = max_records_per_node

    @classmethod
    def from_environment(cls, store) -> EvidenceService:
        return cls(
            store,
            retention=timedelta(
                hours=float(os.getenv("GPU_FAULT_EVIDENCE_RETENTION_HOURS", "24"))
            ),
            max_records_per_node=int(
                os.getenv(
                    "GPU_FAULT_EVIDENCE_MAX_RECORDS_PER_NODE",
                    "10000",
                )
            ),
        )

    def capture(
        self,
        *,
        record_id: str,
        cluster_id: str,
        node_id: str,
        kind: EvidenceKind,
        observed_at: datetime,
        attempt_ids: list[str],
        payload: dict,
    ) -> RawEvidenceRecord:
        ingested_at = datetime.now(timezone.utc)
        record = RawEvidenceRecord(
            record_id=record_id,
            cluster_id=cluster_id,
            node_id=node_id,
            kind=kind,
            observed_at=observed_at,
            ingested_at=ingested_at,
            expires_at=ingested_at + self.retention,
            attempt_ids=attempt_ids,
            payload=payload,
        )
        self.store.save_raw_evidence(
            record,
            max_records_per_node=self.max_records_per_node,
        )
        return record


#: Where the Completion Watcher posts :class:`WorkloadCoverageHeartbeat`. It is
#: deliberately not a ``CHANNEL_REGISTRY`` channel: the heartbeat carries no
#: attempt, needs no receipt and is superseded by the next pass.
ATTEMPT_COVERAGE_PATH = "/v1/attempts/coverage"


class WorkloadCoverageHeartbeat(StrictModel):
    """One watcher's statement that it watched a whole cluster and is current.

    An idle cluster produces no attempt observation, so "no observation" used
    to mean both "nobody is watching" and "there is nothing to see", and the
    resolver had to fail closed on UNKNOWN for both (completion-watcher F4).
    This is the second statement made explicit: the watcher completed a full
    pass at ``observed_at`` and found ``watched_attempts`` attempts across
    ``watched_pods`` managed Pods -- zero being the interesting case.

    ``resource_version`` is the Kubernetes list revision the pass started from
    and ``watcher_instance`` names the process, so an operator can tell a
    heartbeat from a watcher that has since been replaced from a current one.
    Neither is read by the resolver: coverage is decided by ``observed_at``
    alone, which is what keeps a rolled-back watcher from claiming coverage
    with a stale clock.
    """

    cluster_id: str
    observed_at: datetime
    watched_pods: int = Field(default=0, ge=0)
    watched_attempts: int = Field(default=0, ge=0)
    resource_version: str | None = None
    watcher_instance: str


class WorkloadContext(StrictModel):
    workload_state: str
    workload_ids: list[str] = Field(default_factory=list)
    job_ids: list[str] = Field(default_factory=list)
    attempt_ids: list[str] = Field(default_factory=list)
    job_attempt_ids: list[tuple[str, str]] = Field(default_factory=list)
    runtime_profile_version: str | None = None
    ranks: list[int] = Field(default_factory=list)
    gpu_uuids: list[str] = Field(default_factory=list)


class WorkloadTopologyService:
    def __init__(
        self,
        store,
        *,
        max_age_seconds: int = 120,
        freshness_seconds: float = 600,
        coverage_freshness_seconds: float | None = None,
    ) -> None:
        if freshness_seconds <= 0:
            raise ValueError("freshness_seconds must be positive")
        if freshness_seconds < max_age_seconds:
            raise ValueError(
                "freshness_seconds must not be shorter than max_age_seconds: an "
                "observation young enough to match a node must also count as "
                "coverage of its cluster"
            )
        if coverage_freshness_seconds is None:
            coverage_freshness_seconds = freshness_seconds
        if coverage_freshness_seconds <= 0:
            raise ValueError("coverage_freshness_seconds must be positive")
        self.store = store
        self.max_age_seconds = max_age_seconds
        # How recently the cluster must have produced any attempt observation
        # for "nothing matches this node" to mean IDLE rather than UNKNOWN.
        self.freshness_seconds = freshness_seconds
        # The same question for the watcher's own coverage heartbeat, which is
        # the only coverage an idle cluster produces. Defaults to the
        # observation window so one setting moves both.
        self.coverage_freshness_seconds = coverage_freshness_seconds

    def observe(self, observation: AttemptObservation) -> None:
        self.store.save_attempt_observation(observation)

    def observe_coverage(self, heartbeat: WorkloadCoverageHeartbeat) -> bool:
        """Record a watcher's full-pass heartbeat; ``False`` when it is stale.

        One row per cluster: the store keeps the newest ``observed_at`` and
        refuses an older one, so the last in-flight heartbeat of a watcher a
        rollout has already replaced cannot move coverage backwards.
        """

        return bool(self.store.save_workload_coverage_heartbeat(heartbeat))

    def _covered_by_heartbeat(self, cluster_id: str, observed_at: datetime) -> bool:
        """Whether a watcher vouched for the whole cluster recently enough.

        Read only when no observation covers the cluster -- on a busy cluster
        this costs nothing, and on an idle one it is a single keyed row.

        The answer is per *cluster*, exactly like observation coverage: the
        watcher lists every managed Pod of the cluster in one pass, so "I found
        no attempt" is a statement about every node at once. That is what the
        planner needs, because the node it asks about is by definition one the
        observations did not name.
        """

        # A store that does not carry heartbeats at all -- a narrow proxy, a
        # double -- means "no coverage", the same answer it gave before the
        # heartbeat existed. Resolving runs on the fault ingest path, so the
        # missing method must not raise there, and the fallback is the closed
        # direction: UNKNOWN blocks, it does not permit.
        read = getattr(self.store, "get_workload_coverage_heartbeat", None)
        heartbeat = read(cluster_id) if read is not None else None
        if heartbeat is None:
            return False
        # The window is symmetric so ordinary clock skew between the watcher
        # and this process cannot silently retire coverage, while a stamp from
        # far in the future -- which the store's monotonic guard would then
        # keep -- stops counting instead of vouching for the cluster for ever.
        age = (observed_at - heartbeat.observed_at).total_seconds()
        return abs(age) <= self.coverage_freshness_seconds

    def resolve(
        self,
        cluster_id: str,
        node_id: str,
        observed_at: datetime,
        *,
        observations: list[AttemptObservation] | None = None,
        target_gpu_uuids: set[str] | None = None,
        pod_uid: str | None = None,
        container_id: str | None = None,
        host_pid: int | None = None,
        cgroup_path: str | None = None,
    ) -> WorkloadContext:
        """Resolve the workload on one node at one instant.

        ``observations`` lets a caller that resolves several nodes of the
        same cluster in one go read the cluster's observations once
        instead of once per node; the age filter below is what bounds
        staleness either way.

        ``workload_state`` fails closed: IDLE needs fresh coverage of the
        cluster that simply does not name this node. Coverage is either a fresh
        attempt observation (any attempt, any phase, within
        ``freshness_seconds``) or -- for a cluster that is running nothing at
        all, and therefore publishes no observation -- the watcher's own
        full-pass heartbeat within ``coverage_freshness_seconds``. A cluster
        with neither -- the watcher down, the feed stale -- is UNKNOWN, which
        the compilers treat as "someone may be using it" (design: monitoring
        loss is Unknown; ARCH-E2E-1 finding 2, completion-watcher F4).
        """

        if observations is None:
            observations = self.store.list_attempt_observations(cluster_id)
        covered = any(
            (observed_at - observation.observed_at).total_seconds()
            <= self.freshness_seconds
            for observation in observations
        ) or self._covered_by_heartbeat(cluster_id, observed_at)
        workloads: list[str] = []
        jobs: list[str] = []
        attempts: list[str] = []
        ranks: list[int] = []
        observed_gpu_uuids: list[str] = []
        profiles: list[str] = []
        for observation in observations:
            age = (observed_at - observation.observed_at).total_seconds()
            if age > self.max_age_seconds or observation.workload_phase not in {
                WorkloadPhase.PENDING,
                WorkloadPhase.RUNNING,
            }:
                continue
            matched = [
                item
                for item in observation.containers
                if item.node_id == node_id and not item.terminated
            ]
            if target_gpu_uuids:
                matched = [
                    item
                    for item in matched
                    if target_gpu_uuids.intersection(item.gpu_uuids)
                ]
            if pod_uid is not None:
                matched = [item for item in matched if item.pod_uid == pod_uid]
            if container_id is not None:
                normalized = container_id.rsplit("://", 1)[-1]
                matched = [
                    item
                    for item in matched
                    if item.container_id is not None
                    and item.container_id.rsplit("://", 1)[-1] == normalized
                ]
            if host_pid is not None:
                matched = [item for item in matched if item.host_pid == host_pid]
            if cgroup_path is not None:
                expected = cgroup_path.rstrip("/")
                matched = [
                    item
                    for item in matched
                    if item.cgroup_path is not None
                    and (
                        item.cgroup_path.rstrip("/") == expected
                        or item.cgroup_path.rstrip("/").startswith(expected + "/")
                        or expected.startswith(item.cgroup_path.rstrip("/") + "/")
                    )
                ]
            if not matched:
                continue
            jobs.append(observation.job_id)
            attempts.append(observation.attempt_id)
            workloads.extend(observation.workload_ids)
            ranks.extend(item.rank for item in matched)
            observed_gpu_uuids.extend(gpu for item in matched for gpu in item.gpu_uuids)
            profiles.append(observation.runtime_profile_version)
        if attempts:
            workload_state = "ACTIVE"
        elif covered:
            workload_state = "IDLE"
        else:
            workload_state = "UNKNOWN"
        return WorkloadContext(
            workload_state=workload_state,
            workload_ids=list(dict.fromkeys(workloads)),
            job_ids=list(dict.fromkeys(jobs)),
            attempt_ids=list(dict.fromkeys(attempts)),
            job_attempt_ids=list(dict.fromkeys(zip(jobs, attempts, strict=True))),
            runtime_profile_version=(
                profiles[0] if profiles and len(set(profiles)) == 1 else None
            ),
            ranks=sorted(set(ranks)),
            gpu_uuids=list(dict.fromkeys(observed_gpu_uuids)),
        )


class NvSwitchPortTopologyService:
    """Resolves an SXID port from fresh, trusted topology telemetry."""

    METRIC_NAME = "nvswitch_port_topology"
    _ACCESS_ONLY_PRODUCTS = {
        "H100",
        "H200",
        "H20",
    }

    def __init__(self, store, *, max_age_seconds: int = 300) -> None:
        self.store = store
        self.max_age_seconds = max_age_seconds

    @staticmethod
    def _switch_key(value: str | None) -> str:
        normalized = (value or "").strip().lower()
        prefix = "nvidia-nvswitch"
        return (
            normalized[len(prefix) :] if normalized.startswith(prefix) else normalized
        )

    @staticmethod
    def _port_key(value: str | None) -> str:
        normalized = (value or "").strip()
        return (
            str(int(normalized, 10)) if normalized.isdecimal() else normalized.lower()
        )

    def resolve(self, event):
        from gpu_fault.policy import SxidLinkScope

        if event.link_scope is not SxidLinkScope.UNKNOWN:
            return event
        if not event.switch_id or not event.port:
            return event

        candidates = []
        for metric in self.store.list_telemetry_metrics_latest(
            event.cluster_id, event.node_id
        ):
            labels = metric.labels
            age = (event.observed_at - metric.observed_at).total_seconds()
            if (
                metric.name != self.METRIC_NAME
                or metric.value != 1
                or age < -30
                or age > self.max_age_seconds
                or labels.get("trusted", "").lower() != "true"
                or (
                    event.switch_id
                    and self._switch_key(labels.get("switch_id"))
                    != self._switch_key(event.switch_id)
                )
                or (
                    event.port
                    and self._port_key(labels.get("port")) != self._port_key(event.port)
                )
            ):
                continue
            raw_scope = labels.get("link_scope", "").upper()
            if raw_scope not in {"ACCESS", "TRUNK"}:
                peer_type = labels.get("peer_type", "").upper()
                if peer_type == "GPU":
                    raw_scope = "ACCESS"
                elif peer_type in {"NVSWITCH", "SWITCH"}:
                    raw_scope = "TRUNK"
                else:
                    continue
            candidates.append((metric, SxidLinkScope(raw_scope)))

        scopes = {scope for _, scope in candidates}
        if len(scopes) == 1:
            scope = scopes.pop()
            updates = {
                "link_scope": scope,
                "link_scope_source": "TRUSTED_NVSWITCH_TOPOLOGY",
            }
            partitions = {
                metric.labels["fabric_partition"]
                for metric, _ in candidates
                if metric.labels.get("fabric_partition")
            }
            if len(partitions) == 1 and not event.fabric_partition:
                updates["fabric_partition"] = partitions.pop()
            gpu_uuids = {
                metric.labels["gpu_uuid"]
                for metric, candidate_scope in candidates
                if (
                    candidate_scope is SxidLinkScope.ACCESS
                    and metric.labels.get("gpu_uuid")
                )
            }
            if gpu_uuids:
                updates["participating_gpu_uuids"] = sorted(
                    set(event.participating_gpu_uuids) | gpu_uuids
                )
            return event.model_copy(update=updates)
        if candidates:
            return event

        # Third-generation HGX H100/H200 baseboards have no trunk links.
        # This product invariant is safe only when the event identifies a port.
        product = (event.product or "").strip().upper()
        if (
            event.port
            and event.classification.value == "FATAL"
            and any(
                product == item or product.startswith(f"NVIDIA {item}")
                for item in self._ACCESS_ONLY_PRODUCTS
            )
        ):
            return event.model_copy(
                update={
                    "link_scope": SxidLinkScope.ACCESS,
                    "link_scope_source": ("NVIDIA_PRODUCT_INVARIANT"),
                }
            )
        return event
