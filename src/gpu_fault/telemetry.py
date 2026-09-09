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
from gpu_fault.watcher import (
    AttemptObservation,
    WorkloadCoverageHeartbeat,
    WorkloadPhase,
)


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
    # How many of a cluster's newest attempt observations one ``resolve`` reads
    # from the store. Matching looks ``max_age_seconds`` (two minutes) back and
    # coverage is decided by the newest observation alone, so the week of
    # attempts the table retains has nothing to add; before this bound every
    # fault event materialised all of them and filtered in Python, O(events x
    # attempts) on a busy cluster (性能 2). The store keeps one row per attempt,
    # so 512 is 512 distinct attempts observed inside two minutes on one
    # cluster -- far above what a watcher scan reports, so the bound is a
    # ceiling on pathological churn rather than something the loop reaches.
    # Too small would drop in-window attempts (a missed ACTIVE); too large only
    # costs the rows it fetches, so it errs generously.
    OBSERVATION_SCAN_LIMIT = 512

    def __init__(
        self,
        store,
        *,
        max_age_seconds: int = 120,
        freshness_seconds: float = 600,
    ) -> None:
        if freshness_seconds <= 0:
            raise ValueError("freshness_seconds must be positive")
        if freshness_seconds < max_age_seconds:
            raise ValueError(
                "freshness_seconds must not be shorter than max_age_seconds: an "
                "observation young enough to match a node must also count as "
                "coverage of its cluster"
            )
        self.store = store
        self.max_age_seconds = max_age_seconds
        # How recently the cluster must have produced any attempt observation
        # for "nothing matches this node" to mean IDLE rather than UNKNOWN.
        self.freshness_seconds = freshness_seconds

    def observe(self, observation: AttemptObservation) -> None:
        self.store.save_attempt_observation(observation)

    def observe_coverage(self, heartbeat: WorkloadCoverageHeartbeat) -> None:
        self.store.save_workload_coverage(heartbeat)

    def _heartbeat_covers(self, cluster_id: str, observed_at: datetime) -> bool:
        heartbeat = self.store.get_workload_coverage(cluster_id)
        return (
            heartbeat is not None
            and (observed_at - heartbeat.scanned_at).total_seconds()
            <= self.freshness_seconds
        )

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
        staleness either way. Such a list is taken as given, in any order.

        Without it the store is read newest-first and bounded to
        ``OBSERVATION_SCAN_LIMIT`` rows. That is safe because the answer only
        depends on the freshest rows: ``covered`` needs one observation within
        ``freshness_seconds``, which the newest either is or nothing is; and a
        match needs an observation within ``max_age_seconds``, so the walk
        stops at the first one older than that -- every later row is older
        still (``newest_first`` orders by ``observed_at`` in every store).
        Observations slightly *ahead* of ``observed_at`` (clock skew between
        watcher and collector) have a negative age, sort first, and match as
        they always did.

        ``workload_state`` fails closed: IDLE needs fresh coverage of the
        cluster -- an attempt observation (any attempt, any phase, observed
        within ``freshness_seconds``) that simply does not name this node, or
        the watcher's scan heartbeat, which is how a cluster with no managed
        attempt at all proves the feed is alive. A cluster with neither --
        the watcher down, the cluster silent, a stale feed -- is UNKNOWN,
        which the compilers treat as "someone may be using it" (design:
        monitoring loss is Unknown; ARCH-E2E-1 finding 2).
        """

        # A caller-supplied list carries no ordering promise, so it is scanned
        # whole; only the store read below is known to be newest-first.
        newest_first = observations is None
        if observations is None:
            observations = [
                state.observation
                for state in self.store.list_attempt_observation_states(
                    cluster_id,
                    limit=self.OBSERVATION_SCAN_LIMIT,
                    newest_first=True,
                )
            ]
        if newest_first:
            covered = bool(observations) and (
                (observed_at - observations[0].observed_at).total_seconds()
                <= self.freshness_seconds
            )
        else:
            covered = any(
                (observed_at - observation.observed_at).total_seconds()
                <= self.freshness_seconds
                for observation in observations
            )
        covered = covered or self._heartbeat_covers(cluster_id, observed_at)
        workloads: list[str] = []
        jobs: list[str] = []
        attempts: list[str] = []
        ranks: list[int] = []
        observed_gpu_uuids: list[str] = []
        profiles: list[str] = []
        for observation in observations:
            age = (observed_at - observation.observed_at).total_seconds()
            if age > self.max_age_seconds:
                if newest_first:
                    # Newest-first: everything after this row is older still.
                    break
                continue
            if observation.workload_phase not in {
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
