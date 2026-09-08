from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from enum import StrEnum

from pydantic import Field, field_validator

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

LOGGER = logging.getLogger(__name__)


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
    pass over every namespace of the cluster at ``observed_at`` and still saw
    ``watched_attempts`` attempts running across ``watched_pods`` running
    managed Pods. Zero and zero is the only case that says anything -- see
    ``WorkloadTopologyService._covered_by_heartbeat``.

    ``resource_version`` is the Kubernetes list revision the pass started from
    and ``watcher_instance`` names the process, so an operator can tell a
    heartbeat from a watcher that has since been replaced from a current one.
    Neither is read by the resolver: coverage is decided by ``observed_at`` and
    the two counts, which is what keeps a rolled-back watcher from claiming
    coverage with a stale clock.
    """

    cluster_id: str
    observed_at: datetime
    #: Managed Pods and attempts the pass still saw *running*. Only a pass that
    #: saw nothing running may be read as coverage of an idle cluster, so both
    #: are part of the statement and both are checked by the resolver.
    watched_pods: int = Field(default=0, ge=0)
    watched_attempts: int = Field(default=0, ge=0)
    resource_version: str | None = None
    watcher_instance: str

    @field_validator("observed_at")
    @classmethod
    def _require_aware_observed_at(cls, value: datetime) -> datetime:
        """Refuse a naive stamp at the boundary and store UTC.

        Everything that reads this row compares it against an aware ``now``,
        and that comparison raises ``TypeError`` on a naive value -- on the
        fault ingest path, for every cluster, until the row is replaced. A
        naive stamp is also not a time: guessing UTC for a watcher that meant
        UTC+8 would either vouch for a cluster eight hours after it went busy
        or freeze the row's monotonic guard for eight hours. So it is rejected
        here, where the answer is a 422 the watcher's own metrics show.
        """

        if value.tzinfo is None:
            raise ValueError("coverage heartbeat observed_at must include a timezone")
        return value.astimezone(timezone.utc)


#: Clusters already reported as carrying an unusable stored heartbeat. Coverage
#: is read on the fault ingest path, so the report is logged once per cluster
#: instead of once per fault.
_UNUSABLE_COVERAGE_HEARTBEATS: set[str] = set()


def warn_unusable_coverage_heartbeat(cluster_id: str, reason: str) -> None:
    """Report once that a stored coverage row cannot be used at all."""

    if cluster_id in _UNUSABLE_COVERAGE_HEARTBEATS:
        return
    _UNUSABLE_COVERAGE_HEARTBEATS.add(cluster_id)
    LOGGER.warning(
        "ignoring the stored coverage heartbeat of cluster %s: %s; the cluster "
        "reads UNKNOWN until a watcher replaces the row",
        cluster_id,
        reason,
    )


def note_usable_coverage_heartbeat(cluster_id: str) -> None:
    """Arm the report again: this cluster's row could be read.

    The suppression above is "once per cluster", not "once per process". A
    watcher that writes an unusable row is a watcher whose whole cluster reads
    UNKNOWN and every node-mutating plan on it is BLOCKED, so a second
    occurrence -- the rolled-back watcher that comes back -- has to be visible
    even when the first one happened days ago.
    """

    _UNUSABLE_COVERAGE_HEARTBEATS.discard(cluster_id)


def coverage_heartbeat_supersedes(
    heartbeat: WorkloadCoverageHeartbeat,
    previous: WorkloadCoverageHeartbeat,
) -> bool:
    """Whether ``heartbeat`` may replace the stored ``previous`` row.

    The newest ``observed_at`` wins outright, so the last in-flight heartbeat
    of a watcher a rollout has already replaced cannot age coverage backwards.

    A stored stamp the new one cannot be compared against -- a naive value from
    an older release, before the model rejected them -- is *replaced* rather
    than defended: the guard exists to keep coverage from moving backwards, and
    a row nothing can compare against is not coverage at all. Refusing here
    would poison the cluster's single row for ever.
    """

    try:
        return heartbeat.observed_at > previous.observed_at
    except TypeError:
        warn_unusable_coverage_heartbeat(
            previous.cluster_id,
            "its observed_at cannot be compared with an aware stamp",
        )
        return True


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
        # Heartbeats the store refused because a newer row already stood. A few
        # are ordinary (a watcher rollout overlaps); a rate near one per pass
        # means two watchers are publishing for one cluster, or one of them has
        # a clock far in the future and is holding the row.
        self.coverage_heartbeats_rejected_total = 0

    def observe(self, observation: AttemptObservation) -> None:
        self.store.save_attempt_observation(observation)

    def observe_coverage(self, heartbeat: WorkloadCoverageHeartbeat) -> bool:
        """Record a watcher's full-pass heartbeat; ``False`` when it is stale.

        One row per cluster: the store keeps the newest ``observed_at`` and
        refuses an older one, so the last in-flight heartbeat of a watcher a
        rollout has already replaced cannot move coverage backwards.
        """

        accepted = bool(self.store.save_workload_coverage_heartbeat(heartbeat))
        if not accepted:
            self.coverage_heartbeats_rejected_total += 1
            LOGGER.warning(
                "refused a coverage heartbeat behind the stored row: cluster=%s "
                "watcher=%s observed_at=%s",
                heartbeat.cluster_id,
                heartbeat.watcher_instance,
                heartbeat.observed_at.isoformat(),
            )
        return accepted

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
        heartbeat: WorkloadCoverageHeartbeat | None = (
            read(cluster_id) if read is not None else None
        )
        if heartbeat is None:
            return False
        # Coverage of an *idle* cluster is the only thing a heartbeat can prove.
        # A pass that saw a running Pod or a live attempt saw workload the
        # control plane should be hearing about in observations, and if it is
        # not hearing about it -- a swallowed per-attempt failure, a dropped
        # observation POST, an observation shed under load while this
        # reserved-capacity path still landed -- then "no observation" means the
        # feed is lossy, not that the cluster is idle. Reading such a heartbeat
        # as coverage would answer IDLE for a node that is training, and IDLE
        # skips CHECKPOINT and STOP_WORKLOADS before a reboot.
        if heartbeat.watched_attempts or heartbeat.watched_pods:
            return False
        # The window is symmetric so ordinary clock skew between the watcher
        # and this process cannot silently retire coverage, while a stamp from
        # far in the future -- which the store's monotonic guard would then
        # keep -- stops counting instead of vouching for the cluster for ever.
        try:
            age = (observed_at - heartbeat.observed_at).total_seconds()
        except TypeError:
            # A row written before the model rejected naive stamps. Absent
            # coverage is the closed answer, and the writer replaces the row.
            warn_unusable_coverage_heartbeat(
                cluster_id,
                "its observed_at carries no timezone",
            )
            return False
        # The row read cleanly, so an unusable one that replaces it later is a
        # new fact and gets reported again.
        note_usable_coverage_heartbeat(cluster_id)
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
        full-pass heartbeat within ``coverage_freshness_seconds`` that saw
        nothing running at all. A cluster
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
