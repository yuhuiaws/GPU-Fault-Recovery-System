"""Records and store views for driving the shared-storage correlator directly.

The correlator reads per-node incidents, the latest
``shared_filesystem_unavailable`` rows and health-signal states, and writes
domain records, back-references and one notification. These builders write
exactly those records into an ``InMemoryStore`` with the shapes host
ingestion gives them (an activation's event id ends in
``-shared_filesystem_unavailable-<mount>``), and the two store views stand in
for what another replica does between this look's read and its write.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any

from gpu_fault.host_health import (
    HostMetricSample,
    HostTelemetryBatch,
    NodeHealthCategory,
    NodeHealthFinding,
    event_id_component,
)
from gpu_fault.models import (
    FaultIncident,
    IncidentState,
    MarkerScope,
    NodeMarker,
    RecoveryAction,
    Severity,
)
from gpu_fault.shared_storage_failure_domain import (
    SHARED_FILESYSTEM_METRIC,
    SHARED_STORAGE_FAILURE_DOMAIN_EVENT_TYPE,
    SHARED_STORAGE_FAILURE_DOMAIN_POLICY_SOURCE,
    SHARED_STORAGE_FAILURE_DOMAIN_POLICY_VERSION,
    is_shared_storage_failure_domain,
)
from gpu_fault.store import InMemoryStore, NotFoundError
from gpu_fault.telemetry_models import TelemetryMetricLatest

CLUSTER = "cluster-a"
EXPORT = "10.0.0.5:/export"
OTHER_EXPORT = "10.0.0.6:/other"
MOUNT = "/mnt/shared"
#: Node clock of the first activation in every case.
T0 = datetime(2026, 11, 1, 8, 0, tzinfo=timezone.utc)


def activation_event_id(batch_id: str, device: str = MOUNT) -> str:
    """The event id host ingestion mints for an activation on ``device``."""

    return f"{batch_id}-{SHARED_FILESYSTEM_METRIC}-{event_id_component(device)}"


def node_incident(
    incident_id: str,
    node_id: str,
    *,
    batch_id: str | None = None,
    event_id: str | None = None,
    event_type: str = "NODE_HEALTH",
    node_ids: list[str] | None = None,
    state: IncidentState = IncidentState.DETECTED,
    effective_action: RecoveryAction | None = RecoveryAction.RUN_DIAGNOSTICS,
    official_action: str | None = None,
    workflow_request_id: str | None = None,
) -> FaultIncident:
    """A per-node incident; by default the one an activation minted."""

    return FaultIncident(
        incident_id=incident_id,
        event_id=event_id or activation_event_id(batch_id or f"{node_id}-1"),
        event_type=event_type,
        cluster_id=CLUSTER,
        node_ids=node_ids if node_ids is not None else [node_id],
        policy_version="site-node-health-policy/v1",
        policy_source="SITE_NODE_HEALTH",
        effective_action=effective_action,
        official_action=official_action,
        state=state,
        workflow_request_id=workflow_request_id,
    )


def domain_record(
    incident_id: str,
    node_ids: list[str],
    *,
    export: str = EXPORT,
    state: IncidentState = IncidentState.DETECTED,
) -> FaultIncident:
    """An open failure-domain record, as the correlator writes one."""

    return FaultIncident(
        incident_id=incident_id,
        event_id=f"event-{incident_id}",
        event_type=SHARED_STORAGE_FAILURE_DOMAIN_EVENT_TYPE,
        event_source=SHARED_FILESYSTEM_METRIC,
        cluster_id=CLUSTER,
        node_ids=sorted(node_ids),
        policy_version=SHARED_STORAGE_FAILURE_DOMAIN_POLICY_VERSION,
        policy_source=SHARED_STORAGE_FAILURE_DOMAIN_POLICY_SOURCE,
        policy_reference=f"shared-filesystem:{export}",
        state=state,
        reasons=[f"common failure domain: shared filesystem {export} unavailable"],
    )


def finding(
    node_id: str,
    observed_at: datetime,
    *,
    batch_id: str | None = None,
    metric: str = SHARED_FILESYSTEM_METRIC,
    export: str | None = EXPORT,
    device: str = MOUNT,
) -> NodeHealthFinding:
    """The CRITICAL activation a node's ``shared_filesystem_unavailable`` mints."""

    event_id = activation_event_id(batch_id or f"{node_id}-1", device)
    return NodeHealthFinding(
        finding_id=f"finding-{event_id}",
        event_id=event_id,
        cluster_id=CLUSTER,
        node_id=node_id,
        observed_at=observed_at,
        category=NodeHealthCategory.STORAGE,
        severity=Severity.CRITICAL,
        reason=f"shared filesystem {device} unavailable",
        recommended_action=RecoveryAction.RUN_DIAGNOSTICS,
        metric_name=metric,
        value=1,
        device=device,
        diagnostic_parameters={} if export is None else {"source": export},
    )


def reading(
    store: InMemoryStore,
    node_id: str,
    value: float,
    observed_at: datetime,
    *,
    export: str = EXPORT,
    device: str = MOUNT,
) -> None:
    """The node's latest gauge row for ``export``, as host ingestion stores it."""

    store.observe_telemetry_metric(
        TelemetryMetricLatest(
            cluster_id=CLUSTER,
            node_id=node_id,
            observed_at=observed_at,
            name=SHARED_FILESYSTEM_METRIC,
            value=value,
            device=device,
            labels={"source": export, "fstype": "nfs4"},
        )
    )


def batch(
    node_id: str,
    value: float,
    observed_at: datetime,
    *,
    export: str = EXPORT,
    received_at: datetime | None = None,
    device: str = MOUNT,
) -> HostTelemetryBatch:
    return HostTelemetryBatch(
        batch_id=f"batch-{node_id}-{observed_at.timestamp():.0f}",
        cluster_id=CLUSTER,
        node_id=node_id,
        observed_at=observed_at,
        received_at=received_at,
        samples=[
            HostMetricSample(
                name=SHARED_FILESYSTEM_METRIC,
                value=value,
                device=device,
                labels={"source": export, "fstype": "nfs4"},
            )
        ],
    )


def routed_marker(
    event_id: str, incident_id: str, node_id: str, observed_at: datetime
) -> NodeMarker:
    """The marker of a finding ingestion routed to ``incident_id``."""

    return NodeMarker(
        marker_id=f"marker-{event_id}",
        source="node-health",
        cluster_id=CLUSTER,
        incident_id=incident_id,
        observed_at=observed_at,
        expires_at=observed_at + timedelta(hours=1),
        scope=MarkerScope(node_ids=[node_id]),
        severity=Severity.CRITICAL,
        recommended_action=RecoveryAction.RUN_DIAGNOSTICS,
        mapping_version="site-node-health-policy/v1",
        event_source=SHARED_FILESYSTEM_METRIC,
    )


def domain_records(store: InMemoryStore) -> list[FaultIncident]:
    return sorted(
        (
            incident
            for incident in store.list_incidents_by_state(CLUSTER, set(IncidentState))
            if is_shared_storage_failure_domain(incident)
        ),
        key=lambda item: item.created_at,
    )


class SiblingReplicaStore:
    """The store while a sibling replica writes the record this look writes.

    ``before_save(incident_id, write)`` arms the sibling: before this look's
    next ``times`` writes of that incident reach the store, the sibling's
    ``write(current)`` lands first, so the store's own compare-and-set refuses
    this look's write as stale. With ``then_vanish`` the record is gone for
    this look afterwards (an archive between its write and its re-read).
    """

    def __init__(self, store: InMemoryStore) -> None:
        self.store = store
        self.sibling_writes = 0
        self._armed: dict[
            str, tuple[Callable[[FaultIncident], FaultIncident], int]
        ] = {}
        self._vanish: set[str] = set()
        self._vanished: set[str] = set()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.store, name)

    def before_save(
        self,
        incident_id: str,
        write: Callable[[FaultIncident], FaultIncident],
        *,
        times: int = 1,
        then_vanish: bool = False,
    ) -> None:
        self._armed[incident_id] = (write, times)
        if then_vanish:
            self._vanish.add(incident_id)

    def save_incident(
        self, incident: FaultIncident, *, expected: FaultIncident | None = None
    ) -> None:
        armed = self._armed.get(incident.incident_id)
        if armed is not None:
            write, remaining = armed
            current = self.store.get_incident(incident.incident_id)
            self.store.save_incident(write(current), expected=current)
            self.sibling_writes += 1
            if remaining > 1:
                self._armed[incident.incident_id] = (write, remaining - 1)
            else:
                del self._armed[incident.incident_id]
            if incident.incident_id in self._vanish:
                self._vanished.add(incident.incident_id)
        self.store.save_incident(incident, expected=expected)

    def get_incident(self, incident_id: str) -> FaultIncident:
        if incident_id in self._vanished:
            raise NotFoundError(incident_id)
        incident: FaultIncident = self.store.get_incident(incident_id)
        return incident


class EarlierListingStore:
    """The store as a look saw it before a concurrent look's record committed:
    listings miss every failure-domain record, point reads see them."""

    def __init__(self, store: InMemoryStore) -> None:
        self.store = store

    def __getattr__(self, name: str) -> Any:
        return getattr(self.store, name)

    def list_incidents_by_state(
        self, cluster_id: str, states: Any, **kwargs: Any
    ) -> list[FaultIncident]:
        incidents: list[FaultIncident] = self.store.list_incidents_by_state(
            cluster_id, states, **kwargs
        )
        return [
            item for item in incidents if not is_shared_storage_failure_domain(item)
        ]
