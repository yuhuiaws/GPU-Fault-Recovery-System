"""One shared filesystem failing on several nodes is one failure domain.

``shared_filesystem_unavailable`` is graded per node (``host_health``): every
node that loses the export mints its own CRITICAL ``RUN_DIAGNOSTICS`` finding,
incident and ``VALIDATE_HOST`` chain, so a dead NFS server or a Lustre MDS in
recovery looked like N independent node problems -- N diagnostic chains and
nothing anywhere saying "common failure domain: shared filesystem X, nodes A,
B" (GF-GENERAL-LIVE-005 ``shared_failure_domain``).

This module is the additive, control-plane-only correlation on top of that:

* **Identity.** Two nodes share a failure domain when they report the *same
  export* unavailable. The export identity comes from the collector's sample
  labels (``source`` -- the mountinfo mount source such as
  ``10.0.0.5:/export`` or ``fs-0abc@tcp:/fsname`` -- or ``server`` +
  ``export``). The mount point is deliberately **not** an identity: it is a
  per-node path, and two nodes may mount different exports at ``/mnt/data``.
  A sample without an identity is UNKNOWN and is never correlated.
* **Record.** A cluster-scoped :class:`FaultIncident` with
  ``event_type=SHARED_STORAGE_FAILURE_DOMAIN``, ``node_ids`` = the member
  nodes, ``policy_reference=shared-filesystem:<identity>`` and reasons naming
  the export, the nodes and every member's per-node incident. It has no
  workflow and no action (``effective_action=None``): it is a record, the
  per-node chains stay the authority for each node and are not changed in any
  way (diagnostic-only, no isolation, no STOP). Every member's per-node
  incident gets one reason line pointing back at the domain record, so a
  later per-node notification carries the reference.
* **Window.** A domain forms when at least two nodes report the same identity
  unavailable within :data:`SHARED_STORAGE_FAILURE_DOMAIN_WINDOW_SECONDS` of
  each other (node clocks); a node whose activation comes later joins the
  open record. One notification per domain record.
* **Clearing.** The record closes (``RECOVERED``) when every member's latest
  reading of the identity is "available" -- positive evidence from each node.
  A member that stops reporting while its last reading was "unavailable" is
  silent, not healthy: once no member still reports a fresh failure the record
  moves to ``ESCALATED`` (the operator owns the close; the departed-node
  cleanup and ``incident-close`` can take it from there), and a later
  "available" from the silent node still closes it.

Store shape: existing records only (incidents, notifications); no new record
kind, no schema change.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from gpu_fault.host_health import (
    HostMetricSample,
    HostTelemetryBatch,
    NodeHealthFinding,
    event_id_component,
)
from gpu_fault.models import (
    AdvisoryNotification,
    FaultIncident,
    IncidentState,
    bounded_reasons,
)
from gpu_fault.orchestration.families.identity import derived_record_id
from gpu_fault.store import NotFoundError
from gpu_fault.store.shared.errors import StaleWriteError
from gpu_fault.store.shared.health_signals import health_signal_key
from gpu_fault.telemetry_models import TelemetryMetricLatest

LOGGER = logging.getLogger(__name__)

SHARED_FILESYSTEM_METRIC = "shared_filesystem_unavailable"
SHARED_STORAGE_FAILURE_DOMAIN_EVENT_TYPE = "SHARED_STORAGE_FAILURE_DOMAIN"
SHARED_STORAGE_FAILURE_DOMAIN_POLICY_SOURCE = "SITE_SHARED_STORAGE_FAILURE_DOMAIN"
SHARED_STORAGE_FAILURE_DOMAIN_POLICY_VERSION = "site-shared-storage-failure-domain/v1"
SHARED_STORAGE_FAILURE_DOMAIN_NOTIFICATION_CATEGORY = "SHARED_STORAGE_FAILURE_DOMAIN"
#: Two nodes' readings of one export count as the same outage when they lie
#: within this many seconds of each other; a member's "unavailable" reading
#: older than this against the current observation is silence, not a fresh
#: failure. The host collector ticks every 15 s and a bounded ``statvfs``
#: adds seconds, so 10 minutes is tens of ticks of slack and still bounded.
SHARED_STORAGE_FAILURE_DOMAIN_WINDOW_SECONDS = 600.0
#: The states an open domain record can be in: DETECTED while a member still
#: reports a fresh failure, ESCALATED once only silent members are left.
OPEN_DOMAIN_STATES = frozenset({IncidentState.DETECTED, IncidentState.ESCALATED})
#: Per-node incident states that still speak for a node when its peers are
#: discovered; RECOVERED ones have closed and say nothing about the present.
_OPEN_NODE_INCIDENT_STATES = frozenset(set(IncidentState) - {IncidentState.RECOVERED})
_IDENTITY_LABELS = ("source", "mount_source")
_WRITE_ATTEMPTS = 3
_RECORD_FAMILY = "shared-storage-failure-domain"


def shared_filesystem_identity(labels: Mapping[str, Any]) -> str | None:
    """The export identity of a shared-filesystem sample, or ``None``.

    ``source`` (or ``mount_source``) is the mount source as the host's
    mountinfo spells it; ``server`` + ``export`` is accepted as the split form.
    Whitespace is stripped and a trailing ``/`` on the export path is dropped
    (``host:/export/`` and ``host:/export`` are one export); nothing else is
    normalised, since export paths and Lustre fsnames are case-sensitive.
    Missing or blank identity is ``None`` -- never the mount point.
    """

    for key in _IDENTITY_LABELS:
        value = _clean_identity(labels.get(key))
        if value:
            return value
    server = _clean_identity(labels.get("server"))
    export = _clean_identity(labels.get("export"))
    if server and export:
        return f"{server}:{export}"
    return None


def _clean_identity(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    while len(text) > 1 and text.endswith("/") and not text.endswith(":/"):
        text = text[:-1]
    return text or None


def is_shared_storage_failure_domain(incident: FaultIncident) -> bool:
    return incident.event_type == SHARED_STORAGE_FAILURE_DOMAIN_EVENT_TYPE


def _policy_reference(identity: str) -> str:
    return f"shared-filesystem:{identity}"


@dataclass(frozen=True)
class _Member:
    node_id: str
    device: str | None
    incident_id: str | None
    observed_at: datetime


class SharedStorageFailureDomainCorrelator:
    """Correlates per-node ``shared_filesystem_unavailable`` signals by export.

    ``send_notification`` is the advisory service's ``send``; it is called
    once per domain record created.
    """

    def __init__(
        self,
        store: Any,
        send_notification: Callable[[str], Any],
        *,
        window_seconds: float = SHARED_STORAGE_FAILURE_DOMAIN_WINDOW_SECONDS,
    ) -> None:
        if window_seconds <= 0:
            raise ValueError(
                "the shared-storage failure-domain window must be positive"
            )
        self.store = store
        self.send_notification = send_notification
        self.window = timedelta(seconds=window_seconds)

    # ------------------------------------------------------------ entry points

    def observe_finding(
        self, finding: NodeHealthFinding, incident: FaultIncident
    ) -> tuple[FaultIncident | None, str | None]:
        """A per-node signal just activated: correlate it with its peers.

        Returns the domain record the node now belongs to (``None`` when the
        node stands alone or carries no identity) and the id of the one
        notification a newly created record sent.
        """

        if finding.metric_name != SHARED_FILESYSTEM_METRIC:
            return None, None
        identity = shared_filesystem_identity(finding.diagnostic_parameters)
        if identity is None:
            return None, None
        member = _Member(
            finding.node_id, finding.device, incident.incident_id, finding.observed_at
        )
        return self._correlate(finding.cluster_id, identity, member)

    def observe_batch(
        self, batch: HostTelemetryBatch, findings: Sequence[NodeHealthFinding]
    ) -> list[str]:
        """Every host batch: clear on "available", re-check on "unavailable".

        A reading of 0 settles the open records naming this node; a reading of
        1 that minted no finding (the signal was already active) is the safety
        net for an activation whose peers were not yet visible -- bounded to
        the correlation window after the activation.
        """

        notifications: list[str] = []
        for sample in batch.samples:
            if sample.name != SHARED_FILESYSTEM_METRIC:
                continue
            identity = shared_filesystem_identity(sample.labels)
            if identity is None:
                continue
            if _available(sample.value):
                self._settle(
                    batch.cluster_id, batch.node_id, identity, batch.observed_at
                )
                continue
            if not _failing(sample.value):
                continue  # neither 0 nor 1: no evidence either way
            if any(
                item.metric_name == SHARED_FILESYSTEM_METRIC
                and item.device == sample.device
                for item in findings
            ):
                continue  # the activation path (observe_finding) handled it
            notification_id = self._recheck(batch, sample, identity)
            if notification_id is not None:
                notifications.append(notification_id)
        return notifications

    # ------------------------------------------------------------- correlation

    def _correlate(
        self, cluster_id: str, identity: str, member: _Member
    ) -> tuple[FaultIncident | None, str | None]:
        domain = self._open_domain(cluster_id, identity, member.node_id)
        if domain is not None:
            if member.node_id not in domain.node_ids:
                domain = self._join(domain, identity, member)
            self._reference_domain(domain, identity, member)
            return domain, None
        peers = self._fresh_failing_peers(cluster_id, identity, member)
        if not peers:
            return None, None
        members = [member, *peers]
        domain = self._create(cluster_id, identity, members)
        for item in members:
            self._reference_domain(domain, identity, item)
        return domain, self._notify(domain, identity, members)

    def _open_domain(
        self, cluster_id: str, identity: str, *node_ids: str
    ) -> FaultIncident | None:
        """The open domain record for ``identity``: the one naming any of
        ``node_ids`` (an indexed read), else the newest open one in the
        cluster."""

        reference = _policy_reference(identity)
        filters: list[set[str] | None] = [set(node_ids)] if node_ids else []
        filters.append(None)
        for node_filter in filters:
            incidents: list[FaultIncident] = self.store.list_incidents_by_state(
                cluster_id, OPEN_DOMAIN_STATES, node_ids=node_filter
            )
            for incident in incidents:
                if (
                    is_shared_storage_failure_domain(incident)
                    and incident.policy_reference == reference
                ):
                    return incident
        return None

    def _fresh_failing_peers(
        self, cluster_id: str, identity: str, member: _Member
    ) -> list[_Member]:
        """Other nodes whose latest reading of ``identity`` is a fresh failure.

        Candidates are the nodes with an open per-node incident for the metric
        (their VALIDATE_HOST holds while the export stays unavailable) and the
        members of open domain records; each candidate's latest telemetry row
        for the identity decides. A node whose reading is older than the
        window, or reads available, is not a peer.
        """

        candidates: dict[str, str | None] = {}
        for incident in self.store.list_incidents_by_state(
            cluster_id, _OPEN_NODE_INCIDENT_STATES
        ):
            if is_shared_storage_failure_domain(incident):
                for node_id in incident.node_ids:
                    candidates.setdefault(node_id, None)
            elif _is_shared_filesystem_incident(incident):
                for node_id in incident.node_ids:
                    # Newest first: the first incident seen for a node is the
                    # one its current activation opened.
                    candidates.setdefault(node_id, incident.incident_id)
        candidates.pop(member.node_id, None)
        peers: list[_Member] = []
        for node_id in sorted(candidates):
            reading = self._latest_reading(cluster_id, node_id, identity)
            if reading is None or not _failing(reading.value):
                continue
            if not self._fresh(reading.observed_at, member.observed_at):
                continue
            incident_id = candidates[node_id] or self._open_node_incident_id(
                cluster_id, node_id, reading.device
            )
            peers.append(
                _Member(node_id, reading.device, incident_id, reading.observed_at)
            )
        return peers

    def _recheck(
        self, batch: HostTelemetryBatch, sample: HostMetricSample, identity: str
    ) -> str | None:
        state = self.store.get_health_signal_state(
            health_signal_key(
                batch.cluster_id, batch.node_id, SHARED_FILESYSTEM_METRIC, sample.device
            )
        )
        if state is None or not state.active or state.active_since is None:
            return None
        now = batch.received_at or datetime.now(timezone.utc)
        if now - state.active_since > self.window:
            return None  # the correlation window of this activation has closed
        domain = self._open_domain(batch.cluster_id, identity, batch.node_id)
        if domain is not None and batch.node_id in domain.node_ids:
            return None  # already a member
        member = _Member(
            batch.node_id,
            sample.device,
            self._open_node_incident_id(batch.cluster_id, batch.node_id, sample.device),
            batch.observed_at,
        )
        _, notification_id = self._correlate(batch.cluster_id, identity, member)
        return notification_id

    # ------------------------------------------------------------------ clears

    def _settle(
        self, cluster_id: str, node_id: str, identity: str, observed_at: datetime
    ) -> None:
        reference = _policy_reference(identity)
        for incident in self.store.list_incidents_by_state(
            cluster_id, OPEN_DOMAIN_STATES, node_ids={node_id}
        ):
            if (
                is_shared_storage_failure_domain(incident)
                and incident.policy_reference == reference
            ):
                self._evaluate(incident, identity, observed_at)

    def _evaluate(self, domain: FaultIncident, identity: str, now: datetime) -> None:
        """Close, escalate or keep the record from its members' latest readings."""

        cleared: list[str] = []
        silent: list[tuple[str, datetime | None]] = []
        for node_id in domain.node_ids:
            reading = self._latest_reading(domain.cluster_id, node_id, identity)
            if reading is None:
                silent.append((node_id, None))
            elif _available(reading.value):
                cleared.append(node_id)
            elif _failing(reading.value) and self._fresh(reading.observed_at, now):
                return  # a member still reports the export unavailable
            else:
                # Stale failure, or a reading that is neither 0 nor 1 (never
                # read as health): the node has not said "available".
                silent.append((node_id, reading.observed_at))
        if not silent:
            self._write(
                domain,
                state=IncidentState.RECOVERED,
                reason=(
                    "cleared: every member node reports shared filesystem "
                    f"{identity} available again ({', '.join(cleared)})"
                ),
            )
        elif domain.state is not IncidentState.ESCALATED:
            silent_text = "; ".join(
                f"{node_id} has not reported since {since.isoformat()} "
                "(last reading unavailable)"
                if since is not None
                else f"{node_id} has no reading of the export"
                for node_id, since in silent
            )
            self._write(
                domain,
                state=IncidentState.ESCALATED,
                reason=(
                    f"{', '.join(cleared) or 'no member'} reports shared filesystem "
                    f"{identity} available again; {silent_text}: silence is not "
                    "recovery, the operator owns the close"
                ),
            )

    # --------------------------------------------------------------- records

    def _create(
        self, cluster_id: str, identity: str, members: Sequence[_Member]
    ) -> FaultIncident:
        nodes = sorted({member.node_id for member in members})
        incident_ids = sorted(
            {member.incident_id for member in members if member.incident_id}
        )
        parts = [cluster_id, identity, *nodes, *incident_ids]
        incident_id = derived_record_id("incident", _RECORD_FAMILY, *parts)
        existing = self._incident(incident_id)
        if existing is not None:
            if existing.state in OPEN_DOMAIN_STATES:
                return existing
            # A closed episode built from the very same per-node incidents
            # (both still open from the last outage): a new record, not an
            # overwrite of the closed one.
            parts.append(min(member.observed_at for member in members).isoformat())
            incident_id = derived_record_id("incident", _RECORD_FAMILY, *parts)
        now = datetime.now(timezone.utc)
        reasons = [
            f"common failure domain: shared filesystem {identity} unavailable on "
            f"{len(nodes)} nodes ({', '.join(nodes)}); a shared-storage outage, "
            "not a per-node hardware fault: every node keeps its own "
            "diagnostic-only chain, no node is isolated or stopped for it",
            *(self._member_line(member) for member in sorted(members, key=_node)),
        ]
        incident = FaultIncident(
            incident_id=incident_id,
            event_id=derived_record_id(_RECORD_FAMILY, _RECORD_FAMILY, *parts),
            event_type=SHARED_STORAGE_FAILURE_DOMAIN_EVENT_TYPE,
            event_source=SHARED_FILESYSTEM_METRIC,
            cluster_id=cluster_id,
            node_ids=nodes,
            policy_version=SHARED_STORAGE_FAILURE_DOMAIN_POLICY_VERSION,
            policy_source=SHARED_STORAGE_FAILURE_DOMAIN_POLICY_SOURCE,
            policy_reference=_policy_reference(identity),
            official_action=None,
            effective_action=None,
            state=IncidentState.DETECTED,
            reasons=bounded_reasons(reasons),
            created_at=now,
            updated_at=now,
        )
        self.store.save_incident(incident)
        LOGGER.info(
            "shared filesystem %s unavailable on %s: failure domain %s recorded",
            identity,
            ",".join(nodes),
            incident_id,
        )
        return incident

    def _join(
        self, domain: FaultIncident, identity: str, member: _Member
    ) -> FaultIncident:
        def update(current: FaultIncident) -> FaultIncident | None:
            if member.node_id in current.node_ids:
                return None
            return current.model_copy(
                update={
                    "node_ids": sorted({*current.node_ids, member.node_id}),
                    "reasons": bounded_reasons(
                        [
                            *current.reasons,
                            f"{member.node_id} joined: shared filesystem {identity} "
                            f"unavailable there as well at "
                            f"{member.observed_at.isoformat()}",
                            self._member_line(member),
                        ]
                    ),
                    "updated_at": datetime.now(timezone.utc),
                }
            )

        return self._compare_and_set(domain, update) or domain

    def _reference_domain(
        self, domain: FaultIncident, identity: str, member: _Member
    ) -> None:
        """One reason line on the member's per-node incident, pointing back."""

        if member.incident_id is None:
            return
        incident = self._incident(member.incident_id)
        if incident is None:
            return
        prefix = f"common failure domain {domain.incident_id}:"
        others = [node for node in domain.node_ids if node != member.node_id]

        def update(current: FaultIncident) -> FaultIncident | None:
            if any(reason.startswith(prefix) for reason in current.reasons):
                return None
            return current.model_copy(
                update={
                    "reasons": bounded_reasons(
                        [
                            *current.reasons,
                            f"{prefix} shared filesystem {identity} is unavailable "
                            f"on {', '.join(others) or 'other nodes'} as well "
                            "(shared-storage outage, not a node fault); the "
                            "diagnostic-only handling of this node is unchanged",
                        ]
                    ),
                    "updated_at": datetime.now(timezone.utc),
                }
            )

        self._compare_and_set(incident, update)

    def _write(
        self, domain: FaultIncident, *, state: IncidentState, reason: str
    ) -> None:
        def update(current: FaultIncident) -> FaultIncident | None:
            if current.state not in OPEN_DOMAIN_STATES or current.state is state:
                return None
            return current.model_copy(
                update={
                    "state": state,
                    "reasons": bounded_reasons([*current.reasons, reason]),
                    "updated_at": datetime.now(timezone.utc),
                }
            )

        if self._compare_and_set(domain, update) is not None:
            LOGGER.info(
                "shared-storage failure domain %s -> %s: %s",
                domain.incident_id,
                state.value,
                reason,
            )

    def _compare_and_set(
        self,
        incident: FaultIncident,
        update: Callable[[FaultIncident], FaultIncident | None],
    ) -> FaultIncident | None:
        """CAS ``incident`` through ``update`` (``None`` = nothing to write),
        re-reading on a concurrent write a bounded number of times."""

        current: FaultIncident | None = incident
        for _attempt in range(_WRITE_ATTEMPTS):
            if current is None:
                return None
            updated = update(current)
            if updated is None:
                return None
            try:
                self.store.save_incident(updated, expected=current)
            except StaleWriteError:
                current = self._incident(incident.incident_id)
                continue
            return updated
        LOGGER.warning(
            "incident %s kept moving while the shared-storage failure domain was "
            "recorded on it; left as is",
            incident.incident_id,
        )
        return None

    def _notify(
        self, domain: FaultIncident, identity: str, members: Sequence[_Member]
    ) -> str | None:
        nodes = sorted({member.node_id for member in members})
        scope = hashlib.sha256(
            "/".join([domain.cluster_id, identity, domain.incident_id]).encode("utf-8")
        ).hexdigest()
        incident_ids = sorted(
            {member.incident_id for member in members if member.incident_id}
        )
        candidate = AdvisoryNotification(
            deduplication_key=f"shared-storage-failure-domain/{scope}",
            cluster_name=domain.cluster_id,
            incident_id=domain.incident_id,
            subject=(
                f"[GPU fault shared storage] {identity} unavailable on "
                f"{len(nodes)} nodes ({domain.cluster_id})"
            ),
            body_text=(
                f"Cluster: {domain.cluster_id}\n"
                f"Shared filesystem: {identity}\n"
                f"Nodes: {', '.join(nodes)}\n"
                f"Failure domain record: {domain.incident_id}\n"
                "Per-node incidents:\n"
                + "".join(
                    f"  {member.node_id}: {member.incident_id or 'no open incident'}"
                    f" (mount {member.device or 'n/a'})\n"
                    for member in sorted(members, key=_node)
                )
                + "Assessment: the same export is unavailable on several nodes "
                "at once -- a shared-storage outage (common failure domain), not "
                "a fault of each node.\n"
                "Node handling: unchanged and diagnostic-only (FREEZE_EVIDENCE, "
                "VALIDATE_HOST); no node is isolated, cordoned or stopped for "
                "this.\n"
                "The record clears by itself once every member reports the "
                "export available again."
            ),
            support_case_draft=(
                f"Shared filesystem {identity} is unavailable on {len(nodes)} "
                f"nodes ({', '.join(nodes)}) in {domain.cluster_id}. Check the "
                "file server / export and the network path to it; the nodes are "
                "not suspected hardware faults."
            ),
            evidence_refs=incident_ids,
            category=SHARED_STORAGE_FAILURE_DOMAIN_NOTIFICATION_CATEGORY,
            priority=150,
        )
        notification = self.store.save_notification_if_absent(candidate)
        if notification.notification_id != candidate.notification_id:
            return None  # the same domain record already mailed once
        self.send_notification(notification.notification_id)
        notification_id: str = notification.notification_id
        return notification_id

    # --------------------------------------------------------------- readers

    def _latest_reading(
        self, cluster_id: str, node_id: str, identity: str
    ) -> TelemetryMetricLatest | None:
        """The node's newest ``shared_filesystem_unavailable`` row for the
        identity; with several mounts of one export, a failing one wins."""

        rows: list[TelemetryMetricLatest] = [
            item
            for item in self.store.list_telemetry_metrics_latest(cluster_id, node_id)
            if item.name == SHARED_FILESYSTEM_METRIC
            and shared_filesystem_identity(item.labels) == identity
        ]
        if not rows:
            return None
        failing = [item for item in rows if _failing(item.value)]
        pool = failing or rows
        return max(pool, key=lambda item: item.observed_at)

    def _open_node_incident_id(
        self, cluster_id: str, node_id: str, device: str | None
    ) -> str | None:
        suffix = f"-{SHARED_FILESYSTEM_METRIC}-{event_id_component(device or 'node')}"
        for incident in self.store.list_incidents_by_state(
            cluster_id, _OPEN_NODE_INCIDENT_STATES, node_ids={node_id}
        ):
            if incident.event_type == "NODE_HEALTH" and incident.event_id.endswith(
                suffix
            ):
                incident_id: str = incident.incident_id
                return incident_id
        return None

    def _incident(self, incident_id: str) -> FaultIncident | None:
        try:
            incident: FaultIncident = self.store.get_incident(incident_id)
        except NotFoundError:
            return None
        return incident

    def _fresh(self, observed_at: datetime, reference: datetime) -> bool:
        return observed_at >= reference - self.window

    @staticmethod
    def _member_line(member: _Member) -> str:
        return (
            f"member {member.node_id}: incident "
            f"{member.incident_id or 'none open'} ({SHARED_FILESYSTEM_METRIC} on "
            f"{member.device or 'n/a'}, observed {member.observed_at.isoformat()})"
        )


def _is_shared_filesystem_incident(incident: FaultIncident) -> bool:
    return (
        incident.event_type == "NODE_HEALTH"
        and f"-{SHARED_FILESYSTEM_METRIC}-" in incident.event_id
    )


def _failing(value: float) -> bool:
    """The gauge reads "unavailable" (the rule's threshold is 1)."""

    return value >= 1


def _available(value: float) -> bool:
    """The gauge reads "available"; NaN is neither this nor failing."""

    return value < 1


def _node(member: _Member) -> str:
    return member.node_id
