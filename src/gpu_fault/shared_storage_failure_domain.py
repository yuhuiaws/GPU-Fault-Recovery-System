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
  per-node incidents stay the authority for each node and are not changed in
  any way (nothing is isolated or stopped for the outage). Every member's
  per-node incident gets one reason line pointing back at the domain record,
  so a later per-node notification carries the reference; it names that
  incident's own handling of the node -- diagnostic-only for the chain a
  shared-storage finding mints for itself, the recovery's action
  (``REBOOT_NODE``, ``RESET_GPU``, ``RESTART_WORKLOAD`` ...) for a member whose
  finding sits on another fault's record, and the record's member line and
  the notification name that action too (b215; :func:`_handling`).
* **Member incident.** A member's per-node incident is the one holding its
  activation, under any label the product gives it: the activation's own
  record, the record of a managed attempt's in-flight recovery the grouping
  folded the finding into (under the attempt's group label and the identity
  of the fault that minted it, possibly not even naming the node; b213), or
  the record of a reboot the product is running on the node, which absorbs
  every finding of the node without re-labelling it (``NODE_HEALTH`` for a
  lost GPU's reboot, the fault family's ``XID``; ``orchestration/
  reboot_window.py``, b215). Only the finding's marker ties the node to the
  last two (``_activation_nodes``); the peer look, the safety net and the
  member-incident lookup all follow it, whichever node reports first.
* **Window.** A domain forms when at least two nodes report the same identity
  unavailable within :data:`SHARED_STORAGE_FAILURE_DOMAIN_WINDOW_SECONDS` of
  each other (node clocks); a node whose activation comes later joins the
  open record. One notification per domain record.
* **Two looks per activation.** A shared outage hits every node in the same
  collector tick, so the activations arrive seconds apart and are ingested in
  separate transactions (one per batch, on whichever replica took the POST;
  READ COMMITTED). The look inside the transaction
  (:meth:`~SharedStorageFailureDomainCorrelator.observe_finding`) sees only
  peers that had already committed, so two activations in flight at once both
  stand alone -- live on HyperPod: two nodes, one NFS export, incidents 3.4 s
  apart, no record. The host collector re-delivers an unchanged "unavailable"
  reading only with its health summary (300 s jittered, ``_edge_reasons``),
  never per tick, so the per-batch safety net
  (:meth:`~SharedStorageFailureDomainCorrelator.observe_batch`) cannot close
  that gap inside a shorter outage. Hence the second look
  (:meth:`~SharedStorageFailureDomainCorrelator.observe_committed`) after the
  batch's own commit: whichever of two concurrent activations commits last
  sees the other. Identical member sets derive identical record ids and the
  notification is deduplicated per record, so two late looks converge on one
  record and one mail.
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
    RecoveryAction,
    bounded_reasons,
)
from gpu_fault.orchestration.families.identity import derived_record_id
from gpu_fault.orchestration.families.incident_labels import (
    GROUPED_INCIDENT_EVENT_TYPES,
    NODE_HEALTH_EVENT_TYPE,
    NODE_HEALTH_INCIDENT_EVENT_TYPES,
    routed_finding_markers,
)
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
#: failure. The host collector ticks every 15 s but *delivers* an unchanged
#: reading only with its health summary (300 s, jittered), so the window
#: spans one summary period with margin: a member's latest row is at most one
#: summary old while its export is still down, and the per-batch safety net
#: gets one summary inside the window.
SHARED_STORAGE_FAILURE_DOMAIN_WINDOW_SECONDS = 600.0
#: The states an open domain record can be in: DETECTED while a member still
#: reports a fresh failure, ESCALATED once only silent members are left.
OPEN_DOMAIN_STATES = frozenset({IncidentState.DETECTED, IncidentState.ESCALATED})
#: The effective actions of a record that only looks at its node: a
#: shared-storage finding's own chain is ``RUN_DIAGNOSTICS`` (FREEZE_EVIDENCE,
#: VALIDATE_HOST). Any other action is a recovery the record runs for a fault
#: of its own (:func:`_handling`).
_DIAGNOSTIC_ONLY_ACTIONS = frozenset(
    {
        RecoveryAction.RUN_DIAGNOSTICS,
        RecoveryAction.COLLECT_EVIDENCE,
        RecoveryAction.VALIDATE_NODE,
    }
)
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
        the correlation window after the activation. The collector delivers
        such a reading only with its next health summary, so this net catches
        outages that outlive a summary period; the shorter ones are the job of
        :meth:`observe_committed`.
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

    def observe_committed(
        self, batch: HostTelemetryBatch, findings: Sequence[NodeHealthFinding]
    ) -> list[str]:
        """After the batch's ingestion transaction committed: look again.

        :meth:`observe_finding` ran inside the transaction and could see only
        peers whose own transaction had committed by then; two nodes losing one
        export are detected by the same collector tick on each, their batches
        land seconds apart on separate replicas, and when both are in flight
        at once each stands alone. Called from the finish half of host
        ingestion, after this batch's incident and latest row are durable, so
        the activation that commits last sees the one before it. Only the
        activations this batch minted are looked at (an incident is what a
        member is); a node that already belongs to a record, or one whose
        peers are still not visible, costs one indexed read. Returns the
        notification ids sent (one per record created).
        """

        notifications: list[str] = []
        for finding in findings:
            if finding.metric_name != SHARED_FILESYSTEM_METRIC:
                continue
            identity = shared_filesystem_identity(finding.diagnostic_parameters)
            if identity is None:
                continue
            incident_id = self._node_incident_id(finding)
            if incident_id is None:
                continue  # minted no incident: nothing a member could be
            member = _Member(
                finding.node_id, finding.device, incident_id, finding.observed_at
            )
            _domain, notification_id = self._correlate(
                batch.cluster_id, identity, member
            )
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
                domain = self._join(
                    domain, identity, member, self._member_handling(member)
                )
            self._reference_domain(domain, identity, member)
            return domain, None
        peers = self._fresh_failing_peers(cluster_id, identity, member)
        if not peers:
            return None, None
        members = [member, *peers]
        handlings = {item.node_id: self._member_handling(item) for item in members}
        domain = self._create(cluster_id, identity, members, handlings)
        for item in members:
            self._reference_domain(domain, identity, item)
        return domain, self._notify(domain, identity, members, handlings)

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

        Candidates are the nodes with an open per-node incident holding an
        activation of the metric (their VALIDATE_HOST holds while the export
        stays unavailable, or the recovery or reboot the activation joined
        runs; :meth:`_activation_nodes`) and the members of open domain
        records; each candidate's latest telemetry row for the identity
        decides. A node whose reading is older than the window, or reads
        available, is not a peer. A node is one candidate however many of its
        records hold an activation, and the look costs one indexed marker read
        per open record that is neither a domain record nor an activation's
        own ``NODE_HEALTH`` chain and has a workflow -- bounded by the open-
        incident read's limit, and run only for an activation no open domain
        record holds yet, its look after the commit, and the safety net
        inside the window.
        """

        candidates: dict[str, str | None] = {}
        for incident in self.store.list_incidents_by_state(
            cluster_id, _OPEN_NODE_INCIDENT_STATES
        ):
            if is_shared_storage_failure_domain(incident):
                for node_id in incident.node_ids:
                    candidates.setdefault(node_id, None)
                continue
            for node_id in self._activation_nodes(incident):
                # Newest first: the first incident seen for a node is the one
                # its current activation opened or was folded into.
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
        self,
        cluster_id: str,
        identity: str,
        members: Sequence[_Member],
        handlings: Mapping[str, str | None],
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
            "not a per-node hardware fault: no node is isolated or stopped for "
            "it, every node keeps its own incident's handling (diagnostic-only "
            "unless its member line names an action)",
            *(
                self._member_line(member, handlings.get(member.node_id))
                for member in sorted(members, key=_node)
            ),
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
        self,
        domain: FaultIncident,
        identity: str,
        member: _Member,
        handling: str | None,
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
                            self._member_line(member, handling),
                        ]
                    ),
                    "updated_at": datetime.now(timezone.utc),
                }
            )

        return self._compare_and_set(domain, update) or domain

    def _reference_domain(
        self, domain: FaultIncident, identity: str, member: _Member
    ) -> None:
        """One reason line on the member's per-node incident, pointing back.

        The line says what stays unchanged in the words of the record it sits
        on (:func:`_handling`): a shared-storage finding's own chain is
        diagnostic-only, a record the finding was folded into or absorbed by
        keeps its own recovery -- an attempt's XID 13 restart, an XID 48 reset,
        the reboot of a lost GPU (b215).
        """

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
            unchanged = _unchanged_handling(_handling(current))
            return current.model_copy(
                update={
                    "reasons": bounded_reasons(
                        [
                            *current.reasons,
                            f"{prefix} shared filesystem {identity} is unavailable "
                            f"on {', '.join(others) or 'other nodes'} as well "
                            f"(shared-storage outage, not a node fault); {unchanged}",
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
        self,
        domain: FaultIncident,
        identity: str,
        members: Sequence[_Member],
        handlings: Mapping[str, str | None],
    ) -> str | None:
        nodes = sorted({member.node_id for member in members})
        scope = hashlib.sha256(
            "/".join([domain.cluster_id, identity, domain.incident_id]).encode("utf-8")
        ).hexdigest()
        incident_ids = sorted(
            {member.incident_id for member in members if member.incident_id}
        )
        # A member whose finding sits on another fault's record (an attempt's
        # recovery, a reboot) is not on a diagnostic chain of its own: the
        # mail names that record's action instead of claiming "diagnostic-only"
        # for every node (b215).
        recovering = any(handlings.get(node) is not None for node in nodes)
        node_handling = (
            "Node handling: unchanged; no node is isolated, cordoned or stopped "
            "for this outage. A node's own shared-storage incident is "
            "diagnostic-only (FREEZE_EVIDENCE, VALIDATE_HOST); a node listed "
            "with an action keeps that incident's own recovery, which this "
            "outage does not change.\n"
            if recovering
            else "Node handling: unchanged and diagnostic-only (FREEZE_EVIDENCE, "
            "VALIDATE_HOST); no node is isolated, cordoned or stopped for "
            "this.\n"
        )
        nodes_assessment = (
            "the outage is not a hardware fault of the nodes (a node listed "
            "with an action is in a recovery of its own for another fault)."
            if recovering
            else "the nodes are not suspected hardware faults."
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
                    f" (mount {member.device or 'n/a'}"
                    f"{_handling_suffix(handlings.get(member.node_id))})\n"
                    for member in sorted(members, key=_node)
                )
                + "Assessment: the same export is unavailable on several nodes "
                "at once -- a shared-storage outage (common failure domain), not "
                "a fault of each node.\n"
                + node_handling
                + "The record clears by itself once every member reports the "
                "export available again."
            ),
            support_case_draft=(
                f"Shared filesystem {identity} is unavailable on {len(nodes)} "
                f"nodes ({', '.join(nodes)}) in {domain.cluster_id}. Check the "
                "file server / export and the network path to it; " + nodes_assessment
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

    def _node_incident_id(self, finding: NodeHealthFinding) -> str | None:
        """The per-node incident holding ``finding``: by its event first (the
        orchestrator may have routed it into an incident with another id),
        else the node's open incident for the metric and device."""

        incident = self.store.get_incident_by_event(finding.event_id)
        if incident is not None and incident.state in _OPEN_NODE_INCIDENT_STATES:
            incident_id: str = incident.incident_id
            return incident_id
        return self._open_node_incident_id(
            finding.cluster_id, finding.node_id, finding.device
        )

    def _open_node_incident_id(
        self, cluster_id: str, node_id: str, device: str | None
    ) -> str | None:
        """The open incident holding ``node_id``'s activation on ``device``.

        Among the incidents naming the node first -- the activation's own
        record, a recovery it was folded into, the reboot of the node it was
        absorbed by (the reboot's record always names the node it reboots);
        then the grouped records that do not: a record-only fold into an
        attempt's recovery leaves the record's node scope alone, so only the
        activation's marker names it.
        """

        suffix = f"-{SHARED_FILESYSTEM_METRIC}-{event_id_component(device or 'node')}"
        for incident in self.store.list_incidents_by_state(
            cluster_id, _OPEN_NODE_INCIDENT_STATES, node_ids={node_id}
        ):
            if node_id in self._activation_nodes(incident, suffix):
                incident_id: str = incident.incident_id
                return incident_id
        for incident in self.store.list_incidents_by_state(
            cluster_id, _OPEN_NODE_INCIDENT_STATES
        ):
            if (
                incident.event_type in GROUPED_INCIDENT_EVENT_TYPES
                and node_id not in incident.node_ids
                and node_id in self._activation_nodes(incident, suffix)
            ):
                incident_id = incident.incident_id
                return incident_id
        return None

    def _activation_nodes(
        self, incident: FaultIncident, suffix: str | None = None
    ) -> list[str]:
        """The nodes whose activation of the metric ``incident`` holds (on the
        device ``suffix`` ends an event id with, when given).

        The record an activation minted names the metric in its event id,
        under whichever label the product's grouping gives it later
        (``incident_labels.NODE_HEALTH_INCIDENT_EVENT_TYPES``). A grouped
        record may instead hold an activation the product *folded* into it --
        a node-health finding of a managed attempt whose recovery is in
        flight joins that attempt's record, under the attempt's group label
        and the identity of the fault that minted it, and a record-only fold
        does not even add the node: the activation's marker, which ingestion
        re-points at the record it routed the finding to, is what names the
        node. A record of any label may hold one the product only *linked* to
        it: a node the product is rebooting has every finding absorbed into
        the reboot's record, which keeps its label and identity
        (``orchestration/reboot_window.py``) -- a lost GPU's ``NODE_HEALTH``
        reboot, an ``XID`` record -- so the marker is what names the
        activation there too (b215). A ``NODE_HEALTH`` record an activation
        minted holds only its own finding: its diagnostic-only chain never
        reboots a node. Neither it, nor a record without a workflow (which
        never owned a reboot), costs a marker read.
        """

        if is_shared_storage_failure_domain(incident):
            return []
        minted = _names_activation(incident.event_id, suffix)
        if minted and incident.event_type == NODE_HEALTH_EVENT_TYPE:
            return list(incident.node_ids)
        if (
            incident.event_type in GROUPED_INCIDENT_EVENT_TYPES
            or incident.workflow_request_id
        ):
            held = {
                node_id
                for marker in routed_finding_markers(self.store, incident)
                if _names_activation(marker.marker_id, suffix)
                for node_id in marker.scope.node_ids
            }
            if held:
                return sorted(held)
        if minted and incident.event_type in NODE_HEALTH_INCIDENT_EVENT_TYPES:
            return list(incident.node_ids)
        return []

    def _incident(self, incident_id: str) -> FaultIncident | None:
        try:
            incident: FaultIncident = self.store.get_incident(incident_id)
        except NotFoundError:
            return None
        return incident

    def _fresh(self, observed_at: datetime, reference: datetime) -> bool:
        return observed_at >= reference - self.window

    def _member_handling(self, member: _Member) -> str | None:
        """:func:`_handling` of the record holding ``member``'s activation."""

        if member.incident_id is None:
            return None
        incident = self._incident(member.incident_id)
        return None if incident is None else _handling(incident)

    @staticmethod
    def _member_line(member: _Member, handling: str | None) -> str:
        noted = (
            ""
            if handling is None
            else f"; that incident's handling: {handling or 'its own'}"
        )
        return (
            f"member {member.node_id}: incident "
            f"{member.incident_id or 'none open'} ({SHARED_FILESYSTEM_METRIC} on "
            f"{member.device or 'n/a'}, observed {member.observed_at.isoformat()}"
            f"{noted})"
        )


def _handling(incident: FaultIncident) -> str | None:
    """What ``incident`` does about its node, when that is more than a look.

    ``None`` for a diagnostic-only record -- the FREEZE_EVIDENCE, VALIDATE_HOST
    chain a shared-storage finding mints for itself (``RUN_DIAGNOSTICS``), or
    any other record whose effective action only observes the node. Else the
    record's effective action -- the recovery of the fault the finding was
    folded into or absorbed by: ``RESTART_WORKLOAD`` (an attempt's XID 13),
    ``RESET_GPU`` (XID 48), ``REBOOT_NODE`` (XID 79, a lost GPU) -- or, for a
    record with none, its official action; ``""`` when it names neither.
    """

    action = incident.effective_action
    if action in _DIAGNOSTIC_ONLY_ACTIONS:
        return None
    if action is not None and action is not RecoveryAction.NO_ACTION:
        return action.value
    return incident.official_action or ""


def _unchanged_handling(handling: str | None) -> str:
    """The back-reference's closing clause for :func:`_handling`."""

    if handling is None:
        return "the diagnostic-only handling of this node is unchanged"
    if handling:
        return f"this incident's {handling} handling of this node is unchanged"
    return "this incident's handling of this node is unchanged"


def _handling_suffix(handling: str | None) -> str:
    """The notification's per-node note for :func:`_handling`."""

    return "" if handling is None else f"; {handling or 'its own handling'}"


def _names_activation(identifier: str, suffix: str | None) -> bool:
    """Whether an event id, or a finding's ``marker-<event id>``, is an
    activation of the metric -- on ``suffix``'s device when given."""

    if suffix is not None:
        return identifier.endswith(suffix)
    return f"-{SHARED_FILESYSTEM_METRIC}-" in identifier


def _failing(value: float) -> bool:
    """The gauge reads "unavailable" (the rule's threshold is 1)."""

    return value >= 1


def _available(value: float) -> bool:
    """The gauge reads "available"; NaN is neither this nor failing."""

    return value < 1


def _node(member: _Member) -> str:
    return member.node_id
