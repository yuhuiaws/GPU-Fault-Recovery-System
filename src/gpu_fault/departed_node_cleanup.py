"""The control plane ends what a provably departed GPU node leaves behind.

A spot reclaim, a scale-in or a provider replacement leaves the same residue
every time (live 2026-10-14: three reclaims in 30 minutes, on both providers):
the node's fleet Agent stays ``ACTIVE`` with an expired lease (``status``
reports "Agent coverage drift" and the release verification rolls back), the
incident its last telemetry raised keeps a workflow nobody can run (the
release preflight's ``workflow_safety`` refuses the deploy) and the incident
stays ``ESCALATED``. Each used to need a hand-run ``workflow-reconcile``. This
module runs the same reconcile by itself, once per ``CLEANUP_INTERVAL_SECONDS``
in the one control-worker process that holds the ``departed-node-cleanup``
periodic task lease, on evidence it reads itself:

(a) Kubernetes: the node is absent from the cluster's latest Node report and
    from every report since at least ``DEPARTED_AGENT_GUARD_AGE`` before it,
    and that report is fresh (``kubernetes_node_inventory``; the executor
    reports every Node of its cluster under the cluster token);
(b) the provider, read by the control plane: HyperPod's ``ListClusterNodes``
    no longer lists the instance the node name carries, or EC2 reports the
    Agent's recorded instance ``terminated`` or not at all
    (``departed_node_provider``; a failed or throttled lookup proves nothing
    and is retried next cycle);
(c) the Agent: no record, ``REVOKED``, or a lease that expired with a last
    heartbeat older than the guard -- the CLI's guard, ``departed_nodes``.

All three, judged by the rule ``workflow-reconcile`` judges with
(``departed_nodes``), make a node proven departed; only then are its records
ended, each through the CLI's own path (``departed_node_cleanup_actions``).
What is not the cleanup's to end -- a workflow that changed a node or has a
remote command, an incident naming a node that is still present, anything on
a cluster the registry does not hold ``ACTIVE``, and a departed spare's
declaration (the deploy host's record, released by ``config spare --release``,
which answers the finding with a ``DepartedNodeSpareRelease``) -- is reported
instead: one ledger line per node
(``DepartedNodeCleanupRecord``, read by ``gpu-fault-admin status``), one
notification naming the node and the exact administrator command, and the
refusal counter. A cleaned node gets one informational notification.

``GPU_FAULT_DEPARTED_NODE_CLEANUP_ENABLED`` (admin config
``spec.workflow.departedNodeCleanup``, default on) switches the writes off;
the CLI paths stay the manual fallback either way. Logs carry node names and
counts, never tokens.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import Any, Callable, Mapping

from gpu_fault.departed_node_cleanup_actions import (
    ACTION_KINDS,
    AGENT_ALIVE,
    BLOCKER_KINDS,
    CLUSTER_NOT_ACTIVE,
    DECLARED_SPARE,
    KUBERNETES_UNPROVEN,
    OPEN_INCIDENT_STATES,
    PROVIDER_IDENTITY_UNKNOWN,
    PROVIDER_STILL_LISTS,
    PROVIDER_UNKNOWN,
    WAITING_KINDS,
    Blocker,
    CleanupOutcome,
    DepartureWrites,
    retire_command,
    spare_command,
)
from gpu_fault.departed_node_provider import ProviderLookupError, ProviderMembership
from gpu_fault.departed_nodes import (
    DEPARTED_AGENT_GUARD_AGE,
    EC2_SOURCE,
    EKS_PROVIDER,
    HYPERPOD_SOURCE,
    NO_HYPERPOD_INSTANCE_BLOCKER,
    NO_RECORDED_INSTANCE_BLOCKER,
    NOT_FOUND,
    RETIREABLE_LIFECYCLES,
    DepartedNodeCleanupRecord,
    DepartedNodeCleanupStatus,
    DepartedNodeSpareRelease,
    agent_retirement_reasons,
    cluster_provider_of,
    ec2_state_blocker,
    hyperpod_instance_id,
    hyperpod_listing_blocker,
    node_evidence,
    provider_lookup_blocker,
    recorded_instance_id,
)
from gpu_fault.env import env_bool
from gpu_fault.fleet import AgentLifecycleState, AgentRecord
from gpu_fault.kubernetes_node_inventory import (
    NODE_INVENTORY_FRESHNESS,
    KubernetesAbsence,
    KubernetesNodeInventory,
    kubernetes_absence,
)
from gpu_fault.models import AdvisoryNotification

LOGGER = logging.getLogger(__name__)

ENABLED_ENVIRONMENT = "GPU_FAULT_DEPARTED_NODE_CLEANUP_ENABLED"
#: The periodic task lease that makes one process in the fleet the writer.
CLEANUP_TASK_KEY = "departed-node-cleanup"
CLEANUP_INTERVAL_SECONDS = 60.0
#: How long a CLEANED ledger line stays readable in ``status``.
CLEANED_RETENTION = timedelta(days=7)
#: A ledger line whose verdict did not change is rewritten this often, so
#: ``updated_at`` says when the verdict was last confirmed.
LEDGER_REFRESH = timedelta(minutes=10)


def cleanup_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """The admin-config switch as the control-plane Pods read it."""

    # The name stays literal here: the env reference infers the variable's
    # kind (boolean) from this read site.
    return env_bool(
        "GPU_FAULT_DEPARTED_NODE_CLEANUP_ENABLED",
        True,
        environ=os.environ if environ is None else environ,
    )


@dataclass(frozen=True)
class DepartedNodeCleanupConfig:
    enabled: bool = True
    interval_seconds: float = CLEANUP_INTERVAL_SECONDS
    guard: timedelta = DEPARTED_AGENT_GUARD_AGE
    cleaned_retention: timedelta = CLEANED_RETENTION

    @classmethod
    def from_environment(
        cls, environ: Mapping[str, str] | None = None
    ) -> DepartedNodeCleanupConfig:
        return cls(enabled=cleanup_enabled(environ))


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


@dataclass
class DepartureProof:
    """The evidence for one candidate node, and what it does not prove."""

    cluster_id: str
    node_id: str
    provider: str
    kubernetes: KubernetesAbsence
    agent: AgentRecord | None
    instance_id: str | None = None
    provider_state: str | None = None
    provider_read_at: datetime | None = None
    #: The provider was read and no longer has the instance.
    provider_departed: bool = False
    blockers: list[Blocker] = field(default_factory=list)

    @property
    def proven(self) -> bool:
        return not self.blockers

    def block(self, kind: str, text: str, command: str | None = None) -> None:
        self.blockers.append(Blocker(kind, text, command))

    def evidence(self, registration: Any) -> dict[str, Any]:
        """The per-node evidence record, in the CLI's ``node_evidence`` shape."""

        departed: dict[str, Any] = {
            "node_id": self.node_id,
            "exists": not self.kubernetes.proven,
            "provider": self.provider,
            "instance_id": self.instance_id,
            "absent_from_provider": self.provider_departed,
            "restored": self.proven,
            "blockers": [item.text for item in self.blockers],
        }
        if self.provider == EKS_PROVIDER and self.provider_state is not None:
            departed["ec2_state"] = self.provider_state
        source = EC2_SOURCE if self.provider == EKS_PROVIDER else HYPERPOD_SOURCE
        record = node_evidence(
            self.node_id,
            departed=departed,
            target={
                "cluster_id": self.cluster_id,
                "hyperpod_cluster_name": registration.hyperpod_cluster_name,
                "region": registration.region,
            },
            aws_region=registration.region,
        )
        # ``node_evidence`` speaks for a node Kubernetes no longer has; the
        # ledger judges candidates the reports may still list as unproven.
        record["absent_from_kubernetes"] = self.kubernetes.proven
        record["read_at"] = {
            "kubernetes": _iso(self.kubernetes.observed_at),
            source: _iso(self.provider_read_at),
        }
        record["kubernetes_absent_since"] = _iso(self.kubernetes.absent_since)
        record["kubernetes_reported_at"] = _iso(self.kubernetes.received_at)
        record["agent"] = (
            {
                "lifecycle_state": self.agent.lifecycle_state.value,
                "generation": self.agent.generation,
                "last_seen_at": _iso(self.agent.last_seen_at),
                "lease_expires_at": _iso(self.agent.lease_expires_at),
                "node_instance_id": self.agent.node_instance_id,
            }
            if self.agent is not None
            else None
        )
        return record

    def reference(self) -> str:
        """The departure's id: stable across cycles, replicas and restarts.

        Cluster, node and instance only. A proven departure is an instance its
        provider no longer has -- it never comes back -- so they name one
        departure; when the absence began is not part of it, because past the
        inventory's 24-hour memory that moment moves to the retention window.
        """

        digest = hashlib.sha256(
            "/".join((self.cluster_id, self.node_id, self.instance_id or "-")).encode()
        ).hexdigest()
        return f"departed-node-cleanup-{digest[:16]}"


class DepartedNodeCleanup:
    """One cycle per call to ``run_once``; the periodic runner schedules it."""

    def __init__(
        self,
        store: Any,
        *,
        registry: Any,
        incident_closure: Any,
        provider: ProviderMembership,
        notifications: Any | None = None,
        config: DepartedNodeCleanupConfig | None = None,
        waiting_ttl: timedelta | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.store = store
        self.registry = registry
        self.incident_closure = incident_closure
        self.provider = provider
        self.notifications = notifications
        self.config = config or DepartedNodeCleanupConfig()
        self.waiting_ttl = waiting_ttl
        self.now = now or (lambda: datetime.now(timezone.utc))
        self._lock = Lock()
        # Rendered on /metrics (``metric_lines``); process-local, so only the
        # process holding the task lease moves them.
        self.actions_total: Counter[str] = Counter()
        self.refusals_total: Counter[str] = Counter()
        self.cycles_total = 0
        self.last_cycle_timestamp_seconds = 0.0

    @property
    def interval_seconds(self) -> float:
        return self.config.interval_seconds

    def run_once(self) -> list[DepartedNodeCleanupRecord]:
        """Judge every registered cluster's departed nodes and end what is proven."""

        now = self.now()
        records: list[DepartedNodeCleanupRecord] = []
        if self.config.enabled:
            registered: set[str] = set()
            for registration in self.store.list_regional_clusters():
                registered.add(registration.cluster_id)
                if registration.synthetic:
                    continue
                records.extend(self._cluster(registration, now))
            self._prune(registered, now)
        with self._lock:
            self.cycles_total += 1
            self.last_cycle_timestamp_seconds = time.time()
        return records

    # ------------------------------------------------------------ candidates

    def _cluster(
        self, registration: Any, now: datetime
    ) -> list[DepartedNodeCleanupRecord]:
        cluster_id = registration.cluster_id
        inventory = self.store.get_kubernetes_node_inventory(cluster_id)
        agents = {agent.node_id: agent for agent in self.store.list_agents(cluster_id)}
        existing = {
            record.node_id: record
            for record in self.store.list_departed_node_cleanups(cluster_id)
        }
        releases = {
            item.node_id: item
            for item in self.store.list_departed_node_spare_releases(cluster_id)
        }
        listed = None if inventory is None else frozenset(inventory.node_names)
        for node_id in sorted(existing):
            if listed is not None and node_id in listed:
                # The node is in Kubernetes again (it never left, or the name
                # came back with a new instance): not a departure any more.
                self.store.delete_departed_node_cleanup(cluster_id, node_id)
                if node_id in releases:
                    self.store.delete_departed_node_spare_release(cluster_id, node_id)
                del existing[node_id]
        candidates = self._candidates(cluster_id, inventory, agents, existing, now)
        if not candidates:
            return []
        proofs = self._proofs(registration, candidates, inventory, agents, now)
        proven = frozenset(node for node, proof in proofs.items() if proof.proven)
        writes = DepartureWrites(
            store=self.store,
            registry=self.registry,
            incident_closure=self.incident_closure,
            now=now,
            proven=proven,
            pending=frozenset(proofs),
            evidence={node: proofs[node].evidence(registration) for node in proven},
            waiting_ttl=self.waiting_ttl,
        )
        records = []
        for node_id in sorted(proofs):
            proof = proofs[node_id]
            outcome = CleanupOutcome()
            if proof.proven:
                outcome = writes.clean(
                    cluster_id,
                    node_id,
                    agent=proof.agent,
                    reference=proof.reference(),
                )
                spare = self._spare_finding(
                    proof, existing.get(node_id), releases.get(node_id)
                )
                if spare is not None:
                    outcome.blockers.append(spare)
            records.append(
                self._ledger(registration, proof, outcome, existing.get(node_id), now)
            )
        return records

    def _candidates(
        self,
        cluster_id: str,
        inventory: KubernetesNodeInventory | None,
        agents: Mapping[str, AgentRecord],
        existing: Mapping[str, DepartedNodeCleanupRecord],
        now: datetime,
    ) -> set[str]:
        """Nodes the control plane knows of that may have left the cluster.

        A node whose Agent still holds its lease is never one: it is alive,
        whatever a report says. Otherwise an Agent that is not REVOKED, or an
        open incident's node, that the latest fresh report does not list;
        without a fresh report, only an Agent whose lease expired (judged, and
        reported as unproven, so ``status`` shows why nothing happens). A node
        already in the ledger stays a candidate until it is cleaned.
        """

        fresh = inventory is not None and (
            now - inventory.received_at <= NODE_INVENTORY_FRESHNESS
        )
        listed = frozenset(inventory.node_names) if fresh and inventory else None

        def lease_live(node_id: str) -> bool:
            agent = agents.get(node_id)
            return (
                agent is not None
                and agent.lifecycle_state is not AgentLifecycleState.REVOKED
                and agent.lease_expires_at is not None
                and agent.lease_expires_at > now
            )

        nodes: set[str] = set()
        for node_id, agent in agents.items():
            if agent.lifecycle_state is AgentLifecycleState.REVOKED:
                continue
            if lease_live(node_id):
                continue
            if listed is None or node_id not in listed:
                nodes.add(node_id)
        if listed is not None:
            for incident in self.store.list_incidents_by_state(
                cluster_id, OPEN_INCIDENT_STATES
            ):
                nodes.update(
                    node
                    for node in incident.node_ids
                    if node not in listed and not lease_live(node)
                )
        nodes.update(
            node_id
            for node_id, record in existing.items()
            if record.status is not DepartedNodeCleanupStatus.CLEANED
        )
        return nodes

    # ----------------------------------------------------------------- proof

    def _proofs(
        self,
        registration: Any,
        candidates: set[str],
        inventory: KubernetesNodeInventory | None,
        agents: Mapping[str, AgentRecord],
        now: datetime,
    ) -> dict[str, DepartureProof]:
        provider = cluster_provider_of({"provider": registration.provider})
        active = registration.is_active(now)
        guard = self.config.guard
        proofs: dict[str, DepartureProof] = {}
        for node_id in sorted(candidates):
            agent = agents.get(node_id)
            proof = DepartureProof(
                registration.cluster_id,
                node_id,
                provider,
                kubernetes_absence(inventory, node_id, now=now, guard=guard),
                agent,
            )
            if not active:
                proof.block(
                    CLUSTER_NOT_ACTIVE,
                    f"cluster {registration.cluster_id} is "
                    f"{registration.lifecycle_state.value}"
                    + ("" if registration.enabled else " and disabled")
                    + " in the registry; departed-node cleanup acts on ACTIVE "
                    "clusters only",
                    retire_command(node_id),
                )
            if proof.kubernetes.blocker is not None:
                proof.block(KUBERNETES_UNPROVEN, proof.kubernetes.blocker)
            if agent is not None and agent.lifecycle_state.value in (
                RETIREABLE_LIFECYCLES
            ):
                for reason in agent_retirement_reasons(
                    agent.lifecycle_state,
                    agent.last_seen_at,
                    agent.lease_expires_at,
                    evaluated_at=now,
                    guard=guard,
                ):
                    proof.block(AGENT_ALIVE, reason)
            if provider == EKS_PROVIDER:
                proof.instance_id = recorded_instance_id(
                    agent.node_instance_id if agent is not None else None
                )
                if proof.instance_id is None:
                    proof.block(
                        PROVIDER_IDENTITY_UNKNOWN,
                        NO_RECORDED_INSTANCE_BLOCKER,
                        retire_command(node_id),
                    )
            else:
                proof.instance_id = hyperpod_instance_id(node_id)
                if proof.instance_id is None:
                    proof.block(
                        PROVIDER_IDENTITY_UNKNOWN,
                        NO_HYPERPOD_INSTANCE_BLOCKER,
                        retire_command(node_id),
                    )
            proofs[node_id] = proof
        # The provider is asked only about nodes every other source already
        # cleared: one read per cluster per cycle, never for a node in sight.
        ready = [proof for proof in proofs.values() if proof.proven]
        if ready:
            self._read_provider(registration, provider, ready)
        return proofs

    @staticmethod
    def _spare_finding(
        proof: DepartureProof,
        previous: DepartedNodeCleanupRecord | None,
        release: DepartedNodeSpareRelease | None,
    ) -> Blocker | None:
        """The departed spare's declaration, until the administrator releases it.

        A spare's records are ended like any departed node's; its declaration
        lives on the deploy host and is only reported. Once seen the finding
        sticks to the departure (the same instance) even after the inventory
        forgets the label, and a ``DepartedNodeSpareRelease`` for that
        instance answers it.
        """

        declared = proof.kubernetes.was_spare or (
            previous is not None
            and DECLARED_SPARE in previous.blocker_classes
            and previous.instance_id == proof.instance_id
        )
        if not declared:
            return None
        if release is not None and release.instance_id == proof.instance_id:
            return None
        return Blocker(
            DECLARED_SPARE,
            "node was a declared or reserved warm spare when Kubernetes last "
            "listed it; its declaration on the deploy host is the "
            "administrator's to release (the label and the cordon left with "
            "the Node)",
            spare_command(proof.node_id, proof.cluster_id),
        )

    def _read_provider(
        self, registration: Any, provider: str, ready: list[DepartureProof]
    ) -> None:
        read_at = self.now()
        try:
            if provider == EKS_PROVIDER:
                states = self.provider.ec2_instance_states(
                    registration.region,
                    [str(proof.instance_id) for proof in ready],
                )
            else:
                listed = self.provider.hyperpod_instance_ids(
                    registration.hyperpod_cluster_name, registration.region
                )
        except ProviderLookupError as exc:
            LOGGER.warning(
                "departed-node cleanup: %s lookup for cluster %s failed; %d "
                "node(s) stay unproven: %s",
                provider,
                registration.cluster_id,
                len(ready),
                exc,
            )
            for proof in ready:
                proof.block(PROVIDER_UNKNOWN, provider_lookup_blocker(exc))
            return
        for proof in ready:
            instance_id = str(proof.instance_id)
            proof.provider_read_at = read_at
            if provider == EKS_PROVIDER:
                proof.provider_state = states.get(instance_id, NOT_FOUND)
                blocker = ec2_state_blocker(instance_id, proof.provider_state)
            else:
                blocker = hyperpod_listing_blocker(instance_id, listed)
                proof.provider_state = "listed" if blocker else "not-listed"
            if blocker is not None:
                proof.block(PROVIDER_STILL_LISTS, blocker)
            else:
                proof.provider_departed = True

    # ---------------------------------------------------------------- ledger

    def _ledger(
        self,
        registration: Any,
        proof: DepartureProof,
        outcome: CleanupOutcome,
        previous: DepartedNodeCleanupRecord | None,
        now: datetime,
    ) -> DepartedNodeCleanupRecord:
        blockers = [*proof.blockers, *outcome.blockers]
        kinds = sorted({item.kind for item in blockers})
        if not blockers:
            status = DepartedNodeCleanupStatus.CLEANED
        elif any(kind not in WAITING_KINDS for kind in kinds):
            status = DepartedNodeCleanupStatus.REFUSED
        else:
            status = DepartedNodeCleanupStatus.PENDING
        reference = proof.reference() if proof.proven else None
        same_departure = previous is not None and previous.reference == reference
        cleaned_at: datetime | None = None
        if status is DepartedNodeCleanupStatus.CLEANED:
            cleaned_at = (
                previous.cleaned_at if same_departure and previous else None
            ) or now
        actions = [
            *(previous.actions if same_departure and previous is not None else []),
            *outcome.actions,
        ]
        record = DepartedNodeCleanupRecord(
            cluster_id=proof.cluster_id,
            node_id=proof.node_id,
            provider=proof.provider,
            status=status,
            reference=reference,
            instance_id=proof.instance_id,
            absent_since=proof.kubernetes.absent_since,
            blockers=[item.text for item in blockers],
            blocker_classes=kinds,
            admin_commands=sorted(
                {item.command for item in blockers if item.command is not None}
            ),
            actions=list(dict.fromkeys(actions)),
            evidence=proof.evidence(registration),
            first_seen_at=previous.first_seen_at if previous is not None else now,
            updated_at=now,
            cleaned_at=cleaned_at,
        )
        self._count(outcome, kinds, previous)
        if self._changed(record, previous, now):
            self.store.save_departed_node_cleanup(record)
            self._log(record, previous)
            self._notify(record, previous)
        return record

    def _count(
        self,
        outcome: CleanupOutcome,
        kinds: list[str],
        previous: DepartedNodeCleanupRecord | None,
    ) -> None:
        before = set(previous.blocker_classes) if previous is not None else set()
        with self._lock:
            for kind in outcome.action_kinds:
                self.actions_total[kind] += 1
            for kind in kinds:
                if kind not in before:
                    self.refusals_total[kind] += 1

    @staticmethod
    def _changed(
        record: DepartedNodeCleanupRecord,
        previous: DepartedNodeCleanupRecord | None,
        now: datetime,
    ) -> bool:
        if previous is None:
            return True
        verdict = ("status", "reference", "blockers", "admin_commands", "actions")
        if any(getattr(record, name) != getattr(previous, name) for name in verdict):
            return True
        return now - previous.updated_at >= LEDGER_REFRESH

    @staticmethod
    def _log(
        record: DepartedNodeCleanupRecord, previous: DepartedNodeCleanupRecord | None
    ) -> None:
        if previous is not None and previous.status is record.status:
            if record.actions == previous.actions:
                return
        LOGGER.info(
            "departed-node cleanup: %s/%s %s actions=%s blockers=%s",
            record.cluster_id,
            record.node_id,
            record.status.value,
            ",".join(record.actions) or "-",
            ",".join(record.blocker_classes) or "-",
        )

    def _notify(
        self,
        record: DepartedNodeCleanupRecord,
        previous: DepartedNodeCleanupRecord | None,
    ) -> None:
        if self.notifications is None:
            return
        if record.status is DepartedNodeCleanupStatus.CLEANED:
            if previous is not None and previous.status is record.status:
                return
            candidate = cleaned_notification(record)
        elif record.status is DepartedNodeCleanupStatus.REFUSED:
            if (
                previous is not None
                and previous.status is record.status
                and previous.blocker_classes == record.blocker_classes
            ):
                return
            candidate = refused_notification(record)
        else:
            return
        try:
            saved = self.store.save_notification_if_absent(candidate)
            if saved.notification_id != candidate.notification_id:
                return
            self.notifications.send(saved.notification_id)
        except Exception:  # noqa: BLE001 - the cleanup already landed
            LOGGER.exception(
                "departed-node cleanup notification for %s/%s failed",
                record.cluster_id,
                record.node_id,
            )

    def _prune(self, registered: set[str], now: datetime) -> None:
        kept: set[str] = set()
        for record in self.store.list_departed_node_cleanups():
            expired = (
                record.status is DepartedNodeCleanupStatus.CLEANED
                and record.cleaned_at is not None
                and now - record.cleaned_at > self.config.cleaned_retention
            )
            if expired or record.cluster_id not in registered:
                self.store.delete_departed_node_cleanup(
                    record.cluster_id, record.node_id
                )
            else:
                kept.add(record.key)
        # A spare release lives as long as its ledger line, or the retention
        # when the administrator released it before the cleanup judged it.
        for release in self.store.list_departed_node_spare_releases():
            if release.key in kept:
                continue
            if (
                release.cluster_id not in registered
                or now - release.released_at > self.config.cleaned_retention
            ):
                self.store.delete_departed_node_spare_release(
                    release.cluster_id, release.node_id
                )

    # --------------------------------------------------------------- metrics

    def metric_lines(self) -> list[str]:
        with self._lock:
            actions = dict(self.actions_total)
            refusals = dict(self.refusals_total)
            cycles = self.cycles_total
            last_cycle = self.last_cycle_timestamp_seconds
        lines = [
            "# HELP gpu_fault_departed_node_cleanup_enabled Whether the control "
            "plane ends a departed GPU node's records by itself (admin config "
            "spec.workflow.departedNodeCleanup).",
            "# TYPE gpu_fault_departed_node_cleanup_enabled gauge",
            f"gpu_fault_departed_node_cleanup_enabled {int(self.config.enabled)}",
            "# HELP gpu_fault_departed_node_cleanup_cycles_total Departed-node "
            "cleanup cycles this process ran (it holds the task lease).",
            "# TYPE gpu_fault_departed_node_cleanup_cycles_total counter",
            f"gpu_fault_departed_node_cleanup_cycles_total {cycles}",
            "# HELP gpu_fault_departed_node_cleanup_last_cycle_timestamp_seconds "
            "Unix seconds of this process's last departed-node cleanup cycle.",
            "# TYPE gpu_fault_departed_node_cleanup_last_cycle_timestamp_seconds gauge",
            "gpu_fault_departed_node_cleanup_last_cycle_timestamp_seconds "
            f"{last_cycle}",
            "# HELP gpu_fault_departed_node_cleanup_actions_total Records the "
            "departed-node cleanup ended, by action.",
            "# TYPE gpu_fault_departed_node_cleanup_actions_total counter",
            *(
                "gpu_fault_departed_node_cleanup_actions_total"
                f'{{action="{kind}"}} {actions.get(kind, 0)}'
                for kind in ACTION_KINDS
            ),
            "# HELP gpu_fault_departed_node_cleanup_refusals_total Departed "
            "nodes the cleanup did not (yet) end, by reason; counted when a "
            "node's reason first appears.",
            "# TYPE gpu_fault_departed_node_cleanup_refusals_total counter",
            *(
                "gpu_fault_departed_node_cleanup_refusals_total"
                f'{{reason="{kind}"}} {refusals.get(kind, 0)}'
                for kind in sorted(BLOCKER_KINDS)
            ),
        ]
        return lines


def _summary(record: DepartedNodeCleanupRecord) -> str:
    evidence = record.evidence
    read_at = evidence.get("read_at") or {}
    agent = evidence.get("agent") or {}
    return "\n".join(
        [
            f"Cluster: {record.cluster_id}",
            f"Node: {record.node_id}",
            f"Provider: {record.provider}",
            f"Instance: {record.instance_id or 'unknown'}",
            "Absent from Kubernetes since: "
            f"{_iso(record.absent_since) or 'unknown'} "
            f"(latest node report {read_at.get('kubernetes')})",
            "Provider read: "
            f"{read_at.get(EC2_SOURCE) or read_at.get(HYPERPOD_SOURCE) or 'not read'}",
            f"Agent: {agent.get('lifecycle_state') or 'no record'}"
            + (
                f", last heartbeat {agent.get('last_seen_at')}"
                if agent.get("last_seen_at")
                else ""
            ),
            f"Reference: {record.reference or 'none'}",
        ]
    )


def cleaned_notification(record: DepartedNodeCleanupRecord) -> AdvisoryNotification:
    """The informational notice for a node whose records were all ended."""

    actions = "\n".join(f"  - {item}" for item in record.actions) or "  - none left"
    return AdvisoryNotification(
        deduplication_key=(
            f"departed-node-cleaned/{record.cluster_id}/{record.node_id}/"
            f"{record.reference}"
        ),
        cluster_name=record.cluster_id,
        incident_id=f"departed-node-{record.node_id}",
        subject=(
            "[NOTICE][GPU node departed: records ended] "
            f"{record.cluster_id}: {record.node_id}"
        ),
        body_text=(
            "The control plane proved this GPU node gone from Kubernetes and "
            "from its provider and ended the records it left behind. No action "
            "is needed.\n\n" + _summary(record) + f"\nEnded:\n{actions}\n"
        ),
        support_case_draft="",
        category="ACTION_COMPLETED",
        priority=10,
    )


def refused_notification(record: DepartedNodeCleanupRecord) -> AdvisoryNotification:
    """The notice for a departed node whose remaining records are the operator's."""

    classes = ",".join(record.blocker_classes)
    blockers = "\n".join(f"  - {item}" for item in record.blockers)
    commands = "\n".join(f"  - {item}" for item in record.admin_commands) or (
        "  - see gpu-fault-admin status"
    )
    ended = "\n".join(f"  - {item}" for item in record.actions) or "  - nothing yet"
    return AdvisoryNotification(
        deduplication_key=(
            f"departed-node-refused/{record.cluster_id}/{record.node_id}/"
            f"{record.reference}/{hashlib.sha256(classes.encode()).hexdigest()[:12]}"
        ),
        cluster_name=record.cluster_id,
        incident_id=f"departed-node-{record.node_id}",
        subject=(
            "[GPU node departed: operator action needed] "
            f"{record.cluster_id}: {record.node_id}"
        ),
        body_text=(
            "This GPU node left its cluster, but the control plane does not end "
            "every record it left behind by itself.\n\n"
            + _summary(record)
            + f"\nEnded automatically:\n{ended}\nLeft for the operator:\n"
            f"{blockers}\nManual path:\n{commands}\n"
        ),
        support_case_draft="",
        category="OPERATOR_REVIEW",
        priority=50,
    )


def departed_node_cleanup_from_context(context: Any) -> DepartedNodeCleanup | None:
    """The cleanup a regional control plane runs; None anywhere else.

    Outside regional mode, or without the fleet registry, there is no
    executor to report Kubernetes Nodes and no Agent record to retire. Building
    it opens no connection: the provider clients are created on first use.
    """

    if not getattr(context, "regional_mode", False) or context.fleet_registry is None:
        return None
    from gpu_fault.departed_node_provider import Boto3ProviderMembership
    from gpu_fault.models import WorkflowOperation

    return DepartedNodeCleanup(
        context.store,
        registry=context.fleet_registry,
        incident_closure=context.incident_closure,
        provider=Boto3ProviderMembership(),
        notifications=context.advisory_notifications,
        config=DepartedNodeCleanupConfig.from_environment(),
        # The executor's RESTART_WORKLOAD waiting cap, as the CLI's apply
        # passes it: which WAITING restart reservations a close releases.
        waiting_ttl=timedelta(
            seconds=context.production_executor_config.step_waiting_limit(
                WorkflowOperation.RESTART_WORKLOAD
            )
        ),
    )


def departed_node_cleanup_metric_lines(
    cleanup: DepartedNodeCleanup | None,
) -> list[str]:
    """``/metrics`` lines of the cleanup; none where the cleanup does not run."""

    return cleanup.metric_lines() if cleanup is not None else []
