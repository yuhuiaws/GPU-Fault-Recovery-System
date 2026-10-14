"""When a GPU node counts as departed, and how its Agent record is retired.

A spot reclaim, a scale-in or a provider replacement removes a GPU node with
everything on it, and leaves three records behind in the control plane: the
node's fleet Agent stays ``ACTIVE`` with an expired lease, the incident its
last telemetry raised keeps a workflow nobody can run, and the incident stays
open (live 2026-10-14, three reclaims in 30 minutes). Two callers end them on
the same evidence: the administrator's ``gpu-fault-admin workflow-reconcile``
(read on the deploy host) and the control plane's departed-node cleanup
(``gpu_fault.departed_node_cleanup``, read by the control plane itself). This
module is the one definition of the rule they share, so the two cannot drift:

* the node is absent from its cluster's Kubernetes Nodes -- a NotReady or
  cordoned Node is still a Node;
* the provider no longer has it: on HyperPod the instance the node name
  carries (``hyperpod-<instance-id>``) is not in ``ListClusterNodes``; on an
  ``eks-managed`` cluster EC2 reports the instance the fleet recorded for the
  node (the Agent record's ``node_instance_id``) ``terminated`` or not at all;
* its Agent is not alive: no record, a ``REVOKED`` one, or an ``ACTIVE`` /
  ``DRAINING`` one whose lease expired and whose last heartbeat is older than
  ``DEPARTED_AGENT_GUARD_AGE`` -- an Agent that spoke ten minutes ago is
  mid-restart, not gone.

Anything less -- the provider still lists the instance, the instance is
running or stopped, the name carries no instance id, no instance id was
recorded, a lookup failed -- is not proof, and the blocker says which. Nothing
here reads a source: the callers read Kubernetes and the provider their own
way and hand the answers in.

A provider identity that is unknown (no instance id in the HyperPod node
name, no EC2 instance id recorded) can never be put to the provider. Only the
administrator may then accept the departure, explicitly, once every
Kubernetes node report the control plane holds has left the node out for 24
hours or never listed it (``kubernetes_node_inventory.unknown_provider_exit``)
and its Agent is not alive; ``DepartedNodeIdentityAcceptance`` records that,
and the automatic cleanup never accepts one by itself.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping
from datetime import datetime, timedelta
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from pydantic import Field

from gpu_fault.kubernetes_node_inventory import unknown_provider_exit
from gpu_fault.models import StrictModel

if TYPE_CHECKING:
    from gpu_fault.fleet import AgentRecord, FleetRegistry

EKS_PROVIDER = "eks-managed"
HYPERPOD_PROVIDER = "hyperpod"
#: HyperPod names an EKS node after its EC2 instance.
HYPERPOD_NODE_NAME = re.compile(r"^hyperpod-(i-[0-9a-f]{8,17})$")
#: An EC2 instance id as ``spec.providerID`` and ``DescribeInstances`` spell it.
INSTANCE_ID = re.compile(r"^i-[0-9a-f]{8,17}$")
#: The EC2 answers that prove an instance gone: ``terminated``, or nothing.
NOT_FOUND = "not-found"
TERMINATED = "terminated"
DEPARTED_EC2_STATES = frozenset({NOT_FOUND, TERMINATED})
#: The source key of the eks-managed provider proof under ``sources``/``read_at``.
EC2_SOURCE = "ec2"
HYPERPOD_SOURCE = "hyperpod"

DEPARTED_AGENT_GUARD_AGE = timedelta(minutes=10)
#: An Agent record in these states is retired; ``REVOKED`` already is.
RETIREABLE_LIFECYCLES = frozenset({"ACTIVE", "DRAINING"})
GUARD_REASON = "agent heartbeat is younger than the 10-minute departure guard"
LEASE_REASON = "agent lease has not expired"
NO_HYPERPOD_INSTANCE_BLOCKER = (
    "provider membership unknown: node name carries no HyperPod instance id"
)
NO_RECORDED_INSTANCE_BLOCKER = (
    "provider membership unknown: the fleet recorded no EC2 instance id for the "
    "node (Agent node_instance_id)"
)
#: The provider can never be asked about such a node.
UNKNOWN_IDENTITY_BLOCKERS = frozenset(
    {NO_HYPERPOD_INSTANCE_BLOCKER, NO_RECORDED_INSTANCE_BLOCKER}
)
KUBERNETES_PRESENT_BLOCKER = (
    "node is still in Kubernetes; a never-dispatched record is closed only "
    "for a node that left Kubernetes and HyperPod"
)


def cluster_provider_of(target: Mapping[str, Any]) -> str:
    """The site's or the registry's ``provider`` for a cluster; absent is HyperPod."""

    return str(target.get("provider") or HYPERPOD_PROVIDER).strip() or HYPERPOD_PROVIDER


def hyperpod_instance_id(node_id: str) -> str | None:
    """The EC2 instance a HyperPod node name carries, or None."""

    match = HYPERPOD_NODE_NAME.fullmatch(str(node_id))
    return match.group(1) if match is not None else None


def recorded_instance_id(value: object) -> str | None:
    """An Agent record's ``node_instance_id`` when it is an EC2 instance id.

    The installer passes the providerID's instance id on eks-managed nodes; a
    HyperPod-era record carries a Kubernetes Node UID, which is no proof.
    """

    text = value.strip() if isinstance(value, str) else ""
    return text if INSTANCE_ID.fullmatch(text) else None


def provider_identity(provider: object, node_id: str, recorded: object) -> str | None:
    """The instance the provider would be asked about, or None when unknown."""

    if str(provider or "") == EKS_PROVIDER:
        return recorded_instance_id(recorded)
    return hyperpod_instance_id(node_id)


def provider_lookup_blocker(error: object) -> str:
    return f"provider membership unknown: {error}"


def hyperpod_listing_blocker(instance_id: str, listed: Collection[str]) -> str | None:
    """Why HyperPod's node list does not prove ``instance_id`` gone, or None."""

    if instance_id in listed:
        return (
            "node is missing from Kubernetes but HyperPod still lists "
            f"instance {instance_id}"
        )
    return None


def ec2_state_blocker(instance_id: str, state: str) -> str | None:
    """Why EC2's answer does not prove ``instance_id`` gone, or None."""

    if state in DEPARTED_EC2_STATES:
        return None
    return f"node is missing from Kubernetes but EC2 instance {instance_id} is {state}"


def _aware(value: object) -> datetime | None:
    if isinstance(value, datetime):
        parsed: datetime | None = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    return parsed if parsed is not None and parsed.tzinfo is not None else None


def agent_retirement_reasons(
    lifecycle_state: object,
    last_seen_at: object,
    lease_expires_at: object = None,
    *,
    evaluated_at: datetime,
    guard: timedelta = DEPARTED_AGENT_GUARD_AGE,
) -> list[str]:
    """Why an Agent record must *not* be retired as departed; empty when it may.

    Takes the record's fields as stored or as the inventory serialized them
    (ISO strings, a ``Z`` suffix included). The node evidence is the caller's.
    """

    reasons: list[str] = []
    lifecycle = str(getattr(lifecycle_state, "value", lifecycle_state) or "")
    if lifecycle not in RETIREABLE_LIFECYCLES:
        reasons.append(f"agent lifecycle state is {lifecycle or 'unknown'}")
    seen_at = _aware(last_seen_at)
    if seen_at is None:
        reasons.append("agent last_seen_at is missing or not timezone-aware")
    elif seen_at > evaluated_at - guard:
        reasons.append(GUARD_REASON)
    lease = _aware(lease_expires_at) if lease_expires_at is not None else None
    if lease is not None and lease > evaluated_at:
        reasons.append(LEASE_REASON)
    return reasons


def node_evidence(
    node_id: str,
    *,
    departed: Mapping[str, Any] | None,
    target: Mapping[str, Any],
    aws_region: object,
) -> dict[str, Any]:
    """The per-node departed evidence a workflow close or an Agent retire binds.

    ``departed`` is the provider verdict for a node Kubernetes no longer has
    (``exists: False``, ``provider``, ``instance_id``, ``absent_from_provider``
    or ``blockers``), ``None`` while Kubernetes still has it -- a present node
    is a blocker whatever its scheduling state: the records are closed for a
    node that is *gone*, not for one that is clean. ``sources`` names what was
    consulted (``target`` is the cluster entry, for the provider cluster name
    and region) and what each said; a source that could not be read says so
    instead of answering. The provider source is ``hyperpod``
    (``ListClusterNodes``) or, on an ``eks-managed`` cluster, ``ec2``
    (``DescribeInstances`` for the recorded instance id).
    """

    if departed is None:
        return {
            "node_id": node_id,
            "exists": True,
            "absent_from_kubernetes": False,
            "absent_from_provider": False,
            "restored": False,
            "blockers": [KUBERNETES_PRESENT_BLOCKER],
            "sources": {"kubernetes": {"consulted": True, "present": True}},
        }
    eks_managed = departed.get("provider") == EKS_PROVIDER
    provider: dict[str, Any] = {
        "consulted": "absent_from_provider" in departed,
        "cluster_name": (
            str(target.get("cluster_id") or "").strip()
            if eks_managed
            else str(target.get("hyperpod_cluster_name") or "").strip()
        )
        or None,
        "region": str(target.get("region") or aws_region or "").strip() or None,
    }
    if "absent_from_provider" in departed:
        provider["listed"] = not bool(departed["absent_from_provider"])
    if eks_managed and "ec2_state" in departed:
        provider["state"] = departed["ec2_state"]
    return {
        **departed,
        "absent_from_kubernetes": True,
        "absent_from_provider": bool(departed.get("absent_from_provider", False)),
        "sources": {
            "kubernetes": {"consulted": True, "present": False},
            (EC2_SOURCE if eks_managed else HYPERPOD_SOURCE): provider,
        },
    }


def evidence_proves_departure(evidence: Mapping[str, Any], node_id: str) -> bool:
    """Whether one node's evidence record says gone from both sources."""

    return (
        evidence.get("node_id") == node_id
        and evidence.get("absent_from_kubernetes") is True
        and evidence.get("absent_from_provider") is True
    )


def retirement_audit(
    *,
    reference: str,
    node_id: str,
    evidence: Mapping[str, Any],
    recorded_instance: str,
    actor: object,
    plan_sha256: object,
) -> str:
    """The line an Agent retirement records as its transition reason.

    The sources, their verdicts and read times, who retired it and under
    which plan: the record keeps the whole case for its own departure.
    """

    sources = evidence.get("sources") or {}
    source_name = EC2_SOURCE if EC2_SOURCE in sources else HYPERPOD_SOURCE
    label = "EC2" if source_name == EC2_SOURCE else "HyperPod"
    provider = sources.get(source_name) or {}
    read_at = evidence.get("read_at") or {}
    return (
        f"operator reconciliation {reference}: retired Agent "
        f"{node_id} after its node left Kubernetes and {label} "
        f"(instance {evidence.get('instance_id')}, record node_instance_id "
        f"{recorded_instance or 'unset'}, {label} cluster "
        f"{provider.get('cluster_name')} in {provider.get('region')}; "
        f"kubernetes read {read_at.get('kubernetes')}, {source_name} read "
        f"{read_at.get(source_name)}; actor {actor}; admin plan {plan_sha256})"
    )


def accepted_identity_audit(
    *,
    reference: str,
    node_id: str,
    reason: str,
    kubernetes: Mapping[str, Any],
    recorded_instance: str,
    actor: object,
    plan_sha256: object,
) -> str:
    """The line an Agent retired on an accepted unknown provider identity records."""

    if kubernetes.get("never_reported"):
        reports = f"never listed since {kubernetes.get('chain_started_at')}"
    elif kubernetes.get("absent_since"):
        reports = f"absent since {kubernetes.get('absent_since')}"
    else:
        reports = "absent for over 24 hours"
    return (
        f"operator reconciliation {reference}: retired Agent {node_id} after its "
        f"node left Kubernetes; its provider identity is unknown (record "
        f"node_instance_id {recorded_instance or 'unset'}) and the operator "
        f"accepted the departure: {reason} (kubernetes node reports: {reports}, "
        f"latest {kubernetes.get('received_at')}; actor {actor}; admin plan "
        f"{plan_sha256})"
    )


def unknown_identity_refusal(
    store: Any,
    *,
    cluster_id: str,
    node_id: str,
    provider: object,
    recorded_instance: object,
    now: datetime,
    guard: timedelta = DEPARTED_AGENT_GUARD_AGE,
) -> str | None:
    """Why the operator may not accept the node's departure yet; None when they may.

    The identity must really be unknown (a known one is proven through the
    provider) and the reports must allow it, judged on the store's own
    Kubernetes node inventory at ``now``.
    """

    if provider_identity(provider, node_id, recorded_instance) is not None:
        return (
            "the node's provider identity is known; its departure is proven "
            "through the provider, not accepted"
        )
    verdict = unknown_provider_exit(
        store.get_kubernetes_node_inventory(cluster_id),
        node_id,
        now=now,
        guard=guard,
    )
    return verdict.blocker


def retire_departed_agent(
    registry: FleetRegistry,
    store: Any,
    *,
    cluster_id: str,
    node_id: str,
    expected_generation: int,
    expected_lifecycle: str,
    expected_last_seen_at: datetime,
    evidence: Mapping[str, Any],
    transition_id: str,
    reason: str,
    now: datetime,
    guard: timedelta = DEPARTED_AGENT_GUARD_AGE,
    accept_unknown_provider: bool = False,
) -> AgentRecord:
    """Retire one departed node's Agent through the registry's lifecycle path.

    Compare-and-set on what the plan read -- generation, lifecycle state and
    the last heartbeat (compared as instants, not spellings) -- with the guard
    and the lease re-judged on ``now`` and the evidence re-checked, then
    ``drain_agent`` and ``revoke_agent`` under one ``AgentTransitionRequest``.
    A ``DRAINING`` record is finished under the transition that drained it --
    the registry lets only that one revoke -- which is also how a cleanup
    interrupted between the two writes resumes without a second drain.
    Raises ``ValueError`` naming the field that moved or the reason refused.

    ``accept_unknown_provider`` is the administrator's explicit exit for a
    node whose provider identity is unknown: the provider half of the proof
    is replaced by ``unknown_identity_refusal`` -- re-judged here, on the
    store's reports at ``now`` -- and the node must still be gone from
    Kubernetes in the evidence.
    """

    from gpu_fault.fleet import AgentLifecycleState, AgentTransitionRequest

    current = store.get_agent(cluster_id, node_id)
    if current.generation != expected_generation:
        raise ValueError(
            f"agent generation changed: expected {expected_generation}, "
            f"found {current.generation}"
        )
    if current.lifecycle_state.value != expected_lifecycle:
        raise ValueError(
            "agent lifecycle state changed: expected "
            f"{expected_lifecycle}, found {current.lifecycle_state.value}"
        )
    if expected_last_seen_at.tzinfo is None or (
        current.last_seen_at != expected_last_seen_at
    ):
        raise ValueError(
            "agent heartbeat changed: expected "
            f"{expected_last_seen_at.isoformat()}, found "
            f"{current.last_seen_at.isoformat()}"
        )
    refusals = agent_retirement_reasons(
        current.lifecycle_state,
        current.last_seen_at,
        current.lease_expires_at,
        evaluated_at=now,
        guard=guard,
    )
    if refusals:
        raise ValueError("; ".join(refusals))
    if accept_unknown_provider:
        if (
            evidence.get("node_id") != node_id
            or evidence.get("absent_from_kubernetes") is not True
        ):
            raise ValueError(
                "departed-node evidence does not show the node gone from Kubernetes"
            )
        refusal = unknown_identity_refusal(
            store,
            cluster_id=cluster_id,
            node_id=node_id,
            provider=evidence.get("provider"),
            recorded_instance=current.node_instance_id,
            now=now,
            guard=guard,
        )
        if refusal is not None:
            raise ValueError(f"an unknown provider identity is not accepted: {refusal}")
    elif not evidence_proves_departure(evidence, node_id):
        raise ValueError("departed-node evidence does not prove the node gone")
    # Only an EC2-shaped record value can be compared with the evidence; a
    # HyperPod record's Node UID is carried in the audit line instead.
    recorded = recorded_instance_id(current.node_instance_id)
    if recorded and evidence.get("instance_id") and recorded != evidence["instance_id"]:
        raise ValueError(
            f"agent instance changed: the record names {recorded}, "
            f"the evidence {evidence['instance_id']}"
        )
    if current.lifecycle_state is AgentLifecycleState.ACTIVE:
        request = AgentTransitionRequest(
            expected_generation=current.generation,
            transition_id=transition_id,
            reason=reason,
        )
        registry.drain_agent(cluster_id, node_id, request)
    else:
        request = AgentTransitionRequest(
            expected_generation=current.generation,
            transition_id=current.transition_id or transition_id,
            reason=reason,
        )
    return registry.revoke_agent(cluster_id, node_id, request)


class DepartedNodeCleanupStatus(StrEnum):
    """Where the automatic cleanup of one departed node stands."""

    #: Not proven departed yet (guard, lease, a source not read); retried
    #: every cycle and resolves on its own.
    PENDING = "PENDING"
    #: Proven departed, but something on it is the administrator's to end;
    #: ``admin_commands`` names the manual path.
    REFUSED = "REFUSED"
    #: Every record the departure left was ended.
    CLEANED = "CLEANED"


class DepartedNodeCleanupRecord(StrictModel):
    """The control plane's ledger line for one departed (or departing) node.

    Written by the departed-node cleanup each time its verdict changes, read
    by ``gpu-fault-admin status``. ``reference`` is the id the cleanup's
    writes carry (deterministic per departure, so a restarted cleanup resumes
    under the same one); ``actions`` is what it ended, ``blockers`` what it
    did not and why, ``admin_commands`` the exact manual fallback.
    """

    cluster_id: str
    node_id: str
    provider: str = HYPERPOD_PROVIDER
    status: DepartedNodeCleanupStatus
    reference: str | None = None
    instance_id: str | None = None
    absent_since: datetime | None = None
    blockers: list[str] = Field(default_factory=list)
    #: The bounded refusal classes behind ``blockers`` (metric labels).
    blocker_classes: list[str] = Field(default_factory=list)
    admin_commands: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    evidence: dict[str, Any] = Field(default_factory=dict)
    first_seen_at: datetime
    updated_at: datetime
    cleaned_at: datetime | None = None

    @property
    def key(self) -> str:
        return departed_node_cleanup_key(self.cluster_id, self.node_id)


class DepartedNodeSpareRelease(StrictModel):
    """An administrator released the warm-spare declaration of a departed node.

    The declaration is the deploy host's (``<state-dir>/warm-spares/<node>.json``;
    the label and the cordon left with the Node), so the departed-node cleanup
    can only report it. ``gpu-fault-admin config spare --release`` writes this
    once it proved the node gone and released the record; the cleanup then
    takes the spare finding of that departure -- the same node and instance --
    as answered.
    """

    cluster_id: str
    node_id: str
    instance_id: str
    change_reference: str
    actor: str
    released_at: datetime
    #: ``released``: a gpu-fault-admin declaration record was released;
    #: ``none``: the node carried the spare label without one.
    declaration: str

    @property
    def key(self) -> str:
        return departed_node_cleanup_key(self.cluster_id, self.node_id)


class DepartedNodeIdentityAcceptance(StrictModel):
    """An administrator accepted the departure of a node whose provider identity is unknown.

    Written by ``workflow-reconcile --retire-departed-agents
    --accept-unknown-provider-identity`` (which retires the node's Agent in
    the same apply, when there is one) after the reports allowed it
    (``unknown_identity_refusal``). The departed-node cleanup then takes the
    node as departed -- on the administrator's word, recorded in every reason
    it writes -- and ends what the node left like any departure; it never
    accepts one by itself.
    """

    cluster_id: str
    node_id: str
    change_reference: str
    actor: str
    reason: str
    accepted_at: datetime
    #: ``unknown_provider_exit`` as the apply judged it.
    kubernetes: dict[str, Any] = Field(default_factory=dict)

    @property
    def key(self) -> str:
        return departed_node_cleanup_key(self.cluster_id, self.node_id)


def departed_node_cleanup_key(cluster_id: str, node_id: str) -> str:
    return f"{cluster_id}/{node_id}"
