"""``config spare --release`` for a warm spare the cluster no longer has.

A declared spare that a spot reclaim, a scale-in or a provider replacement took
leaves its declaration on the deploy host (``<state-dir>/warm-spares/<node>.json``)
while the label and the cordon went with the Node: the ordinary release has
nothing to restore and its patch would fail on a Node that does not exist. The
control plane's departed-node cleanup reports the declaration with this very
command, so the command has to work for such a node:

* the node must be provably departed, by the rule the cleanup and
  ``workflow-reconcile`` share (``gpu_fault.departed_nodes``): absent from
  Kubernetes (the read that found no Node), absent from its provider --
  HyperPod's ``list-cluster-nodes`` for the instance the node name carries, or
  EC2 for the declared (else the fleet-recorded) instance on an eks-managed
  cluster -- and an Agent record that is not alive;
* the declaration record, when there is one, is released without touching a
  Node: ``released_at`` and the approval as for any release, the departure as
  the release survey, ``restored_state`` None and ``release_skipped`` saying
  why no label or cordon was put back;
* the control plane is told (``DepartedNodeSpareRelease``), which answers the
  cleanup's spare finding for that departure, the same node and instance.

A node the cluster still has takes the ordinary release; a departure short of
proof is refused with its reason, and nothing is written.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from gpu_fault.admin import operator_identity, warm_spare
from gpu_fault.admin import workflow_reconcile as reconcile
from gpu_fault.admin.atomic_json import write_json_atomic
from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.site import RenderedSite
from gpu_fault.admin.warm_spare import (
    AgentLookup,
    AgentState,
    WarmSpareError,
    WarmSpareRequest,
    _approved_reference,
    _confirmed,
    now,
    read_record,
    site_cluster,
    without_survey,
)
from gpu_fault.cluster_scope import cluster_provider
from gpu_fault.departed_nodes import (
    EKS_PROVIDER,
    agent_retirement_reasons,
    evidence_proves_departure,
    hyperpod_instance_id,
    node_evidence,
    recorded_instance_id,
)

# Runs inside a CPU ingress Pod; prints one JSON object. A control plane from
# before departed-node spare releases has no such record: said, not raised.
SPARE_RELEASE_SCRIPT = """
import json
import sys

from gpu_fault.app import ApplicationContext

payload = json.load(sys.stdin)
try:
    from gpu_fault.departed_nodes import DepartedNodeSpareRelease
except ImportError:
    print(json.dumps({"acknowledged": False, "reason": "the control plane predates departed-node spare releases"}))
else:
    store = ApplicationContext.from_environment().store
    store.save_departed_node_spare_release(
        DepartedNodeSpareRelease.model_validate(payload["release"])
    )
    print(json.dumps({"acknowledged": True}))
"""
# What ``node_scheduling_evidence`` says of any node Kubernetes lacks; it is
# the premise here, not a reason to refuse.
MISSING_NODE = "node is missing"


def declared_instance(record: Mapping[str, Any] | None) -> str | None:
    """The EC2 instance the declaration's survey saw behind the node, if any."""

    survey = (record or {}).get("pre_declaration_survey") or {}
    node = survey.get("node") if isinstance(survey, Mapping) else None
    provider_id = str((node or {}).get("provider_id") or "")
    return recorded_instance_id(provider_id.rstrip("/").rsplit("/", 1)[-1])


def _instant(value: str | None) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def agent_blockers(agent: AgentState) -> list[str]:
    """Why the node's Agent does not count as departed; empty when it does."""

    if agent.error:
        return [
            "Node Agent state could not be read from the control-plane store: "
            + agent.error
        ]
    if agent.lifecycle_state in (None, "REVOKED"):
        return []
    return agent_retirement_reasons(
        agent.lifecycle_state,
        agent.last_seen_at,
        agent.lease_expires_at,
        evaluated_at=_instant(agent.evaluated_at) or datetime.now(timezone.utc),
    )


def departed_spare_proof(
    site: RenderedSite,
    cluster_id: str,
    node: str,
    record: Mapping[str, Any] | None,
    agent: AgentState,
) -> dict[str, Any]:
    """The departed-node proof for a spare the cluster has no Node for.

    Kubernetes was just read (``kubectl get node`` answered NotFound); the
    provider is read here the way the never-changed reconcile reads it; the
    Agent is judged on the record the Pod returned, on the Pod's clock.
    """

    target = site_cluster(site, cluster_id)
    declared = declared_instance(record)
    blockers: list[str] = []
    if cluster_provider(target.get("provider")).value == EKS_PROVIDER:
        instance = declared or recorded_instance_id(agent.node_instance_id)
    else:
        instance = hyperpod_instance_id(node)
        if declared and instance and declared != instance:
            blockers.append(
                f"the declaration names instance {declared}, the node name {instance}"
            )
    read_at = {"kubernetes": now()}
    try:
        departed = reconcile._departed_node_evidence(
            site, cluster_id, node, {}, read_at, instance_id=instance
        )
    except BootstrapError as exc:
        departed = {
            "node_id": node,
            "exists": False,
            "restored": False,
            "blockers": [f"provider membership unknown: {exc}"],
        }
    evidence = node_evidence(
        node,
        departed=departed,
        target=target,
        aws_region=site.release_config.get("aws_region"),
    )
    evidence["read_at"] = dict(read_at)
    evidence["agent"] = {
        "lifecycle_state": agent.lifecycle_state,
        "last_seen_at": agent.last_seen_at,
        "lease_expires_at": agent.lease_expires_at,
    }
    blockers.extend(
        str(item) for item in departed.get("blockers") or [] if item != MISSING_NODE
    )
    blockers.extend(agent_blockers(agent))
    return {
        "proven": not blockers and evidence_proves_departure(evidence, node),
        "instance_id": instance,
        "blockers": blockers,
        "evidence": evidence,
    }


Prove = Callable[[str, str, Mapping[str, Any] | None, AgentState], dict[str, Any]]
Acknowledge = Callable[[Mapping[str, Any]], dict[str, Any]]


class DepartedSpareChannel:
    """``warm_spare.DepartedSpares``: how the proof is read and the release told.

    ``prove(cluster_id, node, record, agent)`` returns the departure proof
    (``proven``, ``instance_id``, ``blockers``, ``evidence``);
    ``acknowledge(release)`` stores the ``DepartedNodeSpareRelease`` mapping
    in the control plane and returns what happened.
    """

    def __init__(self, prove: Prove, acknowledge: Acknowledge) -> None:
        self.prove = prove
        self.acknowledge = acknowledge

    def release(
        self,
        request: WarmSpareRequest,
        *,
        cluster_id: str,
        record: Path,
        agent_lookup: AgentLookup,
        actor: str | None,
        provider: str | None,
    ) -> dict[str, Any]:
        return release_departed_spare(
            request,
            cluster_id=cluster_id,
            record=record,
            agent_lookup=agent_lookup,
            actor=actor,
            provider=provider,
            departed=self,
        )


def site_departed_spares(site: RenderedSite) -> DepartedSpareChannel:
    """The channel bound to the managed site and its control plane."""

    def prove(
        cluster_id: str,
        node: str,
        record: Mapping[str, Any] | None,
        agent: AgentState,
    ) -> dict[str, Any]:
        return departed_spare_proof(site, cluster_id, node, record, agent)

    def acknowledge(release: Mapping[str, Any]) -> dict[str, Any]:
        try:
            return reconcile._run_reconcile(
                site, {"release": dict(release)}, script=SPARE_RELEASE_SCRIPT
            )
        except BootstrapError as exc:
            return {"acknowledged": False, "reason": str(exc)}

    return DepartedSpareChannel(prove, acknowledge)


def _record_refusals(
    document: Mapping[str, Any] | None, *, node: str, cluster_id: str
) -> list[str]:
    if document is None or document.get("released_at"):
        return []
    refusals = []
    if document.get("node") != node:
        refusals.append(f"the warm-spare record names node {document.get('node')}")
    if document.get("cluster_id") != cluster_id:
        refusals.append("warm-spare record belongs to a different or unknown cluster")
    return refusals


def release_departed_spare(
    request: WarmSpareRequest,
    *,
    cluster_id: str,
    record: Path,
    agent_lookup: AgentLookup,
    actor: str | None,
    provider: str | None,
    departed: DepartedSpareChannel,
) -> dict[str, Any]:
    """Release the declaration of a spare the cluster no longer has.

    Refused, writing nothing, unless the node is provably departed. Then the
    unreleased record, if any, is released without a Node patch, and the
    control plane is told; the report says what was released, skipped and
    acknowledged.
    """

    agent = agent_lookup(cluster_id, request.node)
    document = read_record(record)
    report: dict[str, Any] = {
        "observed_at": now(),
        "cluster_id": cluster_id,
        "provider": cluster_provider(provider).value,
        "mode": "release",
        "record": str(record),
        "node": None,
        "departed": True,
        "agent": {"lifecycle_state": agent.lifecycle_state, "error": agent.error},
    }
    proof = departed.prove(cluster_id, request.node, document, agent)
    report["departure"] = proof["evidence"]
    refusals = [
        f"node is not in the cluster and its departure is unproven: {item}"
        for item in proof["blockers"]
    ]
    if not proof["proven"] and not proof["blockers"]:
        refusals.append("node is not in the cluster and its departure is unproven")
    refusals.extend(
        _record_refusals(document, node=request.node, cluster_id=cluster_id)
    )
    report["refusals"] = refusals
    report["ready"] = not refusals
    _confirmed(request)
    reference = _approved_reference(request)
    if refusals:
        raise WarmSpareError("node cannot be released: " + "; ".join(refusals))
    if actor is None:
        actor = operator_identity.resolve_operator_identity(
            fallback=operator_identity.local_operator_identity()
        )
    report["reference"] = reference
    report["actor"] = actor
    instance = str(proof["instance_id"])
    declaration = "none"
    if document is not None and not document.get("released_at"):
        document["released_at"] = now()
        document["release_reference"] = reference
        document["release_actor"] = actor
        document["release_survey"] = dict(report)
        document["restored_state"] = None
        document["release_skipped"] = (
            f"the node is no longer in the cluster and its provider no longer has "
            f"instance {instance}: the spare label and the cordon left with the "
            "Node, so there is nothing to restore"
        )
        write_json_atomic(record, document)
        report["release"] = without_survey(document)
        declaration = "released"
    else:
        report["release"] = {
            "declaration": "none",
            "reason": (
                "no unreleased gpu-fault-admin declaration for the node; the "
                "spare label left with the Node"
            ),
        }
    report["control_plane"] = departed.acknowledge(
        {
            "cluster_id": cluster_id,
            "node_id": request.node,
            "instance_id": instance,
            "change_reference": reference,
            "actor": actor,
            "released_at": datetime.now(timezone.utc).isoformat(),
            "declaration": declaration,
        }
    )
    return report


def config_spare(arguments: argparse.Namespace, *, site: RenderedSite) -> int:
    """``gpu-fault-admin config spare``, a departed spare's release bound in."""

    return warm_spare.run_config_spare_command(
        arguments, site=site, departed_for=site_departed_spares
    )
