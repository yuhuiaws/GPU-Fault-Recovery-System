"""Proof, read from the deploy host, that a node the release recorded has left.

A release records each cluster's Agent node set before it mutates anything
(``capture_agent_identities``) and holds the fleet to that set: the rollback
verification wants one ACTIVE, aligned Agent per recorded node, the resume
validation wants the live node set to equal the recorded one, and a fleet wave
refuses while a destructive workflow is non-terminal. A spot node the provider
reclaims *during* the release breaks all three at once (live 2026-10-06: a
HyperPod node left between waves 1 and 2; the never-dispatched PENDING
workflow the product had opened for it blocked wave 2 and the automatic
rollback, and every resumed rollback ended in ``rollback Agent heartbeat
mismatch`` because the snapshot still named four nodes while three existed).

This module is the one definition of "departed" those gates share, and it is
read-only. A recorded node counts as departed only when every source agrees:

* it is absent from the cluster's Kubernetes Nodes -- a NotReady or cordoned
  Node is still a Node and stays expected;
* the provider no longer has it: on HyperPod the instance id the node name
  carries is not in ``aws sagemaker list-cluster-nodes``; on eks-managed the
  EC2 instance id the fleet recorded for the node (``node_instance_id``) is
  ``terminated`` or unknown to ``aws ec2 describe-instances``;
* its Agent record holds no live lease: ``REVOKED`` (retired with
  ``workflow-reconcile --retire-departed-agents``), absent, or expired.

Anything less -- the provider still lists it, the instance is running or
stopped, a lookup failed, the name carries no instance id, the record still
heartbeats -- keeps the node expected, and the refusal names the node and the
reason. The proof is recorded in the release state (``departed_nodes``) and in
the deploy log, so the commit carries why a recorded node was not verified.
"""

from __future__ import annotations

import json
import subprocess
import threading
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Callable, Mapping

from gpu_fault.admin.bootstrap_common import BootstrapError
from gpu_fault.admin.workflow_reconcile_departed_nodes import (
    HYPERPOD_NODE_NAME,
    INSTANCE_ID,
    NOT_FOUND,
    TERMINATED,
    ec2_instance_states,
    hyperpod_instance_ids,
)
from gpu_fault.cluster_scope import is_eks_managed
from gpu_fault_release.regional_release_config import ClusterTarget, ReleaseError
from gpu_fault_release.regional_release_gpu_rollout import gpu_node_items
from gpu_fault_release.regional_release_narration import narrate_step
from gpu_fault_release.regional_release_probes import probe_source
from gpu_fault_release.regional_release_runtime_identity import exec_cpu_ingress_probe

#: Release-state key of the recorded proofs, ``{cluster_id: {release_id, nodes}}``.
DEPARTED_NODES_STATE_KEY = "departed_nodes"
#: Named in the evidence and the log for a departed node whose Agent record
#: is not yet ``REVOKED``: the next release's coverage check counts it.
RETIRE_DEPARTED_AGENT_HINT = (
    "retire the Agent record with gpu-fault-admin workflow-reconcile "
    "--state-dir <state-dir> --retire-departed-agents --node {node_id} "
    "--dry-run, then without --dry-run and with --reference"
)
#: Wave safety evidence keys the probe reports for never-dispatched records.
NEVER_DISPATCHED_WORKFLOWS_KEY = "never_dispatched_destructive_workflows"
# The rollback verification fans out per cluster; two clusters recording at
# once must not lose each other's rows in the read-modify-write below.
_STATE_LOCK = threading.Lock()


@dataclass(frozen=True)
class DepartureProof:
    """One cluster's verdict on the recorded nodes Kubernetes no longer has."""

    #: Evidence rows, one per provably departed node.
    departed: tuple[dict[str, Any], ...]
    #: ``<node>: <reason>`` for each node missing from Kubernetes that is not
    #: provably gone; the caller refuses on these.
    unproven: tuple[str, ...]

    @property
    def departed_node_ids(self) -> frozenset[str]:
        return frozenset(str(item["node_id"]) for item in self.departed)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _command_runner(release: Any) -> Callable[..., subprocess.CompletedProcess[str]]:
    """Adapt the release runner to the admin module's ``run_command`` shape."""

    def run(
        command: Sequence[str], *, timeout_seconds: float | None = None
    ) -> subprocess.CompletedProcess[str]:
        arguments = list(command)
        try:
            output = release.runner.run(
                arguments, capture=True, timeout_seconds=timeout_seconds
            )
        except ReleaseError as exc:
            return subprocess.CompletedProcess(arguments, 1, "", str(exc))
        return subprocess.CompletedProcess(arguments, 0, str(output or ""), "")

    return run


def agent_lifecycle_facts(
    release: Any, target: ClusterTarget, node_ids: Sequence[str]
) -> dict[str, Any]:
    """The Agent records of ``node_ids`` whatever their lifecycle state.

    ``active_agent_identities`` lists live Agents only; a node that left has a
    ``REVOKED`` or lease-expired record, which is exactly what has to be read.
    """

    raw = exec_cpu_ingress_probe(
        release,
        script=probe_source("agent_lifecycle_facts"),
        failure="Agent lifecycle facts",
        input_text=json.dumps(
            {"cluster_id": target.cluster_id, "node_ids": sorted(set(node_ids))}
        ),
    )
    try:
        result = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ReleaseError("Agent lifecycle facts are invalid") from exc
    if not isinstance(result, dict) or not isinstance(result.get("agents"), dict):
        raise ReleaseError("Agent lifecycle facts are invalid")
    return result


def _agent_verdict(
    node_id: str, record: Mapping[str, Any] | None
) -> tuple[dict[str, Any] | None, str | None, str | None]:
    """``(evidence, blocker, hint)`` for the node's Agent record."""

    if record is None:
        return None, None, None
    lifecycle = str(record.get("lifecycle_state") or "")
    evidence = {
        "lifecycle_state": lifecycle,
        "lease_expires_at": record.get("lease_expires_at"),
        "last_seen_at": record.get("last_seen_at"),
        "lease_live": bool(record.get("lease_live")),
        "node_instance_id": record.get("node_instance_id"),
    }
    if lifecycle == "REVOKED":
        return evidence, None, None
    if evidence["lease_live"]:
        return (
            evidence,
            f"Agent record is {lifecycle} with a live lease "
            f"(expires {record.get('lease_expires_at')}); the node is not gone",
            None,
        )
    return evidence, None, RETIRE_DEPARTED_AGENT_HINT.format(node_id=node_id)


def _hyperpod_membership(
    release: Any, target: ClusterTarget, node_ids: Sequence[str]
) -> dict[str, tuple[dict[str, Any] | None, str | None]]:
    """Per node: ``(provider evidence, blocker)`` from one inventory read."""

    verdicts: dict[str, tuple[dict[str, Any] | None, str | None]] = {}
    matches = {node_id: HYPERPOD_NODE_NAME.fullmatch(node_id) for node_id in node_ids}
    for node_id, match in matches.items():
        if match is None:
            verdicts[node_id] = (
                None,
                "node name carries no HyperPod instance id; provider membership "
                "is unknown",
            )
    named = [node_id for node_id, match in matches.items() if match is not None]
    if not named:
        return verdicts
    read_at = _now()
    try:
        listed = hyperpod_instance_ids(
            SimpleNamespace(release_config={"aws_region": target.region}),
            {
                "hyperpod_cluster_name": target.hyperpod_cluster_name,
                "region": target.region,
            },
            target.cluster_id,
            run_command=_command_runner(release),
        )
    except BootstrapError as exc:
        for node_id in named:
            verdicts[node_id] = (None, f"provider membership unknown: {exc}")
        return verdicts
    for node_id in named:
        match = matches[node_id]
        assert match is not None  # narrowed by ``named``
        instance_id = match.group(1)
        evidence = {
            "source": "hyperpod",
            "cluster_name": target.hyperpod_cluster_name,
            "region": target.region,
            "instance_id": instance_id,
            "listed": instance_id in listed,
            "read_at": read_at,
        }
        blocker = (
            f"HyperPod cluster {target.hyperpod_cluster_name} still lists "
            f"instance {instance_id}"
            if instance_id in listed
            else None
        )
        verdicts[node_id] = (evidence, blocker)
    return verdicts


def _ec2_membership(
    release: Any,
    target: ClusterTarget,
    node_ids: Sequence[str],
    agents: Mapping[str, Any],
) -> dict[str, tuple[dict[str, Any] | None, str | None]]:
    """Per node: ``(provider evidence, blocker)`` from one EC2 read."""

    verdicts: dict[str, tuple[dict[str, Any] | None, str | None]] = {}
    instance_ids: dict[str, str] = {}
    for node_id in node_ids:
        record = agents.get(node_id)
        raw = record.get("node_instance_id") if isinstance(record, dict) else None
        value = raw.strip() if isinstance(raw, str) else ""
        if not value or not INSTANCE_ID.fullmatch(value):
            verdicts[node_id] = (
                None,
                "the fleet recorded no EC2 instance id for the node (Agent "
                "node_instance_id); provider membership is unknown",
            )
            continue
        instance_ids[node_id] = value
    if not instance_ids:
        return verdicts
    read_at = _now()
    try:
        states = ec2_instance_states(
            target.region,
            sorted(set(instance_ids.values())),
            run_command=_command_runner(release),
        )
    except BootstrapError as exc:
        for node_id in instance_ids:
            verdicts[node_id] = (None, f"provider membership unknown: {exc}")
        return verdicts
    for node_id, instance_id in instance_ids.items():
        state = states.get(instance_id, NOT_FOUND)
        evidence = {
            "source": "ec2",
            "region": target.region,
            "instance_id": instance_id,
            "state": state,
            "read_at": read_at,
        }
        blocker = (
            None
            if state in {NOT_FOUND, TERMINATED}
            else f"EC2 instance {instance_id} is {state}"
        )
        verdicts[node_id] = (evidence, blocker)
    return verdicts


def kubernetes_node_names(items: Sequence[Mapping[str, Any]]) -> frozenset[str]:
    return frozenset(
        name
        for item in items
        for name in (str((item.get("metadata") or {}).get("name") or ""),)
        if name
    )


def departure_proof(
    release: Any,
    target: ClusterTarget,
    node_ids: Sequence[str],
    *,
    present_node_names: Collection[str] | None = None,
) -> DepartureProof:
    """Judge the recorded ``node_ids`` the cluster no longer has.

    ``present_node_names`` is the cluster's fresh Kubernetes node list when the
    caller already read it; otherwise it is read here. Nothing is executed in
    the Pod or against AWS unless a recorded node is missing from Kubernetes.
    """

    expected = sorted({str(node_id) for node_id in node_ids if str(node_id)})
    present = (
        frozenset(str(name) for name in present_node_names)
        if present_node_names is not None
        else kubernetes_node_names(gpu_node_items(release, target, fresh=True))
    )
    kubernetes_read_at = _now()
    missing = [node_id for node_id in expected if node_id not in present]
    if not missing:
        return DepartureProof((), ())
    agents = agent_lifecycle_facts(release, target, missing)["agents"]
    provider = "eks-managed" if is_eks_managed(target) else "hyperpod"
    membership = (
        _ec2_membership(release, target, missing, agents)
        if provider == "eks-managed"
        else _hyperpod_membership(release, target, missing)
    )
    departed: list[dict[str, Any]] = []
    unproven: list[str] = []
    for node_id in missing:
        record = agents.get(node_id)
        agent, agent_blocker, hint = _agent_verdict(
            node_id, record if isinstance(record, dict) else None
        )
        provider_evidence, provider_blocker = membership[node_id]
        blockers = [item for item in (provider_blocker, agent_blocker) if item]
        if blockers:
            unproven.append(f"{node_id}: " + "; ".join(blockers))
            continue
        assert provider_evidence is not None  # no blocker means it was read
        departed.append(
            {
                "node_id": node_id,
                "provider": provider,
                "instance_id": provider_evidence.get("instance_id"),
                "kubernetes": {"present": False, "read_at": kubernetes_read_at},
                "provider_inventory": provider_evidence,
                "agent": agent,
                "retire_hint": hint,
            }
        )
    return DepartureProof(tuple(departed), tuple(unproven))


def record_departed_nodes(
    release: Any,
    target: ClusterTarget,
    departed: Sequence[Mapping[str, Any]],
    *,
    phase: str,
) -> None:
    """Keep the proof in the state the next checkpoint writes, and say so."""

    if not departed:
        return
    narrate_step(
        "departed-node-excluded",
        cluster=target.cluster_id,
        phase=phase,
        nodes=",".join(str(item.get("node_id")) for item in departed),
        agents=",".join(
            str((item.get("agent") or {}).get("lifecycle_state") or "absent")
            for item in departed
        ),
        retire=",".join(
            str(item.get("node_id")) for item in departed if item.get("retire_hint")
        )
        or "-",
    )
    state = getattr(release, "state", None)
    if not isinstance(state, dict):
        return
    with _STATE_LOCK:
        existing = state.get(DEPARTED_NODES_STATE_KEY)
        kept = {
            cluster_id: entry
            for cluster_id, entry in (
                existing.items() if isinstance(existing, dict) else ()
            )
            if isinstance(entry, dict) and entry.get("release_id") == release.release_id
        }
        current = kept.get(target.cluster_id) or {}
        nodes = {
            str(item.get("node_id")): dict(item)
            for item in (current.get("nodes") or [])
            if isinstance(item, dict)
        }
        for item in departed:
            nodes[str(item.get("node_id"))] = dict(item)
        kept[target.cluster_id] = {
            "release_id": release.release_id,
            "phase": phase,
            "nodes": [nodes[node_id] for node_id in sorted(nodes)],
        }
        state[DEPARTED_NODES_STATE_KEY] = kept


def recorded_departed_nodes(
    state: Mapping[str, Any] | None, cluster_id: str
) -> tuple[str, ...]:
    """The node ids a checkpoint recorded as departed for ``cluster_id``."""

    entries = (state or {}).get(DEPARTED_NODES_STATE_KEY)
    entry = entries.get(cluster_id) if isinstance(entries, dict) else None
    if not isinstance(entry, dict):
        return ()
    recorded_release = (state or {}).get("release_id")
    if recorded_release and entry.get("release_id") != recorded_release:
        return ()
    return tuple(
        sorted(
            str(item["node_id"])
            for item in entry.get("nodes") or []
            if isinstance(item, dict) and item.get("node_id")
        )
    )


def departed_nodes_summary(state: Mapping[str, Any]) -> dict[str, Any]:
    """The recorded proofs of this release, for ``status``; empty when none."""

    entries = state.get(DEPARTED_NODES_STATE_KEY)
    if not isinstance(entries, dict):
        return {}
    mine = {
        str(cluster_id): entry
        for cluster_id, entry in entries.items()
        if isinstance(entry, dict)
        and entry.get("release_id") == state.get("release_id")
    }
    return {DEPARTED_NODES_STATE_KEY: mine} if mine else {}


def departed_node_ids(
    release: Any,
    target: ClusterTarget,
    node_ids: Sequence[str],
    *,
    phase: str,
    what: str,
    present_node_names: Collection[str] | None = None,
) -> frozenset[str]:
    """The provably departed subset of ``node_ids``, recorded; refuse otherwise.

    ``what`` names the set being judged in the refusal (``rollback Agent node
    set``, ``resume previous node set``), so the operator reads which gate
    stopped and on which node.
    """

    proof = departure_proof(
        release, target, node_ids, present_node_names=present_node_names
    )
    if proof.unproven:
        raise ReleaseError(
            f"{target.cluster_id} {what} names nodes missing from Kubernetes that "
            "are not provably departed: " + "; ".join(proof.unproven)
        )
    record_departed_nodes(release, target, proof.departed, phase=phase)
    return proof.departed_node_ids


def departed_workflow_exclusions(
    release: Any, snapshot: Mapping[str, Any]
) -> tuple[frozenset[str], tuple[str, ...]]:
    """Never-dispatched destructive workflows whose every node provably departed.

    The wave safety probe reports, under ``never_dispatched_destructive_workflows``,
    the PENDING records no adapter was ever handed a step of (the product's
    ``never_dispatched_reconciliation_reasons`` minus the departed-node proof),
    with the incident's cluster and nodes. Such a record cannot be mid-anything
    on an Agent a wave restarts once its nodes are gone, so it is set aside
    here -- and only here; the probe keeps counting it, so an engine without
    this rule still refuses. Returns the excluded request ids and, for the
    records kept, why.
    """

    entries = snapshot.get(NEVER_DISPATCHED_WORKFLOWS_KEY) or []
    targets = {target.cluster_id: target for target in release.config.clusters}
    by_cluster: dict[str, set[str]] = {}
    valid: list[tuple[str, str, frozenset[str]]] = []
    notes: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        request_id = str(entry.get("request_id") or "")
        cluster_id = str(entry.get("cluster_id") or "")
        node_ids = frozenset(
            str(node_id) for node_id in entry.get("node_ids") or [] if str(node_id)
        )
        if not request_id:
            continue
        if cluster_id not in targets:
            notes.append(f"{request_id}: cluster {cluster_id!r} is not in this release")
            continue
        if not node_ids:
            notes.append(f"{request_id}: the incident names no node")
            continue
        by_cluster.setdefault(cluster_id, set()).update(node_ids)
        valid.append((request_id, cluster_id, node_ids))
    proofs: dict[str, DepartureProof] = {}
    for cluster_id, cluster_nodes in by_cluster.items():
        proof = departure_proof(release, targets[cluster_id], sorted(cluster_nodes))
        record_departed_nodes(
            release, targets[cluster_id], proof.departed, phase="fleet-wave"
        )
        proofs[cluster_id] = proof
    excluded: set[str] = set()
    for request_id, cluster_id, node_ids in valid:
        proof = proofs[cluster_id]
        remaining = sorted(node_ids - proof.departed_node_ids)
        if not remaining:
            excluded.add(request_id)
            continue
        reasons = [
            item for item in proof.unproven if item.split(":", 1)[0] in remaining
        ] or [f"{node_id}: node is still in Kubernetes" for node_id in remaining]
        notes.append(f"{request_id}: " + "; ".join(reasons))
    return frozenset(excluded), tuple(notes)
